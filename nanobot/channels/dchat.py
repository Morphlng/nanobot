"""DChat channel using HTTP webhook inbound and outbound push API."""

from __future__ import annotations

import asyncio
import base64
import json
import re
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import DChatConfig

MAX_PAYLOAD_BYTES = 512 * 1024
DEDUP_TTL_SECONDS = 10 * 60
DEDUP_MAX_KEYS = 2000


class DChatChannel(BaseChannel):
    """DChat channel implemented with webhook callbacks and outbound HTTP push."""

    name = "dchat"
    _MENTION_RE = re.compile(r"@<=#\d+=>\s*")

    def __init__(
        self,
        config: DChatConfig,
        bus: MessageBus,
        *,
        listen_host: str = "0.0.0.0",
        listen_port: int = 18790,
    ):
        super().__init__(config, bus)
        self.config: DChatConfig = config
        self._listen_host = listen_host
        self._listen_port = listen_port
        self._webhook_path = self._normalize_path(config.webhook_path)

        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: ThreadingHTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        self._http: httpx.AsyncClient | None = None

        self._seen_message_keys: dict[str, float] = {}
        self._seen_order: deque[str] = deque()
        self._seen_lock = threading.Lock()

    async def start(self) -> None:
        """Start webhook server and keep channel alive."""
        self._running = True
        self._loop = asyncio.get_running_loop()
        self._webhook_path = self._normalize_path(self.config.webhook_path)

        if not self._start_webhook_server():
            self._running = False
            self._loop = None
            return

        logger.info(
            "DChat webhook listening on {}:{}{}",
            self._listen_host,
            self._listen_port,
            self._webhook_path,
        )
        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        """Stop webhook server and HTTP clients."""
        self._running = False

        if self._server:
            try:
                self._server.shutdown()
            except Exception as e:
                logger.debug("DChat webhook shutdown warning: {}", e)
            try:
                self._server.server_close()
            except Exception as e:
                logger.debug("DChat server_close warning: {}", e)
            self._server = None

        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=3.0)
        self._server_thread = None

        if self._http:
            await self._http.aclose()
            self._http = None
        self._loop = None

    async def send(self, msg: OutboundMessage) -> None:
        """Push a text message to DChat outbound API."""
        if not msg.content or not msg.content.strip():
            return
        await self._push_text(msg.chat_id, msg.content)

    def is_allowed(self, sender_id: str) -> bool:
        """DChat permissions are handled via dm.policy, not allow_from."""
        return True

    @classmethod
    def _strip_mention(cls, text: str) -> str:
        return cls._MENTION_RE.sub("", text).strip()

    @staticmethod
    def _normalize_path(path: str) -> str:
        raw = (path or "").strip()
        if not raw:
            raw = "/dchat"
        if not raw.startswith("/"):
            raw = f"/{raw}"
        normalized = raw.rstrip("/")
        return normalized or "/"

    @staticmethod
    def _path_only(path: str) -> str:
        base = (path or "/").split("?", 1)[0]
        normalized = base.rstrip("/")
        return normalized or "/"

    @staticmethod
    def _coerce_receive_id(value: str) -> int | str:
        cleaned = str(value).strip()
        return int(cleaned) if cleaned.isdigit() else cleaned

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=20.0)
        return self._http

    async def _push_text(self, receive_id: str, text: str) -> None:
        outbound = self.config.outbound
        if not outbound.url:
            logger.warning("DChat outbound.url is not configured; skipping send")
            return
        if not receive_id or not text.strip():
            return

        credentials = base64.b64encode(
            f"{outbound.username}:{outbound.password}".encode("utf-8")
        ).decode("ascii")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Basic {credentials}",
            "X-Bot-Id": outbound.bot_id,
            "X-Bot-Type": outbound.bot_type or "bot_user",
        }
        payload = {
            "receive_id": self._coerce_receive_id(receive_id),
            "receive_id_type": "1",
            "text": text,
        }

        try:
            client = await self._get_http()
            resp = await client.post(outbound.url, headers=headers, json=payload)
            if resp.status_code >= 400:
                body = (resp.text or "")[:500]
                logger.error(
                    "DChat outbound API error: {} {} {}",
                    resp.status_code,
                    resp.reason_phrase,
                    body,
                )
        except Exception as e:
            logger.error("DChat outbound send failed: {}", e)

    def _start_webhook_server(self) -> bool:
        handler_cls = self._build_handler()
        try:
            server = ThreadingHTTPServer((self._listen_host, self._listen_port), handler_cls)
            server.daemon_threads = True
        except OSError as e:
            logger.error(
                "Failed to bind DChat webhook server at {}:{}: {}",
                self._listen_host,
                self._listen_port,
                e,
            )
            return False

        self._server = server
        self._server_thread = threading.Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.5},
            name="dchat-webhook-server",
            daemon=True,
        )
        self._server_thread.start()
        return True

    def _build_handler(self) -> type[BaseHTTPRequestHandler]:
        channel = self

        class DChatWebhookHandler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - HTTP verb name
                channel._handle_http_request(self)

            def do_GET(self) -> None:  # noqa: N802 - HTTP verb name
                channel._handle_http_request(self)

            def do_PUT(self) -> None:  # noqa: N802 - HTTP verb name
                channel._handle_http_request(self)

            def do_DELETE(self) -> None:  # noqa: N802 - HTTP verb name
                channel._handle_http_request(self)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
                return

        return DChatWebhookHandler

    @staticmethod
    def _write_json(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    @staticmethod
    def _write_text(handler: BaseHTTPRequestHandler, status: int, text: str) -> None:
        body = text.encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "text/plain; charset=utf-8")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    def _handle_http_request(self, handler: BaseHTTPRequestHandler) -> None:
        path = self._path_only(handler.path)
        if path != self._webhook_path:
            self._write_text(handler, 404, "Not Found")
            return

        if handler.command != "POST":
            handler.send_response(405)
            handler.send_header("Allow", "POST")
            handler.end_headers()
            return

        content_length_raw = handler.headers.get("Content-Length", "0")
        try:
            content_length = int(content_length_raw)
        except ValueError:
            self._write_text(handler, 400, "invalid content length")
            return

        if content_length < 0 or content_length > MAX_PAYLOAD_BYTES:
            self._write_text(handler, 413, "payload too large")
            return

        raw = handler.rfile.read(content_length)
        if not raw:
            self._write_text(handler, 400, "empty payload")
            return

        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            self._write_text(handler, 400, "invalid json")
            return

        if not isinstance(payload, dict):
            self._write_text(handler, 400, "invalid payload")
            return

        sender_id = str(payload.get("user_id", "")).strip()
        text = str(payload.get("text", "")).strip()
        if not sender_id or not text:
            self._write_json(handler, 400, {"error": "Missing required fields: user_id, text"})
            return

        self._write_json(handler, 200, {"text": self.config.placeholder_text})
        self._schedule_inbound(payload)

    def _schedule_inbound(self, payload: dict[str, Any]) -> None:
        if not self._running or self._loop is None:
            logger.warning("DChat channel is not running; dropping inbound payload")
            return
        try:
            fut = asyncio.run_coroutine_threadsafe(self._process_inbound(payload), self._loop)
            fut.add_done_callback(self._log_inbound_future)
        except Exception as e:
            logger.error("Failed to schedule DChat inbound processing: {}", e)

    @staticmethod
    def _log_inbound_future(fut: "asyncio.Future[Any]") -> None:
        try:
            fut.result()
        except Exception as e:
            logger.error("Error processing DChat inbound message: {}", e)

    def _is_duplicate_message_key(self, message_key: str) -> bool:
        if not message_key:
            return False
        now = time.monotonic()
        cutoff = now - DEDUP_TTL_SECONDS

        with self._seen_lock:
            while self._seen_order:
                oldest = self._seen_order[0]
                ts = self._seen_message_keys.get(oldest)
                if ts is None:
                    self._seen_order.popleft()
                    continue
                if ts >= cutoff:
                    break
                self._seen_order.popleft()
                self._seen_message_keys.pop(oldest, None)

            if message_key in self._seen_message_keys:
                return True

            self._seen_message_keys[message_key] = now
            self._seen_order.append(message_key)
            while len(self._seen_message_keys) > DEDUP_MAX_KEYS and self._seen_order:
                oldest = self._seen_order.popleft()
                self._seen_message_keys.pop(oldest, None)
            return False

    async def _process_inbound(self, payload: dict[str, Any]) -> None:
        sender_id = str(payload.get("user_id", "")).strip()
        sender_name = str(payload.get("username", "")).strip()
        message_text = self._strip_mention(str(payload.get("text", "")))
        message_key = str(payload.get("message_key", "")).strip()
        conversation_id = str(payload.get("channel_id", "")).strip() or sender_id

        if not sender_id or not message_text:
            return
        if message_key and self._is_duplicate_message_key(message_key):
            return

        policy = (self.config.dm.policy or "open").lower()
        allow_from = [str(item) for item in (self.config.dm.allow_from or [])]

        if policy == "disabled":
            await self._push_text(sender_id, "This channel is currently disabled.")
            return

        if policy == "allowlist":
            allowed = "*" in allow_from or sender_id in allow_from
            if not allowed:
                await self._push_text(sender_id, "You are not authorized to use this channel.")
                return

        await self._handle_message(
            sender_id=sender_id,
            chat_id=sender_id,
            content=message_text,
            metadata={
                "dchat": {
                    "message_key": message_key or None,
                    "conversation_id": conversation_id,
                    "sender_name": sender_name or None,
                    "raw": payload,
                }
            },
            session_key=f"dchat:{conversation_id}",
        )
