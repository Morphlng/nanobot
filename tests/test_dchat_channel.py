import asyncio
import base64
import socket
from unittest.mock import AsyncMock

import httpx
import pytest

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.dchat import DChatChannel, DChatConfig
from nanobot.channels.manager import ChannelManager
from nanobot.config.schema import Config


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def _start_channel(channel: DChatChannel) -> asyncio.Task:
    task = asyncio.create_task(channel.start())
    for _ in range(100):
        if channel._server is not None:
            return task
        await asyncio.sleep(0.02)
    task.cancel()
    raise RuntimeError("DChat webhook server did not start in time")


async def _stop_channel(channel: DChatChannel, task: asyncio.Task) -> None:
    await channel.stop()
    try:
        await asyncio.wait_for(task, timeout=1.0)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def _build_payload(**overrides):
    payload = {
        "vchannel_id": "test",
        "subtype": "normal",
        "text": "hello",
        "message_key": "msg-1",
        "user_id": "user-1",
        "username": "Alice",
        "channel_id": "conv-1",
        "channel_name": "Test",
        "bot_id": "100",
        "bot_name": "bot",
        "bot_type": "bot_user",
        "is_quick_reply": False,
        "timestamp": 0,
        "content": {},
    }
    payload.update(overrides)
    return payload


def test_dchat_config_defaults() -> None:
    cfg = DChatConfig()
    assert cfg.webhook_path == "/dchat"
    assert cfg.dm.policy == "open"
    assert cfg.placeholder_text == "正在思考中，请稍候..."


def test_dchat_config_camel_case_deserialization() -> None:
    cfg = DChatConfig.model_validate(
        {
            "enabled": True,
            "webhookPath": "/hook",
            "placeholderText": "thinking",
            "dm": {"policy": "allowlist", "allowFrom": ["u1"]},
            "outbound": {
                "url": "http://example.com/api",
                "username": "app",
                "password": "secret",
                "botId": "42",
                "botType": "bot_user",
            },
        }
    )

    assert cfg.enabled is True
    assert cfg.webhook_path == "/hook"
    assert cfg.placeholder_text == "thinking"
    assert cfg.dm.policy == "allowlist"
    assert cfg.dm.allow_from == ["u1"]
    assert cfg.outbound.bot_id == "42"


def test_dchat_default_config_for_onboard() -> None:
    cfg = DChatChannel.default_config()
    assert isinstance(cfg, dict)
    assert cfg["enabled"] is False
    assert cfg["webhookPath"] == "/dchat"
    assert cfg["placeholderText"] == "正在思考中，请稍候..."
    assert cfg["dm"]["policy"] == "open"
    assert cfg["outbound"]["url"] == ""


@pytest.mark.asyncio
async def test_dchat_webhook_immediate_response_and_inbound_dispatch() -> None:
    bus = MessageBus()
    port = _free_port()
    cfg = DChatConfig(enabled=True, webhook_path="/dchat")
    channel = DChatChannel(cfg, bus, listen_host="127.0.0.1", listen_port=port)
    task = await _start_channel(channel)

    try:
        payload = _build_payload(text="@<=#123=> hello world")
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.post(f"http://127.0.0.1:{port}/dchat", json=payload)

        assert resp.status_code == 200
        assert resp.json() == {"text": "正在思考中，请稍候..."}

        msg = await asyncio.wait_for(bus.consume_inbound(), timeout=1.0)
        assert msg.sender_id == "user-1"
        assert msg.chat_id == "user-1"
        assert msg.content == "hello world"
        assert msg.session_key == "dchat:conv-1"
        assert msg.metadata["dchat"]["message_key"] == "msg-1"
    finally:
        await _stop_channel(channel, task)


@pytest.mark.asyncio
async def test_dchat_allowlist_denied_sends_unauthorized_and_skips_inbound() -> None:
    bus = MessageBus()
    port = _free_port()
    cfg = DChatConfig(enabled=True)
    cfg.dm.policy = "allowlist"
    cfg.dm.allow_from = ["other-user"]
    channel = DChatChannel(cfg, bus, listen_host="127.0.0.1", listen_port=port)
    channel._push_text = AsyncMock()  # type: ignore[method-assign]
    task = await _start_channel(channel)

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.post(
                f"http://127.0.0.1:{port}/dchat", json=_build_payload()
            )
        assert resp.status_code == 200
        await asyncio.sleep(0.1)

        channel._push_text.assert_awaited_once_with(
            "user-1", "You are not authorized to use this channel."
        )
        assert bus.inbound_size == 0
    finally:
        await _stop_channel(channel, task)


@pytest.mark.asyncio
async def test_dchat_disabled_policy_sends_disabled_and_skips_inbound() -> None:
    bus = MessageBus()
    port = _free_port()
    cfg = DChatConfig(enabled=True)
    cfg.dm.policy = "disabled"
    channel = DChatChannel(cfg, bus, listen_host="127.0.0.1", listen_port=port)
    channel._push_text = AsyncMock()  # type: ignore[method-assign]
    task = await _start_channel(channel)

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.post(
                f"http://127.0.0.1:{port}/dchat", json=_build_payload()
            )
        assert resp.status_code == 200
        await asyncio.sleep(0.1)

        channel._push_text.assert_awaited_once_with(
            "user-1", "This channel is currently disabled."
        )
        assert bus.inbound_size == 0
    finally:
        await _stop_channel(channel, task)


class _FakeResponse:
    def __init__(self, status_code: int = 200, text: str = ""):
        self.status_code = status_code
        self.reason_phrase = "OK"
        self.text = text


class _FakeHttpClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def post(self, url: str, headers: dict, json: dict):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return _FakeResponse()


@pytest.mark.asyncio
async def test_dchat_send_uses_expected_headers_and_payload() -> None:
    cfg = DChatConfig(enabled=True)
    cfg.outbound.url = "http://im-server/api/v3/message.create"
    cfg.outbound.username = "app_key"
    cfg.outbound.password = "app_secret"
    cfg.outbound.bot_id = "999"
    cfg.outbound.bot_type = "bot_user"

    channel = DChatChannel(cfg, MessageBus())
    fake_http = _FakeHttpClient()
    channel._http = fake_http  # type: ignore[assignment]

    await channel.send(OutboundMessage(channel="dchat", chat_id="12345", content="hi"))

    assert len(fake_http.calls) == 1
    call = fake_http.calls[0]
    expected_auth = base64.b64encode(b"app_key:app_secret").decode("ascii")
    assert call["url"] == "http://im-server/api/v3/message.create"
    assert call["headers"]["Authorization"] == f"Basic {expected_auth}"
    assert call["headers"]["X-Bot-Id"] == "999"
    assert call["headers"]["X-Bot-Type"] == "bot_user"
    assert call["json"]["receive_id"] == 12345
    assert call["json"]["receive_id_type"] == "1"
    assert call["json"]["text"] == "hi"


@pytest.mark.asyncio
async def test_dchat_message_key_dedup() -> None:
    bus = MessageBus()
    port = _free_port()
    cfg = DChatConfig(enabled=True, webhook_path="/dchat")
    channel = DChatChannel(cfg, bus, listen_host="127.0.0.1", listen_port=port)
    task = await _start_channel(channel)

    try:
        payload = _build_payload(message_key="dup-key")
        async with httpx.AsyncClient(timeout=3.0) as client:
            first = await client.post(f"http://127.0.0.1:{port}/dchat", json=payload)
            second = await client.post(f"http://127.0.0.1:{port}/dchat", json=payload)

        assert first.status_code == 200
        assert second.status_code == 200
        _ = await asyncio.wait_for(bus.consume_inbound(), timeout=1.0)
        await asyncio.sleep(0.2)
        assert bus.inbound_size == 0
    finally:
        await _stop_channel(channel, task)


def test_channel_manager_registers_dchat_with_gateway_binding() -> None:
    cfg = Config.model_validate(
        {
            "channels": {
                "dchat": {
                    "enabled": True,
                    "outbound": {"url": "http://im-server/api/v3/message.create"},
                }
            },
            "gateway": {"host": "127.0.0.1", "port": 18888},
        }
    )

    manager = ChannelManager(cfg, MessageBus())
    channel = manager.get_channel("dchat")

    assert isinstance(channel, DChatChannel)
    assert channel._listen_host == "127.0.0.1"
    assert channel._listen_port == 18888
