"""Same-loop ASGI websocket client for FastAPI tests."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import FastAPI


class AsgiWebSocket:
    def __init__(self) -> None:
        self._to_app: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._from_app: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None

    async def send_json(self, data: object) -> None:
        await self._to_app.put(
            {"type": "websocket.receive", "text": json.dumps(data)}
        )

    async def receive_json(self, timeout: float = 4.0) -> Any:
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError("no websocket message")
            msg = await asyncio.wait_for(self._from_app.get(), remaining)
            kind = msg.get("type")
            if kind == "websocket.send":
                if "text" in msg:
                    return json.loads(msg["text"])
                if "bytes" in msg:
                    return json.loads(msg["bytes"])
            if kind == "websocket.close":
                raise ConnectionError(f"websocket closed: {msg.get('code')}")

    async def close(self) -> None:
        await self._to_app.put({"type": "websocket.disconnect", "code": 1000})
        if self._task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self._task), 1.0)
            except (TimeoutError, asyncio.CancelledError):
                self._task.cancel()


async def connect_asgi_ws(
    app: FastAPI,
    path: str,
    *,
    query_string: str = "",
) -> AsgiWebSocket:
    client = AsgiWebSocket()
    scope: dict[str, Any] = {
        "type": "websocket",
        "asgi": {"spec_version": "2.3", "version": "3.0"},
        "http_version": "1.1",
        "scheme": "ws",
        "server": ("testserver", 80),
        "client": ("testclient", 123),
        "root_path": "",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": query_string.encode("ascii"),
        "headers": [(b"host", b"testserver")],
        "subprotocols": [],
        "extensions": {},
        "app": app,
        "state": {},
    }

    async def receive() -> dict[str, Any]:
        return await client._to_app.get()

    async def send(message: dict[str, Any]) -> None:
        await client._from_app.put(message)

    client._task = asyncio.create_task(app(scope, receive, send))
    await client._to_app.put({"type": "websocket.connect"})
    first = await asyncio.wait_for(client._from_app.get(), timeout=2.0)
    if first.get("type") == "websocket.close":
        raise ConnectionError(f"websocket rejected: {first.get('code')}")
    if first.get("type") != "websocket.accept":
        raise RuntimeError(f"expected websocket.accept, got {first}")
    return client
