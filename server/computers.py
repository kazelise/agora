"""Computer registry: pairing tokens and Redis-backed presence.

Sockets stay on the worker that accepted them. Presence is a Redis key
so GET /computers and wake routing see a Computer connected to another
worker. The value is that worker's id; disconnect only deletes the key
if we still own it (another worker may have taken the socket).

last_seen_at remains the heartbeat fallback for the list. Wake routing
asks Redis first; a Redis error falls back to the local socket.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from datetime import datetime, timedelta, timezone
from uuid import UUID

import redis.asyncio as redis
from fastapi import WebSocket
from starlette.websockets import WebSocketState

from server.bus import clear_presence, has_presence, mark_presence

logger = logging.getLogger("agora.computers")

ONLINE_GRACE_S = 30


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def is_recent(last_seen_at: datetime | None, *, now: datetime | None = None) -> bool:
    if last_seen_at is None:
        return False
    clock = now or datetime.now(timezone.utc)
    if last_seen_at.tzinfo is None:
        last_seen_at = last_seen_at.replace(tzinfo=timezone.utc)
    return clock - last_seen_at <= timedelta(seconds=ONLINE_GRACE_S)


class ComputerHub:
    def __init__(
        self,
        redis_client: redis.Redis | None = None,
        worker_id: str | None = None,
    ) -> None:
        self._sockets: dict[UUID, WebSocket] = {}
        self._redis = redis_client
        self.worker_id = worker_id or secrets.token_hex(8)

    async def connect(self, computer_id: UUID, ws: WebSocket) -> None:
        await ws.accept()
        previous = self._sockets.get(computer_id)
        self._sockets[computer_id] = ws
        if self._redis is not None:
            await mark_presence(self._redis, computer_id, self.worker_id)
        if previous is not None and previous is not ws:
            if previous.client_state == WebSocketState.CONNECTED:
                try:
                    await previous.close(code=1000)
                except Exception:
                    logger.warning("failed to close replaced computer socket")

    async def disconnect(self, computer_id: UUID, ws: WebSocket) -> None:
        if self._sockets.get(computer_id) is ws:
            del self._sockets[computer_id]
            if self._redis is not None:
                await clear_presence(self._redis, computer_id, self.worker_id)

    def is_online(self, computer_id: UUID) -> bool:
        ws = self._sockets.get(computer_id)
        return ws is not None and ws.client_state == WebSocketState.CONNECTED

    async def is_present(self, computer_id: UUID) -> bool:
        if self._redis is not None:
            found = await has_presence(self._redis, computer_id)
            if found is not None:
                return found
        return self.is_online(computer_id)

    async def listed_online(
        self, computer_id: UUID, last_seen_at: datetime | None
    ) -> bool:
        return await self.is_present(computer_id) or is_recent(last_seen_at)

    async def touch(self, computer_id: UUID) -> None:
        if self._redis is not None and self.is_online(computer_id):
            await mark_presence(self._redis, computer_id, self.worker_id)

    async def send_wake(self, computer_id: UUID, payload: dict) -> bool:
        ws = self._sockets.get(computer_id)
        if ws is None or ws.client_state != WebSocketState.CONNECTED:
            return False
        try:
            await ws.send_json(payload)
            return True
        except Exception:
            logger.warning(
                "wake send failed for computer %s — dropping socket",
                computer_id,
                exc_info=True,
            )
            await self.disconnect(computer_id, ws)
            return False
