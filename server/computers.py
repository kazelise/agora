"""Computer registry: pairing tokens and in-process presence.

Presence is a dict of connected websockets. That is a deliberate
single-instance simplification — a multi-worker deploy would need
Redis-backed presence. last_seen_at is the heartbeat fallback for
GET /computers (online if the socket is up, or a ping landed within
ONLINE_GRACE_S). Wake routing itself is stricter: only a live socket
counts as connected; otherwise the agent is sleeping.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from uuid import UUID

from fastapi import WebSocket
from starlette.websockets import WebSocketState

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
    def __init__(self) -> None:
        self._sockets: dict[UUID, WebSocket] = {}

    async def connect(self, computer_id: UUID, ws: WebSocket) -> None:
        await ws.accept()
        previous = self._sockets.get(computer_id)
        self._sockets[computer_id] = ws
        if previous is not None and previous is not ws:
            if previous.client_state == WebSocketState.CONNECTED:
                try:
                    await previous.close(code=1000)
                except Exception:
                    logger.warning("failed to close replaced computer socket")

    def disconnect(self, computer_id: UUID, ws: WebSocket) -> None:
        if self._sockets.get(computer_id) is ws:
            del self._sockets[computer_id]

    def is_online(self, computer_id: UUID) -> bool:
        ws = self._sockets.get(computer_id)
        return ws is not None and ws.client_state == WebSocketState.CONNECTED

    def listed_online(self, computer_id: UUID, last_seen_at: datetime | None) -> bool:
        return self.is_online(computer_id) or is_recent(last_seen_at)

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
            self.disconnect(computer_id, ws)
            return False
