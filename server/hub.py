from __future__ import annotations

import logging
from collections import defaultdict
from uuid import UUID

from fastapi import WebSocket
from starlette.websockets import WebSocketState

logger = logging.getLogger("agora.hub")


class RoomHub:
    def __init__(self) -> None:
        self._clients: dict[UUID, set[WebSocket]] = defaultdict(set)

    async def connect(self, room_id: UUID, ws: WebSocket) -> None:
        await ws.accept()
        self._clients[room_id].add(ws)

    def disconnect(self, room_id: UUID, ws: WebSocket) -> None:
        self._clients[room_id].discard(ws)
        if not self._clients[room_id]:
            del self._clients[room_id]

    async def broadcast(self, room_id: UUID, payload: dict) -> None:
        dead: list[WebSocket] = []
        for ws in list(self._clients[room_id]):
            if ws.client_state != WebSocketState.CONNECTED:
                dead.append(ws)
                continue
            try:
                await ws.send_json(payload)
            except Exception:
                logger.warning("websocket send failed; dropping client")
                dead.append(ws)
        for ws in dead:
            self.disconnect(room_id, ws)
