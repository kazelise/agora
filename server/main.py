from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect

from server import db
from server.config import get_settings
from server.hub import RoomHub
from server.models import (
    CreateMessageRequest,
    CreateParticipantRequest,
    CreateRoomRequest,
    MessageListOut,
    MessageOut,
    ParticipantOut,
    RoomOut,
)
from server.scheduler import Scheduler, publish_wake, run_subscriber


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    pool = await db.create_pool(settings.database_url)
    await db.migrate(pool)
    publisher = redis.from_url(settings.redis_url, decode_responses=True)
    subscriber = redis.from_url(settings.redis_url, decode_responses=True)
    await publisher.ping()

    scheduler = Scheduler(pool)
    hub = RoomHub()
    ready = asyncio.Event()
    stop = asyncio.Event()
    task = asyncio.create_task(run_subscriber(subscriber, scheduler, ready, stop))
    await ready.wait()

    app.state.settings = settings
    app.state.pool = pool
    app.state.redis = publisher
    app.state.scheduler = scheduler
    app.state.hub = hub

    try:
        yield
    finally:
        stop.set()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await subscriber.aclose()
        await publisher.aclose()
        await pool.close()


def create_app() -> FastAPI:
    app = FastAPI(title="agora", lifespan=lifespan)
    register_routes(app)
    return app


def register_routes(app: FastAPI) -> None:
    @app.post("/rooms", response_model=RoomOut)
    async def post_room(body: CreateRoomRequest) -> RoomOut:
        room = await db.create_room(app.state.pool, body.name)
        return RoomOut(id=room.id, name=room.name, created_at=room.created_at)

    @app.post("/rooms/{room_id}/participants", response_model=ParticipantOut)
    async def post_participant(room_id: UUID, body: CreateParticipantRequest) -> ParticipantOut:
        try:
            row = await db.add_participant(
                app.state.pool, room_id, body.kind, body.name, body.persona
            )
        except db.NotFoundError as exc:
            raise HTTPException(status_code=404, detail=exc.detail) from exc
        return ParticipantOut(
            id=row.id,
            room_id=row.room_id,
            kind=row.kind,
            name=row.name,
            persona=row.persona,
            created_at=row.created_at,
        )

    @app.post("/rooms/{room_id}/messages", response_model=MessageOut)
    async def post_message(room_id: UUID, body: CreateMessageRequest) -> MessageOut:
        try:
            row = await db.insert_message(app.state.pool, room_id, body.author_id, body.body)
        except db.NotFoundError as exc:
            raise HTTPException(status_code=404, detail=exc.detail) from exc
        await app.state.hub.broadcast(room_id, row.as_ws())
        await publish_wake(app.state.redis, room_id, body.author_id, row.seq)
        return row.as_out()

    @app.get("/rooms/{room_id}/messages", response_model=MessageListOut)
    async def get_messages(
        room_id: UUID,
        since_seq: int = Query(default=0, ge=0),
    ) -> MessageListOut:
        try:
            rows = await db.list_messages(app.state.pool, room_id, since_seq=since_seq)
        except db.NotFoundError as exc:
            raise HTTPException(status_code=404, detail=exc.detail) from exc
        return MessageListOut(messages=[m.as_out() for m in rows])

    @app.websocket("/ws/rooms/{room_id}")
    async def ws_room(websocket: WebSocket, room_id: UUID) -> None:
        try:
            await db.get_room(app.state.pool, room_id)
        except db.NotFoundError:
            await websocket.close(code=1008)
            return
        hub: RoomHub = app.state.hub
        await hub.connect(room_id, websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            hub.disconnect(room_id, websocket)


app = create_app()
