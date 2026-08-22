from __future__ import annotations

import asyncio
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect

from brain.world import WorldMessage
from server import db
from server.computers import ComputerHub
from server.config import Settings, get_settings
from server.hub import RoomHub
from server.k8s import JobLauncher, K8sJobLauncher
from server.models import (
    ComputerCreatedOut,
    ComputerOut,
    CreateComputerRequest,
    CreateMessageRequest,
    CreateParticipantRequest,
    CreateRoomRequest,
    MessageListOut,
    MessageOut,
    ParticipantOut,
    RoomOut,
)
from server.runtime import computer_ws, router as runtime_router
from server.scheduler import Scheduler, TurnFn, fanout_message, run_subscriber


def create_app(
    *,
    stub_turns: bool = False,
    settings: Settings | None = None,
    job_launcher: JobLauncher | None = None,
    run_turn: TurnFn | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        cfg = settings or get_settings()
        if cfg.k8s_enabled and not cfg.cluster_token and job_launcher is None and run_turn is None:
            raise RuntimeError("AGORA_K8S_ENABLED requires AGORA_CLUSTER_TOKEN")
        pool = await db.create_pool(cfg.database_url)
        await db.migrate(pool)
        publisher = redis.from_url(cfg.redis_url, decode_responses=True)
        subscriber = redis.from_url(cfg.redis_url, decode_responses=True)
        await publisher.ping()

        hub = RoomHub()
        computers = ComputerHub()
        launcher = job_launcher
        if stub_turns:
            scheduler = Scheduler(pool, computers=computers)
        elif run_turn is not None:
            scheduler = Scheduler(pool, run_turn=run_turn, computers=computers)
        elif launcher is not None or cfg.k8s_enabled:
            if launcher is None:
                launcher = K8sJobLauncher.from_settings(cfg)
            scheduler = Scheduler(
                pool, run_turn=launcher.run_turn, computers=computers
            )
        else:
            from brain.graph import make_turn_fn

            async def on_committed(row: WorldMessage) -> None:
                await fanout_message(hub, publisher, row)

            scheduler = Scheduler(
                pool,
                run_turn=make_turn_fn(pool, publisher, on_committed=on_committed),
                computers=computers,
            )

        ready = asyncio.Event()
        stop = asyncio.Event()
        task = asyncio.create_task(run_subscriber(subscriber, scheduler, ready, stop))
        await ready.wait()

        app.state.settings = cfg
        app.state.pool = pool
        app.state.redis = publisher
        app.state.scheduler = scheduler
        app.state.hub = hub
        app.state.computers = computers
        app.state.job_launcher = launcher

        try:
            yield
        finally:
            stop.set()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            if launcher is not None:
                close = getattr(launcher, "aclose", None)
                if close is not None:
                    await close()
            await subscriber.aclose()
            await publisher.aclose()
            await pool.close()

    app = FastAPI(title="agora", lifespan=lifespan)
    register_routes(app)
    return app


def register_routes(app: FastAPI) -> None:
    @app.post("/computers", response_model=ComputerCreatedOut)
    async def post_computer(body: CreateComputerRequest) -> ComputerCreatedOut:
        token = secrets.token_urlsafe(32)
        row = await db.create_computer(app.state.pool, body.name, token)
        return ComputerCreatedOut(id=row.id, name=row.name, token=token)

    @app.get("/computers", response_model=list[ComputerOut])
    async def get_computers() -> list[ComputerOut]:
        hub: ComputerHub = app.state.computers
        rows = await db.list_computers(app.state.pool)
        return [
            ComputerOut(
                id=row.id,
                name=row.name,
                created_at=row.created_at,
                last_seen_at=row.last_seen_at,
                online=hub.listed_online(row.id, row.last_seen_at),
            )
            for row in rows
        ]

    @app.post("/rooms", response_model=RoomOut)
    async def post_room(body: CreateRoomRequest) -> RoomOut:
        room = await db.create_room(app.state.pool, body.name)
        return RoomOut(id=room.id, name=room.name, created_at=room.created_at)

    @app.post("/rooms/{room_id}/participants", response_model=ParticipantOut)
    async def post_participant(room_id: UUID, body: CreateParticipantRequest) -> ParticipantOut:
        try:
            row = await db.add_participant(
                app.state.pool,
                room_id,
                body.kind,
                body.name,
                body.persona,
                body.computer_id,
            )
        except db.NotFoundError as exc:
            raise HTTPException(status_code=404, detail=exc.detail) from exc
        return ParticipantOut(
            id=row.id,
            room_id=row.room_id,
            kind=row.kind,
            name=row.name,
            persona=row.persona,
            computer_id=row.computer_id,
            created_at=row.created_at,
        )

    @app.post("/rooms/{room_id}/messages", response_model=MessageOut)
    async def post_message(room_id: UUID, body: CreateMessageRequest) -> MessageOut:
        try:
            row = await db.insert_message(app.state.pool, room_id, body.author_id, body.body)
        except db.NotFoundError as exc:
            raise HTTPException(status_code=404, detail=exc.detail) from exc
        await fanout_message(app.state.hub, app.state.redis, row)
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

    @app.websocket("/ws/computers/{computer_id}")
    async def ws_computer_route(
        websocket: WebSocket,
        computer_id: UUID,
        token: str | None = Query(default=None),
    ) -> None:
        if not token:
            await websocket.close(code=1008)
            return
        await computer_ws(websocket, computer_id, token)

    app.include_router(runtime_router)


app = create_app()
