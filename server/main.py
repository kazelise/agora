from __future__ import annotations

import asyncio
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

import httpx
import redis.asyncio as redis
from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse

from brain.world import WorldMessage
from server import db
from server.computers import ComputerHub
from server.config import Settings, get_settings
from server.hub import RoomHub
from server.jwtutil import check_oauth_state, issue_user_token, make_oauth_state
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
    TokenOut,
    UserOut,
    UserRow,
)
from server.oauth import GitHubClient, HttpGitHub, authorize_url
from server.runtime import computer_ws, router as runtime_router
from server.scheduler import Scheduler, TurnFn, fanout_message, run_subscriber
from server.users import optional_user, require_room_access


def create_app(
    *,
    stub_turns: bool = False,
    settings: Settings | None = None,
    job_launcher: JobLauncher | None = None,
    run_turn: TurnFn | None = None,
    github: GitHubClient | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        cfg = settings or get_settings()
        if cfg.k8s_enabled and not cfg.cluster_token and job_launcher is None and run_turn is None:
            raise RuntimeError("AGORA_K8S_ENABLED requires AGORA_CLUSTER_TOKEN")
        if cfg.auth_enabled and (not cfg.github_client_secret or not cfg.jwt_secret):
            raise RuntimeError(
                "AGORA_GITHUB_CLIENT_ID requires AGORA_GITHUB_CLIENT_SECRET and AGORA_JWT_SECRET"
            )
        pool = await db.create_pool(cfg.database_url)
        await db.migrate(pool)
        publisher = redis.from_url(cfg.redis_url, decode_responses=True)
        subscriber = redis.from_url(cfg.redis_url, decode_responses=True)
        await publisher.ping()

        worker_id = secrets.token_hex(8)
        hub = RoomHub()
        computers = ComputerHub(redis_client=publisher, worker_id=worker_id)
        launcher = job_launcher
        if stub_turns:
            scheduler = Scheduler(
                pool,
                computers=computers,
                redis_client=publisher,
                worker_id=worker_id,
            )
        elif run_turn is not None:
            scheduler = Scheduler(
                pool,
                run_turn=run_turn,
                computers=computers,
                redis_client=publisher,
                worker_id=worker_id,
            )
        elif launcher is not None or cfg.k8s_enabled:
            if launcher is None:
                launcher = K8sJobLauncher.from_settings(cfg)
            scheduler = Scheduler(
                pool,
                run_turn=launcher.run_turn,
                computers=computers,
                redis_client=publisher,
                worker_id=worker_id,
            )
        else:
            from brain.graph import make_turn_fn

            async def on_committed(row: WorldMessage) -> None:
                await fanout_message(hub, publisher, row)

            scheduler = Scheduler(
                pool,
                run_turn=make_turn_fn(pool, publisher, on_committed=on_committed),
                computers=computers,
                redis_client=publisher,
                worker_id=worker_id,
            )

        ready = asyncio.Event()
        stop = asyncio.Event()
        task = asyncio.create_task(
            run_subscriber(
                subscriber, scheduler, ready, stop, hub=hub, computers=computers
            )
        )
        await ready.wait()

        app.state.settings = cfg
        app.state.pool = pool
        app.state.redis = publisher
        app.state.scheduler = scheduler
        app.state.hub = hub
        app.state.computers = computers
        app.state.job_launcher = launcher
        app.state.worker_id = worker_id
        github_client = github
        github_http: httpx.AsyncClient | None = None
        if github_client is None and cfg.auth_enabled:
            github_http = httpx.AsyncClient(timeout=15.0)
            github_client = HttpGitHub(cfg, client=github_http)
        app.state.github = github_client

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
            if github_http is not None:
                await github_http.aclose()
            await subscriber.aclose()
            await publisher.aclose()
            await pool.close()

    app = FastAPI(title="agora", lifespan=lifespan)
    register_routes(app)
    return app


def register_routes(app: FastAPI) -> None:
    @app.get("/auth/github")
    async def auth_github() -> RedirectResponse:
        cfg: Settings = app.state.settings
        if not cfg.auth_enabled:
            raise HTTPException(status_code=404, detail="github oauth is not configured")
        state = make_oauth_state(cfg.jwt_secret)
        return RedirectResponse(authorize_url(cfg, state), status_code=302)

    @app.get("/auth/github/callback", response_model=None)
    async def auth_github_callback(
        code: str | None = None,
        state: str | None = None,
    ) -> TokenOut | RedirectResponse:
        cfg: Settings = app.state.settings
        if not cfg.auth_enabled:
            raise HTTPException(status_code=404, detail="github oauth is not configured")
        if not code or not state or not check_oauth_state(state, cfg.jwt_secret):
            raise HTTPException(status_code=400, detail="invalid oauth state")
        github: GitHubClient | None = app.state.github
        if github is None:
            raise HTTPException(status_code=503, detail="github client missing")
        try:
            gh_token = await github.exchange_code(code)
            profile = await github.fetch_user(gh_token)
        except Exception as exc:
            raise HTTPException(status_code=401, detail="github login failed") from exc
        user = await db.upsert_user(
            app.state.pool,
            profile.github_id,
            profile.login,
            profile.name,
            profile.avatar_url,
        )
        token = issue_user_token(user.id, user.login, cfg.jwt_secret, cfg.jwt_ttl_s)
        if cfg.oauth_success_url:
            sep = "&" if "?" in cfg.oauth_success_url else "?"
            return RedirectResponse(f"{cfg.oauth_success_url}{sep}access_token={token}")
        return TokenOut(access_token=token, user=user.as_out())

    @app.get("/auth/me", response_model=UserOut)
    async def auth_me(user: UserRow | None = Depends(optional_user)) -> UserOut:
        if user is None:
            raise HTTPException(status_code=404, detail="github oauth is not configured")
        return user.as_out()

    @app.post("/computers", response_model=ComputerCreatedOut)
    async def post_computer(
        body: CreateComputerRequest,
        user: UserRow | None = Depends(optional_user),
    ) -> ComputerCreatedOut:
        token = secrets.token_urlsafe(32)
        owner = None if user is None else user.id
        row = await db.create_computer(app.state.pool, body.name, token, created_by=owner)
        return ComputerCreatedOut(id=row.id, name=row.name, token=token)

    @app.get("/computers", response_model=list[ComputerOut])
    async def get_computers(
        user: UserRow | None = Depends(optional_user),
    ) -> list[ComputerOut]:
        hub: ComputerHub = app.state.computers
        owner = None if user is None else user.id
        rows = await db.list_computers(app.state.pool, created_by=owner)
        return [
            ComputerOut(
                id=row.id,
                name=row.name,
                created_at=row.created_at,
                last_seen_at=row.last_seen_at,
                online=await hub.listed_online(row.id, row.last_seen_at),
                created_by=row.created_by,
            )
            for row in rows
        ]

    @app.post("/rooms", response_model=RoomOut)
    async def post_room(
        body: CreateRoomRequest,
        user: UserRow | None = Depends(optional_user),
    ) -> RoomOut:
        owner = None if user is None else user.id
        room = await db.create_room(app.state.pool, body.name, created_by=owner)
        return RoomOut(
            id=room.id,
            name=room.name,
            created_at=room.created_at,
            created_by=room.created_by,
        )

    @app.get("/rooms", response_model=list[RoomOut])
    async def get_rooms(
        user: UserRow | None = Depends(optional_user),
    ) -> list[RoomOut]:
        owner = None if user is None else user.id
        rows = await db.list_rooms(app.state.pool, created_by=owner)
        return [
            RoomOut(
                id=row.id,
                name=row.name,
                created_at=row.created_at,
                created_by=row.created_by,
            )
            for row in rows
        ]

    @app.post("/rooms/{room_id}/participants", response_model=ParticipantOut)
    async def post_participant(
        request: Request,
        room_id: UUID,
        body: CreateParticipantRequest,
        user: UserRow | None = Depends(optional_user),
    ) -> ParticipantOut:
        await require_room_access(request, room_id, user)
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
    async def post_message(
        request: Request,
        room_id: UUID,
        body: CreateMessageRequest,
        user: UserRow | None = Depends(optional_user),
    ) -> MessageOut:
        await require_room_access(request, room_id, user)
        try:
            row = await db.insert_message(app.state.pool, room_id, body.author_id, body.body)
        except db.NotFoundError as exc:
            raise HTTPException(status_code=404, detail=exc.detail) from exc
        await fanout_message(app.state.hub, app.state.redis, row)
        return row.as_out()

    @app.get("/rooms/{room_id}/messages", response_model=MessageListOut)
    async def get_messages(
        request: Request,
        room_id: UUID,
        since_seq: int = Query(default=0, ge=0),
        user: UserRow | None = Depends(optional_user),
    ) -> MessageListOut:
        await require_room_access(request, room_id, user)
        rows = await db.list_messages(app.state.pool, room_id, since_seq=since_seq)
        return MessageListOut(messages=[m.as_out() for m in rows])

    @app.websocket("/ws/rooms/{room_id}")
    async def ws_room(
        websocket: WebSocket,
        room_id: UUID,
        access_token: str | None = Query(default=None),
    ) -> None:
        cfg: Settings = app.state.settings
        user: UserRow | None = None
        if cfg.auth_enabled:
            if not access_token:
                await websocket.close(code=1008)
                return
            from server.jwtutil import JWTError, decode_jwt

            try:
                claims = decode_jwt(access_token, cfg.jwt_secret)
                user = await db.get_user(app.state.pool, UUID(str(claims["sub"])))
            except (JWTError, db.NotFoundError, ValueError):
                await websocket.close(code=1008)
                return
        try:
            room = await db.get_room(app.state.pool, room_id)
        except db.NotFoundError:
            await websocket.close(code=1008)
            return
        if user is not None and room.created_by != user.id:
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
