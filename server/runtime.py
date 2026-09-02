"""Token-authed runtime API for a host (paired computer or cluster Job).

The host holds the model key and talks to these endpoints. A computer
token may only act for agents on that computer; the cluster token may
only act for unhosted (cloud) agents. The server never sees a provider
key. Usage lands in the same llm_calls ledger (purpose + tokens + model).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from brain.wake_hints import consume_called_on
from server import db
from server.auth import hosted_agent, require_host
from server.computers import ComputerHub
from server.models import MessageOut
from server.runtime_types import Host
from server.scheduler import fanout_message  # noqa: F401 (re-exported for tests)

logger = logging.getLogger("agora.runtime")

router = APIRouter(prefix="/runtime", tags=["runtime"])


class RuntimeReplyRequest(BaseModel):
    agent_id: UUID
    room_id: UUID
    body: str = Field(min_length=1)
    not_after_seq: int


class RuntimeClaimRequest(BaseModel):
    agent_id: UUID
    room_id: UUID
    task_key: str = Field(min_length=1)


class RuntimeLlmCallRequest(BaseModel):
    agent_id: UUID
    room_id: UUID
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    purpose: str


class RuntimeLastReadRequest(BaseModel):
    agent_id: UUID
    room_id: UUID
    last_read_seq: int


class RuntimeDecisionRequest(BaseModel):
    agent_id: UUID
    room_id: UUID
    trigger_seq: int
    action: Literal["call_on", "say", "silence"]
    target_id: UUID | None = None


class RuntimeDecisionOut(BaseModel):
    status: Literal["won", "already_decided"]
    action: str
    target_id: UUID | None = None


class RuntimeParticipantOut(BaseModel):
    id: UUID
    name: str
    kind: str
    persona: str | None = None
    role: str = "member"


class RuntimeMessageOut(BaseModel):
    id: UUID
    room_id: UUID
    author_id: UUID
    author_name: str
    body: str
    seq: int
    created_at: datetime


class TurnContextOut(BaseModel):
    agent_id: UUID
    agent_name: str
    persona: str
    last_read_seq: int
    seen_seq: int
    participants: list[RuntimeParticipantOut]
    inbox: list[RuntimeMessageOut]
    agent_only_stretch: int = 0
    room_mode: str = "open"
    agent_kind: str = "agent"
    agent_role: str = "member"
    called_on: bool = False
    called_on_seq: int | None = None


async def named_since(
    request: Request, room_id: UUID, since_seq: int
) -> list[RuntimeMessageOut]:
    pool = request.app.state.pool
    rows = await db.list_messages(pool, room_id, since_seq=since_seq)
    people = await db.list_participants(pool, room_id)
    names = {p.id: p.name for p in people}
    return [
        RuntimeMessageOut(
            id=row.id,
            room_id=row.room_id,
            author_id=row.author_id,
            author_name=names.get(row.author_id, "?"),
            body=row.body,
            seq=row.seq,
            created_at=row.created_at,
        )
        for row in rows
    ]


@router.get("/turn-context", response_model=TurnContextOut)
async def turn_context(
    request: Request,
    agent_id: UUID,
    room_id: UUID,
    host: Host = Depends(require_host),
) -> TurnContextOut:
    agent = await hosted_agent(request, host, agent_id, room_id)
    last_read = await db.get_last_read(request.app.state.pool, agent_id, room_id)
    people = await db.list_participants(request.app.state.pool, room_id)
    inbox = await named_since(request, room_id, last_read)
    seen_seq = max((m.seq for m in inbox), default=last_read)
    try:
        room = await db.get_room(request.app.state.pool, room_id)
        room_mode = room.mode
    except db.NotFoundError:
        room_mode = "open"
    hint = await consume_called_on(request.app.state.redis, agent_id, room_id)
    return TurnContextOut(
        agent_id=agent.id,
        agent_name=agent.name,
        persona=agent.persona or "",
        last_read_seq=last_read,
        seen_seq=seen_seq,
        participants=[
            RuntimeParticipantOut(
                id=p.id, name=p.name, kind=p.kind, persona=p.persona, role=p.role
            )
            for p in people
        ],
        inbox=inbox,
        agent_only_stretch=await db.count_agent_only_stretch(
            request.app.state.pool, room_id
        ),
        room_mode=room_mode,
        agent_kind=agent.kind,
        agent_role=agent.role,
        called_on=hint is not None,
        called_on_seq=hint,
    )


@router.get("/room-seq")
async def room_seq(
    request: Request,
    agent_id: UUID,
    room_id: UUID,
    host: Host = Depends(require_host),
) -> dict[str, int]:
    await hosted_agent(request, host, agent_id, room_id)
    try:
        last_seq = await db.get_room_last_seq(request.app.state.pool, room_id)
    except db.NotFoundError as exc:
        raise HTTPException(status_code=404, detail=exc.detail) from exc
    return {"last_seq": last_seq}


@router.get("/messages", response_model=list[RuntimeMessageOut])
async def messages_since(
    request: Request,
    agent_id: UUID,
    room_id: UUID,
    host: Host = Depends(require_host),
    since_seq: int = Query(default=0, ge=0),
) -> list[RuntimeMessageOut]:
    await hosted_agent(request, host, agent_id, room_id)
    try:
        return await named_since(request, room_id, since_seq)
    except db.NotFoundError as exc:
        raise HTTPException(status_code=404, detail=exc.detail) from exc


@router.post("/reply", response_model=MessageOut)
async def reply(
    request: Request,
    body: RuntimeReplyRequest,
    host: Host = Depends(require_host),
) -> MessageOut:
    await hosted_agent(request, host, body.agent_id, body.room_id)
    try:
        row = await db.insert_message(
            request.app.state.pool,
            body.room_id,
            body.agent_id,
            body.body,
            not_after_seq=body.not_after_seq,
        )
    except db.StaleWriteError as exc:
        newer = await named_since(request, body.room_id, body.not_after_seq)
        raise HTTPException(
            status_code=409,
            detail={
                "error": "stale_write",
                "last_seq": exc.last_seq,
                "messages": [m.model_dump(mode="json") for m in newer],
            },
        ) from exc
    except db.DuplicateReplyError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "duplicate_reply",
                "peer_seq": exc.peer_seq,
            },
        ) from exc
    except db.NotFoundError as exc:
        raise HTTPException(status_code=404, detail=exc.detail) from exc
    await request.app.state.fanout_with_stall_reset(
        request.app.state.hub, request.app.state.redis, row
    )
    return row.as_out()


@router.post("/claim")
async def claim(
    request: Request,
    body: RuntimeClaimRequest,
    host: Host = Depends(require_host),
) -> dict[str, bool]:
    await hosted_agent(request, host, body.agent_id, body.room_id)
    won = await db.try_claim(
        request.app.state.pool, body.room_id, body.task_key, body.agent_id
    )
    return {"won": won}


@router.post("/release-claim")
async def release_claim(
    request: Request,
    body: RuntimeClaimRequest,
    host: Host = Depends(require_host),
) -> dict[str, bool]:
    await hosted_agent(request, host, body.agent_id, body.room_id)
    released = await db.release_claim(
        request.app.state.pool, body.room_id, body.task_key, body.agent_id
    )
    return {"released": released}


@router.post("/llm-call")
async def llm_call(
    request: Request,
    body: RuntimeLlmCallRequest,
    host: Host = Depends(require_host),
) -> dict[str, str]:
    await hosted_agent(request, host, body.agent_id, body.room_id)
    await db.insert_llm_call(
        request.app.state.pool,
        body.agent_id,
        body.room_id,
        body.model,
        body.prompt_tokens,
        body.completion_tokens,
        body.purpose,
    )
    return {"status": "ok"}


@router.get("/authored-since")
async def authored_since(
    request: Request,
    agent_id: UUID,
    room_id: UUID,
    host: Host = Depends(require_host),
    since_seq: int = Query(default=0, ge=0),
) -> dict[str, bool]:
    await hosted_agent(request, host, agent_id, room_id)
    found = await db.has_authored_since(
        request.app.state.pool, agent_id, room_id, since_seq
    )
    return {"authored": found}


@router.get("/last-read")
async def get_last_read(
    request: Request,
    agent_id: UUID,
    room_id: UUID,
    host: Host = Depends(require_host),
) -> dict[str, int]:
    await hosted_agent(request, host, agent_id, room_id)
    value = await db.get_last_read(request.app.state.pool, agent_id, room_id)
    return {"last_read_seq": value}


@router.put("/last-read")
async def put_last_read(
    request: Request,
    body: RuntimeLastReadRequest,
    host: Host = Depends(require_host),
) -> dict[str, str]:
    await hosted_agent(request, host, body.agent_id, body.room_id)
    await db.set_last_read(
        request.app.state.pool, body.agent_id, body.room_id, body.last_read_seq
    )
    return {"status": "ok"}


@router.post("/decision", response_model=RuntimeDecisionOut)
async def decision(
    request: Request,
    body: RuntimeDecisionRequest,
    host: Host = Depends(require_host),
) -> RuntimeDecisionOut:
    """Record a moderator decision. The SERVER owns any call_on wake.

    BYOA must not reach into server lanes: this endpoint writes the
    unique (room, trigger_seq) row and, on a freshly won call_on,
    dispatches the target through Scheduler.wake_one.
    """
    agent = await hosted_agent(request, host, body.agent_id, body.room_id)
    if agent.role != "moderator":
        raise HTTPException(status_code=403, detail="only the moderator can record a decision")
    try:
        room = await db.get_room(request.app.state.pool, body.room_id)
    except db.NotFoundError as exc:
        raise HTTPException(status_code=404, detail=exc.detail) from exc
    if room.mode != "moderated":
        # 409 is reserved for freshness/dup on the daemon client
        # (_raise_for_status treats any 409 as StaleWrite).
        raise HTTPException(status_code=400, detail="room is not moderated")
    if body.action == "call_on":
        if body.target_id is None:
            raise HTTPException(status_code=400, detail="call_on requires target_id")
        try:
            target = await db.get_participant(request.app.state.pool, body.target_id)
        except db.NotFoundError as exc:
            raise HTTPException(status_code=404, detail=exc.detail) from exc
        if target.room_id != body.room_id or target.kind != "agent":
            raise HTTPException(
                status_code=400, detail="call_on target must be an agent in this room"
            )
        if target.id == agent.id:
            raise HTTPException(status_code=400, detail="call_on cannot target the moderator")
    try:
        status, row = await db.record_decision(
            request.app.state.pool,
            body.room_id,
            body.agent_id,
            body.trigger_seq,
            body.action,
            body.target_id,
        )
    except db.InvalidTriggerSeqError as exc:
        raise HTTPException(status_code=422, detail=exc.detail) from exc
    except db.NotFoundError as exc:
        raise HTTPException(status_code=404, detail=exc.detail) from exc
    if status == "won" and row.action == "call_on" and row.target_id is not None:
        try:
            await request.app.state.scheduler.wake_one(
                body.room_id,
                row.target_id,
                called_on_seq=row.trigger_seq,
            )
        except Exception:
            # Row is committed; _undelivered_call_on redelivers on the
            # next stall nudge if the target's last_read still trails.
            logger.warning(
                "on_call_on wake failed room=%s target=%s — fail-open",
                body.room_id,
                row.target_id,
                exc_info=True,
            )
    return RuntimeDecisionOut(
        status=status, action=row.action, target_id=row.target_id
    )


async def computer_ws(websocket: WebSocket, computer_id: UUID, token: str) -> None:
    app = websocket.app
    if not await db.computer_token_matches(app.state.pool, computer_id, token):
        await websocket.close(code=1008)
        return
    hub: ComputerHub = app.state.computers
    await hub.connect(computer_id, websocket)
    await db.touch_computer(app.state.pool, computer_id)
    try:
        while True:
            raw = await websocket.receive_json()
            if isinstance(raw, dict) and raw.get("type") == "ping":
                await db.touch_computer(app.state.pool, computer_id)
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        hub.disconnect(computer_id, websocket)
    except RuntimeError:
        # A reconnect replaced this socket and the hub closed it; reading
        # from the closed socket raises RuntimeError. Presence still
        # points at the NEW socket, so there is nothing to disconnect —
        # the old loop just retires quietly instead of raising into the
        # ASGI stack.
        hub.disconnect(computer_id, websocket)
    except Exception:
        hub.disconnect(computer_id, websocket)
        raise
