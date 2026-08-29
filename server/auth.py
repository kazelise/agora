"""Runtime principals: a paired Computer, or the cluster Job host.

A cluster token is a server-issued service credential. It is not a
Computer row and cannot ride a BYOA agent's /runtime/* path. Jobs use
it to act for agents whose computer_id is NULL — the cloud lane.
"""

from __future__ import annotations

import hmac
from uuid import UUID

from fastapi import HTTPException, Request

from server import db
from server.computers import hash_token
from server.models import ComputerRow, ParticipantRow
from server.runtime_types import ClusterHost, Host


def tokens_equal(left: str, right: str) -> bool:
    """Constant-time compare that stays same-length via sha256."""
    return hmac.compare_digest(hash_token(left), hash_token(right))


def bearer_token(authorization: str | None) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing computer token")
    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status_code=401, detail="missing computer token")
    return token


async def require_host(request: Request) -> Host:
    token = bearer_token(request.headers.get("authorization"))
    settings = getattr(request.app.state, "settings", None)
    cluster = getattr(settings, "cluster_token", "") if settings is not None else ""
    if cluster and tokens_equal(token, cluster):
        return ClusterHost()
    computer = await db.get_computer_by_token(request.app.state.pool, token)
    if computer is None:
        raise HTTPException(status_code=401, detail="invalid computer token")
    return computer


async def hosted_agent(
    request: Request,
    host: Host,
    agent_id: UUID,
    room_id: UUID | None = None,
) -> ParticipantRow:
    try:
        agent = await db.get_participant(request.app.state.pool, agent_id)
    except db.NotFoundError as exc:
        raise HTTPException(status_code=404, detail=exc.detail) from exc
    if isinstance(host, ClusterHost):
        if agent.computer_id is not None:
            raise HTTPException(
                status_code=403,
                detail="cluster host cannot run a BYOA agent",
            )
    else:
        computer: ComputerRow = host
        if agent.computer_id != computer.id:
            raise HTTPException(
                status_code=403, detail="agent is not hosted on this computer"
            )
    if room_id is not None and agent.room_id != room_id:
        raise HTTPException(status_code=403, detail="agent is not in this room")
    return agent
