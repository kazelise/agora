"""Human admission: Bearer JWT when GitHub OAuth is configured.

Auth off (empty AGORA_GITHUB_CLIENT_ID) is a first-class mode, not a
backdoor: tests and curl demos stay the Phase 4b/4c path. Auth on is
fail-closed. The two host tokens never satisfy these dependencies.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import HTTPException, Request

from server import db
from server.jwtutil import JWTError, decode_jwt
from server.models import RoomRow, UserRow


def jwt_from_request(request: Request) -> str | None:
    header = request.headers.get("authorization")
    if header and header.startswith("Bearer "):
        token = header.removeprefix("Bearer ").strip()
        if token:
            return token
    query = request.query_params.get("access_token")
    if query:
        return query
    return None


async def optional_user(request: Request) -> UserRow | None:
    settings = request.app.state.settings
    if not settings.auth_enabled:
        return None
    token = jwt_from_request(request)
    if not token:
        raise HTTPException(status_code=401, detail="missing access token")
    try:
        claims = decode_jwt(token, settings.jwt_secret)
    except JWTError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    if claims.get("iss") != "agora" or not claims.get("sub"):
        raise HTTPException(status_code=401, detail="invalid token")
    try:
        return await db.get_user(request.app.state.pool, UUID(str(claims["sub"])))
    except db.NotFoundError as exc:
        raise HTTPException(status_code=401, detail="user not found") from exc


async def require_room_access(
    request: Request, room_id: UUID, user: UserRow | None
) -> RoomRow:
    """404 if missing; 403 if auth is on and this user does not own it."""
    try:
        room = await db.get_room(request.app.state.pool, room_id)
    except db.NotFoundError as exc:
        raise HTTPException(status_code=404, detail=exc.detail) from exc
    if user is None:
        return room
    if room.created_by != user.id:
        raise HTTPException(status_code=403, detail="not your room")
    return room
