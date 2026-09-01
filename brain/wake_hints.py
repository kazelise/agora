"""Per-wake `called_on` hint: a call_on is a protocol fact, not inbox mail.

A member woken by the moderator's decide(call_on) must skip triage. The
flag rides the wake (WS payload / in-process lane) and is also parked
in Redis so a host that only calls `Brain.run(agent, room)` — K8s Job,
a daemon that missed the frame field — still sees it on load_turn.

Fail-open: Redis down means the hint is lost and the member runs a
normal turn (triage included). That is a weaker skip, not a blocked
room — the same class of miss as a dropped pub/sub wake.
"""

from __future__ import annotations

import logging
from uuid import UUID

logger = logging.getLogger("agora.wake_hints")

# Fail-open: an expired hint is a normal triage turn, never a stuck
# room. 600s covers a cold K8s Job reaching /runtime/turn-context.
TTL_SECONDS = 600
KEY_PREFIX = "agora:called_on"


def _key(agent_id: UUID, room_id: UUID) -> str:
    return f"{KEY_PREFIX}:{agent_id}:{room_id}"


async def set_called_on(redis: object, agent_id: UUID, room_id: UUID, called_on: bool) -> None:
    # False must not erase a pending True (OR-merge with the in-process
    # flag). Only a consume after the turn, or TTL, drops the hint.
    if redis is None or not called_on:
        return
    key = _key(agent_id, room_id)
    try:
        await redis.set(key, "1", ex=TTL_SECONDS)  # type: ignore[union-attr]
    except Exception:
        logger.warning("called_on hint write failed — fail-open", exc_info=True)


async def consume_called_on(redis: object, agent_id: UUID, room_id: UUID) -> bool:
    if redis is None:
        return False
    try:
        value = await redis.getdel(_key(agent_id, room_id))  # type: ignore[union-attr]
        return bool(value)
    except Exception:
        logger.warning("called_on hint read failed — fail-open", exc_info=True)
        return False
