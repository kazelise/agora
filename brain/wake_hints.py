"""Per-wake `called_on` hint: a call_on is a protocol fact, not inbox mail.

A member woken by the moderator's decide(call_on) must skip triage.
The hint carries the decision's trigger_seq so a later turn can tell
whether this agent has already spoken since being called.

It is parked in Redis only for hosts that cannot receive the in-process
value — BYOA daemons and K8s Jobs. In-process lanes use Scheduler's
dict and never write or read this key.

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


def hint_key(agent_id: UUID, room_id: UUID) -> str:
    return f"{KEY_PREFIX}:{agent_id}:{room_id}"


async def set_called_on(
    redis: object, agent_id: UUID, room_id: UUID, trigger_seq: int
) -> None:
    # Max-merge: a later call_on must not be erased by an earlier one.
    # Only a consume after the turn, or TTL, drops the hint.
    if redis is None or trigger_seq <= 0:
        return
    key = hint_key(agent_id, room_id)
    try:
        raw = await redis.get(key)  # type: ignore[union-attr]
        current = int(raw) if raw is not None else 0
        if trigger_seq > current:
            await redis.set(key, str(trigger_seq), ex=TTL_SECONDS)  # type: ignore[union-attr]
        elif current > 0:
            await redis.expire(key, TTL_SECONDS)  # type: ignore[union-attr]
    except Exception:
        logger.warning("called_on hint write failed — fail-open", exc_info=True)


async def consume_called_on(
    redis: object, agent_id: UUID, room_id: UUID
) -> int | None:
    """Return the parked trigger_seq, or None if unset / Redis failed."""
    if redis is None:
        return None
    try:
        value = await redis.getdel(hint_key(agent_id, room_id))  # type: ignore[union-attr]
    except Exception:
        logger.warning("called_on hint read failed — fail-open", exc_info=True)
        return None
    if value is None:
        return None
    try:
        seq = int(value)
    except (TypeError, ValueError):
        return None
    return seq if seq > 0 else None
