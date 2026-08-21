"""Redis seen-cursor: compose-window high-water for an (agent, room).

Freshness HOLD itself compares Postgres last_seq to the in-state seen_seq
that was shown to the model. This module exists so *other* processes
(future BYOA daemon) can ask \"what has this agent already been shown?\"
without sharing conversation_reads (that column is the inbox cursor).

Fail-open: every Redis error is logged and treated as fresh / unset.
A missed HOLD is a possible collision — the failure we try to reduce —
never a blocked turn or a failed INSERT.
"""

from __future__ import annotations

import logging
from uuid import UUID

import redis.asyncio as redis

logger = logging.getLogger("agora.seen")

TTL_SECONDS = 600
KEY_PREFIX = "agora:seen"

# Monotonic SET + TTL refresh. Two concurrent record_seen calls converge
# on the larger seq; a smaller write never regresses the cursor.
_MONOTONIC_SET = """
local cur = tonumber(redis.call('GET', KEYS[1])) or 0
local newv = tonumber(ARGV[1]) or 0
local ttl = tonumber(ARGV[2])
if newv > cur then
  redis.call('SET', KEYS[1], newv, 'EX', ttl)
  return newv
end
if cur > 0 then
  redis.call('EXPIRE', KEYS[1], ttl)
end
return cur
"""


def _key(agent_id: UUID, room_id: UUID) -> str:
    return f"{KEY_PREFIX}:{agent_id}:{room_id}"


async def record_seen(
    client: redis.Redis,
    agent_id: UUID,
    room_id: UUID,
    seq: int,
) -> None:
    if seq <= 0:
        return
    try:
        await client.eval(_MONOTONIC_SET, 1, _key(agent_id, room_id), seq, TTL_SECONDS)
    except Exception:
        logger.warning(
            "record_seen(%s, %s, %s) failed — fail-open",
            agent_id,
            room_id,
            seq,
            exc_info=True,
        )


async def get_seen(
    client: redis.Redis,
    agent_id: UUID,
    room_id: UUID,
) -> int | None:
    """High-water seq shown to this agent, or None if unset / Redis failed."""
    try:
        raw = await client.get(_key(agent_id, room_id))
    except Exception:
        logger.warning(
            "get_seen(%s, %s) failed — fail-open",
            agent_id,
            room_id,
            exc_info=True,
        )
        return None
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


async def is_fresh(
    client: redis.Redis,
    agent_id: UUID,
    room_id: UUID,
    latest_seq: int,
) -> bool:
    """True if latest_seq is not ahead of the Redis cursor.

    Fail-open: Redis errors and a missing key both return True (do not HOLD).
    The turn hot path should prefer in-state seen_seq vs Postgres last_seq;
    this helper is for out-of-process readers.
    """
    seen = await get_seen(client, agent_id, room_id)
    if seen is None:
        return True
    return latest_seq <= seen
