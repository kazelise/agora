"""Redis bus for multi-worker coordination.

Wakes, room broadcasts, host wakes, presence, and per-agent lanes all
share Redis. Every one of these is a coordination signal: a Redis error
misses a turn or a frame, it does not block INSERT. Admission (OAuth)
does not live here.

A single process still uses this path so the one-worker and N-worker
shapes stay the same. Tests that spin two apps against one Redis are
the proof.
"""

from __future__ import annotations

import logging
from uuid import UUID

import redis.asyncio as redis

logger = logging.getLogger("agora.bus")

WAKE_CHANNEL = "agora:wake"
MESSAGE_CHANNEL = "agora:messages"
HOST_WAKE_CHANNEL = "agora:host-wake"

PRESENCE_PREFIX = "agora:presence"
LANE_PREFIX = "agora:lane"
DIRTY_PREFIX = "agora:lane-dirty"
DISPATCH_PREFIX = "agora:dispatch"

PRESENCE_TTL_S = 45
LANE_TTL_S = 300
DISPATCH_TTL_S = 120

_RELEASE_IF_OWNER = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  redis.call('DEL', KEYS[1])
  return 1
end
return 0
"""

# After a turn: dirty means rerun-once; otherwise the owner drops the lane.
_LANE_AFTER_TURN = """
if redis.call('GET', KEYS[2]) then
  redis.call('DEL', KEYS[2])
  return 1
end
if redis.call('GET', KEYS[1]) == ARGV[1] then
  redis.call('DEL', KEYS[1])
end
return 0
"""


def presence_key(computer_id: UUID) -> str:
    return f"{PRESENCE_PREFIX}:{computer_id}"


def lane_keys(agent_id: UUID) -> tuple[str, str]:
    return f"{LANE_PREFIX}:{agent_id}", f"{DIRTY_PREFIX}:{agent_id}"


def dispatch_key(agent_id: UUID, seq: int) -> str:
    return f"{DISPATCH_PREFIX}:{agent_id}:{seq}"


async def try_claim(client: redis.Redis, key: str, owner: str, ttl: int) -> bool:
    """SET NX. Redis errors are a miss (fail-open), not a crash."""
    try:
        return bool(await client.set(key, owner, nx=True, ex=ttl))
    except Exception:
        logger.warning("claim %s failed — fail-open miss", key, exc_info=True)
        return False


async def mark_presence(client: redis.Redis, computer_id: UUID, worker_id: str) -> None:
    try:
        await client.set(presence_key(computer_id), worker_id, ex=PRESENCE_TTL_S)
    except Exception:
        logger.warning(
            "presence set %s failed — fail-open", computer_id, exc_info=True
        )


async def clear_presence(
    client: redis.Redis, computer_id: UUID, worker_id: str
) -> None:
    try:
        await client.eval(
            _RELEASE_IF_OWNER, 1, presence_key(computer_id), worker_id
        )
    except Exception:
        logger.warning(
            "presence clear %s failed — fail-open", computer_id, exc_info=True
        )


async def has_presence(client: redis.Redis, computer_id: UUID) -> bool | None:
    """True/False from Redis, or None if Redis failed (caller falls back)."""
    try:
        return bool(await client.exists(presence_key(computer_id)))
    except Exception:
        logger.warning(
            "presence get %s failed — fail-open", computer_id, exc_info=True
        )
        return None


async def acquire_lane(client: redis.Redis, agent_id: UUID, worker_id: str) -> bool:
    key, _ = lane_keys(agent_id)
    return await try_claim(client, key, worker_id, LANE_TTL_S)


async def release_lane(
    client: redis.Redis, agent_id: UUID, worker_id: str
) -> None:
    key, _ = lane_keys(agent_id)
    try:
        await client.eval(_RELEASE_IF_OWNER, 1, key, worker_id)
    except Exception:
        logger.warning(
            "lane release %s failed — fail-open", agent_id, exc_info=True
        )


async def mark_lane_dirty(client: redis.Redis, agent_id: UUID) -> None:
    _, dirty = lane_keys(agent_id)
    try:
        await client.set(dirty, "1", ex=LANE_TTL_S)
    except Exception:
        logger.warning("lane dirty %s failed — fail-open", agent_id, exc_info=True)


async def refresh_lane(client: redis.Redis, agent_id: UUID) -> None:
    key, _ = lane_keys(agent_id)
    try:
        await client.expire(key, LANE_TTL_S)
    except Exception:
        logger.warning("lane expire %s failed — fail-open", agent_id, exc_info=True)


async def lane_should_rerun(
    client: redis.Redis, agent_id: UUID, worker_id: str
) -> bool:
    lane, dirty = lane_keys(agent_id)
    try:
        raw = await client.eval(_LANE_AFTER_TURN, 2, lane, dirty, worker_id)
    except Exception:
        logger.warning(
            "lane finish %s failed — fail-open stop", agent_id, exc_info=True
        )
        return False
    return int(raw or 0) == 1
