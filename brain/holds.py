"""Redis hold tokens: a HOLD must be acknowledged, never pre-emptively skipped.

Cumora's lesson (their docs/COORDINATION.md §5d): an unconditional override
flag gets adopted pre-emptively by cost-optimizing brains, and the gate
silently stops existing. The flag is only honored when the server has
actually SHOWN this agent a HOLD for the same scope, and consuming the
token is atomic. The token is seq-bound: it acknowledges exactly the room
state the agent was shown, so a stale ack cannot skip messages the agent
has never seen.

Fail-closed: coordination signal, not a correctness invariant, so a Redis
outage must not UPGRADE the agent's authority — but the flag itself only
ever exists to bypass a gate this module owns. If Redis is down we cannot
verify a token, so the ack is refused and the turn keeps its ordinary
HOLD + re-decide path (bounded by MAX_HOLDS, worst case held_exhausted).
That is one missed override, not a blocked conversation. Honoring the flag
on unverifiable state is what §5d exists to prevent.
"""

from __future__ import annotations

import logging
from uuid import UUID

import redis.asyncio as redis

logger = logging.getLogger("agora.holds")

TTL_SECONDS = 120
KEY_PREFIX = "agora:hold"

# Atomic consume: the token exists exactly once; the second reader gets none.
# Returns the seq the token is bound to (0 = unbound), or nil if absent.
_CONSUME_SCRIPT = """
local v = redis.call('GET', KEYS[1])
if v then
  redis.call('DEL', KEYS[1])
end
return v
"""


def _key(agent_id: UUID, room_id: UUID) -> str:
    return f"{KEY_PREFIX}:{agent_id}:{room_id}"


async def record_hold(
    client: redis.Redis,
    agent_id: UUID,
    room_id: UUID,
    seq: int,
) -> None:
    """Arm (or refresh) the hold token for this (agent, room) scope.

    `seq` is the max peer seq the HELD envelope showed — the token only
    acknowledges room state up to that point. A successful send clears
    any lingering token (see clear_hold).
    """
    if seq <= 0:
        return
    try:
        await client.set(_key(agent_id, room_id), str(seq), ex=TTL_SECONDS)
    except Exception:
        logger.warning(
            "record_hold(%s, %s, %s) failed — fail-open",
            agent_id,
            room_id,
            seq,
            exc_info=True,
        )


async def consume_hold(
    client: redis.Redis,
    agent_id: UUID,
    room_id: UUID,
) -> int | None:
    """Atomically pop the token; return the seq it acknowledged, or None.

    None means no verifiable server-shown HOLD exists for this scope —
    absent token OR Redis failure — and a send_anyway at this moment is
    an unverified bypass: refused. The gate still runs, so the cost is
    one extra HOLD, never a blocked turn.
    """
    try:
        raw = await client.eval(_CONSUME_SCRIPT, 1, _key(agent_id, room_id))
    except Exception:
        logger.warning(
            "consume_hold(%s, %s) failed — fail-closed to gate",
            agent_id,
            room_id,
            exc_info=True,
        )
        return None
    if raw is None:
        return None
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


async def clear_hold(
    client: redis.Redis,
    agent_id: UUID,
    room_id: UUID,
) -> None:
    """A successful send invalidates any lingering acknowledgement."""
    try:
        await client.delete(_key(agent_id, room_id))
    except Exception:
        logger.warning(
            "clear_hold(%s, %s) failed — fail-open",
            agent_id,
            room_id,
            exc_info=True,
        )
