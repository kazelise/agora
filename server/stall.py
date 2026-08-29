"""Stall pipeline: proactive wake for rooms that went quiet with work owed.

Agora's turns are reactive — a wake happens only when a message lands.
Cumora's §5c adds the missing liveness leg: when a room is stalled (the
latest message is from a peer an agent owes a reply to, and nobody has
spoken for a while), a periodic sweep picks EXACTLY ONE agent and wakes
it. Without this, a released claim + a silent room is a deadlock nobody
ever wakes from.

Design constraints (from this repo's layered principles):
- The sweep is DETERMINISTIC: eligibility is arithmetic on committed rows
  (who spoke last, how long ago, who has read what). It never classifies
  content. The brain still decides what, if anything, to say.
- Exactly one agent is nudged per stall, chosen deterministically
  (oldest-participant first among eligible agents). No NX claim needed:
  the scheduler's per-agent lane already serializes turns, and the
  Redis-pubsub topology means exactly one process runs the sweep.
- A per-room decline cap bounds the burn: if the nudged agents keep
  declining (turns end without a new message), the sweep stops after
  STALL_MAX_NUDGES attempts until ANY new message lands (state changed,
  fresh budget). This is Cumora's decline cap (their e1d83e7).
- Fail-open: any sweep error is a missed nudge, never a crashed server.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from uuid import UUID

logger = logging.getLogger("agora.stall")

# A room counts as stalled only after this much silence.
STALL_MIN_S = 20.0
# ...and never nudges rooms quiet longer than this (stale rooms are dead,
# not stalled; nudging them just burns tokens against an abandoned task).
STALL_MAX_S = 3600.0
# Declines (nudges that end without a new message) allowed per stall.
STALL_MAX_NUDGES = 3
# How often the sweep runs.
SWEEP_INTERVAL_S = 10.0

# WakeFn(agent_id, room_id) — matches Scheduler.lane(...).notify(room, agent)
# arity after binding; the sweeper always calls with the agent first.
WakeFn = Callable[[UUID, UUID], Awaitable[None]]


class StallSweeper:
    """Periodic deterministic sweep over active rooms."""

    def __init__(
        self,
        pool: Any,
        wake: WakeFn,
        *,
        interval_s: float = SWEEP_INTERVAL_S,
        min_s: float = STALL_MIN_S,
        max_s: float = STALL_MAX_S,
        max_nudges: int = STALL_MAX_NUDGES,
    ) -> None:
        self._pool = pool
        self._wake = wake
        self._interval = interval_s
        self._min_s = min_s
        self._max_s = max_s
        self._max_nudges = max_nudges
        # room_id -> declines since the last message landed there.
        self._declines: dict[UUID, int] = {}
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="agora-stall-sweep")

    async def stop(self) -> None:
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    def on_message(self, room_id: UUID) -> None:
        """A message landed: the decline budget resets (state changed)."""
        self._declines.pop(room_id, None)

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            try:
                await self.sweep_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("stall sweep failed — skipping this pass", exc_info=True)

    async def sweep_once(self) -> int:
        """One pass; returns how many nudges fired (tests call this directly)."""
        now = datetime.now(UTC)
        nudged = 0
        for stall in await self._stalled_rooms(now):
            if self._declines.get(stall["room_id"], 0) >= self._max_nudges:
                continue
            agent_id = stall["agent_id"]
            try:
                await self._wake(agent_id, stall["room_id"])
            except Exception:
                logger.warning(
                    "stall wake failed room=%s agent=%s",
                    stall["room_id"],
                    agent_id,
                    exc_info=True,
                )
                continue
            nudged += 1
            self._declines[stall["room_id"]] = (
                self._declines.get(stall["room_id"], 0) + 1
            )
            logger.info(
                "stall nudge room=%s agent=%s quiet_for=%.0fs last_author=%s "
                "declines=%s/%s",
                stall["room_id"],
                agent_id,
                stall["quiet_for_s"],
                stall["last_author_id"],
                self._declines[stall["room_id"]],
                self._max_nudges,
            )
        return nudged

    async def _stalled_rooms(self, now: datetime) -> list[dict[str, Any]]:
        """Rooms where an agent owes the last word and nobody has spoken.

        Eligibility is arithmetic on committed state only:
        - the room's latest message is older than min_s (silence)
        - ...but younger than max_s (not an abandoned room)
        - the agent being nudged is not the author of the last message
          (you don't owe yourself a reply)
        - the agent has already READ that message (waking an agent with an
          unread inbox is not a stall — its own lane missed a wake; that is
          a different failure, and the cursor may still deliver it)
        Among eligible agents the OLDEST participant wins: deterministic,
        and rotation across sweeps comes from `declines` advancing.
        """
        from server import db

        out: list[dict[str, Any]] = []
        for room in await db.list_active_rooms(self._pool):
            room_id = UUID(str(room["id"]))
            last_at = room["last_at"]
            if last_at is None:
                continue
            quiet_for = (now - last_at).total_seconds()
            if quiet_for < self._min_s or quiet_for > self._max_s:
                continue
            last_author = UUID(str(room["author_id"]))
            agents = await db.list_agent_participants(self._pool, room_id)
            candidates = [
                a
                for a in agents
                if a.id != last_author
                and await db.get_last_read(self._pool, a.id, room_id)
                >= int(room["seq"])
            ]
            if not candidates:
                continue
            out.append(
                {
                    "room_id": room_id,
                    "agent_id": candidates[0].id,
                    "quiet_for_s": quiet_for,
                    "last_author_id": last_author,
                    "last_author_kind": str(room["author_kind"]),
                }
            )
        return out
