"""Stall pipeline: proactive wake for rooms that went quiet with work owed.

Agora's turns are reactive — a wake happens only when a message lands.
Cumora's §5c adds the missing liveness leg: when a room is stalled (the
latest message is from a peer an agent owes a reply to, and nobody has
spoken for a while), a periodic sweep nudges the room's NON-author agents
through the same dispatch path a real message takes. Without this, a
released claim + a silent room is a deadlock nobody ever wakes from.

    Design constraints (from this repo's layered principles):
- The sweep is DETERMINISTIC: eligibility is arithmetic on committed rows
  (who spoke last, how long ago, who has read what). It never classifies
  content. The brain still decides what, if anything, to say — including
  whether silence is the right reply on a proactive turn.
- Nudges travel through dispatch (room_id + nominal author), so BYOA
  computers and K8s jobs are woken over their own transports. Routing the
  nudge into a server-side brain lane would silently strip a BYOA agent
  of its model boundary.
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
# How long a room where NOBODY has read the last message stays classified
# as "delivery in flight" (a wake may still land) before the sweeper
# treats the wake as lost and nudges anyway. Must comfortably exceed one
# turn (model latency + retries); losing a wake with no follow-up event
# would otherwise starve the room forever (the reactive path has no
# retry — the pub/sub wake is fire-and-forget).
STALL_UNREAD_GRACE_S = 120.0

# NudgeFn(room_id, author_id) — the same arity as Scheduler.dispatch. The
# sweeper names the LAST AUTHOR as the nominal "message sender": dispatch
# then routes each non-author agent through its real host (in-process
# lane, BYOA websocket, or K8s job). Waking through the server-side lane
# directly would silently turn BYOA agents into server-brained agents.
NudgeFn = Callable[[UUID, UUID], Awaitable[None]]


class StallSweeper:
    """Periodic deterministic sweep over active rooms."""

    def __init__(
        self,
        pool: Any,
        nudge: NudgeFn,
        *,
        interval_s: float = SWEEP_INTERVAL_S,
        min_s: float = STALL_MIN_S,
        max_s: float = STALL_MAX_S,
        max_nudges: int = STALL_MAX_NUDGES,
        unread_grace_s: float = STALL_UNREAD_GRACE_S,
    ) -> None:
        self._pool = pool
        self._nudge = nudge
        self._interval = interval_s
        self._min_s = min_s
        self._max_s = max_s
        self._max_nudges = max_nudges
        self._unread_grace_s = unread_grace_s
        # room_id -> declines since the last message landed there.
        self._declines: dict[UUID, int] = {}
        self._task: asyncio.Task[None] | None = None

    def start(self) -> asyncio.Task[None]:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="agora-stall-sweep")
        return self._task

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
            try:
                # Route through dispatch semantics, not the server lane: a
                # nudge must reach BYOA / K8s agents the same way a real
                # message would. The room's last author is the nominal
                # "sender"; dispatch wakes every agent except them, each
                # through its own host (lane, computer ws, or cloud job).
                await self._nudge(stall["room_id"], stall["last_author_id"])
            except Exception:
                logger.warning(
                    "stall nudge failed room=%s last_author=%s",
                    stall["room_id"],
                    stall["last_author_id"],
                    exc_info=True,
                )
                continue
            nudged += 1
            self._declines[stall["room_id"]] = (
                self._declines.get(stall["room_id"], 0) + 1
            )
            logger.info(
                "stall nudge room=%s last_author=%s quiet_for=%.0fs "
                "declines=%s/%s",
                stall["room_id"],
                stall["last_author_id"],
                stall["quiet_for_s"],
                self._declines[stall["room_id"]],
                self._max_nudges,
            )
        return nudged

    async def _stalled_rooms(self, now: datetime) -> list[dict[str, Any]]:
        """Rooms where an agent owes the last word and nobody has spoken.

        Eligibility is arithmetic on committed state only:
        - the room's latest message is older than min_s (silence)
        - ...but younger than max_s (not an abandoned room)
        - at least one non-author agent has READ the last message (waking
          agents with an unread inbox is not a stall — their own lane
          missed a wake; the cursor will deliver it). Unread agents are
          still woken by the dispatch — they get the reactive delivery —
          but a room where nobody has read counts as undelivered, not
          stalled.

        The unread rule has a starvation hole: a room whose ONLY non-author
        agents are unread (an offline BYOA host that missed the pub/sub
        wake — it is fire-and-forget) never becomes eligible, and no
        further event will ever re-dispatch. So an unread room graduates
        to stalled once the last message is older than unread_grace_s —
        longer than any legitimate turn, so a live lane cannot still be
        "in flight"; the wake was lost and the nudge is the retry.
        Individual wake routing (in-process lane vs computer ws vs cloud
        job) is dispatch's job; the sweeper only judges the room.
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
            # The last author never owes themselves a reply. Everyone else
            # is woken — through dispatch — if at least one of them has
            # already read the room's last word.
            readers = [
                a
                for a in agents
                if a.id != last_author
                and await db.get_last_read(self._pool, a.id, room_id)
                >= int(room["seq"])
            ]
            if not readers and quiet_for < self._unread_grace_s:
                continue
            out.append(
                {
                    "room_id": room_id,
                    "quiet_for_s": quiet_for,
                    "last_author_id": last_author,
                    "last_author_kind": str(room["author_kind"]),
                }
            )
        return out
