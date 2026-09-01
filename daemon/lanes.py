"""Per-agent serial lane with rerun-once coalescing.

Copied from server.scheduler.AgentLane rather than imported: that module
pulls asyncpg and redis, which this process must not depend on. The
daemon owns BYOA lanes; the server does not queue wakes for us.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import UUID

TurnFn = Callable[[UUID, UUID], Awaitable[Any]]


class AgentLane:
    """One in-flight turn per agent, plus a single pending-wake slot.

    Wakes that arrive while a turn is pending/running do not enqueue N
    extra turns. They overwrite `_pending` with the latest (room_id,
    agent_id). When the current turn finishes we rerun at most once —
    against the LATEST wake, not the first: an agent in several rooms
    whose wake for room B lands while it is mid-turn for room A must
    rerun in room B (its cursor-based inbox will also cover any A
    updates that raced in, on the turn after that).

    One caveat kept honest: with wakes for TWO other rooms racing the
    current turn, only the last one survives the coalesce. The other
    room's delivery waits for the next event — the same eventual
    delivery a missed pub/sub wake gets. Bounding the rerun to one is
    what keeps a message burst from queueing N model calls.
    """

    def __init__(self, run_turn: TurnFn) -> None:
        self._run_turn = run_turn
        self._lock = asyncio.Lock()
        self._pending: tuple[UUID, UUID] | None = None
        self._task: asyncio.Task[None] | None = None

    async def notify(self, room_id: UUID, agent_id: UUID) -> tuple[UUID, UUID] | None:
        """Returns the overwritten pending (room, agent), if any."""
        async with self._lock:
            if self._task is not None and not self._task.done():
                old = self._pending
                self._pending = (room_id, agent_id)
                return old
            self._pending = None
            self._task = asyncio.create_task(
                self._loop(room_id, agent_id),
                name=f"agora-byoa-{agent_id}",
            )
            return None

    async def _loop(self, room_id: UUID, agent_id: UUID) -> None:
        while True:
            await self._run_turn(agent_id, room_id)
            async with self._lock:
                if self._pending is None:
                    self._task = None
                    return
                room_id, agent_id = self._pending
                self._pending = None

    async def wait_idle(self) -> None:
        while True:
            async with self._lock:
                task = self._task
            if task is None:
                return
            await task
