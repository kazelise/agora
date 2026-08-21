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
    """One in-flight turn per agent, plus a single dirty bit.

    Wakes that arrive while a turn is pending/running do not enqueue N
    extra turns. They set `_dirty`. When the current turn finishes we
    rerun at most once against the latest inbox (rerun-once).
    """

    def __init__(self, run_turn: TurnFn) -> None:
        self._run_turn = run_turn
        self._lock = asyncio.Lock()
        self._dirty = False
        self._task: asyncio.Task[None] | None = None

    async def notify(self, room_id: UUID, agent_id: UUID) -> None:
        async with self._lock:
            if self._task is not None and not self._task.done():
                self._dirty = True
                return
            self._dirty = False
            self._task = asyncio.create_task(
                self._loop(room_id, agent_id),
                name=f"agora-byoa-{agent_id}",
            )

    async def _loop(self, room_id: UUID, agent_id: UUID) -> None:
        while True:
            await self._run_turn(agent_id, room_id)
            async with self._lock:
                if not self._dirty:
                    self._task = None
                    return
                self._dirty = False

    async def wait_idle(self) -> None:
        while True:
            async with self._lock:
                task = self._task
            if task is None:
                return
            await task
