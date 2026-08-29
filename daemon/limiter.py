"""Per-computer LLM concurrency cap (Cumora §2/§3a).

When many agents on one computer wake on the same fan-out, the pacer
spaces their call *starts* but does not bound how many run
*concurrently* — a 7-agent broadcast room can hold 7 model calls open at
once, hitting the provider's short-window burst limit in lockstep
(Cumora's observed 130 rate-limit hits in 17 minutes). The cap is a
semaphore: at most N model calls in flight per process, both model
classes sharing one budget (small-brain triage and big-brain turns pull
from the same provider account — capping one layer without the other
just moves the herd up, Cumora §3a).

Fail-open by construction: if the semaphore code ever fails, turns still
run — the cap only adds waiting, never changes a decision. One shared
instance per process, like the pacer.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

# Cumora's default, tuned for persistent sessions + a pacer; a single
# process rarely wants more concurrent model calls than this.
DEFAULT_MAX_CONCURRENT = 6


class ConcurrencyLimiter:
    """Bounded concurrent model calls, shared across model classes."""

    def __init__(self, max_concurrent: int = DEFAULT_MAX_CONCURRENT) -> None:
        self._sem = asyncio.Semaphore(max_concurrent)
        self.max_concurrent = max_concurrent
        self._in_flight = 0

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        await self._sem.acquire()
        self._in_flight += 1
        try:
            yield
        finally:
            self._in_flight -= 1
            self._sem.release()
