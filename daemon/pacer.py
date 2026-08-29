"""Adaptive spawn pacing for the daemon's model calls.

Cumora's lesson (§3b): when many agents on one computer wake on the same
fan-out, they hit the provider's burst limit in lockstep. Random jitter is
probabilistic — four simultaneous wakes can all roll low — so the base
rate is a deterministic minimum interval between LLM call starts. When a
call comes back rate-limited, the interval doubles (capped); consecutive
clean calls halve it back toward the base.

This is a coordination signal, not a correctness invariant: pacing adds
latency, never changes what a turn decides. Everything here must be safe
under concurrent asyncio tasks — the pacer is one shared instance per
process, guarded by a lock.
"""

from __future__ import annotations

import asyncio
import time

BASE_INTERVAL_S = 0.5
MAX_INTERVAL_S = 8.0
CLEAN_CALLS_TO_HALVE = 5


class AdaptivePacer:
    """Deterministic minimum interval between model-call starts, with
    exponential adaptation on rate-limit feedback."""

    def __init__(
        self,
        *,
        base_interval_s: float = BASE_INTERVAL_S,
        max_interval_s: float = MAX_INTERVAL_S,
    ) -> None:
        self._base = base_interval_s
        self._max = max_interval_s
        self._interval = base_interval_s
        self._next_ok_at = 0.0
        self._clean = 0
        self._lock = asyncio.Lock()

    @property
    def interval_s(self) -> float:
        return self._interval

    async def wait_turn(self) -> float:
        """Block until this caller's slot; return the interval actually waited.

        FIFO by lock acquisition order: each caller reserves the next slot
        at least `interval` seconds after the previous one, so a burst of N
        wakes is spread at a hard 1/interval rate by construction.
        """
        async with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_ok_at - now)
            if delay:
                await asyncio.sleep(delay)
                now = time.monotonic()
            self._next_ok_at = now + self._interval
            return delay

    def on_rate_limited(self) -> float:
        """Double the interval (capped). Returns the new interval."""
        self._clean = 0
        self._interval = min(self._interval * 2, self._max)
        return self._interval

    def on_ok(self) -> float:
        """After enough consecutive clean calls, halve toward the base."""
        self._clean += 1
        if self._clean >= CLEAN_CALLS_TO_HALVE and self._interval > self._base:
            self._interval = max(self._interval / 2, self._base)
            self._clean = 0
        return self._interval
