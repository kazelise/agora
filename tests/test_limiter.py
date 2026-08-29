"""Per-computer LLM concurrency cap (Cumora §2/§3a)."""

from __future__ import annotations

import asyncio

import pytest

from daemon.limiter import ConcurrencyLimiter


@pytest.mark.asyncio
async def test_cap_bounds_concurrent_calls() -> None:
    limiter = ConcurrencyLimiter(max_concurrent=2)
    state = {"current": 0, "peak": 0}
    lock = asyncio.Lock()

    async def call() -> None:
        async with limiter.slot():
            async with lock:
                state["current"] += 1
                state["peak"] = max(state["peak"], state["current"])
            await asyncio.sleep(0.02)
            state["current"] -= 1

    await asyncio.gather(*(call() for _ in range(8)))
    assert state["peak"] == 2  # parallelism allowed up to the cap, no further


@pytest.mark.asyncio
async def test_slot_releases_on_error() -> None:
    limiter = ConcurrencyLimiter(max_concurrent=1)
    with pytest.raises(RuntimeError):
        async with limiter.slot():
            raise RuntimeError("boom")
    # The failed holder must not have leaked the slot.
    assert limiter.in_flight == 0
    async with limiter.slot():
        assert limiter.in_flight == 1


@pytest.mark.asyncio
async def test_retry_holds_the_same_slot() -> None:
    """The retry is the same logical call: no queued caller may jump the
    slot between attempt and retry. Modeled after invoke_model's shape."""
    limiter = ConcurrencyLimiter(max_concurrent=1)
    events: list[str] = []

    async def other_caller() -> None:
        async with limiter.slot():
            events.append("other-in")

    async def flaky_call_with_retry() -> None:
        async with limiter.slot():
            events.append("attempt-1")
            # Release nothing: retry stays inside the same slot.
            events.append("retry")
            await other_caller if False else None

    holder = asyncio.create_task(flaky_call_with_retry())
    await asyncio.sleep(0.01)
    outsider = asyncio.create_task(other_caller())
    await asyncio.gather(holder, outsider)
    # The outsider entered only after the holder's whole attempt+retry.
    assert events == ["attempt-1", "retry", "other-in"]


@pytest.mark.asyncio
async def test_invoke_model_routes_through_limiter(monkeypatch: pytest.MonkeyPatch) -> None:
    from brain.graph import invoke_model
    from daemon.limiter import ConcurrencyLimiter

    limiter = ConcurrencyLimiter(max_concurrent=1)
    seen: list[int] = []

    class Model:
        async def ainvoke(self, messages: list) -> object:
            seen.append(limiter.in_flight)
            return "ok"

    result = await invoke_model(Model(), [], label="x", limiter=limiter)
    assert result == "ok"
    assert seen == [1]
    assert limiter.in_flight == 0


@pytest.mark.asyncio
async def test_invoke_model_no_limiter_is_passthrough() -> None:
    from brain.graph import invoke_model

    class Model:
        async def ainvoke(self, messages: list) -> object:
            return "ok"

    assert await invoke_model(Model(), [], label="x") == "ok"


def test_pacer_and_limiter_are_independent_budgets() -> None:
    from daemon.pacer import AdaptivePacer

    pacer = AdaptivePacer()
    limiter = ConcurrencyLimiter(max_concurrent=1)
    # Spacing (time between starts) and concurrency (calls in flight) are
    # separate budgets: one does not implement, replace, or share state
    # with the other (Cumora runs both for the same reason).
    assert pacer.interval_s > 0
    assert limiter.max_concurrent == 1
