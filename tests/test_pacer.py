"""AdaptivePacer: deterministic spawn spacing + rate-limit adaptation."""

from __future__ import annotations

import asyncio
import time

import pytest

from daemon.pacer import AdaptivePacer


@pytest.mark.asyncio
async def test_burst_is_spread_at_hard_interval() -> None:
    """5 concurrent callers start at >= interval apart, by construction."""
    pacer = AdaptivePacer(base_interval_s=0.1)
    starts: list[float] = []

    async def caller() -> None:
        await pacer.wait_turn()
        starts.append(time.monotonic())

    await asyncio.gather(*(caller() for _ in range(5)))
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert all(g >= 0.09 for g in gaps), gaps


@pytest.mark.asyncio
async def test_no_wait_when_calls_are_spaced() -> None:
    pacer = AdaptivePacer(base_interval_s=0.05)
    await pacer.wait_turn()
    await asyncio.sleep(0.06)
    waited = await pacer.wait_turn()
    assert waited == 0.0


def test_rate_limit_doubles_then_caps() -> None:
    pacer = AdaptivePacer(base_interval_s=0.5, max_interval_s=2.0)
    assert pacer.on_rate_limited() == 1.0
    assert pacer.on_rate_limited() == 2.0
    assert pacer.on_rate_limited() == 2.0


def test_clean_calls_halve_back_to_base() -> None:
    pacer = AdaptivePacer(base_interval_s=0.5, max_interval_s=2.0)
    pacer.on_rate_limited()
    pacer.on_rate_limited()  # interval 2.0
    for _ in range(5):
        pacer.on_ok()
    assert pacer.interval_s == 1.0
    for _ in range(5):
        pacer.on_ok()
    assert pacer.interval_s == 0.5
    # Never below base.
    for _ in range(5):
        pacer.on_ok()
    assert pacer.interval_s == 0.5


@pytest.mark.asyncio
async def test_rate_limited_call_widens_pacer_through_invoke_model() -> None:
    """The graph's invoke_model feeds 429s into the pacer."""
    from brain.graph import invoke_model

    class RateLimitedModel:
        async def ainvoke(self, messages: list, **_kwargs: object) -> object:
            raise RuntimeError("Error code: 429 - rate limit exceeded")

    pacer = AdaptivePacer(base_interval_s=0.01)
    import brain.graph as graph_mod

    original = graph_mod.LLM_RETRY_BACKOFF_S
    graph_mod.LLM_RETRY_BACKOFF_S = 0
    try:
        result = await invoke_model(
            RateLimitedModel(), [], label="t", pacer=pacer
        )
    finally:
        graph_mod.LLM_RETRY_BACKOFF_S = original
    assert result is None
    # Both the first attempt and the retry were rate-limited → doubled twice.
    assert pacer.interval_s == 0.04


@pytest.mark.asyncio
async def test_clean_call_restores_pacer_through_invoke_model() -> None:
    from brain.graph import invoke_model

    class OkModel:
        async def ainvoke(self, messages: list, **_kwargs: object) -> object:
            return "ok"

    pacer = AdaptivePacer(base_interval_s=0.01)
    pacer.on_rate_limited()
    result = await invoke_model(OkModel(), [], label="t", pacer=pacer)
    assert result == "ok"
