"""Stall pipeline: deterministic proactive wake for quiet rooms with work owed."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import pytest

from server import db
from server.stall import StallSweeper


@pytest.fixture
async def pool(require_services: None) -> AsyncIterator[Any]:
    import tests.conftest as cfg

    created = await db.create_pool(cfg.DSN)
    await db.migrate(created)
    await db.truncate_all(created)
    yield created
    await created.close()


def _old(seconds: float) -> datetime:
    return datetime.now(UTC) - timedelta(seconds=seconds)


async def _setup_room(
    pool: Any,
    *,
    agents: int = 2,
    last_author: str = "human",
    age_s: float = 60.0,
    read_by_agents: bool = True,
) -> tuple[UUID, UUID, list[UUID]]:
    """Room whose last message landed `age_s` ago from `last_author`."""
    room = await db.create_room(pool, "stall-room")
    human = await db.add_participant(pool, room.id, "human", "Ada", None)
    # DB order is (created_at, id) — insertion order here, since each
    # participant INSERT is its own transaction with a fresh now(). The
    # sweeper picks the first eligible in that order.
    agent_rows = [
        await db.add_participant(pool, room.id, "agent", f"A{i}", None)
        for i in range(agents)
    ]
    agent_ids = [p.id for p in agent_rows]
    last: UUID
    if last_author == "human":
        last = human.id
    else:
        last = agent_ids[int(last_author)]
    row = await db.insert_message(pool, room.id, last, "anyone there?")
    # Backdate the message so the room looks quiet.
    await pool.execute(
        "UPDATE messages SET created_at = $1 WHERE id = $2",
        _old(age_s),
        row.id,
    )
    if read_by_agents:
        for agent_id in agent_ids:
            await db.set_last_read(pool, agent_id, room.id, row.seq)
    return room.id, human.id, agent_ids


@pytest.mark.asyncio
async def test_stalled_room_wakes_one_oldest_agent(pool: Any) -> None:
    room_id, _human, agent_ids = await _setup_room(pool, last_author="human")
    woken: list[tuple[UUID, UUID]] = []

    async def wake(agent_id: UUID, r: UUID) -> None:
        woken.append((agent_id, r))

    sweeper = StallSweeper(pool, wake, interval_s=9999)
    fired = await sweeper.sweep_once()

    assert fired == 1
    assert woken == [(agent_ids[0], room_id)]


@pytest.mark.asyncio
async def test_decline_cap_stops_nudging_until_new_message(pool: Any) -> None:
    room_id, _human, agent_ids = await _setup_room(pool, last_author="human")
    woken: list[UUID] = []

    async def wake(agent_id: UUID, _r: UUID) -> None:
        woken.append(agent_id)

    sweeper = StallSweeper(pool, wake, interval_s=9999, max_nudges=2)
    assert await sweeper.sweep_once() == 1
    assert await sweeper.sweep_once() == 1
    # Declines exhausted — the judgment has converged; stop burning tokens.
    assert await sweeper.sweep_once() == 0
    assert len(woken) == 2

    # A new message lands → budget resets.
    sweeper.on_message(room_id)
    assert await sweeper.sweep_once() == 1
    assert len(woken) == 3


@pytest.mark.asyncio
async def test_fresh_room_and_stale_room_are_not_stalled(pool: Any) -> None:
    # Too fresh: the human just spoke.
    await _setup_room(pool, last_author="human", age_s=1.0)
    # Too stale: abandoned long ago.
    await _setup_room(pool, last_author="human", age_s=7200.0)
    woken: list[UUID] = []

    async def wake(agent_id: UUID, _r: UUID) -> None:
        woken.append(agent_id)

    sweeper = StallSweeper(pool, wake, interval_s=9999)
    assert await sweeper.sweep_once() == 0
    assert woken == []


@pytest.mark.asyncio
async def test_unread_agents_are_not_stalled(pool: Any) -> None:
    """Agents that never read the last message have a delivery problem,
    not a stall — waking them here would double-wake a live lane."""
    room_id, _human, _agent_ids = await _setup_room(
        pool, last_author="human", read_by_agents=False
    )
    woken: list[UUID] = []

    async def wake(agent_id: UUID, _r: UUID) -> None:
        woken.append(agent_id)

    sweeper = StallSweeper(pool, wake, interval_s=9999)
    assert await sweeper.sweep_once() == 0


@pytest.mark.asyncio
async def test_last_author_is_not_nudged(pool: Any) -> None:
    """The agent who spoke last doesn't owe itself a reply; a peer does."""
    room_id, _human, agent_ids = await _setup_room(pool, last_author="0")
    woken: list[UUID] = []

    async def wake(agent_id: UUID, _r: UUID) -> None:
        woken.append(agent_id)

    sweeper = StallSweeper(pool, wake, interval_s=9999)
    assert await sweeper.sweep_once() == 1
    assert woken == [agent_ids[1]]


@pytest.mark.asyncio
async def test_wake_error_is_fail_open(pool: Any) -> None:
    room_id, _human, _agent_ids = await _setup_room(pool, last_author="human")

    async def boom(_agent_id: UUID, _r: UUID) -> None:
        raise RuntimeError("lane exploded")

    sweeper = StallSweeper(pool, boom, interval_s=9999)
    # Does not raise; does not count as a nudge (no decline spent).
    assert await sweeper.sweep_once() == 0


@pytest.mark.asyncio
async def test_sweep_loop_runs_and_stops() -> None:
    sweeper = StallSweeper(None, None, interval_s=0.01)  # type: ignore[arg-type]
    # Patch sweep_once so the loop never touches the (None) pool.
    calls = 0

    async def fake_sweep() -> int:
        nonlocal calls
        calls += 1
        if calls >= 3:
            raise _StopSweep()

    class _StopSweep(Exception):
        pass

    sweeper.sweep_once = fake_sweep  # type: ignore[method-assign]
    sweeper.start()
    await asyncio.sleep(0.2)
    await sweeper.stop()
    assert calls >= 3


@pytest.mark.asyncio
async def test_stall_nudge_runs_real_turn(app_client: tuple) -> None:
    """End-to-end: a quiet room with an unread human ask gets woken by the
    sweep through the scheduler lane (stub turns record the wake)."""
    app, client = app_client
    app.state.stalls._min_s = 0.0  # don't wait 20s in the test
    room = (await client.post("/rooms", json={"name": "sweep-room"})).json()
    human = (
        await client.post(
            f"/rooms/{room['id']}/participants",
            json={"kind": "human", "name": "Ada"},
        )
    ).json()
    agent = (
        await client.post(
            f"/rooms/{room['id']}/participants",
            json={"kind": "agent", "name": "Jules"},
        )
    ).json()
    posted = await client.post(
        f"/rooms/{room['id']}/messages",
        json={"author_id": human["id"], "body": "hello?"},
    )
    posted.raise_for_status()
    # Agent reads it (cursor moves) but never replies → genuinely stalled.
    await db.set_last_read(
        app.state.pool, UUID(agent["id"]), UUID(room["id"]), 1
    )

    fired = await app.state.stalls.sweep_once()
    await app.state.scheduler.wait_idle()

    assert fired == 1
    # The human POST already caused a reactive wake (turn 1). The sweep
    # must add a PROACTIVE turn on top — inbox now empty, cursor current —
    # proving the room was re-woken from silence, not just re-delivered.
    turns = app.state.scheduler.turns
    assert len(turns) == 2
    assert turns[-1].agent_id == UUID(agent["id"])
    assert turns[-1].room_id == UUID(room["id"])
    assert turns[-1].inbox_count == 0
