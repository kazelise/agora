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
async def test_stalled_room_nudges_via_dispatch_sender(pool: Any) -> None:
    """The nudge carries (room, last_author) — dispatch semantics, so the
    caller routes every non-author agent through its own host."""
    room_id, human_id, _agent_ids = await _setup_room(pool, last_author="human")
    nudges: list[tuple[UUID, UUID]] = []

    async def nudge(room: UUID, author: UUID) -> None:
        nudges.append((room, author))

    sweeper = StallSweeper(pool, nudge, interval_s=9999)
    fired = await sweeper.sweep_once()

    assert fired == 1
    assert nudges == [(room_id, human_id)]


@pytest.mark.asyncio
async def test_nudge_sender_is_last_author_agent_too(pool: Any) -> None:
    """An agent's own abandoned last word still nudges its peers (the
    nominal sender is the last author, whoever it was)."""
    room_id, _human, agent_ids = await _setup_room(pool, last_author="0")
    nudges: list[tuple[UUID, UUID]] = []

    async def nudge(room: UUID, author: UUID) -> None:
        nudges.append((room, author))

    sweeper = StallSweeper(pool, nudge, interval_s=9999)
    assert await sweeper.sweep_once() == 1
    assert nudges == [(room_id, agent_ids[0])]


@pytest.mark.asyncio
async def test_decline_cap_stops_nudging_until_new_message(pool: Any) -> None:
    room_id, human_id, _agent_ids = await _setup_room(pool, last_author="human")
    nudges: list[UUID] = []

    async def nudge(room: UUID, _author: UUID) -> None:
        nudges.append(room)

    sweeper = StallSweeper(pool, nudge, interval_s=9999, max_nudges=2)
    assert await sweeper.sweep_once() == 1
    assert await sweeper.sweep_once() == 1
    # Declines exhausted — the judgment has converged; stop burning tokens.
    assert await sweeper.sweep_once() == 0
    assert len(nudges) == 2

    # A new message lands → budget resets.
    sweeper.on_message(room_id)
    assert await sweeper.sweep_once() == 1
    assert len(nudges) == 3


@pytest.mark.asyncio
async def test_fresh_room_and_stale_room_are_not_stalled(pool: Any) -> None:
    # Too fresh: the human just spoke.
    await _setup_room(pool, last_author="human", age_s=1.0)
    # Too stale: abandoned long ago.
    await _setup_room(pool, last_author="human", age_s=7200.0)
    nudges: list[UUID] = []

    async def nudge(room: UUID, _author: UUID) -> None:
        nudges.append(room)

    sweeper = StallSweeper(pool, nudge, interval_s=9999)
    assert await sweeper.sweep_once() == 0
    assert nudges == []


@pytest.mark.asyncio
async def test_unread_rooms_are_not_stalled(pool: Any) -> None:
    """Agents that never read the last message have a delivery problem,
    not a stall — their reactive wake may still be in flight; the cursor
    will deliver the message. Nudging here would double-wake a live lane."""
    room_id, _human, _agent_ids = await _setup_room(
        pool, last_author="human", read_by_agents=False
    )
    nudges: list[UUID] = []

    async def nudge(room: UUID, _author: UUID) -> None:
        nudges.append(room)

    sweeper = StallSweeper(pool, nudge, interval_s=9999)
    assert await sweeper.sweep_once() == 0


@pytest.mark.asyncio
async def test_unread_room_graduates_after_unread_grace(pool: Any) -> None:
    """The unread rule's starvation hole: an offline BYOA host that
    missed the fire-and-forget pub/sub wake never reads, and no further
    event ever re-dispatches — the room starves. Past the unread grace
    (longer than any legitimate turn), the lost wake is treated as
    undelivered and the nudge becomes the retry."""
    room_id, human_id, _agent_ids = await _setup_room(
        pool, last_author="human", read_by_agents=False
    )
    nudges: list[tuple[UUID, UUID]] = []

    async def nudge(room: UUID, author: UUID) -> None:
        nudges.append((room, author))

    sweeper = StallSweeper(
        pool, nudge, interval_s=9999, unread_grace_s=300.0
    )
    # 60s silence: inside the grace — a live lane may still be running.
    assert await sweeper.sweep_once() == 0
    assert nudges == []
    # Age the last message past the grace: the wake is lost, retry it.
    await pool.execute(
        "UPDATE messages SET created_at = $1 WHERE room_id = $2",
        _old(400.0),
        room_id,
    )
    assert await sweeper.sweep_once() == 1
    assert nudges == [(room_id, human_id)]

@pytest.mark.asyncio
async def test_nudge_error_is_fail_open(pool: Any) -> None:
    room_id, _human, _agent_ids = await _setup_room(pool, last_author="human")

    async def boom(_room: UUID, _author: UUID) -> None:
        raise RuntimeError("dispatch exploded")

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
async def test_stall_nudge_dispatches_and_proactive_turn_decides(
    app_client: tuple,
) -> None:
    """End-to-end: a quiet room with a read-but-unanswered human ask gets
    re-woken by the sweep through dispatch, and the proactive turn runs
    triage (via the stub scheduler's turn records) instead of no-oping."""
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
    # must add a PROACTIVE turn on top — inbox empty, cursor current —
    # proving the room was re-woken from silence, not just re-delivered.
    turns = app.state.scheduler.turns
    assert len(turns) == 2
    assert turns[-1].agent_id == UUID(agent["id"])
    assert turns[-1].room_id == UUID(room["id"])
    assert turns[-1].inbox_count == 0


@pytest.mark.asyncio
async def test_proactive_turn_skips_offline_byoa_like_dispatch(
    app_client: tuple,
) -> None:
    """A stall nudge must not invent turns for agents whose host is
    offline — the same sleep rule as a landed message."""
    app, client = app_client
    app.state.stalls._min_s = 0.0
    room = (await client.post("/rooms", json={"name": "offline-room"})).json()
    human = (
        await client.post(
            f"/rooms/{room['id']}/participants",
            json={"kind": "human", "name": "Ada"},
        )
    ).json()
    agent = (
        await client.post(
            f"/rooms/{room['id']}/participants",
            json={"kind": "agent", "name": "Remote", "computer_id": None},
        )
    ).json()
    # Give the agent a computer_id pointing at a computer that never
    # connects → dispatch sees it offline and logs "sleeping".
    computer = (await client.post("/computers", json={"name": "ghost"})).json()
    await app.state.pool.execute(
        "UPDATE participants SET computer_id = $1 WHERE id = $2",
        UUID(computer["id"]),
        UUID(agent["id"]),
    )
    posted = await client.post(
        f"/rooms/{room['id']}/messages",
        json={"author_id": human["id"], "body": "anyone home?"},
    )
    posted.raise_for_status()
    await db.set_last_read(
        app.state.pool, UUID(agent["id"]), UUID(room["id"]), 1
    )

    fired = await app.state.stalls.sweep_once()
    await app.state.scheduler.wait_idle()

    assert fired == 1
    # Only the reactive turn (from the human POST) happened; the nudge
    # found the host offline and did not queue anything.
    turns = [
        t
        for t in app.state.scheduler.turns
        if t.agent_id == UUID(agent["id"])
    ]
    assert turns == []


@pytest.mark.asyncio
async def test_stall_nudge_reaches_byoa_computer_over_ws(
    app_client: tuple,
) -> None:
    """The nudge must ride dispatch so a BYOA agent is woken over its
    computer websocket (its brain lives on the laptop), never through the
    server-side lane. Regression guard for the host-boundary mutation:
    routing the nudge into scheduler.lane directly would run BYOA brains
    on the server and strip the model boundary."""
    from tests.asgi_ws import connect_asgi_ws

    app, client = app_client
    app.state.stalls._min_s = 0.0
    computer = (await client.post("/computers", json={"name": "laptop"})).json()
    room = (await client.post("/rooms", json={"name": "byoa-stall"})).json()
    human = (
        await client.post(
            f"/rooms/{room['id']}/participants",
            json={"kind": "human", "name": "Ada"},
        )
    ).json()
    agent = (
        await client.post(
            f"/rooms/{room['id']}/participants",
            json={
                "kind": "agent",
                "name": "Jules",
                "computer_id": computer["id"],
            },
        )
    ).json()
    posted = await client.post(
        f"/rooms/{room['id']}/messages",
        json={"author_id": human["id"], "body": "jules?"},
    )
    posted.raise_for_status()
    await db.set_last_read(
        app.state.pool, UUID(agent["id"]), UUID(room["id"]), 1
    )

    ws = await connect_asgi_ws(
        app,
        f"/ws/computers/{computer['id']}",
        query_string=f"token={computer['token']}",
    )
    try:
        fired = await app.state.stalls.sweep_once()
        assert fired == 1
        # The nudge surfaces as a wake frame on the computer socket.
        frame = await ws.receive_json(timeout=4.0)
        assert frame == {
            "type": "wake",
            "agent_id": agent["id"],
            "room_id": room["id"],
        }
        # And the server-side scheduler never ran a turn for the BYOA
        # agent — its brain is not the server's to run.
        await app.state.scheduler.wait_idle()
        assert all(
            t.agent_id != UUID(agent["id"])
            for t in app.state.scheduler.turns
        )
    finally:
        await ws.close()
