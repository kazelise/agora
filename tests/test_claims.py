"""Claims: atomic claim/release, crash-recovery TTL steal, race safety."""

from __future__ import annotations

import asyncio

import pytest

from server import db
from server.db import CLAIM_TTL_S
from tests.conftest import DSN


@pytest.fixture
async def pool():
    pool = await db.create_pool(DSN)
    await db.migrate(pool)
    await db.truncate_all(pool)
    yield pool
    await pool.close()


@pytest.fixture
async def room(pool):
    room = await db.create_room(pool, "claims")
    a = await db.add_participant(pool, room.id, "agent", "A", None)
    b = await db.add_participant(pool, room.id, "agent", "B", None)
    return room.id, a.id, b.id


@pytest.mark.asyncio
async def test_claim_insert_then_conflict_loses(pool, room) -> None:
    room_id, a_id, b_id = room
    assert await db.try_claim(pool, room_id, "t1", a_id) is True
    assert await db.try_claim(pool, room_id, "t1", b_id) is False
    # Same holder re-claiming is a no-op win (idempotent refresh).
    assert await db.try_claim(pool, room_id, "t1", a_id) is True


@pytest.mark.asyncio
async def test_release_only_by_holder(pool, room) -> None:
    room_id, a_id, b_id = room
    assert await db.try_claim(pool, room_id, "t1", a_id) is True
    # B cannot release A's claim.
    assert await db.release_claim(pool, room_id, "t1", b_id) is False
    assert await db.release_claim(pool, room_id, "t1", a_id) is True
    # Now B can claim it.
    assert await db.try_claim(pool, room_id, "t1", b_id) is True


@pytest.mark.asyncio
async def test_fresh_claim_cannot_be_stolen(pool, room) -> None:
    """A claim younger than the TTL is a live lock — nobody may steal it.
    Regression guard for the TTL steal: the WHERE clause must compare
    age, not just exist."""
    room_id, a_id, b_id = room
    assert await db.try_claim(pool, room_id, "t1", a_id) is True
    assert await db.try_claim(pool, room_id, "t1", b_id) is False
    row = await pool.fetchrow(
        "SELECT claimed_by FROM claims WHERE room_id = $1 AND task_key = $2",
        room_id,
        "t1",
    )
    assert str(row["claimed_by"]) == str(a_id)


@pytest.mark.asyncio
async def test_expired_claim_is_stolen_atomically(pool, room) -> None:
    """A crashed holder's claim must not pin the task key forever: after
    the TTL another agent steals it in one statement (no window where the
    key is free and two stealers both 'win')."""
    room_id, a_id, b_id = room
    assert await db.try_claim(pool, room_id, "t1", a_id) is True
    # Age the claim past the TTL (simulate a holder that crashed long ago).
    await pool.execute(
        """
        UPDATE claims
        SET created_at = now() - make_interval(secs => $3)
        WHERE room_id = $1 AND task_key = $2
        """,
        room_id,
        "t1",
        CLAIM_TTL_S * 2,
    )
    assert await db.try_claim(pool, room_id, "t1", b_id) is True
    row = await pool.fetchrow(
        "SELECT claimed_by FROM claims WHERE room_id = $1 AND task_key = $2",
        room_id,
        "t1",
    )
    assert str(row["claimed_by"]) == str(b_id)


@pytest.mark.asyncio
async def test_expired_claim_steal_is_race_safe(pool, room) -> None:
    """Many agents stealing the same expired claim concurrently: exactly
    one wins, the rest are told lost."""
    room_id, _a_id, b_id = room
    assert await db.try_claim(pool, room_id, "t1", _a_id) is True
    await pool.execute(
        """
        UPDATE claims
        SET created_at = now() - make_interval(secs => $3)
        WHERE room_id = $1 AND task_key = $2
        """,
        room_id,
        "t1",
        CLAIM_TTL_S * 2,
    )
    # The claim is expired: every challenger is a candidate stealer.
    challengers = [_a_id, b_id]
    for i in range(5):
        p = await db.add_participant(pool, room_id, "agent", f"C{i}", None)
        challengers.append(p.id)

    results = await asyncio.gather(
        *[db.try_claim(pool, room_id, "t1", c) for c in challengers]
    )
    assert sum(results) == 1
    row = await pool.fetchrow(
        "SELECT claimed_by FROM claims WHERE room_id = $1 AND task_key = $2",
        room_id,
        "t1",
    )
    winner_index = results.index(True)
    assert str(row["claimed_by"]) == str(challengers[winner_index])
