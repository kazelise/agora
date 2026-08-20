from __future__ import annotations

import asyncio

import asyncpg
import pytest

from server import db
from tests.conftest import DSN


@pytest.fixture
async def pool(require_services: None) -> asyncpg.Pool:
    pool = await db.create_pool(DSN)
    await db.migrate(pool)
    await db.truncate_all(pool)
    yield pool
    await pool.close()


@pytest.mark.asyncio
async def test_per_room_seq_monotonic_and_gapless_under_concurrency(pool: asyncpg.Pool) -> None:
    room = await db.create_room(pool, "seq-race")
    author = await db.add_participant(pool, room.id, "human", "Ada", None)

    async def post(i: int) -> int:
        row = await db.insert_message(pool, room.id, author.id, f"msg-{i}")
        return row.seq

    seqs = await asyncio.gather(*[post(i) for i in range(20)])
    assert sorted(seqs) == list(range(1, 21))
    assert len(set(seqs)) == 20

    stored = await pool.fetch(
        "SELECT seq FROM messages WHERE room_id = $1 ORDER BY seq",
        room.id,
    )
    assert [row["seq"] for row in stored] == list(range(1, 21))
    last = await pool.fetchval("SELECT last_seq FROM rooms WHERE id = $1", room.id)
    assert last == 20
