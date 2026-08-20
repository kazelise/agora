from __future__ import annotations

from pathlib import Path
from typing import cast
from uuid import UUID, uuid4

import asyncpg

from server.models import MessageRow, ParticipantKind, ParticipantRow, RoomRow

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class NotFoundError(Exception):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


async def create_pool(dsn: str) -> asyncpg.Pool:
    return await asyncpg.create_pool(dsn, min_size=2, max_size=20)


async def migrate(pool: asyncpg.Pool) -> None:
    sql = SCHEMA_PATH.read_text()
    async with pool.acquire() as conn:
        await conn.execute(sql)


def _room(row: asyncpg.Record) -> RoomRow:
    return RoomRow(id=row["id"], name=row["name"], created_at=row["created_at"])


def _participant(row: asyncpg.Record) -> ParticipantRow:
    return ParticipantRow(
        id=row["id"],
        room_id=row["room_id"],
        kind=cast(ParticipantKind, row["kind"]),
        name=row["name"],
        persona=row["persona"],
        created_at=row["created_at"],
    )


def _message(row: asyncpg.Record) -> MessageRow:
    return MessageRow(
        id=row["id"],
        room_id=row["room_id"],
        author_id=row["author_id"],
        body=row["body"],
        seq=row["seq"],
        created_at=row["created_at"],
    )


async def create_room(pool: asyncpg.Pool, name: str) -> RoomRow:
    row = await pool.fetchrow(
        """
        INSERT INTO rooms (id, name)
        VALUES ($1, $2)
        RETURNING id, name, created_at
        """,
        uuid4(),
        name,
    )
    assert row is not None
    return _room(row)


async def get_room(pool: asyncpg.Pool, room_id: UUID) -> RoomRow:
    row = await pool.fetchrow(
        "SELECT id, name, created_at FROM rooms WHERE id = $1",
        room_id,
    )
    if row is None:
        raise NotFoundError(f"room {room_id} not found")
    return _room(row)


async def add_participant(
    pool: asyncpg.Pool,
    room_id: UUID,
    kind: ParticipantKind,
    name: str,
    persona: str | None,
) -> ParticipantRow:
    async with pool.acquire() as conn:
        async with conn.transaction():
            exists = await conn.fetchval("SELECT 1 FROM rooms WHERE id = $1", room_id)
            if exists is None:
                raise NotFoundError(f"room {room_id} not found")
            row = await conn.fetchrow(
                """
                INSERT INTO participants (id, room_id, kind, name, persona)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id, room_id, kind, name, persona, created_at
                """,
                uuid4(),
                room_id,
                kind,
                name,
                persona,
            )
    assert row is not None
    return _participant(row)


async def get_participant(pool: asyncpg.Pool, participant_id: UUID) -> ParticipantRow:
    row = await pool.fetchrow(
        """
        SELECT id, room_id, kind, name, persona, created_at
        FROM participants
        WHERE id = $1
        """,
        participant_id,
    )
    if row is None:
        raise NotFoundError(f"participant {participant_id} not found")
    return _participant(row)


async def list_agent_participants(pool: asyncpg.Pool, room_id: UUID) -> list[ParticipantRow]:
    rows = await pool.fetch(
        """
        SELECT id, room_id, kind, name, persona, created_at
        FROM participants
        WHERE room_id = $1 AND kind = 'agent'
        ORDER BY created_at
        """,
        room_id,
    )
    return [_participant(r) for r in rows]


async def insert_message(
    pool: asyncpg.Pool,
    room_id: UUID,
    author_id: UUID,
    body: str,
) -> MessageRow:
    # Per-room counter, not a Postgres SEQUENCE: SEQUENCE is table-global
    # (or needs one sequence object per room). UPDATE ... RETURNING takes
    # a row lock on the room, so concurrent inserts serialize and rollback
    # of a failed insert also undoes the increment — seq stays gapless.
    async with pool.acquire() as conn:
        async with conn.transaction():
            belongs = await conn.fetchval(
                "SELECT 1 FROM participants WHERE id = $1 AND room_id = $2",
                author_id,
                room_id,
            )
            if belongs is None:
                exists = await conn.fetchval("SELECT 1 FROM rooms WHERE id = $1", room_id)
                if exists is None:
                    raise NotFoundError(f"room {room_id} not found")
                raise NotFoundError(f"author {author_id} is not in room {room_id}")
            seq = await conn.fetchval(
                """
                UPDATE rooms
                SET last_seq = last_seq + 1
                WHERE id = $1
                RETURNING last_seq
                """,
                room_id,
            )
            if seq is None:
                raise NotFoundError(f"room {room_id} not found")
            row = await conn.fetchrow(
                """
                INSERT INTO messages (id, room_id, author_id, body, seq)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id, room_id, author_id, body, seq, created_at
                """,
                uuid4(),
                room_id,
                author_id,
                body,
                seq,
            )
    assert row is not None
    return _message(row)


async def list_messages(
    pool: asyncpg.Pool,
    room_id: UUID,
    since_seq: int = 0,
) -> list[MessageRow]:
    exists = await pool.fetchval("SELECT 1 FROM rooms WHERE id = $1", room_id)
    if exists is None:
        raise NotFoundError(f"room {room_id} not found")
    rows = await pool.fetch(
        """
        SELECT id, room_id, author_id, body, seq, created_at
        FROM messages
        WHERE room_id = $1 AND seq > $2
        ORDER BY seq ASC
        """,
        room_id,
        since_seq,
    )
    return [_message(r) for r in rows]


async def get_last_read(pool: asyncpg.Pool, agent_id: UUID, room_id: UUID) -> int:
    value = await pool.fetchval(
        """
        SELECT last_read_seq
        FROM conversation_reads
        WHERE agent_id = $1 AND room_id = $2
        """,
        agent_id,
        room_id,
    )
    return int(value) if value is not None else 0


async def set_last_read(
    pool: asyncpg.Pool,
    agent_id: UUID,
    room_id: UUID,
    last_read_seq: int,
) -> None:
    await pool.execute(
        """
        INSERT INTO conversation_reads (agent_id, room_id, last_read_seq)
        VALUES ($1, $2, $3)
        ON CONFLICT (agent_id, room_id)
        DO UPDATE SET last_read_seq = EXCLUDED.last_read_seq
        """,
        agent_id,
        room_id,
        last_read_seq,
    )


async def truncate_all(pool: asyncpg.Pool) -> None:
    await pool.execute("TRUNCATE rooms CASCADE")
