from __future__ import annotations

from pathlib import Path
from typing import cast
from uuid import UUID, uuid4

import asyncpg

from server.computers import hash_token
from server.models import (
    ComputerRow,
    MessageRow,
    ParticipantKind,
    ParticipantRow,
    RoomRow,
)

_PARTICIPANT_COLS = (
    "id, room_id, kind, name, persona, computer_id, created_at"
)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class NotFoundError(Exception):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class StaleWriteError(Exception):
    """INSERT refused: room last_seq moved past the writer's seen cursor.

    Raised inside the same transaction that holds the room row lock, so
    the check and the would-be insert are one critical section. `last_seq`
    is the pre-increment value that failed the freshness test.
    """

    def __init__(self, last_seq: int) -> None:
        super().__init__(f"room last_seq {last_seq} is after the write's seen cursor")
        self.last_seq = last_seq


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
        computer_id=row["computer_id"],
    )


def _computer(row: asyncpg.Record) -> ComputerRow:
    return ComputerRow(
        id=row["id"],
        name=row["name"],
        created_at=row["created_at"],
        last_seen_at=row["last_seen_at"],
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
    computer_id: UUID | None = None,
) -> ParticipantRow:
    async with pool.acquire() as conn:
        async with conn.transaction():
            exists = await conn.fetchval("SELECT 1 FROM rooms WHERE id = $1", room_id)
            if exists is None:
                raise NotFoundError(f"room {room_id} not found")
            if computer_id is not None:
                hosted = await conn.fetchval(
                    "SELECT 1 FROM computers WHERE id = $1", computer_id
                )
                if hosted is None:
                    raise NotFoundError(f"computer {computer_id} not found")
            row = await conn.fetchrow(
                f"""
                INSERT INTO participants (id, room_id, kind, name, persona, computer_id)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING {_PARTICIPANT_COLS}
                """,
                uuid4(),
                room_id,
                kind,
                name,
                persona,
                computer_id,
            )
    assert row is not None
    return _participant(row)


async def get_participant(pool: asyncpg.Pool, participant_id: UUID) -> ParticipantRow:
    row = await pool.fetchrow(
        f"""
        SELECT {_PARTICIPANT_COLS}
        FROM participants
        WHERE id = $1
        """,
        participant_id,
    )
    if row is None:
        raise NotFoundError(f"participant {participant_id} not found")
    return _participant(row)


async def list_participants(pool: asyncpg.Pool, room_id: UUID) -> list[ParticipantRow]:
    rows = await pool.fetch(
        f"""
        SELECT {_PARTICIPANT_COLS}
        FROM participants
        WHERE room_id = $1
        ORDER BY created_at
        """,
        room_id,
    )
    return [_participant(r) for r in rows]


async def list_agent_participants(pool: asyncpg.Pool, room_id: UUID) -> list[ParticipantRow]:
    rows = await pool.fetch(
        f"""
        SELECT {_PARTICIPANT_COLS}
        FROM participants
        WHERE room_id = $1 AND kind = 'agent'
        ORDER BY created_at
        """,
        room_id,
    )
    return [_participant(r) for r in rows]


async def create_computer(pool: asyncpg.Pool, name: str, token: str) -> ComputerRow:
    row = await pool.fetchrow(
        """
        INSERT INTO computers (id, name, token_hash)
        VALUES ($1, $2, $3)
        RETURNING id, name, created_at, last_seen_at
        """,
        uuid4(),
        name,
        hash_token(token),
    )
    assert row is not None
    return _computer(row)


async def get_computer(pool: asyncpg.Pool, computer_id: UUID) -> ComputerRow:
    row = await pool.fetchrow(
        """
        SELECT id, name, created_at, last_seen_at
        FROM computers
        WHERE id = $1
        """,
        computer_id,
    )
    if row is None:
        raise NotFoundError(f"computer {computer_id} not found")
    return _computer(row)


async def get_computer_by_token(pool: asyncpg.Pool, token: str) -> ComputerRow | None:
    row = await pool.fetchrow(
        """
        SELECT id, name, created_at, last_seen_at
        FROM computers
        WHERE token_hash = $1
        """,
        hash_token(token),
    )
    return None if row is None else _computer(row)


async def computer_token_matches(
    pool: asyncpg.Pool, computer_id: UUID, token: str
) -> bool:
    stored = await pool.fetchval(
        "SELECT token_hash FROM computers WHERE id = $1",
        computer_id,
    )
    return stored is not None and stored == hash_token(token)


async def list_computers(pool: asyncpg.Pool) -> list[ComputerRow]:
    rows = await pool.fetch(
        """
        SELECT id, name, created_at, last_seen_at
        FROM computers
        ORDER BY created_at
        """
    )
    return [_computer(r) for r in rows]


async def touch_computer(pool: asyncpg.Pool, computer_id: UUID) -> None:
    await pool.execute(
        "UPDATE computers SET last_seen_at = now() WHERE id = $1",
        computer_id,
    )


async def insert_message(
    pool: asyncpg.Pool,
    room_id: UUID,
    author_id: UUID,
    body: str,
    *,
    not_after_seq: int | None = None,
) -> MessageRow:
    # Per-room counter, not a Postgres SEQUENCE: SEQUENCE is table-global
    # (or needs one sequence object per room). SELECT ... FOR UPDATE takes
    # the room row lock first; the later increment + INSERT share that
    # critical section. Rollback undoes the increment — seq stays gapless.
    # not_after_seq is the transactional freshness invariant: if another
    # writer already advanced last_seq past the caller's seen cursor,
    # raise StaleWriteError instead of inserting.
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
            current = await conn.fetchval(
                "SELECT last_seq FROM rooms WHERE id = $1 FOR UPDATE",
                room_id,
            )
            if current is None:
                raise NotFoundError(f"room {room_id} not found")
            if not_after_seq is not None and int(current) > not_after_seq:
                raise StaleWriteError(int(current))
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


async def get_room_last_seq(pool: asyncpg.Pool, room_id: UUID) -> int:
    value = await pool.fetchval("SELECT last_seq FROM rooms WHERE id = $1", room_id)
    if value is None:
        raise NotFoundError(f"room {room_id} not found")
    return int(value)


async def try_claim(
    pool: asyncpg.Pool,
    room_id: UUID,
    task_key: str,
    claimed_by: UUID,
) -> bool:
    row = await pool.fetchrow(
        """
        INSERT INTO claims (id, room_id, task_key, claimed_by)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (room_id, task_key) DO NOTHING
        RETURNING id
        """,
        uuid4(),
        room_id,
        task_key,
        claimed_by,
    )
    return row is not None


async def release_claim(
    pool: asyncpg.Pool,
    room_id: UUID,
    task_key: str,
    claimed_by: UUID,
) -> bool:
    """Drop a claim only if this agent still holds it.

    An unfulfilled winner must not pin the task_key forever — losers
    already yielded, and a dead lock starves the room.
    """
    row = await pool.fetchrow(
        """
        DELETE FROM claims
        WHERE room_id = $1 AND task_key = $2 AND claimed_by = $3
        RETURNING id
        """,
        room_id,
        task_key,
        claimed_by,
    )
    return row is not None


async def insert_llm_call(
    pool: asyncpg.Pool,
    agent_id: UUID,
    room_id: UUID,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    purpose: str,
) -> None:
    await pool.execute(
        """
        INSERT INTO llm_calls (
            id, agent_id, room_id, model, prompt_tokens, completion_tokens, purpose
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        """,
        uuid4(),
        agent_id,
        room_id,
        model,
        prompt_tokens,
        completion_tokens,
        purpose,
    )


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
    await pool.execute("TRUNCATE rooms, computers CASCADE")
