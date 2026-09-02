from __future__ import annotations

from pathlib import Path
from typing import Literal, cast
from uuid import UUID, uuid4

import asyncpg

from server.computers import hash_token
from server.models import (
    ComputerRow,
    DecisionRow,
    MessageRow,
    ParticipantKind,
    ParticipantRole,
    ParticipantRow,
    RoomMode,
    RoomRow,
)

_PARTICIPANT_COLS = (
    "id, room_id, kind, name, persona, computer_id, role, created_at"
)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class NotFoundError(Exception):
    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class DuplicateModeratorError(Exception):
    """A room may have at most one moderator seat."""

    def __init__(self, detail: str = "room already has a moderator") -> None:
        super().__init__(detail)
        self.detail = detail


class InvalidParticipantRoleError(Exception):
    """role=moderator is an agent seat; a human cannot hold it."""

    def __init__(self, detail: str = "role=moderator requires kind=agent") -> None:
        super().__init__(detail)
        self.detail = detail


class DuplicateParticipantNameError(Exception):
    """Names address @-mentions and call_on targets; one per room."""

    def __init__(self, detail: str = "participant name already used in this room") -> None:
        super().__init__(detail)
        self.detail = detail


class InvalidTriggerSeqError(Exception):
    """trigger_seq must be a committed room seq (1..last_seq)."""

    def __init__(self, detail: str = "trigger_seq is outside the room's committed seq range") -> None:
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
    mode = row["mode"] if "mode" in row.keys() else "open"
    return RoomRow(
        id=row["id"],
        name=row["name"],
        created_at=row["created_at"],
        mode=str(mode or "open"),
    )


def _participant(row: asyncpg.Record) -> ParticipantRow:
    role = row["role"] if "role" in row.keys() else "member"
    return ParticipantRow(
        id=row["id"],
        room_id=row["room_id"],
        kind=cast(ParticipantKind, row["kind"]),
        name=row["name"],
        persona=row["persona"],
        created_at=row["created_at"],
        computer_id=row["computer_id"],
        role=str(role or "member"),
    )


def _decision(row: asyncpg.Record) -> DecisionRow:
    return DecisionRow(
        id=row["id"],
        room_id=row["room_id"],
        moderator_id=row["moderator_id"],
        trigger_seq=int(row["trigger_seq"]),
        action=str(row["action"]),
        target_id=row["target_id"],
        created_at=row["created_at"],
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


async def create_room(
    pool: asyncpg.Pool, name: str, mode: RoomMode = "open"
) -> RoomRow:
    row = await pool.fetchrow(
        """
        INSERT INTO rooms (id, name, mode)
        VALUES ($1, $2, $3)
        RETURNING id, name, created_at, mode
        """,
        uuid4(),
        name,
        mode,
    )
    assert row is not None
    return _room(row)


async def get_room(pool: asyncpg.Pool, room_id: UUID) -> RoomRow:
    row = await pool.fetchrow(
        "SELECT id, name, created_at, mode FROM rooms WHERE id = $1",
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
    role: ParticipantRole = "member",
) -> ParticipantRow:
    async with pool.acquire() as conn:
        async with conn.transaction():
            exists = await conn.fetchval("SELECT 1 FROM rooms WHERE id = $1", room_id)
            if exists is None:
                raise NotFoundError(f"room {room_id} not found")
            if role == "moderator" and kind != "agent":
                raise InvalidParticipantRoleError()
            if computer_id is not None:
                hosted = await conn.fetchval(
                    "SELECT 1 FROM computers WHERE id = $1", computer_id
                )
                if hosted is None:
                    raise NotFoundError(f"computer {computer_id} not found")
            if role == "moderator":
                taken = await conn.fetchval(
                    """
                    SELECT 1 FROM participants
                    WHERE room_id = $1 AND role = 'moderator'
                    """,
                    room_id,
                )
                if taken is not None:
                    raise DuplicateModeratorError()
            try:
                row = await conn.fetchrow(
                    f"""
                    INSERT INTO participants
                        (id, room_id, kind, name, persona, computer_id, role)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    RETURNING {_PARTICIPANT_COLS}
                    """,
                    uuid4(),
                    room_id,
                    kind,
                    name,
                    persona,
                    computer_id,
                    role,
                )
            except asyncpg.UniqueViolationError as exc:
                constraint = exc.constraint_name or ""
                if "one_moderator" in constraint:
                    raise DuplicateModeratorError() from exc
                raise DuplicateParticipantNameError() from exc
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
        ORDER BY created_at, id
        """,
        room_id,
    )
    return [_participant(r) for r in rows]


async def list_active_rooms(pool: asyncpg.Pool) -> list[Any]:
    """Every room with its latest message, for the stall sweep.

    One query instead of N+1 per room: the LATERAL join fetches each
    room's newest row (with the author's kind) alongside the room row.
    Rooms with no messages are skipped by the sweep (nothing can be owed
    about silence).
    """
    return await pool.fetch(
        """
        SELECT r.id, r.name, r.created_at,
               latest.created_at AS last_at,
               latest.seq, latest.body, latest.author_id, latest.author_kind
        FROM rooms r
        LEFT JOIN LATERAL (
            SELECT m.seq, m.body, m.created_at, m.author_id, p.kind AS author_kind
            FROM messages m
            JOIN participants p ON p.id = m.author_id
            WHERE m.room_id = r.id
            ORDER BY m.seq DESC
            LIMIT 1
        ) latest ON true
        WHERE latest.seq IS NOT NULL
        """
    )


async def count_agent_only_stretch(pool: asyncpg.Pool, room_id: UUID) -> int:
    """Agent messages since the room's last human message (loop-cap cursor).

    Room-level, not per-inbox: the inbox is this turn's delivery batch, the
    stretch is the room's conversation state — a quick ping-pong keeps each
    inbox tiny while the room circles, so the cap must count across turns.
    One statement, ordered by the gapless per-room seq. Agents only: a
    human message resets the count by definition; a room with no human
    message yet (the MOST loop-prone shape) counts everything.
    """
    return int(
        await pool.fetchval(
            """
            SELECT COUNT(*) FILTER (WHERE author_kind = 'agent')
            FROM (
                SELECT p.kind AS author_kind
                FROM messages m
                JOIN participants p ON p.id = m.author_id
                WHERE m.room_id = $1
                  AND m.seq > COALESCE((
                        SELECT m2.seq
                        FROM messages m2
                        JOIN participants p2 ON p2.id = m2.author_id
                        WHERE m2.room_id = $1 AND p2.kind = 'human'
                        ORDER BY m2.seq DESC
                        LIMIT 1
                  ), 0)
            ) stretch
            """,
            room_id,
        )
        or 0
    )


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


class DuplicateReplyError(Exception):
    """INSERT refused: body is verbatim-identical to the latest peer message.

    Checked inside the same room-row-locked transaction as the seq claim,
    so it sees committed peers and is race-free (the TOCTOU window that
    defeats a pre-INSERT check). There is no legitimate use case for
    repeating the immediately-prior peer message verbatim — not even a
    HOLD override — so this gate is non-bypassable.
    """

    def __init__(self, peer_seq: int) -> None:
        super().__init__(f"message duplicates peer seq {peer_seq} verbatim")
        self.peer_seq = peer_seq


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
            # One query for both the membership check and the dup-gate's
            # agent/human dispatch (the two reads cannot disagree: kind is
            # immutable in this repo — no code path UPDATEs or DELETEs it).
            kind = await conn.fetchval(
                """
                SELECT kind FROM participants
                WHERE id = $1 AND room_id = $2
                """,
                author_id,
                room_id,
            )
            if kind is None:
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
            # Verbatim-dup gate, inside the row lock: two agents composing
            # the same "3" both pass any pre-transaction snapshot, but only
            # the first INSERT commits; the second sees it here. Agents
            # only — a human echoing the number (grading, joining in) is
            # a legitimate move, and humans POST through the API directly.
            peer = None
            if kind == "agent":
                peer = await conn.fetchrow(
                    """
                    SELECT body, seq FROM messages
                    WHERE room_id = $1 AND author_id <> $2
                    ORDER BY seq DESC
                    LIMIT 1
                    """,
                    room_id,
                    author_id,
                )
            if (
                peer is not None
                and peer["body"].strip() == body.strip()
                and body.strip() != ""
            ):
                raise DuplicateReplyError(int(peer["seq"]))
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


async def get_message_by_seq(
    pool: asyncpg.Pool, room_id: UUID, seq: int
) -> MessageRow | None:
    row = await pool.fetchrow(
        """
        SELECT id, room_id, author_id, body, seq, created_at
        FROM messages
        WHERE room_id = $1 AND seq = $2
        """,
        room_id,
        seq,
    )
    return None if row is None else _message(row)


async def record_decision(
    pool: asyncpg.Pool,
    room_id: UUID,
    moderator_id: UUID,
    trigger_seq: int,
    action: str,
    target_id: UUID | None,
) -> tuple[Literal["won", "already_decided"], DecisionRow]:
    """Insert the decision, or return the existing row for this trigger.

    UNIQUE (room_id, trigger_seq) makes a moderator rerun for the same
    trigger idempotent: the second attempt is "already decided", not an
    error. ON CONFLICT DO NOTHING + a follow-up SELECT is one critical
    section under the unique index — two concurrent inserts cannot both
    report won. trigger_seq must already be a committed room seq so a
    daemon cannot pre-book future keys.
    """
    async with pool.acquire() as conn:
        async with conn.transaction():
            last_seq = await conn.fetchval(
                "SELECT last_seq FROM rooms WHERE id = $1 FOR SHARE",
                room_id,
            )
            if last_seq is None:
                raise NotFoundError(f"room {room_id} not found")
            if trigger_seq <= 0 or int(trigger_seq) > int(last_seq):
                raise InvalidTriggerSeqError(
                    f"trigger_seq {trigger_seq} is outside 1..{int(last_seq)}"
                )
            row = await conn.fetchrow(
                """
                INSERT INTO moderator_decisions (
                    id, room_id, moderator_id, trigger_seq, action, target_id
                )
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (room_id, trigger_seq) DO NOTHING
                RETURNING id, room_id, moderator_id, trigger_seq, action,
                          target_id, created_at
                """,
                uuid4(),
                room_id,
                moderator_id,
                trigger_seq,
                action,
                target_id,
            )
            if row is not None:
                return "won", _decision(row)
            existing = await conn.fetchrow(
                """
                SELECT id, room_id, moderator_id, trigger_seq, action,
                       target_id, created_at
                FROM moderator_decisions
                WHERE room_id = $1 AND trigger_seq = $2
                """,
                room_id,
                trigger_seq,
            )
    assert existing is not None
    return "already_decided", _decision(existing)


async def get_latest_decision(
    pool: asyncpg.Pool, room_id: UUID
) -> DecisionRow | None:
    row = await pool.fetchrow(
        """
        SELECT id, room_id, moderator_id, trigger_seq, action,
               target_id, created_at
        FROM moderator_decisions
        WHERE room_id = $1
        ORDER BY created_at DESC, trigger_seq DESC
        LIMIT 1
        """,
        room_id,
    )
    return None if row is None else _decision(row)


async def list_decisions(pool: asyncpg.Pool, room_id: UUID) -> list[DecisionRow]:
    rows = await pool.fetch(
        """
        SELECT id, room_id, moderator_id, trigger_seq, action,
               target_id, created_at
        FROM moderator_decisions
        WHERE room_id = $1
        ORDER BY trigger_seq ASC
        """,
        room_id,
    )
    return [_decision(r) for r in rows]


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


async def count_call_ons_since_human(pool: asyncpg.Pool, room_id: UUID) -> int:
    """call_on rows since the last human message (decision-layer poll cap).

    Floor is last human seq minus one so the call_on that answers that
    message (trigger_seq == human seq) counts; no human yet counts from
    seq 0, matching count_agent_only_stretch. Mentions are not rows.
    """
    return int(
        await pool.fetchval(
            """
            SELECT COUNT(*)
            FROM moderator_decisions d
            WHERE d.room_id = $1
              AND d.action = 'call_on'
              AND d.trigger_seq > COALESCE((
                    SELECT m.seq - 1
                    FROM messages m
                    JOIN participants p ON p.id = m.author_id
                    WHERE m.room_id = $1 AND p.kind = 'human'
                    ORDER BY m.seq DESC
                    LIMIT 1
              ), 0)
            """,
            room_id,
        )
        or 0
    )


async def has_authored_since(
    pool: asyncpg.Pool, author_id: UUID, room_id: UUID, since_seq: int
) -> bool:
    """True if this author has a committed message with seq > since_seq."""
    return bool(
        await pool.fetchval(
            """
            SELECT EXISTS(
                SELECT 1 FROM messages
                WHERE room_id = $1 AND author_id = $2 AND seq > $3
            )
            """,
            room_id,
            author_id,
            since_seq,
        )
    )


async def get_room_last_seq(pool: asyncpg.Pool, room_id: UUID) -> int:
    value = await pool.fetchval("SELECT last_seq FROM rooms WHERE id = $1", room_id)
    if value is None:
        raise NotFoundError(f"room {room_id} not found")
    return int(value)


CLAIM_TTL_S = 300.0
"""Claims older than this may be stolen by another agent.

A claim is a coordination signal, not a correctness invariant: its only
job is to keep two agents from doing the same task concurrently. The
normal leak valves are in-graph (the obligation nudge, the end-of-turn
release), but a process that crashes mid-turn leaves its won claim
pinned forever — release_claim only runs at the end of a turn. After
the TTL another agent may steal it atomically; the old winner's in-flight
reply, if any, still lands (dup-gate and freshness gate remain the
correctness boundary), it just no longer holds the lock.
"""


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
        ON CONFLICT (room_id, task_key) DO UPDATE
            SET claimed_by = EXCLUDED.claimed_by,
                created_at = EXCLUDED.created_at
            WHERE claims.claimed_by = EXCLUDED.claimed_by
               OR claims.created_at < now() - make_interval(secs => $5)
        RETURNING id
        """,
        uuid4(),
        room_id,
        task_key,
        claimed_by,
        CLAIM_TTL_S,
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


async def room_digest(
    pool: asyncpg.Pool,
    room_id: UUID,
) -> dict | None:
    """Everything the markdown digest needs, in a handful of independent
    queries (room + roster, transcript, claims, spend; decisions only
    when the room is moderated). No shared snapshot — same as before.
    """
    room = await pool.fetchrow(
        "SELECT id, name, created_at, mode FROM rooms WHERE id = $1", room_id
    )
    if room is None:
        return None
    people = await pool.fetch(
        """
        SELECT id, kind, name, role FROM participants
        WHERE room_id = $1
        ORDER BY created_at
        """,
        room_id,
    )
    messages = await pool.fetch(
        """
        SELECT m.seq, m.body, m.created_at, m.author_id, p.name AS author_name, p.kind
        FROM messages m
        JOIN participants p ON p.id = m.author_id
        WHERE m.room_id = $1
        ORDER BY m.seq ASC
        """,
        room_id,
    )
    claims = await pool.fetch(
        """
        SELECT c.task_key, c.claimed_by, c.created_at, p.name AS claimed_by_name
        FROM claims c
        JOIN participants p ON p.id = c.claimed_by
        WHERE c.room_id = $1
        ORDER BY c.created_at ASC
        """,
        room_id,
    )
    usage = await pool.fetch(
        """
        SELECT purpose, model,
               SUM(prompt_tokens) AS prompt_tokens,
               SUM(completion_tokens) AS completion_tokens,
               COUNT(*) AS calls
        FROM llm_calls
        WHERE room_id = $1
        GROUP BY purpose, model
        ORDER BY purpose, model
        """,
        room_id,
    )
    mode = room["mode"] if "mode" in room.keys() else "open"
    decisions = (
        await list_decisions(pool, room_id) if str(mode or "open") == "moderated" else None
    )
    return {
        "room": room,
        "participants": people,
        "messages": messages,
        "claims": claims,
        "usage": usage,
        "decisions": decisions,
    }
