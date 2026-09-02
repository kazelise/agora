"""In-process World: the cloud host's asyncpg pool + Redis.

Behavior is the pre-BYOA path. Used by the server-side brain lane and
by existing tests (construct DirectWorld(pool, redis)).
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from uuid import UUID

import asyncpg
import redis.asyncio as redis

from brain.world import (
    DecisionResult,
    DuplicateReply,
    StaleWrite,
    TurnContext,
    WorldMessage,
    WorldParticipant,
)
from server import db
from server.models import MessageRow, ParticipantRow

logger = logging.getLogger("agora.world")

# (room_id, target_id) or (room_id, target_id, trigger_seq).
CallOnFn = Callable[..., Awaitable[None]]


def _participant(row: ParticipantRow) -> WorldParticipant:
    return WorldParticipant(
        id=row.id,
        name=row.name,
        persona=row.persona,
        kind=row.kind,
        computer_id=row.computer_id,
        role=row.role,
    )


def _message(row: MessageRow, author_name: str) -> WorldMessage:
    return WorldMessage(
        id=row.id,
        room_id=row.room_id,
        author_id=row.author_id,
        body=row.body,
        seq=row.seq,
        created_at=row.created_at,
        author_name=author_name,
    )


class DirectWorld:
    def __init__(
        self,
        pool: asyncpg.Pool,
        redis_client: redis.Redis,
        *,
        on_call_on: CallOnFn | None = None,
    ) -> None:
        self.pool = pool
        self.redis = redis_client
        # Server wires this to Scheduler.wake_one(..., called_on_seq=).
        # Tests pass a spy. Absent: call_on still writes the row.
        self.on_call_on = on_call_on

    async def _names(self, room_id: UUID) -> dict[str, str]:
        people = await db.list_participants(self.pool, room_id)
        return {str(p.id): p.name for p in people}

    async def _named_messages(
        self, room_id: UUID, since_seq: int
    ) -> list[WorldMessage]:
        rows = await db.list_messages(self.pool, room_id, since_seq=since_seq)
        names = await self._names(room_id)
        return [_message(row, names.get(str(row.author_id), "?")) for row in rows]

    async def load_turn(
        self,
        agent_id: UUID,
        room_id: UUID,
        *,
        called_on_seq: int | None = None,
    ) -> TurnContext:
        agent = await db.get_participant(self.pool, agent_id)
        last_read = await db.get_last_read(self.pool, agent_id, room_id)
        people = await db.list_participants(self.pool, room_id)
        names = {str(p.id): p.name for p in people}
        rows = await db.list_messages(self.pool, room_id, since_seq=last_read)
        inbox = [_message(row, names.get(str(row.author_id), "?")) for row in rows]
        seen_seq = max((m.seq for m in inbox), default=last_read)
        stretch = await db.count_agent_only_stretch(self.pool, room_id)
        try:
            room = await db.get_room(self.pool, room_id)
            room_mode = room.mode
        except db.NotFoundError:
            room_mode = "open"
        # In-process lanes pass the dict value here. Never read Redis:
        # that hint is for hosts that cannot receive the in-process
        # value (BYOA / K8s Jobs via /runtime/turn-context).
        return TurnContext(
            agent=_participant(agent),
            inbox=inbox,
            participants=[_participant(p) for p in people],
            last_read_seq=last_read,
            seen_seq=seen_seq,
            agent_only_stretch=stretch,
            room_mode=room_mode,
            called_on=called_on_seq is not None,
            called_on_seq=called_on_seq,
        )

    async def get_last_read(self, agent_id: UUID, room_id: UUID) -> int:
        return await db.get_last_read(self.pool, agent_id, room_id)

    async def set_last_read(
        self, agent_id: UUID, room_id: UUID, last_read_seq: int
    ) -> None:
        await db.set_last_read(self.pool, agent_id, room_id, last_read_seq)

    async def get_room_last_seq(self, room_id: UUID) -> int:
        return await db.get_room_last_seq(self.pool, room_id)

    async def list_messages_since(
        self, room_id: UUID, since_seq: int
    ) -> list[WorldMessage]:
        return await self._named_messages(room_id, since_seq)

    async def insert_message(
        self,
        room_id: UUID,
        author_id: UUID,
        body: str,
        *,
        not_after_seq: int,
    ) -> WorldMessage:
        try:
            row = await db.insert_message(
                self.pool,
                room_id,
                author_id,
                body,
                not_after_seq=not_after_seq,
            )
        except db.StaleWriteError as exc:
            newer = await self._named_messages(room_id, not_after_seq)
            raise StaleWrite(exc.last_seq, newer) from exc
        except db.DuplicateReplyError as exc:
            raise DuplicateReply(exc.peer_seq) from exc
        names = await self._names(room_id)
        return _message(row, names.get(str(row.author_id), "?"))

    async def try_claim(
        self, room_id: UUID, task_key: str, claimed_by: UUID
    ) -> bool:
        return await db.try_claim(self.pool, room_id, task_key, claimed_by)

    async def release_claim(
        self, room_id: UUID, task_key: str, claimed_by: UUID
    ) -> bool:
        return await db.release_claim(self.pool, room_id, task_key, claimed_by)

    async def record_llm_call(
        self,
        agent_id: UUID,
        room_id: UUID,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        purpose: str,
    ) -> None:
        await db.insert_llm_call(
            self.pool,
            agent_id,
            room_id,
            model,
            prompt_tokens,
            completion_tokens,
            purpose,
        )

    async def record_decision(
        self,
        room_id: UUID,
        moderator_id: UUID,
        trigger_seq: int,
        action: str,
        target_id: UUID | None,
    ) -> DecisionResult:
        status, row = await db.record_decision(
            self.pool, room_id, moderator_id, trigger_seq, action, target_id
        )
        result = DecisionResult(
            status=status, action=row.action, target_id=row.target_id
        )
        # Wake only on a freshly won call_on. A replay must not double-wake
        # (the unique key already decided who was called). The row is
        # already committed: a wake failure is a missed coordination
        # signal, same class as publish_wake. _undelivered_call_on
        # redelivers this call_on on the next stall nudge if the
        # target's last_read still trails trigger_seq.
        if (
            status == "won"
            and row.action == "call_on"
            and row.target_id is not None
            and self.on_call_on is not None
        ):
            try:
                await self._invoke_on_call_on(room_id, row.target_id, row.trigger_seq)
            except Exception:
                logger.warning(
                    "on_call_on wake failed room=%s target=%s — fail-open",
                    room_id,
                    row.target_id,
                    exc_info=True,
                )
        return result

    async def _invoke_on_call_on(
        self, room_id: UUID, target_id: UUID, trigger_seq: int
    ) -> None:
        assert self.on_call_on is not None
        try:
            await self.on_call_on(room_id, target_id, trigger_seq)
        except TypeError:
            await self.on_call_on(room_id, target_id)

    async def has_authored_since(
        self, agent_id: UUID, room_id: UUID, since_seq: int
    ) -> bool:
        return await db.has_authored_since(self.pool, agent_id, room_id, since_seq)
