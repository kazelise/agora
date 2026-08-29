"""Host-agnostic I/O surface for the brain.

The graph never talks to Postgres or Redis directly. A World implementation
is the only side-effect boundary: the in-process cloud host uses DirectWorld
(pool + redis); a BYOA daemon uses HttpWorld (token-authed HTTP). Swapping
the host swaps the transport and who holds the model key — not the graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from uuid import UUID


class StaleWrite(Exception):
    """INSERT refused: room last_seq moved past the writer's seen cursor.

    `last_seq` is the pre-increment room value that failed the check.
    `newer` is the messages with seq > the writer's not_after_seq, when
    the host already has them (runtime 409, or DirectWorld after the
    transactional reject). HOLD can apply them without another fetch.
    """

    def __init__(self, last_seq: int, newer: list[WorldMessage] | None = None) -> None:
        super().__init__(f"room last_seq {last_seq} is after the write's seen cursor")
        self.last_seq = last_seq
        self.newer = newer


class DuplicateReply(Exception):
    """INSERT refused: body verbatim-duplicates the latest peer message.

    Non-bypassable (no flag routes around it): the graph re-decides inside
    the same turn — either rewrite the reply or yield.
    """

    def __init__(self, peer_seq: int) -> None:
        super().__init__(f"message duplicates peer seq {peer_seq} verbatim")
        self.peer_seq = peer_seq


@dataclass(frozen=True)
class WorldMessage:
    id: UUID
    room_id: UUID
    author_id: UUID
    body: str
    seq: int
    created_at: datetime
    author_name: str = ""

    def as_ws(self) -> dict:
        return {
            "type": "message",
            "id": str(self.id),
            "room_id": str(self.room_id),
            "author_id": str(self.author_id),
            "body": self.body,
            "seq": self.seq,
            "created_at": self.created_at.isoformat(),
        }


@dataclass(frozen=True)
class WorldParticipant:
    id: UUID
    name: str
    persona: str | None
    kind: str = "agent"
    computer_id: UUID | None = None


@dataclass(frozen=True)
class TurnContext:
    agent: WorldParticipant
    inbox: list[WorldMessage]
    participants: list[WorldParticipant]
    last_read_seq: int
    seen_seq: int
    # Room-level count of agent messages since the last human message
    # (the loop-cap cursor; 0 when the latest message is human's). The
    # graph's agent↔agent loop cap reads this, never the inbox size:
    # the inbox is a per-turn delivery batch, the stretch is the room's
    # conversation state. Default 0 keeps ad-hoc TurnContexts valid.
    agent_only_stretch: int = 0


class World(Protocol):
    async def load_turn(self, agent_id: UUID, room_id: UUID) -> TurnContext: ...

    async def get_last_read(self, agent_id: UUID, room_id: UUID) -> int: ...

    async def set_last_read(
        self, agent_id: UUID, room_id: UUID, last_read_seq: int
    ) -> None: ...

    async def get_room_last_seq(self, room_id: UUID) -> int: ...

    async def list_messages_since(
        self, room_id: UUID, since_seq: int
    ) -> list[WorldMessage]: ...

    async def insert_message(
        self,
        room_id: UUID,
        author_id: UUID,
        body: str,
        *,
        not_after_seq: int,
    ) -> WorldMessage: ...

    async def try_claim(
        self, room_id: UUID, task_key: str, claimed_by: UUID
    ) -> bool: ...

    async def release_claim(
        self, room_id: UUID, task_key: str, claimed_by: UUID
    ) -> bool: ...

    async def record_llm_call(
        self,
        agent_id: UUID,
        room_id: UUID,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        purpose: str,
    ) -> None: ...

    async def record_seen(self, agent_id: UUID, room_id: UUID, seq: int) -> None: ...
