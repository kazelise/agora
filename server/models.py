from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

ParticipantKind = Literal["human", "agent"]


class CreateRoomRequest(BaseModel):
    name: str = Field(min_length=1)


class CreateParticipantRequest(BaseModel):
    kind: ParticipantKind
    name: str = Field(min_length=1)
    persona: str | None = None


class CreateMessageRequest(BaseModel):
    author_id: UUID
    body: str = Field(min_length=1)


class RoomOut(BaseModel):
    id: UUID
    name: str
    created_at: datetime


class ParticipantOut(BaseModel):
    id: UUID
    room_id: UUID
    kind: ParticipantKind
    name: str
    persona: str | None
    created_at: datetime


class MessageOut(BaseModel):
    id: UUID
    room_id: UUID
    author_id: UUID
    body: str
    seq: int
    created_at: datetime


class MessageListOut(BaseModel):
    messages: list[MessageOut]


@dataclass(frozen=True)
class RoomRow:
    id: UUID
    name: str
    created_at: datetime


@dataclass(frozen=True)
class ParticipantRow:
    id: UUID
    room_id: UUID
    kind: ParticipantKind
    name: str
    persona: str | None
    created_at: datetime


@dataclass(frozen=True)
class MessageRow:
    id: UUID
    room_id: UUID
    author_id: UUID
    body: str
    seq: int
    created_at: datetime

    def as_out(self) -> MessageOut:
        return MessageOut(
            id=self.id,
            room_id=self.room_id,
            author_id=self.author_id,
            body=self.body,
            seq=self.seq,
            created_at=self.created_at,
        )

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
