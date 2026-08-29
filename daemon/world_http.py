"""HTTP World: the daemon's only view of the server.

Talks to /runtime/* with a computer token. Redis and Postgres stay on
the server; this process depends on httpx, not the infra drivers.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

import httpx

from brain.world import (
    DuplicateReply,
    StaleWrite,
    TurnContext,
    WorldMessage,
    WorldParticipant,
)


def _dt(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _uuid(value: str | UUID) -> UUID:
    return value if isinstance(value, UUID) else UUID(str(value))


def _message(data: dict[str, Any]) -> WorldMessage:
    return WorldMessage(
        id=_uuid(data["id"]),
        room_id=_uuid(data["room_id"]),
        author_id=_uuid(data["author_id"]),
        body=data["body"],
        seq=int(data["seq"]),
        created_at=_dt(data["created_at"]),
        author_name=str(data.get("author_name") or ""),
    )


def _participant(data: dict[str, Any]) -> WorldParticipant:
    raw_computer = data.get("computer_id")
    return WorldParticipant(
        id=_uuid(data["id"]),
        name=data["name"],
        persona=data.get("persona"),
        kind=str(data.get("kind") or "agent"),
        computer_id=_uuid(raw_computer) if raw_computer else None,
    )


class HttpWorld:
    def __init__(self, client: httpx.AsyncClient, token: str) -> None:
        self._client = client
        self._token = token
        self._actor: UUID | None = None

    def bind_actor(self, agent_id: UUID) -> None:
        self._actor = agent_id

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"}

    def _actor_id(self, agent_id: UUID | None = None) -> UUID:
        if agent_id is not None:
            self._actor = agent_id
            return agent_id
        if self._actor is None:
            raise RuntimeError("HttpWorld has no acting agent; call load_turn first")
        return self._actor

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> httpx.Response:
        resp = await self._client.request(
            method,
            path,
            params=params,
            json=json,
            headers=self._headers(),
        )
        return resp

    def _raise_for_status(self, resp: httpx.Response) -> None:
        if resp.status_code == 409:
            detail = resp.json().get("detail") or {}
            if detail.get("error") == "duplicate_reply":
                raise DuplicateReply(int(detail["peer_seq"]))
            newer_raw = detail.get("messages") or []
            newer = [_message(item) for item in newer_raw]
            raise StaleWrite(int(detail["last_seq"]), newer)
        resp.raise_for_status()

    async def load_turn(self, agent_id: UUID, room_id: UUID) -> TurnContext:
        self._actor = agent_id
        resp = await self._request(
            "GET",
            "/runtime/turn-context",
            params={"agent_id": str(agent_id), "room_id": str(room_id)},
        )
        self._raise_for_status(resp)
        data = resp.json()
        inbox = [_message(item) for item in data.get("inbox") or []]
        people = [_participant(item) for item in data.get("participants") or []]
        return TurnContext(
            agent=WorldParticipant(
                id=agent_id,
                name=data["agent_name"],
                persona=data.get("persona") or "",
                kind="agent",
            ),
            inbox=inbox,
            participants=people,
            last_read_seq=int(data["last_read_seq"]),
            seen_seq=int(data["seen_seq"]),
        )

    async def get_last_read(self, agent_id: UUID, room_id: UUID) -> int:
        self._actor = agent_id
        resp = await self._request(
            "GET",
            "/runtime/last-read",
            params={"agent_id": str(agent_id), "room_id": str(room_id)},
        )
        self._raise_for_status(resp)
        return int(resp.json()["last_read_seq"])

    async def set_last_read(
        self, agent_id: UUID, room_id: UUID, last_read_seq: int
    ) -> None:
        self._actor = agent_id
        resp = await self._request(
            "PUT",
            "/runtime/last-read",
            json={
                "agent_id": str(agent_id),
                "room_id": str(room_id),
                "last_read_seq": last_read_seq,
            },
        )
        self._raise_for_status(resp)

    async def get_room_last_seq(self, room_id: UUID) -> int:
        agent_id = self._actor_id()
        resp = await self._request(
            "GET",
            "/runtime/room-seq",
            params={"agent_id": str(agent_id), "room_id": str(room_id)},
        )
        self._raise_for_status(resp)
        return int(resp.json()["last_seq"])

    async def list_messages_since(
        self, room_id: UUID, since_seq: int
    ) -> list[WorldMessage]:
        agent_id = self._actor_id()
        resp = await self._request(
            "GET",
            "/runtime/messages",
            params={
                "agent_id": str(agent_id),
                "room_id": str(room_id),
                "since_seq": since_seq,
            },
        )
        self._raise_for_status(resp)
        return [_message(item) for item in resp.json()]

    async def insert_message(
        self,
        room_id: UUID,
        author_id: UUID,
        body: str,
        *,
        not_after_seq: int,
    ) -> WorldMessage:
        self._actor = author_id
        resp = await self._request(
            "POST",
            "/runtime/reply",
            json={
                "agent_id": str(author_id),
                "room_id": str(room_id),
                "body": body,
                "not_after_seq": not_after_seq,
            },
        )
        self._raise_for_status(resp)
        data = resp.json()
        if "author_name" not in data:
            data = {**data, "author_name": ""}
        return _message(data)

    async def try_claim(
        self, room_id: UUID, task_key: str, claimed_by: UUID
    ) -> bool:
        self._actor = claimed_by
        resp = await self._request(
            "POST",
            "/runtime/claim",
            json={
                "agent_id": str(claimed_by),
                "room_id": str(room_id),
                "task_key": task_key,
            },
        )
        self._raise_for_status(resp)
        return bool(resp.json()["won"])

    async def release_claim(
        self, room_id: UUID, task_key: str, claimed_by: UUID
    ) -> bool:
        self._actor = claimed_by
        resp = await self._request(
            "POST",
            "/runtime/release-claim",
            json={
                "agent_id": str(claimed_by),
                "room_id": str(room_id),
                "task_key": task_key,
            },
        )
        self._raise_for_status(resp)
        return bool(resp.json()["released"])

    async def record_llm_call(
        self,
        agent_id: UUID,
        room_id: UUID,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        purpose: str,
    ) -> None:
        self._actor = agent_id
        resp = await self._request(
            "POST",
            "/runtime/llm-call",
            json={
                "agent_id": str(agent_id),
                "room_id": str(room_id),
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "purpose": purpose,
            },
        )
        self._raise_for_status(resp)

    async def record_seen(self, agent_id: UUID, room_id: UUID, seq: int) -> None:
        self._actor = agent_id
        resp = await self._request(
            "POST",
            "/runtime/seen",
            json={
                "agent_id": str(agent_id),
                "room_id": str(room_id),
                "seq": seq,
            },
        )
        self._raise_for_status(resp)
