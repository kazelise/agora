from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI

from server.db import truncate_all
from server.main import create_app
from server.scheduler import Scheduler, TurnRecord


@dataclass
class Harness:
    app: FastAPI
    client: httpx.AsyncClient

    @property
    def scheduler(self) -> Scheduler:
        return self.app.state.scheduler


async def _wait_turns(
    scheduler: Scheduler,
    pred: Callable[[list[TurnRecord]], bool],
    timeout: float = 4.0,
) -> list[TurnRecord]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if pred(scheduler.turns):
            await scheduler.wait_idle()
            return list(scheduler.turns)
        await asyncio.sleep(0.05)
    raise AssertionError(f"timed out waiting for turns: {scheduler.turns}")


@pytest.fixture
async def harness(require_services: None) -> AsyncIterator[Harness]:
    app = create_app(stub_turns=True)
    async with app.router.lifespan_context(app):
        await truncate_all(app.state.pool)
        app.state.scheduler.turns.clear()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield Harness(app=app, client=client)


async def _setup_room(client: httpx.AsyncClient) -> tuple[UUID, UUID, UUID, UUID]:
    room = (await client.post("/rooms", json={"name": "wake-room"})).json()
    room_id = UUID(room["id"])
    human = (
        await client.post(
            f"/rooms/{room_id}/participants",
            json={"kind": "human", "name": "Ada"},
        )
    ).json()
    iris = (
        await client.post(
            f"/rooms/{room_id}/participants",
            json={"kind": "agent", "name": "Iris", "persona": "brief"},
        )
    ).json()
    marcus = (
        await client.post(
            f"/rooms/{room_id}/participants",
            json={"kind": "agent", "name": "Marcus", "persona": "brief"},
        )
    ).json()
    return room_id, UUID(human["id"]), UUID(iris["id"]), UUID(marcus["id"])


@pytest.mark.asyncio
async def test_human_post_wakes_both_agents_and_agent_post_excludes_author(
    harness: Harness,
) -> None:
    room_id, human_id, iris_id, marcus_id = await _setup_room(harness.client)

    posted = await harness.client.post(
        f"/rooms/{room_id}/messages",
        json={"author_id": str(human_id), "body": "hello team"},
    )
    assert posted.status_code == 200
    assert posted.json()["seq"] == 1

    first = await _wait_turns(
        harness.scheduler,
        lambda turns: {t.agent_id for t in turns} == {iris_id, marcus_id},
    )
    assert {t.agent_id for t in first} == {iris_id, marcus_id}
    assert all(t.inbox_count == 1 for t in first)

    before = len(harness.scheduler.turns)
    as_iris = await harness.client.post(
        f"/rooms/{room_id}/messages",
        json={"author_id": str(iris_id), "body": "Iris here"},
    )
    assert as_iris.status_code == 200

    later = await _wait_turns(
        harness.scheduler,
        lambda turns: any(t.agent_id == marcus_id for t in turns[before:]),
    )
    new = later[before:]
    assert {t.agent_id for t in new} == {marcus_id}
    assert iris_id not in {t.agent_id for t in new}
