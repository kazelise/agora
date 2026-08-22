"""Phase 5: two workers, one Redis — presence, broadcast, one dispatch."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI

from server.db import truncate_all
from server.main import create_app
from tests.asgi_ws import connect_asgi_ws


class Pair:
    def __init__(self, a: FastAPI, b: FastAPI, ca: httpx.AsyncClient, cb: httpx.AsyncClient) -> None:
        self.a = a
        self.b = b
        self.ca = ca
        self.cb = cb


@pytest.fixture
async def pair(require_services: None) -> AsyncIterator[Pair]:
    app_a = create_app(stub_turns=True)
    app_b = create_app(stub_turns=True)
    async with app_a.router.lifespan_context(app_a):
        await truncate_all(app_a.state.pool)
        async with app_b.router.lifespan_context(app_b):
            ta = httpx.ASGITransport(app=app_a)
            tb = httpx.ASGITransport(app=app_b)
            async with (
                httpx.AsyncClient(transport=ta, base_url="http://a") as ca,
                httpx.AsyncClient(transport=tb, base_url="http://b") as cb,
            ):
                yield Pair(app_a, app_b, ca, cb)


async def _wait(pred, timeout: float = 4.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if pred():
            return
        await asyncio.sleep(0.05)
    raise AssertionError("timed out")


@pytest.mark.asyncio
async def test_one_wake_one_turn_across_two_workers(pair: Pair) -> None:
    room = (await pair.ca.post("/rooms", json={"name": "cluster"})).json()
    human = (
        await pair.ca.post(
            f"/rooms/{room['id']}/participants",
            json={"kind": "human", "name": "Ada"},
        )
    ).json()
    iris = (
        await pair.ca.post(
            f"/rooms/{room['id']}/participants",
            json={"kind": "agent", "name": "Iris"},
        )
    ).json()
    marcus = (
        await pair.ca.post(
            f"/rooms/{room['id']}/participants",
            json={"kind": "agent", "name": "Marcus"},
        )
    ).json()
    posted = await pair.ca.post(
        f"/rooms/{room['id']}/messages",
        json={"author_id": human["id"], "body": "hello both workers"},
    )
    assert posted.status_code == 200

    await _wait(
        lambda: len(pair.a.state.scheduler.turns) + len(pair.b.state.scheduler.turns)
        >= 2
    )
    await pair.a.state.scheduler.wait_idle()
    await pair.b.state.scheduler.wait_idle()

    turns = [*pair.a.state.scheduler.turns, *pair.b.state.scheduler.turns]
    assert {t.agent_id for t in turns} == {UUID(iris["id"]), UUID(marcus["id"])}
    assert len(turns) == 2


@pytest.mark.asyncio
async def test_room_ws_on_other_worker_gets_the_message(pair: Pair) -> None:
    room = (await pair.ca.post("/rooms", json={"name": "fanout"})).json()
    human = (
        await pair.ca.post(
            f"/rooms/{room['id']}/participants",
            json={"kind": "human", "name": "Ada"},
        )
    ).json()
    ws = await connect_asgi_ws(pair.b, f"/ws/rooms/{room['id']}")
    try:
        posted = await pair.ca.post(
            f"/rooms/{room['id']}/messages",
            json={"author_id": human["id"], "body": "across the aisle"},
        )
        assert posted.status_code == 200
        frame = await ws.receive_json(timeout=4.0)
        assert frame["type"] == "message"
        assert frame["body"] == "across the aisle"
        assert frame["seq"] == 1
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_presence_visible_on_the_other_worker(pair: Pair) -> None:
    computer = (await pair.ca.post("/computers", json={"name": "laptop"})).json()
    listed = await pair.cb.get("/computers")
    assert listed.json()[0]["online"] is False

    ws = await connect_asgi_ws(
        pair.b,
        f"/ws/computers/{computer['id']}",
        query_string=f"token={computer['token']}",
    )
    try:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 4.0
        online = False
        while loop.time() < deadline:
            rows = (await pair.ca.get("/computers")).json()
            if rows and rows[0]["online"]:
                online = True
                break
            await asyncio.sleep(0.05)
        assert online is True
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_byoa_wake_reaches_socket_on_the_other_worker(pair: Pair) -> None:
    computer = (await pair.ca.post("/computers", json={"name": "laptop"})).json()
    room = (await pair.ca.post("/rooms", json={"name": "byoa-cluster"})).json()
    human = (
        await pair.ca.post(
            f"/rooms/{room['id']}/participants",
            json={"kind": "human", "name": "Ada"},
        )
    ).json()
    byoa = (
        await pair.ca.post(
            f"/rooms/{room['id']}/participants",
            json={
                "kind": "agent",
                "name": "Jules",
                "computer_id": computer["id"],
            },
        )
    ).json()
    ws = await connect_asgi_ws(
        pair.b,
        f"/ws/computers/{computer['id']}",
        query_string=f"token={computer['token']}",
    )
    try:
        posted = await pair.ca.post(
            f"/rooms/{room['id']}/messages",
            json={"author_id": human["id"], "body": "wake the other worker"},
        )
        assert posted.status_code == 200
        frame = await ws.receive_json(timeout=4.0)
        assert frame["type"] == "wake"
        assert frame["agent_id"] == byoa["id"]
        assert frame["room_id"] == room["id"]
    finally:
        await ws.close()
