"""Phase 4b: runtime World, wake routing, and an in-process BYOA loop."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI

from brain.graph import Brain
from daemon.world_http import HttpWorld
from server.db import truncate_all
from server.main import create_app
from tests.asgi_ws import connect_asgi_ws
from tests.fakes import ScriptedChatModel, tool_call, triage_message


class Harness:
    def __init__(self, app: FastAPI, client: httpx.AsyncClient) -> None:
        self.app = app
        self.client = client


@pytest.fixture
async def harness(require_services: None) -> AsyncIterator[Harness]:
    app = create_app(stub_turns=True)
    async with app.router.lifespan_context(app):
        await truncate_all(app.state.pool)
        app.state.scheduler.turns.clear()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield Harness(app=app, client=client)


async def _pair(client: httpx.AsyncClient, name: str = "laptop") -> dict[str, Any]:
    resp = await client.post("/computers", json={"name": name})
    resp.raise_for_status()
    return resp.json()


async def _room_with_hosts(
    client: httpx.AsyncClient,
    *,
    computer_id: str | None,
    cloud: bool = True,
) -> dict[str, Any]:
    room = (await client.post("/rooms", json={"name": "byoa-room"})).json()
    room_id = room["id"]
    human = (
        await client.post(
            f"/rooms/{room_id}/participants",
            json={"kind": "human", "name": "Ada"},
        )
    ).json()
    cloud_agent = None
    if cloud:
        cloud_agent = (
            await client.post(
                f"/rooms/{room_id}/participants",
                json={"kind": "agent", "name": "Iris", "persona": "cloud"},
            )
        ).json()
    byoa = (
        await client.post(
            f"/rooms/{room_id}/participants",
            json={
                "kind": "agent",
                "name": "Jules",
                "persona": "local",
                "computer_id": computer_id,
            },
        )
    ).json()
    return {
        "room": room,
        "human": human,
        "cloud": cloud_agent,
        "byoa": byoa,
    }


def _world(client: httpx.AsyncClient, token: str) -> HttpWorld:
    return HttpWorld(client, token)


@pytest.mark.asyncio
async def test_http_world_turn_context_and_fresh_reply(harness: Harness) -> None:
    computer = await _pair(harness.client)
    setup = await _room_with_hosts(harness.client, computer_id=computer["id"], cloud=False)
    room_id = UUID(setup["room"]["id"])
    human_id = UUID(setup["human"]["id"])
    agent_id = UUID(setup["byoa"]["id"])

    posted = await harness.client.post(
        f"/rooms/{room_id}/messages",
        json={"author_id": str(human_id), "body": "hello jules"},
    )
    assert posted.status_code == 200
    assert posted.json()["seq"] == 1

    world = _world(harness.client, computer["token"])
    ctx = await world.load_turn(agent_id, room_id)
    assert ctx.agent.name == "Jules"
    assert ctx.seen_seq == 1
    assert [m.body for m in ctx.inbox] == ["hello jules"]
    assert {p.name for p in ctx.participants} == {"Ada", "Jules"}

    row = await world.insert_message(
        room_id, agent_id, "jules here", not_after_seq=1
    )
    assert row.seq == 2
    stored = await harness.client.get(f"/rooms/{room_id}/messages")
    assert [m["body"] for m in stored.json()["messages"]] == [
        "hello jules",
        "jules here",
    ]


@pytest.mark.asyncio
async def test_http_world_stale_reply_returns_409_with_newer(
    harness: Harness,
) -> None:
    computer = await _pair(harness.client)
    setup = await _room_with_hosts(harness.client, computer_id=computer["id"], cloud=False)
    room_id = UUID(setup["room"]["id"])
    human_id = UUID(setup["human"]["id"])
    agent_id = UUID(setup["byoa"]["id"])
    await harness.client.post(
        f"/rooms/{room_id}/messages",
        json={"author_id": str(human_id), "body": "first"},
    )
    world = _world(harness.client, computer["token"])
    ctx = await world.load_turn(agent_id, room_id)
    assert ctx.seen_seq == 1

    await harness.client.post(
        f"/rooms/{room_id}/messages",
        json={"author_id": str(human_id), "body": "peer slipped in"},
    )

    from brain.world import StaleWrite

    with pytest.raises(StaleWrite) as exc:
        await world.insert_message(room_id, agent_id, "too late", not_after_seq=1)
    assert exc.value.last_seq == 2
    assert exc.value.newer is not None
    assert [m.body for m in exc.value.newer] == ["peer slipped in"]
    assert exc.value.newer[0].seq == 2
    assert exc.value.newer[0].author_name == "Ada"

    row = await world.insert_message(
        room_id, agent_id, "after hold", not_after_seq=2
    )
    assert row.seq == 3


@pytest.mark.asyncio
async def test_http_world_claim_and_llm_call_row(harness: Harness) -> None:
    computer = await _pair(harness.client)
    setup = await _room_with_hosts(harness.client, computer_id=computer["id"], cloud=False)
    room_id = UUID(setup["room"]["id"])
    agent_id = UUID(setup["byoa"]["id"])
    world = _world(harness.client, computer["token"])
    world.bind_actor(agent_id)

    assert await world.try_claim(room_id, "t1", agent_id) is True
    # Same holder re-claiming is an idempotent refresh, not a loss
    # (cross-agent conflict is tested in tests/test_claims.py).
    assert await world.try_claim(room_id, "t1", agent_id) is True

    await world.record_llm_call(
        agent_id, room_id, "gpt-test", 11, 5, "turn"
    )
    rows = await harness.app.state.pool.fetch(
        """
        SELECT model, prompt_tokens, completion_tokens, purpose
        FROM llm_calls
        WHERE agent_id = $1
        """,
        agent_id,
    )
    assert len(rows) == 1
    assert rows[0]["purpose"] == "turn"
    assert rows[0]["model"] == "gpt-test"
    assert rows[0]["prompt_tokens"] == 11
    assert rows[0]["completion_tokens"] == 5


@pytest.mark.asyncio
async def test_runtime_rejects_wrong_token_and_cross_computer(
    harness: Harness,
) -> None:
    alpha = await _pair(harness.client, "alpha")
    beta = await _pair(harness.client, "beta")
    setup = await _room_with_hosts(harness.client, computer_id=alpha["id"], cloud=False)
    room_id = setup["room"]["id"]
    agent_id = setup["byoa"]["id"]

    wrong = await harness.client.get(
        "/runtime/turn-context",
        params={"agent_id": agent_id, "room_id": room_id},
        headers={"Authorization": "Bearer not-a-real-token"},
    )
    assert wrong.status_code == 401

    cross = await harness.client.get(
        "/runtime/turn-context",
        params={"agent_id": agent_id, "room_id": room_id},
        headers={"Authorization": f"Bearer {beta['token']}"},
    )
    assert cross.status_code == 403

    claim = await harness.client.post(
        "/runtime/claim",
        json={"agent_id": agent_id, "room_id": room_id, "task_key": "t1"},
        headers={"Authorization": f"Bearer {beta['token']}"},
    )
    assert claim.status_code == 403


@pytest.mark.asyncio
async def test_wake_routing_byoa_over_ws_not_in_process_lane(
    harness: Harness,
) -> None:
    computer = await _pair(harness.client)
    setup = await _room_with_hosts(harness.client, computer_id=computer["id"], cloud=True)
    room_id = setup["room"]["id"]
    human_id = setup["human"]["id"]
    byoa_id = UUID(setup["byoa"]["id"])
    cloud_id = UUID(setup["cloud"]["id"])

    ws = await connect_asgi_ws(
        harness.app,
        f"/ws/computers/{computer['id']}",
        query_string=f"token={computer['token']}",
    )
    try:
        posted = await harness.client.post(
            f"/rooms/{room_id}/messages",
            json={"author_id": human_id, "body": "hello both"},
        )
        assert posted.status_code == 200
        frame = await ws.receive_json(timeout=4.0)
        assert frame["type"] == "wake"
        assert frame["agent_id"] == str(byoa_id)
        assert frame["room_id"] == room_id

        await harness.app.state.scheduler.wait_idle()
        turned = {t.agent_id for t in harness.app.state.scheduler.turns}
        assert cloud_id in turned
        assert byoa_id not in turned
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_wake_routing_offline_computer_sleeps(
    harness: Harness, caplog: pytest.LogCaptureFixture
) -> None:
    computer = await _pair(harness.client)
    setup = await _room_with_hosts(harness.client, computer_id=computer["id"], cloud=True)
    room_id = setup["room"]["id"]
    human_id = setup["human"]["id"]
    byoa_id = UUID(setup["byoa"]["id"])
    cloud_id = UUID(setup["cloud"]["id"])

    with caplog.at_level(logging.INFO, logger="agora.scheduler"):
        posted = await harness.client.post(
            f"/rooms/{room_id}/messages",
            json={"author_id": human_id, "body": "anyone home"},
        )
        assert posted.status_code == 200
        # Pub/sub is async: wait until the cloud lane has actually run
        # so we know dispatch has decided Jules is sleeping.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 4.0
        while loop.time() < deadline:
            if any(t.agent_id == cloud_id for t in harness.app.state.scheduler.turns):
                break
            await asyncio.sleep(0.05)
        await harness.app.state.scheduler.wait_idle()

    assert "agent Jules is sleeping (computer offline)" in caplog.text
    turned = {t.agent_id for t in harness.app.state.scheduler.turns}
    assert turned == {cloud_id}
    assert byoa_id not in turned


@pytest.mark.asyncio
async def test_byoa_loop_mocked_model_reply_and_ledger(harness: Harness) -> None:
    computer = await _pair(harness.client)
    setup = await _room_with_hosts(harness.client, computer_id=computer["id"], cloud=True)
    room_id = UUID(setup["room"]["id"])
    human_id = setup["human"]["id"]
    byoa_id = UUID(setup["byoa"]["id"])
    cloud_id = UUID(setup["cloud"]["id"])

    ws = await connect_asgi_ws(
        harness.app,
        f"/ws/computers/{computer['id']}",
        query_string=f"token={computer['token']}",
    )
    try:
        posted = await harness.client.post(
            f"/rooms/{room_id}/messages",
            json={"author_id": human_id, "body": "please reply"},
        )
        assert posted.status_code == 200
        frame = await ws.receive_json(timeout=4.0)
        assert frame == {
            "type": "wake",
            "agent_id": str(byoa_id),
            "room_id": str(room_id),
        }

        world = _world(harness.client, computer["token"])
        small = ScriptedChatModel(
            [triage_message(actionable=True, reason="addressed", response_mode="me")]
        )
        big = ScriptedChatModel([tool_call("reply", {"body": "jules from the laptop"})])
        brain = Brain(world, small_model=small, big_model=big)
        result = await brain.run(byoa_id, room_id)
        assert result.outcome == "replied"
        assert result.reply_body == "jules from the laptop"

        listed = (await harness.client.get(f"/rooms/{room_id}/messages")).json()
        messages = listed["messages"]
        assert messages[-1]["seq"] == 2
        assert messages[-1]["author_id"] == str(byoa_id)
        assert messages[-1]["body"] == "jules from the laptop"

        # Author is excluded from the wake fan-out of their own reply.
        with pytest.raises(TimeoutError):
            await ws.receive_json(timeout=0.4)

        await harness.app.state.scheduler.wait_idle()
        # Cloud agent woke on the human post and again on Jules's reply.
        cloud_turns = [
            t for t in harness.app.state.scheduler.turns if t.agent_id == cloud_id
        ]
        assert len(cloud_turns) >= 2
        assert all(t.agent_id != byoa_id for t in harness.app.state.scheduler.turns)

        rows = await harness.app.state.pool.fetch(
            """
            SELECT purpose FROM llm_calls
            WHERE agent_id = $1
            ORDER BY created_at, id
            """,
            byoa_id,
        )
        purposes = [r["purpose"] for r in rows]
        assert "turn" in purposes
        assert purposes[0] == "triage"
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_ws_rejects_bad_token(harness: Harness) -> None:
    computer = await _pair(harness.client)
    with pytest.raises(ConnectionError, match="1008"):
        await connect_asgi_ws(
            harness.app,
            f"/ws/computers/{computer['id']}",
            query_string="token=nope",
        )
