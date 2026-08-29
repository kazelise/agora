"""Coordination hardening borrowed from Cumora and yuanzhuo.

- verbatim-dup: atomic in-transaction duplicate suppression (Cumora §5b)
- hold tokens: send_anyway must acknowledge a server-shown HOLD (§5d)
- loop cap: agent-only runs past a counted floor are stale (§6)
- digest: yuanzhuo-style markdown export of the room
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from uuid import UUID

import asyncpg
import httpx
import pytest
import redis.asyncio as redis

from brain.holds import consume_hold, record_hold
from brain.world import DuplicateReply
from brain.world_direct import DirectWorld
from server import db
from server.main import create_app
from tests.conftest import DSN, REDIS_URL
from tests.fakes import ScriptedChatModel, text_message, tool_call, triage_message
from brain.graph import Brain


@pytest.fixture
async def pool(require_services: None) -> AsyncIterator[asyncpg.Pool]:
    created = await db.create_pool(DSN)
    await db.migrate(created)
    await db.truncate_all(created)
    yield created
    await created.close()


@pytest.fixture
async def redis_client(require_services: None) -> AsyncIterator[redis.Redis]:
    client = redis.from_url(REDIS_URL, decode_responses=True)
    await client.flushdb()
    yield client
    await client.aclose()


async def _room(pool: asyncpg.Pool, *, agents: int = 2) -> tuple[UUID, UUID, list[UUID]]:
    room = await db.create_room(pool, "hardening-room")
    human = await db.add_participant(pool, room.id, "human", "Ada", None)
    agent_ids = []
    for i in range(agents):
        agent = await db.add_participant(pool, room.id, "agent", f"Agent{i}", None)
        agent_ids.append(agent.id)
    return room.id, human.id, agent_ids


# ── verbatim-dup gate ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_verbatim_dup_rejected_in_transaction(pool: asyncpg.Pool) -> None:
    room_id, human_id, agent_ids = await _room(pool)
    iris_id, marcus_id = agent_ids
    await db.insert_message(pool, room_id, human_id, "pick a number")
    await db.insert_message(pool, room_id, iris_id, "3")

    with pytest.raises(db.DuplicateReplyError) as exc:
        await db.insert_message(pool, room_id, marcus_id, "3")
    assert exc.value.peer_seq == 2
    stored = await db.list_messages(pool, room_id)
    assert [m.body for m in stored].count("3") == 1


@pytest.mark.asyncio
async def test_verbatim_dup_race_only_first_lands(pool: asyncpg.Pool) -> None:
    """Two concurrent inserts of the same body: exactly one commits."""
    room_id, human_id, agent_ids = await _room(pool)
    iris_id, marcus_id = agent_ids
    await db.insert_message(pool, room_id, human_id, "pick a number")

    results = await asyncio.gather(
        db.insert_message(pool, room_id, iris_id, "3"),
        db.insert_message(pool, room_id, marcus_id, "3"),
        return_exceptions=True,
    )
    bodies = [m.body for m in await db.list_messages(pool, room_id)]
    assert bodies.count("3") == 1
    assert any(isinstance(r, db.DuplicateReplyError) for r in results)


@pytest.mark.asyncio
async def test_whitespace_trimmed_comparison(pool: asyncpg.Pool) -> None:
    room_id, human_id, agent_ids = await _room(pool)
    iris_id, _marcus_id = agent_ids
    await db.insert_message(pool, room_id, human_id, "go on")
    other = await db.add_participant(pool, room_id, "agent", "Third", None)
    # Trimmed comparison: " go on\n" is still a verbatim dup of the peer row.
    with pytest.raises(db.DuplicateReplyError):
        await db.insert_message(pool, room_id, other.id, " go on\n")


@pytest.mark.asyncio
async def test_graph_duplicate_reply_redecides_and_replies(
    pool: asyncpg.Pool, redis_client: redis.Redis
) -> None:
    """The real dup path: the brain SAW the peer's "3" (freshness passes,
    seen includes it) and still parrots it. The transactional dup gate
    rejects, the fact goes back, and the brain re-decides in the same
    turn — no holds spent on a semantics error."""
    room_id, human_id, agent_ids = await _room(pool, agents=2)
    iris_id, marcus_id = agent_ids
    await db.insert_message(pool, room_id, human_id, "pick a number")
    await db.insert_message(pool, room_id, marcus_id, "3")

    small = ScriptedChatModel(
        [triage_message(actionable=True, reason="each", response_mode="each")]
    )
    big = ScriptedChatModel(
        [
            tool_call("reply", {"body": "3"}),
            tool_call("reply", {"body": "4 — 3 is taken, next after 3"}),
        ]
    )
    brain = Brain(DirectWorld(pool, redis_client), small_model=small, big_model=big)
    result = await brain.run(iris_id, room_id)

    assert result.outcome == "replied"
    assert result.reply_body == "4 — 3 is taken, next after 3"
    assert result.hold_count == 0
    dup_prompts = [
        " ".join(str(getattr(m, "content", "")) for m in call)
        for call in big.calls
    ]
    assert any("verbatim-duplicates" in p for p in dup_prompts)
    stored = await db.list_messages(pool, room_id)
    assert [(m.author_id, m.body) for m in stored][-1] == (
        iris_id,
        "4 — 3 is taken, next after 3",
    )


@pytest.mark.asyncio
async def test_graph_duplicate_reply_no_holds_spent(
    pool: asyncpg.Pool, redis_client: redis.Redis
) -> None:
    """The dup gate costs zero holds: the turn ends as skipped after the
    brain declines to rewrite — hold exhaustion is never touched."""
    room_id, human_id, agent_ids = await _room(pool, agents=2)
    iris_id, marcus_id = agent_ids
    await db.insert_message(pool, room_id, human_id, "say something")
    await db.insert_message(pool, room_id, marcus_id, "hello")

    small = ScriptedChatModel(
        [triage_message(actionable=True, reason="each", response_mode="each")]
    )
    big = ScriptedChatModel(
        [
            tool_call("reply", {"body": "hello"}),
            text_message("peer already said hello; I will stay quiet"),
        ]
    )
    brain = Brain(DirectWorld(pool, redis_client), small_model=small, big_model=big)
    result = await brain.run(iris_id, room_id)

    assert result.outcome == "skipped"
    assert result.hold_count == 0
    stored = [m.body for m in await db.list_messages(pool, room_id)]
    assert stored.count("hello") == 1


# ── hold tokens ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hold_token_roundtrip(
    pool: asyncpg.Pool, redis_client: redis.Redis
) -> None:
    room_id, _human_id, agent_ids = await _room(pool)
    agent_id = agent_ids[0]
    assert await consume_hold(redis_client, agent_id, room_id) is None
    await record_hold(redis_client, agent_id, room_id, 7)
    assert await consume_hold(redis_client, agent_id, room_id) == 7
    # Atomic consume: the second read gets nothing.
    assert await consume_hold(redis_client, agent_id, room_id) is None


@pytest.mark.asyncio
async def test_send_anyway_without_token_does_not_bypass(
    pool: asyncpg.Pool, redis_client: redis.Redis
) -> None:
    """Pre-emptive send_anyway is ignored: the gate still HOLDs."""
    room_id, human_id, agent_ids = await _room(pool)
    iris_id, marcus_id = agent_ids
    await db.insert_message(pool, room_id, human_id, "next number after 2")

    async def peer_first(_messages: list) -> object:
        await db.insert_message(pool, room_id, marcus_id, "3")
        return tool_call("reply", {"body": "3", "send_anyway": True})

    small = ScriptedChatModel(
        [triage_message(actionable=True, reason="each", response_mode="each")]
    )
    big = ScriptedChatModel([peer_first, tool_call("reply", {"body": "4"})])
    brain = Brain(
        DirectWorld(pool, redis_client),
        small_model=small,
        big_model=big,
        hold_redis=redis_client,
    )

    result = await brain.run(iris_id, room_id)

    assert result.hold_count == 1
    assert result.reply_body == "4"


@pytest.mark.asyncio
async def test_send_anyway_with_token_bypasses_hold(
    pool: asyncpg.Pool, redis_client: redis.Redis
) -> None:
    """HELD once → token armed; the retry with send_anyway ships without
    a second HOLD (the legitimate flow)."""
    room_id, human_id, agent_ids = await _room(pool)
    iris_id, marcus_id = agent_ids
    await db.insert_message(pool, room_id, human_id, "next number after 2")

    async def peer_first(_messages: list) -> object:
        await db.insert_message(pool, room_id, marcus_id, "3")
        return tool_call("reply", {"body": "3"})

    async def retry_with_flag(_messages: list) -> object:
        # The HELD envelope armed a token bound to seq 2; acknowledging it
        # is legitimate — the agent has seen everything up to that point.
        # (The body differs from the peer's, so the dup gate stays clear.)
        return tool_call("reply", {"body": "3, going with 4", "send_anyway": True})

    small = ScriptedChatModel(
        [triage_message(actionable=True, reason="each", response_mode="each")]
    )
    big = ScriptedChatModel([peer_first, retry_with_flag])
    brain = Brain(
        DirectWorld(pool, redis_client),
        small_model=small,
        big_model=big,
        hold_redis=redis_client,
    )

    result = await brain.run(iris_id, room_id)

    assert result.outcome == "replied"
    assert result.reply_body == "3, going with 4"
    assert result.hold_count == 1
    stored = await db.list_messages(pool, room_id)
    assert [(m.author_id, m.body) for m in stored][-1] == (
        iris_id,
        "3, going with 4",
    )


@pytest.mark.asyncio
async def test_successful_send_clears_lingering_token(
    pool: asyncpg.Pool, redis_client: redis.Redis
) -> None:
    room_id, human_id, agent_ids = await _room(pool)
    iris_id, _marcus_id = agent_ids
    await db.insert_message(pool, room_id, human_id, "say hi")
    await record_hold(redis_client, iris_id, room_id, 1)

    small = ScriptedChatModel(
        [triage_message(actionable=True, reason="me", response_mode="me")]
    )
    big = ScriptedChatModel([tool_call("reply", {"body": "hi!"})])
    brain = Brain(
        DirectWorld(pool, redis_client),
        small_model=small,
        big_model=big,
        hold_redis=redis_client,
    )
    result = await brain.run(iris_id, room_id)

    assert result.outcome == "replied"
    assert await consume_hold(redis_client, iris_id, room_id) is None


# ── loop cap ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_agent_only_run_past_loop_cap_stays_silent(
    pool: asyncpg.Pool, redis_client: redis.Redis
) -> None:
    """4-agent-only messages (cap 4 x 1 agent... here 2 agents, cap 8) —
    we insert 8 agent messages and no human; triage must skip without
    spending the small model."""
    room_id, _human_id, agent_ids = await _room(pool, agents=2)
    iris_id, marcus_id = agent_ids
    for i in range(4):
        await db.insert_message(pool, room_id, marcus_id, f"agent chatter {i}")
    await db.insert_message(pool, room_id, iris_id, "agent chatter 4")
    for i in range(3):
        await db.insert_message(pool, room_id, marcus_id, f"agent chatter {i + 5}")

    small = ScriptedChatModel()
    big = ScriptedChatModel()
    brain = Brain(DirectWorld(pool, redis_client), small_model=small, big_model=big)

    result = await brain.run(iris_id, room_id)

    assert result.outcome == "skipped"
    assert small.calls == []
    assert big.calls == []
    assert "loop cap" in (result.triage_reason or "")


@pytest.mark.asyncio
async def test_human_message_resets_loop_cap(
    pool: asyncpg.Pool, redis_client: redis.Redis
) -> None:
    """A human message in the inbox lifts the cap — the model gate runs."""
    room_id, human_id, agent_ids = await _room(pool, agents=2)
    iris_id, marcus_id = agent_ids
    for i in range(5):
        await db.insert_message(pool, room_id, marcus_id, f"agent chatter {i}")
    await db.insert_message(pool, room_id, human_id, "ok everyone stop")

    small = ScriptedChatModel(
        [triage_message(actionable=False, reason="not mine", response_mode="me")]
    )
    big = ScriptedChatModel()
    brain = Brain(DirectWorld(pool, redis_client), small_model=small, big_model=big)

    result = await brain.run(iris_id, room_id)

    assert result.outcome == "skipped"
    assert len(small.calls) == 1


# ── digest ───────────────────────────────────────────────────────────────


@pytest.fixture
async def app_client(require_services: None) -> AsyncIterator[tuple]:
    app = create_app(stub_turns=True)
    async with app.router.lifespan_context(app):
        await db.truncate_all(app.state.pool)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield app, client


@pytest.mark.asyncio
async def test_digest_endpoint_renders_transcript_claims_and_spend(
    app_client: tuple,
) -> None:
    app, client = app_client
    room = (await client.post("/rooms", json={"name": "review"})).json()
    human = (
        await client.post(
            f"/rooms/{room['id']}/participants",
            json={"kind": "human", "name": "Ada"},
        )
    ).json()
    agent = (
        await client.post(
            f"/rooms/{room['id']}/participants",
            json={"kind": "agent", "name": "Jules", "persona": "analyst"},
        )
    ).json()
    await client.post(
        f"/rooms/{room['id']}/messages",
        json={"author_id": human["id"], "body": "review the plan | first"},
    )
    await client.post(
        f"/rooms/{room['id']}/messages",
        json={"author_id": agent["id"], "body": "on it"},
    )
    await db.try_claim(app.state.pool, UUID(room["id"]), "t1:review", UUID(agent["id"]))
    await db.insert_llm_call(
        app.state.pool, UUID(agent["id"]), UUID(room["id"]), "m-small", 10, 2, "triage"
    )
    await db.insert_llm_call(
        app.state.pool, UUID(agent["id"]), UUID(room["id"]), "m-big", 100, 50, "turn"
    )

    resp = await client.get(f"/rooms/{room['id']}/digest")
    assert resp.status_code == 200
    body = resp.text
    assert "# review" in body
    assert "review the plan \\| first" in body
    assert "`t1:review` — held by **Jules**" in body
    assert "| triage | m-small | 1 | 10 | 2 |" in body
    assert "**110**" in body


@pytest.mark.asyncio
async def test_digest_404_for_missing_room(app_client: tuple) -> None:
    _app, client = app_client
    import uuid as uuid_mod

    resp = await client.get(f"/rooms/{uuid_mod.uuid4()}/digest")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_duplicate_reply_over_runtime_returns_409(app_client: tuple) -> None:
    _app, client = app_client
    computer = (await client.post("/computers", json={"name": "laptop"})).json()
    room = (await client.post("/rooms", json={"name": "dup-room"})).json()
    human = (
        await client.post(
            f"/rooms/{room['id']}/participants",
            json={"kind": "human", "name": "Ada"},
        )
    ).json()
    byoa = (
        await client.post(
            f"/rooms/{room['id']}/participants",
            json={
                "kind": "agent",
                "name": "Jules",
                "computer_id": computer["id"],
            },
        )
    ).json()
    await client.post(
        f"/rooms/{room['id']}/messages",
        json={"author_id": human["id"], "body": "pick a number"},
    )
    agent_id = UUID(byoa["id"])

    async def post_reply(body: str, not_after_seq: int) -> httpx.Response:
        return await client.post(
            "/runtime/reply",
            json={
                "agent_id": str(agent_id),
                "room_id": room["id"],
                "body": body,
                "not_after_seq": not_after_seq,
            },
            headers={"Authorization": f"Bearer {computer['token']}"},
        )

    first = await post_reply("3", 1)
    assert first.status_code == 200
    # Fresh cursor (seen up to seq 2), but body duplicates the peer's "3"
    # — wait, the peer here is the agent's own first reply; author rows
    # are excluded, so re-post "3" when the human echoes it first.
    await client.post(
        f"/rooms/{room['id']}/messages",
        json={"author_id": human["id"], "body": "3"},
    )
    second = await post_reply("3", 3)
    assert second.status_code == 409
    assert second.json()["detail"]["error"] == "duplicate_reply"
    assert second.json()["detail"]["peer_seq"] == 3

    from daemon.world_http import HttpWorld

    world = HttpWorld(client, computer["token"])
    world.bind_actor(agent_id)
    with pytest.raises(DuplicateReply):
        await world.insert_message(UUID(room["id"]), agent_id, "3", not_after_seq=3)
