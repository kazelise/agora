"""Phase 7: moderated rooms — routing, decide tool, idempotent decisions."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from uuid import UUID, uuid4

import asyncpg
import httpx
import pytest
import redis.asyncio as redis

import brain.graph as graph_mod
from brain.graph import (
    DECIDE_CALLS_EXHAUSTED,
    DECIDE_TARGET_ERROR,
    MODERATION_NOTE,
    MODERATOR_SAY_ERROR,
    Brain,
)
from brain.wake_hints import consume_called_on, hint_key
from brain.world_direct import DirectWorld
from server import db
from server.digest import build_room_digest
from server.main import create_app
from server.mentions import mentioned_name
from server.scheduler import Scheduler
from tests.conftest import DSN, REDIS_URL
from tests.fakes import ScriptedChatModel, text_message, tool_call, triage_message


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
    yield client
    await client.aclose()


async def _moderated_room(
    pool: asyncpg.Pool,
    *,
    members: int = 1,
) -> tuple[UUID, UUID, UUID, list[UUID]]:
    room = await db.create_room(pool, "roundtable", mode="moderated")
    human = await db.add_participant(pool, room.id, "human", "Ada", None)
    chair = await db.add_participant(
        pool, room.id, "agent", "Chair", "keeps time", role="moderator"
    )
    member_ids: list[UUID] = []
    names = ["Iris", "Marcus", "Jules"]
    for i in range(members):
        row = await db.add_participant(
            pool, room.id, "agent", names[i], f"persona-{i}"
        )
        member_ids.append(row.id)
    return room.id, human.id, chair.id, member_ids


async def _calls(pool: asyncpg.Pool, room_id: UUID) -> list[asyncpg.Record]:
    return await pool.fetch(
        """
        SELECT purpose, model, prompt_tokens, completion_tokens
        FROM llm_calls
        WHERE room_id = $1
        ORDER BY created_at, id
        """,
        room_id,
    )


async def _decisions(pool: asyncpg.Pool, room_id: UUID) -> list[asyncpg.Record]:
    return await pool.fetch(
        """
        SELECT trigger_seq, action, target_id, moderator_id
        FROM moderator_decisions
        WHERE room_id = $1
        ORDER BY created_at, id
        """,
        room_id,
    )


async def _chair_call_on(
    pool: asyncpg.Pool,
    redis_client: redis.Redis,
    room_id: UUID,
    chair_id: UUID,
    target: str,
) -> object:
    brain = Brain(
        DirectWorld(pool, redis_client),
        small_model=ScriptedChatModel(),
        big_model=ScriptedChatModel(
            [tool_call("decide", {"action": "call_on", "target": target})]
        ),
    )
    return await brain.run(chair_id, room_id)


async def _wait_turns(
    scheduler: Scheduler,
    pred: Callable[[list], bool],
    timeout: float = 4.0,
) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if pred(scheduler.turns):
            await scheduler.wait_idle()
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"timed out waiting for turns: {scheduler.turns}")


# ── mention protocol ─────────────────────────────────────────────────────


def test_mention_longest_match_is_case_sensitive() -> None:
    names = ["Iris", "IrisLee", "Marcus"]
    assert mentioned_name("please @IrisLee take this", names) == "IrisLee"
    assert mentioned_name("please @Iris take this", names) == "Iris"
    assert mentioned_name("please @iris take this", names) is None
    assert mentioned_name("no one named", names) is None


def test_mention_earliest_position_wins() -> None:
    names = ["Bob", "Alexander"]
    assert mentioned_name("@Bob please ask @Alexander", names) == "Bob"
    assert mentioned_name("please ask @Alexander then @Bob", names) == "Alexander"


def test_mention_cjk_and_email_boundaries() -> None:
    names = ["张三", "张三丰", "Bob"]
    assert mentioned_name("请 @张三丰 发言", names) == "张三丰"
    assert mentioned_name("请 @张三 发言", names) == "张三"
    assert mentioned_name("foo@Bob later", names) is None
    assert mentioned_name("hi @Bob.", names) == "Bob"


# ── API ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_mode_and_role_round_trip(app_client: tuple) -> None:
    _app, client = app_client
    room = (
        await client.post("/rooms", json={"name": "rt", "mode": "moderated"})
    ).json()
    assert room["mode"] == "moderated"
    chair = (
        await client.post(
            f"/rooms/{room['id']}/participants",
            json={"kind": "agent", "name": "Chair", "role": "moderator"},
        )
    ).json()
    assert chair["role"] == "moderator"
    member = (
        await client.post(
            f"/rooms/{room['id']}/participants",
            json={"kind": "agent", "name": "Iris"},
        )
    ).json()
    assert member["role"] == "member"
    opened = (await client.post("/rooms", json={"name": "plain"})).json()
    assert opened["mode"] == "open"


@pytest.mark.asyncio
async def test_second_moderator_is_409(app_client: tuple) -> None:
    _app, client = app_client
    room = (
        await client.post("/rooms", json={"name": "rt", "mode": "moderated"})
    ).json()
    first = await client.post(
        f"/rooms/{room['id']}/participants",
        json={"kind": "agent", "name": "Chair", "role": "moderator"},
    )
    assert first.status_code == 200
    second = await client.post(
        f"/rooms/{room['id']}/participants",
        json={"kind": "agent", "name": "Vice", "role": "moderator"},
    )
    assert second.status_code == 409


@pytest.mark.asyncio
async def test_human_cannot_be_moderator(app_client: tuple) -> None:
    _app, client = app_client
    room = (
        await client.post("/rooms", json={"name": "rt", "mode": "moderated"})
    ).json()
    resp = await client.post(
        f"/rooms/{room['id']}/participants",
        json={"kind": "human", "name": "Ada", "role": "moderator"},
    )
    assert resp.status_code == 400


# ── routing ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_moderated_room_wakes_moderator_only(app_client: tuple) -> None:
    app, client = app_client
    room = (
        await client.post("/rooms", json={"name": "rt", "mode": "moderated"})
    ).json()
    human = (
        await client.post(
            f"/rooms/{room['id']}/participants",
            json={"kind": "human", "name": "Ada"},
        )
    ).json()
    chair = (
        await client.post(
            f"/rooms/{room['id']}/participants",
            json={"kind": "agent", "name": "Chair", "role": "moderator"},
        )
    ).json()
    iris = (
        await client.post(
            f"/rooms/{room['id']}/participants",
            json={"kind": "agent", "name": "Iris"},
        )
    ).json()
    posted = await client.post(
        f"/rooms/{room['id']}/messages",
        json={"author_id": human["id"], "body": "let us begin"},
    )
    assert posted.status_code == 200
    await _wait_turns(
        app.state.scheduler,
        lambda turns: any(t.agent_id == UUID(chair["id"]) for t in turns),
    )
    woken = {t.agent_id for t in app.state.scheduler.turns}
    assert woken == {UUID(chair["id"])}
    assert UUID(iris["id"]) not in woken


@pytest.mark.asyncio
async def test_mention_wakes_only_named_agent(app_client: tuple) -> None:
    app, client = app_client
    room = (
        await client.post("/rooms", json={"name": "rt", "mode": "moderated"})
    ).json()
    human = (
        await client.post(
            f"/rooms/{room['id']}/participants",
            json={"kind": "human", "name": "Ada"},
        )
    ).json()
    await client.post(
        f"/rooms/{room['id']}/participants",
        json={"kind": "agent", "name": "Chair", "role": "moderator"},
    )
    iris = (
        await client.post(
            f"/rooms/{room['id']}/participants",
            json={"kind": "agent", "name": "Iris"},
        )
    ).json()
    marcus = (
        await client.post(
            f"/rooms/{room['id']}/participants",
            json={"kind": "agent", "name": "Marcus"},
        )
    ).json()
    posted = await client.post(
        f"/rooms/{room['id']}/messages",
        json={"author_id": human["id"], "body": "@Iris what is the next number?"},
    )
    assert posted.status_code == 200
    await _wait_turns(
        app.state.scheduler,
        lambda turns: any(t.agent_id == UUID(iris["id"]) for t in turns),
    )
    woken = {t.agent_id for t in app.state.scheduler.turns}
    assert woken == {UUID(iris["id"])}
    assert UUID(marcus["id"]) not in woken


@pytest.mark.asyncio
async def test_author_never_self_wakes_in_moderated_room(app_client: tuple) -> None:
    app, client = app_client
    room = (
        await client.post("/rooms", json={"name": "rt", "mode": "moderated"})
    ).json()
    chair = (
        await client.post(
            f"/rooms/{room['id']}/participants",
            json={"kind": "agent", "name": "Chair", "role": "moderator"},
        )
    ).json()
    posted = await client.post(
        f"/rooms/{room['id']}/messages",
        json={"author_id": chair["id"], "body": "opening the table"},
    )
    assert posted.status_code == 200
    await asyncio.sleep(0.3)
    await app.state.scheduler.wait_idle()
    assert app.state.scheduler.turns == []
    # Mention of the author themselves is also a no-op (author exclusion).
    before = len(app.state.scheduler.turns)
    await client.post(
        f"/rooms/{room['id']}/messages",
        json={"author_id": chair["id"], "body": "@Chair notes"},
    )
    await asyncio.sleep(0.3)
    await app.state.scheduler.wait_idle()
    assert len(app.state.scheduler.turns) == before


# ── decide tool ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_call_on_writes_row_wakes_target_skips_triage(
    pool: asyncpg.Pool, redis_client: redis.Redis
) -> None:
    room_id, human_id, chair_id, member_ids = await _moderated_room(pool)
    iris_id = member_ids[0]
    await db.insert_message(pool, room_id, human_id, "who should answer?")

    wakes: list[tuple[UUID, UUID]] = []

    async def on_call_on(rid: UUID, tid: UUID, _trigger_seq: int) -> None:
        wakes.append((rid, tid))

    small = ScriptedChatModel()
    big = ScriptedChatModel(
        [tool_call("decide", {"action": "call_on", "target": "Iris"})]
    )
    chair_brain = Brain(
        DirectWorld(pool, redis_client, on_call_on=on_call_on),
        small_model=small,
        big_model=big,
    )
    result = await chair_brain.run(chair_id, room_id)

    assert result.outcome == "moderated_call"
    assert small.calls == []
    assert big.bound_tools == ["decide"]
    assert wakes == [(room_id, iris_id)]
    rows = await _decisions(pool, room_id)
    assert len(rows) == 1
    assert rows[0]["action"] == "call_on"
    assert rows[0]["target_id"] == iris_id
    assert rows[0]["trigger_seq"] == 1
    ledger = await _calls(pool, room_id)
    assert [r["purpose"] for r in ledger] == ["moderate"]

    member_small = ScriptedChatModel()
    member_big = ScriptedChatModel([tool_call("reply", {"body": "Iris answering"})])
    member_brain = Brain(
        DirectWorld(pool, redis_client),
        small_model=member_small,
        big_model=member_big,
    )
    called = await member_brain.run(iris_id, room_id, called_on=True)

    assert called.outcome == "replied"
    assert called.response_mode == "me"
    assert member_small.calls == []
    assert member_big.bound_tools == ["reply", "claim"]
    assert called.reply_body == "Iris answering"
    prompt = " ".join(str(getattr(m, "content", "")) for m in member_big.calls[0])
    assert "moderator called on you" in prompt


@pytest.mark.asyncio
async def test_say_goes_through_freshness_hold(
    pool: asyncpg.Pool, redis_client: redis.Redis
) -> None:
    room_id, human_id, chair_id, member_ids = await _moderated_room(pool)
    iris_id = member_ids[0]
    await db.insert_message(pool, room_id, human_id, "open the table")

    async def first_say(_messages: list) -> object:
        await db.insert_message(pool, room_id, iris_id, "Iris slipped in")
        return tool_call("decide", {"action": "say", "body": "welcome everyone"})

    small = ScriptedChatModel()
    big = ScriptedChatModel(
        [
            first_say,
            tool_call("decide", {"action": "say", "body": "welcome, after hold"}),
            tool_call("decide", {"action": "silence"}),
        ]
    )
    brain = Brain(
        DirectWorld(pool, redis_client), small_model=small, big_model=big
    )
    result = await brain.run(chair_id, room_id)

    assert result.hold_count == 1
    # HOLD advances seen_seq, so the retry is a new trigger — the gate
    # still fired, and the first raced body never landed. say is
    # non-terminal: the turn ends on silence, reply_body is the said text.
    assert result.outcome == "moderated_silence"
    assert result.reply_body == "welcome, after hold"
    second_prompt = " ".join(
        str(getattr(m, "content", "")) for m in big.calls[1]
    )
    assert "New messages landed while you were composing" in second_prompt
    stored = [m.body for m in await db.list_messages(pool, room_id)]
    assert "welcome everyone" not in stored
    assert "welcome, after hold" in stored
    assert "Iris slipped in" in stored
    rows = await _decisions(pool, room_id)
    assert [r["action"] for r in rows] == ["say", "silence"]
    assert rows[0]["trigger_seq"] < rows[1]["trigger_seq"]


@pytest.mark.asyncio
async def test_silence_writes_row_and_no_message(
    pool: asyncpg.Pool, redis_client: redis.Redis
) -> None:
    room_id, human_id, chair_id, _members = await _moderated_room(pool)
    await db.insert_message(pool, room_id, human_id, "carry on")
    small = ScriptedChatModel()
    big = ScriptedChatModel([tool_call("decide", {"action": "silence"})])
    brain = Brain(
        DirectWorld(pool, redis_client), small_model=small, big_model=big
    )
    result = await brain.run(chair_id, room_id)

    assert result.outcome == "moderated_silence"
    assert result.reply_body is None
    assert [m.body for m in await db.list_messages(pool, room_id)] == ["carry on"]
    rows = await _decisions(pool, room_id)
    assert len(rows) == 1
    assert rows[0]["action"] == "silence"
    assert rows[0]["target_id"] is None


@pytest.mark.asyncio
async def test_invalid_target_is_tool_error_then_retry(
    pool: asyncpg.Pool, redis_client: redis.Redis
) -> None:
    room_id, human_id, chair_id, member_ids = await _moderated_room(pool)
    iris_id = member_ids[0]
    await db.insert_message(pool, room_id, human_id, "pick someone")
    small = ScriptedChatModel()
    big = ScriptedChatModel(
        [
            tool_call("decide", {"action": "call_on", "target": "Ada"}),
            tool_call("decide", {"action": "call_on", "target": "Iris"}),
        ]
    )
    wakes: list[UUID] = []

    async def on_call_on(_rid: UUID, tid: UUID, _trigger_seq: int) -> None:
        wakes.append(tid)

    brain = Brain(
        DirectWorld(pool, redis_client, on_call_on=on_call_on),
        small_model=small,
        big_model=big,
    )
    result = await brain.run(chair_id, room_id)

    assert result.outcome == "moderated_call"
    assert wakes == [iris_id]
    told = [getattr(m, "content", "") for m in big.calls[1]]
    assert any(DECIDE_TARGET_ERROR in str(t) for t in told)
    assert any(
        "unknown or non-callable target 'Ada'" in str(t)
        and "call_on accepts one of: Iris" in str(t)
        and "humans cannot be called on" in str(t)
        for t in told
    )
    rows = await _decisions(pool, room_id)
    assert [r["target_id"] for r in rows] == [iris_id]


@pytest.mark.asyncio
async def test_no_tool_call_moderation_is_invalid_and_writes_no_row(
    pool: asyncpg.Pool, redis_client: redis.Redis
) -> None:
    room_id, human_id, chair_id, _members = await _moderated_room(pool)
    await db.insert_message(pool, room_id, human_id, "thoughts?")
    small = ScriptedChatModel()
    big = ScriptedChatModel(
        [text_message("I think Iris should go"), text_message("still no tool")]
    )
    brain = Brain(
        DirectWorld(pool, redis_client), small_model=small, big_model=big
    )
    result = await brain.run(chair_id, room_id)

    assert result.outcome == "invalid_moderation"
    assert await _decisions(pool, room_id) == []
    assert len(big.calls) == 2
    second = [getattr(m, "content", "") for m in big.calls[1]]
    assert any(MODERATION_NOTE in str(t) for t in second)


@pytest.mark.asyncio
async def test_same_trigger_seq_is_idempotent_no_double_wake(
    pool: asyncpg.Pool, redis_client: redis.Redis
) -> None:
    room_id, human_id, chair_id, member_ids = await _moderated_room(pool)
    iris_id = member_ids[0]
    await db.insert_message(pool, room_id, human_id, "pick a speaker")
    wakes: list[UUID] = []

    async def on_call_on(_rid: UUID, tid: UUID, _trigger_seq: int) -> None:
        wakes.append(tid)

    def brain() -> Brain:
        return Brain(
            DirectWorld(pool, redis_client, on_call_on=on_call_on),
            small_model=ScriptedChatModel(),
            big_model=ScriptedChatModel(
                [tool_call("decide", {"action": "call_on", "target": "Iris"})]
            ),
        )

    first = await brain().run(chair_id, room_id)
    assert first.outcome == "moderated_call"
    await db.set_last_read(pool, chair_id, room_id, 0)
    second = await brain().run(chair_id, room_id)
    assert second.outcome == "decision_replayed"
    assert wakes == [iris_id]
    assert len(await _decisions(pool, room_id)) == 1


@pytest.mark.asyncio
async def test_moderator_over_loop_cap_silences_with_zero_llm(
    pool: asyncpg.Pool, redis_client: redis.Redis
) -> None:
    room_id, _human_id, chair_id, member_ids = await _moderated_room(pool)
    iris_id = member_ids[0]
    # 2 agents → cap 8. Stretch is room-level since the last human; there
    # is a human participant but they have not spoken, so every agent
    # message counts (the most loop-prone shape).
    for i in range(8):
        await db.insert_message(pool, room_id, iris_id, f"circle {i}")
    small = ScriptedChatModel()
    big = ScriptedChatModel()
    brain = Brain(
        DirectWorld(pool, redis_client), small_model=small, big_model=big
    )
    result = await brain.run(chair_id, room_id)

    assert result.outcome == "moderated_silence"
    assert small.calls == []
    assert big.calls == []
    assert await _calls(pool, room_id) == []
    rows = await _decisions(pool, room_id)
    assert len(rows) == 1
    assert rows[0]["action"] == "silence"


# ── BYOA decision path ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_http_world_decision_wakes_byoa_target_over_ws(
    app_client: tuple,
) -> None:
    """Daemon-side moderator records via /runtime/decision; the server
    wakes the computer-hosted target. No server-side brain turn for that
    target — same host-boundary as stall nudges."""
    from daemon.world_http import HttpWorld
    from tests.asgi_ws import connect_asgi_ws

    app, client = app_client
    computer = (await client.post("/computers", json={"name": "laptop"})).json()
    room = (
        await client.post("/rooms", json={"name": "byoa-rt", "mode": "moderated"})
    ).json()
    human = (
        await client.post(
            f"/rooms/{room['id']}/participants",
            json={"kind": "human", "name": "Ada"},
        )
    ).json()
    chair = (
        await client.post(
            f"/rooms/{room['id']}/participants",
            json={
                "kind": "agent",
                "name": "Chair",
                "role": "moderator",
                "computer_id": computer["id"],
            },
        )
    ).json()
    iris = (
        await client.post(
            f"/rooms/{room['id']}/participants",
            json={
                "kind": "agent",
                "name": "Iris",
                "computer_id": computer["id"],
            },
        )
    ).json()
    posted = await client.post(
        f"/rooms/{room['id']}/messages",
        json={"author_id": human["id"], "body": "begin"},
    )
    posted.raise_for_status()

    ws = await connect_asgi_ws(
        app,
        f"/ws/computers/{computer['id']}",
        query_string=f"token={computer['token']}",
    )
    try:
        # Human post wakes the (BYOA) moderator over WS — not Iris.
        first = await ws.receive_json(timeout=4.0)
        assert first == {
            "type": "wake",
            "agent_id": chair["id"],
            "room_id": room["id"],
        }

        world = HttpWorld(client, computer["token"])
        recorded = await world.record_decision(
            UUID(room["id"]),
            UUID(chair["id"]),
            trigger_seq=1,
            action="call_on",
            target_id=UUID(iris["id"]),
        )
        assert recorded.status == "won"
        assert recorded.action == "call_on"
        assert recorded.target_id == UUID(iris["id"])

        frame = await ws.receive_json(timeout=4.0)
        assert frame == {
            "type": "wake",
            "agent_id": iris["id"],
            "room_id": room["id"],
            "called_on": True,
            "called_on_seq": 1,
        }
        await app.state.scheduler.wait_idle()
        assert all(
            t.agent_id != UUID(iris["id"]) for t in app.state.scheduler.turns
        )
        assert all(
            t.agent_id != UUID(chair["id"]) for t in app.state.scheduler.turns
        )
    finally:
        await ws.close()


@pytest.mark.asyncio
async def test_human_moderator_on_missing_room_is_404(app_client: tuple) -> None:
    _app, client = app_client
    resp = await client.post(
        f"/rooms/{uuid4()}/participants",
        json={"kind": "human", "name": "Ada", "role": "moderator"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_duplicate_participant_name_is_409(app_client: tuple) -> None:
    _app, client = app_client
    room = (await client.post("/rooms", json={"name": "rt"})).json()
    first = await client.post(
        f"/rooms/{room['id']}/participants",
        json={"kind": "agent", "name": "Iris"},
    )
    assert first.status_code == 200
    second = await client.post(
        f"/rooms/{room['id']}/participants",
        json={"kind": "human", "name": "Iris"},
    )
    assert second.status_code == 409


@pytest.mark.asyncio
async def test_decision_on_open_room_is_400_not_409(app_client: tuple) -> None:
    app, client = app_client
    computer = (await client.post("/computers", json={"name": "laptop"})).json()
    room = (await client.post("/rooms", json={"name": "open-rt"})).json()
    human = (
        await client.post(
            f"/rooms/{room['id']}/participants",
            json={"kind": "human", "name": "Ada"},
        )
    ).json()
    chair = (
        await client.post(
            f"/rooms/{room['id']}/participants",
            json={
                "kind": "agent",
                "name": "Chair",
                "role": "moderator",
                "computer_id": computer["id"],
            },
        )
    ).json()
    posted = await client.post(
        f"/rooms/{room['id']}/messages",
        json={"author_id": human["id"], "body": "hello"},
    )
    assert posted.status_code == 200
    resp = await client.post(
        "/runtime/decision",
        headers={"Authorization": f"Bearer {computer['token']}"},
        json={
            "agent_id": chair["id"],
            "room_id": room["id"],
            "trigger_seq": 1,
            "action": "silence",
        },
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "room is not moderated"


@pytest.mark.asyncio
async def test_future_trigger_seq_is_422(
    pool: asyncpg.Pool, app_client: tuple
) -> None:
    room_id, human_id, chair_id, _ = await _moderated_room(pool)
    await db.insert_message(pool, room_id, human_id, "now")
    with pytest.raises(db.InvalidTriggerSeqError):
        await db.record_decision(pool, room_id, chair_id, 99, "silence", None)
    with pytest.raises(db.InvalidTriggerSeqError):
        await db.record_decision(pool, room_id, chair_id, 0, "silence", None)

    _app, client = app_client
    computer = (await client.post("/computers", json={"name": "laptop"})).json()
    room = (
        await client.post("/rooms", json={"name": "seq-rt", "mode": "moderated"})
    ).json()
    human = (
        await client.post(
            f"/rooms/{room['id']}/participants",
            json={"kind": "human", "name": "Ada"},
        )
    ).json()
    chair = (
        await client.post(
            f"/rooms/{room['id']}/participants",
            json={
                "kind": "agent",
                "name": "Chair",
                "role": "moderator",
                "computer_id": computer["id"],
            },
        )
    ).json()
    posted = await client.post(
        f"/rooms/{room['id']}/messages",
        json={"author_id": human["id"], "body": "begin"},
    )
    assert posted.status_code == 200
    resp = await client.post(
        "/runtime/decision",
        headers={"Authorization": f"Bearer {computer['token']}"},
        json={
            "agent_id": chair["id"],
            "room_id": room["id"],
            "trigger_seq": 99,
            "action": "silence",
        },
    )
    assert resp.status_code == 422


# ── liveness review fixes ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_nudge_wakes_moderator_who_is_last_author(
    pool: asyncpg.Pool,
) -> None:
    room_id, human_id, chair_id, _ = await _moderated_room(pool)
    await db.insert_message(pool, room_id, human_id, "open")
    await db.insert_message(pool, room_id, chair_id, "I already spoke")
    wakes: list[tuple[UUID, bool]] = []

    async def run_turn(
        agent_id: UUID, room_id: UUID, *, called_on: bool = False
    ) -> None:
        wakes.append((agent_id, called_on))

    scheduler = Scheduler(pool, run_turn=run_turn)
    await scheduler.dispatch(room_id, chair_id, seq=None)
    await scheduler.wait_idle()
    assert wakes == [(chair_id, False)]


@pytest.mark.asyncio
async def test_nudge_redelivers_lost_call_on_once(pool: asyncpg.Pool) -> None:
    room_id, human_id, chair_id, member_ids = await _moderated_room(pool)
    iris_id = member_ids[0]
    await db.insert_message(pool, room_id, human_id, "pick a speaker")
    status, _ = await db.record_decision(
        pool, room_id, chair_id, 1, "call_on", iris_id
    )
    assert status == "won"
    wakes: list[tuple[UUID, bool]] = []

    async def run_turn(
        agent_id: UUID, room_id: UUID, *, called_on: bool = False
    ) -> None:
        wakes.append((agent_id, called_on))

    scheduler = Scheduler(pool, run_turn=run_turn)
    await scheduler.dispatch(room_id, human_id, seq=None)
    await scheduler.wait_idle()
    assert wakes == [(iris_id, True)]

    await db.set_last_read(pool, iris_id, room_id, 1)
    wakes.clear()
    await scheduler.dispatch(room_id, human_id, seq=None)
    await scheduler.wait_idle()
    assert wakes == [(chair_id, False)]


@pytest.mark.asyncio
async def test_crashed_say_leaves_trigger_open(
    pool: asyncpg.Pool, redis_client: redis.Redis
) -> None:
    room_id, human_id, chair_id, _ = await _moderated_room(pool)
    await db.insert_message(pool, room_id, human_id, "please greet")

    class CrashOnceWorld(DirectWorld):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]
            self.crashes = 0

        async def insert_message(self, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
            if self.crashes == 0:
                self.crashes += 1
                raise RuntimeError("crash before land")
            return await super().insert_message(*args, **kwargs)

    world = CrashOnceWorld(pool, redis_client)
    big = ScriptedChatModel(
        [
            tool_call("decide", {"action": "say", "body": "welcome everyone"}),
            tool_call("decide", {"action": "say", "body": "welcome everyone"}),
            tool_call("decide", {"action": "silence"}),
        ]
    )
    brain = Brain(world, small_model=ScriptedChatModel(), big_model=big)
    with pytest.raises(RuntimeError, match="crash before land"):
        await brain.run(chair_id, room_id)
    assert await _decisions(pool, room_id) == []

    result = await brain.run(chair_id, room_id)
    assert result.outcome == "moderated_silence"
    assert result.reply_body == "welcome everyone"
    rows = await _decisions(pool, room_id)
    assert [r["action"] for r in rows] == ["say", "silence"]
    stored = [m.body for m in await db.list_messages(pool, room_id)]
    assert stored.count("welcome everyone") == 1


@pytest.mark.asyncio
async def test_non_decide_tool_is_acked_before_corrective_note(
    pool: asyncpg.Pool, redis_client: redis.Redis
) -> None:
    room_id, human_id, chair_id, _members = await _moderated_room(pool)
    await db.insert_message(pool, room_id, human_id, "thoughts?")
    from langchain_core.messages import AIMessage

    extra = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "reply",
                "args": {"body": "I will answer"},
                "id": "call_reply",
            }
        ],
    )
    big = ScriptedChatModel(
        [
            extra,
            tool_call("decide", {"action": "call_on", "target": "Iris"}),
        ]
    )
    brain = Brain(
        DirectWorld(pool, redis_client),
        small_model=ScriptedChatModel(),
        big_model=big,
    )
    result = await brain.run(chair_id, room_id)
    assert result.outcome == "moderated_call"
    second = big.calls[1]
    texts = [str(getattr(m, "content", "")) for m in second]
    assert any("unknown tool reply" in t for t in texts)
    assert any(MODERATION_NOTE in t for t in texts)
    # Tool result must precede the corrective HumanMessage.
    kinds = [type(m).__name__ for m in second]
    tool_at = next(i for i, k in enumerate(kinds) if k == "ToolMessage")
    human_at = next(
        i
        for i, (k, t) in enumerate(zip(kinds, texts, strict=False))
        if k == "HumanMessage" and MODERATION_NOTE in t
    )
    assert tool_at < human_at


@pytest.mark.asyncio
async def test_dispatch_call_on_runs_target_via_real_wake(
    require_services: None,
) -> None:
    """Human POST → real dispatch → scripted decide(call_on) → target
    turn via Scheduler.wake_one, not a manual Brain.run(called_on=True)."""
    brains: dict[UUID, Brain] = {}

    async def run_turn(
        agent_id: UUID,
        room_id: UUID,
        *,
        called_on: bool = False,
        called_on_seq: int | None = None,
    ) -> object:
        return await brains[agent_id].run(
            agent_id,
            room_id,
            called_on=called_on,
            called_on_seq=called_on_seq,
        )

    app = create_app(run_turn=run_turn)
    async with app.router.lifespan_context(app):
        await db.truncate_all(app.state.pool)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            room = (
                await client.post(
                    "/rooms", json={"name": "e2e-rt", "mode": "moderated"}
                )
            ).json()
            human = (
                await client.post(
                    f"/rooms/{room['id']}/participants",
                    json={"kind": "human", "name": "Ada"},
                )
            ).json()
            chair = (
                await client.post(
                    f"/rooms/{room['id']}/participants",
                    json={
                        "kind": "agent",
                        "name": "Chair",
                        "role": "moderator",
                        "persona": "keeps time",
                    },
                )
            ).json()
            iris = (
                await client.post(
                    f"/rooms/{room['id']}/participants",
                    json={"kind": "agent", "name": "Iris", "persona": "brief"},
                )
            ).json()
            chair_id = UUID(chair["id"])
            iris_id = UUID(iris["id"])
            room_id = UUID(room["id"])

            async def on_call_on(
                rid: UUID, tid: UUID, trigger_seq: int
            ) -> None:
                await app.state.scheduler.wake_one(
                    rid, tid, called_on_seq=trigger_seq
                )

            chair_small = ScriptedChatModel()
            chair_big = ScriptedChatModel(
                [tool_call("decide", {"action": "call_on", "target": "Iris"})]
            )
            iris_small = ScriptedChatModel()
            iris_big = ScriptedChatModel(
                [tool_call("reply", {"body": "Iris via wake"})]
            )
            brains[chair_id] = Brain(
                DirectWorld(
                    app.state.pool, app.state.redis, on_call_on=on_call_on
                ),
                small_model=chair_small,
                big_model=chair_big,
            )
            brains[iris_id] = Brain(
                DirectWorld(app.state.pool, app.state.redis),
                small_model=iris_small,
                big_model=iris_big,
            )

            posted = await client.post(
                f"/rooms/{room['id']}/messages",
                json={"author_id": human["id"], "body": "who speaks?"},
            )
            assert posted.status_code == 200

            loop = asyncio.get_running_loop()
            deadline = loop.time() + 4.0
            while loop.time() < deadline:
                if any(
                    getattr(r, "agent_id", None) == iris_id
                    and getattr(r, "outcome", None) == "replied"
                    for r in app.state.scheduler.brain_results
                ):
                    await app.state.scheduler.wait_idle()
                    break
                await asyncio.sleep(0.05)
            else:
                raise AssertionError(
                    f"timed out: {app.state.scheduler.brain_results}"
                )

            assert chair_small.calls == []
            assert iris_small.calls == []
            assert chair_big.bound_tools == ["decide"]
            assert iris_big.bound_tools == ["reply", "claim"]
            stored = await db.list_messages(app.state.pool, room_id)
            assert any(m.body == "Iris via wake" for m in stored)


# ── Phase 7c: called-on decline is an explicit pass ──────────────────────


class _BoomModel:
    def __init__(self) -> None:
        self.calls: list[list] = []
        self.bound_tools: list[str] = []

    def bind_tools(self, _tools: object, **_kwargs: object) -> _BoomModel:
        return self

    async def ainvoke(self, messages: list, **_kwargs: object) -> object:
        self.calls.append(list(messages))
        raise RuntimeError("relay 400: Bad Request")


@pytest.mark.asyncio
async def test_called_on_decline_posts_pass_and_moderator_redirects(
    require_services: None,
) -> None:
    """call_on Iris → Iris declines → `Iris passes.` lands → Chair is
    woken on the new seq and can call_on Marcus. Scheduler/lane path,
    not a manual Brain.run for the redirect."""
    brains: dict[UUID, Brain] = {}

    async def run_turn(
        agent_id: UUID,
        room_id: UUID,
        *,
        called_on: bool = False,
        called_on_seq: int | None = None,
    ) -> object:
        return await brains[agent_id].run(
            agent_id,
            room_id,
            called_on=called_on,
            called_on_seq=called_on_seq,
        )

    app = create_app(run_turn=run_turn)
    async with app.router.lifespan_context(app):
        await db.truncate_all(app.state.pool)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            room = (
                await client.post(
                    "/rooms", json={"name": "pass-rt", "mode": "moderated"}
                )
            ).json()
            human = (
                await client.post(
                    f"/rooms/{room['id']}/participants",
                    json={"kind": "human", "name": "Ada"},
                )
            ).json()
            chair = (
                await client.post(
                    f"/rooms/{room['id']}/participants",
                    json={
                        "kind": "agent",
                        "name": "Chair",
                        "role": "moderator",
                    },
                )
            ).json()
            iris = (
                await client.post(
                    f"/rooms/{room['id']}/participants",
                    json={"kind": "agent", "name": "Iris"},
                )
            ).json()
            marcus = (
                await client.post(
                    f"/rooms/{room['id']}/participants",
                    json={"kind": "agent", "name": "Marcus"},
                )
            ).json()
            chair_id = UUID(chair["id"])
            iris_id = UUID(iris["id"])
            marcus_id = UUID(marcus["id"])
            room_id = UUID(room["id"])

            async def on_call_on(
                rid: UUID, tid: UUID, trigger_seq: int
            ) -> None:
                await app.state.scheduler.wake_one(
                    rid, tid, called_on_seq=trigger_seq
                )

            async def on_committed(row: object) -> None:
                await app.state.fanout_with_stall_reset(
                    app.state.hub, app.state.redis, row
                )

            brains[chair_id] = Brain(
                DirectWorld(
                    app.state.pool, app.state.redis, on_call_on=on_call_on
                ),
                small_model=ScriptedChatModel(),
                big_model=ScriptedChatModel(
                    [
                        tool_call("decide", {"action": "call_on", "target": "Iris"}),
                        tool_call(
                            "decide", {"action": "call_on", "target": "Marcus"}
                        ),
                    ]
                ),
                on_committed=on_committed,
            )
            brains[iris_id] = Brain(
                DirectWorld(app.state.pool, app.state.redis),
                small_model=ScriptedChatModel(),
                big_model=ScriptedChatModel(
                    [text_message("I'll sit this out")]
                ),
                on_committed=on_committed,
            )
            brains[marcus_id] = Brain(
                DirectWorld(app.state.pool, app.state.redis),
                small_model=ScriptedChatModel(),
                big_model=ScriptedChatModel(
                    [tool_call("reply", {"body": "Marcus after redirect"})]
                ),
                on_committed=on_committed,
            )

            posted = await client.post(
                f"/rooms/{room['id']}/messages",
                json={"author_id": human["id"], "body": "who speaks?"},
            )
            assert posted.status_code == 200

            loop = asyncio.get_running_loop()
            deadline = loop.time() + 4.0
            while loop.time() < deadline:
                stored = await db.list_messages(app.state.pool, room_id)
                decisions = await _decisions(app.state.pool, room_id)
                if (
                    any(m.body == "Iris passes." for m in stored)
                    and any(m.body == "Marcus after redirect" for m in stored)
                    and [r["action"] for r in decisions] == ["call_on", "call_on"]
                ):
                    await app.state.scheduler.wait_idle()
                    break
                await asyncio.sleep(0.05)
            else:
                raise AssertionError(
                    f"timed out: messages={[m.body for m in await db.list_messages(app.state.pool, room_id)]}"
                    f" decisions={await _decisions(app.state.pool, room_id)}"
                    f" results={app.state.scheduler.brain_results}"
                )

            stored = await db.list_messages(app.state.pool, room_id)
            bodies = [m.body for m in stored]
            assert "Iris passes." in bodies
            assert "Marcus after redirect" in bodies
            decisions = await _decisions(app.state.pool, room_id)
            assert [r["action"] for r in decisions] == ["call_on", "call_on"]
            assert [r["target_id"] for r in decisions] == [iris_id, marcus_id]
            assert decisions[1]["trigger_seq"] > decisions[0]["trigger_seq"]


@pytest.mark.asyncio
async def test_two_member_passes_both_land(
    pool: asyncpg.Pool, redis_client: redis.Redis
) -> None:
    room_id, human_id, _chair_id, member_ids = await _moderated_room(
        pool, members=2
    )
    iris_id, marcus_id = member_ids
    await db.insert_message(pool, room_id, human_id, "who speaks?")

    iris = Brain(
        DirectWorld(pool, redis_client),
        small_model=ScriptedChatModel(),
        big_model=ScriptedChatModel([text_message("no")]),
    )
    marcus = Brain(
        DirectWorld(pool, redis_client),
        small_model=ScriptedChatModel(),
        big_model=ScriptedChatModel([text_message("no")]),
    )
    first = await iris.run(iris_id, room_id, called_on=True)
    second = await marcus.run(marcus_id, room_id, called_on=True)
    assert first.outcome == "skipped"
    assert second.outcome == "skipped"
    bodies = [m.body for m in await db.list_messages(pool, room_id)]
    assert bodies.count("Iris passes.") == 1
    assert bodies.count("Marcus passes.") == 1


@pytest.mark.asyncio
async def test_called_on_llm_error_leaves_last_read_for_redelivery(
    pool: asyncpg.Pool,
    redis_client: redis.Redis,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(graph_mod, "LLM_RETRY_BACKOFF_S", 0)
    room_id, human_id, chair_id, member_ids = await _moderated_room(pool)
    iris_id = member_ids[0]
    await db.insert_message(pool, room_id, human_id, "pick a speaker")
    status, _ = await db.record_decision(
        pool, room_id, chair_id, 1, "call_on", iris_id
    )
    assert status == "won"
    assert await db.get_last_read(pool, iris_id, room_id) == 0

    brain = Brain(
        DirectWorld(pool, redis_client),
        small_model=ScriptedChatModel(),
        big_model=_BoomModel(),
    )
    result = await brain.run(iris_id, room_id, called_on=True)
    assert result.outcome == "llm_error"
    assert await db.get_last_read(pool, iris_id, room_id) == 0
    assert "Iris passes." not in [m.body for m in await db.list_messages(pool, room_id)]

    scheduler = Scheduler(pool)
    targets = await scheduler._route_wake(room_id, human_id, None)
    assert [(t.id, seq) for t, seq in targets] == [(iris_id, 1)]


@pytest.mark.asyncio
async def test_loop_cap_skip_does_not_post_pass(
    pool: asyncpg.Pool, redis_client: redis.Redis
) -> None:
    room_id, _human_id, _chair_id, member_ids = await _moderated_room(pool)
    iris_id = member_ids[0]
    # 2 agents → cap 8. Stretch is room-level since the last human.
    for i in range(8):
        await db.insert_message(pool, room_id, iris_id, f"circle {i}")
    brain = Brain(
        DirectWorld(pool, redis_client),
        small_model=ScriptedChatModel(),
        big_model=ScriptedChatModel(),
    )
    result = await brain.run(iris_id, room_id, called_on=True)
    assert result.outcome == "skipped"
    assert "loop cap" in (result.triage_reason or "")
    bodies = [m.body for m in await db.list_messages(pool, room_id)]
    assert "Iris passes." not in bodies


@pytest.mark.asyncio
async def test_stale_called_on_pass_is_dropped(
    pool: asyncpg.Pool, redis_client: redis.Redis
) -> None:
    room_id, human_id, _chair_id, member_ids = await _moderated_room(
        pool, members=2
    )
    iris_id, marcus_id = member_ids
    await db.insert_message(pool, room_id, human_id, "pick a speaker")

    async def decline_after_race(_messages: list) -> object:
        await db.insert_message(pool, room_id, marcus_id, "Marcus already spoke")
        return text_message("I'll sit this out")

    brain = Brain(
        DirectWorld(pool, redis_client),
        small_model=ScriptedChatModel(),
        big_model=ScriptedChatModel([decline_after_race]),
    )
    result = await brain.run(iris_id, room_id, called_on=True)
    assert result.outcome == "skipped"
    bodies = [m.body for m in await db.list_messages(pool, room_id)]
    assert "Iris passes." not in bodies
    assert "Marcus already spoke" in bodies


@pytest.mark.asyncio
async def test_digest_shows_pass_as_transcript_row(
    pool: asyncpg.Pool, redis_client: redis.Redis
) -> None:
    room_id, human_id, _chair_id, member_ids = await _moderated_room(pool)
    iris_id = member_ids[0]
    await db.insert_message(pool, room_id, human_id, "pick a speaker")
    brain = Brain(
        DirectWorld(pool, redis_client),
        small_model=ScriptedChatModel(),
        big_model=ScriptedChatModel([text_message("no")]),
    )
    await brain.run(iris_id, room_id, called_on=True)
    markdown = await build_room_digest(pool, room_id)
    assert markdown is not None
    assert "| Iris | Iris passes. |" in markdown


# ── Phase 7c: moderator say is non-terminal ──────────────────────────────


@pytest.mark.asyncio
async def test_say_then_call_on_in_one_turn(
    pool: asyncpg.Pool, redis_client: redis.Redis
) -> None:
    room_id, human_id, chair_id, member_ids = await _moderated_room(pool)
    iris_id = member_ids[0]
    await db.insert_message(pool, room_id, human_id, "open the table")
    wakes: list[UUID] = []

    async def on_call_on(_rid: UUID, tid: UUID, _trigger_seq: int) -> None:
        wakes.append(tid)

    brain = Brain(
        DirectWorld(pool, redis_client, on_call_on=on_call_on),
        small_model=ScriptedChatModel(),
        big_model=ScriptedChatModel(
            [
                tool_call("decide", {"action": "say", "body": "welcome"}),
                tool_call("decide", {"action": "call_on", "target": "Iris"}),
            ]
        ),
    )
    result = await brain.run(chair_id, room_id)
    assert result.outcome == "moderated_call"
    assert result.reply_body == "welcome"
    assert wakes == [iris_id]
    stored = [m.body for m in await db.list_messages(pool, room_id)]
    assert "welcome" in stored
    rows = await _decisions(pool, room_id)
    assert [r["action"] for r in rows] == ["say", "call_on"]
    assert rows[0]["trigger_seq"] == 1
    assert rows[1]["trigger_seq"] == 2
    assert rows[1]["target_id"] == iris_id


@pytest.mark.asyncio
async def test_say_budget_second_say_errors_then_call_on(
    pool: asyncpg.Pool, redis_client: redis.Redis
) -> None:
    room_id, human_id, chair_id, member_ids = await _moderated_room(pool)
    iris_id = member_ids[0]
    await db.insert_message(pool, room_id, human_id, "open the table")
    wakes: list[UUID] = []

    async def on_call_on(_rid: UUID, tid: UUID, _trigger_seq: int) -> None:
        wakes.append(tid)

    big = ScriptedChatModel(
        [
            tool_call("decide", {"action": "say", "body": "first"}),
            tool_call("decide", {"action": "say", "body": "second"}),
            tool_call("decide", {"action": "call_on", "target": "Iris"}),
        ]
    )
    brain = Brain(
        DirectWorld(pool, redis_client, on_call_on=on_call_on),
        small_model=ScriptedChatModel(),
        big_model=big,
    )
    result = await brain.run(chair_id, room_id)
    assert result.outcome == "moderated_call"
    assert result.reply_body == "first"
    assert wakes == [iris_id]
    # The error ToolMessage is on the hop after the rejected second say.
    told = [str(getattr(m, "content", "")) for m in big.calls[2]]
    assert any(MODERATOR_SAY_ERROR in t for t in told)
    stored = [m.body for m in await db.list_messages(pool, room_id)]
    assert stored.count("first") == 1
    assert "second" not in stored
    assert [r["action"] for r in await _decisions(pool, room_id)] == [
        "say",
        "call_on",
    ]


@pytest.mark.asyncio
async def test_say_then_silence_ends_with_no_wake(
    pool: asyncpg.Pool, redis_client: redis.Redis
) -> None:
    room_id, human_id, chair_id, _members = await _moderated_room(pool)
    await db.insert_message(pool, room_id, human_id, "carry on")
    wakes: list[UUID] = []

    async def on_call_on(_rid: UUID, tid: UUID, _trigger_seq: int) -> None:
        wakes.append(tid)

    brain = Brain(
        DirectWorld(pool, redis_client, on_call_on=on_call_on),
        small_model=ScriptedChatModel(),
        big_model=ScriptedChatModel(
            [
                tool_call("decide", {"action": "say", "body": "noted"}),
                tool_call("decide", {"action": "silence"}),
            ]
        ),
    )
    result = await brain.run(chair_id, room_id)
    assert result.outcome == "moderated_silence"
    assert result.reply_body == "noted"
    assert wakes == []
    assert [r["action"] for r in await _decisions(pool, room_id)] == [
        "say",
        "silence",
    ]


# ── review follow-up: called_on trigger_seq, Redis hint, silence nudge ──


@pytest.mark.asyncio
async def test_redelivered_call_on_after_reply_does_not_pass(
    pool: asyncpg.Pool, redis_client: redis.Redis
) -> None:
    """Construction (a): stall redelivers call_on while the member is
    still on their first turn. After the reply lands, the coalesced
    rerun must not post a false pass."""
    room_id, human_id, chair_id, member_ids = await _moderated_room(pool)
    iris_id = member_ids[0]
    await db.insert_message(pool, room_id, human_id, "pick a speaker")
    status, _ = await db.record_decision(
        pool, room_id, chair_id, 1, "call_on", iris_id
    )
    assert status == "won"

    started = asyncio.Event()
    release = asyncio.Event()
    turns = 0
    brain = Brain(
        DirectWorld(pool, redis_client),
        small_model=ScriptedChatModel(),
        big_model=ScriptedChatModel(
            [
                tool_call("reply", {"body": "Iris answering"}),
                text_message("already said my piece"),
            ]
        ),
    )

    async def run_turn(
        agent_id: UUID,
        rid: UUID,
        *,
        called_on: bool = False,
        called_on_seq: int | None = None,
    ) -> object:
        nonlocal turns
        if agent_id != iris_id:
            return None
        turns += 1
        if turns == 1:
            started.set()
            await release.wait()
        return await brain.run(
            agent_id, rid, called_on=called_on, called_on_seq=called_on_seq
        )

    scheduler = Scheduler(pool, run_turn=run_turn, redis_client=redis_client)
    await scheduler.wake_one(room_id, iris_id, called_on_seq=1)
    await started.wait()
    await scheduler.dispatch(room_id, human_id, seq=None)
    release.set()
    await scheduler.wait_idle()

    bodies = [m.body for m in await db.list_messages(pool, room_id)]
    assert "Iris answering" in bodies
    assert "Iris passes." not in bodies
    assert turns == 2


@pytest.mark.asyncio
async def test_say_mention_then_call_on_same_member_no_false_pass(
    require_services: None,
) -> None:
    """Construction (b): Chair say `@Iris` then call_on Iris. Mention
    fan-out plus call_on must not produce a false pass."""
    brains: dict[UUID, Brain] = {}

    async def run_turn(
        agent_id: UUID,
        room_id: UUID,
        *,
        called_on: bool = False,
        called_on_seq: int | None = None,
    ) -> object:
        return await brains[agent_id].run(
            agent_id,
            room_id,
            called_on=called_on,
            called_on_seq=called_on_seq,
        )

    app = create_app(run_turn=run_turn)
    async with app.router.lifespan_context(app):
        await db.truncate_all(app.state.pool)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            room = (
                await client.post(
                    "/rooms",
                    json={"name": "mention-call", "mode": "moderated"},
                )
            ).json()
            human = (
                await client.post(
                    f"/rooms/{room['id']}/participants",
                    json={"kind": "human", "name": "Ada"},
                )
            ).json()
            chair = (
                await client.post(
                    f"/rooms/{room['id']}/participants",
                    json={"kind": "agent", "name": "Chair", "role": "moderator"},
                )
            ).json()
            iris = (
                await client.post(
                    f"/rooms/{room['id']}/participants",
                    json={"kind": "agent", "name": "Iris"},
                )
            ).json()
            marcus = (
                await client.post(
                    f"/rooms/{room['id']}/participants",
                    json={"kind": "agent", "name": "Marcus"},
                )
            ).json()
            chair_id = UUID(chair["id"])
            iris_id = UUID(iris["id"])
            marcus_id = UUID(marcus["id"])
            room_id = UUID(room["id"])

            async def on_call_on(
                rid: UUID, tid: UUID, trigger_seq: int
            ) -> None:
                await app.state.scheduler.wake_one(
                    rid, tid, called_on_seq=trigger_seq
                )

            async def on_committed(row: object) -> None:
                await app.state.fanout_with_stall_reset(
                    app.state.hub, app.state.redis, row
                )

            brains[chair_id] = Brain(
                DirectWorld(
                    app.state.pool, app.state.redis, on_call_on=on_call_on
                ),
                small_model=ScriptedChatModel(),
                big_model=ScriptedChatModel(
                    [
                        tool_call(
                            "decide",
                            {"action": "say", "body": "@Iris welcome"},
                        ),
                        tool_call(
                            "decide", {"action": "call_on", "target": "Iris"}
                        ),
                        tool_call("decide", {"action": "silence"}),
                    ]
                ),
                on_committed=on_committed,
            )
            brains[iris_id] = Brain(
                DirectWorld(app.state.pool, app.state.redis),
                small_model=ScriptedChatModel(
                    [
                        triage_message(
                            actionable=True, reason="named", response_mode="me"
                        )
                    ]
                ),
                big_model=ScriptedChatModel(
                    [
                        tool_call("reply", {"body": "Iris here"}),
                        text_message("already answered"),
                    ]
                ),
                on_committed=on_committed,
            )
            brains[marcus_id] = Brain(
                DirectWorld(app.state.pool, app.state.redis),
                small_model=ScriptedChatModel(),
                big_model=ScriptedChatModel([text_message("no")]),
                on_committed=on_committed,
            )

            posted = await client.post(
                f"/rooms/{room['id']}/messages",
                json={"author_id": human["id"], "body": "open the table"},
            )
            assert posted.status_code == 200

            loop = asyncio.get_running_loop()
            deadline = loop.time() + 4.0
            while loop.time() < deadline:
                stored = await db.list_messages(app.state.pool, room_id)
                if any(m.body == "Iris here" for m in stored):
                    await app.state.scheduler.wait_idle()
                    break
                await asyncio.sleep(0.05)
            else:
                raise AssertionError(
                    f"timed out: {[m.body for m in await db.list_messages(app.state.pool, room_id)]}"
                )

            stored = await db.list_messages(app.state.pool, room_id)
            bodies = [m.body for m in stored]
            assert bodies.count("Iris here") == 1
            assert "Iris passes." not in bodies
            assert "Marcus passes." not in bodies
            member_replies = [
                m
                for m in stored
                if m.author_id in {iris_id, marcus_id}
            ]
            assert [m.body for m in member_replies] == ["Iris here"]


@pytest.mark.asyncio
async def test_in_process_wake_writes_no_redis_hint(
    pool: asyncpg.Pool, redis_client: redis.Redis
) -> None:
    room_id, _human_id, _chair_id, member_ids = await _moderated_room(pool)
    iris_id = member_ids[0]
    started = asyncio.Event()
    release = asyncio.Event()

    async def run_turn(
        _agent_id: UUID,
        _rid: UUID,
        *,
        called_on: bool = False,
        called_on_seq: int | None = None,
    ) -> None:
        started.set()
        await release.wait()

    scheduler = Scheduler(pool, run_turn=run_turn, redis_client=redis_client)
    await scheduler.wake_one(room_id, iris_id, called_on_seq=1)
    await started.wait()
    assert not await redis_client.exists(hint_key(iris_id, room_id))
    release.set()
    await scheduler.wait_idle()


@pytest.mark.asyncio
async def test_coalesce_overwrite_leaves_no_stale_redis_hint(
    pool: asyncpg.Pool, redis_client: redis.Redis
) -> None:
    room_id, _human_id, _chair_id, member_ids = await _moderated_room(pool)
    iris_id = member_ids[0]
    started = asyncio.Event()
    release = asyncio.Event()
    seen: list[tuple[bool, int | None]] = []

    async def run_turn(
        _agent_id: UUID,
        _rid: UUID,
        *,
        called_on: bool = False,
        called_on_seq: int | None = None,
    ) -> None:
        seen.append((called_on, called_on_seq))
        if len(seen) == 1:
            started.set()
            await release.wait()

    scheduler = Scheduler(pool, run_turn=run_turn, redis_client=redis_client)
    await scheduler.wake_one(room_id, iris_id, called_on_seq=1)
    await started.wait()
    await scheduler.wake_one(room_id, iris_id, called_on_seq=3)
    assert not await redis_client.exists(hint_key(iris_id, room_id))
    release.set()
    await scheduler.wait_idle()
    assert await consume_called_on(redis_client, iris_id, room_id) is None
    assert seen[0] == (True, 1)
    assert seen[1] == (True, 3)


@pytest.mark.asyncio
async def test_nudge_skips_moderator_after_silence_at_last_seq(
    pool: asyncpg.Pool,
) -> None:
    room_id, human_id, chair_id, _ = await _moderated_room(pool)
    await db.insert_message(pool, room_id, human_id, "carry on")
    status, _ = await db.record_decision(
        pool, room_id, chair_id, 1, "silence", None
    )
    assert status == "won"
    wakes: list[tuple[UUID, bool]] = []

    async def run_turn(
        agent_id: UUID, rid: UUID, *, called_on: bool = False
    ) -> None:
        wakes.append((agent_id, called_on))

    scheduler = Scheduler(pool, run_turn=run_turn)
    await scheduler.dispatch(room_id, human_id, seq=None)
    await scheduler.wait_idle()
    assert wakes == []
    targets = await scheduler._route_wake(room_id, human_id, None)
    assert targets == []


@pytest.mark.asyncio
async def test_duplicate_called_on_pass_is_dropped(
    pool: asyncpg.Pool, redis_client: redis.Redis
) -> None:
    room_id, human_id, _chair_id, member_ids = await _moderated_room(
        pool, members=2
    )
    iris_id, marcus_id = member_ids
    await db.insert_message(pool, room_id, human_id, "pick a speaker")
    await db.insert_message(pool, room_id, marcus_id, "Iris passes.")
    brain = Brain(
        DirectWorld(pool, redis_client),
        small_model=ScriptedChatModel(),
        big_model=ScriptedChatModel([text_message("no")]),
    )
    result = await brain.run(iris_id, room_id, called_on=True)
    assert result.outcome == "skipped"
    bodies = [m.body for m in await db.list_messages(pool, room_id)]
    assert bodies.count("Iris passes.") == 1
    assert await db.get_last_read(pool, iris_id, room_id) > 0


@pytest.mark.asyncio
async def test_say_logs_when_decision_row_blocked(
    pool: asyncpg.Pool,
    redis_client: redis.Redis,
    caplog: pytest.LogCaptureFixture,
) -> None:
    room_id, human_id, chair_id, _ = await _moderated_room(pool)
    await db.insert_message(pool, room_id, human_id, "please greet")
    status, _ = await db.record_decision(
        pool, room_id, chair_id, 1, "say", None
    )
    assert status == "won"
    brain = Brain(
        DirectWorld(pool, redis_client),
        small_model=ScriptedChatModel(),
        big_model=ScriptedChatModel(
            [
                tool_call("decide", {"action": "say", "body": "welcome"}),
                tool_call("decide", {"action": "silence"}),
            ]
        ),
    )
    with caplog.at_level(logging.INFO, logger="agora.brain"):
        result = await brain.run(chair_id, room_id)
    assert result.outcome == "moderated_silence"
    assert "welcome" in [m.body for m in await db.list_messages(pool, room_id)]
    assert "decision row blocked" in caplog.text


# ── Phase 7d: call_on budget per human message ───────────────────────────


@pytest.mark.asyncio
async def test_call_on_budget_fourth_is_rejected(
    pool: asyncpg.Pool, redis_client: redis.Redis
) -> None:
    room_id, human_id, chair_id, member_ids = await _moderated_room(
        pool, members=2
    )
    iris_id, marcus_id = member_ids
    await db.insert_message(pool, room_id, human_id, "exactly one answer")
    world = DirectWorld(pool, redis_client)

    targets = ["Iris", "Marcus", "Iris"]
    speakers = [iris_id, marcus_id, iris_id]
    for i, (target, speaker) in enumerate(zip(targets, speakers, strict=True)):
        result = await _chair_call_on(
            pool, redis_client, room_id, chair_id, target
        )
        assert result.outcome == "moderated_call", i
        await db.insert_message(pool, room_id, speaker, f"answer {i}")

    assert await world.call_ons_since_human(room_id) == 3
    big = ScriptedChatModel(
        [
            tool_call("decide", {"action": "call_on", "target": "Marcus"}),
            tool_call("decide", {"action": "silence"}),
        ]
    )
    brain = Brain(
        DirectWorld(pool, redis_client), small_model=ScriptedChatModel(), big_model=big
    )
    fourth = await brain.run(chair_id, room_id)
    assert fourth.outcome == "moderated_silence"
    told = [str(getattr(m, "content", "")) for m in big.calls[1]]
    assert any(DECIDE_CALLS_EXHAUSTED in t for t in told)
    rows = await _decisions(pool, room_id)
    assert [r["action"] for r in rows] == ["call_on", "call_on", "call_on", "silence"]
    assert await world.call_ons_since_human(room_id) == 3


@pytest.mark.asyncio
async def test_call_on_budget_resets_on_new_human(
    pool: asyncpg.Pool, redis_client: redis.Redis
) -> None:
    room_id, human_id, chair_id, member_ids = await _moderated_room(
        pool, members=2
    )
    iris_id, marcus_id = member_ids
    await db.insert_message(pool, room_id, human_id, "first ask")
    for target, speaker in (
        ("Iris", iris_id),
        ("Marcus", marcus_id),
        ("Iris", iris_id),
    ):
        result = await _chair_call_on(
            pool, redis_client, room_id, chair_id, target
        )
        assert result.outcome == "moderated_call"
        await db.insert_message(pool, room_id, speaker, f"done {target}")

    await db.insert_message(pool, room_id, human_id, "new question")
    world = DirectWorld(pool, redis_client)
    assert await world.call_ons_since_human(room_id) == 0
    again = await _chair_call_on(pool, redis_client, room_id, chair_id, "Marcus")
    assert again.outcome == "moderated_call"
    assert await world.call_ons_since_human(room_id) == 1
    rows = await _decisions(pool, room_id)
    assert [r["action"] for r in rows] == ["call_on"] * 4


@pytest.mark.asyncio
async def test_call_on_budget_does_not_block_say_or_silence(
    pool: asyncpg.Pool, redis_client: redis.Redis
) -> None:
    room_id, human_id, chair_id, member_ids = await _moderated_room(
        pool, members=2
    )
    iris_id, marcus_id = member_ids
    await db.insert_message(pool, room_id, human_id, "ask")
    for target, speaker in (
        ("Iris", iris_id),
        ("Marcus", marcus_id),
        ("Iris", iris_id),
    ):
        result = await _chair_call_on(
            pool, redis_client, room_id, chair_id, target
        )
        assert result.outcome == "moderated_call"
        await db.insert_message(pool, room_id, speaker, f"heard {target}")

    brain = Brain(
        DirectWorld(pool, redis_client),
        small_model=ScriptedChatModel(),
        big_model=ScriptedChatModel(
            [
                tool_call("decide", {"action": "say", "body": "closing"}),
                tool_call("decide", {"action": "silence"}),
            ]
        ),
    )
    result = await brain.run(chair_id, room_id)
    assert result.outcome == "moderated_silence"
    assert result.reply_body == "closing"
    assert "closing" in [m.body for m in await db.list_messages(pool, room_id)]
    actions = [r["action"] for r in await _decisions(pool, room_id)]
    assert actions == ["call_on", "call_on", "call_on", "say", "silence"]


@pytest.mark.asyncio
async def test_http_world_call_ons_since_human(app_client: tuple) -> None:
    from daemon.world_http import HttpWorld

    _app, client = app_client
    computer = (await client.post("/computers", json={"name": "laptop"})).json()
    room = (
        await client.post("/rooms", json={"name": "cap-rt", "mode": "moderated"})
    ).json()
    human = (
        await client.post(
            f"/rooms/{room['id']}/participants",
            json={"kind": "human", "name": "Ada"},
        )
    ).json()
    chair = (
        await client.post(
            f"/rooms/{room['id']}/participants",
            json={
                "kind": "agent",
                "name": "Chair",
                "role": "moderator",
                "computer_id": computer["id"],
            },
        )
    ).json()
    iris = (
        await client.post(
            f"/rooms/{room['id']}/participants",
            json={"kind": "agent", "name": "Iris"},
        )
    ).json()
    marcus = (
        await client.post(
            f"/rooms/{room['id']}/participants",
            json={"kind": "agent", "name": "Marcus"},
        )
    ).json()
    posted = await client.post(
        f"/rooms/{room['id']}/messages",
        json={"author_id": human["id"], "body": "begin"},
    )
    posted.raise_for_status()
    assert posted.json()["seq"] == 1

    world = HttpWorld(client, computer["token"])
    world.bind_actor(UUID(chair["id"]))
    room_id = UUID(room["id"])
    chair_id = UUID(chair["id"])
    targets = [UUID(iris["id"]), UUID(marcus["id"]), UUID(iris["id"])]
    speakers = [iris, marcus, iris]
    for i, (target_id, speaker) in enumerate(zip(targets, speakers, strict=True)):
        recorded = await world.record_decision(
            room_id, chair_id, trigger_seq=i + 1, action="call_on", target_id=target_id
        )
        assert recorded.status == "won"
        follow = await client.post(
            f"/rooms/{room['id']}/messages",
            json={"author_id": speaker["id"], "body": f"answer {i}"},
        )
        follow.raise_for_status()

    assert await world.call_ons_since_human(room_id) == 3
    raw = await client.get(
        "/runtime/call-ons-since-human",
        params={"agent_id": chair["id"], "room_id": room["id"]},
        headers={"Authorization": f"Bearer {computer['token']}"},
    )
    assert raw.status_code == 200
    assert raw.json() == {"count": 3}

    reset = await client.post(
        f"/rooms/{room['id']}/messages",
        json={"author_id": human["id"], "body": "new ask"},
    )
    reset.raise_for_status()
    assert await world.call_ons_since_human(room_id) == 0
    assert (
        await client.get(
            "/runtime/call-ons-since-human",
            params={"agent_id": chair["id"], "room_id": room["id"]},
            headers={"Authorization": f"Bearer {computer['token']}"},
        )
    ).json() == {"count": 0}
