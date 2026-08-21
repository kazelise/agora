from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from uuid import UUID

import asyncpg
import pytest
import redis.asyncio as redis

import brain.graph as graph_mod
from brain.graph import CLAIM_KEY_ERROR, Brain
from brain.policy import big_model_name
from brain.world_direct import DirectWorld
from server import db
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


async def _room(
    pool: asyncpg.Pool,
    *,
    agents: int = 1,
) -> tuple[UUID, UUID, list[UUID]]:
    room = await db.create_room(pool, "brain-room")
    human = await db.add_participant(pool, room.id, "human", "Ada", None)
    agent_ids: list[UUID] = []
    for i in range(agents):
        agent = await db.add_participant(
            pool, room.id, "agent", f"Agent{i}", f"persona-{i}"
        )
        agent_ids.append(agent.id)
    return room.id, human.id, agent_ids


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


@pytest.mark.asyncio
async def test_triage_not_actionable_skips_big_model(
    pool: asyncpg.Pool, redis_client: redis.Redis
) -> None:
    room_id, human_id, agent_ids = await _room(pool)
    await db.insert_message(pool, room_id, human_id, "just chatting")
    small = ScriptedChatModel(
        [triage_message(actionable=False, reason="small talk", response_mode="me")]
    )
    big = ScriptedChatModel()
    brain = Brain(DirectWorld(pool, redis_client), small_model=small, big_model=big)

    result = await brain.run(agent_ids[0], room_id)

    assert result.outcome == "skipped"
    assert big.calls == []
    messages = await db.list_messages(pool, room_id)
    assert [m.body for m in messages] == ["just chatting"]
    rows = await _calls(pool, room_id)
    assert len(rows) == 1
    assert rows[0]["purpose"] == "triage"


@pytest.mark.asyncio
async def test_empty_inbox_makes_zero_llm_calls(
    pool: asyncpg.Pool, redis_client: redis.Redis
) -> None:
    room_id, _human_id, agent_ids = await _room(pool)
    small = ScriptedChatModel()
    big = ScriptedChatModel()
    brain = Brain(DirectWorld(pool, redis_client), small_model=small, big_model=big)

    result = await brain.run(agent_ids[0], room_id)

    assert result.outcome == "empty"
    assert small.calls == []
    assert big.calls == []
    assert await _calls(pool, room_id) == []


@pytest.mark.asyncio
async def test_freshness_hold_then_different_reply(
    pool: asyncpg.Pool, redis_client: redis.Redis
) -> None:
    room_id, human_id, agent_ids = await _room(pool, agents=2)
    iris_id, marcus_id = agent_ids
    await db.insert_message(pool, room_id, human_id, "next number after 2")

    async def first_reply(_messages: list) -> object:
        await db.insert_message(pool, room_id, marcus_id, "3")
        return tool_call("reply", {"body": "3"})

    small = ScriptedChatModel(
        [triage_message(actionable=True, reason="counting", response_mode="each")]
    )
    big = ScriptedChatModel([first_reply, tool_call("reply", {"body": "4"})])
    brain = Brain(DirectWorld(pool, redis_client), small_model=small, big_model=big)

    result = await brain.run(iris_id, room_id)

    assert result.outcome == "replied"
    assert result.hold_count == 1
    assert result.reply_body == "4"
    assert len(big.calls) == 2
    second_prompt = " ".join(
        getattr(m, "content", "") if not isinstance(getattr(m, "content", ""), list) else ""
        for m in big.calls[1]
    )
    assert "New messages landed while you were composing" in second_prompt
    assert "[seq=" in second_prompt and "3" in second_prompt
    stored = await db.list_messages(pool, room_id)
    assert [(m.author_id, m.body) for m in stored] == [
        (human_id, "next number after 2"),
        (marcus_id, "3"),
        (iris_id, "4"),
    ]


@pytest.mark.asyncio
async def test_claim_race_one_winner_loser_is_told(
    pool: asyncpg.Pool, redis_client: redis.Redis
) -> None:
    room_id, human_id, agent_ids = await _room(pool, agents=2)
    await db.insert_message(pool, room_id, human_id, "one of you answer q1")

    def after_claim(messages: list) -> object:
        texts = [getattr(m, "content", "") for m in messages]
        if any(t == "lost" for t in texts):
            return text_message("sitting out")
        return tool_call("reply", {"body": "the answer"})

    def make_big() -> ScriptedChatModel:
        return ScriptedChatModel(
            [tool_call("claim", {"task_key": "t1:answer-q1"}), after_claim]
        )

    small_a = ScriptedChatModel(
        [triage_message(actionable=True, reason="pick one", response_mode="one-of-us")]
    )
    small_b = ScriptedChatModel(
        [triage_message(actionable=True, reason="pick one", response_mode="one-of-us")]
    )
    big_a = make_big()
    big_b = make_big()
    brain_a = Brain(DirectWorld(pool, redis_client), small_model=small_a, big_model=big_a)
    brain_b = Brain(DirectWorld(pool, redis_client), small_model=small_b, big_model=big_b)

    results = await asyncio.gather(
        brain_a.run(agent_ids[0], room_id),
        brain_b.run(agent_ids[1], room_id),
    )

    claims = await pool.fetch(
        "SELECT claimed_by FROM claims WHERE room_id = $1 AND task_key = $2",
        room_id,
        "t1",
    )
    assert len(claims) == 1
    winner_id = claims[0]["claimed_by"]
    winner = next(r for r in results if r.agent_id == winner_id)
    loser = next(r for r in results if r.agent_id != winner_id)
    assert winner.claims == (("t1", "won"),)
    assert loser.claims == (("t1", "lost"),)
    loser_big = big_a if loser.agent_id == agent_ids[0] else big_b
    assert len(loser_big.calls) == 2
    told = [getattr(m, "content", "") for m in loser_big.calls[1]]
    assert "lost" in told


@pytest.mark.asyncio
async def test_ledger_records_triage_plus_two_turn_hops(
    pool: asyncpg.Pool, redis_client: redis.Redis
) -> None:
    room_id, human_id, agent_ids = await _room(pool)
    await db.insert_message(pool, room_id, human_id, "pick a speaker")
    small = ScriptedChatModel(
        [triage_message(actionable=True, reason="one-of-us", response_mode="one-of-us")]
    )
    big = ScriptedChatModel(
        [
            tool_call("claim", {"task_key": "t1:intro"}),
            tool_call("reply", {"body": "I will intro"}),
        ]
    )
    brain = Brain(DirectWorld(pool, redis_client), small_model=small, big_model=big)

    await brain.run(agent_ids[0], room_id)

    rows = await _calls(pool, room_id)
    assert [r["purpose"] for r in rows] == ["triage", "turn", "turn"]
    assert rows[0]["model"]
    assert rows[1]["purpose"] == "turn"
    assert rows[2]["purpose"] == "turn"


def test_triage_rejects_big_model_name() -> None:
    with pytest.raises(ValueError, match="triage must not use the big model"):
        Brain(
            world=None,  # type: ignore[arg-type]
            small_model=object(),
            big_model=object(),
            small_model_name=big_model_name(),
        )


@pytest.mark.asyncio
async def test_claim_freeform_key_rejected_then_anchored_wins(
    pool: asyncpg.Pool, redis_client: redis.Redis
) -> None:
    room_id, human_id, agent_ids = await _room(pool)
    await db.insert_message(pool, room_id, human_id, "one of you intro")
    small = ScriptedChatModel(
        [triage_message(actionable=True, reason="one-of-us", response_mode="one-of-us")]
    )
    big = ScriptedChatModel(
        [
            tool_call("claim", {"task_key": "room-purpose-intro"}),
            tool_call("claim", {"task_key": "t1:intro"}),
            tool_call("reply", {"body": "this room is agora"}),
        ]
    )
    brain = Brain(DirectWorld(pool, redis_client), small_model=small, big_model=big)

    result = await brain.run(agent_ids[0], room_id)

    assert result.outcome == "replied"
    assert result.claims == (("t1", "won"),)
    rows = await pool.fetch("SELECT task_key FROM claims WHERE room_id = $1", room_id)
    assert [r["task_key"] for r in rows] == ["t1"]
    told = [getattr(m, "content", "") for m in big.calls[1]]
    assert any(CLAIM_KEY_ERROR in str(t) for t in told)


@pytest.mark.asyncio
async def test_commit_race_after_freshness_holds(
    pool: asyncpg.Pool,
    redis_client: redis.Redis,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Peer's row lands after freshness passed but before commit.

    The cheap freshness node is shown a stale last_seq; the transactional
    insert is the one that must HOLD.
    """
    room_id, human_id, agent_ids = await _room(pool, agents=2)
    iris_id, marcus_id = agent_ids
    await db.insert_message(pool, room_id, human_id, "next number after 2")

    real_last_seq = db.get_room_last_seq
    freshness_reads = {"n": 0}

    async def stale_then_real(
        pool_arg: asyncpg.Pool, rid: UUID
    ) -> int:
        freshness_reads["n"] += 1
        if freshness_reads["n"] == 1:
            seen = await real_last_seq(pool_arg, rid)
            await db.insert_message(pool_arg, rid, marcus_id, "3")
            return seen
        return await real_last_seq(pool_arg, rid)

    monkeypatch.setattr(db, "get_room_last_seq", stale_then_real)

    small = ScriptedChatModel(
        [triage_message(actionable=True, reason="counting", response_mode="each")]
    )
    big = ScriptedChatModel(
        [
            tool_call("reply", {"body": "3"}),
            tool_call("reply", {"body": "4"}),
        ]
    )
    brain = Brain(DirectWorld(pool, redis_client), small_model=small, big_model=big)

    result = await brain.run(iris_id, room_id)

    assert result.outcome == "replied"
    assert result.hold_count == 1
    assert result.reply_body == "4"
    assert freshness_reads["n"] >= 1
    stored = await db.list_messages(pool, room_id)
    assert [(m.author_id, m.body) for m in stored] == [
        (human_id, "next number after 2"),
        (marcus_id, "3"),
        (iris_id, "4"),
    ]
    iris_bodies = [m.body for m in stored if m.author_id == iris_id]
    assert iris_bodies == ["4"]


class _BoomModel:
    """Raises on every ainvoke. bind_tools returns self so the graph stays here."""

    def __init__(self) -> None:
        self.calls: list[list] = []

    def bind_tools(self, _tools: object, **_kwargs: object) -> _BoomModel:
        return self

    async def ainvoke(self, messages: list, **_kwargs: object) -> object:
        self.calls.append(list(messages))
        raise RuntimeError("relay 400: Bad Request")


@pytest.mark.asyncio
async def test_big_model_failure_ends_turn_as_llm_error(
    pool: asyncpg.Pool,
    redis_client: redis.Redis,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(graph_mod, "LLM_RETRY_BACKOFF_S", 0)
    room_id, human_id, agent_ids = await _room(pool)
    await db.insert_message(pool, room_id, human_id, "please reply")
    small = ScriptedChatModel(
        [triage_message(actionable=True, reason="me", response_mode="me")]
    )
    big = _BoomModel()
    brain = Brain(DirectWorld(pool, redis_client), small_model=small, big_model=big)

    result = await brain.run(agent_ids[0], room_id)

    assert result.outcome == "llm_error"
    assert result.reply_body is None
    assert len(big.calls) == 2
    stored = await db.list_messages(pool, room_id)
    assert [m.body for m in stored] == ["please reply"]
    rows = await _calls(pool, room_id)
    assert [r["purpose"] for r in rows] == ["triage"]
