"""Coordination hardening borrowed from Cumora and yuanzhuo.

- verbatim-dup: atomic in-transaction duplicate suppression (Cumora §5b)
- hold tokens: send_anyway must acknowledge a server-shown HOLD (§5d)
- loop cap: agent-only runs past a counted floor are stale (§6)
- digest: yuanzhuo-style markdown export of the room
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from uuid import UUID, uuid4

import asyncpg
import httpx
import pytest
import redis.asyncio as redis

import brain.graph as graph_mod
import brain.holds as holds_mod
from brain.holds import consume_hold, record_hold
from brain.world import DuplicateReply, StaleWrite
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


@pytest.mark.asyncio
async def test_dup_rejection_retry_after_room_moved_takes_hold_path(
    pool: asyncpg.Pool, redis_client: redis.Redis
) -> None:
    """A room that moves while the brain re-decides after a dup rejection
    must SHOW the interfering row before the retry can commit: the
    re-decided reply goes through the HOLD path (the fresh row lands in
    the prompt) — never a silent commit over unseen state."""
    room_id, human_id, agent_ids = await _room(pool, agents=2)
    iris_id, marcus_id = agent_ids
    await db.insert_message(pool, room_id, human_id, "pick a number")
    # seq 2: the peer "3" the agent will parrot.
    await db.insert_message(pool, room_id, marcus_id, "3")

    async def after_dup(_messages: list) -> object:
        # While the brain re-decides, the human races to say EXACTLY what
        # the agent was about to say (seq 3): the reworded retry must be
        # re-judged against that row, not committed blind.
        texts = [str(getattr(m, "content", "")) for m in _messages]
        if any("verbatim-duplicates" in t for t in texts):
            await db.insert_message(pool, room_id, human_id, "4 — next after 3")
        return tool_call("reply", {"body": "4 — next after 3"})

    small = ScriptedChatModel(
        [triage_message(actionable=True, reason="each", response_mode="each")]
    )
    big = ScriptedChatModel(
        [
            tool_call("reply", {"body": "3"}),
            after_dup,
            tool_call("reply", {"body": "Ada took 4 — I'll take 5"}),
        ]
    )
    brain = Brain(DirectWorld(pool, redis_client), small_model=small, big_model=big)
    result = await brain.run(iris_id, room_id)

    # Stale/HOLD path: the interfering row was SHOWN (one HOLD), then the
    # revised reply committed — the re-decide prompt carries the shown row.
    assert result.outcome == "replied"
    assert result.hold_count == 1
    redecide_prompt = " ".join(
        str(getattr(m, "content", "")) for m in big.calls[2]
    )
    assert "New messages landed while you were composing" in redecide_prompt
    assert "[seq=3" in redecide_prompt
    stored = [m.body for m in await db.list_messages(pool, room_id)]
    assert stored[-1] == "Ada took 4 — I'll take 5"
    # Cursor advanced to the committed row's seq (3 human + 4 agent).
    assert await db.get_last_read(pool, iris_id, room_id) == 4


# ── hold tokens ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hold_token_roundtrip(redis_client: redis.Redis) -> None:
    room_id, agent_id = uuid4(), uuid4()
    assert await consume_hold(redis_client, agent_id, room_id) is None
    await record_hold(redis_client, agent_id, room_id, 7)
    assert await consume_hold(redis_client, agent_id, room_id) == 7
    # Atomic consume: the second read gets nothing.
    assert await consume_hold(redis_client, agent_id, room_id) is None


@pytest.mark.asyncio
async def test_consume_hold_fail_closed_when_redis_down(
    redis_client: redis.Redis, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Redis outage → the ack is REFUSED (None), never honored: an
    unverifiable token must not upgrade the flag into a free pass. The
    gate keeps running, so the cost is one extra HOLD."""
    room_id, agent_id = uuid4(), uuid4()
    await record_hold(redis_client, agent_id, room_id, 5)

    class BrokenClient:
        def __getattr__(self, name: str):
            raise redis.ConnectionError("redis down")

        async def eval(self, *args: object, **kwargs: object) -> object:
            raise redis.ConnectionError("redis down")

    assert await consume_hold(BrokenClient(), agent_id, room_id) is None


@pytest.mark.asyncio
async def test_send_anyway_without_token_does_not_bypass(
    pool: asyncpg.Pool, redis_client: redis.Redis, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-emptive send_anyway is ignored: the gate still HOLDs, and the
    preemptive flag never touches the token store (no consume, no skip)."""
    room_id, human_id, agent_ids = await _room(pool)
    iris_id, marcus_id = agent_ids
    await db.insert_message(pool, room_id, human_id, "next number after 2")

    consumed: list[int | None] = []
    _real_consume = consume_hold

    async def spy_consume(client: object, agent_id: UUID, room_id: UUID) -> int | None:
        result = await _real_consume(client, agent_id, room_id)
        consumed.append(result)
        return result

    monkeypatch.setattr("brain.graph.consume_hold", spy_consume)

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
    # The preemptive flag consumed NOTHING: the void path found no token
    # (None — nothing to spend, ack refused) and armed no authority.
    assert consumed == [None]


@pytest.mark.asyncio
async def test_send_anyway_with_token_bypasses_hold(
    pool: asyncpg.Pool, redis_client: redis.Redis, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HELD once → token armed; the retry with send_anyway ships without
    a second HOLD — the legitimate flow, and one that only works while
    the room is exactly at the acknowledged state (seq binding)."""
    room_id, human_id, agent_ids = await _room(pool)
    iris_id, marcus_id = agent_ids
    await db.insert_message(pool, room_id, human_id, "next number after 2")

    consumed: list[int | None] = []
    _real_consume = consume_hold

    async def spy_consume(client: object, agent_id: UUID, room_id: UUID) -> int | None:
        result = await _real_consume(client, agent_id, room_id)
        consumed.append(result)
        return result

    monkeypatch.setattr("brain.graph.consume_hold", spy_consume)

    async def peer_first(_messages: list) -> object:
        await db.insert_message(pool, room_id, marcus_id, "3")
        return tool_call("reply", {"body": "3"})

    async def retry_with_flag(_messages: list) -> object:
        # The HELD envelope armed a token bound to seq 2 (marcus's "3"),
        # and no new message has landed since — the ack covers exactly the
        # shown state. (The body differs from the peer's, so the dup gate
        # stays clear.) Post-commit the graph re-enters tool_loop; the
        # third scripted response keeps that hop silent.
        return tool_call("reply", {"body": "3, going with 4", "send_anyway": True})

    small = ScriptedChatModel(
        [triage_message(actionable=True, reason="each", response_mode="each")]
    )
    # 1) peer_first (reply without flag) → HOLD; 2) retry_with_flag → token
    # acked, ships; 3) post-commit re-decide (loop continues after commit)
    # → silent.
    big = ScriptedChatModel(
        [peer_first, retry_with_flag, text_message("peer raced ahead; silence")]
    )
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
    # The ack was REAL: the flagged reply spent the token the HOLD armed.
    # (The mutation "flag never reaches the gate" kills this — consume
    # never fires — and the mutation "token never armed" kills the
    # roundtrip assertion upstream.)
    assert consumed == [2]


@pytest.mark.asyncio
async def test_send_anyway_with_stale_token_re_holds(
    pool: asyncpg.Pool, redis_client: redis.Redis, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cumora §5d seq binding: the room moved past the acknowledged state,
    so the ack is void — the gate runs a fresh HOLD (showing the truly-new
    rows and re-arming the token) instead of letting the flag skip them."""
    room_id, human_id, agent_ids = await _room(pool)
    iris_id, marcus_id = agent_ids
    await db.insert_message(pool, room_id, human_id, "next number after 2")

    consumed: list[int | None] = []
    _real_consume = consume_hold

    async def spy_consume(client: object, agent_id: UUID, room_id: UUID) -> int | None:
        result = await _real_consume(client, agent_id, room_id)
        consumed.append(result)
        return result

    monkeypatch.setattr("brain.graph.consume_hold", spy_consume)

    async def peer_first(_messages: list) -> object:
        await db.insert_message(pool, room_id, marcus_id, "3")
        return tool_call("reply", {"body": "3"})

    async def retry_with_flag(_messages: list) -> object:
        # New message AFTER the HOLD armed the token at seq 2: latest is
        # now 3, token acks 2 < 3 → void. A fresh HOLD must follow.
        await db.insert_message(pool, room_id, marcus_id, "wait, 4 is mine")
        return tool_call("reply", {"body": "3, going with 4", "send_anyway": True})

    async def after_rehold(_messages: list) -> object:
        # Second HELD showed seq 3 and re-armed the token at 3; the room
        # has not moved since, so this ack is honored.
        return tool_call("reply", {"body": "4 then", "send_anyway": True})

    small = ScriptedChatModel(
        [triage_message(actionable=True, reason="each", response_mode="each")]
    )
    # 1) peer_first → HOLD#1 (token@2); 2) flagged retry → void, HOLD#2
    # (token@3, shows the new row); 3) flagged again → acked, ships;
    # 4) post-commit re-decide → silent.
    big = ScriptedChatModel(
        [
            peer_first,
            retry_with_flag,
            after_rehold,
            text_message("peer raced ahead; silence"),
        ]
    )
    brain = Brain(
        DirectWorld(pool, redis_client),
        small_model=small,
        big_model=big,
        hold_redis=redis_client,
    )

    result = await brain.run(iris_id, room_id)

    assert result.outcome == "replied"
    assert result.reply_body == "4 then"
    assert result.hold_count == 2
    stored = await db.list_messages(pool, room_id)
    assert [(m.author_id, m.body) for m in stored][-1] == (iris_id, "4 then")
    # The flagged retry SPENT the stale token@2 before the fresh HOLD
    # re-armed at 3 — the ack was consumed, not silently ignored, and the
    # second flagged reply spent the re-armed token@3.
    assert consumed == [2, 3]


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


@pytest.mark.asyncio
async def test_turn_end_clears_token_when_turn_ends_without_commit(
    pool: asyncpg.Pool, redis_client: redis.Redis
) -> None:
    """S1: a token must not outlive the turn that earned it. HOLD arms
    the token, then the turn ends WITHOUT a commit (the model declines
    to re-reply) — a future turn's preemptive send_anyway must find
    nothing to spend."""
    room_id, human_id, agent_ids = await _room(pool)
    iris_id, marcus_id = agent_ids
    await db.insert_message(pool, room_id, human_id, "next number after 2")

    async def peer_first(_messages: list) -> object:
        await db.insert_message(pool, room_id, marcus_id, "3")
        return tool_call("reply", {"body": "3"})

    small = ScriptedChatModel(
        [triage_message(actionable=True, reason="each", response_mode="each")]
    )
    # Hop 1 triggers the HOLD; hop 2 declines (plain text, no reply) —
    # the turn ends skipped, never reaching commit.
    big = ScriptedChatModel([peer_first, text_message("no number from me")])
    brain = Brain(
        DirectWorld(pool, redis_client),
        small_model=small,
        big_model=big,
        hold_redis=redis_client,
    )

    result = await brain.run(iris_id, room_id)

    assert result.outcome == "skipped"
    assert result.hold_count == 1
    assert await consume_hold(redis_client, iris_id, room_id) is None


@pytest.mark.asyncio
async def test_turn_crash_mid_graph_still_reaps_hold_token(
    pool: asyncpg.Pool,
    redis_client: redis.Redis,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An in-graph crash (GraphRecursionError, a world transport error
    outside the LLM retry path) must not leave a HOLD-minted token
    armed: the next turn's send_anyway could spend an acknowledgement
    earned in a different turn. run() reaps the token before re-raising."""
    room_id, human_id, agent_ids = await _room(pool)
    iris_id, marcus_id = agent_ids
    await db.insert_message(pool, room_id, human_id, "next number after 2")

    async def peer_first(_messages: list) -> object:
        await db.insert_message(pool, room_id, marcus_id, "3")
        return tool_call("reply", {"body": "3"})

    small = ScriptedChatModel(
        [triage_message(actionable=True, reason="each", response_mode="each")]
    )
    big = ScriptedChatModel([peer_first])
    brain = Brain(
        DirectWorld(pool, redis_client),
        small_model=small,
        big_model=big,
        hold_redis=redis_client,
    )

    # LLM failures are folded into llm_error by invoke_model, so the crash
    # must come from outside that path: blow up right after the HOLD mints
    # its token — the token exists, the turn never ends normally.
    real_record_hold = holds_mod.record_hold

    async def mint_then_explode(*args: object, **kwargs: object) -> None:
        await real_record_hold(*args, **kwargs)  # type: ignore[arg-type]
        raise RuntimeError("transport exploded right after minting")

    # graph.py imports record_hold by name; patch both sides.
    monkeypatch.setattr(graph_mod, "record_hold", mint_then_explode)

    with pytest.raises(RuntimeError, match="transport exploded"):
        await brain.run(iris_id, room_id)

    # The token minted by the crashed turn is reaped.
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
async def test_loop_cap_counts_across_turns_not_per_inbox(
    pool: asyncpg.Pool, redis_client: redis.Redis
) -> None:
    """The stretch is ROOM-level: an earlier turn's replies accumulate in
    the room count even though each wake's inbox holds a single message.
    Per-inbox counting (the pre-room-level bug) would keep waking the
    model gate here forever; the room cursor trips the cap instead."""
    room_id, _human_id, agent_ids = await _room(pool, agents=2)
    iris_id, marcus_id = agent_ids
    # Seed 7 agent-only messages (cap is 8 for this room): turn 1 sits
    # one below the cap, so the model gate still runs.
    for i in range(6):
        await db.insert_message(pool, room_id, marcus_id, f"agent chatter {i}")
    await db.insert_message(pool, room_id, iris_id, "agent chatter 6")

    small = ScriptedChatModel(
        [triage_message(actionable=False, reason="calm", response_mode="me")]
    )
    big = ScriptedChatModel()
    brain = Brain(DirectWorld(pool, redis_client), small_model=small, big_model=big)
    result = await brain.run(iris_id, room_id)
    assert result.outcome == "skipped"
    assert len(small.calls) == 1
    assert "loop cap" not in (result.triage_reason or "")

    # One more agent reply lands. The new wake's inbox holds exactly ONE
    # message — under any per-inbox reading — but the ROOM has now run 8
    # agent messages with no human: the cap fires without the model.
    await db.insert_message(pool, room_id, marcus_id, "agent chatter 7")
    result = await brain.run(iris_id, room_id)
    assert result.outcome == "skipped"
    assert len(small.calls) == 1
    assert "loop cap" in (result.triage_reason or "")
    assert big.calls == []


@pytest.mark.asyncio
async def test_burst_inbox_under_room_cap_still_runs_model_gate(
    pool: asyncpg.Pool, redis_client: redis.Redis
) -> None:
    """A coalesced burst can push the per-turn inbox far past the cap on
    a room whose agent-only stretch is still young (the human just
    spoke, or a short run). The room cursor — not the batch size —
    decides, so the model gate runs instead of a silent skip."""
    room_id, human_id, agent_ids = await _room(pool, agents=2)
    iris_id, marcus_id = agent_ids
    await db.insert_message(pool, room_id, human_id, "plans, quickly")
    for i in range(6):
        await db.insert_message(pool, room_id, marcus_id, f"burst {i}")

    small = ScriptedChatModel(
        [triage_message(actionable=False, reason="calm", response_mode="me")]
    )
    big = ScriptedChatModel()
    brain = Brain(DirectWorld(pool, redis_client), small_model=small, big_model=big)

    result = await brain.run(iris_id, room_id)

    assert result.outcome == "skipped"
    assert len(small.calls) == 1
    assert "loop cap" not in (result.triage_reason or "")


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


@pytest.mark.asyncio
async def test_loop_cap_arms_in_human_free_room(
    pool: asyncpg.Pool, redis_client: redis.Redis
) -> None:
    """No human participant at all: the cap still arms (this is the MOST
    loop-prone shape — nobody will ever reset the counter)."""
    room = await db.create_room(pool, "no-human-room")
    iris = await db.add_participant(pool, room.id, "agent", "Iris", None)
    marcus = await db.add_participant(pool, room.id, "agent", "Marcus", None)
    for i in range(4):
        await db.insert_message(pool, room.id, str(marcus.id), f"agent chatter {i}")
    for i in range(4):
        await db.insert_message(pool, room.id, str(iris.id), f"agent chatter {i + 4}")

    small = ScriptedChatModel()
    big = ScriptedChatModel()
    brain = Brain(DirectWorld(pool, redis_client), small_model=small, big_model=big)

    result = await brain.run(iris.id, room.id)

    assert result.outcome == "skipped"
    assert small.calls == []
    assert "loop cap" in (result.triage_reason or "")


@pytest.mark.asyncio
async def test_second_human_message_resets_loop_cap(
    pool: asyncpg.Pool, redis_client: redis.Redis
) -> None:
    """Every human resets the counter, not just the first one found."""
    room = await db.create_room(pool, "two-humans")
    ada = await db.add_participant(pool, room.id, "human", "Ada", None)
    bob = await db.add_participant(pool, room.id, "human", "Bob", None)
    iris = await db.add_participant(pool, room.id, "agent", "Iris", None)
    marcus = await db.add_participant(pool, room.id, "agent", "Marcus", None)
    for i in range(4):
        await db.insert_message(pool, room.id, marcus.id, f"chatter {i}")
    # Bob (the SECOND human) speaks: counter resets even though the
    # first-human-only heuristic would have kept counting.
    await db.insert_message(pool, room.id, str(bob.id), "keep going")
    for i in range(3):
        await db.insert_message(pool, room.id, marcus.id, f"more {i}")

    small = ScriptedChatModel(
        [triage_message(actionable=False, reason="calm", response_mode="me")]
    )
    big = ScriptedChatModel()
    brain = Brain(DirectWorld(pool, redis_client), small_model=small, big_model=big)

    result = await brain.run(iris.id, room.id)

    assert result.outcome == "skipped"
    assert len(small.calls) == 1
    assert "loop cap" not in (result.triage_reason or "")

    # And with Ada silent the whole time, her id never gates anything.
    assert str(ada) != str(bob)


# ── digest ───────────────────────────────────────────────────────────────


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
    assert "`t1:review` — held by **Jules** (" in body
    assert "| triage | m-small | 1 | 10 | 2 |" in body
    assert "**110**" in body


@pytest.mark.asyncio
async def test_digest_escapes_pipe_and_newline_in_names(app_client: tuple) -> None:
    """A name containing a pipe or newline must not corrupt the table."""
    app, client = app_client
    room = (await client.post("/rooms", json={"name": "pipe | room"})).json()
    agent = (
        await client.post(
            f"/rooms/{room['id']}/participants",
            json={"kind": "agent", "name": "a | b"},
        )
    ).json()
    await client.post(
        f"/rooms/{room['id']}/messages",
        json={"author_id": agent["id"], "body": "hello"},
    )

    resp = await client.get(f"/rooms/{room['id']}/digest")
    assert resp.status_code == 200
    body = resp.text
    # Exactly one table row for the message, with the pipe escaped.
    assert "| 1 | a \\| b | hello |" in body
    # The H1 title is not inside a table; the pipe stays literal there.
    assert "# pipe | room" in body


@pytest.mark.asyncio
async def test_digest_marks_stale_claim_as_crash_orphan(app_client: tuple) -> None:
    """A claim past the steal TTL is a crash orphan, not a live
    obligation: the digest labels it instead of presenting a dead lock as
    'held by X'."""
    app, client = app_client
    room = (await client.post("/rooms", json={"name": "stale-claim"})).json()
    agent = (
        await client.post(
            f"/rooms/{room['id']}/participants",
            json={"kind": "agent", "name": "Jules"},
        )
    ).json()
    await db.try_claim(app.state.pool, UUID(room["id"]), "t1:orphan", UUID(agent["id"]))
    # Age the claim past the steal TTL, as if the holder crashed right
    # after claiming.
    await app.state.pool.execute(
        "UPDATE claims SET created_at = now() - interval '400 seconds'"
    )

    resp = await client.get(f"/rooms/{room['id']}/digest")
    assert resp.status_code == 200
    body = resp.text
    assert "`t1:orphan`" in body
    assert "stale" in body
    assert "stealable" in body


@pytest.mark.asyncio
async def test_digest_404_for_missing_room(app_client: tuple) -> None:
    _app, client = app_client
    import uuid as uuid_mod

    resp = await client.get(f"/rooms/{uuid_mod.uuid4()}/digest")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_digest_moderated_renders_decisions_in_seq_order(
    app_client: tuple,
) -> None:
    """Moderated rooms insert a 决策 table between transcript and claims.

    Rows follow trigger_seq, not insert order. Target names reuse the
    same table-cell escape as transcript bodies.
    """
    app, client = app_client
    room = (
        await client.post("/rooms", json={"name": "roundtable", "mode": "moderated"})
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
                "persona": "keeps time",
                "role": "moderator",
            },
        )
    ).json()
    member = (
        await client.post(
            f"/rooms/{room['id']}/participants",
            json={"kind": "agent", "name": "Iris | Lee"},
        )
    ).json()
    for body in ("q1", "q2", "q3"):
        posted = await client.post(
            f"/rooms/{room['id']}/messages",
            json={"author_id": human["id"], "body": body},
        )
        posted.raise_for_status()

    room_id = UUID(room["id"])
    chair_id = UUID(chair["id"])
    member_id = UUID(member["id"])
    # Insert out of trigger order so the digest, not the writer, sorts.
    await db.record_decision(app.state.pool, room_id, chair_id, 3, "silence", None)
    await db.record_decision(app.state.pool, room_id, chair_id, 1, "call_on", member_id)
    await db.record_decision(app.state.pool, room_id, chair_id, 2, "say", None)

    resp = await client.get(f"/rooms/{room['id']}/digest")
    assert resp.status_code == 200
    body = resp.text
    assert "- Mode: `moderated`" in body
    assert "## 决策 — Chair" in body
    assert "| trigger_seq | action | target | created_at |" in body
    call_on = body.index("| 1 | call_on | Iris \\| Lee |")
    say = body.index("| 2 | say | — |")
    silence = body.index("| 3 | silence | — |")
    assert call_on < say < silence
    transcript_at = body.index("## Transcript")
    decisions_at = body.index("## 决策")
    claims_at = body.index("## Action items (claims)")
    assert transcript_at < decisions_at < claims_at


@pytest.mark.asyncio
async def test_digest_flattens_newline_in_moderator_name(
    app_client: tuple,
) -> None:
    """A moderator name with a newline must not forge digest headings.

    `_esc` flattens newlines so both the 决策 heading and the pre-existing
    `- Agents:` line stay on one line.
    """
    _app, client = app_client
    room = (
        await client.post("/rooms", json={"name": "inject", "mode": "moderated"})
    ).json()
    await client.post(
        f"/rooms/{room['id']}/participants",
        json={
            "kind": "agent",
            "name": "Chair\n## forged heading",
            "role": "moderator",
        },
    )

    resp = await client.get(f"/rooms/{room['id']}/digest")
    assert resp.status_code == 200
    body = resp.text
    assert "## 决策 — Chair ## forged heading" in body
    assert "- Agents: **Chair ## forged heading**" in body
    assert "\n## forged heading" not in body
    assert body.count("## Action items (claims)") == 1


@pytest.mark.asyncio
async def test_digest_moderated_empty_decisions_is_placeholder(
    app_client: tuple,
) -> None:
    """A moderated room with no rows still shows 决策, as a placeholder
    rather than an empty table."""
    _app, client = app_client
    room = (
        await client.post("/rooms", json={"name": "empty-decisions", "mode": "moderated"})
    ).json()
    await client.post(
        f"/rooms/{room['id']}/participants",
        json={"kind": "agent", "name": "Chair", "role": "moderator"},
    )

    resp = await client.get(f"/rooms/{room['id']}/digest")
    assert resp.status_code == 200
    body = resp.text
    assert "## 决策 — Chair" in body
    assert "_(no decisions)_" in body
    assert "| trigger_seq | action | target | created_at |" not in body


@pytest.mark.asyncio
async def test_digest_open_room_omits_decisions_section(
    app_client: tuple,
) -> None:
    """Open rooms must not grow an empty 决策 table — the section is
    omitted byte-wise, not rendered blank."""
    _app, client = app_client
    room = (await client.post("/rooms", json={"name": "plain"})).json()
    human = (
        await client.post(
            f"/rooms/{room['id']}/participants",
            json={"kind": "human", "name": "Ada"},
        )
    ).json()
    await client.post(
        f"/rooms/{room['id']}/messages",
        json={"author_id": human["id"], "body": "hello"},
    )

    resp = await client.get(f"/rooms/{room['id']}/digest")
    assert resp.status_code == 200
    body = resp.text
    assert "- Mode: `open`" in body
    assert "决策" not in body
    assert "trigger_seq" not in body


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
    # The dup gate's peer is the latest OTHER message. After the agent's
    # "3", the human echoes "3" (legitimate — humans bypass the gate),
    # which becomes the new peer; the agent re-posting "3" now hits it.
    echo = await client.post(
        f"/rooms/{room['id']}/messages",
        json={"author_id": human["id"], "body": "3"},
    )
    assert echo.status_code == 200
    second = await post_reply("3", 3)
    assert second.status_code == 409
    assert second.json()["detail"]["error"] == "duplicate_reply"
    assert second.json()["detail"]["peer_seq"] == 3

    from daemon.world_http import HttpWorld

    world = HttpWorld(client, computer["token"])
    world.bind_actor(agent_id)
    with pytest.raises(DuplicateReply):
        await world.insert_message(UUID(room["id"]), agent_id, "3", not_after_seq=3)


@pytest.mark.asyncio
async def test_http_world_survives_foreign_409_shape(app_client: tuple) -> None:
    """A 409 rewritten by a proxy/gateway (plain text, no detail dict)
    must surface as StaleWrite with empty details, not crash parsing."""
    from daemon.world_http import HttpWorld

    _app, client = app_client

    class RewritingTransport(httpx.AsyncBaseTransport):
        """Wraps ASGI and flattens any 409 body to plain text."""

        def __init__(self, app: object) -> None:
            self._inner = httpx.ASGITransport(app=app)  # type: ignore[arg-type]

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            resp = await self._inner.handle_async_request(request)
            if resp.status_code == 409:
                return httpx.Response(409, text="gateway says conflict")
            return resp

    app, _client = app_client
    computer = (await client.post("/computers", json={"name": "laptop"})).json()
    room = (await client.post("/rooms", json={"name": "proxy-room"})).json()
    human = (
        await client.post(
            f"/rooms/{room['id']}/participants",
            json={"kind": "human", "name": "Ada"},
        )
    ).json()
    await client.post(
        f"/rooms/{room['id']}/messages",
        json={"author_id": human["id"], "body": "pick a number"},
    )
    agent = (
        await client.post(
            f"/rooms/{room['id']}/participants",
            json={
                "kind": "agent",
                "name": "Jules",
                "computer_id": computer["id"],
            },
        )
    ).json()

    world = HttpWorld(
        httpx.AsyncClient(transport=RewritingTransport(app), base_url="http://test"),
        computer["token"],
    )
    world.bind_actor(UUID(agent["id"]))
    # last_seq 0: unknown, but the cursor can never regress below what
    # the turn has already seen — the graph holds on max(seen, latest).
    with pytest.raises(StaleWrite) as exc:
        await world.insert_message(
            UUID(room["id"]), UUID(agent["id"]), "3", not_after_seq=0
        )
    assert exc.value.last_seq == 0
    assert exc.value.newer == []
