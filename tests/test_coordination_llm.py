"""Real-relay coordination tests. Invariants, not wording.

Skipped when OPENAI_BASE_URL is unset or the relay does not answer /v1/models.
"""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI

from server.main import create_app

_INT = re.compile(r"-?\d+")

COUNTING_PERSONA = (
    "You are playing a counting game with the other agents. "
    "This is not a one-of-us task: do not claim; just reply. "
    "Each message you post must be exactly one Arabic numeral and nothing else "
    "(example: 3). No words, no other digits. "
    "Only numerals posted by agents count as moves. "
    "Digits that appear in the human's rules (start at 1, stop at 6) are not moves. "
    "If no agent has posted a numeral yet, post 1. "
    "Otherwise continue from the latest agent numeral. "
    "Never repeat a number. Never skip. Stop after an agent has posted 6."
)

INTRO_PERSONA = (
    "You share this room with other agents. Answer in the room's language. "
    "If exactly one of you should speak, claim first, then reply only if you won. "
    "If anyone has already introduced the room, stay silent."
)


def _require_relay() -> str:
    base = (os.environ.get("OPENAI_BASE_URL") or "").rstrip("/")
    if not base:
        pytest.skip(
            "OPENAI_BASE_URL unset; real-relay coordination tests skipped"
        )
    url = f"{base}/models"
    key = os.environ.get("OPENAI_API_KEY") or "relay-no-key"
    os.environ.setdefault("OPENAI_API_KEY", key)
    os.environ.setdefault("OPENAI_API_BASE", base)
    try:
        with httpx.Client(timeout=10.0) as client:
            client.get(
                url,
                headers={"Authorization": f"Bearer {key}"},
            )
    except httpx.RequestError as exc:
        pytest.skip(f"OpenAI-compatible relay unreachable at {url}: {exc}")
    return base


@pytest.fixture
async def live(
    require_services: None,
) -> AsyncIterator[tuple[FastAPI, httpx.AsyncClient]]:
    _require_relay()
    app = create_app()
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://agora") as client:
            yield app, client


async def _add_agent(
    client: httpx.AsyncClient,
    room_id: str,
    name: str,
    persona: str,
    *,
    role: str = "member",
) -> dict[str, Any]:
    resp = await client.post(
        f"/rooms/{room_id}/participants",
        json={"kind": "agent", "name": name, "persona": persona, "role": role},
    )
    resp.raise_for_status()
    return resp.json()


def _agent_ints(messages: list[dict[str, Any]], agent_ids: set[str]) -> list[int]:
    out: list[int] = []
    for message in messages:
        if message["author_id"] not in agent_ids:
            continue
        out.extend(int(m) for m in _INT.findall(message["body"]))
    return out


def _count_done(nums: list[int]) -> bool:
    if not nums:
        return False
    if nums != list(range(nums[0], nums[0] + len(nums))):
        return False
    return 6 in nums or set(nums) >= {1, 2, 3, 4, 5, 6} or len(nums) >= 6


async def _messages(client: httpx.AsyncClient, room_id: str) -> list[dict[str, Any]]:
    resp = await client.get(f"/rooms/{room_id}/messages")
    resp.raise_for_status()
    return resp.json()["messages"]


async def _wait_quiescent(
    app: FastAPI,
    client: httpx.AsyncClient,
    room_id: str,
    *,
    settle: float = 3.0,
    timeout: float = 90.0,
) -> list[dict[str, Any]]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        await app.state.scheduler.wait_idle()
        before = await _messages(client, room_id)
        await asyncio.sleep(settle)
        await app.state.scheduler.wait_idle()
        after = await _messages(client, room_id)
        if len(after) == len(before) and after == before:
            return after
    raise TimeoutError(
        f"room {room_id} did not go quiet after {timeout}s; "
        f"last={[m['body'] for m in after]}"
    )


@pytest.mark.llm
@pytest.mark.asyncio
async def test_counting_game_no_dup_no_gap(
    live: tuple[FastAPI, httpx.AsyncClient],
) -> None:
    app, client = live
    room = (await client.post("/rooms", json={"name": "count-game"})).json()
    room_id = room["id"]
    human = (
        await client.post(
            f"/rooms/{room_id}/participants",
            json={"kind": "human", "name": "Ada"},
        )
    ).json()
    agents = [
        await _add_agent(client, room_id, name, COUNTING_PERSONA)
        for name in ("Iris", "Marcus", "Jules")
    ]
    agent_ids = {a["id"] for a in agents}

    posted = await client.post(
        f"/rooms/{room_id}/messages",
        json={
            "author_id": human["id"],
            "body": "我们玩报数，从1开始，每人每条消息只报一个数，报到6为止。谁先来？",
        },
    )
    posted.raise_for_status()

    loop = asyncio.get_running_loop()
    deadline = loop.time() + 120.0
    listed: list[dict[str, Any]] = []
    nums: list[int] = []
    while loop.time() < deadline:
        listed = await _messages(client, room_id)
        nums = _agent_ints(listed, agent_ids)
        if _count_done(nums):
            break
        await asyncio.sleep(0.4)

    # Catch a late collision that lands after we first saw a complete run.
    await app.state.scheduler.wait_idle()
    await asyncio.sleep(2.0)
    await app.state.scheduler.wait_idle()
    listed = await _messages(client, room_id)
    nums = _agent_ints(listed, agent_ids)

    transcript = [(m["seq"], m["author_id"][:8], m["body"]) for m in listed]
    assert len(nums) >= 3, (
        f"fewer than 3 numbers landed before timeout: {nums!r} transcript={transcript!r}"
    )
    assert len(nums) == len(set(nums)), f"duplicate integers: {nums} transcript={transcript!r}"
    assert nums == list(range(nums[0], nums[0] + len(nums))), (
        f"sequence is not strictly increasing/gapless: {nums} transcript={transcript!r}"
    )


@pytest.mark.llm
@pytest.mark.asyncio
async def test_one_of_us_exactly_one_agent_reply(
    live: tuple[FastAPI, httpx.AsyncClient],
) -> None:
    app, client = live
    room = (await client.post("/rooms", json={"name": "one-of-us"})).json()
    room_id = room["id"]
    human = (
        await client.post(
            f"/rooms/{room_id}/participants",
            json={"kind": "human", "name": "Ada"},
        )
    ).json()
    agents = [
        await _add_agent(client, room_id, name, INTRO_PERSONA)
        for name in ("Iris", "Marcus", "Jules")
    ]
    agent_ids = {a["id"] for a in agents}

    posted = await client.post(
        f"/rooms/{room_id}/messages",
        json={
            "author_id": human["id"],
            "body": "请你们中恰好一个人用一句话介绍这个房间。",
        },
    )
    posted.raise_for_status()

    listed = await _wait_quiescent(app, client, room_id)
    agent_msgs = [m for m in listed if m["author_id"] in agent_ids]
    transcript = [(m["seq"], m["author_id"][:8], m["body"]) for m in listed]
    assert len(agent_msgs) == 1, (
        f"expected exactly one agent message, got {len(agent_msgs)}: {transcript!r}"
    )

    rows = await app.state.pool.fetch(
        "SELECT task_key FROM claims WHERE room_id = $1",
        UUID(room_id),
    )
    keys = [r["task_key"] for r in rows]
    assert any(k.startswith("t1") for k in keys), (
        f"no claim key starts with t1: {keys!r} transcript={transcript!r}"
    )


CHAIR_PERSONA = (
    "You are a terse Chinese-speaking moderator. "
    "Use the decide tool: call_on the one member whose specialty matches "
    "the question. Do not answer the substance yourself. "
    "say only for procedure; silence if nothing is needed."
)

IRIS_BACKEND = (
    "You are a backend engineer. Answer only implementation, database, "
    "and API questions, in Chinese. Stay silent on product or design."
)

MARCUS_PRODUCT = (
    "You are a product manager. Answer only product, priority, and user "
    "questions, in Chinese. Stay silent on backend or design."
)

JULES_DESIGN = (
    "You are a designer. Answer only interaction and visual-design "
    "questions, in Chinese. Stay silent on backend or product."
)


@pytest.mark.llm
@pytest.mark.asyncio
async def test_moderated_one_call_one_answer(
    live: tuple[FastAPI, httpx.AsyncClient],
) -> None:
    """Moderated room: every member message corresponds to a call_on.

    The correspondence is a code invariant (routing). The count is
    model behavior: the Chair may legitimately call on a second member,
    so we require at least one member message, not exactly one.
    """
    app, client = live
    room = (
        await client.post("/rooms", json={"name": "moderated-one", "mode": "moderated"})
    ).json()
    room_id = room["id"]
    human = (
        await client.post(
            f"/rooms/{room_id}/participants",
            json={"kind": "human", "name": "Ada"},
        )
    ).json()
    await _add_agent(client, room_id, "Chair", CHAIR_PERSONA, role="moderator")
    members = [
        await _add_agent(client, room_id, name, persona)
        for name, persona in (
            ("Iris", IRIS_BACKEND),
            ("Marcus", MARCUS_PRODUCT),
            ("Jules", JULES_DESIGN),
        )
    ]
    member_ids = {a["id"] for a in members}

    posted = await client.post(
        f"/rooms/{room_id}/messages",
        json={
            "author_id": human["id"],
            "body": (
                "PostgreSQL 的房间序号为什么用行上的计数器而不是 SEQUENCE？"
                "请恰好一个人回答。"
            ),
        },
    )
    posted.raise_for_status()

    listed = await _wait_quiescent(app, client, room_id)
    member_msgs = [m for m in listed if m["author_id"] in member_ids]
    transcript = [(m["seq"], m["author_id"][:8], m["body"]) for m in listed]
    # Mechanism: >= 1 member post. Count beyond that is the Chair's
    # choice, not a kernel invariant.
    assert len(member_msgs) >= 1, (
        f"expected at least one member message, got {len(member_msgs)}: "
        f"{transcript!r}"
    )

    rows = await app.state.pool.fetch(
        """
        SELECT action, target_id FROM moderator_decisions
        WHERE room_id = $1
        """,
        UUID(room_id),
    )
    call_ons = [r for r in rows if r["action"] == "call_on"]
    assert call_ons, (
        f"expected at least one call_on row, got {[(r['action'], r['target_id']) for r in rows]!r} "
        f"transcript={transcript!r}"
    )
    called = {str(r["target_id"]) for r in call_ons if r["target_id"] is not None}
    for message in member_msgs:
        assert message["author_id"] in called, (
            f"member {message['author_id'][:8]} posted without a matching "
            f"call_on target; called={called!r} transcript={transcript!r}"
        )


@pytest.mark.llm
@pytest.mark.asyncio
async def test_moderated_mention_bypasses_moderator(
    live: tuple[FastAPI, httpx.AsyncClient],
) -> None:
    """@Name in a moderated room wakes that member directly. The named
    member replies; no other member does. The mention path writes no
    decision for that trigger_seq — if the chair was woken and call_on'd
    Marcus, routing is dead and this must fail."""
    app, client = live
    room = (
        await client.post(
            "/rooms", json={"name": "moderated-mention", "mode": "moderated"}
        )
    ).json()
    room_id = room["id"]
    human = (
        await client.post(
            f"/rooms/{room_id}/participants",
            json={"kind": "human", "name": "Ada"},
        )
    ).json()
    await _add_agent(client, room_id, "Chair", CHAIR_PERSONA, role="moderator")
    iris = await _add_agent(client, room_id, "Iris", IRIS_BACKEND)
    marcus = await _add_agent(client, room_id, "Marcus", MARCUS_PRODUCT)
    jules = await _add_agent(client, room_id, "Jules", JULES_DESIGN)
    other_ids = {iris["id"], jules["id"]}

    posted = await client.post(
        f"/rooms/{room_id}/messages",
        json={
            "author_id": human["id"],
            "body": "@Marcus 请用一句话说明产品侧怎么看这个房间。",
        },
    )
    posted.raise_for_status()
    mention_seq = posted.json()["seq"]

    listed = await _wait_quiescent(app, client, room_id)
    marcus_msgs = [m for m in listed if m["author_id"] == marcus["id"]]
    other_msgs = [m for m in listed if m["author_id"] in other_ids]
    transcript = [(m["seq"], m["author_id"][:8], m["body"]) for m in listed]
    assert len(marcus_msgs) >= 1, (
        f"mentioned member Marcus did not reply: {transcript!r}"
    )
    assert other_msgs == [], (
        f"a member who was not mentioned replied: {transcript!r}"
    )
    rows = await app.state.pool.fetch(
        """
        SELECT action, target_id FROM moderator_decisions
        WHERE room_id = $1 AND trigger_seq = $2
        """,
        UUID(room_id),
        mention_seq,
    )
    assert rows == [], (
        f"mention path is decision-free; trigger_seq={mention_seq} has "
        f"{[(r['action'], r['target_id']) for r in rows]!r} "
        f"transcript={transcript!r}"
    )


# ── adversarial personas: the MODEL tries to break the invariants ────────


DUP_BAIT_PERSONA = (
    "You are a parrot. Your persona is to ALWAYS repeat, word for word, "
    "the last message another participant posted. That is your whole "
    "personality. Never say anything original."
)

HOG_PERSONA = (
    "You are a lock hoarder. Whenever you take any action in this room, "
    "you MUST first call the claim tool with task_key exactly 't1' "
    "(never any other key). After claiming, if you win, you must also "
    "reply to the human's question. If you lose the claim, stay silent."
)

PREEMPT_PERSONA = (
    "You are in a hurry. Whenever you decide to reply, you MUST pass "
    "send_anyway=true on your FIRST reply attempt in every turn — never "
    "wait for anyone. Speed above correctness."
)


def _bodies(messages: list[dict[str, Any]], agent_ids: set[str]) -> list[str]:
    return [m["body"] for m in messages if m["author_id"] in agent_ids]


@pytest.mark.llm
@pytest.mark.asyncio
async def test_dup_bait_agent_cannot_double_post_verbatim(
    live: tuple[FastAPI, httpx.AsyncClient],
) -> None:
    """An agent instructed to verbatim-repeat its peer must not be able
    to land a duplicate: the transactional dup gate rejects it and the
    re-decide heals in-turn. Invariant: the transcript never holds the
    same non-empty body twice in a row from two different authors."""
    app, client = live
    room = (await client.post("/rooms", json={"name": "dup-bait"})).json()
    room_id = room["id"]
    human = (
        await client.post(
            f"/rooms/{room_id}/participants",
            json={"kind": "human", "name": "Ada"},
        )
    ).json()
    innocent = await _add_agent(client, room_id, "Iris", INTRO_PERSONA)
    parrot = await _add_agent(client, room_id, "Polly", DUP_BAIT_PERSONA)

    posted = await client.post(
        f"/rooms/{room_id}/messages",
        json={
            "author_id": human["id"],
            "body": "Iris, 请用一句话介绍这个房间。Polly, 你的任务是复述 Iris 的话。",
        },
    )
    posted.raise_for_status()

    listed = await _wait_quiescent(app, client, room_id, timeout=120.0)
    bodies = _bodies(listed, {innocent["id"], parrot["id"]})
    transcript = [(m["seq"], m["author_id"][:8], m["body"]) for m in listed]

    # The invariant under attack: no two consecutive agent messages with
    # the SAME non-empty body (the verbatim-dup gate's exact shape).
    for prev, cur in zip(bodies, bodies[1:]):
        assert not (
            prev.strip() and cur.strip() == prev.strip()
        ), f"verbatim duplicate landed: {transcript!r}"

    # The gate was actually exercised: the parrot woke and spent model
    # calls (proves the room ran, not that the model politely stayed
    # quiet — the gate's own re-decide heals the rest).
    rows = await app.state.pool.fetch(
        """
        SELECT COUNT(*) AS n FROM llm_calls
        WHERE room_id = $1 AND purpose = 'triage'
        """,
        UUID(room_id),
    )
    assert rows[0]["n"] >= 2, "parrot never woke — test exercised nothing"


@pytest.mark.llm
@pytest.mark.asyncio
async def test_claim_hog_two_agents_one_lock_one_reply(
    live: tuple[FastAPI, httpx.AsyncClient],
) -> None:
    """Two agents both instructed to claim t1 and answer: exactly one
    wins the lock and exactly one reply lands. The loser must not
    double-answer, and the room must not deadlock."""
    app, client = live
    room = (await client.post("/rooms", json={"name": "claim-hog"})).json()
    room_id = room["id"]
    human = (
        await client.post(
            f"/rooms/{room_id}/participants",
            json={"kind": "human", "name": "Ada"},
        )
    ).json()
    hog_a = await _add_agent(client, room_id, "Bella", HOG_PERSONA)
    hog_b = await _add_agent(client, room_id, "Cain", HOG_PERSONA)
    hog_ids = {hog_a["id"], hog_b["id"]}

    posted = await client.post(
        f"/rooms/{room_id}/messages",
        json={
            "author_id": human["id"],
            "body": "你们两个都想回答：请先用 t1 认领，赢的人用一句话报出今天的日期意义。",
        },
    )
    posted.raise_for_status()

    listed = await _wait_quiescent(app, client, room_id, timeout=120.0)
    replies = _bodies(listed, hog_ids)
    transcript = [(m["seq"], m["author_id"][:8], m["body"]) for m in listed]

    rows = await app.state.pool.fetch(
        "SELECT task_key, claimed_by FROM claims WHERE room_id = $1",
        UUID(room_id),
    )
    keys = [r["task_key"] for r in rows]
    assert keys == ["t1"], f"expected exactly the t1 lock, got {keys!r}"
    assert len(replies) == 1, (
        f"expected exactly one hog reply, got {len(replies)}: {transcript!r}"
    )


@pytest.mark.llm
@pytest.mark.asyncio
async def test_preemptive_send_anyway_cannot_skip_freshness(
    live: tuple[FastAPI, httpx.AsyncClient],
) -> None:
    """An agent told to pass send_anyway on every first attempt must
    still have its writes gated: replies that raced a peer's row are
    refused (409) and re-decided, so the transcript stays collision-free
    for the counting task. send_anyway is an acknowledgement, never a
    pass."""
    app, client = live
    room = (await client.post("/rooms", json={"name": "preempt"})).json()
    room_id = room["id"]
    human = (
        await client.post(
            f"/rooms/{room_id}/participants",
            json={"kind": "human", "name": "Ada"},
        )
    ).json()
    agents = [
        await _add_agent(client, room_id, name, persona)
        for name, persona in (
            ("Iris", COUNTING_PERSONA),
            ("Racer", PREEMPT_PERSONA + " " + COUNTING_PERSONA),
        )
    ]
    agent_ids = {a["id"] for a in agents}

    posted = await client.post(
        f"/rooms/{room_id}/messages",
        json={
            "author_id": human["id"],
            "body": "我们玩报数，从1开始，每人每条消息只报一个数，报到4为止。越快越好。",
        },
    )
    posted.raise_for_status()

    loop = asyncio.get_running_loop()
    deadline = loop.time() + 120.0
    listed: list[dict[str, Any]] = []
    nums: list[int] = []
    while loop.time() < deadline:
        listed = await _messages(client, room_id)
        nums = _agent_ints(listed, agent_ids)
        if _count_done(nums):
            break
        await asyncio.sleep(0.4)
    await app.state.scheduler.wait_idle()
    await asyncio.sleep(2.0)
    await app.state.scheduler.wait_idle()
    listed = await _messages(client, room_id)
    nums = _agent_ints(listed, agent_ids)

    transcript = [(m["seq"], m["author_id"][:8], m["body"]) for m in listed]
    assert len(nums) >= 3, (
        f"fewer than 3 numbers landed: {nums!r} transcript={transcript!r}"
    )
    assert len(nums) == len(set(nums)), (
        f"duplicate integers with a send_anyway abuser: {nums} "
        f"transcript={transcript!r}"
    )
    assert nums == list(range(nums[0], nums[0] + len(nums))), (
        f"sequence broken under send_anyway abuse: {nums} transcript={transcript!r}"
    )
