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
        with httpx.Client(timeout=3.0) as client:
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
    client: httpx.AsyncClient, room_id: str, name: str, persona: str
) -> dict[str, Any]:
    resp = await client.post(
        f"/rooms/{room_id}/participants",
        json={"kind": "agent", "name": name, "persona": persona},
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
