"""Phase 7 demo: moderated room, Chair decides, @-mention bypass, decline→pass.

Needs a real OPENAI_API_KEY / OPENAI_BASE_URL (same contract as demo_phase2)
and AGORA_DATABASE_URL.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from uuid import UUID

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI  # noqa: E402

from server import db  # noqa: E402
from server.main import create_app  # noqa: E402


def _say(text: str = "") -> None:
    print(text, flush=True)


def _relay_hint() -> str:
    return (
        "Local relay example:\n"
        "  export OPENAI_API_KEY=relay-no-key\n"
        "  export OPENAI_BASE_URL=http://192.168.1.100:8317/v1\n"
        "  export OPENAI_API_BASE=$OPENAI_BASE_URL\n"
        "  export AGORA_DATABASE_URL=postgresql://agora:agora@127.0.0.1:5433/agora\n"
        "  export AGORA_REDIS_URL=redis://127.0.0.1:6379/0\n"
        "Then: uv run python scripts/demo_phase7.py\n"
        "Models default to AGORA_SMALL_MODEL=gpt-5.6-luna and "
        "AGORA_BIG_MODEL=gpt-5.6-terra (override either if you want)."
    )


def _probe_relay() -> None:
    """Same /v1/models probe as tests.test_coordination_llm._require_relay."""
    base = (os.environ.get("OPENAI_BASE_URL") or "").rstrip("/")
    if not base:
        raise SystemExit(
            "demo_phase7 needs OPENAI_BASE_URL (OpenAI-compatible relay).\n"
            + _relay_hint()
        )
    url = f"{base}/models"
    key = os.environ.get("OPENAI_API_KEY") or "relay-no-key"
    os.environ.setdefault("OPENAI_API_KEY", key)
    os.environ.setdefault("OPENAI_API_BASE", base)
    try:
        with httpx.Client(timeout=5.0) as client:
            client.get(url, headers={"Authorization": f"Bearer {key}"})
    except httpx.RequestError as exc:
        raise SystemExit(
            f"OpenAI-compatible relay unreachable at {url}: {exc}\n"
            + _relay_hint()
        ) from exc


async def _wait_brain(app: FastAPI, minimum: int, timeout: float = 90.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if len(app.state.scheduler.brain_results) >= minimum:
            await app.state.scheduler.wait_idle()
            return
        await asyncio.sleep(0.1)
    raise TimeoutError(
        f"only {len(app.state.scheduler.brain_results)} brain turns after {timeout}s, "
        f"wanted {minimum}"
    )


async def _wait_quiet(
    app: FastAPI,
    client: httpx.AsyncClient,
    room_id: str,
    *,
    settle: float = 3.0,
    timeout: float = 90.0,
) -> list[dict]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    after: list[dict] = []
    while loop.time() < deadline:
        await app.state.scheduler.wait_idle()
        before = (await client.get(f"/rooms/{room_id}/messages")).json()["messages"]
        await asyncio.sleep(settle)
        await app.state.scheduler.wait_idle()
        after = (await client.get(f"/rooms/{room_id}/messages")).json()["messages"]
        if after == before:
            return after
    raise TimeoutError(
        f"room {room_id} did not go quiet after {timeout}s"
    )


def _role_of(participant: dict) -> str:
    if participant["kind"] == "human":
        return "human"
    return participant.get("role") or "member"


async def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "demo_phase7 needs OPENAI_API_KEY and OPENAI_BASE_URL in the environment.\n"
            + _relay_hint()
        )
    _probe_relay()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    app = create_app()
    try:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://agora") as client:
                await _run(app, client)
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(
            f"demo failed ({exc}). Start dependencies first:\n"
            "  docker compose up -d\n"
            "  uv sync\n"
            + _relay_hint()
        ) from exc


async def _run(app: FastAPI, client: httpx.AsyncClient) -> None:
    room = (
        await client.post("/rooms", json={"name": "phase7-demo", "mode": "moderated"})
    ).json()
    room_id = room["id"]
    human = (
        await client.post(
            f"/rooms/{room_id}/participants",
            json={"kind": "human", "name": "Ada"},
        )
    ).json()
    chair = (
        await client.post(
            f"/rooms/{room_id}/participants",
            json={
                "kind": "agent",
                "name": "Chair",
                "role": "moderator",
                "persona": (
                    "简洁的中文主持人。用 decide 点名最合适的成员作答，"
                    "自己不回答实质问题。正文点到座位名就优先 call_on 那个座位。"
                    "程序说明才 say，否则 silence。"
                ),
            },
        )
    ).json()
    iris = (
        await client.post(
            f"/rooms/{room_id}/participants",
            json={
                "kind": "agent",
                "name": "Iris",
                "persona": "后端工程师。只谈实现、数据库、API。用中文简短回答。",
            },
        )
    ).json()
    marcus = (
        await client.post(
            f"/rooms/{room_id}/participants",
            json={
                "kind": "agent",
                "name": "Marcus",
                "persona": "产品经理。只谈需求、优先级、用户。用中文简短回答。",
            },
        )
    ).json()
    lex = (
        await client.post(
            f"/rooms/{room_id}/participants",
            json={
                "kind": "agent",
                "name": "Lex",
                "persona": (
                    "法务合规。只谈合同、隐私、监管风险。"
                    "产品优先级、排期、实现不是你的座位——"
                    "被点到这类问题就保持沉默，不要 reply。"
                ),
            },
        )
    ).json()

    roster = {p["id"]: p for p in (human, chair, iris, marcus, lex)}

    _say()
    _say(f"room     {room['name']}  mode={room['mode']}  {room_id}")
    _say(f"human    {human['name']}    {human['id']}")
    _say(f"moderator {chair['name']}  {chair['id']}")
    _say(f"member   {iris['name']}   {iris['id']}")
    _say(f"member   {marcus['name']} {marcus['id']}")
    _say(f"member   {lex['name']}    {lex['id']}")
    _say()
    _say("--- Ada asks a backend question (Chair should call_on Iris) ---")

    posted = await client.post(
        f"/rooms/{room_id}/messages",
        json={
            "author_id": human["id"],
            "body": (
                "Postgres 里房间序号为什么用行上的计数器而不是 SEQUENCE？"
                "请恰好一个人回答。"
            ),
        },
    )
    posted.raise_for_status()
    _say(f"message seq={posted.json()['seq']}  {posted.json()['body']}")

    await _wait_brain(app, 1)
    listed = await _wait_quiet(app, client, room_id)

    names = {
        human["id"]: "Ada",
        chair["id"]: "Chair",
        iris["id"]: "Iris",
        marcus["id"]: "Marcus",
        lex["id"]: "Lex",
    }

    def _print_transcript(messages: list[dict]) -> None:
        for message in messages:
            who = roster.get(message["author_id"])
            label = names.get(message["author_id"], message["author_id"][:8])
            role = _role_of(who) if who else "?"
            _say(f"  [{message['seq']}] {label} ({role}): {message['body']}")

    _say()
    _say("--- per-agent brain ---")
    for turn in app.state.scheduler.brain_results:
        claims = ", ".join(f"{k}={v}" for k, v in turn.claims) or "-"
        _say(
            f"  {turn.agent_name:8}  outcome={turn.outcome:16}  "
            f"triage={turn.triage_actionable} {turn.response_mode or '-'}  "
            f"({turn.triage_reason or '-'})"
        )
        _say(
            f"           claims={claims}  holds={turn.hold_count}  "
            f"hops={turn.hop_count}  reply={turn.reply_body!r}"
        )

    _say()
    _say("--- room transcript ---")
    _print_transcript(listed)

    decisions = await db.list_decisions(app.state.pool, UUID(room_id))
    _say()
    _say("--- moderator_decisions ---")
    if not decisions:
        _say("  (none)")
    else:
        id_to_name = {UUID(p["id"]): p["name"] for p in roster.values()}
        for row in decisions:
            target = (
                "—"
                if row.target_id is None
                else id_to_name.get(row.target_id, str(row.target_id))
            )
            _say(
                f"  trigger_seq={row.trigger_seq}  action={row.action}  "
                f"target={target}  at={row.created_at.isoformat(timespec='seconds')}"
            )

    digest = await client.get(f"/rooms/{room_id}/digest")
    digest.raise_for_status()
    _say()
    _say("--- digest ---")
    _say(digest.text)

    before_n = len(listed)
    _say("--- Ada @-mentions Marcus (mention bypasses the chair) ---")
    second = await client.post(
        f"/rooms/{room_id}/messages",
        json={
            "author_id": human["id"],
            "body": "@Marcus 请用一句话说明产品侧怎么看这个房间。",
        },
    )
    second.raise_for_status()
    _say(f"message seq={second.json()['seq']}  {second.json()['body']}")

    await _wait_brain(app, len(app.state.scheduler.brain_results) + 1)
    after = await _wait_quiet(app, client, room_id)

    _say()
    _say("--- delta ---")
    _print_transcript(after[before_n:])

    digest2 = await client.get(f"/rooms/{room_id}/digest")
    digest2.raise_for_status()
    _say()
    _say("--- digest (after mention) ---")
    _say(digest2.text)

    # Decline → pass → redirect. Persona (not a scripted model) is the
    # only lever: Lex speaks only on legal topics; Ada asks a product
    # question and names Lex so Chair should call_on that seat.
    before_decline = len(after)
    trigger_floor = after[-1]["seq"] if after else 0
    _say()
    _say("--- decline → pass → redirect ---")
    _say(
        "不变量：被点名成员沉默必须落地一条带名字的 pass 推进 seq；"
        "主持被新 seq 叫醒后才能写新的 decide。"
        "从「没回」推断弃权是慢 turn 下的 TOCTOU。"
    )
    third = await client.post(
        f"/rooms/{room_id}/messages",
        json={
            "author_id": human["id"],
            "body": "请 Lex 先说：消息表的 seq 列该用 BIGINT 还是 INTEGER？给个实现建议。",
        },
    )
    third.raise_for_status()
    _say(f"message seq={third.json()['seq']}  {third.json()['body']}")

    await _wait_brain(app, len(app.state.scheduler.brain_results) + 1)
    declined = await _wait_quiet(app, client, room_id)

    decisions_after = await db.list_decisions(app.state.pool, UUID(room_id))
    id_to_name = {UUID(p["id"]): p["name"] for p in roster.values()}
    lex_id = UUID(lex["id"])
    pass_body = f"{lex['name']} passes."
    pass_row = next(
        (
            m
            for m in declined
            if m["author_id"] == lex["id"]
            and m["body"] == pass_body
            and m["seq"] > trigger_floor
        ),
        None,
    )
    first_call = next(
        (
            d
            for d in decisions_after
            if d.action == "call_on"
            and d.target_id == lex_id
            and d.trigger_seq >= trigger_floor
        ),
        None,
    )
    redirect = None
    if first_call is not None:
        redirect = next(
            (
                d
                for d in decisions_after
                if d.action == "call_on"
                and d.trigger_seq > first_call.trigger_seq
                and d.target_id is not None
                and d.target_id != lex_id
            ),
            None,
        )
    answer = None
    if redirect is not None and pass_row is not None:
        target = str(redirect.target_id)
        answer = next(
            (
                m
                for m in declined
                if m["author_id"] == target
                and m["seq"] > pass_row["seq"]
                and not str(m["body"]).endswith(" passes.")
            ),
            None,
        )

    if (
        first_call is None
        or pass_row is None
        or redirect is None
        or answer is None
    ):
        _say(
            "decline path was not exercised this run "
            "(model did not decline, or Chair did not redirect) — not faked."
        )
        _say("--- delta ---")
        _print_transcript(declined[before_decline:])
    else:
        _say()
        _say("--- call_on (Lex) ---")
        target = id_to_name.get(first_call.target_id, str(first_call.target_id))
        _say(
            f"  trigger_seq={first_call.trigger_seq}  action={first_call.action}  "
            f"target={target}  at={first_call.created_at.isoformat(timespec='seconds')}"
        )
        _say("--- pass ---")
        _print_transcript([pass_row])
        _say("--- call_on (redirect) ---")
        rtarget = id_to_name.get(redirect.target_id, str(redirect.target_id))
        _say(
            f"  trigger_seq={redirect.trigger_seq}  action={redirect.action}  "
            f"target={rtarget}  at={redirect.created_at.isoformat(timespec='seconds')}"
        )
        _say("--- second member ---")
        _print_transcript([answer])

    digest3 = await client.get(f"/rooms/{room_id}/digest")
    digest3.raise_for_status()
    _say()
    _say("--- digest (final) ---")
    _say(digest3.text)


if __name__ == "__main__":
    asyncio.run(main())
