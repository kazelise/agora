"""Phase 2 demo: two agents, one-of-us intro. Needs a real OPENAI_API_KEY."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI  # noqa: E402

from server.main import create_app  # noqa: E402


def _say(text: str = "") -> None:
    print(text, flush=True)


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


async def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "demo_phase2 needs OPENAI_API_KEY and OPENAI_BASE_URL in the environment.\n"
            "Local relay example:\n"
            "  export OPENAI_API_KEY=relay-no-key\n"
            "  export OPENAI_BASE_URL=http://192.168.1.100:8317/v1\n"
            "  export OPENAI_API_BASE=$OPENAI_BASE_URL\n"
            "Then: uv run python scripts/demo_phase2.py\n"
            "Models default to AGORA_SMALL_MODEL=gpt-5.6-luna and "
            "AGORA_BIG_MODEL=gpt-5.6-terra (override either if you want)."
        )

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
            "  export OPENAI_API_KEY=relay-no-key\n"
            "  export OPENAI_BASE_URL=http://192.168.1.100:8317/v1\n"
            "  export OPENAI_API_BASE=$OPENAI_BASE_URL\n"
            "  export AGORA_DATABASE_URL=postgresql://agora:agora@127.0.0.1:5433/agora\n"
            "  export AGORA_REDIS_URL=redis://127.0.0.1:6379/0"
        ) from exc


async def _run(app: FastAPI, client: httpx.AsyncClient) -> None:
    room = (await client.post("/rooms", json={"name": "phase2-demo"})).json()
    room_id = room["id"]
    human = (
        await client.post(
            f"/rooms/{room_id}/participants",
            json={"kind": "human", "name": "Ada"},
        )
    ).json()
    iris = (
        await client.post(
            f"/rooms/{room_id}/participants",
            json={
                "kind": "agent",
                "name": "Iris",
                "persona": "Concise coordinator. Answer in the room's language.",
            },
        )
    ).json()
    marcus = (
        await client.post(
            f"/rooms/{room_id}/participants",
            json={
                "kind": "agent",
                "name": "Marcus",
                "persona": "Warm explainer. Answer in the room's language.",
            },
        )
    ).json()

    _say()
    _say(f"room     {room['name']}  {room_id}")
    _say(f"human    {human['name']}    {human['id']}")
    _say(f"agent    {iris['name']}   {iris['id']}")
    _say(f"agent    {marcus['name']} {marcus['id']}")
    _say()
    _say("--- Ada asks one of them to introduce the room ---")

    posted = await client.post(
        f"/rooms/{room_id}/messages",
        json={
            "author_id": human["id"],
            "body": "大家好，请你们中的一个人介绍一下这个房间是干什么的",
        },
    )
    posted.raise_for_status()
    _say(f"message seq={posted.json()['seq']}  {posted.json()['body']}")

    await _wait_brain(app, 2)

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

    listed = await client.get(f"/rooms/{room_id}/messages")
    _say()
    _say("--- room transcript ---")
    for message in listed.json()["messages"]:
        author = {
            human["id"]: "Ada",
            iris["id"]: "Iris",
            marcus["id"]: "Marcus",
        }.get(message["author_id"], message["author_id"][:8])
        _say(f"  [{message['seq']}] {author}: {message['body']}")


if __name__ == "__main__":
    asyncio.run(main())
