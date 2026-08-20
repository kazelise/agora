"""Phase 1 demo: one human, two agents, a post, then a burst to show coalescing."""

from __future__ import annotations

import asyncio
import logging
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


async def _wait_turns(app: FastAPI, minimum: int, timeout: float = 3.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if len(app.state.scheduler.turns) >= minimum:
            await app.state.scheduler.wait_idle()
            return
        await asyncio.sleep(0.02)
    raise TimeoutError(
        f"only {len(app.state.scheduler.turns)} turns after {timeout}s, wanted {minimum}"
    )


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    app = create_app()
    try:
        async with app.router.lifespan_context(app):
            app.state.scheduler.turn_delay_s = 0.2
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://agora") as client:
                await _run(app, client)
    except Exception as exc:
        raise SystemExit(
            f"demo failed ({exc}). Start dependencies first:\n"
            "  docker compose up -d\n"
            "  uv sync"
        ) from exc


async def _run(app: FastAPI, client: httpx.AsyncClient) -> None:
    room = (await client.post("/rooms", json={"name": "phase1-demo"})).json()
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
            json={"kind": "agent", "name": "Iris", "persona": "terse teammate"},
        )
    ).json()
    marcus = (
        await client.post(
            f"/rooms/{room_id}/participants",
            json={
                "kind": "agent",
                "name": "Marcus",
                "persona": "also terse teammate",
            },
        )
    ).json()

    _say()
    _say(f"room     {room['name']}  {room_id}")
    _say(f"human    {human['name']}    {human['id']}")
    _say(f"agent    {iris['name']}   {iris['id']}")
    _say(f"agent    {marcus['name']} {marcus['id']}")
    _say()
    _say("--- Ada posts once (Iris and Marcus should each wake once) ---")

    first = await client.post(
        f"/rooms/{room_id}/messages",
        json={"author_id": human["id"], "body": "hello team, please stand by"},
    )
    first.raise_for_status()
    _say(f"message seq={first.json()['seq']}  {first.json()['body']}")
    await _wait_turns(app, 2)

    _say()
    _say("--- Ada posts 5 more concurrently (in-flight + one rerun, not 5 turns) ---")
    await asyncio.gather(
        *[
            client.post(
                f"/rooms/{room_id}/messages",
                json={"author_id": human["id"], "body": f"burst-{i + 1}"},
            )
            for i in range(5)
        ]
    )
    await _wait_turns(app, 4)

    _say()
    _say("--- wake / coalescing log ---")
    by_agent: dict[str, int] = {}
    for turn in app.state.scheduler.turns:
        by_agent[turn.agent_name] = by_agent.get(turn.agent_name, 0) + 1
        _say(
            f"  {turn.agent_name:8}  inbox={turn.inbox_count}  "
            f"since_seq={turn.since_seq}  last_read={turn.last_read_seq}"
        )
    _say()
    _say(f"turns per agent: {by_agent}")
    listed = await client.get(f"/rooms/{room_id}/messages")
    _say(f"messages in room: {len(listed.json()['messages'])}")


if __name__ == "__main__":
    asyncio.run(main())
