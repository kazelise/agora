"""BYOA money-shot: one room, one cloud agent, one local daemon, then sleep."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import sys
from pathlib import Path

import httpx
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.main import create_app  # noqa: E402


def _say(text: str = "") -> None:
    print(text, flush=True)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def _wait_online(
    client: httpx.AsyncClient, computer_id: str, timeout: float = 15.0
) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        rows = (await client.get("/computers")).json()
        row = next((c for c in rows if c["id"] == computer_id), None)
        if row is not None and row["online"]:
            return
        await asyncio.sleep(0.1)
    raise TimeoutError("daemon did not come online")


async def _messages(client: httpx.AsyncClient, room_id: str) -> list[dict]:
    return (await client.get(f"/rooms/{room_id}/messages")).json()["messages"]


def _spoken(listed: list[dict], authors: set[str], since_seq: int = 0) -> set[str]:
    return {
        m["author_id"]
        for m in listed
        if m["seq"] > since_seq and m["author_id"] in authors
    }


async def _wait_authors(
    client: httpx.AsyncClient,
    room_id: str,
    authors: set[str],
    *,
    since_seq: int = 0,
    timeout: float = 90.0,
) -> list[dict]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    listed: list[dict] = []
    while loop.time() < deadline:
        listed = await _messages(client, room_id)
        if authors <= _spoken(listed, authors, since_seq):
            return listed
        await asyncio.sleep(0.4)
    missing = authors - _spoken(listed, authors, since_seq)
    raise TimeoutError(
        f"still waiting for {missing}; last={[m['body'] for m in listed]}"
    )


async def _pump(proc: asyncio.subprocess.Process) -> None:
    assert proc.stdout is not None
    while True:
        line = await proc.stdout.readline()
        if not line:
            return
        _say(f"  daemon | {line.decode().rstrip()}")


async def _stop(proc: asyncio.subprocess.Process | None) -> None:
    if proc is None or proc.returncode is not None:
        return
    proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), 5)
    except TimeoutError:
        proc.kill()
        await proc.wait()


async def _demo(client: httpx.AsyncClient, server_url: str) -> None:
    computer = (await client.post("/computers", json={"name": "Ada's laptop"})).json()
    room = (await client.post("/rooms", json={"name": "byoa-demo"})).json()
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
                "persona": (
                    "You are Iris, hosted on the agora cloud. "
                    "When asked to speak, you MUST reply with exactly one short "
                    "sentence in the room's language, even if another agent "
                    "already posted. Mention that you run on the cloud host. "
                    "Do not claim; this is not a one-of-us task."
                ),
            },
        )
    ).json()
    jules = (
        await client.post(
            f"/rooms/{room_id}/participants",
            json={
                "kind": "agent",
                "name": "Jules",
                "persona": (
                    "You are Jules, hosted on Ada's laptop (BYOA). "
                    "When asked to speak, you MUST reply with exactly one short "
                    "sentence in the room's language, even if another agent "
                    "already posted. Mention that you run on the local computer. "
                    "Do not claim; this is not a one-of-us task."
                ),
                "computer_id": computer["id"],
            },
        )
    ).json()

    names = {
        human["id"]: "Ada",
        iris["id"]: "Iris",
        jules["id"]: "Jules",
    }
    hosts = {
        human["id"]: "human",
        iris["id"]: "cloud",
        jules["id"]: "byoa",
    }

    _say()
    _say("=== Agora BYOA ===")
    _say(f"server     {server_url}")
    _say(f"computer   {computer['name']}  {computer['id']}")
    _say(f"token      (shown once)  {computer['token'][:12]}…")
    _say(f"room       {room['name']}  {room_id}")
    _say(f"human      Ada     {human['id']}")
    _say(f"agent      Iris    {iris['id']}   host=cloud")
    _say(f"agent      Jules   {jules['id']}   host=byoa / Ada's laptop")
    _say()
    _say("--- spawn daemon (its own OPENAI_* ; server never sees the key) ---")

    env = os.environ.copy()
    env["AGORA_SERVER_URL"] = server_url
    env["AGORA_COMPUTER_ID"] = computer["id"]
    env["AGORA_COMPUTER_TOKEN"] = computer["token"]
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "daemon",
        cwd=ROOT,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    pump = asyncio.create_task(_pump(proc))
    try:
        await _wait_online(client, computer["id"])
        _say(f"daemon     pid={proc.pid}  status=online")
        _say()
        _say("--- Ada asks both hosts to speak ---")

        first = await client.post(
            f"/rooms/{room_id}/messages",
            json={
                "author_id": human["id"],
                "body": (
                    "请你们每个人都发言：各用一句话介绍这个房间，并说明你跑在哪台 "
                    "Computer 上。即使别人已经说过，你也必须再说一句。"
                ),
            },
        )
        first.raise_for_status()
        _say(f"message    seq={first.json()['seq']}  Ada: {first.json()['body']}")

        both = {iris["id"], jules["id"]}
        try:
            listed = await _wait_authors(client, room_id, both, timeout=18.0)
        except TimeoutError:
            listed = await _messages(client, room_id)
            missing = both - _spoken(listed, both)
            nudges = {
                iris["id"]: "Iris，请你现在用一句话说明你跑在云端 Computer 上。",
                jules["id"]: "Jules，请你现在用一句话说明你跑在本地 Computer 上。",
            }
            for agent_id in (iris["id"], jules["id"]):
                if agent_id not in missing:
                    continue
                _say(
                    f"NUDGE FIRED: {names[agent_id]} did not answer the each-question; "
                    "Ada posts a fallback"
                )
                nudge = await client.post(
                    f"/rooms/{room_id}/messages",
                    json={"author_id": human["id"], "body": nudges[agent_id]},
                )
                nudge.raise_for_status()
                _say(f"message    seq={nudge.json()['seq']}  Ada: {nudge.json()['body']}")
            listed = await _wait_authors(client, room_id, both, timeout=60.0)
        _say()
        _say("--- who answered (both hosts up) ---")
        for message in listed:
            if message["author_id"] == human["id"]:
                continue
            _say(
                f"  [{hosts[message['author_id']]:5}] {names[message['author_id']]}: "
                f"{message['body']}"
            )

        _say()
        _say("--- kill daemon; Jules should sleep, Iris still answers ---")
        await _stop(proc)
        await asyncio.sleep(0.4)
        _say(f"daemon     pid={proc.pid}  status=offline")

        last_seq = max(m["seq"] for m in listed)
        second = await client.post(
            f"/rooms/{room_id}/messages",
            json={
                "author_id": human["id"],
                "body": "Jules 的电脑断了。谁还醒着，请回一声。",
            },
        )
        second.raise_for_status()
        _say(f"message    seq={second.json()['seq']}  Ada: {second.json()['body']}")

        await _wait_authors(
            client, room_id, {iris["id"]}, since_seq=last_seq, timeout=90.0
        )
        await asyncio.sleep(2.0)
        later = (await client.get(f"/rooms/{room_id}/messages")).json()["messages"]

        _say()
        _say("--- who answered (Jules sleeping) ---")
        new_byoa = [
            m
            for m in later
            if m["seq"] > last_seq and m["author_id"] == jules["id"]
        ]
        new_cloud = [
            m
            for m in later
            if m["seq"] > last_seq and m["author_id"] == iris["id"]
        ]
        for message in new_cloud:
            _say(f"  [cloud] Iris: {message['body']}")
        if new_byoa:
            _say("  unexpected BYOA reply while the computer was offline:")
            for message in new_byoa:
                _say(f"  [byoa ] Jules: {message['body']}")
        else:
            _say("  [byoa ] Jules is sleeping (computer offline) — wake dropped")

        _say()
        _say("--- room transcript ---")
        for message in later:
            author = names.get(message["author_id"], message["author_id"][:8])
            host = hosts.get(message["author_id"], "?")
            _say(f"  [{message['seq']}] {author:6} ({host}): {message['body']}")

        if not new_cloud:
            raise SystemExit("cloud agent Iris did not answer after the daemon died")
        if new_byoa:
            raise SystemExit("BYOA agent Jules answered while its computer was offline")
    finally:
        pump.cancel()
        await _stop(proc)


async def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "demo_byoa needs OPENAI_API_KEY and OPENAI_BASE_URL in the environment.\n"
            "The daemon subprocess inherits them as *its* key — the server never sees it.\n"
            "  export OPENAI_API_KEY=relay-no-key\n"
            "  export OPENAI_BASE_URL=http://192.168.1.100:8317/v1\n"
            "  export OPENAI_API_BASE=$OPENAI_BASE_URL\n"
            "Then: uv run python scripts/demo_byoa.py"
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    port = _free_port()
    server_url = f"http://127.0.0.1:{port}"
    app = create_app()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    serve_task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)

    try:
        async with httpx.AsyncClient(base_url=server_url, timeout=60.0) as client:
            await _demo(client, server_url)
    finally:
        server.should_exit = True
        await serve_task


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except SystemExit:
        raise
    except TimeoutError as exc:
        raise SystemExit(f"demo timed out: {exc}") from exc
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
