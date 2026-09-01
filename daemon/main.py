"""BYOA daemon: websocket wakes + the same Brain, local ChatOpenAI."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Any
from urllib.parse import urlparse, urlunparse
from uuid import UUID

import httpx
from langchain_openai import ChatOpenAI

from brain.graph import Brain, TurnResult
from brain.policy import big_model_name, small_model_name
from daemon.lanes import AgentLane
from daemon.limiter import DEFAULT_MAX_CONCURRENT, ConcurrencyLimiter
from daemon.pacer import BASE_INTERVAL_S, MAX_INTERVAL_S, AdaptivePacer
from daemon.world_http import HttpWorld

logger = logging.getLogger("agora.daemon")

HEARTBEAT_S = 10.0
BACKOFF_CAP_S = 30.0


def _ws_url(server: str, computer_id: str, token: str) -> str:
    parsed = urlparse(server)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    netloc = parsed.netloc or parsed.path
    path = f"/ws/computers/{computer_id}"
    return urlunparse((scheme, netloc, path, "", f"token={token}", ""))


def _openai_base() -> str:
    return (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or ""
    )


def build_brain(
    world: HttpWorld,
    *,
    pacer: AdaptivePacer | None = None,
    limiter: ConcurrencyLimiter | None = None,
) -> Brain:
    small = small_model_name()
    big = big_model_name()
    # ChatOpenAI reads OPENAI_API_KEY / OPENAI_BASE_URL from this process.
    # The server never sees them.
    return Brain(
        world,
        small_model=ChatOpenAI(model=small),
        big_model=ChatOpenAI(model=big),
        small_model_name=small,
        big_model_name=big,
        pacer=pacer,
        limiter=limiter,
    )


def _log_turn(result: TurnResult) -> None:
    claims = ", ".join(f"{k}={v}" for k, v in result.claims) or "-"
    logger.info(
        "turn %s outcome=%s triage=%s %s (%s) holds=%s hops=%s claims=%s reply=%r",
        result.agent_name,
        result.outcome,
        result.triage_actionable,
        result.response_mode or "-",
        result.triage_reason or "-",
        result.hold_count,
        result.hop_count,
        claims,
        result.reply_body,
    )


class Daemon:
    def __init__(
        self,
        server: str,
        computer_id: str,
        token: str,
        *,
        brain: Brain | None = None,
        world: HttpWorld | None = None,
        http: httpx.AsyncClient | None = None,
        pacer: AdaptivePacer | None = None,
        limiter: ConcurrencyLimiter | None = None,
    ) -> None:
        self.server = server.rstrip("/")
        self.computer_id = computer_id
        self.token = token
        self._http = http
        self._world = world
        self._brain = brain
        self._pacer = pacer
        self._limiter = limiter
        self._lanes: dict[UUID, AgentLane] = {}
        self._called_on: dict[tuple[UUID, UUID], bool] = {}
        self._owns_http = http is None

    def _lane(self, agent_id: UUID) -> AgentLane:
        lane = self._lanes.get(agent_id)
        if lane is None:
            lane = AgentLane(self._run_turn)
            self._lanes[agent_id] = lane
        return lane

    async def _run_turn(self, agent_id: UUID, room_id: UUID) -> None:
        assert self._brain is not None
        called_on = self._called_on.pop((agent_id, room_id), False)
        try:
            result = await self._brain.run(agent_id, room_id, called_on=called_on)
        except Exception:
            logger.exception("turn failed agent=%s room=%s", agent_id, room_id)
            return
        _log_turn(result)

    async def handle_frame(self, raw: str) -> None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("non-json frame: %s", raw[:120])
            return
        if data.get("type") == "wake":
            agent_id = UUID(data["agent_id"])
            room_id = UUID(data["room_id"])
            key = (agent_id, room_id)
            self._called_on[key] = self._called_on.get(key, False) or bool(
                data.get("called_on")
            )
            logger.info(
                "wake agent=%s room=%s called_on=%s",
                agent_id,
                room_id,
                self._called_on[key],
            )
            overwritten = await self._lane(agent_id).notify(room_id, agent_id)
            if overwritten is not None:
                old_room, old_agent = overwritten
                if (old_agent, old_room) != (agent_id, room_id):
                    self._called_on.pop((old_agent, old_room), None)
        elif data.get("type") == "pong":
            logger.debug("pong")

    async def _session(self, connect: Any) -> None:
        uri = _ws_url(self.server, self.computer_id, self.token)
        async with connect(uri) as ws:
            logger.info("connected to %s", uri.split("?", 1)[0])
            stop = asyncio.Event()

            async def heartbeat() -> None:
                while not stop.is_set():
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=HEARTBEAT_S)
                        return
                    except TimeoutError:
                        await ws.send(json.dumps({"type": "ping"}))

            beat = asyncio.create_task(heartbeat())
            try:
                async for message in ws:
                    if isinstance(message, bytes):
                        message = message.decode()
                    await self.handle_frame(message)
            finally:
                stop.set()
                beat.cancel()
                try:
                    await beat
                except asyncio.CancelledError:
                    pass

    async def run_forever(self) -> None:
        try:
            from websockets.asyncio.client import connect
        except ImportError:  # pragma: no cover
            from websockets import connect  # type: ignore[no-redef]

        owns = self._http is None
        client = self._http or httpx.AsyncClient(base_url=self.server, timeout=60.0)
        try:
            world = self._world or HttpWorld(client, self.token)
            self._world = world
            # One pacer + one limiter per computer: triage (small) and turn
            # (big) models share the same provider account, so both classes
            # must flow through the same rate budget (Cumora §3/§3a: capping
            # one layer without the other just moves the thundering herd up).
            pacer = self._pacer or AdaptivePacer()
            self._pacer = pacer
            limiter = self._limiter or ConcurrencyLimiter()
            self._limiter = limiter
            self._brain = self._brain or build_brain(world, pacer=pacer, limiter=limiter)
            delay = 1.0
            while True:
                try:
                    await self._session(connect)
                    delay = 1.0
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning(
                        "ws disconnected (%s: %s); reconnect in %.0fs",
                        type(exc).__name__,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, BACKOFF_CAP_S)
        finally:
            if owns:
                await client.aclose()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="daemon", description="Agora BYOA computer")
    parser.add_argument(
        "--server",
        default=os.environ.get("AGORA_SERVER_URL", "http://127.0.0.1:8000"),
    )
    parser.add_argument(
        "--computer-id",
        default=os.environ.get("AGORA_COMPUTER_ID"),
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("AGORA_COMPUTER_TOKEN"),
    )
    parser.add_argument(
        "--pacer-base-s",
        type=float,
        default=float(os.environ.get("AGORA_PACER_BASE_S", BASE_INTERVAL_S)),
        help="minimum spacing between LLM call starts (default 0.5)",
    )
    parser.add_argument(
        "--pacer-max-s",
        type=float,
        default=float(os.environ.get("AGORA_PACER_MAX_S", MAX_INTERVAL_S)),
        help="cap for the adaptive interval after rate limits (default 8.0)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=int(
            os.environ.get("AGORA_MAX_CONCURRENT", DEFAULT_MAX_CONCURRENT)
        ),
        help=(
            "max model calls in flight on this computer, both model classes "
            "combined (default 6); lower to 2-4 on tight provider quotas"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.computer_id or not args.token:
        raise SystemExit(
            "need --computer-id and --token "
            "(or AGORA_COMPUTER_ID / AGORA_COMPUTER_TOKEN)"
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
    logger.info(
        "daemon starting server=%s computer=%s base_url=%s small=%s big=%s",
        args.server,
        args.computer_id,
        _openai_base() or "(default)",
        small_model_name(),
        big_model_name(),
    )
    daemon = Daemon(
        args.server,
        args.computer_id,
        args.token,
        pacer=AdaptivePacer(
            base_interval_s=args.pacer_base_s, max_interval_s=args.pacer_max_s
        ),
        limiter=ConcurrencyLimiter(max_concurrent=args.max_concurrent),
    )
    try:
        asyncio.run(daemon.run_forever())
    except KeyboardInterrupt:
        logger.info("daemon stopped")


if __name__ == "__main__":
    main()
