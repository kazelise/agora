"""One-shot cloud turn: the container command for a K8s Job.

Same graph as the daemon. Transport is HttpWorld with the cluster token.
The process holds OPENAI_* ; the server never sees those values on this
path. Exit 0 after one turn, 1 if the turn raises.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from typing import Any
from uuid import UUID

import httpx
from langchain_openai import ChatOpenAI

from brain.graph import Brain, TurnResult
from brain.policy import big_model_name, small_model_name
from daemon.world_http import HttpWorld

logger = logging.getLogger("agora.job")


def _openai_base() -> str:
    return (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or ""
    )


def build_brain(world: HttpWorld, **kwargs: Any) -> Brain:
    small = small_model_name()
    big = big_model_name()
    return Brain(
        world,
        small_model=kwargs.pop("small_model", None) or ChatOpenAI(model=small),
        big_model=kwargs.pop("big_model", None) or ChatOpenAI(model=big),
        small_model_name=small,
        big_model_name=big,
        **kwargs,
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


async def run_once(
    server: str,
    token: str,
    agent_id: UUID,
    room_id: UUID,
    *,
    brain: Brain | None = None,
    http: httpx.AsyncClient | None = None,
) -> TurnResult:
    owns = http is None
    client = http or httpx.AsyncClient(base_url=server.rstrip("/"), timeout=60.0)
    try:
        world = HttpWorld(client, token)
        engine = brain or build_brain(world)
        result = await engine.run(agent_id, room_id)
        _log_turn(result)
        return result
    finally:
        if owns:
            await client.aclose()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="brain.job",
        description="Agora one-shot cloud turn (K8s Job entrypoint)",
    )
    parser.add_argument(
        "--server",
        default=os.environ.get("AGORA_SERVER_URL", "http://127.0.0.1:8000"),
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("AGORA_CLUSTER_TOKEN")
        or os.environ.get("AGORA_COMPUTER_TOKEN"),
    )
    parser.add_argument(
        "--agent-id",
        default=os.environ.get("AGORA_AGENT_ID"),
    )
    parser.add_argument(
        "--room-id",
        default=os.environ.get("AGORA_ROOM_ID"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.token or not args.agent_id or not args.room_id:
        raise SystemExit(
            "need --token, --agent-id, --room-id "
            "(or AGORA_CLUSTER_TOKEN / AGORA_AGENT_ID / AGORA_ROOM_ID)"
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
        "job starting server=%s agent=%s room=%s base_url=%s small=%s big=%s",
        args.server,
        args.agent_id,
        args.room_id,
        _openai_base() or "(default)",
        small_model_name(),
        big_model_name(),
    )
    try:
        asyncio.run(
            run_once(
                args.server,
                args.token,
                UUID(args.agent_id),
                UUID(args.room_id),
            )
        )
    except KeyboardInterrupt:
        logger.info("job interrupted")
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
