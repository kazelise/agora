"""One LangGraph: triage → tool_loop → freshness HOLD → commit."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal, TypedDict
from uuid import UUID, uuid4

import asyncpg
import redis.asyncio as redis
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from brain import policy
from brain.seen import record_seen
from server import db
from server.models import MessageRow

logger = logging.getLogger("agora.brain")

MAX_HOPS = 6
MAX_HOLDS = 2
LLM_RETRY_BACKOFF_S = 1.0

# Protocol parse of the model's claim key — not content classification.
# Free-form names never converge across two models; seq is objective.
_TASK_KEY = re.compile(r"^t(\d+)(?::[A-Za-z0-9_-]+)?$")
CLAIM_KEY_ERROR = (
    "claim rejected: task_key must be t<seq> or t<seq>:<slug> "
    "(e.g. t1 or t1:intro), where <seq> is the seq of the message "
    "you are responding to. Retry with that format."
)


def canonical_task_key(task_key: str) -> str | None:
    """Return `t<seq>` if `task_key` is `t<seq>` or `t<seq>:<slug>`, else None.

    The slug is for the model; the lock is the seq prefix. Two agents
    writing t1 and t1:intro must contend on the same claims row.
    """
    match = _TASK_KEY.fullmatch(task_key.strip())
    if match is None:
        return None
    return f"t{match.group(1)}"


class TriageVerdict(BaseModel):
    actionable: bool
    reason: str
    response_mode: Literal["me", "each", "one-of-us"]


class InboxItem(TypedDict):
    seq: int
    body: str
    author_id: str
    author_name: str


class ClaimRecord(TypedDict):
    task_key: str
    result: str


class BrainState(TypedDict, total=False):
    agent_id: str
    agent_name: str
    persona: str
    room_id: str
    inbox: list[InboxItem]
    author_names: dict[str, str]
    seen_seq: int
    triage_actionable: bool
    triage_reason: str
    response_mode: str
    messages: list[BaseMessage]
    hold_count: int
    hop_count: int
    pending_reply: str
    outcome: str
    claims: list[ClaimRecord]


@tool
def reply(body: str) -> str:
    """Post your message to the room."""
    return body


@tool
def claim(task_key: str) -> str:
    """Atomically claim a task. task_key must be t<seq> or t<seq>:<slug>."""
    return task_key


TOOLS = [reply, claim]


@dataclass(frozen=True)
class TurnResult:
    agent_id: UUID
    agent_name: str
    room_id: UUID
    outcome: str
    hold_count: int
    hop_count: int
    seen_seq: int
    inbox_count: int
    triage_actionable: bool | None
    triage_reason: str | None
    response_mode: str | None
    claims: tuple[tuple[str, str], ...]
    reply_body: str | None


def token_usage(message: Any) -> tuple[int, int]:
    meta = getattr(message, "usage_metadata", None) or {}
    if meta:
        return int(meta.get("input_tokens") or 0), int(meta.get("output_tokens") or 0)
    resp = getattr(message, "response_metadata", None) or {}
    usage = resp.get("token_usage") or resp.get("usage") or {}
    return (
        int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
        int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
    )


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        return "".join(parts)
    return str(content or "")


def _parse_triage(text: str) -> TriageVerdict:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return TriageVerdict.model_validate_json(raw)


def _format_inbox(items: list[InboxItem]) -> str:
    if not items:
        return "(empty)"
    return "\n".join(f"[seq={m['seq']} {m['author_name']}] {m['body']}" for m in items)


def _tool_prompt(state: BrainState) -> str:
    return (
        f"You are {state['agent_name']}. {state.get('persona') or ''}\n\n"
        f"Triage: actionable={state.get('triage_actionable')} "
        f"response_mode={state.get('response_mode')} ({state.get('triage_reason')}).\n\n"
        f"Room:\n{_format_inbox(state.get('inbox') or [])}\n\n"
        "Tools: reply(body) posts to the room; claim(task_key) atomically "
        "claims a task and returns won or lost.\n"
        "If response_mode is one-of-us: claim first, reply only if you won.\n"
        "If response_mode is me or each: do not claim; reply.\n"
        "claim task_key must be t<seq> or t<seq>:<slug> "
        "(seq = the message you are responding to)."
    )


def _tc_get(tc: Any, key: str, default: Any = None) -> Any:
    if isinstance(tc, dict):
        return tc.get(key, default)
    return getattr(tc, key, default)


def _payload_shape(messages: list[Any]) -> str:
    """Compact role/length dump so a 400 can be diagnosed without dumping bodies."""
    bits: list[str] = []
    for i, message in enumerate(messages):
        role = getattr(message, "type", None) or type(message).__name__
        content = _text(getattr(message, "content", ""))
        calls = getattr(message, "tool_calls", None) or []
        names = ",".join(str(_tc_get(tc, "name") or "?") for tc in calls)
        bits.append(f"{i}:{role}:chars={len(content)}:tools={names or '-'}")
    return "[" + " ".join(bits) + "]"


async def invoke_model(model: Any, messages: list[Any], *, label: str) -> Any | None:
    """Call the model; retry once after a short backoff; None if both fail.

    Fail-open: a dead relay is a missed reply, not a crashed turn. No
    caller should write a ledger row unless this returns a message.
    """
    try:
        return await model.ainvoke(messages)
    except Exception as first:
        logger.warning(
            "LLM %s failed (%s: %s); retrying once; payload=%s",
            label,
            type(first).__name__,
            first,
            _payload_shape(messages),
        )
        await asyncio.sleep(LLM_RETRY_BACKOFF_S)
        try:
            return await model.ainvoke(messages)
        except Exception as second:
            logger.warning(
                "LLM %s retry failed (%s: %s) — ending turn llm_error; payload=%s",
                label,
                type(second).__name__,
                second,
                _payload_shape(messages),
            )
            return None


class Brain:
    def __init__(
        self,
        pool: asyncpg.Pool,
        redis_client: redis.Redis,
        *,
        small_model: Any | None = None,
        big_model: Any | None = None,
        small_model_name: str | None = None,
        big_model_name: str | None = None,
        checkpointer: Any | None = None,
        on_committed: Callable[[MessageRow], Awaitable[None]] | None = None,
    ) -> None:
        self.small_model_name = policy.assert_triage_model(
            policy.small_model_name() if small_model_name is None else small_model_name
        )
        self.big_model_name = (
            policy.big_model_name() if big_model_name is None else big_model_name
        )
        self.pool = pool
        self.redis = redis_client
        self.small = small_model or ChatOpenAI(model=self.small_model_name)
        self.big = big_model or ChatOpenAI(model=self.big_model_name)
        # InMemorySaver: langgraph-checkpoint-postgres wants psycopg/libpq;
        # this repo is asyncpg-only. HOLD is an in-graph loop, so a durable
        # saver is not required for correctness. Swap via this arg later.
        self.checkpointer = MemorySaver() if checkpointer is None else checkpointer
        self.on_committed = on_committed
        self.graph = self._compile()

    def _compile(self) -> Any:
        graph = StateGraph(BrainState)
        graph.add_node("triage", self._triage)
        graph.add_node("tool_loop", self._tool_loop)
        graph.add_node("freshness", self._freshness)
        graph.add_node("commit", self._commit)
        graph.add_edge(START, "triage")
        graph.add_conditional_edges("triage", self._after_triage)
        graph.add_conditional_edges("tool_loop", self._after_tools)
        graph.add_conditional_edges("freshness", self._after_fresh)
        graph.add_conditional_edges("commit", self._after_commit)
        return graph.compile(checkpointer=self.checkpointer)

    def _after_triage(self, state: BrainState) -> str:
        if state.get("outcome") in {"skipped", "llm_error"}:
            return END
        if not state.get("triage_actionable"):
            return END
        return "tool_loop"

    def _after_tools(self, state: BrainState) -> str:
        if state.get("outcome") == "llm_error":
            return END
        if state.get("pending_reply"):
            return "freshness"
        if state.get("outcome"):
            return END
        return "tool_loop"

    def _after_fresh(self, state: BrainState) -> str:
        if state.get("outcome") == "held_exhausted":
            return END
        if state.get("pending_reply"):
            return "commit"
        return "tool_loop"

    def _after_commit(self, state: BrainState) -> str:
        if state.get("outcome") in {"held_exhausted", "replied"}:
            return END
        return "tool_loop"

    async def _ledger(self, state: BrainState, purpose: str, message: Any) -> None:
        prompt_tokens, completion_tokens = token_usage(message)
        model = self.small_model_name if purpose == "triage" else self.big_model_name
        await db.insert_llm_call(
            self.pool,
            UUID(state["agent_id"]),
            UUID(state["room_id"]),
            model,
            prompt_tokens,
            completion_tokens,
            purpose,
        )

    async def _triage(self, state: BrainState) -> dict[str, Any]:
        prompt = (
            f"You are {state['agent_name']}. Persona: {state.get('persona') or 'none'}.\n\n"
            f"New messages since you last read:\n{_format_inbox(state.get('inbox') or [])}\n\n"
            "Decide whether you should act. Reply with a single JSON object, no markdown:\n"
            '{"actionable": bool, "reason": str, "response_mode": "me"|"each"|"one-of-us"}\n'
            "me = addressed to you; each = everyone should respond; "
            "one-of-us = exactly one agent should respond.\n"
            "If another participant already completed the request, actionable=false."
        )
        ai = await invoke_model(
            self.small, [SystemMessage(content=prompt)], label="triage"
        )
        if ai is None:
            return {
                "outcome": "llm_error",
                "triage_actionable": False,
                "triage_reason": "llm_error",
                "response_mode": "",
            }
        await self._ledger(state, "triage", ai)
        try:
            verdict = _parse_triage(_text(getattr(ai, "content", "")))
        except Exception:
            logger.warning("triage output was not valid JSON — skipping", exc_info=True)
            return {
                "outcome": "skipped",
                "triage_actionable": False,
                "triage_reason": "invalid triage output",
                "response_mode": "",
            }
        logger.info(
            "triage %s actionable=%s mode=%s (%s)",
            state["agent_name"],
            verdict.actionable,
            verdict.response_mode,
            verdict.reason,
        )
        if not verdict.actionable:
            return {
                "outcome": "skipped",
                "triage_actionable": False,
                "triage_reason": verdict.reason,
                "response_mode": verdict.response_mode,
            }
        return {
            "triage_actionable": True,
            "triage_reason": verdict.reason,
            "response_mode": verdict.response_mode,
        }

    async def _tool_loop(self, state: BrainState) -> dict[str, Any]:
        hop = int(state.get("hop_count") or 0)
        if hop >= MAX_HOPS:
            return {"outcome": "hop_exhausted"}

        history = list(state.get("messages") or [])
        if not history:
            history = [SystemMessage(content=_tool_prompt(state))]

        bound = self.big.bind_tools(TOOLS) if hasattr(self.big, "bind_tools") else self.big
        ai = await invoke_model(bound, history, label="tool_loop")
        if ai is None:
            return {"outcome": "llm_error"}
        await self._ledger(state, "turn", ai)
        hop += 1

        after: list[BaseMessage] = [*history, ai]
        claims = list(state.get("claims") or [])
        pending = ""
        for tc in getattr(ai, "tool_calls", None) or []:
            name = _tc_get(tc, "name")
            args = _tc_get(tc, "args") or {}
            call_id = str(_tc_get(tc, "id") or "call")
            if name == "claim":
                raw_key = str(args.get("task_key") or "")
                task_key = canonical_task_key(raw_key)
                if task_key is None:
                    after.append(
                        ToolMessage(content=CLAIM_KEY_ERROR, tool_call_id=call_id)
                    )
                    logger.info(
                        "claim %s rejected key %r", state["agent_name"], raw_key
                    )
                    continue
                won = await db.try_claim(
                    self.pool,
                    UUID(state["room_id"]),
                    task_key,
                    UUID(state["agent_id"]),
                )
                result = "won" if won else "lost"
                claims.append({"task_key": task_key, "result": result})
                after.append(ToolMessage(content=result, tool_call_id=call_id))
                logger.info("claim %s %s -> %s", state["agent_name"], task_key, result)
            elif name == "reply":
                body = str(args.get("body") or "")
                if body:
                    pending = body
                    after.append(ToolMessage(content="queued", tool_call_id=call_id))
                else:
                    after.append(ToolMessage(content="empty body ignored", tool_call_id=call_id))
            else:
                after.append(ToolMessage(content=f"unknown tool {name}", tool_call_id=call_id))

        update: dict[str, Any] = {
            "messages": after,
            "hop_count": hop,
            "claims": claims,
            "pending_reply": pending,
        }
        if pending:
            return update
        if not getattr(ai, "tool_calls", None) or hop >= MAX_HOPS:
            update["outcome"] = "skipped" if hop < MAX_HOPS else "hop_exhausted"
        return update

    async def _hold(self, state: BrainState, latest: int) -> dict[str, Any]:
        seen = int(state["seen_seq"])
        hold_count = int(state.get("hold_count") or 0)
        if hold_count >= MAX_HOLDS:
            logger.info(
                "HOLD exhausted for %s (seen=%s latest=%s)",
                state["agent_name"],
                seen,
                latest,
            )
            return {"outcome": "held_exhausted", "pending_reply": ""}

        room_id = UUID(state["room_id"])
        newer = await db.list_messages(self.pool, room_id, since_seq=seen)
        names = dict(state.get("author_names") or {})
        lines: list[str] = []
        extra: list[InboxItem] = []
        for message in newer:
            name = names.get(str(message.author_id))
            if name is None:
                author = await db.get_participant(self.pool, message.author_id)
                name = author.name
                names[str(message.author_id)] = name
            lines.append(f"[seq={message.seq} {name}] {message.body}")
            extra.append(
                {
                    "seq": message.seq,
                    "body": message.body,
                    "author_id": str(message.author_id),
                    "author_name": name,
                }
            )

        # User role, not system: some OpenAI-compatible backends reject a
        # system message that is not at position 0 (mid-conversation HOLD).
        note = HumanMessage(
            content="New messages landed while you were composing; re-decide.\n"
            + "\n".join(lines)
        )
        await record_seen(self.redis, UUID(state["agent_id"]), room_id, latest)
        logger.info(
            "HOLD %s seen=%s latest=%s hold=%s",
            state["agent_name"],
            seen,
            latest,
            hold_count + 1,
        )
        return {
            "hold_count": hold_count + 1,
            "hop_count": 0,
            "pending_reply": "",
            "seen_seq": latest,
            "author_names": names,
            "inbox": [*(state.get("inbox") or []), *extra],
            "messages": [*(state.get("messages") or []), note],
        }

    async def _freshness(self, state: BrainState) -> dict[str, Any]:
        # Cheap first check. The transactional insert is the invariant.
        room_id = UUID(state["room_id"])
        latest = await db.get_room_last_seq(self.pool, room_id)
        seen = int(state["seen_seq"])
        if latest <= seen:
            return {}
        return await self._hold(state, latest)

    async def _commit(self, state: BrainState) -> dict[str, Any]:
        body = state.get("pending_reply") or ""
        try:
            row = await db.insert_message(
                self.pool,
                UUID(state["room_id"]),
                UUID(state["agent_id"]),
                body,
                not_after_seq=int(state["seen_seq"]),
            )
        except db.StaleWriteError as exc:
            return await self._hold(state, exc.last_seq)
        if self.on_committed is not None:
            await self.on_committed(row)
        logger.info("commit %s seq=%s", state["agent_name"], row.seq)
        return {"outcome": "replied"}

    async def run(self, agent_id: UUID, room_id: UUID) -> TurnResult:
        agent = await db.get_participant(self.pool, agent_id)
        since = await db.get_last_read(self.pool, agent_id, room_id)
        rows = await db.list_messages(self.pool, room_id, since_seq=since)
        participants = await db.list_participants(self.pool, room_id)
        names = {str(p.id): p.name for p in participants}
        inbox: list[InboxItem] = [
            {
                "seq": message.seq,
                "body": message.body,
                "author_id": str(message.author_id),
                "author_name": names.get(str(message.author_id), "?"),
            }
            for message in rows
        ]
        seen_seq = max((message.seq for message in rows), default=since)
        if not inbox:
            return TurnResult(
                agent_id=agent_id,
                agent_name=agent.name,
                room_id=room_id,
                outcome="empty",
                hold_count=0,
                hop_count=0,
                seen_seq=seen_seq,
                inbox_count=0,
                triage_actionable=None,
                triage_reason=None,
                response_mode=None,
                claims=(),
                reply_body=None,
            )

        # Inbox is about to be shown to the model — high-water goes to Redis.
        await record_seen(self.redis, agent_id, room_id, seen_seq)
        initial: BrainState = {
            "agent_id": str(agent_id),
            "agent_name": agent.name,
            "persona": agent.persona or "",
            "room_id": str(room_id),
            "inbox": inbox,
            "author_names": names,
            "seen_seq": seen_seq,
            "triage_actionable": False,
            "triage_reason": "",
            "response_mode": "",
            "messages": [],
            "hold_count": 0,
            "hop_count": 0,
            "pending_reply": "",
            "outcome": "",
            "claims": [],
        }
        final = await self.graph.ainvoke(
            initial,
            {
                "configurable": {"thread_id": str(uuid4())},
                "recursion_limit": 40,
            },
        )
        last_read = int(final.get("seen_seq") or seen_seq)
        await db.set_last_read(self.pool, agent_id, room_id, last_read)
        claims = tuple(
            (item["task_key"], item["result"]) for item in (final.get("claims") or [])
        )
        outcome = str(final.get("outcome") or "skipped")
        reply = final.get("pending_reply") or None
        if outcome != "replied":
            reply = None
        return TurnResult(
            agent_id=agent_id,
            agent_name=agent.name,
            room_id=room_id,
            outcome=outcome,
            hold_count=int(final.get("hold_count") or 0),
            hop_count=int(final.get("hop_count") or 0),
            seen_seq=last_read,
            inbox_count=len(inbox),
            triage_actionable=bool(final.get("triage_actionable")),
            triage_reason=final.get("triage_reason") or None,
            response_mode=final.get("response_mode") or None,
            claims=claims,
            reply_body=reply,
        )


def build_brain(
    pool: asyncpg.Pool,
    redis_client: redis.Redis,
    **kwargs: Any,
) -> Brain:
    return Brain(pool, redis_client, **kwargs)


def make_turn_fn(
    pool: asyncpg.Pool,
    redis_client: redis.Redis,
    **kwargs: Any,
) -> Callable[[UUID, UUID], Awaitable[TurnResult]]:
    brain = Brain(pool, redis_client, **kwargs)
    return brain.run
