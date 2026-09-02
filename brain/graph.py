"""One LangGraph: triage → tool_loop → freshness HOLD → commit."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal, Self, TypedDict
from uuid import UUID, uuid4

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
from pydantic import BaseModel, model_validator

from brain import policy
from brain.holds import clear_hold, consume_hold, record_hold
from brain.world import DecisionResult, DuplicateReply, StaleWrite, World, WorldMessage

logger = logging.getLogger("agora.brain")

MAX_HOPS = 6
MAX_HOLDS = 2
LLM_RETRY_BACKOFF_S = 1.0
# One extra hop after a won-claim obligation nudge. Bounded and obvious:
# the winner gets exactly one more chance to reply before we release.
CLAIM_OBLIGATION_HOPS = 1
# Agent↔agent loop cap: an agent-only run that has circled this many full
# rounds (x agent count) without a human is stale by count. The small
# model is still the primary wind-down; this is the deterministic floor.
AGENT_LOOP_CAP = 4
# One say per moderator turn. HOLD resets hop_count, so hop budget
# cannot bound repeated says; the verbatim-dup gate also ignores
# same-author repeats. A second say is a ToolMessage, not a post.
MODERATOR_SAY_BUDGET = 1
# Moderated-room analogue of AGENT_LOOP_CAP: bounds "moderator polls
# everyone" at the decision layer, before the message-level loop cap
# has to fire. A call_on counts when its trigger_seq is after the last
# human message, including the decision that answers that message
# (trigger_seq > last_human_seq - 1; floor 0 if no human has spoken).
# Mentions (@Name) are not call_ons and do not count.
MODERATED_CALLS_PER_HUMAN = 3
# Messages of recent room history shown to triage on a PROACTIVE turn
# (stall nudge: the inbox itself is empty — the agent already read
# everything). A bounded tail, not the whole room, keeps the prompt small.
INBOX_TAIL = 10

# Protocol parse of the model's claim key — not content classification.
# Free-form names never converge across two models; seq is objective.
_TASK_KEY = re.compile(r"^t(\d+)(?::[A-Za-z0-9_-]+)?$")
CLAIM_KEY_ERROR = (
    "claim rejected: task_key must be t<seq> or t<seq>:<slug> "
    "(e.g. t1 or t1:intro), where <seq> is the seq of the message "
    "you are responding to. Retry with that format."
)

DUPLICATE_REPLY_ERROR = (
    "reply rejected: it verbatim-duplicates the latest peer message. "
    "Do not restate a peer. Either say something materially different "
    "or stay silent."
)

DECIDE_TARGET_ERROR = "unknown or non-callable target"

DECIDE_CALLS_EXHAUSTED = (
    f"decide rejected: call_on budget exhausted this human turn "
    f"({MODERATED_CALLS_PER_HUMAN} call_ons since the last human message). "
    "The round is over; only silence (or a single say) remains."
)

MODERATOR_SAY_ERROR = (
    "decide rejected: say already used this turn. "
    "call_on a member or silence."
)

MODERATION_NOTE = (
    "A moderation turn must call the decide tool exactly once. "
    "Free text is not a valid decision."
)


def claim_obligation_text(task_key: str) -> str:
    return (
        f"You won claim {task_key}; you must reply now or the claim will be released."
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
    response_mode: Literal["me", "each", "one-of-us"] | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_response_mode(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if data.get("actionable"):
            return data
        out = dict(data)
        out["response_mode"] = None
        return out

    @model_validator(mode="after")
    def _require_mode_when_actionable(self) -> Self:
        if self.actionable and self.response_mode is None:
            raise ValueError("response_mode is required when actionable")
        return self


class InboxItem(TypedDict):
    seq: int
    body: str
    author_id: str
    author_name: str


class ClaimRecord(TypedDict):
    task_key: str
    result: str


class RosterEntry(TypedDict):
    id: str
    name: str
    kind: str
    role: str


class BrainState(TypedDict, total=False):
    agent_id: str
    agent_name: str
    persona: str
    room_id: str
    inbox: list[InboxItem]
    author_names: dict[str, str]
    agent_count: int
    agent_only_stretch: int
    proactive: bool
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
    claim_nudged: bool
    send_anyway: bool
    room_mode: str
    agent_role: str
    roster: list[RosterEntry]
    moderate: bool
    called_on: bool
    moderation_nudged: bool
    moderator_says: int
    said_body: str


@tool
def reply(body: str, send_anyway: bool = False) -> str:
    """Post your message to the room.

    send_anyway acknowledges the freshness HOLD this turn showed you
    (the HELD reply armed a token). It never skips the gate: if the
    room moved past the acknowledged state, the ack is void and a
    fresh HOLD shows you the truly-new rows for a re-decide. Without a
    token the flag does nothing and the gate still runs.
    """
    return body


@tool
def claim(task_key: str) -> str:
    """Atomically claim a task. task_key must be t<seq> or t<seq>:<slug>."""
    return task_key


@tool
def decide(
    action: Literal["call_on", "say", "silence"],
    target: str = "",
    body: str = "",
) -> str:
    """Return a moderation decision.

    call_on: target is an agent member's name (never the human); ends
    the turn. After the called-on member has answered the human, prefer
    silence over another call_on.
    say: body is posted through the ordinary reply path and does not
    end the turn — follow with call_on or silence. At most once, and
    only when the table needs your voice.
    silence: end without calling on anyone. The default once the ask
    has been answered.
    """
    return action


TOOLS = [reply, claim]
MODERATOR_TOOLS = [decide]


@dataclass(frozen=True)
class TurnResult:
    """Result of one brain turn.

    `outcome` is why the turn ended. A moderator turn that posted a
    `say` mid-turn still reports the *final* decision (`moderated_call`
    / `moderated_silence` / ...) and puts the said text in `reply_body`
    (or None if the moderator never said). A member who did not reply
    also has `reply_body` None. A member reply keeps
    `outcome="replied"` and `reply_body` the post.
    """

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


def _hold_guidance(response_mode: str) -> str:
    if response_mode == "each":
        return (
            "a peer's reply does not release you; if you have not yet "
            "done the task yourself, you still must do it.\n"
        )
    if response_mode == "one-of-us":
        return "if a peer already completed the request, stay silent.\n"
    if response_mode == "me":
        return "the request was addressed to you; new messages rarely change that.\n"
    return ""


def _format_inbox(items: list[InboxItem]) -> str:
    if not items:
        return "(empty)"
    return "\n".join(f"[seq={m['seq']} {m['author_name']}] {m['body']}" for m in items)


def _tool_prompt(state: BrainState) -> str:
    called = ""
    if state.get("called_on"):
        called = (
            "The moderator called on you to respond to the room. "
            "response_mode=me.\n\n"
        )
    return (
        f"You are {state['agent_name']}. {state.get('persona') or ''}\n\n"
        f"{called}"
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


def _moderator_prompt(state: BrainState) -> str:
    # Shape-level: who you are and what one decide call is. No scenario
    # list — the model picks the seat; code only parses the name.
    roster = state.get("roster") or []
    seats = ", ".join(
        f"{p['name']} ({p['kind']}, {p['role']})" for p in roster
    )
    return (
        f"You are {state['agent_name']}, the moderator of this room. "
        f"{state.get('persona') or ''}\n\n"
        "You never answer substantive questions yourself. "
        "say posts one short line and does not end the turn; "
        "then you must call_on or silence. A second say is rejected.\n"
        "call_on only an agent in the roster (never the human), and "
        "only when the ask is still unanswered or the human asked for "
        "a discussion.\n"
        "After the member you called on has answered the human, choose "
        "silence — do not thank, do not summarize, do not call on "
        "someone else unless the answer was clearly incomplete.\n"
        "A line like \"<Name> passes.\" means that member declined: the "
        "ask is still unanswered, so call_on another agent who can "
        "answer it, or silence only if nobody in the roster can.\n"
        "say at most once and only when the table needs your voice.\n"
        "Free text without a decide call is not a valid moderation turn.\n\n"
        f"Seats: {seats}\n\n"
        f"Room:\n{_format_inbox(state.get('inbox') or [])}\n"
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


def _is_rate_limited(exc: BaseException) -> bool:
    """Best-effort detection of provider throttling across SDK shapes.

    OpenAI SDK: RateLimitError (subclass of APIStatusError, status 429);
    httpx-style payloads put "rate limit" in the message. Unrecognized
    errors are treated as ordinary failures — pacing degrades to nothing.
    """
    cls = type(exc)
    if cls.__name__ == "RateLimitError":
        return True
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    text = str(exc).lower()
    return "rate limit" in text or "429" in text


async def invoke_model(
    model: Any,
    messages: list[Any],
    *,
    label: str,
    pacer: Any | None = None,
    limiter: Any | None = None,
) -> Any | None:
    """Call the model; retry once after a short backoff; None if both fail.

    When a pacer is given, every call start first waits for its slot; a
    rate-limited response doubles the global interval, clean calls restore
    it. When a limiter is given, the call runs inside a concurrency slot —
    a per-computer cap shared by BOTH model classes (Cumora §2/§3a: the
    pacer spaces call starts, the semaphore bounds calls in flight; the
    retry stays inside the same slot, it is the same logical call).
    Fail-open: a dead relay is a missed reply, not a crashed turn. No
    caller should write a ledger row unless this returns a message.
    """
    if limiter is not None:
        async with limiter.slot():
            return await _invoke_with_pacer(model, messages, label=label, pacer=pacer)
    return await _invoke_with_pacer(model, messages, label=label, pacer=pacer)


async def _invoke_with_pacer(
    model: Any,
    messages: list[Any],
    *,
    label: str,
    pacer: Any | None,
) -> Any | None:
    if pacer is not None:
        waited = await pacer.wait_turn()
        if waited:
            logger.info("pacer %s waited %.1fs", label, waited)
    try:
        result = await model.ainvoke(messages)
        if pacer is not None:
            pacer.on_ok()
        return result
    except Exception as first:
        if pacer is not None and _is_rate_limited(first):
            logger.warning(
                "LLM %s rate-limited — pacer interval now %.1fs",
                label,
                pacer.on_rate_limited(),
            )
        logger.warning(
            "LLM %s failed (%s: %s); retrying once; payload=%s",
            label,
            type(first).__name__,
            first,
            _payload_shape(messages),
        )
        await asyncio.sleep(LLM_RETRY_BACKOFF_S)
        if pacer is not None:
            await pacer.wait_turn()
        try:
            result = await model.ainvoke(messages)
            if pacer is not None:
                pacer.on_ok()
            return result
        except Exception as second:
            if pacer is not None and _is_rate_limited(second):
                logger.warning(
                    "LLM %s rate-limited on retry — pacer interval now %.1fs",
                    label,
                    pacer.on_rate_limited(),
                )
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
        world: World,
        *,
        small_model: Any | None = None,
        big_model: Any | None = None,
        small_model_name: str | None = None,
        big_model_name: str | None = None,
        checkpointer: Any | None = None,
        on_committed: Callable[[WorldMessage], Awaitable[None]] | None = None,
        hold_redis: Any | None = None,
        pacer: Any | None = None,
        limiter: Any | None = None,
    ) -> None:
        self.small_model_name = policy.assert_triage_model(
            policy.small_model_name() if small_model_name is None else small_model_name
        )
        self.big_model_name = (
            policy.big_model_name() if big_model_name is None else big_model_name
        )
        self.world = world
        self.small = small_model or ChatOpenAI(model=self.small_model_name)
        self.big = big_model or ChatOpenAI(model=self.big_model_name)
        # Hold tokens live in Redis (server-side). The daemon constructs its
        # Brain without one; its freshness gate is the runtime 409 path.
        self._hold_redis = hold_redis
        # Optional AdaptivePacer (daemon-side): deterministic spacing between
        # LLM call starts plus exponential backoff on provider rate limits.
        self._pacer = pacer
        # Optional ConcurrencyLimiter (daemon-side): per-computer cap on
        # model calls in flight, shared by both model classes.
        self._limiter = limiter
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
        # Wiring, not content classification: the moderator IS the gate,
        # and a call_on is a protocol fact on the wake. Both skip triage.
        graph.add_conditional_edges(
            START,
            self._from_start,
            {"triage": "triage", "tool_loop": "tool_loop"},
        )
        graph.add_conditional_edges("triage", self._after_triage)
        graph.add_conditional_edges("tool_loop", self._after_tools)
        graph.add_conditional_edges("freshness", self._after_fresh)
        graph.add_conditional_edges("commit", self._after_commit)
        return graph.compile(checkpointer=self.checkpointer)

    def _from_start(self, state: BrainState) -> str:
        if state.get("moderate") or state.get("called_on"):
            return "tool_loop"
        return "triage"

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
        if state.get("outcome") == "held_exhausted":
            return END
        # A member reply ends the turn. A moderator say does not: the
        # same turn continues until call_on or silence.
        if state.get("outcome") == "replied" and not state.get("moderate"):
            return END
        return "tool_loop"

    async def _ledger(self, state: BrainState, purpose: str, message: Any) -> None:
        prompt_tokens, completion_tokens = token_usage(message)
        model = self.small_model_name if purpose == "triage" else self.big_model_name
        if purpose == "moderate":
            policy.assert_moderate_uses_big(self.big_model_name)
        await self.world.record_llm_call(
            UUID(state["agent_id"]),
            UUID(state["room_id"]),
            model,
            prompt_tokens,
            completion_tokens,
            purpose,
        )

    async def _triage(self, state: BrainState) -> dict[str, Any]:
        # Agent↔agent loop cap (deterministic backstop under the model
        # gate): an agent-only stretch past the cap is stale by COUNT, not
        # by wording — the one class of triage that is arithmetic, not
        # classification (Cumora §6 loop floors; this repo's rule that a
        # non-model short-circuit must be counting, never content). The
        # count is ROOM-level — every agent message since the last human
        # message, across turns — supplied by the World in TurnContext.
        # Counting the per-turn inbox instead mis-scopes both directions:
        # quick ping-pong replies each stay under one inbox batch yet the
        # room is circling, and a single huge coalesced burst trips the
        # cap on a room that was fine one message ago. No humans at all
        # is the MOST loop-prone shape, so the cap arms hardest there;
        # any human's message resets the counter at the source.
        if state.get("proactive"):
            framing = (
                "The room has gone quiet after the messages below (you have "
                "read them all). Someone may still be owed a reply — the task "
                "may be unfinished. Decide whether to speak now; staying "
                "silent is a valid, common answer when nothing is owed."
            )
        else:
            framing = "Decide whether you should act."
        prompt = (
            f"You are {state['agent_name']}. Persona: {state.get('persona') or 'none'}.\n\n"
            f"{framing}\n"
            f"Recent room messages:\n"
            f"{_format_inbox(state.get('inbox') or [])}\n\n"
            "Reply with a single JSON object, no markdown:\n"
            '{"actionable": bool, "reason": str, "response_mode": "me"|"each"|"one-of-us"}\n'
            "Omit response_mode when actionable is false.\n"
            "me = addressed to you; new messages from others rarely mean you should skip.\n"
            "each = everyone should respond; a peer already speaking does not "
            "complete the request for you.\n"
            "one-of-us = exactly one agent should respond; if a peer already "
            "completed the request, actionable=false."
        )
        ai = await invoke_model(
            self.small,
            [SystemMessage(content=prompt)],
            label="triage",
            pacer=self._pacer,
            limiter=self._limiter,
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

    def _hop_budget(self, state: BrainState) -> int:
        extra = CLAIM_OBLIGATION_HOPS if state.get("claim_nudged") else 0
        return MAX_HOPS + extra

    def _won_keys(self, claims: list[ClaimRecord]) -> list[str]:
        return [c["task_key"] for c in claims if c.get("result") == "won"]

    def _claim_nudge(
        self,
        state: BrainState,
        after: list[BaseMessage],
        claims: list[ClaimRecord],
        hop: int,
    ) -> dict[str, Any] | None:
        """One protocol reminder if a winner is about to leave without replying."""
        if state.get("claim_nudged"):
            return None
        won = self._won_keys(claims)
        if not won:
            return None
        key = won[-1]
        note = HumanMessage(content=claim_obligation_text(key))
        logger.info("claim obligation %s %s — one more hop", state["agent_name"], key)
        return {
            "messages": [*after, note],
            "hop_count": hop,
            "claims": claims,
            "pending_reply": "",
            "claim_nudged": True,
        }

    def _callable_member_names(self, state: BrainState) -> list[str]:
        me = state["agent_id"]
        return [
            person["name"]
            for person in state.get("roster") or []
            if (
                person.get("kind") == "agent"
                and person.get("role") == "member"
                and person.get("id") != me
            )
        ]

    def _decide_target_error(self, state: BrainState, name: str) -> str:
        seats = ", ".join(self._callable_member_names(state)) or "(none)"
        return (
            f"{DECIDE_TARGET_ERROR} {name!r}; "
            f"call_on accepts one of: {seats} "
            f"(agents only — humans cannot be called on)"
        )

    def _resolve_call_on_target(self, state: BrainState, name: str) -> UUID | None:
        me = state["agent_id"]
        for person in state.get("roster") or []:
            if (
                person["name"] == name
                and person["kind"] == "agent"
                and person["id"] != me
            ):
                return UUID(person["id"])
        return None

    async def _apply_decision(
        self,
        state: BrainState,
        action: str,
        target_name: str,
        body: str,
        call_id: str,
        after: list[BaseMessage],
        hop: int,
    ) -> dict[str, Any]:
        trigger_seq = int(state["seen_seq"])
        room_id = UUID(state["room_id"])
        moderator_id = UUID(state["agent_id"])
        target_id: UUID | None = None
        if action == "call_on":
            target_id = self._resolve_call_on_target(state, target_name)
            if target_id is None:
                after.append(
                    ToolMessage(
                        content=self._decide_target_error(state, target_name),
                        tool_call_id=call_id,
                    )
                )
                logger.info(
                    "decide %s rejected target %r", state["agent_name"], target_name
                )
                return {
                    "messages": after,
                    "hop_count": hop,
                    "pending_reply": "",
                }
            used = await self.world.call_ons_since_human(room_id)
            if used >= MODERATED_CALLS_PER_HUMAN:
                after.append(
                    ToolMessage(content=DECIDE_CALLS_EXHAUSTED, tool_call_id=call_id)
                )
                logger.info(
                    "decide %s call_on exhausted (%s >= %s)",
                    state["agent_name"],
                    used,
                    MODERATED_CALLS_PER_HUMAN,
                )
                return {
                    "messages": after,
                    "hop_count": hop,
                    "pending_reply": "",
                }
        elif action == "say" and not body.strip():
            after.append(
                ToolMessage(content="empty body ignored", tool_call_id=call_id)
            )
            return {
                "messages": after,
                "hop_count": hop,
                "pending_reply": "",
            }
        elif (
            action == "say"
            and int(state.get("moderator_says") or 0) >= MODERATOR_SAY_BUDGET
        ):
            after.append(
                ToolMessage(content=MODERATOR_SAY_ERROR, tool_call_id=call_id)
            )
            return {
                "messages": after,
                "hop_count": hop,
                "pending_reply": "",
            }
        elif action not in {"call_on", "say", "silence"}:
            after.append(
                ToolMessage(content=f"unknown action {action}", tool_call_id=call_id)
            )
            return {
                "messages": after,
                "hop_count": hop,
                "pending_reply": "",
            }

        # call_on is row-first: the unique key is the wake-dedup, and a
        # lost wake is redelivered by cursor arithmetic in _route_wake.
        # say/silence are row-after-side-effect: a crash before the row
        # leaves the trigger open so the next wake re-decides. Freshness
        # and verbatim-dup already block a double post if the message
        # landed. Recording say here would starve the room on replay.
        if action == "say":
            after.append(ToolMessage(content="say", tool_call_id=call_id))
            return {
                "messages": after,
                "hop_count": hop,
                "pending_reply": body.strip(),
                "send_anyway": False,
            }

        recorded: DecisionResult = await self.world.record_decision(
            room_id, moderator_id, trigger_seq, action, target_id
        )
        if recorded.status == "already_decided":
            after.append(ToolMessage(content="already_decided", tool_call_id=call_id))
            logger.info(
                "decide %s trigger=%s replayed action=%s",
                state["agent_name"],
                trigger_seq,
                recorded.action,
            )
            return {
                "messages": after,
                "hop_count": hop,
                "pending_reply": "",
                "outcome": "decision_replayed",
            }

        after.append(ToolMessage(content=recorded.action, tool_call_id=call_id))
        if action == "call_on":
            return {
                "messages": after,
                "hop_count": hop,
                "pending_reply": "",
                "outcome": "moderated_call",
            }
        return {
            "messages": after,
            "hop_count": hop,
            "pending_reply": "",
            "outcome": "moderated_silence",
        }

    async def _tool_loop(self, state: BrainState) -> dict[str, Any]:
        hop = int(state.get("hop_count") or 0)
        budget = self._hop_budget(state)
        if hop >= budget:
            return {"outcome": "hop_exhausted"}

        moderate = bool(state.get("moderate"))
        history = list(state.get("messages") or [])
        if not history:
            prompt = _moderator_prompt(state) if moderate else _tool_prompt(state)
            history = [SystemMessage(content=prompt)]

        tools = MODERATOR_TOOLS if moderate else TOOLS
        if moderate:
            # Decide is a tool call on the big model — never the triage wiring.
            policy.assert_moderate_uses_big(self.big_model_name)
        bound = self.big.bind_tools(tools) if hasattr(self.big, "bind_tools") else self.big
        ai = await invoke_model(
            bound, history, label="tool_loop", pacer=self._pacer, limiter=self._limiter
        )
        if ai is None:
            return {"outcome": "llm_error"}
        await self._ledger(state, "moderate" if moderate else "turn", ai)
        hop += 1

        after: list[BaseMessage] = [*history, ai]
        claims = list(state.get("claims") or [])
        pending = ""
        send_anyway = False
        calls = getattr(ai, "tool_calls", None) or []

        if moderate:
            decide_calls = [tc for tc in calls if _tc_get(tc, "name") == "decide"]
            chosen_id = (
                str(_tc_get(decide_calls[0], "id") or "call") if decide_calls else None
            )
            # Every unanswered tool_call must get a ToolMessage before any
            # HumanMessage. OpenAI-compatible backends 400 on a corrective
            # note that skips the tool-result turn (the member loop already
            # does this; the same rule applies to extra calls beside a
            # valid decide — say can re-enter via HOLD).
            for tc in calls:
                name = str(_tc_get(tc, "name") or "")
                call_id = str(_tc_get(tc, "id") or "call")
                if chosen_id is not None and call_id == chosen_id:
                    continue
                if name == "decide":
                    after.append(
                        ToolMessage(
                            content="only one decide per turn", tool_call_id=call_id
                        )
                    )
                else:
                    after.append(
                        ToolMessage(
                            content=f"unknown tool {name}", tool_call_id=call_id
                        )
                    )
            if not decide_calls:
                if not state.get("moderation_nudged") and hop < budget:
                    after.append(HumanMessage(content=MODERATION_NOTE))
                    return {
                        "messages": after,
                        "hop_count": hop,
                        "moderation_nudged": True,
                        "pending_reply": "",
                    }
                return {
                    "messages": after,
                    "hop_count": hop,
                    "pending_reply": "",
                    "outcome": "invalid_moderation",
                }
            tc = decide_calls[0]
            args = _tc_get(tc, "args") or {}
            return await self._apply_decision(
                state,
                str(args.get("action") or ""),
                str(args.get("target") or ""),
                str(args.get("body") or ""),
                str(_tc_get(tc, "id") or "call"),
                after,
                hop,
            )

        for tc in calls:
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
                won = await self.world.try_claim(
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
                    send_anyway = send_anyway or bool(_tc_get(args, "send_anyway"))
                    after.append(ToolMessage(content="queued", tool_call_id=call_id))
                else:
                    after.append(ToolMessage(content="empty body ignored", tool_call_id=call_id))
            else:
                after.append(ToolMessage(content=f"unknown tool {name}", tool_call_id=call_id))

        # send_anyway must travel in the returned update: LangGraph copies
        # channel state between nodes, so an in-place write to `state`
        # never reaches `_freshness` (verified by a minimal two-node repro).
        # Writing it every round both sets it on a flagged call and clears
        # it afterwards, so a stale flag cannot leak into the next hop.
        update: dict[str, Any] = {
            "messages": after,
            "hop_count": hop,
            "claims": claims,
            "pending_reply": pending,
            "send_anyway": send_anyway,
        }
        if pending:
            return update
        if not calls or hop >= budget:
            nudged = self._claim_nudge(state, after, claims, hop)
            if nudged is not None:
                return nudged
            update["outcome"] = "skipped" if hop < budget else "hop_exhausted"
        return update

    async def _hold(
        self,
        state: BrainState,
        latest: int,
        newer: list[WorldMessage] | None = None,
    ) -> dict[str, Any]:
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
        landed = newer if newer is not None else await self.world.list_messages_since(
            room_id, seen
        )
        names = dict(state.get("author_names") or {})
        lines: list[str] = []
        extra: list[InboxItem] = []
        for message in landed:
            name = message.author_name or names.get(str(message.author_id), "?")
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
        # Mode line is semantic guidance; the code does not classify bodies.
        note = HumanMessage(
            content=(
                "New messages landed while you were composing; re-decide.\n"
                f"{_hold_guidance(str(state.get('response_mode') or ''))}"
                + "\n".join(lines)
            )
        )
        # Arm the hold token: the HELD envelope showed this agent state up
        # to `latest`, so a later send_anyway THIS TURN may ack exactly that.
        if self._hold_redis is not None:
            await record_hold(self._hold_redis, UUID(state["agent_id"]), room_id, latest)
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
            # max(): the cursor never regresses. In every legitimate path
            # latest > seen already; the max only matters for the defensive
            # foreign-409 shape (unknown last_seq → 0), where regressing
            # seen_seq would re-serve the whole room as inbox next turn.
            "seen_seq": max(seen, latest),
            "author_names": names,
            "inbox": [*(state.get("inbox") or []), *extra],
            "messages": [*(state.get("messages") or []), note],
        }

    async def _freshness(self, state: BrainState) -> dict[str, Any]:
        # Cheap first check. The transactional insert is the invariant.
        room_id = UUID(state["room_id"])
        agent_id = UUID(state["agent_id"])
        latest = await self.world.get_room_last_seq(room_id)
        seen = int(state["seen_seq"])
        if latest <= seen:
            if state.get("send_anyway"):
                # Nothing new since the HELD envelope: the ack spends here.
                # Consuming (rather than leaving the token armed) closes
                # Cumora's double-spend hole — a yielded token must not be
                # reusable by a later turn's preemptive flag.
                acked = await consume_hold(self._hold_redis, agent_id, room_id)
                if acked is not None:
                    logger.info(
                        "send_anyway %s acked held state (token seq=%s, room unchanged)",
                        state["agent_name"],
                        acked,
                    )
                else:
                    logger.info(
                        "send_anyway %s no-op (room unchanged since last read)",
                        state["agent_name"],
                    )
                return {"send_anyway": False}
            return {}
        if state.get("send_anyway") and self._hold_redis is None:
            # This process has no hold-token store (BYOA daemon / K8s job).
            # Deliberate: the runtime 409 freshness path is their gate, and
            # a flag with nothing to ack must never look like a pass.
            logger.info(
                "send_anyway %s ignored (no hold-token store in this process)",
                state["agent_name"],
            )
            return {"send_anyway": False, **await self._hold(state, latest)}
        if state.get("send_anyway"):
            # The room moved past the shown state. Arming and showing are
            # one step in _hold, so any token this process armed acks at
            # most `seen` — the ack is void by CONSTRUCTION here, not by a
            # seq comparison (an acked >= latest accept branch would be
            # dead code). Spend the attempt — the atomic consume is also
            # the crash-recovery cleanup for a token left armed by a turn
            # that died before its end-of-turn clear — then run the fresh
            # HOLD that shows the truly-new rows and re-arms.
            await consume_hold(self._hold_redis, agent_id, room_id)
            logger.info(
                "send_anyway %s void (room at %s, shown state %s) — fresh HOLD",
                state["agent_name"],
                latest,
                seen,
            )
            return {"send_anyway": False, **await self._hold(state, latest)}
        return await self._hold(state, latest)

    async def _commit(self, state: BrainState) -> dict[str, Any]:
        body = state.get("pending_reply") or ""
        try:
            row = await self.world.insert_message(
                UUID(state["room_id"]),
                UUID(state["agent_id"]),
                body,
                not_after_seq=int(state["seen_seq"]),
            )
        except StaleWrite as exc:
            return await self._hold(state, exc.last_seq, newer=exc.newer)
        except DuplicateReply as exc:
            # Non-bypassable gate: the brain is shown the fact and re-decides
            # in the same turn. No holds are spent — this is a semantics
            # error, not a race. No seen_seq sync either: the stale check
            # runs BEFORE the dup check inside the same insert transaction,
            # so at rejection time peer_seq <= seen_seq is an invariant —
            # the cursor can never trail the peer row, and a room that has
            # moved since is correctly routed to the stale/HOLD path.
            logger.info(
                "duplicate reply %s peer seq=%s — re-decide",
                state["agent_name"],
                exc.peer_seq,
            )
            note = HumanMessage(content=DUPLICATE_REPLY_ERROR)
            return {
                "outcome": "",
                "pending_reply": "",
                "messages": [*(state.get("messages") or []), note],
            }
        if self.on_committed is not None:
            await self.on_committed(row)
        if state.get("moderate"):
            # Row after the message: the trigger stays open if we crash
            # between insert and here; the next wake re-decides. seen_seq
            # advances to the say, so a later call_on records at N+1.
            recorded = await self.world.record_decision(
                UUID(state["room_id"]),
                UUID(state["agent_id"]),
                int(state["seen_seq"]),
                "say",
                None,
            )
            if recorded.status == "already_decided":
                logger.info(
                    "say landed but decision row blocked room=%s trigger=%s",
                    state["room_id"],
                    state["seen_seq"],
                )
            if self._hold_redis is not None:
                await clear_hold(
                    self._hold_redis, UUID(state["agent_id"]), UUID(state["room_id"])
                )
            logger.info("commit %s seq=%s", state["agent_name"], row.seq)
            return {
                "outcome": "",
                "pending_reply": "",
                "seen_seq": int(row.seq),
                "said_body": body,
                "moderator_says": int(state.get("moderator_says") or 0) + 1,
            }
        if self._hold_redis is not None:
            await clear_hold(
                self._hold_redis, UUID(state["agent_id"]), UUID(state["room_id"])
            )
        logger.info("commit %s seq=%s", state["agent_name"], row.seq)
        # The cursor must swallow the committed row: the agent authored it,
        # so it has by definition seen it. Left behind, the next turn's
        # inbox (seq > last_read, no author filter) re-serves the agent its
        # own message as new mail — confusing triage and wasting tokens.
        return {"outcome": "replied", "seen_seq": int(row.seq)}

    async def _release_unfulfilled(
        self,
        agent_id: UUID,
        room_id: UUID,
        agent_name: str,
        claims: list[ClaimRecord],
    ) -> None:
        seen: set[str] = set()
        for item in claims:
            if item.get("result") != "won":
                continue
            key = item["task_key"]
            if key in seen:
                continue
            seen.add(key)
            released = await self.world.release_claim(room_id, key, agent_id)
            if released:
                logger.warning(
                    "released unfulfilled claim %s by %s — winner did not reply",
                    key,
                    agent_name,
                )

    async def _post_called_on_pass(
        self,
        agent_id: UUID,
        agent_name: str,
        room_id: UUID,
        seen_seq: int,
    ) -> WorldMessage | None:
        """Land an explicit pass so a silent called-on member advances seq.

        The body includes the author name because the verbatim-dup gate
        compares against the latest other-author message: a bare pass
        from a second member would collide and recreate the deadlock.
        DirectWorld has no server-side fan-out, so a landed pass must
        fire on_committed exactly like _commit; HttpWorld's /runtime/reply
        already fans out. A stale pass (room moved past seen_seq) or a
        verbatim-dup against the latest peer drops silently — the
        moderator is woken by whatever landed.
        """
        try:
            row = await self.world.insert_message(
                room_id,
                agent_id,
                f"{agent_name} passes.",
                not_after_seq=seen_seq,
            )
        except (StaleWrite, DuplicateReply):
            return None
        if self.on_committed is not None:
            await self.on_committed(row)
        logger.info("pass %s seq=%s (called-on decline)", agent_name, row.seq)
        return row

    def _loop_cap_reason(self, stretch: int, agent_count: int) -> str:
        return (
            f"agent-only run past loop cap "
            f"({stretch} agent messages without a human; "
            f"cap {AGENT_LOOP_CAP} x {agent_count} agents)"
        )

    def _empty_result(
        self,
        agent_id: UUID,
        agent_name: str,
        room_id: UUID,
        seen_seq: int,
        *,
        outcome: str,
        inbox_count: int = 0,
        triage_reason: str | None = None,
    ) -> TurnResult:
        return TurnResult(
            agent_id=agent_id,
            agent_name=agent_name,
            room_id=room_id,
            outcome=outcome,
            hold_count=0,
            hop_count=0,
            seen_seq=seen_seq,
            inbox_count=inbox_count,
            triage_actionable=None if outcome == "empty" else False,
            triage_reason=triage_reason,
            response_mode=None,
            claims=(),
            reply_body=None,
        )

    async def run(
        self,
        agent_id: UUID,
        room_id: UUID,
        *,
        called_on: bool = False,
        called_on_seq: int | None = None,
    ) -> TurnResult:
        ctx = await self.world.load_turn(
            agent_id, room_id, called_on_seq=called_on_seq
        )
        names = {str(p.id): p.name for p in ctx.participants}
        inbox: list[InboxItem] = [
            {
                "seq": message.seq,
                "body": message.body,
                "author_id": str(message.author_id),
                "author_name": message.author_name or names.get(str(message.author_id), "?"),
            }
            for message in ctx.inbox
        ]
        seen_seq = ctx.seen_seq
        # A proactive turn (stall nudge) arrives with an empty inbox: the
        # agent has read everything. That is a PROMPT to speak — "the room
        # went quiet with work owed" — not a reason to no-op. The triage
        # model sees the recent room tail and decides whether silence is
        # right; the deterministic loop cap still bounds it. Reactive
        # wakes with a real inbox run the same graph below; only the
        # triage prompt differs.
        proactive = not inbox
        if proactive:
            tail = await self.world.list_messages_since(
                room_id, max(seen_seq - INBOX_TAIL, 0)
            )
            inbox = [
                {
                    "seq": message.seq,
                    "body": message.body,
                    "author_id": str(message.author_id),
                    "author_name": message.author_name
                    or names.get(str(message.author_id), "?"),
                }
                for message in tail
            ]
            if not inbox:
                # Nothing to judge — a silent room with no history (or the
                # agent has no persona to act on). Keep the cheap no-op.
                return TurnResult(
                    agent_id=agent_id,
                    agent_name=ctx.agent.name,
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

        agent_count = sum(1 for p in ctx.participants if p.kind == "agent")
        stretch = int(ctx.agent_only_stretch)
        is_moderator = (
            ctx.room_mode == "moderated" and ctx.agent.role == "moderator"
        )
        trigger = (
            called_on_seq if called_on_seq is not None else ctx.called_on_seq
        )
        called_on = bool(called_on or trigger is not None)
        # Counting, not classification: the same stretch cap as triage,
        # applied here so skip-triage paths (moderator / call_on) still
        # hit the floor without a model call.
        if agent_count > 0 and stretch >= AGENT_LOOP_CAP * agent_count:
            reason = self._loop_cap_reason(stretch, agent_count)
            logger.info(
                "loop cap %s (%s >= %s) — staying silent",
                ctx.agent.name,
                stretch,
                AGENT_LOOP_CAP * agent_count,
            )
            outcome = "moderated_silence" if is_moderator else "skipped"
            if is_moderator:
                await self.world.record_decision(
                    room_id, agent_id, seen_seq, "silence", None
                )
            await self.world.set_last_read(agent_id, room_id, seen_seq)
            if self._hold_redis is not None:
                await clear_hold(self._hold_redis, agent_id, room_id)
            return self._empty_result(
                agent_id,
                ctx.agent.name,
                room_id,
                seen_seq,
                outcome=outcome,
                inbox_count=len(inbox),
                triage_reason=reason,
            )

        roster: list[RosterEntry] = [
            {
                "id": str(p.id),
                "name": p.name,
                "kind": p.kind,
                "role": p.role,
            }
            for p in ctx.participants
        ]
        initial: BrainState = {
            "agent_id": str(agent_id),
            "agent_name": ctx.agent.name,
            "persona": ctx.agent.persona or "",
            "room_id": str(room_id),
            "inbox": inbox,
            "author_names": names,
            "agent_count": agent_count,
            "agent_only_stretch": stretch,
            "proactive": proactive,
            "seen_seq": seen_seq,
            "triage_actionable": True if (is_moderator or called_on) else False,
            "triage_reason": (
                "moderator called on you" if called_on else ""
            ),
            "response_mode": "me" if called_on else "",
            "messages": [],
            "hold_count": 0,
            "hop_count": 0,
            "pending_reply": "",
            "outcome": "",
            "claims": [],
            "claim_nudged": False,
            "send_anyway": False,
            "room_mode": ctx.room_mode,
            "agent_role": ctx.agent.role,
            "roster": roster,
            "moderate": is_moderator,
            "called_on": called_on,
            "moderation_nudged": False,
            "moderator_says": 0,
            "said_body": "",
        }
        final: dict[str, Any]
        try:
            final = await self.graph.ainvoke(
                initial,
                {
                    "configurable": {"thread_id": str(uuid4())},
                    "recursion_limit": 40,
                },
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            # An in-graph crash (GraphRecursionError, a world transport
            # error outside the LLM retry path) must not skip cleanup:
            # a HOLD minted this turn would otherwise survive until its
            # TTL and get spent by a future turn's send_anyway. Claims
            # have their own TTL steal, so only the token is reaped here;
            # the turn surfaces as a raised error, same as before.
            if self._hold_redis is not None:
                await clear_hold(self._hold_redis, agent_id, room_id)
            raise
        last_read = int(final.get("seen_seq") or seen_seq)
        claims = tuple(
            (item["task_key"], item["result"]) for item in (final.get("claims") or [])
        )
        outcome = str(final.get("outcome") or "skipped")
        said = str(final.get("said_body") or "")
        if outcome == "replied":
            reply = final.get("pending_reply") or None
        elif is_moderator:
            reply = said or None
        else:
            reply = None
            await self._release_unfulfilled(
                agent_id, room_id, ctx.agent.name, final.get("claims") or []
            )
        # A called-on llm_error is not a decline: leave last_read behind
        # so _undelivered_call_on redelivers on the next stall nudge.
        # Loop-cap mute returns earlier and never reaches this wrap-up,
        # so it cannot be confused with a model skip.
        if called_on and outcome == "llm_error":
            pass
        else:
            if called_on and outcome in {"skipped", "hop_exhausted"}:
                floor = trigger if trigger is not None else 0
                already = await self.world.has_authored_since(
                    agent_id, room_id, floor
                )
                if already:
                    logger.info(
                        "skip pass %s — already spoken since trigger %s",
                        ctx.agent.name,
                        floor,
                    )
                else:
                    landed = await self._post_called_on_pass(
                        agent_id, ctx.agent.name, room_id, last_read
                    )
                    if landed is not None:
                        last_read = landed.seq
            await self.world.set_last_read(agent_id, room_id, last_read)
        # A token outlives only a COMMITTED send (which cleared it in
        # _commit). Any other ending — skipped, held_exhausted, llm_error,
        # crash — must drop it, or the 2-minute TTL window lets a future
        # turn's preemptive send_anyway spend an acknowledgement that was
        # shown in a different turn (Cumora §5d). It is cleanup, not
        # authority: clearing can fail (Redis down) without affecting the
        # result — a stale token still has to beat the seq check.
        if self._hold_redis is not None:
            await clear_hold(self._hold_redis, agent_id, room_id)
        return TurnResult(
            agent_id=agent_id,
            agent_name=ctx.agent.name,
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


def build_brain(world: World, **kwargs: Any) -> Brain:
    return Brain(world, **kwargs)


def make_turn_fn(
    pool: Any,
    redis_client: Any,
    **kwargs: Any,
) -> Callable[..., Awaitable[TurnResult]]:
    # Lazy import: DirectWorld pulls asyncpg/redis. The daemon constructs
    # Brain(HttpWorld, ...) and must not load the cloud host's drivers.
    from brain.world_direct import DirectWorld

    on_call_on = kwargs.pop("on_call_on", None)
    world = DirectWorld(pool, redis_client, on_call_on=on_call_on)
    brain = Brain(
        world,
        hold_redis=redis_client,
        **kwargs,
    )

    async def run(
        agent_id: UUID,
        room_id: UUID,
        *,
        called_on: bool = False,
        called_on_seq: int | None = None,
    ) -> TurnResult:
        return await brain.run(
            agent_id,
            room_id,
            called_on=called_on,
            called_on_seq=called_on_seq,
        )

    run.world = world  # type: ignore[attr-defined]
    run.brain = brain  # type: ignore[attr-defined]
    return run
