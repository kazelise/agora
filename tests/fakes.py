"""Scripted chat models for Phase 2 tests. No network."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Sequence
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage

Response = AIMessage | Callable[..., AIMessage | Awaitable[AIMessage]]


def usage(prompt: int = 7, completion: int = 3) -> dict[str, int]:
    return {
        "input_tokens": prompt,
        "output_tokens": completion,
        "total_tokens": prompt + completion,
    }


def triage_message(
    *,
    actionable: bool,
    reason: str,
    response_mode: str = "me",
) -> AIMessage:
    return AIMessage(
        content=json.dumps(
            {
                "actionable": actionable,
                "reason": reason,
                "response_mode": response_mode,
            }
        ),
        usage_metadata=usage(),
    )


def tool_call(name: str, args: dict[str, Any], call_id: str | None = None) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": args,
                "id": call_id or f"call_{uuid4().hex[:8]}",
            }
        ],
        usage_metadata=usage(11, 5),
    )


def text_message(content: str) -> AIMessage:
    return AIMessage(content=content, usage_metadata=usage(4, 2))


class ScriptedChatModel:
    """Queue of AIMessages (or callables that produce one). Records every ainvoke."""

    def __init__(self, responses: Sequence[Response] | None = None) -> None:
        self._responses = list(responses or [])
        self.calls: list[list[Any]] = []

    def bind_tools(self, _tools: Any, **_kwargs: Any) -> ScriptedChatModel:
        return self

    def push(self, response: Response) -> None:
        self._responses.append(response)

    async def ainvoke(self, messages: list[Any], **_kwargs: Any) -> AIMessage:
        self.calls.append(list(messages))
        if not self._responses:
            raise AssertionError("unexpected LLM call with no scripted response")
        item = self._responses.pop(0)
        if callable(item):
            result = item(messages)
            if hasattr(result, "__await__"):
                result = await result  # type: ignore[misc]
            return result  # type: ignore[return-value]
        return item
