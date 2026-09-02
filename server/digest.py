"""Room digest: the transcript rendered as a Markdown brief.

Borrowed from yuanzhuo's secretary export: a roundtable is only useful if
the discussion becomes a durable artifact. Agora's equivalent takes the
room transcript, the active claims (the action items — claims exist only
for real shared work), and the llm_calls ledger, and renders one
self-contained Markdown document. Deterministic formatting only: no model
call, no content interpretation — a summarizer model can sit on top later,
but the export itself must never require one.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from server import db
from server.db import CLAIM_TTL_S


def _esc(text: str) -> str:
    return (
        text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ").strip()
    )


def _cell(text: str) -> str:
    """Make a string safe inside a Markdown table cell."""
    return _esc(text).replace("|", "\\|").replace("\n", " ")


def render_digest(data: dict[str, Any]) -> str:
    room = data["room"]
    people = data["participants"]
    messages = data["messages"]
    claims = data["claims"]
    usage = data["usage"]
    decisions = data.get("decisions")

    humans = [p for p in people if p["kind"] == "human"]
    agents = [p for p in people if p["kind"] == "agent"]
    names = {p["id"]: p["name"] for p in people}
    mode = room["mode"] if "mode" in room else "open"

    lines: list[str] = []
    lines.append(f"# {_esc(room['name'])}")
    lines.append("")
    lines.append(f"- Room id: `{room['id']}`")
    lines.append(f"- Mode: `{mode}`")
    lines.append(f"- Started: {room['created_at'].isoformat(timespec='seconds')}")
    lines.append(
        f"- Participants: {len(humans)} human(s), {len(agents)} agent(s)"
    )
    if agents:
        lines.append(
            "- Agents: " + ", ".join(f"**{_esc(p['name'])}**" for p in agents)
        )
    lines.append("")

    lines.append("## Transcript")
    lines.append("")
    if not messages:
        lines.append("_(no messages)_")
    else:
        lines.append("| seq | author | message |")
        lines.append("|---|---|---|")
        for m in messages:
            lines.append(
                f"| {m['seq']} | {_cell(m['author_name'])} | {_cell(m['body'])} |"
            )
    lines.append("")

    if decisions is not None:
        moderator_name = next(
            (p["name"] for p in people if "role" in p and p["role"] == "moderator"),
            None,
        )
        heading = (
            f"## 决策 — {_esc(moderator_name)}" if moderator_name else "## 决策"
        )
        lines.append(heading)
        lines.append("")
        if not decisions:
            lines.append("_(no decisions)_")
        else:
            lines.append("| trigger_seq | action | target | created_at |")
            lines.append("|---|---|---|---|")
            for d in decisions:
                target_id = d.target_id
                if target_id is None:
                    target = "—"
                else:
                    known = names.get(target_id)
                    target = _cell(known if known is not None else str(target_id))
                created_s = d.created_at.isoformat(timespec="seconds")
                lines.append(
                    f"| {d.trigger_seq} | {_cell(d.action)} | {target} | {created_s} |"
                )
        lines.append("")

    lines.append("## Action items (claims)")
    lines.append("")
    if not claims:
        lines.append("_(no claims — no shared work was claimed in this room)_")
    else:
        for c in claims:
            owner = c["claimed_by_name"] or names.get(c["claimed_by"], "?")
            age_s = max(0.0, (datetime.now(UTC) - c["created_at"]).total_seconds())
            age = f"{age_s:.0f}s"
            # A claim older than the steal TTL is a crash orphan (the
            # holder died before release/fulfil): label it, so the
            # action-item list does not present a dead lock as an
            # obligation someone is actively holding.
            mark = " (stale — holder likely crashed; stealable)" if age_s > CLAIM_TTL_S else ""
            lines.append(f"- `{c['task_key']}` — held by **{_esc(owner)}** ({age}){mark}")
    lines.append("")

    lines.append("## Model spend")
    lines.append("")
    if not usage:
        lines.append("_(no model calls recorded)_")
    else:
        lines.append("| purpose | model | calls | prompt tokens | completion tokens |")
        lines.append("|---|---|---|---|---|")
        total_prompt = 0
        total_completion = 0
        for u in usage:
            prompt = int(u["prompt_tokens"] or 0)
            completion = int(u["completion_tokens"] or 0)
            total_prompt += prompt
            total_completion += completion
            lines.append(
                f"| {u['purpose']} | {_cell(str(u['model']))} | {u['calls']} | {prompt} | {completion} |"
            )
        lines.append(f"| **total** | — | — | **{total_prompt}** | **{total_completion}** |")
    lines.append("")
    return "\n".join(lines)


async def build_room_digest(pool: Any, room_id: Any) -> str | None:
    data = await db.room_digest(pool, room_id)
    if data is None:
        return None
    return render_digest(data)
