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

from typing import Any

from server import db


def _esc(text: str) -> str:
    return text.replace("\r\n", "\n").strip()


def render_digest(data: dict[str, Any]) -> str:
    room = data["room"]
    people = data["participants"]
    messages = data["messages"]
    claims = data["claims"]
    usage = data["usage"]

    humans = [p for p in people if p["kind"] == "human"]
    agents = [p for p in people if p["kind"] == "agent"]
    names = {p["id"]: p["name"] for p in people}

    lines: list[str] = []
    lines.append(f"# {room['name']}")
    lines.append("")
    lines.append(f"- Room id: `{room['id']}`")
    lines.append(f"- Started: {room['created_at'].isoformat(timespec='seconds')}")
    lines.append(
        f"- Participants: {len(humans)} human(s), {len(agents)} agent(s)"
    )
    if agents:
        lines.append(
            "- Agents: " + ", ".join(f"**{p['name']}**" for p in agents)
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
            body = _esc(m["body"]).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {m['seq']} | {m['author_name']} | {body} |")
    lines.append("")

    lines.append("## Action items (claims)")
    lines.append("")
    if not claims:
        lines.append("_(no claims — no shared work was claimed in this room)_")
    else:
        for c in claims:
            owner = c["claimed_by_name"] or names.get(c["claimed_by"], "?")
            lines.append(f"- `{c['task_key']}` — held by **{owner}**")
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
                f"| {u['purpose']} | {u['model']} | {u['calls']} | {prompt} | {completion} |"
            )
        lines.append(f"| **total** | — | — | **{total_prompt}** | **{total_completion}** |")
    lines.append("")
    return "\n".join(lines)


async def build_room_digest(pool: Any, room_id: Any) -> str | None:
    data = await db.room_digest(pool, room_id)
    if data is None:
        return None
    return render_digest(data)
