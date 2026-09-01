"""@-mention protocol parse — the one hardcoded interaction rule.

This is the same class of work as `task_key` parsing: extract a token
that the protocol defined, not classify the message's meaning. Earliest
`@Name` in the body wins; length only breaks a tie at the same offset
(`@IrisLee` over `@Iris`). Comparison is case-sensitive; the body is
not rewritten.
"""

from __future__ import annotations

import re


def mentioned_name(body: str, names: list[str]) -> str | None:
    """Return the roster name of the earliest `@Name` hit, or None.

    Earliest start offset wins so `@Bob … @Alexander` is Bob, not the
    longer later name. Length only breaks a tie at the same offset
    (`@IrisLee` vs `@Iris`). Left/right boundaries use Unicode word
    chars so `@张三` does not eat `@张三丰`, and `foo@Bob` is not a mention.
    """
    if not body or not names:
        return None
    # Longest first so a shared start offset prefers the longer seat.
    ordered = sorted({n for n in names if n}, key=len, reverse=True)
    best: str | None = None
    best_at = -1
    for name in ordered:
        pattern = re.compile(
            r"(?<![A-Za-z0-9_@])@" + re.escape(name) + r"(?!\w)"
        )
        match = pattern.search(body)
        if match is None:
            continue
        at = match.start()
        # Earliest offset wins. Same offset: longest-first already stored
        # the longer seat, so a later prefix at that offset is ignored.
        if best is None or at < best_at:
            best = name
            best_at = at
    return best
