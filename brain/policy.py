"""Which model may run which node. Enforced in code, not in the prompt."""

from __future__ import annotations

import os

DEFAULT_SMALL_MODEL = "gpt-5.6-luna"
DEFAULT_BIG_MODEL = "gpt-5.6-terra"


def small_model_name() -> str:
    return os.environ.get("AGORA_SMALL_MODEL", DEFAULT_SMALL_MODEL)


def big_model_name() -> str:
    return os.environ.get("AGORA_BIG_MODEL", DEFAULT_BIG_MODEL)


def assert_triage_model(model_name: str) -> str:
    """Triage is the cheap gate. Passing it the big model is a wiring bug."""
    if model_name == big_model_name():
        raise ValueError(
            f"triage must not use the big model ({model_name!r}); "
            f"expected AGORA_SMALL_MODEL ({small_model_name()!r})"
        )
    return model_name
