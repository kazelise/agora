"""Daemon wiring: env-tunable pacer + limiter parameters."""

from __future__ import annotations

from daemon.limiter import DEFAULT_MAX_CONCURRENT
from daemon.main import parse_args
from daemon.pacer import BASE_INTERVAL_S, MAX_INTERVAL_S


def _argv() -> list[str]:
    return ["--computer-id", "c", "--token", "t", "--server", "http://x"]


def test_pacer_defaults() -> None:
    args = parse_args(_argv())
    assert args.pacer_base_s == BASE_INTERVAL_S
    assert args.pacer_max_s == MAX_INTERVAL_S
    assert args.max_concurrent == DEFAULT_MAX_CONCURRENT


def test_pacer_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("AGORA_PACER_BASE_S", "1.5")
    monkeypatch.setenv("AGORA_PACER_MAX_S", "20")
    monkeypatch.setenv("AGORA_MAX_CONCURRENT", "3")
    args = parse_args(_argv())
    assert args.pacer_base_s == 1.5
    assert args.pacer_max_s == 20.0
    assert args.max_concurrent == 3


def test_pacer_flag_beats_env(monkeypatch) -> None:
    monkeypatch.setenv("AGORA_PACER_BASE_S", "1.5")
    args = parse_args([*_argv(), "--pacer-base-s", "0.2"])
    assert args.pacer_base_s == 0.2
