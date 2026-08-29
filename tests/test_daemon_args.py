"""Daemon wiring: env-tunable pacer parameters."""

from __future__ import annotations

from daemon.main import parse_args
from daemon.pacer import BASE_INTERVAL_S, MAX_INTERVAL_S


def test_pacer_defaults() -> None:
    args = parse_args(
        ["--computer-id", "c", "--token", "t", "--server", "http://x"]
    )
    assert args.pacer_base_s == BASE_INTERVAL_S
    assert args.pacer_max_s == MAX_INTERVAL_S


def test_pacer_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("AGORA_PACER_BASE_S", "1.5")
    monkeypatch.setenv("AGORA_PACER_MAX_S", "20")
    args = parse_args(
        ["--computer-id", "c", "--token", "t", "--server", "http://x"]
    )
    assert args.pacer_base_s == 1.5
    assert args.pacer_max_s == 20.0


def test_pacer_flag_beats_env(monkeypatch) -> None:
    monkeypatch.setenv("AGORA_PACER_BASE_S", "1.5")
    args = parse_args(
        [
            "--computer-id",
            "c",
            "--token",
            "t",
            "--server",
            "http://x",
            "--pacer-base-s",
            "0.2",
        ]
    )
    assert args.pacer_base_s == 0.2
