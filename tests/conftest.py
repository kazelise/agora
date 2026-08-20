from __future__ import annotations

import os
import socket
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

import pytest

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("AGORA_DATABASE_URL", "postgresql://agora:agora@127.0.0.1:5433/agora")
os.environ.setdefault("AGORA_REDIS_URL", "redis://127.0.0.1:6379/0")

DSN = os.environ["AGORA_DATABASE_URL"]
REDIS_URL = os.environ["AGORA_REDIS_URL"]


def _host_port(url: str, default_port: int) -> tuple[str, int]:
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or default_port
    return host, port


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def _reachable() -> bool:
    pg_host, pg_port = _host_port(DSN, 5432)
    rd_host, rd_port = _host_port(REDIS_URL, 6379)
    return _port_open(pg_host, pg_port) and _port_open(rd_host, rd_port)


def _try_compose() -> None:
    subprocess.run(
        ["docker", "compose", "-f", str(ROOT / "docker-compose.yml"), "up", "-d", "--wait"],
        cwd=ROOT,
        check=False,
        timeout=120,
    )


def _ensure_services() -> bool:
    if _reachable():
        return True
    try:
        _try_compose()
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return False
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if _reachable():
            return True
        time.sleep(1)
    return False


@pytest.fixture(scope="session")
def require_services() -> None:
    if not _ensure_services():
        pytest.skip(
            "Postgres/Redis unreachable and docker compose could not start them"
        )
