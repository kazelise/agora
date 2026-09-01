"""Daemon-side AgentLane: same pending-wake semantics, no infra imports."""

from __future__ import annotations

import asyncio
from uuid import UUID, uuid4

import pytest

from daemon.lanes import AgentLane


@pytest.mark.asyncio
async def test_burst_coalesces_to_in_flight_plus_one_rerun() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    runs = 0

    async def stub(_agent_id, _room_id) -> None:
        nonlocal runs
        runs += 1
        started.set()
        await release.wait()

    lane = AgentLane(stub)
    agent_id = uuid4()

    await lane.notify(uuid4(), agent_id)
    await started.wait()
    for _ in range(3):
        await lane.notify(uuid4(), agent_id)
    release.set()
    await lane.wait_idle()

    assert runs == 2


@pytest.mark.asyncio
async def test_rerun_uses_latest_wake_room() -> None:
    """A wake for room B landing mid-turn for room A reruns in room B —
    the pending slot carries the latest (room, agent) pair."""
    started = asyncio.Event()
    release = asyncio.Event()
    rooms: list[UUID] = []

    async def stub(_agent_id, room_id) -> None:
        rooms.append(room_id)
        started.set()
        await release.wait()

    lane = AgentLane(stub)
    agent_id = uuid4()
    room_a = uuid4()
    room_b = uuid4()

    await lane.notify(room_a, agent_id)
    await started.wait()
    await lane.notify(room_b, agent_id)
    release.set()
    await lane.wait_idle()

    assert rooms == [room_a, room_b]
