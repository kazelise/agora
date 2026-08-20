from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from server.scheduler import AgentLane


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
    room_id = uuid4()

    await lane.notify(room_id, agent_id)
    await started.wait()
    for _ in range(4):
        await lane.notify(room_id, agent_id)
    release.set()
    await lane.wait_idle()

    assert runs == 2
