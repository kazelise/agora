"""Phase 5 bus: Redis errors are a miss, never a raise."""

from __future__ import annotations

from uuid import uuid4

import pytest

from server.bus import (
    acquire_lane,
    clear_presence,
    has_presence,
    lane_should_rerun,
    mark_lane_dirty,
    mark_presence,
    refresh_lane,
    release_lane,
    try_claim,
)
from server.scheduler import fanout_message, publish_json


class Boom:
    async def set(self, *args, **kwargs):
        raise ConnectionError("redis down")

    async def eval(self, *args, **kwargs):
        raise ConnectionError("redis down")

    async def exists(self, *args, **kwargs):
        raise ConnectionError("redis down")

    async def expire(self, *args, **kwargs):
        raise ConnectionError("redis down")

    async def publish(self, *args, **kwargs):
        raise ConnectionError("redis down")


class Row:
    def __init__(self) -> None:
        self.room_id = uuid4()
        self.author_id = uuid4()
        self.seq = 1

    def as_ws(self) -> dict:
        return {
            "type": "message",
            "room_id": str(self.room_id),
            "author_id": str(self.author_id),
            "seq": self.seq,
            "body": "ignored",
        }


@pytest.mark.asyncio
async def test_claim_and_presence_fail_open() -> None:
    broken = Boom()
    computer_id = uuid4()
    agent_id = uuid4()

    assert await try_claim(broken, "agora:dispatch:x", "w1", 30) is False
    assert await acquire_lane(broken, agent_id, "w1") is False
    assert await has_presence(broken, computer_id) is None
    assert await lane_should_rerun(broken, agent_id, "w1") is False

    await mark_presence(broken, computer_id, "w1")
    await clear_presence(broken, computer_id, "w1")
    await mark_lane_dirty(broken, agent_id)
    await refresh_lane(broken, agent_id)
    await release_lane(broken, agent_id, "w1")


@pytest.mark.asyncio
async def test_fanout_fail_open_never_blocks() -> None:
    broken = Boom()
    await publish_json(broken, "agora:wake", {"seq": 1})
    await fanout_message(None, broken, Row())
