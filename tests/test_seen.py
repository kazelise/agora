from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
import redis.asyncio as redis

from brain.seen import get_seen, is_fresh, record_seen
from tests.conftest import REDIS_URL


@pytest.fixture
async def redis_client(require_services: None) -> AsyncIterator[redis.Redis]:
    client = redis.from_url(REDIS_URL, decode_responses=True)
    yield client
    await client.aclose()


@pytest.mark.asyncio
async def test_record_seen_is_monotonic(redis_client: redis.Redis) -> None:
    agent_id, room_id = uuid4(), uuid4()
    await record_seen(redis_client, agent_id, room_id, 3)
    await record_seen(redis_client, agent_id, room_id, 2)
    assert await get_seen(redis_client, agent_id, room_id) == 3
    await record_seen(redis_client, agent_id, room_id, 5)
    assert await get_seen(redis_client, agent_id, room_id) == 5
    assert await is_fresh(redis_client, agent_id, room_id, 5) is True
    assert await is_fresh(redis_client, agent_id, room_id, 6) is False


@pytest.mark.asyncio
async def test_seen_fail_open_never_raises() -> None:
    class Boom:
        async def eval(self, *args, **kwargs):
            raise ConnectionError("redis down")

        async def get(self, *args, **kwargs):
            raise ConnectionError("redis down")

    broken = Boom()  # type: ignore[assignment]
    agent_id, room_id = uuid4(), uuid4()
    await record_seen(broken, agent_id, room_id, 4)
    assert await get_seen(broken, agent_id, room_id) is None
    assert await is_fresh(broken, agent_id, room_id, 99) is True
