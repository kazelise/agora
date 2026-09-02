"""Phase 7c crash containment: subscriber, lane, watchdog, call_on wake."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from uuid import UUID, uuid4

import asyncpg
import pytest
import redis.asyncio as redis

from brain.world_direct import DirectWorld
from server import db
from server.scheduler import (
    WAKE_CHANNEL,
    AgentLane,
    log_unexpected_task_exit,
    run_subscriber,
)
from tests.conftest import DSN, REDIS_URL


@pytest.fixture
async def pool(require_services: None) -> AsyncIterator[asyncpg.Pool]:
    created = await db.create_pool(DSN)
    await db.migrate(created)
    await db.truncate_all(created)
    yield created
    await created.close()


@pytest.fixture
async def redis_client(require_services: None) -> AsyncIterator[redis.Redis]:
    client = redis.from_url(REDIS_URL, decode_responses=True)
    yield client
    await client.aclose()


class _FakePubSub:
    def __init__(
        self,
        queue: asyncio.Queue[dict | None],
        *,
        fail_listen: bool,
        fail_subscribe: bool = False,
    ) -> None:
        self._queue = queue
        self.fail_listen = fail_listen
        self.fail_subscribe = fail_subscribe
        self.subscribe_calls = 0
        self.unsubscribe_calls = 0
        self.aclose_calls = 0

    async def subscribe(self, channel: str) -> None:
        self.subscribe_calls += 1
        assert channel == WAKE_CHANNEL
        if self.fail_subscribe:
            raise ConnectionError("redis down")

    async def listen(self):
        if self.fail_listen:
            raise ConnectionError("pubsub dropped")
        while True:
            item = await self._queue.get()
            if item is None:
                return
            yield item

    async def unsubscribe(self, channel: str) -> None:
        self.unsubscribe_calls += 1

    async def aclose(self) -> None:
        self.aclose_calls += 1


class _FakeRedis:
    def __init__(
        self, *, listen_failures: int = 0, fail_first_subscribe: bool = False
    ) -> None:
        self.queue: asyncio.Queue[dict | None] = asyncio.Queue()
        self.created: list[_FakePubSub] = []
        self._remaining_failures = listen_failures
        self._fail_first_subscribe = fail_first_subscribe

    def pubsub(self) -> _FakePubSub:
        fail_listen = self._remaining_failures > 0
        if fail_listen:
            self._remaining_failures -= 1
        fail_subscribe = self._fail_first_subscribe
        self._fail_first_subscribe = False
        pubsub = _FakePubSub(
            self.queue, fail_listen=fail_listen, fail_subscribe=fail_subscribe
        )
        self.created.append(pubsub)
        return pubsub


class _RecordingScheduler:
    def __init__(self, *, fail_first: bool = False) -> None:
        self.calls: list[tuple[UUID, UUID, int | None]] = []
        self._fail_first = fail_first

    async def dispatch(
        self, room_id: UUID, author_id: UUID, *, seq: int | None = None
    ) -> None:
        self.calls.append((room_id, author_id, seq))
        if self._fail_first:
            self._fail_first = False
            raise RuntimeError("dispatch exploded")


def _wake_payload(room_id: UUID, author_id: UUID, seq: int) -> dict:
    return {
        "type": "message",
        "data": json.dumps(
            {
                "room_id": str(room_id),
                "author_id": str(author_id),
                "seq": seq,
            }
        ),
    }


@pytest.mark.asyncio
async def test_subscriber_survives_dispatch_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("server.scheduler.SUBSCRIBER_BACKOFF_S", 0)
    room_a, author_a = uuid4(), uuid4()
    room_b, author_b = uuid4(), uuid4()
    client = _FakeRedis()
    scheduler = _RecordingScheduler(fail_first=True)
    ready = asyncio.Event()
    stop = asyncio.Event()
    task = asyncio.create_task(run_subscriber(client, scheduler, ready, stop))
    try:
        await asyncio.wait_for(ready.wait(), timeout=1.0)
        await client.queue.put(_wake_payload(room_a, author_a, 1))
        await client.queue.put(_wake_payload(room_b, author_b, 2))
        deadline = asyncio.get_running_loop().time() + 2.0
        while len(scheduler.calls) < 2:
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError(f"only dispatched {scheduler.calls}")
            await asyncio.sleep(0.02)
        assert scheduler.calls == [
            (room_a, author_a, 1),
            (room_b, author_b, 2),
        ]
    finally:
        stop.set()
        await client.queue.put(None)
        await asyncio.wait_for(task, timeout=1.0)


@pytest.mark.asyncio
async def test_subscriber_resubscribes_after_listen_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("server.scheduler.SUBSCRIBER_BACKOFF_S", 0)
    room_id, author_id = uuid4(), uuid4()
    client = _FakeRedis(listen_failures=1)
    scheduler = _RecordingScheduler()
    ready = asyncio.Event()
    stop = asyncio.Event()
    task = asyncio.create_task(run_subscriber(client, scheduler, ready, stop))
    try:
        await asyncio.wait_for(ready.wait(), timeout=1.0)
        deadline = asyncio.get_running_loop().time() + 2.0
        while len(client.created) < 2:
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError(f"did not reconnect: {len(client.created)}")
            await asyncio.sleep(0.02)
        assert [ps.subscribe_calls for ps in client.created] == [1, 1]
        await client.queue.put(_wake_payload(room_id, author_id, 7))
        deadline = asyncio.get_running_loop().time() + 2.0
        while not scheduler.calls:
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("message after reconnect was not dispatched")
            await asyncio.sleep(0.02)
        assert scheduler.calls == [(room_id, author_id, 7)]
    finally:
        stop.set()
        await client.queue.put(None)
        await asyncio.wait_for(task, timeout=1.0)


@pytest.mark.asyncio
async def test_subscriber_first_subscribe_failure_raises() -> None:
    client = _FakeRedis(fail_first_subscribe=True)
    ready = asyncio.Event()
    stop = asyncio.Event()
    with pytest.raises(ConnectionError, match="redis down"):
        await run_subscriber(client, _RecordingScheduler(), ready, stop)
    assert not ready.is_set()
    assert len(client.created) == 1
    assert client.created[0].unsubscribe_calls == 0
    assert client.created[0].aclose_calls == 1


@pytest.mark.asyncio
async def test_lane_logs_and_continues_to_pending_after_turn_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    rooms: list[UUID] = []

    async def stub(_agent_id: UUID, room_id: UUID) -> None:
        rooms.append(room_id)
        if len(rooms) == 1:
            started.set()
            await release.wait()
            raise RuntimeError("turn exploded")

    lane = AgentLane(stub)
    agent_id = uuid4()
    room_a = uuid4()
    room_b = uuid4()

    with caplog.at_level(logging.ERROR, logger="agora.scheduler"):
        await lane.notify(room_a, agent_id)
        await started.wait()
        await lane.notify(room_b, agent_id)
        release.set()
        await lane.wait_idle()

    assert rooms == [room_a, room_b]
    assert "turn exploded" in caplog.text
    assert str(room_a) in caplog.text
    assert str(agent_id) in caplog.text


@pytest.mark.asyncio
async def test_done_callback_logs_critical_on_unexpected_exit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def boom() -> None:
        raise RuntimeError("subscriber died")

    with caplog.at_level(logging.CRITICAL, logger="agora.scheduler"):
        task = asyncio.create_task(boom(), name="agora-wake-subscriber")
        task.add_done_callback(log_unexpected_task_exit)
        with pytest.raises(RuntimeError, match="subscriber died"):
            await task

    assert "agora-wake-subscriber" in caplog.text
    assert "subscriber died" in caplog.text


@pytest.mark.asyncio
async def test_record_decision_returns_when_on_call_on_raises(
    pool: asyncpg.Pool, redis_client: redis.Redis
) -> None:
    room = await db.create_room(pool, "rt", mode="moderated")
    await db.add_participant(pool, room.id, "human", "Ada", None)
    chair = await db.add_participant(
        pool, room.id, "agent", "Chair", "keeps time", role="moderator"
    )
    iris = await db.add_participant(pool, room.id, "agent", "Iris", "brief")
    await db.insert_message(pool, room.id, chair.id, "open")

    async def boom_wake(_room_id: UUID, _target_id: UUID) -> None:
        raise RuntimeError("get_participant blip")

    world = DirectWorld(pool, redis_client, on_call_on=boom_wake)
    result = await world.record_decision(
        room.id, chair.id, 1, "call_on", iris.id
    )
    assert result.status == "won"
    assert result.action == "call_on"
    assert result.target_id == iris.id
    rows = await pool.fetch(
        "SELECT action, target_id FROM moderator_decisions WHERE room_id = $1",
        room.id,
    )
    assert len(rows) == 1
    assert rows[0]["target_id"] == iris.id
