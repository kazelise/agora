from __future__ import annotations

import asyncio
import json
import logging
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import asyncpg
import redis.asyncio as redis

from server import db
from server.bus import (
    DISPATCH_TTL_S,
    HOST_WAKE_CHANNEL,
    MESSAGE_CHANNEL,
    WAKE_CHANNEL,
    acquire_lane,
    dispatch_key,
    lane_should_rerun,
    mark_lane_dirty,
    refresh_lane,
    release_lane,
    try_claim,
)
from server.computers import ComputerHub

logger = logging.getLogger("agora.scheduler")

TurnFn = Callable[[UUID, UUID], Awaitable[Any]]


@dataclass(frozen=True)
class TurnRecord:
    agent_id: UUID
    agent_name: str
    room_id: UUID
    room_name: str
    inbox_count: int
    since_seq: int
    last_read_seq: int


class AgentLane:
    """One in-flight turn per agent, plus a single dirty bit.

    Wakes that arrive while a turn is pending/running do not enqueue N
    extra turns. They set `_dirty`. When the current turn finishes we
    rerun at most once against the latest inbox (rerun-once). A 5-message
    burst is therefore 1 in-flight + 1 coalesced rerun, not 5.
    """

    def __init__(self, run_turn: TurnFn) -> None:
        self._run_turn = run_turn
        self._lock = asyncio.Lock()
        self._dirty = False
        self._task: asyncio.Task[None] | None = None

    async def notify(self, room_id: UUID, agent_id: UUID) -> None:
        async with self._lock:
            if self._task is not None and not self._task.done():
                self._dirty = True
                return
            self._dirty = False
            self._task = asyncio.create_task(
                self._loop(room_id, agent_id),
                name=f"agora-turn-{agent_id}",
            )

    async def _loop(self, room_id: UUID, agent_id: UUID) -> None:
        while True:
            await self._run_turn(agent_id, room_id)
            async with self._lock:
                if not self._dirty:
                    self._task = None
                    return
                self._dirty = False

    async def wait_idle(self) -> None:
        while True:
            async with self._lock:
                task = self._task
            if task is None:
                return
            await task


class ClusterAgentLane:
    """AgentLane, but the lock and dirty bit live in Redis.

    Two workers can both see the same wake. The dispatch claim (agent,
    seq) makes one of them call notify. A later seq on the other worker
    sets dirty; the owner reruns. Same 1 in-flight + 1 coalesced rerun.
    """

    def __init__(
        self, client: redis.Redis, worker_id: str, run_turn: TurnFn
    ) -> None:
        self._redis = client
        self._worker_id = worker_id
        self._run_turn = run_turn
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None

    async def notify(self, room_id: UUID, agent_id: UUID) -> None:
        acquired = await acquire_lane(self._redis, agent_id, self._worker_id)
        if not acquired:
            await mark_lane_dirty(self._redis, agent_id)
            return
        # Redis release and `_task = None` share this lock, so a finishing
        # loop cannot drop the key while still looking in-flight.
        async with self._lock:
            if self._task is not None and not self._task.done():
                await mark_lane_dirty(self._redis, agent_id)
                return
            self._task = asyncio.create_task(
                self._loop(room_id, agent_id),
                name=f"agora-turn-{agent_id}",
            )

    async def _loop(self, room_id: UUID, agent_id: UUID) -> None:
        try:
            while True:
                await refresh_lane(self._redis, agent_id)
                await self._run_turn(agent_id, room_id)
                async with self._lock:
                    if not await lane_should_rerun(
                        self._redis, agent_id, self._worker_id
                    ):
                        self._task = None
                        return
        except Exception:
            async with self._lock:
                await release_lane(self._redis, agent_id, self._worker_id)
                self._task = None
            raise

    async def wait_idle(self) -> None:
        while True:
            async with self._lock:
                task = self._task
            if task is None:
                return
            await task


class Scheduler:
    def __init__(
        self,
        pool: asyncpg.Pool,
        run_turn: TurnFn | None = None,
        computers: ComputerHub | None = None,
        redis_client: redis.Redis | None = None,
        worker_id: str | None = None,
    ) -> None:
        self._pool = pool
        self._run_turn = run_turn or self.run_turn_stub
        self._computers = computers
        self._redis = redis_client
        self._worker_id = worker_id or secrets.token_hex(8)
        self._lanes: dict[UUID, AgentLane | ClusterAgentLane] = {}
        self.turns: list[TurnRecord] = []
        self.brain_results: list[Any] = []
        # Demo-only: keep a stub turn in-flight so a burst can hit the dirty bit.
        self.turn_delay_s = 0.0

    def lane(self, agent_id: UUID) -> AgentLane | ClusterAgentLane:
        lane = self._lanes.get(agent_id)
        if lane is None:
            if self._redis is not None:
                lane = ClusterAgentLane(
                    self._redis, self._worker_id, self._tracked_turn
                )
            else:
                lane = AgentLane(self._tracked_turn)
            self._lanes[agent_id] = lane
        return lane

    async def _tracked_turn(self, agent_id: UUID, room_id: UUID) -> None:
        result = await self._run_turn(agent_id, room_id)
        if result is not None:
            self.brain_results.append(result)

    async def dispatch(
        self, room_id: UUID, author_id: UUID, seq: int = 0
    ) -> None:
        agents = await db.list_agent_participants(self._pool, room_id)
        for agent in agents:
            if agent.id == author_id:
                continue
            if self._redis is not None:
                won = await try_claim(
                    self._redis,
                    dispatch_key(agent.id, seq),
                    self._worker_id,
                    DISPATCH_TTL_S,
                )
                if not won:
                    continue
            if agent.computer_id is None:
                await self.lane(agent.id).notify(room_id, agent.id)
                continue
            present = False
            if self._computers is not None:
                present = await self._computers.is_present(agent.computer_id)
            if present and self._redis is not None:
                await publish_json(
                    self._redis,
                    HOST_WAKE_CHANNEL,
                    {
                        "type": "wake",
                        "computer_id": str(agent.computer_id),
                        "agent_id": str(agent.id),
                        "room_id": str(room_id),
                    },
                )
                continue
            if present and self._computers is not None:
                sent = await self._computers.send_wake(
                    agent.computer_id,
                    {
                        "type": "wake",
                        "agent_id": str(agent.id),
                        "room_id": str(room_id),
                    },
                )
                if sent:
                    continue
            logger.info(
                "agent %s is sleeping (computer offline)",
                agent.name,
            )

    async def run_turn_stub(self, agent_id: UUID, room_id: UUID) -> None:
        agent = await db.get_participant(self._pool, agent_id)
        room = await db.get_room(self._pool, room_id)
        since = await db.get_last_read(self._pool, agent_id, room_id)
        inbox = await db.list_messages(self._pool, room_id, since_seq=since)
        last_read = max((m.seq for m in inbox), default=since)
        if inbox:
            await db.set_last_read(self._pool, agent_id, room_id, last_read)
        record = TurnRecord(
            agent_id=agent_id,
            agent_name=agent.name,
            room_id=room_id,
            room_name=room.name,
            inbox_count=len(inbox),
            since_seq=since,
            last_read_seq=last_read,
        )
        self.turns.append(record)
        logger.info(
            "agent %s woke for room %s, inbox has %s new messages since last_read seq %s",
            agent.name,
            room.name,
            len(inbox),
            since,
        )
        if self.turn_delay_s:
            await asyncio.sleep(self.turn_delay_s)

    async def wait_idle(self) -> None:
        await asyncio.gather(*(lane.wait_idle() for lane in list(self._lanes.values())))


async def publish_json(client: redis.Redis, channel: str, payload: dict) -> None:
    try:
        await client.publish(channel, json.dumps(payload))
    except Exception:
        logger.warning("publish %s failed — fail-open", channel, exc_info=True)


async def fanout_message(hub: Any, client: redis.Redis, row: Any) -> None:
    """Broadcast + wake. Broadcast goes through Redis so every worker's
    RoomHub sees it; the publishing worker is also a subscriber.
    `hub` is unused on purpose — kept so call sites stay stable.
    """
    _ = hub
    await publish_json(client, MESSAGE_CHANNEL, row.as_ws())
    await publish_wake(client, row.room_id, row.author_id, row.seq)


async def publish_wake(
    client: redis.Redis,
    room_id: UUID,
    author_id: UUID,
    seq: int,
) -> None:
    await publish_json(
        client,
        WAKE_CHANNEL,
        {
            "room_id": str(room_id),
            "author_id": str(author_id),
            "seq": seq,
        },
    )


async def run_subscriber(
    client: redis.Redis,
    scheduler: Scheduler,
    ready: asyncio.Event,
    stop: asyncio.Event,
    *,
    hub: Any = None,
    computers: ComputerHub | None = None,
) -> None:
    pubsub = client.pubsub()
    await pubsub.subscribe(WAKE_CHANNEL, MESSAGE_CHANNEL, HOST_WAKE_CHANNEL)
    ready.set()
    try:
        async for message in pubsub.listen():
            if stop.is_set():
                break
            if message.get("type") != "message":
                continue
            raw = message.get("data")
            if not isinstance(raw, str):
                continue
            channel = message.get("channel")
            try:
                data = json.loads(raw)
                if channel == WAKE_CHANNEL:
                    await scheduler.dispatch(
                        UUID(data["room_id"]),
                        UUID(data["author_id"]),
                        int(data.get("seq") or 0),
                    )
                elif channel == MESSAGE_CHANNEL and hub is not None:
                    await hub.broadcast(UUID(data["room_id"]), data)
                elif channel == HOST_WAKE_CHANNEL and computers is not None:
                    await computers.send_wake(UUID(data["computer_id"]), data)
            except Exception:
                logger.warning(
                    "bus handler failed on %s — fail-open skip",
                    channel,
                    exc_info=True,
                )
    finally:
        await pubsub.unsubscribe(WAKE_CHANNEL, MESSAGE_CHANNEL, HOST_WAKE_CHANNEL)
        await pubsub.aclose()
