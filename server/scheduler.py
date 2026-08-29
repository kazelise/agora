from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import asyncpg
import redis.asyncio as redis

from server import db
from server.computers import ComputerHub

logger = logging.getLogger("agora.scheduler")

WAKE_CHANNEL = "agora:wake"

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


class Scheduler:
    def __init__(
        self,
        pool: asyncpg.Pool,
        run_turn: TurnFn | None = None,
        computers: ComputerHub | None = None,
    ) -> None:
        self._pool = pool
        self._run_turn = run_turn or self.run_turn_stub
        self._computers = computers
        self._lanes: dict[UUID, AgentLane] = {}
        self.turns: list[TurnRecord] = []
        self.brain_results: list[Any] = []
        # Demo-only: keep a stub turn in-flight so a burst can hit the dirty bit.
        self.turn_delay_s = 0.0

    def lane(self, agent_id: UUID) -> AgentLane:
        lane = self._lanes.get(agent_id)
        if lane is None:
            lane = AgentLane(self._tracked_turn)
            self._lanes[agent_id] = lane
        return lane

    async def _tracked_turn(self, agent_id: UUID, room_id: UUID) -> None:
        result = await self._run_turn(agent_id, room_id)
        if result is not None:
            self.brain_results.append(result)

    async def dispatch(self, room_id: UUID, author_id: UUID) -> None:
        agents = await db.list_agent_participants(self._pool, room_id)
        for agent in agents:
            if agent.id == author_id:
                continue
            if agent.computer_id is None:
                await self.lane(agent.id).notify(room_id, agent.id)
                continue
            hub = self._computers
            if hub is not None and hub.is_online(agent.computer_id):
                sent = await hub.send_wake(
                    agent.computer_id,
                    {
                        "type": "wake",
                        "agent_id": str(agent.id),
                        "room_id": str(room_id),
                    },
                )
                if sent:
                    continue
            # BYOA host is offline. Do not queue: the inbox is cursor-based,
            # so a missed wake is a missed turn and the agent catches up on
            # the next one. Unbounded offline queues would just grow.
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


async def fanout_message(
    hub: Any,
    client: redis.Redis,
    row: Any,
    *,
    on_committed: Callable[[UUID], Awaitable[None]] | None = None,
) -> None:
    """Broadcast + wake, the same path human POST and a runtime reply share.

    `on_committed` (room_id) fires for every landed message; the stall
    sweeper uses it to reset its decline budget (state changed, fresh
    nudge budget).
    """
    await hub.broadcast(row.room_id, row.as_ws())
    await publish_wake(client, row.room_id, row.author_id, row.seq)
    if on_committed is not None:
        try:
            await on_committed(UUID(str(row.room_id)))
        except Exception:
            logger.warning("on_committed hook failed — fail-open", exc_info=True)


async def publish_wake(
    client: redis.Redis,
    room_id: UUID,
    author_id: UUID,
    seq: int,
) -> None:
    payload = json.dumps(
        {
            "room_id": str(room_id),
            "author_id": str(author_id),
            "seq": seq,
        }
    )
    try:
        await client.publish(WAKE_CHANNEL, payload)
    except Exception:
        # Coordination signal, not a correctness invariant: the message
        # is already committed. Losing a wake is a missed turn, not data loss.
        logger.warning("wake publish failed — fail-open", exc_info=True)


async def run_subscriber(
    client: redis.Redis,
    scheduler: Scheduler,
    ready: asyncio.Event,
    stop: asyncio.Event,
) -> None:
    pubsub = client.pubsub()
    await pubsub.subscribe(WAKE_CHANNEL)
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
            data = json.loads(raw)
            await scheduler.dispatch(UUID(data["room_id"]), UUID(data["author_id"]))
    finally:
        await pubsub.unsubscribe(WAKE_CHANNEL)
        await pubsub.aclose()
