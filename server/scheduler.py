from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import asyncpg
import redis.asyncio as redis

from brain.wake_hints import set_called_on
from server import db
from server.computers import ComputerHub
from server.mentions import mentioned_name
from server.models import ParticipantRow

logger = logging.getLogger("agora.scheduler")

WAKE_CHANNEL = "agora:wake"
# Same reconnect shape as daemon/main.py (~202-214): start small, cap
# at a few seconds so a flapping Redis cannot stall the process forever.
SUBSCRIBER_BACKOFF_S = 1.0
SUBSCRIBER_BACKOFF_CAP_S = 4.0

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
    """One in-flight turn per agent, plus a single pending-wake slot.

    Wakes that arrive while a turn is pending/running do not enqueue N
    extra turns. They overwrite `_pending` with the latest (room_id,
    agent_id). When the current turn finishes we rerun at most once —
    against the LATEST wake, not the first: an agent in several rooms
    whose wake for room B lands while it is mid-turn for room A must
    rerun in room B (its cursor-based inbox will also cover any A
    updates that raced in, on the turn after that).

    One caveat kept honest: with wakes for TWO other rooms racing the
    current turn, only the last one survives the coalesce. The other
    room's delivery waits for the next event — the same eventual
    delivery a missed pub/sub wake gets. Bounding the rerun to one is
    what keeps a message burst from queueing N model calls.
    """

    def __init__(self, run_turn: TurnFn) -> None:
        self._run_turn = run_turn
        self._lock = asyncio.Lock()
        self._pending: tuple[UUID, UUID] | None = None
        self._task: asyncio.Task[None] | None = None

    async def notify(self, room_id: UUID, agent_id: UUID) -> tuple[UUID, UUID] | None:
        """Returns the overwritten pending (room, agent), if any."""
        async with self._lock:
            if self._task is not None and not self._task.done():
                old = self._pending
                self._pending = (room_id, agent_id)
                return old
            self._pending = None
            self._task = asyncio.create_task(
                self._loop(room_id, agent_id),
                name=f"agora-turn-{agent_id}",
            )
            return None

    async def _loop(self, room_id: UUID, agent_id: UUID) -> None:
        while True:
            try:
                await self._run_turn(agent_id, room_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "turn crashed room=%s agent=%s — continuing",
                    room_id,
                    agent_id,
                )
            async with self._lock:
                if self._pending is None:
                    self._task = None
                    return
                room_id, agent_id = self._pending
                self._pending = None

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
    ) -> None:
        self._pool = pool
        self._run_turn = run_turn or self.run_turn_stub
        self._computers = computers
        self._redis = redis_client
        self._lanes: dict[UUID, AgentLane] = {}
        # (agent, room) → trigger_seq of a pending call_on. Max-merged
        # and consumed by the in-process lane. Hosts that cannot receive
        # this value (BYOA websocket, K8s Job via HttpWorld) get the
        # same seq in Redis + the wake frame; /runtime/turn-context
        # consumes it. In-process DirectWorld never reads Redis.
        self._called_on: dict[tuple[UUID, UUID], int] = {}
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
        called_on_seq = self._called_on.pop((agent_id, room_id), None)
        result = await self._invoke_turn(agent_id, room_id, called_on_seq)
        if result is not None:
            self.brain_results.append(result)

    async def _invoke_turn(
        self, agent_id: UUID, room_id: UUID, called_on_seq: int | None
    ) -> Any:
        fn = self._run_turn
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            params = {}
        kwargs: dict[str, Any] = {}
        if "called_on_seq" in params:
            kwargs["called_on_seq"] = called_on_seq
        if "called_on" in params:
            kwargs["called_on"] = called_on_seq is not None
        if kwargs:
            return await fn(agent_id, room_id, **kwargs)  # type: ignore[call-arg]
        return await fn(agent_id, room_id)

    async def dispatch(
        self, room_id: UUID, author_id: UUID, *, seq: int | None = None
    ) -> None:
        """Wake whoever the room's mode says should run.

        Routing lives here — not in the stall sweeper, not in fan-out —
        so a landed message and a stall nudge share one decision. Open
        rooms fan out to every non-author agent (today's behavior).
        Moderated rooms are arithmetic/protocol: an `@Name` against the
        roster wakes that agent; otherwise only the moderator. The
        author is never woken by their own message.
        """
        targets = await self._route_wake(room_id, author_id, seq)
        for agent, called_on_seq in targets:
            await self._wake_agent(room_id, agent, called_on_seq=called_on_seq)

    async def wake_one(
        self,
        room_id: UUID,
        agent_id: UUID,
        *,
        called_on_seq: int | None = None,
    ) -> None:
        """Targeted wake (moderator call_on). Same host routing as dispatch.

        `called_on_seq` is the only marker: an int is a call_on at that
        trigger; None is an ordinary targeted wake.
        """
        agent = await db.get_participant(self._pool, agent_id)
        if agent.kind != "agent":
            return
        await self._wake_agent(room_id, agent, called_on_seq=called_on_seq)

    async def _route_wake(
        self, room_id: UUID, author_id: UUID, seq: int | None
    ) -> list[tuple[ParticipantRow, int | None]]:
        agents = await db.list_agent_participants(self._pool, room_id)
        try:
            room = await db.get_room(self._pool, room_id)
        except db.NotFoundError:
            return []
        if room.mode != "moderated":
            return [(a, None) for a in agents if a.id != author_id]
        # Stall nudges pass seq=None: no new body, and a nudge is not a
        # self-wake on the last author's own message. A landed-message
        # wake carries the triggering seq and still excludes the author.
        if seq is None:
            undelivered = await self._undelivered_call_on(room_id, agents)
            if undelivered is not None:
                return [undelivered]
            # silence at last_seq: nobody owes a word. Replaying the
            # moderator only burns the decline budget.
            latest = await db.get_latest_decision(self._pool, room_id)
            if latest is not None and latest.action == "silence":
                last_seq = await db.get_room_last_seq(self._pool, room_id)
                if latest.trigger_seq == last_seq:
                    return []
            mods = [a for a in agents if a.role == "moderator"]
            if not mods:
                logger.warning(
                    "moderated room %s (%s) has no moderator — wake dropped",
                    room.name,
                    room_id,
                )
                return []
            return [(mods[0], None)]
        message = await db.get_message_by_seq(self._pool, room_id, seq)
        if message is not None:
            hit = mentioned_name(message.body, [a.name for a in agents])
            if hit is not None:
                named = next(a for a in agents if a.name == hit)
                if named.id == author_id:
                    return []
                return [(named, None)]
        mods = [
            a for a in agents if a.role == "moderator" and a.id != author_id
        ]
        if not mods:
            if not any(a.role == "moderator" for a in agents):
                logger.warning(
                    "moderated room %s (%s) has no moderator — wake dropped",
                    room.name,
                    room_id,
                )
            return []
        return [(mods[0], None)]

    async def _undelivered_call_on(
        self, room_id: UUID, agents: list[ParticipantRow]
    ) -> tuple[ParticipantRow, int] | None:
        """Redeliver a lost call_on: committed row, unread cursor.

        If the latest decision is call_on and the target has not read
        past trigger_seq, the original wake was dropped (offline host,
        restart, coalesce). This is counting, not an outbox.
        """
        latest = await db.get_latest_decision(self._pool, room_id)
        if latest is None or latest.action != "call_on" or latest.target_id is None:
            return None
        target = next((a for a in agents if a.id == latest.target_id), None)
        if target is None:
            return None
        last_read = await db.get_last_read(self._pool, target.id, room_id)
        if last_read < latest.trigger_seq:
            return target, latest.trigger_seq
        return None

    def _wants_redis_hint(self, agent: ParticipantRow) -> bool:
        # BYOA hosts cannot see the in-process dict. K8s Jobs also
        # cannot: the lane runs launcher.run_turn, but the container
        # talks HttpWorld and reads /runtime/turn-context.
        if agent.computer_id is not None:
            return True
        owner = getattr(self._run_turn, "__self__", None)
        return bool(getattr(owner, "remote_called_on_hint", False))

    async def _wake_agent(
        self,
        room_id: UUID,
        agent: ParticipantRow,
        *,
        called_on_seq: int | None,
    ) -> None:
        if agent.computer_id is None:
            if called_on_seq is not None:
                key = (agent.id, room_id)
                current = self._called_on.get(key, 0)
                self._called_on[key] = max(current, called_on_seq)
            if (
                called_on_seq is not None
                and self._redis is not None
                and self._wants_redis_hint(agent)
            ):
                await set_called_on(self._redis, agent.id, room_id, called_on_seq)
            overwritten = await self.lane(agent.id).notify(room_id, agent.id)
            if overwritten is not None:
                old_room, old_agent = overwritten
                if (old_agent, old_room) != (agent.id, room_id):
                    self._called_on.pop((old_agent, old_room), None)
            return
        if called_on_seq is not None and self._redis is not None:
            await set_called_on(self._redis, agent.id, room_id, called_on_seq)
        hub = self._computers
        payload: dict[str, Any] = {
            "type": "wake",
            "agent_id": str(agent.id),
            "room_id": str(room_id),
        }
        # Only attach the flag when it is set so existing exact-match
        # wake-frame tests (open-room BYOA) stay byte-identical.
        if called_on_seq is not None:
            payload["called_on"] = True
            payload["called_on_seq"] = called_on_seq
        if hub is not None and hub.is_online(agent.computer_id):
            sent = await hub.send_wake(agent.computer_id, payload)
            if sent:
                return
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


def log_unexpected_task_exit(task: asyncio.Task[Any]) -> None:
    """Lifespan watchdog: a background task died, and it was not a stop."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is None:
        return
    logger.critical(
        "background task %s exited unexpectedly: %s",
        task.get_name(),
        exc,
        exc_info=exc,
    )


async def run_subscriber(
    client: redis.Redis,
    scheduler: Scheduler,
    ready: asyncio.Event,
    stop: asyncio.Event,
) -> None:
    """Listen for wake publishes; reconnect with backoff after the first subscribe.

    The first subscribe is not retried: a failure raises so startup
    fails loudly instead of hanging on ready.wait(). After one
    successful subscribe, an outer loop recreates the pubsub,
    resubscribes, and resumes listen. Delay resets after every
    successful subscribe. Per-message dispatch is isolated so one bad
    payload cannot kill the loop. finally only aclose() — unsubscribe
    on a dead connection can hang shutdown.
    """
    delay = SUBSCRIBER_BACKOFF_S
    subscribed_once = False
    while not stop.is_set():
        pubsub = client.pubsub()
        try:
            await pubsub.subscribe(WAKE_CHANNEL)
            ready.set()
            subscribed_once = True
            delay = SUBSCRIBER_BACKOFF_S
            async for message in pubsub.listen():
                if stop.is_set():
                    return
                if message.get("type") != "message":
                    continue
                raw = message.get("data")
                if not isinstance(raw, str):
                    continue
                try:
                    data = json.loads(raw)
                    seq_raw = data.get("seq")
                    seq = int(seq_raw) if seq_raw is not None else None
                    await scheduler.dispatch(
                        UUID(data["room_id"]), UUID(data["author_id"]), seq=seq
                    )
                except Exception:
                    logger.exception("wake dispatch failed — continuing")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not subscribed_once:
                raise
            if stop.is_set():
                return
            logger.warning(
                "wake subscriber disconnected (%s: %s); reconnect in %.0fs",
                type(exc).__name__,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, SUBSCRIBER_BACKOFF_CAP_S)
        finally:
            try:
                await pubsub.aclose()
            except Exception:
                logger.warning("wake pubsub close failed — fail-open", exc_info=True)
