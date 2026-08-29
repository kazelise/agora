"""Phase 4c: cluster token, Job manifest, launcher, one-shot runner."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI

from brain.graph import Brain
from brain.job import parse_args, run_once
from daemon.world_http import HttpWorld
from server.config import Settings
from server.db import truncate_all
from server.k8s import (
    K8sConfigError,
    K8sJobLauncher,
    K8sJobsApi,
    job_finished,
    resolve_k8s_endpoint,
    turn_job_manifest,
)
from server.main import create_app
from tests.asgi_ws import connect_asgi_ws
from tests.fakes import ScriptedChatModel, tool_call, triage_message


def _settings(**kwargs: Any) -> Settings:
    return Settings(
        cluster_token="cluster-secret",
        k8s_namespace="agora",
        k8s_image="agora:local",
        k8s_server_url="http://agora.agora.svc.cluster.local:8000",
        k8s_ttl_s=120,
        **kwargs,
    )


def test_turn_job_manifest_is_one_shot_never_restart() -> None:
    agent_id = UUID("11111111-1111-1111-1111-111111111111")
    room_id = UUID("22222222-2222-2222-2222-222222222222")
    body = turn_job_manifest(
        agent_id, room_id, settings=_settings(), name="agora-turn-11111111-abcd1234"
    )
    assert body["apiVersion"] == "batch/v1"
    assert body["kind"] == "Job"
    assert body["metadata"]["name"] == "agora-turn-11111111-abcd1234"
    assert body["metadata"]["namespace"] == "agora"
    assert body["metadata"]["labels"]["agora.agent-id"] == str(agent_id)
    assert body["spec"]["backoffLimit"] == 0
    assert body["spec"]["ttlSecondsAfterFinished"] == 120
    pod = body["spec"]["template"]["spec"]
    assert pod["restartPolicy"] == "Never"
    container = pod["containers"][0]
    assert container["command"] == ["python", "-m", "brain.job"]
    assert container["image"] == "agora:local"
    env = {item["name"]: item["value"] for item in container["env"]}
    assert env["AGORA_AGENT_ID"] == str(agent_id)
    assert env["AGORA_ROOM_ID"] == str(room_id)
    assert env["AGORA_CLUSTER_TOKEN"] == "cluster-secret"
    assert env["AGORA_SERVER_URL"] == "http://agora.agora.svc.cluster.local:8000"


def test_turn_job_manifest_uses_secret_refs_when_named() -> None:
    body = turn_job_manifest(
        uuid4(),
        uuid4(),
        settings=_settings(k8s_secret_name="agora-cloud"),
        name="agora-turn-x",
    )
    env = {item["name"]: item for item in body["spec"]["template"]["spec"]["containers"][0]["env"]}
    assert env["AGORA_CLUSTER_TOKEN"]["valueFrom"]["secretKeyRef"] == {
        "name": "agora-cloud",
        "key": "cluster-token",
    }
    assert "value" not in env["AGORA_CLUSTER_TOKEN"]
    assert env["OPENAI_API_KEY"]["valueFrom"]["secretKeyRef"]["key"] == "openai-api-key"


def test_job_finished() -> None:
    assert job_finished(None) is None
    assert job_finished({}) is None
    assert job_finished({"succeeded": 1}) == "succeeded"
    assert job_finished({"failed": 1}) == "failed"
    assert job_finished(
        {"conditions": [{"type": "Failed", "status": "True"}]}
    ) == "failed"


def test_resolve_k8s_endpoint_requires_api_or_incluster() -> None:
    with pytest.raises(K8sConfigError, match="not in-cluster"):
        resolve_k8s_endpoint(_settings())


def test_resolve_k8s_endpoint_from_settings() -> None:
    url, token, verify = resolve_k8s_endpoint(
        _settings(k8s_api_url="https://127.0.0.1:6443", k8s_token="tok", k8s_insecure=True)
    )
    assert url == "https://127.0.0.1:6443"
    assert token == "tok"
    assert verify is False


def test_parse_job_args_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGORA_SERVER_URL", "http://svc:8000")
    monkeypatch.setenv("AGORA_CLUSTER_TOKEN", "tok")
    monkeypatch.setenv("AGORA_AGENT_ID", "11111111-1111-1111-1111-111111111111")
    monkeypatch.setenv("AGORA_ROOM_ID", "22222222-2222-2222-2222-222222222222")
    args = parse_args([])
    assert args.server == "http://svc:8000"
    assert args.token == "tok"
    assert args.agent_id.startswith("11111111")


class FakeK8s:
    def __init__(self, outcomes: list[dict[str, Any]] | None = None) -> None:
        self.created: list[dict[str, Any]] = []
        self.gets = 0
        self._outcomes = list(outcomes or [{"status": {"succeeded": 1}}])

    async def create_job(self, namespace: str, body: dict[str, Any]) -> dict[str, Any]:
        self.created.append({"namespace": namespace, "body": body})
        return body

    async def get_job(self, namespace: str, name: str) -> dict[str, Any]:
        self.gets += 1
        if not self._outcomes:
            return {"metadata": {"name": name}, "status": {}}
        return self._outcomes.pop(0)


@pytest.mark.asyncio
async def test_launcher_creates_job_and_waits_for_success() -> None:
    api = FakeK8s(
        [
            {"status": {}},
            {"status": {"succeeded": 1}},
        ]
    )
    launcher = K8sJobLauncher(_settings(k8s_poll_s=0.0), api=api)  # type: ignore[arg-type]
    agent_id = uuid4()
    room_id = uuid4()
    await launcher.run_turn(agent_id, room_id)
    assert len(api.created) == 1
    assert api.created[0]["namespace"] == "agora"
    env = {
        item["name"]: item["value"]
        for item in api.created[0]["body"]["spec"]["template"]["spec"]["containers"][0]["env"]
    }
    assert env["AGORA_AGENT_ID"] == str(agent_id)
    assert api.gets == 2


@pytest.mark.asyncio
async def test_launcher_failed_job_is_fail_open() -> None:
    api = FakeK8s([{"status": {"failed": 1}}])
    launcher = K8sJobLauncher(_settings(), api=api)  # type: ignore[arg-type]
    await launcher.run_turn(uuid4(), uuid4())
    assert api.created  # launched, did not raise


@pytest.mark.asyncio
async def test_launcher_create_error_is_fail_open() -> None:
    class Boom:
        async def create_job(self, namespace: str, body: dict[str, Any]) -> dict[str, Any]:
            raise httpx.HTTPStatusError(
                "no", request=httpx.Request("POST", "http://k8s"), response=httpx.Response(500)
            )

        async def get_job(self, namespace: str, name: str) -> dict[str, Any]:
            raise AssertionError("should not poll after create failed")

    launcher = K8sJobLauncher(_settings(), api=Boom())  # type: ignore[arg-type]
    await launcher.run_turn(uuid4(), uuid4())


@pytest.mark.asyncio
async def test_launcher_timeout_is_fail_open() -> None:
    settings = _settings(k8s_job_timeout_s=0.0, k8s_poll_s=0.0)
    api = FakeK8s([])  # never finishes
    launcher = K8sJobLauncher(settings, api=api)  # type: ignore[arg-type]
    await launcher.run_turn(uuid4(), uuid4())
    assert api.created


class Harness:
    def __init__(self, app: FastAPI, client: httpx.AsyncClient) -> None:
        self.app = app
        self.client = client


@pytest.fixture
async def harness(require_services: None) -> AsyncIterator[Harness]:
    app = create_app(stub_turns=True, settings=_settings())
    async with app.router.lifespan_context(app):
        await truncate_all(app.state.pool)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield Harness(app=app, client=client)


async def _cloud_room(client: httpx.AsyncClient) -> dict[str, Any]:
    room = (await client.post("/rooms", json={"name": "k8s-room"})).json()
    human = (
        await client.post(
            f"/rooms/{room['id']}/participants",
            json={"kind": "human", "name": "Ada"},
        )
    ).json()
    agent = (
        await client.post(
            f"/rooms/{room['id']}/participants",
            json={"kind": "agent", "name": "Iris", "persona": "cloud"},
        )
    ).json()
    return {"room": room, "human": human, "agent": agent}


@pytest.mark.asyncio
async def test_cluster_token_can_run_unhosted_agent(harness: Harness) -> None:
    setup = await _cloud_room(harness.client)
    room_id = UUID(setup["room"]["id"])
    human_id = UUID(setup["human"]["id"])
    agent_id = UUID(setup["agent"]["id"])
    await harness.client.post(
        f"/rooms/{room_id}/messages",
        json={"author_id": str(human_id), "body": "hello iris"},
    )
    world = HttpWorld(harness.client, "cluster-secret")
    ctx = await world.load_turn(agent_id, room_id)
    assert ctx.agent.name == "Iris"
    assert [m.body for m in ctx.inbox] == ["hello iris"]
    row = await world.insert_message(room_id, agent_id, "iris here", not_after_seq=1)
    assert row.seq == 2


@pytest.mark.asyncio
async def test_cluster_token_cannot_run_byoa_agent(harness: Harness) -> None:
    computer = (await harness.client.post("/computers", json={"name": "laptop"})).json()
    room = (await harness.client.post("/rooms", json={"name": "mixed"})).json()
    byoa = (
        await harness.client.post(
            f"/rooms/{room['id']}/participants",
            json={
                "kind": "agent",
                "name": "Jules",
                "computer_id": computer["id"],
            },
        )
    ).json()
    denied = await harness.client.get(
        "/runtime/turn-context",
        params={"agent_id": byoa["id"], "room_id": room["id"]},
        headers={"Authorization": "Bearer cluster-secret"},
    )
    assert denied.status_code == 403
    assert "BYOA" in denied.json()["detail"]


@pytest.mark.asyncio
async def test_computer_token_cannot_run_cloud_agent(harness: Harness) -> None:
    computer = (await harness.client.post("/computers", json={"name": "laptop"})).json()
    setup = await _cloud_room(harness.client)
    denied = await harness.client.get(
        "/runtime/turn-context",
        params={"agent_id": setup["agent"]["id"], "room_id": setup["room"]["id"]},
        headers={"Authorization": f"Bearer {computer['token']}"},
    )
    assert denied.status_code == 403


@pytest.mark.asyncio
async def test_wrong_cluster_token_is_401(harness: Harness) -> None:
    setup = await _cloud_room(harness.client)
    denied = await harness.client.get(
        "/runtime/turn-context",
        params={"agent_id": setup["agent"]["id"], "room_id": setup["room"]["id"]},
        headers={"Authorization": "Bearer nope"},
    )
    assert denied.status_code == 401


@pytest.mark.asyncio
async def test_job_runner_replies_via_cluster_token(harness: Harness) -> None:
    setup = await _cloud_room(harness.client)
    room_id = UUID(setup["room"]["id"])
    human_id = UUID(setup["human"]["id"])
    agent_id = UUID(setup["agent"]["id"])
    await harness.client.post(
        f"/rooms/{room_id}/messages",
        json={"author_id": str(human_id), "body": "please reply"},
    )
    small = ScriptedChatModel(
        [triage_message(actionable=True, reason="addressed", response_mode="me")]
    )
    big = ScriptedChatModel([tool_call("reply", {"body": "iris from a job"})])
    world = HttpWorld(harness.client, "cluster-secret")
    brain = Brain(world, small_model=small, big_model=big)
    result = await run_once(
        "http://test",
        "cluster-secret",
        agent_id,
        room_id,
        brain=brain,
        http=harness.client,
    )
    assert result.outcome == "replied"
    assert result.reply_body == "iris from a job"
    listed = (await harness.client.get(f"/rooms/{room_id}/messages")).json()["messages"]
    assert listed[-1]["body"] == "iris from a job"
    rows = await harness.app.state.pool.fetch(
        "SELECT purpose FROM llm_calls WHERE agent_id = $1 ORDER BY created_at, id",
        agent_id,
    )
    assert [r["purpose"] for r in rows][0] == "triage"


class RecordingLauncher:
    def __init__(self) -> None:
        self.calls: list[tuple[UUID, UUID]] = []
        self.seen = asyncio.Event()

    async def run_turn(self, agent_id: UUID, room_id: UUID) -> None:
        self.calls.append((agent_id, room_id))
        self.seen.set()


@pytest.mark.asyncio
async def test_k8s_enabled_routes_cloud_agent_to_launcher(
    require_services: None,
) -> None:
    launcher = RecordingLauncher()
    app = create_app(settings=_settings(k8s_enabled=True), job_launcher=launcher)
    async with app.router.lifespan_context(app):
        await truncate_all(app.state.pool)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            setup = await _cloud_room(client)
            posted = await client.post(
                f"/rooms/{setup['room']['id']}/messages",
                json={"author_id": setup["human"]["id"], "body": "wake the job"},
            )
            assert posted.status_code == 200
            await asyncio.wait_for(launcher.seen.wait(), timeout=4.0)
            await app.state.scheduler.wait_idle()
            assert launcher.calls == [
                (UUID(setup["agent"]["id"]), UUID(setup["room"]["id"]))
            ]
            # In-process stub/brain did not run.
            assert app.state.scheduler.turns == []


@pytest.mark.asyncio
async def test_k8s_enabled_still_wakes_byoa_over_ws(
    require_services: None,
) -> None:
    launcher = RecordingLauncher()
    app = create_app(settings=_settings(k8s_enabled=True), job_launcher=launcher)
    async with app.router.lifespan_context(app):
        await truncate_all(app.state.pool)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            computer = (await client.post("/computers", json={"name": "laptop"})).json()
            room = (await client.post("/rooms", json={"name": "mixed"})).json()
            human = (
                await client.post(
                    f"/rooms/{room['id']}/participants",
                    json={"kind": "human", "name": "Ada"},
                )
            ).json()
            cloud = (
                await client.post(
                    f"/rooms/{room['id']}/participants",
                    json={"kind": "agent", "name": "Iris"},
                )
            ).json()
            byoa = (
                await client.post(
                    f"/rooms/{room['id']}/participants",
                    json={
                        "kind": "agent",
                        "name": "Jules",
                        "computer_id": computer["id"],
                    },
                )
            ).json()
            ws = await connect_asgi_ws(
                app,
                f"/ws/computers/{computer['id']}",
                query_string=f"token={computer['token']}",
            )
            try:
                posted = await client.post(
                    f"/rooms/{room['id']}/messages",
                    json={"author_id": human["id"], "body": "hello both"},
                )
                assert posted.status_code == 200
                frame = await ws.receive_json(timeout=4.0)
                assert frame["type"] == "wake"
                assert frame["agent_id"] == byoa["id"]
                await asyncio.wait_for(launcher.seen.wait(), timeout=4.0)
                await app.state.scheduler.wait_idle()
                assert launcher.calls == [(UUID(cloud["id"]), UUID(room["id"]))]
            finally:
                await ws.close()


@pytest.mark.asyncio
async def test_k8s_enabled_without_token_refuses_to_start() -> None:
    app = create_app(settings=Settings(k8s_enabled=True, cluster_token=""))
    with pytest.raises(RuntimeError, match="CLUSTER_TOKEN"):
        async with app.router.lifespan_context(app):
            pass
