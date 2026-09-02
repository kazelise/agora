"""Cloud host: one Kubernetes Job per coalesced turn.

The scheduler lane still serializes. This module is the turn function
when AGORA_K8S_ENABLED=1: create a Job, wait until it finishes, return.
A failed create or a failed Job is a missed turn — fail-open, same as a
dropped Redis wake. k8s retries are not turn retries (backoffLimit=0);
the dirty bit is the only rerun.

The Job talks HttpWorld with the cluster token. Reply goes through
/runtime/reply so WebSocket fan-out stays on the server.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID, uuid4

import httpx

from server.config import Settings

logger = logging.getLogger("agora.k8s")

INCLUSTER_TOKEN = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
INCLUSTER_CA = Path("/var/run/secrets/kubernetes.io/serviceaccount/ca.crt")


class JobLauncher(Protocol):
    async def run_turn(self, agent_id: UUID, room_id: UUID) -> None: ...


class K8sConfigError(RuntimeError):
    """k8s is enabled but we have no API endpoint."""


def job_name(agent_id: UUID) -> str:
    return f"agora-turn-{agent_id.hex[:8]}-{uuid4().hex[:8]}"


def resolve_k8s_endpoint(settings: Settings) -> tuple[str, str, str | bool]:
    """Return (api_url, token, verify) for the Jobs API.

    verify is a CA path, True (system CAs), or False (insecure, kind).
    """
    if settings.k8s_api_url:
        token = settings.k8s_token
        if settings.k8s_insecure:
            verify: str | bool = False
        elif settings.k8s_ca_path:
            verify = settings.k8s_ca_path
        else:
            verify = True
        return settings.k8s_api_url.rstrip("/"), token, verify

    host = os.environ.get("KUBERNETES_SERVICE_HOST")
    port = os.environ.get("KUBERNETES_SERVICE_PORT", "443")
    if INCLUSTER_TOKEN.is_file() and host:
        return (
            f"https://{host}:{port}",
            INCLUSTER_TOKEN.read_text().strip(),
            str(INCLUSTER_CA) if INCLUSTER_CA.is_file() else True,
        )
    raise K8sConfigError(
        "AGORA_K8S_ENABLED but no AGORA_K8S_API_URL and not in-cluster"
    )


def _env_value(name: str, value: str) -> dict[str, str]:
    return {"name": name, "value": value}


def _env_secret(name: str, secret: str, key: str) -> dict[str, Any]:
    return {
        "name": name,
        "valueFrom": {"secretKeyRef": {"name": secret, "key": key}},
    }


def turn_job_manifest(
    agent_id: UUID,
    room_id: UUID,
    *,
    settings: Settings,
    name: str | None = None,
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
) -> dict[str, Any]:
    """Build a batch/v1 Job that runs `python -m brain.job` once."""
    job = name or job_name(agent_id)
    secret = settings.k8s_secret_name
    env: list[dict[str, Any]] = [
        _env_value("AGORA_SERVER_URL", settings.k8s_server_url),
        _env_value("AGORA_AGENT_ID", str(agent_id)),
        _env_value("AGORA_ROOM_ID", str(room_id)),
        _env_value("AGORA_SMALL_MODEL", settings.small_model),
        _env_value("AGORA_BIG_MODEL", settings.big_model),
    ]
    if secret:
        env.append(_env_secret("AGORA_CLUSTER_TOKEN", secret, "cluster-token"))
        env.append(_env_secret("OPENAI_API_KEY", secret, "openai-api-key"))
        env.append(_env_secret("OPENAI_BASE_URL", secret, "openai-base-url"))
        env.append(_env_secret("OPENAI_API_BASE", secret, "openai-base-url"))
    else:
        # kind / tests: literals. Production should set AGORA_K8S_SECRET_NAME.
        env.append(_env_value("AGORA_CLUSTER_TOKEN", settings.cluster_token))
        key = openai_api_key if openai_api_key is not None else os.environ.get(
            "OPENAI_API_KEY", ""
        )
        base = openai_base_url if openai_base_url is not None else (
            os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
            or ""
        )
        env.append(_env_value("OPENAI_API_KEY", key))
        if base:
            env.append(_env_value("OPENAI_BASE_URL", base))
            env.append(_env_value("OPENAI_API_BASE", base))

    labels = {
        "app": "agora",
        "agora.role": "turn",
        "agora.agent-id": str(agent_id),
        "agora.room-id": str(room_id),
    }
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job,
            "namespace": settings.k8s_namespace,
            "labels": labels,
        },
        "spec": {
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": settings.k8s_ttl_s,
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [
                        {
                            "name": "turn",
                            "image": settings.k8s_image,
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["python", "-m", "brain.job"],
                            "env": env,
                        }
                    ],
                },
            },
        },
    }


class K8sJobsApi:
    """Thin batch/v1 client. Tests inject a fake."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def create_job(self, namespace: str, body: dict[str, Any]) -> dict[str, Any]:
        resp = await self._client.post(
            f"/apis/batch/v1/namespaces/{namespace}/jobs",
            json=body,
        )
        resp.raise_for_status()
        return resp.json()

    async def get_job(self, namespace: str, name: str) -> dict[str, Any]:
        resp = await self._client.get(
            f"/apis/batch/v1/namespaces/{namespace}/jobs/{name}"
        )
        resp.raise_for_status()
        return resp.json()


def job_finished(status: dict[str, Any] | None) -> str | None:
    """Return 'succeeded', 'failed', or None if still running."""
    if not status:
        return None
    if int(status.get("succeeded") or 0) >= 1:
        return "succeeded"
    if int(status.get("failed") or 0) >= 1:
        return "failed"
    for condition in status.get("conditions") or []:
        if condition.get("type") == "Failed" and condition.get("status") == "True":
            return "failed"
    return None


class K8sJobLauncher:
    # Job process talks HttpWorld and cannot see Scheduler's dict.
    remote_called_on_hint = True

    def __init__(
        self,
        settings: Settings,
        *,
        api: K8sJobsApi | None = None,
        sleep: Any = asyncio.sleep,
    ) -> None:
        self.settings = settings
        self._api = api
        self._sleep = sleep
        self._owns_client = False
        self._client: httpx.AsyncClient | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> K8sJobLauncher:
        api_url, token, verify = resolve_k8s_endpoint(settings)
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        client = httpx.AsyncClient(
            base_url=api_url,
            headers=headers,
            verify=verify,
            timeout=30.0,
        )
        launcher = cls(settings, api=K8sJobsApi(client))
        launcher._owns_client = True
        launcher._client = client
        return launcher

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def run_turn(self, agent_id: UUID, room_id: UUID) -> None:
        """TurnFn for the scheduler. Fail-open: errors are a missed turn."""
        try:
            await self._run(agent_id, room_id)
        except Exception:
            logger.warning(
                "k8s job failed agent=%s room=%s — fail-open",
                agent_id,
                room_id,
                exc_info=True,
            )

    async def _run(self, agent_id: UUID, room_id: UUID) -> None:
        if self._api is None:
            raise K8sConfigError("K8sJobLauncher has no API client")
        name = job_name(agent_id)
        body = turn_job_manifest(agent_id, room_id, settings=self.settings, name=name)
        logger.info("creating job %s for agent %s room %s", name, agent_id, room_id)
        await self._api.create_job(self.settings.k8s_namespace, body)
        deadline = asyncio.get_running_loop().time() + self.settings.k8s_job_timeout_s
        while True:
            job = await self._api.get_job(self.settings.k8s_namespace, name)
            outcome = job_finished(job.get("status"))
            if outcome == "succeeded":
                logger.info("job %s succeeded", name)
                return
            if outcome == "failed":
                logger.warning("job %s failed — missed turn", name)
                return
            if asyncio.get_running_loop().time() >= deadline:
                logger.warning("job %s timed out — missed turn", name)
                return
            await self._sleep(self.settings.k8s_poll_s)
