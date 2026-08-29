"""Phase 4a: GitHub OAuth, JWT admission, room ownership."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI

from server.config import Settings
from server.db import truncate_all
from server.jwtutil import (
    JWTError,
    check_oauth_state,
    decode_jwt,
    encode_jwt,
    issue_user_token,
    make_oauth_state,
)
from server.main import create_app
from server.oauth import GitHubUser
from tests.asgi_ws import connect_asgi_ws


def test_jwt_roundtrip_and_expiry() -> None:
    secret = "s3cret"
    token = encode_jwt({"sub": "u1", "exp": int(time.time()) + 60}, secret)
    assert decode_jwt(token, secret)["sub"] == "u1"
    expired = encode_jwt({"sub": "u1", "exp": int(time.time()) - 1}, secret)
    with pytest.raises(JWTError, match="expired"):
        decode_jwt(expired, secret)
    with pytest.raises(JWTError, match="invalid signature"):
        decode_jwt(token, "other")
    with pytest.raises(JWTError, match="malformed"):
        decode_jwt("not-a-jwt", secret)


def test_oauth_state_is_signed_and_expires() -> None:
    secret = "s3cret"
    state = make_oauth_state(secret)
    assert check_oauth_state(state, secret)
    assert not check_oauth_state(state, "other")
    assert not check_oauth_state("nope", secret)
    ts, nonce, sig = state.rsplit(":", 2)
    old = f"{int(time.time()) - 10_000}:{nonce}"
    import hashlib
    import hmac

    old_sig = hmac.new(secret.encode(), old.encode(), hashlib.sha256).hexdigest()
    assert not check_oauth_state(f"{old}:{old_sig}", secret, max_age_s=600)


class FakeGitHub:
    def __init__(self) -> None:
        self.profiles: dict[str, GitHubUser] = {
            "alice": GitHubUser(1, "alice", "Alice", "https://example.com/a.png"),
            "bob": GitHubUser(2, "bob", "Bob", None),
        }

    async def exchange_code(self, code: str) -> str:
        if code not in self.profiles:
            raise RuntimeError("bad code")
        return f"gh-{code}"

    async def fetch_user(self, access_token: str) -> GitHubUser:
        return self.profiles[access_token.removeprefix("gh-")]


def _auth_settings(**kwargs: Any) -> Settings:
    return Settings(
        github_client_id="client",
        github_client_secret="ghs",
        jwt_secret="jwt-secret",
        jwt_ttl_s=3600,
        **kwargs,
    )


class Harness:
    def __init__(self, app: FastAPI, client: httpx.AsyncClient, github: FakeGitHub) -> None:
        self.app = app
        self.client = client
        self.github = github


@pytest.fixture
async def harness(require_services: None) -> AsyncIterator[Harness]:
    github = FakeGitHub()
    app = create_app(stub_turns=True, settings=_auth_settings(), github=github)
    async with app.router.lifespan_context(app):
        await truncate_all(app.state.pool)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield Harness(app=app, client=client, github=github)


async def _login(client: httpx.AsyncClient, code: str = "alice") -> dict[str, Any]:
    start = await client.get("/auth/github")
    assert start.status_code == 302
    loc = urlparse(start.headers["location"])
    assert loc.scheme == "https"
    assert loc.netloc == "github.com"
    qs = parse_qs(loc.query)
    assert qs["client_id"] == ["client"]
    assert qs["scope"] == ["read:user"]
    state = qs["state"][0]
    done = await client.get(
        "/auth/github/callback",
        params={"code": code, "state": state},
    )
    assert done.status_code == 200, done.text
    return done.json()


@pytest.mark.asyncio
async def test_github_login_issues_jwt_and_me(harness: Harness) -> None:
    body = await _login(harness.client, "alice")
    assert body["token_type"] == "bearer"
    assert body["user"]["login"] == "alice"
    assert body["user"]["github_id"] == 1
    claims = decode_jwt(body["access_token"], "jwt-secret")
    assert claims["iss"] == "agora"
    assert claims["login"] == "alice"
    me = await harness.client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {body['access_token']}"},
    )
    assert me.status_code == 200
    assert me.json()["login"] == "alice"


@pytest.mark.asyncio
async def test_callback_rejects_bad_state_and_github_failure(harness: Harness) -> None:
    start = await harness.client.get("/auth/github")
    state = parse_qs(urlparse(start.headers["location"]).query)["state"][0]
    bad_state = await harness.client.get(
        "/auth/github/callback",
        params={"code": "alice", "state": "tampered"},
    )
    assert bad_state.status_code == 400
    failed = await harness.client.get(
        "/auth/github/callback",
        params={"code": "nope", "state": state},
    )
    assert failed.status_code == 401


@pytest.mark.asyncio
async def test_management_api_requires_jwt_when_auth_on(harness: Harness) -> None:
    denied = await harness.client.post("/rooms", json={"name": "secret"})
    assert denied.status_code == 401
    body = await _login(harness.client)
    token = body["access_token"]
    created = await harness.client.post(
        "/rooms",
        json={"name": "secret"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert created.status_code == 200
    assert created.json()["created_by"] == body["user"]["id"]


@pytest.mark.asyncio
async def test_second_user_cannot_see_or_write_first_users_room(
    harness: Harness,
) -> None:
    alice = await _login(harness.client, "alice")
    room = (
        await harness.client.post(
            "/rooms",
            json={"name": "alice-room"},
            headers={"Authorization": f"Bearer {alice['access_token']}"},
        )
    ).json()
    human = (
        await harness.client.post(
            f"/rooms/{room['id']}/participants",
            json={"kind": "human", "name": "Ada"},
            headers={"Authorization": f"Bearer {alice['access_token']}"},
        )
    ).json()
    bob = await _login(harness.client, "bob")
    listed = await harness.client.get(
        "/rooms",
        headers={"Authorization": f"Bearer {bob['access_token']}"},
    )
    assert listed.json() == []
    peek = await harness.client.get(
        f"/rooms/{room['id']}/messages",
        headers={"Authorization": f"Bearer {bob['access_token']}"},
    )
    assert peek.status_code == 403
    write = await harness.client.post(
        f"/rooms/{room['id']}/messages",
        json={"author_id": human["id"], "body": "intruder"},
        headers={"Authorization": f"Bearer {bob['access_token']}"},
    )
    assert write.status_code == 403


@pytest.mark.asyncio
async def test_computer_token_is_not_a_user_jwt(harness: Harness) -> None:
    alice = await _login(harness.client, "alice")
    computer = (
        await harness.client.post(
            "/computers",
            json={"name": "laptop"},
            headers={"Authorization": f"Bearer {alice['access_token']}"},
        )
    ).json()
    as_host = await harness.client.post(
        "/rooms",
        json={"name": "nope"},
        headers={"Authorization": f"Bearer {computer['token']}"},
    )
    assert as_host.status_code == 401
    listed = await harness.client.get(
        "/computers",
        headers={"Authorization": f"Bearer {alice['access_token']}"},
    )
    assert [c["name"] for c in listed.json()] == ["laptop"]


@pytest.mark.asyncio
async def test_user_jwt_cannot_call_runtime(harness: Harness) -> None:
    alice = await _login(harness.client, "alice")
    room = (
        await harness.client.post(
            "/rooms",
            json={"name": "r"},
            headers={"Authorization": f"Bearer {alice['access_token']}"},
        )
    ).json()
    agent = (
        await harness.client.post(
            f"/rooms/{room['id']}/participants",
            json={"kind": "agent", "name": "Iris"},
            headers={"Authorization": f"Bearer {alice['access_token']}"},
        )
    ).json()
    runtime = await harness.client.get(
        "/runtime/turn-context",
        params={"agent_id": agent["id"], "room_id": room["id"]},
        headers={"Authorization": f"Bearer {alice['access_token']}"},
    )
    assert runtime.status_code == 401


@pytest.mark.asyncio
async def test_room_ws_requires_owner_jwt(harness: Harness) -> None:
    alice = await _login(harness.client, "alice")
    room = (
        await harness.client.post(
            "/rooms",
            json={"name": "ws"},
            headers={"Authorization": f"Bearer {alice['access_token']}"},
        )
    ).json()
    with pytest.raises(ConnectionError, match="1008"):
        await connect_asgi_ws(harness.app, f"/ws/rooms/{room['id']}")
    bob = await _login(harness.client, "bob")
    with pytest.raises(ConnectionError, match="1008"):
        await connect_asgi_ws(
            harness.app,
            f"/ws/rooms/{room['id']}",
            query_string=f"access_token={bob['access_token']}",
        )
    ws = await connect_asgi_ws(
        harness.app,
        f"/ws/rooms/{room['id']}",
        query_string=f"access_token={alice['access_token']}",
    )
    await ws.close()


@pytest.mark.asyncio
async def test_second_login_same_github_id_reuses_user(harness: Harness) -> None:
    first = await _login(harness.client, "alice")
    second = await _login(harness.client, "alice")
    assert first["user"]["id"] == second["user"]["id"]


@pytest.mark.asyncio
async def test_auth_off_keeps_anonymous_curl_path(require_services: None) -> None:
    app = create_app(stub_turns=True, settings=Settings())
    async with app.router.lifespan_context(app):
        await truncate_all(app.state.pool)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            missing = await client.get("/auth/github")
            assert missing.status_code == 404
            room = await client.post("/rooms", json={"name": "anon"})
            assert room.status_code == 200
            assert room.json()["created_by"] is None


@pytest.mark.asyncio
async def test_auth_enabled_without_secrets_refuses_to_start() -> None:
    app = create_app(
        stub_turns=True,
        settings=Settings(github_client_id="client", github_client_secret="", jwt_secret=""),
    )
    with pytest.raises(RuntimeError, match="JWT_SECRET"):
        async with app.router.lifespan_context(app):
            pass


def test_issue_user_token_sub_is_uuid() -> None:
    user_id = uuid4()
    token = issue_user_token(user_id, "alice", "jwt-secret", 60)
    assert decode_jwt(token, "jwt-secret")["sub"] == str(user_id)
