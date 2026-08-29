"""GitHub OAuth for humans. The host plane does not use this.

Login is fail-closed: a bad state, a failed token exchange, or a
missing user payload is 400/401, not a guest session. That is the
opposite of a Redis wake. Coordination can drop; admission cannot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlencode

import httpx

from server.config import Settings


@dataclass(frozen=True)
class GitHubUser:
    github_id: int
    login: str
    name: str | None
    avatar_url: str | None


class GitHubClient(Protocol):
    async def exchange_code(self, code: str) -> str: ...

    async def fetch_user(self, access_token: str) -> GitHubUser: ...


class HttpGitHub:
    def __init__(self, settings: Settings, *, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._client = client

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("HttpGitHub has no client; bind one in lifespan")
        return self._client

    async def exchange_code(self, code: str) -> str:
        resp = await self._http().post(
            self.settings.github_token_url,
            headers={"Accept": "application/json"},
            data={
                "client_id": self.settings.github_client_id,
                "client_secret": self.settings.github_client_secret,
                "code": code,
                "redirect_uri": self.settings.oauth_redirect_uri,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        token = data.get("access_token")
        if not token:
            raise RuntimeError(data.get("error_description") or "github token exchange failed")
        return str(token)

    async def fetch_user(self, access_token: str) -> GitHubUser:
        resp = await self._http().get(
            self.settings.github_user_url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {access_token}",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        github_id = data.get("id")
        login = data.get("login")
        if github_id is None or not login:
            raise RuntimeError("github user payload missing id/login")
        return GitHubUser(
            github_id=int(github_id),
            login=str(login),
            name=data.get("name"),
            avatar_url=data.get("avatar_url"),
        )


def authorize_url(settings: Settings, state: str) -> str:
    query = urlencode(
        {
            "client_id": settings.github_client_id,
            "redirect_uri": settings.oauth_redirect_uri,
            "scope": "read:user",
            "state": state,
        }
    )
    return f"{settings.github_authorize_url}?{query}"
