"""HS256 JWT for the human admission layer. No third-party JWT library.

Host tokens (computer / cluster) never go through this module. A JWT
cannot satisfy /runtime/*; a computer token cannot satisfy /auth/me.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any
from uuid import UUID


class JWTError(ValueError):
    pass


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


def encode_jwt(claims: dict[str, Any], secret: str) -> str:
    header = _b64url_encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload = _b64url_encode(
        json.dumps(claims, separators=(",", ":"), default=str).encode()
    )
    signing = f"{header}.{payload}".encode()
    sig = hmac.new(secret.encode("utf-8"), signing, hashlib.sha256).digest()
    return f"{header}.{payload}.{_b64url_encode(sig)}"


def decode_jwt(token: str, secret: str) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) != 3:
        raise JWTError("malformed token")
    header_b, payload_b, sig_b = parts
    signing = f"{header_b}.{payload_b}".encode()
    expected = hmac.new(secret.encode("utf-8"), signing, hashlib.sha256).digest()
    try:
        given = _b64url_decode(sig_b)
    except Exception as exc:
        raise JWTError("malformed signature") from exc
    if not hmac.compare_digest(expected, given):
        raise JWTError("invalid signature")
    try:
        header = json.loads(_b64url_decode(header_b))
        claims = json.loads(_b64url_decode(payload_b))
    except Exception as exc:
        raise JWTError("malformed payload") from exc
    if header.get("alg") != "HS256":
        raise JWTError("unsupported alg")
    exp = claims.get("exp")
    if exp is not None and int(exp) <= int(time.time()):
        raise JWTError("expired")
    return claims


def issue_user_token(user_id: UUID, login: str, secret: str, ttl_s: int) -> str:
    now = int(time.time())
    return encode_jwt(
        {
            "iss": "agora",
            "sub": str(user_id),
            "login": login,
            "iat": now,
            "exp": now + ttl_s,
        },
        secret,
    )


def make_oauth_state(secret: str) -> str:
    raw = f"{int(time.time())}:{_b64url_encode(hashlib.sha256(str(time.time_ns()).encode()).digest()[:12])}"
    sig = hmac.new(secret.encode("utf-8"), raw.encode(), hashlib.sha256).hexdigest()
    return f"{raw}:{sig}"


def check_oauth_state(state: str, secret: str, *, max_age_s: int = 600) -> bool:
    parts = state.rsplit(":", 2)
    if len(parts) != 3:
        return False
    ts, nonce, sig = parts
    raw = f"{ts}:{nonce}"
    expected = hmac.new(secret.encode("utf-8"), raw.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig):
        return False
    try:
        age = time.time() - int(ts)
    except ValueError:
        return False
    return 0 <= age <= max_age_s
