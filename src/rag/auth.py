"""Identify the user behind a request from a Supabase access token."""

import logging
import time
from dataclasses import dataclass
from typing import Protocol

import httpx

logger = logging.getLogger(__name__)

_CACHE_SECONDS = 60
_TIMEOUT_SECONDS = 10


@dataclass(frozen=True, slots=True)
class User:
    id: str
    email: str = ""


# Used when authentication is not configured: a single-user installation.
LOCAL_USER = User(id="local")


class AuthUnavailable(RuntimeError):
    """The identity provider could not be reached, so the token is neither valid nor invalid."""


class Authenticator(Protocol):
    def verify(self, token: str) -> User | None:
        """Return the user a token belongs to, or ``None`` if it is not valid."""
        ...


class SupabaseAuthenticator:
    """Asks Supabase who a token belongs to.

    Supabase stays the single place that decides whether a session is valid, which
    also covers sign-outs and banned users. Answers are cached briefly so that one
    page of requests costs one lookup.
    """

    def __init__(self, url: str, anon_key: str, client: httpx.Client | None = None) -> None:
        self._endpoint = f"{url.rstrip('/')}/auth/v1/user"
        self._anon_key = anon_key
        self._client = client or httpx.Client(timeout=_TIMEOUT_SECONDS)
        self._cache: dict[str, tuple[float, User]] = {}

    def verify(self, token: str) -> User | None:
        now = time.monotonic()
        cached = self._cache.get(token)
        if cached and cached[0] > now:
            return cached[1]

        try:
            response = self._client.get(
                self._endpoint,
                headers={"apikey": self._anon_key, "Authorization": f"Bearer {token}"},
            )
        except httpx.HTTPError as error:
            raise AuthUnavailable("Could not reach the authentication service.") from error
        if response.status_code in (401, 403):
            self._cache.pop(token, None)
            return None
        if response.status_code != 200:
            raise AuthUnavailable(f"Authentication service returned HTTP {response.status_code}.")

        body = response.json()
        user = User(id=body["id"], email=body.get("email") or "")
        if len(self._cache) > 1000:
            self._cache.clear()
        self._cache[token] = (now + _CACHE_SECONDS, user)
        return user
