"""Upstream forwarding.

The read timeout scales with the requested generation length. A flat timeout
would make the gateway itself abort legitimately expensive prompts, and those
aborts would show up in the analysis as mitigation-induced failures.
"""

from __future__ import annotations

from typing import AsyncIterator

import httpx

from .config import UpstreamConfig

# Stripped in both directions; these describe a single hop, not the payload.
HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "content-length",
        "host",
    }
)


def filter_headers(headers) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP}


class UpstreamClient:
    def __init__(self, config: UpstreamConfig, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._config = config
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        limits = httpx.Limits(
            max_connections=self._config.max_connections,
            max_keepalive_connections=self._config.max_connections,
        )
        self._client = httpx.AsyncClient(
            base_url=self._config.base_url,
            limits=limits,
            transport=self._transport,
            timeout=httpx.Timeout(
                connect=self._config.connect_timeout_s,
                read=self._config.read_timeout_base_s,
                write=self._config.connect_timeout_s,
                pool=self._config.connect_timeout_s,
            ),
        )

    async def stop(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("UpstreamClient.start() was not awaited")
        return self._client

    def read_timeout_for(self, max_tokens: int) -> float:
        config = self._config
        return min(
            config.read_timeout_max_s,
            config.read_timeout_base_s + config.read_timeout_per_token_s * max_tokens,
        )

    def _timeout(self, max_tokens: int) -> httpx.Timeout:
        return httpx.Timeout(
            connect=self._config.connect_timeout_s,
            read=self.read_timeout_for(max_tokens),
            write=self._config.connect_timeout_s,
            pool=self._config.connect_timeout_s,
        )

    async def request(
        self,
        method: str,
        path: str,
        *,
        headers: dict[str, str] | None = None,
        content: bytes | None = None,
        max_tokens: int = 0,
    ) -> httpx.Response:
        return await self.client.request(
            method,
            path,
            headers=headers,
            content=content,
            timeout=self._timeout(max_tokens),
        )

    async def stream(
        self,
        method: str,
        path: str,
        *,
        headers: dict[str, str] | None = None,
        content: bytes | None = None,
        max_tokens: int = 0,
    ) -> AsyncIterator[tuple[httpx.Response, AsyncIterator[bytes]]]:
        """Yield the response and its body iterator without buffering it."""
        request = self.client.build_request(
            method,
            path,
            headers=headers,
            content=content,
            timeout=self._timeout(max_tokens),
        )
        response = await self.client.send(request, stream=True)
        yield response, response.aiter_raw()
