"""Shared shape of the ``lightpanda serve`` wrappers (``CDPServer``, ``BiDiServer``).

A serve wrapper owns one ``lightpanda serve`` process on a localhost port
and hands out endpoints for third-party clients; the subclasses differ only
in the ``--protocol`` flags they pass and the endpoints they expose.
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.request
from collections.abc import Sequence
from typing import Generic, TypeVar

from .client import _HOST, _spawn, _terminate, find_binary
from .errors import LightpandaError

_HTTP_TIMEOUT = 5.0


class _ServeProcess:
    _protocol: tuple[str, ...] = ()  # ``serve`` flags placed before the user's ``args``

    def __init__(
        self,
        binary: str | os.PathLike | None = None,
        env: dict[str, str] | None = None,
        verbose: bool = False,
        args: Sequence[str] = (),
        port: int | None = None,
    ):
        """Spawn the server process.

        Args:
            binary: Path to a lightpanda binary. When omitted, resolved from
                the ``LIGHTPANDA_BIN`` environment variable, then the binary
                bundled in the package, then ``PATH``.
            env: Extra environment variables for the spawned process.
            verbose: Let the browser's own logging through to stderr.
            args: Extra ``lightpanda serve`` flags; pass ``port=`` rather
                than ``--port``.
            port: Pin the listening port. Defaults to a free one.
        """
        self._proc, self._port = _spawn(
            find_binary(binary), "serve", [*self._protocol, *args], env, verbose, port=port
        )

    @property
    def port(self) -> int:
        """The port the server listens on."""
        return self._port

    @property
    def http_endpoint(self) -> str:
        """``http://127.0.0.1:<port>``, the server's HTTP root: what Puppeteer
        (``browserURL``) and Playwright (``connect_over_cdp`` with an http URL)
        discover the CDP WebSocket from, and Selenium's ``command_executor``."""
        return f"http://{_HOST}:{self._port}"

    def _get_json(self, path: str) -> dict:
        """GET ``path`` and parse the JSON body. urllib sends an IP-literal
        ``Host`` and no ``Origin``, which is what the browser accepts."""
        if self._proc is None:
            raise LightpandaError("server closed")
        url = f"{self.http_endpoint}{path}"
        try:
            with urllib.request.urlopen(url, timeout=_HTTP_TIMEOUT) as resp:
                return json.loads(resp.read())
        except (OSError, ValueError) as err:  # URLError is an OSError
            raise LightpandaError(f"GET {url} failed: {err}") from err

    def close(self) -> None:
        """Stop the server process. Idempotent."""
        if self._proc is not None:
            _terminate(self._proc)
            self._proc = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


_S = TypeVar("_S", bound=_ServeProcess)


class _AsyncServeProcess(Generic[_S]):
    """asyncio twin of a :class:`_ServeProcess`: the process is spawned by
    :meth:`start`, called automatically on ``async with`` entry. Subclasses
    set ``_sync_cls`` and re-declare their protocol-specific members."""

    _sync_cls: type[_S]

    def __init__(
        self,
        binary: str | os.PathLike | None = None,
        env: dict[str, str] | None = None,
        verbose: bool = False,
        args: Sequence[str] = (),
        port: int | None = None,
    ):
        """Prepare the facade; the process is spawned by :meth:`start`.

        Args:
            binary: Forwarded to the sync class.
            env: Forwarded to the sync class.
            verbose: Forwarded to the sync class.
            args: Forwarded to the sync class.
            port: Forwarded to the sync class.
        """
        self._kwargs = dict(binary=binary, env=env, verbose=verbose, args=args, port=port)
        self._server: _S | None = None
        self._start_lock = asyncio.Lock()

    async def start(self):
        """Spawn the server process. Idempotent."""
        async with self._start_lock:
            if self._server is None:
                self._server = await asyncio.to_thread(self._sync_cls, **self._kwargs)
        return self

    def _started(self) -> _S:
        if self._server is None:
            raise LightpandaError("server not started; use `async with` or `await start()`")
        return self._server

    @property
    def port(self) -> int:
        """The port the server listens on."""
        return self._started().port

    @property
    def http_endpoint(self) -> str:
        """``http://127.0.0.1:<port>``, see the sync class."""
        return self._started().http_endpoint

    async def close(self) -> None:
        """Stop the server process. Idempotent."""
        if self._server is not None:
            server, self._server = self._server, None
            await asyncio.to_thread(server.close)

    async def __aenter__(self):
        return await self.start()

    async def __aexit__(self, *exc):
        await self.close()

