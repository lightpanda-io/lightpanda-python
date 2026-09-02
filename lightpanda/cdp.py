"""Chrome DevTools Protocol: run ``lightpanda serve`` and hand out its endpoint.

The browser has a native CDP server (``lightpanda serve``): a WebSocket on
``/`` plus the ``/json/version`` discovery endpoint Chrome exposes. Anything
that speaks CDP connects to it directly — Playwright (``connect_over_cdp``),
Puppeteer (``connect({browserWSEndpoint})``), chromedp, cdp-use — with no
Chromium involved. :class:`CDPServer` spawns that server on a free localhost
port and owns the process; :class:`AsyncCDPServer` is the asyncio twin.

This is a separate process from :class:`lightpanda.Browser`: the binary
cannot serve MCP and CDP from one process, and MCP sessions and CDP
browser contexts are unrelated anyway.
"""

from __future__ import annotations

import asyncio
import json
import os
import urllib.request

from .client import _HOST, _spawn, _terminate, find_binary
from .errors import LightpandaError

_VERSION_TIMEOUT = 5.0


def _get_version(port: int) -> dict:
    """GET ``/json/version``. urllib sends an IP-literal ``Host`` and no
    ``Origin``, which is what the browser's handshake accepts."""
    url = f"http://{_HOST}:{port}/json/version"
    try:
        with urllib.request.urlopen(url, timeout=_VERSION_TIMEOUT) as resp:
            return json.loads(resp.read())
    except (OSError, ValueError) as err:  # URLError is an OSError
        raise LightpandaError(f"GET {url} failed: {err}") from err


class CDPServer:
    """A lightpanda process serving the Chrome DevTools Protocol on 127.0.0.1.

    ```python
    from lightpanda import CDPServer
    from playwright.sync_api import sync_playwright

    with CDPServer() as server, sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(server.ws_endpoint)
        page = browser.new_context().new_page()
        page.goto("https://example.com")
    ```

    Every connected client gets its own browser; up to 16 connect at once
    by default (``args=["--cdp-max-connections", "N"]`` to change). The
    process is stopped by :meth:`close` / leaving the ``with`` block, and on
    Linux also when the interpreter dies.
    """

    def __init__(
        self,
        binary: str | os.PathLike | None = None,
        env: dict[str, str] | None = None,
        verbose: bool = False,
        args: tuple[str, ...] | list[str] = (),
        port: int | None = None,
    ):
        """``port`` pins the listening port (default: a free one). ``args``
        are extra ``lightpanda serve`` flags (``--http-proxy``, ``--cookie``,
        ``--cdp-max-connections``, ...); pass ``port=`` rather than
        ``--port``. ``verbose`` lets the browser log through to stderr."""
        self._proc, self._port = _spawn(find_binary(binary), "serve", args, env, verbose, port=port)

    @property
    def port(self) -> int:
        return self._port

    @property
    def ws_endpoint(self) -> str:
        """The CDP WebSocket URL, ``ws://127.0.0.1:<port>/``.

        Keep it as is: the server only upgrades on path ``/`` and only
        accepts an IP-literal or ``localhost`` host."""
        return f"ws://{_HOST}:{self._port}/"

    @property
    def http_endpoint(self) -> str:
        """``http://127.0.0.1:<port>``, for clients that discover the
        WebSocket through ``/json/version`` (Puppeteer's ``browserURL``,
        Playwright's ``connect_over_cdp`` with an http URL)."""
        return f"http://{_HOST}:{self._port}"

    def version(self) -> dict:
        """The ``/json/version`` document (browser, protocol version,
        ``webSocketDebuggerUrl``)."""
        if self._proc is None:
            raise LightpandaError("server closed")
        return _get_version(self._port)

    def close(self) -> None:
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


class AsyncCDPServer:
    """:class:`CDPServer` for asyncio: the process is spawned by
    :meth:`start`, called automatically on ``async with`` entry.

    ```python
    async with AsyncCDPServer() as server, async_playwright() as p:
        browser = await p.chromium.connect_over_cdp(server.ws_endpoint)
    ```
    """

    def __init__(
        self,
        binary: str | os.PathLike | None = None,
        env: dict[str, str] | None = None,
        verbose: bool = False,
        args: tuple[str, ...] | list[str] = (),
        port: int | None = None,
    ):
        """Arguments are forwarded to :class:`CDPServer`."""
        self._kwargs = dict(binary=binary, env=env, verbose=verbose, args=args, port=port)
        self._server: CDPServer | None = None
        self._start_lock = asyncio.Lock()

    async def start(self) -> AsyncCDPServer:
        """Spawn the server process. Idempotent."""
        if self._server is None:
            async with self._start_lock:
                if self._server is None:
                    self._server = await asyncio.to_thread(CDPServer, **self._kwargs)
        return self

    def _started(self) -> CDPServer:
        if self._server is None:
            raise LightpandaError("server not started; use `async with` or `await start()`")
        return self._server

    @property
    def port(self) -> int:
        return self._started().port

    @property
    def ws_endpoint(self) -> str:
        """See :attr:`CDPServer.ws_endpoint`."""
        return self._started().ws_endpoint

    @property
    def http_endpoint(self) -> str:
        """See :attr:`CDPServer.http_endpoint`."""
        return self._started().http_endpoint

    async def version(self) -> dict:
        """See :meth:`CDPServer.version`."""
        return await asyncio.to_thread(self._started().version)

    async def close(self) -> None:
        if self._server is not None:
            server, self._server = self._server, None
            await asyncio.to_thread(server.close)

    async def __aenter__(self):
        return await self.start()

    async def __aexit__(self, *exc):
        await self.close()


__all__ = ["CDPServer", "AsyncCDPServer"]
