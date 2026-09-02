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

from ._serve import _AsyncServeProcess, _ServeProcess
from .client import _HOST, _documented


@_documented
class CDPServer(_ServeProcess):
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

    _protocol = ("--protocol", "cdp")  # explicit, so args=["--protocol", "webdriver"] is additive

    @property
    def ws_endpoint(self) -> str:
        """The CDP WebSocket URL, ``ws://127.0.0.1:<port>/``.

        Keep it as is: the server only upgrades on path ``/`` and only
        accepts an IP-literal or ``localhost`` host."""
        return f"ws://{_HOST}:{self._port}/"

    def version(self) -> dict:
        """The ``/json/version`` document (browser, protocol version,
        ``webSocketDebuggerUrl``)."""
        return self._get_json("/json/version")


@_documented
class AsyncCDPServer(_AsyncServeProcess[CDPServer]):
    """:class:`CDPServer` for asyncio: the process is spawned by
    :meth:`start`, called automatically on ``async with`` entry.

    ```python
    async with AsyncCDPServer() as server, async_playwright() as p:
        browser = await p.chromium.connect_over_cdp(server.ws_endpoint)
    ```
    """

    _sync_cls = CDPServer

    @property
    def ws_endpoint(self) -> str:
        """See :attr:`CDPServer.ws_endpoint`."""
        return self._started().ws_endpoint

    async def version(self) -> dict:
        """See :meth:`CDPServer.version`."""
        return await asyncio.to_thread(self._started().version)


__all__ = ["CDPServer", "AsyncCDPServer"]
