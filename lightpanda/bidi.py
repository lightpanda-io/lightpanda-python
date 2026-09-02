"""WebDriver BiDi: run ``lightpanda serve --protocol webdriver`` for Selenium and co.

:class:`BiDiServer` spawns the browser's WebDriver BiDi server on a free
localhost port and owns the process; :class:`AsyncBiDiServer` is the asyncio
twin. See :class:`BiDiServer` for what the browser serves.
"""

from __future__ import annotations

import asyncio

from ._serve import _AsyncServeProcess, _ServeProcess
from .client import _HOST, _documented


@_documented
class BiDiServer(_ServeProcess):
    """A lightpanda process serving WebDriver BiDi on 127.0.0.1.

    ```python
    from lightpanda import BiDiServer
    from selenium import webdriver
    from selenium.webdriver.common.options import ArgOptions

    options = ArgOptions()
    options.web_socket_url = True  # ask for a WebDriver BiDi session

    with BiDiServer() as server:
        driver = webdriver.Remote(command_executor=server.http_endpoint, options=options)
        context = driver.browsing_context.create(type="tab")
        driver.browsing_context.navigate(context=context, url="https://example.com", wait="complete")
        print(driver.script.execute("() => document.title", context_id=context)["value"])
        driver.quit()
    ```

    :attr:`http_endpoint` is Selenium's ``command_executor``. The browser
    serves the BiDi modules (``session``, ``browser``, ``browsingContext``,
    ``script``, ``input``) over the WebSocket plus the classic session
    bootstrap (``GET /status``, ``POST /session`` with the ``webSocketUrl``
    capability, ``DELETE /session/<id>``); other classic WebDriver commands
    such as Selenium's ``driver.get`` or ``find_element`` are not served, so
    drive the page through ``driver.browsing_context`` and ``driver.script``
    with an explicit context, created first as above. Pass
    ``args=["--protocol", "cdp"]`` to serve CDP on the same port as well
    (``--protocol`` is additive). The process is stopped by :meth:`close` /
    leaving the ``with`` block, and on Linux also when the interpreter dies.
    """

    _protocol = ("--protocol", "webdriver")

    @property
    def bidi_endpoint(self) -> str:
        """The session-less BiDi WebSocket URL, ``ws://127.0.0.1:<port>/session``,
        for clients that speak BiDi directly (``session.new`` over the socket).
        A session bootstrapped through ``POST /session`` gets its own socket at
        ``<bidi_endpoint>/<sessionId>``, returned as the ``webSocketUrl``
        capability.

        Keep the IP literal: the WebSocket upgrade rejects any ``Origin``
        header and only accepts an IP-literal or ``localhost`` host."""
        return f"ws://{_HOST}:{self._port}/session"

    def status(self) -> dict:
        """The ``GET /status`` value, ``{"ready": True, "message": ""}``."""
        return self._get_json("/status")["value"]


@_documented
class AsyncBiDiServer(_AsyncServeProcess[BiDiServer]):
    """:class:`BiDiServer` for asyncio: the process is spawned by
    :meth:`start`, called automatically on ``async with`` entry."""

    _sync_cls = BiDiServer

    @property
    def bidi_endpoint(self) -> str:
        """See :attr:`BiDiServer.bidi_endpoint`."""
        return self._started().bidi_endpoint

    async def status(self) -> dict:
        """See :meth:`BiDiServer.status`."""
        return await asyncio.to_thread(self._started().status)


__all__ = ["BiDiServer", "AsyncBiDiServer"]
