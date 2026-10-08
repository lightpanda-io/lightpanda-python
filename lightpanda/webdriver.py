"""WebDriver: run ``lightpanda serve --protocol webdriver`` for Selenium and co.

:class:`WebDriverServer` spawns the browser's WebDriver server (classic and BiDi)
on a free localhost port and owns the process; :class:`AsyncWebDriverServer` is the
asyncio twin. See :class:`WebDriverServer` for what the browser serves.
"""

from __future__ import annotations

import asyncio

from ._serve import _AsyncServeProcess, _ServeProcess
from .client import _HOST, _documented


@_documented
class WebDriverServer(_ServeProcess):
    """A lightpanda process serving WebDriver, classic and BiDi, on 127.0.0.1.

    ```python
    from lightpanda import WebDriverServer
    from selenium import webdriver
    from selenium.webdriver.common.by import By

    with WebDriverServer() as server:
        # Selenium requires an options object; Lightpanda accepts any browser's.
        driver = webdriver.Remote(command_executor=server.http_endpoint, options=webdriver.ChromeOptions())
        driver.get("https://example.com")
        print(driver.find_element(By.CSS_SELECTOR, "h1").text)
        driver.quit()
    ```

    :attr:`http_endpoint` is Selenium's ``command_executor``. The browser
    serves classic WebDriver over HTTP (navigation, element lookup and
    interaction, script execution, actions, cookies) and, for a session
    created with the ``webSocketUrl`` capability (Selenium's
    ``options.web_socket_url = True``), the BiDi modules (``session``,
    ``browser``, ``browsingContext``, ``script``, ``input``) over the
    WebSocket, reached through ``driver.browsing_context`` and
    ``driver.script``. Pass ``args=["--protocol", "cdp"]`` to serve CDP on the
    same port as well (``--protocol`` is additive). The process is stopped by
    :meth:`close` / leaving the ``with`` block, and on Linux also when the
    interpreter dies.
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
class AsyncWebDriverServer(_AsyncServeProcess[WebDriverServer]):
    """:class:`WebDriverServer` for asyncio: the process is spawned by
    :meth:`start`, called automatically on ``async with`` entry."""

    _sync_cls = WebDriverServer

    @property
    def bidi_endpoint(self) -> str:
        """See :attr:`WebDriverServer.bidi_endpoint`."""
        return self._started().bidi_endpoint

    async def status(self) -> dict:
        """See :meth:`WebDriverServer.status`."""
        return await asyncio.to_thread(self._started().status)


__all__ = ["WebDriverServer", "AsyncWebDriverServer"]
