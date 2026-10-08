"""Lightpanda for Python: a lightweight headless browser.

```python
from lightpanda import Browser

with Browser() as b:
    page = b.new_session()
    page.goto(url="https://example.com")
    data = page.extract(schema={"title": "h1"})
```

The same API is available for asyncio:

```python
from lightpanda import AsyncBrowser

async with AsyncBrowser() as b:
    page = await b.new_session()
    await page.goto(url="https://example.com")
    data = await page.extract(schema={"title": "h1"})
```

For Playwright or Puppeteer code, ``CDPServer`` runs the browser's own
Chrome DevTools Protocol server and hands you the endpoint to connect to
(see its docs for an example). For Selenium, ``WebDriverServer`` serves
WebDriver, classic and BiDi, the same way and hands you the
``command_executor`` URL.

The browser can enforce ``robots.txt``, which is off by default. Pass
``args=["--obey-robots"]`` to any of these and a disallowed request fails
instead of being sent, surfacing as whatever your client raises: a
:class:`ToolError` here, a navigation error in Playwright or Selenium. Note
that the rule applies to every request, not only the page you asked for, so a
permitted page whose assets sit under a disallowed path will load without them.
"""

__docformat__ = "google"

from .async_browser import AsyncBrowser, AsyncSession, run_script_async
from .webdriver import AsyncWebDriverServer, WebDriverServer
from .browser import Browser, PageResult, Session, run_script
from .cdp import AsyncCDPServer, CDPServer
from .errors import LightpandaError, ProcessError, ProtocolError, ScriptError, ToolError

__all__ = [
    "Browser",
    "Session",
    "PageResult",
    "run_script",
    "AsyncBrowser",
    "AsyncSession",
    "run_script_async",
    "CDPServer",
    "AsyncCDPServer",
    "WebDriverServer",
    "AsyncWebDriverServer",
    "LightpandaError",
    "ProcessError",
    "ProtocolError",
    "ScriptError",
    "ToolError",
]
