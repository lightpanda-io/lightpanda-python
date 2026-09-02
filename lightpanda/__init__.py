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
(see its docs for an example). For Selenium, ``BiDiServer`` serves WebDriver
BiDi the same way and hands you the ``command_executor`` URL.
"""

from .async_browser import AsyncBrowser, AsyncSession, run_script_async
from .bidi import AsyncBiDiServer, BiDiServer
from .browser import Browser, Session, run_script
from .cdp import AsyncCDPServer, CDPServer
from .errors import LightpandaError, ProcessError, ProtocolError, ScriptError, ToolError

__all__ = [
    "Browser",
    "Session",
    "run_script",
    "AsyncBrowser",
    "AsyncSession",
    "run_script_async",
    "CDPServer",
    "AsyncCDPServer",
    "BiDiServer",
    "AsyncBiDiServer",
    "LightpandaError",
    "ProcessError",
    "ProtocolError",
    "ScriptError",
    "ToolError",
]
