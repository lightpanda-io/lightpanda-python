# lightpanda-python

Lightpanda for Python: scrape and automate the web with `pip install lightpanda`. Browser included, no separate download.

[Lightpanda](https://lightpanda.io) is an open-source headless browser built
for web scraping, automation, and AI agents. It executes JavaScript and renders
pages like a full browser, but starts in milliseconds and uses an order of
magnitude less memory than Chrome stacks. The wheel bundles the browser binary;
there is no second install step.

```bash
uv add lightpanda        # or: uv pip install lightpanda / pip install lightpanda
```

`example.py`:

```python
from lightpanda import Browser

with Browser() as b:
    page = b.new_session()
    page.goto(url="https://example.com")
    data = page.extract(schema={"title": "h1"})
    print(data)
```

Run it like any Python script — no driver to install, no browser to download:

```bash
$ python example.py
{'title': 'Example Domain'}
```

The same API is available for asyncio — `AsyncBrowser` spawns the browser on
first use and runs sessions concurrently. `async_example.py`:

```python
import asyncio

from lightpanda import AsyncBrowser


async def main():
    async with AsyncBrowser() as b:
        page = await b.new_session()
        await page.goto(url="https://example.com")
        data = await page.extract(schema={"title": "h1"})
        print(data)


asyncio.run(main())
```

```bash
python async_example.py
```

The package also puts the full `lightpanda` CLI on PATH — agent REPL, fetch,
serve, and the rest: see the
[command reference](https://lightpanda.io/docs/run-locally/commands).

## Drive it with Playwright or Puppeteer

Lightpanda has its own Chrome DevTools Protocol server, so existing
Playwright/Puppeteer code works against it without Chromium. `CDPServer`
starts `lightpanda serve` on a free localhost port and hands you the
endpoint:

```python
from lightpanda import CDPServer
from playwright.sync_api import sync_playwright

with CDPServer() as server, sync_playwright() as p:
    browser = p.chromium.connect_over_cdp(server.ws_endpoint)
    page = browser.new_context().new_page()
    page.goto("https://example.com")
    print(page.title())
```

`AsyncCDPServer` is the asyncio twin (`async with AsyncCDPServer() as server`).
`server.ws_endpoint` is a plain CDP WebSocket, so Node clients connect to it
too: Puppeteer with `puppeteer.connect({ browserWSEndpoint })` and Playwright
with `chromium.connectOverCDP(...)`. Pass `port=` to pin the port, and
`args=` for `lightpanda serve` flags such as `--cdp-max-connections` or
`--http-proxy`. This is a separate process from `Browser` (the binary
cannot serve MCP and CDP from the same one). The package itself needs only
Python's standard library; Playwright is a dev-only test dependency and
`connect_over_cdp` never downloads a browser.

## Drive it with Selenium (WebDriver BiDi)

The browser also speaks [WebDriver BiDi](https://w3c.github.io/webdriver-bidi/).
`BiDiServer` starts `lightpanda serve --protocol webdriver` and hands you the
URL Selenium's `webdriver.Remote` takes as `command_executor`:

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

`AsyncBiDiServer` is the asyncio twin, and `server.bidi_endpoint`
(`ws://127.0.0.1:<port>/session`) is the raw BiDi WebSocket for clients that
speak the protocol directly. The browser serves the BiDi modules plus the
classic session bootstrap Selenium needs, not the classic WebDriver
commands: `driver.get`, `find_element` and friends are not implemented, so
drive the page through `driver.browsing_context` and `driver.script` with an
explicit context, created first as above. Pass `args=["--protocol", "cdp"]`
to serve CDP on the same port as well.

## How the bindings work

Every browser tool is a `Session` method, typed and documented in your IDE.
The methods are not written by hand: they are generated from the bundled
browser's MCP tool schemas (`scripts/generate_methods.py` →
`lightpanda/_methods.py`), so signatures and docstrings come straight from the
binary, with tool and parameter names in snake_case (`waitForSelector` →
`wait_for_selector`, `backendNodeId` → `backend_node_id`). `Session.call` is
the escape hatch that takes the raw tool and parameter names as the MCP
server declares them. The full API reference is published at
[lightpanda.io/lightpanda-python](https://lightpanda.io/lightpanda-python/).

The bindings follow Lightpanda's development and the package version tracks
browser releases — there is no backwards-compatibility guarantee: when the
browser's tools change, the Python methods change with them.

## Examples

[`examples/`](examples/) scrapes a JavaScript-rendered site that `requests`
cannot read, analyses the result with pandas/matplotlib, and measures the same
job against Selenium + headless Chrome (100 quotes in ~3 s / 35 MB vs ~6 s /
1.2 GB, median of 5 runs). Each script runs standalone: `uv run examples/quotes_analysis.py`.

## License

This client library is Apache-2.0. The bundled Lightpanda browser binary is
licensed separately under AGPL-3.0 — see
[lightpanda-io/browser](https://github.com/lightpanda-io/browser).

## Development

The runtime binary is resolved from `LIGHTPANDA_BIN`, the package directory,
then PATH. For development, clone
[lightpanda-io/browser](https://github.com/lightpanda-io/browser) as a sibling
checkout and build it (`zig build`) — or point `LIGHTPANDA_BIN` at any
lightpanda binary. Then:

```bash
uv run --group dev pytest tests
```

The `dev` group includes `playwright` and `selenium` as clients for the CDP
and BiDi tests (their pip packages only; no browser or driver download).
Those tests skip when the client is absent.

Regenerate the tool methods (`lightpanda/_methods.py`) and the API docs:

```bash
uv run --no-project python scripts/generate_methods.py
uv run --no-project --with pdoc python scripts/build_docs.py   # writes docs/
```

## Building a wheel

With a binary available (`LIGHTPANDA_BIN`, or the sibling checkout built
`ReleaseFast`):

```bash
uv build --wheel                                    # -> dist/lightpanda-*-py3-none-<plat>.whl
LIGHTPANDA_PLAT=manylinux_2_35_x86_64 uv build --wheel   # CI: tag for the release runner's glibc
```

The wheel is platform-specific (it carries the binary) but works on every
Python ≥3.10 — hence the `py3-none-<plat>` tag. Release wheels must bundle a
`ReleaseFast` binary; a debug build works but is several times larger and
slower.

## Releasing

CI (`.github/workflows/wheels.yml`) builds and tests wheels for all four
platforms on every pull request and push to `main`, bundling the browser's
`nightly` release.

Publishing is automatic: when the browser repo builds a version release, its
release workflow (the `update-python-package` job in
[browser's `release.yml`](https://github.com/lightpanda-io/browser/blob/main/.github/workflows/release.yml))
creates the matching release here. That release event starts a wheel run that
builds from that browser release, derives the wheel version from the tag,
tests on all platforms, and publishes to PyPI via trusted publishing — after a
maintainer approves the `pypi` environment deployment on the run page. A
release can also be triggered by hand: create a GitHub release here whose tag
matches a browser release tag, or use the `workflow_dispatch` path (any
browser tag, publish to TestPyPI or PyPI) for dry runs.

The same release event runs `.github/workflows/docs.yml`, which regenerates
the API reference from that browser release with pdoc and deploys it to GitHub
Pages. It can also be dispatched by hand for any browser tag to preview.
