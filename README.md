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

## Drive it with Selenium (WebDriver)

The browser also speaks [WebDriver](https://www.w3.org/TR/webdriver2/), both
classic and [BiDi](https://w3c.github.io/webdriver-bidi/). `WebDriverServer`
starts `lightpanda serve --protocol webdriver` and hands you the URL
Selenium's `webdriver.Remote` takes as `command_executor`, so no chromedriver
is involved:

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

In a pytest suite, start the server once in a session-scoped fixture and open
a driver per test; with pytest-xdist each worker gets its own server on its
own free port.

Set `options.web_socket_url = True` to also get a WebDriver BiDi session and
use `driver.browsing_context` / `driver.script`. `server.bidi_endpoint`
(`ws://127.0.0.1:<port>/session`) is the raw BiDi WebSocket for clients that
speak the protocol directly. `AsyncWebDriverServer` is the asyncio twin, and
`args=["--protocol", "cdp"]` serves CDP on the same port as well.

## Respecting robots.txt

The browser can enforce `robots.txt` for you. It is off by default, matching
the `lightpanda` binary's own default, and every wrapper takes browser flags
through `args=`:

```python
Browser(args=["--obey-robots"])                    # also AsyncBrowser
CDPServer(args=["--obey-robots"])                  # also WebDriverServer, and the async twins
run_script("saved.js", args=["--obey-robots"])
```

A request the site disallows then fails rather than being sent: a tool call
raises `ToolError: navigation failed: RobotsBlocked`, and `run_script` raises
`ScriptError`.

One thing to know before turning it on, because it is easy to mistake for a
bug: the rule is applied to **every** request, not just the page you asked for.
Plenty of sites disallow the directory their own assets live in, so a page you
are allowed to fetch can load with its scripts and styles missing, and render
blank or empty. That is `robots.txt` being honoured, not a failure — the site
is asking you not to fetch those files. If a page comes back strangely empty
under `--obey-robots`, read the site's `robots.txt` before assuming otherwise.

Whether a given scrape is acceptable is not a question `robots.txt` alone
answers: a site's terms of service can forbid what its `robots.txt` permits.

## How the bindings work

Every browser tool is a `Session` method, typed and documented in your IDE.
The methods are not written by hand: they are generated from the bundled
browser's MCP tool schemas (`scripts/generate_methods.py` →
`lightpanda/_methods.py`), so signatures and docstrings come straight from the
binary, with tool and parameter names in snake_case (`waitForSelector` →
`wait_for_selector`, `backendNodeId` → `backend_node_id`). `Session.call` is
the escape hatch that takes the raw tool and parameter names as the MCP
server declares them. The full API reference is published at
[lightpanda.io/docs/reference/python](https://lightpanda.io/docs/reference/python).

The bindings follow Lightpanda's development and the package version tracks
browser releases — there is no backwards-compatibility guarantee: when the
browser's tools change, the Python methods change with them.

A tool returning JSON gives you the parsed value — `extract` a dict, `links`
a list. `goto` and the actions answer with a sentence describing what they
did, which also carries where it left the page, so a 404 is a field rather
than something to find in prose:

```python
r = page.goto(url="https://example.com/missing")
print(r)          # Navigated successfully. HTTP 404 Not Found.
if r.http_status >= 400:
    raise SystemExit(f"{r.url} is an error page, not content")
```

## Examples

[`examples/`](examples/) scrapes a JavaScript-rendered site that `requests`
cannot read, analyses the result with pandas/matplotlib, measures the same job
against Selenium + headless Chrome (100 quotes in ~3 s / 35 MB vs ~6 s / 1.2 GB,
median of 5 runs), and reads five years of NeurIPS — 19,219 paper abstracts that
never appear in the HTML at all — then clusters them to find what the field
started working on.
Each script runs standalone: `uv run examples/quotes_analysis.py`.

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
dispatches the wheels workflow here with that tag. The run builds from that
browser release, derives the wheel version from the tag, tests on all
platforms, and publishes to PyPI via trusted publishing — after a maintainer
approves the `pypi` environment deployment on the run page. Once published,
the run's `release` job records the matching GitHub release here.

That job is what creates the release, rather than the browser repo doing it
directly: the browser repo authenticates with a GitHub App that can start
workflows here but deliberately has no write access to repository contents,
so the release is created in-repo with the run's own `GITHUB_TOKEN`. A
release created that way triggers no workflow, so it cannot loop back into
the `release: published` build.

A release can also be triggered by hand: run the workflow from the Actions
tab (any browser tag, publish to TestPyPI or PyPI) for dry runs, or create a
GitHub release here whose tag matches a browser release tag — the
`release: published` trigger is kept as an escape hatch and does everything
the dispatch does.

To ship a packaging or client-side fix without waiting for a new browser
release, dispatch the workflow with the browser tag and a `post` number, e.g.
browser `0.4.0` with post `1`: it bundles browser `0.4.0` again and publishes
as the PEP 440 post-release `0.4.0.post1`. Bare `pip install lightpanda`,
`>=` and `~=` requirements pick post-releases up; an exact `==0.4.0` pin
deliberately does not, so pin with `~=0.4.0` to receive them. Creating a
release tagged `0.4.0-1` (or `0.4.0.post1`) by hand works too.

The API reference at
[lightpanda.io/docs/reference/python](https://lightpanda.io/docs/reference/python)
is regenerated daily from this repository's `main` branch by the
[docs repo's `python-reference` workflow](https://github.com/lightpanda-io/docs/blob/main/.github/workflows/python-reference.yml),
so a merged docstring or signature change shows up there with no step on this
side.
