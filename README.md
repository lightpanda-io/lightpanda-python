# lightpanda-python

Lightpanda for Python: scrape and automate the web with `pip install lightpanda`. Browser included, no separate download.

[Lightpanda](https://lightpanda.io) is a lightweight headless browser — an order of magnitude less memory than Chrome stacks. The wheel bundles the browser binary; there is no second install step.

```bash
uv add lightpanda        # or: uv pip install lightpanda / pip install lightpanda
```

```python
from lightpanda import Browser

with Browser() as b:
    page = b.new_session()
    page.goto(url="https://example.com")
    data = page.extract(schema={"title": "h1"})
```

The same API is available for asyncio — `AsyncBrowser` spawns the browser on
first use and runs sessions concurrently:

```python
from lightpanda import AsyncBrowser

async with AsyncBrowser() as b:
    page = await b.new_session()
    await page.goto(url="https://example.com")
    data = await page.extract(schema={"title": "h1"})
```

Every browser tool is a `Session` method (both `waitForSelector` and
`wait_for_selector` work), typed and documented in your IDE. Replay a saved
lightpanda agent script without any LLM:

```python
from lightpanda import run_script

run_script("hn.lp.js", env={"LP_HN_USERNAME": "me"})
```

(`run_script_async` is the awaitable variant.)

The package also puts the full `lightpanda` CLI on PATH (agent REPL, fetch,
serve).

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

Publishing is automatic: a daily job (`.github/workflows/track-browser.yml`)
watches [lightpanda-io/browser releases](https://github.com/lightpanda-io/browser/releases)
and, for a new tag, creates the matching release here and starts a wheel run
that builds from that browser release, derives the wheel version from the tag,
tests on all platforms, and publishes to PyPI via trusted publishing — after a
maintainer approves the `pypi` environment deployment on the run page. A
release can also be triggered by hand: create a GitHub release here whose tag
matches a browser release tag, or use the `workflow_dispatch` path (any
browser tag, publish to TestPyPI or PyPI) for dry runs.
