# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Python client for the Lightpanda headless browser (`pip install lightpanda`). Pure Python — no C extension. The wheel bundles the `lightpanda` binary (built from the sibling `../browser` Zig repo) and the package spawns it as a subprocess, speaking MCP JSON-RPC over local HTTP. The client library is Apache-2.0; the bundled binary is AGPL-3.0.

## Commands

Everything needs a lightpanda binary. It is resolved in order: explicit `binary=` arg, `$LIGHTPANDA_BIN`, the package directory (bundled copy), then PATH (`lightpanda/client.py:find_binary`). Tests additionally look at the sibling checkout's `../browser/zig-out/bin/lightpanda` and **skip** (not fail) if no binary is found — a silently-skipping suite usually means no binary.

```bash
uv run --group dev pytest tests                                  # full test suite
uv run --group dev pytest tests/test_browser.py::test_evaluate   # single test
uv run --no-project python scripts/generate_methods.py           # regenerate lightpanda/_methods.py
uv run --no-project --with pdoc python scripts/build_docs.py     # regenerate methods + docs/ (pdoc)
uv build --wheel                                                 # platform wheel with bundled binary
LIGHTPANDA_PLAT=manylinux_2_35_x86_64 uv build --wheel           # override the wheel's platform tag (CI)
```

Tests use pytest with `asyncio_mode = "auto"` (configured in `pyproject.toml` — async tests are silently skipped without it). Fixtures in `tests/conftest.py` are session-scoped: one `Browser` for the whole run, with `abrowser` wrapping it via `AsyncBrowser.wrap` rather than spawning a second process, and a local `http.server` serving `tests/fixtures/`.

Release wheels are built by `.github/workflows/wheels.yml` (manual `workflow_dispatch`): all platforms build on one Ubuntu runner by downloading the prebuilt binary from a `lightpanda-io/browser` GitHub release, then native runners test the installed wheel. Release wheels must bundle a `ReleaseFast` binary.

## Architecture

**`lightpanda/client.py` — process + transport.** `Client` spawns the binary in `mcp --port <n>` mode (MCP Streamable HTTP on 127.0.0.1) and sends each JSON-RPC message as one POST. Session routing is the `Mcp-Session-Id` header; the server creates a session on first use of an unknown id, so the client mints its own ids (`py1`, `py2`, …) and `DELETE` tears one down. On Linux, `prctl(PR_SET_PDEATHSIG)` guarantees the subprocess dies with the interpreter.

**`lightpanda/browser.py` — sync public API.** `Browser` fetches `tools/list` once at startup; `Session.call(tool, **kwargs)` is the single funnel every tool call goes through. It drops `None` kwargs, JSON-serializes dict/list values for params the schema declares as strings (e.g. `extract`'s `schema`), and parses JSON tool results back into Python values. Both camelCase (`waitForSelector`) and snake_case (`wait_for_selector`) names resolve to the same tool.

**`lightpanda/_methods.py` — generated, never edit by hand.** `scripts/generate_methods.py` emits one concrete typed/documented method per browser tool (from the binary's `tools/list` schemas) in both `SessionMethods` and `AsyncSessionMethods`, all forwarding to `call`. Regenerate it whenever the browser's tool schemas change. `Session` also has a `__getattr__` fallback that routes any listed tool through `call`, so the package works even before/without `_methods.py` (this is how the generator bootstraps — see `_generated()` in browser.py). `_attach_generated` copies the methods onto the class so pdoc and IDEs see them as its own.

**`lightpanda/async_browser.py` — asyncio facade.** Not a native async transport: every blocking call runs on a browser-owned `ThreadPoolExecutor` (`max_concurrency` caps in-flight tool calls). `AsyncBrowser` starts lazily (`async with` or first `new_session`); `AsyncBrowser.wrap(browser)` adopts an existing sync `Browser` without owning its lifetime.

**`lightpanda/cli.py`** puts the full `lightpanda` CLI on PATH by `execv`-ing the bundled binary. `run_script` / `run_script_async` replay saved PandaScript agent scripts via `lightpanda run` as a plain subprocess (no MCP).

**`setup.py` — wheel packaging glue.** `build_py` copies the binary into the package; the `bdist_wheel` subclass tags wheels `py3-none-<plat>` (platform-specific because of the binary, but Python-version independent). A source install without a binary still works and falls back to `LIGHTPANDA_BIN`/PATH at runtime.

## Conventions

- Errors: everything derives from `LightpandaError`; `ProtocolError` (JSON-RPC layer), `ToolError` (a tool reported failure), `ScriptError` (script replay exit ≠ 0). Raise the most specific one.
- The wheel version tracks the browser release tag (CI rewrites `pyproject.toml`'s version from the tag).
- New public surface must appear in `lightpanda/__init__.py`'s `__all__` and keep the sync/async APIs mirrored.
