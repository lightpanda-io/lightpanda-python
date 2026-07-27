# lightpanda-python

Lightpanda for Python: scrape and automate the web with `pip install lightpanda`. Browser included, no separate download.

[Lightpanda](https://lightpanda.io) is a lightweight headless browser — an order of magnitude less memory than Chrome stacks. The wheel bundles the browser binary; there is no second install step.

```python
from lightpanda import Browser

with Browser() as b:
    page = b.new_session()
    page.goto(url="https://example.com")
    data = page.extract(schema={"title": "h1"})
```

Every browser tool is a `Session` method (both `waitForSelector` and
`wait_for_selector` work), typed and documented in your IDE. Replay a saved
lightpanda agent script without any LLM:

```python
from lightpanda import run_script

run_script("hn.lp.js", env={"LP_HN_USERNAME": "me"})
```

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
