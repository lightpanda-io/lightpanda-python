import functools
import http.server
import os
import threading
from pathlib import Path

import pytest

from lightpanda import AsyncBrowser, Browser, CDPServer

BROWSER_CHECKOUT = Path(__file__).parent.parent.parent / "browser"
FIXTURES = Path(__file__).parent / "fixtures"


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


@pytest.fixture(scope="session")
def binary() -> str:
    env = os.environ.get("LIGHTPANDA_BIN")
    if env and Path(env).is_file():
        return env
    built = BROWSER_CHECKOUT / "zig-out" / "bin" / "lightpanda"
    if built.is_file():
        return str(built)
    pytest.skip("no lightpanda binary (set LIGHTPANDA_BIN or build zig-out/bin/lightpanda)")


@pytest.fixture(scope="session")
def fixture_url():
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(FIXTURES))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


@pytest.fixture(scope="session")
def browser(binary):
    with Browser(binary=binary) as b:
        yield b


# Wraps the session browser rather than spawning a second process; the
# spawn path is covered by test_lazy_start_and_tools_property.
@pytest.fixture(scope="session")
async def abrowser(browser):
    async with AsyncBrowser.wrap(browser) as b:
        yield b


@pytest.fixture(scope="session")
def cdp_server(binary):
    with CDPServer(binary=binary) as server:
        yield server
