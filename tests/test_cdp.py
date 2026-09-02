"""CDPServer / AsyncCDPServer lifecycle, with the standard library only.
A real CDP client is exercised in test_cdp_playwright.py; the PDEATHSIG
test lives in test_browser.py."""

import json
import socket
import urllib.request

import pytest
from conftest import alive

from lightpanda import AsyncCDPServer, CDPServer, LightpandaError
from lightpanda.client import _reserve_port


def test_endpoints_and_version(cdp_server):
    assert cdp_server.port > 0
    assert cdp_server.ws_endpoint == f"ws://127.0.0.1:{cdp_server.port}/"
    assert cdp_server.http_endpoint == f"http://127.0.0.1:{cdp_server.port}"

    version = cdp_server.version()
    assert version["Browser"].startswith("Lightpanda")
    assert version["webSocketDebuggerUrl"] == cdp_server.ws_endpoint

    with urllib.request.urlopen(f"{cdp_server.http_endpoint}/json/list", timeout=5) as resp:
        assert json.loads(resp.read()) == []


def test_fixed_port(binary):
    port = _reserve_port()
    with CDPServer(binary=binary, port=port) as server:
        assert server.port == port


def test_port_in_use_raises(binary):
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        with pytest.raises(LightpandaError, match="in use"):
            CDPServer(binary=binary, port=taken.getsockname()[1])


def test_extra_args_passthrough(binary):
    with CDPServer(binary=binary, args=["--advertise-host", "localhost"]) as server:
        assert server.version()["webSocketDebuggerUrl"] == f"ws://localhost:{server.port}/"


def test_close_kills_process(binary):
    server = CDPServer(binary=binary)
    pid = server._proc.pid
    assert alive(pid)

    server.close()  # terminates and reaps
    assert not alive(pid)
    with pytest.raises(LightpandaError, match="closed"):
        server.version()
    server.close()  # idempotent


async def test_async_lazy_start(binary):
    server = AsyncCDPServer(binary=binary)
    with pytest.raises(LightpandaError, match="not started"):
        server.ws_endpoint
    async with server:
        assert server.ws_endpoint == f"ws://127.0.0.1:{server.port}/"
        version = await server.version()
        assert version["webSocketDebuggerUrl"] == server.ws_endpoint
    await server.close()  # idempotent
