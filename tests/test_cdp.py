"""CDPServer's endpoints and options, with the standard library only. A real
CDP client is exercised in test_cdp_playwright.py; the shared process
lifecycle in test_serve.py and the PDEATHSIG test in test_browser.py."""

import socket

import pytest

from lightpanda import CDPServer, LightpandaError
from lightpanda.client import _reserve_port


def test_endpoints_and_version(cdp_server):
    assert cdp_server.port > 0
    assert cdp_server.ws_endpoint == f"ws://127.0.0.1:{cdp_server.port}/"
    assert cdp_server.http_endpoint == f"http://127.0.0.1:{cdp_server.port}"

    version = cdp_server.version()
    assert version["Browser"].startswith("Lightpanda")
    assert version["webSocketDebuggerUrl"] == cdp_server.ws_endpoint


def test_fixed_port(binary):
    port = _reserve_port(0)
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
