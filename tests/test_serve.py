"""Lifecycle shared by the `lightpanda serve` wrappers, tested once per
class pair. Protocol-specific behaviour lives in test_cdp.py / test_bidi.py."""

import pytest
from conftest import alive

from lightpanda import AsyncBiDiServer, AsyncCDPServer, BiDiServer, CDPServer, LightpandaError

PAIRS = [
    pytest.param(CDPServer, AsyncCDPServer, "version", "ws_endpoint", id="cdp"),
    pytest.param(BiDiServer, AsyncBiDiServer, "status", "bidi_endpoint", id="bidi"),
]


@pytest.mark.parametrize("sync_cls, async_cls, probe, endpoint", PAIRS)
def test_close_kills_process(binary, sync_cls, async_cls, probe, endpoint):
    server = sync_cls(binary=binary)
    pid = server._proc.pid
    assert alive(pid)

    server.close()  # terminates and reaps
    assert not alive(pid)
    with pytest.raises(LightpandaError, match="closed"):
        getattr(server, probe)()
    server.close()  # idempotent


@pytest.mark.parametrize("sync_cls, async_cls, probe, endpoint", PAIRS)
async def test_async_lazy_start(binary, sync_cls, async_cls, probe, endpoint):
    server = async_cls(binary=binary)
    with pytest.raises(LightpandaError, match="not started"):
        getattr(server, endpoint)
    async with server:
        assert getattr(server, endpoint).startswith(f"ws://127.0.0.1:{server.port}/")
        assert await getattr(server, probe)()
    await server.close()  # idempotent
