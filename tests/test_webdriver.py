"""WebDriverServer's endpoints and the WebDriver session bootstrap they promise,
with the standard library only. A real BiDi client is exercised in
test_webdriver_selenium.py; the shared process lifecycle in test_serve.py."""

import json
import urllib.error
import urllib.request

from lightpanda import WebDriverServer

BIDI_CAPS = {"capabilities": {"alwaysMatch": {"webSocketUrl": True}}}


def _request(method, url, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as err:
        return err.code, None


def _with_session(server):
    """The `POST /session` value; call the returned closer when done."""
    status, payload = _request("POST", f"{server.http_endpoint}/session", BIDI_CAPS)
    assert status == 200, payload
    value = payload["value"]
    return value, lambda: _request("DELETE", f"{server.http_endpoint}/session/{value['sessionId']}")


def test_endpoints_and_status(webdriver_server):
    assert webdriver_server.port > 0
    assert webdriver_server.bidi_endpoint == f"ws://127.0.0.1:{webdriver_server.port}/session"
    assert webdriver_server.http_endpoint == f"http://127.0.0.1:{webdriver_server.port}"
    assert webdriver_server.status() == {"ready": True, "message": ""}


def test_session_bootstrap_advertises_bidi_endpoint(webdriver_server):
    value, done = _with_session(webdriver_server)
    try:
        assert value["capabilities"]["browserName"] == "Lightpanda"
        assert value["capabilities"]["webSocketUrl"] == f"{webdriver_server.bidi_endpoint}/{value['sessionId']}"
    finally:
        assert done() == (200, {"value": None})


def test_extra_args_passthrough(webdriver_server, binary):
    assert _request("GET", f"{webdriver_server.http_endpoint}/json/version")[0] == 404  # CDP off by default
    args = ["--protocol", "cdp", "--advertise-host", "localhost"]
    with WebDriverServer(binary=binary, args=args) as server:
        status, version = _request("GET", f"{server.http_endpoint}/json/version")  # --protocol is additive
        assert status == 200 and version["Browser"].startswith("Lightpanda")
        value, done = _with_session(server)
        done()
        assert value["capabilities"]["webSocketUrl"].startswith(f"ws://localhost:{server.port}/session/")
