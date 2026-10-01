import http.server
import json
import threading

import pytest

from lightpanda import Browser


class MockTypeSafe(http.server.BaseHTTPRequestHandler):
    """Answers every question asked: 0.9 for a noul, the first option for a choice."""

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        answers = {}
        for key, question in request["questions"].items():
            if question["type"] == "noul":
                answers[key] = {"type": "noul", "noul": 0.9}
            else:
                options = list(question["criteria"])
                rest = 0.2 / max(len(options) - 1, 1)
                answers[key] = {
                    "type": "choice",
                    "choice": options[0],
                    "probabilities": {o: 0.8 if i == 0 else rest for i, o in enumerate(options)},
                    "confidence": 0.8,
                }
        body = json.dumps({"model": "jev-1.13.0", "answers": answers}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def typesafe_url():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MockTypeSafe)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


@pytest.fixture(scope="module")
def classify_browser(binary, typesafe_url):
    env = {"TYPESAFE_API_KEY": "test-key", "TYPESAFE_BASE_URL": typesafe_url}
    with Browser(binary=binary, env=env) as b:
        if "classify" not in b.tools:
            pytest.skip("binary has no classify tool")
        yield b


def test_classify_categories(classify_browser, fixture_url):
    with classify_browser.new_session() as page:
        page.goto(url=f"{fixture_url}/index.html")
        assert page.classify(questions=["home", "article"]) == "home"


def test_classify_presets_and_questions(classify_browser, fixture_url):
    with classify_browser.new_session() as page:
        page.goto(url=f"{fixture_url}/index.html")
        result = page.classify(
            questions={
                "isBlocked": True,
                "has_list": "Does the page show a list of links?",
                "kind": {"question": "What kind of page?", "options": ["home", "article"]},
            },
            selector="body",
        )
        assert result["isBlocked"] == 0.9
        assert result["has_list"] == 0.9
        assert result["kind"]["choice"] == "home"



def test_judge(classify_browser, fixture_url):
    with classify_browser.new_session() as page:
        page.goto(url=f"{fixture_url}/index.html")
        verdict = page.judge()
        assert verdict.is_blocked and verdict.is_paywall and not verdict.ok
        assert set(verdict.scores) >= {"is_blocked", "is_login_wall", "is_unsupported_browser"}
        assert set(verdict.scores.values()) == {0.9}
        assert verdict.reason in verdict.scores
        assert page.judge(threshold=0.95).ok
        assert page.judge(threshold=0.95).reason is None


async def test_judge_async(classify_browser, fixture_url):
    from lightpanda import AsyncBrowser

    async with AsyncBrowser.wrap(classify_browser).session() as page:
        await page.goto(url=f"{fixture_url}/index.html")
        assert (await page.judge()).is_captcha
