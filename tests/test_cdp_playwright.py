"""Drive a CDPServer with a real CDP client. Playwright is a dev-only
dependency: `connect_over_cdp` needs its pip package, not a browser
download, and these tests skip when it is not installed."""

import pytest

sync_api = pytest.importorskip("playwright.sync_api")


@pytest.mark.parametrize("endpoint", ["ws_endpoint", "http_endpoint"])
def test_connect_over_cdp(cdp_server, fixture_url, endpoint):
    with sync_api.sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(getattr(cdp_server, endpoint))
        context = browser.new_context()
        page = context.new_page()
        page.goto(f"{fixture_url}/index.html")
        assert page.text_content("#headline") == "Hello from the fixture"
        assert page.locator(".item").count() == 3
        context.close()
        browser.close()
