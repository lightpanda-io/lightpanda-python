"""Drive a BiDiServer with Selenium. Selenium is a dev-only dependency:
`webdriver.Remote` needs its pip package, not a driver or browser download,
and this test skips when it is not installed.

Only the WebDriver BiDi modules are served, so every step goes over the
websocket with an explicit browsing context."""

import pytest

webdriver = pytest.importorskip("selenium.webdriver")
from selenium.webdriver.common.options import ArgOptions  # noqa: E402


def test_selenium_bidi_session(bidi_server, fixture_url):
    options = ArgOptions()
    options.web_socket_url = True
    driver = webdriver.Remote(command_executor=bidi_server.http_endpoint, options=options)
    try:
        assert driver.caps["browserName"] == "Lightpanda"
        assert driver.caps["webSocketUrl"] == f"{bidi_server.bidi_endpoint}/{driver.session_id}"

        assert driver.browsing_context.get_tree() == []  # first BiDi access opens the websocket
        context = driver.browsing_context.create(type="tab")
        assert [c.context for c in driver.browsing_context.get_tree()] == [context]

        driver.browsing_context.navigate(context=context, url=f"{fixture_url}/index.html", wait="complete")

        nodes = driver.browsing_context.locate_nodes(context=context, locator={"type": "css", "value": ".item"})
        assert len(nodes) == 3

        headline = driver.script.execute("() => document.querySelector('#headline').textContent", context_id=context)
        assert headline["value"] == "Hello from the fixture"

        result = driver.script.evaluate(expression="document.title", target={"context": context}, await_promise=False)
        assert result["result"]["value"] == "Fixture Home"
    finally:
        driver.quit()  # closes the websocket, then DELETE /session/<id>
