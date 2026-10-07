"""Drive a BiDiServer with Selenium, over classic WebDriver and over BiDi.
Selenium is a dev-only dependency: `webdriver.Remote` needs its pip package,
not a driver or browser download, and these tests skip when it is not
installed."""

import pytest

webdriver = pytest.importorskip("selenium.webdriver")
from selenium.webdriver.common.by import By  # noqa: E402
from selenium.webdriver.common.options import ArgOptions  # noqa: E402

from lightpanda import BiDiServer, WebDriverServer  # noqa: E402


def test_webdriver_server_alias():
    assert WebDriverServer is BiDiServer


def test_selenium_classic_session(bidi_server, fixture_url):
    driver = webdriver.Remote(command_executor=bidi_server.http_endpoint, options=webdriver.ChromeOptions())
    try:
        driver.get(f"{fixture_url}/index.html")
        assert driver.title == "Fixture Home"
        assert driver.find_element(By.ID, "headline").text == "Hello from the fixture"

        items = driver.find_elements(By.CSS_SELECTOR, ".item a")
        assert [i.text for i in items] == ["First item", "Second item", "Third item"]

        items[0].click()
        assert driver.current_url == f"{fixture_url}/other.html"
        driver.back()
        assert driver.current_url == f"{fixture_url}/index.html"

        assert driver.execute_script("return document.querySelectorAll('.item').length") == 3
    finally:
        driver.quit()  # DELETE /session/<id>


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
