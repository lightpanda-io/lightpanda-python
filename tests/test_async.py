"""Async-facade tests. The tool-call machinery is the sync Session's (covered
by test_browser.py); these cover what is new: the await surface, error
propagation across the thread hop, concurrency, and lazy startup."""

import asyncio

import pytest

from lightpanda import AsyncBrowser, LightpandaError, ToolError, run_script_async


async def test_goto_markdown_and_alias(abrowser, fixture_url):
    async with abrowser.session() as page:
        await page.goto(url=f"{fixture_url}/index.html")
        assert "Hello from the fixture" in await page.markdown()
        assert await page.get_url() == await page.getUrl()


async def test_concurrent_sessions_are_isolated(abrowser, fixture_url):
    async def visit(path):
        async with abrowser.session() as page:
            await page.goto(url=f"{fixture_url}/{path}")
            return await page.get_url()

    urls = await asyncio.gather(*(visit(p) for p in ("index.html", "other.html", "index.html")))
    assert [u.rsplit("/", 1)[1] for u in urls] == ["index.html", "other.html", "index.html"]


async def test_tool_error_raises(abrowser, fixture_url):
    async with abrowser.session() as page:
        await page.goto(url=f"{fixture_url}/index.html")
        with pytest.raises(ToolError):
            await page.extract(schema={"nope": "#does-not-exist"})


async def test_lazy_start_and_tools_property(binary):
    browser = AsyncBrowser(binary=binary)
    with pytest.raises(LightpandaError, match="not started"):
        browser.tools
    try:
        page = await browser.new_session()
        assert "goto" in browser.tools
        await page.close()
    finally:
        await browser.close()


async def test_run_script_async(binary, fixture_url, tmp_path):
    script = tmp_path / "visit.js"
    script.write_text('const page = new Page();\nawait page.goto("$LP_TEST_URL");\n')
    await run_script_async(script, env={"LP_TEST_URL": f"{fixture_url}/index.html"}, binary=binary)
