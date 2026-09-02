import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from conftest import alive
from lightpanda import Browser, LightpandaError, ToolError, run_script, client


def test_goto_and_markdown(browser, fixture_url):
    page = browser.new_session()
    page.goto(url=f"{fixture_url}/index.html")
    text = page.markdown()
    assert "Hello from the fixture" in text
    page.close()


def test_extract_dict_schema(browser, fixture_url):
    with browser.new_session() as page:
        page.goto(url=f"{fixture_url}/index.html")
        data = page.extract(schema={"headline": "#headline", "items": [".item a"]})
        assert data["headline"] == "Hello from the fixture"
        assert data["items"] == ["First item", "Second item", "Third item"]


def test_evaluate(browser, fixture_url):
    with browser.new_session() as page:
        page.goto(url=f"{fixture_url}/index.html")
        assert page.evaluate(script="1 + 2") == 3
        assert page.evaluate(script="document.title") == "Fixture Home"


def test_screenshot_inline_returns_png_bytes(browser, fixture_url):
    with browser.new_session() as page:
        page.goto(url=f"{fixture_url}/index.html")
        png = page.screenshot()
        assert isinstance(png, bytes)
        assert png.startswith(b"\x89PNG\r\n\x1a\n")


def test_snake_case_only(browser, fixture_url):
    with browser.new_session() as page:
        page.goto(url=f"{fixture_url}/index.html")
        assert page.get_url().endswith("/index.html")
        assert not hasattr(page, "getUrl")
        # Parameters are snake_case too; call() also takes the schema's names.
        shallow = page.tree(max_depth=1)
        assert shallow == page.call("tree", maxDepth=1)
        assert len(str(shallow)) < len(str(page.tree()))


def test_sessions_are_isolated(browser, fixture_url):
    with browser.new_session() as a, browser.new_session() as b:
        a.goto(url=f"{fixture_url}/index.html")
        b.goto(url=f"{fixture_url}/other.html")
        assert a.get_url().endswith("/index.html")
        assert b.get_url().endswith("/other.html")


def test_tool_error_raises(browser, fixture_url):
    with browser.new_session() as page:
        page.goto(url=f"{fixture_url}/index.html")
        with pytest.raises(ToolError):
            page.extract(schema={"nope": "#does-not-exist"})


def test_unknown_tool_raises(browser):
    with browser.new_session() as page:
        with pytest.raises(ToolError, match="unknown tool"):
            page.call("teleport", where="moon")


def test_extra_args_passthrough(binary, fixture_url, tmp_path):
    cache_dir = tmp_path / "cache"
    with Browser(binary=binary, args=["--http-cache-dir", str(cache_dir)]) as b:
        with b.new_session() as page:
            page.goto(url=f"{fixture_url}/index.html")
            assert "Hello" in page.markdown()
    assert cache_dir.exists()


@pytest.mark.skipif(sys.platform != "linux", reason="PDEATHSIG is Linux-only")
@pytest.mark.parametrize(
    "spawn",
    [
        "from lightpanda import Browser; b = Browser(binary=sys.argv[1]); pid = b._client._proc.pid",
        "from lightpanda import CDPServer; s = CDPServer(binary=sys.argv[1]); pid = s._proc.pid",
    ],
    ids=["Browser", "CDPServer"],
)
def test_sidecar_dies_with_killed_parent(binary, tmp_path, spawn):
    child_src = tmp_path / "spawn_and_hang.py"
    child_src.write_text(f"import sys, time\n{spawn}\nprint(pid, flush=True)\ntime.sleep(60)\n")
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parent.parent))
    child = subprocess.Popen(
        [sys.executable, str(child_src), binary],
        stdout=subprocess.PIPE, text=True, env=env,
    )
    sidecar_pid = int(child.stdout.readline())
    assert alive(sidecar_pid)

    os.kill(child.pid, signal.SIGKILL)
    child.wait()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and alive(sidecar_pid):
        time.sleep(0.1)
    assert not alive(sidecar_pid), "sidecar survived its parent's SIGKILL"


def test_run_script(binary, fixture_url, tmp_path):
    script = tmp_path / "visit.js"
    script.write_text('const page = new Page();\nawait page.goto("$LP_TEST_URL");\n')
    run_script(script, env={"LP_TEST_URL": f"{fixture_url}/index.html"}, binary=binary)


def test_find_binary_skips_own_console_script(tmp_path, monkeypatch):
    # `uv run` puts the venv's bin dir first on PATH, where the package's
    # `lightpanda` entry point shadows a real binary further along.
    monkeypatch.delenv("LIGHTPANDA_BIN", raising=False)
    monkeypatch.setattr(client, "BINARY_NAME", "lightpanda-probe")
    venv_bin, real_bin = tmp_path / "venv", tmp_path / "real"
    venv_bin.mkdir()
    real_bin.mkdir()
    script = venv_bin / "lightpanda-probe"
    script.write_text("#!/usr/bin/python3\nfrom lightpanda.cli import main\nmain()\n")
    script.chmod(0o755)

    monkeypatch.setenv("PATH", str(venv_bin))
    with pytest.raises(LightpandaError, match="could not find"):
        client.find_binary()

    real = real_bin / "lightpanda-probe"
    real.write_bytes(b"\x7fELF not really")
    real.chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join([str(venv_bin), str(real_bin)]))
    assert client.find_binary() == real
