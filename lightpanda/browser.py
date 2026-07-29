"""Public API: Browser, Session, run_script.

Session tool methods are generated from the server's ``tools/list`` schemas —
one method per browser tool, kwargs exactly the tool's schema properties.
Both the original tool name (``waitForSelector``) and its snake_case form
(``wait_for_selector``) resolve to the same method.
"""

from __future__ import annotations

import functools
import importlib
import itertools
import json
import os
import re
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .client import Client, find_binary
from .errors import ProtocolError, ScriptError, ToolError

_SESSION_TOOLS = {"save", "session_new", "session_list", "session_close"}


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _generated(name: str) -> type:
    """The named class from ._methods, or an empty stub while
    scripts/generate_methods.py bootstraps (it imports this package before
    _methods.py exists; tool calls then rely on __getattr__)."""
    try:
        return getattr(importlib.import_module("._methods", __package__), name)
    except (ImportError, AttributeError):
        return type(name, (), {})


def _attach_generated(cls: type, methods: type) -> None:
    """pdoc hides members inherited from a private module; re-attach the
    generated tool methods as the class's own so they document (and
    introspect) directly."""
    for name, member in vars(methods).items():
        if not name.startswith("_") and name != "call":
            setattr(cls, name, member)


SessionMethods = _generated("SessionMethods")


class Session(SessionMethods):
    """One isolated browsing context (own page, cookies, memory).

    Do not construct directly — use :meth:`Browser.new_session`.
    """

    def __init__(self, client: Client, session_id: str, tools: dict[str, dict]):
        self._client = client
        self._id = session_id
        self._tools = tools
        self._snake_map = {_snake(name): name for name in tools}
        self._closed = False

        self._client.request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "lightpanda-python", "version": _version()},
            },
            session_id=self._id,
        )
        self._client.notify("notifications/initialized", session_id=self._id)

    @property
    def id(self) -> str:
        return self._id

    def call(self, tool: str, **kwargs):
        """Invoke a browser tool by name. The generated methods route here."""
        if self._closed:
            raise ToolError(f"session {self._id} is closed")
        name = self._snake_map.get(tool, tool)
        if name not in self._tools:
            raise ToolError(f"unknown tool {tool!r}")
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        # The server declares JSON-carrying params (e.g. extract's schema) as
        # strings; serialize dict/list values passed for any such param.
        properties = self._tools[name]["schema"].get("properties", {})
        for key, value in kwargs.items():
            if isinstance(value, (dict, list)) and properties.get(key, {}).get("type") == "string":
                kwargs[key] = json.dumps(value)

        result = self._client.request(
            "tools/call", {"name": name, "arguments": kwargs}, session_id=self._id
        )
        content = result.get("content") or []
        text = "\n".join(part.get("text", "") for part in content if part.get("type") == "text")
        if result.get("isError"):
            raise ToolError(text or f"{name} failed")
        # Tool results are text; JSON payloads (extract, links, tree, ...) are
        # returned parsed, anything else as the raw string.
        try:
            return json.loads(text)
        except ValueError:
            return text

    def _resolve(self, attr: str) -> str | None:
        """The tool name behind a public attribute (snake or camel), if any."""
        name = self._snake_map.get(attr, attr)
        return name if name in self._tools and name not in _SESSION_TOOLS else None

    def _tool_attrs(self) -> set[str]:
        names = set()
        for snake, name in self._snake_map.items():
            if name not in _SESSION_TOOLS:
                names.update((snake, name))
        return names

    def __getattr__(self, attr: str):
        if self.__dict__.get("_tools") and (name := self._resolve(attr)):
            return functools.partial(self.call, name)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {attr!r}")

    def __dir__(self):
        return sorted(set(super().__dir__()) | self._tool_attrs())

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._client.delete_session(self._id)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


_attach_generated(Session, SessionMethods)


class Browser:
    """A lightpanda browser process. Spawns the bundled binary on first use.

    Not fork-inheritable: after ``os.fork()``/``multiprocessing``, create a
    fresh Browser in the child.
    """

    def __init__(
        self,
        binary: str | os.PathLike | None = None,
        env: dict[str, str] | None = None,
        timeout: float = 300.0,
        verbose: bool = False,
        args: tuple[str, ...] | list[str] = (),
    ):
        """``args`` are extra CLI flags for the spawned browser process
        (e.g. ``["--http-cache-dir", path]`` or cookie flags)."""
        self._client = Client(binary=binary, env=env, timeout=timeout, verbose=verbose, args=args)
        self._seq = itertools.count(1)
        listed = self._client.request("tools/list")
        self._tools = {
            tool["name"]: {
                "description": tool.get("description", ""),
                "schema": tool.get("inputSchema") or {},
            }
            for tool in listed.get("tools", [])
        }

    @property
    def tools(self) -> dict[str, dict]:
        """Tool name → {description, schema}, as reported by the browser."""
        return self._tools

    def new_session(self) -> Session:
        # itertools.count is atomic, so concurrent callers (the async facade's
        # worker threads) can't mint duplicate session ids.
        return Session(self._client, f"py{next(self._seq)}", self._tools)

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def run_script(
    script: str | os.PathLike,
    env: dict[str, str] | None = None,
    binary: str | os.PathLike | None = None,
    timeout: float | None = None,
) -> str:
    """Replay a saved lightpanda script (no LLM) and return its stdout.

    ``env`` entries (e.g. ``LP_*`` placeholder values) are added to the
    child's environment. Raises :class:`ScriptError` on a non-zero exit.
    """
    path = Path(script)
    if not path.is_file():
        raise ScriptError(f"script not found: {path}", returncode=-1)
    proc = subprocess.run(
        [str(find_binary(binary)), "run", str(path)],
        env=os.environ | (env or {}),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "no output"
        raise ScriptError(
            f"{path.name} failed (exit {proc.returncode}): {detail}",
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
    return proc.stdout


@functools.cache
def _version() -> str:
    try:
        return version("lightpanda")
    except PackageNotFoundError:
        return "0.0.0.dev0"


__all__ = ["Browser", "Session", "run_script", "ProtocolError", "ScriptError", "ToolError"]
