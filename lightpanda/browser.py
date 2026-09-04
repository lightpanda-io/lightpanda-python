"""Public API: Browser, Session, run_script.

Session tool methods are generated from the server's ``tools/list`` schemas —
one method per browser tool, with the tool name and its schema properties
in snake_case (``waitForSelector`` → ``wait_for_selector``, ``backendNodeId``
→ ``backend_node_id``). :meth:`Session.call` is the escape hatch that also
accepts the raw tool and property names.
"""

from __future__ import annotations

import base64
import functools
import importlib
import itertools
import json
import os
import re
import subprocess
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .client import Client, _documented, find_binary
from .errors import ScriptError, ToolError

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


SessionMethods = _generated("SessionMethods")


@_documented
class Session(SessionMethods):
    """One isolated browsing context (own page, cookies, memory).

    Do not construct directly — use :meth:`Browser.new_session`.
    """

    def __init__(self, browser: Browser, session_id: str):
        self._client = browser._client
        self._tools = browser._tools
        self._snake_map = browser._snake_map
        self._id = session_id
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
        """The session id, as the browser knows it."""
        return self._id

    def call(self, tool: str, **kwargs):
        """Invoke a browser tool by name. The generated methods route here.

        Accepts the tool and argument names as the browser declares them
        (``waitForSelector``, ``backendNodeId``) as well as their snake_case
        forms. Returns parsed JSON for JSON-carrying tools, ``bytes`` for
        image results (``screenshot`` without ``path``), otherwise the result
        text. Raises :class:`ToolError` when the tool reports a failure.

        Args:
            tool: The tool name.
            **kwargs: The tool's arguments; ``None`` values are omitted.
        """
        if self._closed:
            raise ToolError(f"session {self._id} is closed")
        name = self._snake_map.get(tool, tool)
        if name not in self._tools:
            raise ToolError(f"unknown tool {tool!r}")
        # Params are sent under the schema's (camelCase) property names; the
        # generated methods take their snake_case forms, so map those back.
        properties = self._tools[name]["schema"].get("properties", {})
        snake_props = {_snake(prop): prop for prop in properties}
        kwargs = {snake_props.get(k, k): v for k, v in kwargs.items() if v is not None}
        # The server declares JSON-carrying params (e.g. extract's schema) as
        # strings; serialize dict/list values passed for any such param.
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
        # Binary results (an inline screenshot) arrive as base64 image parts:
        # return the decoded bytes. Text results carrying JSON (extract, links,
        # tree, ...) are returned parsed, anything else as the raw string.
        images = [base64.b64decode(part["data"]) for part in content if part.get("type") == "image"]
        if images:
            return images[0] if len(images) == 1 else images
        try:
            return json.loads(text)
        except ValueError:
            return text

    def _resolve(self, attr: str) -> str | None:
        """The tool name behind a snake_case public attribute, if any."""
        name = self._snake_map.get(attr)
        return name if name is not None and name not in _SESSION_TOOLS else None

    def __getattr__(self, attr: str):
        if self.__dict__.get("_tools") and (name := self._resolve(attr)):
            return functools.partial(self.call, name)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {attr!r}")

    def close(self) -> None:
        """Release the session's page. Idempotent; calls made after this
        raise :class:`ToolError`. Closing the browser closes every session."""
        if not self._closed:
            self._closed = True
            self._client.delete_session(self._id)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()



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
        args: Sequence[str] = (),
    ):
        """Spawn the browser process and fetch its tool list.

        Args:
            binary: Path to a lightpanda binary. When omitted, resolved from
                the ``LIGHTPANDA_BIN`` environment variable, then the binary
                bundled in the package, then ``PATH``.
            env: Extra environment variables for the spawned process.
            timeout: Seconds to wait for a response to any request before
                raising :class:`ProtocolError`.
            verbose: Let the browser's own logging through to stderr.
            args: Extra CLI flags for the spawned browser process, e.g.
                ``["--http-cache-dir", path]`` or cookie flags.
        """
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
        self._snake_map = {_snake(name): name for name in self._tools}

    @property
    def tools(self) -> dict[str, dict]:
        """Tool name → {description, schema}, as reported by the browser."""
        return self._tools

    def new_session(self) -> Session:
        """Open a new isolated browsing context: its own page, cookies and
        memory. Close it with :meth:`Session.close` or a ``with`` block."""
        # itertools.count is atomic, so concurrent callers (the async facade's
        # worker threads) can't mint duplicate session ids.
        return Session(self, f"py{next(self._seq)}")

    def close(self) -> None:
        """Stop the browser process, closing every session with it."""
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


__all__ = ["Browser", "Session", "run_script"]
