"""Subprocess management and MCP-over-HTTP JSON-RPC transport.

The bundled ``lightpanda`` binary is spawned in ``mcp --port <n>`` mode
(MCP Streamable HTTP on 127.0.0.1). Each JSON-RPC message is one POST;
session routing uses the ``Mcp-Session-Id`` header — an unknown id creates
the session on first use, so the client mints its own ids. DELETE with the
header tears a session down.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path

from .errors import ProcessError, ProtocolError

BINARY_NAME = "lightpanda.exe" if sys.platform == "win32" else "lightpanda"

_HOST = "127.0.0.1"
_SPAWN_ATTEMPTS = 3
_READY_TIMEOUT = 15.0


def _die_with_parent():
    """Linux: ask the kernel to SIGTERM the sidecar when the parent dies,
    so a SIGKILLed interpreter can't leak a browser process."""
    try:
        import ctypes
        import signal as _signal

        PR_SET_PDEATHSIG = 1
        ctypes.CDLL(None, use_errno=True).prctl(PR_SET_PDEATHSIG, _signal.SIGTERM)
    except Exception:
        pass


def _is_console_script(path: Path) -> bool:
    """True for this package's own ``lightpanda`` entry point (the trampoline
    pip/uv put on PATH), which must not be mistaken for the browser."""
    try:
        with open(path, "rb") as f:
            head = f.read(512)
    except OSError:
        return False
    return head.startswith(b"#!") and b"lightpanda.cli" in head


def find_binary(explicit: str | os.PathLike | None = None) -> Path:
    """Locate the lightpanda binary: explicit arg, $LIGHTPANDA_BIN, the
    bundled package copy, then PATH (skipping this package's own console
    script, which shadows a real binary when a venv's bin dir comes first)."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("LIGHTPANDA_BIN")
    if env:
        candidates.append(Path(env))
    candidates.append(Path(__file__).parent / BINARY_NAME)
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if entry:
            candidates.append(Path(entry) / BINARY_NAME)

    for candidate in candidates:
        if candidate.is_file() and not _is_console_script(candidate):
            return candidate
    raise ProcessError(
        "could not find the lightpanda binary; reinstall the package, set "
        "LIGHTPANDA_BIN, or put `lightpanda` on PATH"
    )


def _reserve_port(port: int) -> int:
    """Bind-and-release ``port`` (0: any free one) and return it. SO_REUSEADDR
    mirrors the browser's own bind, so a TIME_WAIT leftover doesn't count as
    a collision."""
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((_HOST, port))
        except OSError as err:
            raise ProcessError(f"port {port} is already in use") from err
        return s.getsockname()[1]


def _wait_ready(proc: subprocess.Popen, port: int) -> None:
    deadline = time.monotonic() + _READY_TIMEOUT
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise ProcessError(f"lightpanda exited during startup (code {proc.returncode})")
        try:
            with socket.create_connection((_HOST, port), timeout=0.25):
                return
        except OSError:
            time.sleep(0.005)  # a refused localhost connect fails instantly
    raise ProcessError("timed out waiting for lightpanda to listen")


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _spawn(
    binary: Path,
    mode: str,
    args: Sequence[str],
    env: dict[str, str] | None,
    verbose: bool,
    port: int | None = None,
) -> tuple[subprocess.Popen, int]:
    """Start ``lightpanda <mode> --port N <args>`` and wait until it listens.

    With ``port=None`` a free port is picked; a collision (the browser exits
    with "address already in use") is retried on a fresh port. A caller-chosen
    ``port`` gets a single attempt.
    """
    child_env = os.environ | (env or {})
    stderr = None if verbose else subprocess.DEVNULL
    last_error: Exception | None = None
    for _ in range(_SPAWN_ATTEMPTS if port is None else 1):
        chosen = _reserve_port(port or 0)
        proc = subprocess.Popen(
            [str(binary), mode, "--port", str(chosen), *args],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=stderr,
            env=child_env,
            preexec_fn=_die_with_parent if sys.platform == "linux" else None,
        )
        try:
            _wait_ready(proc, chosen)
        except ProcessError as err:
            last_error = err
            _terminate(proc)
            continue
        return proc, chosen
    raise ProcessError(
        f"failed to start `lightpanda {mode}`: {last_error}; "
        "run with verbose=True to see the browser log"
    )


class Client:
    """Owns one lightpanda subprocess and speaks JSON-RPC to it."""

    def __init__(
        self,
        binary: str | os.PathLike | None = None,
        env: dict[str, str] | None = None,
        timeout: float = 300.0,
        verbose: bool = False,
        args: Sequence[str] = (),
    ):
        self._timeout = timeout
        self._lock = threading.Lock()
        self._id = 0
        self._proc: subprocess.Popen | None
        self._proc, self._port = _spawn(find_binary(binary), "mcp", args, env, verbose)

    @property
    def base_url(self) -> str:
        return f"http://{_HOST}:{self._port}/"

    def _http(self, method: str, body: bytes | None, session_id: str | None) -> tuple[int, bytes]:
        headers = {"Content-Type": "application/json"}
        if session_id:
            headers["Mcp-Session-Id"] = session_id
        req = urllib.request.Request(self.base_url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as err:
            return err.code, err.read()
        except OSError as err:
            if self._proc is None:
                state = "closed"
            elif self._proc.poll() is None:
                state = "running"
            else:
                state = f"exited (code {self._proc.returncode})"
            raise ProcessError(f"lost connection to lightpanda ({state}): {err}") from err

    def request(self, method: str, params: dict | None = None, session_id: str | None = None):
        """Send one JSON-RPC request and return its ``result``."""
        with self._lock:
            self._id += 1
            rpc_id = self._id
        message: dict = {"jsonrpc": "2.0", "id": rpc_id, "method": method}
        if params is not None:
            message["params"] = params

        status, body = self._post(message, session_id)
        if not body:
            raise ProtocolError(f"empty response (HTTP {status}) for {method}")
        try:
            payload = json.loads(body)
        except ValueError as err:
            raise ProtocolError(f"invalid JSON-RPC response for {method}: {body[:200]!r}") from err
        if error := payload.get("error"):
            raise ProtocolError(f"{error.get('message', 'error')} (code {error.get('code')})", code=error.get("code"))
        return payload.get("result")

    def notify(self, method: str, session_id: str | None = None) -> None:
        self._post({"jsonrpc": "2.0", "method": method}, session_id)

    def _post(self, message: dict, session_id: str | None) -> tuple[int, bytes]:
        return self._http("POST", json.dumps(message).encode(), session_id)

    def delete_session(self, session_id: str) -> None:
        self._http("DELETE", None, session_id)

    def close(self) -> None:
        if self._proc is not None:
            _terminate(self._proc)
            self._proc = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _documented(cls: type) -> type:
    """Copy the public members (and ``__init__``) that ``cls`` inherits from
    bases defined in private modules onto ``cls`` itself. pdoc hides what is
    inherited from a private module, so the generated tool methods
    (``_methods``) and the serve-wrapper base (``_serve``) would otherwise be
    missing from the public classes' docs and IDE signatures."""
    for base in cls.__mro__[1:]:
        if not base.__module__.rpartition(".")[2].startswith("_"):
            continue
        for name, member in vars(base).items():
            if (name == "__init__" or not name.startswith("_")) and name not in vars(cls):
                setattr(cls, name, member)
    return cls
