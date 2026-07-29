"""Async public API: AsyncBrowser, AsyncSession, run_script_async.

A thin asyncio facade over the synchronous implementation: every blocking
operation runs on a browser-owned thread pool, so the event loop is never
blocked and concurrency isn't capped by (or contended with) asyncio's
default executor. The surface mirrors the sync API with ``await`` at each
call site; a native asyncio transport could replace the internals later
without changing it.
"""

from __future__ import annotations

import asyncio
import functools
import os
from concurrent.futures import ThreadPoolExecutor

from .browser import Browser, Session, _attach_generated, _generated, run_script
from .errors import LightpandaError

AsyncSessionMethods = _generated("AsyncSessionMethods")


async def _run(executor: ThreadPoolExecutor, fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, functools.partial(fn, *args, **kwargs))


class AsyncSession(AsyncSessionMethods):
    """One isolated browsing context (own page, cookies, memory), async.

    Do not construct directly — use :meth:`AsyncBrowser.new_session`.
    """

    def __init__(self, session: Session, executor: ThreadPoolExecutor):
        self._session = session
        self._executor = executor

    @property
    def id(self) -> str:
        return self._session.id

    async def call(self, tool: str, **kwargs):
        """Invoke a browser tool by name. The generated methods route here."""
        return await _run(self._executor, self._session.call, tool, **kwargs)

    def __getattr__(self, attr: str):
        session = self.__dict__.get("_session")
        if session is not None and session._resolve(attr):
            return functools.partial(self.call, attr)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {attr!r}")

    def __dir__(self):
        return sorted(set(super().__dir__()) | self._session._tool_attrs())

    async def close(self) -> None:
        await _run(self._executor, self._session.close)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()


_attach_generated(AsyncSession, AsyncSessionMethods)


class AsyncBrowser:
    """A lightpanda browser process, driven from asyncio.

    The subprocess is spawned by :meth:`start` — called automatically on
    ``async with`` entry and by :meth:`new_session`. Not fork-inheritable,
    same as :class:`Browser`.
    """

    def __init__(
        self,
        binary: str | os.PathLike | None = None,
        env: dict[str, str] | None = None,
        timeout: float = 300.0,
        verbose: bool = False,
        args: tuple[str, ...] | list[str] = (),
        max_concurrency: int = 32,
    ):
        """``binary``/``env``/``timeout``/``verbose``/``args`` are forwarded
        to :class:`Browser`. ``max_concurrency`` caps concurrently executing
        tool calls across this browser's sessions (worker threads are
        created lazily)."""
        self._kwargs = dict(binary=binary, env=env, timeout=timeout, verbose=verbose, args=args)
        self._browser: Browser | None = None
        self._start_lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_concurrency, thread_name_prefix="lightpanda")

    async def start(self) -> AsyncBrowser:
        """Spawn the browser process and fetch its tool list. Idempotent."""
        if self._browser is None:
            async with self._start_lock:
                if self._browser is None:
                    self._browser = await _run(self._executor, Browser, **self._kwargs)
        return self

    @property
    def tools(self) -> dict[str, dict]:
        """Tool name → {description, schema}, as reported by the browser."""
        if self._browser is None:
            raise LightpandaError("browser not started; use `async with` or `await start()`")
        return self._browser.tools

    async def new_session(self) -> AsyncSession:
        await self.start()
        return AsyncSession(await _run(self._executor, self._browser.new_session), self._executor)

    async def close(self) -> None:
        if self._browser is not None:
            browser, self._browser = self._browser, None
            await _run(self._executor, browser.close)
        self._executor.shutdown(wait=False)

    async def __aenter__(self):
        return await self.start()

    async def __aexit__(self, *exc):
        await self.close()


async def run_script_async(
    script: str | os.PathLike,
    env: dict[str, str] | None = None,
    binary: str | os.PathLike | None = None,
    timeout: float | None = None,
) -> str:
    """Async variant of :func:`lightpanda.run_script` (runs in a worker thread)."""
    return await asyncio.to_thread(run_script, script, env=env, binary=binary, timeout=timeout)


__all__ = ["AsyncBrowser", "AsyncSession", "run_script_async"]
