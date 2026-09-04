class LightpandaError(Exception):
    """Base error for the lightpanda package."""


class ProcessError(LightpandaError):
    """The browser binary could not be found, started, or reached."""


class ProtocolError(LightpandaError):
    """JSON-RPC level failure (invalid request, timeout, internal error)."""

    def __init__(self, message: str, code: int | None = None):
        super().__init__(message)
        self.code = code
        """The JSON-RPC error code, when the server sent one."""


class ToolError(LightpandaError):
    """A browser tool reported failure (bad selector, JS exception, ...)."""


class ScriptError(LightpandaError):
    """A script replay (`run_script`) exited with a failure."""

    def __init__(self, message: str, returncode: int, stdout: str = "", stderr: str = ""):
        super().__init__(message)
        self.returncode = returncode
        """The process exit status, or ``-1`` when the script file does not exist."""
        self.stdout = stdout
        """What the script wrote to stdout before failing."""
        self.stderr = stderr
        """What the script wrote to stderr."""
