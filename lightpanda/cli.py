"""Console entry point: trampoline into the bundled lightpanda binary."""

import os
import subprocess
import sys

from .client import find_binary
from .errors import LightpandaError


def main():
    try:
        binary = find_binary()
    except LightpandaError as err:
        print(f"error: {err}", file=sys.stderr)
        sys.exit(1)

    args = [str(binary)] + sys.argv[1:]
    try:
        if sys.platform == "win32":
            try:
                sys.exit(subprocess.run(args).returncode)
            except KeyboardInterrupt:
                sys.exit(130)
        os.execv(binary, args)
    except OSError as err:
        print(f"error executing {binary}: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
