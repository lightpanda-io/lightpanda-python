"""Build glue: bundle the lightpanda binary into the wheel.

The binary is taken from, in order: the LIGHTPANDA_BIN environment variable,
or a sibling browser checkout's zig-out/bin/lightpanda (built with
`zig build -Doptimize=ReleaseFast`). Wheel CI downloads a release artifact and
points LIGHTPANDA_BIN at it. A source install without a binary still works —
the runtime falls back to LIGHTPANDA_BIN or PATH.
"""

import os
import shutil
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_py import build_py
from setuptools.dist import Distribution

HERE = Path(__file__).parent
BROWSER_CHECKOUT = HERE.parent / "browser"
BINARY_NAME = "lightpanda.exe" if sys.platform == "win32" else "lightpanda"


def find_binary():
    env = os.environ.get("LIGHTPANDA_BIN")
    if env and Path(env).is_file():
        return Path(env)
    built = BROWSER_CHECKOUT / "zig-out" / "bin" / BINARY_NAME
    if built.is_file():
        return built
    return None


class BuildPyWithBinary(build_py):
    def run(self):
        super().run()
        binary = find_binary()
        dest = Path(self.build_lib) / "lightpanda" / BINARY_NAME
        if binary is None:
            print(
                "warning: no lightpanda binary found (set LIGHTPANDA_BIN or run "
                "`zig build -Doptimize=ReleaseFast`); the package will rely on "
                "LIGHTPANDA_BIN or PATH at runtime",
                file=sys.stderr,
            )
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(binary, dest)
        dest.chmod(0o755)


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


class BinaryWheel(bdist_wheel):
    """The wheel is platform-specific (bundled binary) but Python-version
    independent — tag it py3-none-<plat>. LIGHTPANDA_PLAT overrides the
    platform (CI sets e.g. manylinux_2_35_x86_64 to match the release
    runner's glibc floor)."""

    def get_tag(self):
        _, _, plat = super().get_tag()
        return "py3", "none", os.environ.get("LIGHTPANDA_PLAT", plat)


setup(
    cmdclass={"build_py": BuildPyWithBinary, "bdist_wheel": BinaryWheel},
    distclass=BinaryDistribution,
)
