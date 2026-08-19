"""Install the backend runtime dependencies into the current Python environment.

Use uv when available because uv-managed virtual environments may intentionally
omit pip. Fall back to pip only for conventional Python environments.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


REQUIREMENTS = Path(__file__).with_name("requirements-runtime.txt")


def run(command: list[str]) -> None:
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.check_call(command)


def main() -> int:
    if not REQUIREMENTS.is_file():
        raise FileNotFoundError(f"Backend requirements not found: {REQUIREMENTS}")

    uv = shutil.which("uv")
    if uv:
        run([uv, "pip", "install", "--python", sys.executable, "-r", str(REQUIREMENTS)])
    else:
        try:
            import pip  # noqa: F401
        except ImportError:
            run([sys.executable, "-m", "ensurepip", "--upgrade"])
        run([sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)])

    # Fail immediately if the original import chain is still incomplete.
    run(
        [
            sys.executable,
            "-c",
            (
                "import datasets, einops, fastapi, pyarrow.dataset, pydantic, sklearn, transformer_lens; "
                "print('Backend dependency import check passed')"
            ),
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
