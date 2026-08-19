"""Phase 1247: regression test for the backend's optional dataset import boundary.

The visualization server needs TransformerLens model APIs, but it does not need
Hugging Face datasets during startup.  This test verifies that the optional
datasets/pyarrow stack remains lazy, while explicit evaluation imports and a
real FastAPI server launch continue to work.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = ROOT / "tests" / "glm5" / "result" / "phase1247_backend_import_boundary.json"


def run_python(code: str, *, extra_env: dict[str, str] | None = None) -> dict[str, object]:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    return {
        "returncode": completed.returncode,
        "seconds": round(time.perf_counter() - started, 4),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def find_debugpy_lib() -> Path | None:
    extension_root = Path.home() / ".vscode" / "extensions"
    candidates = sorted(
        extension_root.glob("ms-python.debugpy-*/bundled/libs"),
        key=lambda path: path.as_posix(),
        reverse=True,
    )
    return candidates[0] if candidates else None


def free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def live_server_check() -> dict[str, object]:
    port = free_local_port()
    env = os.environ.copy()
    env.update(
        {
            "AI2050_HOST": "127.0.0.1",
            "AI2050_PORT": str(port),
            "AI2050_RELOAD": "0",
            "PYTHONIOENCODING": "utf-8",
        }
    )
    process = subprocess.Popen(
        [sys.executable, str(ROOT / "server" / "server.py")],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    started = time.perf_counter()
    response_body = ""
    error = ""
    try:
        deadline = started + 45
        while time.perf_counter() < deadline:
            if process.poll() is not None:
                error = f"server exited early with code {process.returncode}"
                break
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                    response_body = response.read().decode("utf-8")
                    if response.status == 200:
                        break
            except Exception as exc:  # the socket is expected to refuse while booting
                error = str(exc)
                time.sleep(0.25)
        else:
            error = "server health endpoint did not become ready within 45 seconds"
    finally:
        process.terminate()
        try:
            output, _ = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate(timeout=10)

    ok = bool(response_body)
    return {
        "ok": ok,
        "port": port,
        "seconds_to_health": round(time.perf_counter() - started, 4),
        "response": response_body,
        "last_error": "" if ok else error,
        "server_output_tail": output[-2000:].strip(),
    }


def main() -> int:
    lazy_import = run_python(
        "import sys, transformer_lens; "
        "assert 'datasets' not in sys.modules; "
        "assert 'pyarrow.dataset' not in sys.modules; "
        "print('optional dataset modules stayed lazy')"
    )
    explicit_dataset = run_python(
        "import datasets, pyarrow.dataset; "
        "print('explicit dataset import succeeded')"
    )
    explicit_evals = run_python(
        "from transformer_lens import evals; "
        "print(evals.__name__)"
    )

    debugpy_lib = find_debugpy_lib()
    if debugpy_lib:
        debugpy_env = {"PYTHONPATH": os.pathsep.join([str(debugpy_lib), os.environ.get("PYTHONPATH", "")])}
        debugpy_import = run_python(
            "import debugpy, server.server as server; "
            "assert server.app is not None; print('debugpy server import succeeded')",
            extra_env=debugpy_env,
        )
    else:
        debugpy_import = {"skipped": True, "reason": "VS Code debugpy extension not found"}

    live_server = live_server_check()
    checks = {
        "lazy_transformer_lens_import": lazy_import,
        "explicit_dataset_import": explicit_dataset,
        "explicit_evals_import": explicit_evals,
        "debugpy_server_import": debugpy_import,
        "live_server_health": live_server,
    }
    failures = [
        name
        for name, check in checks.items()
        if not check.get("skipped")
        and (check.get("returncode", 0) != 0 or check.get("ok", True) is not True)
    ]
    report = {
        "phase": 1247,
        "python": sys.version,
        "executable": sys.executable,
        "versions": {
            "datasets": importlib.metadata.version("datasets"),
            "pyarrow": importlib.metadata.version("pyarrow"),
            # transformer_lens is vendored in this repository rather than
            # installed as a wheel, so package metadata is intentionally absent.
            "transformer_lens": "local-repository-copy",
        },
        "checks": checks,
        "failures": failures,
        "passed": not failures,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
