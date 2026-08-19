"""Validate the live Vite frontend and the VS Code full-stack launch contract."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[2]


def fetch(url: str, timeout: float = 5.0) -> tuple[int, str, str]:
    with urlopen(url, timeout=timeout) as response:
        return (
            response.status,
            response.headers.get("Content-Type", ""),
            response.read().decode("utf-8", errors="replace"),
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:5173")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "tests/glm5/result/phase1241_frontend_availability/phase1241_summary.json",
    )
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    root_status, root_type, root_body = fetch(f"{base_url}/")
    client_status, client_type, client_body = fetch(f"{base_url}/@vite/client")
    main_status, main_type, main_body = fetch(f"{base_url}/src/main.jsx")

    launch_path = ROOT / ".vscode/launch.json"
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    compounds = {item["name"]: item for item in launch.get("compounds", [])}
    frontend_configs = {
        item["name"]: item
        for item in launch.get("configurations", [])
        if item.get("name") == "Frontend"
    }
    full_stack = compounds.get("Full Stack (Backend + Frontend)", {})
    frontend = frontend_configs.get("Frontend", {})

    checks = {
        "root_http_200": root_status == 200,
        "root_mount_present": 'id="root"' in root_body,
        "vite_client_http_200": client_status == 200,
        "vite_client_transformed": "createHotContext" in client_body,
        "main_module_http_200": main_status == 200,
        "main_module_transformed": "react-refresh" in main_body or "createRoot" in main_body,
        "full_stack_does_not_stop_all": full_stack.get("stopAll") is False,
        "frontend_uses_repository_launcher": "start_visualization.ps1"
        in frontend.get("command", ""),
    }
    passed = all(checks.values())
    result = {
        "schema_version": "1.0.0",
        "phase_id": "Phase1241",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "base_url": base_url,
        "passed": passed,
        "checks": checks,
        "responses": {
            "root": {"status": root_status, "content_type": root_type, "bytes": len(root_body)},
            "vite_client": {
                "status": client_status,
                "content_type": client_type,
                "bytes": len(client_body),
            },
            "main_module": {
                "status": main_status,
                "content_type": main_type,
                "bytes": len(main_body),
            },
        },
        "model_runs": 0,
        "gpu_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
