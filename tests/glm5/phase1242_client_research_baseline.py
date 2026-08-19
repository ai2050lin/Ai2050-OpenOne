"""Audit the client-facing research baseline against Phase1236/C001-WP00."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.current_research_progress import build_current_research_progress


def fetch_text(url: str, timeout: float = 5.0) -> tuple[int, str]:
    with urlopen(url, timeout=timeout) as response:
        return response.status, response.read().decode("utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:5173")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "tests/glm5/result/phase1242_client_research_baseline/phase1242_summary.json",
    )
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    snapshot = build_current_research_progress(ROOT)
    current_state_source = (
        ROOT / "frontend/src/researchKernel/currentResearchState.js"
    ).read_text(encoding="utf-8")
    progress_source = (
        ROOT / "frontend/src/blueprint/ResearchProgressTab.jsx"
    ).read_text(encoding="utf-8")
    system_source = (
        ROOT / "frontend/src/blueprint/SystemStatusTab.jsx"
    ).read_text(encoding="utf-8")
    route_source = (
        ROOT / "frontend/src/blueprint/theoryRouteLatestData.js"
    ).read_text(encoding="utf-8")

    root_status, root_body = fetch_text(f"{base_url}/")
    state_status, live_state = fetch_text(
        f"{base_url}/src/researchKernel/currentResearchState.js"
    )
    progress_status, live_progress = fetch_text(
        f"{base_url}/src/blueprint/ResearchProgressTab.jsx"
    )

    checks = {
        "snapshot_phase_1236": snapshot.get("current_phase") == 1236,
        "contract_preregistered": snapshot["systemic"]["experiment_contract"]["status"]
        == "preregistered",
        "run_ready_false": snapshot["systemic"]["experiment_contract"]["run_ready"]
        is False,
        "no_numeric_convergence_claim": snapshot["systemic"]["convergence_index"] is None,
        "registry_counts_match": snapshot["systemic"]["registry_counts"]["evidence"]
        == 14
        and snapshot["systemic"]["registry_counts"]["constructs"] == 4,
        "client_state_phase_1236": "phase: 1236" in current_state_source,
        "client_contract_blocked": "runReady: false" in current_state_source,
        "typed_gate_visible": "1,0,1,1" in progress_source,
        "wp01_preflight_visible": "WP01 无模型冻结与反泄漏预审计" in progress_source,
        "system_status_is_c001": "C001 全局结构辨识系统状态" in system_source,
        "glm5_route_is_current": "Phase 1236 / C001-WP00" in route_source,
        "live_root_http_200": root_status == 200 and 'id="root"' in root_body,
        "live_state_http_200": state_status == 200 and "phase: 1236" in live_state,
        "live_progress_http_200": progress_status == 200
        and "WP01 无模型冻结与反泄漏预审计" in live_progress,
    }
    passed = all(checks.values())
    result = {
        "schema_version": "1.0.0",
        "phase_id": "Phase1242",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "passed": passed,
        "checks": checks,
        "scientific_baseline": "Phase1236/C001-WP00",
        "latest_engineering_phase": 1241,
        "contract_id": "EXP-C001-WP01-001",
        "contract_status": "preregistered",
        "run_ready": False,
        "model_runs": 0,
        "gpu_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
