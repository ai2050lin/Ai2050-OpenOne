#!/usr/bin/env python3
"""Audit C111 role-state observation integration in the parameter atlas client."""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1612_c111_value_identity_role_coalition_observation"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    closure = core.load(OUT / "analysis/closure.json")
    source = core.load(OUT / "audit/independent_closure_audit.json")
    payload = core.load(PUBLIC)
    component = (ROOT / "frontend/src/components/app/ResearchHeatmapRoute.jsx").read_text(encoding="utf-8")
    with urllib.request.urlopen("http://127.0.0.1:5173/vis_data/research_kernel/c109_role_state_field_atlas.json", timeout=30) as response:
        status = response.status
        length = int(response.headers.get("Content-Length") or 0)
    checks = {
        "source": source["all_checks_passed"],
        "schema": payload["phase"] == 1614,
        "hash": core.sha(PUBLIC) == closure["heatmap"]["sha256"],
        "component": "c111TrajectoryRows" in component and "C111跨词表角色形成图" in component,
        "trajectory": len(payload["c111_observation"]["trajectory_rows"]) == 518,
        "coordinates": payload["dimensions"] == list(range(2560)),
        "build": bool(list((ROOT / "frontend/dist/assets").glob("main-*.js"))),
        "http": status == 200 and length == PUBLIC.stat().st_size,
    }
    result = {"phase": 1614, "campaign": "C111", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "asset_bytes": PUBLIC.stat().st_size, "asset_sha256": core.sha(PUBLIC), "url": "http://127.0.0.1:5173/", "visual_browser_check": "unavailable: no connected browser; production build and exact HTTP asset checks passed"}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/visualization_client_integration_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
