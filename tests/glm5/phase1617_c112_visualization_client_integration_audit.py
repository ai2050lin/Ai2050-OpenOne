#!/usr/bin/env python3
"""Audit the C112 response lattice in the parameter-level atlas client."""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1615_c112_value_identity_role_lattice"
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
        "schema": payload["phase"] == 1617,
        "hash": core.sha(PUBLIC) == closure["heatmap"]["sha256"],
        "component": "c112ModeRows" in component and "C112等L2坐标与角色响应格" in component,
        "summaries": len(payload["c112_batch"]["summaries"]) == 8,
        "coordinates": payload["dimensions"] == list(range(2560)),
        "build": bool(list((ROOT / "frontend/dist/assets").glob("main-*.js"))),
        "http": status == 200 and length == PUBLIC.stat().st_size,
    }
    result = {"phase": 1617, "campaign": "C112", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "asset_bytes": PUBLIC.stat().st_size, "asset_sha256": core.sha(PUBLIC), "url": "http://127.0.0.1:5173/", "visual_browser_check": "unavailable: no connected browser; production build and exact HTTP asset checks passed"}
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/visualization_client_integration_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
