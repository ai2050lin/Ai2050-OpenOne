#!/usr/bin/env python3
"""Audit the C109 parameter-level heatmap client integration."""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1603_c109_fresh_role_state_field_atlas"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    closure = core.load(OUT / "analysis/closure.json")
    source_audit = core.load(OUT / "audit/independent_closure_audit.json")
    payload = core.load(PUBLIC)
    route = (ROOT / "frontend/src/researchKernel/heatmapResearchRoute.js").read_text(encoding="utf-8")
    hook = (ROOT / "frontend/src/researchKernel/useResearchKernel.js").read_text(encoding="utf-8")
    component = (ROOT / "frontend/src/components/app/ResearchHeatmapRoute.jsx").read_text(encoding="utf-8")
    app = (ROOT / "frontend/src/App.jsx").read_text(encoding="utf-8")
    with urllib.request.urlopen("http://127.0.0.1:5173/vis_data/research_kernel/c109_role_state_field_atlas.json", timeout=30) as response:
        http_status = response.status
        content_length = int(response.headers.get("Content-Length") or 0)
    dist_assets = list((ROOT / "frontend/dist/assets").glob("main-*.js"))
    checks = {
        "source": source_audit["all_checks_passed"],
        "schema": payload["schema"] == "c109_role_state_field_atlas.v1",
        "hash": core.sha(PUBLIC) == closure["heatmap"]["sha256"],
        "route": "C109_ROLE_STATE_FIELD_ATLAS_ROUTE" in route and "c109_role_state_field_atlas.json" in route,
        "hook": "c109RoleStateFieldAtlas" in hook and "setC109RoleStateFieldAtlas" in hook,
        "component": "buildC109RoleStateFieldAtlasData" in component and "C109 Fresh Role-State Field Atlas" in component,
        "app": "c109RoleStateFieldAtlas={realResearchTrace.c109RoleStateFieldAtlas}" in app,
        "all_coordinates": payload["dimensions"] == list(range(2560)) and "全部参数" in component,
        "embedding_hidden": {row["state_kind"] for row in payload["raw_rows"]} == {"embedding", "hidden_state"},
        "build": bool(dist_assets) and (ROOT / "frontend/dist/index.html").exists(),
        "http": http_status == 200 and content_length == PUBLIC.stat().st_size,
    }
    result = {
        "phase": 1606,
        "campaign": "C109",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "asset_bytes": PUBLIC.stat().st_size,
        "asset_sha256": core.sha(PUBLIC),
        "url": "http://127.0.0.1:5173/",
        "visual_browser_check": "unavailable: no in-app or extension browser was connected; production build and HTTP asset checks passed",
    }
    if not result["all_checks_passed"]:
        raise RuntimeError(result)
    core.save(OUT / "audit/visualization_client_integration_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
