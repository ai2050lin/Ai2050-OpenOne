#!/usr/bin/env python3
"""Audit C114 structural-atlas client integration."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1623_c114_existing_data_structural_atlas"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
ROUTE = ROOT / "frontend/src/researchKernel/heatmapResearchRoute.js"
VIEW = ROOT / "frontend/src/components/app/ResearchHeatmapRoute.jsx"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    closure = core.load(OUT / "analysis/closure.json")
    independent = core.load(OUT / "audit/independent_closure_audit.json")
    payload = core.load(PUBLIC)
    route = ROUTE.read_text(encoding="utf-8")
    view = VIEW.read_text(encoding="utf-8")
    checks = {
        "closure": independent["all_checks_passed"],
        "identity": core.sha(PUBLIC) == closure["heatmap"]["sha256"],
        "payload": payload["phase"] == 1624 and len(payload["c114_structural_atlas"]["cells"]) == 16,
        "coordinates": payload["dimensions"] == list(range(2560)),
        "route": "C109-C114 Coordinate Assignment / Multi-Position Atlas" in route,
        "view_data": "c114Definitions" in view and "c114Rows" in view and "c114Scale" in view,
        "view_labels": "C114跨词汇坐标规律" in view and "C114跨 C112-C113 描述性结构图谱" in view,
        "boundary": "只压缩已暴露" in route and "自然传输路径" in route,
    }
    report = {"phase": 1624, "campaign": "C114", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "producer_sha256": core.sha(Path(__file__)), "asset_sha256": core.sha(PUBLIC), "authorization": "production_build_append_memo_and_close"}
    if not report["all_checks_passed"]:
        raise RuntimeError(report)
    core.save(OUT / "audit/visualization_client_integration_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
