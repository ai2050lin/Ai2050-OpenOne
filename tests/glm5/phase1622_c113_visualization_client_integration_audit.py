#!/usr/bin/env python3
"""Audit C113 parameter-level heatmap client integration."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1618_c113_fourth_lexicon_role_lattice_replication"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
ROUTE = ROOT / "frontend/src/researchKernel/heatmapResearchRoute.js"
VIEW = ROOT / "frontend/src/components/app/ResearchHeatmapRoute.jsx"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def main() -> None:
    closure = core.load(OUT / "analysis/closure.json")
    independent = core.load(OUT / "audit/independent_closure_audit.json")
    asset = core.load(PUBLIC)
    canonical = OUT / "visualization/c109_c113_coordinate_multi_position_atlas.json"
    route = ROUTE.read_text(encoding="utf-8")
    view = VIEW.read_text(encoding="utf-8")
    checks = {
        "closure": independent["all_checks_passed"],
        "historical_identity": core.sha(canonical) == closure["heatmap"]["sha256"],
        "current_public_superset": "c113_batch" in asset and asset["dimensions"] == list(range(2560)),
        "dimensions": asset["dimensions"] == list(range(2560)),
        "c113_effect_rows": len([row for row in asset["effect_rows"] if row.get("dataset") == "C113"]) == 280,
        "c113_raw_rows": len([row for row in asset["raw_rows"] if row.get("dataset") == "C113"]) == 28,
        "route": "C109-C114 Coordinate Assignment / Multi-Position Atlas" in route and "sourcePath: '/vis_data/research_kernel/c109_role_state_field_atlas.json'" in route,
        "view_data": "c113ModeDefinitions" in view and "c113ModeRows" in view and "c113Scale" in view,
        "view_labels": "C113第四词汇场" in view and "C113留一角色" in view and "等L2坐标、分阶段联盟与留一角色" in view,
        "boundary": "自然传输路径" in route and "语义神经元" in route,
    }
    report = {"phase": 1622, "campaign": "C113", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "producer_sha256": core.sha(Path(__file__)), "asset_sha256": core.sha(PUBLIC), "authorization": "frontend_production_build_and_memo_audit"}
    if not report["all_checks_passed"]:
        raise RuntimeError(report)
    core.save(OUT / "audit/visualization_client_integration_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
