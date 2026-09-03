#!/usr/bin/env python3
"""Audit the C117 all-coordinate heatmap client integration."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1635_c117_whole_part_exception_observation_campaign"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c109_role_state_field_atlas.json"
ROUTE = ROOT / "frontend/src/researchKernel/heatmapResearchRoute.js"
COMPONENT = ROOT / "frontend/src/components/app/ResearchHeatmapRoute.jsx"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core


def main() -> None:
    closure = core.load(OUT / "analysis/closure.json")
    payload = core.load(PUBLIC)
    route = ROUTE.read_text(encoding="utf-8")
    component = COMPONENT.read_text(encoding="utf-8")
    effects = [row for row in payload["effect_rows"] if row.get("dataset") == "C117"]
    raw = [row for row in payload["raw_rows"] if row.get("dataset") == "C117"]
    nomination = payload["c117_batch"]["nomination"]
    checks = {
        "closure": core.load(OUT / "audit/independent_closure_audit.json")["all_checks_passed"],
        "asset_hash": core.sha(PUBLIC) == closure["heatmap"]["sha256"],
        "full_coordinates": payload["dimensions"] == list(range(2560)) and all(len(row["values"]) == 2560 for row in [*effects, *raw]),
        "c117": len(effects) in (210, 231) and len(raw) == 24 and len(payload["c117_batch"]["summaries"]) == 4,
        "candidate": any(row["role"] == nomination["role"] and row["state"] == nomination["state"] for row in effects),
        "residual": "common_component_residual" in payload["c117_batch"]["validation"],
        "route": "C109-C117 Relation-Role-State Activation Atlas" in route and "正交子空间" in route and "拓扑" in route,
        "component": all(term in component for term in ("C117默认-例外场", "C117共同分量/残差", "c117Rows", "c117Batch")),
        "boundary": "not weights" in closure["claim_boundary"] and "attention/MLP" in closure["claim_boundary"] and "topology" in closure["claim_boundary"],
    }
    report = {
        "phase": 1639,
        "campaign": "C117",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "asset_sha256": core.sha(PUBLIC),
        "route_sha256": core.sha(ROUTE),
        "component_sha256": core.sha(COMPONENT),
    }
    if not report["all_checks_passed"]:
        raise RuntimeError(report)
    core.save(OUT / "audit/visualization_client_integration_audit.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
