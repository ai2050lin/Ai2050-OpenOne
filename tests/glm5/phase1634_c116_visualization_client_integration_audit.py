#!/usr/bin/env python3
"""Audit the C115-C116 parameter-level heatmap client integration."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1630_c116_negation_scope_observation_campaign"
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
    c115_effects = [row for row in payload["effect_rows"] if row.get("dataset") == "C115"]
    c116_effects = [row for row in payload["effect_rows"] if row.get("dataset") == "C116"]
    c116_raw = [row for row in payload["raw_rows"] if row.get("dataset") == "C116"]
    nomination = payload["c116_batch"]["nomination"]
    checks = {
        "closure": core.load(OUT / "audit/independent_closure_audit.json")["all_checks_passed"],
        "asset_hash": core.sha(PUBLIC) == closure["heatmap"]["sha256"],
        "full_coordinates": payload["dimensions"] == list(range(2560)) and all(len(row["values"]) == 2560 for row in [*c115_effects, *c116_effects, *c116_raw]),
        "c115": len(c115_effects) == 280 and len(payload["c115_batch"]["summaries"]) == 8,
        "c116": len(c116_effects) == 231 and len(c116_raw) == 24 and len(payload["c116_batch"]["summaries"]) == 4,
        "candidate": any(row["role"] == nomination["role"] and row["state"] == nomination["state"] for row in c116_effects),
        "route": "C109-C116 Relation-Role-State Activation Atlas" in route and "all 2560" not in route,
        "component": all(term in component for term in ("C115第五词汇场", "C116否定作用域场", "c115Rows", "c116Rows")),
        "boundary": "语义神经元" in route and "新数学" in route,
    }
    report = {"phase": 1634, "campaign": "C116", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "asset_sha256": core.sha(PUBLIC), "route_sha256": core.sha(ROUTE), "component_sha256": core.sha(COMPONENT)}
    if not report["all_checks_passed"]:
        raise RuntimeError(report)
    core.save(OUT / "audit/visualization_client_integration_audit.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
