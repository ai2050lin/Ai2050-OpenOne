#!/usr/bin/env python3
"""Independent audit for Phase1678/C144."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"; RESULT = TESTS / "result"
OUT = RESULT / "phase1678_c144_dual_graph_composition_reconstruction"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

freeze = core.load(OUT / "protocol/frozen_composition_model.json")
report = core.load(OUT / "analysis/confirmation_reconstruction.json")
graph = core.load(OUT / "analysis/dual_graph.json")
checks = {
    "discovery_internal": core.load(OUT / "audit/internal_discovery_audit.json")["all_checks_passed"],
    "confirmation_internal": core.load(OUT / "audit/internal_confirmation_audit.json")["all_checks_passed"],
    "closure_internal": core.load(OUT / "audit/internal_closure_audit.json")["all_checks_passed"],
    "frozen_order": report["frozen_order"] == freeze["frozen_order"],
    "five_arms": len(report["arm_results"]) == 5,
    "three_orders": set(report["summary"]) == {"1", "2", "3"},
    "typed_edges": len(graph["edges"]) == 35 and len(graph["language_nodes"]) == 7,
    "finite": all(np.isfinite(x["median_relative_error"]) for x in report["summary"].values()),
}
audit = {"phase": 1678, "campaign": "C144", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "scientific_composition_gate_passed": report["composition_gate_passed"], "authorization": "start_C145"}
core.save(OUT / "audit/independent_closure_audit.json", audit)
print(json.dumps(audit, indent=2))
