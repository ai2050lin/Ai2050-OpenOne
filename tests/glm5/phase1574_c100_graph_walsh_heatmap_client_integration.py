#!/usr/bin/env python3
"""Phase1574: verify C100 graph-Walsh heatmap client integration."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
SOURCE = TESTS / "result/phase1573_c100_graph_field_analysis_adapter"
OUT = TESTS / "result/phase1574_c100_graph_walsh_heatmap_client_integration"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

FILES = {
    "route": ROOT / "frontend/src/researchKernel/heatmapResearchRoute.js",
    "hook": ROOT / "frontend/src/researchKernel/useResearchKernel.js",
    "card": ROOT / "frontend/src/components/app/ResearchHeatmapRoute.jsx",
    "app": ROOT / "frontend/src/App.jsx",
    "asset": ROOT / "frontend/public/vis_data/research_kernel/c100_graph_walsh_heatmap.json",
    "built_asset": ROOT / "frontend/dist/vis_data/research_kernel/c100_graph_walsh_heatmap.json",
}


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1574 exists")
    parent = core.load(SOURCE / "analysis/final.json")
    audit = core.load(SOURCE / "audit/independent_final_audit.json")
    texts = {key: path.read_text(encoding="utf-8") for key, path in FILES.items() if key in {"route", "hook", "card", "app"}}
    asset = core.load(FILES["asset"])
    checks = {
        "parent": parent["all_checks_passed"] and audit["all_checks_passed"],
        "schema": asset["schema"] == "graph_walsh_heatmap.v1" and asset["result_type"] == "graph_walsh_heatmap",
        "dimensions": len(asset["dimensions"]) == 64,
        "rows": len(asset["rows"]) == 648,
        "route": "GRAPH_WALSH_HEATMAP_ROUTE" in texts["route"] and "c100_graph_walsh_heatmap.json" in texts["route"],
        "hook": "graphWalshHeatmap" in texts["hook"] and "setGraphWalshHeatmap" in texts["hook"],
        "card": "buildGraphWalshHeatmapData" in texts["card"] and "Directed Graph Walsh Heatmap" in texts["card"],
        "app": "graphWalshHeatmap={realResearchTrace.graphWalshHeatmap}" in texts["app"],
        "public_identity": core.sha(FILES["asset"]) == parent["visualization"]["sha256"],
        "build_identity": FILES["built_asset"].exists() and core.sha(FILES["asset"]) == core.sha(FILES["built_asset"]),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    c101_requirements = {
        "campaign": "C101",
        "status": "requirements_frozen_not_started",
        "scientific_order": ["observe", "find_structure", "freeze_prediction", "validate", "intervene"],
        "shared_constraints": {
            "models": ["Qwen3-4B first; GLM4 and DS7B only in separate sequential breadth runs"],
            "representation_scope": "token embeddings and all hidden states only",
            "forbidden_primary_methods": ["attention decomposition", "MLP decomposition", "PCA", "learned probes"],
            "required_controls": [
                "semantic uniqueness audit",
                "naturalness audit with typed human-review missingness",
                "exact positional and answer-code balance",
                "causal-prefix numerical zero test",
                "design-preserving raw-cell null",
            ],
            "claim_rule": "behavior failure types hidden results as diagnostic; it does not erase observations",
        },
        "confirmation_arm": {
            "purpose": "fresh preregistered test of the post-hoc C100 late boundary pattern",
            "material": "new graph entities and templates with no lexical overlap with C100",
            "primary_event": "answer boundary",
            "primary_state": 24,
            "secondary_states": [31, 32],
            "primary_effect": "Walsh xy relation-match contrast",
            "primary_rule": "evaluate the frozen full-coordinate contrast without selecting a state or role on new data",
            "support_rule": "C100 discovery support may be evaluated as a fixed diagnostic but cannot replace the full-vector primary",
        },
        "breadth_observation_arm": {
            "purpose": "map where related conditional response structure appears across language pattern families",
            "pattern_families": [
                "attribute binding",
                "agent-patient role reversal",
                "negation and scope",
                "whole-part with licensed exceptions",
            ],
            "states": "all",
            "roles": "all semantically compiled token roles plus answer boundary",
            "analysis": "raw-coordinate finite differences and typed missingness; no universal gate during observation",
            "claim_rule": "patterns found here remain exploratory until a later fresh-material confirmation",
        },
        "start_condition": "independent zero-model and material audit passes before any model load",
        "forbidden_after_reveal": "changing the confirmation state, role, effect, split, null, or threshold",
    }
    core.save(OUT / "protocol/c101_requirements.json", c101_requirements)
    report = {
        "phase": 1574,
        "campaign": "C100",
        "status": "graph_walsh_heatmap_client_integration_verified",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "external_commands": {
            "targeted_eslint": "passed",
            "vite_production_build": "passed_with_existing_chunk_size_warning",
            "full_App_eslint": "not_claimed_due_preexisting_unrelated_errors",
        },
        "files": {key: {"path": str(path.relative_to(ROOT)), "sha256": core.sha(path)} for key, path in FILES.items()},
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": "append_phase1571_1574_major_stage_memo",
        "next_campaign": "C101 requirements frozen; model run not started",
    }
    core.save(OUT / "analysis/client_integration.json", report)
    core.save(OUT / "analysis/final.json", {"phase": 1574, "campaign": "C100", "status": report["status"], "authorization": report["authorization"]})
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
