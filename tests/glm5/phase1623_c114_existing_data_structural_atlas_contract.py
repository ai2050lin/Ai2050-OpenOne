#!/usr/bin/env python3
"""Phase1623 / C114: freeze a descriptive C112-C113 structural atlas."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
C112 = TESTS / "result/phase1615_c112_value_identity_role_lattice"
C113 = TESTS / "result/phase1618_c113_fourth_lexicon_role_lattice_replication"
OUT = TESTS / "result/phase1623_c114_existing_data_structural_atlas"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"C114 already exists: {OUT}")
    closure = core.load(C113 / "analysis/closure.json")
    audit = core.load(C113 / "audit/independent_closure_audit.json")
    if not audit["all_checks_passed"] or not closure["next_authorization"].startswith("C114 existing-data structural atlas"):
        raise RuntimeError("C114 authorization missing")
    sources = {
        "c112_summary": C112 / "analysis/batch_intervention_summary.jsonl",
        "c112_results": C112 / "analysis/batch_intervention_results.jsonl",
        "c112_closure": C112 / "analysis/closure.json",
        "c113_summary": C113 / "analysis/intervention_summary.jsonl",
        "c113_results": C113 / "analysis/intervention_results.jsonl",
        "c113_closure": C113 / "analysis/closure.json",
    }
    contract = {
        "phase": 1623, "campaign": "C114", "created_at_utc": now(),
        "status": "existing_data_structural_atlas_contract_frozen",
        "object": "descriptively separate cross-lexicon coordinate-assignment grades, query-centered role gains, and protocol-stage gains without new model runs",
        "datasets": ["C112", "C113"], "families": ["attribute_binding", "agent_patient"],
        "cells": 16, "pairs": 384, "independent_lexical_units": 48,
        "common_modes": ["frozen_support", "single_focus_pre", "single_focus_record", "single_focus_post", "single_query_focus", "single_query_anchor", "single_code_instruction", "single_boundary", "coalition_record_to_query_path", "coalition_all_registered_roles"],
        "registered_views": {
            "coordinate_assignment": ["correct_minus_permutation_median", "conservative_rank_among_correct_plus_eight_permutations", "strict_win_all"],
            "multi_position": ["path_minus_query", "all_minus_path", "single_role_positive_cells"],
            "c113_only_composition": ["leave_one_path_losses", "code_over_path", "boundary_over_path_code", "focus_pre_over_staged_all"],
        },
        "analysis_policy": "descriptive counts, ranges, signs, and exact cell tables only; no PCA, clustering, p-value, significance, or confirmatory upgrade",
        "missingness": "C112 did not preregister leave-one or staged coalitions; those fields remain C113-only and cannot be backfilled",
        "claim_boundary": "existing exposed Qwen3 controlled-English activation interventions; no independent replication, natural-route, minimality, semantic-neuron, weight, attention/MLP, or universal-language claim",
        "source_paths": {name: str(path) for name, path in sources.items()},
        "source_hashes": {name: core.sha(path) for name, path in sources.items()},
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "execute_phase1624_c114_structural_atlas_and_freeze_c115_predictions",
    }
    core.save(OUT / "protocol/preregistration.json", contract)
    checks = {
        "sources": all(Path(contract["source_paths"][name]).exists() and core.sha(Path(contract["source_paths"][name])) == digest for name, digest in contract["source_hashes"].items()),
        "counts": contract["cells"] == 16 and contract["pairs"] == 384 and contract["independent_lexical_units"] == 48,
        "views": set(contract["registered_views"]) == {"coordinate_assignment", "multi_position", "c113_only_composition"},
        "descriptive": "no PCA" in contract["analysis_policy"] and "no independent replication" in contract["claim_boundary"],
        "missingness": contract["missingness"].startswith("C112 did not preregister"),
        "authorization": contract["authorization"] == "execute_phase1624_c114_structural_atlas_and_freeze_c115_predictions",
    }
    report = {"phase": 1623, "campaign": "C114", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "producer_sha256": core.sha(Path(__file__)), "protocol_sha256": core.sha(OUT / "protocol/preregistration.json"), "authorization": contract["authorization"]}
    if not report["all_checks_passed"]:
        raise RuntimeError(report)
    core.save(OUT / "audit/internal_contract_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
