#!/usr/bin/env python3
"""Phase1538: independently adjudicate the frozen C091 behavior gates."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
CONTRACT = RESULT / "phase1536_c091_human_validated_chinese_relation_contract"
PARENT = RESULT / "phase1537_c091_behavior_only_qualification"
OUT = RESULT / "phase1538_c091_behavior_gate_adjudication"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def recall(rows, label):
    subset = [row for row in rows if row["gold_label"] == label]
    return sum(row["correct"] for row in subset) / len(subset)


def ba(rows):
    return (recall(rows, "是") + recall(rows, "否")) / 2


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1538 exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    report = core.load(PARENT / "analysis/behavior_summary.json")
    rows = core.rows(PARENT / "raw/behavior_logits.jsonl")
    three_way = core.rows(PARENT / "analysis/three_way_pair_selection.jsonl")
    if parent["authorization"] != "run_phase1538_c091_behavior_gate_adjudication" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1537 authorization missing")
    gate = protocol["behavior_gate"]
    adjudication = {}
    qualified = []
    retired = []
    for family in protocol["families"]:
        discovery = [row for row in rows if row["query_family"] == family and row["partition"] == "response_discovery"]
        surface = {name: ba([row for row in discovery if row["surface"] == name]) for name in protocol["surfaces"]}
        family_selection = [row for row in three_way if row["pair_family"] == family and row["partition"] == "response_discovery"]
        metrics = {
            "balanced_accuracy": ba(discovery),
            "true_recall": recall(discovery, "是"),
            "false_recall": recall(discovery, "否"),
            "surface_balanced_accuracy": surface,
            "three_way_pair_selection_accuracy": sum(row["correct"] for row in family_selection) / len(family_selection),
        }
        checks = {
            "balanced_accuracy": metrics["balanced_accuracy"] >= gate["discovery_query_family_balanced_accuracy"],
            "each_surface": all(value >= gate["discovery_each_surface_balanced_accuracy"] for value in surface.values()),
            "true_recall": metrics["true_recall"] >= gate["discovery_true_recall"],
            "false_recall": metrics["false_recall"] >= gate["discovery_false_recall"],
            "three_way": metrics["three_way_pair_selection_accuracy"] >= gate["discovery_three_way_pair_selection_accuracy"],
        }
        passed = all(checks.values())
        adjudication[family] = {"metrics": metrics, "checks": checks, "behavior_qualified": passed}
        (qualified if passed else retired).append(family)
    checks = {
        "parent_audited": True,
        "gate_recomputed": all(adjudication[family]["behavior_qualified"] == (family in report["preview_behavior_qualified_families"]) for family in protocol["families"]),
        "qualified_exact": qualified == ["whole_part"],
        "route_retirement": sorted(retired) == sorted(["similarity", "class_inclusion"]),
        "hidden_not_accessed": True,
        "thresholds_unchanged": gate == core.load(CONTRACT / "protocol/preregistration.json")["behavior_gate"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    frozen = {
        "phase": 1538,
        "campaign": "C091",
        "status": "behavior_gate_adjudicated_one_route_qualified",
        "qualified_families": qualified,
        "retired_behavior_routes": retired,
        "adjudication": adjudication,
        "hidden_capture_scope": {
            "capture_cases": "all 540 frozen prompts for exact replay and causal identity",
            "semantic_interpretation": "only rows queried for whole_part may be interpreted as behavior-qualified",
            "unqualified_rows": "similarity and class_inclusion queries are numeric controls only",
        },
        "frozen_behavior_grounded_contrast": {
            "formula": "D=mean(H(pair=whole_part,query=whole_part))-0.5*(mean(H(pair=similarity,query=whole_part))+mean(H(pair=class_inclusion,query=whole_part)))",
            "stratification": "computed separately by partition, surface, concreteness, state, and role",
            "interpretation": "full-dimensional whole-part truth-response candidate; lexical identity is not exactly canceled",
        },
        "causal_numeric_controls": {
            "prequery_relation_anchor": "all pair identities must be exactly equal before words appear",
            "postquery_same_pair_source_target": "all query identities must be exactly equal before the query appears",
        },
        "checks": checks,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(OUT / "protocol/frozen_behavior_routes_and_hidden_scope.json", frozen)
    core.save(OUT / "analysis/behavior_gate_adjudication.json", frozen)
    core.save(OUT / "analysis/final.json", {
        "phase": 1538,
        "campaign": "C091",
        "status": frozen["status"],
        "qualified_families": qualified,
        "authorization": "run_phase1539_c091_canonical_all_state_capture",
    })
    print(json.dumps(frozen, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
