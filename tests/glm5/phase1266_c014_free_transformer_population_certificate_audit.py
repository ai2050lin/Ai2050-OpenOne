"""Independent audit for Phase1266 C014 WP03."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
MAIN_PATH = ROOT / "tests/glm5/phase1266_c014_free_transformer_population_certificate.py"
SPEC = importlib.util.spec_from_file_location("phase1266_main", MAIN_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load Phase1266 main module")
main = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = main
SPEC.loader.exec_module(main)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def preaudit() -> dict[str, Any]:
    protocol = main.read_json(main.PROTOCOL)
    rows = main.read_jsonl(main.MATERIAL)
    expected = main.protocol_payload(rows)
    dependency = main.read_json(main.PHASE1264_FINAL)
    dependency_audit = main.read_json(main.PHASE1264_AUDIT)
    invalid_predecessor = main.read_json(main.PHASE1265_INVALID)
    counts = {name: sum(row["partition"] == name for row in rows) for name in main.PARTITION_COUNTS}
    oracle = [row for row in rows if row["partition"] == "oracle"]
    expected_radius = math.sqrt(
        math.log(2.0 * main.MAX_REGISTERED_EVENTS * len(main.CANDIDATES) / main.GLOBAL_ERROR_BUDGET)
        / (2.0 * main.SELECTION_DRAWS)
    )
    row_digests = True
    for row in rows:
        value = dict(row)
        stored = value.pop("row_digest")
        row_digests = row_digests and main.digest(value) == stored
    oracle_keys = {
        (
            row["source_code"],
            row["target_code"],
            row["shift0"],
            row["shift1"],
            tuple(row["codebook_order"]),
        )
        for row in oracle
    }
    checks = {
        "dependency_passed": dependency.get("passed") and dependency_audit.get("all_checks_passed"),
        "invalid_predecessor_registered": invalid_predecessor.get("status") == "invalid_engineering_run",
        "protocol_digest": protocol.get("protocol_digest") == expected.get("protocol_digest"),
        "source_hashes": protocol.get("source_hashes") == expected.get("source_hashes"),
        "partition_counts": counts == main.PARTITION_COUNTS,
        "row_digests": row_digests,
        "oracle_unique_complete": len(oracle_keys) == main.PARTITION_COUNTS["oracle"] == 3456,
        "factorial_panels": protocol.get("panels") == list(main.PANELS),
        "candidate_order": protocol.get("candidate_order") == list(main.CANDIDATES) + ["abstain"],
        "radius_recomputed": abs(protocol.get("candidate_camera", {}).get("radius", -1.0) - expected_radius) < 1.0e-15,
        "exact_truth_cannot_select": any("Exact population risks cannot select" in item for item in protocol.get("hard_stops", [])),
        "causal_selection_separate": "causal_selection" in protocol.get("partitions", {}),
        "donor_separate": "donor" in protocol.get("partitions", {}),
        "same_executor_registered": all(name in protocol.get("source_hashes", {}) for name in ("same_executor", "executor", "task")),
        "shape_safe_rff_rank_0": main.feature_matrix(
            "rff",
            torch.zeros((4, 0), dtype=torch.float64),
            torch.zeros((4, 0), dtype=torch.float64),
        ).shape == (4, 1 + 2 * main.RFF_WIDTH),
        "shape_safe_rff_rank_3": main.feature_matrix(
            "rff",
            torch.zeros((4, 1), dtype=torch.float64),
            torch.zeros((4, 2), dtype=torch.float64),
        ).shape == (4, 1 + 3 + 2 + 2 * main.RFF_WIDTH),
        "pretrained_forbidden": "Qwen3" in protocol.get("forbidden_claims", []),
        "memo_not_in_instrument": all(not path.endswith(".md") for path in protocol.get("source_hashes", {})),
    }
    return {
        "mode": "pre",
        "created_at_utc": utc_now(),
        "checks": checks,
        "passed_checks": sum(checks.values()),
        "total_checks": len(checks),
        "all_checks_passed": all(checks.values()),
    }


def causal_passes(metrics: dict[str, Any]) -> bool:
    return (
        metrics["state_relative_error"] <= main.THRESHOLDS["causal_state_relative_error_max"]
        and metrics["correct_output"]["cosine"] >= main.THRESHOLDS["causal_output_cosine_min"]
        and metrics["correct_accuracy"] >= main.THRESHOLDS["causal_correct_accuracy_min"]
        and metrics["wrong_identity_accuracy"] >= main.THRESHOLDS["wrong_identity_accuracy_min"]
        and metrics["wrong_false_target"] <= main.THRESHOLDS["wrong_false_target_max"]
        and metrics["oracle_patch_accuracy"] >= main.THRESHOLDS["oracle_patch_accuracy_min"]
        and metrics["reverse_block_accuracy"] >= main.THRESHOLDS["reverse_block_accuracy_min"]
    )


def final_audit() -> dict[str, Any]:
    protocol = main.read_json(main.PROTOCOL)
    rows = main.read_jsonl(main.MODELS)
    summary_file = main.read_json(main.SUMMARY)
    marker = main.read_json(main.COMPLETE)
    final = main.read_json(main.FINAL)
    final_without_digest = dict(final)
    expected_final_digest = final_without_digest.pop("final_digest")
    row_math = True
    selection_scope = True
    confirmation_math = True
    for row in rows:
        if not row["behavior_qualified"]:
            row_math = row_math and not row["event_ledger"] and not row["selected_events"]
            continue
        for event in row["event_ledger"]:
            exact_class, exact_reason = main.select_point(event["population_risks"])
            point_class, _point_reason = main.select_point(event["sample_risks"])
            bounds = main.confidence_bounds(event["sample_risks"])
            certificate_class, certificate_reason = main.select_certificate(bounds)
            rescue = (
                certificate_class != "abstain"
                and bounds[certificate_class]["upper"] <= main.THRESHOLDS["rescue_authorization_upper_max"]
            )
            row_math = row_math and (
                event["exact_class"] == exact_class
                and event["exact_reason"] == exact_reason
                and event["point_class"] == point_class
                and event["confidence_bounds"] == bounds
                and event["certificate_class"] == certificate_class
                and event["certificate_reason"] == certificate_reason
                and event["rescue_authorized"] == rescue
            )
        admissible = sorted(
            int(layer)
            for layer, metrics in row.get("causal_selection", {}).items()
            if metrics["answer_state_causally_admissible"]
        )
        expected_selected = []
        if admissible:
            expected_selected.append(admissible[0])
            if admissible[-1] != admissible[0]:
                expected_selected.append(admissible[-1])
        selection_scope = selection_scope and row["selected_events"] == expected_selected
        confirmation_math = confirmation_math and all(
            metrics["passed"] == causal_passes(metrics) for metrics in row["causal_confirmations"]
        )
    recomputed = main.summarize(rows)
    expected_gate_set = {
        "G-BEHAVIOR",
        "G-ZERO-FALSE-AUTHORIZATION",
        "G-ROBUST-COVERAGE",
        "G-AMBIGUOUS-ABSTENTION",
        "G-NONTRIVIAL-EVENTS",
        "G-CLASS-DIVERSITY",
        "G-INDEPENDENT-DONOR-CAUSAL",
        "G-CROSS-DEPTH-BREADTH",
    }
    checks = {
        "formal_marker": marker.get("status") == "formal_run_complete",
        "model_count": len(rows) == len(main.MODEL_SEEDS) == summary_file.get("models"),
        "models_hash": main.file_sha256(main.MODELS) == summary_file.get("models_hash") == marker.get("models_hash") == final.get("models_hash"),
        "run_digest": main.digest(rows) == summary_file.get("run_digest") == marker.get("run_digest") == final.get("run_digest"),
        "protocol_digest": protocol.get("protocol_digest") == summary_file.get("protocol_digest") == final.get("protocol_digest"),
        "event_certificate_math": row_math,
        "causal_selection_scope": selection_scope,
        "causal_confirmation_math": confirmation_math,
        "summary_recomputed": all(final.get(key) == value for key, value in recomputed.items()),
        "registered_gate_set": set(final.get("gates", {})) == expected_gate_set,
        "verdict_matches_gates": final.get("passed") == all(final.get("gates", {}).values()),
        "authorization_matches": final.get("authorization", {}).get("new_pretrained_contract") == final.get("passed"),
        "no_automatic_qwen": not final.get("authorization", {}).get("qwen3_automatic"),
        "no_pretrained_loaded": summary_file.get("pretrained_model_loaded") is False,
        "final_digest": main.digest(final_without_digest) == expected_final_digest,
        "scope_narrow": "synthetic cyclic-code" in final.get("claim_boundary", "") and "not natural language" in final.get("claim_boundary", ""),
    }
    return {
        "mode": "final",
        "created_at_utc": utc_now(),
        "checks": checks,
        "passed_checks": sum(checks.values()),
        "total_checks": len(checks),
        "all_checks_passed": all(checks.values()),
        "recomputed": recomputed,
    }


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pre", "final"), required=True)
    args = parser.parse_args()
    result = preaudit() if args.mode == "pre" else final_audit()
    main.atomic_json(main.PREAUDIT if args.mode == "pre" else main.FINAL_AUDIT, result)
    print(json.dumps({"mode": args.mode, "checks": f"{result['passed_checks']}/{result['total_checks']}", "passed": result["all_checks_passed"]}, separators=(",", ":")))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
