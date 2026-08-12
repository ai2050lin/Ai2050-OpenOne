#!/usr/bin/env python3
"""Independent pre-model and result audit for Phase 1224."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1224_final_layer_patch_construct_audit as p1224


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def first_divergence(left: list[int], right: list[int]) -> int:
    for index, (a, b) in enumerate(zip(left, right)):
        if int(a) != int(b):
            return index
    return min(len(left), len(right))


def median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else float("nan")


def fraction(values: list[bool]) -> float:
    return float(sum(bool(value) for value in values) / len(values)) if values else float("nan")


def audit_document(stage: str, checks: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        "phase": p1224.PHASE,
        "stage": stage,
        "created_at": now(),
        "checks": checks,
        "passed_count": sum(bool(row["passed"]) for row in checks),
        "check_count": len(checks),
        "all_checks_passed": all(bool(row["passed"]) for row in checks),
    }
    result["audit_digest"] = digest(result)
    return result


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def preaudit() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    protocol = read_json(p1224.PROTOCOL_PATH)
    manifest = read_jsonl(p1224.MANIFEST_PATH)
    source_final = read_json(p1224.SOURCE_FINAL)
    source_audit = read_json(p1224.SOURCE_RESULT_AUDIT)

    add(checks, "source_final_digest", source_final.get("final_digest") == p1224.EXPECTED_SOURCE_FINAL_DIGEST, source_final.get("final_digest"))
    add(checks, "source_audit", source_audit.get("all_checks_passed") and source_audit.get("audit_digest") == p1224.EXPECTED_SOURCE_RESULT_AUDIT_DIGEST, source_audit.get("audit_digest"))
    source_paths = {
        "script": p1224.SCRIPT,
        "audit_script": p1224.AUDIT_SCRIPT,
        "phase1223_final": p1224.SOURCE_FINAL,
        "phase1223_result_audit": p1224.SOURCE_RESULT_AUDIT,
        "phase1223_protocol": p1224.SOURCE_PROTOCOL,
        "phase1223_pairs": p1224.SOURCE_PAIRS,
        "phase1223_states": p1224.SOURCE_STATES,
        "phase1223_arrays": p1224.SOURCE_ARRAYS,
    }
    hash_ok = all(protocol["source_hashes"].get(key) == file_sha256(path) for key, path in source_paths.items())
    add(checks, "source_hashes", hash_ok, protocol["source_hashes"])
    payload = {key: value for key, value in protocol.items() if key != "protocol_digest"}
    add(checks, "protocol_digest", protocol.get("protocol_digest") == digest(payload), protocol.get("protocol_digest"))
    add(checks, "manifest_digest", protocol["material"]["manifest_digest"] == digest(manifest), protocol["material"]["manifest_digest"])
    add(checks, "cardinality", len(manifest) == 160 and len({row["pair_id"] for row in manifest}) == 160, len(manifest))
    add(checks, "row_digests", all(row.get("row_digest") == digest({key: value for key, value in row.items() if key != "row_digest"}) for row in manifest), "all rows")

    static_ok = True
    first_count = shared_count = discovery_first = holdout_first = 0
    for row in manifest:
        candidates = row["candidate_token_ids"]
        lengths = {len(value) for value in candidates.values()}
        lcp = first_divergence(candidates[row["recipient_gold"]], candidates[row["donor_gold"]])
        static_ok = static_ok and len(lengths) == 1 and lcp == row["gold_first_divergence"]
        static_ok = static_ok and bool(lcp == 0) == row["gold_first_token_discriminative"]
        static_ok = static_ok and row["generation_boundary"] == row["recipient_prompt_length"] - 1
        first_count += int(lcp == 0)
        shared_count += int(lcp > 0)
        discovery_first += int(row["split"] == "discovery" and lcp == 0)
        holdout_first += int(row["split"] in p1224.HOLDOUT_SPLITS and lcp == 0)
    add(checks, "static_token_fields", static_ok, {"first": first_count, "shared": shared_count})
    expected_static = protocol["material"]["static_token_audit"]
    counts_ok = (
        first_count == expected_static["first_token_discriminative_count"] == 10
        and shared_count == expected_static["shared_first_token_count"] == 150
        and discovery_first == expected_static["discovery_first_token_discriminative_count"] == 4
        and holdout_first == expected_static["holdout_first_token_discriminative_count"] == 6
    )
    add(checks, "static_token_counts", counts_ok, {"first": first_count, "shared": shared_count, "discovery_first": discovery_first, "holdout_first": holdout_first})
    split_ok = all(sum(row["split"] == split for row in manifest) == 40 for split in p1224.SPLITS)
    add(checks, "split_balance", split_ok, {split: sum(row["split"] == split for row in manifest) for split in p1224.SPLITS})
    add(checks, "fixed_intervention", protocol["interventions"]["fixed_depth"] == 36 and protocol["interventions"]["fixed_component"] == "final decoder layer whole residual output", protocol["interventions"])
    add(checks, "condition_registry", tuple(protocol["interventions"]["conditions"].keys()) == p1224.CONDITIONS, list(protocol["interventions"]["conditions"]))
    add(checks, "no_search", protocol["interventions"]["no_layer_search"] and protocol["interventions"]["no_role_search"] and protocol["interventions"]["no_head_or_neuron_search"], protocol["interventions"])
    add(checks, "fixed_readout", protocol["readouts"]["fixed_margin"] == "S(donor_gold)-S(recipient_gold) for recipient, donor, and every patched condition", protocol["readouts"]["fixed_margin"])
    add(checks, "thresholds", protocol["thresholds"] == p1224.THRESHOLDS, protocol["thresholds"])
    add(checks, "claim_scope", protocol["claim_boundary"]["construct_validity_not_language_mechanism"] and not protocol["authorization"]["qwen_new_mechanism_scan"], protocol["claim_boundary"])
    no_outputs = not p1224.RECORD_PATH.exists() and not p1224.RUN_SUMMARY_PATH.exists() and not p1224.FINAL_PATH.exists()
    add(checks, "no_model_output_before_preaudit", no_outputs, no_outputs)
    result = audit_document("pre", checks)
    write_json(p1224.PREAUDIT_PATH, result)
    return result


def recompute(records: list[dict[str, Any]]) -> dict[str, Any]:
    holdout = [row for row in records if row["split"] in p1224.HOLDOUT_SPLITS]
    lcp0 = [row for row in holdout if row["gold_first_token_discriminative"]]
    lcp_pos = [row for row in holdout if not row["gold_first_token_discriminative"]]
    conditions = [condition for row in records for condition in row["conditions"].values()]
    boundary = [row["conditions"]["boundary_live"] for row in records]
    sustained = [row["conditions"]["all_scoring_live"] for row in records]
    zero = [row["conditions"]["all_scoring_zero"] for row in records]
    divergence = [row["conditions"]["divergence_live"] for row in records]
    lcp0_values = [row["conditions"]["boundary_live"]["completion"] for row in lcp0]
    lcp_pos_values = [abs(row["conditions"]["boundary_live"]["completion"]) for row in lcp_pos]
    metrics = {
        "record_count": len(records),
        "finite_fraction": fraction([row["finite"] and all(c["finite"] for c in row["conditions"].values()) for row in records]),
        "hook_write_max_abs": max(c["hook_write_max_abs"] for c in conditions),
        "boundary_live_logit_max_abs": max(c["boundary_logit_max_abs_vs_donor"] for c in boundary),
        "boundary_live_top1_agreement": min(c["boundary_top1_agreement"] for c in boundary),
        "all_scoring_score_max_abs": max(c["score_max_abs_vs_donor"] for c in sustained),
        "all_scoring_completion_median": median([c["completion"] for c in sustained]),
        "zero_score_max_abs": max(abs(c["scores"][candidate] - row["recipient_scores"][candidate]) for row, c in zip(records, zero) for candidate in row["recipient_scores"]),
        "prompt_full_logit_max_abs": max(max(row["donor_prompt_full_logit_max_abs"], row["recipient_prompt_full_logit_max_abs"]) for row in records),
        "stored_live_hidden_relative_max": max(row["stored_live_hidden_relative"] for row in records),
        "target_shift_abs_min": min(row["target_shift_abs"] for row in records),
        "divergence_token_score_max_abs": max(c["divergence_token_score_max_abs_vs_donor"] for c in divergence),
        "holdout_lcp0_count": len(lcp0),
        "holdout_lcp0_boundary_completion_median": median(lcp0_values),
        "holdout_lcp0_positive_fraction": fraction([value > 0 for value in lcp0_values]),
        "holdout_lcp_positive_count": len(lcp_pos),
        "holdout_lcp_positive_abs_completion_median": median(lcp_pos_values),
        "discovery_lcp0_boundary_completion_median": median([row["conditions"]["boundary_live"]["completion"] for row in records if row["split"] == "discovery" and row["gold_first_token_discriminative"]]),
    }
    t = p1224.THRESHOLDS
    gates = {
        "finite": metrics["finite_fraction"] >= t["finite_fraction_min"],
        "hook_write": metrics["hook_write_max_abs"] <= t["hook_write_max_abs_max"],
        "boundary_next_logit": metrics["boundary_live_logit_max_abs"] <= t["boundary_live_logit_max_abs_max"],
        "boundary_top1": metrics["boundary_live_top1_agreement"] >= t["boundary_live_top1_agreement_min"],
        "sustained_score": metrics["all_scoring_score_max_abs"] <= t["all_scoring_score_max_abs_max"],
        "sustained_completion": metrics["all_scoring_completion_median"] >= t["all_scoring_completion_median_min"],
        "zero_identity": metrics["zero_score_max_abs"] <= t["zero_score_max_abs_max"],
        "prompt_full_parity": metrics["prompt_full_logit_max_abs"] <= t["prompt_full_logit_max_abs_max"],
        "stored_replay": metrics["stored_live_hidden_relative_max"] <= t["stored_live_hidden_relative_max"],
        "denominator": metrics["target_shift_abs_min"] >= t["target_shift_abs_min"],
        "divergence_token": metrics["divergence_token_score_max_abs"] <= t["divergence_token_score_max_abs_max"],
        "holdout_lcp0_completion": metrics["holdout_lcp0_boundary_completion_median"] >= t["holdout_lcp0_boundary_completion_median_min"],
        "holdout_lcp0_positive": metrics["holdout_lcp0_positive_fraction"] >= t["holdout_lcp0_positive_fraction_min"],
        "holdout_lcp_positive_near_zero": metrics["holdout_lcp_positive_abs_completion_median"] <= t["holdout_lcp_positive_abs_completion_median_max"],
    }
    return {"metrics": metrics, "gates": gates, "passed": all(gates.values())}


def result_audit() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    pre = read_json(p1224.PREAUDIT_PATH)
    protocol = read_json(p1224.PROTOCOL_PATH)
    manifest = read_jsonl(p1224.MANIFEST_PATH)
    records = read_jsonl(p1224.RECORD_PATH)
    summary = read_json(p1224.RUN_SUMMARY_PATH)
    final = read_json(p1224.FINAL_PATH)
    add(checks, "preaudit_passed", pre.get("all_checks_passed"), pre.get("audit_digest"))
    add(checks, "record_cardinality", len(records) == len(manifest) == 160 and len({row["pair_id"] for row in records}) == 160, len(records))
    add(checks, "record_digest", summary["record_digest"] == digest(records), summary["record_digest"])
    add(checks, "record_row_digests", all(row["record_digest"] == digest({key: value for key, value in row.items() if key != "record_digest"}) for row in records), "all records")
    manifest_by_id = {row["pair_id"]: row for row in manifest}
    structure_ok = True
    formula_ok = True
    token_sum_ok = True
    for row in records:
        item = manifest_by_id[row["pair_id"]]
        structure_ok = structure_ok and row["scope"] == item["scope"] and row["split"] == item["split"]
        structure_ok = structure_ok and set(row["conditions"]) == set(p1224.CONDITIONS)
        structure_ok = structure_ok and row["gold_first_divergence"] == item["gold_first_divergence"]
        recipient_margin = row["recipient_scores"][row["donor_gold"]] - row["recipient_scores"][row["recipient_gold"]]
        donor_margin = row["donor_scores"][row["donor_gold"]] - row["donor_scores"][row["recipient_gold"]]
        target_shift = donor_margin - recipient_margin
        formula_ok = formula_ok and math.isclose(recipient_margin, row["recipient_margin"], abs_tol=1e-9)
        formula_ok = formula_ok and math.isclose(donor_margin, row["donor_margin"], abs_tol=1e-9)
        formula_ok = formula_ok and math.isclose(target_shift, row["target_shift"], abs_tol=1e-9)
        for condition in row["conditions"].values():
            margin = condition["scores"][row["donor_gold"]] - condition["scores"][row["recipient_gold"]]
            completion = (margin - recipient_margin) / target_shift
            formula_ok = formula_ok and math.isclose(margin, condition["fixed_margin"], abs_tol=1e-9)
            formula_ok = formula_ok and math.isclose(completion, condition["completion"], abs_tol=1e-9)
            for candidate, score in condition["scores"].items():
                token_sum_ok = token_sum_ok and math.isclose(sum(condition["token_scores"][candidate]), score, abs_tol=1e-9)
        for candidate, score in row["recipient_scores"].items():
            token_sum_ok = token_sum_ok and math.isclose(sum(row["recipient_token_scores"][candidate]), score, abs_tol=1e-9)
        for candidate, score in row["donor_scores"].items():
            token_sum_ok = token_sum_ok and math.isclose(sum(row["donor_token_scores"][candidate]), score, abs_tol=1e-9)
    add(checks, "record_structure", structure_ok, "manifest and conditions")
    add(checks, "fixed_margin_and_completion", formula_ok, "same donor-minus-recipient readout")
    add(checks, "per_token_score_sums", token_sum_ok, "all clean and patched scores")
    add(checks, "patch_calls", all(condition["patch_calls"] == 1 for row in records for condition in row["conditions"].values()), "one hook call each")

    independent = recompute(records)
    add(checks, "metric_recomputation", canonical_json(independent["metrics"]) == canonical_json(final["result"]["metrics"]), independent["metrics"])
    add(checks, "gate_recomputation", independent["gates"] == final["result"]["gates"] and independent["passed"] == final["result"]["passed"], independent["gates"])
    precision = summary["precision_audit"]
    precision_ok = set(precision["parameter_dtypes"]) == {"float16"} and not precision["has_bf16_parameters"] and not precision["has_quantized_modules"]
    add(checks, "fp16_nonquantized", precision_ok, precision)
    summary_payload = {key: value for key, value in summary.items() if key != "summary_digest"}
    add(checks, "summary_digest", summary["summary_digest"] == digest(summary_payload), summary["summary_digest"])
    final_payload = {key: value for key, value in final.items() if key != "final_digest"}
    add(checks, "final_digest", final["final_digest"] == digest(final_payload), final["final_digest"])
    expected_grade = "E3-METHOD" if independent["passed"] else "E3-INSTRUMENT-BOUNDARY"
    add(checks, "claim_scope", final["k_item"]["evidence_grade"] == expected_grade and final["k200_scope_refinement"]["frozen_k200_not_deleted"] and not final["authorized_next"]["qwen_new_mechanism_scan"], final["k_item"])
    result = audit_document("result", checks)
    write_json(p1224.RESULT_AUDIT_PATH, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pre", "result"), required=True)
    args = parser.parse_args()
    result = preaudit() if args.stage == "pre" else result_audit()
    print(canonical_json({"stage": args.stage, "passed": result["all_checks_passed"], "checks": f"{result['passed_count']}/{result['check_count']}", "audit_digest": result["audit_digest"]}))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
