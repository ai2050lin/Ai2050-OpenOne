#!/usr/bin/env python3
"""Independent audit for Phase1258."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/glm5/result/phase1258_c010_qwen3_natural_factorial_head_coalition"
PROTOCOL = OUT / "protocol/preregistration.json"
ENVIRONMENT = OUT / "protocol/environment_snapshot.json"
MATERIAL = OUT / "material/frozen_natural_factorial_worlds.jsonl"
PREAUDIT = OUT / "audit/independent_preaudit.json"
DETAILS = OUT / "raw/head_coalition_result.json"
SUMMARY = OUT / "raw/run_summary.json"
COMPLETE = OUT / "raw/FORMAL_RUN_COMPLETE.json"
ANALYSIS = OUT / "analysis/natural_factorial_adjudication.json"
FINAL = OUT / "analysis/final.json"
FINAL_AUDIT = OUT / "audit/independent_final_audit.json"


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rows() -> list[dict[str, Any]]:
    return [json.loads(line) for line in MATERIAL.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def preaudit() -> None:
    protocol = read(PROTOCOL)
    material = rows()
    partition_counts = {name: sum(row["partition"] == name for row in material) for name in protocol["partitions"]}
    template_texts = protocol["template_families"]
    checks = {
        "contract": protocol.get("contract_id") == "EXP-C010-WP02-001",
        "phase1257_dependencies": set(protocol.get("dependencies", {})) == {"phase1257_final", "phase1257_audit"},
        "row_count": len(material) == 256 and partition_counts == protocol["partitions"],
        "five_panels": all(set(row["panels"]) == {"base", "target", "wrong", "null", "joint"} for row in material),
        "factorial_values_distinct": all(len(set(row["values"].values())) == 5 for row in material),
        "single_token_values": protocol["token_audit"]["all_values_single_token"],
        "single_token_names": protocol["token_audit"]["all_names_single_token"],
        "one_token_factorial": protocol["token_audit"]["factorial_pairs_differ_by_one_token"],
        "within_world_equal_length": protocol["token_audit"]["within_world_panel_lengths_equal"],
        "template_holdout": len({text for values in template_texts.values() for text in values}) == sum(len(values) for values in template_texts.values()),
        "all_components_registered": protocol["component_ontology"]["component_count"] == 2340,
        "balanced_shortlist": protocol["selection"]["observational_shortlist"] == {"q": 12, "ov": 24, "mlp": 12},
        "direct_identity_gate": "direct_identity_separation_min" in protocol["thresholds"],
        "decomposed_null_gates": all(key in protocol["thresholds"] for key in ("null_parallel_abs_max", "null_orthogonal_max", "null_total_max")),
        "confirmation_disjoint": any("disjoint" in item for item in protocol["hard_stops"]),
        "no_old_hotspot_seed": any("eight-layer" in item for item in protocol["hard_stops"]),
        "single_formal_run": protocol["budgets"]["max_formal_runs"] == 1 and protocol["budgets"]["max_adaptive_rounds"] == 0,
        "source_hashes": set(protocol.get("source_hashes", {})) == {"main", "auditor"},
    }
    result = {"stage": "preaudit", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    write(PREAUDIT, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


def final_audit() -> None:
    protocol = read(PROTOCOL)
    details = read(DETAILS)
    summary = read(SUMMARY)
    complete = read(COMPLETE)
    analysis = read(ANALYSIS)
    final = read(FINAL)
    behavior = details["behavior"]
    confirmation = details.get("confirmation")
    checks = {
        "formal_marker": complete.get("status") == "formal_run_complete",
        "artifact_links": complete.get("details_sha256") == sha(DETAILS) and complete.get("summary_sha256") == sha(SUMMARY),
        "run_digest_link": complete.get("run_digest") == summary.get("run_digest"),
        "fp16_unquantized": set(details["precision_audit"]["parameter_dtypes"]) == {"float16"} and not details["precision_audit"]["has_quantized_modules"],
        "behavior_accounted": len(behavior["cell_accuracy"]) == 15 and behavior["candidate_finite_fraction"] == 1.0,
        "behavior_first": (not details["traces_captured"]) if not behavior["passed"] else details["traces_captured"],
        "registered_count": details["registered_component_count"] == 2340,
        "shortlist_frozen": (not behavior["passed"]) or len(details["observational_shortlist"]) == 48,
        "causal_discovery_target_only": (not behavior["passed"]) or len(details["causal_discovery"]) == 48,
        "selection_prefixes": (not behavior["passed"]) or [item["size"] for item in details["selection"]] == [1, 2, 4, 8, 12],
        "response_tensor_saved": (not behavior["passed"]) or (confirmation is not None and len(confirmation["response_tensor"]["row_ids"]) == 128),
        "null_decomposed": (not behavior["passed"]) or set(confirmation["null"]) == {"parallel_fraction", "orthogonal_fraction", "total_fraction"},
        "direct_identity_saved": (not behavior["passed"]) or all(key in confirmation for key in ("direct_correct_ratio", "direct_wrong_ratio", "direct_identity_separation")),
        "conditional_saved": (not behavior["passed"]) or "conditional" in confirmation,
        "verdict_consistent": analysis["verdict"] == final["verdict"] and (
            analysis["verdict"] == "qwen3_natural_factorial_head_coalition_confirmed"
        ) == bool(details["passed"]),
        "scope_limited": analysis["authorization"]["glm4_or_ds7b"] is False and analysis["authorization"]["semantic_mechanism_claim"] is False,
        "artifact_set": set(final["artifact_hashes"]) == {"protocol", "environment", "material", "preaudit", "details", "summary", "complete", "analysis"},
    }
    result = {"stage": "final_audit", "checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values())}
    write(FINAL_AUDIT, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preaudit", "final"))
    args = parser.parse_args()
    if args.stage == "preaudit":
        preaudit()
    else:
        final_audit()


if __name__ == "__main__":
    main()
