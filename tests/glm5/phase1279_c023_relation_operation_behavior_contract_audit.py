#!/usr/bin/env python3
"""Independent zero-model audit for Phase1279."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/glm5/result/phase1279_c023_relation_operation_behavior_contract"
PROTOCOL = OUT / "protocol/preregistration.json"
MATERIAL = OUT / "material/frozen_relation_worlds.jsonl"
FINAL = OUT / "analysis/final.json"
AUDIT = OUT / "audit/independent_final_audit.json"
OPERATIONS = ("contrast", "addition", "cause", "sequence")
PANELS = ("base", "target", "wrong", "null", "joint", "surface", "implicit")
COUNTS = {"discovery": 64, "selection": 64, "confirmation": 128}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def check(name: str, passed: bool, detail: Any = None) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def run() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in MATERIAL.read_text(encoding="utf-8").splitlines() if line.strip()]
    final = json.loads(FINAL.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []
    checks.append(check("row_count", len(rows) == 256, len(rows)))
    counts = Counter(row["partition"] for row in rows)
    checks.append(check("partition_counts", dict(counts) == COUNTS, dict(counts)))
    checks.append(check("row_ids_unique", len({row["row_id"] for row in rows}) == len(rows)))
    checks.append(check("content_ids_unique", len({row["content_id"] for row in rows}) == len(rows)))
    checks.append(check("row_digests", all(digest({key: value for key, value in row.items() if key != "row_digest"}) == row["row_digest"] for row in rows)))
    checks.append(check("panel_registry", all(set(row["panels"]) == set(PANELS) for row in rows)))
    checks.append(check("expected_registry", all(set(row["expected"]) == set(PANELS) for row in rows)))
    checks.append(check("factorial_semantics", all(
        row["expected"]["base"] == row["operations"]["base"]
        and row["expected"]["target"] == row["operations"]["target"]
        and row["expected"]["wrong"] == row["operations"]["wrong"]
        and row["expected"]["null"] == row["operations"]["base"]
        and row["expected"]["joint"] == row["operations"]["target"]
        for row in rows
    )))
    target_balance = {
        partition: Counter(row["operations"]["target"] for row in rows if row["partition"] == partition)
        for partition in COUNTS
    }
    checks.append(check("target_balance", all(max(value.values()) - min(value.values()) == 0 for value in target_balance.values()), target_balance))
    pair_coverage = {
        partition: len({(row["operations"]["base"], row["operations"]["target"]) for row in rows if row["partition"] == partition})
        for partition in COUNTS
    }
    checks.append(check("all_ordered_base_target_pairs", all(value == 12 for value in pair_coverage.values()), pair_coverage))
    checks.append(check("operations_distinct", all(len(set(row["operations"].values())) == 3 for row in rows)))
    checks.append(check("quoted_null_contains_target_word", all(
        row["markers"]["target"] in row["panels"]["null"]
        and row["markers"]["base"] in row["panels"]["null"] for row in rows
    )))
    checks.append(check("partition_content_prefix", all(row["content_id"].startswith(row["partition"] + ".") for row in rows)))
    token = protocol["token_audit"]
    checks.append(check("tokenizer_agreement", token["fast_slow_tokenization_equal"]))
    checks.append(check("single_token_candidates_and_markers", token["candidate_suffix_single_token"] and token["operative_markers_single_token"]))
    checks.append(check("one_token_factorials", token["active_panels_equal_length_one_token_difference"] and token["null_joint_equal_length_one_token_difference"]))
    checks.append(check("event_coverage", all(
        set(token["event_token_indices"][row["row_id"]]) == {"base", "target", "wrong", "null", "joint", "surface"}
        for row in rows
    )))
    checks.append(check("thresholds_frozen", protocol["thresholds"] == {
        "candidate_finite_fraction_min": 1.0,
        "factorial_cell_accuracy_min": 0.9,
        "surface_cell_accuracy_min": 0.85,
        "implicit_cell_accuracy_min": 0.8,
        "operation_macro_accuracy_min": 0.85,
        "gold_margin_median_min": 0.5,
    }))
    checks.append(check("zero_model_claim", final["verdict"] == "relation_operation_behavior_contract_frozen" and final["authorization"] == "phase1280_qwen3_behavior_only"))
    result = {
        "phase": 1279,
        "audit_type": "independent_zero_model_contract_audit",
        "checks": checks,
        "passed_count": sum(item["passed"] for item in checks),
        "check_count": len(checks),
        "all_checks_passed": all(item["passed"] for item in checks),
        "authorization": "phase1280_qwen3_behavior_only" if all(item["passed"] for item in checks) else "deny_model_run",
    }
    atomic_json(AUDIT, result)
    print(canonical_json(result))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
