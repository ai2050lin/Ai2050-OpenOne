#!/usr/bin/env python3
"""Independent semantic and mechanical audit for Phase1281."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/glm5/result/phase1281_c024_expectation_reversal_contract"
PROTOCOL = OUT / "protocol/preregistration.json"
MATERIAL = OUT / "material/frozen_expectation_worlds.jsonl"
FINAL = OUT / "analysis/final.json"
AUDIT = OUT / "audit/independent_final_audit.json"
COUNTS = {"discovery": 64, "selection": 64, "confirmation": 128}
SURFACES = ("coordination", "adverbial", "expectation", "evaluation", "report")
PANELS = ("consistency", "contrast", "carrier_consistency", "lexical_consistency", "carrier_contrast", "lexical_contrast")


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
    final = json.loads(FINAL.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in MATERIAL.read_text(encoding="utf-8").splitlines() if line.strip()]
    counts = Counter(row["partition"] for row in rows)
    token = protocol["token_audit"]
    semantic = protocol["semantic_audit"]
    checks = [
        check("row_count", len(rows) == 256, len(rows)),
        check("partition_counts", dict(counts) == COUNTS, dict(counts)),
        check("row_digest", all(digest({key: value for key, value in row.items() if key != "row_digest"}) == row["row_digest"] for row in rows)),
        check("row_ids_unique", len({row["row_id"] for row in rows}) == len(rows)),
        check("worlds_unique", semantic["world_descriptions_unique"]),
        check("surface_panel_registry", all(set(row["contexts"]) == set(SURFACES) and all(set(row["contexts"][surface]) == set(PANELS) for surface in SURFACES) for row in rows)),
        check("axis_orientation_balance", semantic["orientation_balanced"]),
        check("antonym_roles_distinct", semantic["expected_opposite_distinct"]),
        check("context_contains_only_expected", semantic["expected_present_opposite_absent"]),
        check("candidate_multitoken_equal", token["all_candidates_multitoken"] and token["candidate_lengths_equal_within_world"]),
        check("prefix_and_tokenizer", token["context_prefix_stable_under_continuation"] and token["fast_slow_tokenization_equal"]),
        check("matched_note_lengths", token["carrier_lexical_context_lengths_equal"] and token["contrast_carrier_cue_token_lengths_matched"]),
        check("event_registry", all(
            set(protocol["token_audit"]["event_token_indices"][row["row_id"]]) == set(SURFACES)
            and all(set(protocol["token_audit"]["event_token_indices"][row["row_id"]][surface]) == set(PANELS) for surface in SURFACES)
            for row in rows
        )),
        check("explicit_scope_limit", semantic["independent_human_labels"] is False and "no claim" in semantic["semantic_scope"].lower()),
        check("authorization", final["authorization"] == "phase1282_qwen3_multitoken_behavior_and_generation"),
    ]
    result = {
        "phase": 1281,
        "audit_type": "independent_semantic_and_mechanical_contract_audit",
        "checks": checks,
        "passed_count": sum(item["passed"] for item in checks),
        "check_count": len(checks),
        "all_checks_passed": all(item["passed"] for item in checks),
        "authorization": "phase1282_qwen3_multitoken_behavior_and_generation" if all(item["passed"] for item in checks) else "deny_model_run",
    }
    atomic_json(AUDIT, result)
    print(canonical_json(result))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
