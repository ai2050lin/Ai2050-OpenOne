#!/usr/bin/env python3
"""Independent replay audit for the frozen Phase1289 C028 contract."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/glm5/result/phase1289_c028_typed_complement_contract"
PROTOCOL = OUT / "protocol/preregistration.json"
ENVIRONMENT = OUT / "protocol/environment_snapshot.json"
MATERIAL = OUT / "material/frozen_typed_complement_material.jsonl"
SEMANTIC_REVIEW = OUT / "material/pre_model_semantic_naturalness_review.json"
FINAL = OUT / "analysis/final.json"
AUDIT = OUT / "audit/independent_final_audit.json"

EXPECTED_PARTITIONS = ("discovery", "selection", "confirmation")
EXPECTED_FAMILIES = ("case_record", "lab_log", "field_report")
EXPECTED_SURFACES = tuple(f"{family}_{variant}" for family in EXPECTED_FAMILIES for variant in ("a", "b"))
EXPECTED_PANELS = ("identity", "single_complement", "double_complement", "lexical_null", "scope_null")
EXPECTED_THRESHOLDS = {
    "finite_fraction_min": 1.0,
    "overall_candidate_accuracy_min": 0.95,
    "partition_candidate_accuracy_min": 0.93,
    "surface_candidate_accuracy_min": 0.90,
    "active_panel_accuracy_min": 0.90,
    "median_gold_margin_per_active_panel_min": 0.25,
    "active_triple_all_correct_rate_min": 0.85,
    "identity_double_both_correct_rate_min": 0.90,
    "identity_single_opposition_both_correct_rate_min": 0.90,
    "lexical_null_preservation_rate_min": 0.90,
    "scope_null_preservation_rate_min": 0.90,
    "surface_variant_both_correct_rate_min": 0.88,
    "base_side_accuracy_min": 0.90,
    "generation_coverage_min": 0.85,
    "generation_exact_accuracy_min": 0.85,
    "generation_active_triple_rate_min": 0.80,
    "shortcut_program_accuracy_max": 0.70,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def check(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def main() -> None:
    protocol = read_json(PROTOCOL)
    rows = read_jsonl(MATERIAL)
    review = read_json(SEMANTIC_REVIEW)
    environment = read_json(ENVIRONMENT)
    final = read_json(FINAL)
    checks: list[dict[str, Any]] = []

    timeless = {key: value for key, value in protocol.items() if key not in {"created_at_utc", "protocol_digest"}}
    checks.append(check("protocol_digest_recomputes", digest(timeless) == protocol["protocol_digest"], protocol["protocol_digest"]))
    checks.append(check("material_hash_matches", file_sha256(MATERIAL) == protocol["material_sha256"], protocol["material_sha256"]))
    checks.append(check("semantic_review_hash_matches", file_sha256(SEMANTIC_REVIEW) == protocol["semantic_review_sha256"], protocol["semantic_review_sha256"]))
    checks.append(check("phase_campaign_contract", (protocol["phase"], protocol["campaign"], protocol["contract_id"]) == (1289, "C028", "EXP-C028-WP00-001"), [protocol["phase"], protocol["campaign"], protocol["contract_id"]]))
    checks.append(check("partitions_exact", tuple(protocol["partitions_order"]) == EXPECTED_PARTITIONS, protocol["partitions_order"]))
    checks.append(check("families_exact", tuple(protocol["families"]) == EXPECTED_FAMILIES, protocol["families"]))
    checks.append(check("surfaces_exact", tuple(protocol["surfaces"]) == EXPECTED_SURFACES, protocol["surfaces"]))
    checks.append(check("panels_exact", tuple(protocol["panels"]) == EXPECTED_PANELS, protocol["panels"]))
    checks.append(check("thresholds_exact", protocol["thresholds"] == EXPECTED_THRESHOLDS, protocol["thresholds"]))
    checks.append(check("single_model_fp16_only", protocol["models"] == {"behavior": ["qwen3-4b-fp16-cuda-no-quantization"], "other_models_authorized": False, "formal_behavior_runs": 1}, protocol["models"]))

    counts = Counter(row["partition"] for row in rows)
    checks.append(check("row_count_144", len(rows) == 144, len(rows)))
    checks.append(check("partition_counts_48", counts == Counter({value: 48 for value in EXPECTED_PARTITIONS}), dict(counts)))
    checks.append(check("unique_row_ids", len({row["row_id"] for row in rows}) == len(rows), len({row["row_id"] for row in rows})))
    checks.append(check("unique_row_digests", len({row["row_digest"] for row in rows}) == len(rows), len({row["row_digest"] for row in rows})))
    checks.append(check("row_digests_recompute", all(digest({k: v for k, v in row.items() if k != "row_digest"}) == row["row_digest"] for row in rows), "all rows"))
    checks.append(check("axes_8_per_partition", all(len({row["axis"] for row in rows if row["partition"] == part}) == 8 for part in EXPECTED_PARTITIONS), {part: len({row["axis"] for row in rows if row["partition"] == part}) for part in EXPECTED_PARTITIONS}))
    checks.append(check("world_factorization", all(len([row for row in rows if row["partition"] == part and row["axis"] == axis]) == 6 for part in EXPECTED_PARTITIONS for axis in {row["axis"] for row in rows if row["partition"] == part}), "3 items x 2 base sides"))
    checks.append(check("base_sides_balanced", Counter(row["base_side"] for row in rows) == Counter({0: 72, 1: 72}), dict(Counter(row["base_side"] for row in rows))))
    checks.append(check("listed_orders_balanced", abs(sum(row["listed_order"][0] == row["left_label"] for row in rows) - 72) == 0, sum(row["listed_order"][0] == row["left_label"] for row in rows)))

    schema_ok = True
    gold_ok = True
    events_ok = True
    lexical_null_ok = True
    scope_null_ok = True
    candidate_ok = True
    context_unique = True
    all_contexts: list[str] = []
    for row in rows:
        schema_ok &= set(row["contexts"]) == set(EXPECTED_SURFACES)
        schema_ok &= all(set(row["contexts"][surface]) == set(EXPECTED_PANELS) for surface in EXPECTED_SURFACES)
        gold = row["gold_by_panel"]
        gold_ok &= gold["identity"] == row["base_label"] == gold["double_complement"]
        gold_ok &= gold["single_complement"] == row["opposite_label"] != row["base_label"]
        gold_ok &= gold["lexical_null"] == row["base_label"] == gold["scope_null"]
        candidate_ok &= set(row["candidate_continuations"]) == {"left", "right"}
        candidate_ok &= row["left_label"] in row["candidate_continuations"]["left"]
        candidate_ok &= row["right_label"] in row["candidate_continuations"]["right"]
        for surface in EXPECTED_SURFACES:
            for panel in EXPECTED_PANELS:
                text = row["contexts"][surface][panel]
                all_contexts.append(text)
                typed = row["typed_events"][surface][panel]
                spans = [typed["source_value"], typed["query_object"], typed["answer_boundary"]]
                spans.extend(typed["operator_events"])
                spans.extend(typed["null_operator_events"])
                events_ok &= all(0 <= a < b <= len(text) for a, b in spans)
                events_ok &= len(typed["operator_events"]) == (2 if panel == "double_complement" else 1)
            lexical_null_ok &= len(row["typed_events"][surface]["lexical_null"]["null_operator_events"]) >= 1
            scope_null_ok &= row["distractor"] in row["contexts"][surface]["scope_null"]
            scope_null_ok &= row["item"] != row["distractor"]
    context_unique &= len(set(all_contexts)) == len(all_contexts)
    checks.append(check("material_schema_exact", schema_ok, "surfaces and panels"))
    checks.append(check("gold_algebra_exact", gold_ok, "identity=double=base; single=opposite; null=base"))
    checks.append(check("candidate_schema_exact", candidate_ok, "left/right sentence continuations"))
    checks.append(check("typed_event_spans_valid", events_ok, "one event except two ordered events for double complement"))
    checks.append(check("lexical_null_contains_nonoperative_cue", lexical_null_ok, "all surface/world cells"))
    checks.append(check("scope_null_targets_distinct_distractor", scope_null_ok, "all surface/world cells"))
    checks.append(check("all_contexts_unique", context_unique, len(set(all_contexts))))

    token = protocol["token_audit"]
    checks.append(check("token_prefix_and_suffix", token["context_prefix_stable_under_candidates"] and token["candidate_suffix_nonempty"], token))
    checks.append(check("equal_length_single_token_labels", token["candidate_lengths_equal_within_context"] and token["all_state_labels_single_token_with_leading_space"], token))
    checks.append(check("typed_token_audit", token["typed_character_events_valid"], token["typed_character_events_valid"]))
    checks.append(check("semantic_unique", review["all_rows_have_unique_gold"] and not review["ambiguity_flags"], {"unique": review["all_rows_have_unique_gold"], "flags": review["ambiguity_flags"]}))
    checks.append(check("naturalness_limitation_explicit", not review["independent_human_blind_labels"] and "researcher-constructed" in review["naturalness_scope_limit"], review["naturalness_scope_limit"]))
    checks.append(check("prior_c027_overlap_empty", protocol["prior_material_overlap"].get("label_overlap") == [] and protocol["prior_material_overlap"].get("item_overlap") == [] and protocol["prior_material_overlap"].get("row_digest_overlap") == [], protocol["prior_material_overlap"]))
    checks.append(check("behavior_before_hidden_stop", any("Behavior and exact free generation" in value for value in protocol["hard_stops"]), protocol["hard_stops"]))
    checks.append(check("all_failure_branches_close", protocol["branching"]["any_phase1290_behavior_or_generation_ledger_fails"] == "close_c028_without_hidden" and protocol["branching"]["future_response_prediction_fails"] == "close_c028" and protocol["branching"]["path_or_independent_rescue_fails"] == "close_c028", protocol["branching"]))
    checks.append(check("no_weights_loaded", environment["model_weights_loaded"] is False and final["model_weights_loaded"] is False, [environment["model_weights_loaded"], final["model_weights_loaded"]]))
    checks.append(check("source_hashes_match", all(file_sha256(ROOT / path) == expected for path, expected in {
        "tests/glm5/phase1289_c028_typed_complement_contract.py": protocol["source_hashes"]["main"],
        "tests/glm5/phase1289_c028_typed_complement_contract_audit.py": protocol["source_hashes"]["auditor"],
    }.items()), protocol["source_hashes"]))

    duplicate_pair_use: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        duplicate_pair_use[(row["left_label"], row["right_label"])].add(row["partition"])
    checks.append(check("state_pairs_partition_disjoint", all(len(parts) == 1 for parts in duplicate_pair_use.values()) and len(duplicate_pair_use) == 24, {"pairs": len(duplicate_pair_use)}))

    all_passed = all(value["passed"] for value in checks)
    audit = {
        "phase": 1289,
        "campaign": "C028",
        "schema_version": "phase1289.c028.independent_audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "auditor_imports_main": False,
        "checks": checks,
        "passed_count": sum(value["passed"] for value in checks),
        "total_count": len(checks),
        "all_checks_passed": all_passed,
        "authorization": "phase1290_qwen3_typed_complement_behavior" if all_passed else "none",
        "protocol_digest": protocol["protocol_digest"],
        "material_sha256": file_sha256(MATERIAL),
    }
    atomic_json(AUDIT, audit)
    if all_passed:
        final.update({
            "verdict": "contract_frozen_and_independently_audited",
            "authorization": "phase1290_qwen3_typed_complement_behavior",
            "audit_passed": True,
            "audit_path": str(AUDIT.relative_to(ROOT)).replace("\\", "/"),
        })
        atomic_json(FINAL, final)
    print(canonical_json({
        "phase": 1289,
        "passed": audit["passed_count"],
        "total": audit["total_count"],
        "all_checks_passed": all_passed,
        "authorization": audit["authorization"],
    }))
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
