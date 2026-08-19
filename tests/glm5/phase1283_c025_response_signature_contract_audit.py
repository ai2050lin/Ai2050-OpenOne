#!/usr/bin/env python3
"""Independent mechanical and semantic-boundary audit for Phase1283."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/glm5/result/phase1283_c025_response_signature_contract"
PROTOCOL = OUT / "protocol/preregistration.json"
MATERIAL = OUT / "material/frozen_response_worlds.jsonl"
REVIEW = OUT / "material/pre_model_semantic_review.json"
FINAL = OUT / "analysis/final.json"
AUDIT = OUT / "audit/independent_final_audit.json"
COUNTS = {"discovery": 64, "selection": 64, "confirmation": 64}
PARTITION_SURFACES = {
    "discovery": {"test_confirmation", "forecast_agreement"},
    "selection": {"evidence_support", "outcome_match"},
    "confirmation": {"measurement_validation", "finding_consistency"},
}
PANELS = {
    "consistency", "reversal", "carrier_consistency", "lexical_consistency",
    "role_consistency", "role_reversal",
}
ROLES = {"expected_0", "expected_1", "opposite_0", "opposite_1", "control_0", "control_1"}
C024_WORDS = {
    "light", "heavy", "small", "large", "cold", "hot", "smooth", "rough", "quiet", "loud",
    "bright", "dim", "weak", "strong", "rigid", "flexible", "slow", "fast", "clean", "dirty",
    "dry", "wet", "soft", "hard", "short", "long", "open", "closed", "empty", "full", "narrow", "wide",
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


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def check(name: str, passed: bool, detail: Any = None) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def run() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in MATERIAL.read_text(encoding="utf-8").splitlines() if line.strip()]
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    final = json.loads(FINAL.read_text(encoding="utf-8"))
    counts = Counter(row["partition"] for row in rows)
    axes = {partition: {row["axis"] for row in rows if row["partition"] == partition} for partition in COUNTS}
    all_words = {
        word for row in rows for word in row["expected_terms"] + row["opposite_terms"]
    }
    surface_ok = all(
        set(row["contexts"]) == PARTITION_SURFACES[row["partition"]]
        and all(set(panels) == PANELS for panels in row["contexts"].values())
        for row in rows
    )
    lexical_roles_ok = all(
        set(row["candidate_continuations"]) == ROLES
        and set(row["expected_terms"]).isdisjoint(row["opposite_terms"])
        and all(row["candidate_continuations"][role].startswith(" ") and row["candidate_continuations"][role].endswith(".") for role in ROLES)
        for row in rows
    )
    semantic_context_ok = True
    for row in rows:
        expected = row["expected_terms"][0].lower()
        forbidden = {value.lower() for value in row["opposite_terms"]}
        for panels in row["contexts"].values():
            for context in panels.values():
                words = set(context.lower().replace(".", " ").replace(",", " ").split())
                semantic_context_ok &= expected in words and forbidden.isdisjoint(words)
    token = protocol["token_audit"]
    axis_review_ok = len(review["axis_reviews"]) == 24 and all(
        value["opposition_unambiguous"]
        and value["all_candidate_item_combinations_natural"]
        and value["naturalness_score_1_to_5"] >= 4
        for value in review["axis_reviews"]
    )
    surface_review_ok = len(review["surface_reviews"]) == 6 and all(
        value["consistency_entails_expected_side"]
        and value["reversal_entails_opposite_side"]
        and value["slot_requires_predicative_description"]
        and value["naturalness_score_1_to_5"] >= 4
        for value in review["surface_reviews"]
    )
    event_ok = all(
        set(protocol["token_audit"]["qwen_event_token_ends"][row["row_id"]]) == PARTITION_SURFACES[row["partition"]]
        and all(set(protocol["token_audit"]["qwen_event_token_ends"][row["row_id"]][surface]) == PANELS for surface in PARTITION_SURFACES[row["partition"]])
        for row in rows
    )
    checks = [
        check("row_and_partition_counts", len(rows) == 192 and dict(counts) == COUNTS, {"rows": len(rows), "partitions": dict(counts)}),
        check("row_ids_and_digests", len({row["row_id"] for row in rows}) == len(rows) and all(digest({key: value for key, value in row.items() if key != "row_digest"}) == row["row_digest"] for row in rows)),
        check("axis_partition_disjointness", all(len(value) == 8 for value in axes.values()) and not (axes["discovery"] & axes["selection"] or axes["discovery"] & axes["confirmation"] or axes["selection"] & axes["confirmation"]), {key: sorted(value) for key, value in axes.items()}),
        check("c024_lexicon_excluded", C024_WORDS.isdisjoint(all_words), sorted(C024_WORDS & all_words)),
        check("surface_and_panel_registry", surface_ok),
        check("candidate_role_and_opposition_registry", lexical_roles_ok),
        check("expected_only_in_context", semantic_context_ok),
        check("manual_axis_semantic_review", axis_review_ok),
        check("manual_surface_naturalness_review", surface_review_ok),
        check("review_scope_honest", review["reviewed_before_any_weight_run"] and review["independent_human_labels"] is False and review["ambiguity_flags"] == [] and "not an independent human" in review["scope_limit"]),
        check("lexical_carrier_token_match", token["lexical_carrier_context_lengths_equal_all_models"]),
        check("role_null_token_match", token["role_context_lengths_equal_all_models"]),
        check("candidate_prefix_and_suffix", token["context_prefix_stable_under_all_candidates_all_models"] and token["candidate_suffix_nonempty_all_models"]),
        check("typed_event_registry", event_ok),
        check("material_and_review_hashes", protocol["material_sha256"] == file_sha256(MATERIAL) and protocol["semantic_review_sha256"] == file_sha256(REVIEW)),
        check("response_object_and_stops", protocol["response_definition"]["template"] == [-1, -1, 1, 1, 0, 0] and protocol["formal_qwen_run_budget"] == 1 and len(protocol["hard_stops"]) >= 7),
        check("forbidden_claims", len(protocol["claims_forbidden"]) >= 4),
        check("pending_authorization", final["authorization"] == "pending_independent_phase1283_audit"),
    ]
    passed = all(item["passed"] for item in checks)
    result = {
        "phase": 1283,
        "campaign": "C025",
        "audit_type": "independent_contract_semantic_boundary_and_mechanical_audit",
        "checks": checks,
        "passed_count": sum(item["passed"] for item in checks),
        "check_count": len(checks),
        "all_checks_passed": passed,
        "authorization": "phase1284_qwen3_response_signature_behavior" if passed else "deny_model_run",
    }
    atomic_json(AUDIT, result)
    if passed:
        final["verdict"] = "c025_response_signature_contract_frozen"
        final["authorization"] = "phase1284_qwen3_response_signature_behavior"
        atomic_json(FINAL, final)
    print(canonical_json(result))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
