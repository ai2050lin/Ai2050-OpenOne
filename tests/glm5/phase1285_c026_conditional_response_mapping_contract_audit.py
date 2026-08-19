#!/usr/bin/env python3
"""Independent mechanical and construct-boundary audit for Phase1285."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/glm5/result/phase1285_c026_conditional_response_mapping_contract"
PROTOCOL = OUT / "protocol/preregistration.json"
MATERIAL = OUT / "material/frozen_binary_status_worlds.jsonl"
REVIEW = OUT / "material/pre_model_semantic_naturalness_review.json"
FINAL = OUT / "analysis/final.json"
AUDIT = OUT / "audit/independent_final_audit.json"

PARTITIONS = {"discovery", "selection", "confirmation"}
SURFACES = {"official_decision", "binary_audit", "closed_review", "signed_assessment"}
PANELS = {
    "consistency", "reversal", "lexical_consistency", "lexical_reversal",
    "role_consistency", "role_reversal",
}
ROLES = {"expected_0", "expected_1", "opposite_0", "opposite_1", "control_0", "control_1"}


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
    axes = {partition: {row["axis"] for row in rows if row["partition"] == partition} for partition in PARTITIONS}
    items = {partition: {row["item"] for row in rows if row["partition"] == partition} for partition in PARTITIONS}
    names = {partition: {row["name"] for row in rows if row["partition"] == partition} for partition in PARTITIONS}
    expected_sides = Counter((row["partition"], row["axis"], row["expected_side"]) for row in rows)
    order_counts = Counter(tuple(row["listed_order"]) == (row["left_label"], row["right_label"]) for row in rows)

    material_structure = all(
        set(row["contexts"]) == SURFACES
        and all(set(panels) == PANELS for panels in row["contexts"].values())
        and set(row["candidate_continuations"]) == ROLES
        and row["expected_label"] != row["opposite_label"]
        and set(row["listed_order"]) == {row["left_label"], row["right_label"]}
        for row in rows
    )
    logical_contract = all(
        ("exactly one of two" in row["contexts"]["official_decision"]["consistency"])
        and ("mutually exclusive" in row["contexts"]["binary_audit"]["consistency"])
        and ("could return only" in row["contexts"]["closed_review"]["consistency"])
        and ("two exclusive outcomes" in row["contexts"]["signed_assessment"]["consistency"])
        and all(row["expected_label"] in panels["consistency"] for panels in row["contexts"].values())
        and all(row["opposite_label"] in panels["consistency"] for panels in row["contexts"].values())
        for row in rows
    )
    lexical_null_semantics = all(
        "without reporting this case" in panels["lexical_consistency"]
        and "without reporting this case" in panels["lexical_reversal"]
        and panels["lexical_consistency"].endswith(panels["consistency"])
        and panels["lexical_reversal"].endswith(panels["consistency"])
        for row in rows for panels in row["contexts"].values()
    )
    role_null_semantics = all(
        "did not concern the target below" in panels["role_consistency"]
        and "did not concern the target below" in panels["role_reversal"]
        and panels["role_consistency"].endswith(panels["consistency"])
        and panels["role_reversal"].endswith(panels["consistency"])
        for row in rows for panels in row["contexts"].values()
    )
    event_registry = all(
        set(row["typed_events"]) == SURFACES
        and all(set(events) == {"consistency", "reversal"} for events in row["typed_events"].values())
        and all(
            set(event) == {"expected_label", "relation_cue", "context_end"}
            and all(0 <= span[0] < span[1] <= len(row["contexts"][surface][panel]) for span in event.values())
            for surface, events in row["typed_events"].items() for panel, event in events.items()
        )
        for row in rows
    )
    axis_review = review["axis_reviews"]
    surface_review = review["surface_reviews"]
    checks = [
        check("phase_campaign_and_contract", protocol["phase"] == 1285 and protocol["campaign"] == "C026" and protocol["contract_id"] == "EXP-C026-WP00-001"),
        check("row_partition_counts", len(rows) == 192 and counts == Counter({"discovery": 64, "selection": 64, "confirmation": 64}), {"rows": len(rows), "counts": dict(counts)}),
        check("row_ids_and_digests", len({row["row_id"] for row in rows}) == 192 and all(digest({key: value for key, value in row.items() if key != "row_digest"}) == row["row_digest"] for row in rows)),
        check("axis_partition_disjointness", all(len(value) == 8 for value in axes.values()) and not (axes["discovery"] & axes["selection"] or axes["discovery"] & axes["confirmation"] or axes["selection"] & axes["confirmation"]), {key: sorted(value) for key, value in axes.items()}),
        check("item_partition_disjointness", not (items["discovery"] & items["selection"] or items["discovery"] & items["confirmation"] or items["selection"] & items["confirmation"])),
        check("name_partition_disjointness", not (names["discovery"] & names["selection"] or names["discovery"] & names["confirmation"] or names["selection"] & names["confirmation"])),
        check("expected_side_balance", len(expected_sides) == 48 and all(value == 4 for value in expected_sides.values())),
        check("listed_order_balance", order_counts[True] == order_counts[False] == 96, dict(order_counts)),
        check("surface_panel_role_registry", material_structure),
        check("explicit_binary_semantic_uniqueness", logical_contract),
        check("lexical_null_preserves_target_gold", lexical_null_semantics),
        check("role_null_preserves_target_gold", role_null_semantics),
        check("typed_event_registry", event_registry),
        check("semantic_axis_review", len(axis_review) == 24 and all(value["explicit_closed_binary_contract_makes_gold_unique"] and value["labels_are_distinct"] and value["all_item_combinations_grammatical"] and value["naturalness_score_1_to_5"] >= 4 for value in axis_review)),
        check("surface_naturalness_review", len(surface_review) == 4 and all(value["consistency_selects_expected_under_closed_binary_contract"] and value["reversal_selects_only_remaining_label_under_closed_binary_contract"] and value["answer_slot_is_predicative"] and value["naturalness_score_1_to_5"] >= 4 for value in surface_review)),
        check("review_scope_honest", review["reviewed_before_any_c026_weight_run"] and review["independent_human_labels"] is False and "not an independent human" in review["scope_limit"] and review["ambiguity_flags"] == []),
        check("token_prefix_and_suffix", protocol["score_accounts"]["context_prefix_stable_under_all_candidates_all_models"] and protocol["score_accounts"]["candidate_suffix_nonempty_all_models"]),
        check("length_account_frozen", protocol["score_accounts"]["primary_score"] == "continuation_mean_log_probability_per_token" and protocol["score_accounts"]["secondary_score"] == "continuation_total_log_probability"),
        check("material_review_hashes", protocol["material_sha256"] == file_sha256(MATERIAL) and protocol["semantic_review_sha256"] == file_sha256(REVIEW)),
        check("hypothesis_and_zero_model_registry", set(protocol["hypotheses"]) == {"H0_constant", "H1_identity", "H2_diagonal_affine", "H3_full_affine"} and len(protocol["zero_models"]["fixed_role_permutations"]) == 3),
        check("discovery_selection_confirmation_order", protocol["partition_roles"]["discovery"].startswith("fit") and protocol["partition_roles"]["selection"].startswith("choose") and "once" in protocol["partition_roles"]["confirmation"]),
        check("sequential_model_authorization", protocol["models"]["order"] == ["qwen3", "glm4", "deepseek7b"] and "denied unless" in protocol["models"]["authorization"]),
        check("single_run_and_stops", protocol["formal_run_budget"] == {"qwen3": 1, "glm4": 1, "deepseek7b": 1, "qwen3_hidden": 1} and len(protocol["hard_stops"]) >= 9),
        check("forbidden_overclaims", len(protocol["claims_forbidden"]) >= 6),
        check("pending_authorization", final["authorization"] == "pending_independent_phase1285_audit"),
    ]
    passed = all(value["passed"] for value in checks)
    result = {
        "phase": 1285,
        "campaign": "C026",
        "audit_type": "independent_contract_construct_boundary_and_mechanical_audit",
        "checks": checks,
        "passed_count": sum(value["passed"] for value in checks),
        "check_count": len(checks),
        "all_checks_passed": passed,
        "authorization": "phase1286_qwen3_conditional_response_mapping_behavior" if passed else "deny_model_run",
    }
    atomic_json(AUDIT, result)
    if passed:
        final["verdict"] = "c026_conditional_response_mapping_contract_frozen"
        final["authorization"] = "phase1286_qwen3_conditional_response_mapping_behavior"
        atomic_json(FINAL, final)
    print(canonical_json(result))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
