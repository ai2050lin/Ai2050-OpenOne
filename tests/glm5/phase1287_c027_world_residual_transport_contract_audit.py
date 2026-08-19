#!/usr/bin/env python3
"""Pure replay audit for the frozen Phase1287 C027 contract."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))
from phase1287_c027_world_residual_transport_contract import (  # noqa: E402
    CAMPAIGN, FAMILIES, PANELS, PARTITIONS, ROLE_ORDER, SCRIPT, SEMANTIC_REVIEW,
    SURFACES, SURFACE_ORDER, THRESHOLDS, VARIANTS, canonical_json, digest,
)


OUT = ROOT / "tests/glm5/result/phase1287_c027_world_residual_transport_contract"
PROTOCOL = OUT / "protocol/preregistration.json"
MATERIAL = OUT / "material/frozen_world_residual_material.jsonl"
FINAL = OUT / "analysis/final.json"
AUDIT = OUT / "audit/independent_final_audit.json"
C026_MATERIAL = ROOT / "tests/glm5/result/phase1285_c026_conditional_response_mapping_contract/material/frozen_binary_status_worlds.jsonl"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    protocol = read_json(PROTOCOL)
    rows = read_jsonl(MATERIAL)
    review = read_json(SEMANTIC_REVIEW)
    final = read_json(FINAL)
    old_rows = read_jsonl(C026_MATERIAL)

    old_axes = {row["axis"] for row in old_rows}
    old_labels = {value for row in old_rows for value in (row["left_label"], row["right_label"])}
    old_items = {row["item"] for row in old_rows}
    new_axes = {row["axis"] for row in rows}
    new_labels = {value for row in rows for value in (row["left_label"], row["right_label"])}
    new_items = {row["item"] for row in rows}

    row_ids = [row["row_id"] for row in rows]
    row_digests_valid = all(
        row["row_digest"] == digest({key: value for key, value in row.items() if key != "row_digest"})
        for row in rows
    )
    partition_axes = {
        partition: {row["axis"] for row in rows if row["partition"] == partition}
        for partition in PARTITIONS
    }
    partition_items = {
        partition: {row["item"] for row in rows if row["partition"] == partition}
        for partition in PARTITIONS
    }
    partition_labels = {
        partition: {value for row in rows if row["partition"] == partition for value in (row["left_label"], row["right_label"])}
        for partition in PARTITIONS
    }

    contexts_semantically_typed = True
    variants_distinct = True
    candidate_schema_valid = True
    feature_schema_valid = True
    for row in rows:
        candidate_schema_valid &= set(row["candidate_continuations"]) == set(ROLE_ORDER)
        feature_schema_valid &= set(row["content_features"]) == set(protocol["content_feature_order"])
        for surface in SURFACE_ORDER:
            panels = row["contexts"][surface]
            contexts_semantically_typed &= set(panels) == set(PANELS)
            contexts_semantically_typed &= SURFACES[surface]["consistency"] in panels["consistency"]
            contexts_semantically_typed &= SURFACES[surface]["reversal"] in panels["reversal"]
            contexts_semantically_typed &= panels["lexical_consistency"].endswith(panels["consistency"])
            contexts_semantically_typed &= panels["lexical_reversal"].endswith(panels["consistency"])
            contexts_semantically_typed &= panels["role_consistency"].endswith(panels["consistency"])
            contexts_semantically_typed &= panels["role_reversal"].endswith(panels["consistency"])
        for family in FAMILIES:
            variants_distinct &= row["contexts"][f"{family}_a"]["consistency"] != row["contexts"][f"{family}_b"]["consistency"]

    timeless = {key: value for key, value in protocol.items() if key not in ("created_at_utc", "protocol_digest")}
    checks = {
        "phase_campaign": protocol["phase"] == 1287 and protocol["campaign"] == CAMPAIGN,
        "protocol_digest": protocol["protocol_digest"] == digest(timeless),
        "main_source_hash": protocol["source_hashes"]["main"] == file_sha256(SCRIPT),
        "auditor_source_hash": protocol["source_hashes"]["auditor"] == file_sha256(Path(__file__)),
        "material_hash": protocol["material_sha256"] == file_sha256(MATERIAL),
        "semantic_review_hash": protocol["semantic_review_sha256"] == file_sha256(SEMANTIC_REVIEW),
        "row_count": len(rows) == 162,
        "row_ids_unique": len(set(row_ids)) == len(row_ids),
        "row_digests": row_digests_valid,
        "partition_counts": all(sum(row["partition"] == partition for row in rows) == 54 for partition in PARTITIONS),
        "axis_counts": all(len(partition_axes[partition]) == 9 for partition in PARTITIONS),
        "partition_axes_disjoint": not any(partition_axes[a] & partition_axes[b] for i, a in enumerate(PARTITIONS) for b in PARTITIONS[i + 1:]),
        "partition_items_disjoint": not any(partition_items[a] & partition_items[b] for i, a in enumerate(PARTITIONS) for b in PARTITIONS[i + 1:]),
        "partition_labels_disjoint": not any(partition_labels[a] & partition_labels[b] for i, a in enumerate(PARTITIONS) for b in PARTITIONS[i + 1:]),
        "c026_axes_disjoint": not (new_axes & old_axes),
        "c026_labels_disjoint": not (new_labels & old_labels),
        "c026_items_disjoint": not (new_items & old_items),
        "closed_binary_unique_gold": all(row["expected_label"] != row["opposite_label"] for row in rows),
        "listed_order_exact": all(set(row["listed_order"]) == {row["left_label"], row["right_label"]} for row in rows),
        "surface_count": len(SURFACE_ORDER) == 8 and len(SURFACES) == 8,
        "family_variant_factorial": all(all(f"{family}_{variant}" in SURFACE_ORDER for variant in VARIANTS) for family in FAMILIES),
        "variant_wordings_distinct": variants_distinct,
        "panel_semantics": contexts_semantically_typed,
        "candidate_schema": candidate_schema_valid,
        "content_feature_schema": feature_schema_valid,
        "token_prefix_stability": protocol["token_audit"]["context_prefix_stable_under_all_candidates"],
        "token_suffix_nonempty": protocol["token_audit"]["candidate_suffix_nonempty"],
        "typed_events_valid": protocol["token_audit"]["typed_character_events_valid"],
        "semantic_review_unique": review["all_rows_have_unique_gold"] and not review["ambiguity_flags"],
        "semantic_review_scope_honest": review["independent_human_blind_labels"] is False,
        "thresholds_exact": protocol["thresholds"] == THRESHOLDS,
        "model_frozen": protocol["model"] == {
            "name": "qwen3", "precision": "FP16 CUDA, no quantization", "formal_runs": 1, "other_models_authorized": False,
        },
        "hypotheses_frozen": set(protocol["hypotheses"]) == {"H0_zero", "HC_content", "H1_identity", "H2_diagonal", "H3_full"},
        "zero_models_frozen": set(protocol["zero_models"]) == {
            "zero_center", "content_features", "wrong_world_offsets", "role_permutations", "lexical_null", "role_null",
        },
        "authorization_frozen": final["authorization"] == "phase1288_qwen3_world_residual_behavior_after_audit",
        "hard_stop_present": any("failed C027 ledger closes" in value for value in protocol["hard_stops"]),
        "claims_forbidden": len(protocol["claims_forbidden"]) >= 5,
    }
    result = {
        "phase": 1287,
        "campaign": CAMPAIGN,
        "audit_kind": "pure_replay_no_mutable_authorization_precondition",
        "checks": checks,
        "passed": sum(bool(value) for value in checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "material_summary": {
            "rows": len(rows),
            "axes": len(new_axes),
            "surface_variants": len(SURFACE_ORDER),
            "c026_overlap": {
                "axes": sorted(new_axes & old_axes),
                "labels": sorted(new_labels & old_labels),
                "items": sorted(new_items & old_items),
            },
        },
        "authorization": "phase1288_qwen3_world_residual_behavior" if all(checks.values()) else "stop_before_weights",
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(canonical_json({"passed": result["passed"], "total": result["total"], "authorization": result["authorization"]}))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
