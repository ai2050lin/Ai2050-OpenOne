#!/usr/bin/env python3
"""Audit Phase564-566 claims against the frozen result rows."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EDGE_DIR = ROOT / "tests/gpt5/result/phase564_source_conditioned_edge"
RESIDUAL_DIR = ROOT / "tests/gpt5/result/phase565_residual_multiposition_operator"
OUT_PATH = ROOT / "tests/gpt5/result/phase567_multi_relation_binding/phase567_prior_evidence_audit.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def audit() -> dict[str, Any]:
    final = read_json(RESIDUAL_DIR / "phase566_final_audit.json")
    edge = read_json(EDGE_DIR / "phase564_source_edge_confirmation_analysis.json")
    residual = read_json(RESIDUAL_DIR / "phase565_residual_operator_analysis.json")
    rows = read_jsonl(RESIDUAL_DIR / "phase565_residual_operator_rows.jsonl")
    behavior = {
        row["case_id"]: row
        for row in read_jsonl(EDGE_DIR / "phase564_qwen3_edge_behavior_rows.jsonl")
    }
    semantic_rows = [row for row in rows if row["position_block"] == "semantic7"]
    position_counts = Counter(int(row["position_count"]) for row in semantic_rows)
    identity_rows = [row for row in rows if row["condition"] == "same_case_restore"]
    paired_rows = [row for row in rows if row["condition"] == "paired_donor_residual_replace"]
    invariants = Counter()
    for row in paired_rows:
        recipient = behavior[row["recipient_case_id"]]
        donor = behavior[row["donor_case_id"]]
        invariants["same_query_object"] += recipient["query_object"] == donor["query_object"]
        invariants["same_query_object_index"] += (
            recipient["query_object_index"] == donor["query_object_index"]
        )
        invariants["same_surface"] += recipient["surface_id"] == donor["surface_id"]
        invariants["same_fact_order"] += recipient["fact_order"] == donor["fact_order"]
        invariants["same_object_lexicon"] += (
            {recipient["object_a"], recipient["object_b"]}
            == {donor["object_a"], donor["object_b"]}
        )
        invariants["same_color_lexicon"] += (
            {recipient["color_a"], recipient["color_b"]}
            == {donor["color_a"], donor["color_b"]}
        )
        invariants["different_target"] += recipient["target"] != donor["target"]
    checks = {
        "phase566_final_audit_valid": bool(final["valid"]),
        "phase564_confirmation_edges_zero": (
            edge["confirmation_passing_candidate_count"] == 0
        ),
        "phase565_rows_complete": len(rows) == 71424,
        "semantic7_labels_are_six_unique_coordinates": (
            set(position_counts) == {6}
        ),
        "same_case_restore_is_exact_identity_write": (
            identity_rows
            and max(abs(float(row["donor_switch_effect"])) for row in identity_rows) == 0.0
            and all(row["intervention_scores"] == row["baseline_scores"] for row in identity_rows)
        ),
        "paired_donor_keeps_query_lexicon_fixed": all(
            invariants[key] == len(paired_rows)
            for key in (
                "same_query_object", "same_query_object_index", "same_surface",
                "same_fact_order", "same_object_lexicon", "same_color_lexicon",
            )
        ),
        "natural_necessity_not_tested": not residual["natural_necessity_tested"],
        "compute_edges_zero": residual["compute_edge_count"] == 0,
        "sealed_unread": bool(final["checks"]["sealed_unread"]),
    }
    payload = {
        "schema_version": "phase567_prior_evidence_audit.v1",
        "phase_id": "Phase567",
        "created_at": now(),
        "valid": all(checks.values()),
        "checks": checks,
        "objective_counts": {
            "phase565_rows": len(rows),
            "semantic7_label_rows": len(semantic_rows),
            "semantic7_unique_coordinate_count_distribution": {
                str(key): value for key, value in sorted(position_counts.items())
            },
            "same_case_identity_rows": len(identity_rows),
            "paired_donor_rows": len(paired_rows),
            "paired_lexical_invariant_counts": dict(sorted(invariants.items())),
            "qualified_state_sufficiency_operators": residual["qualified_operator_count"],
            "qualified_compute_edges": residual["compute_edge_count"],
        },
        "evidence_corrections": {
            "semantic7_name": "seven semantic labels but six unique physical coordinates",
            "same_case_restore": "same-layer identity write; numeric control, not damage-restore",
            "paired_donor_query_state": (
                "hidden states at fixed query relation/object tokens are replaced; "
                "query lexical identity is not changed"
            ),
            "wrong_state_replacement": "counterfactual sensitivity, not natural necessity",
            "phase564_numeric_error": (
                "analytic reconstruction mismatch under BF16 execution; not an independently "
                "measured no-op repeat-forward distribution"
            ),
            "phase565_positive_scope": "distributed task-state sufficiency only",
        },
    }
    write_json(OUT_PATH, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


if __name__ == "__main__":
    audit()
