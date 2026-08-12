#!/usr/bin/env python3
"""Prospective target-by-donor causal response matrix.

Phase1141 showed that one matched cross-item donor often moves the answer as
strongly as the same-item donor.  This phase changes the object of study from a
single hot position to a complete, paired target-by-donor response matrix.

Every off-diagonal donor is evaluated beside a fresh copy of the target's
correct donor in the same forward batch.  Alpha zero and alpha one are also in
that batch.  This makes the content-specific contrast a within-batch quantity
and directly addresses the FP16 batch-composition boundary recorded as K80.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests/glm5"
sys.path.insert(0, str(TEST_ROOT))

from model_utils import get_layers  # noqa: E402
from phase1023_fp16_utils import quantization_audit, release_fp16  # noqa: E402
import phase1135_temporal_binding_intervention as source  # noqa: E402
import phase1138_temporal_residual_onset as phase1138  # noqa: E402
import phase1140_temporal_position_scope_sufficiency as phase1140  # noqa: E402
import phase1141_first_lexical_divergence_boundary as phase1141  # noqa: E402


PHASE = 1142
MODELS = ("qwen3_4b", "qwen3_14b")
SPLITS = ("discovery", "confirmation")
PROPERTIES = ("P54", "P286", "P6")
ITEMS_PER_PROPERTY = 4
COHORT_SIZE = len(PROPERTIES) * ITEMS_PER_PROPERTY
RESERVE_PER_PROPERTY = 4
REQUESTED_FRACTION = 0.7
ALPHAS = (0.0, 1.0)
PANELS = {
    "original": ("original_pre", "original_post", "old", "new"),
    "swapped": ("swapped_pre", "swapped_post", "new", "old"),
}
MATRIX_COMPARISONS = COHORT_SIZE * (COHORT_SIZE - 1) * len(PANELS)
CONTROL_COMPARISONS = COHORT_SIZE * 2
EXPECTED_COMPARISONS = MATRIX_COMPARISONS + CONTROL_COMPARISONS
RECORDS_PER_COMPARISON = 4
EXPECTED_RECORDS = EXPECTED_COMPARISONS * RECORDS_PER_COMPARISON
OUT_ROOT = ROOT / "tests/glm5/result/phase1142_causal_donor_response_matrix"
SOURCE_ITEMS = source.SOURCE
SOURCE1141 = phase1141.OUT_ROOT
EPSILON = 1e-8

THRESHOLDS = {
    "finite_fraction": 0.99,
    "paired_alpha0_max_abs_margin_difference": 0.005,
    "diagonal_baseline_valid_fraction": 0.99,
    "diagonal_endpoint_flip_fraction": 0.95,
    "diagonal_positive_change_fraction": 0.99,
    "same_answer_endpoint_flip_fraction": 0.10,
    "self_identity_max_abs_margin_change": 0.005,
    "diagonal_to_same_answer_abs_ratio": 2.0,
    "item_advantage_median": 0.20,
    "item_advantage_positive_fraction": 0.75,
    "item_advantage_same_relation_median": 0.15,
    "item_advantage_cross_relation_median": 0.15,
    "diagonal_minus_offdiagonal_flip_fraction": 0.20,
    "diagonal_top1_fraction": 0.60,
    "per_property_item_advantage_median": 0.10,
    "relation_advantage_median": 0.15,
    "relation_advantage_positive_fraction": 0.75,
    "same_minus_cross_relation_flip_fraction": 0.20,
    "per_property_relation_advantage_median": 0.10,
    "global_cross_relation_flip_fraction": 0.75,
    "global_cross_relation_positive_fraction": 0.90,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def median(values: Iterable[float | None]) -> float | None:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return statistics.median(finite) if finite else None


def fraction(rows: Iterable[dict[str, Any]], key: str) -> float:
    rows = list(rows)
    return sum(bool(row[key]) for row in rows) / max(len(rows), 1)


def stable_order(label: str, values: Iterable[str]) -> list[str]:
    return sorted(
        values,
        key=lambda value: hashlib.sha256(
            f"phase1142|{label}|{value}".encode("utf-8")
        ).hexdigest(),
    )


def prior_used_ids() -> set[str]:
    prereg = read_json(SOURCE1141 / "protocol/preregistration.json")
    used = set(phase1141.excluded_hidden_ids())
    used.update(prereg["material"]["cohorts"]["discovery"])
    used.update(prereg["material"]["cohorts"]["confirmation"])
    used.update(prereg["material"]["reserve_item_ids"])
    return used


def freeze_material() -> tuple[dict[str, list[str]], list[str], dict[str, Any]]:
    items = read_jsonl(SOURCE_ITEMS)
    item_index = {str(row["item_id"]): row for row in items}
    q4 = phase1141.behavior_all_four_by_split(phase1141.Q4_DECISIONS)
    q14 = phase1141.behavior_all_four_by_split(phase1141.Q14_DECISIONS)
    shared = {split: q4[split] & q14[split] for split in SPLITS}
    token_meta = phase1141.token_metadata()
    excluded = prior_used_ids()
    eligible: dict[str, dict[str, list[str]]] = {}
    cohorts: dict[str, list[str]] = {}
    selected: set[str] = set()

    for split in SPLITS:
        eligible[split] = {}
        cohort = []
        for property_id in PROPERTIES:
            rows = {
                item_id
                for item_id in shared[split]
                if item_id not in excluded
                and item_id in token_meta
                and str(item_index[item_id]["split"]) == split
                and str(token_meta[item_id]["property_id"]) == property_id
                and bool(token_meta[item_id]["first_token_informative"])
                and not bool(token_meta[item_id]["prefix_containment"])
            }
            ordered = stable_order(f"{split}|{property_id}", rows)
            if len(ordered) < ITEMS_PER_PROPERTY:
                raise RuntimeError(
                    f"insufficient fresh {property_id} rows for {split}: {len(ordered)}"
                )
            eligible[split][property_id] = ordered
            cohort.extend(ordered[:ITEMS_PER_PROPERTY])
        cohort = stable_order(f"{split}|cohort", cohort)
        if len(cohort) != COHORT_SIZE or len(set(cohort)) != COHORT_SIZE:
            raise RuntimeError(f"cohort drift for {split}")
        cohorts[split] = cohort
        selected.update(cohort)

    reserve = []
    for property_id in PROPERTIES:
        candidates = {
            item_id
            for split in SPLITS
            for item_id in eligible[split][property_id]
            if item_id not in selected
        }
        ordered = stable_order(f"reserve|{property_id}", candidates)
        if len(ordered) < RESERVE_PER_PROPERTY:
            raise RuntimeError(f"insufficient reserve for {property_id}")
        reserve.extend(ordered[:RESERVE_PER_PROPERTY])
    reserve = stable_order("reserve|all", reserve)

    audit = {
        "available": {
            split: {
                property_id: len(eligible[split][property_id])
                for property_id in PROPERTIES
            }
            for split in SPLITS
        },
        "selected": {
            split: {
                property_id: sum(
                    str(token_meta[item_id]["property_id"]) == property_id
                    for item_id in cohorts[split]
                )
                for property_id in PROPERTIES
            }
            for split in SPLITS
        },
        "cohort_sizes": {split: len(cohorts[split]) for split in SPLITS},
        "cohorts_disjoint": set(cohorts["discovery"]).isdisjoint(
            cohorts["confirmation"]
        ),
        "avoid_all_prior_hidden_and_reserve": selected.isdisjoint(excluded),
        "all_lcp_zero": all(
            int(token_meta[item_id]["common_prefix_length"]) == 0
            for item_id in selected
        ),
        "reserve_count": len(reserve),
        "reserve_by_property": {
            property_id: sum(
                str(token_meta[item_id]["property_id"]) == property_id
                for item_id in reserve
            )
            for property_id in PROPERTIES
        },
        "reserve_disjoint": set(reserve).isdisjoint(selected),
        "shared_behavior_counts": {split: len(shared[split]) for split in SPLITS},
    }
    return cohorts, reserve, audit


def protocol_command() -> None:
    if (OUT_ROOT / "runs").exists():
        raise RuntimeError("refusing to rewrite Phase1142 after model output exists")
    cohorts, reserve, material_audit = freeze_material()
    prior_prereg = read_json(SOURCE1141 / "protocol/preregistration.json")
    prior_final = read_json(SOURCE1141 / "analysis/final.json")
    prior_audit = read_json(SOURCE1141 / "audit/independent_result_audit.json")
    behavior1135 = read_json(
        ROOT
        / "tests/glm5/result/phase1135_temporal_binding_intervention"
        / "analysis/behavior_authorization.json"
    )
    checks = {
        "phase1141_audit_passed": bool(prior_audit["all_checks_passed"]),
        "phase1141_failed_and_stopped": (
            prior_final["outcome"] == "discovery_failed"
            and prior_final["auto_continue"] is False
        ),
        "fresh_12_per_split": all(
            len(cohorts[split]) == COHORT_SIZE for split in SPLITS
        ),
        "four_per_property": all(
            material_audit["selected"][split][property_id] == ITEMS_PER_PROPERTY
            for split in SPLITS
            for property_id in PROPERTIES
        ),
        "cohorts_disjoint": material_audit["cohorts_disjoint"],
        "avoid_prior_hidden_and_reserve": material_audit[
            "avoid_all_prior_hidden_and_reserve"
        ],
        "all_lcp_zero": material_audit["all_lcp_zero"],
        "reserve_12": len(reserve) == len(PROPERTIES) * RESERVE_PER_PROPERTY,
        "reserve_disjoint": material_audit["reserve_disjoint"],
        "matrix_count_264": MATRIX_COMPARISONS == 264,
        "control_count_24": CONTROL_COMPARISONS == 24,
        "expected_records_1152": EXPECTED_RECORDS == 1152,
        "paired_alpha_grid": ALPHAS == (0.0, 1.0),
        "depth_frozen_0_7": REQUESTED_FRACTION == 0.7,
        "qwen_tokenizers_identical": sha256_file(
            ROOT / "models/hf/qwen3-4b/tokenizer.json"
        )
        == sha256_file(ROOT / "models/hf/Qwen3-14B/tokenizer.json"),
        "glm4_ds7b_prior_hidden_denied": behavior1135[
            "authorized_models"
        ] == ["qwen3"],
        "no_output_before_protocol": not (OUT_ROOT / "runs").exists(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase1142 protocol checks failed: {checks}")

    core = {
        "schema_version": "phase1142_causal_donor_matrix_preregistration.v1",
        "phase": PHASE,
        "created_at_utc": utc_now(),
        "objective": (
            "Estimate a complete paired target-by-donor causal response matrix and "
            "separate item-diagonal, relation-block, and globally reusable temporal-answer "
            "effects without selecting another hidden-state hotspot."
        ),
        "epistemic_scope": (
            "Same-family Qwen3 FP16 whole-residual sufficiency at relative depth 0.7 on "
            "fresh, immediate-divergence Wikidata templates. It does not identify a semantic "
            "vector, component, necessity, natural generation mechanism, or cross-architecture invariant."
        ),
        "source": {
            "items_sha256": sha256_file(SOURCE_ITEMS),
            "qwen4_decisions_sha256": sha256_file(phase1141.Q4_DECISIONS),
            "qwen14_decisions_sha256": sha256_file(phase1141.Q14_DECISIONS),
            "token_cases_sha256": sha256_file(phase1141.TOKEN_CASES),
            "phase1141_protocol_digest": prior_prereg["protocol_digest"],
            "phase1141_final_digest": prior_final["final_digest"],
            "phase1141_audit_digest": prior_audit["audit_digest"],
            "phase1135_behavior_authorization_digest": behavior1135[
                "authorization_digest"
            ],
            "script_sha256": sha256_file(Path(__file__)),
        },
        "models": prior_prereg["models"],
        "cross_architecture_status": {
            "qwen3_4b": "authorized by all-four-state behavior qualification",
            "qwen3_14b": "authorized by all-four-state behavior qualification",
            "glm4": "hidden intervention denied by frozen Phase1135 FP16 behavior gate",
            "deepseek7b": "hidden intervention denied by frozen Phase1135 FP16 behavior gate",
        },
        "material": {
            "properties": list(PROPERTIES),
            "items_per_property": ITEMS_PER_PROPERTY,
            "cohort_size": COHORT_SIZE,
            "cohorts": cohorts,
            "reserve_item_ids": reserve,
            "material_audit": material_audit,
            "selection_rule": (
                "fresh all-four-correct Qwen3-4B/Qwen3-14B items, never used in Phase1138-1141 "
                "hidden/cohort/reserve data, LCP=0, deterministic hash order"
            ),
        },
        "intervention": {
            "component": "whole residual stream after frozen relative depth 0.7 layer",
            "position": (
                "first lexical divergence; all selected candidates have LCP=0, so this equals "
                "the answer-boundary prediction position and removes source-offset confounding"
            ),
            "operation": "live-state absolute interpolation",
            "alphas": list(ALPHAS),
            "panels": {
                key: {
                    "target_state": value[0],
                    "source_state": value[1],
                    "base_key": value[2],
                    "desired_key": value[3],
                }
                for key, value in PANELS.items()
            },
            "matrix": (
                "all 12x12 target-donor cells in both panels; each off-diagonal cell is paired "
                "with a fresh same-item diagonal reference in the identical forward batch"
            ),
            "controls": [
                "same-item same-answer early-date donor paired with correct donor",
                "same-state self donor paired with correct donor",
                "alpha zero and alpha one paired in every forward batch",
            ],
            "matrix_comparisons": MATRIX_COMPARISONS,
            "control_comparisons": CONTROL_COMPARISONS,
            "expected_records_per_model_split": EXPECTED_RECORDS,
        },
        "thresholds": THRESHOLDS,
        "predictions": {
            "P1": "all protocol, source, model, material, precision, pairing, and count audits pass",
            "P2": "paired alpha-zero margins agree within 0.005 and self replacement is inert",
            "P3_item": (
                "the same-item diagonal exceeds same-relation and cross-relation donors by the "
                "frozen magnitude, fraction, flip, rank, and per-property gates"
            ),
            "P4_relation": (
                "if item identity fails, same-relation off-diagonal donors may still exceed "
                "cross-relation donors by the separately frozen relation-block gates"
            ),
            "P5_global": (
                "high cross-relation flip and positive-change fractions without P3/P4 support "
                "a role-compatible reusable field, not a single proven universal vector"
            ),
            "P6": (
                "only a route passing in both discovery models authorizes untouched confirmation; "
                "only independent confirmation authorizes donor-difference injection"
            ),
        },
        "decision_rule": {
            "discovery": (
                "authorize confirmation iff both models pass the item route or both models pass "
                "the relation route; freeze the highest supported claim level"
            ),
            "confirmation": "repeat the identical route and thresholds on the untouched source split",
            "next_phase": (
                "confirmed item or relation selectivity authorizes a separately preregistered "
                "donor-difference injection phase on reserve items"
            ),
        },
        "hard_stops": [
            "do not consume Phase1141 confirmation or reserve",
            "do not add another depth or position after seeing the matrix",
            "do not change cohort, matrix, pairing, thresholds, controls, or precision after output",
            "do not remove weak targets or donors",
            "do not call high off-diagonal response one fixed universal vector",
            "do not run confirmation unless one route passes both discovery models",
            "do not run attention, MLP, head, neuron, SAE, Jacobian, or necessity searches",
            "do not run donor-difference injection before independent confirmation",
            "do not upgrade external-machine consensus to human gold",
        ],
    }
    prereg = dict(core)
    prereg["protocol_digest"] = digest(core)
    audit_core = {
        "schema_version": "phase1142_protocol_audit.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(bool(value) for value in checks.values()),
        "all_checks_passed": all(checks.values()),
    }
    audit = dict(audit_core)
    audit["audit_digest"] = digest(audit_core)
    write_json(OUT_ROOT / "protocol/preregistration.json", prereg)
    write_json(OUT_ROOT / "protocol/audit.json", audit)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "command": "protocol",
                "checks": f"{audit['passed_count']}/{audit['check_count']}",
                "cohorts": {split: len(cohorts[split]) for split in SPLITS},
                "matrix_comparisons": MATRIX_COMPARISONS,
                "expected_records": EXPECTED_RECORDS,
                "protocol_digest": prereg["protocol_digest"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def entry(
    comparison_id: str,
    comparison_kind: str,
    arm: str,
    item_id: str,
    target_property_id: str,
    panel: str,
    target_state: str,
    source_item_id: str,
    source_property_id: str,
    source_state: str,
    base_key: str,
    desired_key: str,
    alpha: float,
) -> dict[str, Any]:
    curve_id = f"{comparison_id}|arm={arm}"
    return {
        "schema_version": "phase1142_donor_matrix_record.v1",
        "phase": PHASE,
        "comparison_id": comparison_id,
        "comparison_kind": comparison_kind,
        "arm": arm,
        "curve_id": curve_id,
        "item_id": item_id,
        "source_item_id": source_item_id,
        "curve_kind": arm,
        "panel": panel,
        "stratum": "immediate_divergence",
        "property_id": target_property_id,
        "source_property_id": source_property_id,
        "same_relation": target_property_id == source_property_id,
        "common_prefix_length": 0,
        "source_common_prefix_length": 0,
        "target_state": target_state,
        "target_case_id": source.state_id(item_id, target_state),
        "source_state": source_state,
        "source_case_id": source.state_id(source_item_id, source_state),
        "base_key": base_key,
        "desired_key": desired_key,
        "scope": "first_lexical_divergence",
        "alpha": alpha,
        "record_id": f"{curve_id}|alpha={alpha:.2f}",
    }


def paired_block(
    comparison_id: str,
    comparison_kind: str,
    item_id: str,
    target_property_id: str,
    panel: str,
    target_state: str,
    reference_source_state: str,
    challenger_item_id: str,
    challenger_property_id: str,
    challenger_source_state: str,
    base_key: str,
    desired_key: str,
) -> list[dict[str, Any]]:
    rows = []
    for arm, source_item_id, property_id, source_state in (
        (
            "paired_correct_donor",
            item_id,
            target_property_id,
            reference_source_state,
        ),
        (
            "challenger_donor",
            challenger_item_id,
            challenger_property_id,
            challenger_source_state,
        ),
    ):
        for alpha in ALPHAS:
            rows.append(
                entry(
                    comparison_id,
                    comparison_kind,
                    arm,
                    item_id,
                    target_property_id,
                    panel,
                    target_state,
                    source_item_id,
                    property_id,
                    source_state,
                    base_key,
                    desired_key,
                    alpha,
                )
            )
    return rows


def build_blocks(
    item_ids: list[str], token_meta: dict[str, dict[str, Any]]
) -> list[list[dict[str, Any]]]:
    blocks = []
    ordered_items = stable_order("matrix-targets", item_ids)
    for item_id in ordered_items:
        target_property = str(token_meta[item_id]["property_id"])
        donors = stable_order(
            f"matrix-donors|{item_id}",
            [donor for donor in ordered_items if donor != item_id],
        )
        for panel, specification in PANELS.items():
            target_state, source_state, base_key, desired_key = specification
            for donor_id in donors:
                donor_property = str(token_meta[donor_id]["property_id"])
                comparison_id = f"matrix|{item_id}|{panel}|donor={donor_id}"
                blocks.append(
                    paired_block(
                        comparison_id,
                        "matrix",
                        item_id,
                        target_property,
                        panel,
                        target_state,
                        source_state,
                        donor_id,
                        donor_property,
                        source_state,
                        base_key,
                        desired_key,
                    )
                )

        blocks.append(
            paired_block(
                f"same_answer|{item_id}",
                "same_answer_temporal_control",
                item_id,
                target_property,
                "original",
                "original_pre",
                "original_post",
                item_id,
                target_property,
                "original_pre_early",
                "old",
                "new",
            )
        )
        blocks.append(
            paired_block(
                f"self_identity|{item_id}",
                "self_identity_control",
                item_id,
                target_property,
                "original",
                "original_pre",
                "original_post",
                item_id,
                target_property,
                "original_pre",
                "old",
                "new",
            )
        )
    if len(blocks) != EXPECTED_COMPARISONS:
        raise RuntimeError(f"comparison count drift: {len(blocks)}")
    if any(len(block) != RECORDS_PER_COMPARISON for block in blocks):
        raise RuntimeError("paired block record count drift")
    return blocks


def curve_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["curve_id"])].append(row)
    curves = []
    for curve_id, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: float(row["alpha"]))
        if [float(row["alpha"]) for row in ordered] != list(ALPHAS):
            raise RuntimeError(f"alpha grid drift for {curve_id}")
        base, endpoint = ordered
        finite = bool(base["finite"] and endpoint["finite"])
        base_margin = base["full_oriented_margin"]
        endpoint_margin = endpoint["full_oriented_margin"]
        change = (
            float(endpoint_margin) - float(base_margin)
            if finite and base_margin is not None and endpoint_margin is not None
            else None
        )
        curves.append(
            {
                "curve_id": curve_id,
                "comparison_id": base["comparison_id"],
                "comparison_kind": base["comparison_kind"],
                "arm": base["arm"],
                "model": base["model"],
                "split": base["split"],
                "item_id": base["item_id"],
                "source_item_id": base["source_item_id"],
                "property_id": base["property_id"],
                "source_property_id": base["source_property_id"],
                "same_relation": bool(base["same_relation"]),
                "panel": base["panel"],
                "finite": finite,
                "baseline_margin": base_margin,
                "endpoint_margin": endpoint_margin,
                "margin_change": change,
                "baseline_valid": bool(
                    finite and base_margin is not None and float(base_margin) < 0.0
                ),
                "endpoint_flip": bool(
                    finite and endpoint_margin is not None and float(endpoint_margin) > 0.0
                ),
                "positive_change": bool(change is not None and change > 0.0),
            }
        )
    return curves


def comparison_rows(curves: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in curves:
        grouped[str(row["comparison_id"])][str(row["arm"])] = row
    result = []
    for comparison_id, arms in sorted(grouped.items()):
        if set(arms) != {"paired_correct_donor", "challenger_donor"}:
            raise RuntimeError(f"comparison arm drift for {comparison_id}")
        reference = arms["paired_correct_donor"]
        challenger = arms["challenger_donor"]
        finite = bool(reference["finite"] and challenger["finite"])
        advantage = (
            float(reference["margin_change"]) - float(challenger["margin_change"])
            if finite
            and reference["margin_change"] is not None
            and challenger["margin_change"] is not None
            else None
        )
        alpha0_difference = (
            abs(float(reference["baseline_margin"]) - float(challenger["baseline_margin"]))
            if finite
            and reference["baseline_margin"] is not None
            and challenger["baseline_margin"] is not None
            else None
        )
        result.append(
            {
                "comparison_id": comparison_id,
                "comparison_kind": reference["comparison_kind"],
                "model": reference["model"],
                "split": reference["split"],
                "item_id": reference["item_id"],
                "source_item_id": challenger["source_item_id"],
                "property_id": reference["property_id"],
                "source_property_id": challenger["source_property_id"],
                "same_relation": bool(challenger["same_relation"]),
                "panel": reference["panel"],
                "finite": finite,
                "paired_alpha0_abs_margin_difference": alpha0_difference,
                "diagonal_change": reference["margin_change"],
                "challenger_change": challenger["margin_change"],
                "diagonal_endpoint_flip": bool(reference["endpoint_flip"]),
                "challenger_endpoint_flip": bool(challenger["endpoint_flip"]),
                "diagonal_positive_change": bool(reference["positive_change"]),
                "challenger_positive_change": bool(challenger["positive_change"]),
                "diagonal_baseline_valid": bool(reference["baseline_valid"]),
                "challenger_baseline_valid": bool(challenger["baseline_valid"]),
                "diagonal_advantage": advantage,
                "diagonal_advantage_positive": bool(
                    advantage is not None and advantage > 0.0
                ),
            }
        )
    return result


def target_panel_rows(matrix: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in matrix:
        grouped[(str(row["item_id"]), str(row["panel"]))].append(row)
    result = []
    for (item_id, panel), rows in sorted(grouped.items()):
        same = [row for row in rows if row["same_relation"]]
        cross = [row for row in rows if not row["same_relation"]]
        if len(rows) != COHORT_SIZE - 1 or len(same) != ITEMS_PER_PROPERTY - 1:
            raise RuntimeError(f"target-panel matrix drift for {item_id} {panel}")
        diagonal = median(row["diagonal_change"] for row in rows)
        donor_values = [float(row["challenger_change"]) for row in rows]
        rank = (
            1 + sum(value >= float(diagonal) for value in donor_values)
            if diagonal is not None
            else None
        )
        same_median = median(row["challenger_change"] for row in same)
        cross_median = median(row["challenger_change"] for row in cross)
        relation_advantage = (
            float(same_median) - float(cross_median)
            if same_median is not None and cross_median is not None
            else None
        )
        result.append(
            {
                "item_id": item_id,
                "panel": panel,
                "property_id": rows[0]["property_id"],
                "diagonal_change_median": diagonal,
                "same_relation_change_median": same_median,
                "cross_relation_change_median": cross_median,
                "relation_advantage": relation_advantage,
                "relation_advantage_positive": bool(
                    relation_advantage is not None and relation_advantage > 0.0
                ),
                "diagonal_rank": rank,
                "diagonal_top1": rank == 1,
                "diagonal_top3": rank is not None and rank <= 3,
            }
        )
    return result


def analyze_records(
    model_name: str, split: str, records: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    curves = curve_rows(records)
    comparisons = comparison_rows(curves)
    matrix = [row for row in comparisons if row["comparison_kind"] == "matrix"]
    same = [row for row in matrix if row["same_relation"]]
    cross = [row for row in matrix if not row["same_relation"]]
    temporal = [
        row
        for row in comparisons
        if row["comparison_kind"] == "same_answer_temporal_control"
    ]
    self_controls = [
        row
        for row in comparisons
        if row["comparison_kind"] == "self_identity_control"
    ]
    target_panels = target_panel_rows(matrix)

    diagonal_change = median(row["diagonal_change"] for row in matrix)
    temporal_abs = median(
        abs(float(row["challenger_change"]))
        for row in temporal
        if row["challenger_change"] is not None
    )
    paired_alpha0 = [
        float(row["paired_alpha0_abs_margin_difference"])
        for row in comparisons
        if row["paired_alpha0_abs_margin_difference"] is not None
    ]
    self_abs = [
        abs(float(row["challenger_change"]))
        for row in self_controls
        if row["challenger_change"] is not None
    ]
    per_property_item = {
        property_id: median(
            row["diagonal_advantage"]
            for row in matrix
            if row["property_id"] == property_id
        )
        for property_id in PROPERTIES
    }
    per_property_relation = {
        property_id: median(
            row["relation_advantage"]
            for row in target_panels
            if row["property_id"] == property_id
        )
        for property_id in PROPERTIES
    }
    donor_columns = {}
    for donor_id in sorted({str(row["source_item_id"]) for row in matrix}):
        donor_columns[donor_id] = {
            "property_id": next(
                str(row["source_property_id"])
                for row in matrix
                if str(row["source_item_id"]) == donor_id
            ),
            "median_change": median(
                row["challenger_change"]
                for row in matrix
                if str(row["source_item_id"]) == donor_id
            ),
            "flip_fraction": fraction(
                [
                    row
                    for row in matrix
                    if str(row["source_item_id"]) == donor_id
                ],
                "challenger_endpoint_flip",
            ),
        }

    metrics = {
        "model": model_name,
        "split": split,
        "record_count": len(records),
        "curve_count": len(curves),
        "comparison_count": len(comparisons),
        "matrix_comparison_count": len(matrix),
        "same_relation_matrix_count": len(same),
        "cross_relation_matrix_count": len(cross),
        "finite_fraction": sum(bool(row["finite"]) for row in records)
        / max(len(records), 1),
        "paired_alpha0_max_abs_margin_difference": max(paired_alpha0)
        if paired_alpha0
        else None,
        "diagonal_baseline_valid_fraction": fraction(
            matrix, "diagonal_baseline_valid"
        ),
        "diagonal_endpoint_flip_fraction": fraction(
            matrix, "diagonal_endpoint_flip"
        ),
        "diagonal_positive_change_fraction": fraction(
            matrix, "diagonal_positive_change"
        ),
        "diagonal_change_median": diagonal_change,
        "offdiagonal_change_median": median(
            row["challenger_change"] for row in matrix
        ),
        "same_relation_change_median": median(
            row["challenger_change"] for row in same
        ),
        "cross_relation_change_median": median(
            row["challenger_change"] for row in cross
        ),
        "offdiagonal_endpoint_flip_fraction": fraction(
            matrix, "challenger_endpoint_flip"
        ),
        "same_relation_endpoint_flip_fraction": fraction(
            same, "challenger_endpoint_flip"
        ),
        "cross_relation_endpoint_flip_fraction": fraction(
            cross, "challenger_endpoint_flip"
        ),
        "cross_relation_positive_change_fraction": fraction(
            cross, "challenger_positive_change"
        ),
        "item_advantage_median": median(
            row["diagonal_advantage"] for row in matrix
        ),
        "item_advantage_positive_fraction": fraction(
            matrix, "diagonal_advantage_positive"
        ),
        "item_advantage_same_relation_median": median(
            row["diagonal_advantage"] for row in same
        ),
        "item_advantage_cross_relation_median": median(
            row["diagonal_advantage"] for row in cross
        ),
        "diagonal_minus_offdiagonal_flip_fraction": fraction(
            matrix, "diagonal_endpoint_flip"
        )
        - fraction(matrix, "challenger_endpoint_flip"),
        "diagonal_top1_fraction": fraction(target_panels, "diagonal_top1"),
        "diagonal_top3_fraction": fraction(target_panels, "diagonal_top3"),
        "diagonal_rank_median": median(row["diagonal_rank"] for row in target_panels),
        "per_property_item_advantage_median": per_property_item,
        "relation_advantage_median": median(
            row["relation_advantage"] for row in target_panels
        ),
        "relation_advantage_positive_fraction": fraction(
            target_panels, "relation_advantage_positive"
        ),
        "same_minus_cross_relation_flip_fraction": fraction(
            same, "challenger_endpoint_flip"
        )
        - fraction(cross, "challenger_endpoint_flip"),
        "per_property_relation_advantage_median": per_property_relation,
        "same_answer_endpoint_flip_fraction": fraction(
            temporal, "challenger_endpoint_flip"
        ),
        "same_answer_abs_change_median": temporal_abs,
        "self_identity_max_abs_margin_change": max(self_abs) if self_abs else None,
        "diagonal_to_same_answer_abs_ratio": (
            float(diagonal_change) / max(float(temporal_abs), EPSILON)
            if diagonal_change is not None and temporal_abs is not None
            else None
        ),
        "donor_columns": donor_columns,
    }
    instrument = {
        "finite": metrics["finite_fraction"] >= THRESHOLDS["finite_fraction"],
        "paired_alpha0": (
            metrics["paired_alpha0_max_abs_margin_difference"] is not None
            and metrics["paired_alpha0_max_abs_margin_difference"]
            <= THRESHOLDS["paired_alpha0_max_abs_margin_difference"]
        ),
        "diagonal_baseline": metrics["diagonal_baseline_valid_fraction"]
        >= THRESHOLDS["diagonal_baseline_valid_fraction"],
        "diagonal_endpoint": metrics["diagonal_endpoint_flip_fraction"]
        >= THRESHOLDS["diagonal_endpoint_flip_fraction"],
        "diagonal_positive": metrics["diagonal_positive_change_fraction"]
        >= THRESHOLDS["diagonal_positive_change_fraction"],
        "same_answer": metrics["same_answer_endpoint_flip_fraction"]
        <= THRESHOLDS["same_answer_endpoint_flip_fraction"],
        "self_identity": (
            metrics["self_identity_max_abs_margin_change"] is not None
            and metrics["self_identity_max_abs_margin_change"]
            <= THRESHOLDS["self_identity_max_abs_margin_change"]
        ),
        "temporal_ratio": (
            metrics["diagonal_to_same_answer_abs_ratio"] is not None
            and metrics["diagonal_to_same_answer_abs_ratio"]
            >= THRESHOLDS["diagonal_to_same_answer_abs_ratio"]
        ),
    }
    item = {
        "instrument": all(instrument.values()),
        "median": metrics["item_advantage_median"] is not None
        and metrics["item_advantage_median"] >= THRESHOLDS["item_advantage_median"],
        "positive_fraction": metrics["item_advantage_positive_fraction"]
        >= THRESHOLDS["item_advantage_positive_fraction"],
        "same_relation": metrics["item_advantage_same_relation_median"] is not None
        and metrics["item_advantage_same_relation_median"]
        >= THRESHOLDS["item_advantage_same_relation_median"],
        "cross_relation": metrics["item_advantage_cross_relation_median"] is not None
        and metrics["item_advantage_cross_relation_median"]
        >= THRESHOLDS["item_advantage_cross_relation_median"],
        "flip_advantage": metrics["diagonal_minus_offdiagonal_flip_fraction"]
        >= THRESHOLDS["diagonal_minus_offdiagonal_flip_fraction"],
        "top1": metrics["diagonal_top1_fraction"]
        >= THRESHOLDS["diagonal_top1_fraction"],
        "each_property": all(
            per_property_item[property_id] is not None
            and float(per_property_item[property_id])
            >= THRESHOLDS["per_property_item_advantage_median"]
            for property_id in PROPERTIES
        ),
    }
    relation = {
        "instrument": all(instrument.values()),
        "median": metrics["relation_advantage_median"] is not None
        and metrics["relation_advantage_median"]
        >= THRESHOLDS["relation_advantage_median"],
        "positive_fraction": metrics["relation_advantage_positive_fraction"]
        >= THRESHOLDS["relation_advantage_positive_fraction"],
        "flip_advantage": metrics["same_minus_cross_relation_flip_fraction"]
        >= THRESHOLDS["same_minus_cross_relation_flip_fraction"],
        "each_property": all(
            per_property_relation[property_id] is not None
            and float(per_property_relation[property_id])
            >= THRESHOLDS["per_property_relation_advantage_median"]
            for property_id in PROPERTIES
        ),
    }
    global_direction = {
        "cross_relation_flip": metrics["cross_relation_endpoint_flip_fraction"]
        >= THRESHOLDS["global_cross_relation_flip_fraction"],
        "cross_relation_positive": metrics["cross_relation_positive_change_fraction"]
        >= THRESHOLDS["global_cross_relation_positive_fraction"],
        "item_route_failed": not all(item.values()),
        "relation_route_failed": not all(relation.values()),
    }
    metrics["instrument_gate_checks"] = instrument
    metrics["instrument_qualified"] = all(instrument.values())
    metrics["item_gate_checks"] = item
    metrics["item_selectivity_qualified"] = all(item.values())
    metrics["relation_gate_checks"] = relation
    metrics["relation_selectivity_qualified"] = all(relation.values())
    metrics["global_direction_checks"] = global_direction
    metrics["global_role_compatible_field_observed"] = all(global_direction.values())
    metrics["curve_digest"] = digest(curves)
    metrics["comparison_digest"] = digest(comparisons)
    return metrics, curves, comparisons


def run_command(model_name: str, split: str) -> None:
    if model_name not in MODELS or split not in SPLITS:
        raise RuntimeError("invalid Phase1142 endpoint")
    prereg = read_json(OUT_ROOT / "protocol/preregistration.json")
    protocol_audit = read_json(OUT_ROOT / "protocol/audit.json")
    if not protocol_audit["all_checks_passed"]:
        raise RuntimeError("Phase1142 protocol audit failed")
    if split == "confirmation":
        selection = read_json(OUT_ROOT / "analysis/discovery_selection.json")
        if not selection["confirmation_authorized"]:
            raise RuntimeError("Phase1142 discovery denied confirmation")
    output_root = OUT_ROOT / "runs" / split / model_name
    if output_root.exists():
        raise RuntimeError(f"refusing to overwrite {output_root}")

    selected_ids = list(prereg["material"]["cohorts"][split])
    items = read_jsonl(SOURCE_ITEMS)
    selected_items, cases = source.causal_cases(items, split, selected_ids)
    all_token_meta = phase1141.token_metadata()
    selected_token_meta = {
        item_id: all_token_meta[item_id] for item_id in selected_ids
    }
    blocks = build_blocks(selected_ids, selected_token_meta)
    model = None
    started = time.time()
    records: list[dict[str, Any]] = []
    try:
        model, tokenizer, device, placement = phase1138.load_model(model_name, prereg)
        precision = quantization_audit(model)
        expected = prereg["models"][model_name]
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count != int(expected["expected_parameter_count"]):
            raise RuntimeError(f"{model_name} parameter count mismatch")
        if (
            precision["has_quantized_modules"]
            or precision["has_bf16_parameters"]
            or not precision["has_fp16_parameters"]
        ):
            raise RuntimeError(f"{model_name} FP16/no-quantization gate failed")
        layers = get_layers(model)
        depth_rows = [
            row
            for row in phase1138.depth_rows_for_model(len(layers))
            if math.isclose(
                float(row["requested_fraction"]),
                REQUESTED_FRACTION,
                abs_tol=1e-12,
            )
        ]
        if len(depth_rows) != 1:
            raise RuntimeError("frozen depth did not map to one layer")
        depth = int(depth_rows[0]["depth"])
        layer = layers[depth - 1]
        pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        capture_batch_size = int(expected["batch_size"])
        vectors = phase1140.capture_candidate_paths(
            model,
            layer,
            cases,
            tokenizer,
            int(pad_id),
            device,
            capture_batch_size,
        )

        # A block is eight expanded sequences: two arms x two alphas x two candidates.
        blocks_per_forward = max(1, capture_batch_size // 8)
        for start in range(0, len(blocks), blocks_per_forward):
            current_blocks = blocks[start : start + blocks_per_forward]
            current_entries = [row for block in current_blocks for row in block]
            for row in current_entries:
                row["model"] = model_name
                row["split"] = split
                row["depth"] = depth
                row["relative_depth"] = depth / len(layers)
                row["requested_fraction"] = REQUESTED_FRACTION
            records.extend(
                phase1141.score_intervention_batch(
                    model,
                    layer,
                    current_entries,
                    cases,
                    vectors,
                    selected_token_meta,
                    tokenizer,
                    int(pad_id),
                    device,
                )
            )
            completed = min(start + blocks_per_forward, len(blocks))
            if completed % 25 == 0 or completed == len(blocks):
                print(
                    json.dumps(
                        {
                            "phase": PHASE,
                            "model": model_name,
                            "split": split,
                            "comparisons": completed,
                            "records": len(records),
                        }
                    ),
                    flush=True,
                )
        if len(records) != EXPECTED_RECORDS:
            raise RuntimeError(f"record count drift: {len(records)}")
        metrics, curves, comparisons = analyze_records(model_name, split, records)
        core = {
            "schema_version": "phase1142_donor_matrix_run_summary.v1",
            "phase": PHASE,
            "model": model_name,
            "split": split,
            "protocol_digest": prereg["protocol_digest"],
            "precision": precision,
            "parameter_count": parameter_count,
            "placement": placement,
            "layer_count": len(layers),
            "depth": depth,
            "relative_depth": depth / len(layers),
            "requested_fraction": REQUESTED_FRACTION,
            "item_count": len(selected_items),
            "record_count": len(records),
            "metrics": metrics,
            "elapsed_seconds": time.time() - started,
            "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "record_digest": digest(records),
            "evidence_scope": "same_family_fresh_split_paired_causal_donor_matrix",
        }
        summary = dict(core)
        summary["summary_digest"] = digest(core)
        write_jsonl(output_root / "records.jsonl", records)
        write_jsonl(output_root / "curves.jsonl", curves)
        write_jsonl(output_root / "comparisons.jsonl", comparisons)
        write_json(output_root / "summary.json", summary)
        print(
            json.dumps(
                {
                    "phase": PHASE,
                    "command": "run",
                    "model": model_name,
                    "split": split,
                    "instrument": metrics["instrument_qualified"],
                    "item": metrics["item_selectivity_qualified"],
                    "relation": metrics["relation_selectivity_qualified"],
                    "global_field": metrics[
                        "global_role_compatible_field_observed"
                    ],
                    "item_advantage": metrics["item_advantage_median"],
                    "relation_advantage": metrics["relation_advantage_median"],
                    "summary_digest": summary["summary_digest"],
                }
            ),
            flush=True,
        )
    finally:
        if model is not None:
            if model_name == "qwen3_4b":
                release_fp16(model)
            else:
                del model
                gc.collect()
                torch.cuda.empty_cache()


def select_command() -> None:
    prereg = read_json(OUT_ROOT / "protocol/preregistration.json")
    summaries = {
        model: read_json(OUT_ROOT / "runs/discovery" / model / "summary.json")
        for model in MODELS
    }
    if any(
        summary["protocol_digest"] != prereg["protocol_digest"]
        for summary in summaries.values()
    ):
        raise RuntimeError("Phase1142 discovery protocol digest drift")
    item = {
        model: bool(summary["metrics"]["item_selectivity_qualified"])
        for model, summary in summaries.items()
    }
    relation = {
        model: bool(summary["metrics"]["relation_selectivity_qualified"])
        for model, summary in summaries.items()
    }
    item_authorized = all(item.values())
    relation_authorized = all(relation.values())
    confirmation_authorized = item_authorized or relation_authorized
    claim_level = (
        "item"
        if item_authorized
        else "relation"
        if relation_authorized
        else None
    )
    core = {
        "schema_version": "phase1142_discovery_selection.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "models": {model: summary["metrics"] for model, summary in summaries.items()},
        "item_qualified": item,
        "relation_qualified": relation,
        "item_route_authorized": item_authorized,
        "relation_route_authorized": relation_authorized,
        "selected_claim_level": claim_level,
        "confirmation_authorized": confirmation_authorized,
        "selection_rule_followed": True,
    }
    result = dict(core)
    result["selection_digest"] = digest(core)
    write_json(OUT_ROOT / "analysis/discovery_selection.json", result)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "command": "select",
                "item": item,
                "relation": relation,
                "claim_level": claim_level,
                "confirmation_authorized": confirmation_authorized,
                "selection_digest": result["selection_digest"],
            }
        ),
        flush=True,
    )


def finalize_command() -> None:
    prereg = read_json(OUT_ROOT / "protocol/preregistration.json")
    selection = read_json(OUT_ROOT / "analysis/discovery_selection.json")
    confirmations = {}
    if selection["confirmation_authorized"]:
        for model in MODELS:
            path = OUT_ROOT / "runs/confirmation" / model / "summary.json"
            if not path.exists():
                raise RuntimeError(f"missing confirmation output for {model}")
            confirmations[model] = read_json(path)
    selected = selection["selected_claim_level"]
    confirmed = bool(
        confirmations
        and selected is not None
        and all(
            bool(
                confirmations[model]["metrics"][
                    f"{selected}_selectivity_qualified"
                ]
            )
            for model in MODELS
        )
    )
    if not selection["confirmation_authorized"]:
        outcome = "discovery_no_identity_or_relation_selectivity"
    elif confirmed:
        outcome = f"{selected}_selectivity_independently_confirmed"
    else:
        outcome = f"{selected}_selectivity_confirmation_failed"
    global_field_discovery = {
        model: bool(
            selection["models"][model]["global_role_compatible_field_observed"]
        )
        for model in MODELS
    }
    core = {
        "schema_version": "phase1142_donor_matrix_final.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "selection_digest": selection["selection_digest"],
        "confirmation_run": bool(confirmations),
        "selected_claim_level": selected,
        "selected_route_confirmed": confirmed,
        "global_role_compatible_field_discovery": global_field_discovery,
        "outcome": outcome,
        "donor_difference_injection_authorized": confirmed,
        "component_search_authorized": False,
        "necessity_claim_authorized": False,
        "semantic_vector_claim_authorized": False,
        "cross_architecture_claim_authorized": False,
        "auto_continue": confirmed,
        "claim_boundary": (
            "A confirmed item route identifies same-family whole-residual donor identity "
            "selectivity; a confirmed relation route identifies relation-compatible block "
            "selectivity. Neither is a component, necessary mechanism, semantic vector, natural "
            "generation result, or cross-architecture invariant. High off-diagonal response alone "
            "is only a reusable role-compatible field observation."
        ),
    }
    final = dict(core)
    final["final_digest"] = digest(core)
    write_json(OUT_ROOT / "analysis/final.json", final)
    print(
        json.dumps(
            {
                "phase": PHASE,
                "command": "finalize",
                "outcome": outcome,
                "confirmed": confirmed,
                "auto_continue": confirmed,
                "final_digest": final["final_digest"],
            }
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("protocol")
    run = sub.add_parser("run")
    run.add_argument("model", choices=MODELS)
    run.add_argument("split", choices=SPLITS)
    sub.add_parser("select")
    sub.add_parser("finalize")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "protocol":
        protocol_command()
    elif args.command == "run":
        run_command(args.model, args.split)
    elif args.command == "select":
        select_command()
    elif args.command == "finalize":
        finalize_command()
    else:
        raise RuntimeError(f"unknown command {args.command}")


if __name__ == "__main__":
    main()
