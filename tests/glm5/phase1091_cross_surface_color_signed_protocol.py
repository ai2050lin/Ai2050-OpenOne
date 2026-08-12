#!/usr/bin/env python3
"""Freeze Phase1091 English/Chinese cross-surface signed color map."""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1090_cross_surface_color_behavior_protocol as base
import phase1089_truth_matched_color_binding_protocol as signed_base


PHASE = 1091
PROTOCOL_REVISION = 1
MODELS = base.MODELS
PRECISION = "fp16"
QUANTIZATION = "none"
COLORS = base.COLORS
COLOR_PAIRS = base.COLOR_PAIRS
OPERATIONS = base.OPERATIONS
BASE_WORLDS = base.BASE_WORLDS
SURFACE_ROUTES = ("en_en", "zh_zh", "en_zh", "zh_en")
WORLDS = tuple(
    f"{world}@{route}"
    for world in BASE_WORLDS
    for route in SURFACE_ROUTES
)
CELLS = tuple(
    f"{operation}__{world}"
    for operation in OPERATIONS
    for world in WORLDS
)
FAMILIES = CELLS
SPLITS = base.SPLITS
PANELS = base.PANELS
TEMPLATE_IDS = (0,)
OUTPUT_SET_IDS = (0,)
STATES = tuple(
    f"t0_c{panel}_m{target}_q{binding}_w0"
    for panel in PANELS
    for target in (0, 1)
    for binding in (0, 1)
)
ITEMS_PER_CELL_SPLIT = base.ITEMS_PER_CELL_SPLIT
ASSISTANT_PREFILL = base.ASSISTANT_PREFILL
CAPTURE_ROLES = signed_base.CAPTURE_ROLES
PRIMARY_PROFILE_ROLES = signed_base.PRIMARY_PROFILE_ROLES
PRE_QUERY_ROLES = signed_base.PRE_QUERY_ROLES
TARGET_RELATIVE_DEPTH_MIN = signed_base.TARGET_RELATIVE_DEPTH_MIN
TARGET_RELATIVE_DEPTH_MAX = signed_base.TARGET_RELATIVE_DEPTH_MAX
SIGNED_PROJECTION_DIM = signed_base.SIGNED_PROJECTION_DIM
SIGNED_PROJECTION_REPLICATES = signed_base.SIGNED_PROJECTION_REPLICATES
SIGNED_PROJECTION_SEED = 1091001
SIGNED_FIELDS = ("active_binding", "field_null", "content")
OUT_ROOT = (
    ROOT / "tests" / "glm5" / "result"
    / "phase1091_cross_surface_color_signed"
)
SOURCE_PHASE1090 = base.OUT_ROOT

EVIDENCE_THRESHOLDS = {
    **signed_base.EVIDENCE_THRESHOLDS,
    "minimum_cross_surface_pair_directions": 8,
    "minimum_cross_surface_pair_gram_cosine": 0.50,
    "minimum_cross_surface_content_advantage": 0.10,
    "minimum_cross_surface_models": 2,
    "minimum_route_split_top1": 6,
}

PROSPECTIVE_PREDICTIONS = {
    "P1": "The Phase1090 behavior authorization and all Phase1091 static audits pass.",
    "P2": "Both signed projections and FP16 numeric audits pass in at least two models.",
    "P3": (
        "Within-route centered pair identity retrieves at least six of eight "
        "pairs across independent splits for all four routes in two models."
    ),
    "P4": (
        "At least eight of twelve directed surface-route transfers retrieve "
        "six of eight canonical pairs and beat the matched null in two models."
    ),
    "P5": (
        "At least eight of twelve directed surface-route pair Gram comparisons "
        "reach 0.50 and exceed matched-null geometry by 0.10 in two models."
    ),
    "P6": (
        "A common color-binding centroid transfers across lexical routes and "
        "beats the matched null in two models."
    ),
    "P7": (
        "At least two directed healthy model pairs preserve cross-surface pair "
        "geometry with a 0.10 matched-null advantage."
    ),
    "P8": (
        "The preregistered relative-depth 0.30-0.45 query/answer band contains "
        "the strongest repeatable cross-surface pair map in two models."
    ),
}

write_json = base.write_json
write_jsonl = base.write_jsonl
read_json = base.read_json
read_jsonl = base.read_jsonl
digest = base.digest
tokenizer_for = base.tokenizer_for


def split_world(value: str) -> tuple[str, str]:
    return base.split_world(value)


def split_cell(cell: str) -> tuple[str, str]:
    return base.split_cell(cell)


def operation_colors(operation: str) -> tuple[str, str]:
    return base.operation_colors(operation)


def old_state(new_state: str) -> str:
    if not new_state.startswith("t0_") or not new_state.endswith("_w0"):
        raise ValueError(new_state)
    return new_state[3:-3]


def split_items(cell: str, split: str) -> tuple[dict[str, Any], ...]:
    return base.split_items(cell, split)


def build_case(
    tokenizer,
    model_name: str,
    cell: str,
    split: str,
    item: dict[str, Any],
    state: str,
    case_index: int,
) -> dict[str, Any]:
    row = base.build_case(
        tokenizer, model_name, cell, split, item, old_state(state), case_index
    )
    row.update({
        "schema_version": "phase1091_case.v1",
        "phase": PHASE,
        "state": state,
        "template": 0,
        "output_set": 0,
        "label_swap": 0,
        "record_id": f"{model_name}.{cell}.{split}.{item['item_id']}.{state}",
    })
    return row


def signed_pair_records(state_tensor, values, template: int, output_set: int):
    return signed_base.signed_pair_records(
        state_tensor, values, template, output_set
    )


def audit_model(model_name: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        by_unit[str(row["unit_id"])].append(row)
    checks: dict[str, bool] = {}
    checks["complete_factorial_units"] = all(
        {row["state"] for row in rows} == set(STATES)
        for rows in by_unit.values()
    )
    checks["selected_routes_only"] = {
        row["surface_route"] for row in cases
    } == set(SURFACE_ROUTES)
    checks["active_truth_formula"] = all(
        row["semantic_answer_index"]
        == int(int(row["binding"]) != int(row["target_variant"]))
        for row in cases if row["panel"] == "active"
    )
    checks["null_truth_formula"] = all(
        row["semantic_answer_index"]
        == int(int(row["anchor_variant"]) != int(row["target_variant"]))
        for row in cases if row["panel"] == "field_null"
    )
    checks["one_true_one_false_every_side"] = all(
        sorted(
            row["semantic_answer_index"]
            for row in rows
            if row["panel"] == panel and row["binding"] == binding
        ) == [0, 1]
        for rows in by_unit.values()
        for panel in PANELS
        for binding in (0, 1)
    )
    checks["role_positions_valid"] = all(
        all(0 <= int(value) < len(row["input_ids"])
            for value in row["role_positions"].values())
        for row in cases
    )
    checks["single_template_output_axes"] = all(
        row["template"] == 0 and row["output_set"] == 0 for row in cases
    )
    checks["behavior_routes_authorized"] = set(SURFACE_ROUTES).issubset(
        set(read_json(
            SOURCE_PHASE1090 / "analysis" / "final_summary.json"
        )["selected_routes_for_phase1091"])
    )
    checks["all_checks_boolean"] = all(
        isinstance(value, bool) for value in checks.values()
    )
    return {
        "schema_version": "phase1091_protocol_model_audit.v1",
        "phase": PHASE,
        "model": model_name,
        "case_count": len(cases),
        "unit_count": len(by_unit),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "case_digest": digest(cases),
    }


def build_model_cases(model_name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tokenizer = tokenizer_for(model_name)
    cases = []
    case_index = 0
    for cell in CELLS:
        for split in SPLITS:
            for item in split_items(cell, split):
                for state in STATES:
                    cases.append(build_case(
                        tokenizer, model_name, cell, split, item, state, case_index
                    ))
                    case_index += 1
    return cases, audit_model(model_name, cases)


def main() -> None:
    protocol_root = OUT_ROOT / "protocol"
    model_case_digests = {}
    model_audits = {}
    for model_name in MODELS:
        cases, audit = build_model_cases(model_name)
        if not audit["all_checks_passed"]:
            failed = [name for name, value in audit["checks"].items() if not value]
            raise RuntimeError(f"protocol audit failed for {model_name}: {failed}")
        write_jsonl(protocol_root / f"cases.{model_name}.jsonl", cases)
        write_json(protocol_root / f"audit.{model_name}.json", audit)
        model_case_digests[model_name] = audit["case_digest"]
        model_audits[model_name] = audit

    behavior_final = read_json(
        SOURCE_PHASE1090 / "analysis" / "final_summary.json"
    )
    prereg = {
        "schema_version": "phase1091_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "sequential_model_order": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "operations": list(OPERATIONS),
        "color_pairs": [list(pair) for pair in COLOR_PAIRS],
        "base_worlds": list(BASE_WORLDS),
        "surface_routes": list(SURFACE_ROUTES),
        "worlds": list(WORLDS),
        "splits": list(SPLITS),
        "panels": list(PANELS),
        "states": list(STATES),
        "template_ids": list(TEMPLATE_IDS),
        "output_set_ids": list(OUTPUT_SET_IDS),
        "capture_roles": list(CAPTURE_ROLES),
        "relative_depth_range": [
            TARGET_RELATIVE_DEPTH_MIN, TARGET_RELATIVE_DEPTH_MAX
        ],
        "signed_fields": list(SIGNED_FIELDS),
        "projection": {
            "type": "deterministic_rademacher",
            "dimension_per_replicate": SIGNED_PROJECTION_DIM,
            "replicates": SIGNED_PROJECTION_REPLICATES,
            "seed": SIGNED_PROJECTION_SEED,
            "cross_model_rule": "Compare only within-model pair Gram geometry.",
        },
        "items_per_cell_split": ITEMS_PER_CELL_SPLIT,
        "case_count_per_model": len(CELLS) * len(SPLITS) * ITEMS_PER_CELL_SPLIT * len(STATES),
        "unit_count_per_model": len(CELLS) * len(SPLITS) * ITEMS_PER_CELL_SPLIT,
        "evidence_thresholds": EVIDENCE_THRESHOLDS,
        "prospective_predictions": PROSPECTIVE_PREDICTIONS,
        "model_case_digests": model_case_digests,
        "source_phase1090_summary_digest": behavior_final["summary_digest"],
        "behavior_healthy_models": [
            name for name, row in behavior_final["models"].items()
            if row["candidate_finite_fraction"]
            >= base.EVIDENCE_THRESHOLDS["minimum_candidate_finite_fraction"]
        ],
        "interpretation_limits": [
            "A cross-surface map can still reflect a generic equality task rather than stored color qualia.",
            "English/Chinese labels are natural but tokenization and training frequency differ.",
            "GLM4 Phase1090 FP16 behavior was numerically unhealthy and remains exploratory.",
            "The scan covers only the preregistered middle band and is descriptive, not causal.",
        ],
        "automatic_next": {
            "causal_if": "Never from Phase1091 alone.",
            "next_family_if": "Only if cross-surface pair identity and Gram gates P3-P5 pass in two healthy models.",
            "otherwise": "Retain the descriptive map and stop color escalation.",
        },
        "model_audits": model_audits,
    }
    prereg["protocol_digest"] = digest(prereg)
    write_json(protocol_root / "preregistration.json", prereg)
    audit = {
        "schema_version": "phase1091_protocol_audit.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "model_audits": model_audits,
        "all_checks_passed": all(row["all_checks_passed"] for row in model_audits.values()),
    }
    audit["audit_digest"] = digest(audit)
    write_json(protocol_root / "audit.json", audit)
    authorization = {
        "schema_version": "phase1091_behavior_authorization.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "source_phase1090_summary_digest": behavior_final["summary_digest"],
        "models": {
            name: {
                "candidate_finite_fraction": row["candidate_finite_fraction"],
                "behavior_healthy": name in prereg["behavior_healthy_models"],
            }
            for name, row in behavior_final["models"].items()
        },
        "hidden_scan_authorized": (
            audit["all_checks_passed"]
            and behavior_final["hidden_protocol_authorized"]
        ),
        "causal_authorized": False,
    }
    authorization["authorization_digest"] = digest(authorization)
    write_json(OUT_ROOT / "analysis" / "behavior_authorization.json", authorization)
    print({
        "phase": PHASE,
        "case_count_per_model": prereg["case_count_per_model"],
        "unit_count_per_model": prereg["unit_count_per_model"],
        "behavior_healthy_models": prereg["behavior_healthy_models"],
        "hidden_scan_authorized": authorization["hidden_scan_authorized"],
        "protocol_digest": prereg["protocol_digest"],
    })


if __name__ == "__main__":
    main()
