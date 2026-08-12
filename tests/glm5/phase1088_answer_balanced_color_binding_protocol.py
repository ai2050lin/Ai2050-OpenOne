#!/usr/bin/env python3
"""Freeze Phase1088 answer-balanced color-binding field analysis.

Phase1088 reuses every frozen Phase1087 prompt and behavior result.  It changes
only the hidden-state contrast: each binding side averages one true and one
false query before the two binding states are compared.  This removes a direct
true-versus-false axis from the primary field.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1087_color_relation_protocol as base


PHASE = 1088
PROTOCOL_REVISION = 1
MODELS = base.MODELS
PRECISION = base.PRECISION
QUANTIZATION = base.QUANTIZATION
COLORS = base.COLORS
COLOR_PAIRS = base.COLOR_PAIRS
OPERATIONS = base.OPERATIONS
WORLDS = base.WORLDS
CELLS = base.CELLS
FAMILIES = base.FAMILIES
SPLITS = base.SPLITS
PANELS = base.PANELS
OUTPUT_PAIRS = base.OUTPUT_PAIRS
CODE_WORDS = base.CODE_WORDS
STATES = base.STATES
TARGET_RELATIVE_DEPTH_MIN = base.TARGET_RELATIVE_DEPTH_MIN
TARGET_RELATIVE_DEPTH_MAX = base.TARGET_RELATIVE_DEPTH_MAX
CAPTURE_ROLES = base.CAPTURE_ROLES
PRIMARY_PROFILE_ROLES = base.PRIMARY_PROFILE_ROLES
PRE_QUERY_ROLES = base.PRE_QUERY_ROLES
ITEMS_PER_CELL_SPLIT = base.ITEMS_PER_CELL_SPLIT
GENERATION_UNITS_PER_FAMILY_SPLIT = base.GENERATION_UNITS_PER_FAMILY_SPLIT
GENERATION_STEPS = base.GENERATION_STEPS
ASSISTANT_PREFILL = base.ASSISTANT_PREFILL
SIGNED_PROJECTION_DIM = base.SIGNED_PROJECTION_DIM
SIGNED_PROJECTION_REPLICATES = base.SIGNED_PROJECTION_REPLICATES
SIGNED_PROJECTION_SEED = 1088001
SIGNED_FIELDS = ("active_binding", "field_null", "content")
OUT_ROOT = (
    ROOT / "tests" / "glm5" / "result"
    / "phase1088_answer_balanced_color_binding"
)
SOURCE_ROOT = base.OUT_ROOT


EVIDENCE_THRESHOLDS = {
    **base.EVIDENCE_THRESHOLDS,
    "minimum_numeric_models": 2,
}

PROSPECTIVE_PREDICTIONS = {
    "P1": (
        "Phase1087 static protocol and three-model behavior authorization are "
        "reused byte-for-state; only the registered hidden contrast changes."
    ),
    "P2": "Both answer-balanced signed sketches pass norm-distortion audits.",
    "P3": (
        "The answer-balanced binding-content field repeats across independent "
        "splits in three worlds and beats the binding null in two models."
    ),
    "P4": (
        "At least eight directed entity-world pairs repeat the answer-balanced "
        "field with a 0.10 null advantage in two models."
    ),
    "P5": (
        "The field transfers across surfaces and output words, with surface "
        "and output magnitudes no larger than content."
    ),
    "P6": (
        "Centered color-pair residuals retrieve at least six of eight pairs "
        "across independent samples in two models."
    ),
    "P7": (
        "Color-pair residuals transfer to three held-out entity worlds and "
        "beat the matched binding null."
    ),
    "P8": (
        "At least two directed healthy model pairs repeat color-pair Gram "
        "geometry and beat null geometry by 0.10."
    ),
    "P9": "At least two models pass all FP16 and finite-value audits.",
    "P10": (
        "A seven-pair centroid predicts a held-out eighth pair in at least "
        "24 of 32 pair-world cells in both sketches for two models."
    ),
}


write_json = base.write_json
write_jsonl = base.write_jsonl
read_json = base.read_json
read_jsonl = base.read_jsonl
digest = base.digest


def signed_pair_records(
    state_tensor,
    values,
    template: int,
    output_set: int,
):
    """Return answer-balanced binding, matched null, and interaction pairs."""
    active_left = 0.5 * (
        state_tensor(values, template, "active", 0, 0, output_set)
        + state_tensor(values, template, "active", 1, 0, output_set)
    )
    active_right = 0.5 * (
        state_tensor(values, template, "active", 0, 1, output_set)
        + state_tensor(values, template, "active", 1, 1, output_set)
    )
    null_left = 0.5 * (
        state_tensor(values, template, "field_null", 0, 0, output_set)
        + state_tensor(values, template, "field_null", 1, 0, output_set)
    )
    null_right = 0.5 * (
        state_tensor(values, template, "field_null", 0, 1, output_set)
        + state_tensor(values, template, "field_null", 1, 1, output_set)
    )
    return (
        ("active_binding", active_left, active_right, 0),
        ("field_null", null_left, null_right, 0),
        (
            "content",
            0.5 * (active_left + null_right),
            0.5 * (active_right + null_left),
            0,
        ),
    )


def main() -> None:
    protocol_root = OUT_ROOT / "protocol"
    model_case_digests = {}
    model_audits = {}
    for model_name in MODELS:
        source_cases = read_jsonl(
            SOURCE_ROOT / "protocol" / f"cases.{model_name}.jsonl"
        )
        cases = []
        for row in source_cases:
            current = dict(row)
            current["phase"] = PHASE
            current["schema_version"] = "phase1088_case.v1"
            current["source_phase1087_record_id"] = row["record_id"]
            cases.append(current)
        case_digest = digest(cases)
        source_audit = read_json(
            SOURCE_ROOT / "protocol" / f"audit.{model_name}.json"
        )
        audit = {
            "schema_version": "phase1088_protocol_model_audit.v1",
            "phase": PHASE,
            "model": model_name,
            "case_count": len(cases),
            "unit_count": source_audit["unit_count"],
            "checks": {
                **source_audit["checks"],
                "source_phase1087_audit_passed": bool(
                    source_audit["all_checks_passed"]
                ),
                "answer_balanced_builder_registered": True,
            },
            "all_checks_passed": bool(source_audit["all_checks_passed"]),
            "source_phase1087_case_digest": source_audit["case_digest"],
            "case_digest": case_digest,
        }
        write_jsonl(protocol_root / f"cases.{model_name}.jsonl", cases)
        write_json(protocol_root / f"audit.{model_name}.json", audit)
        model_case_digests[model_name] = case_digest
        model_audits[model_name] = audit

    source_prereg = read_json(
        SOURCE_ROOT / "protocol" / "preregistration.json"
    )
    source_final = read_json(
        SOURCE_ROOT / "analysis" / "final_summary.json"
    )
    prereg = {
        "schema_version": "phase1088_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "sequential_model_order": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "operations": list(OPERATIONS),
        "color_pairs": [list(pair) for pair in COLOR_PAIRS],
        "worlds": list(WORLDS),
        "splits": list(SPLITS),
        "panels": list(PANELS),
        "states": list(STATES),
        "capture_roles": list(CAPTURE_ROLES),
        "primary_profile_roles": list(PRIMARY_PROFILE_ROLES),
        "relative_depth_range": [
            TARGET_RELATIVE_DEPTH_MIN, TARGET_RELATIVE_DEPTH_MAX
        ],
        "signed_fields": list(SIGNED_FIELDS),
        "projection": {
            "type": "deterministic_rademacher",
            "dimension_per_replicate": SIGNED_PROJECTION_DIM,
            "replicates": SIGNED_PROJECTION_REPLICATES,
            "seed": SIGNED_PROJECTION_SEED,
            "cross_model_rule": (
                "Compare only within-model color-pair Gram geometry."
            ),
        },
        "contrast_definition": {
            "binding_left": "mean(active target0/binding0 true, active target1/binding0 false)",
            "binding_right": "mean(active target0/binding1 false, active target1/binding1 true)",
            "field_null": "the same binding averages while querying the unaffected anchor",
            "content": "active binding direction minus field-null binding direction",
        },
        "items_per_cell_split": ITEMS_PER_CELL_SPLIT,
        "case_count_per_model": source_prereg["case_count_per_model"],
        "unit_count_per_model": source_prereg["unit_count_per_model"],
        "evidence_thresholds": EVIDENCE_THRESHOLDS,
        "prospective_predictions": PROSPECTIVE_PREDICTIONS,
        "model_case_digests": model_case_digests,
        "source_phase1087_protocol_digest": source_prereg["protocol_digest"],
        "source_phase1087_summary_digest": source_final["summary_digest"],
        "interpretation_limits": [
            "Answer balancing removes a direct truth contrast but not every nonlinear truth-by-binding interaction.",
            "The field may be generic entity-binding transport rather than color semantics.",
            "The anchor null matches word movement but uses an always-false query.",
            "A repeated field is descriptive until all matched controls and held-out gates pass.",
            "No result establishes a neuron code, brain homology, optimality, or new mathematics.",
        ],
        "automatic_next": {
            "causal_authorization": "Only if P1-P10 all pass prospectively.",
            "otherwise": "Retain the map and stop before component or neuron selection.",
        },
        "model_audits": model_audits,
    }
    prereg["protocol_digest"] = digest(prereg)
    write_json(protocol_root / "preregistration.json", prereg)
    global_audit = {
        "schema_version": "phase1088_protocol_audit.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "source_phase1087_protocol_digest": source_prereg["protocol_digest"],
        "model_audits": model_audits,
        "all_checks_passed": all(
            row["all_checks_passed"] for row in model_audits.values()
        ),
    }
    global_audit["audit_digest"] = digest(global_audit)
    write_json(protocol_root / "audit.json", global_audit)

    source_authorization = read_json(
        SOURCE_ROOT / "analysis" / "behavior_authorization.json"
    )
    authorization = {
        "schema_version": "phase1088_behavior_authorization.v1",
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "source_phase1087_authorization_digest": source_authorization[
            "authorization_digest"
        ],
        "predictions": {
            "P1": {
                "passed": (
                    global_audit["all_checks_passed"]
                    and source_authorization["hidden_scan_authorized"]
                ),
                "reused_behavior_models": source_authorization[
                    "predictions"
                ]["P2"]["passing_models"],
            },
        },
        "models": source_authorization["models"],
        "hidden_scan_authorized": (
            global_audit["all_checks_passed"]
            and source_authorization["hidden_scan_authorized"]
        ),
        "full_atlas_authorized": False,
        "causal_authorized": False,
        "reason": (
            "The input states and behavior are unchanged from Phase1087; only "
            "the preregistered hidden-state contrast is new."
        ),
    }
    authorization["authorization_digest"] = digest(authorization)
    write_json(
        OUT_ROOT / "analysis" / "behavior_authorization.json",
        authorization,
    )
    print({
        "phase": PHASE,
        "case_count_per_model": prereg["case_count_per_model"],
        "all_checks_passed": global_audit["all_checks_passed"],
        "hidden_scan_authorized": authorization["hidden_scan_authorized"],
        "protocol_digest": prereg["protocol_digest"],
    })


if __name__ == "__main__":
    main()
