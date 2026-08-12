#!/usr/bin/env python3
"""Freeze an independent, larger-name-world replication of Phase1101 Revision2."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

from phase1018_language_pattern_protocol import tokenizer_for
import phase1101_relation_identity_routing_protocol as base


PHASE = 1102
PROTOCOL_REVISION = 1
OUT_ROOT = ROOT / "tests" / "glm5" / "result" / "phase1102_relation_identity_routing_replication"
SOURCE_PHASE1101_ROOT = base.OUT_ROOT
SOURCE_PHASE1101_AUTHORIZATION = SOURCE_PHASE1101_ROOT / "analysis" / "behavior_authorization.json"
PHASE1099_PREREG = base.phase1099.OUT_ROOT / "protocol" / "preregistration.json"
ITEMS_PER_TEMPLATE = 4

# Re-export the frozen design. The only intentional changes are phase/output,
# item count, and a disjoint name world selected before any Phase1102 result.
MODELS = base.MODELS
FORMAL_MODELS = base.FORMAL_MODELS
PRECISION = base.PRECISION
QUANTIZATION = base.QUANTIZATION
SURFACES = base.SURFACES
TEMPLATES = base.TEMPLATES
TEMPLATES_BY_SPLIT = base.TEMPLATES_BY_SPLIT
SPLITS = base.SPLITS
ROUTE_TYPES = base.ROUTE_TYPES
CONGRUENCES = base.CONGRUENCES
TARGET_RELATIONS = base.TARGET_RELATIONS
RELATION_ORDERS = base.RELATION_ORDERS
ORIENTATIONS = base.ORIENTATIONS
ASSISTANT_PREFILL = base.ASSISTANT_PREFILL
CONTINUATION_PREFIX = base.CONTINUATION_PREFIX
GENERATION_STEPS = base.GENERATION_STEPS
GENERATION_ITEMS_PER_CELL = base.GENERATION_ITEMS_PER_CELL
CAPTURE_ROLES = base.CAPTURE_ROLES
FIELDS = base.FIELDS
PRIMARY_FIELD = base.PRIMARY_FIELD
MATCHED_CONTROLS = base.MATCHED_CONTROLS
PRIMARY_ROLE = base.PRIMARY_ROLE
DEPTH_FRACTIONS = base.DEPTH_FRACTIONS
COMPONENTS = base.COMPONENTS
SOURCE_ROOT = base.SOURCE_ROOT
SOURCE_PHASE1100 = base.SOURCE_PHASE1100
SOURCE_PHASE1100_AUDIT = base.SOURCE_PHASE1100_AUDIT
RELATION_ROWS = base.RELATION_ROWS
RELATIONS = base.RELATIONS
RELATION_FAMILY = base.RELATION_FAMILY
RELATION_LABELS = base.RELATION_LABELS
RELATION_PAIRS = base.RELATION_PAIRS
PAIR_RELATIONS = base.PAIR_RELATIONS
PAIR_FAMILY = base.PAIR_FAMILY
FAMILIES = base.FAMILIES
STATES = base.STATES
THRESHOLDS = base.THRESHOLDS
PROSPECTIVE_PREDICTIONS = dict(base.PROSPECTIVE_PREDICTIONS)
SHELLS = base.SHELLS
ORDINAL_SELECTORS = base.ORDINAL_SELECTORS
write_json = base.write_json
write_jsonl = base.write_jsonl
read_json = base.read_json
read_jsonl = base.read_jsonl
digest = base.digest
sha256_text = base.sha256_text
state_name = base.state_name
state_factors = base.state_factors
split_for_template = base.split_for_template
render_prompt = base.render_prompt


def configure_base() -> None:
    base.PHASE = PHASE
    base.PROTOCOL_REVISION = PROTOCOL_REVISION
    base.OUT_ROOT = OUT_ROOT
    base.ITEMS_PER_TEMPLATE = ITEMS_PER_TEMPLATE


configure_base()


def selected_names(tokenizers: dict[str, Any]) -> tuple[str, ...]:
    if not PHASE1099_PREREG.exists():
        raise RuntimeError("Phase1099 frozen name world is missing")
    names = tuple(read_json(PHASE1099_PREREG)["selected_names"])
    if len(names) != len(TEMPLATES) * ITEMS_PER_TEMPLATE * 2:
        raise RuntimeError("Phase1099 name world does not contain 32 labels")
    phase1101_names = set(
        read_json(SOURCE_PHASE1101_ROOT / "protocol" / "preregistration.json")[
            "selected_names"
        ]
    )
    if phase1101_names & set(names):
        raise RuntimeError("Phase1102 name world overlaps Phase1101")
    for model, tokenizer in tokenizers.items():
        ids = [tokenizer.encode(" " + name, add_special_tokens=False) for name in names]
        if any(len(values) != 1 for values in ids):
            raise RuntimeError(f"Phase1102 name tokenization drift for {model}")
        if len({int(values[0]) for values in ids}) != len(ids):
            raise RuntimeError(f"Phase1102 name token collision for {model}")
    return names


def build_model_cases(tokenizer, model_name: str, names: tuple[str, ...]) -> list[dict[str, Any]]:
    configure_base()
    rows = base.build_model_cases(tokenizer, model_name, names)
    for row in rows:
        row["schema_version"] = "phase1102_relation_identity_routing_replication_case.v1"
        row["phase"] = PHASE
        for key in ("record_id", "unit_id", "superunit_id"):
            row[key] = str(row[key]).replace("phase1101.", "phase1102.", 1)
    return rows


def audit_model(model_name: str, rows: list[dict[str, Any]], names: tuple[str, ...]) -> dict[str, Any]:
    configure_base()
    original_old_names = base.old_names
    try:
        base.old_names = lambda: set()
        audit = base.audit_model(model_name, rows, names)
    finally:
        base.old_names = original_old_names
    phase1101_names = set(
        read_json(SOURCE_PHASE1101_ROOT / "protocol" / "preregistration.json")[
            "selected_names"
        ]
    )
    audit["schema_version"] = "phase1102_protocol_model_audit.v1"
    audit["checks"]["name_world_disjoint_from_phase1101"] = not (
        phase1101_names & set(names)
    )
    audit["checks"]["larger_sample_count"] = ITEMS_PER_TEMPLATE == 4
    audit["all_checks_passed"] = all(audit["checks"].values())
    audit["case_digest"] = digest(rows)
    return audit


def main() -> None:
    configure_base()
    if not SOURCE_PHASE1101_AUTHORIZATION.exists():
        raise RuntimeError("Phase1101 Revision2 behavior authorization is missing")
    source_authorization = read_json(SOURCE_PHASE1101_AUTHORIZATION)
    if source_authorization["hidden_scan_authorized"]:
        raise RuntimeError("Phase1102 replication is only allowed after Phase1101 behavior stop")
    tokenizers = {model: tokenizer_for(model) for model in MODELS}
    names = selected_names(tokenizers)
    model_digests = {}
    model_audits = {}
    for model in MODELS:
        rows = build_model_cases(tokenizers[model], model, names)
        audit = audit_model(model, rows, names)
        if not audit["all_checks_passed"]:
            raise RuntimeError(f"Phase1102 protocol audit failed for {model}: {audit}")
        write_jsonl(OUT_ROOT / "protocol" / f"cases.{model}.jsonl", rows)
        write_json(OUT_ROOT / "protocol" / f"audit.{model}.json", audit)
        model_digests[model] = audit["case_digest"]
        model_audits[model] = audit
    prereg = {
        "schema_version": "phase1102_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "formal_models": list(FORMAL_MODELS),
        "sequential_model_order": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "families": list(FAMILIES),
        "relations": list(RELATIONS),
        "relation_pairs": list(RELATION_PAIRS),
        "pair_relations": {key: list(value) for key, value in PAIR_RELATIONS.items()},
        "pair_family": PAIR_FAMILY,
        "surfaces": list(SURFACES),
        "templates": list(TEMPLATES),
        "templates_by_split": {key: list(value) for key, value in TEMPLATES_BY_SPLIT.items()},
        "items_per_template": ITEMS_PER_TEMPLATE,
        "states": list(STATES),
        "selected_names": list(names),
        "case_count_per_model": len(read_jsonl(OUT_ROOT / "protocol" / f"cases.{MODELS[0]}.jsonl")),
        "generation_steps": GENERATION_STEPS,
        "generation_items_per_cell": GENERATION_ITEMS_PER_CELL,
        "capture_roles": list(CAPTURE_ROLES),
        "fields": list(FIELDS),
        "primary_field": PRIMARY_FIELD,
        "matched_controls": list(MATCHED_CONTROLS),
        "primary_role": PRIMARY_ROLE,
        "sampled_event_grid": {
            "components": list(COMPONENTS),
            "relative_depths": list(DEPTH_FRACTIONS),
        },
        "primary_object": "Independent larger-name-world replication of the frozen Phase1101 Revision2 behavior-necessary relation-address routing task.",
        "replication_constraints": {
            "prompt_text_unchanged": True,
            "thresholds_unchanged": True,
            "relation_pairs_unchanged": True,
            "factorial_states_unchanged": True,
            "name_world_disjoint_from_phase1101": True,
            "items_per_template_phase1101": 3,
            "items_per_template_phase1102": ITEMS_PER_TEMPLATE,
            "further_behavior_revision_authorized": False,
        },
        "behavioral_necessity": "Conflict states require selecting the late named relation because the two records have opposite winners; congruent states and ordinal routing remain matched controls.",
        "source_object": "Phase1100 input-query-polarity pair differences, r1 minus r0, for the same 15 relation pairs, used only if behavior authorizes hidden scanning.",
        "forbidden_primary_inputs": [
            "candidate logits", "output margins", "generation scores", "PCA",
            "learned probes", "post-hoc components", "post-hoc roles",
        ],
        "evidence_thresholds": THRESHOLDS,
        "prospective_predictions": PROSPECTIVE_PREDICTIONS,
        "automatic_next_rule": "Only the unchanged Phase1101 behavior gates authorize Phase1102 hidden scanning; only P1-P7 then authorize causality.",
        "source_phase1101_authorization_digest": source_authorization[
            "authorization_digest"
        ],
        "model_case_digests": model_digests,
        "model_audits": model_audits,
    }
    prereg["protocol_digest"] = digest(prereg)
    write_json(OUT_ROOT / "protocol" / "preregistration.json", prereg)
    common_audit = {
        "schema_version": "phase1102_protocol_audit.v1",
        "phase": PHASE,
        "checks": {
            "all_model_audits_pass": all(
                row["all_checks_passed"] for row in model_audits.values()
            ),
            "phase1101_hidden_scan_stopped": not source_authorization[
                "hidden_scan_authorized"
            ],
            "same_prompts_and_thresholds": SHELLS == base.SHELLS
            and THRESHOLDS == base.THRESHOLDS,
            "independent_name_world": not (
                set(names)
                & set(read_json(SOURCE_PHASE1101_ROOT / "protocol" / "preregistration.json")["selected_names"])
            ),
            "larger_sample": ITEMS_PER_TEMPLATE == 4,
            "fp16_no_quantization": PRECISION == "fp16"
            and QUANTIZATION == "none",
        },
        "model_case_digests": model_digests,
        "protocol_digest": prereg["protocol_digest"],
    }
    common_audit["all_checks_passed"] = all(common_audit["checks"].values())
    common_audit["audit_digest"] = digest(common_audit)
    write_json(OUT_ROOT / "protocol" / "audit.json", common_audit)
    print(json.dumps({
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "audit_digest": common_audit["audit_digest"],
        "case_count_per_model": prereg["case_count_per_model"],
        "selected_names": names,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
