#!/usr/bin/env python3
"""Freeze the independent Phase1033 source/query alliance replication."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1032_span_alliance_protocol as base
from phase1018_language_pattern_protocol import tokenizer_for


PHASE = 1033
PROTOCOL_REVISION = 1
MODELS = base.MODELS
WORLD_CODES = base.WORLD_CODES
DONOR_OFFSETS = base.DONOR_OFFSETS
SPAN_ROLES = base.SPAN_ROLES
SELECTED_DEPTHS = base.SELECTED_DEPTHS
CONDITIONS = base.CONDITIONS
CATEGORY_LABELS = base.CATEGORY_LABELS

OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1033_alliance_independent_replication"
)

CONCEPT_BANKS = {
    "single": (
        ("pear", "fruit"),
        ("tiger", "animal"),
        ("train", "vehicle"),
        ("doctor", "job"),
        ("beach", "place"),
        ("lamp", "object"),
        ("red", "color"),
        ("foot", "body"),
    ),
    "double": (
        ("yellow pear", "fruit"),
        ("white tiger", "animal"),
        ("fast train", "vehicle"),
        ("family doctor", "job"),
        ("sandy beach", "place"),
        ("desk lamp", "object"),
        ("dark red", "color"),
        ("right foot", "body"),
    ),
}

NONCE_PAIRS = (
    ("arvik", "benor"),
    ("calen", "elvan"),
    ("feron", "gavin"),
    ("helor", "invar"),
    ("kavel", "lemor"),
    ("norik", "oprel"),
    ("peron", "quvin"),
    ("ravel", "sevik"),
)

TEMPLATES = (
    (
        'Codebook facts: "{nonce_a}" refers to {concept_a}, whereas '
        '"{nonce_b}" refers to {concept_b}. State the broad category '
        'of "{query_nonce}":'
    ),
    (
        'Read the mapping. Assign {concept_a} to symbol "{nonce_a}" '
        'and {concept_b} to symbol "{nonce_b}". The symbol '
        '"{query_nonce}" belongs to which general kind? Reply:'
    ),
)

QUERY_STARTS = (
    "State the broad category",
    "The symbol",
)

canonical = base.canonical
digest = base.digest
write_json = base.write_json
write_jsonl = base.write_jsonl
read_json = base.read_json
read_jsonl = base.read_jsonl


def configure_base() -> None:
    """Point the audited Phase1032 generator at the frozen replication data."""
    base.PHASE = PHASE
    base.PROTOCOL_REVISION = PROTOCOL_REVISION
    base.MODELS = MODELS
    base.WORLD_CODES = WORLD_CODES
    base.DONOR_OFFSETS = DONOR_OFFSETS
    base.SPAN_ROLES = SPAN_ROLES
    base.SELECTED_DEPTHS = SELECTED_DEPTHS
    base.CONDITIONS = CONDITIONS
    base.CATEGORY_LABELS = CATEGORY_LABELS
    base.CONCEPT_BANKS = CONCEPT_BANKS
    base.NONCE_PAIRS = NONCE_PAIRS
    base.TEMPLATES = TEMPLATES
    base.QUERY_STARTS = QUERY_STARTS
    base.OUT_ROOT = OUT_ROOT


def phase_schema(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: phase_schema(item) for key, item in value.items()
        }
    if isinstance(value, list):
        return [phase_schema(item) for item in value]
    if isinstance(value, str):
        return value.replace("phase1032", "phase1033")
    return value


configure_base()


def main() -> None:
    configure_base()
    units, common_cases = base.build_units_and_cases()
    units = phase_schema(units)
    common_cases = phase_schema(common_cases)
    common = phase_schema(base.common_audit(units, common_cases))
    if not common["all_checks_passed"]:
        raise RuntimeError(f"common audit failed: {common}")

    protocol_dir = OUT_ROOT / "protocol"
    write_jsonl(protocol_dir / "units.jsonl", units)
    write_jsonl(protocol_dir / "cases.common.jsonl", common_cases)
    write_json(protocol_dir / "common_audit.json", common)

    model_audits = {}
    for model_name in MODELS:
        tokenizer = tokenizer_for(model_name)
        rows = [
            phase_schema(base.model_case(
                tokenizer, model_name, row
            ))
            for row in common_cases
        ]
        audit = phase_schema(
            base.model_audit(model_name, units, rows)
        )
        if not audit["all_checks_passed"]:
            raise RuntimeError(
                f"{model_name} tokenization audit failed: {audit}"
            )
        write_jsonl(protocol_dir / f"cases.{model_name}.jsonl", rows)
        write_json(protocol_dir / f"audit.{model_name}.json", audit)
        model_audits[model_name] = audit
        del tokenizer

    prereg_core = {
        "schema_version": "phase1033_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "title": (
            "Independent replication of the span-aware conditional "
            "source/query alliance"
        ),
        "models": list(MODELS),
        "precision": "fp16",
        "quantization": "none",
        "sequential_model_order": list(MODELS),
        "unit_count": len(units),
        "case_count": len(common_cases),
        "world_codes": list(WORLD_CODES),
        "conditions": list(CONDITIONS),
        "selected_depths_frozen_from_phase1029": SELECTED_DEPTHS,
        "independence_from_phase1032": {
            "new_concept_surfaces": True,
            "new_nonce_surfaces": True,
            "new_templates": True,
            "same_category_roles": True,
            "same_unit_count": True,
            "same_span_balance": True,
            "same_depths": True,
            "same_conditions": True,
            "same_thresholds": True,
        },
        "balanced_design": {
            "templates": len(TEMPLATES),
            "nonce_pairs": len(NONCE_PAIRS),
            "span_banks": {
                "single": "exactly one concept token",
                "double": "exactly two concept tokens",
            },
            "units_per_bank": 256,
            "units_per_template": 256,
            "candidate_categories": list(CATEGORY_LABELS),
        },
        "span_rules": {
            "primary": (
                "Patch every token one-to-one in complete equal-length "
                "spans; no pooling, truncation, or posthoc alignment."
            ),
            "endpoint_comparator": (
                "Patch only the final token on the same units."
            ),
            "pre_output_role": (
                "The generation boundary remains a receiver/readout and "
                "is not patched as part of the upstream alliance."
            ),
        },
        "primary_readout": (
            "Within-template leave-surface concept prototypes."
        ),
        "secondary_readout": (
            "Eight preregistered one-token category logits, reported only "
            "on rows where all candidate logits are finite."
        ),
        "replication_target": {
            "selected_source": (
                "Selected full source span exceeds the unselected source "
                "by at least 0.30 in both templates."
            ),
            "composition": (
                "Source-pair plus query span raises base Top1 by at least "
                "0.10 over both constituent patches in both templates."
            ),
            "cross_model": (
                "The directional rule must hold in at least two models."
            ),
        },
        "claim_limit": (
            "A repeated result supports a local sufficient state alliance "
            "for this artificial two-binding retrieval pattern. It does "
            "not identify the natural attention/MLP route, establish "
            "next-token closure, generalize to all knowledge or grammar, "
            "or validate brain/LLM optimality hypotheses."
        ),
    }
    prereg = dict(prereg_core)
    prereg["protocol_digest"] = digest(prereg_core)
    prereg["common_audit"] = common
    prereg["model_tokenization_audits"] = model_audits
    write_json(protocol_dir / "preregistration.json", prereg)
    print(json.dumps({
        "phase": PHASE,
        "protocol_digest": prereg["protocol_digest"],
        "unit_count": len(units),
        "case_count": len(common_cases),
        "common_audit": common,
        "model_audits": model_audits,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
