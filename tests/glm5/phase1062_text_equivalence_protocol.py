#!/usr/bin/env python3
"""Freeze decoded-text equivalence for translation behavior and transport."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1040_expanded_mlp_replication_protocol as material
import phase1061_translation_equivalence_protocol as source


PHASE = 1062
PROTOCOL_REVISION = 1
MODELS = source.MODELS
PRECISION = "fp16"
QUANTIZATION = "none"
SOURCE_ROOT = source.OUT_ROOT
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1062_text_equivalence"
)
GENERATION_STEPS = source.GENERATION_STEPS
PAIR_LIMIT = source.PAIR_LIMIT
CONTROL_PAIR_LIMIT = source.CONTROL_PAIR_LIMIT
PAIR_FAMILIES = source.PAIR_FAMILIES
GATES = {
    "accepted_case_count_min": 80,
    "valid_pair_count_per_family_min": 50,
    "phrase_post_text_exact_rate_min": 0.50,
    "component_post_text_exact_rate_min": 0.35,
    "source_minus_control_text_rate_min": 0.30,
    "minimum_repeated_models": 2,
}


write_json = material.write_json
write_jsonl = material.write_jsonl
read_json = material.read_json
read_jsonl = material.read_jsonl
digest = material.digest


def main() -> None:
    source_prereg = read_json(
        SOURCE_ROOT / "protocol" / "preregistration.json"
    )
    source_aggregate = read_json(SOURCE_ROOT / "aggregate.json")
    if source_aggregate["automatic_next_decision"][
        "should_continue_automatically"
    ]:
        raise RuntimeError("Phase1061 unexpectedly authorized sentences")
    model_plans = {}
    model_audits = {}
    for model_name in MODELS:
        cases = read_jsonl(
            SOURCE_ROOT / "protocol" / f"cases.{model_name}.jsonl"
        )
        targets = read_jsonl(
            SOURCE_ROOT / "protocol" / f"targets.{model_name}.jsonl"
        )
        for row in cases:
            row["schema_version"] = "phase1062_model_case.v1"
            row["phase"] = PHASE
        for row in targets:
            row["schema_version"] = "phase1062_target.v1"
            row["phase"] = PHASE
        write_jsonl(
            OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl",
            cases,
        )
        write_jsonl(
            OUT_ROOT / "protocol" / f"targets.{model_name}.jsonl",
            targets,
        )
        model_plans[model_name] = source_prereg["model_plans"][model_name]
        case_ids = {int(row["semantic_case_index"]) for row in cases}
        target_counts = Counter(
            str(row["pair_family"]) for row in targets
        )
        invalid_targets = [
            int(row["target_index"])
            for row in targets
            if int(row["target_case_index"]) not in case_ids
            or int(row["cross_case_index"]) not in case_ids
        ]
        model_audits[model_name] = {
            "case_count": len(cases),
            "target_counts": dict(target_counts),
            "invalid_target_count": len(invalid_targets),
            "empty_label_count": sum(
                not row["acceptable_labels"] for row in cases
            ),
        }
    audit = {
        "schema_version": "phase1062_protocol_audit.v1",
        "phase": PHASE,
        "normalization": {
            "unicode": "NFC",
            "outer_whitespace": "strip",
            "internal_whitespace": "collapse",
            "case": "casefold",
            "punctuation": "preserve",
            "edit_distance": "disabled",
            "unregistered_synonyms": "reject",
        },
        "models": model_audits,
    }
    audit["all_checks_passed"] = all(
        row["case_count"] == 112
        and row["invalid_target_count"] == 0
        and row["empty_label_count"] == 0
        and all(
            row["target_counts"].get(family, 0) == 112
            for family in PAIR_FAMILIES
        )
        for row in model_audits.values()
    )
    write_json(OUT_ROOT / "protocol" / "audit.json", audit)
    if not audit["all_checks_passed"]:
        raise RuntimeError(f"Phase1062 protocol audit failed: {audit}")
    payload = {
        "schema_version": "phase1062_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "source_phase1061_digest": source_prereg["protocol_digest"],
        "source_phase1061_route": source_aggregate[
            "automatic_next_decision"
        ],
        "authorization": (
            "Automatic instrument repair after observing identical decoded "
            "text with non-identical token segmentations in Phase1061."
        ),
        "models": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "sequential_model_order": list(MODELS),
        "normalization": audit["normalization"],
        "generation_steps": GENERATION_STEPS,
        "pair_limit": PAIR_LIMIT,
        "control_pair_limit": CONTROL_PAIR_LIMIT,
        "pair_families": list(PAIR_FAMILIES),
        "model_plans": model_plans,
        "gates": GATES,
        "primary_outcome": (
            "Both patched arms must equal the opposite clean arm after the "
            "frozen decoded-text normalization."
        ),
        "secondary_outcome": (
            "Both patched arms exactly match the opposite raw clean token "
            "sequence through EOS."
        ),
        "automatic_next": {
            "if_two_models_repeat": "phase1063_sentence_role_transport",
            "otherwise": "stop_with_real_lexical_behavior_limit",
        },
        "interpretation_limits": [
            "Text normalization repairs representation identity, not semantics.",
            "The finite synonym list remains incomplete.",
            "Punctuation and word order are not relaxed.",
            "Raw token-sequence matching remains separately reported.",
            "K/V replacement is sufficient under intervention, not unique.",
            "No result tests biological optimality or brain homology.",
        ],
    }
    payload["protocol_digest"] = digest(payload)
    write_json(
        OUT_ROOT / "protocol" / "preregistration.json",
        payload,
    )
    print(
        f"Phase{PHASE} protocol frozen: {payload['protocol_digest']} "
        f"models={len(MODELS)} cases_per_model=112"
    )


if __name__ == "__main__":
    main()
