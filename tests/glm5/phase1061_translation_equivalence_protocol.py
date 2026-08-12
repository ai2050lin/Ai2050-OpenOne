#!/usr/bin/env python3
"""Freeze multi-reference equivalence classes for the new phrase panel."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1040_expanded_mlp_replication_protocol as material
import phase1051_natural_behavior_protocol as behavior
import phase1060_lexicon_template_factorial_protocol as source


PHASE = 1061
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
    / "phase1061_translation_equivalence"
)
GENERATION_STEPS = 10
PAIR_LIMIT = 80
CONTROL_PAIR_LIMIT = 48
PAIR_FAMILIES = ("phrase", "color", "noun")
GATES = {
    "accepted_case_count_min": 80,
    "valid_pair_count_per_family_min": 50,
    "phrase_post_eos_exact_rate_min": 0.50,
    "component_post_eos_exact_rate_min": 0.35,
    "source_minus_control_rate_min": 0.30,
    "minimum_repeated_models": 2,
}


write_json = material.write_json
write_jsonl = material.write_jsonl
read_json = material.read_json
read_jsonl = material.read_jsonl
digest = material.digest


def acceptable_labels(row: dict[str, Any]) -> list[str]:
    noun = str(row["expected_label"]).split(" ", 1)[0]
    color = str(row["color_id"])
    gender = str(row["gender"])
    if color == "brown":
        adjectives = ["marron", "brun" if gender == "m" else "brune"]
    elif color == "beige":
        adjectives = ["beige"]
    elif color == "golden":
        adjectives = [
            "doré" if gender == "m" else "dorée",
            "d'or",
            "en or",
        ]
    elif color == "silver":
        adjectives = [
            "argenté" if gender == "m" else "argentée",
            "d'argent",
            "en argent",
        ]
    else:
        raise RuntimeError(f"unexpected Phase1061 color: {color}")
    return list(dict.fromkeys(f"{noun} {value}" for value in adjectives))


def main() -> None:
    source_prereg = read_json(
        SOURCE_ROOT / "protocol" / "preregistration.json"
    )
    source_aggregate = read_json(SOURCE_ROOT / "aggregate.json")
    if source_aggregate["automatic_next_decision"][
        "should_continue_automatically"
    ]:
        raise RuntimeError("Phase1060 unexpectedly authorized sentences")
    model_plans = {}
    model_audits = {}
    for model_name in MODELS:
        tokenizer = source.tokenizer_for(model_name)
        source_cases = read_jsonl(
            SOURCE_ROOT / "protocol" / f"cases.{model_name}.jsonl"
        )
        source_targets = read_jsonl(
            SOURCE_ROOT / "protocol" / f"targets.{model_name}.jsonl"
        )
        cases = []
        for source_row in source_cases:
            if source_row["cell"] != "new_old":
                continue
            row = dict(source_row)
            labels = acceptable_labels(row)
            row["schema_version"] = "phase1061_model_case.v1"
            row["phase"] = PHASE
            row["acceptable_labels"] = labels
            row["acceptable_token_ids"] = [
                behavior.continuation_ids(
                    tokenizer,
                    str(row["rendered_prompt"]),
                    source.CONTINUATION_PREFIX,
                    label,
                )
                for label in labels
            ]
            cases.append(row)
        targets = []
        for source_row in source_targets:
            if source_row["cell"] != "new_old":
                continue
            row = dict(source_row)
            row["schema_version"] = "phase1061_target.v1"
            row["phase"] = PHASE
            targets.append(row)
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
        color_alternative_counts = {
            color: sorted({
                len(row["acceptable_labels"])
                for row in cases
                if row["color_id"] == color
            })
            for color in ("brown", "beige", "golden", "silver")
        }
        invalid_targets = [
            int(row["target_index"])
            for row in targets
            if int(row["target_case_index"]) not in case_ids
            or int(row["cross_case_index"]) not in case_ids
        ]
        model_audits[model_name] = {
            "case_count": len(cases),
            "target_counts": dict(target_counts),
            "color_alternative_counts": color_alternative_counts,
            "invalid_target_count": len(invalid_targets),
            "empty_alternative_count": sum(
                not row["acceptable_token_ids"] for row in cases
            ),
        }
    audit = {
        "schema_version": "phase1061_protocol_audit.v1",
        "phase": PHASE,
        "equivalence_rules": {
            "brown": ["marron", "brun/brune"],
            "beige": ["beige"],
            "golden": ["doré/dorée", "d'or", "en or"],
            "silver": ["argenté/argentée", "d'argent", "en argent"],
        },
        "models": model_audits,
    }
    audit["all_checks_passed"] = all(
        row["case_count"] == 112
        and row["invalid_target_count"] == 0
        and row["empty_alternative_count"] == 0
        and all(
            row["target_counts"].get(family, 0) == 112
            for family in PAIR_FAMILIES
        )
        and row["color_alternative_counts"]["brown"] == [2]
        and row["color_alternative_counts"]["beige"] == [1]
        and row["color_alternative_counts"]["golden"] == [3]
        and row["color_alternative_counts"]["silver"] == [3]
        for row in model_audits.values()
    )
    write_json(OUT_ROOT / "protocol" / "audit.json", audit)
    if not audit["all_checks_passed"]:
        raise RuntimeError(f"Phase1061 protocol audit failed: {audit}")
    payload = {
        "schema_version": "phase1061_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "source_phase1060_digest": source_prereg["protocol_digest"],
        "source_phase1060_route": source_aggregate[
            "automatic_next_decision"
        ],
        "authorization": (
            "Automatic behavior-equivalence audit requested by the user "
            "after the Phase1060 lexical-panel diagnosis."
        ),
        "models": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "sequential_model_order": list(MODELS),
        "cell": "new_old",
        "equivalence_rules": audit["equivalence_rules"],
        "generation_steps": GENERATION_STEPS,
        "pair_limit": PAIR_LIMIT,
        "control_pair_limit": CONTROL_PAIR_LIMIT,
        "pair_families": list(PAIR_FAMILIES),
        "model_plans": model_plans,
        "gates": GATES,
        "primary_outcome": (
            "Patched sequence must exactly equal the opposite arm's clean "
            "EOS-censored sequence. Multi-reference labels only determine "
            "whether a clean arm is behavior-qualified."
        ),
        "automatic_next": {
            "if_two_models_repeat": "phase1062_sentence_role_transport",
            "otherwise": "stop_with_unresolved_translation_equivalence",
        },
        "interpretation_limits": [
            "The equivalence list is finite and may omit legitimate French.",
            "Multi-reference qualification does not relax donor-clean matching.",
            "Lexical alternatives are behavior labels, not latent variables.",
            "K/V replacement remains a sufficient intervention, not unique.",
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
