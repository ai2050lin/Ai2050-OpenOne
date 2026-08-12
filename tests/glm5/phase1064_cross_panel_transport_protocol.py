#!/usr/bin/env python3
"""Freeze cross-panel K/V transport after the Phase1063 behavior gate."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1040_expanded_mlp_replication_protocol as material
import phase1063_lexical_behavior_atlas_protocol as source


PHASE = 1064
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
    / "phase1064_cross_panel_transport"
)
PANELS = ("anchor_common", "novel_noun")
PAIR_FAMILIES = source.PAIR_FAMILIES
PAIR_LIMITS = {
    "anchor_common": 72,
    "novel_noun": 96,
}
CONTROL_PAIR_LIMITS = {
    "anchor_common": 48,
    "novel_noun": 64,
}
GATES = {
    "clean_replay_parity_min": 1.0,
    "phrase_post_text_rate_min": 0.50,
    "component_post_text_rate_min": 0.40,
    "source_minus_control_text_rate_min": 0.30,
    "minimum_repeated_models": 2,
}
CONDITION_BLUEPRINT = (
    "phrase_post_kv",
    "color_post_kv",
    "noun_post_kv",
    "phrase_early_kv",
    "phrase_all_kv",
    "phrase_post_k_only",
    "phrase_post_v_only",
    "phrase_late_half_kv",
    "phrase_late_quarter_kv",
    "phrase_frozen_rectangle",
    "operator_post_kv",
    "target_language_post_kv",
)


write_json = material.write_json
read_json = material.read_json
read_jsonl = material.read_jsonl
digest = material.digest


def main() -> None:
    source_prereg = read_json(
        SOURCE_ROOT / "protocol" / "preregistration.json"
    )
    source_audit = read_json(
        SOURCE_ROOT / "protocol" / "audit.json"
    )
    source_aggregate = read_json(SOURCE_ROOT / "aggregate.json")
    if not source_audit["all_checks_passed"]:
        raise RuntimeError("Phase1063 source audit failed")
    decision = source_aggregate["automatic_next_decision"]
    if not decision["should_continue_automatically"]:
        raise RuntimeError("Phase1063 did not authorize Phase1064")
    if decision["route"] != "continue_to_phase1064_cross_panel_transport":
        raise RuntimeError("unexpected Phase1063 route")
    source_models = set(source_aggregate["primary_passing_models"])
    if not {"qwen3", "glm4"}.issubset(source_models):
        raise RuntimeError("required cross-model behavior pool absent")
    for model_name in MODELS:
        summary = read_json(
            SOURCE_ROOT / "atlas" / model_name / "summary.json"
        )
        if summary["protocol_digest"] != source_prereg["protocol_digest"]:
            raise RuntimeError(
                f"source digest drift for {model_name}"
            )
        for panel in PANELS:
            for family in PAIR_FAMILIES:
                key = f"{panel}.{family}"
                if (
                    model_name in source_models
                    and summary["valid_pair_counts"][key]
                    < PAIR_LIMITS[panel]
                ):
                    raise RuntimeError(
                        f"insufficient frozen pairs for {model_name} {key}"
                    )

    payload = {
        "schema_version": "phase1064_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "sequential_model_order": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "generation_steps": source_prereg["generation_steps"],
        "panels": list(PANELS),
        "pair_families": list(PAIR_FAMILIES),
        "pair_limits": PAIR_LIMITS,
        "control_pair_limits": CONTROL_PAIR_LIMITS,
        "condition_blueprint": list(CONDITION_BLUEPRINT),
        "gates": GATES,
        "model_plans": source_prereg["model_plans"],
        "source_phase1063_digest": source_prereg["protocol_digest"],
        "source_phase1063_route": decision["route"],
        "source_behavior_passing_models": sorted(source_models),
        "primary_outcome": (
            "Whether the postsource K/V transport topology repeats on "
            "familiar nouns and forty lexically new nouns in at least two "
            "behavior-qualified models."
        ),
        "secondary_outcomes": [
            "Text and raw-token bidirectional transport rates.",
            "Early, postsource, and all-depth phase response.",
            "K-only, V-only, and joint K/V response.",
            "Late-half, late-quarter, and frozen-rectangle sufficiency.",
            "Operator and target-language role controls.",
        ],
        "interpretation_limits": [
            "The two panels are protocol strata, not corpus-frequency bins.",
            "A rate difference is not a lexical difficulty scaling law.",
            "Successful state replacement shows intervention sufficiency.",
            "Failure of selected cuts does not prove every omitted unit is necessary.",
            "K/V channel names do not establish address/content semantics.",
            "No result tests brain optimality or biological homology.",
        ],
        "automatic_next": {
            "stop_after": (
                "The three-model cross-panel replication and audit."
            ),
            "reason": (
                "A positive result is a lexical transport milestone; the "
                "next pattern family requires a separately frozen protocol."
            ),
        },
    }
    payload["protocol_digest"] = digest(payload)
    write_json(
        OUT_ROOT / "protocol" / "preregistration.json",
        payload,
    )
    audit = {
        "schema_version": "phase1064_protocol_audit.v1",
        "phase": PHASE,
        "source_audit_passed": True,
        "source_route_authorized": True,
        "source_behavior_passing_models": sorted(source_models),
        "condition_count_per_panel": len(CONDITION_BLUEPRINT),
        "condition_count_total": (
            len(PANELS) * len(CONDITION_BLUEPRINT)
        ),
        "all_pair_limits_supported": True,
        "all_checks_passed": True,
    }
    write_json(OUT_ROOT / "protocol" / "audit.json", audit)
    print(
        f"Phase{PHASE} protocol frozen: "
        f"{payload['protocol_digest']} conditions="
        f"{audit['condition_count_total']}"
    )


if __name__ == "__main__":
    main()
