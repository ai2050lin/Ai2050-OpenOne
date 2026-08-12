#!/usr/bin/env python3
"""Freeze a natural English-to-French K/V transfer protocol."""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1018_language_pattern_protocol import tokenizer_for
from phase1021_natural_language_atlas_protocol import offset_token_spans
import phase1022_ability_family_protocol as lexical
import phase1040_expanded_mlp_replication_protocol as material
import phase1051_natural_behavior_protocol as behavior
import phase1052_full_vocab_kv_bridge_protocol as bridge
import phase1054_joint_kv_rollout_protocol as source


PHASE = 1055
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
    / "phase1055_pattern_family_transfer"
)
SURFACES = (
    {
        "template": (
            "Translate the English word below into French. Return exactly "
            "one French word and nothing else.\nEnglish word: {term}"
        ),
        "operator": "Translate",
        "target_language": "French",
    },
    {
        "template": (
            "Convert this English lexical item to French. Give only the "
            "French equivalent.\nSource item: {term}"
        ),
        "operator": "Convert",
        "target_language": "French",
    },
)
VARIANTS = (
    {
        "variant": "raw_translation",
        "render_mode": "raw",
        "assistant_prefill": "\nFrench translation:",
        "continuation_prefix": " ",
    },
    {
        "variant": "chat_plain",
        "render_mode": "native_chat",
        "assistant_prefill": "",
        "continuation_prefix": "",
    },
    {
        "variant": "chat_translation_prefill",
        "render_mode": "native_chat",
        "assistant_prefill": "Translation:",
        "continuation_prefix": " ",
    },
)
VARIANT_ORDER = tuple(row["variant"] for row in VARIANTS)
PAIR_OFFSET = 1
ROLLOUT_STEPS = 8
ROLLOUT_PAIR_LIMIT = 48
CONDITION_ORDER = (
    "source_fact_rectangle",
    "source_all_groups_postsource",
    "source_all_groups_all_layers",
    "operator_all_groups_all_layers",
    "target_language_all_groups_all_layers",
)
GATES = {
    "behavior_correct_pair_count_min": 40,
    "behavior_pair_accuracy_min": 0.60,
    "behavior_concept_coverage_min": 20,
    "broad_both_counterfactual_rate_min": 0.30,
    "broad_both_counterfactual_count_min": 20,
    "source_minus_control_rate_min": 0.20,
    "fact_rectangle_rate_min": 0.20,
    "fact_rectangle_retained_fraction_min": 0.50,
    "rollout_pair_count_min": 20,
    "eos_censored_both_match_rate_min": 0.50,
    "minimum_repeated_models": 2,
}


write_json = material.write_json
write_jsonl = material.write_jsonl
read_json = material.read_json
read_jsonl = material.read_jsonl
digest = material.digest


def fragment(
    text: str,
    value: str,
    *,
    occurrence: str,
) -> tuple[int, int, str]:
    if occurrence == "first":
        start = text.find(value)
    elif occurrence == "last":
        start = text.rfind(value)
    else:
        raise ValueError(occurrence)
    if start < 0:
        raise RuntimeError(f"missing fragment {value!r}")
    return start, start + len(value), value


def concept_rows(split: str) -> list[dict[str, Any]]:
    return [
        row for row in lexical.CONCEPTS
        if row["split"] == split
    ]


def build_model_cases(
    model_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tokenizer = tokenizer_for(model_name)
    cases = []
    for split in ("discovery", "confirmation"):
        for surface_index, surface in enumerate(SURFACES):
            for concept in concept_rows(split):
                source_term = str(concept["terms"]["en"][0])
                target_term = str(concept["terms"]["fr"][0])
                content = surface["template"].format(term=source_term)
                fragments = {
                    "source_term": fragment(
                        content, source_term, occurrence="last"
                    ),
                    "operator": fragment(
                        content,
                        str(surface["operator"]),
                        occurrence="first",
                    ),
                    "target_language": fragment(
                        content,
                        str(surface["target_language"]),
                        occurrence="first",
                    ),
                }
                for variant in VARIANTS:
                    if variant["render_mode"] == "raw":
                        rendered = content
                    else:
                        rendered = behavior.render_native(
                            tokenizer,
                            model_name,
                            content,
                            with_system=False,
                        )
                    rendered += str(variant["assistant_prefill"])
                    input_ids = [
                        int(value) for value in tokenizer.encode(
                            rendered, add_special_tokens=False
                        )
                    ]
                    role_spans = {
                        role: [int(start), int(end)]
                        for role, (start, end) in offset_token_spans(
                            tokenizer,
                            rendered,
                            content,
                            fragments,
                        ).items()
                    }
                    role_spans["selected_concept"] = list(
                        role_spans["source_term"]
                    )
                    target_ids = behavior.continuation_ids(
                        tokenizer,
                        rendered,
                        str(variant["continuation_prefix"]),
                        target_term,
                    )
                    cases.append({
                        "schema_version": "phase1055_model_case.v1",
                        "phase": PHASE,
                        "model": model_name,
                        "semantic_case_index": len(cases),
                        "case_key": (
                            f"{split}.s{surface_index}."
                            f"{concept['concept_id']}."
                            f"{variant['variant']}"
                        ),
                        "split": split,
                        "surface_index": surface_index,
                        "variant": variant["variant"],
                        "concept_id": str(concept["concept_id"]),
                        "category": str(concept["category"]),
                        "source_term": source_term,
                        "expected_label": target_term,
                        "rendered_prompt": rendered,
                        "input_ids": input_ids,
                        "role_spans": role_spans,
                        "expected_token_ids": target_ids,
                        "expected_first_token_id": int(target_ids[0]),
                    })

    targets = []
    by_panel: dict[
        tuple[str, int, str, int],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in cases:
        width = (
            int(row["role_spans"]["source_term"][1])
            - int(row["role_spans"]["source_term"][0])
            + 1
        )
        by_panel[
            (
                str(row["split"]),
                int(row["surface_index"]),
                str(row["variant"]),
                width,
            )
        ].append(row)
    for panel, rows in sorted(by_panel.items()):
        rows.sort(key=lambda row: str(row["concept_id"]))
        if len(rows) < 2:
            continue
        for index, left in enumerate(rows):
            right = rows[(index + PAIR_OFFSET) % len(rows)]
            if (
                int(left["expected_first_token_id"])
                == int(right["expected_first_token_id"])
            ):
                continue
            targets.append({
                "schema_version": "phase1055_target.v1",
                "phase": PHASE,
                "model": model_name,
                "target_index": len(targets),
                "split": panel[0],
                "surface_index": panel[1],
                "variant": panel[2],
                "source_token_count": panel[3],
                "target_case_index": int(
                    left["semantic_case_index"]
                ),
                "cross_case_index": int(
                    right["semantic_case_index"]
                ),
                "target_concept_id": str(left["concept_id"]),
                "cross_concept_id": str(right["concept_id"]),
                "target_expected_label": str(
                    left["expected_label"]
                ),
                "cross_expected_label": str(
                    right["expected_label"]
                ),
            })
    return cases, targets


def main() -> None:
    source_aggregate = read_json(SOURCE_ROOT / "aggregate.json")
    next_decision = source_aggregate["automatic_next_decision"]
    if (
        not next_decision["should_continue_automatically"]
        or next_decision["route"]
        != "phase1055_pattern_family_transfer"
    ):
        raise RuntimeError(
            f"Phase1054 did not authorize transfer: {next_decision}"
        )
    source_prereg = read_json(
        SOURCE_ROOT / "protocol" / "preregistration.json"
    )
    bridge_prereg = read_json(
        bridge.OUT_ROOT / "protocol" / "preregistration.json"
    )
    model_plans = {}
    model_audits = {}
    for model_name in MODELS:
        cases, targets = build_model_cases(model_name)
        write_jsonl(
            OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl",
            cases,
        )
        write_jsonl(
            OUT_ROOT / "protocol" / f"targets.{model_name}.jsonl",
            targets,
        )
        source_summary = read_json(
            SOURCE_ROOT / "atlas" / model_name / "summary.json"
        )
        source_plan = source_prereg["model_plans"][model_name]
        bridge_plan = bridge_prereg["model_plans"][model_name]
        n_layers = int(bridge_plan["n_layers"])
        conditions = {
            "source_fact_rectangle": {
                "site": "source_term",
                "groups": [
                    int(value)
                    for value in source_summary["frozen_groups"]
                ],
                "depths": [
                    int(value)
                    for value in source_summary["frozen_depths"]
                ],
            },
            "source_all_groups_postsource": {
                "site": "source_term",
                "groups": [
                    int(value) for value in source_plan["all_groups"]
                ],
                "depths": [
                    int(value)
                    for value in source_plan[
                        "all_postsource_depths"
                    ]
                ],
            },
            "source_all_groups_all_layers": {
                "site": "source_term",
                "groups": [
                    int(value) for value in source_plan["all_groups"]
                ],
                "depths": list(range(1, n_layers + 1)),
            },
            "operator_all_groups_all_layers": {
                "site": "operator",
                "groups": [
                    int(value) for value in source_plan["all_groups"]
                ],
                "depths": list(range(1, n_layers + 1)),
            },
            "target_language_all_groups_all_layers": {
                "site": "target_language",
                "groups": [
                    int(value) for value in source_plan["all_groups"]
                ],
                "depths": list(range(1, n_layers + 1)),
            },
        }
        if tuple(conditions) != CONDITION_ORDER:
            raise RuntimeError("condition order drift")
        model_plans[model_name] = {
            "n_layers": n_layers,
            "n_kv_heads": int(source_plan["n_kv_heads"]),
            "conditions": conditions,
        }
        counts = Counter(
            (row["split"], row["variant"]) for row in targets
        )
        source_widths = [
            end - start + 1
            for row in cases
            for start, end in [
                row["role_spans"]["source_term"]
            ]
        ]
        model_audits[model_name] = {
            "case_count": len(cases),
            "target_count": len(targets),
            "target_counts": {
                f"{split}|{variant}": count
                for (split, variant), count in counts.items()
            },
            "maximum_source_span": max(source_widths),
            "minimum_panel_pair_count": min(counts.values()),
            "distinct_confirmation_concepts": len({
                value
                for row in targets
                if row["split"] == "confirmation"
                for value in (
                    row["target_concept_id"],
                    row["cross_concept_id"],
                )
            }),
        }

    audit = {
        "schema_version": "phase1055_protocol_audit.v1",
        "phase": PHASE,
        "models": model_audits,
    }
    audit["all_checks_passed"] = all(
        row["maximum_source_span"] <= bridge.MAX_ROLE_SPAN
        and row["minimum_panel_pair_count"] >= 20
        and row["distinct_confirmation_concepts"] >= 20
        for row in model_audits.values()
    )
    write_json(OUT_ROOT / "protocol" / "audit.json", audit)
    if not audit["all_checks_passed"]:
        raise RuntimeError(f"Phase1055 protocol audit failed: {audit}")

    payload = {
        "schema_version": "phase1055_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "source_phase1054_digest": source_prereg["protocol_digest"],
        "models": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "sequential_model_order": list(MODELS),
        "language_direction": "English_to_French",
        "surfaces": [dict(row) for row in SURFACES],
        "variants": [dict(row) for row in VARIANTS],
        "variant_order": list(VARIANT_ORDER),
        "variant_selection_rule": (
            "On discovery concepts only, maximize full-vocabulary clean "
            "pair accuracy, then correct pair count, concept coverage, "
            "and finally use the frozen variant order."
        ),
        "disjoint_lexical_splits": {
            "discovery": len(concept_rows("discovery")),
            "confirmation": len(concept_rows("confirmation")),
        },
        "model_plans": model_plans,
        "condition_order": list(CONDITION_ORDER),
        "rollout_steps": ROLLOUT_STEPS,
        "rollout_pair_limit": ROLLOUT_PAIR_LIMIT,
        "gates": GATES,
        "automatic_next": {
            "broad_translation_bridge_repeats": (
                "phase1056_translation_coalition_localization"
            ),
            "otherwise": "stop_with_pattern_specific_difference",
        },
        "interpretation_limits": [
            "This is one English-to-French lexical translation family.",
            "First-token transport is weaker than full-word transport.",
            "A reused fact rectangle would show physical reuse, not identity.",
            "A new broad bridge would show topology reuse, not one circuit.",
            "Operation and target-language controls precede the source term.",
            "No result establishes a universal translation algorithm.",
        ],
    }
    payload["protocol_digest"] = digest(payload)
    write_json(OUT_ROOT / "protocol" / "preregistration.json", payload)
    print(
        f"Phase{PHASE} protocol frozen: {payload['protocol_digest']} "
        f"models={len(MODELS)}"
    )


if __name__ == "__main__":
    main()
