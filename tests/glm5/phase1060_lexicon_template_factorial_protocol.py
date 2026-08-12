#!/usr/bin/env python3
"""Freeze a lexicon-by-template factorial repair of Phase1059."""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase1018_language_pattern_protocol import tokenizer_for
from phase1021_natural_language_atlas_protocol import offset_token_spans
import phase1040_expanded_mlp_replication_protocol as material
import phase1051_natural_behavior_protocol as behavior
import phase1058_multitoken_translation_protocol as legacy
import phase1059_lexically_heldout_composition_protocol as source


PHASE = 1060
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
    / "phase1060_lexicon_template_factorial"
)
ASSISTANT_PREFILL = "Translation:"
CONTINUATION_PREFIX = " "
MAX_ROLE_SPAN = 8
GENERATION_STEPS = 10
PAIR_LIMIT = 80
CONTROL_PAIR_LIMIT = 48
PAIR_FAMILIES = ("phrase", "color", "noun")
CELLS = ("old_old", "old_new", "new_old", "new_new")
GATES = {
    "exact_case_count_per_primary_cell_min": 60,
    "valid_pair_count_per_primary_family_min": 50,
    "phrase_post_eos_exact_rate_min": 0.50,
    "component_post_eos_exact_rate_min": 0.35,
    "source_minus_control_rate_min": 0.30,
    "minimum_repeated_models": 2,
}


OLD_COLORS = tuple(legacy.COLORS[:4])
OLD_NOUNS = (
    legacy.NOUNS[0],
    legacy.NOUNS[1],
    legacy.NOUNS[2],
    legacy.NOUNS[3],
    legacy.NOUNS[4],
    legacy.NOUNS[5],
    legacy.NOUNS[6],
    legacy.NOUNS[10],
    legacy.NOUNS[11],
    legacy.NOUNS[12],
    legacy.NOUNS[13],
    legacy.NOUNS[14],
    legacy.NOUNS[15],
    legacy.NOUNS[16],
)
NEW_COLORS = tuple(source.COLORS["confirmation"])
NEW_NOUNS = tuple(source.NOUNS["confirmation"])
OLD_TEMPLATES = tuple(legacy.SURFACES[:2])
NEW_TEMPLATES = tuple(source.SURFACES["confirmation"])


write_json = material.write_json
write_jsonl = material.write_jsonl
read_json = material.read_json
read_jsonl = material.read_jsonl
digest = material.digest


def cell_parts(
    cell: str,
) -> tuple[
    tuple[tuple[str, str, str], ...],
    tuple[tuple[str, str, str], ...],
    tuple[dict[str, str], ...],
]:
    lexicon, templates = cell.split("_")
    colors = OLD_COLORS if lexicon == "old" else NEW_COLORS
    nouns = OLD_NOUNS if lexicon == "old" else NEW_NOUNS
    surfaces = (
        OLD_TEMPLATES if templates == "old" else NEW_TEMPLATES
    )
    return colors, nouns, surfaces


def fragment(
    text: str,
    value: str,
    *,
    occurrence: str,
) -> tuple[int, int, str]:
    start = text.find(value) if occurrence == "first" else text.rfind(value)
    if start < 0:
        raise RuntimeError(f"missing fragment {value!r}")
    return start, start + len(value), value


def span_width(row: dict[str, Any], role: str) -> int:
    start, end = row["role_spans"][role]
    return int(end) - int(start) + 1


def pair_candidate(
    left: dict[str, Any],
    pool: list[dict[str, Any]],
    family: str,
) -> dict[str, Any] | None:
    if family == "color":
        site = "source_color"
        candidates = [
            row for row in pool
            if row["noun_id"] == left["noun_id"]
            and row["color_id"] != left["color_id"]
        ]
    elif family == "noun":
        site = "source_noun"
        candidates = [
            row for row in pool
            if row["color_id"] == left["color_id"]
            and row["noun_id"] != left["noun_id"]
        ]
    elif family == "phrase":
        site = "source_phrase"
        candidates = [
            row for row in pool
            if row["color_id"] != left["color_id"]
            and row["noun_id"] != left["noun_id"]
        ]
    else:
        raise ValueError(f"unknown family: {family}")
    candidates = [
        row for row in candidates
        if len(row["input_ids"]) == len(left["input_ids"])
        and span_width(row, site) == span_width(left, site)
        and row["expected_token_ids"] != left["expected_token_ids"]
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda row: str(row["composition_id"]))
    return candidates[
        int(left["semantic_case_index"]) % len(candidates)
    ]


def build_model_cases(
    model_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tokenizer = tokenizer_for(model_name)
    cases = []
    for cell in CELLS:
        colors, nouns, surfaces = cell_parts(cell)
        for color_en, color_m, color_f in colors:
            for noun_en, noun_fr, gender in nouns:
                adjective = color_m if gender == "m" else color_f
                source_phrase = f"{color_en} {noun_en}"
                target_phrase = f"{noun_fr} {adjective}"
                composition_id = f"{color_en}_{noun_en}"
                for surface_index, surface in enumerate(surfaces):
                    content = str(surface["template"]).format(
                        phrase=source_phrase
                    )
                    phrase_start = content.rfind(source_phrase)
                    if phrase_start < 0:
                        raise RuntimeError(
                            f"missing source phrase {source_phrase}"
                        )
                    fragments = {
                        "source_phrase": (
                            phrase_start,
                            phrase_start + len(source_phrase),
                            source_phrase,
                        ),
                        "source_color": (
                            phrase_start,
                            phrase_start + len(color_en),
                            color_en,
                        ),
                        "source_noun": (
                            phrase_start + len(color_en) + 1,
                            phrase_start + len(source_phrase),
                            noun_en,
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
                    rendered = behavior.render_native(
                        tokenizer,
                        model_name,
                        content,
                        with_system=False,
                    )
                    rendered += ASSISTANT_PREFILL
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
                    expected_ids = behavior.continuation_ids(
                        tokenizer,
                        rendered,
                        CONTINUATION_PREFIX,
                        target_phrase,
                    )
                    cases.append({
                        "schema_version": "phase1060_model_case.v1",
                        "phase": PHASE,
                        "model": model_name,
                        "semantic_case_index": len(cases),
                        "case_key": (
                            f"{cell}.s{surface_index}.{composition_id}"
                        ),
                        "cell": cell,
                        "lexicon_panel": cell.split("_")[0],
                        "template_panel": cell.split("_")[1],
                        "surface_index": surface_index,
                        "composition_id": composition_id,
                        "color_id": color_en,
                        "noun_id": noun_en,
                        "gender": gender,
                        "source_phrase": source_phrase,
                        "expected_label": target_phrase,
                        "rendered_prompt": rendered,
                        "input_ids": input_ids,
                        "role_spans": role_spans,
                        "expected_token_ids": expected_ids,
                        "expected_first_token_id": int(expected_ids[0]),
                        "multitoken_eligible": (
                            span_width(
                                {"role_spans": role_spans},
                                "source_phrase",
                            )
                            >= 2
                            and len(expected_ids) >= 2
                        ),
                    })

    panels: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        panels[(str(row["cell"]), int(row["surface_index"]))].append(row)
    targets = []
    for panel in sorted(panels):
        pool = panels[panel]
        for left in sorted(pool, key=lambda row: str(row["composition_id"])):
            if not left["multitoken_eligible"]:
                continue
            for family in PAIR_FAMILIES:
                right = pair_candidate(left, pool, family)
                if right is None or not right["multitoken_eligible"]:
                    continue
                site = {
                    "phrase": "source_phrase",
                    "color": "source_color",
                    "noun": "source_noun",
                }[family]
                targets.append({
                    "schema_version": "phase1060_target.v1",
                    "phase": PHASE,
                    "model": model_name,
                    "target_index": len(targets),
                    "cell": panel[0],
                    "surface_index": panel[1],
                    "pair_family": family,
                    "site": site,
                    "source_token_count": span_width(left, site),
                    "target_case_index": int(left["semantic_case_index"]),
                    "cross_case_index": int(right["semantic_case_index"]),
                    "target_composition_id": str(left["composition_id"]),
                    "cross_composition_id": str(right["composition_id"]),
                })
    return cases, targets


def main() -> None:
    source_prereg = read_json(
        SOURCE_ROOT / "protocol" / "preregistration.json"
    )
    source_aggregate = read_json(SOURCE_ROOT / "aggregate.json")
    if source_aggregate["automatic_next_decision"][
        "should_continue_automatically"
    ]:
        raise RuntimeError("Phase1059 unexpectedly authorized sentences")
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
        source_plan = source_prereg["model_plans"][model_name]
        model_plans[model_name] = {
            key: source_plan[key]
            for key in (
                "n_layers",
                "n_kv_heads",
                "all_groups",
                "all_layers",
                "early_depths",
                "postsource_depths",
                "late_half_depths",
                "late_quarter_depths",
                "even_groups",
                "odd_groups",
                "frozen_groups",
                "frozen_depths",
            )
        }
        target_counts = Counter(
            (str(row["cell"]), str(row["pair_family"]))
            for row in targets
        )
        answer_leaks = [
            str(row["case_key"])
            for row in cases
            if str(row["expected_label"]).casefold()
            in str(row["rendered_prompt"]).casefold()
        ]
        widths = [
            span_width(row, role)
            for row in cases
            for role in ("source_phrase", "source_color", "source_noun")
        ]
        model_audits[model_name] = {
            "case_count": len(cases),
            "case_counts_by_cell": dict(Counter(
                str(row["cell"]) for row in cases
            )),
            "target_counts": {
                f"{cell}.{family}": count
                for (cell, family), count in sorted(target_counts.items())
            },
            "maximum_role_span": max(widths),
            "multitoken_eligible_case_count": sum(
                bool(row["multitoken_eligible"]) for row in cases
            ),
            "answer_leak_count": len(answer_leaks),
        }

    old_color_ids = {str(row[0]) for row in OLD_COLORS}
    new_color_ids = {str(row[0]) for row in NEW_COLORS}
    old_noun_ids = {str(row[0]) for row in OLD_NOUNS}
    new_noun_ids = {str(row[0]) for row in NEW_NOUNS}
    old_template_ids = {str(row["template"]) for row in OLD_TEMPLATES}
    new_template_ids = {str(row["template"]) for row in NEW_TEMPLATES}
    audit = {
        "schema_version": "phase1060_protocol_audit.v1",
        "phase": PHASE,
        "old_new_color_overlap": sorted(old_color_ids & new_color_ids),
        "old_new_noun_overlap": sorted(old_noun_ids & new_noun_ids),
        "old_new_template_overlap": sorted(
            old_template_ids & new_template_ids
        ),
        "models": model_audits,
    }
    audit["all_checks_passed"] = (
        not audit["old_new_color_overlap"]
        and not audit["old_new_noun_overlap"]
        and not audit["old_new_template_overlap"]
        and all(
            row["case_count"] == 448
            and row["maximum_role_span"] <= MAX_ROLE_SPAN
            and row["multitoken_eligible_case_count"] == 448
            and row["answer_leak_count"] == 0
            and all(
                row["case_counts_by_cell"].get(cell, 0) == 112
                for cell in CELLS
            )
            and all(
                row["target_counts"].get(f"{cell}.{family}", 0)
                >= 80
                for cell in CELLS
                for family in PAIR_FAMILIES
            )
            for row in model_audits.values()
        )
    )
    write_json(OUT_ROOT / "protocol" / "audit.json", audit)
    if not audit["all_checks_passed"]:
        raise RuntimeError(f"Phase1060 protocol audit failed: {audit}")

    payload = {
        "schema_version": "phase1060_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "source_phase1059_digest": source_prereg["protocol_digest"],
        "source_phase1059_route": source_aggregate[
            "automatic_next_decision"
        ],
        "authorization": (
            "Automatic protocol repair requested by the user after the "
            "Phase1059 behavior-gate failure."
        ),
        "models": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "sequential_model_order": list(MODELS),
        "factorial_cells": {
            "old_old": "Phase1058-like lexicon and Phase1058 templates",
            "old_new": "Phase1058-like lexicon and Phase1059 templates",
            "new_old": "Phase1059 confirmation lexicon and Phase1058 templates",
            "new_new": "Phase1059 confirmation lexicon and Phase1059 templates",
        },
        "primary_causal_cell": "new_old",
        "behavior_reference_cell": "old_old",
        "generation_steps": GENERATION_STEPS,
        "pair_limit": PAIR_LIMIT,
        "control_pair_limit": CONTROL_PAIR_LIMIT,
        "pair_families": list(PAIR_FAMILIES),
        "model_plans": model_plans,
        "gates": GATES,
        "automatic_next": {
            "if_two_models_repeat_new_old": (
                "phase1061_sentence_role_transport"
            ),
            "otherwise": "stop_with_lexicon_or_protocol_limit",
        },
        "interpretation_limits": [
            "The 2x2 matrix separates panel effects but not every word effect.",
            "Exact behavior can vary by lexical collocation and tokenization.",
            "Only behavior-qualified cells support intervention inference.",
            "K/V replacement is sufficient under intervention, not unique.",
            "Support-width cuts are descriptive and are not minimum circuits.",
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
        f"models={len(MODELS)} cases_per_model=448"
    )


if __name__ == "__main__":
    main()
