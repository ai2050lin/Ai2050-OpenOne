#!/usr/bin/env python3
"""Freeze a compositional multi-token English-to-French protocol."""

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
import phase1057_translation_trajectory_protocol as source


PHASE = 1058
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
    / "phase1058_multitoken_translation"
)
SURFACES = (
    {
        "template": (
            "Translate this English phrase into French. Return exactly "
            "the French phrase and nothing else.\nEnglish phrase: {phrase}"
        ),
        "operator": "Translate",
        "target_language": "French",
    },
    {
        "template": (
            "Convert the English noun phrase below to French. Give only "
            "the French equivalent.\nSource phrase: {phrase}"
        ),
        "operator": "Convert",
        "target_language": "French",
    },
    {
        "template": (
            "Provide the French equivalent of this English phrase. "
            "Output only the translated phrase.\nPhrase: {phrase}"
        ),
        "operator": "equivalent",
        "target_language": "French",
    },
)
ASSISTANT_PREFILL = "Translation:"
CONTINUATION_PREFIX = " "
MAX_ROLE_SPAN = 8
GENERATION_STEPS = 10
PAIR_LIMIT = 72
CONTROL_PAIR_LIMIT = 48
CACHE_PARITY_PAIR_LIMIT = 8
PAIR_FAMILIES = ("phrase", "color", "noun")
GATES = {
    "discovery_exact_case_count_min": 90,
    "confirmation_exact_case_count_min": 90,
    "confirmation_pair_count_per_family_min": 30,
    "phrase_post_eos_exact_rate_min": 0.50,
    "component_post_eos_exact_rate_min": 0.40,
    "source_minus_control_rate_min": 0.30,
    "cache_parity_rate_min": 0.99,
    "minimum_repeated_models": 2,
}


# Six colors x twenty nouns produce 120 unique compositions. The same
# lexical components occur on both splits, while no complete composition
# crosses the split boundary.
COLORS = (
    ("red", "rouge", "rouge"),
    ("blue", "bleu", "bleue"),
    ("green", "vert", "verte"),
    ("black", "noir", "noire"),
    ("white", "blanc", "blanche"),
    ("yellow", "jaune", "jaune"),
)
NOUNS = (
    ("cat", "chat", "m"),
    ("dog", "chien", "m"),
    ("bird", "oiseau", "m"),
    ("horse", "cheval", "m"),
    ("book", "livre", "m"),
    ("pen", "stylo", "m"),
    ("hat", "chapeau", "m"),
    ("coat", "manteau", "m"),
    ("bag", "sac", "m"),
    ("wall", "mur", "m"),
    ("house", "maison", "f"),
    ("car", "voiture", "f"),
    ("chair", "chaise", "f"),
    ("cup", "tasse", "f"),
    ("shirt", "chemise", "f"),
    ("skirt", "jupe", "f"),
    ("road", "route", "f"),
    ("mountain", "montagne", "f"),
    ("flower", "fleur", "f"),
    ("door", "porte", "f"),
)


write_json = material.write_json
write_jsonl = material.write_jsonl
read_json = material.read_json
read_jsonl = material.read_jsonl
digest = material.digest


def compositions() -> list[dict[str, Any]]:
    rows = []
    for color_index, (color_en, color_m, color_f) in enumerate(COLORS):
        for noun_index, (noun_en, noun_fr, gender) in enumerate(NOUNS):
            adjective = color_m if gender == "m" else color_f
            rows.append({
                "composition_id": f"{color_en}_{noun_en}",
                "split": (
                    "discovery"
                    if (color_index + noun_index) % 2 == 0
                    else "confirmation"
                ),
                "color_id": color_en,
                "noun_id": noun_en,
                "gender": gender,
                "source_phrase": f"{color_en} {noun_en}",
                "target_phrase": f"{noun_fr} {adjective}",
            })
    return rows


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
        raise ValueError(f"unknown pair family: {family}")
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
    for composition in compositions():
        for surface_index, surface in enumerate(SURFACES):
            source_phrase = str(composition["source_phrase"])
            target_phrase = str(composition["target_phrase"])
            color = str(composition["color_id"])
            noun = str(composition["noun_id"])
            content = str(surface["template"]).format(
                phrase=source_phrase
            )
            phrase_start = content.rfind(source_phrase)
            if phrase_start < 0:
                raise RuntimeError(f"missing source phrase {source_phrase}")
            fragments = {
                "source_phrase": (
                    phrase_start,
                    phrase_start + len(source_phrase),
                    source_phrase,
                ),
                "source_color": (
                    phrase_start,
                    phrase_start + len(color),
                    color,
                ),
                "source_noun": (
                    phrase_start + len(color) + 1,
                    phrase_start + len(source_phrase),
                    noun,
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
                "schema_version": "phase1058_model_case.v1",
                "phase": PHASE,
                "model": model_name,
                "semantic_case_index": len(cases),
                "case_key": (
                    f"{composition['split']}.s{surface_index}."
                    f"{composition['composition_id']}"
                ),
                "split": composition["split"],
                "surface_index": surface_index,
                "composition_id": composition["composition_id"],
                "color_id": color,
                "noun_id": noun,
                "gender": composition["gender"],
                "source_phrase": source_phrase,
                "expected_label": target_phrase,
                "rendered_prompt": rendered,
                "input_ids": input_ids,
                "role_spans": role_spans,
                "expected_token_ids": expected_ids,
                "expected_first_token_id": int(expected_ids[0]),
                "multitoken_eligible": (
                    span_width({"role_spans": role_spans}, "source_phrase")
                    >= 2
                    and len(expected_ids) >= 2
                ),
            })

    panels: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        panels[(str(row["split"]), int(row["surface_index"]))].append(row)
    targets = []
    for panel in sorted(panels):
        pool = panels[panel]
        for left in sorted(
            pool, key=lambda row: str(row["composition_id"])
        ):
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
                    "schema_version": "phase1058_target.v1",
                    "phase": PHASE,
                    "model": model_name,
                    "target_index": len(targets),
                    "split": panel[0],
                    "surface_index": panel[1],
                    "pair_family": family,
                    "site": site,
                    "source_token_count": span_width(left, site),
                    "target_case_index": int(
                        left["semantic_case_index"]
                    ),
                    "cross_case_index": int(
                        right["semantic_case_index"]
                    ),
                    "target_composition_id": str(
                        left["composition_id"]
                    ),
                    "cross_composition_id": str(
                        right["composition_id"]
                    ),
                })
    return cases, targets


def main() -> None:
    source_aggregate = read_json(SOURCE_ROOT / "aggregate.json")
    route = source_aggregate["automatic_next_decision"]
    if (
        not route["should_continue_automatically"]
        or route["route"] != "phase1058_multitoken_translation"
    ):
        raise RuntimeError(f"Phase1057 did not authorize Phase1058: {route}")
    source_prereg = read_json(
        SOURCE_ROOT / "protocol" / "preregistration.json"
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
                "frozen_groups",
                "frozen_depths",
            )
        }
        split_compositions = {
            split: {
                str(row["composition_id"])
                for row in cases
                if row["split"] == split
            }
            for split in ("discovery", "confirmation")
        }
        target_counts = Counter(
            (str(row["split"]), str(row["pair_family"]))
            for row in targets
        )
        source_widths = [
            span_width(row, "source_phrase") for row in cases
        ]
        expected_widths = [
            len(row["expected_token_ids"]) for row in cases
        ]
        answer_leaks = [
            str(row["case_key"])
            for row in cases
            if str(row["expected_label"]).casefold()
            in str(row["rendered_prompt"]).casefold()
        ]
        model_audits[model_name] = {
            "case_count": len(cases),
            "target_counts": {
                f"{split}.{family}": count
                for (split, family), count in sorted(
                    target_counts.items()
                )
            },
            "minimum_source_phrase_span": min(source_widths),
            "maximum_source_phrase_span": max(source_widths),
            "minimum_expected_token_count": min(expected_widths),
            "maximum_expected_token_count": max(expected_widths),
            "multitoken_eligible_case_count": sum(
                bool(row["multitoken_eligible"]) for row in cases
            ),
            "split_composition_overlap": sorted(
                split_compositions["discovery"]
                & split_compositions["confirmation"]
            ),
            "distinct_compositions": len(
                split_compositions["discovery"]
                | split_compositions["confirmation"]
            ),
            "answer_leak_count": len(answer_leaks),
            "answer_leak_examples": answer_leaks[:5],
        }

    audit = {
        "schema_version": "phase1058_protocol_audit.v1",
        "phase": PHASE,
        "composition_count": len(compositions()),
        "models": model_audits,
    }
    audit["all_checks_passed"] = (
        audit["composition_count"] == 120
        and all(
            row["case_count"] == 360
            and row["minimum_source_phrase_span"] >= 2
            and row["maximum_source_phrase_span"] <= MAX_ROLE_SPAN
            and row["minimum_expected_token_count"] >= 2
            and row["multitoken_eligible_case_count"] == 360
            and not row["split_composition_overlap"]
            and row["distinct_compositions"] == 120
            and row["answer_leak_count"] == 0
            and all(
                row["target_counts"].get(
                    f"{split}.{family}", 0
                ) >= 100
                for split in ("discovery", "confirmation")
                for family in PAIR_FAMILIES
            )
            for row in model_audits.values()
        )
    )
    write_json(OUT_ROOT / "protocol" / "audit.json", audit)
    if not audit["all_checks_passed"]:
        raise RuntimeError(f"Phase1058 protocol audit failed: {audit}")

    payload = {
        "schema_version": "phase1058_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "source_phase1057_digest": source_prereg["protocol_digest"],
        "source_phase1057_route": route,
        "models": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "sequential_model_order": list(MODELS),
        "language_direction": "English_to_French",
        "composition_design": (
            "Six colors crossed with twenty gendered nouns; complete "
            "compositions are split by parity while lexical components "
            "are deliberately reused across both splits."
        ),
        "source_order": "color_then_noun",
        "target_order": "noun_then_gender_inflected_color",
        "surfaces": [dict(row) for row in SURFACES],
        "assistant_prefill": ASSISTANT_PREFILL,
        "continuation_prefix": CONTINUATION_PREFIX,
        "generation_steps": GENERATION_STEPS,
        "pair_limit": PAIR_LIMIT,
        "control_pair_limit": CONTROL_PAIR_LIMIT,
        "cache_parity_pair_limit": CACHE_PARITY_PAIR_LIMIT,
        "pair_families": list(PAIR_FAMILIES),
        "model_plans": model_plans,
        "condition_families": {
            "component_transport": [
                "phrase_post_kv",
                "color_post_kv",
                "noun_post_kv",
            ],
            "phase": [
                "phrase_early_kv",
                "phrase_post_kv",
                "phrase_all_kv",
            ],
            "channels": [
                "phrase_post_k_only",
                "phrase_post_v_only",
                "phrase_post_kv",
            ],
            "frozen_rectangle": "Phase1056/1057 frozen rectangle",
            "role_controls": [
                "operator_post_kv",
                "target_language_post_kv",
            ],
        },
        "primary_observations": [
            "EOS-censored exact donor-sequence transport",
            "content-only exact donor-sequence transport",
            "first distinguishing token transport",
            "donor-prefix agreement by generation step",
            "termination-step agreement",
        ],
        "gates": GATES,
        "automatic_next": {
            "decision": (
                "Stop after this major compositional milestone. A move "
                "to sentences or a different language pattern requires "
                "a separately audited protocol rather than an automatic "
                "extension of the lexical intervention."
            ),
            "should_continue_automatically": False,
        },
        "interpretation_limits": [
            "Clean-behavior filtering uses no intervention outcome.",
            "The vocabulary is common and remains in-distribution.",
            "K/V replacement is a sufficient graph cut, not a unique circuit.",
            "Exact sequence transport does not prove a symbolic phrase object.",
            "Color and noun reuse here does not establish all language rules.",
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
        f"models={len(MODELS)} compositions={len(compositions())}"
    )


if __name__ == "__main__":
    main()
