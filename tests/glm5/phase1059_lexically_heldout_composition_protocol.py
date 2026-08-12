#!/usr/bin/env python3
"""Freeze a fully held-out compositional translation protocol."""

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
import phase1058_multitoken_translation_protocol as source


PHASE = 1059
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
    / "phase1059_lexically_heldout_composition"
)
ASSISTANT_PREFILL = "Translation:"
CONTINUATION_PREFIX = " "
MAX_ROLE_SPAN = 8
GENERATION_STEPS = 10
PAIR_LIMIT = 80
CONTROL_PAIR_LIMIT = 48
PAIR_FAMILIES = ("phrase", "color", "noun")
GATES = {
    "exact_case_count_per_split_min": 60,
    "valid_pair_count_per_split_family_min": 50,
    "phrase_post_eos_exact_rate_min": 0.50,
    "component_post_eos_exact_rate_min": 0.35,
    "source_minus_control_rate_min": 0.30,
    "minimum_repeated_models": 2,
}


SURFACES = {
    "discovery": (
        {
            "template": (
                "Translate the following English noun phrase into French. "
                "Reply with only the French noun phrase.\nInput: {phrase}"
            ),
            "operator": "Translate",
            "target_language": "French",
        },
        {
            "template": (
                "Write the exact French translation of this English phrase "
                "and no commentary.\nEnglish: {phrase}"
            ),
            "operator": "translation",
            "target_language": "French",
        },
    ),
    "confirmation": (
        {
            "template": (
                "Render this English noun phrase in French. Output the "
                "translation alone.\nPhrase to translate: {phrase}"
            ),
            "operator": "Render",
            "target_language": "French",
        },
        {
            "template": (
                "What is the French equivalent of the English noun phrase "
                "below? Give only the answer.\nNoun phrase: {phrase}"
            ),
            "operator": "equivalent",
            "target_language": "French",
        },
    ),
}


# These color words and nouns do not occur in the Phase1058 panel. Discovery
# and confirmation use disjoint lexical items as well as disjoint templates.
COLORS = {
    "discovery": (
        ("orange", "orange", "orange"),
        ("purple", "violet", "violette"),
        ("pink", "rose", "rose"),
        ("gray", "gris", "grise"),
    ),
    "confirmation": (
        ("brown", "marron", "marron"),
        ("beige", "beige", "beige"),
        ("golden", "doré", "dorée"),
        ("silver", "argenté", "argentée"),
    ),
}
NOUNS = {
    "discovery": (
        ("tree", "arbre", "m"),
        ("train", "train", "m"),
        ("boat", "bateau", "m"),
        ("garden", "jardin", "m"),
        ("castle", "château", "m"),
        ("knife", "couteau", "m"),
        ("basket", "panier", "m"),
        ("table", "table", "f"),
        ("bottle", "bouteille", "f"),
        ("river", "rivière", "f"),
        ("beach", "plage", "f"),
        ("forest", "forêt", "f"),
        ("street", "rue", "f"),
        ("plate", "assiette", "f"),
    ),
    "confirmation": (
        ("mirror", "miroir", "m"),
        ("floor", "sol", "m"),
        ("roof", "toit", "m"),
        ("bed", "lit", "m"),
        ("pencil", "crayon", "m"),
        ("bridge", "pont", "m"),
        ("lake", "lac", "m"),
        ("window", "fenêtre", "f"),
        ("kitchen", "cuisine", "f"),
        ("room", "chambre", "f"),
        ("dress", "robe", "f"),
        ("shoe", "chaussure", "f"),
        ("pocket", "poche", "f"),
        ("fork", "fourchette", "f"),
    ),
}


write_json = material.write_json
write_jsonl = material.write_jsonl
read_json = material.read_json
read_jsonl = material.read_jsonl
digest = material.digest


def compositions() -> list[dict[str, Any]]:
    rows = []
    for split in ("discovery", "confirmation"):
        for color_en, color_m, color_f in COLORS[split]:
            for noun_en, noun_fr, gender in NOUNS[split]:
                adjective = color_m if gender == "m" else color_f
                rows.append({
                    "composition_id": f"{color_en}_{noun_en}",
                    "split": split,
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
    surface_number = 0
    for composition in compositions():
        split = str(composition["split"])
        for local_surface, surface in enumerate(SURFACES[split]):
            source_phrase = str(composition["source_phrase"])
            target_phrase = str(composition["target_phrase"])
            color = str(composition["color_id"])
            noun = str(composition["noun_id"])
            content = str(surface["template"]).format(phrase=source_phrase)
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
                "schema_version": "phase1059_model_case.v1",
                "phase": PHASE,
                "model": model_name,
                "semantic_case_index": len(cases),
                "case_key": (
                    f"{split}.s{local_surface}."
                    f"{composition['composition_id']}"
                ),
                "split": split,
                "surface_index": surface_number + local_surface,
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
        surface_number += 0

    panels: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        panels[(str(row["split"]), int(row["surface_index"]))].append(row)
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
                    "schema_version": "phase1059_target.v1",
                    "phase": PHASE,
                    "model": model_name,
                    "target_index": len(targets),
                    "split": panel[0],
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


def tail(values: list[int], fraction: float) -> list[int]:
    count = max(1, int(len(values) * fraction + 0.999999))
    return values[-count:]


def main() -> None:
    source_prereg = read_json(
        SOURCE_ROOT / "protocol" / "preregistration.json"
    )
    source_aggregate = read_json(SOURCE_ROOT / "aggregate.json")
    old_colors = {str(row[0]) for row in source.COLORS}
    old_nouns = {str(row[0]) for row in source.NOUNS}
    all_new_colors = {
        str(row[0]) for split in COLORS.values() for row in split
    }
    all_new_nouns = {
        str(row[0]) for split in NOUNS.values() for row in split
    }
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
        post = [int(value) for value in source_plan["postsource_depths"]]
        groups = [int(value) for value in source_plan["all_groups"]]
        model_plans[model_name] = {
            "n_layers": int(source_plan["n_layers"]),
            "n_kv_heads": int(source_plan["n_kv_heads"]),
            "all_groups": groups,
            "all_layers": [
                int(value) for value in source_plan["all_layers"]
            ],
            "early_depths": [
                int(value) for value in source_plan["early_depths"]
            ],
            "postsource_depths": post,
            "late_half_depths": tail(post, 0.50),
            "late_quarter_depths": tail(post, 0.25),
            "even_groups": groups[::2],
            "odd_groups": groups[1::2],
            "frozen_groups": [
                int(value) for value in source_plan["frozen_groups"]
            ],
            "frozen_depths": [
                int(value) for value in source_plan["frozen_depths"]
            ],
        }
        target_counts = Counter(
            (str(row["split"]), str(row["pair_family"]))
            for row in targets
        )
        span_widths = [
            span_width(row, role)
            for row in cases
            for role in ("source_phrase", "source_color", "source_noun")
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
                for (split, family), count in sorted(target_counts.items())
            },
            "maximum_role_span": max(span_widths),
            "multitoken_eligible_case_count": sum(
                bool(row["multitoken_eligible"]) for row in cases
            ),
            "answer_leak_count": len(answer_leaks),
            "answer_leak_examples": answer_leaks[:5],
        }

    discovery_colors = {str(row[0]) for row in COLORS["discovery"]}
    confirmation_colors = {
        str(row[0]) for row in COLORS["confirmation"]
    }
    discovery_nouns = {str(row[0]) for row in NOUNS["discovery"]}
    confirmation_nouns = {str(row[0]) for row in NOUNS["confirmation"]}
    discovery_templates = {
        str(row["template"]) for row in SURFACES["discovery"]
    }
    confirmation_templates = {
        str(row["template"]) for row in SURFACES["confirmation"]
    }
    audit = {
        "schema_version": "phase1059_protocol_audit.v1",
        "phase": PHASE,
        "composition_count": len(compositions()),
        "phase1058_color_overlap": sorted(old_colors & all_new_colors),
        "phase1058_noun_overlap": sorted(old_nouns & all_new_nouns),
        "split_color_overlap": sorted(
            discovery_colors & confirmation_colors
        ),
        "split_noun_overlap": sorted(
            discovery_nouns & confirmation_nouns
        ),
        "split_template_overlap": sorted(
            discovery_templates & confirmation_templates
        ),
        "models": model_audits,
    }
    audit["all_checks_passed"] = (
        audit["composition_count"] == 112
        and not audit["phase1058_color_overlap"]
        and not audit["phase1058_noun_overlap"]
        and not audit["split_color_overlap"]
        and not audit["split_noun_overlap"]
        and not audit["split_template_overlap"]
        and all(
            row["case_count"] == 224
            and row["maximum_role_span"] <= MAX_ROLE_SPAN
            and row["multitoken_eligible_case_count"] == 224
            and row["answer_leak_count"] == 0
            and all(
                row["target_counts"].get(
                    f"{split}.{family}", 0
                ) >= 80
                for split in ("discovery", "confirmation")
                for family in PAIR_FAMILIES
            )
            for row in model_audits.values()
        )
    )
    write_json(OUT_ROOT / "protocol" / "audit.json", audit)
    if not audit["all_checks_passed"]:
        raise RuntimeError(f"Phase1059 protocol audit failed: {audit}")

    payload = {
        "schema_version": "phase1059_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "source_phase1058_digest": source_prereg["protocol_digest"],
        "source_phase1058_stop": source_aggregate[
            "automatic_next_decision"
        ],
        "authorization": (
            "User-authorized major follow-up after the Phase1058 stop gate."
        ),
        "models": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "sequential_model_order": list(MODELS),
        "language_direction": "English_to_French",
        "holdout_design": (
            "Discovery and confirmation have disjoint color words, nouns, "
            "complete compositions, and prompt templates. All lexical items "
            "are also absent from the Phase1058 panel."
        ),
        "surfaces": {
            split: [dict(row) for row in rows]
            for split, rows in SURFACES.items()
        },
        "assistant_prefill": ASSISTANT_PREFILL,
        "continuation_prefix": CONTINUATION_PREFIX,
        "generation_steps": GENERATION_STEPS,
        "pair_limit": PAIR_LIMIT,
        "control_pair_limit": CONTROL_PAIR_LIMIT,
        "pair_families": list(PAIR_FAMILIES),
        "model_plans": model_plans,
        "fixed_support_cuts": [
            "all_post_all_groups",
            "late_half_all_groups",
            "late_quarter_all_groups",
            "all_post_even_groups",
            "all_post_odd_groups",
            "phase1058_frozen_rectangle",
        ],
        "primary_observations": [
            "EOS-censored exact donor-sequence transport",
            "content-only exact donor-sequence transport",
            "first distinguishing token transport",
            "donor-prefix agreement",
            "termination-step agreement",
            "retention under fixed depth and group cuts",
        ],
        "gates": GATES,
        "automatic_next": {
            "if_two_models_repeat": "phase1060_sentence_role_transport",
            "otherwise": "stop_and_revise_behavior_or_holdout_protocol",
        },
        "interpretation_limits": [
            "The support cuts are fixed probes, not a minimum-circuit search.",
            "A low compact-cut rate does not prove every omitted component is necessary.",
            "A high group-half rate does not identify a unique natural module.",
            "K/V replacement remains a sufficient intervention, not a decoder.",
            "French agreement may be carried with lexical and collocational cues.",
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
