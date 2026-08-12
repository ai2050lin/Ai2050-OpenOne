#!/usr/bin/env python3
"""Freeze a stratified lexical behavior atlas for phrase translation."""

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
import phase1058_multitoken_translation_protocol as anchor
import phase1059_lexically_heldout_composition_protocol as heldout
import phase1062_text_equivalence_protocol as source


PHASE = 1063
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
    / "phase1063_lexical_behavior_atlas"
)
GENERATION_STEPS = 10
PAIR_FAMILIES = ("phrase", "color", "noun")
PRIMARY_PANELS = ("anchor_common", "novel_noun")
SURFACE = {
    "template": (
        "Translate this English phrase into French. Return exactly "
        "the French phrase and nothing else.\nEnglish phrase: {phrase}"
    ),
    "operator": "Translate",
    "target_language": "French",
}
ASSISTANT_PREFILL = "Translation:"
CONTINUATION_PREFIX = " "
MAX_ROLE_SPAN = 12
GATES = {
    "panel_accepted_case_min": {
        "anchor_common": 90,
        "novel_noun": 160,
        "ambiguous_color": 55,
    },
    "panel_valid_pair_min": {
        "anchor_common": 80,
        "novel_noun": 100,
        "ambiguous_color": 40,
    },
    "minimum_repeated_models": 2,
}


COMMON_COLORS = (
    ("red", ("rouge",), ("rouge",)),
    ("blue", ("bleu",), ("bleue",)),
    ("green", ("vert",), ("verte",)),
    ("black", ("noir",), ("noire",)),
    ("white", ("blanc",), ("blanche",)),
    ("yellow", ("jaune",), ("jaune",)),
)
AMBIGUOUS_COLORS = (
    ("brown", ("marron", "brun"), ("marron", "brune")),
    ("beige", ("beige",), ("beige",)),
    ("golden", ("doré", "d'or", "en or"), ("dorée", "d'or", "en or")),
    (
        "silver",
        ("argenté", "d'argent", "en argent"),
        ("argentée", "d'argent", "en argent"),
    ),
)


def noun(
    english: str,
    french: str,
    gender: str,
    *alternatives: tuple[str, str],
) -> tuple[str, tuple[tuple[str, str], ...]]:
    return english, ((french, gender), *alternatives)


ANCHOR_NOUNS = tuple(
    noun(english, french, gender)
    for english, french, gender in anchor.NOUNS
)
NOVEL_NOUNS = (
    noun("rabbit", "lapin", "m"),
    noun("duck", "canard", "m"),
    noun("lion", "lion", "m"),
    noun("tiger", "tigre", "m"),
    noun("bear", "ours", "m"),
    noun("monkey", "singe", "m"),
    noun("fox", "renard", "m"),
    noun("wolf", "loup", "m"),
    noun("fish", "poisson", "m"),
    noun("elephant", "éléphant", "m"),
    noun("phone", "téléphone", "m"),
    noun("computer", "ordinateur", "m"),
    noun("glass", "verre", "m"),
    noun("bowl", "bol", "m"),
    noun("newspaper", "journal", "m"),
    noun("notebook", "cahier", "m", ("carnet", "m")),
    noun("brush", "pinceau", "m"),
    noun("hammer", "marteau", "m"),
    noun("oven", "four", "m"),
    noun("umbrella", "parapluie", "m"),
    noun("lamp", "lampe", "f"),
    noun("key", "clé", "f"),
    noun("clock", "horloge", "f", ("pendule", "f")),
    noun("box", "boîte", "f"),
    noun("spoon", "cuillère", "f"),
    noun("apple", "pomme", "f"),
    noun("pear", "poire", "f"),
    noun("banana", "banane", "f"),
    noun("strawberry", "fraise", "f"),
    noun("suitcase", "valise", "f"),
    noun("tower", "tour", "f"),
    noun("island", "île", "f"),
    noun("valley", "vallée", "f"),
    noun("school", "école", "f"),
    noun("church", "église", "f"),
    noun("farm", "ferme", "f"),
    noun("kettle", "bouilloire", "f"),
    noun("pan", "casserole", "f", ("poêle", "f")),
    noun("ball", "balle", "f"),
    noun("bell", "cloche", "f"),
)
PANELS = {
    "anchor_common": {
        "colors": COMMON_COLORS,
        "nouns": ANCHOR_NOUNS,
        "purpose": "Phase1058 lexical anchor under one frozen surface.",
    },
    "novel_noun": {
        "colors": COMMON_COLORS,
        "nouns": NOVEL_NOUNS,
        "purpose": (
            "New noun identities with stable color expressions; isolates "
            "noun novelty from ambiguous color translation."
        ),
    },
    "ambiguous_color": {
        "colors": AMBIGUOUS_COLORS,
        "nouns": ANCHOR_NOUNS,
        "purpose": (
            "Multiple predeclared color expressions on familiar nouns; "
            "diagnostic only."
        ),
    },
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
    occurrence: str = "first",
) -> tuple[int, int, str]:
    start = text.find(value) if occurrence == "first" else text.rfind(value)
    if start < 0:
        raise RuntimeError(f"missing fragment {value!r}")
    return start, start + len(value), value


def labels_for(
    color: tuple[str, tuple[str, ...], tuple[str, ...]],
    noun_row: tuple[str, tuple[tuple[str, str], ...]],
) -> list[str]:
    _, masculine, feminine = color
    _, noun_forms = noun_row
    labels = []
    for noun_fr, gender in noun_forms:
        adjectives = masculine if gender == "m" else feminine
        labels.extend(f"{noun_fr} {adjective}" for adjective in adjectives)
    return list(dict.fromkeys(labels))


def compositions() -> list[dict[str, Any]]:
    rows = []
    for panel_name, panel in PANELS.items():
        for color in panel["colors"]:
            for noun_row in panel["nouns"]:
                labels = labels_for(color, noun_row)
                rows.append({
                    "panel": panel_name,
                    "composition_id": f"{color[0]}_{noun_row[0]}",
                    "color_id": color[0],
                    "noun_id": noun_row[0],
                    "source_phrase": f"{color[0]} {noun_row[0]}",
                    "canonical_label": labels[0],
                    "acceptable_labels": labels,
                })
    return rows


def pair_candidate(
    left: dict[str, Any],
    pool: list[dict[str, Any]],
    family: str,
) -> dict[str, Any] | None:
    if family == "color":
        candidates = [
            row for row in pool
            if row["noun_id"] == left["noun_id"]
            and row["color_id"] != left["color_id"]
        ]
    elif family == "noun":
        candidates = [
            row for row in pool
            if row["color_id"] == left["color_id"]
            and row["noun_id"] != left["noun_id"]
        ]
    elif family == "phrase":
        candidates = [
            row for row in pool
            if row["color_id"] != left["color_id"]
            and row["noun_id"] != left["noun_id"]
        ]
    else:
        raise ValueError(f"unknown pair family: {family}")
    if not candidates:
        return None
    candidates.sort(key=lambda row: str(row["composition_id"]))
    return candidates[
        int(left["semantic_case_index"]) % len(candidates)
    ]


def dedupe_token_sequences(values: list[list[int]]) -> list[list[int]]:
    output = []
    seen = set()
    for row in values:
        key = tuple(int(value) for value in row)
        if key not in seen:
            seen.add(key)
            output.append(list(key))
    return output


def build_model_cases(
    model_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tokenizer = tokenizer_for(model_name)
    cases = []
    for composition in compositions():
        source_phrase = str(composition["source_phrase"])
        color = str(composition["color_id"])
        noun_id = str(composition["noun_id"])
        content = str(SURFACE["template"]).format(phrase=source_phrase)
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
                noun_id,
            ),
            "operator": fragment(
                content, str(SURFACE["operator"]), occurrence="first"
            ),
            "target_language": fragment(
                content,
                str(SURFACE["target_language"]),
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
            int(value)
            for value in tokenizer.encode(rendered, add_special_tokens=False)
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
        acceptable_labels = [
            str(value) for value in composition["acceptable_labels"]
        ]
        acceptable_token_ids = dedupe_token_sequences([
            behavior.continuation_ids(
                tokenizer,
                rendered,
                CONTINUATION_PREFIX,
                label,
            )
            for label in acceptable_labels
        ])
        canonical_ids = behavior.continuation_ids(
            tokenizer,
            rendered,
            CONTINUATION_PREFIX,
            str(composition["canonical_label"]),
        )
        cases.append({
            "schema_version": "phase1063_model_case.v1",
            "phase": PHASE,
            "model": model_name,
            "semantic_case_index": len(cases),
            "case_key": (
                f"{composition['panel']}."
                f"{composition['composition_id']}"
            ),
            "panel": composition["panel"],
            "composition_id": composition["composition_id"],
            "color_id": color,
            "noun_id": noun_id,
            "source_phrase": source_phrase,
            "expected_label": composition["canonical_label"],
            "acceptable_labels": acceptable_labels,
            "rendered_prompt": rendered,
            "input_ids": input_ids,
            "role_spans": role_spans,
            "expected_token_ids": canonical_ids,
            "acceptable_token_ids": acceptable_token_ids,
            "source_token_width": (
                role_spans["source_phrase"][1]
                - role_spans["source_phrase"][0]
                + 1
            ),
            "canonical_target_token_width": len(canonical_ids),
            "multitoken_eligible": (
                role_spans["source_phrase"][1]
                - role_spans["source_phrase"][0] + 1 >= 2
                and len(canonical_ids) >= 2
            ),
        })

    panels: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        panels[str(row["panel"])].append(row)
    targets = []
    for panel_name, pool in sorted(panels.items()):
        for left in sorted(
            pool, key=lambda row: str(row["composition_id"])
        ):
            if not left["multitoken_eligible"]:
                continue
            for family in PAIR_FAMILIES:
                right = pair_candidate(left, pool, family)
                if right is None or not right["multitoken_eligible"]:
                    continue
                targets.append({
                    "schema_version": "phase1063_pair_target.v1",
                    "phase": PHASE,
                    "model": model_name,
                    "target_index": len(targets),
                    "panel": panel_name,
                    "pair_family": family,
                    "site": {
                        "phrase": "source_phrase",
                        "color": "source_color",
                        "noun": "source_noun",
                    }[family],
                    "target_case_index": int(
                        left["semantic_case_index"]
                    ),
                    "cross_case_index": int(
                        right["semantic_case_index"]
                    ),
                    "target_case_key": left["case_key"],
                    "cross_case_key": right["case_key"],
                })
    return cases, targets


def audit_model(
    model_name: str,
    cases: list[dict[str, Any]],
    targets: list[dict[str, Any]],
) -> dict[str, Any]:
    panel_counts = Counter(str(row["panel"]) for row in cases)
    target_counts = Counter(
        (str(row["panel"]), str(row["pair_family"]))
        for row in targets
    )
    expected_panel_counts = {
        name: len(panel["colors"]) * len(panel["nouns"])
        for name, panel in PANELS.items()
    }
    empty_roles = sum(
        int(end) < int(start)
        for row in cases
        for start, end in row["role_spans"].values()
    )
    role_too_wide = sum(
        int(end) - int(start) + 1 > MAX_ROLE_SPAN
        for row in cases
        for start, end in row["role_spans"].values()
    )
    invalid_pairs = 0
    by_index = {
        int(row["semantic_case_index"]): row for row in cases
    }
    for target in targets:
        left = by_index[int(target["target_case_index"])]
        right = by_index[int(target["cross_case_index"])]
        family = str(target["pair_family"])
        valid = left["panel"] == right["panel"]
        if family == "color":
            valid = (
                valid
                and left["noun_id"] == right["noun_id"]
                and left["color_id"] != right["color_id"]
            )
        elif family == "noun":
            valid = (
                valid
                and left["color_id"] == right["color_id"]
                and left["noun_id"] != right["noun_id"]
            )
        elif family == "phrase":
            valid = (
                valid
                and left["color_id"] != right["color_id"]
                and left["noun_id"] != right["noun_id"]
            )
        invalid_pairs += not valid
    all_labels_frozen = all(
        row["acceptable_labels"] and row["acceptable_token_ids"]
        for row in cases
    )
    panel_count_ok = dict(panel_counts) == expected_panel_counts
    target_count_ok = all(
        target_counts[(panel, family)] >= (
            expected_panel_counts[panel] - 2
        )
        for panel in PANELS
        for family in PAIR_FAMILIES
    )
    return {
        "model": model_name,
        "case_count": len(cases),
        "target_count": len(targets),
        "panel_counts": dict(panel_counts),
        "expected_panel_counts": expected_panel_counts,
        "target_counts": {
            f"{panel}.{family}": count
            for (panel, family), count in sorted(target_counts.items())
        },
        "empty_role_count": empty_roles,
        "role_too_wide_count": role_too_wide,
        "invalid_pair_count": invalid_pairs,
        "all_labels_frozen": all_labels_frozen,
        "panel_count_ok": panel_count_ok,
        "target_count_ok": target_count_ok,
        "all_checks_passed": (
            panel_count_ok
            and target_count_ok
            and empty_roles == 0
            and role_too_wide == 0
            and invalid_pairs == 0
            and all_labels_frozen
        ),
    }


def lexical_audit() -> dict[str, Any]:
    anchor_words = {row[0] for row in ANCHOR_NOUNS}
    novel_words = {row[0] for row in NOVEL_NOUNS}
    phase1059_words = {
        row[0]
        for split_rows in heldout.NOUNS.values()
        for row in split_rows
    }
    common_colors = {row[0] for row in COMMON_COLORS}
    ambiguous_colors = {row[0] for row in AMBIGUOUS_COLORS}
    expected_common = {row[0] for row in anchor.COLORS}
    expected_ambiguous = {
        row[0] for row in heldout.COLORS["confirmation"]
    }
    checks = {
        "novel_anchor_noun_overlap": sorted(
            novel_words & anchor_words
        ),
        "novel_phase1059_noun_overlap": sorted(
            novel_words & phase1059_words
        ),
        "common_colors_match_phase1058": (
            common_colors == expected_common
        ),
        "ambiguous_colors_match_phase1059_confirmation": (
            ambiguous_colors == expected_ambiguous
        ),
        "novel_noun_count": len(novel_words),
        "novel_noun_gender_counts": dict(Counter(
            gender
            for _, forms in NOVEL_NOUNS
            for _, gender in forms[:1]
        )),
    }
    checks["all_checks_passed"] = (
        not checks["novel_anchor_noun_overlap"]
        and not checks["novel_phase1059_noun_overlap"]
        and checks["common_colors_match_phase1058"]
        and checks["ambiguous_colors_match_phase1059_confirmation"]
        and checks["novel_noun_count"] == 40
        and checks["novel_noun_gender_counts"] == {"m": 20, "f": 20}
    )
    return checks


def serializable_panels() -> dict[str, Any]:
    return {
        name: {
            "purpose": panel["purpose"],
            "colors": [
                {
                    "english": row[0],
                    "masculine_forms": list(row[1]),
                    "feminine_forms": list(row[2]),
                }
                for row in panel["colors"]
            ],
            "nouns": [
                {
                    "english": row[0],
                    "french_forms": [
                        {"text": text, "gender": gender}
                        for text, gender in row[1]
                    ],
                }
                for row in panel["nouns"]
            ],
        }
        for name, panel in PANELS.items()
    }


def main() -> None:
    source_prereg = read_json(
        SOURCE_ROOT / "protocol" / "preregistration.json"
    )
    source_aggregate = read_json(SOURCE_ROOT / "aggregate.json")
    lexical = lexical_audit()
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
        model_audits[model_name] = audit_model(
            model_name, cases, targets
        )
    audit = {
        "schema_version": "phase1063_protocol_audit.v1",
        "phase": PHASE,
        "lexical": lexical,
        "models": model_audits,
        "all_checks_passed": (
            lexical["all_checks_passed"]
            and all(
                row["all_checks_passed"]
                for row in model_audits.values()
            )
        ),
    }
    write_json(OUT_ROOT / "protocol" / "audit.json", audit)
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1063 protocol audit failed")

    payload = {
        "schema_version": "phase1063_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "models": list(MODELS),
        "sequential_model_order": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "generation_steps": GENERATION_STEPS,
        "pair_families": list(PAIR_FAMILIES),
        "primary_panels": list(PRIMARY_PANELS),
        "surface": dict(SURFACE),
        "assistant_prefill": ASSISTANT_PREFILL,
        "continuation_prefix": CONTINUATION_PREFIX,
        "panels": serializable_panels(),
        "gates": GATES,
        "model_plans": source_prereg["model_plans"],
        "source_phase1062_digest": source_prereg["protocol_digest"],
        "source_phase1062_route": source_aggregate[
            "automatic_next_decision"
        ]["route"],
        "answer_policy": {
            "frozen_before_forward_pass": True,
            "primary_behavior_identity": "normalized_decoded_text",
            "secondary_behavior_identity": "raw_token_sequence",
            "normalization": source_prereg["normalization"],
            "posthoc_synonym_addition": False,
            "edit_distance": False,
        },
        "primary_outcome": (
            "Whether familiar colors crossed with forty entirely new nouns "
            "produce at least 100 behavior-qualified bidirectional pairs "
            "per family in at least two models."
        ),
        "interpretation_limits": [
            "Panel labels describe protocol construction, not corpus frequency.",
            "Behavior differences do not identify an internal mechanism.",
            "Visible-text equality is not unrestricted semantic equivalence.",
            "A later causal rate comparison is not a scaling law by itself.",
            "Brain plasticity and efficiency optimality are not tested.",
        ],
        "automatic_next": {
            "continue_only_if": (
                "At least two models pass both anchor_common and "
                "novel_noun behavior gates."
            ),
            "next_if_pass": (
                "Phase1064 cross-panel K/V transport replication."
            ),
            "next_if_fail": (
                "Stop causal expansion and retain a lexical behavior atlas."
            ),
        },
    }
    payload["protocol_digest"] = digest(payload)
    write_json(
        OUT_ROOT / "protocol" / "preregistration.json",
        payload,
    )
    print(
        f"Phase{PHASE} protocol frozen: "
        f"{payload['protocol_digest']} cases=440/model"
    )


if __name__ == "__main__":
    main()
