#!/usr/bin/env python3
"""Freeze a fresh translation trajectory and phase-window protocol."""

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
import phase1056_translation_phase_coalition_protocol as source


PHASE = 1057
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
    / "phase1057_translation_trajectory"
)
SURFACES = (
    {
        "template": (
            "Translate this English word into French. Return exactly one "
            "French word and nothing else.\nEnglish word: {term}"
        ),
        "operator": "Translate",
        "target_language": "French",
    },
    {
        "template": (
            "Convert the English lexical item below to French. Give only "
            "the French equivalent.\nSource item: {term}"
        ),
        "operator": "Convert",
        "target_language": "French",
    },
    {
        "template": (
            "Provide the French equivalent of this English word. Output "
            "one word only.\nWord: {term}"
        ),
        "operator": "equivalent",
        "target_language": "French",
    },
)
ASSISTANT_PREFILL = "Translation:"
CONTINUATION_PREFIX = " "
PAIR_OFFSET = 1
MAX_SOURCE_SPAN = bridge.MAX_ROLE_SPAN
TRAJECTORY_PAIR_LIMIT = 120
ROLLOUT_PAIR_LIMIT = 48
ROLLOUT_STEPS = 8
GATES = {
    "discovery_clean_pair_count_min": 100,
    "confirmation_clean_pair_count_min": 100,
    "post_kv_both_counterfactual_rate_min": 0.50,
    "post_kv_both_counterfactual_count_min": 50,
    "source_minus_control_rate_min": 0.30,
    "trajectory_pair_count_min": 80,
    "rollout_pair_count_min": 30,
    "eos_censored_both_match_rate_min": 0.60,
    "minimum_repeated_models": 2,
}


# None of these 120 English source concepts occurs in Phase1055.
# Alternating rows within each category create balanced lexical splits.
CONCEPTS = (
    ("food", "bread", "pain"),
    ("food", "cheese", "fromage"),
    ("food", "milk", "lait"),
    ("food", "water", "eau"),
    ("food", "coffee", "café"),
    ("food", "tea", "thé"),
    ("food", "sugar", "sucre"),
    ("food", "salt", "sel"),
    ("food", "rice", "riz"),
    ("food", "meat", "viande"),
    ("food", "fish", "poisson"),
    ("food", "egg", "œuf"),
    ("animal", "bird", "oiseau"),
    ("animal", "cow", "vache"),
    ("animal", "pig", "cochon"),
    ("animal", "sheep", "mouton"),
    ("animal", "goat", "chèvre"),
    ("animal", "mouse", "souris"),
    ("animal", "bear", "ours"),
    ("animal", "wolf", "loup"),
    ("animal", "fox", "renard"),
    ("animal", "monkey", "singe"),
    ("animal", "snake", "serpent"),
    ("animal", "frog", "grenouille"),
    ("object", "pen", "stylo"),
    ("object", "pencil", "crayon"),
    ("object", "paper", "papier"),
    ("object", "window", "fenêtre"),
    ("object", "door", "porte"),
    ("object", "mirror", "miroir"),
    ("object", "cup", "tasse"),
    ("object", "plate", "assiette"),
    ("object", "fork", "fourchette"),
    ("object", "spoon", "cuillère"),
    ("object", "knife", "couteau"),
    ("object", "bag", "sac"),
    ("place", "house", "maison"),
    ("place", "castle", "château"),
    ("place", "road", "route"),
    ("place", "street", "rue"),
    ("place", "bridge", "pont"),
    ("place", "river", "rivière"),
    ("place", "mountain", "montagne"),
    ("place", "sea", "mer"),
    ("place", "lake", "lac"),
    ("place", "forest", "forêt"),
    ("place", "beach", "plage"),
    ("place", "church", "église"),
    ("nature", "sun", "soleil"),
    ("nature", "moon", "lune"),
    ("nature", "star", "étoile"),
    ("nature", "sky", "ciel"),
    ("nature", "cloud", "nuage"),
    ("nature", "rain", "pluie"),
    ("nature", "snow", "neige"),
    ("nature", "wind", "vent"),
    ("nature", "fire", "feu"),
    ("nature", "earth", "terre"),
    ("nature", "stone", "pierre"),
    ("nature", "flower", "fleur"),
    ("body", "arm", "bras"),
    ("body", "leg", "jambe"),
    ("body", "hair", "cheveux"),
    ("body", "face", "visage"),
    ("body", "neck", "cou"),
    ("body", "back", "dos"),
    ("body", "finger", "doigt"),
    ("body", "tooth", "dent"),
    ("body", "knee", "genou"),
    ("body", "shoulder", "épaule"),
    ("body", "blood", "sang"),
    ("body", "skin", "peau"),
    ("clothing", "shirt", "chemise"),
    ("clothing", "shoe", "chaussure"),
    ("clothing", "hat", "chapeau"),
    ("clothing", "coat", "manteau"),
    ("clothing", "dress", "robe"),
    ("clothing", "skirt", "jupe"),
    ("clothing", "sock", "chaussette"),
    ("clothing", "glove", "gant"),
    ("clothing", "belt", "ceinture"),
    ("clothing", "pocket", "poche"),
    ("clothing", "trousers", "pantalon"),
    ("clothing", "sweater", "pull"),
    ("person", "mother", "mère"),
    ("person", "father", "père"),
    ("person", "brother", "frère"),
    ("person", "sister", "sœur"),
    ("person", "child", "enfant"),
    ("person", "baby", "bébé"),
    ("person", "friend", "ami"),
    ("person", "king", "roi"),
    ("person", "queen", "reine"),
    ("person", "student", "étudiant"),
    ("person", "writer", "écrivain"),
    ("person", "singer", "chanteur"),
    ("home", "bed", "lit"),
    ("home", "kitchen", "cuisine"),
    ("home", "room", "chambre"),
    ("home", "wall", "mur"),
    ("home", "floor", "sol"),
    ("home", "roof", "toit"),
    ("home", "stairs", "escalier"),
    ("home", "garden", "jardin"),
    ("home", "drawer", "tiroir"),
    ("home", "soap", "savon"),
    ("home", "towel", "serviette"),
    ("home", "basket", "panier"),
    ("time", "day", "jour"),
    ("time", "night", "nuit"),
    ("time", "morning", "matin"),
    ("time", "evening", "soir"),
    ("time", "week", "semaine"),
    ("time", "month", "mois"),
    ("time", "year", "année"),
    ("time", "spring", "printemps"),
    ("time", "summer", "été"),
    ("time", "autumn", "automne"),
    ("time", "winter", "hiver"),
    ("time", "hour", "heure"),
)


write_json = material.write_json
write_jsonl = material.write_jsonl
read_json = material.read_json
read_jsonl = material.read_jsonl
digest = material.digest


def split_concepts() -> list[dict[str, str]]:
    category_offsets: dict[str, int] = defaultdict(int)
    rows = []
    for category, source_term, target_term in CONCEPTS:
        local_index = category_offsets[category]
        category_offsets[category] += 1
        rows.append({
            "concept_id": source_term,
            "category": category,
            "split": (
                "discovery" if local_index % 2 == 0
                else "confirmation"
            ),
            "source_term": source_term,
            "target_term": target_term,
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


def build_model_cases(
    model_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tokenizer = tokenizer_for(model_name)
    cases = []
    for concept in split_concepts():
        for surface_index, surface in enumerate(SURFACES):
            source_term = concept["source_term"]
            target_term = concept["target_term"]
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
            role_spans["selected_concept"] = list(
                role_spans["source_term"]
            )
            target_ids = behavior.continuation_ids(
                tokenizer,
                rendered,
                CONTINUATION_PREFIX,
                target_term,
            )
            cases.append({
                "schema_version": "phase1057_model_case.v1",
                "phase": PHASE,
                "model": model_name,
                "semantic_case_index": len(cases),
                "case_key": (
                    f"{concept['split']}.s{surface_index}."
                    f"{concept['concept_id']}"
                ),
                "split": concept["split"],
                "surface_index": surface_index,
                "concept_id": concept["concept_id"],
                "category": concept["category"],
                "source_term": source_term,
                "expected_label": target_term,
                "rendered_prompt": rendered,
                "input_ids": input_ids,
                "role_spans": role_spans,
                "expected_token_ids": target_ids,
                "expected_first_token_id": int(target_ids[0]),
            })

    targets = []
    panels: dict[tuple[str, int, int], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in cases:
        start, end = row["role_spans"]["source_term"]
        panels[
            (
                str(row["split"]),
                int(row["surface_index"]),
                int(end) - int(start) + 1,
            )
        ].append(row)
    for panel, rows in sorted(panels.items()):
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
                "schema_version": "phase1057_target.v1",
                "phase": PHASE,
                "model": model_name,
                "target_index": len(targets),
                "split": panel[0],
                "surface_index": panel[1],
                "source_token_count": panel[2],
                "target_case_index": int(
                    left["semantic_case_index"]
                ),
                "cross_case_index": int(
                    right["semantic_case_index"]
                ),
                "target_concept_id": str(left["concept_id"]),
                "cross_concept_id": str(right["concept_id"]),
                "target_expected_label": str(left["expected_label"]),
                "cross_expected_label": str(right["expected_label"]),
            })
    return cases, targets


def main() -> None:
    source_aggregate = read_json(SOURCE_ROOT / "aggregate.json")
    if source_aggregate["automatic_next_decision"][
        "should_continue_automatically"
    ]:
        raise RuntimeError(
            "Phase1057 is a user-authorized diagnostic after a stop gate"
        )
    source_prereg = read_json(
        SOURCE_ROOT / "protocol" / "preregistration.json"
    )
    old_sources = {
        str(row["concept_id"]) for row in lexical.CONCEPTS
    }
    new_sources = {row["concept_id"] for row in split_concepts()}
    if old_sources & new_sources:
        raise RuntimeError(
            f"Phase1057 lexical overlap: {old_sources & new_sources}"
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
        source_summary = read_json(
            SOURCE_ROOT / "atlas" / model_name / "summary.json"
        )
        all_layers = [
            int(value) for value in source_plan["all_layers"]
        ]
        early_depths = [
            int(value) for value in source_plan["early_depths"]
        ]
        post_depths = [
            int(value)
            for value in source_plan["all_postsource_depths"]
        ]
        slots = [
            [int(value) for value in slot]
            for slot in source_plan["depth_slots"]
        ]
        model_plans[model_name] = {
            "n_layers": len(all_layers),
            "n_kv_heads": int(source_plan["n_kv_heads"]),
            "all_groups": [
                int(value) for value in source_plan["all_groups"]
            ],
            "all_layers": all_layers,
            "early_depths": early_depths,
            "postsource_depths": post_depths,
            "postsource_slots": slots,
            "frozen_groups": [
                int(value)
                for value in source_summary["frozen_groups"]
            ],
            "frozen_depths": [
                int(value)
                for value in source_summary["frozen_depths"]
            ],
            "trajectory_pair_limit": TRAJECTORY_PAIR_LIMIT,
        }
        counts = Counter(str(row["split"]) for row in targets)
        split_ids = {
            split: {
                str(value)
                for row in targets
                if row["split"] == split
                for value in (
                    row["target_concept_id"],
                    row["cross_concept_id"],
                )
            }
            for split in ("discovery", "confirmation")
        }
        widths = [
            int(end) - int(start) + 1
            for row in cases
            for start, end in [row["role_spans"]["source_term"]]
        ]
        answer_leaks = [
            row["case_key"]
            for row in cases
            if str(row["expected_label"]).casefold()
            in str(row["rendered_prompt"]).casefold()
        ]
        model_audits[model_name] = {
            "case_count": len(cases),
            "target_counts": dict(counts),
            "maximum_source_span": max(widths),
            "discovery_confirmation_overlap": sorted(
                split_ids["discovery"] & split_ids["confirmation"]
            ),
            "distinct_concepts": len(
                split_ids["discovery"] | split_ids["confirmation"]
            ),
            "answer_leak_count": len(answer_leaks),
            "answer_leak_examples": answer_leaks[:5],
        }

    audit = {
        "schema_version": "phase1057_protocol_audit.v1",
        "phase": PHASE,
        "new_concept_count": len(new_sources),
        "old_source_overlap": sorted(old_sources & new_sources),
        "models": model_audits,
    }
    audit["all_checks_passed"] = (
        len(new_sources) == 120
        and not audit["old_source_overlap"]
        and all(
            row["maximum_source_span"] <= MAX_SOURCE_SPAN
            and not row["discovery_confirmation_overlap"]
            and row["distinct_concepts"] == 120
            and row["answer_leak_count"] == 0
            and min(row["target_counts"].values()) >= 150
            for row in model_audits.values()
        )
    )
    write_json(OUT_ROOT / "protocol" / "audit.json", audit)
    if not audit["all_checks_passed"]:
        raise RuntimeError(f"Phase1057 protocol audit failed: {audit}")

    payload = {
        "schema_version": "phase1057_preregistration.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "source_phase1056_digest": source_prereg["protocol_digest"],
        "source_phase1056_route": source_aggregate[
            "automatic_next_decision"
        ],
        "models": list(MODELS),
        "precision": PRECISION,
        "quantization": QUANTIZATION,
        "sequential_model_order": list(MODELS),
        "language_direction": "English_to_French",
        "fresh_concept_count": len(new_sources),
        "lexical_split_rule": (
            "Alternate concepts within each of ten semantic categories; "
            "no English source concept appeared in Phase1055."
        ),
        "surfaces": [dict(row) for row in SURFACES],
        "assistant_prefill": ASSISTANT_PREFILL,
        "continuation_prefix": CONTINUATION_PREFIX,
        "model_plans": model_plans,
        "condition_families": {
            "phase": [
                "source_early_kv",
                "source_post_kv",
                "source_all_kv",
            ],
            "channels": [
                "source_post_k_only",
                "source_post_v_only",
                "source_post_kv",
            ],
            "slots": (
                "Each normalized postsource slot alone, plus early depths "
                "union cumulative postsource prefixes."
            ),
            "frozen_rectangle": "Phase1056 frozen groups and depths",
            "role_controls": [
                "operator_post_kv",
                "target_language_post_kv",
            ],
        },
        "trajectory_objects": [
            "source_position_KV",
            "answer_boundary_residual",
        ],
        "trajectory_conditions": [
            "clean",
            "source_early_kv",
            "source_post_kv",
            "source_all_kv",
        ],
        "trajectory_pair_limit": TRAJECTORY_PAIR_LIMIT,
        "rollout_pair_limit": ROLLOUT_PAIR_LIMIT,
        "rollout_steps": ROLLOUT_STEPS,
        "gates": GATES,
        "automatic_next": {
            "fresh_bridge_repeats_and_one_rollout_passes": (
                "phase1058_multitoken_translation"
            ),
            "otherwise": "stop_with_unstable_fresh_translation_bridge",
        },
        "interpretation_limits": [
            "A donor-closer downstream state is a trajectory observation.",
            "Downstream change alone does not establish a natural gate.",
            "K/V replacement can create off-manifold hybrid states.",
            "This is lexical translation, not sentence translation.",
            "A graph cut is sufficient under intervention, not unique.",
            "No result tests brain optimality or biological homology.",
        ],
    }
    payload["protocol_digest"] = digest(payload)
    write_json(
        OUT_ROOT / "protocol" / "preregistration.json",
        payload,
    )
    print(
        f"Phase{PHASE} protocol frozen: {payload['protocol_digest']} "
        f"models={len(MODELS)} concepts={len(new_sources)}"
    )


if __name__ == "__main__":
    main()
