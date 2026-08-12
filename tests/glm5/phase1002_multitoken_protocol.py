#!/usr/bin/env python3
"""Freeze the Phase 1002 generation-aligned multi-token protocol.

This module creates a new factorial denominator with names and templates that
were not used by Phase 1001. It freezes model-specific answer surfaces before
any Phase 1002 causal result is observed.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from phase548_shared_attention_compute_protocol import render_chat, tokenizer_for


PHASE = 1002
PROTOCOL_REVISION = 2
MODELS = ("qwen3", "glm4", "deepseek7b")
COLORS = ("red", "blue", "green", "yellow")
NAMES = (
    "Aaron", "Bella", "Chloe", "Ethan", "Fiona", "Gavin", "Hazel", "Isaac",
    "Julia", "Kevin", "Laura", "Mason", "Nina", "Owen", "Paige", "Ryan",
    "Sofia", "Tyler", "Vera", "Wendy", "Xavier", "Zoe", "Amber", "Caleb",
    "Daisy", "Elena", "Felix", "Holly", "Jason", "Kara", "Lucas", "Megan",
    "Noah", "Olivia", "Peter", "Rachel", "Sarah", "Simon", "Victor", "Wyatt",
    "Adam", "Brian", "Cindy", "Derek", "Emily", "George", "Hannah", "Jacob",
    "Katie", "Luke", "Maria", "Nathan", "Penny", "Robert", "Susan", "Thomas",
    "Violet", "Walter",
)
WORLD_COUNT_PER_SPLIT = 32
PAIRS_PER_STRATUM = 4
ANSWER_PREFIX = {
    "qwen3": "",
    "glm4": "\n",
    "deepseek7b": "",
}
STEP_ROLES = {
    "qwen3": (
        "determiner",
        "object_noun",
        "copula",
        "semantic_value",
        "punctuation",
    ),
    "glm4": (
        "protocol_newline",
        "determiner",
        "object_noun",
        "copula",
        "semantic_value",
        "punctuation",
    ),
    "deepseek7b": (
        "determiner",
        "object_noun",
        "copula",
        "semantic_value",
        "punctuation",
    ),
}
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1002_multitoken_scpg_r2"
)
PHASE1001_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1001_attention_physical_decomposition"
    / "cross_model_topology_causal_screen"
)


def canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def stable_order(value: str, salt: str) -> str:
    return hashlib.sha256(
        f"phase1002:{salt}:{value}".encode("utf-8")
    ).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")
    temp.replace(path)


def one_token_id(tokenizer, text: str) -> int:
    values = tokenizer.encode(text, add_special_tokens=False)
    if len(values) != 1:
        raise RuntimeError(f"expected one token for {text!r}, got {values}")
    return int(values[0])


def positions_of(ids: list[int], token_id: int) -> list[int]:
    return [index for index, value in enumerate(ids) if value == token_id]


def render_user_prompt(
    template: int,
    entity0: str,
    color0: str,
    entity1: str,
    color1: str,
    query: str,
) -> str:
    instruction = (
        "Answer with exactly four words in this form: "
        "The marker is [color]. Replace [color] with the answer."
    )
    if template == 0:
        return (
            f"Marker ledger: {entity0} is linked to {color0}; "
            f"{entity1} is linked to {color1}.\n"
            f"What marker color is linked to {query}?\n{instruction}"
        )
    if template == 1:
        return (
            f"Inventory entries say {entity0} has a {color0} marker and "
            f"{entity1} has a {color1} marker.\n"
            f"Report the marker color for {query}.\n{instruction}"
        )
    if template == 2:
        return (
            f"The {color0} marker is assigned to {entity0}. "
            f"The {color1} marker is assigned to {entity1}.\n"
            f"Which marker color is assigned to {query}?\n{instruction}"
        )
    return (
        f"During this trial, the marker assigned to {entity0} is {color0}, "
        f"whereas the marker assigned to {entity1} is {color1}.\n"
        f"Which marker color belongs to {query}?\n{instruction}"
    )


def answer_text(model_name: str, color: str) -> str:
    return f"{ANSWER_PREFIX[model_name]}The marker is {color}."


def select_pairs(
    pairs: list[dict[str, Any]], split: str
) -> list[dict[str, Any]]:
    strata: dict[tuple[int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        if pair["split"] != split:
            continue
        key = (
            int(pair["template"]),
            int(pair["display_order"]),
            int(pair["value_swap"]),
            int(pair["query_role"]),
        )
        strata[key].append(pair)
    selected = []
    for key, values in sorted(strata.items()):
        ordered = sorted(
            values,
            key=lambda row: stable_order(row["pair_id"], f"{split}:{key}"),
        )
        if len(ordered) < PAIRS_PER_STRATUM:
            raise RuntimeError(f"underfilled {split} stratum {key}")
        selected.extend(ordered[:PAIRS_PER_STRATUM])
    expected = 32 * PAIRS_PER_STRATUM
    if len(selected) != expected:
        raise RuntimeError(f"{split}: selected {len(selected)} != {expected}")
    return selected


def frozen_topology(model_name: str) -> dict[str, Any]:
    path = PHASE1001_ROOT / model_name / "frozen_topology.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    primary_ids = list(value["joint_event_ids"])
    variants = {"primary": primary_ids}
    if model_name == "qwen3":
        variants["preregistered_k2_secondary"] = [
            item["event_id"] for item in value["ranked_receivers"][:2]
        ]
    return {
        "source_phase": 1001,
        "source_depth": int(value["source_depth"]),
        "variants": variants,
        "selection_uses_phase1002": False,
        "source_digest": digest(value),
    }


def build_model(model_name: str) -> dict[str, Any]:
    tokenizer = tokenizer_for(model_name)
    prompt_name_ids = {
        name: one_token_id(tokenizer, " " + name) for name in NAMES
    }
    prompt_color_ids = {
        color: one_token_id(tokenizer, " " + color) for color in COLORS
    }
    answer_ids = {
        color: [
            int(value)
            for value in tokenizer.encode(
                answer_text(model_name, color), add_special_tokens=False
            )
        ]
        for color in COLORS
    }
    step_roles = list(STEP_ROLES[model_name])
    if any(len(values) != len(step_roles) for values in answer_ids.values()):
        raise RuntimeError(f"{model_name}: answer step count drift {answer_ids}")
    color_step = step_roles.index("semantic_value")
    candidate_ids = {
        color: int(answer_ids[color][color_step]) for color in COLORS
    }
    for step in range(len(step_roles)):
        values = {answer_ids[color][step] for color in COLORS}
        expected = 4 if step == color_step else 1
        if len(values) != expected:
            raise RuntimeError(
                f"{model_name}: answer role drift step={step}, values={values}"
            )

    rng = random.Random(1002_20260723)
    name_pairs = list(itertools.combinations(NAMES, 2))
    rng.shuffle(name_pairs)
    color_pairs = list(itertools.combinations(COLORS, 2))
    total_worlds = WORLD_COUNT_PER_SPLIT * 2
    cases: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    token_lengths: dict[int, set[int]] = defaultdict(set)

    for world in range(total_worlds):
        split = (
            "discovery" if world < WORLD_COUNT_PER_SPLIT else "confirmation"
        )
        split_world = (
            world
            if split == "discovery"
            else world - WORLD_COUNT_PER_SPLIT
        )
        world_id = f"{split[:1]}w{split_world:02d}"
        base_entities = list(name_pairs[world])
        base_colors = list(color_pairs[world % len(color_pairs)])
        for template, display_order, value_swap, query_role in itertools.product(
            range(4), (0, 1), (0, 1), (0, 1)
        ):
            arms = []
            for entity_swap in (0, 1):
                slot_entities = (
                    list(base_entities)
                    if entity_swap == 0
                    else [base_entities[1], base_entities[0]]
                )
                slot_colors = (
                    list(base_colors)
                    if value_swap == 0
                    else [base_colors[1], base_colors[0]]
                )
                query_entity = base_entities[query_role]
                query_slot = slot_entities.index(query_entity)
                gold = slot_colors[query_slot]
                foil = slot_colors[1 - query_slot]
                first_slot, second_slot = (
                    (0, 1) if display_order == 0 else (1, 0)
                )
                raw_prompt = render_user_prompt(
                    template,
                    slot_entities[first_slot],
                    slot_colors[first_slot],
                    slot_entities[second_slot],
                    slot_colors[second_slot],
                    query_entity,
                )
                rendered = render_chat(tokenizer, model_name, raw_prompt)
                ids = [
                    int(value)
                    for value in tokenizer.encode(
                        rendered, add_special_tokens=False
                    )
                ]
                expected_answer = answer_ids[gold]
                extended = [
                    int(value)
                    for value in tokenizer.encode(
                        rendered + answer_text(model_name, gold),
                        add_special_tokens=False,
                    )
                ]
                if extended != ids + expected_answer:
                    raise RuntimeError(
                        f"{model_name}: answer boundary drift "
                        f"{world_id}/t{template}/{gold}"
                    )

                fact_positions = {}
                for entity in base_entities:
                    found = positions_of(ids, prompt_name_ids[entity])
                    expected_count = 2 if entity == query_entity else 1
                    if len(found) != expected_count:
                        raise RuntimeError(
                            f"{model_name}: entity position drift "
                            f"{world_id}/{entity}/{found}"
                        )
                    fact_positions[entity] = found[0]
                query_positions = positions_of(
                    ids, prompt_name_ids[query_entity]
                )
                color_positions = {
                    color: positions_of(ids, prompt_color_ids[color])
                    for color in base_colors
                }
                if any(len(values) != 1 for values in color_positions.values()):
                    raise RuntimeError(
                        f"{model_name}: color position drift "
                        f"{world_id}/{color_positions}"
                    )

                record_id = (
                    f"{model_name}.{world_id}.t{template}.o{display_order}."
                    f"v{value_swap}.q{query_role}.e{entity_swap}"
                )
                row = {
                    "schema_version": "phase1002_multitoken_case.v1",
                    "phase": PHASE,
                    "model": model_name,
                    "record_id": record_id,
                    "world": world,
                    "world_id": world_id,
                    "split": split,
                    "template": template,
                    "display_order": display_order,
                    "value_swap": value_swap,
                    "query_role": query_role,
                    "entity_swap": entity_swap,
                    "base_entities": base_entities,
                    "base_colors": base_colors,
                    "slot_entities": slot_entities,
                    "slot_colors": slot_colors,
                    "query_entity": query_entity,
                    "query_slot": query_slot,
                    "gold": gold,
                    "foil": foil,
                    "raw_prompt": raw_prompt,
                    "rendered_prompt": rendered,
                    "input_ids": ids,
                    "input_token_count": len(ids),
                    "answer_text": answer_text(model_name, gold),
                    "answer_token_ids": expected_answer,
                    "answer_step_roles": step_roles,
                    "semantic_step": color_step,
                    "candidate_token_ids": candidate_ids,
                    "role_positions": {
                        "slot0_entity": fact_positions[slot_entities[0]],
                        "slot0_color": color_positions[slot_colors[0]][0],
                        "slot1_entity": fact_positions[slot_entities[1]],
                        "slot1_color": color_positions[slot_colors[1]][0],
                        "query_name": query_positions[-1],
                        "answer_boundary": len(ids) - 1,
                    },
                }
                cases.append(row)
                arms.append(row)
                token_lengths[template].add(len(ids))

            arm0, arm1 = arms
            changed = [
                index
                for index, (left, right) in enumerate(
                    zip(arm0["input_ids"], arm1["input_ids"])
                )
                if left != right
            ]
            expected_changed = sorted((
                arm0["role_positions"]["slot0_entity"],
                arm0["role_positions"]["slot1_entity"],
            ))
            if changed != expected_changed:
                raise RuntimeError(
                    f"{model_name}: counterfactual drift "
                    f"{world_id}/{changed}/{expected_changed}"
                )
            if arm0["gold"] != arm1["foil"] or arm1["gold"] != arm0["foil"]:
                raise RuntimeError(f"{model_name}: answer swap drift {world_id}")
            pair_id = (
                f"{model_name}.{world_id}.t{template}.o{display_order}."
                f"v{value_swap}.q{query_role}"
            )
            pairs.append({
                "schema_version": "phase1002_multitoken_pair.v1",
                "phase": PHASE,
                "model": model_name,
                "pair_id": pair_id,
                "factor": "entity",
                "split": split,
                "world_id": world_id,
                "template": template,
                "display_order": display_order,
                "value_swap": value_swap,
                "query_role": query_role,
                "arm0_record_id": arm0["record_id"],
                "arm1_record_id": arm1["record_id"],
                "changed_positions": changed,
            })

    if any(len(values) != 1 for values in token_lengths.values()):
        raise RuntimeError(
            f"{model_name}: template prompt length drift {token_lengths}"
        )
    discovery_selected = select_pairs(pairs, "discovery")
    confirmation_selected = select_pairs(pairs, "confirmation")
    model_root = OUT_ROOT / "protocol" / model_name
    write_jsonl(model_root / "cases.jsonl", cases)
    write_jsonl(model_root / "pairs.jsonl", pairs)
    write_jsonl(
        model_root / "discovery_selected_pairs.jsonl", discovery_selected
    )
    write_jsonl(
        model_root / "confirmation_selected_pairs.jsonl",
        confirmation_selected,
    )
    audit = {
        "schema_version": "phase1002_multitoken_protocol_audit.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "model": model_name,
        "case_count": len(cases),
        "pair_count": len(pairs),
        "case_count_per_split": len(cases) // 2,
        "pair_count_per_split": len(pairs) // 2,
        "selected_pair_count_per_split": len(discovery_selected),
        "selected_direction_count_per_split": 2 * len(discovery_selected),
        "world_count_per_split": WORLD_COUNT_PER_SPLIT,
        "new_name_count": len(NAMES),
        "answer_prefix": ANSWER_PREFIX[model_name],
        "answer_ids": answer_ids,
        "answer_step_roles": step_roles,
        "semantic_step": color_step,
        "candidate_token_ids": candidate_ids,
        "template_prompt_lengths": {
            str(key): sorted(values)
            for key, values in token_lengths.items()
        },
        "all_counterfactuals_change_only_two_entity_tokens": True,
        "all_counterfactuals_swap_gold_and_foil": True,
        "case_digest": digest(cases),
        "pair_digest": digest(pairs),
        "discovery_selection_digest": digest(discovery_selected),
        "confirmation_selection_digest": digest(confirmation_selected),
    }
    write_json(model_root / "protocol_audit.json", audit)
    return audit


def main() -> None:
    audits = {model_name: build_model(model_name) for model_name in MODELS}
    topology = {
        model_name: frozen_topology(model_name)
        for model_name in MODELS
    }
    preregistration = {
        "schema_version": "phase1002_preregistered_protocol.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "revision_audit": {
            "r1_result": (
                "Qwen3 failed the behavior gate because the color-to-entity "
                "arrow notation in template 2 was ambiguous."
            ),
            "repair_scope": "Template 2 wording only.",
            "repair_selection": (
                "Three explicit color-first variants were tested on eight "
                "calibration-only examples per model. All variants scored "
                "8/8 on qwen3, glm4, and deepseek7b; the first preregistered "
                "variant was selected."
            ),
            "internal_results_observed_before_repair": False,
            "r1_results_retained": True,
        },
        "research_order": (
            "new denominator -> behavior gate -> frozen Phase1001 topology "
            "replication -> temporal controls -> only then new discovery"
        ),
        "calibration_data_reused_in_formal_test": False,
        "models": list(MODELS),
        "protocol_audits": audits,
        "frozen_phase1001_topology": topology,
        "primary_thresholds": {
            "clean_candidate_accuracy": 0.95,
            "clean_exact_sentence_rate": 0.95,
            "source_do_semantic_flip_rate": 0.80,
            "frozen_topology_semantic_restore_rate": 0.50,
            "semantic_mediation_median": 0.30,
            "wrong_step_restore_target_max": 0.10,
            "cache_full_recompute_token_agreement": 0.99,
            "cross_model_minimum_pass_count": 2,
        },
        "non_gates": {
            "minimum_event_count": (
                "No requirement that a multi-token cut be smaller than the "
                "Phase1001 six-head set."
            ),
            "nonsemantic_steps": (
                "No requirement that entity binding alter protocol tokens. "
                "Temporal localization is reported rather than forced."
            ),
            "theory_formula": (
                "No global formula is fitted or used for event selection."
            ),
        },
        "claim_boundary": (
            "The primary test reuses only Phase1001-frozen functional events "
            "on new Phase1002 data. It tests repeated topology, not shared "
            "neurons or a language law."
        ),
        "preregistration_digest": None,
    }
    preregistration["preregistration_digest"] = digest({
        key: value
        for key, value in preregistration.items()
        if key != "preregistration_digest"
    })
    write_json(OUT_ROOT / "preregistered_protocol.json", preregistration)
    print(json.dumps(preregistration, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
