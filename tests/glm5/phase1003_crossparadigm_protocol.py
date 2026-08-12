#!/usr/bin/env python3
"""Freeze the Phase 1003 cross-paradigm causal-map denominator.

The protocol is deliberately role based rather than formula based.  It creates
four independent two-fact query tasks with disjoint discovery/confirmation
entity vocabularies.  Internal coordinates are not observed here.
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
from phase1002_multitoken_protocol import NAMES as PHASE1002_NAMES


PHASE = 1003
PROTOCOL_REVISION = 4
MODELS = ("qwen3", "glm4", "deepseek7b")
DOMAINS = {
    "color": ("purple", "orange", "black", "white"),
    "shape": ("circle", "square", "triangle", "diamond"),
    "size": ("tiny", "small", "large", "huge"),
    "category": ("fruit", "animal", "vehicle", "tool"),
}
DISCOVERY_NAMES = (
    "Alice", "Amy", "Andrew", "Angela", "Anna", "Arthur", "Blake",
    "Brandon", "Bruce", "Carla", "Carol", "Charles", "Daniel", "David",
    "Diana", "Donna", "Edward", "Eric", "Eva", "Frank", "Grace", "Henry",
    "Ian", "Irene", "Jack", "Jane", "Jennifer", "John", "Joseph", "Karen",
    "Kelly", "Liam",
)
CONFIRMATION_NAMES = (
    "Linda", "Lisa", "Mark", "Martin", "Michael", "Michelle", "Mike",
    "Nancy", "Neil", "Nicole", "Pamela", "Paul", "Philip", "Rebecca",
    "Richard", "Robin", "Sam", "Samuel", "Sandra", "Scott", "Sean",
    "Steven", "Tina", "Todd", "Tony", "Tracy", "William", "Atlas",
    "Beacon", "Cedar", "Delta", "Ember",
)
CALIBRATION_NAMES = (
    "Aster", "Birch", "Coral", "Elm", "Flint", "Grove", "Heath", "Iris",
)
WORLD_COUNT_PER_SPLIT = 16
SELECTED_PAIRS_PER_STRATUM = 1
ANCHOR_ROLES = (
    "slot0_entity",
    "slot0_value",
    "slot1_entity",
    "slot1_value",
    "query_name",
)
ANSWER_PREFIX = {
    "qwen3": "",
    "glm4": "\n",
    "deepseek7b": "",
}
OUT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1003_crossparadigm_causal_map"
)
PHASE1002_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1002_multitoken_scpg_r2"
)


def canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def stable_order(value: str, salt: str) -> str:
    return hashlib.sha256(
        f"phase1003:{salt}:{value}".encode("utf-8")
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def one_token_id(tokenizer, text: str) -> int:
    values = tokenizer.encode(text, add_special_tokens=False)
    if len(values) != 1:
        raise RuntimeError(f"expected one token for {text!r}, got {values}")
    return int(values[0])


def positions_of(ids: list[int], token_id: int) -> list[int]:
    return [index for index, value in enumerate(ids) if value == token_id]


def render_user_prompt(
    template: int,
    domain: str,
    entity0: str,
    value0: str,
    entity1: str,
    value1: str,
    query: str,
) -> str:
    instruction = (
        "Answer exactly in this form: Answer: [value] Replace [value] with "
        "the lowercase answer. Do not add punctuation or other words."
    )
    if template == 0:
        body = (
            f"Registry: {entity0} has {domain} {value0}; "
            f"{entity1} has {domain} {value1}.\n"
            f"What is the {domain} of {query}?"
        )
    elif template == 1:
        body = (
            f"The recorded {domain} for {entity0} is {value0}. "
            f"The recorded {domain} for {entity1} is {value1}.\n"
            f"Report the {domain} of {query}."
        )
    elif template == 2:
        body = (
            f"For {domain}, {value0} is assigned to {entity0}, and "
            f"{value1} is assigned to {entity1}.\n"
            f"Which {domain} is assigned to {query}?"
        )
    else:
        body = (
            f"Entry one records {entity0} with {value0} as the {domain}. "
            f"Entry two records {entity1} with {value1} as the {domain}.\n"
            f"Give the {domain} recorded for {query}."
        )
    return f"{body}\n{instruction}"


def answer_text(model_name: str, value: str) -> str:
    return f"{ANSWER_PREFIX[model_name]}Answer: {value}"


def select_pairs(
    pairs: list[dict[str, Any]],
    domain: str,
    split: str,
) -> list[dict[str, Any]]:
    strata: dict[tuple[int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        if pair["domain"] != domain or pair["split"] != split:
            continue
        key = (
            int(pair["template"]),
            int(pair["display_order"]),
            int(pair["value_swap"]),
            int(pair["query_role"]),
        )
        strata[key].append(pair)
    selected: list[dict[str, Any]] = []
    for key, values in sorted(strata.items()):
        ordered = sorted(
            values,
            key=lambda row: stable_order(
                row["pair_id"], f"{domain}:{split}:{key}"
            ),
        )
        if len(ordered) < SELECTED_PAIRS_PER_STRATUM:
            raise RuntimeError(f"underfilled {domain}/{split} stratum {key}")
        selected.extend(ordered[:SELECTED_PAIRS_PER_STRATUM])
    expected = 32 * SELECTED_PAIRS_PER_STRATUM
    if len(selected) != expected:
        raise RuntimeError(
            f"{domain}/{split}: selected {len(selected)} != {expected}"
        )
    return selected


def selected_directional_rows(
    model_name: str,
    domain: str,
    split: str,
) -> list[dict[str, Any]]:
    model_root = OUT_ROOT / "protocol" / model_name
    cases = {
        row["record_id"]: row
        for row in read_jsonl(model_root / "cases.jsonl")
    }
    pairs = read_jsonl(
        model_root / f"{domain}_{split}_selected_pairs.jsonl"
    )
    result = []
    for pair in pairs:
        arm0 = cases[pair["arm0_record_id"]]
        arm1 = cases[pair["arm1_record_id"]]
        for direction, source, target in (
            ("arm0_to_arm1", arm0, arm1),
            ("arm1_to_arm0", arm1, arm0),
        ):
            result.append({
                "pair_id": pair["pair_id"],
                "model": model_name,
                "domain": domain,
                "split": split,
                "direction": direction,
                "template": pair["template"],
                "source": source,
                "target": target,
            })
    return result


def build_model(model_name: str) -> dict[str, Any]:
    if set(DISCOVERY_NAMES) & set(CONFIRMATION_NAMES):
        raise RuntimeError("discovery/confirmation name overlap")
    if (
        (set(DISCOVERY_NAMES) | set(CONFIRMATION_NAMES))
        & set(CALIBRATION_NAMES)
    ):
        raise RuntimeError("formal/calibration name overlap")
    if (
        (set(DISCOVERY_NAMES) | set(CONFIRMATION_NAMES))
        & set(PHASE1002_NAMES)
    ):
        raise RuntimeError("Phase1002/Phase1003 formal name overlap")
    tokenizer = tokenizer_for(model_name)
    all_names = (
        DISCOVERY_NAMES + CONFIRMATION_NAMES + CALIBRATION_NAMES
    )
    prompt_name_ids = {
        name: one_token_id(tokenizer, " " + name) for name in all_names
    }
    prompt_value_ids = {
        domain: {
            value: one_token_id(tokenizer, " " + value)
            for value in values
        }
        for domain, values in DOMAINS.items()
    }
    answer_ids = {
        domain: {
            value: [
                int(token_id)
                for token_id in tokenizer.encode(
                    answer_text(model_name, value),
                    add_special_tokens=False,
                )
            ]
            for value in values
        }
        for domain, values in DOMAINS.items()
    }
    semantic_steps = {}
    candidate_ids = {}
    answer_step_counts = {}
    for domain, by_value in answer_ids.items():
        lengths = {len(token_ids) for token_ids in by_value.values()}
        if len(lengths) != 1:
            raise RuntimeError(
                f"{model_name}/{domain}: answer length drift {by_value}"
            )
        width = next(iter(lengths))
        varying_steps = [
            step
            for step in range(width)
            if len({ids[step] for ids in by_value.values()}) > 1
        ]
        if len(varying_steps) != 1:
            raise RuntimeError(
                f"{model_name}/{domain}: semantic step drift {varying_steps}"
            )
        step = varying_steps[0]
        if len({ids[step] for ids in by_value.values()}) != len(by_value):
            raise RuntimeError(
                f"{model_name}/{domain}: candidate token collision"
            )
        semantic_steps[domain] = step
        candidate_ids[domain] = {
            value: int(ids[step]) for value, ids in by_value.items()
        }
        answer_step_counts[domain] = width

    rng = random.Random(1003_20260723)
    split_worlds = {}
    for split, names in (
        ("discovery", DISCOVERY_NAMES),
        ("confirmation", CONFIRMATION_NAMES),
    ):
        shuffled_names = list(names)
        rng.shuffle(shuffled_names)
        split_worlds[split] = [
            tuple(shuffled_names[index : index + 2])
            for index in range(0, len(shuffled_names), 2)
        ]
        if len(split_worlds[split]) != WORLD_COUNT_PER_SPLIT:
            raise RuntimeError(f"{split}: world count drift")

    cases: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    token_lengths: dict[tuple[str, int], set[int]] = defaultdict(set)
    value_pairs = {
        domain: list(itertools.combinations(values, 2))
        for domain, values in DOMAINS.items()
    }
    for domain, values in DOMAINS.items():
        for split, worlds in split_worlds.items():
            for world_index, name_pair in enumerate(worlds):
                world_id = f"{domain}.{split[:1]}w{world_index:02d}"
                base_entities = list(name_pair)
                base_values = list(
                    value_pairs[domain][
                        world_index % len(value_pairs[domain])
                    ]
                )
                for (
                    template,
                    display_order,
                    value_swap,
                    query_role,
                ) in itertools.product(range(4), (0, 1), (0, 1), (0, 1)):
                    arms = []
                    for entity_swap in (0, 1):
                        slot_entities = (
                            list(base_entities)
                            if entity_swap == 0
                            else [base_entities[1], base_entities[0]]
                        )
                        slot_values = (
                            list(base_values)
                            if value_swap == 0
                            else [base_values[1], base_values[0]]
                        )
                        query_entity = base_entities[query_role]
                        query_slot = slot_entities.index(query_entity)
                        gold = slot_values[query_slot]
                        foil = slot_values[1 - query_slot]
                        first_slot, second_slot = (
                            (0, 1) if display_order == 0 else (1, 0)
                        )
                        raw_prompt = render_user_prompt(
                            template,
                            domain,
                            slot_entities[first_slot],
                            slot_values[first_slot],
                            slot_entities[second_slot],
                            slot_values[second_slot],
                            query_entity,
                        )
                        rendered = render_chat(
                            tokenizer, model_name, raw_prompt
                        )
                        ids = [
                            int(token_id)
                            for token_id in tokenizer.encode(
                                rendered, add_special_tokens=False
                            )
                        ]
                        expected_answer = answer_ids[domain][gold]
                        extended = [
                            int(token_id)
                            for token_id in tokenizer.encode(
                                rendered + answer_text(model_name, gold),
                                add_special_tokens=False,
                            )
                        ]
                        if extended != ids + expected_answer:
                            raise RuntimeError(
                                f"{model_name}/{world_id}: answer boundary drift"
                            )

                        fact_positions = {}
                        for entity in base_entities:
                            found = positions_of(
                                ids, prompt_name_ids[entity]
                            )
                            expected = 2 if entity == query_entity else 1
                            if len(found) != expected:
                                raise RuntimeError(
                                    f"{model_name}/{world_id}/{entity}: "
                                    f"entity positions {found}"
                                )
                            fact_positions[entity] = found[0]
                        query_positions = positions_of(
                            ids, prompt_name_ids[query_entity]
                        )
                        value_positions = {
                            value: positions_of(
                                ids, prompt_value_ids[domain][value]
                            )
                            for value in base_values
                        }
                        if any(
                            len(found) != 1
                            for found in value_positions.values()
                        ):
                            raise RuntimeError(
                                f"{model_name}/{world_id}: "
                                f"value positions {value_positions}"
                            )

                        record_id = (
                            f"{model_name}.{world_id}.t{template}."
                            f"o{display_order}.v{value_swap}."
                            f"q{query_role}.e{entity_swap}"
                        )
                        row = {
                            "schema_version": (
                                "phase1003_crossparadigm_case.v1"
                            ),
                            "phase": PHASE,
                            "protocol_revision": PROTOCOL_REVISION,
                            "model": model_name,
                            "record_id": record_id,
                            "domain": domain,
                            "world_id": world_id,
                            "world_index": world_index,
                            "split": split,
                            "template": template,
                            "display_order": display_order,
                            "value_swap": value_swap,
                            "query_role": query_role,
                            "entity_swap": entity_swap,
                            "base_entities": base_entities,
                            "base_values": base_values,
                            "slot_entities": slot_entities,
                            "slot_values": slot_values,
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
                            "semantic_step": semantic_steps[domain],
                            "candidate_token_ids": candidate_ids[domain],
                            "role_positions": {
                                "slot0_entity": fact_positions[
                                    slot_entities[0]
                                ],
                                "slot0_value": value_positions[
                                    slot_values[0]
                                ][0],
                                "slot1_entity": fact_positions[
                                    slot_entities[1]
                                ],
                                "slot1_value": value_positions[
                                    slot_values[1]
                                ][0],
                                "query_name": query_positions[-1],
                                "answer_boundary": len(ids) - 1,
                            },
                        }
                        cases.append(row)
                        arms.append(row)
                        token_lengths[(domain, template)].add(len(ids))

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
                            f"{model_name}/{world_id}: "
                            f"counterfactual drift {changed}"
                        )
                    if (
                        arm0["gold"] != arm1["foil"]
                        or arm1["gold"] != arm0["foil"]
                    ):
                        raise RuntimeError(
                            f"{model_name}/{world_id}: answer swap drift"
                        )
                    pair_id = (
                        f"{model_name}.{world_id}.t{template}."
                        f"o{display_order}.v{value_swap}.q{query_role}"
                    )
                    pairs.append({
                        "schema_version": (
                            "phase1003_crossparadigm_pair.v1"
                        ),
                        "phase": PHASE,
                        "model": model_name,
                        "pair_id": pair_id,
                        "domain": domain,
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
            f"{model_name}: domain/template length drift {token_lengths}"
        )
    model_root = OUT_ROOT / "protocol" / model_name
    write_jsonl(model_root / "cases.jsonl", cases)
    write_jsonl(model_root / "pairs.jsonl", pairs)
    selections = {}
    for domain, split in itertools.product(
        DOMAINS, ("discovery", "confirmation")
    ):
        selected = select_pairs(pairs, domain, split)
        write_jsonl(
            model_root / f"{domain}_{split}_selected_pairs.jsonl",
            selected,
        )
        selections[f"{domain}:{split}"] = digest(selected)

    audit = {
        "schema_version": "phase1003_crossparadigm_protocol_audit.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "model": model_name,
        "domains": list(DOMAINS),
        "case_count": len(cases),
        "pair_count": len(pairs),
        "selected_pair_count_per_domain_split": (
            32 * SELECTED_PAIRS_PER_STRATUM
        ),
        "selected_direction_count_per_domain_split": (
            64 * SELECTED_PAIRS_PER_STRATUM
        ),
        "world_count_per_domain_split": WORLD_COUNT_PER_SPLIT,
        "discovery_name_count": len(DISCOVERY_NAMES),
        "confirmation_name_count": len(CONFIRMATION_NAMES),
        "discovery_confirmation_name_overlap": sorted(
            set(DISCOVERY_NAMES) & set(CONFIRMATION_NAMES)
        ),
        "phase1002_name_overlap": sorted(
            (set(DISCOVERY_NAMES) | set(CONFIRMATION_NAMES))
            & set(PHASE1002_NAMES)
        ),
        "answer_prefix": ANSWER_PREFIX[model_name],
        "answer_ids": answer_ids,
        "answer_step_counts": answer_step_counts,
        "semantic_steps": semantic_steps,
        "candidate_token_ids": candidate_ids,
        "template_prompt_lengths": {
            f"{domain}:t{template}": sorted(values)
            for (domain, template), values in token_lengths.items()
        },
        "all_counterfactuals_change_only_two_entity_tokens": True,
        "all_counterfactuals_swap_gold_and_foil": True,
        "case_digest": digest(cases),
        "pair_digest": digest(pairs),
        "selection_digests": selections,
    }
    write_json(model_root / "protocol_audit.json", audit)
    return audit


def main() -> None:
    audits = {
        model_name: build_model(model_name) for model_name in MODELS
    }
    phase1002_prereg = read_json(
        PHASE1002_ROOT / "preregistered_protocol.json"
    )
    layer_summary = read_json(
        PHASE1002_ROOT
        / "kv_value_layer_localization"
        / "summary.json"
    )
    frozen_value_layers = {
        model_name: {
            "layer_numbers": layer_summary["models"][model_name][
                "selection"
            ]["selected_layer_numbers"],
            "source_phase": 1002,
            "selection_uses_phase1003": False,
        }
        for model_name in MODELS
    }
    source_depths = {
        model_name: int(
            phase1002_prereg["frozen_phase1001_topology"][model_name][
                "source_depth"
            ]
        )
        for model_name in MODELS
    }
    preregistration = {
        "schema_version": "phase1003_preregistered_protocol.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "research_order": (
            "independent denominator -> behavior gate -> exhaustive anchor "
            "subsets -> frozen natural confirmation -> cross-paradigm cache "
            "transport -> frozen value-layer replication -> head/channel "
            "decomposition only where its parent instrument passes"
        ),
        "revision_audit": {
            "revision_1": (
                "Independent calibration found that Qwen3 capitalized many "
                "shape, size, and category values. Exact-answer rate was "
                "0.15625 even when the semantic choice was often visibly "
                "correct. No internal state had been observed."
            ),
            "revision_2": (
                "Only the answer instruction was changed to require copying "
                "the value's lowercase spelling exactly. Qwen3 then often "
                "lowercased the whole answer; exact rate was 0.0859375."
            ),
            "revision_3": (
                "The frozen answer surface was changed to an explicitly "
                "all-lowercase four-word sentence. Revisions 1 and 2 are "
                "retained. No internal state had been observed."
            ),
            "revision_4": (
                "A calibration-only, four-surface probe compared one word, "
                "Answer: value, The value is value, and the lowercase "
                "sentence. Answer: value was the first preregistered surface "
                "with exact rate at least 0.95 in all three models "
                "(1.0/0.96875/0.96875). It was frozen before formal behavior "
                "or any internal intervention."
            ),
            "formal_data_used_in_calibration": False,
            "internal_results_observed_before_revision": False,
        },
        "models_in_required_execution_order": list(MODELS),
        "domains": {
            key: list(values) for key, values in DOMAINS.items()
        },
        "anchor_roles": list(ANCHOR_ROLES),
        "anchor_subset_count": 2 ** len(ANCHOR_ROLES),
        "protocol_audits": audits,
        "source_depths": source_depths,
        "frozen_phase1002_value_layers": frozen_value_layers,
        "primary_thresholds": {
            "behavior_candidate_accuracy": 0.95,
            "behavior_exact_answer_rate": 0.95,
            "full_anchor_donor_rate": 0.80,
            "subset_donor_rate": 0.80,
            "subset_median_normalized_transfer": 0.50,
            "noop_prediction_agreement": 0.99,
            "cache_value_donor_rate": 0.80,
            "frozen_value_layer_sufficiency_rate": 0.70,
            "frozen_value_layer_restore_rate": 0.70,
            "head_joint_sufficiency_rate": 0.70,
            "head_joint_restore_rate": 0.70,
            "cross_model_minimum_pass_count": 2,
            "cross_domain_minimum_pass_count": 2,
        },
        "selection_rules": {
            "anchor_subsets": (
                "Enumerate all 32 subsets. On discovery, retain every subset "
                "at the smallest cardinality meeting both frozen sufficiency "
                "thresholds. Report every retained subset on confirmation; "
                "confirmation never chooses among them."
            ),
            "heads_and_channels": (
                "Use direct causal ablation and transplantation. Rank "
                "lexicographically by restoration mediation, then "
                "sufficiency transfer, then physical coordinate. Do not "
                "combine metrics with fitted or arbitrary weights."
            ),
            "null_result": (
                "No smaller anchor subset, no sparse head set, or no sparse "
                "channel set is a valid no-go result and does not fail data "
                "integrity."
            ),
        },
        "controls": {
            "real_donor_states_only": True,
            "donor_answer_excludes_recipient_answer": True,
            "target_state_noop": True,
            "discovery_confirmation_entity_vocabulary_disjoint": True,
            "confirmation_used_for_selection": False,
            "parent_instrument_required_before_decomposition": True,
        },
        "claim_boundary": (
            "This phase can identify repeated causal role maps and physical "
            "cache coordinates for four controlled attribute paradigms. It "
            "cannot establish an open-language law, shared neuron identity, "
            "or a complete language mechanism."
        ),
        "forbidden_claims": [
            "A progress percentage without a defined denominator.",
            "A weighted causal score as a language formula.",
            "A requirement that a smaller subset must exist.",
            "Calling the working closure framework a completed theory.",
        ],
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
