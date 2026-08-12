#!/usr/bin/env python3
"""Freeze the Phase1004 blind causal-state-basis denominator.

Selection in this phase receives physical coordinates and causal responses
only. Semantic role positions are retained in a sealed audit field and are
revealed only after every source/receiver selection has been frozen.
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
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from phase548_shared_attention_compute_protocol import render_chat, tokenizer_for
from phase1002_multitoken_protocol import NAMES as PHASE1002_NAMES
from phase1003_crossparadigm_protocol import (
    CALIBRATION_NAMES as PHASE1003_CALIBRATION_NAMES,
    CONFIRMATION_NAMES as PHASE1003_CONFIRMATION_NAMES,
    DISCOVERY_NAMES as PHASE1003_DISCOVERY_NAMES,
)


PHASE = 1004
PROTOCOL_REVISION = 3
MODELS = ("qwen3", "glm4", "deepseek7b")
SPLITS = ("discovery", "confirmation")
DOMAINS = {
    "color": {
        "discovery": ("red", "blue", "green", "yellow"),
        "confirmation": ("pink", "brown", "gray", "cyan"),
    },
    "shape": {
        "discovery": ("oval", "sphere", "cube", "cone"),
        "confirmation": ("star", "heart", "ring", "cross"),
    },
}
DISCOVERY_NAMES = (
    "Adrian", "Albert", "Alex", "Alexa", "Alfred", "Alicia", "Allison",
    "Amanda", "Amelia", "Andre", "Andrea", "Anita", "Anthony", "April",
    "Ariel", "Arnold", "Ashley", "Audrey", "Austin", "Autumn", "Barbara",
    "Barry", "Bernard", "Beth", "Betty", "Beverly", "Bonnie", "Bradley",
    "Brenda", "Brittany", "Bryan", "Calvin",
)
CONFIRMATION_NAMES = (
    "Cameron", "Carmen", "Carrie", "Casey", "Catherine", "Chad", "Cheryl",
    "Claire", "Clara", "Clarence", "Claude", "Clayton", "Cody", "Colin",
    "Connor", "Craig", "Crystal", "Curtis", "Dakota", "Dale", "Darren",
    "Dawn", "Dean", "Dennis", "Devin", "Dominic", "Douglas", "Dylan",
    "Earl", "Edgar", "Edwin", "Elaine",
)
TEMPLATES_BY_SPLIT = {
    "discovery": (0, 1),
    "confirmation": (2, 3),
}
WORLD_COUNT_PER_SPLIT = 16
SELECTED_PAIRS_PER_STRATUM = 2
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
    / "phase1004_blind_causal_state_basis"
)


def canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def stable_order(value: str, salt: str) -> str:
    return hashlib.sha256(
        f"phase1004:{salt}:{value}".encode("utf-8")
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
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(f"expected one token for {text!r}, got {ids}")
    return int(ids[0])


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
            f"Ledger: {entity0} has {domain} {value0}; "
            f"{entity1} has {domain} {value1}.\n"
            f"What is the {domain} of {query}?"
        )
    elif template == 1:
        body = (
            f"For {domain}, the record pairs {entity0} with {value0} and "
            f"{entity1} with {value1}.\n"
            f"Return the {domain} paired with {query}."
        )
    elif template == 2:
        body = (
            f"The {domain} table lists {value0} beside {entity0}. "
            f"It lists {value1} beside {entity1}.\n"
            f"Report the {domain} listed beside {query}."
        )
    elif template == 3:
        body = (
            f"Item A says {entity0}: {domain} = {value0}. "
            f"Item B says {entity1}: {domain} = {value1}.\n"
            f"Which {domain} belongs to {query}?"
        )
    else:
        raise KeyError(template)
    return f"{body}\n{instruction}"


def answer_text(model_name: str, value: str) -> str:
    return f"{ANSWER_PREFIX[model_name]}Answer: {value}"


def semantic_case(case: dict[str, Any]) -> dict[str, Any]:
    result = dict(case)
    step = int(case["semantic_step"])
    result["input_ids"] = (
        list(case["input_ids"]) + list(case["answer_token_ids"][:step])
    )
    result["input_token_count"] = len(result["input_ids"])
    result["answer_boundary"] = result["input_token_count"] - 1
    return result


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
                "source": source,
                "target": target,
            })
    return result


def select_pairs(
    pairs: list[dict[str, Any]],
    domain: str,
    split: str,
) -> list[dict[str, Any]]:
    strata: dict[tuple[int, int, int, int], list[dict[str, Any]]] = (
        defaultdict(list)
    )
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
    selected = []
    for key, values in sorted(strata.items()):
        ordered = sorted(
            values,
            key=lambda row: stable_order(
                row["pair_id"], f"pair:{domain}:{split}:{key}"
            ),
        )
        if len(ordered) < SELECTED_PAIRS_PER_STRATUM:
            raise RuntimeError(f"underfilled stratum {domain}/{split}/{key}")
        selected.extend(ordered[:SELECTED_PAIRS_PER_STRATUM])
    expected = 16 * SELECTED_PAIRS_PER_STRATUM
    if len(selected) != expected:
        raise RuntimeError(
            f"{domain}/{split}: selected {len(selected)} != {expected}"
        )
    return selected


def build_model(model_name: str) -> dict[str, Any]:
    prior_names = (
        set(PHASE1002_NAMES)
        | set(PHASE1003_DISCOVERY_NAMES)
        | set(PHASE1003_CONFIRMATION_NAMES)
        | set(PHASE1003_CALIBRATION_NAMES)
    )
    formal_names = set(DISCOVERY_NAMES) | set(CONFIRMATION_NAMES)
    if set(DISCOVERY_NAMES) & set(CONFIRMATION_NAMES):
        raise RuntimeError("discovery/confirmation name overlap")
    if formal_names & prior_names:
        raise RuntimeError(
            f"prior phase name overlap: {sorted(formal_names & prior_names)}"
        )

    tokenizer = tokenizer_for(model_name)
    prompt_name_ids = {
        name: one_token_id(tokenizer, " " + name)
        for name in DISCOVERY_NAMES + CONFIRMATION_NAMES
    }
    if len(set(prompt_name_ids.values())) != len(prompt_name_ids):
        raise RuntimeError(f"{model_name}: name token collision")

    prompt_value_ids = {}
    answer_ids = {}
    semantic_steps = {}
    candidate_ids = {}
    for domain, split_values in DOMAINS.items():
        prompt_value_ids[domain] = {}
        answer_ids[domain] = {}
        semantic_steps[domain] = {}
        candidate_ids[domain] = {}
        for split, values in split_values.items():
            prompt_value_ids[domain][split] = {
                value: one_token_id(tokenizer, " " + value)
                for value in values
            }
            encoded = {
                value: [
                    int(token_id)
                    for token_id in tokenizer.encode(
                        answer_text(model_name, value),
                        add_special_tokens=False,
                    )
                ]
                for value in values
            }
            widths = {len(ids) for ids in encoded.values()}
            if len(widths) != 1:
                raise RuntimeError(
                    f"{model_name}/{domain}/{split}: answer width drift"
                )
            width = next(iter(widths))
            varying = [
                index
                for index in range(width)
                if len({ids[index] for ids in encoded.values()}) > 1
            ]
            if len(varying) != 1:
                raise RuntimeError(
                    f"{model_name}/{domain}/{split}: semantic step {varying}"
                )
            semantic_step = varying[0]
            answer_ids[domain][split] = encoded
            semantic_steps[domain][split] = semantic_step
            candidate_ids[domain][split] = {
                value: int(ids[semantic_step])
                for value, ids in encoded.items()
            }

    rng = random.Random(1004_20260724)
    worlds = {}
    for split, names in (
        ("discovery", DISCOVERY_NAMES),
        ("confirmation", CONFIRMATION_NAMES),
    ):
        shuffled = list(names)
        rng.shuffle(shuffled)
        worlds[split] = [
            tuple(shuffled[index : index + 2])
            for index in range(0, len(shuffled), 2)
        ]
        if len(worlds[split]) != WORLD_COUNT_PER_SPLIT:
            raise RuntimeError(f"{split}: world count drift")

    cases = []
    pairs = []
    prompt_lengths: dict[tuple[str, str, int], set[int]] = defaultdict(set)
    for domain, split_values in DOMAINS.items():
        for split, values in split_values.items():
            value_pairs = list(itertools.combinations(values, 2))
            for world_index, base_entities_tuple in enumerate(worlds[split]):
                base_entities = list(base_entities_tuple)
                base_values = list(value_pairs[world_index % len(value_pairs)])
                world_id = f"{domain}.{split[:1]}w{world_index:02d}"
                for template, display_order, value_swap, query_role in (
                    itertools.product(
                        TEMPLATES_BY_SPLIT[split],
                        (0, 1),
                        (0, 1),
                        (0, 1),
                    )
                ):
                    arms = []
                    for entity_swap in (0, 1):
                        slot_entities = (
                            list(base_entities)
                            if not entity_swap
                            else [base_entities[1], base_entities[0]]
                        )
                        slot_values = (
                            list(base_values)
                            if not value_swap
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
                        answer_token_ids = answer_ids[domain][split][gold]
                        extended = [
                            int(token_id)
                            for token_id in tokenizer.encode(
                                rendered + answer_text(model_name, gold),
                                add_special_tokens=False,
                            )
                        ]
                        if extended != ids + answer_token_ids:
                            raise RuntimeError(
                                f"{model_name}/{world_id}: answer boundary drift"
                            )
                        fact_entity_positions = {}
                        for entity in base_entities:
                            found = positions_of(
                                ids, prompt_name_ids[entity]
                            )
                            expected = 2 if entity == query_entity else 1
                            if len(found) != expected:
                                raise RuntimeError(
                                    f"{model_name}/{world_id}/{entity}: {found}"
                                )
                            fact_entity_positions[entity] = found[0]
                        query_positions = positions_of(
                            ids, prompt_name_ids[query_entity]
                        )
                        value_positions = {
                            value: positions_of(
                                ids,
                                prompt_value_ids[domain][split][value],
                            )
                            for value in base_values
                        }
                        if any(
                            len(found) != 1
                            for found in value_positions.values()
                        ):
                            raise RuntimeError(
                                f"{model_name}/{world_id}: value positions"
                            )
                        audit_positions = {
                            "slot0_entity": fact_entity_positions[
                                slot_entities[0]
                            ],
                            "slot0_value": value_positions[
                                slot_values[0]
                            ][0],
                            "slot1_entity": fact_entity_positions[
                                slot_entities[1]
                            ],
                            "slot1_value": value_positions[
                                slot_values[1]
                            ][0],
                            "query_name": query_positions[-1],
                        }
                        record_id = (
                            f"{model_name}.{world_id}.t{template}."
                            f"o{display_order}.v{value_swap}."
                            f"q{query_role}.e{entity_swap}"
                        )
                        row = {
                            "schema_version": (
                                "phase1004_blind_causal_case.v1"
                            ),
                            "phase": PHASE,
                            "protocol_revision": PROTOCOL_REVISION,
                            "model": model_name,
                            "record_id": record_id,
                            "domain": domain,
                            "split": split,
                            "world_id": world_id,
                            "world_index": world_index,
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
                            "gold": gold,
                            "foil": foil,
                            "raw_prompt": raw_prompt,
                            "rendered_prompt": rendered,
                            "input_ids": ids,
                            "input_token_count": len(ids),
                            "answer_text": answer_text(model_name, gold),
                            "answer_token_ids": answer_token_ids,
                            "semantic_step": semantic_steps[domain][split],
                            "candidate_token_ids": candidate_ids[domain][split],
                            "answer_boundary": len(ids) - 1,
                            # Forbidden to all selection functions. It exists
                            # only for the post-freeze reconstruction audit.
                            "sealed_semantic_role_positions": audit_positions,
                        }
                        cases.append(row)
                        arms.append(row)
                        prompt_lengths[(domain, split, template)].add(
                            len(ids)
                        )

                    arm0, arm1 = arms
                    changed = [
                        index
                        for index, (left, right) in enumerate(
                            zip(arm0["input_ids"], arm1["input_ids"])
                        )
                        if left != right
                    ]
                    expected = sorted((
                        arm0["sealed_semantic_role_positions"][
                            "slot0_entity"
                        ],
                        arm0["sealed_semantic_role_positions"][
                            "slot1_entity"
                        ],
                    ))
                    if changed != expected:
                        raise RuntimeError(
                            f"{model_name}/{world_id}: pair drift {changed}"
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
                            "phase1004_blind_causal_pair.v1"
                        ),
                        "phase": PHASE,
                        "protocol_revision": PROTOCOL_REVISION,
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

    if any(len(values) != 1 for values in prompt_lengths.values()):
        raise RuntimeError(f"{model_name}: prompt length drift")
    model_root = OUT_ROOT / "protocol" / model_name
    write_jsonl(model_root / "cases.jsonl", cases)
    write_jsonl(model_root / "pairs.jsonl", pairs)
    selections = {}
    for domain, split in itertools.product(DOMAINS, SPLITS):
        selected = select_pairs(pairs, domain, split)
        write_jsonl(
            model_root / f"{domain}_{split}_selected_pairs.jsonl",
            selected,
        )
        selections[f"{domain}:{split}"] = digest(selected)
    audit = {
        "schema_version": "phase1004_protocol_audit.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "model": model_name,
        "case_count": len(cases),
        "pair_count": len(pairs),
        "selected_pair_count_per_domain_split": (
            16 * SELECTED_PAIRS_PER_STRATUM
        ),
        "selected_direction_count_per_domain_split": (
            32 * SELECTED_PAIRS_PER_STRATUM
        ),
        "world_count_per_domain_split": WORLD_COUNT_PER_SPLIT,
        "discovery_confirmation_name_overlap": sorted(
            set(DISCOVERY_NAMES) & set(CONFIRMATION_NAMES)
        ),
        "prior_phase_name_overlap": sorted(formal_names & prior_names),
        "template_overlap": sorted(
            set(TEMPLATES_BY_SPLIT["discovery"])
            & set(TEMPLATES_BY_SPLIT["confirmation"])
        ),
        "value_overlap_by_domain": {
            domain: sorted(
                set(values["discovery"])
                & set(values["confirmation"])
            )
            for domain, values in DOMAINS.items()
        },
        "prompt_lengths": {
            f"{domain}:{split}:t{template}": sorted(values)
            for (domain, split, template), values in prompt_lengths.items()
        },
        "all_pair_counterfactuals_change_only_two_entity_tokens": True,
        "all_pairs_swap_gold_and_foil": True,
        "case_digest": digest(cases),
        "pair_digest": digest(pairs),
        "selection_digests": selections,
    }
    write_json(model_root / "protocol_audit.json", audit)
    return audit


def main() -> None:
    audits = {
        model_name: build_model(model_name)
        for model_name in MODELS
    }
    preregistration = {
        "schema_version": "phase1004_preregistered_protocol.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "title": (
            "Blind causal state basis and repeated functional subgraph"
        ),
        "epistemic_order": [
            "freeze independent denominator",
            "behavior qualification",
            "diagnostic prediction trajectory",
            "blind all-position source fingerprints",
            "freeze deterministic source reconstruction rule",
            "blind component receiver fingerprints",
            "cross-domain/cross-template/cross-model repetition audit",
            "natural rollout and EOS confirmation",
            "BF16 boundary audit",
            "only then reveal semantic roles",
        ],
        "models_in_required_execution_order": list(MODELS),
        "domains": DOMAINS,
        "templates_by_split": {
            key: list(value)
            for key, value in TEMPLATES_BY_SPLIT.items()
        },
        "protocol_audits": audits,
        "event_universe": {
            "source": (
                "every absolute position inside the original rendered prompt "
                "at residual depth 1; teacher-forced Answer-prefix positions "
                "are excluded; event ids are p000... and carry no semantic "
                "label"
            ),
            "receiver_screen": (
                "twelve preregistered relative-depth checkpoints x "
                "attention/MLP/residual x answer-boundary position"
            ),
            "terminal": "native candidate logits, natural rollout, EOS",
            "future_extension_not_authorized": (
                "head and subspace decomposition requires a repeated "
                "component-level parent event"
            ),
        },
        "source_reconstruction_rule": {
            "ranking": [
                "descending leave-one-out restored-target rate",
                "descending median leave-one-out mediation",
                "descending single-position donor rate",
                "ascending physical position",
            ],
            "joint_build": (
                "add ranked positions until donor rate >= 0.80 and "
                "median normalized transfer >= 0.50; then run one "
                "reverse-delete pass without changing thresholds"
            ),
            "confirmation_contract": (
                "apply the identical algorithm and thresholds independently "
                "to unseen names, values, and templates; confirmation may "
                "not tune order, thresholds, or event types"
            ),
        },
        "revision_audit": {
            "revision_1": (
                "The first static protocol allowed every position in the "
                "teacher-forced semantic-decision input. During the first "
                "Qwen3 trial, before reading any result summary, it was "
                "recognized that this included Answer-prefix positions. "
                "Such positions occur after the natural prompt and cannot "
                "serve as a pre-output source for rollout."
            ),
            "revision_1_result_used": False,
            "revision_1_partial_artifacts_retained_at": (
                "blind_source_pre_prompt_scope_fix"
            ),
            "revision_2": (
                "The source universe is restricted to all physical "
                "positions in the original rendered prompt. The semantic "
                "decision input still appends the frozen Answer prefix only "
                "to expose the value decision boundary."
            ),
            "internal_result_used_to_choose_revision_2": False,
            "revision_3": (
                "The first complete Qwen3 execution attempt stopped before "
                "template 3 because the selected 32-pair recipient subset "
                "did not always contain a complementary-value donor. The "
                "donor contract was not relaxed. Donors are now assigned "
                "from the complete frozen protocol denominator while formal "
                "recipients remain the preregistered selected pairs."
            ),
            "revision_2_partial_artifacts_retained_at": (
                "blind_source_pre_full_donor_pool_fix"
            ),
            "revision_2_result_used": False,
            "revision_3_donor_pool": (
                "complete frozen same-model/domain/split/template case pool"
            ),
            "internal_result_used_to_choose_revision_3": False,
        },
        "receiver_rule": {
            "discovery_screen_n": 16,
            "confirmation_n": 64,
            "maximum_frozen_events_per_model_domain": 12,
            "ranking": [
                "descending median restoration mediation",
                "descending mean sufficiency transfer",
                "descending restored-target rate",
                "ascending relative depth then component",
            ],
            "repeated_parent_gate": (
                "same component class and relative-depth half must have "
                "positive median mediation on both domains and both splits "
                "in at least two models"
            ),
        },
        "thresholds": {
            "behavior_candidate_accuracy": 0.95,
            "source_joint_donor_rate": 0.80,
            "source_joint_median_transfer": 0.50,
            "noop_candidate_agreement": 0.99,
            "same_answer_control_target_rate": 0.95,
            "receiver_positive_median_mediation": 0.10,
            "receiver_mean_sufficiency_transfer": 0.10,
            "natural_donor_semantic_rate": 0.70,
            "natural_eos_rate": 0.95,
            "cross_domain_min_models": 2,
        },
        "diagnostic_lens_boundary": (
            "Layerwise normalized-logit trajectories are diagnostic "
            "observer outputs only. They cannot select causal events and "
            "cannot establish a phase transition or mechanism."
        ),
        "controls": {
            "real_donor_states_only": True,
            "same_answer_cross_world_donor": True,
            "target_state_noop": True,
            "discovery_confirmation_name_disjoint": True,
            "discovery_confirmation_value_disjoint": True,
            "discovery_confirmation_template_disjoint": True,
            "confirmation_used_to_tune_algorithm": False,
            "semantic_role_labels_visible_during_selection": False,
        },
        "valid_no_go_results": [
            "no sparse source position set",
            "no repeated component receiver class",
            "prediction trajectory not aligned with causal events",
            "no head/subspace authorization",
            "no neuron-level localization",
        ],
        "forbidden_claims": [
            "raw logit lens is an internal native probability",
            "entropy drop proves a mechanism phase transition",
            "a fixed layer or neuron is a cross-model functional identity",
            "an automatically reconstructed set is a complete language law",
            "a progress percentage without a defined denominator",
        ],
        "claim_boundary": (
            "This phase may discover repeated, label-blind causal event "
            "classes in two controlled paradigms. It cannot establish "
            "open-language syntax, general reasoning, a shared neuron map, "
            "or a complete intelligence theory."
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
