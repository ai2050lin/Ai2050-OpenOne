#!/usr/bin/env python3
"""Freeze the Phase421 balanced relation-boundary denominator.

Phase421 changes the prediction target instead of adding another patch to the
Phase420 late-layer candidates.  It crosses current-support multiplicity,
explicit history-reliability cues and compatible/conflicting/irrelevant
history relations.  These are registered prompt operations, not assumed
latent scalar variables.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from model_registry import get_model_spec  # noqa: E402


PHASE_ID = "Phase421-BalancedBoundaryDenominator"
SCHEMA_VERSION = "94.0.0"
MODELS = ("qwen3", "glm4", "deepseek7b")
INTERFACES = ("chat", "completion")
CURRENT_IDENTITIES = ("a", "b")
SUPPORT_LEVELS = (1, 2, 3)
RELIABILITY_LEVELS = (1, 2, 3)
HISTORY_RELATIONS = ("compatible", "conflict", "irrelevant")
PHYSICAL_SUPPORT_LEVELS = (1, 3)
PHYSICAL_RELIABILITY_LEVELS = (1, 3)
SHARED_SUFFIX = "\nFinal answer:"

SOURCE = ROOT / "tests/gpt5/result/phase345_three_core_protocol/three_core_protocol_qualification/phase345_registered_cases.jsonl"
PHASE420_GROUPS = ROOT / "tests/gpt5/result/phase420_typed_path_atlas/phase420_frozen_groups.jsonl"
OUT = ROOT / "tests/gpt5/result/phase421_balanced_boundary_atlas"

SOURCE_SPLIT_TO_PHASE421 = {
    "discovery": "discovery",
    "calibration": "calibration",
    "heldout": "behavior_holdout",
    "private_heldout": "physical_holdout",
}

# The Phase345 source bank has only nine independent private items per family.
# Keeping Phase420-exposed items out of the new sealed split therefore permits
# two genuinely fresh groups per family, not the proposed four.
SPLIT_QUOTAS = {
    "discovery": {family: 13 for family in ("knowledge_network", "reasoning", "grammar", "protocol_control")},
    "calibration": {family: 5 for family in ("knowledge_network", "reasoning", "grammar", "protocol_control")},
    "behavior_holdout": {family: 4 for family in ("knowledge_network", "reasoning", "grammar", "protocol_control")},
    "physical_holdout": {family: 2 for family in ("knowledge_network", "reasoning", "grammar", "protocol_control")},
}

PHYSICAL_GROUP_QUOTAS = {
    "discovery": 3,
    "calibration": 2,
    "behavior_holdout": 2,
}

NEUTRAL_ANSWER_CANDIDATES = (
    "unknown",
    "maybe",
    "other",
    "none",
    "void",
    "neither",
    "irrelevant",
    "unclear",
    "not stated",
    "outside scope",
    "not applicable",
    "no answer",
    "unrelated item",
    "neutral response",
    "separate topic",
    "not in context",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def tokenizer_for(model: str, fast: bool = False) -> Any:
    spec = get_model_spec(model)
    return AutoTokenizer.from_pretrained(
        str(spec.local_dir),
        trust_remote_code=spec.trust_remote_code,
        local_files_only=True,
        use_fast=fast,
    )


def base_semantic_id(value: str) -> str:
    return value.removesuffix("_registered_negative").removesuffix("_registered_positive")


def augment_reasoning_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    augmented = [dict(row) for row in rows]
    for row in rows:
        mechanism = row["mechanism_id"]
        if mechanism in {"direct_fact_control", "two_hop_entailment"}:
            query = row["query_fragment"]
            match = list(re.finditer(r"set(\d+)", query))
            if not match:
                raise RuntimeError(f"Cannot build reasoning counterfactual: {query}")
            final = match[-1]
            replacement = f"set{int(final.group(1)) + 100}"
            negative_query = query[: final.start()] + replacement + query[final.end() :]
            negative_raw = row["raw_prompt"].replace(query, negative_query, 1)
            counterpart = {
                **row,
                "case_id": row["case_id"] + "_registered_negative",
                "semantic_case_id": row["semantic_case_id"] + "_registered_negative",
                "item_index": int(row["item_index"]) + 100,
                "raw_prompt": negative_raw,
                "query_fragment": negative_query,
                "target": "no",
                "target_aliases": ["no"],
                "distractors": ["yes", "unknown"],
                "phase421_counterfactual_current_state": True,
                "phase421_base_semantic_case_id": row["semantic_case_id"],
            }
            augmented.append(counterpart)
        elif mechanism == "missing_condition_check":
            old_source = row["source_fragment"]
            new_source = old_source.replace(
                "No fact says it is locked.",
                "Fact: it is locked.",
            )
            if new_source == old_source:
                raise RuntimeError(f"Cannot build missing-condition positive: {old_source}")
            positive_raw = row["raw_prompt"].replace(old_source, new_source, 1)
            counterpart = {
                **row,
                "case_id": row["case_id"] + "_registered_positive",
                "semantic_case_id": row["semantic_case_id"] + "_registered_positive",
                "item_index": int(row["item_index"]) + 100,
                "raw_prompt": positive_raw,
                "source_fragment": new_source,
                "target": "yes",
                "target_aliases": ["yes"],
                "distractors": ["no", "unknown"],
                "phase421_counterfactual_current_state": True,
                "phase421_base_semantic_case_id": row["semantic_case_id"],
            }
            augmented.append(counterpart)
    for row in augmented:
        row.setdefault("phase421_counterfactual_current_state", False)
        row.setdefault("phase421_base_semantic_case_id", base_semantic_id(row["semantic_case_id"]))
    return augmented


def load_source_rows() -> list[dict[str, Any]]:
    return augment_reasoning_rows(read_jsonl(SOURCE))


def continuation_ids(tokenizer: Any, text: str) -> list[int]:
    # The registered continuation contract uses one leading ASCII space.
    return [int(value) for value in tokenizer(" " + text, add_special_tokens=False)["input_ids"]]


def answer_signature(tokenizers: dict[str, Any], text: str) -> tuple[int, ...]:
    return tuple(
        len(tokenizers[model](text, add_special_tokens=False)["input_ids"])
        for model in MODELS
    )


def neutral_by_signature(tokenizers: dict[str, Any]) -> dict[tuple[int, ...], list[str]]:
    output: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for answer in NEUTRAL_ANSWER_CANDIDATES:
        output[answer_signature(tokenizers, answer)].append(answer)
    return output


def build_candidates(
    rows: list[dict[str, Any]],
    tokenizers: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    index = {(row["model"], row["semantic_case_id"]): row for row in rows}
    qwen_rows = [row for row in rows if row["model"] == "qwen3"]
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in qwen_rows:
        grouped[(row["family_id"], row["mechanism_id"], row["template_id"], row["split"])].append(row)
    neutrals = neutral_by_signature(tokenizers)
    audit = Counter()
    candidates: list[dict[str, Any]] = []
    for (family, mechanism, template, source_split), values in sorted(grouped.items()):
        for left, right in combinations(sorted(values, key=lambda item: (int(item["item_index"]), item["semantic_case_id"])), 2):
            audit["candidate_pair_count"] += 1
            if left["target"] == right["target"]:
                audit["same_target_rejection_count"] += 1
                continue
            target_counts: dict[str, dict[str, int]] = {}
            target_shape_valid = True
            for model in MODELS:
                left_model = index[(model, left["semantic_case_id"])]
                right_model = index[(model, right["semantic_case_id"])]
                tokenizer = tokenizers[model]
                left_count = len(tokenizer(left_model["target"], add_special_tokens=False)["input_ids"])
                right_count = len(tokenizer(right_model["target"], add_special_tokens=False)["input_ids"])
                target_counts[model] = {"a": left_count, "b": right_count}
                if left_count != right_count:
                    target_shape_valid = False
                target_shape_valid = target_shape_valid and (
                    continuation_ids(tokenizer, left_model["target"])
                    != continuation_ids(tokenizer, right_model["target"])
                )
            if not target_shape_valid:
                audit["target_shape_rejection_count"] += 1
                continue
            signature = tuple(target_counts[model]["a"] for model in MODELS)
            neutral_options = [
                answer
                for answer in neutrals.get(signature, [])
                if answer not in {left["target"], right["target"]}
            ]
            if not neutral_options:
                audit["neutral_shape_rejection_count"] += 1
                continue
            matched_neutral = None
            for neutral_answer in neutral_options:
                full_prompt_shape_pass = True
                for model in MODELS:
                    left_model = index[(model, left["semantic_case_id"])]
                    right_model = index[(model, right["semantic_case_id"])]
                    tokenizer = tokenizers[model]
                    for current_model in (left_model, right_model):
                        for interface in INTERFACES:
                            prompt_counts = {
                                len(
                                    tokenizer(
                                        serialize_prompt(
                                            tokenizer,
                                            current_model["raw_prompt"],
                                            current_model["source_fragment"],
                                            interface,
                                            answer,
                                            1,
                                            1,
                                        ),
                                        add_special_tokens=True,
                                    )["input_ids"]
                                )
                                for answer in (
                                    left_model["target"],
                                    right_model["target"],
                                    neutral_answer,
                                )
                            }
                            full_prompt_shape_pass = full_prompt_shape_pass and len(prompt_counts) == 1
                if full_prompt_shape_pass:
                    matched_neutral = neutral_answer
                    break
            if matched_neutral is None:
                audit["neutral_full_prompt_shape_rejection_count"] += 1
                continue
            candidates.append(
                {
                    "family_id": family,
                    "mechanism_id": mechanism,
                    "template_id": template,
                    "source_split": source_split,
                    "split": SOURCE_SPLIT_TO_PHASE421[source_split],
                    "semantic_case_a": left["semantic_case_id"],
                    "semantic_case_b": right["semantic_case_id"],
                    "base_semantic_case_a": left["phase421_base_semantic_case_id"],
                    "base_semantic_case_b": right["phase421_base_semantic_case_id"],
                    "item_index_a": int(left["item_index"]),
                    "item_index_b": int(right["item_index"]),
                    "target_a": left["target"],
                    "target_b": right["target"],
                    "irrelevant_answer": matched_neutral,
                    "target_token_counts": target_counts,
                    "history_answer_token_signature": list(signature),
                    "target_continuations_distinct": True,
                }
            )
            audit["qualified_candidate_pair_count"] += 1
    return candidates, dict(audit)


def phase420_exposed_base_ids() -> set[str]:
    if not PHASE420_GROUPS.exists():
        return set()
    exposed = set()
    for row in read_jsonl(PHASE420_GROUPS):
        exposed.add(base_semantic_id(row["semantic_case_a"]))
        exposed.add(base_semantic_id(row["semantic_case_b"]))
    return exposed


def select_groups(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    exposed = phase420_exposed_base_ids()
    selected: list[dict[str, Any]] = []
    used_bases: set[str] = set()
    for split, family_quotas in SPLIT_QUOTAS.items():
        for family, quota in family_quotas.items():
            mechanism_use = Counter()
            template_use = Counter()
            for _ in range(quota):
                eligible = [
                    row
                    for row in candidates
                    if row["split"] == split
                    and row["family_id"] == family
                    and row["base_semantic_case_a"] not in used_bases
                    and row["base_semantic_case_b"] not in used_bases
                    and row not in selected
                ]
                if not eligible:
                    raise RuntimeError(f"No disjoint Phase421 group left for {split}/{family}")
                choice = min(
                    eligible,
                    key=lambda row: (
                        int(row["base_semantic_case_a"] in exposed)
                        + int(row["base_semantic_case_b"] in exposed),
                        mechanism_use[row["mechanism_id"]],
                        template_use[row["template_id"]],
                        abs(row["item_index_a"] - row["item_index_b"]),
                        row["item_index_a"],
                        row["item_index_b"],
                    ),
                )
                selected.append(choice)
                used_bases.add(choice["base_semantic_case_a"])
                used_bases.add(choice["base_semantic_case_b"])
                mechanism_use[choice["mechanism_id"]] += 1
                template_use[choice["template_id"]] += 1
    return selected


def current_prompt(raw_prompt: str, source_fragment: str, support_count: int) -> str:
    lines = [f"Current support count: {support_count}."]
    for evidence_index in range(2, support_count + 1):
        lines.append(f"Additional current evidence {evidence_index}: {source_fragment}")
    lines.extend(["Current task:", raw_prompt])
    return "\n".join(lines)


def serialize_prompt(
    tokenizer: Any,
    raw_prompt: str,
    source_fragment: str,
    interface: str,
    history_answer: str,
    support_count: int,
    reliability_score: int,
) -> str:
    history_header = (
        "Earlier source record. "
        f"Source reliability score: {reliability_score} of 3."
    )
    current = current_prompt(raw_prompt, source_fragment, support_count)
    messages = [
        {"role": "user", "content": history_header},
        {"role": "assistant", "content": history_answer},
        {"role": "user", "content": current},
    ]
    if interface == "chat":
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    elif interface == "completion":
        prompt = "\n".join(
            [
                "Earlier source record:",
                history_header,
                f"Answer: {history_answer}",
                current,
            ]
        )
    else:
        raise KeyError(interface)
    return prompt.rstrip() + SHARED_SUFFIX


def relation_answer(
    current_identity: str,
    relation: str,
    target_a: str,
    target_b: str,
    irrelevant_answer: str,
) -> tuple[str, str]:
    current_target = target_a if current_identity == "a" else target_b
    opposite_target = target_b if current_identity == "a" else target_a
    if relation == "compatible":
        return current_target, "current_target"
    if relation == "conflict":
        return opposite_target, "opposite_target"
    if relation == "irrelevant":
        return irrelevant_answer, "irrelevant_target"
    raise KeyError(relation)


def freeze() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    source_rows = load_source_rows()
    index = {(row["model"], row["semantic_case_id"]): row for row in source_rows}
    tokenizers = {model: tokenizer_for(model) for model in MODELS}
    candidates, candidate_audit = build_candidates(source_rows, tokenizers)
    selected = select_groups(candidates)
    created_at = now()
    physical_group_counter = Counter()
    frozen_groups: list[dict[str, Any]] = []
    conditions: list[dict[str, Any]] = []
    exposed = phase420_exposed_base_ids()
    for group_index, pair in enumerate(selected):
        physical_panel = False
        if pair["split"] in PHYSICAL_GROUP_QUOTAS:
            key = (pair["split"], pair["family_id"])
            family_limit = PHYSICAL_GROUP_QUOTAS[pair["split"]]
            if physical_group_counter[key] < family_limit:
                physical_panel = True
                physical_group_counter[key] += 1
        group_id = (
            f"phase421_group_{group_index:03d}_{pair['mechanism_id']}_"
            f"{pair['item_index_a']:03d}_{pair['item_index_b']:03d}_{pair['template_id']}"
        )
        historical_exposure_count = int(pair["base_semantic_case_a"] in exposed) + int(
            pair["base_semantic_case_b"] in exposed
        )
        frozen_groups.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE_ID,
                "created_at": created_at,
                "group_id": group_id,
                "group_index": group_index,
                **pair,
                "phase420_historical_exposure_count": historical_exposure_count,
                "physical_development_panel": physical_panel,
                "behavior_generation_current_identity": "a" if group_index % 2 == 0 else "b",
                "causal": False,
            }
        )
        for model in MODELS:
            source_a = index[(model, pair["semantic_case_a"])]
            source_b = index[(model, pair["semantic_case_b"])]
            for interface in INTERFACES:
                for current_identity in CURRENT_IDENTITIES:
                    current_source = source_a if current_identity == "a" else source_b
                    current_target = source_a["target"] if current_identity == "a" else source_b["target"]
                    opposite_target = source_b["target"] if current_identity == "a" else source_a["target"]
                    for support_count in SUPPORT_LEVELS:
                        for reliability_score in RELIABILITY_LEVELS:
                            for relation in HISTORY_RELATIONS:
                                history_answer, history_role = relation_answer(
                                    current_identity,
                                    relation,
                                    source_a["target"],
                                    source_b["target"],
                                    pair["irrelevant_answer"],
                                )
                                prompt = serialize_prompt(
                                    tokenizers[model],
                                    current_source["raw_prompt"],
                                    current_source["source_fragment"],
                                    interface,
                                    history_answer,
                                    support_count,
                                    reliability_score,
                                )
                                prompt_count = len(
                                    tokenizers[model](prompt, add_special_tokens=True)["input_ids"]
                                )
                                physical_condition = bool(
                                    physical_panel
                                    and support_count in PHYSICAL_SUPPORT_LEVELS
                                    and reliability_score in PHYSICAL_RELIABILITY_LEVELS
                                )
                                generation_panel = bool(
                                    support_count == 2
                                    and reliability_score == 2
                                    and current_identity
                                    == ("a" if group_index % 2 == 0 else "b")
                                )
                                condition_id = (
                                    f"phase421_{model}_{group_index:03d}_{interface}_"
                                    f"current_{current_identity}_support_{support_count}_"
                                    f"reliability_{reliability_score}_{relation}"
                                )
                                conditions.append(
                                    {
                                        **current_source,
                                        "schema_version": SCHEMA_VERSION,
                                        "phase_id": PHASE_ID,
                                        "created_at": created_at,
                                        "phase421_condition_id": condition_id,
                                        "group_id": group_id,
                                        "group_index": group_index,
                                        "split": pair["split"],
                                        "source_split": pair["source_split"],
                                        "interface": interface,
                                        "current_identity": current_identity,
                                        "current_support_count": support_count,
                                        "history_reliability_score": reliability_score,
                                        "history_relation": relation,
                                        "history_answer": history_answer,
                                        "history_answer_role": history_role,
                                        "irrelevant_answer": pair["irrelevant_answer"],
                                        "target": current_target,
                                        "target_aliases": [current_target],
                                        "opposite_identity_target": opposite_target,
                                        "identity_target_a": source_a["target"],
                                        "identity_target_b": source_b["target"],
                                        "semantic_case_a": pair["semantic_case_a"],
                                        "semantic_case_b": pair["semantic_case_b"],
                                        "registered_prompt_token_count": prompt_count,
                                        "registered_target_token_count": pair["target_token_counts"][model][current_identity],
                                        "phase420_historical_exposure_count": historical_exposure_count,
                                        "behavior_margin_collection_authorized": True,
                                        "behavior_generation_panel": generation_panel,
                                        "physical_development_panel": physical_condition,
                                        "physical_holdout_sealed": pair["split"] == "physical_holdout",
                                        "causal_intervention_authorized": False,
                                        "single_neuron_scan_authorized": False,
                                    }
                                )

    conditions.sort(
        key=lambda row: (
            MODELS.index(row["model"]),
            row["group_index"],
            INTERFACES.index(row["interface"]),
            CURRENT_IDENTITIES.index(row["current_identity"]),
            row["current_support_count"],
            row["history_reliability_score"],
            HISTORY_RELATIONS.index(row["history_relation"]),
        )
    )
    prompt_shape_groups: dict[tuple[Any, ...], set[int]] = defaultdict(set)
    for row in conditions:
        prompt_shape_groups[
            (
                row["model"],
                row["group_id"],
                row["interface"],
                row["current_identity"],
                row["current_support_count"],
            )
        ].add(int(row["registered_prompt_token_count"]))
    split_counts = Counter(row["split"] for row in frozen_groups)
    family_counts = Counter(row["family_id"] for row in frozen_groups)
    model_counts = Counter(row["model"] for row in conditions)
    physical_counts = Counter(
        row["model"] for row in conditions if row["physical_development_panel"]
    )
    generation_counts = Counter(
        row["model"] for row in conditions if row["behavior_generation_panel"]
    )
    holdout_exposure = sum(
        row["phase420_historical_exposure_count"]
        for row in frozen_groups
        if row["split"] in {"calibration", "behavior_holdout", "physical_holdout"}
    )
    valid = bool(
        len(frozen_groups) == 96
        and len(conditions) == 31_104
        and split_counts
        == Counter({"discovery": 52, "calibration": 20, "behavior_holdout": 16, "physical_holdout": 8})
        and all(family_counts[family] == 24 for family in SPLIT_QUOTAS["discovery"])
        and all(model_counts[model] == 10_368 for model in MODELS)
        and all(physical_counts[model] == 1_344 for model in MODELS)
        and all(generation_counts[model] == 576 for model in MODELS)
        and all(len(counts) == 1 for counts in prompt_shape_groups.values())
        and holdout_exposure == 0
        and len({row["phase421_condition_id"] for row in conditions}) == len(conditions)
    )
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE_ID,
        "created_at": created_at,
        "objective": "balance_history_relation_boundary_before_incremental_typed_path_prediction",
        "model_order": list(MODELS),
        "group_count": len(frozen_groups),
        "condition_count": len(conditions),
        "split_count": dict(split_counts),
        "family_count": dict(family_counts),
        "registered_prompt_operations": {
            "current_support_multiplicity": list(SUPPORT_LEVELS),
            "history_reliability_numeric_cue": list(RELIABILITY_LEVELS),
            "history_relation": list(HISTORY_RELATIONS),
            "operations_are_not_assumed_latent_scalar_ground_truth": True,
        },
        "behavior_contract": {
            "primary": "continuous_first_step_target_minus_opposite_logit_margin",
            "secondary_generation_panel_per_model": 576,
            "generation_horizon_initial": 12,
            "generation_horizon_extended": 24,
            "balance_near_zero_absolute_margin": 0.25,
            "balance_each_sign_min_rate": 0.20,
            "balance_largest_sign_max_rate": 0.70,
        },
        "physical_contract": {
            "pre_registered_group_count": 28,
            "conditions_per_model": 1_344,
            "support_levels": list(PHYSICAL_SUPPORT_LEVELS),
            "reliability_levels": list(PHYSICAL_RELIABILITY_LEVELS),
            "physical_holdout_requires_incremental_prediction": True,
            "physical_holdout_is_sealed": True,
            "causal_intervention_authorized": False,
        },
        "geometry_contract": {
            "parallel_gain": "dot(delta_mlp,delta_attn)/(norm(delta_attn)^2+eps)",
            "orthogonal_rewrite": "norm(delta_mlp-g_parallel*delta_attn)/(norm(delta_attn)+eps)",
            "total_write_ratio": "norm(delta_mlp)/(norm(delta_attn)+eps)",
            "repeat_noise_floor_required": True,
            "minimum_numeric_floor": 1e-6,
        },
        "prediction_contract": {
            "baseline_and_path_fit_on_discovery_only": True,
            "calibration_and_behavior_holdout_reported_separately": True,
            "minimum_relative_squared_error_reduction": 0.05,
            "both_holdouts_must_pass": True,
        },
        "claim_boundary": "balanced_behavior_and_development_physical_prediction_no_causal_or_neuron_claim",
    }
    qualification = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase421-DenominatorQualification",
        "created_at": created_at,
        "valid": valid,
        "candidate_audit": candidate_audit,
        "selected_group_count": len(frozen_groups),
        "condition_count": len(conditions),
        "split_count": dict(split_counts),
        "family_count": dict(family_counts),
        "model_condition_count": dict(model_counts),
        "model_physical_development_condition_count": dict(physical_counts),
        "model_generation_panel_count": dict(generation_counts),
        "history_relation_and_reliability_prompt_token_count_exact": all(
            len(counts) == 1 for counts in prompt_shape_groups.values()
        ),
        "calibration_behavior_and_physical_holdout_phase420_exposure_count": holdout_exposure,
        "behavior_collection_authorized": valid,
        "physical_development_collection_requires_behavior_gate": True,
        "physical_holdout_collection_authorized": False,
        "causal_intervention_authorized": False,
        "single_neuron_scan_authorized": False,
    }
    return frozen_groups, conditions, protocol, qualification


def main() -> None:
    groups, conditions, protocol, qualification = freeze()
    write_jsonl(OUT / "phase421_frozen_groups.jsonl", groups)
    write_jsonl(OUT / "phase421_registered_conditions.jsonl", conditions)
    write_json(OUT / "phase421_protocol.json", protocol)
    write_json(OUT / "phase421_denominator_qualification.json", qualification)
    print(json.dumps(qualification, ensure_ascii=False, indent=2))
    if not qualification["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
