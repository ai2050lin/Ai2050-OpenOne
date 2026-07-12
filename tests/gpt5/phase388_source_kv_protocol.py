#!/usr/bin/env python3
"""Freeze fresh relation-binding cases for a causally admissible K/V test."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / "tests/gpt5"))

from model_registry import get_model_spec  # noqa: E402
from phase333_dynamic_case_bank import interface_prompt  # noqa: E402

OUT = ROOT / "tests/gpt5/result/phase388_source_kv_transport"
OLD_CASES = (
    ROOT
    / "tests/gpt5/result/phase386_multitime_relation_atlas"
    / "protocol/private/phase386_candidate_execution_cases.jsonl"
)
MODELS = ("qwen3", "glm4", "deepseek7b")
GROUP_COUNT = 24

ENTITY_PAIRS = (
    ("Mira", "Tova"),
    ("Lena", "Nora"),
    ("Kira", "Vera"),
    ("Rina", "Sonia"),
    ("Dara", "Faye"),
    ("Gina", "Hera"),
    ("Iris", "Juno"),
    ("Kara", "Lola"),
    ("Maya", "Nina"),
    ("Opal", "Pia"),
    ("Rosa", "Sara"),
    ("Tina", "Uma"),
    ("Vina", "Willa"),
    ("Xena", "Yara"),
    ("Ada", "Bella"),
    ("Cora", "Dina"),
    ("Elsa", "Freya"),
    ("Greta", "Hilda"),
    ("Ida", "Julia"),
    ("Lara", "Mona"),
    ("Olga", "Petra"),
    ("Rita", "Selma"),
    ("Thea", "Una"),
    ("Vada", "Wendy"),
)

ITEM_PAIRS = (
    ("amber", "cobalt"),
    ("copper", "silver"),
    ("lemon", "peach"),
    ("violin", "flute"),
    ("cedar", "maple"),
    ("ruby", "jade"),
    ("falcon", "heron"),
    ("velvet", "linen"),
    ("comet", "planet"),
    ("coral", "pearl"),
    ("basil", "thyme"),
    ("marble", "granite"),
    ("saffron", "pepper"),
    ("willow", "birch"),
    ("piano", "cello"),
    ("orange", "plum"),
    ("bronze", "nickel"),
    ("tulip", "orchid"),
    ("salmon", "trout"),
    ("cotton", "silk"),
    ("saturn", "venus"),
    ("mint", "ginger"),
    ("quartz", "onyx"),
    ("raven", "swan"),
)


def stable_id(prefix: str, *parts: str) -> str:
    value = "\x1f".join(parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(value).hexdigest()[:20]}"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prompt(entity_a: str, entity_b: str, item_a: str, item_b: str) -> str:
    return (
        "Answer with only one item.\n"
        "Facts:\n"
        f"{item_a} is assigned to {entity_a}.\n"
        f"{item_b} is assigned to {entity_b}.\n"
        f"Question: Which item is assigned to {entity_a}?\n"
        "Answer:"
    )


def main() -> None:
    if len(ENTITY_PAIRS) != GROUP_COUNT or len(ITEM_PAIRS) != GROUP_COUNT:
        raise RuntimeError("Phase388 source pools must match the frozen group count")
    old_prompts: set[str] = set()
    if OLD_CASES.is_file():
        for line in OLD_CASES.read_text(encoding="utf-8").splitlines():
            if line.strip():
                old_prompts.add(json.loads(line)["prompt"])

    created_at = datetime.now(timezone.utc).isoformat()
    tokenizers = {}
    for model in MODELS:
        spec = get_model_spec(model)
        tokenizers[model] = AutoTokenizer.from_pretrained(
            str(spec.local_dir),
            trust_remote_code=spec.trust_remote_code,
            local_files_only=True,
            use_fast=False,
        )
    rows: list[dict[str, Any]] = []
    public_groups: list[dict[str, Any]] = []
    for index, ((entity_a, entity_b), (item_a, item_b)) in enumerate(
        zip(ENTITY_PAIRS, ITEM_PAIRS, strict=True)
    ):
        group_id = stable_id("p388g", str(index), entity_a, entity_b, item_a, item_b)
        variants = (
            ("mapping_a", item_a, item_b, item_a, item_b),
            ("mapping_b", item_b, item_a, item_b, item_a),
        )
        group_rows = []
        for condition, first_item, second_item, target, distractor in variants:
            raw_text = prompt(entity_a, entity_b, first_item, second_item)
            condition_id = stable_id("p388c", group_id, condition)
            group_rows.append(
                {
                    "condition": condition,
                    "condition_id": condition_id,
                    "target": target,
                    "distractor": distractor,
                }
            )
            for model in MODELS:
                text, add_special, answer_phase = interface_prompt(
                    tokenizers[model], model, raw_text, "answer_aligned_chat"
                )
                if text in old_prompts:
                    raise RuntimeError("Phase388 prompt overlaps Phase386")
                rows.append(
                    {
                        "schema_version": "62.0.0",
                        "phase_id": "Phase388-Protocol",
                        "created_at": created_at,
                        "private_execution_model": model,
                        "blind_case_id": stable_id("p388case", condition_id, model),
                        "parallel_group_id": group_id,
                        "group_priority": index,
                        "condition": condition,
                        "prompt": text,
                        "raw_prompt": raw_text,
                        "interface": "answer_aligned_chat",
                        "answer_phase": answer_phase,
                        "tokenization_add_special_tokens": add_special,
                        "target": target,
                        "target_aliases": [target],
                        "distractors": [distractor],
                        "source_entity": entity_a,
                        "wrong_source_entity": entity_b,
                        "first_fact_item": first_item,
                        "second_fact_item": second_item,
                    }
                )
        public_groups.append(
            {
                "parallel_group_id": group_id,
                "group_priority": index,
                "condition_count": 2,
                "conditions": group_rows,
            }
        )

    private_path = OUT / "protocol/private/phase388_candidate_execution_cases.jsonl"
    write_jsonl(private_path, rows)
    protocol = {
        "schema_version": "62.0.0",
        "phase_id": "Phase388-Protocol",
        "created_at": created_at,
        "objective": (
            "test_causally_admissible_source_layer_input_derived_kv_to_query_"
            "attention_transport_on_fresh_relation_binding_cases"
        ),
        "denominator": {
            "candidate_group_count": GROUP_COUNT,
            "conditions_per_group": 2,
            "models": list(MODELS),
            "candidate_case_count": len(rows),
            "instrument_groups_required": 2,
            "causal_test_groups_required": 16,
        },
        "case_design": {
            "same_entity_and_item_sets_across_pair": True,
            "only_item_to_entity_binding_is_swapped": True,
            "source_patch_token_identity_is_held_constant": True,
            "source_state_occurs_after_bound_item_in_causal_text_order": True,
            "prompt_overlap_with_phase386": False,
            "model_native_answer_aligned_interface": True,
        },
        "runtime_contract": {
            "execution_batch_size": 1,
            "reason": (
                "preserve the previously audited single-sample position path; "
                "batched left padding is not qualified for GLM4"
            ),
            "model_order": list(MODELS),
        },
        "frozen_interventions": [
            "no_intervention",
            "identity_source_kv",
            "donor_source_k_only",
            "donor_source_v_only",
            "donor_source_kv",
            "donor_wrong_source_kv",
            "donor_source_kv_at_terminal_control_depth",
        ],
        "frozen_outcomes": [
            "query_attention_state_projection_toward_donor",
            "donor_vs_recipient_first_token_logit_margin_shift",
            "patched_generation_switch_to_donor_target",
        ],
        "selection_rule": (
            "Select the first two preordered groups with exact one-item behavior in "
            "both conditions and all models for instrumentation; select the next "
            "sixteen for the causal test. Do not replace groups after intervention starts."
        ),
        "physical_holdout_reuse": False,
        "single_neuron_scan_authorized": False,
        "groups": public_groups,
    }
    write_json(OUT / "phase388_protocol.json", protocol)
    write_json(
        OUT / "phase388_interface_amendment.json",
        {
            "schema_version": "62.0.1",
            "phase_id": "Phase388-InterfaceAmendment",
            "created_at": created_at,
            "retired_contract": "plain_raw_completion_without_model_native_interface",
            "retirement_evidence": {
                "qwen3_exact": {"numerator": 45, "denominator": 48},
                "glm4_exact": {"numerator": 18, "denominator": 48},
                "glm4_exclamation_only_failure_count": 30,
            },
            "replacement_contract": "model_native_answer_aligned_chat",
            "all_models_rerun_required": True,
            "selective_failure_rerun_allowed": False,
            "internal_intervention_started_before_amendment": False,
        },
    )
    write_json(
        OUT / "phase388_runtime_amendment.json",
        {
            "schema_version": "62.0.2",
            "phase_id": "Phase388-RuntimeAmendment",
            "created_at": created_at,
            "retired_contract": "batched_left_padded_behavior_generation",
            "retirement_evidence": {
                "qwen3_exact_at_batch_8": {"numerator": 48, "denominator": 48},
                "glm4_exact_at_batch_8": {"numerator": 18, "denominator": 48},
                "glm4_failures_clustered_by_prompt_length_group": True,
            },
            "replacement_contract": "single_sample_generation_batch_size_1",
            "all_models_rerun_required": True,
            "selective_failure_rerun_allowed": False,
            "internal_intervention_started_before_amendment": False,
        },
    )
    print(json.dumps(protocol["denominator"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
