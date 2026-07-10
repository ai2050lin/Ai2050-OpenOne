#!/usr/bin/env python3
"""Phase329 full-vocabulary blocker and tokenwise query mediation atlas."""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase327_natural_retrieval_path as phase327  # noqa: E402
import phase329_full_vocabulary_case_bank as case_bank  # noqa: E402
from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402


PHASE = "Phase329"
SCHEMA_VERSION = "7.0.0"
MODELS = ("qwen3", "glm4", "deepseek7b")
MECHANISMS = ("color_retrieval", "category_retrieval", "habitat_retrieval")
ROUND_DEFAULT = "full_vocabulary_mediation"
OUT = ROOT / "tests/gpt5/result/phase329_full_vocabulary_mediation"
PHASE326 = ROOT / "tests/gpt5/result/phase326_distributed_carrier_atlas/distributed_carrier_atlas"
PHASE327 = ROOT / "tests/gpt5/result/phase327_natural_retrieval_path/natural_retrieval_path"
ROLE = "query"
TOP_K = 50


CONDITIONS = (
    "correct_baseline",
    "correct_carrier_joint_zero",
    "recipient_baseline",
    "recipient_tokenwise_correct",
    "recipient_pooled_correct",
    "recipient_tokenwise_same_target",
    "recipient_tokenwise_wrong_target",
    "recipient_tokenwise_unrelated",
    "recipient_tokenwise_norm_matched_unrelated",
    "recipient_tokenwise_shuffled",
    "recipient_tokenwise_correct_wrong_layer",
    "recipient_natural_carrier_correct",
)


GENERATION_CONDITIONS = (
    "correct_baseline",
    "correct_carrier_joint_zero",
    "recipient_baseline",
    "recipient_tokenwise_correct",
    "recipient_pooled_correct",
    "recipient_natural_carrier_correct",
)


FUNCTION_WORDS = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "is", "it", "its",
    "of", "on", "or", "that", "the", "this", "to", "was", "with",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_mean(values: list[float]) -> float:
    return round(mean(values), 6) if values else 0.0


def rate(values: list[bool]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def freeze_residual_selection(
    model: str,
    mechanism: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    eligible = [
        row for row in rows
        if row["model"] == model
        and row["mechanism_id"] == mechanism
        and row["split"] == "registered_primary"
        and row["position_role"] == ROLE
    ]
    if not eligible:
        raise ValueError(f"No Phase327 residual rows for {model}/{mechanism}")
    max_layer = max(int(row["layer"]) for row in eligible)
    layer_limit = round(max_layer * 0.70)
    grouped: dict[int, dict[str, list[float]]] = defaultdict(lambda: {"same": [], "wrong": []})
    for row in eligible:
        layer = int(row["layer"])
        if layer > layer_limit:
            continue
        bucket = "same" if row["comparison_variant"] == "same_target_object" else "wrong"
        grouped[layer][bucket].append(float(row["residual_rms_delta"]))
    candidates = []
    for layer, values in grouped.items():
        same = safe_mean(values["same"])
        wrong = safe_mean(values["wrong"])
        candidates.append({
            "residual_observation_layer": layer,
            "same_target_rms_delta": same,
            "wrong_target_rms_delta": wrong,
            "target_identity_specificity": round(wrong - same, 6),
            "same_target_observation_count": len(values["same"]),
            "wrong_target_observation_count": len(values["wrong"]),
        })
    selected = max(
        candidates,
        key=lambda row: (row["target_identity_specificity"], row["residual_observation_layer"]),
    )
    observation_layer = int(selected["residual_observation_layer"])
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "model": model,
        "family_id": "content_knowledge",
        "mechanism_id": mechanism,
        "position_role": ROLE,
        **selected,
        "intervention_input_layer": observation_layer + 1,
        "selection_split": "phase327_registered_primary_only",
        "validation_split": "phase329_registered_independent_only",
        "layer_search_limit": layer_limit,
        "selection_updates_allowed": False,
        "positive_identity_at_selection": selected["target_identity_specificity"] > 0.0,
        "alignment_correction": (
            "Phase327 hidden state Lk is the output of block k and is transplanted into the input "
            "of block k+1. This corrects the Phase328 one-layer interface offset."
        ),
        "selection_boundary": (
            "Maximum target-identity specificity within the first 70% of layers. A non-positive "
            "maximum is retained as a registered negative branch, not promoted as a mechanism."
        ),
    }


def cases_for(mechanism: str) -> list[dict[str, Any]]:
    return [case for case in case_bank.build_cases() if case["mechanism_id"] == mechanism]


def carrier_specs_for(selections: list[dict[str, Any]], case: dict[str, Any]) -> list[dict[str, Any]]:
    return phase327.selections_for_case(selections, case)


def capture_donor(
    loaded: Any,
    case: dict[str, Any],
    variant: str,
    capture_specs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    donor_case = phase327.variant_case(case, variant)
    return phase327.forward_probe(
        loaded,
        donor_case,
        case["target"],
        case["distractors"],
        capture_specs=capture_specs or [],
        capture_hidden=True,
        capture_hidden_tokens=True,
    )


def token_state(result: dict[str, Any], observation_layer: int) -> torch.Tensor:
    key = f"L{observation_layer}:{ROLE}"
    if key not in result["hidden_token_vectors"]:
        raise KeyError(f"Missing tokenwise residual state {key}")
    return result["hidden_token_vectors"][key]


def norm_matched(reference: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
    if reference.shape != control.shape:
        raise ValueError(f"Norm-match shape mismatch: {reference.shape} != {control.shape}")
    reference_norm = torch.linalg.vector_norm(reference, dim=1, keepdim=True)
    control_norm = torch.linalg.vector_norm(control, dim=1, keepdim=True).clamp_min(1e-8)
    return control * (reference_norm / control_norm)


def set_similarity(reference: dict[str, torch.Tensor], current: dict[str, torch.Tensor]) -> float:
    keys = sorted(set(reference) & set(current))
    return safe_mean([phase327.cosine(reference[key], current[key]) for key in keys])


def effective_captures(
    condition: str,
    result: dict[str, Any],
    reference: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if condition == "recipient_natural_carrier_correct":
        return reference
    if condition == "correct_carrier_joint_zero":
        return {key: torch.zeros_like(value) for key, value in reference.items()}
    return result["captures"]


def global_metrics(loaded: Any, logits: torch.Tensor, target: str) -> dict[str, Any]:
    target_id = phase327.target_token_ids(loaded.tokenizer, target)[0]
    target_logit = float(logits[target_id].item())
    target_rank = int((logits > target_logit).sum().item()) + 1
    top_ids = torch.topk(logits, k=TOP_K).indices.tolist()
    return {
        "target_first_token_id": target_id,
        "global_target_logit": round(target_logit, 6),
        "global_target_rank": target_rank,
        "global_blocker_count": target_rank - 1,
        "global_target_top1": int(top_ids[0]) == target_id,
        "global_target_top5": target_id in top_ids[:5],
        "global_target_top50": target_id in top_ids,
        "global_top1_token_id": int(top_ids[0]),
        "global_top1_token": loaded.tokenizer.decode([int(top_ids[0])]),
    }


def normalized_token(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def token_category(
    loaded: Any,
    token_id: int,
    token_text: str,
    case: dict[str, Any],
) -> str:
    target_id = phase327.target_token_ids(loaded.tokenizer, case["target"])[0]
    alias_ids = {
        phase327.target_token_ids(loaded.tokenizer, alias)[0]
        for alias in case["target_aliases"]
    }
    wrong_ids = {
        phase327.target_token_ids(loaded.tokenizer, value)[0]
        for value in case["distractors"]
    }
    normalized = normalized_token(token_text)
    if token_id == target_id:
        return "target"
    if token_id in alias_ids:
        return "target_alias"
    if token_id in wrong_ids:
        return "registered_wrong_answer"
    if any(marker in token_text for marker in ("\n", "\r", "\t", "<|", "###", "**", ":")):
        return "protocol_or_format"
    if not normalized:
        return "punctuation_or_whitespace"
    if normalized in FUNCTION_WORDS:
        return "continuation_function"
    subject_words = set(re.findall(r"[a-z0-9]+", case["subject"].lower()))
    if normalized in subject_words:
        return "subject_copy"
    return "semantic_content_other"


def top50_rows(
    loaded: Any,
    case: dict[str, Any],
    condition: str,
    logits: torch.Tensor,
    target_rank: int,
) -> list[dict[str, Any]]:
    values, ids = torch.topk(logits, k=TOP_K)
    target_id = phase327.target_token_ids(loaded.tokenizer, case["target"])[0]
    target_logit = float(logits[target_id].item())
    rows = []
    for rank_index, (value, token_id) in enumerate(zip(values.tolist(), ids.tolist()), start=1):
        token_id = int(token_id)
        token_text = loaded.tokenizer.decode([token_id])
        category = token_category(loaded, token_id, token_text, case)
        rows.append({
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE,
            "created_at": now(),
            "model": loaded.key,
            "case_id": case["case_id"],
            "base_case_id": case["base_case_id"],
            "family_id": case["family_id"],
            "mechanism_id": case["mechanism_id"],
            "template_id": case["template_id"],
            "condition": condition,
            "evaluation_target": case["target"],
            "target_global_rank": target_rank,
            "top50_rank": rank_index,
            "token_id": token_id,
            "token_text": token_text,
            "token_normalized": normalized_token(token_text),
            "token_logit": round(float(value), 6),
            "logit_above_target": round(float(value) - target_logit, 6),
            "is_full_vocabulary_blocker": float(value) > target_logit,
            "blocker_category": category,
            "taxonomy_registered_before_model_run": True,
        })
    return rows


def member_rows(
    loaded: Any,
    case: dict[str, Any],
    condition: str,
    specs: list[dict[str, Any]],
    reference: dict[str, torch.Tensor],
    recipient_baseline: dict[str, torch.Tensor],
    current: dict[str, torch.Tensor],
    global_row: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for spec in specs:
        key = phase327.spec_key(spec)
        if key not in reference or key not in recipient_baseline or key not in current:
            continue
        reference_value = reference[key]
        baseline_value = recipient_baseline[key]
        current_value = current[key]
        baseline_cosine = phase327.cosine(reference_value, baseline_value)
        current_cosine = phase327.cosine(reference_value, current_value)
        rows.append({
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE,
            "created_at": now(),
            "model": loaded.key,
            "case_id": case["case_id"],
            "base_case_id": case["base_case_id"],
            "family_id": case["family_id"],
            "mechanism_id": case["mechanism_id"],
            "template_id": case["template_id"],
            "condition": condition,
            "member_key": key,
            "component_type": spec["component_type"],
            "component_layer": int(spec["component_layer"]),
            "position_role": spec["position_role"],
            "component_index": int(spec["component_index"]),
            "component_start": int(spec["component_start"]),
            "component_end": int(spec["component_end"]),
            "recipient_baseline_cosine_to_correct": baseline_cosine,
            "current_cosine_to_correct": current_cosine,
            "cosine_gain_toward_correct": round(current_cosine - baseline_cosine, 6),
            "recipient_baseline_rms_to_correct": phase327.rms_delta(reference_value, baseline_value),
            "current_rms_to_correct": phase327.rms_delta(reference_value, current_value),
            "global_target_rank_gain": global_row["global_target_rank_gain"],
            "global_blocker_decline": global_row["global_blocker_decline"],
            "capture_timing": (
                "effective_direct_intervention_state"
                if condition in {"correct_carrier_joint_zero", "recipient_natural_carrier_correct"}
                else "carrier_input_after_upstream_condition"
            ),
            "single_unit_causal": False,
        })
    return rows


@torch.inference_mode()
def greedy_rollout(
    loaded: Any,
    prompt_case: dict[str, Any],
    *,
    mutate_specs: list[dict[str, Any]] | None = None,
    transplant_specs: list[dict[str, Any]] | None = None,
    transplant_values: dict[str, torch.Tensor] | None = None,
    residual_patch: dict[str, Any] | None = None,
    max_new_tokens: int = 4,
) -> dict[str, Any]:
    batch = loaded.tokenizer(prompt_case["prompt"], return_tensors="pt", truncation=True, max_length=96)
    input_ids = batch["input_ids"].to(loaded.input_device)
    generated: list[int] = []
    eos = loaded.tokenizer.eos_token_id
    eos_generated = False
    for _step in range(max_new_tokens):
        dynamic = loaded.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        logits = phase327._generation_forward(
            loaded,
            {**prompt_case, "prompt": dynamic},
            prompt_case,
            mutate_specs or [],
            transplant_specs or [],
            transplant_values or {},
            input_ids,
            residual_patch,
        )
        next_id = int(torch.argmax(logits).item())
        generated.append(next_id)
        input_ids = torch.cat(
            (input_ids, torch.tensor([[next_id]], dtype=input_ids.dtype, device=input_ids.device)),
            dim=1,
        )
        if eos is not None and next_id == eos:
            eos_generated = True
            break
    return {
        "generated_token_ids": generated,
        "generated_text": loaded.tokenizer.decode(generated, skip_special_tokens=True).strip(),
        "generated_token_count": len(generated),
        "eos_generated": eos_generated,
    }


def alias_match(text: str, aliases: list[str]) -> tuple[bool, str | None]:
    words = phase327.normalized_words(text)
    for alias in aliases:
        alias_words = phase327.normalized_words(alias)
        if alias_words and words[: len(alias_words)] == alias_words:
            return True, alias
    return False, None


def generation_row(
    loaded: Any,
    case: dict[str, Any],
    condition: str,
    rollout: dict[str, Any],
) -> dict[str, Any]:
    words = phase327.normalized_words(rollout["generated_text"])
    target_words = phase327.normalized_words(case["target"])
    target_prefix = bool(target_words and words[: len(target_words)] == target_words)
    alias, matched_alias = alias_match(rollout["generated_text"], case["target_aliases"])
    if rollout["eos_generated"]:
        stop_state = "eos"
    elif target_prefix and len(words) > len(target_words):
        stop_state = "target_then_continue"
    elif target_prefix:
        stop_state = "target_at_token_limit"
    else:
        stop_state = "non_target_at_token_limit"
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "model": loaded.key,
        "case_id": case["case_id"],
        "base_case_id": case["base_case_id"],
        "family_id": case["family_id"],
        "mechanism_id": case["mechanism_id"],
        "template_id": case["template_id"],
        "condition": condition,
        "evaluation_target": case["target"],
        "target_aliases": case["target_aliases"],
        **rollout,
        "normalized_words": words,
        "target_prefix_match": target_prefix,
        "target_alias_prefix_match": alias,
        "matched_alias": matched_alias,
        "one_word_protocol_compliant": len(words) == 1,
        "continued_after_target": target_prefix and len(words) > len(target_words),
        "stop_continue_state": stop_state,
        "single_unit_causal": False,
    }


def run_model_mechanism(model_key: str, mechanism: str, round_name: str) -> dict[str, Any]:
    output = OUT / round_name
    phase327_rows = read_jsonl(PHASE327 / f"phase327_{model_key}_residual_path_rows.jsonl")
    selection = freeze_residual_selection(model_key, mechanism, phase327_rows)
    carrier_selections = read_jsonl(PHASE326 / f"phase326_{model_key}_carrier_sets.jsonl")
    cases = cases_for(mechanism)
    loaded = None
    try:
        loaded = load_probe_model(model_key)
        n_layers = len(get_layers(loaded.model))
        selected_observation_layer = int(selection["residual_observation_layer"])
        selected_input_layer = int(selection["intervention_input_layer"])
        if selected_input_layer >= n_layers:
            raise ValueError("Selected residual observation has no downstream input layer")
        wrong_observation_layer = min(
            n_layers - 2,
            selected_observation_layer + max(2, n_layers // 5),
        )
        if wrong_observation_layer == selected_observation_layer:
            wrong_observation_layer = max(0, selected_observation_layer - max(2, n_layers // 5))
        wrong_input_layer = wrong_observation_layer + 1
        selection["wrong_control_residual_observation_layer"] = wrong_observation_layer
        selection["wrong_control_intervention_input_layer"] = wrong_input_layer

        rank_rows: list[dict[str, Any]] = []
        competitor_rows: list[dict[str, Any]] = []
        carrier_rows: list[dict[str, Any]] = []
        generation_rows: list[dict[str, Any]] = []

        for number, case in enumerate(cases, start=1):
            specs = carrier_specs_for(carrier_selections, case)
            correct = capture_donor(loaded, case, "correct_object", specs)
            same_target = capture_donor(loaded, case, "same_target_object")
            wrong_target = capture_donor(loaded, case, "same_semantic_wrong_target")
            unrelated = capture_donor(loaded, case, "unrelated_wrong_target")
            recipient_case = phase327.variant_case(case, "same_semantic_wrong_target")
            recipient = phase327.forward_probe(
                loaded,
                recipient_case,
                case["target"],
                case["distractors"],
                capture_specs=specs,
            )

            correct_tokens = token_state(correct, selected_observation_layer)
            same_tokens = token_state(same_target, selected_observation_layer)
            wrong_tokens = token_state(wrong_target, selected_observation_layer)
            unrelated_tokens = token_state(unrelated, selected_observation_layer)
            correct_wrong_layer_tokens = token_state(correct, wrong_observation_layer)
            if not (
                correct_tokens.shape == same_tokens.shape == wrong_tokens.shape == unrelated_tokens.shape
            ):
                raise ValueError(f"Tokenwise query shape mismatch for {case['case_id']}")
            reference_captures = correct["captures"]
            recipient_captures = recipient["captures"]
            patch = lambda layer, value: {  # noqa: E731
                "layer": layer,
                "position_role": ROLE,
                "value": value,
            }
            settings: dict[str, dict[str, Any]] = {
                "recipient_tokenwise_correct": {
                    "residual_patch": patch(selected_input_layer, correct_tokens),
                },
                "recipient_pooled_correct": {
                    "residual_patch": patch(selected_input_layer, correct_tokens.mean(dim=0)),
                },
                "recipient_tokenwise_same_target": {
                    "residual_patch": patch(selected_input_layer, same_tokens),
                },
                "recipient_tokenwise_wrong_target": {
                    "residual_patch": patch(selected_input_layer, wrong_tokens),
                },
                "recipient_tokenwise_unrelated": {
                    "residual_patch": patch(selected_input_layer, unrelated_tokens),
                },
                "recipient_tokenwise_norm_matched_unrelated": {
                    "residual_patch": patch(
                        selected_input_layer,
                        norm_matched(correct_tokens, unrelated_tokens),
                    ),
                },
                "recipient_tokenwise_shuffled": {
                    "residual_patch": patch(selected_input_layer, correct_tokens.flip(0)),
                },
                "recipient_tokenwise_correct_wrong_layer": {
                    "residual_patch": patch(wrong_input_layer, correct_wrong_layer_tokens),
                },
                "recipient_natural_carrier_correct": {
                    "transplant_specs": specs,
                    "transplant_values": reference_captures,
                },
            }

            recipient_global = global_metrics(loaded, recipient["first_logits"], case["target"])
            baseline_similarity = set_similarity(reference_captures, recipient_captures)
            for condition in CONDITIONS:
                if condition == "correct_baseline":
                    result = correct
                elif condition == "correct_carrier_joint_zero":
                    result = phase327.forward_probe(
                        loaded,
                        case,
                        case["target"],
                        case["distractors"],
                        mutate_specs=specs,
                        capture_specs=specs,
                    )
                elif condition == "recipient_baseline":
                    result = recipient
                else:
                    result = phase327.forward_probe(
                        loaded,
                        recipient_case,
                        case["target"],
                        case["distractors"],
                        capture_specs=specs,
                        **settings[condition],
                    )
                metrics = global_metrics(loaded, result["first_logits"], case["target"])
                current_captures = effective_captures(condition, result, reference_captures)
                similarity = set_similarity(reference_captures, current_captures)
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE,
                    "created_at": now(),
                    "model": model_key,
                    "case_id": case["case_id"],
                    "base_case_id": case["base_case_id"],
                    "family_id": case["family_id"],
                    "mechanism_id": mechanism,
                    "split": case["split"],
                    "template_id": case["template_id"],
                    "condition": condition,
                    "subject": case["subject"],
                    "recipient_subject": recipient_case["subject"],
                    "recipient_natural_target": recipient_case["natural_target"],
                    "evaluation_target": case["target"],
                    "residual_observation_layer": selected_observation_layer,
                    "intervention_input_layer": (
                        wrong_input_layer
                        if condition == "recipient_tokenwise_correct_wrong_layer"
                        else selected_input_layer
                    ),
                    "query_token_count": int(correct_tokens.shape[0]),
                    "frozen_carrier_member_count": len(specs),
                    "recipient_baseline_phrase_logprob_sum": recipient["phrase_logprob_sum"],
                    "phrase_logprob_sum": result["phrase_logprob_sum"],
                    "phrase_logprob_gain": round(
                        result["phrase_logprob_sum"] - recipient["phrase_logprob_sum"], 6
                    ),
                    "recipient_baseline_target_margin": recipient["metrics"]["target_margin"],
                    "target_margin": result["metrics"]["target_margin"],
                    "target_margin_gain": round(
                        result["metrics"]["target_margin"] - recipient["metrics"]["target_margin"], 6
                    ),
                    **metrics,
                    "recipient_baseline_global_target_rank": recipient_global["global_target_rank"],
                    "global_target_rank_gain": (
                        recipient_global["global_target_rank"] - metrics["global_target_rank"]
                    ),
                    "recipient_baseline_global_blocker_count": recipient_global["global_blocker_count"],
                    "global_blocker_decline": (
                        recipient_global["global_blocker_count"] - metrics["global_blocker_count"]
                    ),
                    "correct_carrier_set_similarity": similarity,
                    "recipient_baseline_correct_carrier_set_similarity": baseline_similarity,
                    "carrier_set_similarity_gain": round(similarity - baseline_similarity, 6),
                    "js_divergence_from_recipient": (
                        0.0
                        if condition == "recipient_baseline"
                        else phase327.phase326.js_divergence(
                            recipient["first_logits"], result["first_logits"]
                        )
                    ),
                    "causal_scope": (
                        "none"
                        if condition in {"correct_baseline", "recipient_baseline"}
                        else "registered_distributed_state_or_residual_intervention"
                    ),
                    "single_unit_causal": False,
                }
                rank_rows.append(row)
                competitor_rows.extend(
                    top50_rows(
                        loaded,
                        case,
                        condition,
                        result["first_logits"],
                        metrics["global_target_rank"],
                    )
                )
                carrier_rows.extend(
                    member_rows(
                        loaded,
                        case,
                        condition,
                        specs,
                        reference_captures,
                        recipient_captures,
                        current_captures,
                        row,
                    )
                )

            if case["template_id"] == "template_i":
                generation_settings = {
                    "correct_baseline": (case, {}, None),
                    "correct_carrier_joint_zero": (case, {"mutate_specs": specs}, None),
                    "recipient_baseline": (recipient_case, {}, None),
                    "recipient_tokenwise_correct": (
                        recipient_case,
                        {},
                        patch(selected_input_layer, correct_tokens),
                    ),
                    "recipient_pooled_correct": (
                        recipient_case,
                        {},
                        patch(selected_input_layer, correct_tokens.mean(dim=0)),
                    ),
                    "recipient_natural_carrier_correct": (
                        recipient_case,
                        {
                            "transplant_specs": specs,
                            "transplant_values": reference_captures,
                        },
                        None,
                    ),
                }
                for condition in GENERATION_CONDITIONS:
                    prompt_case, kwargs, residual = generation_settings[condition]
                    rollout = greedy_rollout(
                        loaded,
                        prompt_case,
                        residual_patch=residual,
                        **kwargs,
                    )
                    generation_rows.append(generation_row(loaded, case, condition, rollout))

            if number % 4 == 0 or number == len(cases):
                print(f"[{model_key}/{mechanism}] {number}/{len(cases)}", flush=True)

        expected = {
            "rank": len(cases) * len(CONDITIONS),
            "competitor": len(cases) * len(CONDITIONS) * TOP_K,
            "carrier": len(cases) * len(CONDITIONS) * len(
                carrier_specs_for(carrier_selections, cases[0])
            ),
            "generation": len([case for case in cases if case["template_id"] == "template_i"])
            * len(GENERATION_CONDITIONS),
        }
        actual = {
            "rank": len(rank_rows),
            "competitor": len(competitor_rows),
            "carrier": len(carrier_rows),
            "generation": len(generation_rows),
        }
        if actual != expected:
            raise ValueError(f"Incomplete Phase329 rows: expected={expected}, actual={actual}")
        prefix = f"phase329_{model_key}_{mechanism}"
        write_json(output / f"{prefix}_residual_selection.json", selection)
        write_jsonl(output / f"{prefix}_rank_rows.jsonl", rank_rows)
        write_jsonl(output / f"{prefix}_top50_rows.jsonl", competitor_rows)
        write_jsonl(output / f"{prefix}_carrier_member_rows.jsonl", carrier_rows)
        write_jsonl(output / f"{prefix}_generation_rows.jsonl", generation_rows)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE,
            "created_at": now(),
            "model": model_key,
            "model_name_or_path": str(loaded.spec.local_dir),
            "mechanism_id": mechanism,
            "registered_prompt_count": len(cases),
            "registered_independent_object_count": len({case["base_case_id"] for case in cases}),
            "row_counts": actual,
            "selection": selection,
            "single_unit_causal_count": 0,
        }
        write_json(output / f"{prefix}_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return summary
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def condition_rows(rows: list[dict[str, Any]], condition: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["condition"] == condition]


def audit_model_mechanism(
    model: str,
    mechanism: str,
    rank_rows: list[dict[str, Any]],
    carrier_rows: list[dict[str, Any]],
    generation_rows: list[dict[str, Any]],
    selection: dict[str, Any],
) -> dict[str, Any]:
    by_condition = {condition: condition_rows(rank_rows, condition) for condition in CONDITIONS}

    def avg(condition: str, field: str) -> float:
        return safe_mean([float(row[field]) for row in by_condition[condition]])

    def bool_rate(condition: str, field: str) -> float:
        return rate([bool(row[field]) for row in by_condition[condition]])

    token_phrase = avg("recipient_tokenwise_correct", "phrase_logprob_gain")
    pooled_phrase = avg("recipient_pooled_correct", "phrase_logprob_gain")
    token_rank = avg("recipient_tokenwise_correct", "global_target_rank_gain")
    pooled_rank = avg("recipient_pooled_correct", "global_target_rank_gain")
    same_phrase = avg("recipient_tokenwise_same_target", "phrase_logprob_gain")
    controls = (
        "recipient_tokenwise_wrong_target",
        "recipient_tokenwise_unrelated",
        "recipient_tokenwise_norm_matched_unrelated",
        "recipient_tokenwise_shuffled",
        "recipient_tokenwise_correct_wrong_layer",
    )
    best_control_phrase = max(avg(condition, "phrase_logprob_gain") for condition in controls)
    best_control_rank = max(avg(condition, "global_target_rank_gain") for condition in controls)
    phrase_specificity = round(min(token_phrase, same_phrase) - best_control_phrase, 6)
    rank_specificity = round(
        min(token_rank, avg("recipient_tokenwise_same_target", "global_target_rank_gain"))
        - best_control_rank,
        6,
    )
    tokenwise_consistency = rate([
        float(row["phrase_logprob_gain"]) > 0 and float(row["global_target_rank_gain"]) > 0
        for row in by_condition["recipient_tokenwise_correct"]
    ])
    blocker_decline = avg("recipient_tokenwise_correct", "global_blocker_decline")
    blocker_consistency = rate([
        float(row["global_blocker_decline"]) > 0
        for row in by_condition["recipient_tokenwise_correct"]
    ])
    recipient_top1 = bool_rate("recipient_baseline", "global_target_top1")
    tokenwise_top1 = bool_rate("recipient_tokenwise_correct", "global_target_top1")
    best_control_top1 = max(bool_rate(condition, "global_target_top1") for condition in controls)
    tokenwise_beats_pooled = (
        token_phrase > 0.0
        and token_rank > 0.0
        and token_phrase > pooled_phrase
        and token_rank > pooled_rank
    )
    donor_specificity_pass = phrase_specificity > 0.0 and rank_specificity > 0.0
    blocker_pass = blocker_decline > 0.0 and blocker_consistency >= 0.65
    top1_unlock = tokenwise_top1 > recipient_top1 and tokenwise_top1 > best_control_top1
    mean_js = avg("recipient_tokenwise_correct", "js_divergence_from_recipient")
    side_effect_pass = mean_js <= 0.05

    member_case_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in carrier_rows:
        member_case_group[(row["case_id"], row["condition"])].append(row)
    case_carrier_gain: dict[str, list[float]] = defaultdict(list)
    for (case_id, condition), values in member_case_group.items():
        case_carrier_gain[condition].append(
            safe_mean([float(row["cosine_gain_toward_correct"]) for row in values])
        )
    correct_carrier_gain = safe_mean(case_carrier_gain["recipient_tokenwise_correct"])
    same_carrier_gain = safe_mean(case_carrier_gain["recipient_tokenwise_same_target"])
    control_carrier_gain = max(safe_mean(case_carrier_gain[condition]) for condition in controls)
    carrier_specificity = round(
        min(correct_carrier_gain, same_carrier_gain) - control_carrier_gain,
        6,
    )
    correct_members = condition_rows(carrier_rows, "recipient_tokenwise_correct")
    member_positive_rate = rate([
        float(row["cosine_gain_toward_correct"]) > 0 for row in correct_members
    ])
    cochange_by_case = []
    for row in by_condition["recipient_tokenwise_correct"]:
        member_values = member_case_group[(row["case_id"], "recipient_tokenwise_correct")]
        member_gain = safe_mean([
            float(member["cosine_gain_toward_correct"]) for member in member_values
        ])
        rank_gain = float(row["global_target_rank_gain"])
        cochange_by_case.append((rank_gain > 0 and member_gain > 0) or (rank_gain <= 0 and member_gain <= 0))
    member_rank_cochange_consistency = rate(cochange_by_case)
    member_mediation_pass = (
        correct_carrier_gain > 0.0
        and carrier_specificity > 0.0
        and member_positive_rate >= 0.65
        and member_rank_cochange_consistency >= 0.65
        and token_rank > 0.0
    )

    generation_by_condition = {
        condition: condition_rows(generation_rows, condition)
        for condition in GENERATION_CONDITIONS
    }

    def generation_rate(condition: str, field: str) -> float:
        return rate([bool(row[field]) for row in generation_by_condition[condition]])

    recipient_generation = generation_rate("recipient_baseline", "target_alias_prefix_match")
    token_generation = generation_rate("recipient_tokenwise_correct", "target_alias_prefix_match")
    pooled_generation = generation_rate("recipient_pooled_correct", "target_alias_prefix_match")
    correct_generation = generation_rate("correct_baseline", "target_alias_prefix_match")
    zero_generation = generation_rate("correct_carrier_joint_zero", "target_alias_prefix_match")
    natural_carrier_generation = generation_rate(
        "recipient_natural_carrier_correct", "target_alias_prefix_match"
    )
    generation_improvement_pass = token_generation > recipient_generation
    full_candidate = (
        bool(selection["positive_identity_at_selection"])
        and tokenwise_beats_pooled
        and donor_specificity_pass
        and tokenwise_consistency >= 0.65
        and blocker_pass
        and member_mediation_pass
        and top1_unlock
        and generation_improvement_pass
        and side_effect_pass
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "model": model,
        "family_id": "content_knowledge",
        "mechanism_id": mechanism,
        "registered_prompt_count": len(rank_rows) // len(CONDITIONS),
        "residual_observation_layer": selection["residual_observation_layer"],
        "intervention_input_layer": selection["intervention_input_layer"],
        "residual_target_identity_specificity": selection["target_identity_specificity"],
        "positive_identity_at_selection": selection["positive_identity_at_selection"],
        "recipient_phrase_logprob_gain_tokenwise": token_phrase,
        "recipient_phrase_logprob_gain_pooled": pooled_phrase,
        "recipient_rank_gain_tokenwise": token_rank,
        "recipient_rank_gain_pooled": pooled_rank,
        "tokenwise_beats_pooled": tokenwise_beats_pooled,
        "tokenwise_phrase_donor_specificity": phrase_specificity,
        "tokenwise_rank_donor_specificity": rank_specificity,
        "tokenwise_donor_specificity_pass": donor_specificity_pass,
        "tokenwise_positive_consistency": tokenwise_consistency,
        "mean_full_vocabulary_blocker_decline": blocker_decline,
        "blocker_decline_positive_consistency": blocker_consistency,
        "blocker_decline_pass": blocker_pass,
        "recipient_baseline_top1_rate": recipient_top1,
        "recipient_tokenwise_top1_rate": tokenwise_top1,
        "best_control_top1_rate": best_control_top1,
        "top1_unlock_pass": top1_unlock,
        "mean_js_divergence": mean_js,
        "side_effect_threshold": 0.05,
        "low_side_effect_pass": side_effect_pass,
        "correct_carrier_member_similarity_gain": correct_carrier_gain,
        "same_target_carrier_member_similarity_gain": same_carrier_gain,
        "best_control_carrier_member_similarity_gain": control_carrier_gain,
        "carrier_member_donor_specificity": carrier_specificity,
        "carrier_member_positive_rate": member_positive_rate,
        "member_rank_cochange_consistency": member_rank_cochange_consistency,
        "carrier_member_mediation_pass": member_mediation_pass,
        "correct_generation_alias_rate": correct_generation,
        "correct_carrier_zero_generation_alias_rate": zero_generation,
        "recipient_generation_alias_rate": recipient_generation,
        "recipient_tokenwise_generation_alias_rate": token_generation,
        "recipient_pooled_generation_alias_rate": pooled_generation,
        "recipient_natural_carrier_generation_alias_rate": natural_carrier_generation,
        "generation_improvement_pass": generation_improvement_pass,
        "full_chain_candidate": full_candidate,
        "single_unit_intervention_gate_open": full_candidate,
        "single_unit_causal": False,
        "evidence_boundary": (
            "distributed full-vocabulary chain candidate; no single-neuron evidence"
            if full_candidate
            else "one or more identity, tokenwise, blocker, member, top1, generation, or side-effect criteria failed"
        ),
    }


def collect_model(model: str, round_name: str) -> dict[str, Any]:
    output = OUT / round_name
    rank_rows: list[dict[str, Any]] = []
    competitor_rows: list[dict[str, Any]] = []
    carrier_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    audits = []
    selections = []
    for mechanism in MECHANISMS:
        prefix = f"phase329_{model}_{mechanism}"
        selection = read_json(output / f"{prefix}_residual_selection.json")
        selection_rows = read_jsonl(output / f"{prefix}_rank_rows.jsonl")
        selection_competitors = read_jsonl(output / f"{prefix}_top50_rows.jsonl")
        selection_carriers = read_jsonl(output / f"{prefix}_carrier_member_rows.jsonl")
        selection_generation = read_jsonl(output / f"{prefix}_generation_rows.jsonl")
        expected = (288, 14400, 3456, 72)
        actual = (
            len(selection_rows),
            len(selection_competitors),
            len(selection_carriers),
            len(selection_generation),
        )
        if actual != expected:
            raise ValueError(f"Incomplete {model}/{mechanism}: expected={expected}, actual={actual}")
        selections.append(selection)
        rank_rows.extend(selection_rows)
        competitor_rows.extend(selection_competitors)
        carrier_rows.extend(selection_carriers)
        generation_rows.extend(selection_generation)
        audits.append(
            audit_model_mechanism(
                model,
                mechanism,
                selection_rows,
                selection_carriers,
                selection_generation,
                selection,
            )
        )
    write_jsonl(output / f"phase329_{model}_rank_rows.jsonl", rank_rows)
    write_jsonl(output / f"phase329_{model}_top50_rows.jsonl", competitor_rows)
    write_jsonl(output / f"phase329_{model}_carrier_member_rows.jsonl", carrier_rows)
    write_jsonl(output / f"phase329_{model}_generation_rows.jsonl", generation_rows)
    write_jsonl(output / f"phase329_{model}_mechanism_audits.jsonl", audits)
    write_jsonl(output / f"phase329_{model}_residual_selections.jsonl", selections)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "model": model,
        "registered_prompt_count": 72,
        "registered_independent_object_count": 36,
        "rank_transition_row_count": len(rank_rows),
        "top50_competitor_row_count": len(competitor_rows),
        "carrier_member_row_count": len(carrier_rows),
        "generation_row_count": len(generation_rows),
        "mechanism_audits": audits,
        "tokenwise_beats_pooled_count": sum(row["tokenwise_beats_pooled"] for row in audits),
        "blocker_decline_pass_count": sum(row["blocker_decline_pass"] for row in audits),
        "carrier_member_mediation_pass_count": sum(row["carrier_member_mediation_pass"] for row in audits),
        "top1_unlock_pass_count": sum(row["top1_unlock_pass"] for row in audits),
        "generation_improvement_pass_count": sum(row["generation_improvement_pass"] for row in audits),
        "full_chain_candidate_count": sum(row["full_chain_candidate"] for row in audits),
        "single_unit_causal_count": 0,
    }
    write_json(output / f"phase329_{model}_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def blocker_type_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["is_full_vocabulary_blocker"]:
            grouped[(
                row["model"],
                row["mechanism_id"],
                row["condition"],
                row["blocker_category"],
            )].append(row)
    result = []
    for (model, mechanism, condition, category), values in sorted(grouped.items()):
        result.append({
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE,
            "created_at": now(),
            "model": model,
            "family_id": "content_knowledge",
            "mechanism_id": mechanism,
            "condition": condition,
            "blocker_category": category,
            "top50_blocker_observation_count": len(values),
            "independent_case_count": len({row["case_id"] for row in values}),
            "mean_logit_above_target": safe_mean([
                float(row["logit_above_target"]) for row in values
            ]),
            "taxonomy_scope": "registered top50 blocker categories; exact all-vocabulary blocker count is rank minus one",
        })
    return result


def condition_summaries(
    rank_rows: list[dict[str, Any]],
    competitor_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    top1_category = {
        (row["model"], row["case_id"], row["condition"]): row["blocker_category"]
        for row in competitor_rows
        if int(row["top50_rank"]) == 1
    }
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rank_rows:
        grouped[(row["model"], row["mechanism_id"], row["condition"])].append(row)
    result = []
    for (model, mechanism, condition), values in sorted(grouped.items()):
        top1_counts = Counter(
            top1_category[(model, row["case_id"], condition)] for row in values
        )
        result.append({
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE,
            "created_at": now(),
            "model": model,
            "family_id": "content_knowledge",
            "mechanism_id": mechanism,
            "condition": condition,
            "registered_prompt_count": len(values),
            "mean_phrase_logprob_gain": safe_mean([
                float(row["phrase_logprob_gain"]) for row in values
            ]),
            "mean_global_target_rank": safe_mean([
                float(row["global_target_rank"]) for row in values
            ]),
            "mean_global_target_rank_gain": safe_mean([
                float(row["global_target_rank_gain"]) for row in values
            ]),
            "mean_global_blocker_decline": safe_mean([
                float(row["global_blocker_decline"]) for row in values
            ]),
            "mean_carrier_set_similarity_gain": safe_mean([
                float(row["carrier_set_similarity_gain"]) for row in values
            ]),
            "mean_js_divergence_from_recipient": safe_mean([
                float(row["js_divergence_from_recipient"]) for row in values
            ]),
            "target_top1_rate": rate([bool(row["global_target_top1"]) for row in values]),
            "target_top5_rate": rate([bool(row["global_target_top5"]) for row in values]),
            "target_top50_rate": rate([bool(row["global_target_top50"]) for row in values]),
            "top1_category_counts": dict(sorted(top1_counts.items())),
        })
    return result


def rank_band(rank_value: int) -> str:
    if rank_value == 1:
        return "rank_1"
    if rank_value <= 5:
        return "rank_2_5"
    if rank_value <= 50:
        return "rank_6_50"
    if rank_value <= 100:
        return "rank_51_100"
    if rank_value <= 1000:
        return "rank_101_1000"
    return "rank_above_1000"


def rank_band_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["mechanism_id"], row["condition"])].append(row)
    result = []
    for (model, mechanism, condition), values in sorted(grouped.items()):
        counts = Counter(rank_band(int(row["global_target_rank"])) for row in values)
        result.append({
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE,
            "created_at": now(),
            "model": model,
            "family_id": "content_knowledge",
            "mechanism_id": mechanism,
            "condition": condition,
            "registered_prompt_count": len(values),
            "rank_band_counts": {
                band: counts.get(band, 0)
                for band in (
                    "rank_1", "rank_2_5", "rank_6_50", "rank_51_100",
                    "rank_101_1000", "rank_above_1000",
                )
            },
        })
    return result


def annotate_generation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        words = phase327.normalized_words(row["generated_text"])
        matched = None
        for alias in row["target_aliases"]:
            alias_words = phase327.normalized_words(alias)
            if alias_words and any(
                words[index:index + len(alias_words)] == alias_words
                for index in range(max(0, len(words) - len(alias_words) + 1))
            ):
                matched = alias
                break
        result.append({
            **row,
            "target_alias_anywhere_match": matched is not None,
            "matched_alias_anywhere": matched,
            "prefix_match_required_for_generation_success": True,
        })
    return result


def collect(round_name: str) -> dict[str, Any]:
    output = OUT / round_name
    summaries = [read_json(output / f"phase329_{model}_summary.json") for model in MODELS]
    rank_rows = [
        row for model in MODELS
        for row in read_jsonl(output / f"phase329_{model}_rank_rows.jsonl")
    ]
    competitor_rows = [
        row for model in MODELS
        for row in read_jsonl(output / f"phase329_{model}_top50_rows.jsonl")
    ]
    carrier_rows = [
        row for model in MODELS
        for row in read_jsonl(output / f"phase329_{model}_carrier_member_rows.jsonl")
    ]
    generation_rows = annotate_generation_rows([
        row for model in MODELS
        for row in read_jsonl(output / f"phase329_{model}_generation_rows.jsonl")
    ])
    audits = [row for summary in summaries for row in summary["mechanism_audits"]]
    selections = [
        row for model in MODELS
        for row in read_jsonl(output / f"phase329_{model}_residual_selections.jsonl")
    ]
    expected = {
        "rank": 2592,
        "competitor": 129600,
        "carrier": 31104,
        "generation": 648,
        "audits": 9,
        "selections": 9,
    }
    actual = {
        "rank": len(rank_rows),
        "competitor": len(competitor_rows),
        "carrier": len(carrier_rows),
        "generation": len(generation_rows),
        "audits": len(audits),
        "selections": len(selections),
    }
    if actual != expected:
        raise ValueError(f"Incomplete cross-model Phase329 rows: expected={expected}, actual={actual}")

    mechanism_rows = []
    for mechanism in MECHANISMS:
        rows = [row for row in audits if row["mechanism_id"] == mechanism]
        mechanism_rows.append({
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE,
            "created_at": now(),
            "family_id": "content_knowledge",
            "mechanism_id": mechanism,
            "model_count": len(rows),
            "positive_residual_identity_models": [
                row["model"] for row in rows if row["positive_identity_at_selection"]
            ],
            "tokenwise_beats_pooled_models": [
                row["model"] for row in rows if row["tokenwise_beats_pooled"]
            ],
            "donor_specificity_models": [
                row["model"] for row in rows if row["tokenwise_donor_specificity_pass"]
            ],
            "blocker_decline_models": [
                row["model"] for row in rows if row["blocker_decline_pass"]
            ],
            "carrier_member_mediation_models": [
                row["model"] for row in rows if row["carrier_member_mediation_pass"]
            ],
            "top1_unlock_models": [row["model"] for row in rows if row["top1_unlock_pass"]],
            "generation_improvement_models": [
                row["model"] for row in rows if row["generation_improvement_pass"]
            ],
            "full_chain_candidate_models": [
                row["model"] for row in rows if row["full_chain_candidate"]
            ],
        })
    for row in mechanism_rows:
        row["cross_model_tokenwise_beats_pooled"] = len(row["tokenwise_beats_pooled_models"]) >= 2
        row["cross_model_blocker_decline"] = len(row["blocker_decline_models"]) >= 2
        row["cross_model_carrier_member_mediation"] = len(row["carrier_member_mediation_models"]) >= 2
        row["cross_model_top1_unlock"] = len(row["top1_unlock_models"]) >= 2
        row["cross_model_generation_improvement"] = len(row["generation_improvement_models"]) >= 2
        row["cross_model_full_chain_candidate"] = len(row["full_chain_candidate_models"]) >= 2

    blocker_summaries = blocker_type_summaries(competitor_rows)
    condition_summary_rows = condition_summaries(rank_rows, competitor_rows)
    rank_band_rows = rank_band_summaries(rank_rows)
    cross = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "models": list(MODELS),
        "mechanisms": list(MECHANISMS),
        "registered_prompt_model_count": 216,
        "registered_independent_object_model_count": 108,
        "row_counts": actual,
        "mechanism_results": mechanism_rows,
        "cross_model_tokenwise_beats_pooled_mechanism_count": sum(
            row["cross_model_tokenwise_beats_pooled"] for row in mechanism_rows
        ),
        "cross_model_blocker_decline_mechanism_count": sum(
            row["cross_model_blocker_decline"] for row in mechanism_rows
        ),
        "cross_model_carrier_member_mediation_mechanism_count": sum(
            row["cross_model_carrier_member_mediation"] for row in mechanism_rows
        ),
        "cross_model_top1_unlock_mechanism_count": sum(
            row["cross_model_top1_unlock"] for row in mechanism_rows
        ),
        "cross_model_generation_improvement_mechanism_count": sum(
            row["cross_model_generation_improvement"] for row in mechanism_rows
        ),
        "cross_model_full_chain_candidate_count": sum(
            row["cross_model_full_chain_candidate"] for row in mechanism_rows
        ),
        "single_unit_intervention_gate_open": any(
            row["cross_model_full_chain_candidate"] for row in mechanism_rows
        ),
        "single_unit_causal_count": 0,
        "closure_claim": False,
        "small_model_boundary": (
            "These are three local small-model implementations. Cross-model agreement is evidence "
            "of recurrence, not proof that larger language models use the same physical path."
        ),
    }
    write_jsonl(output / "phase329A_rank_transition_rows.jsonl", rank_rows)
    write_jsonl(output / "phase329A_top50_competitors.jsonl", competitor_rows)
    write_jsonl(output / "phase329A_blocker_types.jsonl", blocker_summaries)
    write_jsonl(output / "phase329A_condition_summaries.jsonl", condition_summary_rows)
    write_jsonl(output / "phase329A_rank_band_summaries.jsonl", rank_band_rows)
    write_jsonl(
        output / "phase329B_tokenwise_query_transplant_rows.jsonl",
        [row for row in rank_rows if "tokenwise" in row["condition"]],
    )
    write_jsonl(
        output / "phase329B_control_rows.jsonl",
        [row for row in rank_rows if row["condition"] not in {
            "recipient_tokenwise_correct", "recipient_pooled_correct"
        }],
    )
    write_jsonl(output / "phase329B_layer_position_rows.jsonl", selections)
    write_jsonl(output / "phase329C_carrier_member_mediation_rows.jsonl", carrier_rows)
    write_jsonl(output / "phase329C_member_similarity_rows.jsonl", audits)
    write_jsonl(output / "phase329C_member_rank_effect_rows.jsonl", audits)
    write_jsonl(output / "phase329D_full_generation_rows.jsonl", generation_rows)
    write_jsonl(
        output / "phase329D_alias_match_rows.jsonl",
        [{
            "model": row["model"],
            "case_id": row["case_id"],
            "mechanism_id": row["mechanism_id"],
            "condition": row["condition"],
            "evaluation_target": row["evaluation_target"],
            "target_aliases": row["target_aliases"],
            "target_alias_prefix_match": row["target_alias_prefix_match"],
            "matched_alias": row["matched_alias"],
        } for row in generation_rows],
    )
    write_jsonl(output / "phase329D_rollout_rows.jsonl", generation_rows)
    write_jsonl(output / "phase329_model_mechanism_audits.jsonl", audits)
    write_jsonl(output / "phase329_cross_model_mechanisms.jsonl", mechanism_rows)
    write_json(output / "phase329_cross_model_summary.json", cross)
    print(json.dumps(cross, ensure_ascii=False, indent=2), flush=True)
    return cross


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--mechanism", choices=MECHANISMS)
    parser.add_argument("--round", default=ROUND_DEFAULT)
    parser.add_argument("--collect-model", action="store_true")
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    if args.collect:
        collect(args.round)
    elif args.collect_model:
        if not args.model:
            parser.error("--collect-model requires --model")
        collect_model(args.model, args.round)
    else:
        if not args.model or not args.mechanism:
            parser.error("model execution requires --model and --mechanism")
        run_model_mechanism(args.model, args.mechanism, args.round)


if __name__ == "__main__":
    main()
