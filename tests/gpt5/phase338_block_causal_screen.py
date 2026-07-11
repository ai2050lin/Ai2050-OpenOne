#!/usr/bin/env python3
"""Run staged Phase338 coarse-block phrase and rollout interventions."""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase331_refined_mechanism_audit import target_match  # noqa: E402
from phase333_dynamic_survey import continuation_ids  # noqa: E402
from phase334_natural_contrast_survey import component_tensor, role_positions  # noqa: E402
from phase338_block_causal_case_bank import (  # noqa: E402
    DEPTH_BINS, OUT, PHASE, POSITION_ROLES, ROUND_DEFAULT, SCHEMA_VERSION,
)


MODELS = ("qwen3", "glm4", "deepseek7b")
DISCOVERY_CONDITION = "zero"
CALIBRATION_CONDITIONS = ("zero", "half", "permutation")
HELDOUT_CONDITIONS = (
    "baseline", "correct_zero", "correct_half", "correct_permutation",
    "wrong_depth_zero", "wrong_position_zero",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def replace_output(output: Any, tensor: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return tensor
    if isinstance(output, tuple):
        return (tensor, *output[1:])
    if isinstance(output, list):
        return [tensor, *output[1:]]
    raise TypeError(f"Unsupported output type {type(output).__name__}")


def layers_for_bin(layer_count: int, depth_bin: str) -> list[int]:
    boundaries = (0, layer_count // 3, (2 * layer_count) // 3, layer_count)
    index = DEPTH_BINS.index(depth_bin)
    return list(range(boundaries[index], boundaries[index + 1]))


def mutate_vectors(vectors: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "zero":
        return torch.zeros_like(vectors)
    if mode == "half":
        return vectors * 0.5
    if mode == "permutation":
        return torch.roll(vectors, shifts=max(1, vectors.shape[-1] // 7), dims=-1)
    raise KeyError(mode)


def install_block_hooks(
    loaded: Any, block: dict[str, Any], positions: list[int], mode: str
) -> list[Any]:
    layers = get_layers(loaded.model)
    selected = set(layers_for_bin(len(layers), block["depth_bin"]))
    component = block["component"]
    handles = []
    layer_inputs: dict[int, torch.Tensor] = {}
    position_tensor = torch.tensor(positions, dtype=torch.long, device=loaded.input_device)

    for layer_index, layer in enumerate(layers):
        if layer_index not in selected:
            continue

        def pre(_module: Any, inputs: tuple[Any, ...], idx: int = layer_index) -> None:
            if inputs and torch.is_tensor(inputs[0]):
                layer_inputs[idx] = inputs[0].detach()

        def patch_component(output: Any, idx: int, kind: str) -> Any:
            tensor = component_tensor(output)
            if tensor.ndim != 3 or tensor.shape[0] != position_tensor.numel():
                return output
            if int(position_tensor.max().item()) >= tensor.shape[1]:
                return output
            patched = tensor.clone()
            batch = torch.arange(tensor.shape[0], device=tensor.device)
            pos = position_tensor.to(tensor.device)
            natural = tensor[batch, pos]
            if kind == "residual_increment":
                before = layer_inputs[idx].to(tensor.device)
                increment = natural - before[batch, pos]
                patched[batch, pos] = before[batch, pos] + mutate_vectors(increment, mode)
            else:
                patched[batch, pos] = mutate_vectors(natural, mode)
            return replace_output(output, patched)

        def attention(
            _module: Any, _inputs: tuple[Any, ...], output: Any, idx: int = layer_index
        ) -> Any:
            return patch_component(output, idx, "attention_output") if component == "attention_output" else output

        def mlp(
            _module: Any, _inputs: tuple[Any, ...], output: Any, idx: int = layer_index
        ) -> Any:
            return patch_component(output, idx, "mlp_output") if component == "mlp_output" else output

        def residual(
            _module: Any, _inputs: tuple[Any, ...], output: Any, idx: int = layer_index
        ) -> Any:
            return patch_component(output, idx, "residual_increment") if component == "residual_increment" else output

        handles.append(layer.register_forward_pre_hook(pre))
        if component == "attention_output":
            handles.append(layer.self_attn.register_forward_hook(attention))
        elif component == "mlp_output":
            handles.append(layer.mlp.register_forward_hook(mlp))
        else:
            handles.append(layer.register_forward_hook(residual))
    return handles


def prompt_ids(loaded: Any, case: dict[str, Any]) -> list[int]:
    ids = loaded.tokenizer(
        case["prompt"], add_special_tokens=bool(case["tokenization_add_special_tokens"]),
        truncation=True, max_length=256,
    )["input_ids"]
    return [int(value) for value in ids]


@torch.inference_mode()
def score_cases(
    loaded: Any, cases: list[dict[str, Any]], block: dict[str, Any] | None, mode: str | None
) -> list[dict[str, float]]:
    pad = int(loaded.tokenizer.pad_token_id)
    sequences: list[list[int]] = []
    prompt_lengths: list[int] = []
    answer_ids: list[list[int]] = []
    positions: list[int] = []
    option_case_indices: list[int] = []
    option_indices: list[int] = []
    for case_index, case in enumerate(cases):
        prompt = prompt_ids(loaded, case)
        role_map = role_positions(loaded, case, prompt)
        values = [case["target"], *case["distractors"]]
        for option_index, value in enumerate(values):
            continuation = continuation_ids(loaded, case, value)
            sequences.append(prompt + continuation)
            prompt_lengths.append(len(prompt))
            answer_ids.append(continuation)
            positions.append(role_map[block["position_role"]][0] if block else len(prompt) - 1)
            option_case_indices.append(case_index)
            option_indices.append(option_index)
    width = max(map(len, sequences))
    input_ids = torch.full(
        (len(sequences), width), pad, dtype=torch.long, device=loaded.input_device
    )
    attention_mask = torch.zeros_like(input_ids)
    for index, sequence in enumerate(sequences):
        input_ids[index, :len(sequence)] = torch.tensor(sequence, device=loaded.input_device)
        attention_mask[index, :len(sequence)] = 1
    handles = install_block_hooks(loaded, block, positions, mode) if block and mode else []
    try:
        output = loaded.model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True
        )
    finally:
        for handle in handles:
            handle.remove()
    log_probs = torch.log_softmax(output.logits.detach().float(), dim=-1)
    values_by_case: dict[int, dict[int, float]] = defaultdict(dict)
    for row_index, (case_index, option_index, prompt_length, ids) in enumerate(zip(
        option_case_indices, option_indices, prompt_lengths, answer_ids, strict=True
    )):
        values = [
            float(log_probs[row_index, prompt_length + offset - 1, token_id].item())
            for offset, token_id in enumerate(ids)
        ]
        values_by_case[case_index][option_index] = mean(values)
    results = []
    for case_index in range(len(cases)):
        option = values_by_case[case_index]
        distractor = max(option[1], option[2])
        results.append({
            "target_phrase_mean_logprob": option[0],
            "best_distractor_phrase_mean_logprob": distractor,
            "phrase_margin": option[0] - distractor,
        })
    del output, log_probs, input_ids, attention_mask
    return results


def row_base(case: dict[str, Any], model: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
        "model": model, "case_id": case["case_id"],
        "semantic_case_id": case["semantic_case_id"], "family_id": case["family_id"],
        "mechanism_id": case["mechanism_id"], "item_index": case["item_index"],
        "split": case["split"], "template_id": case["template_id"],
        "interface": case["interface"], "target": case["target"],
    }


def summarize_block_rows(
    rows: list[dict[str, Any]], protocol: dict[str, Any], stage: str
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["condition"] != "baseline":
            grouped[row["block_id"]].append(row)
    summaries = []
    for block_id, values in sorted(grouped.items()):
        zero = [row for row in values if row["condition"] == "zero"]
        losses = [row["phrase_margin_loss_vs_baseline"] for row in zero]
        zero_mean = mean(losses) if losses else 0.0
        positive_rate = sum(value > 0 for value in losses) / len(losses) if losses else 0.0
        summary = {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "model": values[0]["model"], "stage": stage, "block_id": block_id,
            "component": values[0]["component"], "depth_bin": values[0]["depth_bin"],
            "position_role": values[0]["position_role"], "case_count": len(zero),
            "mean_zero_phrase_margin_loss": round(zero_mean, 7),
            "zero_positive_case_rate": round(positive_rate, 7),
        }
        if stage == "discovery":
            threshold = protocol["thresholds"]
            summary["stage_gate_pass"] = bool(
                zero_mean >= threshold["discovery_mean_phrase_loss_min"]
                and positive_rate >= threshold["discovery_positive_case_rate_min"]
            )
        else:
            half = [row["phrase_margin_loss_vs_baseline"] for row in values if row["condition"] == "half"]
            permutation = [
                row["phrase_margin_loss_vs_baseline"] for row in values
                if row["condition"] == "permutation"
            ]
            half_mean = mean(half) if half else 0.0
            permutation_mean = mean(permutation) if permutation else 0.0
            summary.update({
                "mean_half_phrase_margin_loss": round(half_mean, 7),
                "mean_permutation_phrase_margin_loss": round(permutation_mean, 7),
                "zero_minus_permutation": round(zero_mean - permutation_mean, 7),
                "joint_structural_score": round(min(zero_mean, permutation_mean), 7),
            })
            threshold = protocol["thresholds"]
            summary["stage_gate_pass"] = bool(
                zero_mean >= threshold["calibration_mean_phrase_loss_min"]
                and positive_rate >= threshold["calibration_positive_case_rate_min"]
                and permutation_mean >= threshold["calibration_permutation_phrase_loss_min"]
            )
        summaries.append(summary)
    return summaries


def run_phrase_stage(model: str, stage: str, round_name: str) -> dict[str, Any]:
    root = OUT / round_name
    protocol = read_json(root / "phase338_registered_protocol.json")
    all_cases = [row for row in read_jsonl(root / "phase338_registered_cases.jsonl") if row["model"] == model]
    cases = [row for row in all_cases if row["split"] == stage]
    blocks = read_jsonl(root / "phase338_registered_blocks.jsonl")
    model_root = root / "models" / model
    if stage == "calibration":
        blocks = read_jsonl(model_root / "phase338_discovery_top_blocks.jsonl")
        conditions = CALIBRATION_CONDITIONS
    else:
        conditions = (DISCOVERY_CONDITION,)
    loaded = None
    rows: list[dict[str, Any]] = []
    try:
        loaded = load_probe_model(model)
        for batch_start in range(0, len(cases), 6):
            batch = cases[batch_start:batch_start + 6]
            baseline = score_cases(loaded, batch, None, None)
            for case, score in zip(batch, baseline, strict=True):
                rows.append({
                    **row_base(case, model), "stage": stage, "condition": "baseline",
                    "block_id": None, "component": None, "depth_bin": None,
                    "position_role": None, **{key: round(value, 7) for key, value in score.items()},
                    "phrase_margin_loss_vs_baseline": 0.0,
                })
            for block in blocks:
                for condition in conditions:
                    scores = score_cases(loaded, batch, block, condition)
                    for case, score, base in zip(batch, scores, baseline, strict=True):
                        rows.append({
                            **row_base(case, model), "stage": stage, "condition": condition,
                            **block, **{key: round(value, 7) for key, value in score.items()},
                            "phrase_margin_loss_vs_baseline": round(
                                base["phrase_margin"] - score["phrase_margin"], 7
                            ),
                            "evidence_level": "L3_coarse_block_intervention",
                            "single_unit_causal": False,
                        })
            print(f"[{model}] {stage} {min(batch_start + 6, len(cases))}/{len(cases)}", flush=True)
        summaries = summarize_block_rows(rows, protocol, stage)
        if stage == "discovery":
            frozen = sorted(
                summaries,
                key=lambda row: (
                    row["stage_gate_pass"], row["mean_zero_phrase_margin_loss"],
                    row["zero_positive_case_rate"],
                ), reverse=True,
            )[:3]
            write_jsonl(model_root / "phase338_discovery_top_blocks.jsonl", frozen)
        else:
            passing = [row for row in summaries if row["stage_gate_pass"]]
            frozen = sorted(
                passing,
                key=lambda row: (
                    row["joint_structural_score"], row["mean_zero_phrase_margin_loss"],
                ), reverse=True,
            )[:1]
            write_jsonl(model_root / "phase338_frozen_heldout_block.jsonl", frozen)
        write_jsonl(model_root / f"phase338_{stage}_rows.jsonl", rows)
        write_jsonl(model_root / f"phase338_{stage}_block_summary.jsonl", summaries)
        complete = {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "model": model, "stage": stage, "case_count": len(cases),
            "block_count": len(blocks), "condition_row_count": len(rows),
            "stage_gate_block_count": sum(row["stage_gate_pass"] for row in summaries),
            "frozen_next_stage_block_count": len(frozen), "valid": True,
        }
        write_json(model_root / f"phase338_{stage}_complete.json", complete)
        return complete
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def wrong_block(block: dict[str, Any], axis: str) -> dict[str, Any]:
    result = dict(block)
    if axis == "depth":
        index = (DEPTH_BINS.index(block["depth_bin"]) + 1) % len(DEPTH_BINS)
        result["depth_bin"] = DEPTH_BINS[index]
    else:
        index = (POSITION_ROLES.index(block["position_role"]) + 1) % len(POSITION_ROLES)
        result["position_role"] = POSITION_ROLES[index]
    result["block_id"] = f"{result['component']}__{result['depth_bin']}__{result['position_role']}"
    return result


def first_nonempty_line(text: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


@torch.inference_mode()
def generate_case(
    loaded: Any, case: dict[str, Any], block: dict[str, Any] | None,
    mode: str | None, max_new_tokens: int,
) -> dict[str, Any]:
    prompt = prompt_ids(loaded, case)
    encoded = {
        "input_ids": torch.tensor([prompt], dtype=torch.long, device=loaded.input_device),
        "attention_mask": torch.ones((1, len(prompt)), dtype=torch.long, device=loaded.input_device),
    }
    role_map = role_positions(loaded, case, prompt)
    positions = [role_map[block["position_role"]][0]] if block else [len(prompt) - 1]
    handles = install_block_hooks(loaded, block, positions, mode) if block and mode else []
    try:
        generated = loaded.model.generate(
            **encoded, max_new_tokens=max_new_tokens, do_sample=False, use_cache=False,
            pad_token_id=loaded.tokenizer.pad_token_id,
            eos_token_id=loaded.tokenizer.eos_token_id,
        )
    finally:
        for handle in handles:
            handle.remove()
    ids = [int(value) for value in generated[0, len(prompt):].tolist()]
    text = loaded.tokenizer.decode(ids, skip_special_tokens=False)
    head = first_nonempty_line(text)
    return {
        "generated_text": text, "generated_token_ids": ids,
        "generated_token_count": len(ids), "answer_head_text": head,
        "answer_head_semantic_correct": target_match(head, case["target_aliases"]),
    }


def run_heldout(model: str, round_name: str, max_new_tokens: int) -> dict[str, Any]:
    root = OUT / round_name
    all_cases = [row for row in read_jsonl(root / "phase338_registered_cases.jsonl") if row["model"] == model]
    cases = [row for row in all_cases if row["split"] in {"heldout", "private_heldout"}]
    model_root = root / "models" / model
    frozen = read_jsonl(model_root / "phase338_frozen_heldout_block.jsonl")
    loaded = None
    rows: list[dict[str, Any]] = []
    if not frozen:
        complete = {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "model": model, "stage": "heldout", "case_count": 0, "condition_row_count": 0,
            "frozen_block_count": 0, "valid": True, "stopped_after_calibration": True,
        }
        write_json(model_root / "phase338_heldout_complete.json", complete)
        write_jsonl(model_root / "phase338_heldout_rows.jsonl", [])
        return complete
    block = frozen[0]
    wrong_depth = wrong_block(block, "depth")
    wrong_position = wrong_block(block, "position")
    condition_specs = {
        "baseline": (None, None), "correct_zero": (block, "zero"),
        "correct_half": (block, "half"), "correct_permutation": (block, "permutation"),
        "wrong_depth_zero": (wrong_depth, "zero"),
        "wrong_position_zero": (wrong_position, "zero"),
    }
    try:
        loaded = load_probe_model(model)
        for index, case in enumerate(cases, 1):
            baseline_score = score_cases(loaded, [case], None, None)[0]
            case_rows = []
            for condition in HELDOUT_CONDITIONS:
                selected, mode = condition_specs[condition]
                score = baseline_score if condition == "baseline" else score_cases(
                    loaded, [case], selected, mode
                )[0]
                rollout = generate_case(loaded, case, selected, mode, max_new_tokens)
                case_rows.append({
                    **row_base(case, model), "stage": "heldout", "condition": condition,
                    "selected_block_id": block["block_id"],
                    "intervened_block_id": selected["block_id"] if selected else None,
                    "component": selected["component"] if selected else None,
                    "depth_bin": selected["depth_bin"] if selected else None,
                    "position_role": selected["position_role"] if selected else None,
                    **{key: round(value, 7) for key, value in score.items()}, **rollout,
                    "phrase_margin_loss_vs_baseline": round(
                        baseline_score["phrase_margin"] - score["phrase_margin"], 7
                    ),
                    "single_unit_causal": False,
                })
            baseline = next(row for row in case_rows if row["condition"] == "baseline")
            for row in case_rows:
                row["behavior_lost_vs_baseline"] = bool(
                    baseline["answer_head_semantic_correct"]
                    and not row["answer_head_semantic_correct"]
                )
                rows.append(row)
            print(f"[{model}] heldout {index}/{len(cases)}", flush=True)
        write_jsonl(model_root / "phase338_heldout_rows.jsonl", rows)
        complete = {
            "schema_version": SCHEMA_VERSION, "phase_id": PHASE, "created_at": now(),
            "model": model, "stage": "heldout", "case_count": len(cases),
            "condition_row_count": len(rows), "frozen_block_count": 1,
            "heldout_case_count": sum(row["split"] == "heldout" for row in cases),
            "private_heldout_case_count": sum(row["split"] == "private_heldout" for row in cases),
            "valid": len(rows) == len(cases) * len(HELDOUT_CONDITIONS),
            "stopped_after_calibration": False,
        }
        write_json(model_root / "phase338_heldout_complete.json", complete)
        return complete
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--stage", choices=("discovery", "calibration", "heldout"), required=True)
    parser.add_argument("--round", default=ROUND_DEFAULT)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args()
    result = (
        run_heldout(args.model, args.round, args.max_new_tokens)
        if args.stage == "heldout"
        else run_phrase_stage(args.model, args.stage, args.round)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
