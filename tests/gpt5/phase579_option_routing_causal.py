#!/usr/bin/env python3
"""Run frozen Phase579 option-routing interventions on untouched holdout worlds."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))
os.environ.setdefault("PROBE_TORCH_DTYPE", "bfloat16")

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase548_shared_attention_compute_protocol import render_chat  # noqa: E402
from phase573_coarse_message_causal import edge_contribution  # noqa: E402
from phase578_natural_trace import projected_states, span_positions  # noqa: E402
import phase577_natural_choice_protocol as choice  # noqa: E402
import phase578_choice_world_protocol as source  # noqa: E402
import phase579_option_routing_causal_protocol as causal_protocol  # noqa: E402


DISCOVERY_DECISION_PATH = (
    source.OUT_DIR / "phase579_option_routing_causal_discovery_decision.json"
)
RESTORE_CONDITIONS = {
    "option_score_swap_restore",
    "option_weight_swap_restore",
}
PATCH_CONDITIONS = {
    "option_score_swap",
    "option_score_equalize",
    "object_relation_score_swap_control",
    "option_weight_swap",
    "option_value_swap_positive_control",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite(value: float) -> float:
    return float(value) if math.isfinite(value) else 0.0


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def paths(model: str, split: str) -> dict[str, Path]:
    stem = source.OUT_DIR / f"phase579_{model}_{split}_option_routing_causal"
    return {
        "rows": stem.with_name(stem.name + "_rows.jsonl.gz"),
        "summary": stem.with_name(stem.name + "_summary.json"),
        "contract": stem.with_name(stem.name + "_contract.json"),
    }


def load_worlds(
    model: str,
    split: str,
) -> list[tuple[str, list[dict[str, Any]]]]:
    frozen = read_json(causal_protocol.PROTOCOL_PATH)
    if model not in frozen["authorized_models"]:
        raise RuntimeError(f"Phase579 is not authorized for {model}")
    selected = list(
        frozen["causal_holdout_world_ids_by_model_and_split"][model][split]
    )
    allowed_relations = set(choice.RELATIONS)
    if split == "causal_confirmation":
        if not DISCOVERY_DECISION_PATH.exists():
            raise RuntimeError("Phase579 confirmation requires a discovery decision")
        decision = read_json(DISCOVERY_DECISION_PATH)
        allowed_relations = set(
            decision["authorized_confirmation_relations_by_model"].get(model, [])
        )
        if not allowed_relations:
            raise RuntimeError(f"Phase579 confirmation has no authorized branch for {model}")

    selected_set = set(selected)
    bank: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in iter_jsonl(source.SOURCE_CASES_PATH):
        if row["world_id"] not in selected_set:
            continue
        if row["split"] != split or row["sealed"]:
            raise RuntimeError("Phase579 attempted a forbidden source row")
        bank[row["world_id"]][row["variant"]] = row
    worlds = []
    for world_id in selected:
        variants = bank.get(world_id, {})
        if set(variants) != set(choice.VARIANTS):
            raise RuntimeError(f"Phase579 incomplete causal world: {world_id}")
        ordered = [variants[variant] for variant in choice.VARIANTS]
        if ordered[0]["relation"] in allowed_relations:
            worlds.append((world_id, ordered))
    if not worlds:
        raise RuntimeError(f"Phase579 selected no worlds for {model}/{split}")
    return worlds


def prepare_batch(
    loaded: Any,
    model: str,
    worlds: list[tuple[str, list[dict[str, Any]]]],
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    rows = [row for _, variants in worlds for row in variants]
    prompts = [render_chat(loaded.tokenizer, model, row["raw_prompt"]) for row in rows]
    positions = [
        span_positions(loaded.tokenizer, prompt, row)
        for prompt, row in zip(prompts, rows, strict=True)
    ]
    loaded.tokenizer.padding_side = "right"
    encoded = loaded.tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    for batch_index, item in enumerate(positions):
        active = encoded["input_ids"][batch_index][
            encoded["attention_mask"][batch_index].bool()
        ].tolist()
        if [int(value) for value in active] != item["ids"]:
            raise RuntimeError("Phase579 tokenization drift")
    position_ids = encoded["attention_mask"].long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(encoded["attention_mask"] == 0, 0)
    encoded["position_ids"] = position_ids
    metadata = []
    for batch_index, row in enumerate(rows):
        metadata.append(
            {
                "batch_index": batch_index,
                "world_id": row["world_id"],
                "case_id": row["case_id"],
                "variant": row["variant"],
                "object_id": row["object_id"],
                "is_fruit": row["is_fruit"],
                "relation": row["relation"],
                "target": row["target"],
                "foil": row["foil"],
                "target_token_ids": row["candidate_token_ids_by_model"][model][
                    row["target"]
                ],
                "foil_token_ids": row["candidate_token_ids_by_model"][model][
                    row["foil"]
                ],
                "positions": positions[batch_index],
            }
        )
    return encoded, metadata


def replace_attention_output(
    output: Any,
    primary: torch.Tensor,
    weights: torch.Tensor,
) -> Any:
    if not isinstance(output, tuple):
        return primary
    return (primary, weights, *output[2:])


def attention_mask_row(
    attention_mask: torch.Tensor | None,
    batch_index: int,
    receiver: int,
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    return attention_mask[batch_index, 0, receiver, :]


def normalized_weights(
    score_row: torch.Tensor,
    mask_row: torch.Tensor | None,
) -> torch.Tensor:
    masked = score_row if mask_row is None else score_row + mask_row
    return torch.softmax(masked.float(), dim=-1).to(score_row.dtype)


def swap_group_means(
    tensor: torch.Tensor,
    first: list[int],
    second: list[int],
) -> torch.Tensor:
    output = tensor.clone()
    first_mean = output[:, first].mean(dim=-1, keepdim=True)
    second_mean = output[:, second].mean(dim=-1, keepdim=True)
    output[:, first] += second_mean - first_mean
    output[:, second] += first_mean - second_mean
    return output


def equalize_group_means(
    tensor: torch.Tensor,
    first: list[int],
    second: list[int],
) -> torch.Tensor:
    output = tensor.clone()
    first_mean = output[:, first].mean(dim=-1, keepdim=True)
    second_mean = output[:, second].mean(dim=-1, keepdim=True)
    pooled = (first_mean + second_mean) / 2.0
    output[:, first] += pooled - first_mean
    output[:, second] += pooled - second_mean
    return output


def swap_group_weight_mass(
    tensor: torch.Tensor,
    first: list[int],
    second: list[int],
) -> torch.Tensor:
    output = tensor.clone()
    first_values = output[:, first].clone()
    second_values = output[:, second].clone()
    first_mass = first_values.sum(dim=-1, keepdim=True)
    second_mass = second_values.sum(dim=-1, keepdim=True)
    output[:, first] = first_values * (
        second_mass / first_mass.clamp_min(1e-12)
    )
    output[:, second] = second_values * (
        first_mass / second_mass.clamp_min(1e-12)
    )
    return output / output.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def swap_group_value_means(
    tensor: torch.Tensor,
    first: list[int],
    second: list[int],
) -> torch.Tensor:
    output = tensor.clone()
    first_mean = output[:, first, :].mean(dim=1, keepdim=True)
    second_mean = output[:, second, :].mean(dim=1, keepdim=True)
    output[:, first, :] += second_mean - first_mean
    output[:, second, :] += first_mean - second_mean
    return output


def route_metrics(
    module: Any,
    raw_score_row: torch.Tensor,
    weights: torch.Tensor,
    values: torch.Tensor,
    batch_index: int,
    receiver: int,
    target_positions: list[int],
    foil_positions: list[int],
) -> dict[str, float]:
    target_score = raw_score_row[:, target_positions].float().mean()
    foil_score = raw_score_row[:, foil_positions].float().mean()
    target_weight = weights[
        batch_index, :, receiver, target_positions
    ].float().sum(dim=-1).mean()
    foil_weight = weights[
        batch_index, :, receiver, foil_positions
    ].float().sum(dim=-1).mean()
    target_message = edge_contribution(
        module, weights, values, batch_index, receiver, target_positions
    )
    foil_message = edge_contribution(
        module, weights, values, batch_index, receiver, foil_positions
    )
    return {
        "option_score_margin": finite(float((target_score - foil_score).item())),
        "option_weight_margin": finite(float((target_weight - foil_weight).item())),
        "target_weight_mass": finite(float(target_weight.item())),
        "foil_weight_mass": finite(float(foil_weight.item())),
        "option_message_norm_margin": finite(
            float((target_message.float().norm() - foil_message.float().norm()).item())
        ),
    }


def forward_condition(
    loaded: Any,
    layers: list[Any],
    encoded_cpu: dict[str, torch.Tensor],
    metadata: list[dict[str, Any]],
    selected_layers: dict[str, int],
    condition: str,
) -> tuple[list[dict[str, Any]], float]:
    encoded = {key: value.to(loaded.input_device) for key, value in encoded_cpu.items()}
    captured: dict[int, dict[str, float]] = {}
    reconstruction_max = 0.0

    def hook_for(layer_index: int):
        def hook(
            module: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: Any,
        ) -> Any:
            nonlocal reconstruction_max
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            position_embeddings = kwargs.get("position_embeddings")
            attention_mask = kwargs.get("attention_mask")
            if (
                hidden is None
                or position_embeddings is None
                or not isinstance(output, tuple)
                or output[1] is None
            ):
                raise RuntimeError("Phase579 requires eager attention weights")
            query, key, value = projected_states(module, hidden, position_embeddings)
            raw_scores = torch.matmul(query, key.transpose(2, 3)) * module.scaling
            masked = raw_scores if attention_mask is None else raw_scores + attention_mask
            reconstructed = torch.softmax(masked.float(), dim=-1).to(query.dtype)
            reconstruction_max = max(
                reconstruction_max,
                finite(float((reconstructed - output[1]).float().abs().max().item())),
            )
            primary = output[0].clone()
            weights = output[1].clone()
            values = value.clone() if condition == "option_value_swap_positive_control" else value
            modified = condition in PATCH_CONDITIONS

            for item in metadata:
                if selected_layers[item["relation"]] != layer_index:
                    continue
                batch_index = int(item["batch_index"])
                positions = item["positions"]
                receiver = int(positions["answer_boundary"][-1])
                target_positions = positions["target_option"]
                foil_positions = positions["foil_option"]
                score_row = raw_scores[batch_index, :, receiver, :].clone()
                weight_row = weights[batch_index, :, receiver, :].clone()
                value_row = values[batch_index]

                if condition == "option_score_swap":
                    score_row = swap_group_means(
                        score_row, target_positions, foil_positions
                    )
                    weight_row = normalized_weights(
                        score_row,
                        attention_mask_row(attention_mask, batch_index, receiver),
                    )
                elif condition == "option_score_equalize":
                    score_row = equalize_group_means(
                        score_row, target_positions, foil_positions
                    )
                    weight_row = normalized_weights(
                        score_row,
                        attention_mask_row(attention_mask, batch_index, receiver),
                    )
                elif condition == "object_relation_score_swap_control":
                    score_row = swap_group_means(
                        score_row,
                        positions["object"],
                        positions["relation"],
                    )
                    weight_row = normalized_weights(
                        score_row,
                        attention_mask_row(attention_mask, batch_index, receiver),
                    )
                elif condition == "option_weight_swap":
                    weight_row = swap_group_weight_mass(
                        weight_row, target_positions, foil_positions
                    )
                elif condition == "option_value_swap_positive_control":
                    value_row = swap_group_value_means(
                        value_row, target_positions, foil_positions
                    )
                    values[batch_index] = value_row
                elif condition not in {"natural_baseline", *RESTORE_CONDITIONS}:
                    raise RuntimeError(f"Unknown Phase579 condition: {condition}")

                if modified:
                    head_output = torch.einsum("hs,hsd->hd", weight_row, value_row)
                    projected = module.o_proj(head_output.reshape(1, -1)).squeeze(0)
                    primary[batch_index, receiver, :] = projected
                    weights[batch_index, :, receiver, :] = weight_row
                captured[batch_index] = route_metrics(
                    module,
                    score_row,
                    weights,
                    values,
                    batch_index,
                    receiver,
                    target_positions,
                    foil_positions,
                )
            if modified:
                return replace_attention_output(output, primary, weights)
            return None

        return hook

    handles = [
        layers[layer_index].self_attn.register_forward_hook(
            hook_for(layer_index), with_kwargs=True
        )
        for layer_index in sorted(set(selected_layers.values()))
    ]
    try:
        with torch.inference_mode():
            result = loaded.model(
                **encoded,
                use_cache=False,
                output_attentions=True,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    if len(captured) != len(metadata):
        raise RuntimeError(
            f"Phase579 incomplete capture for {condition}: {len(captured)}/{len(metadata)}"
        )

    outcomes = []
    for item in metadata:
        batch_index = int(item["batch_index"])
        receiver = int(item["positions"]["answer_boundary"][-1])
        logits = result.logits[batch_index, receiver].float()
        target_score = finite(float(logits[int(item["target_token_ids"][0])].item()))
        foil_score = finite(float(logits[int(item["foil_token_ids"][0])].item()))
        outcomes.append(
            {
                **item,
                **captured[batch_index],
                "target_candidate_score": target_score,
                "foil_candidate_score": foil_score,
                "candidate_margin": target_score - foil_score,
                "candidate_winner": (
                    item["target"] if target_score > foil_score else item["foil"]
                ),
                "target_candidate_single_token": len(item["target_token_ids"]) == 1,
                "foil_candidate_single_token": len(item["foil_token_ids"]) == 1,
            }
        )
    del result, encoded
    return outcomes, reconstruction_max


def causal_rows(
    model: str,
    split: str,
    condition: str,
    baseline: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    selected_layers: dict[str, int],
) -> list[dict[str, Any]]:
    rows = []
    for base, outcome in zip(baseline, outcomes, strict=True):
        if base["case_id"] != outcome["case_id"]:
            raise RuntimeError("Phase579 baseline/intervention alignment drift")
        rows.append(
            {
                "schema_version": "phase579_option_routing_causal_row.v1",
                "phase_id": causal_protocol.PHASE,
                "created_at": now(),
                "model": model,
                "split": split,
                "world_id": outcome["world_id"],
                "case_id": outcome["case_id"],
                "variant": outcome["variant"],
                "object_id": outcome["object_id"],
                "is_fruit": outcome["is_fruit"],
                "relation": outcome["relation"],
                "target": outcome["target"],
                "foil": outcome["foil"],
                "selected_layer": selected_layers[outcome["relation"]],
                "condition": condition,
                "baseline_candidate_margin": base["candidate_margin"],
                "intervention_candidate_margin": outcome["candidate_margin"],
                "candidate_margin_effect": (
                    outcome["candidate_margin"] - base["candidate_margin"]
                ),
                "baseline_candidate_winner": base["candidate_winner"],
                "intervention_candidate_winner": outcome["candidate_winner"],
                "intervention_foil_wins": outcome["candidate_winner"] == outcome["foil"],
                "maximum_candidate_score_delta": max(
                    abs(outcome["target_candidate_score"] - base["target_candidate_score"]),
                    abs(outcome["foil_candidate_score"] - base["foil_candidate_score"]),
                ),
                "baseline_option_score_margin": base["option_score_margin"],
                "intervention_option_score_margin": outcome["option_score_margin"],
                "option_score_margin_effect": (
                    outcome["option_score_margin"] - base["option_score_margin"]
                ),
                "baseline_option_weight_margin": base["option_weight_margin"],
                "intervention_option_weight_margin": outcome["option_weight_margin"],
                "option_weight_margin_effect": (
                    outcome["option_weight_margin"] - base["option_weight_margin"]
                ),
                "baseline_option_message_norm_margin": base[
                    "option_message_norm_margin"
                ],
                "intervention_option_message_norm_margin": outcome[
                    "option_message_norm_margin"
                ],
                "target_candidate_single_token": outcome[
                    "target_candidate_single_token"
                ],
                "foil_candidate_single_token": outcome["foil_candidate_single_token"],
                "recipient_values_preserved": condition
                != "option_value_swap_positive_control",
                "direct_restore": condition in RESTORE_CONDITIONS,
                "all_heads_patched": condition in PATCH_CONDITIONS,
                "head_channel_parameter_neuron_scan_executed": False,
                "sealed": False,
            }
        )
    return rows


def run(model: str, split: str, restart: bool) -> Path:
    frozen = read_json(causal_protocol.PROTOCOL_PATH)
    if frozen["source_cases_sha256"] != sha256_file(source.SOURCE_CASES_PATH):
        raise RuntimeError("Phase579 source case hash drift")
    if split not in source.OPEN_SPLITS:
        raise RuntimeError(f"Phase579 invalid split: {split}")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase579 causal test requires CUDA")
    output = paths(model, split)
    if restart:
        for path in output.values():
            path.unlink(missing_ok=True)
    worlds = load_worlds(model, split)
    selected_layers = {
        relation: int(item["layer"])
        for relation, item in frozen[
            "selected_coordinates_by_model_and_relation"
        ][model].items()
    }
    write_json(
        output["contract"],
        {
            "schema_version": "phase579_option_routing_causal_contract.v1",
            "phase_id": causal_protocol.PHASE,
            "created_at": now(),
            "model": model,
            "split": split,
            "world_count": len(worlds),
            "world_ids": [world_id for world_id, _ in worlds],
            "selected_layers_by_relation": selected_layers,
            "conditions": frozen["conditions"],
            "protocol_sha256": sha256_file(causal_protocol.PROTOCOL_PATH),
            "causal_discovery_internal_state_read": split == "causal_discovery",
            "causal_confirmation_internal_state_read": split == "causal_confirmation",
            "sealed_split_read": False,
        },
    )

    loaded = None
    rows_out: list[dict[str, Any]] = []
    reconstruction_max = 0.0
    started = time.monotonic()
    try:
        loaded = load_probe_model(model)
        if loaded.input_device.type != "cuda":
            raise RuntimeError(f"Phase579 requires CUDA, got {loaded.input_device}")
        dtype = str(next(loaded.model.parameters()).dtype)
        if dtype != frozen["execution"]["torch_dtype"]:
            raise RuntimeError(f"Phase579 dtype drift: {dtype}")
        loaded.model.config._attn_implementation = "eager"
        layers = get_layers(loaded.model)
        world_batch_size = int(frozen["execution"]["world_batch_size"])
        for start in range(0, len(worlds), world_batch_size):
            world_batch = worlds[start : start + world_batch_size]
            encoded, metadata = prepare_batch(loaded, model, world_batch)
            baseline, error = forward_condition(
                loaded,
                layers,
                encoded,
                metadata,
                selected_layers,
                "natural_baseline",
            )
            reconstruction_max = max(reconstruction_max, error)
            for condition in frozen["conditions"]:
                if condition == "natural_baseline":
                    outcomes = baseline
                    condition_error = error
                else:
                    outcomes, condition_error = forward_condition(
                        loaded,
                        layers,
                        encoded,
                        metadata,
                        selected_layers,
                        condition,
                    )
                reconstruction_max = max(reconstruction_max, condition_error)
                rows_out.extend(
                    causal_rows(
                        model,
                        split,
                        condition,
                        baseline,
                        outcomes,
                        selected_layers,
                    )
                )
            print(
                f"[{time.strftime('%H:%M:%S')}] {model} Phase579 {split} "
                f"{min(start + world_batch_size, len(worlds))}/{len(worlds)}",
                flush=True,
            )
            del encoded, baseline
        write_jsonl(output["rows"], rows_out)
        summary = {
            "schema_version": "phase579_option_routing_causal_summary.v1",
            "phase_id": causal_protocol.PHASE,
            "created_at": now(),
            "status": "complete",
            "model": model,
            "split": split,
            "device_type": loaded.input_device.type,
            "torch_dtype": dtype,
            "world_count": len(worlds),
            "case_count": len(worlds) * len(choice.VARIANTS),
            "condition_count": len(frozen["conditions"]),
            "row_count": len(rows_out),
            "attention_weight_reconstruction_max_abs_error": reconstruction_max,
            "attention_reconstruction_pass": reconstruction_max <= 0.01,
            "runtime_seconds": time.monotonic() - started,
            "rows_sha256": sha256_file(output["rows"]),
            "causal_discovery_internal_state_read": split == "causal_discovery",
            "causal_confirmation_internal_state_read": split == "causal_confirmation",
            "sealed_split_read": False,
            "head_channel_parameter_neuron_scan_executed": False,
        }
        write_json(output["summary"], summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return output["summary"]
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=source.MODELS)
    parser.add_argument("--split", choices=source.OPEN_SPLITS, required=True)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    run(args.model, args.split, args.restart)


if __name__ == "__main__":
    main()
