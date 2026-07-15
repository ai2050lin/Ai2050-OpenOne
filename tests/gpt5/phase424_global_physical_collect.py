#!/usr/bin/env python3
"""Collect the Phase424 formation-transport-competition physical census.

Only aggregate states required for the registered path are retained.  The
collector does not dump full neuron activations.  Attention transport is
reconstructed from the model's actual probabilities, value states and output
projection, then checked against the executed attention output.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import get_layers, load_probe_model, release_loaded, vram_gb  # noqa: E402
from phase358_multiresolution_component_conservation import (  # noqa: E402
    install_hooks,
    module_attr,
    relative_error,
)
from phase371b_anchor_qk_collection import capture_actual_qkv, repeat_key_value  # noqa: E402
from phase420_typed_path_trace import cosine, mlp_relation, source_writes  # noqa: E402
from phase424_global_physical_protocol import (  # noqa: E402
    MODELS,
    OUT,
    SCHEMA_VERSION,
)


PHASE_ID = "Phase424-GlobalPhysicalPathCollection"
REGISTERED = OUT / "phase424_registered_conditions.jsonl"
PAIR_FILE = OUT / "phase424_registered_pairs.jsonl"
LEDGER_THRESHOLD = 0.01
FEATURES = (
    "formation_specificity",
    "transport_contrast_specificity",
    "source_mass_specificity",
    "source_target_specificity",
    "query_target_alignment",
    "cancellation_index",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def clean(value: float) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"Phase424 non-finite scalar: {value}")
    return round(float(value), 10)


def hash_rows(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def depth_bin(layer: int, layer_count: int) -> str:
    relative = layer / max(1, layer_count - 1)
    if relative < 1 / 3:
        return "early"
    if relative < 2 / 3:
        return "middle"
    return "late"


def tensor_cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left = left.float()
    right = right.float()
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    if float(left_norm.item()) <= 1e-12 or float(right_norm.item()) <= 1e-12:
        return None
    return clean(float(torch.dot(left, right).item() / (left_norm.item() * right_norm.item())))


def mean_vector(tensor: torch.Tensor, positions: list[int]) -> torch.Tensor:
    index = torch.tensor(positions, dtype=torch.long, device=tensor.device)
    return tensor[0].index_select(0, index).float().mean(dim=0)


def normalized_delta(left: torch.Tensor, right: torch.Tensor) -> float:
    numerator = float(torch.linalg.vector_norm(left.float() - right.float()).item())
    denominator = 0.5 * (
        float(torch.linalg.vector_norm(left.float()).item())
        + float(torch.linalg.vector_norm(right.float()).item())
    )
    return clean(numerator / max(denominator, 1e-8))


def executed_ids(loaded: Any, row: dict[str, Any]) -> list[int]:
    prompt_ids = [
        int(value)
        for value in loaded.tokenizer(row["prompt"], add_special_tokens=True)["input_ids"]
    ]
    if len(prompt_ids) != int(row["base_prompt_token_count"]):
        raise RuntimeError(f"Prompt token contract changed: {row['condition_id']}")
    output = [*prompt_ids, *[int(value) for value in row["common_branch_prefix_token_ids"]]]
    digest = hashlib.sha256(
        json.dumps(output, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    if digest != row["executed_token_ids_sha256"]:
        raise RuntimeError(f"Executed token hash changed: {row['condition_id']}")
    return output


def output_direction(loaded: Any, target_id: int, opposite_id: int) -> torch.Tensor:
    embedding = loaded.model.get_output_embeddings()
    if embedding is None or not hasattr(embedding, "weight"):
        raise RuntimeError(f"Cannot locate output embedding for {loaded.key}")
    weight = embedding.weight
    return weight[target_id].float() - weight[opposite_id].float()


def collect_condition(
    loaded: Any,
    layers: list[Any],
    captures: dict[tuple[str, int], torch.Tensor],
    row: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[int, dict[str, torch.Tensor]], dict[str, Any]]:
    ids = executed_ids(loaded, row)
    input_ids = torch.tensor([ids], dtype=torch.long, device=loaded.input_device)
    attention_mask = torch.ones_like(input_ids)
    qpos = len(ids) - 1
    if qpos != int(row["prediction_position"]):
        raise RuntimeError(f"Prediction position changed: {row['condition_id']}")
    target_id = int(row["target_branch_token_id"])
    opposite_id = int(row["opposite_branch_token_id"])
    direction = output_direction(loaded, target_id, opposite_id)

    def execute() -> Any:
        captures.clear()
        return loaded.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_attentions=True,
            output_hidden_states=False,
            return_dict=True,
        )

    numerical_retry_count = 0
    result = execute()
    final_logits = result.logits[0, qpos].float()
    if not bool(torch.isfinite(final_logits).all().item()):
        numerical_retry_count = 1
        del result, final_logits
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        result = execute()
        final_logits = result.logits[0, qpos].float()
    if not bool(torch.isfinite(final_logits).all().item()):
        raise RuntimeError(
            f"Persistent non-finite logits after one retry: {row['condition_id']}"
        )
    final_margin = clean(float((final_logits[target_id] - final_logits[opposite_id]).item()))

    scalar_rows: list[dict[str, Any]] = []
    tensors: dict[int, dict[str, torch.Tensor]] = {}
    max_ledger_error = 0.0
    for layer_index, layer in enumerate(layers):
        probabilities = captures.get(("attention_probabilities", layer_index))
        if probabilities is None:
            raise RuntimeError(f"Missing actual attention probabilities at layer {layer_index}")
        value = captures[("value", layer_index)]
        head_count = int(probabilities.shape[1])
        repeated_value = repeat_key_value(value, head_count)
        head_dim = int(repeated_value.shape[-1])
        o_proj = module_attr(layer.self_attn, ("o_proj", "dense"))
        output_blocks = o_proj.weight.float().view(o_proj.weight.shape[0], head_count, head_dim)

        role_payload: dict[str, dict[str, Any]] = {}
        for role, positions in (
            ("source", row["source_positions"]),
            ("control", row["instruction_control_positions"]),
            ("query", row["query_positions"]),
        ):
            mass, head_norms, vector = source_writes(
                probabilities,
                repeated_value,
                output_blocks,
                [int(value) for value in positions],
                qpos,
            )
            token_count = max(1, len(positions))
            vector = vector.float() / token_count
            role_payload[role] = {
                "mass_per_token": clean(float(mass.mean().item()) / token_count),
                "write": vector,
                "write_norm": clean(float(torch.linalg.vector_norm(vector).item())),
                "target_alignment": tensor_cosine(vector, direction.detach().cpu()),
                "top_head_index": int(torch.argmax(head_norms).item()),
                "top_head_write_norm_per_token": clean(
                    float(torch.max(head_norms).item()) / token_count
                ),
            }

        layer_input = captures[("layer_input", layer_index)].float()
        attention_output = captures[("attention_output", layer_index)].float()
        mlp_output = captures[("mlp_output", layer_index)].float()
        layer_output = captures[("layer_output", layer_index)].float()
        source_state = mean_vector(layer_input, row["source_positions"])
        control_state = mean_vector(layer_input, row["instruction_control_positions"])
        query_state = mean_vector(layer_input, row["query_positions"])
        attention_q = attention_output[0, qpos]
        mlp_q = mlp_output[0, qpos]
        output_q = layer_output[0, qpos]

        all_positions = list(range(probabilities.shape[-1]))
        _, _, replay_vector = source_writes(
            probabilities,
            repeated_value,
            output_blocks,
            all_positions,
            qpos,
        )
        if o_proj.bias is not None:
            replay_vector = replay_vector + o_proj.bias.detach().float().cpu()
        _, attention_error = relative_error(
            attention_q.detach().cpu(), replay_vector
        )
        _, block_error = relative_error(
            output_q.detach().cpu(),
            (
                layer_input[0, qpos]
                + attention_q
                + mlp_q
            ).detach().cpu(),
        )
        probability_error = float(
            (probabilities[0, :, qpos].float().sum(dim=-1) - 1.0).abs().max().item()
        )
        max_ledger_error = max(
            max_ledger_error,
            probability_error,
            float(attention_error),
            float(block_error),
        )
        rewrite = mlp_relation(
            attention_q.detach().float().cpu(), mlp_q.detach().float().cpu()
        )
        direction_cpu = direction.detach().cpu()
        scalar_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE_ID,
                "created_at": now(),
                "model": loaded.key,
                "condition_id": row["condition_id"],
                "pair_id": row["pair_id"],
                "pair_identity": row["pair_identity"],
                "family_id": row["family_id"],
                "mechanism_id": row["mechanism_id"],
                "split": row["split"],
                "layer": layer_index,
                "relative_depth": clean(layer_index / max(1, len(layers) - 1)),
                "depth_bin": depth_bin(layer_index, len(layers)),
                "source_state_norm": clean(float(torch.linalg.vector_norm(source_state).item())),
                "control_state_norm": clean(float(torch.linalg.vector_norm(control_state).item())),
                "query_state_norm": clean(float(torch.linalg.vector_norm(query_state).item())),
                "source_state_target_alignment": tensor_cosine(source_state, direction),
                "control_state_target_alignment": tensor_cosine(control_state, direction),
                "query_state_target_alignment": tensor_cosine(query_state, direction),
                "source_attention_mass_per_token": role_payload["source"]["mass_per_token"],
                "control_attention_mass_per_token": role_payload["control"]["mass_per_token"],
                "query_attention_mass_per_token": role_payload["query"]["mass_per_token"],
                "source_write_norm_per_token": role_payload["source"]["write_norm"],
                "control_write_norm_per_token": role_payload["control"]["write_norm"],
                "query_write_norm_per_token": role_payload["query"]["write_norm"],
                "source_write_target_alignment": role_payload["source"]["target_alignment"],
                "control_write_target_alignment": role_payload["control"]["target_alignment"],
                "query_write_target_alignment": role_payload["query"]["target_alignment"],
                "source_top_head_index": role_payload["source"]["top_head_index"],
                "source_top_head_write_norm_per_token": role_payload["source"][
                    "top_head_write_norm_per_token"
                ],
                "attention_output_target_alignment": tensor_cosine(attention_q, direction),
                "mlp_output_target_alignment": tensor_cosine(mlp_q, direction),
                "layer_output_target_alignment": tensor_cosine(output_q, direction),
                "attention_probability_sum_max_error": clean(probability_error),
                "attention_replay_relative_error": clean(float(attention_error)),
                "block_replay_relative_error": clean(float(block_error)),
                **rewrite,
                "final_target_branch_margin": final_margin,
                "numerical_retry_count": numerical_retry_count,
                "physical": True,
                "observer_overlay": True,
                "predictive": False,
                "causal": False,
            }
        )
        tensors[layer_index] = {
            "source_state": source_state.detach().cpu(),
            "control_state": control_state.detach().cpu(),
            "query_state": query_state.detach().cpu(),
            "source_write": role_payload["source"]["write"],
            "control_write": role_payload["control"]["write"],
            "query_write": role_payload["query"]["write"],
            "direction": direction_cpu,
        }
        del output_blocks, repeated_value

    condition_summary = {
        "condition_id": row["condition_id"],
        "pair_id": row["pair_id"],
        "pair_identity": row["pair_identity"],
        "final_target_branch_margin": final_margin,
        "branch_correct": final_margin > 0.0,
        "executed_token_count": len(ids),
        "source_token_count": int(row["source_token_count"]),
        "query_token_count": int(row["query_token_count"]),
        "control_token_count": int(row["instruction_control_token_count"]),
        "target_word_count": int(row["target_word_count"]),
        "target_absent_from_prompt": bool(row["target_absent_from_prompt"]),
        "max_component_ledger_relative_error": clean(max_ledger_error),
        "numerical_retry_count": numerical_retry_count,
    }
    del result, input_ids, attention_mask, final_logits, direction
    return scalar_rows, tensors, condition_summary


def combine_pair(
    model: str,
    pair: dict[str, Any],
    rows_a: list[dict[str, Any]],
    tensors_a: dict[int, dict[str, torch.Tensor]],
    summary_a: dict[str, Any],
    rows_b: list[dict[str, Any]],
    tensors_b: dict[int, dict[str, torch.Tensor]],
    summary_b: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_layer_a = {int(row["layer"]): row for row in rows_a}
    by_layer_b = {int(row["layer"]): row for row in rows_b}
    layer_count = len(by_layer_a)
    layer_rows: list[dict[str, Any]] = []
    for layer in range(layer_count):
        a = by_layer_a[layer]
        b = by_layer_b[layer]
        ta = tensors_a[layer]
        tb = tensors_b[layer]
        source_contrast = normalized_delta(ta["source_state"], tb["source_state"])
        control_contrast = normalized_delta(ta["control_state"], tb["control_state"])
        query_contrast = normalized_delta(ta["query_state"], tb["query_state"])
        source_write_contrast = normalized_delta(ta["source_write"], tb["source_write"])
        control_write_contrast = normalized_delta(ta["control_write"], tb["control_write"])
        source_mass_specificity = statistics.fmean(
            [
                float(a["source_attention_mass_per_token"])
                - float(a["control_attention_mass_per_token"]),
                float(b["source_attention_mass_per_token"])
                - float(b["control_attention_mass_per_token"]),
            ]
        )
        source_target_specificity = statistics.fmean(
            [
                float(a["source_write_target_alignment"] or 0.0)
                - float(a["control_write_target_alignment"] or 0.0),
                float(b["source_write_target_alignment"] or 0.0)
                - float(b["control_write_target_alignment"] or 0.0),
            ]
        )
        query_target_alignment = statistics.fmean(
            [
                float(a["query_state_target_alignment"] or 0.0),
                float(b["query_state_target_alignment"] or 0.0),
            ]
        )
        layer_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase424-PairedPhysicalPath",
                "created_at": now(),
                "model": model,
                "pair_id": pair["pair_id"],
                "pair_index": pair["pair_index"],
                "family_id": pair["family_id"],
                "mechanism_id": pair["mechanism_id"],
                "split": pair["split"],
                "layer": layer,
                "relative_depth": a["relative_depth"],
                "depth_bin": a["depth_bin"],
                "source_state_contrast": source_contrast,
                "control_state_contrast": control_contrast,
                "query_state_contrast": query_contrast,
                "formation_specificity": clean(source_contrast - control_contrast),
                "source_write_contrast": source_write_contrast,
                "control_write_contrast": control_write_contrast,
                "transport_contrast_specificity": clean(
                    source_write_contrast - control_write_contrast
                ),
                "source_mass_specificity": clean(source_mass_specificity),
                "source_target_specificity": clean(source_target_specificity),
                "query_target_alignment": clean(query_target_alignment),
                "attention_target_alignment": clean(
                    statistics.fmean(
                        [
                            float(a["attention_output_target_alignment"] or 0.0),
                            float(b["attention_output_target_alignment"] or 0.0),
                        ]
                    )
                ),
                "mlp_target_alignment": clean(
                    statistics.fmean(
                        [
                            float(a["mlp_output_target_alignment"] or 0.0),
                            float(b["mlp_output_target_alignment"] or 0.0),
                        ]
                    )
                ),
                "layer_output_target_alignment": clean(
                    statistics.fmean(
                        [
                            float(a["layer_output_target_alignment"] or 0.0),
                            float(b["layer_output_target_alignment"] or 0.0),
                        ]
                    )
                ),
                "cancellation_index": clean(
                    statistics.fmean(
                        [float(a["cancellation_index"]), float(b["cancellation_index"])]
                    )
                ),
                "rewrite_novelty": clean(
                    statistics.fmean(
                        [float(a["rewrite_novelty"]), float(b["rewrite_novelty"])]
                    )
                ),
                "behavior_margin_mean": clean(
                    statistics.fmean(
                        [
                            float(summary_a["final_target_branch_margin"]),
                            float(summary_b["final_target_branch_margin"]),
                        ]
                    )
                ),
                "both_branches_correct": bool(
                    summary_a["branch_correct"] and summary_b["branch_correct"]
                ),
                "physical": True,
                "observer_overlay": True,
                "predictive": False,
                "causal": False,
            }
        )

    depth_features: dict[str, float] = {}
    for depth in ("early", "middle", "late"):
        depth_values = [row for row in layer_rows if row["depth_bin"] == depth]
        for feature in FEATURES:
            depth_features[f"{depth}_{feature}_median"] = clean(
                statistics.median(float(row[feature]) for row in depth_values)
            )
            depth_features[f"{depth}_{feature}_max"] = clean(
                max(float(row[feature]) for row in depth_values)
            )

    def peak(feature: str) -> dict[str, Any]:
        row = max(layer_rows, key=lambda value: float(value[feature]))
        return {
            "layer": int(row["layer"]),
            "relative_depth": float(row["relative_depth"]),
            "depth_bin": row["depth_bin"],
            "value": float(row[feature]),
        }

    formation_peak = peak("formation_specificity")
    transport_peak = peak("transport_contrast_specificity")
    competition_peak = peak("source_target_specificity")
    pair_summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": "Phase424-PairSummary",
        "created_at": now(),
        "model": model,
        "pair_id": pair["pair_id"],
        "pair_index": pair["pair_index"],
        "family_id": pair["family_id"],
        "mechanism_id": pair["mechanism_id"],
        "split": pair["split"],
        "source_cases_previously_exposed": bool(pair["source_cases_previously_exposed"]),
        "strict_double_blind_eligible": False,
        "behavior_margin_a": summary_a["final_target_branch_margin"],
        "behavior_margin_b": summary_b["final_target_branch_margin"],
        "behavior_margin_mean": clean(
            statistics.fmean(
                [
                    float(summary_a["final_target_branch_margin"]),
                    float(summary_b["final_target_branch_margin"]),
                ]
            )
        ),
        "both_branches_correct": bool(
            summary_a["branch_correct"] and summary_b["branch_correct"]
        ),
        "executed_token_count_mean": clean(
            statistics.fmean(
                [summary_a["executed_token_count"], summary_b["executed_token_count"]]
            )
        ),
        "source_token_count_mean": clean(
            statistics.fmean(
                [summary_a["source_token_count"], summary_b["source_token_count"]]
            )
        ),
        "query_token_count_mean": clean(
            statistics.fmean(
                [summary_a["query_token_count"], summary_b["query_token_count"]]
            )
        ),
        "control_token_count_mean": clean(
            statistics.fmean(
                [summary_a["control_token_count"], summary_b["control_token_count"]]
            )
        ),
        "target_word_count_mean": clean(
            statistics.fmean(
                [summary_a["target_word_count"], summary_b["target_word_count"]]
            )
        ),
        "target_leak_fraction": clean(
            statistics.fmean(
                [
                    not summary_a["target_absent_from_prompt"],
                    not summary_b["target_absent_from_prompt"],
                ]
            )
        ),
        "max_component_ledger_relative_error": max(
            summary_a["max_component_ledger_relative_error"],
            summary_b["max_component_ledger_relative_error"],
        ),
        "numerical_retry_count": int(summary_a["numerical_retry_count"])
        + int(summary_b["numerical_retry_count"]),
        "formation_peak": formation_peak,
        "transport_peak": transport_peak,
        "competition_peak": competition_peak,
        "peak_order_with_tolerance": bool(
            formation_peak["relative_depth"]
            <= transport_peak["relative_depth"] + 0.15
            and transport_peak["relative_depth"]
            <= competition_peak["relative_depth"] + 0.15
        ),
        **depth_features,
        "physical": True,
        "observer_overlay": True,
        "predictive": False,
        "causal": False,
    }
    return layer_rows, pair_summary


def run_model(model: str) -> dict[str, Any]:
    protocol = read_json(OUT / "phase424_protocol.json")
    if not protocol["validation"]["valid"]:
        raise RuntimeError("Phase424 protocol is not qualified")
    all_conditions = read_jsonl(REGISTERED)
    conditions = [row for row in all_conditions if row["model"] == model]
    if len(conditions) != 1728:
        raise RuntimeError(f"Expected 1728 conditions for {model}, got {len(conditions)}")
    condition_index = {row["condition_id"]: row for row in conditions}
    pairs = read_jsonl(PAIR_FILE)
    model_root = OUT / "models" / model
    complete_path = model_root / "phase424_collection_complete.json"
    if complete_path.exists():
        existing = read_json(complete_path)
        if existing.get("all_rows_complete"):
            print(json.dumps(existing, ensure_ascii=False, indent=2))
            return existing
    checkpoint_root = model_root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint_contract_path = checkpoint_root / "phase424_checkpoint_contract.json"
    expected_dtype = protocol["execution_dtype_by_model"][model]
    if checkpoint_contract_path.exists():
        checkpoint_contract = read_json(checkpoint_contract_path)
        if checkpoint_contract.get("execution_dtype") != expected_dtype:
            raise RuntimeError(
                f"Checkpoint dtype mismatch for {model}: "
                f"{checkpoint_contract.get('execution_dtype')} != {expected_dtype}"
            )
    completed_pair_ids: set[str] = set()
    for path in sorted(checkpoint_root.glob("phase424_pair_summaries_*.jsonl")):
        completed_pair_ids.update(row["pair_id"] for row in read_jsonl(path))
    loaded = None
    handles: list[Any] = []
    started = time.monotonic()
    shard_condition_rows: list[dict[str, Any]] = []
    shard_pair_layer_rows: list[dict[str, Any]] = []
    shard_pair_summaries: list[dict[str, Any]] = []
    shard_start: int | None = None

    def flush_shard(end_index: int) -> None:
        nonlocal shard_start
        if not shard_pair_summaries or shard_start is None:
            return
        stem = f"{shard_start:04d}_{end_index:04d}"
        write_jsonl(
            checkpoint_root / f"phase424_condition_rows_{stem}.jsonl",
            shard_condition_rows,
        )
        write_jsonl(
            checkpoint_root / f"phase424_pair_layers_{stem}.jsonl",
            shard_pair_layer_rows,
        )
        write_jsonl(
            checkpoint_root / f"phase424_pair_summaries_{stem}.jsonl",
            shard_pair_summaries,
        )
        shard_condition_rows.clear()
        shard_pair_layer_rows.clear()
        shard_pair_summaries.clear()
        shard_start = None
    try:
        print(
            f"[Phase424] loading {model}; pairs={len(pairs)} "
            f"checkpointed={len(completed_pair_ids)}",
            flush=True,
        )
        loaded = load_probe_model(model)
        actual_dtype = str(next(loaded.model.parameters()).dtype).removeprefix("torch.")
        if actual_dtype != expected_dtype:
            raise RuntimeError(
                f"Execution dtype mismatch for {model}: {actual_dtype} != {expected_dtype}"
            )
        write_json(
            checkpoint_contract_path,
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE_ID,
                "model": model,
                "execution_dtype": actual_dtype,
                "protocol_pair_rows_sha256": protocol["pair_rows_sha256"],
                "protocol_condition_rows_sha256": protocol["condition_rows_sha256"],
            },
        )
        layers = get_layers(loaded.model)
        captures: dict[tuple[str, int], torch.Tensor] = {}
        handles = install_hooks(layers, captures)
        with capture_actual_qkv(loaded.key, tuple(range(len(layers))), captures):
            for pair_number, pair in enumerate(pairs, start=1):
                if pair["pair_id"] in completed_pair_ids:
                    continue
                if shard_start is None:
                    shard_start = pair_number - 1
                row_a = condition_index[f"{pair['pair_id']}__a__{model}"]
                row_b = condition_index[f"{pair['pair_id']}__b__{model}"]
                scalar_a, tensors_a, summary_a = collect_condition(
                    loaded, layers, captures, row_a
                )
                captures.clear()
                scalar_b, tensors_b, summary_b = collect_condition(
                    loaded, layers, captures, row_b
                )
                shard_condition_rows.extend(scalar_a)
                shard_condition_rows.extend(scalar_b)
                combined, summary = combine_pair(
                    model,
                    pair,
                    scalar_a,
                    tensors_a,
                    summary_a,
                    scalar_b,
                    tensors_b,
                    summary_b,
                )
                shard_pair_layer_rows.extend(combined)
                shard_pair_summaries.append(summary)
                captures.clear()
                del scalar_a, tensors_a, scalar_b, tensors_b, combined
                if len(shard_pair_summaries) == 24 or pair_number == len(pairs):
                    flush_shard(pair_number - 1)
                    print(
                        f"[Phase424:{model}] pairs={pair_number}/{len(pairs)} "
                        f"checkpointed={len(completed_pair_ids) + pair_number}",
                        flush=True,
                    )
        condition_rows: list[dict[str, Any]] = []
        pair_layer_rows: list[dict[str, Any]] = []
        pair_summaries: list[dict[str, Any]] = []
        for path in sorted(checkpoint_root.glob("phase424_condition_rows_*.jsonl")):
            condition_rows.extend(read_jsonl(path))
        for path in sorted(checkpoint_root.glob("phase424_pair_layers_*.jsonl")):
            pair_layer_rows.extend(read_jsonl(path))
        for path in sorted(checkpoint_root.glob("phase424_pair_summaries_*.jsonl")):
            pair_summaries.extend(read_jsonl(path))
        if len({row["pair_id"] for row in pair_summaries}) != len(pair_summaries):
            raise RuntimeError(f"Duplicate Phase424 checkpoint pairs for {model}")
        write_jsonl(model_root / "phase424_condition_layer_rows.jsonl", condition_rows)
        write_jsonl(model_root / "phase424_pair_layer_rows.jsonl", pair_layer_rows)
        write_jsonl(model_root / "phase424_pair_summary_rows.jsonl", pair_summaries)
        expected_condition_layers = 1728 * len(layers)
        expected_pair_layers = 864 * len(layers)
        max_error = max(
            float(row["max_component_ledger_relative_error"])
            for row in pair_summaries
        )
        complete = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE_ID,
            "created_at": now(),
            "model": model,
            "layer_count": len(layers),
            "execution_dtype": actual_dtype,
            "condition_count": len(conditions),
            "pair_count": len(pair_summaries),
            "condition_layer_row_count": len(condition_rows),
            "pair_layer_row_count": len(pair_layer_rows),
            "expected_condition_layer_row_count": expected_condition_layers,
            "expected_pair_layer_row_count": expected_pair_layers,
            "branch_correct_pair_count": sum(
                bool(row["both_branches_correct"]) for row in pair_summaries
            ),
            "numerical_retry_condition_count": sum(
                int(row["numerical_retry_count"]) for row in pair_summaries
            ),
            "max_component_ledger_relative_error": clean(max_error),
            "component_ledger_gate_pass": max_error <= LEDGER_THRESHOLD,
            "all_rows_complete": bool(
                len(condition_rows) == expected_condition_layers
                and len(pair_layer_rows) == expected_pair_layers
                and len(pair_summaries) == 864
                and max_error <= LEDGER_THRESHOLD
            ),
            "condition_rows_sha256": hash_rows(condition_rows),
            "pair_layer_rows_sha256": hash_rows(pair_layer_rows),
            "pair_summary_rows_sha256": hash_rows(pair_summaries),
            "elapsed_seconds": clean(time.monotonic() - started),
            "vram_gb": list(vram_gb()),
            "physical": True,
            "observer_overlay": True,
            "predictive": False,
            "causal": False,
            "strict_double_blind_holdout_collected": False,
        }
        write_json(model_root / "phase424_collection_complete.json", complete)
        print(json.dumps(complete, ensure_ascii=False, indent=2))
        return complete
    finally:
        for handle in handles:
            handle.remove()
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS)
    args = parser.parse_args()
    run_model(args.model)


if __name__ == "__main__":
    main()
