#!/usr/bin/env python3
"""Capture all-layer natural choice trajectories for Phase578."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import importlib
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
from phase569_role_position_utils import token_indices_for_span  # noqa: E402
from phase573_coarse_message_causal import edge_contribution  # noqa: E402
import phase577_natural_choice_protocol as source  # noqa: E402
import phase578_choice_world_protocol as protocol  # noqa: E402
import phase578_natural_trace_protocol as trace_protocol  # noqa: E402


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def paths(model: str) -> dict[str, Path]:
    stem = protocol.OUT_DIR / f"phase578_{model}_natural_trace"
    return {
        "rows": stem.with_name(stem.name + "_rows.jsonl.gz"),
        "snapshots": stem.with_name(stem.name + "_snapshots.pt"),
        "summary": stem.with_name(stem.name + "_summary.json"),
        "contract": stem.with_name(stem.name + "_contract.json"),
    }


def load_worlds(model: str) -> list[tuple[str, str, list[dict[str, Any]]]]:
    frozen = read_json(trace_protocol.TRACE_PROTOCOL_PATH)
    if model not in frozen["authorized_models"]:
        raise RuntimeError(f"Phase578 natural trace is not authorized for {model}")
    selected = frozen["natural_trace_world_ids_by_model_and_split"][model]
    selected_ids = set().union(*(set(values) for values in selected.values()))
    bank: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in iter_jsonl(protocol.SOURCE_CASES_PATH):
        if row["world_id"] not in selected_ids:
            continue
        if row["split"] not in protocol.OPEN_SPLITS or row["sealed"]:
            raise RuntimeError("Phase578 natural trace attempted forbidden rows")
        bank[row["world_id"]][row["variant"]] = row
    worlds = []
    for split in protocol.OPEN_SPLITS:
        for world_id in selected[split]:
            variants = bank.get(world_id, {})
            if set(variants) != set(source.VARIANTS):
                raise RuntimeError(f"Phase578 incomplete world: {world_id}")
            worlds.append((
                split,
                world_id,
                [variants[variant] for variant in source.VARIANTS],
            ))
    expected = len(protocol.OPEN_SPLITS) * protocol.NATURAL_TRACE_WORLDS_PER_SPLIT
    if len(worlds) != expected:
        raise RuntimeError(f"Phase578 world count drift: {len(worlds)}/{expected}")
    return worlds


def span_positions(tokenizer: Any, prompt: str, row: dict[str, Any]) -> dict[str, list[int]]:
    encoded = tokenizer(prompt, add_special_tokens=True, return_offsets_mapping=True)
    ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    raw_start = prompt.index(row["raw_prompt"])
    raw = row["raw_prompt"]

    def positions(fragment: str, *, last: bool = False) -> list[int]:
        local = raw.rfind(fragment) if last else raw.find(fragment)
        if local < 0:
            raise RuntimeError(f"Phase578 missing fragment {fragment!r}")
        start = raw_start + local
        return token_indices_for_span(offsets, start, start + len(fragment))

    groups = {
        "object": positions(row["object_label"]),
        "relation": positions(row["question_phrase"]),
        "target_option": positions(row["target"], last=True),
        "foil_option": positions(row["foil"], last=True),
        "answer_boundary": [len(ids) - 1],
    }
    if any(not value for value in groups.values()):
        raise RuntimeError("Phase578 empty role positions")
    return {"ids": ids, **groups}


def apply_model_rotary(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    model_module = importlib.import_module(module.__class__.__module__)
    apply_rotary = getattr(model_module, "apply_rotary_pos_emb")
    cos, sin = position_embeddings
    return apply_rotary(query, key, cos, sin)


def projected_states(
    module: Any,
    hidden: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_shape = hidden.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    query = module.q_proj(hidden).view(hidden_shape)
    key = module.k_proj(hidden).view(hidden_shape)
    if hasattr(module, "q_norm"):
        query = module.q_norm(query)
    if hasattr(module, "k_norm"):
        key = module.k_norm(key)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = module.v_proj(hidden).view(hidden_shape).transpose(1, 2)
    query, key = apply_model_rotary(module, query, key, position_embeddings)
    key = key.repeat_interleave(module.num_key_value_groups, dim=1)
    value = value.repeat_interleave(module.num_key_value_groups, dim=1)
    return query, key, value


def source_metrics(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    raw_scores: torch.Tensor,
    weights: torch.Tensor,
    batch_index: int,
    receiver: int,
    source_positions: list[int],
    keep_vectors: bool,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    keys = key[batch_index, :, source_positions, :].float().mean(dim=1)
    values = value[batch_index, :, source_positions, :].float().mean(dim=1)
    scores = raw_scores[batch_index, :, receiver, source_positions].float()
    masses = weights[batch_index, :, receiver, source_positions].float()
    message = edge_contribution(
        module, weights, value, batch_index, receiver, source_positions
    ).detach()
    scalars = {
        "key_norm": finite(float(keys.norm().item())),
        "value_norm": finite(float(values.norm().item())),
        "score_mean": finite(float(scores.mean().item())),
        "score_max_mean": finite(float(scores.max(dim=-1).values.mean().item())),
        "weight_mass_mean": finite(float(masses.sum(dim=-1).mean().item())),
        "message_norm": finite(float(message.float().norm().item())),
    }
    vectors = {}
    if keep_vectors:
        vectors = {
            "key": keys.detach().cpu(),
            "value": values.detach().cpu(),
            "score_by_head": scores.mean(dim=-1).detach().cpu(),
            "weight_by_head": masses.sum(dim=-1).detach().cpu(),
            "message": message.detach().cpu(),
        }
    return scalars, vectors


def candidate_margin(
    loaded: Any,
    state: torch.Tensor,
    target_id: int,
    foil_id: int,
) -> float:
    normalized = loaded.model.model.norm(state)
    weight = loaded.model.lm_head.weight
    target = torch.dot(normalized.float(), weight[target_id].float())
    foil = torch.dot(normalized.float(), weight[foil_id].float())
    return finite(float((target - foil).item()))


def nested_numeric_max_delta(left: Any, right: Any) -> float:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return float("inf")
        return max(
            (nested_numeric_max_delta(left[key], right[key]) for key in left if key != "created_at"),
            default=0.0,
        )
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return float("inf")
        return max(
            (nested_numeric_max_delta(a, b) for a, b in zip(left, right)),
            default=0.0,
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    return 0.0 if left == right else float("inf")


def trace_world(
    loaded: Any,
    layers: list[Any],
    split: str,
    world_id: str,
    rows: list[dict[str, Any]],
    keep_snapshot: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, float]:
    prompts = [render_chat(loaded.tokenizer, loaded.key, row["raw_prompt"]) for row in rows]
    positions = [span_positions(loaded.tokenizer, prompt, row) for prompt, row in zip(prompts, rows)]
    encoded_cpu = loaded.tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    for batch_index, item in enumerate(positions):
        active = encoded_cpu["input_ids"][batch_index][encoded_cpu["attention_mask"][batch_index].bool()].tolist()
        if [int(value) for value in active] != item["ids"]:
            raise RuntimeError("Phase578 tokenization drift")
    position_ids = encoded_cpu["attention_mask"].long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(encoded_cpu["attention_mask"] == 0, 0)
    encoded_cpu["position_ids"] = position_ids
    encoded = {key: value.to(loaded.input_device) for key, value in encoded_cpu.items()}

    attention_captures: dict[int, dict[str, Any]] = {}
    mlp_captures: dict[int, torch.Tensor] = {}
    vector_captures: dict[int, dict[str, Any]] = {}
    reconstruction_error = 0.0

    def attention_hook(layer_index: int):
        def hook(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
            nonlocal reconstruction_error
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            position_embeddings = kwargs.get("position_embeddings")
            attention_mask = kwargs.get("attention_mask")
            if hidden is None or position_embeddings is None or not isinstance(output, tuple) or output[1] is None:
                raise RuntimeError("Phase578 requires eager attention weights")
            primary, weights = output[0], output[1]
            query, key, value = projected_states(module, hidden, position_embeddings)
            raw_scores = torch.matmul(query, key.transpose(2, 3)) * module.scaling
            masked = raw_scores if attention_mask is None else raw_scores + attention_mask
            reconstructed = torch.softmax(masked.float(), dim=-1).to(query.dtype)
            reconstruction_error = max(
                reconstruction_error,
                finite(float((reconstructed - weights).float().abs().max().item())),
            )
            variants = {}
            vectors = {}
            for batch_index, row in enumerate(rows):
                receiver = positions[batch_index]["answer_boundary"][-1]
                source_out = {}
                source_vectors = {}
                for source_name in trace_protocol.SOURCE_GROUPS:
                    scalars, vector = source_metrics(
                        module,
                        query,
                        key,
                        value,
                        raw_scores,
                        weights,
                        batch_index,
                        receiver,
                        positions[batch_index][source_name],
                        keep_snapshot,
                    )
                    source_out[source_name] = scalars
                    source_vectors[source_name] = vector
                variants[row["variant"]] = {
                    "query_norm": finite(float(query[batch_index, :, receiver].float().norm().item())),
                    "attention_output_norm": finite(float(primary[batch_index, receiver].float().norm().item())),
                    "sources": source_out,
                    "option_score_margin": source_out["target_option"]["score_mean"] - source_out["foil_option"]["score_mean"],
                    "option_weight_margin": source_out["target_option"]["weight_mass_mean"] - source_out["foil_option"]["weight_mass_mean"],
                    "option_message_norm_margin": source_out["target_option"]["message_norm"] - source_out["foil_option"]["message_norm"],
                }
                if keep_snapshot:
                    vectors[row["variant"]] = {
                        "query": query[batch_index, :, receiver].detach().cpu(),
                        "attention_output": primary[batch_index, receiver].detach().cpu(),
                        "sources": source_vectors,
                    }
            attention_captures[layer_index] = variants
            vector_captures[layer_index] = vectors
        return hook

    def mlp_hook(layer_index: int):
        def hook(module: Any, args: tuple[Any, ...], output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            mlp_captures[layer_index] = tensor.detach()
        return hook

    handles = []
    for layer_index, layer in enumerate(layers):
        handles.append(layer.self_attn.register_forward_hook(attention_hook(layer_index), with_kwargs=True))
        handles.append(layer.mlp.register_forward_hook(mlp_hook(layer_index)))
    try:
        with torch.inference_mode():
            result = loaded.model(
                **encoded,
                use_cache=False,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    if set(attention_captures) != set(range(len(layers))) or set(mlp_captures) != set(range(len(layers))):
        raise RuntimeError("Phase578 incomplete layer capture")

    output_rows = []
    snapshot = {"layers": vector_captures} if keep_snapshot else None
    for layer_index in range(len(layers)):
        variants = {}
        for batch_index, row in enumerate(rows):
            receiver = positions[batch_index]["answer_boundary"][-1]
            target_ids = row["candidate_token_ids_by_model"][loaded.key][row["target"]]
            foil_ids = row["candidate_token_ids_by_model"][loaded.key][row["foil"]]
            if not target_ids or not foil_ids:
                raise RuntimeError("Phase578 missing candidate token ids")
            layer_input = result.hidden_states[layer_index][batch_index, receiver]
            layer_output = result.hidden_states[layer_index + 1][batch_index, receiver]
            mlp_output = mlp_captures[layer_index][batch_index, receiver]
            base = attention_captures[layer_index][row["variant"]]
            variants[row["variant"]] = {
                **base,
                "layer_input_norm": finite(float(layer_input.float().norm().item())),
                "mlp_output_norm": finite(float(mlp_output.float().norm().item())),
                "layer_output_norm": finite(float(layer_output.float().norm().item())),
                "candidate_input_logit_margin": candidate_margin(
                    loaded, layer_input, int(target_ids[0]), int(foil_ids[0])
                ),
                "candidate_output_logit_margin": candidate_margin(
                    loaded, layer_output, int(target_ids[0]), int(foil_ids[0])
                ),
                "target_candidate_single_token": len(target_ids) == 1,
                "foil_candidate_single_token": len(foil_ids) == 1,
            }
            if keep_snapshot:
                snapshot["layers"][layer_index][row["variant"]].update({
                    "layer_input": layer_input.detach().cpu(),
                    "mlp_output": mlp_output.detach().cpu(),
                    "layer_output": layer_output.detach().cpu(),
                })
        output_rows.append({
            "schema_version": "phase578_natural_trace_row.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "model": loaded.key,
            "split": split,
            "world_id": world_id,
            "object_id": rows[0]["object_id"],
            "is_fruit": rows[0]["is_fruit"],
            "relation": rows[0]["relation"],
            "target": rows[0]["target"],
            "foil": rows[0]["foil"],
            "surface_id": rows[0]["surface_id"],
            "surface_order": rows[0]["surface_order"],
            "layer": layer_index,
            "normalized_depth": layer_index / max(1, len(layers) - 1),
            "variants": variants,
            "natural_behavior_qualified": True,
            "causal_intervention_executed": False,
            "causal_holdout_internal_state_read": False,
            "sealed": False,
        })
    del result, encoded, encoded_cpu, attention_captures, mlp_captures, vector_captures
    return output_rows, snapshot, reconstruction_error


def run(model: str, restart: bool) -> Path:
    frozen = read_json(trace_protocol.TRACE_PROTOCOL_PATH)
    if model not in frozen["authorized_models"]:
        raise RuntimeError(f"Phase578 trace unauthorized for {model}")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase578 natural trace requires CUDA")
    output = paths(model)
    if restart:
        for path in output.values():
            path.unlink(missing_ok=True)
    worlds = load_worlds(model)
    write_json(output["contract"], {
        "schema_version": "phase578_natural_trace_contract.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "model": model,
        "trace_protocol_sha256": sha256_file(trace_protocol.TRACE_PROTOCOL_PATH),
        "source_case_sha256": sha256_file(protocol.SOURCE_CASES_PATH),
        "world_count": len(worlds),
        "world_ids": [world_id for _, world_id, _ in worlds],
        "trace_every_layer": True,
        "causal_holdout_internal_state_read": False,
        "sealed_split_read": False,
    })
    loaded = None
    rows_out = []
    snapshots = {}
    reconstruction_max = 0.0
    duplicate_max = 0.0
    duplicate_count = 0
    snapshot_count = 0
    started = time.monotonic()
    try:
        loaded = load_probe_model(model)
        if loaded.input_device.type != "cuda":
            raise RuntimeError(f"Phase578 requires CUDA, got {loaded.input_device}")
        loaded.tokenizer.padding_side = "right"
        dtype = str(next(loaded.model.parameters()).dtype)
        if dtype != "torch.bfloat16":
            raise RuntimeError(f"Phase578 requires BF16, got {dtype}")
        layers = get_layers(loaded.model)
        if len(layers) != frozen["layer_count_by_model"][model]:
            raise RuntimeError("Phase578 layer count drift")
        loaded.model.config._attn_implementation = "eager"
        for world_index, (split, world_id, rows) in enumerate(worlds):
            keep_snapshot = split == protocol.OPEN_SPLITS[0] and snapshot_count < frozen[
                "full_vector_snapshot_worlds_in_discovery_per_model"
            ]
            traced, snapshot, error = trace_world(
                loaded, layers, split, world_id, rows, keep_snapshot
            )
            rows_out.extend(traced)
            reconstruction_max = max(reconstruction_max, error)
            if snapshot is not None:
                snapshots[world_id] = snapshot
                snapshot_count += 1
            if split == protocol.OPEN_SPLITS[0] and duplicate_count < frozen[
                "duplicate_trace_audit_worlds_in_discovery_per_model"
            ]:
                duplicate_rows, _, duplicate_error = trace_world(
                    loaded, layers, split, world_id, rows, False
                )
                duplicate_max = max(
                    duplicate_max,
                    nested_numeric_max_delta(traced, duplicate_rows),
                    abs(error - duplicate_error),
                )
                duplicate_count += 1
            if (world_index + 1) % 12 == 0:
                print(
                    f"[{time.strftime('%H:%M:%S')}] {model} Phase578 natural trace "
                    f"{world_index + 1}/{len(worlds)}",
                    flush=True,
                )
        write_jsonl(output["rows"], rows_out)
        torch.save(snapshots, output["snapshots"])
        gates = {
            "attention_reconstruction_pass": reconstruction_max <= frozen[
                "attention_weight_reconstruction_max_abs_error"
            ],
            "duplicate_trace_exact_pass": duplicate_max == frozen[
                "duplicate_trace_max_abs_delta"
            ],
        }
        summary = {
            "schema_version": "phase578_natural_trace_summary.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "status": "complete",
            "model": model,
            "device_type": loaded.input_device.type,
            "torch_dtype": dtype,
            "world_count": len(worlds),
            "world_count_by_split": {
                split: sum(value == split for value, _, _ in worlds)
                for split in protocol.OPEN_SPLITS
            },
            "layer_count": len(layers),
            "trace_row_count": len(rows_out),
            "snapshot_world_count": len(snapshots),
            "duplicate_trace_world_count": duplicate_count,
            "attention_weight_reconstruction_max_abs_error": reconstruction_max,
            "duplicate_trace_max_abs_delta": duplicate_max,
            "quality_gates": gates,
            "natural_structure_analysis_authorized": all(gates.values()),
            "runtime_seconds": time.monotonic() - started,
            "rows_sha256": sha256_file(output["rows"]),
            "snapshots_sha256": sha256_file(output["snapshots"]),
            "causal_holdout_internal_state_read": False,
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
    parser.add_argument("model", choices=protocol.MODELS)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    run(args.model, args.restart)


if __name__ == "__main__":
    main()
