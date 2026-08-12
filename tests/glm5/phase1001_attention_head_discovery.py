#!/usr/bin/env python3
"""Phase 1001 attention-head discovery on frozen Phase 1000 validation pairs.

The script intervenes on the real concatenated head output immediately before
Qwen3 attention o_proj. It first verifies that restoring all heads is
equivalent to restoring the full attention output, then screens all 96 heads
from layers 25, 30, and 31 for mediation. Only validation data select the
frozen head set.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, get_model_info, load_model, release_model
from phase1000_factorial_binding_behavior import eos_ids, parse_generated, strip_at_eos
from phase1000_factorial_binding_protocol import COLORS, MODEL, PHASE
from phase1000_scpg_discovery import (
    batches_by_template,
    candidate_tensor,
    capture_residuals,
    case_tensors,
    directional_pairs,
    prediction_colors,
    read_jsonl,
    register_source_patch,
    replace_positions,
    semantic_margin,
    source_patch_spec,
    write_rows,
)
from phase1000_source_control_audit import valid_derangement_shifts


PHASE_ID = 1001
SOURCE_DEPTH = 1
TARGET_LAYERS = (25, 30, 31)
HEAD_COUNT = 32
HEAD_DIM = 128
SELECT_PER_LAYER = 4
SELECT_LIMIT = 16
JOINT_SIZES = (1, 2, 4, 8, 12, 16)
HEAD_THRESHOLDS = {
    "instrument_mean_abs_margin_error": 1e-3,
    "instrument_prediction_agreement": 0.999,
    "single_median_mediation": 0.05,
    "single_mean_sufficiency_transfer": 0.02,
    "single_wrong_o_excess": 0.01,
    "joint_median_mediation": 0.30,
    "joint_natural_restoration_rate": 0.50,
}
RESULT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1001_attention_physical_decomposition"
)
PHASE1000_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1000_factorial_binding_scpg"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def head_events() -> list[dict[str, Any]]:
    return [
        {
            "event_id": f"l{layer_number:02d}.h{head_index:02d}",
            "layer_number": layer_number,
            "block_index": layer_number - 1,
            "head_index": head_index,
            "role": "answer_boundary",
        }
        for layer_number in TARGET_LAYERS
        for head_index in range(HEAD_COUNT)
    ]


def selected_phase1000_inputs(scope: str):
    if scope == "smoke":
        protocol_root = PHASE1000_ROOT / "smoke"
        selection_root = PHASE1000_ROOT / "smoke_discovery"
        behavior_root = PHASE1000_ROOT / "smoke_behavior"
        output_root = RESULT_ROOT / "smoke_head_discovery"
        partition = "smoke"
    else:
        protocol_root = PHASE1000_ROOT / "protocol"
        selection_root = PHASE1000_ROOT / "discovery"
        behavior_root = PHASE1000_ROOT / "behavior"
        output_root = RESULT_ROOT / "head_discovery"
        partition = "validation"
    cases = read_jsonl(protocol_root / "cases.jsonl")
    selected_pairs = read_jsonl(selection_root / "selected_pairs.jsonl")
    protocol = read_json(protocol_root / "protocol.json")
    behavior = read_json(behavior_root / "summary.json")
    case_by_id = {row["record_id"]: row for row in cases}
    directional = directional_pairs(
        selected_pairs, case_by_id, partition, bidirectional=True
    )
    return protocol, behavior, selected_pairs, directional, output_root


def capture_attention_states(
    model,
    layers,
    device,
    rows: list[dict[str, Any]],
    candidate_ids: dict[str, int],
    source_patch: dict[str, Any] | None = None,
) -> tuple[
    torch.Tensor,
    dict[int, torch.Tensor],
    dict[int, torch.Tensor],
]:
    input_ids, attention = case_tensors(rows, device)
    head_inputs: dict[int, torch.Tensor] = {}
    attention_outputs: dict[int, torch.Tensor] = {}
    counts: dict[str, int] = defaultdict(int)
    handles = []
    source_handle = None
    try:
        source_handle, source_count = register_source_patch(
            layers, source_patch, full_width=None
        )
        for layer_number in TARGET_LAYERS:
            layer = layers[layer_number - 1]
            positions = torch.tensor(
                [row["role_positions"]["answer_boundary"] for row in rows],
                dtype=torch.long,
                device=device,
            )

            def make_o_pre(number, pos):
                def hook(module, args):
                    value = args[0]
                    batch_index = torch.arange(value.shape[0], device=value.device)
                    selected = value[
                        batch_index, pos.to(value.device), :
                    ].reshape(value.shape[0], HEAD_COUNT, HEAD_DIM)
                    head_inputs[number] = selected.detach()
                    counts[f"o/{number}"] += 1

                return hook

            def make_attn_hook(number, pos):
                def hook(module, args, output):
                    value = output[0] if isinstance(output, tuple) else output
                    batch_index = torch.arange(value.shape[0], device=value.device)
                    attention_outputs[number] = value[
                        batch_index, pos.to(value.device), :
                    ].detach()
                    counts[f"attn/{number}"] += 1
                    return output

                return hook

            handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(
                    make_o_pre(layer_number, positions)
                )
            )
            handles.append(
                layer.self_attn.register_forward_hook(
                    make_attn_hook(layer_number, positions)
                )
            )
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
                return_dict=True,
            )
        candidates = candidate_tensor(output.logits[:, -1, :], candidate_ids).detach()
        if source_patch is not None and source_count[0] != 1:
            raise RuntimeError(f"source hook count drift: {source_count[0]}")
        expected = {
            f"{kind}/{layer_number}"
            for layer_number in TARGET_LAYERS
            for kind in ("o", "attn")
        }
        bad = {key: counts[key] for key in expected if counts[key] != 1}
        if bad:
            raise RuntimeError(f"attention capture count drift: {bad}")
        del output, input_ids, attention
        return candidates, head_inputs, attention_outputs
    finally:
        for handle in reversed(handles):
            handle.remove()
        if source_handle is not None:
            source_handle.remove()


def register_head_patches(
    layers,
    rows: list[dict[str, Any]],
    patches: list[dict[str, Any]],
    device,
    full_width: int | None,
):
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for patch in patches:
        grouped[int(patch["event"]["layer_number"])].append(patch)
    handles = []
    counts: dict[int, list[int]] = {}
    for layer_number, layer_patches in grouped.items():
        positions = torch.tensor(
            [row["role_positions"]["answer_boundary"] for row in rows],
            dtype=torch.long,
            device=device,
        )
        counter = [0]
        counts[layer_number] = counter

        def make_hook(items, pos, count):
            def hook(module, args):
                value = args[0]
                if full_width is not None and value.shape[1] != full_width:
                    return None
                patched = value.clone()
                batch_index = torch.arange(value.shape[0], device=value.device)
                for item in items:
                    head = int(item["event"]["head_index"])
                    start = head * HEAD_DIM
                    stop = start + HEAD_DIM
                    vectors = item["vectors"].to(
                        device=value.device, dtype=value.dtype
                    )
                    patched[
                        batch_index,
                        pos.to(value.device),
                        start:stop,
                    ] = vectors
                count[0] += 1
                return (patched,) + tuple(args[1:])

            return hook

        handles.append(
            layers[layer_number - 1].self_attn.o_proj.register_forward_pre_hook(
                make_hook(layer_patches, positions, counter)
            )
        )
    return handles, counts


def register_full_attention_patches(
    layers,
    rows: list[dict[str, Any]],
    vectors_by_layer: dict[int, torch.Tensor],
    device,
    full_width: int | None,
):
    handles = []
    counts: dict[int, list[int]] = {}
    for layer_number, vectors in vectors_by_layer.items():
        positions = torch.tensor(
            [row["role_positions"]["answer_boundary"] for row in rows],
            dtype=torch.long,
            device=device,
        )
        counter = [0]
        counts[layer_number] = counter

        def make_hook(pos, replacement, count):
            def hook(module, args, output):
                value = output[0] if isinstance(output, tuple) else output
                if full_width is not None and value.shape[1] != full_width:
                    return output
                count[0] += 1
                return replace_positions(
                    output,
                    {"answer": pos},
                    {"answer": replacement},
                )

            return hook

        handles.append(
            layers[layer_number - 1].self_attn.register_forward_hook(
                make_hook(positions, vectors, counter)
            )
        )
    return handles, counts


def forward_with_patches(
    model,
    layers,
    device,
    rows: list[dict[str, Any]],
    candidate_ids: dict[str, int],
    source_patch: dict[str, Any] | None = None,
    head_patches: list[dict[str, Any]] | None = None,
    attention_patches: dict[int, torch.Tensor] | None = None,
) -> torch.Tensor:
    input_ids, attention = case_tensors(rows, device)
    source_handle = None
    head_handles = []
    attention_handles = []
    try:
        source_handle, source_count = register_source_patch(
            layers, source_patch, full_width=None
        )
        head_handles, head_counts = register_head_patches(
            layers,
            rows,
            head_patches or [],
            device,
            full_width=None,
        )
        attention_handles, attention_counts = register_full_attention_patches(
            layers,
            rows,
            attention_patches or {},
            device,
            full_width=None,
        )
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
                return_dict=True,
            )
        result = candidate_tensor(output.logits[:, -1, :], candidate_ids).detach()
        if source_patch is not None and source_count[0] != 1:
            raise RuntimeError(f"source forward count drift: {source_count[0]}")
        if any(counter[0] != 1 for counter in head_counts.values()):
            raise RuntimeError(
                f"head patch count drift: "
                f"{ {key: value[0] for key, value in head_counts.items()} }"
            )
        if any(counter[0] != 1 for counter in attention_counts.values()):
            raise RuntimeError(
                f"attention patch count drift: "
                f"{ {key: value[0] for key, value in attention_counts.items()} }"
            )
        del output, input_ids, attention
        return result
    finally:
        for handle in reversed(attention_handles):
            handle.remove()
        for handle in reversed(head_handles):
            handle.remove()
        if source_handle is not None:
            source_handle.remove()


def generate_with_patches(
    model,
    layers,
    tokenizer,
    device,
    rows: list[dict[str, Any]],
    source_patch: dict[str, Any] | None,
    head_patches: list[dict[str, Any]] | None,
    attention_patches: dict[int, torch.Tensor] | None,
    effective_eos: list[int],
    budget: int,
) -> list[dict[str, Any]]:
    input_ids, attention = case_tensors(rows, device)
    full_width = input_ids.shape[1]
    source_handle = None
    head_handles = []
    attention_handles = []
    try:
        source_handle, source_count = register_source_patch(
            layers, source_patch, full_width=full_width
        )
        head_handles, head_counts = register_head_patches(
            layers,
            rows,
            head_patches or [],
            device,
            full_width=full_width,
        )
        attention_handles, attention_counts = register_full_attention_patches(
            layers,
            rows,
            attention_patches or {},
            device,
            full_width=full_width,
        )
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                max_new_tokens=budget,
                eos_token_id=effective_eos,
                pad_token_id=int(tokenizer.pad_token_id),
                return_dict_in_generate=True,
                output_scores=False,
                output_hidden_states=False,
                output_attentions=False,
            )
        if source_patch is not None and source_count[0] != 1:
            raise RuntimeError(f"source generation count drift: {source_count[0]}")
        if any(counter[0] != 1 for counter in head_counts.values()):
            raise RuntimeError("head generation patch count drift")
        if any(counter[0] != 1 for counter in attention_counts.values()):
            raise RuntimeError("attention generation patch count drift")
        suffixes = generated.sequences[:, full_width:].detach().cpu().tolist()
        eos_set = set(effective_eos)
        results = []
        for suffix in suffixes:
            suffix = [int(value) for value in suffix]
            before, eos_position = strip_at_eos(suffix, eos_set)
            text = tokenizer.decode(
                before,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            parsed = parse_generated(text)
            results.append(
                {
                    "prediction": parsed["first_color"],
                    "text": text,
                    "suffix": suffix,
                    "eos_seen": eos_position is not None,
                    "exact_short": parsed["exact_short"],
                }
            )
        del generated, input_ids, attention
        return results
    finally:
        for handle in reversed(attention_handles):
            handle.remove()
        for handle in reversed(head_handles):
            handle.remove()
        if source_handle is not None:
            source_handle.remove()


def response_rows_for_batch(
    model,
    batch: list[dict[str, Any]],
    source_heads: dict[int, torch.Tensor],
    target_heads: dict[int, torch.Tensor],
    do_heads: dict[int, torch.Tensor],
    candidate_ids: dict[str, int],
) -> list[dict[str, Any]]:
    color_index = {color: index for index, color in enumerate(COLORS)}
    output_weight = model.get_output_embeddings().weight.detach().float()
    candidate_unembed = output_weight[
        torch.tensor(
            [candidate_ids[color] for color in COLORS],
            dtype=torch.long,
            device=output_weight.device,
        )
    ]
    source_indices = torch.tensor(
        [color_index[item["source"]["gold"]] for item in batch],
        dtype=torch.long,
        device=output_weight.device,
    )
    target_indices = torch.tensor(
        [color_index[item["target"]["gold"]] for item in batch],
        dtype=torch.long,
        device=output_weight.device,
    )
    unembed_direction = (
        candidate_unembed[source_indices] - candidate_unembed[target_indices]
    )
    rows = []
    for layer_number in TARGET_LAYERS:
        w_o = (
            model.model.layers[layer_number - 1]
            .self_attn.o_proj.weight.detach()
            .float()
        )
        for head_index in range(HEAD_COUNT):
            natural_delta = (
                source_heads[layer_number][:, head_index, :]
                - target_heads[layer_number][:, head_index, :]
            ).float()
            do_delta = (
                do_heads[layer_number][:, head_index, :]
                - target_heads[layer_number][:, head_index, :]
            ).float()
            cosine = F.cosine_similarity(
                do_delta, natural_delta, dim=-1, eps=1e-8
            )
            natural_norm = torch.linalg.vector_norm(natural_delta, dim=-1)
            do_norm = torch.linalg.vector_norm(do_delta, dim=-1)
            start = head_index * HEAD_DIM
            stop = start + HEAD_DIM
            residual_delta = do_delta @ w_o[:, start:stop].T
            direct_logit = torch.sum(
                residual_delta * unembed_direction, dim=-1
            )
            for index, item in enumerate(batch):
                rows.append(
                    {
                        "schema_version": "phase1001_head_response_row.v1",
                        "phase": PHASE_ID,
                        "model": MODEL,
                        "partition": item["partition"],
                        "pair_id": item["pair_id"],
                        "direction": item["direction"],
                        "event_id": f"l{layer_number:02d}.h{head_index:02d}",
                        "layer_number": layer_number,
                        "head_index": head_index,
                        "natural_delta_norm": float(natural_norm[index]),
                        "source_driven_delta_norm": float(do_norm[index]),
                        "source_to_natural_cosine": float(cosine[index]),
                        "magnitude_fraction": float(
                            do_norm[index]
                            / torch.clamp(natural_norm[index], min=1e-8)
                        ),
                        "direct_source_vs_target_logit_effect": float(
                            direct_logit[index]
                        ),
                    }
                )
    return rows


def mediation_row(
    item: dict[str, Any],
    event: dict[str, Any],
    source_margin: float,
    target_margin: float,
    do_margin: float,
    restored_margin: float,
    prediction: str,
) -> dict[str, Any]:
    source_effect = do_margin - target_margin
    return {
        "schema_version": "phase1001_head_mediation_row.v1",
        "phase": PHASE_ID,
        "model": MODEL,
        "partition": item["partition"],
        "pair_id": item["pair_id"],
        "direction": item["direction"],
        **event,
        "source_gold": item["source"]["gold"],
        "target_gold": item["target"]["gold"],
        "source_margin": source_margin,
        "target_margin": target_margin,
        "do_source_margin": do_margin,
        "restored_margin": restored_margin,
        "mediation_fraction": (
            (do_margin - restored_margin) / max(abs(source_effect), 1e-8)
        ),
        "restored_prediction": prediction,
        "restored_to_target": prediction == item["target"]["gold"],
    }


def summarize_response(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["event_id"]].append(row)
    result = {}
    for event_id, values in groups.items():
        result[event_id] = {
            "event_id": event_id,
            "layer_number": values[0]["layer_number"],
            "head_index": values[0]["head_index"],
            "n": len(values),
            "median_source_to_natural_cosine": float(
                np.median([row["source_to_natural_cosine"] for row in values])
            ),
            "median_magnitude_fraction": float(
                np.median([row["magnitude_fraction"] for row in values])
            ),
            "mean_direct_source_vs_target_logit_effect": float(
                np.mean(
                    [row["direct_source_vs_target_logit_effect"] for row in values]
                )
            ),
            "positive_direct_effect_rate": float(
                np.mean(
                    [
                        row["direct_source_vs_target_logit_effect"] > 0
                        for row in values
                    ]
                )
            ),
        }
    return result


def summarize_mediation(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["event_id"]].append(row)
    return {
        event_id: {
            "event_id": event_id,
            "layer_number": values[0]["layer_number"],
            "head_index": values[0]["head_index"],
            "n": len(values),
            "median_mediation_fraction": float(
                np.median([row["mediation_fraction"] for row in values])
            ),
            "mean_mediation_fraction": float(
                np.mean([row["mediation_fraction"] for row in values])
            ),
            "positive_mediation_rate": float(
                np.mean([row["mediation_fraction"] > 0 for row in values])
            ),
            "restored_to_target_rate": float(
                np.mean([row["restored_to_target"] for row in values])
            ),
        }
        for event_id, values in groups.items()
    }


def select_heads(
    response: dict[str, dict[str, Any]],
    mediation: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    selected_ids = set()
    selected = []

    def add(event_id: str):
        if event_id in selected_ids or len(selected) >= SELECT_LIMIT:
            return
        item = {
            **response[event_id],
            **mediation[event_id],
        }
        selected.append(item)
        selected_ids.add(event_id)

    for layer_number in TARGET_LAYERS:
        layer_events = [
            item
            for item in mediation.values()
            if item["layer_number"] == layer_number
        ]
        ordered = sorted(
            layer_events,
            key=lambda item: (
                -item["median_mediation_fraction"],
                -item["mean_mediation_fraction"],
                item["head_index"],
            ),
        )
        for item in ordered[:SELECT_PER_LAYER]:
            add(item["event_id"])
    global_order = sorted(
        mediation.values(),
        key=lambda item: (
            -item["median_mediation_fraction"],
            -item["mean_mediation_fraction"],
            item["layer_number"],
            item["head_index"],
        ),
    )
    for item in global_order:
        if len(selected) >= SELECT_LIMIT:
            break
        if item["median_mediation_fraction"] > 0:
            add(item["event_id"])
    for rank, item in enumerate(selected, 1):
        item["mediation_selection_rank"] = rank
        item["selection_partition"] = "validation"
        item["selection_uses_holdout"] = False
    return selected


def selected_head_controls(
    model,
    layers,
    device,
    batch: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    candidate_ids: dict[str, int],
    source_logits: torch.Tensor,
    target_logits: torch.Tensor,
    do_logits: torch.Tensor,
    source_patch: dict[str, Any],
    target_heads: dict[int, torch.Tensor],
    do_heads: dict[int, torch.Tensor],
) -> list[dict[str, Any]]:
    target_cases = [row["target"] for row in batch]
    source_margin = semantic_margin(source_logits, batch)
    target_margin = semantic_margin(target_logits, batch)
    do_margin = semantic_margin(do_logits, batch)
    pair_safe_shift = valid_derangement_shifts(batch, 1)[0]
    rows = []
    for event in selected:
        layer_number = int(event["layer_number"])
        head_index = int(event["head_index"])
        vector_target = target_heads[layer_number][:, head_index, :]
        vector_do = do_heads[layer_number][:, head_index, :]
        delta = vector_do - vector_target
        suff_logits = forward_with_patches(
            model,
            layers,
            device,
            target_cases,
            candidate_ids,
            head_patches=[{"event": event, "vectors": vector_do}],
        )
        wrong_head = (head_index + 1) % HEAD_COUNT
        wrong_event = {
            **event,
            "event_id": f"l{layer_number:02d}.h{wrong_head:02d}",
            "head_index": wrong_head,
        }
        wrong_vectors = (
            target_heads[layer_number][:, wrong_head, :] + delta
        )
        wrong_logits = forward_with_patches(
            model,
            layers,
            device,
            target_cases,
            candidate_ids,
            head_patches=[{"event": wrong_event, "vectors": wrong_vectors}],
        )
        null_vectors = torch.roll(vector_do, shifts=pair_safe_shift, dims=0)
        null_logits = forward_with_patches(
            model,
            layers,
            device,
            target_cases,
            candidate_ids,
            head_patches=[{"event": event, "vectors": null_vectors}],
        )
        suff_margin = semantic_margin(suff_logits, batch)
        wrong_margin = semantic_margin(wrong_logits, batch)
        null_margin = semantic_margin(null_logits, batch)
        suff_predictions = prediction_colors(suff_logits)
        wrong_predictions = prediction_colors(wrong_logits)
        null_predictions = prediction_colors(null_logits)
        for index, item in enumerate(batch):
            semantic_denominator = float(
                source_margin[index] - target_margin[index]
            )
            rows.append(
                {
                    "schema_version": "phase1001_head_control_row.v1",
                    "phase": PHASE_ID,
                    "model": MODEL,
                    "partition": item["partition"],
                    "pair_id": item["pair_id"],
                    "direction": item["direction"],
                    "event_id": event["event_id"],
                    "layer_number": layer_number,
                    "head_index": head_index,
                    "wrong_head_index": wrong_head,
                    "source_gold": item["source"]["gold"],
                    "target_gold": item["target"]["gold"],
                    "source_margin": float(source_margin[index]),
                    "target_margin": float(target_margin[index]),
                    "do_source_margin": float(do_margin[index]),
                    "sufficiency_margin": float(suff_margin[index]),
                    "wrong_o_margin": float(wrong_margin[index]),
                    "cross_pair_null_margin": float(null_margin[index]),
                    "sufficiency_transfer": float(
                        (suff_margin[index] - target_margin[index])
                        / max(abs(semantic_denominator), 1e-8)
                    ),
                    "wrong_o_transfer": float(
                        (wrong_margin[index] - target_margin[index])
                        / max(abs(semantic_denominator), 1e-8)
                    ),
                    "cross_pair_null_transfer": float(
                        (null_margin[index] - target_margin[index])
                        / max(abs(semantic_denominator), 1e-8)
                    ),
                    "sufficiency_prediction": suff_predictions[index],
                    "wrong_o_prediction": wrong_predictions[index],
                    "cross_pair_null_prediction": null_predictions[index],
                    "sufficiency_flipped": (
                        suff_predictions[index] == item["source"]["gold"]
                    ),
                    "wrong_o_flipped": (
                        wrong_predictions[index] == item["source"]["gold"]
                    ),
                    "cross_pair_null_flipped": (
                        null_predictions[index] == item["source"]["gold"]
                    ),
                }
            )
        del suff_logits, wrong_logits, null_logits
    return rows


def summarize_controls(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["event_id"]].append(row)
    return {
        event_id: {
            "event_id": event_id,
            "layer_number": values[0]["layer_number"],
            "head_index": values[0]["head_index"],
            "n": len(values),
            "mean_sufficiency_transfer": float(
                np.mean([row["sufficiency_transfer"] for row in values])
            ),
            "sufficiency_flip_rate": float(
                np.mean([row["sufficiency_flipped"] for row in values])
            ),
            "mean_wrong_o_transfer": float(
                np.mean([row["wrong_o_transfer"] for row in values])
            ),
            "wrong_o_flip_rate": float(
                np.mean([row["wrong_o_flipped"] for row in values])
            ),
            "mean_cross_pair_null_transfer": float(
                np.mean([row["cross_pair_null_transfer"] for row in values])
            ),
            "cross_pair_null_flip_rate": float(
                np.mean([row["cross_pair_null_flipped"] for row in values])
            ),
        }
        for event_id, values in groups.items()
    }


def rank_heads(
    selected: list[dict[str, Any]],
    controls: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    values = []
    for item in selected:
        control = controls[item["event_id"]]
        score = (
            max(0.0, item["median_mediation_fraction"])
            + max(0.0, control["mean_sufficiency_transfer"])
            + max(
                0.0,
                control["mean_sufficiency_transfer"]
                - control["mean_wrong_o_transfer"],
            )
        )
        values.append({**item, **control, "causal_score": score})
    ranked = sorted(
        values,
        key=lambda item: (
            -item["causal_score"],
            -item["median_mediation_fraction"],
            item["layer_number"],
            item["head_index"],
        ),
    )
    for rank, item in enumerate(ranked, 1):
        item["causal_rank"] = rank
    return ranked


def joint_rows_for_batch(
    model,
    layers,
    device,
    batch: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    candidate_ids: dict[str, int],
    source_logits: torch.Tensor,
    target_logits: torch.Tensor,
    do_logits: torch.Tensor,
    source_patch: dict[str, Any],
    target_heads: dict[int, torch.Tensor],
    do_heads: dict[int, torch.Tensor],
) -> list[dict[str, Any]]:
    target_cases = [row["target"] for row in batch]
    source_margin = semantic_margin(source_logits, batch)
    target_margin = semantic_margin(target_logits, batch)
    do_margin = semantic_margin(do_logits, batch)
    sizes = sorted({min(size, len(ranked)) for size in JOINT_SIZES})
    rows = []
    for size in sizes:
        events = ranked[:size]
        restore_patches = [
            {
                "event": event,
                "vectors": target_heads[event["layer_number"]][
                    :, event["head_index"], :
                ],
            }
            for event in events
        ]
        suff_patches = [
            {
                "event": event,
                "vectors": do_heads[event["layer_number"]][
                    :, event["head_index"], :
                ],
            }
            for event in events
        ]
        restored_logits = forward_with_patches(
            model,
            layers,
            device,
            target_cases,
            candidate_ids,
            source_patch=source_patch,
            head_patches=restore_patches,
        )
        suff_logits = forward_with_patches(
            model,
            layers,
            device,
            target_cases,
            candidate_ids,
            head_patches=suff_patches,
        )
        restored_margin = semantic_margin(restored_logits, batch)
        suff_margin = semantic_margin(suff_logits, batch)
        restored_predictions = prediction_colors(restored_logits)
        suff_predictions = prediction_colors(suff_logits)
        for index, item in enumerate(batch):
            semantic_denominator = float(
                source_margin[index] - target_margin[index]
            )
            source_effect = float(do_margin[index] - target_margin[index])
            rows.append(
                {
                    "schema_version": "phase1001_head_joint_row.v1",
                    "phase": PHASE_ID,
                    "model": MODEL,
                    "partition": item["partition"],
                    "pair_id": item["pair_id"],
                    "direction": item["direction"],
                    "joint_size": size,
                    "event_ids": [event["event_id"] for event in events],
                    "source_gold": item["source"]["gold"],
                    "target_gold": item["target"]["gold"],
                    "sufficiency_transfer": float(
                        (suff_margin[index] - target_margin[index])
                        / max(abs(semantic_denominator), 1e-8)
                    ),
                    "mediation_fraction": float(
                        (do_margin[index] - restored_margin[index])
                        / max(abs(source_effect), 1e-8)
                    ),
                    "sufficiency_prediction": suff_predictions[index],
                    "restored_prediction": restored_predictions[index],
                    "sufficiency_flipped": (
                        suff_predictions[index] == item["source"]["gold"]
                    ),
                    "restored_to_target": (
                        restored_predictions[index] == item["target"]["gold"]
                    ),
                }
            )
        del restored_logits, suff_logits
    return rows


def summarize_joint(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[int(row["joint_size"])].append(row)
    return {
        str(size): {
            "joint_size": size,
            "event_ids": values[0]["event_ids"],
            "n": len(values),
            "mean_sufficiency_transfer": float(
                np.mean([row["sufficiency_transfer"] for row in values])
            ),
            "sufficiency_flip_rate": float(
                np.mean([row["sufficiency_flipped"] for row in values])
            ),
            "median_mediation_fraction": float(
                np.median([row["mediation_fraction"] for row in values])
            ),
            "mean_mediation_fraction": float(
                np.mean([row["mediation_fraction"] for row in values])
            ),
            "restored_to_target_rate": float(
                np.mean([row["restored_to_target"] for row in values])
            ),
        }
        for size, values in sorted(groups.items())
    }


def choose_joint_size(summary: dict[str, dict[str, Any]]) -> int:
    best = max(
        item["median_mediation_fraction"] for item in summary.values()
    )
    threshold = 0.95 * best
    eligible = [
        int(size)
        for size, item in summary.items()
        if item["median_mediation_fraction"] >= threshold
    ]
    return min(eligible)


def natural_joint_rows(
    model,
    layers,
    tokenizer,
    device,
    batch: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    joint_size: int,
    candidate_ids: dict[str, int],
    source_patch: dict[str, Any],
    target_heads: dict[int, torch.Tensor],
    target_attn: dict[int, torch.Tensor],
    effective_eos: list[int],
    budget: int,
) -> list[dict[str, Any]]:
    del candidate_ids
    target_cases = [row["target"] for row in batch]
    events = ranked[:joint_size]
    head_restore = [
        {
            "event": event,
            "vectors": target_heads[event["layer_number"]][
                :, event["head_index"], :
            ],
        }
        for event in events
    ]
    conditions = {
        "source_do": {
            "head_patches": [],
            "attention_patches": {},
        },
        "source_plus_frozen_head_restore": {
            "head_patches": head_restore,
            "attention_patches": {},
        },
        "source_plus_all_attention_restore": {
            "head_patches": [],
            "attention_patches": target_attn,
        },
    }
    rows = []
    for condition, patches in conditions.items():
        generated = generate_with_patches(
            model,
            layers,
            tokenizer,
            device,
            target_cases,
            source_patch,
            patches["head_patches"],
            patches["attention_patches"],
            effective_eos,
            budget,
        )
        for index, item in enumerate(batch):
            result = generated[index]
            rows.append(
                {
                    "schema_version": "phase1001_head_natural_row.v1",
                    "phase": PHASE_ID,
                    "model": MODEL,
                    "partition": item["partition"],
                    "pair_id": item["pair_id"],
                    "direction": item["direction"],
                    "condition": condition,
                    "joint_size": joint_size,
                    "event_ids": [event["event_id"] for event in events],
                    "source_gold": item["source"]["gold"],
                    "target_gold": item["target"]["gold"],
                    "prediction": result["prediction"],
                    "flipped_to_source": (
                        result["prediction"] == item["source"]["gold"]
                    ),
                    "restored_to_target": (
                        result["prediction"] == item["target"]["gold"]
                    ),
                    "eos_seen": result["eos_seen"],
                    "exact_short": result["exact_short"],
                    "generated_text": result["text"],
                }
            )
    return rows


def summarize_natural(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["condition"]].append(row)
    return {
        condition: {
            "n": len(values),
            "flip_rate": float(np.mean([row["flipped_to_source"] for row in values])),
            "target_rate": float(np.mean([row["restored_to_target"] for row in values])),
            "eos_rate": float(np.mean([row["eos_seen"] for row in values])),
            "exact_short_rate": float(np.mean([row["exact_short"] for row in values])),
        }
        for condition, values in sorted(groups.items())
    }


def run(scope: str, batch_size: int, natural_budget: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 1001 head discovery requires CUDA")
    (
        protocol,
        behavior,
        selected_pairs,
        directional,
        output_root,
    ) = selected_phase1000_inputs(scope)
    output_root.mkdir(parents=True, exist_ok=True)
    if not behavior.get("behavior_gate_pass"):
        raise RuntimeError("Phase 1000 behavior gate is not open")
    write_rows(output_root / "selected_pairs.jsonl", selected_pairs)
    candidate_ids = {
        color: int(protocol["candidate_token_ids"][color]) for color in COLORS
    }
    events = head_events()
    event_by_id = {event["event_id"]: event for event in events}
    model = tokenizer = None
    started = time.time()
    try:
        model, tokenizer, device = load_model(
            MODEL, dtype=torch.bfloat16, use_8bit=False
        )
        tokenizer.padding_side = "left"
        layers = get_layers(model)
        info = get_model_info(model, MODEL)
        effective_eos = eos_ids(model, tokenizer)
        if (
            model.config.num_attention_heads != HEAD_COUNT
            or model.config.head_dim != HEAD_DIM
        ):
            raise RuntimeError("Qwen3 head geometry drift")

        response_rows = []
        mediation_rows = []
        instrument_rows = []
        batch_cache = []
        batches = list(batches_by_template(directional, batch_size))
        for batch_number, batch in enumerate(batches, 1):
            source_cases = [row["source"] for row in batch]
            target_cases = [row["target"] for row in batch]
            source_logits, source_residuals = capture_residuals(
                model, device, source_cases, (SOURCE_DEPTH,), candidate_ids
            )
            (
                source_candidate,
                source_heads,
                _,
            ) = capture_attention_states(
                model, layers, device, source_cases, candidate_ids
            )
            (
                target_logits,
                target_heads,
                target_attn,
            ) = capture_attention_states(
                model, layers, device, target_cases, candidate_ids
            )
            source_patch = source_patch_spec(
                SOURCE_DEPTH,
                target_cases,
                source_residuals[SOURCE_DEPTH],
                "joint",
            )
            (
                do_logits,
                do_heads,
                _,
            ) = capture_attention_states(
                model,
                layers,
                device,
                target_cases,
                candidate_ids,
                source_patch=source_patch,
            )
            response_rows.extend(
                response_rows_for_batch(
                    model,
                    batch,
                    source_heads,
                    target_heads,
                    do_heads,
                    candidate_ids,
                )
            )
            all_head_restore = [
                {
                    "event": event,
                    "vectors": target_heads[event["layer_number"]][
                        :, event["head_index"], :
                    ],
                }
                for event in events
            ]
            all_head_logits = forward_with_patches(
                model,
                layers,
                device,
                target_cases,
                candidate_ids,
                source_patch=source_patch,
                head_patches=all_head_restore,
            )
            full_attn_logits = forward_with_patches(
                model,
                layers,
                device,
                target_cases,
                candidate_ids,
                source_patch=source_patch,
                attention_patches=target_attn,
            )
            all_margin = semantic_margin(all_head_logits, batch)
            full_margin = semantic_margin(full_attn_logits, batch)
            all_predictions = prediction_colors(all_head_logits)
            full_predictions = prediction_colors(full_attn_logits)
            for index, item in enumerate(batch):
                instrument_rows.append(
                    {
                        "schema_version": "phase1001_head_instrument_row.v1",
                        "phase": PHASE_ID,
                        "model": MODEL,
                        "partition": item["partition"],
                        "pair_id": item["pair_id"],
                        "direction": item["direction"],
                        "all_head_restore_margin": float(all_margin[index]),
                        "full_attention_restore_margin": float(full_margin[index]),
                        "absolute_margin_error": float(
                            abs(all_margin[index] - full_margin[index])
                        ),
                        "all_head_prediction": all_predictions[index],
                        "full_attention_prediction": full_predictions[index],
                        "prediction_agreement": (
                            all_predictions[index] == full_predictions[index]
                        ),
                    }
                )

            source_margin = semantic_margin(source_candidate, batch)
            target_margin = semantic_margin(target_logits, batch)
            do_margin = semantic_margin(do_logits, batch)
            for event in events:
                layer_number = event["layer_number"]
                head_index = event["head_index"]
                restored_logits = forward_with_patches(
                    model,
                    layers,
                    device,
                    target_cases,
                    candidate_ids,
                    source_patch=source_patch,
                    head_patches=[
                        {
                            "event": event,
                            "vectors": target_heads[layer_number][
                                :, head_index, :
                            ],
                        }
                    ],
                )
                restored_margin = semantic_margin(restored_logits, batch)
                restored_predictions = prediction_colors(restored_logits)
                for index, item in enumerate(batch):
                    mediation_rows.append(
                        mediation_row(
                            item,
                            event,
                            float(source_margin[index]),
                            float(target_margin[index]),
                            float(do_margin[index]),
                            float(restored_margin[index]),
                            restored_predictions[index],
                        )
                    )
                del restored_logits
            batch_cache.append(
                {
                    "batch": batch,
                    "source_logits": source_candidate.detach().cpu(),
                    "target_logits": target_logits.detach().cpu(),
                    "do_logits": do_logits.detach().cpu(),
                }
            )
            del (
                source_logits,
                source_residuals,
                source_candidate,
                source_heads,
                target_logits,
                target_heads,
                target_attn,
                do_logits,
                do_heads,
                all_head_logits,
                full_attn_logits,
            )
            if batch_number % 2 == 0 or batch_number == len(batches):
                print(
                    f"[head-mediation] {batch_number}/{len(batches)} batches",
                    flush=True,
                )

        response_summary = summarize_response(response_rows)
        mediation_summary = summarize_mediation(mediation_rows)
        selected = select_heads(response_summary, mediation_summary)
        write_rows(output_root / "response_rows.jsonl", response_rows)
        write_rows(output_root / "mediation_rows.jsonl", mediation_rows)
        write_rows(output_root / "instrument_rows.jsonl", instrument_rows)
        write_json(output_root / "response_summary.json", response_summary)
        write_json(output_root / "mediation_summary.json", mediation_summary)
        write_json(
            output_root / "selected_heads.json",
            {
                "schema_version": "phase1001_selected_heads.v1",
                "phase": PHASE_ID,
                "selection_partition": directional[0]["partition"],
                "selection_uses_holdout": False,
                "select_per_layer": SELECT_PER_LAYER,
                "select_limit": SELECT_LIMIT,
                "heads": selected,
            },
        )

        control_rows = []
        joint_rows = []
        natural_rows = []
        batches = list(batches_by_template(directional, batch_size))
        for batch_number, batch in enumerate(batches, 1):
            source_cases = [row["source"] for row in batch]
            target_cases = [row["target"] for row in batch]
            source_logits, source_residuals = capture_residuals(
                model, device, source_cases, (SOURCE_DEPTH,), candidate_ids
            )
            (
                source_candidate,
                _,
                _,
            ) = capture_attention_states(
                model, layers, device, source_cases, candidate_ids
            )
            (
                target_logits,
                target_heads,
                target_attn,
            ) = capture_attention_states(
                model, layers, device, target_cases, candidate_ids
            )
            source_patch = source_patch_spec(
                SOURCE_DEPTH,
                target_cases,
                source_residuals[SOURCE_DEPTH],
                "joint",
            )
            (
                do_logits,
                do_heads,
                _,
            ) = capture_attention_states(
                model,
                layers,
                device,
                target_cases,
                candidate_ids,
                source_patch=source_patch,
            )
            control_rows.extend(
                selected_head_controls(
                    model,
                    layers,
                    device,
                    batch,
                    selected,
                    candidate_ids,
                    source_candidate,
                    target_logits,
                    do_logits,
                    source_patch,
                    target_heads,
                    do_heads,
                )
            )
            del source_logits
            if batch_number % 2 == 0 or batch_number == len(batches):
                print(
                    f"[head-controls] {batch_number}/{len(batches)} batches",
                    flush=True,
                )
            del (
                source_residuals,
                source_candidate,
                target_logits,
                target_heads,
                target_attn,
                do_logits,
                do_heads,
            )

        control_summary = summarize_controls(control_rows)
        ranked = rank_heads(selected, control_summary)
        write_rows(output_root / "control_rows.jsonl", control_rows)
        write_json(output_root / "control_summary.json", control_summary)
        write_json(
            output_root / "frozen_head_ranking.json",
            {
                "schema_version": "phase1001_frozen_head_ranking.v1",
                "phase": PHASE_ID,
                "selection_partition": directional[0]["partition"],
                "selection_uses_holdout": False,
                "heads": ranked,
            },
        )

        batches = list(batches_by_template(directional, batch_size))
        for batch_number, batch in enumerate(batches, 1):
            source_cases = [row["source"] for row in batch]
            target_cases = [row["target"] for row in batch]
            _, source_residuals = capture_residuals(
                model, device, source_cases, (SOURCE_DEPTH,), candidate_ids
            )
            source_candidate, _, _ = capture_attention_states(
                model, layers, device, source_cases, candidate_ids
            )
            target_logits, target_heads, target_attn = capture_attention_states(
                model, layers, device, target_cases, candidate_ids
            )
            source_patch = source_patch_spec(
                SOURCE_DEPTH,
                target_cases,
                source_residuals[SOURCE_DEPTH],
                "joint",
            )
            do_logits, do_heads, _ = capture_attention_states(
                model,
                layers,
                device,
                target_cases,
                candidate_ids,
                source_patch=source_patch,
            )
            joint_rows.extend(
                joint_rows_for_batch(
                    model,
                    layers,
                    device,
                    batch,
                    ranked,
                    candidate_ids,
                    source_candidate,
                    target_logits,
                    do_logits,
                    source_patch,
                    target_heads,
                    do_heads,
                )
            )
            del (
                source_residuals,
                source_candidate,
                target_logits,
                target_heads,
                target_attn,
                do_logits,
                do_heads,
            )
            if batch_number % 2 == 0 or batch_number == len(batches):
                print(
                    f"[head-joint] {batch_number}/{len(batches)} batches",
                    flush=True,
                )
        joint_summary = summarize_joint(joint_rows)
        frozen_joint_size = choose_joint_size(joint_summary)
        write_rows(output_root / "joint_rows.jsonl", joint_rows)
        write_json(output_root / "joint_summary.json", joint_summary)

        batches = list(batches_by_template(directional, batch_size))
        for batch_number, batch in enumerate(batches, 1):
            source_cases = [row["source"] for row in batch]
            target_cases = [row["target"] for row in batch]
            _, source_residuals = capture_residuals(
                model, device, source_cases, (SOURCE_DEPTH,), candidate_ids
            )
            target_logits, target_heads, target_attn = capture_attention_states(
                model, layers, device, target_cases, candidate_ids
            )
            del target_logits
            source_patch = source_patch_spec(
                SOURCE_DEPTH,
                target_cases,
                source_residuals[SOURCE_DEPTH],
                "joint",
            )
            natural_rows.extend(
                natural_joint_rows(
                    model,
                    layers,
                    tokenizer,
                    device,
                    batch,
                    ranked,
                    frozen_joint_size,
                    candidate_ids,
                    source_patch,
                    target_heads,
                    target_attn,
                    effective_eos,
                    natural_budget,
                )
            )
            del source_residuals, target_heads, target_attn
            if batch_number % 2 == 0 or batch_number == len(batches):
                print(
                    f"[head-natural] {batch_number}/{len(batches)} batches",
                    flush=True,
                )
        natural_summary = summarize_natural(natural_rows)
        write_rows(output_root / "natural_rows.jsonl", natural_rows)

        instrument_metrics = {
            "n": len(instrument_rows),
            "mean_absolute_margin_error": float(
                np.mean([row["absolute_margin_error"] for row in instrument_rows])
            ),
            "max_absolute_margin_error": float(
                np.max([row["absolute_margin_error"] for row in instrument_rows])
            ),
            "prediction_agreement": float(
                np.mean([row["prediction_agreement"] for row in instrument_rows])
            ),
        }
        single_pass = [
            item["event_id"]
            for item in ranked
            if item["median_mediation_fraction"]
            >= HEAD_THRESHOLDS["single_median_mediation"]
            and item["mean_sufficiency_transfer"]
            >= HEAD_THRESHOLDS["single_mean_sufficiency_transfer"]
            and (
                item["mean_sufficiency_transfer"]
                - item["mean_wrong_o_transfer"]
            )
            >= HEAD_THRESHOLDS["single_wrong_o_excess"]
        ]
        frozen_joint = joint_summary[str(frozen_joint_size)]
        gate_checks = {
            "behavior": bool(behavior["behavior_gate_pass"]),
            "instrument_margin": (
                instrument_metrics["mean_absolute_margin_error"]
                <= HEAD_THRESHOLDS["instrument_mean_abs_margin_error"]
            ),
            "instrument_prediction": (
                instrument_metrics["prediction_agreement"]
                >= HEAD_THRESHOLDS["instrument_prediction_agreement"]
            ),
            "single_head": bool(single_pass),
            "joint_head_mediation": (
                frozen_joint["median_mediation_fraction"]
                >= HEAD_THRESHOLDS["joint_median_mediation"]
            ),
            "joint_head_natural": (
                natural_summary["source_plus_frozen_head_restore"]["target_rate"]
                >= HEAD_THRESHOLDS["joint_natural_restoration_rate"]
            ),
        }
        frozen_spec = {
            "schema_version": "phase1001_frozen_head_spec.v1",
            "phase": PHASE_ID,
            "model": MODEL,
            "source_depth": SOURCE_DEPTH,
            "target_layers": list(TARGET_LAYERS),
            "ranked_head_event_ids": [
                item["event_id"] for item in ranked
            ],
            "frozen_joint_size": frozen_joint_size,
            "frozen_joint_event_ids": [
                item["event_id"] for item in ranked[:frozen_joint_size]
            ],
            "selection_partition": directional[0]["partition"],
            "selection_uses_holdout": False,
            "frozen_before_holdout": True,
        }
        write_json(output_root / "frozen_spec.json", frozen_spec)
        summary = {
            "schema_version": "phase1001_head_discovery_summary.v1",
            "phase": PHASE_ID,
            "model": MODEL,
            "scope": scope,
            "selected_pair_count": len(selected_pairs),
            "direction_count": len(directional),
            "source_depth": SOURCE_DEPTH,
            "target_layers": list(TARGET_LAYERS),
            "head_count_per_layer": HEAD_COUNT,
            "head_dim": HEAD_DIM,
            "screened_head_count": len(events),
            "instrument_metrics": instrument_metrics,
            "response_summary": response_summary,
            "mediation_summary": mediation_summary,
            "selected_heads": selected,
            "control_summary": control_summary,
            "ranked_heads": ranked,
            "single_head_pass_events": single_pass,
            "joint_summary": joint_summary,
            "frozen_joint_size": frozen_joint_size,
            "natural_summary": natural_summary,
            "thresholds": HEAD_THRESHOLDS,
            "gate_checks": gate_checks,
            "head_discovery_gate_pass": all(gate_checks.values()),
            "holdout_not_opened": True,
            "n_layers": info.n_layers,
            "d_model": info.d_model,
            "elapsed_seconds": time.time() - started,
            "cuda_device": torch.cuda.get_device_name(0),
        }
        write_json(output_root / "summary.json", summary)
        return summary
    finally:
        if model is not None:
            release_model(model)
        model = tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("smoke", "formal"), default="formal")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--natural-max-new-tokens", type=int, default=8)
    args = parser.parse_args()
    summary = run(args.scope, args.batch_size, args.natural_max_new_tokens)
    print(
        json.dumps(
            {
                "passed": summary["head_discovery_gate_pass"],
                "scope": args.scope,
                "single_head_pass_events": summary["single_head_pass_events"],
                "frozen_joint_size": summary["frozen_joint_size"],
                "gate_checks": summary["gate_checks"],
                "instrument_metrics": summary["instrument_metrics"],
                "natural_summary": summary["natural_summary"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
