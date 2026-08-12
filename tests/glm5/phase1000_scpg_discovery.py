#!/usr/bin/env python3
"""Phase 1000 source-driven causal propagation graph discovery.

This script uses only validation data to:

1. verify a joint entity-state source while color tokens stay fixed;
2. map source-driven responses in every downstream attention/MLP component;
3. causally screen frozen receiver candidates;
4. test single and joint mediation;
5. test natural one-token output under the selected intervention.

The holdout confirmation is implemented in a separate script.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
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
from phase1000_factorial_binding_behavior import (
    eos_ids,
    parse_generated,
    strip_at_eos,
)
from phase1000_factorial_binding_protocol import (
    COLORS,
    MODEL,
    OUT_ROOT,
    PHASE,
    canonical,
    write_json,
    write_jsonl,
)


SOURCE_DEPTHS = (1, 2, 4, 8, 16)
SOURCE_ROLES = ("slot0_entity", "slot1_entity")
MAP_ROLES = ("query_name", "answer_boundary")
COMPONENTS = ("attn", "mlp")
VALIDATION_PER_STRATUM = 8
SMOKE_PER_STRATUM = 1
RECEIVER_LIMIT = 12
JOINT_SIZES = (1, 2, 4, 8, 12)
SOURCE_THRESHOLDS = {
    "candidate_flip_rate": 0.70,
    "mean_transfer": 0.50,
    "natural_flip_rate": 0.70,
    "max_control_flip_rate": 0.10,
}
EDGE_THRESHOLDS = {
    "response_score": 0.10,
    "single_sufficiency_mean_transfer": 0.05,
    "single_median_mediation": 0.10,
    "single_max_scrambled_flip": 0.10,
    "joint_median_mediation": 0.30,
    "joint_natural_restoration_rate": 0.50,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    write_jsonl(path, rows)


def pair_hash(pair_id: str, salt: str) -> str:
    return hashlib.sha256(f"phase1000:{salt}:{pair_id}".encode("utf-8")).hexdigest()


def select_entity_pairs(
    factor_pairs: list[dict[str, Any]],
    case_by_id: dict[str, dict[str, Any]],
    scope: str,
) -> list[dict[str, Any]]:
    split = "smoke" if scope == "smoke" else "validation"
    per_stratum = SMOKE_PER_STRATUM if scope == "smoke" else VALIDATION_PER_STRATUM
    strata: dict[tuple[int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for pair in factor_pairs:
        if pair["factor"] != "entity" or pair["split"] != split:
            continue
        arm0 = case_by_id[pair["arm0_record_id"]]
        key = (
            int(arm0["template"]),
            int(arm0["display_order"]),
            int(arm0["value_swap"]),
            int(arm0["query_role"]),
        )
        strata[key].append(pair)
    selected: list[dict[str, Any]] = []
    for key, rows in sorted(strata.items()):
        ordered = sorted(rows, key=lambda row: pair_hash(row["pair_id"], "validation"))
        if len(ordered) < per_stratum:
            raise RuntimeError(f"underfilled validation stratum {key}: {len(ordered)}")
        selected.extend(ordered[:per_stratum])
    expected = 32 * per_stratum
    if len(selected) != expected:
        raise RuntimeError(f"selected pair count drift: {len(selected)} != {expected}")
    return selected


def directional_pairs(
    selected: list[dict[str, Any]],
    case_by_id: dict[str, dict[str, Any]],
    partition: str,
    bidirectional: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in selected:
        arm0 = case_by_id[pair["arm0_record_id"]]
        arm1 = case_by_id[pair["arm1_record_id"]]
        rows.append(
            {
                "pair_id": pair["pair_id"],
                "partition": partition,
                "direction": "e0_to_e1",
                "source": arm0,
                "target": arm1,
            }
        )
        if bidirectional:
            rows.append(
                {
                    "pair_id": pair["pair_id"],
                    "partition": partition,
                    "direction": "e1_to_e0",
                    "source": arm1,
                    "target": arm0,
                }
            )
    return rows


def batches_by_template(rows: list[dict[str, Any]], batch_size: int):
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[int(row["source"]["template"])].append(row)
    for template, values in sorted(groups.items()):
        values = sorted(values, key=lambda row: (row["pair_id"], row["direction"]))
        lengths = {
            item["source"]["input_token_count"] for item in values
        } | {item["target"]["input_token_count"] for item in values}
        if len(lengths) != 1:
            raise RuntimeError(f"batch length drift t{template}: {lengths}")
        for start in range(0, len(values), batch_size):
            yield values[start : start + batch_size]


def candidate_tensor(logits: torch.Tensor, candidate_ids: dict[str, int]) -> torch.Tensor:
    ids = torch.tensor(
        [candidate_ids[color] for color in COLORS],
        dtype=torch.long,
        device=logits.device,
    )
    return logits[:, ids].float()


def case_tensors(rows: list[dict[str, Any]], device):
    input_ids = torch.tensor(
        [row["input_ids"] for row in rows], dtype=torch.long, device=device
    )
    attention = torch.ones_like(input_ids)
    return input_ids, attention


def capture_residuals(
    model,
    device,
    rows: list[dict[str, Any]],
    depths: Iterable[int],
    candidate_ids: dict[str, int],
) -> tuple[torch.Tensor, dict[int, dict[str, torch.Tensor]]]:
    input_ids, attention = case_tensors(rows, device)
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    candidates = candidate_tensor(output.logits[:, -1, :], candidate_ids).detach()
    batch_index = torch.arange(len(rows), device=device)
    vectors: dict[int, dict[str, torch.Tensor]] = {}
    for depth in depths:
        vectors[int(depth)] = {}
        for role in SOURCE_ROLES:
            positions = torch.tensor(
                [row["role_positions"][role] for row in rows],
                dtype=torch.long,
                device=device,
            )
            vectors[int(depth)][role] = output.hidden_states[int(depth)][
                batch_index, positions, :
            ].detach()
    del output, input_ids, attention
    return candidates, vectors


def replace_positions(
    output,
    role_positions: dict[str, torch.Tensor],
    role_vectors: dict[str, torch.Tensor],
):
    is_tuple = isinstance(output, tuple)
    value = output[0] if is_tuple else output
    patched = value.clone()
    batch_index = torch.arange(value.shape[0], device=value.device)
    for role, positions in role_positions.items():
        vectors = role_vectors[role].to(device=value.device, dtype=value.dtype)
        patched[
            batch_index,
            positions.to(value.device),
            :,
        ] = vectors
    return (patched,) + output[1:] if is_tuple else patched


def source_patch_spec(
    depth: int,
    rows: list[dict[str, Any]],
    source_vectors: dict[str, torch.Tensor],
    mode: str = "joint",
) -> dict[str, Any]:
    if mode == "joint":
        role_map = {
            "slot0_entity": "slot0_entity",
            "slot1_entity": "slot1_entity",
        }
    elif mode == "single_slot0":
        role_map = {"slot0_entity": "slot0_entity"}
    elif mode == "single_slot1":
        role_map = {"slot1_entity": "slot1_entity"}
    elif mode == "reverse":
        role_map = {
            "slot0_entity": "slot1_entity",
            "slot1_entity": "slot0_entity",
        }
    else:
        raise ValueError(mode)
    return {
        "depth": int(depth),
        "role_positions": {
            target_role: torch.tensor(
                [row["role_positions"][target_role] for row in rows],
                dtype=torch.long,
                device=next(iter(source_vectors.values())).device,
            )
            for target_role in role_map
        },
        "role_vectors": {
            target_role: source_vectors[source_role]
            for target_role, source_role in role_map.items()
        },
    }


def register_source_patch(layers, patch: dict[str, Any] | None, full_width: int | None):
    if patch is None:
        return None, [0]
    count = [0]

    def hook(module, args, output):
        value = output[0] if isinstance(output, tuple) else output
        if full_width is not None and value.shape[1] != full_width:
            return output
        count[0] += 1
        return replace_positions(
            output,
            patch["role_positions"],
            patch["role_vectors"],
        )

    handle = layers[patch["depth"] - 1].register_forward_hook(hook)
    return handle, count


def component_module(layers, event: dict[str, Any]):
    layer = layers[int(event["block_index"])]
    return layer.self_attn if event["component"] == "attn" else layer.mlp


def register_receiver_patches(
    layers,
    rows: list[dict[str, Any]],
    patches: list[dict[str, Any]],
    device,
    full_width: int | None,
):
    handles = []
    counts: list[list[int]] = []
    for patch in patches:
        event = patch["event"]
        positions = torch.tensor(
            [row["role_positions"][event["role"]] for row in rows],
            dtype=torch.long,
            device=device,
        )
        count = [0]

        def make_hook(pos, vectors, counter):
            def hook(module, args, output):
                value = output[0] if isinstance(output, tuple) else output
                if full_width is not None and value.shape[1] != full_width:
                    return output
                counter[0] += 1
                return replace_positions(
                    output,
                    {"receiver": pos},
                    {"receiver": vectors},
                )

            return hook

        handles.append(
            component_module(layers, event).register_forward_hook(
                make_hook(positions, patch["vectors"], count)
            )
        )
        counts.append(count)
    return handles, counts


def forward_candidate(
    model,
    layers,
    device,
    rows: list[dict[str, Any]],
    candidate_ids: dict[str, int],
    source_patch: dict[str, Any] | None = None,
    receiver_patches: list[dict[str, Any]] | None = None,
) -> torch.Tensor:
    input_ids, attention = case_tensors(rows, device)
    source_handle = None
    receiver_handles = []
    receiver_counts: list[list[int]] = []
    try:
        source_handle, source_count = register_source_patch(
            layers, source_patch, full_width=None
        )
        receiver_handles, receiver_counts = register_receiver_patches(
            layers,
            rows,
            receiver_patches or [],
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
            raise RuntimeError(f"source hook count drift: {source_count[0]}")
        if any(counter[0] != 1 for counter in receiver_counts):
            raise RuntimeError(
                f"receiver hook count drift: {[item[0] for item in receiver_counts]}"
            )
        del output, input_ids, attention
        return result
    finally:
        for handle in reversed(receiver_handles):
            handle.remove()
        if source_handle is not None:
            source_handle.remove()


def event_definitions(source_depth: int, n_layers: int) -> list[dict[str, Any]]:
    events = []
    for block_index in range(source_depth, n_layers):
        for component in COMPONENTS:
            for role in MAP_ROLES:
                events.append(
                    {
                        "event_id": (
                            f"l{block_index + 1:02d}.{component}.{role}"
                        ),
                        "block_index": block_index,
                        "layer_number": block_index + 1,
                        "component": component,
                        "role": role,
                    }
                )
    return events


def capture_components(
    model,
    layers,
    device,
    rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
    candidate_ids: dict[str, int],
    source_patch: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    input_ids, attention = case_tensors(rows, device)
    captured: dict[str, torch.Tensor] = {}
    handles = []
    counts: dict[str, int] = defaultdict(int)
    source_handle = None
    try:
        source_handle, source_count = register_source_patch(
            layers, source_patch, full_width=None
        )
        for event in events:
            positions = torch.tensor(
                [row["role_positions"][event["role"]] for row in rows],
                dtype=torch.long,
                device=device,
            )

            def make_hook(event_id, pos):
                def hook(module, args, output):
                    value = output[0] if isinstance(output, tuple) else output
                    batch_index = torch.arange(value.shape[0], device=value.device)
                    captured[event_id] = value[
                        batch_index, pos.to(value.device), :
                    ].detach()
                    counts[event_id] += 1
                    return output

                return hook

            handles.append(
                component_module(layers, event).register_forward_hook(
                    make_hook(event["event_id"], positions)
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
            raise RuntimeError(f"source capture hook drift: {source_count[0]}")
        missing = [
            event["event_id"]
            for event in events
            if counts[event["event_id"]] != 1
        ]
        if missing:
            raise RuntimeError(f"component capture drift: {missing[:5]}")
        del output, input_ids, attention
        return candidates, captured
    finally:
        for handle in reversed(handles):
            handle.remove()
        if source_handle is not None:
            source_handle.remove()


def semantic_margin(
    logits: torch.Tensor,
    batch: list[dict[str, Any]],
) -> torch.Tensor:
    color_index = {color: index for index, color in enumerate(COLORS)}
    source_indices = torch.tensor(
        [color_index[row["source"]["gold"]] for row in batch],
        dtype=torch.long,
        device=logits.device,
    )
    target_indices = torch.tensor(
        [color_index[row["target"]["gold"]] for row in batch],
        dtype=torch.long,
        device=logits.device,
    )
    batch_index = torch.arange(len(batch), device=logits.device)
    return (
        logits[batch_index, source_indices] - logits[batch_index, target_indices]
    )


def prediction_colors(logits: torch.Tensor) -> list[str]:
    indices = torch.argmax(logits, dim=-1).detach().cpu().tolist()
    return [COLORS[int(index)] for index in indices]


def intervention_rows(
    batch: list[dict[str, Any]],
    source_logits: torch.Tensor,
    target_logits: torch.Tensor,
    patched_logits: torch.Tensor,
    condition: str,
    schema: str,
) -> list[dict[str, Any]]:
    source_margin = semantic_margin(source_logits, batch)
    target_margin = semantic_margin(target_logits, batch)
    patched_margin = semantic_margin(patched_logits, batch)
    predictions = prediction_colors(patched_logits)
    rows = []
    for index, item in enumerate(batch):
        denominator = float(source_margin[index] - target_margin[index])
        delta = float(patched_margin[index] - target_margin[index])
        rows.append(
            {
                "schema_version": schema,
                "phase": PHASE,
                "model": MODEL,
                "partition": item["partition"],
                "pair_id": item["pair_id"],
                "direction": item["direction"],
                "condition": condition,
                "source_gold": item["source"]["gold"],
                "target_gold": item["target"]["gold"],
                "source_margin": float(source_margin[index]),
                "target_margin": float(target_margin[index]),
                "patched_margin": float(patched_margin[index]),
                "delta_margin": delta,
                "normalized_transfer": delta / max(abs(denominator), 1e-8),
                "prediction": predictions[index],
                "flipped_to_source": predictions[index] == item["source"]["gold"],
                "remained_target": predictions[index] == item["target"]["gold"],
            }
        )
    return rows


def summarize_interventions(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["condition"]].append(row)
    return {
        condition: {
            "n": len(values),
            "mean_transfer": float(
                np.mean([row["normalized_transfer"] for row in values])
            ),
            "median_transfer": float(
                np.median([row["normalized_transfer"] for row in values])
            ),
            "flip_rate": float(np.mean([row["flipped_to_source"] for row in values])),
            "target_rate": float(np.mean([row["remained_target"] for row in values])),
        }
        for condition, values in sorted(groups.items())
    }


def scan_source_depths(
    model,
    layers,
    device,
    directional: list[dict[str, Any]],
    candidate_ids: dict[str, int],
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    batches = list(batches_by_template(directional, batch_size))
    for batch_number, batch in enumerate(batches, 1):
        source_cases = [row["source"] for row in batch]
        target_cases = [row["target"] for row in batch]
        source_logits, source_vectors = capture_residuals(
            model, device, source_cases, SOURCE_DEPTHS, candidate_ids
        )
        target_logits, _ = capture_residuals(
            model, device, target_cases, SOURCE_DEPTHS, candidate_ids
        )
        for depth in SOURCE_DEPTHS:
            patch = source_patch_spec(
                depth, target_cases, source_vectors[depth], mode="joint"
            )
            patched = forward_candidate(
                model,
                layers,
                device,
                target_cases,
                candidate_ids,
                source_patch=patch,
            )
            rows.extend(
                intervention_rows(
                    batch,
                    source_logits,
                    target_logits,
                    patched,
                    f"depth_{depth}_joint_entity",
                    "phase1000_source_scan_row.v1",
                )
            )
        del source_logits, source_vectors, target_logits
        if batch_number % 4 == 0 or batch_number == len(batches):
            print(f"[source-scan] {batch_number}/{len(batches)} batches", flush=True)
    summary = summarize_interventions(rows)
    eligible = []
    for depth in SOURCE_DEPTHS:
        metric = summary[f"depth_{depth}_joint_entity"]
        if (
            metric["flip_rate"] >= SOURCE_THRESHOLDS["candidate_flip_rate"]
            and metric["mean_transfer"] >= SOURCE_THRESHOLDS["mean_transfer"]
        ):
            eligible.append(depth)
    if eligible:
        selected_depth = min(eligible)
        criterion_pass = True
    else:
        selected_depth = max(
            SOURCE_DEPTHS,
            key=lambda depth: (
                summary[f"depth_{depth}_joint_entity"]["flip_rate"],
                summary[f"depth_{depth}_joint_entity"]["mean_transfer"],
                -depth,
            ),
        )
        criterion_pass = False
    selection = {
        "source_depths_pre_registered": list(SOURCE_DEPTHS),
        "selection_partition": "validation",
        "selection_uses_holdout": False,
        "criterion": {
            "candidate_flip_rate": SOURCE_THRESHOLDS["candidate_flip_rate"],
            "mean_transfer": SOURCE_THRESHOLDS["mean_transfer"],
            "rule": "earliest_depth_meeting_both_thresholds",
        },
        "eligible_depths": eligible,
        "selected_depth": selected_depth,
        "criterion_pass": criterion_pass,
        "condition_summary": summary,
    }
    return rows, selection


def source_controls(
    model,
    layers,
    device,
    directional: list[dict[str, Any]],
    candidate_ids: dict[str, int],
    depth: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    batches = list(batches_by_template(directional, batch_size))
    for batch_number, batch in enumerate(batches, 1):
        source_cases = [row["source"] for row in batch]
        target_cases = [row["target"] for row in batch]
        source_logits, source_residuals = capture_residuals(
            model, device, source_cases, (depth,), candidate_ids
        )
        target_logits, target_residuals = capture_residuals(
            model, device, target_cases, (depth,), candidate_ids
        )
        source_vectors = source_residuals[depth]
        target_vectors = target_residuals[depth]
        scrambled_vectors = {
            role: torch.roll(vector, shifts=1, dims=0)
            for role, vector in source_vectors.items()
        }
        conditions = {
            "joint_entity": source_patch_spec(
                depth, target_cases, source_vectors, "joint"
            ),
            "single_slot0": source_patch_spec(
                depth, target_cases, source_vectors, "single_slot0"
            ),
            "single_slot1": source_patch_spec(
                depth, target_cases, source_vectors, "single_slot1"
            ),
            "reverse_entity": source_patch_spec(
                depth, target_cases, source_vectors, "reverse"
            ),
            "scrambled_pair": source_patch_spec(
                depth, target_cases, scrambled_vectors, "joint"
            ),
            "noop_target": source_patch_spec(
                depth, target_cases, target_vectors, "joint"
            ),
        }
        for condition, patch in conditions.items():
            patched = forward_candidate(
                model,
                layers,
                device,
                target_cases,
                candidate_ids,
                source_patch=patch,
            )
            rows.extend(
                intervention_rows(
                    batch,
                    source_logits,
                    target_logits,
                    patched,
                    condition,
                    "phase1000_source_control_row.v1",
                )
            )
        del source_logits, target_logits, source_residuals, target_residuals
        if batch_number % 4 == 0 or batch_number == len(batches):
            print(f"[source-controls] {batch_number}/{len(batches)} batches", flush=True)
    return rows


def generate_with_interventions(
    model,
    layers,
    tokenizer,
    device,
    rows: list[dict[str, Any]],
    source_patch: dict[str, Any] | None,
    receiver_patches: list[dict[str, Any]] | None,
    effective_eos: list[int],
    budget: int,
) -> list[dict[str, Any]]:
    input_ids, attention = case_tensors(rows, device)
    full_width = input_ids.shape[1]
    source_handle = None
    receiver_handles = []
    try:
        source_handle, source_count = register_source_patch(
            layers, source_patch, full_width=full_width
        )
        receiver_handles, receiver_counts = register_receiver_patches(
            layers,
            rows,
            receiver_patches or [],
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
            raise RuntimeError(f"generation source hook drift: {source_count[0]}")
        if any(counter[0] != 1 for counter in receiver_counts):
            raise RuntimeError(
                "generation receiver hook drift: "
                f"{[counter[0] for counter in receiver_counts]}"
            )
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
                    "suffix": suffix,
                    "before_eos": before,
                    "eos_position": eos_position,
                    "text": text,
                    "prediction": parsed["first_color"],
                    "exact_short": parsed["exact_short"],
                }
            )
        del generated, input_ids, attention
        return results
    finally:
        for handle in reversed(receiver_handles):
            handle.remove()
        if source_handle is not None:
            source_handle.remove()


def natural_source_controls(
    model,
    layers,
    tokenizer,
    device,
    directional: list[dict[str, Any]],
    candidate_ids: dict[str, int],
    depth: int,
    effective_eos: list[int],
    batch_size: int,
    budget: int,
) -> list[dict[str, Any]]:
    del candidate_ids
    result_rows: list[dict[str, Any]] = []
    batches = list(batches_by_template(directional, batch_size))
    for batch_number, batch in enumerate(batches, 1):
        source_cases = [row["source"] for row in batch]
        target_cases = [row["target"] for row in batch]
        _, source_residuals = capture_residuals(
            model,
            device,
            source_cases,
            (depth,),
            {color: int(source_cases[0]["candidate_token_ids"][color]) for color in COLORS},
        )
        _, target_residuals = capture_residuals(
            model,
            device,
            target_cases,
            (depth,),
            {color: int(target_cases[0]["candidate_token_ids"][color]) for color in COLORS},
        )
        source_vectors = source_residuals[depth]
        target_vectors = target_residuals[depth]
        scrambled_vectors = {
            role: torch.roll(vector, shifts=1, dims=0)
            for role, vector in source_vectors.items()
        }
        conditions = {
            "joint_entity": source_patch_spec(
                depth, target_cases, source_vectors, "joint"
            ),
            "single_slot0": source_patch_spec(
                depth, target_cases, source_vectors, "single_slot0"
            ),
            "single_slot1": source_patch_spec(
                depth, target_cases, source_vectors, "single_slot1"
            ),
            "reverse_entity": source_patch_spec(
                depth, target_cases, source_vectors, "reverse"
            ),
            "scrambled_pair": source_patch_spec(
                depth, target_cases, scrambled_vectors, "joint"
            ),
            "noop_target": source_patch_spec(
                depth, target_cases, target_vectors, "joint"
            ),
        }
        for condition, patch in conditions.items():
            generated = generate_with_interventions(
                model,
                layers,
                tokenizer,
                device,
                target_cases,
                patch,
                None,
                effective_eos,
                budget,
            )
            for index, item in enumerate(batch):
                output = generated[index]
                result_rows.append(
                    {
                        "schema_version": "phase1000_source_natural_row.v1",
                        "phase": PHASE,
                        "model": MODEL,
                        "partition": item["partition"],
                        "pair_id": item["pair_id"],
                        "direction": item["direction"],
                        "condition": condition,
                        "source_gold": item["source"]["gold"],
                        "target_gold": item["target"]["gold"],
                        "prediction": output["prediction"],
                        "flipped_to_source": (
                            output["prediction"] == item["source"]["gold"]
                        ),
                        "remained_target": (
                            output["prediction"] == item["target"]["gold"]
                        ),
                        "eos_seen": output["eos_position"] is not None,
                        "exact_short": output["exact_short"],
                        "generated_text": output["text"],
                        "generated_suffix": output["suffix"],
                    }
                )
        del source_residuals, target_residuals
        if batch_number % 4 == 0 or batch_number == len(batches):
            print(f"[source-natural] {batch_number}/{len(batches)} batches", flush=True)
    return result_rows


def summarize_natural(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["condition"]].append(row)
    return {
        condition: {
            "n": len(values),
            "flip_rate": float(np.mean([row["flipped_to_source"] for row in values])),
            "target_rate": float(np.mean([row["remained_target"] for row in values])),
            "eos_rate": float(np.mean([row["eos_seen"] for row in values])),
            "exact_short_rate": float(np.mean([row["exact_short"] for row in values])),
        }
        for condition, values in sorted(groups.items())
    }


def response_map(
    model,
    layers,
    device,
    directional: list[dict[str, Any]],
    candidate_ids: dict[str, int],
    source_depth: int,
    events: list[dict[str, Any]],
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    event_by_id = {event["event_id"]: event for event in events}
    output_weight = model.get_output_embeddings().weight.detach()
    batches = list(batches_by_template(directional, batch_size))
    color_index = {color: index for index, color in enumerate(COLORS)}
    candidate_token_ids = torch.tensor(
        [candidate_ids[color] for color in COLORS],
        dtype=torch.long,
        device=device,
    )
    candidate_unembed = output_weight[candidate_token_ids].float()
    for batch_number, batch in enumerate(batches, 1):
        source_cases = [row["source"] for row in batch]
        target_cases = [row["target"] for row in batch]
        _, source_residuals = capture_residuals(
            model, device, source_cases, (source_depth,), candidate_ids
        )
        source_logits, source_components = capture_components(
            model, layers, device, source_cases, events, candidate_ids
        )
        target_logits, target_components = capture_components(
            model, layers, device, target_cases, events, candidate_ids
        )
        patch = source_patch_spec(
            source_depth,
            target_cases,
            source_residuals[source_depth],
            "joint",
        )
        do_logits, do_components = capture_components(
            model,
            layers,
            device,
            target_cases,
            events,
            candidate_ids,
            source_patch=patch,
        )
        del source_logits, target_logits, do_logits
        source_color_indices = torch.tensor(
            [color_index[item["source"]["gold"]] for item in batch],
            dtype=torch.long,
            device=device,
        )
        target_color_indices = torch.tensor(
            [color_index[item["target"]["gold"]] for item in batch],
            dtype=torch.long,
            device=device,
        )
        unembed_direction = (
            candidate_unembed[source_color_indices]
            - candidate_unembed[target_color_indices]
        )
        for event_id, event in event_by_id.items():
            natural_delta = (
                source_components[event_id] - target_components[event_id]
            ).float()
            do_delta = (do_components[event_id] - target_components[event_id]).float()
            natural_norm = torch.linalg.vector_norm(natural_delta, dim=-1)
            do_norm = torch.linalg.vector_norm(do_delta, dim=-1)
            cosine = F.cosine_similarity(do_delta, natural_delta, dim=-1, eps=1e-8)
            magnitude_fraction = do_norm / torch.clamp(natural_norm, min=1e-8)
            direct_logit = torch.sum(do_delta * unembed_direction, dim=-1)
            for index, item in enumerate(batch):
                rows.append(
                    {
                        "schema_version": "phase1000_response_row.v1",
                        "phase": PHASE,
                        "model": MODEL,
                        "partition": item["partition"],
                        "pair_id": item["pair_id"],
                        "direction": item["direction"],
                        **event,
                        "natural_delta_norm": float(natural_norm[index]),
                        "source_driven_delta_norm": float(do_norm[index]),
                        "source_to_natural_cosine": float(cosine[index]),
                        "magnitude_fraction": float(magnitude_fraction[index]),
                        "direct_source_vs_target_logit_effect": float(
                            direct_logit[index]
                        ),
                    }
                )
        del (
            source_residuals,
            source_components,
            target_components,
            do_components,
            natural_delta,
            do_delta,
        )
        if batch_number % 2 == 0 or batch_number == len(batches):
            print(f"[response-map] {batch_number}/{len(batches)} batches", flush=True)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["event_id"]].append(row)
    metrics: dict[str, dict[str, Any]] = {}
    for event_id, values in groups.items():
        event = event_by_id[event_id]
        median_cosine = float(
            np.median([row["source_to_natural_cosine"] for row in values])
        )
        positive_rate = float(
            np.mean([row["source_to_natural_cosine"] > 0 for row in values])
        )
        median_fraction = float(
            np.median([row["magnitude_fraction"] for row in values])
        )
        response_score = (
            max(0.0, median_cosine)
            * positive_rate
            * min(max(median_fraction, 0.0), 2.0)
        )
        metrics[event_id] = {
            **event,
            "n": len(values),
            "median_source_to_natural_cosine": median_cosine,
            "mean_source_to_natural_cosine": float(
                np.mean([row["source_to_natural_cosine"] for row in values])
            ),
            "positive_alignment_rate": positive_rate,
            "median_magnitude_fraction": median_fraction,
            "mean_source_driven_delta_norm": float(
                np.mean([row["source_driven_delta_norm"] for row in values])
            ),
            "mean_direct_source_vs_target_logit_effect": float(
                np.mean(
                    [row["direct_source_vs_target_logit_effect"] for row in values]
                )
            ),
            "response_score": response_score,
        }
    return rows, metrics


def select_receiver_candidates(
    metrics: dict[str, dict[str, Any]],
    source_depth: int,
    n_layers: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add_best(values: list[dict[str, Any]], count: int) -> None:
        ordered = sorted(
            values,
            key=lambda item: (
                -item["response_score"],
                item["layer_number"],
                item["event_id"],
            ),
        )
        for item in ordered[:count]:
            if item["event_id"] not in selected_ids:
                selected.append(dict(item))
                selected_ids.add(item["event_id"])

    values = list(metrics.values())
    for role in MAP_ROLES:
        for component in COMPONENTS:
            add_best(
                [
                    item
                    for item in values
                    if item["role"] == role and item["component"] == component
                ],
                2,
            )

    downstream = list(range(source_depth, n_layers))
    if downstream:
        boundaries = np.array_split(downstream, 3)
        for component in COMPONENTS:
            for band_index, band in enumerate(boundaries):
                band_set = {int(value) for value in band.tolist()}
                candidates = [
                    item
                    for item in values
                    if item["role"] == "answer_boundary"
                    and item["component"] == component
                    and item["block_index"] in band_set
                ]
                before = len(selected)
                add_best(candidates, 1)
                if len(selected) > before:
                    selected[-1]["stratified_band"] = band_index

    add_best(values, RECEIVER_LIMIT)
    selected = selected[:RECEIVER_LIMIT]
    for rank, item in enumerate(selected, 1):
        item["response_selection_rank"] = rank
        item["selection_partition"] = "validation"
        item["selection_uses_holdout"] = False
    return selected


def receiver_screen(
    model,
    layers,
    device,
    directional: list[dict[str, Any]],
    candidate_ids: dict[str, int],
    source_depth: int,
    selected_events: list[dict[str, Any]],
    batch_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    batches = list(batches_by_template(directional, batch_size))
    for batch_number, batch in enumerate(batches, 1):
        source_cases = [row["source"] for row in batch]
        target_cases = [row["target"] for row in batch]
        source_logits, source_residuals = capture_residuals(
            model, device, source_cases, (source_depth,), candidate_ids
        )
        target_logits, target_components = capture_components(
            model, layers, device, target_cases, selected_events, candidate_ids
        )
        source_patch = source_patch_spec(
            source_depth,
            target_cases,
            source_residuals[source_depth],
            "joint",
        )
        do_logits, do_components = capture_components(
            model,
            layers,
            device,
            target_cases,
            selected_events,
            candidate_ids,
            source_patch=source_patch,
        )
        source_margin = semantic_margin(source_logits, batch)
        target_margin = semantic_margin(target_logits, batch)
        do_margin = semantic_margin(do_logits, batch)
        source_predictions = prediction_colors(source_logits)
        target_predictions = prediction_colors(target_logits)
        do_predictions = prediction_colors(do_logits)
        for event in selected_events:
            event_id = event["event_id"]
            suff_logits = forward_candidate(
                model,
                layers,
                device,
                target_cases,
                candidate_ids,
                receiver_patches=[
                    {"event": event, "vectors": do_components[event_id]}
                ],
            )
            scrambled_logits = forward_candidate(
                model,
                layers,
                device,
                target_cases,
                candidate_ids,
                receiver_patches=[
                    {
                        "event": event,
                        "vectors": torch.roll(
                            do_components[event_id], shifts=1, dims=0
                        ),
                    }
                ],
            )
            restored_logits = forward_candidate(
                model,
                layers,
                device,
                target_cases,
                candidate_ids,
                source_patch=source_patch,
                receiver_patches=[
                    {"event": event, "vectors": target_components[event_id]}
                ],
            )
            suff_margin = semantic_margin(suff_logits, batch)
            scrambled_margin = semantic_margin(scrambled_logits, batch)
            restored_margin = semantic_margin(restored_logits, batch)
            suff_predictions = prediction_colors(suff_logits)
            scrambled_predictions = prediction_colors(scrambled_logits)
            restored_predictions = prediction_colors(restored_logits)
            for index, item in enumerate(batch):
                semantic_denominator = float(
                    source_margin[index] - target_margin[index]
                )
                source_effect = float(do_margin[index] - target_margin[index])
                mediation = float(
                    (do_margin[index] - restored_margin[index])
                    / max(abs(source_effect), 1e-8)
                )
                rows.append(
                    {
                        "schema_version": "phase1000_receiver_causal_row.v1",
                        "phase": PHASE,
                        "model": MODEL,
                        "partition": item["partition"],
                        "pair_id": item["pair_id"],
                        "direction": item["direction"],
                        **{
                            key: event[key]
                            for key in (
                                "event_id",
                                "block_index",
                                "layer_number",
                                "component",
                                "role",
                            )
                        },
                        "source_gold": item["source"]["gold"],
                        "target_gold": item["target"]["gold"],
                        "source_margin": float(source_margin[index]),
                        "target_margin": float(target_margin[index]),
                        "do_source_margin": float(do_margin[index]),
                        "sufficiency_margin": float(suff_margin[index]),
                        "scrambled_sufficiency_margin": float(
                            scrambled_margin[index]
                        ),
                        "restored_margin": float(restored_margin[index]),
                        "source_transfer": source_effect
                        / max(abs(semantic_denominator), 1e-8),
                        "sufficiency_transfer": float(
                            (suff_margin[index] - target_margin[index])
                            / max(abs(semantic_denominator), 1e-8)
                        ),
                        "scrambled_sufficiency_transfer": float(
                            (scrambled_margin[index] - target_margin[index])
                            / max(abs(semantic_denominator), 1e-8)
                        ),
                        "mediation_fraction": mediation,
                        "source_prediction": source_predictions[index],
                        "target_prediction": target_predictions[index],
                        "do_source_prediction": do_predictions[index],
                        "sufficiency_prediction": suff_predictions[index],
                        "scrambled_prediction": scrambled_predictions[index],
                        "restored_prediction": restored_predictions[index],
                        "source_flipped": (
                            do_predictions[index] == item["source"]["gold"]
                        ),
                        "sufficiency_flipped": (
                            suff_predictions[index] == item["source"]["gold"]
                        ),
                        "scrambled_flipped": (
                            scrambled_predictions[index] == item["source"]["gold"]
                        ),
                        "restored_to_target": (
                            restored_predictions[index] == item["target"]["gold"]
                        ),
                    }
                )
            del suff_logits, scrambled_logits, restored_logits
        del (
            source_logits,
            source_residuals,
            target_logits,
            target_components,
            do_logits,
            do_components,
        )
        if batch_number % 2 == 0 or batch_number == len(batches):
            print(f"[receiver-screen] {batch_number}/{len(batches)} batches", flush=True)
    return rows


def summarize_receivers(
    rows: list[dict[str, Any]],
    response_metrics: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["event_id"]].append(row)
    summary = {}
    for event_id, values in groups.items():
        response = response_metrics[event_id]
        mean_sufficiency = float(
            np.mean([row["sufficiency_transfer"] for row in values])
        )
        median_mediation = float(
            np.median([row["mediation_fraction"] for row in values])
        )
        scrambled_flip = float(
            np.mean([row["scrambled_flipped"] for row in values])
        )
        causal_score = (
            max(0.0, mean_sufficiency)
            + max(0.0, median_mediation)
            - scrambled_flip
        )
        summary[event_id] = {
            **{
                key: response[key]
                for key in (
                    "event_id",
                    "block_index",
                    "layer_number",
                    "component",
                    "role",
                    "response_score",
                    "median_source_to_natural_cosine",
                    "median_magnitude_fraction",
                )
            },
            "n": len(values),
            "mean_source_transfer": float(
                np.mean([row["source_transfer"] for row in values])
            ),
            "source_flip_rate": float(
                np.mean([row["source_flipped"] for row in values])
            ),
            "mean_sufficiency_transfer": mean_sufficiency,
            "sufficiency_flip_rate": float(
                np.mean([row["sufficiency_flipped"] for row in values])
            ),
            "mean_scrambled_sufficiency_transfer": float(
                np.mean(
                    [row["scrambled_sufficiency_transfer"] for row in values]
                )
            ),
            "scrambled_flip_rate": scrambled_flip,
            "median_mediation_fraction": median_mediation,
            "mean_mediation_fraction": float(
                np.mean([row["mediation_fraction"] for row in values])
            ),
            "restored_to_target_rate": float(
                np.mean([row["restored_to_target"] for row in values])
            ),
            "causal_score": causal_score,
        }
    return summary


def rank_receivers(
    selected_events: list[dict[str, Any]],
    receiver_metrics: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    ranked = sorted(
        [receiver_metrics[event["event_id"]] for event in selected_events],
        key=lambda item: (
            -item["causal_score"],
            -item["median_mediation_fraction"],
            -item["mean_sufficiency_transfer"],
            item["layer_number"],
            item["event_id"],
        ),
    )
    for rank, item in enumerate(ranked, 1):
        item["causal_rank"] = rank
        item["ranking_partition"] = "validation"
        item["ranking_uses_holdout"] = False
    return ranked


def joint_screen(
    model,
    layers,
    device,
    directional: list[dict[str, Any]],
    candidate_ids: dict[str, int],
    source_depth: int,
    ranked_receivers: list[dict[str, Any]],
    batch_size: int,
) -> list[dict[str, Any]]:
    event_lookup = {item["event_id"]: item for item in ranked_receivers}
    sizes = sorted({min(size, len(ranked_receivers)) for size in JOINT_SIZES})
    all_events = [event_lookup[item["event_id"]] for item in ranked_receivers]
    rows: list[dict[str, Any]] = []
    batches = list(batches_by_template(directional, batch_size))
    for batch_number, batch in enumerate(batches, 1):
        source_cases = [row["source"] for row in batch]
        target_cases = [row["target"] for row in batch]
        source_logits, source_residuals = capture_residuals(
            model, device, source_cases, (source_depth,), candidate_ids
        )
        target_logits, target_components = capture_components(
            model, layers, device, target_cases, all_events, candidate_ids
        )
        source_patch = source_patch_spec(
            source_depth,
            target_cases,
            source_residuals[source_depth],
            "joint",
        )
        do_logits, do_components = capture_components(
            model,
            layers,
            device,
            target_cases,
            all_events,
            candidate_ids,
            source_patch=source_patch,
        )
        source_margin = semantic_margin(source_logits, batch)
        target_margin = semantic_margin(target_logits, batch)
        do_margin = semantic_margin(do_logits, batch)
        for size in sizes:
            events = all_events[:size]
            sufficiency_patches = [
                {"event": event, "vectors": do_components[event["event_id"]]}
                for event in events
            ]
            restore_patches = [
                {"event": event, "vectors": target_components[event["event_id"]]}
                for event in events
            ]
            suff_logits = forward_candidate(
                model,
                layers,
                device,
                target_cases,
                candidate_ids,
                receiver_patches=sufficiency_patches,
            )
            restored_logits = forward_candidate(
                model,
                layers,
                device,
                target_cases,
                candidate_ids,
                source_patch=source_patch,
                receiver_patches=restore_patches,
            )
            suff_margin = semantic_margin(suff_logits, batch)
            restored_margin = semantic_margin(restored_logits, batch)
            suff_predictions = prediction_colors(suff_logits)
            restored_predictions = prediction_colors(restored_logits)
            for index, item in enumerate(batch):
                semantic_denominator = float(
                    source_margin[index] - target_margin[index]
                )
                source_effect = float(do_margin[index] - target_margin[index])
                rows.append(
                    {
                        "schema_version": "phase1000_joint_causal_row.v1",
                        "phase": PHASE,
                        "model": MODEL,
                        "partition": item["partition"],
                        "pair_id": item["pair_id"],
                        "direction": item["direction"],
                        "joint_size": size,
                        "event_ids": [event["event_id"] for event in events],
                        "source_gold": item["source"]["gold"],
                        "target_gold": item["target"]["gold"],
                        "source_margin": float(source_margin[index]),
                        "target_margin": float(target_margin[index]),
                        "do_source_margin": float(do_margin[index]),
                        "sufficiency_margin": float(suff_margin[index]),
                        "restored_margin": float(restored_margin[index]),
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
            del suff_logits, restored_logits
        del (
            source_logits,
            source_residuals,
            target_logits,
            target_components,
            do_logits,
            do_components,
        )
        if batch_number % 2 == 0 or batch_number == len(batches):
            print(f"[joint-screen] {batch_number}/{len(batches)} batches", flush=True)
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


def select_joint_size(joint_summary: dict[str, dict[str, Any]]) -> int:
    return int(
        sorted(
            joint_summary.values(),
            key=lambda item: (
                -item["median_mediation_fraction"],
                -item["restored_to_target_rate"],
                item["joint_size"],
            ),
        )[0]["joint_size"]
    )


def natural_joint_test(
    model,
    layers,
    tokenizer,
    device,
    directional: list[dict[str, Any]],
    candidate_ids: dict[str, int],
    source_depth: int,
    ranked_receivers: list[dict[str, Any]],
    joint_size: int,
    effective_eos: list[int],
    batch_size: int,
    budget: int,
) -> list[dict[str, Any]]:
    selected_events = ranked_receivers[:joint_size]
    rows: list[dict[str, Any]] = []
    batches = list(batches_by_template(directional, batch_size))
    for batch_number, batch in enumerate(batches, 1):
        source_cases = [row["source"] for row in batch]
        target_cases = [row["target"] for row in batch]
        _, source_residuals = capture_residuals(
            model, device, source_cases, (source_depth,), candidate_ids
        )
        _, target_components = capture_components(
            model, layers, device, target_cases, selected_events, candidate_ids
        )
        source_patch = source_patch_spec(
            source_depth,
            target_cases,
            source_residuals[source_depth],
            "joint",
        )
        conditions = {
            "source_do": [],
            "source_plus_joint_restore": [
                {
                    "event": event,
                    "vectors": target_components[event["event_id"]],
                }
                for event in selected_events
            ],
            "source_plus_scrambled_restore": [
                {
                    "event": event,
                    "vectors": torch.roll(
                        target_components[event["event_id"]], shifts=1, dims=0
                    ),
                }
                for event in selected_events
            ],
        }
        for condition, receiver_patches in conditions.items():
            generated = generate_with_interventions(
                model,
                layers,
                tokenizer,
                device,
                target_cases,
                source_patch,
                receiver_patches,
                effective_eos,
                budget,
            )
            for index, item in enumerate(batch):
                output = generated[index]
                rows.append(
                    {
                        "schema_version": "phase1000_joint_natural_row.v1",
                        "phase": PHASE,
                        "model": MODEL,
                        "partition": item["partition"],
                        "pair_id": item["pair_id"],
                        "direction": item["direction"],
                        "condition": condition,
                        "joint_size": joint_size,
                        "event_ids": [
                            event["event_id"] for event in selected_events
                        ],
                        "source_gold": item["source"]["gold"],
                        "target_gold": item["target"]["gold"],
                        "prediction": output["prediction"],
                        "flipped_to_source": (
                            output["prediction"] == item["source"]["gold"]
                        ),
                        "remained_target": (
                            output["prediction"] == item["target"]["gold"]
                        ),
                        "restored_to_target": (
                            output["prediction"] == item["target"]["gold"]
                        ),
                        "eos_seen": output["eos_position"] is not None,
                        "exact_short": output["exact_short"],
                        "generated_text": output["text"],
                    }
                )
        del source_residuals, target_components
        if batch_number % 2 == 0 or batch_number == len(batches):
            print(f"[joint-natural] {batch_number}/{len(batches)} batches", flush=True)
    return rows


def discovery_gate(
    behavior_summary: dict[str, Any],
    source_selection: dict[str, Any],
    source_control_summary: dict[str, Any],
    source_natural_summary: dict[str, Any],
    receiver_metrics: dict[str, dict[str, Any]],
    joint_summary: dict[str, dict[str, Any]],
    best_joint_size: int,
    joint_natural_summary: dict[str, dict[str, float]],
) -> tuple[dict[str, bool], dict[str, Any]]:
    controls = ("reverse_entity", "scrambled_pair", "noop_target")
    max_candidate_control = max(
        source_control_summary[condition]["flip_rate"] for condition in controls
    )
    max_natural_control = max(
        source_natural_summary[condition]["flip_rate"] for condition in controls
    )
    source_candidate = source_control_summary["joint_entity"]
    source_natural = source_natural_summary["joint_entity"]
    single_pass_events = []
    for event_id, metric in receiver_metrics.items():
        if (
            metric["response_score"] >= EDGE_THRESHOLDS["response_score"]
            and metric["mean_sufficiency_transfer"]
            >= EDGE_THRESHOLDS["single_sufficiency_mean_transfer"]
            and metric["median_mediation_fraction"]
            >= EDGE_THRESHOLDS["single_median_mediation"]
            and metric["scrambled_flip_rate"]
            <= EDGE_THRESHOLDS["single_max_scrambled_flip"]
        ):
            single_pass_events.append(event_id)
    best_joint = joint_summary[str(best_joint_size)]
    metrics = {
        "source_depth": source_selection["selected_depth"],
        "source_candidate_flip_rate": source_candidate["flip_rate"],
        "source_candidate_mean_transfer": source_candidate["mean_transfer"],
        "source_natural_flip_rate": source_natural["flip_rate"],
        "max_candidate_control_flip_rate": max_candidate_control,
        "max_natural_control_flip_rate": max_natural_control,
        "single_receiver_pass_events": single_pass_events,
        "best_joint_size": best_joint_size,
        "best_joint_median_mediation": best_joint["median_mediation_fraction"],
        "best_joint_candidate_restoration_rate": best_joint[
            "restored_to_target_rate"
        ],
        "joint_natural_restoration_rate": joint_natural_summary[
            "source_plus_joint_restore"
        ]["target_rate"],
        "joint_natural_scrambled_restoration_rate": joint_natural_summary[
            "source_plus_scrambled_restore"
        ]["target_rate"],
    }
    checks = {
        "G1_behavior": bool(behavior_summary["behavior_gate_pass"]),
        "G2_source_depth_criterion": bool(source_selection["criterion_pass"]),
        "G2_source_candidate": (
            source_candidate["flip_rate"]
            >= SOURCE_THRESHOLDS["candidate_flip_rate"]
            and source_candidate["mean_transfer"]
            >= SOURCE_THRESHOLDS["mean_transfer"]
        ),
        "G2_source_natural": (
            source_natural["flip_rate"]
            >= SOURCE_THRESHOLDS["natural_flip_rate"]
        ),
        "G2_source_controls": (
            max_candidate_control <= SOURCE_THRESHOLDS["max_control_flip_rate"]
            and max_natural_control <= SOURCE_THRESHOLDS["max_control_flip_rate"]
        ),
        "G3_G4_G5_single_receiver": bool(single_pass_events),
        "G5_joint_mediation": (
            best_joint["median_mediation_fraction"]
            >= EDGE_THRESHOLDS["joint_median_mediation"]
        ),
        "G6_natural_restoration": (
            joint_natural_summary["source_plus_joint_restore"]["target_rate"]
            >= EDGE_THRESHOLDS["joint_natural_restoration_rate"]
        ),
    }
    return checks, metrics


def run(
    scope: str,
    batch_size: int,
    natural_budget: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 1000 SCPG discovery requires CUDA")
    protocol_root = OUT_ROOT / ("smoke" if scope == "smoke" else "protocol")
    behavior_root = OUT_ROOT / (
        "smoke_behavior" if scope == "smoke" else "behavior"
    )
    output_root = OUT_ROOT / (
        "smoke_discovery" if scope == "smoke" else "discovery"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    cases = read_jsonl(protocol_root / "cases.jsonl")
    factor_pairs = read_jsonl(protocol_root / "factor_pairs.jsonl")
    protocol = json.loads((protocol_root / "protocol.json").read_text(encoding="utf-8"))
    behavior_summary = json.loads(
        (behavior_root / "summary.json").read_text(encoding="utf-8")
    )
    if not behavior_summary.get("behavior_gate_pass"):
        raise RuntimeError("behavior gate is not open")
    case_by_id = {row["record_id"]: row for row in cases}
    selected_pairs = select_entity_pairs(factor_pairs, case_by_id, scope)
    partition = "smoke" if scope == "smoke" else "validation"
    directional = directional_pairs(
        selected_pairs, case_by_id, partition, bidirectional=True
    )
    write_rows(output_root / "selected_pairs.jsonl", selected_pairs)

    candidate_ids = {
        color: int(protocol["candidate_token_ids"][color]) for color in COLORS
    }
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

        source_scan_rows, source_selection = scan_source_depths(
            model,
            layers,
            device,
            directional,
            candidate_ids,
            batch_size,
        )
        source_depth = int(source_selection["selected_depth"])
        write_rows(output_root / "source_scan_rows.jsonl", source_scan_rows)
        write_json(output_root / "source_selection.json", source_selection)

        source_control_rows = source_controls(
            model,
            layers,
            device,
            directional,
            candidate_ids,
            source_depth,
            batch_size,
        )
        source_control_summary = summarize_interventions(source_control_rows)
        write_rows(output_root / "source_control_rows.jsonl", source_control_rows)

        source_natural_rows = natural_source_controls(
            model,
            layers,
            tokenizer,
            device,
            directional,
            candidate_ids,
            source_depth,
            effective_eos,
            batch_size,
            natural_budget,
        )
        source_natural_summary = summarize_natural(source_natural_rows)
        write_rows(output_root / "source_natural_rows.jsonl", source_natural_rows)

        events = event_definitions(source_depth, info.n_layers)
        response_rows, response_metrics = response_map(
            model,
            layers,
            device,
            directional,
            candidate_ids,
            source_depth,
            events,
            batch_size,
        )
        write_rows(output_root / "response_rows.jsonl", response_rows)
        write_json(output_root / "response_metrics.json", response_metrics)
        selected_receivers = select_receiver_candidates(
            response_metrics, source_depth, info.n_layers
        )
        write_json(
            output_root / "selected_receivers.json",
            {
                "schema_version": "phase1000_selected_receivers.v1",
                "phase": PHASE,
                "model": MODEL,
                "selection_partition": partition,
                "selection_uses_holdout": False,
                "receiver_limit_pre_registered": RECEIVER_LIMIT,
                "receivers": selected_receivers,
            },
        )

        receiver_rows = receiver_screen(
            model,
            layers,
            device,
            directional,
            candidate_ids,
            source_depth,
            selected_receivers,
            batch_size,
        )
        receiver_metrics = summarize_receivers(receiver_rows, response_metrics)
        ranked_receivers = rank_receivers(selected_receivers, receiver_metrics)
        write_rows(output_root / "receiver_causal_rows.jsonl", receiver_rows)
        write_json(output_root / "receiver_metrics.json", receiver_metrics)
        write_json(
            output_root / "frozen_receiver_ranking.json",
            {
                "schema_version": "phase1000_frozen_receiver_ranking.v1",
                "phase": PHASE,
                "model": MODEL,
                "source_depth": source_depth,
                "ranking_partition": partition,
                "ranking_uses_holdout": False,
                "receivers": ranked_receivers,
            },
        )

        joint_rows = joint_screen(
            model,
            layers,
            device,
            directional,
            candidate_ids,
            source_depth,
            ranked_receivers,
            batch_size,
        )
        joint_summary = summarize_joint(joint_rows)
        best_joint_size = select_joint_size(joint_summary)
        write_rows(output_root / "joint_causal_rows.jsonl", joint_rows)

        joint_natural_rows = natural_joint_test(
            model,
            layers,
            tokenizer,
            device,
            directional,
            candidate_ids,
            source_depth,
            ranked_receivers,
            best_joint_size,
            effective_eos,
            batch_size,
            natural_budget,
        )
        joint_natural_summary = summarize_natural(joint_natural_rows)
        write_rows(output_root / "joint_natural_rows.jsonl", joint_natural_rows)

        checks, gate_metrics = discovery_gate(
            behavior_summary,
            source_selection,
            source_control_summary,
            source_natural_summary,
            receiver_metrics,
            joint_summary,
            best_joint_size,
            joint_natural_summary,
        )
        frozen = {
            "schema_version": "phase1000_frozen_discovery.v1",
            "phase": PHASE,
            "model": MODEL,
            "source_depth": source_depth,
            "ranked_receiver_event_ids": [
                item["event_id"] for item in ranked_receivers
            ],
            "best_joint_size": best_joint_size,
            "best_joint_event_ids": [
                item["event_id"]
                for item in ranked_receivers[:best_joint_size]
            ],
            "selection_partition": partition,
            "selection_uses_holdout": False,
            "frozen_before_holdout": True,
        }
        write_json(output_root / "frozen_spec.json", frozen)
        summary = {
            "schema_version": "phase1000_discovery_summary.v1",
            "phase": PHASE,
            "model": MODEL,
            "scope": scope,
            "selected_pair_count": len(selected_pairs),
            "direction_count": len(directional),
            "source_depth": source_depth,
            "source_selection": source_selection,
            "source_control_summary": source_control_summary,
            "source_natural_summary": source_natural_summary,
            "response_event_count": len(events),
            "response_row_count": len(response_rows),
            "selected_receiver_count": len(selected_receivers),
            "receiver_metrics": receiver_metrics,
            "ranked_receivers": ranked_receivers,
            "joint_summary": joint_summary,
            "best_joint_size": best_joint_size,
            "joint_natural_summary": joint_natural_summary,
            "source_thresholds": SOURCE_THRESHOLDS,
            "edge_thresholds": EDGE_THRESHOLDS,
            "gate_checks": checks,
            "gate_metrics": gate_metrics,
            "discovery_gate_pass": all(checks.values()),
            "holdout_not_opened": True,
            "n_layers": info.n_layers,
            "d_model": info.d_model,
            "batch_size": batch_size,
            "natural_max_new_tokens": natural_budget,
            "effective_eos_token_ids": effective_eos,
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
                "passed": summary["discovery_gate_pass"],
                "scope": args.scope,
                "source_depth": summary["source_depth"],
                "best_joint_size": summary["best_joint_size"],
                "gate_checks": summary["gate_checks"],
                "gate_metrics": summary["gate_metrics"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
