#!/usr/bin/env python3
"""Phase 998 causal swap, mediation, restoration, and natural-use tests."""
from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, get_model_info, load_model, release_model
from phase998_minimal_causal_thread_behavior import eos_ids, parse_generated, strip_at_eos
from phase998_minimal_causal_thread_protocol import (
    COLORS,
    MODEL,
    OUT_ROOT,
    PHASE,
    canonical,
    write_json,
)


METHODS = (
    "full",
    "difference_64",
    "difference_256",
    "top_activation_64",
    "top_activation_256",
    "random_64",
    "random_256",
    "noop_difference_256",
    "wrong_position_difference_256",
)
ROLE_NAMES = ("write", "read", "decision")
WRONG_POSITION_ROLE = {
    "write": "foil_color",
    "read": "source_entity",
    "decision": "query_name",
}
THRESHOLDS = {
    "upstream_candidate_flip_rate": 0.70,
    "upstream_natural_flip_rate": 0.60,
    "max_control_flip_rate": 0.10,
    "restoration_median_recovery": 0.70,
    "difference_vs_top_activation_effect_ratio": 2.0,
    "minimum_mediation_fraction": 0.30,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")
    temp.replace(path)


def directional_pairs(
    selected: list[dict[str, Any]],
    case_by_record: dict[str, dict[str, Any]],
    partition: str,
    bidirectional: bool,
) -> list[dict[str, Any]]:
    result = []
    for pair in selected:
        if pair["partition"] != partition:
            continue
        arm0 = case_by_record[pair["arm0_record_id"]]
        arm1 = case_by_record[pair["arm1_record_id"]]
        result.append(
            {
                "pair_id": pair["pair_id"],
                "template": pair["template"],
                "partition": partition,
                "direction": "a0_to_a1",
                "source": arm0,
                "target": arm1,
            }
        )
        if bidirectional:
            result.append(
                {
                    "pair_id": pair["pair_id"],
                    "template": pair["template"],
                    "partition": partition,
                    "direction": "a1_to_a0",
                    "source": arm1,
                    "target": arm0,
                }
            )
    return result


def batches_by_template(rows: list[dict[str, Any]], batch_size: int):
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["template"]].append(row)
    for template, values in sorted(grouped.items()):
        values = sorted(values, key=lambda row: (row["pair_id"], row["direction"]))
        lengths = {
            row["source"]["input_token_count"] for row in values
        } | {row["target"]["input_token_count"] for row in values}
        if len(lengths) != 1:
            raise RuntimeError(f"causal batch length drift: t{template}/{lengths}")
        for start in range(0, len(values), batch_size):
            yield values[start : start + batch_size]


def candidate_tensor(logits: torch.Tensor, candidate_ids: dict[str, int]) -> torch.Tensor:
    ids = torch.tensor(
        [candidate_ids[color] for color in COLORS],
        device=logits.device,
        dtype=torch.long,
    )
    return logits[:, ids].float()


def capture_clean(
    model,
    device,
    rows: list[dict[str, Any]],
    event_specs: dict[str, dict[str, Any]],
    candidate_ids: dict[str, int],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    input_ids = torch.tensor(
        [row["input_ids"] for row in rows], dtype=torch.long, device=device
    )
    attention = torch.ones_like(input_ids)
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
    vectors = {}
    for role, spec in event_specs.items():
        positions = torch.tensor(
            [row["role_positions"][spec["position_role"]] for row in rows],
            dtype=torch.long,
            device=device,
        )
        vectors[role] = output.hidden_states[spec["depth"]][
            batch_index, positions, :
        ].detach()
    del output, input_ids, attention
    return candidates, vectors


def replace_tensor(
    output,
    positions: torch.Tensor,
    vectors: torch.Tensor,
    channels: list[int] | None,
):
    is_tuple = isinstance(output, tuple)
    value = output[0] if is_tuple else output
    patched = value.clone()
    batch_index = torch.arange(value.shape[0], device=value.device)
    positions = positions.to(value.device)
    vectors = vectors.to(device=value.device, dtype=value.dtype)
    if channels is None:
        patched[batch_index, positions, :] = vectors
    else:
        channel_index = torch.tensor(channels, dtype=torch.long, device=value.device)
        patched[
            batch_index[:, None],
            positions[:, None],
            channel_index[None, :],
        ] = vectors[:, channel_index]
    return (patched,) + output[1:] if is_tuple else patched


def patched_forward(
    model,
    layers,
    device,
    rows: list[dict[str, Any]],
    patches: list[dict[str, Any]],
    candidate_ids: dict[str, int],
) -> torch.Tensor:
    input_ids = torch.tensor(
        [row["input_ids"] for row in rows], dtype=torch.long, device=device
    )
    attention = torch.ones_like(input_ids)
    handles = []
    counts = [0 for _ in patches]
    try:
        for patch_index, patch in enumerate(patches):
            positions = torch.tensor(
                [
                    row["role_positions"][patch["position_role"]]
                    for row in rows
                ],
                dtype=torch.long,
                device=device,
            )

            def make_hook(index, pos, vector, channels):
                def hook(module, args, output):
                    counts[index] += 1
                    return replace_tensor(output, pos, vector, channels)

                return hook

            handles.append(
                layers[patch["depth"] - 1].register_forward_hook(
                    make_hook(
                        patch_index,
                        positions,
                        patch["vectors"],
                        patch.get("channels"),
                    )
                )
            )
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
        candidates = candidate_tensor(output.logits[:, -1, :], candidate_ids).detach()
        if any(count != 1 for count in counts):
            raise RuntimeError(f"causal hook count drift: {counts}")
        del output, input_ids, attention
        return candidates
    finally:
        for handle in reversed(handles):
            handle.remove()


def patched_generate(
    model,
    layers,
    tokenizer,
    device,
    rows: list[dict[str, Any]],
    patches: list[dict[str, Any]],
    effective_eos: list[int],
    budget: int,
) -> list[dict[str, Any]]:
    input_ids = torch.tensor(
        [row["input_ids"] for row in rows], dtype=torch.long, device=device
    )
    attention = torch.ones_like(input_ids)
    handles = []
    counts = [0 for _ in patches]
    full_width = input_ids.shape[1]
    try:
        for patch_index, patch in enumerate(patches):
            positions = torch.tensor(
                [
                    row["role_positions"][patch["position_role"]]
                    for row in rows
                ],
                dtype=torch.long,
                device=device,
            )

            def make_hook(index, pos, vector, channels):
                def hook(module, args, output):
                    value = output[0] if isinstance(output, tuple) else output
                    if value.shape[1] != full_width:
                        return output
                    counts[index] += 1
                    return replace_tensor(output, pos, vector, channels)

                return hook

            handles.append(
                layers[patch["depth"] - 1].register_forward_hook(
                    make_hook(
                        patch_index,
                        positions,
                        patch["vectors"],
                        patch.get("channels"),
                    )
                )
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
        suffixes = generated.sequences[:, full_width:].detach().cpu().tolist()
        if any(count != 1 for count in counts):
            raise RuntimeError(f"generation hook count drift: {counts}")
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
        for handle in reversed(handles):
            handle.remove()


def channels_for(
    method: str, role: str, event_specs: dict[str, dict[str, Any]]
) -> list[int] | None:
    if method == "full":
        return None
    base = method
    if method in ("noop_difference_256", "wrong_position_difference_256"):
        base = "difference_256"
    return event_specs[role]["channels"][base]


def build_patch(
    role: str,
    method: str,
    event_specs: dict[str, dict[str, Any]],
    source_vectors: dict[str, torch.Tensor],
    target_vectors: dict[str, torch.Tensor],
) -> dict[str, Any]:
    spec = event_specs[role]
    if method == "noop_difference_256":
        vectors = target_vectors[role]
        position_role = spec["position_role"]
    else:
        vectors = source_vectors[role]
        position_role = (
            WRONG_POSITION_ROLE[role]
            if method == "wrong_position_difference_256"
            else spec["position_role"]
        )
    return {
        "depth": spec["depth"],
        "position_role": position_role,
        "vectors": vectors,
        "channels": channels_for(method, role, event_specs),
    }


def candidate_records(
    batch: list[dict[str, Any]],
    source_logits: torch.Tensor,
    target_logits: torch.Tensor,
    patched_logits: torch.Tensor,
    condition: str,
) -> list[dict[str, Any]]:
    color_index = {color: index for index, color in enumerate(COLORS)}
    rows = []
    for index, item in enumerate(batch):
        source_gold = item["source"]["gold"]
        target_gold = item["target"]["gold"]
        source_id = color_index[source_gold]
        target_id = color_index[target_gold]
        source_margin = float(
            source_logits[index, source_id] - source_logits[index, target_id]
        )
        target_margin = float(
            target_logits[index, source_id] - target_logits[index, target_id]
        )
        patched_margin = float(
            patched_logits[index, source_id] - patched_logits[index, target_id]
        )
        denominator = source_margin - target_margin
        transfer = (patched_margin - target_margin) / max(abs(denominator), 1e-8)
        unrelated = [
            color_index[color]
            for color in COLORS
            if color not in (source_gold, target_gold)
        ]
        collateral = float(
            torch.mean(
                torch.abs(
                    patched_logits[index, unrelated]
                    - target_logits[index, unrelated]
                )
            )
        )
        rows.append(
            {
                "schema_version": "phase998_causal_row.v1",
                "phase": PHASE,
                "model": MODEL,
                "partition": item["partition"],
                "pair_id": item["pair_id"],
                "direction": item["direction"],
                "condition": condition,
                "source_gold": source_gold,
                "target_gold": target_gold,
                "source_margin": source_margin,
                "target_margin": target_margin,
                "patched_margin": patched_margin,
                "delta_margin": patched_margin - target_margin,
                "normalized_transfer": transfer,
                "toward_source": abs(patched_margin - source_margin)
                < abs(target_margin - source_margin),
                "candidate_prediction": COLORS[
                    int(torch.argmax(patched_logits[index]).item())
                ],
                "candidate_flipped_to_source": int(
                    torch.argmax(patched_logits[index]).item()
                )
                == source_id,
                "unrelated_color_logit_change": collateral,
            }
        )
    return rows


def summarize_conditions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)
    result = {}
    for condition, values in sorted(grouped.items()):
        result[condition] = {
            "n": len(values),
            "mean_normalized_transfer": float(
                np.mean([row["normalized_transfer"] for row in values])
            ),
            "median_normalized_transfer": float(
                np.median([row["normalized_transfer"] for row in values])
            ),
            "toward_source_rate": float(
                np.mean([row["toward_source"] for row in values])
            ),
            "candidate_flip_rate": float(
                np.mean([row["candidate_flipped_to_source"] for row in values])
            ),
            "mean_unrelated_color_logit_change": float(
                np.mean([row["unrelated_color_logit_change"] for row in values])
            ),
        }
    return result


def summarize_mediation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["pair_id"], row["direction"])][row["condition"]] = row
    write_fractions = []
    read_fractions = []
    for values in grouped.values():
        write = values["write_difference_256"]
        write_block = values["write_plus_read_target_restore"]
        read = values["read_difference_256"]
        read_block = values["read_plus_decision_target_restore"]
        write_fractions.append(
            1.0
            - abs(write_block["delta_margin"])
            / max(abs(write["delta_margin"]), 1e-8)
        )
        read_fractions.append(
            1.0
            - abs(read_block["delta_margin"])
            / max(abs(read["delta_margin"]), 1e-8)
        )
    return {
        "n": len(grouped),
        "write_to_read_median_mediation_fraction": float(np.median(write_fractions)),
        "write_to_read_mean_mediation_fraction": float(np.mean(write_fractions)),
        "read_to_decision_median_mediation_fraction": float(np.median(read_fractions)),
        "read_to_decision_mean_mediation_fraction": float(np.mean(read_fractions)),
    }


def restoration_records(
    batch: list[dict[str, Any]],
    source_logits: torch.Tensor,
    corrupted_logits: torch.Tensor,
    restored_logits: torch.Tensor,
) -> list[dict[str, Any]]:
    color_index = {color: index for index, color in enumerate(COLORS)}
    rows = []
    for index, item in enumerate(batch):
        source_id = color_index[item["source"]["gold"]]
        target_id = color_index[item["target"]["gold"]]
        clean = float(source_logits[index, source_id] - source_logits[index, target_id])
        corrupt = float(
            corrupted_logits[index, source_id] - corrupted_logits[index, target_id]
        )
        restored = float(
            restored_logits[index, source_id] - restored_logits[index, target_id]
        )
        recovery = (restored - corrupt) / max(abs(clean - corrupt), 1e-8)
        rows.append(
            {
                "schema_version": "phase998_restoration_row.v1",
                "phase": PHASE,
                "model": MODEL,
                "partition": item["partition"],
                "pair_id": item["pair_id"],
                "direction": item["direction"],
                "clean_margin": clean,
                "corrupted_margin": corrupt,
                "restored_margin": restored,
                "corruption_delta": corrupt - clean,
                "restoration_delta": restored - corrupt,
                "recovery_fraction": recovery,
                "restored_prediction": COLORS[
                    int(torch.argmax(restored_logits[index]).item())
                ],
                "restored_to_source": int(torch.argmax(restored_logits[index]).item())
                == source_id,
            }
        )
    return rows


def run_candidate_and_restoration(
    model,
    layers,
    device,
    rows: list[dict[str, Any]],
    event_specs: dict[str, dict[str, Any]],
    candidate_ids: dict[str, int],
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    causal_rows: list[dict[str, Any]] = []
    restoration_rows: list[dict[str, Any]] = []
    batches = list(batches_by_template(rows, batch_size))
    for batch_index, batch in enumerate(batches):
        source_cases = [row["source"] for row in batch]
        target_cases = [row["target"] for row in batch]
        source_logits, source_vectors = capture_clean(
            model, device, source_cases, event_specs, candidate_ids
        )
        target_logits, target_vectors = capture_clean(
            model, device, target_cases, event_specs, candidate_ids
        )
        condition_logits: dict[str, torch.Tensor] = {}
        for role in ROLE_NAMES:
            for method in METHODS:
                condition = f"{role}_{method}"
                patch = build_patch(
                    role, method, event_specs, source_vectors, target_vectors
                )
                patched = patched_forward(
                    model, layers, device, target_cases, [patch], candidate_ids
                )
                condition_logits[condition] = patched
                causal_rows.extend(
                    candidate_records(
                        batch, source_logits, target_logits, patched, condition
                    )
                )

        write_source = build_patch(
            "write", "difference_256", event_specs, source_vectors, target_vectors
        )
        read_target_restore = {
            "depth": event_specs["read"]["depth"],
            "position_role": event_specs["read"]["position_role"],
            "vectors": target_vectors["read"],
            "channels": event_specs["read"]["channels"]["difference_256"],
        }
        write_blocked = patched_forward(
            model,
            layers,
            device,
            target_cases,
            [write_source, read_target_restore],
            candidate_ids,
        )
        condition = "write_plus_read_target_restore"
        condition_logits[condition] = write_blocked
        causal_rows.extend(
            candidate_records(
                batch, source_logits, target_logits, write_blocked, condition
            )
        )

        read_source = build_patch(
            "read", "difference_256", event_specs, source_vectors, target_vectors
        )
        decision_target_restore = {
            "depth": event_specs["decision"]["depth"],
            "position_role": event_specs["decision"]["position_role"],
            "vectors": target_vectors["decision"],
            "channels": event_specs["decision"]["channels"]["difference_256"],
        }
        read_blocked = patched_forward(
            model,
            layers,
            device,
            target_cases,
            [read_source, decision_target_restore],
            candidate_ids,
        )
        condition = "read_plus_decision_target_restore"
        condition_logits[condition] = read_blocked
        causal_rows.extend(
            candidate_records(
                batch, source_logits, target_logits, read_blocked, condition
            )
        )

        corrupt_write = {
            "depth": event_specs["write"]["depth"],
            "position_role": event_specs["write"]["position_role"],
            "vectors": target_vectors["write"],
            "channels": event_specs["write"]["channels"]["difference_256"],
        }
        corrupted = patched_forward(
            model, layers, device, source_cases, [corrupt_write], candidate_ids
        )
        restore_read = {
            "depth": event_specs["read"]["depth"],
            "position_role": event_specs["read"]["position_role"],
            "vectors": source_vectors["read"],
            "channels": event_specs["read"]["channels"]["difference_256"],
        }
        restored = patched_forward(
            model,
            layers,
            device,
            source_cases,
            [corrupt_write, restore_read],
            candidate_ids,
        )
        restoration_rows.extend(
            restoration_records(batch, source_logits, corrupted, restored)
        )

        del source_logits, target_logits, source_vectors, target_vectors
        for tensor in condition_logits.values():
            del tensor
        del corrupted, restored, write_blocked, read_blocked
        if (batch_index + 1) % 4 == 0 or batch_index + 1 == len(batches):
            print(
                f"[causal] {batch_index + 1}/{len(batches)} validation batches",
                flush=True,
            )
    return causal_rows, restoration_rows


def run_natural_holdout(
    model,
    layers,
    tokenizer,
    device,
    rows: list[dict[str, Any]],
    event_specs: dict[str, dict[str, Any]],
    candidate_ids: dict[str, int],
    effective_eos: list[int],
    batch_size: int,
    budget: int,
) -> list[dict[str, Any]]:
    conditions = [
        ("write_difference_256", [("write", "difference_256")]),
        ("write_top_activation_256", [("write", "top_activation_256")]),
        ("write_random_256", [("write", "random_256")]),
        ("write_noop_difference_256", [("write", "noop_difference_256")]),
        (
            "write_wrong_position_difference_256",
            [("write", "wrong_position_difference_256")],
        ),
        ("read_difference_256", [("read", "difference_256")]),
        ("read_top_activation_256", [("read", "top_activation_256")]),
        ("read_random_256", [("read", "random_256")]),
        ("decision_difference_256", [("decision", "difference_256")]),
        ("decision_top_activation_256", [("decision", "top_activation_256")]),
        ("decision_random_256", [("decision", "random_256")]),
    ]
    natural_rows = []
    batches = list(batches_by_template(rows, batch_size))
    for batch_index, batch in enumerate(batches):
        source_cases = [row["source"] for row in batch]
        target_cases = [row["target"] for row in batch]
        _, source_vectors = capture_clean(
            model, device, source_cases, event_specs, candidate_ids
        )
        _, target_vectors = capture_clean(
            model, device, target_cases, event_specs, candidate_ids
        )
        for condition, specifications in conditions:
            patches = [
                build_patch(
                    role, method, event_specs, source_vectors, target_vectors
                )
                for role, method in specifications
            ]
            generated = patched_generate(
                model,
                layers,
                tokenizer,
                device,
                target_cases,
                patches,
                effective_eos,
                budget,
            )
            for item, result in zip(batch, generated, strict=True):
                natural_rows.append(
                    {
                        "schema_version": "phase998_natural_causal_row.v1",
                        "phase": PHASE,
                        "model": MODEL,
                        "partition": item["partition"],
                        "pair_id": item["pair_id"],
                        "direction": item["direction"],
                        "condition": condition,
                        "source_gold": item["source"]["gold"],
                        "target_gold": item["target"]["gold"],
                        "prediction": result["prediction"],
                        "flipped_to_source": result["prediction"]
                        == item["source"]["gold"],
                        "remained_target": result["prediction"]
                        == item["target"]["gold"],
                        "eos_seen": result["eos_position"] is not None,
                        "exact_short": result["exact_short"],
                        "generated_text": result["text"],
                        "generated_suffix": result["suffix"],
                    }
                )
        del source_vectors, target_vectors
        if (batch_index + 1) % 2 == 0 or batch_index + 1 == len(batches):
            print(
                f"[natural] {batch_index + 1}/{len(batches)} holdout batches",
                flush=True,
            )
    return natural_rows


def natural_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)
    return {
        condition: {
            "n": len(values),
            "natural_flip_rate": float(
                np.mean([row["flipped_to_source"] for row in values])
            ),
            "target_retention_rate": float(
                np.mean([row["remained_target"] for row in values])
            ),
            "eos_rate": float(np.mean([row["eos_seen"] for row in values])),
            "exact_short_rate": float(
                np.mean([row["exact_short"] for row in values])
            ),
        }
        for condition, values in sorted(grouped.items())
    }


def final_gate(
    candidate: dict[str, Any],
    natural: dict[str, Any],
    restoration_rows: list[dict[str, Any]],
    mediation: dict[str, Any],
) -> tuple[dict[str, bool], dict[str, float]]:
    upstream_candidate = max(
        candidate["write_difference_256"]["candidate_flip_rate"],
        candidate["read_difference_256"]["candidate_flip_rate"],
    )
    upstream_natural = max(
        natural["write_difference_256"]["natural_flip_rate"],
        natural["read_difference_256"]["natural_flip_rate"],
    )
    control_flip = max(
        candidate["write_noop_difference_256"]["candidate_flip_rate"],
        candidate["write_wrong_position_difference_256"]["candidate_flip_rate"],
        candidate["write_random_256"]["candidate_flip_rate"],
        natural["write_noop_difference_256"]["natural_flip_rate"],
        natural["write_wrong_position_difference_256"]["natural_flip_rate"],
        natural["write_random_256"]["natural_flip_rate"],
    )
    restoration = float(
        np.median([row["recovery_fraction"] for row in restoration_rows])
    )
    diff_effect = max(
        abs(candidate["write_difference_256"]["mean_normalized_transfer"]),
        abs(candidate["read_difference_256"]["mean_normalized_transfer"]),
    )
    top_effect = max(
        abs(candidate["write_top_activation_256"]["mean_normalized_transfer"]),
        abs(candidate["read_top_activation_256"]["mean_normalized_transfer"]),
    )
    effect_ratio = diff_effect / max(top_effect, 1e-8)
    minimum_mediation = min(
        mediation["write_to_read_median_mediation_fraction"],
        mediation["read_to_decision_median_mediation_fraction"],
    )
    metrics = {
        "upstream_candidate_flip_rate": upstream_candidate,
        "upstream_natural_flip_rate": upstream_natural,
        "max_control_flip_rate": control_flip,
        "restoration_median_recovery": restoration,
        "difference_vs_top_activation_effect_ratio": effect_ratio,
        "minimum_mediation_fraction": minimum_mediation,
    }
    checks = {
        "upstream_candidate_flip_rate": upstream_candidate
        >= THRESHOLDS["upstream_candidate_flip_rate"],
        "upstream_natural_flip_rate": upstream_natural
        >= THRESHOLDS["upstream_natural_flip_rate"],
        "max_control_flip_rate": control_flip
        <= THRESHOLDS["max_control_flip_rate"],
        "restoration_median_recovery": restoration
        >= THRESHOLDS["restoration_median_recovery"],
        "difference_vs_top_activation_effect_ratio": effect_ratio
        >= THRESHOLDS["difference_vs_top_activation_effect_ratio"],
        "minimum_mediation_fraction": minimum_mediation
        >= THRESHOLDS["minimum_mediation_fraction"],
    }
    return checks, metrics


def run(batch_size: int, natural_budget: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 998 causal test requires CUDA")
    cases = read_jsonl(OUT_ROOT / "protocol" / "cases.jsonl")
    selected = read_jsonl(OUT_ROOT / "trace" / "selected_pairs.jsonl")
    trace_summary = json.loads(
        (OUT_ROOT / "trace" / "summary.json").read_text(encoding="utf-8")
    )
    channel_sets = json.loads(
        (OUT_ROOT / "trace" / "channel_sets.json").read_text(encoding="utf-8")
    )
    if not trace_summary["observation_gate_pass"]:
        raise RuntimeError("observation gate did not authorize causal testing")
    case_by_record = {row["record_id"]: row for row in cases}
    chain = trace_summary["selected_chain"]
    event_specs = {}
    for role in ROLE_NAMES:
        event = chain[role]
        metric = trace_summary["selected_event_metrics"][role]
        event_specs[role] = {
            "event": event,
            "depth": int(metric["depth"]),
            "position_role": metric["role"],
            "channels": channel_sets[event],
        }
    validation = directional_pairs(selected, case_by_record, "validation", True)
    holdout = directional_pairs(selected, case_by_record, "holdout", False)
    output_root = OUT_ROOT / "causal"
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model = tokenizer = None
    try:
        model, tokenizer, device = load_model(MODEL, dtype=torch.bfloat16, use_8bit=False)
        tokenizer.padding_side = "left"
        layers = get_layers(model)
        info = get_model_info(model, MODEL)
        candidate_ids = {
            color: int(
                json.loads(
                    (OUT_ROOT / "protocol" / "protocol.json").read_text(
                        encoding="utf-8"
                    )
                )["candidate_token_ids"][color]
            )
            for color in COLORS
        }
        effective_eos = eos_ids(model, tokenizer)
        causal_rows, restoration_rows = run_candidate_and_restoration(
            model,
            layers,
            device,
            validation,
            event_specs,
            candidate_ids,
            batch_size,
        )
        candidate = summarize_conditions(causal_rows)
        mediation = summarize_mediation(causal_rows)
        natural_rows = run_natural_holdout(
            model,
            layers,
            tokenizer,
            device,
            holdout,
            event_specs,
            candidate_ids,
            effective_eos,
            batch_size,
            natural_budget,
        )
        natural = natural_summary(natural_rows)
        checks, gate_metrics = final_gate(
            candidate, natural, restoration_rows, mediation
        )
        summary = {
            "schema_version": "phase998_causal_summary.v1",
            "phase": PHASE,
            "model": MODEL,
            "n_layers": info.n_layers,
            "d_model": info.d_model,
            "selected_chain": chain,
            "event_specs": event_specs,
            "validation_direction_count": len(validation),
            "holdout_pair_count": len(holdout),
            "candidate_condition_summary": candidate,
            "natural_condition_summary": natural,
            "mediation_summary": mediation,
            "restoration_summary": {
                "n": len(restoration_rows),
                "median_recovery_fraction": float(
                    np.median(
                        [row["recovery_fraction"] for row in restoration_rows]
                    )
                ),
                "mean_recovery_fraction": float(
                    np.mean(
                        [row["recovery_fraction"] for row in restoration_rows]
                    )
                ),
                "restored_to_source_rate": float(
                    np.mean([row["restored_to_source"] for row in restoration_rows])
                ),
            },
            "thresholds": THRESHOLDS,
            "gate_metrics": gate_metrics,
            "gate_checks": checks,
            "causal_thread_gate_pass": all(checks.values()),
            "elapsed_seconds": time.time() - started,
            "natural_max_new_tokens": natural_budget,
            "effective_eos_token_ids": effective_eos,
            "holdout_used_for_selection": False,
        }
        write_rows(output_root / "causal_rows.jsonl", causal_rows)
        write_rows(output_root / "restoration_rows.jsonl", restoration_rows)
        write_rows(output_root / "natural_rows.jsonl", natural_rows)
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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--natural-max-new-tokens", type=int, default=8)
    args = parser.parse_args()
    summary = run(args.batch_size, args.natural_max_new_tokens)
    print(
        json.dumps(
            {
                "passed": summary["causal_thread_gate_pass"],
                "gate_metrics": summary["gate_metrics"],
                "gate_checks": summary["gate_checks"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
