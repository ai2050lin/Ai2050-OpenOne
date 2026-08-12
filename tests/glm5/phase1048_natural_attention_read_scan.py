#!/usr/bin/env python3
"""Measure natural selected-source Attention and A-times-V contributions."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, get_model_info
from phase1023_fp16_utils import (
    load_fp16,
    quantization_audit,
    release_fp16,
)
import phase1044_natural_recompute_trajectory_scan as common
import phase1048_natural_attention_read_protocol as protocol


BATCH_SIZE = {"qwen3": 8, "glm4": 2, "deepseek7b": 2}


def chunks(rows: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def output_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"unsupported projection output {type(output)!r}")


def attention_tensor(output: Any) -> torch.Tensor:
    if not isinstance(output, (tuple, list)):
        raise TypeError("self-attention did not return a tuple")
    candidates = [
        value
        for value in output
        if torch.is_tensor(value) and value.ndim == 4
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one attention tensor, found {len(candidates)}"
        )
    return candidates[0]


def positions_for_batch(
    targets: list[dict[str, Any]],
    cases: dict[int, dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source_positions = torch.zeros(
        (len(targets), len(protocol.SOURCES), protocol.MAX_SOURCE_SPAN),
        dtype=torch.long,
    )
    source_masks = torch.zeros_like(source_positions, dtype=torch.bool)
    destination_positions = torch.zeros(
        (len(targets), len(protocol.DESTINATIONS)),
        dtype=torch.long,
    )
    for row_slot, target in enumerate(targets):
        case = cases[int(target["target_case_index"])]
        for source_slot, site in enumerate(protocol.SOURCES):
            role = protocol.semantic_role(site, target)
            start, end = (
                int(value) for value in case["anchor_spans"][role]
            )
            active = list(range(start, end + 1))
            source_positions[row_slot, source_slot, :len(active)] = (
                torch.tensor(active, dtype=torch.long)
            )
            source_masks[row_slot, source_slot, :len(active)] = True
        for destination_slot, site in enumerate(protocol.DESTINATIONS):
            start, end = (
                int(value) for value in case["anchor_spans"][site]
            )
            destination_positions[row_slot, destination_slot] = end
    return source_positions, source_masks, destination_positions


def make_batch(
    targets: list[dict[str, Any]],
    cases: dict[int, dict[str, Any]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = [cases[int(target["target_case_index"])] for target in targets]
    max_length = max(len(row["input_ids"]) for row in rows)
    input_ids = torch.full(
        (len(rows), max_length),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    pre_positions = torch.zeros(len(rows), dtype=torch.long, device=device)
    for slot, row in enumerate(rows):
        values = torch.tensor(row["input_ids"], dtype=torch.long, device=device)
        length = len(values)
        input_ids[slot, :length] = values
        attention_mask[slot, :length] = 1
        pre_positions[slot] = int(row["anchor_spans"]["pre_output"][1])
    return input_ids, attention_mask, pre_positions


class NaturalReadCapture:
    def __init__(
        self,
        layers,
        attention_mass: np.ndarray,
        av_norm: np.ndarray,
    ) -> None:
        self.layers = layers
        self.attention_mass = attention_mass
        self.av_norm = av_norm
        self.handles = []
        self.v_cache: dict[int, torch.Tensor] = {}
        self.source_positions: torch.Tensor | None = None
        self.source_masks: torch.Tensor | None = None
        self.destination_positions: torch.Tensor | None = None
        self.target_indices: np.ndarray | None = None

    def register(self) -> None:
        for depth, layer in enumerate(self.layers, start=1):
            self.handles.append(
                layer.self_attn.v_proj.register_forward_hook(
                    self._v_hook(depth)
                )
            )
            self.handles.append(
                layer.self_attn.register_forward_hook(
                    self._attention_hook(depth)
                )
            )

    def begin(
        self,
        source_positions: torch.Tensor,
        source_masks: torch.Tensor,
        destination_positions: torch.Tensor,
        target_indices: np.ndarray,
    ) -> None:
        self.source_positions = source_positions
        self.source_masks = source_masks
        self.destination_positions = destination_positions
        self.target_indices = target_indices
        self.v_cache.clear()

    def end(self) -> None:
        if self.v_cache:
            raise RuntimeError(
                f"unconsumed V caches: {sorted(self.v_cache)}"
            )
        self.source_positions = None
        self.source_masks = None
        self.destination_positions = None
        self.target_indices = None

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _v_hook(self, depth: int):
        def hook(_module, _inputs, output):
            self.v_cache[depth] = output_tensor(output)
        return hook

    def _attention_hook(self, depth: int):
        def hook(module, _inputs, output):
            if (
                self.source_positions is None
                or self.source_masks is None
                or self.destination_positions is None
                or self.target_indices is None
            ):
                raise RuntimeError("capture hook fired outside an active batch")
            values = self.v_cache.pop(depth)
            attention = attention_tensor(output).float()
            batch_size, n_heads, _, _ = attention.shape
            if values.shape[0] != batch_size:
                raise RuntimeError("V/attention batch mismatch")
            head_dim = int(getattr(module, "head_dim", 0))
            if head_dim <= 0:
                head_dim = int(values.shape[-1] // getattr(module, "num_key_value_heads"))
            n_kv_heads = int(values.shape[-1] // head_dim)
            if n_heads % n_kv_heads:
                raise RuntimeError("query/KV head grouping drift")
            values = values.reshape(
                batch_size,
                values.shape[1],
                n_kv_heads,
                head_dim,
            ).float()
            head_to_kv = (
                torch.arange(n_heads, device=values.device)
                // (n_heads // n_kv_heads)
            )
            source_positions = self.source_positions
            source_masks = self.source_masks
            destination_positions = self.destination_positions
            for batch_slot in range(batch_size):
                target_index = int(self.target_indices[batch_slot])
                for destination_slot in range(len(protocol.DESTINATIONS)):
                    destination = int(
                        destination_positions[
                            batch_slot, destination_slot
                        ]
                    )
                    for source_slot in range(len(protocol.SOURCES)):
                        valid = source_masks[
                            batch_slot, source_slot
                        ].nonzero(as_tuple=False).flatten()
                        positions = source_positions[
                            batch_slot, source_slot, valid
                        ].tolist()
                        weights = attention[
                            batch_slot,
                            :,
                            destination,
                            positions,
                        ]
                        if weights.ndim == 1:
                            weights = weights[:, None]
                        selected_values = values[
                            batch_slot, positions, :, :
                        ][:, head_to_kv, :].permute(1, 0, 2)
                        contribution = (
                            weights.to(selected_values.device)[..., None]
                            * selected_values
                        ).sum(dim=1)
                        self.attention_mass[
                            target_index,
                            depth - 1,
                            :n_heads,
                            destination_slot,
                            source_slot,
                        ] = (
                            weights.sum(dim=-1)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        self.av_norm[
                            target_index,
                            depth - 1,
                            :n_heads,
                            destination_slot,
                            source_slot,
                        ] = (
                            contribution.norm(dim=-1)
                            .detach()
                            .cpu()
                            .numpy()
                        )
        return hook


def pair_rows(
    targets: list[dict[str, Any]],
) -> list[tuple[int, int]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in targets:
        grouped[int(row["query_pair_index"])].append(row)
    result = []
    for pair_index in sorted(grouped):
        current = sorted(
            grouped[pair_index], key=lambda row: int(row["query"])
        )
        if len(current) != 2:
            raise RuntimeError("query pair drift")
        result.append(
            (int(current[0]["atlas_index"]), int(current[1]["atlas_index"]))
        )
    return result


def finite_scalar_summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {
            "count": 0,
            "finite_rate": 0.0,
            "median": None,
            "mean": None,
            "positive_rate": None,
        }
    return {
        "count": int(len(finite)),
        "finite_rate": float(len(finite) / len(values)),
        "median": float(np.median(finite)),
        "mean": float(np.mean(finite)),
        "positive_rate": float(np.mean(finite > 0)),
    }


def analyze(
    attention_mass: np.ndarray,
    av_norm: np.ndarray,
    targets: list[dict[str, Any]],
    prereg: dict[str, Any],
    model_name: str,
) -> dict[str, Any]:
    pairs = pair_rows(targets)
    gate = prereg["descriptive_head_gate"]
    layers = prereg["model_info"][model_name]["n_layers"]
    n_heads = attention_mass.shape[2]
    cells = []
    for depth in range(1, layers + 1):
        for head in range(n_heads):
            for destination_slot, destination in enumerate(
                protocol.DESTINATIONS
            ):
                mass_advantage = (
                    attention_mass[
                        :, depth - 1, head, destination_slot, 0
                    ]
                    - attention_mass[
                        :, depth - 1, head, destination_slot, 1
                    ]
                )
                log_ratio = np.log(
                    (
                        av_norm[
                            :, depth - 1, head, destination_slot, 0
                        ]
                        + 1e-8
                    )
                    /
                    (
                        av_norm[
                            :, depth - 1, head, destination_slot, 1
                        ]
                        + 1e-8
                    )
                )
                pair_min_mass = np.asarray([
                    min(mass_advantage[left], mass_advantage[right])
                    for left, right in pairs
                ])
                pair_min_log_ratio = np.asarray([
                    min(log_ratio[left], log_ratio[right])
                    for left, right in pairs
                ])
                mass_summary = finite_scalar_summary(pair_min_mass)
                av_summary = finite_scalar_summary(pair_min_log_ratio)
                passed = (
                    mass_summary["finite_rate"]
                    >= gate["minimum_finite_pair_rate"]
                    and av_summary["finite_rate"]
                    >= gate["minimum_finite_pair_rate"]
                    and mass_summary["median"]
                    > gate[
                        "pair_min_attention_advantage_median_min"
                    ]
                    and mass_summary["positive_rate"]
                    >= gate[
                        "pair_min_attention_advantage_positive_rate_min"
                    ]
                    and av_summary["median"]
                    >= gate["pair_min_av_log_ratio_median_min"]
                    and av_summary["positive_rate"]
                    >= gate[
                        "pair_min_av_ratio_positive_rate_min"
                    ]
                )
                score = None
                if (
                    mass_summary["median"] is not None
                    and av_summary["median"] is not None
                ):
                    score = float(
                        mass_summary["median"]
                        * max(av_summary["median"], 0.0)
                    )
                cells.append({
                    "depth": depth,
                    "head": head,
                    "destination": destination,
                    "pair_min_attention_advantage": mass_summary,
                    "pair_min_av_log_ratio": av_summary,
                    "median_av_ratio": (
                        None
                        if av_summary["median"] is None
                        else float(math.exp(av_summary["median"]))
                    ),
                    "score": score,
                    "descriptive_gate_passed": bool(passed),
                })

    bands = []
    depth_bands = prereg["model_info"][model_name]["read_depth_bands"]
    for slot_text, depths in depth_bands.items():
        for destination in protocol.DESTINATIONS:
            selected = [
                row for row in cells
                if row["depth"] in depths
                and row["destination"] == destination
                and row["descriptive_gate_passed"]
            ]
            scores = [
                float(row["score"])
                for row in selected
                if row["score"] is not None
            ]
            bands.append({
                "normalized_read_slot": int(slot_text),
                "depths": depths,
                "destination": destination,
                "passing_head_cells": len(selected),
                "passing_unique_heads": len({
                    int(row["head"]) for row in selected
                }),
                "maximum_score": max(scores) if scores else None,
                "median_score": (
                    float(np.median(scores)) if scores else None
                ),
                "top_cells": sorted(
                    selected,
                    key=lambda row: float(row["score"] or 0.0),
                    reverse=True,
                )[:12],
            })
    return {
        "query_pair_count": len(pairs),
        "cell_count": len(cells),
        "passing_cell_count": sum(
            row["descriptive_gate_passed"] for row in cells
        ),
        "cells": cells,
        "bands": bands,
    }


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1048 protocol audit failed")
    targets = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / "discovery_targets.jsonl"
    )
    cases_list = protocol.read_jsonl(
        protocol.OUT_ROOT
        / "protocol"
        / f"cases.{model_name}.jsonl"
    )
    cases = {int(row["case_index"]): row for row in cases_list}
    out_dir = protocol.OUT_ROOT / "atlas" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model = tokenizer = None
    try:
        model, tokenizer, device, placement = load_fp16(model_name)
        precision = quantization_audit(model)
        if (
            precision["has_quantized_modules"]
            or precision["has_bf16_parameters"]
            or not precision["has_fp16_parameters"]
        ):
            raise RuntimeError("FP16/no-quantization audit failed")
        layers = get_layers(model)
        info = get_model_info(model, model_name)
        n_heads = int(model.config.num_attention_heads)
        if info.n_layers != int(
            prereg["model_info"][model_name]["n_layers"]
        ):
            raise RuntimeError("model layer count drift")
        pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        candidate_ids = torch.tensor(
            cases_list[0]["candidate_token_ids"], dtype=torch.long
        )
        attention_mass = np.lib.format.open_memmap(
            out_dir / "attention_mass.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(
                len(targets),
                info.n_layers,
                n_heads,
                len(protocol.DESTINATIONS),
                len(protocol.SOURCES),
            ),
        )
        av_norm = np.lib.format.open_memmap(
            out_dir / "av_contribution_norm.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=attention_mass.shape,
        )
        candidate_logits = np.lib.format.open_memmap(
            out_dir / "candidate_logits.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(targets), len(protocol.material.FAMILIES)),
        )
        full_vocab_top1 = np.lib.format.open_memmap(
            out_dir / "full_vocab_top1.int32.npy",
            mode="w+",
            dtype=np.int32,
            shape=(len(targets),),
        )
        attention_mass[:] = np.nan
        av_norm[:] = np.nan
        candidate_logits[:] = np.nan
        full_vocab_top1[:] = -1
        capture = NaturalReadCapture(layers, attention_mass, av_norm)
        capture.register()
        try:
            for target_batch in chunks(
                targets, BATCH_SIZE[model_name]
            ):
                input_ids, attention_mask, pre_positions = make_batch(
                    target_batch,
                    cases,
                    pad_token_id=pad_token_id,
                    device=device,
                )
                source_positions, source_masks, destination_positions = (
                    positions_for_batch(target_batch, cases)
                )
                target_indices = np.asarray(
                    [int(row["atlas_index"]) for row in target_batch],
                    dtype=np.int64,
                )
                capture.begin(
                    source_positions,
                    source_masks,
                    destination_positions,
                    target_indices,
                )
                with torch.inference_mode():
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        output_attentions=True,
                        return_dict=True,
                    )
                capture.end()
                logits = output.logits
                batch = torch.arange(
                    logits.shape[0], device=logits.device
                )
                selected_logits = logits[
                    batch, pre_positions.to(logits.device), :
                ]
                candidate_logits[target_indices] = (
                    selected_logits.float()
                    .index_select(-1, candidate_ids.to(logits.device))
                    .detach()
                    .cpu()
                    .numpy()
                )
                full_vocab_top1[target_indices] = (
                    selected_logits.argmax(dim=-1)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.int32)
                )
                del output, logits, selected_logits
        finally:
            capture.close()
        for array in (
            attention_mass,
            av_norm,
            candidate_logits,
            full_vocab_top1,
        ):
            array.flush()

        analysis = analyze(
            attention_mass,
            av_norm,
            targets,
            prereg,
            model_name,
        )
        expected = np.asarray(
            [int(row["target_family_index"]) for row in targets],
            dtype=np.int64,
        )
        finite_rows = np.isfinite(candidate_logits).all(axis=1)
        behavior_accuracy = float(np.mean(
            np.argmax(candidate_logits[finite_rows], axis=1)
            == expected[finite_rows]
        )) if finite_rows.any() else None
        summary = {
            "schema_version": "phase1048_model_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "precision": precision,
            "placement": placement,
            "model_info": {
                "n_layers": info.n_layers,
                "d_model": info.d_model,
                "n_heads": n_heads,
                "n_kv_heads": int(model.config.num_key_value_heads),
                "model_class": info.model_class,
            },
            "behavior": {
                "finite_candidate_row_rate": float(np.mean(finite_rows)),
                "candidate_accuracy": behavior_accuracy,
            },
            "finite_audits": {
                "attention_mass": common.finite_summary(attention_mass),
                "av_contribution_norm": common.finite_summary(av_norm),
                "candidate_logits": common.finite_summary(candidate_logits),
            },
            "analysis": analysis,
            "elapsed_seconds": float(time.time() - started),
        }
        protocol.write_json(out_dir / "summary.json", summary)
        print(json.dumps({
            "model": model_name,
            "candidate_accuracy": behavior_accuracy,
            "passing_cells": analysis["passing_cell_count"],
            "top_bands": sorted(
                [
                    row for row in analysis["bands"]
                    if row["passing_head_cells"]
                ],
                key=lambda row: float(row["maximum_score"] or 0.0),
                reverse=True,
            )[:5],
            "elapsed_seconds": summary["elapsed_seconds"],
        }, ensure_ascii=False), flush=True)
    finally:
        if model is not None:
            release_fp16(model)
        del tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, choices=protocol.MODELS
    )
    args = parser.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
