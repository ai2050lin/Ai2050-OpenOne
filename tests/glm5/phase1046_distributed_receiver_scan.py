#!/usr/bin/env python3
"""Run the Phase1046 distributed receiver-coalition atlas."""

from __future__ import annotations

import argparse
import json
import sys
import time
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
import phase1044_natural_recompute_trajectory_scan as trajectory_tools
import phase1045_receiver_mediation_scan as source_tools
import phase1046_distributed_receiver_protocol as protocol


TARGET_BATCH_SIZE = {"qwen3": 16, "glm4": 4, "deepseek7b": 4}


def chunks(rows: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def output_tensor(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def replace_output(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (hidden,) + output[1:]
    return hidden


class CoalitionSwap:
    def __init__(self, layer: Any, response_norms: np.memmap):
        self.layer = layer
        self.response_norms = response_norms
        self.positions: torch.Tensor | None = None
        self.masks: torch.Tensor | None = None
        self.full_sequence = False
        self.target_indices: np.ndarray | None = None
        self.depth_slot = -1
        self.mask_slot = -1
        self.count = 0
        self.handle = None

    def begin(
        self,
        positions: torch.Tensor,
        masks: torch.Tensor,
        full_sequence: bool,
        target_indices: np.ndarray,
        depth_slot: int,
        mask_slot: int,
    ) -> None:
        if len(positions) != 2 * len(target_indices):
            raise RuntimeError("coalition pair batch drift")
        self.positions = positions
        self.masks = masks
        self.full_sequence = full_sequence
        self.target_indices = target_indices
        self.depth_slot = depth_slot
        self.mask_slot = mask_slot
        self.count = 0

    def _hook(self, module, args, output):
        if (
            self.positions is None
            or self.masks is None
            or self.target_indices is None
        ):
            raise RuntimeError("coalition swap context missing")
        hidden = output_tensor(output)
        even = torch.arange(
            0, hidden.shape[0], 2, device=hidden.device
        )
        odd = even + 1
        patched = hidden.clone()
        if self.full_sequence:
            difference = hidden[even].float() - hidden[odd].float()
            norms = torch.linalg.vector_norm(
                difference.flatten(start_dim=1), dim=-1
            ) / torch.sqrt(
                torch.full(
                    (len(even),),
                    hidden.shape[1],
                    device=hidden.device,
                    dtype=torch.float32,
                )
            )
            patched[even] = hidden[odd]
            patched[odd] = hidden[even]
        else:
            positions = self.positions.to(hidden.device)
            masks = self.masks.to(hidden.device)
            squared = torch.zeros(
                len(even), device=hidden.device, dtype=torch.float32
            )
            token_count = torch.zeros_like(squared)
            for span_slot in range(protocol.MAX_COALITION_TOKENS):
                active = torch.where(masks[0::2, span_slot])[0]
                if len(active) == 0:
                    continue
                even_rows = even[active]
                odd_rows = odd[active]
                even_pos = positions[even_rows, span_slot]
                odd_pos = positions[odd_rows, span_slot]
                even_value = hidden[even_rows, even_pos, :].clone()
                odd_value = hidden[odd_rows, odd_pos, :].clone()
                squared[active] += torch.sum(
                    (even_value.float() - odd_value.float()) ** 2,
                    dim=-1,
                )
                token_count[active] += 1.0
                patched[even_rows, even_pos, :] = odd_value
                patched[odd_rows, odd_pos, :] = even_value
            norms = torch.sqrt(
                squared / torch.clamp(token_count, min=1.0)
            )
        self.response_norms[
            self.target_indices, self.depth_slot, self.mask_slot
        ] = norms.cpu().numpy()
        self.count += 1
        return replace_output(output, patched)

    def register(self) -> None:
        self.handle = self.layer.register_forward_hook(self._hook)

    def end(self) -> None:
        if self.count != 1:
            raise RuntimeError(
                f"coalition swap hook count drift: {self.count}"
            )
        self.positions = None
        self.masks = None
        self.full_sequence = False
        self.target_indices = None
        self.depth_slot = -1
        self.mask_slot = -1

    def close(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def coalition_positions(
    target_rows: list[dict[str, Any]],
    mask_name: str,
    cases: dict[int, dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    size = 2 * len(target_rows)
    positions = torch.zeros(
        (size, protocol.MAX_COALITION_TOKENS), dtype=torch.long
    )
    masks = torch.zeros_like(positions, dtype=torch.bool)
    if mask_name == "full_sequence_reference":
        return positions, masks, True
    sites = protocol.COALITION_MASKS[mask_name]
    for target_slot, target in enumerate(target_rows):
        row = cases[int(target["target_case_index"])]
        active_positions = []
        for site in sites:
            role = protocol.semantic_role(site, target)
            start, end = (
                int(value) for value in row["anchor_spans"][role]
            )
            active_positions.extend(range(start, end + 1))
        if (
            len(active_positions) != len(set(active_positions))
            or len(active_positions) > protocol.MAX_COALITION_TOKENS
        ):
            raise RuntimeError(
                f"invalid coalition {mask_name} for "
                f"{target['target_index']}"
            )
        for pair_slot in (2 * target_slot, 2 * target_slot + 1):
            positions[
                pair_slot, :len(active_positions)
            ] = torch.tensor(active_positions, dtype=torch.long)
            masks[pair_slot, :len(active_positions)] = True
    return positions, masks, False


def margin_values(
    logits: np.ndarray,
    targets: list[dict[str, Any]],
) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32)
    rows = np.arange(len(targets), dtype=np.int64)
    target_index = np.asarray(
        [int(row["target_family_index"]) for row in targets],
        dtype=np.int64,
    )
    cross_index = np.asarray(
        [int(row["cross_family_index"]) for row in targets],
        dtype=np.int64,
    )
    return values[rows, cross_index] - values[rows, target_index]


def ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.float32)
    denominator = np.asarray(denominator, dtype=np.float32)
    result = np.full(len(numerator), np.nan, dtype=np.float32)
    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator > 1e-8)
    )
    result[valid] = numerator[valid] / denominator[valid]
    return result


def cell_metrics(
    paired_logits: np.ndarray,
    baseline_logits: np.ndarray,
    targets: list[dict[str, Any]],
    prereg: dict[str, Any],
) -> list[dict[str, Any]]:
    source_margin = margin_values(baseline_logits[:, 0, :], targets)
    zero_margin = margin_values(baseline_logits[:, 1, :], targets)
    source_shift = source_margin - zero_margin
    gate = prereg["discovery_gate"]
    rows = []
    for depth_slot, relative_depth in enumerate(
        protocol.RELATIVE_DEPTH_SLOTS
    ):
        for mask_slot, mask_name in enumerate(
            protocol.COALITION_MASKS
        ):
            reset_margin = margin_values(
                paired_logits[:, depth_slot, mask_slot, 0, :],
                targets,
            )
            replay_margin = margin_values(
                paired_logits[:, depth_slot, mask_slot, 1, :],
                targets,
            )
            reset_shift = reset_margin - zero_margin
            replay_shift = replay_margin - zero_margin
            blocked = source_shift - reset_shift
            mediation = ratio(blocked, source_shift)
            recovery = ratio(replay_shift, source_shift)
            summaries = {
                "source_shift": trajectory_tools.scalar_summary(
                    source_shift
                ),
                "reset_shift": trajectory_tools.scalar_summary(
                    reset_shift
                ),
                "replay_shift": trajectory_tools.scalar_summary(
                    replay_shift
                ),
                "blocked_amount": trajectory_tools.scalar_summary(
                    blocked
                ),
                "mediation_fraction": trajectory_tools.scalar_summary(
                    mediation
                ),
                "replay_recovery": trajectory_tools.scalar_summary(
                    recovery
                ),
            }
            if mask_name == "full_sequence_reference":
                passed = (
                    summaries["mediation_fraction"]["median"]
                    >= gate["full_sequence_reset_fraction_min"]
                    and summaries["replay_recovery"]["median"]
                    >= gate["full_sequence_replay_fraction_min"]
                )
                gate_name = "reference_gate"
            else:
                passed = (
                    summaries["source_shift"]["positive_rate"]
                    >= gate["source_shift_positive_rate_min"]
                    and summaries["blocked_amount"]["positive_rate"]
                    >= gate["blocked_positive_rate_min"]
                    and summaries["mediation_fraction"]["median"]
                    >= gate["mediation_fraction_median_min"]
                    and summaries["replay_shift"]["positive_rate"]
                    >= gate["replay_positive_rate_min"]
                    and summaries["replay_recovery"]["median"]
                    >= gate["replay_recovery_median_min"]
                )
                gate_name = "discovery_gate"
            rows.append({
                "relative_depth_slot": relative_depth,
                "coalition_mask": mask_name,
                **summaries,
                gate_name: bool(passed),
            })
    return rows


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1046 protocol audit failed")
    targets = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / "discovery_targets.jsonl"
    )
    all_source_targets = protocol.read_jsonl(
        protocol.SOURCE_ROOT / "protocol" / "targets.jsonl"
    )
    targets_by_index = {
        int(row["target_index"]): row for row in all_source_targets
    }
    cases_list = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl"
    )
    cases = {int(row["case_index"]): row for row in cases_list}
    case_to_local = {
        int(row["case_index"]): index
        for index, row in enumerate(cases_list)
    }
    source_depth = int(
        prereg["model_depths"][model_name]["source_depth"]
    )
    receiver_depths = {
        int(slot): int(depth)
        for slot, depth in prereg["model_depths"][model_name][
            "receiver_depths"
        ].items()
    }
    source_atlas = protocol.SOURCE_ROOT / "atlas" / model_name
    source_cache = np.load(
        source_atlas / "source_states.fp16.npy", mmap_mode="r"
    )
    source_baseline = np.load(
        source_atlas / "paired_candidate_logits.fp32.npy",
        mmap_mode="r",
    )
    confirmation_indices = np.asarray(
        [int(row["confirmation_index"]) for row in targets],
        dtype=np.int64,
    )
    inherited_baseline_logits = np.asarray(
        source_baseline[confirmation_indices, 0, 0, :, :],
        dtype=np.float32,
    )

    atlas_dir = protocol.OUT_ROOT / "atlas" / model_name
    atlas_dir.mkdir(parents=True, exist_ok=True)
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
        pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        candidate_ids = torch.tensor(
            cases_list[0]["candidate_token_ids"], dtype=torch.long
        )

        paired_logits = np.lib.format.open_memmap(
            atlas_dir / "paired_candidate_logits.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(
                len(targets),
                len(protocol.RELATIVE_DEPTH_SLOTS),
                len(protocol.COALITION_MASKS),
                2,
                len(protocol.material.FAMILIES),
            ),
        )
        response_norms = np.lib.format.open_memmap(
            atlas_dir / "coalition_response_norms.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(
                len(targets),
                len(protocol.RELATIVE_DEPTH_SLOTS),
                len(protocol.COALITION_MASKS),
            ),
        )
        baseline_logits = np.lib.format.open_memmap(
            atlas_dir / "paired_baseline_logits.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(
                len(targets),
                2,
                len(protocol.material.FAMILIES),
            ),
        )
        paired_logits[:] = np.nan
        response_norms[:] = np.nan
        baseline_logits[:] = np.nan

        source_patch = trajectory_tools.SourcePatch(
            layers[source_depth - 1]
        )
        source_patch.register()
        try:
            for target_batch in chunks(
                targets, TARGET_BATCH_SIZE[model_name]
            ):
                (
                    input_ids,
                    attention_mask,
                    pre_positions,
                    source_positions,
                    source_masks,
                    payloads,
                    _,
                    _,
                    _,
                    _,
                ) = source_tools.make_paired_batch(
                    target_batch,
                    "cross_selected",
                    "none",
                    targets_by_index,
                    cases,
                    case_to_local,
                    source_cache,
                    pad_token_id=pad_token_id,
                    device=device,
                )
                target_indices = np.asarray(
                    [
                        int(row["coalition_index"])
                        for row in target_batch
                    ],
                    dtype=np.int64,
                )
                source_patch.begin(
                    source_positions, source_masks, payloads
                )
                with torch.inference_mode():
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                source_patch.end()
                logits = output.logits
                batch = torch.arange(
                    logits.shape[0], device=logits.device
                )
                selected = logits[
                    batch,
                    pre_positions.to(logits.device),
                    :,
                ].float().index_select(
                    -1, candidate_ids.to(logits.device)
                )
                pair = selected.reshape(
                    len(target_batch),
                    2,
                    len(protocol.material.FAMILIES),
                )
                baseline_logits[target_indices] = (
                    pair.detach().cpu().numpy()
                )
                del output, logits, selected, pair
            for depth_slot, relative_depth in enumerate(
                protocol.RELATIVE_DEPTH_SLOTS
            ):
                receiver_depth = receiver_depths[relative_depth]
                coalition_swap = CoalitionSwap(
                    layers[receiver_depth - 1], response_norms
                )
                coalition_swap.register()
                try:
                    for mask_slot, mask_name in enumerate(
                        protocol.COALITION_MASKS
                    ):
                        for target_batch in chunks(
                            targets, TARGET_BATCH_SIZE[model_name]
                        ):
                            (
                                input_ids,
                                attention_mask,
                                pre_positions,
                                source_positions,
                                source_masks,
                                payloads,
                                _,
                                _,
                                _,
                                _,
                            ) = source_tools.make_paired_batch(
                                target_batch,
                                "cross_selected",
                                "none",
                                targets_by_index,
                                cases,
                                case_to_local,
                                source_cache,
                                pad_token_id=pad_token_id,
                                device=device,
                            )
                            (
                                coalition_pos,
                                coalition_mask,
                                full_sequence,
                            ) = coalition_positions(
                                target_batch, mask_name, cases
                            )
                            target_indices = np.asarray(
                                [
                                    int(row["coalition_index"])
                                    for row in target_batch
                                ],
                                dtype=np.int64,
                            )
                            source_patch.begin(
                                source_positions, source_masks, payloads
                            )
                            coalition_swap.begin(
                                coalition_pos,
                                coalition_mask,
                                full_sequence,
                                target_indices,
                                depth_slot,
                                mask_slot,
                            )
                            with torch.inference_mode():
                                output = model(
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    use_cache=False,
                                    return_dict=True,
                                )
                            coalition_swap.end()
                            source_patch.end()
                            logits = output.logits
                            batch = torch.arange(
                                logits.shape[0], device=logits.device
                            )
                            selected = logits[
                                batch,
                                pre_positions.to(logits.device),
                                :,
                            ].float().index_select(
                                -1, candidate_ids.to(logits.device)
                            )
                            pair = selected.reshape(
                                len(target_batch),
                                2,
                                len(protocol.material.FAMILIES),
                            )
                            paired_logits[
                                target_indices,
                                depth_slot,
                                mask_slot,
                                :,
                                :,
                            ] = pair.detach().cpu().numpy()
                            del output, logits, selected, pair
                finally:
                    coalition_swap.close()
        finally:
            source_patch.close()
        paired_logits.flush()
        response_norms.flush()
        baseline_logits.flush()

        cells = cell_metrics(
            paired_logits, baseline_logits, targets, prereg
        )
        protocol.write_jsonl(atlas_dir / "coalition_cells.jsonl", cells)
        candidate_cells = [
            row for row in cells
            if row.get("discovery_gate", False)
        ]
        reference_cells = [
            row for row in cells
            if row["coalition_mask"] == "full_sequence_reference"
        ]
        summary = {
            "schema_version": "phase1046_model_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "precision": precision,
            "placement": placement,
            "model_info": {
                "n_layers": info.n_layers,
                "d_model": info.d_model,
                "model_class": info.model_class,
            },
            "source_depth": source_depth,
            "receiver_depths": receiver_depths,
            "baseline_logits_finite": (
                trajectory_tools.finite_summary(baseline_logits)
            ),
            "baseline_vs_phase1045_max_abs": float(
                np.max(
                    np.abs(
                        np.asarray(baseline_logits, dtype=np.float32)
                        - inherited_baseline_logits
                    )
                )
            ),
            "paired_logits_finite": (
                trajectory_tools.finite_summary(paired_logits)
            ),
            "response_norms_finite": (
                trajectory_tools.finite_summary(response_norms)
            ),
            "candidate_cell_count": len(candidate_cells),
            "candidate_cells": candidate_cells,
            "reference_cells": reference_cells,
            "elapsed_seconds": float(time.time() - started),
        }
        protocol.write_json(atlas_dir / "summary.json", summary)
        print(json.dumps({
            "model": model_name,
            "candidate_cell_count": len(candidate_cells),
            "candidate_cells": [
                (
                    row["relative_depth_slot"],
                    row["coalition_mask"],
                    row["mediation_fraction"]["median"],
                    row["replay_recovery"]["median"],
                )
                for row in candidate_cells
            ],
            "reference": [
                (
                    row["relative_depth_slot"],
                    row["mediation_fraction"]["median"],
                    row["replay_recovery"]["median"],
                    row["reference_gate"],
                )
                for row in reference_cells
            ],
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
