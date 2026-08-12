#!/usr/bin/env python3
"""Run independent Phase1047 concept-pair constituent confirmation."""

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
import phase1046_distributed_receiver_scan as coalition_tools
import phase1047_concept_pair_confirmation_protocol as protocol


CLEAN_BATCH_SIZE = {"qwen3": 32, "glm4": 8, "deepseek7b": 8}
TARGET_BATCH_SIZE = {"qwen3": 16, "glm4": 4, "deepseek7b": 4}


def chunks(rows: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def coalition_positions(
    target_rows: list[dict[str, Any]],
    mask_name: str,
    cases: dict[int, dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    size = 2 * len(target_rows)
    positions = torch.zeros(
        (
            size,
            coalition_tools.protocol.MAX_COALITION_TOKENS,
        ),
        dtype=torch.long,
    )
    masks = torch.zeros_like(positions, dtype=torch.bool)
    if mask_name == "full_sequence_reference":
        return positions, masks, True
    for target_slot, target in enumerate(target_rows):
        row = cases[int(target["target_case_index"])]
        active = []
        for site in protocol.CONFIRMATION_MASKS[mask_name]:
            role = protocol.semantic_role(site, target)
            start, end = (
                int(value) for value in row["anchor_spans"][role]
            )
            active.extend(range(start, end + 1))
        if (
            len(active) != len(set(active))
            or len(active) > protocol.MAX_COALITION_TOKENS
        ):
            raise RuntimeError(f"invalid coalition mask {mask_name}")
        for pair_slot in (2 * target_slot, 2 * target_slot + 1):
            positions[pair_slot, :len(active)] = torch.tensor(
                active, dtype=torch.long
            )
            masks[pair_slot, :len(active)] = True
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


def analyze(
    paired_logits: np.ndarray,
    baseline_logits: np.ndarray,
    targets: list[dict[str, Any]],
    prereg: dict[str, Any],
) -> dict[str, Any]:
    source_margin = margin_values(baseline_logits[:, 0, :], targets)
    zero_margin = margin_values(baseline_logits[:, 1, :], targets)
    source_shift = source_margin - zero_margin
    mask_rows = {}
    for mask_slot, mask_name in enumerate(protocol.CONFIRMATION_MASKS):
        reset_margin = margin_values(
            paired_logits[:, mask_slot, 0, :], targets
        )
        replay_margin = margin_values(
            paired_logits[:, mask_slot, 1, :], targets
        )
        reset_shift = reset_margin - zero_margin
        replay_shift = replay_margin - zero_margin
        blocked = source_shift - reset_shift
        mask_rows[mask_name] = {
            "source_shift": trajectory_tools.scalar_summary(
                source_shift
            ),
            "reset_shift": trajectory_tools.scalar_summary(reset_shift),
            "replay_shift": trajectory_tools.scalar_summary(
                replay_shift
            ),
            "blocked_amount": trajectory_tools.scalar_summary(blocked),
            "mediation_fraction": trajectory_tools.scalar_summary(
                ratio(blocked, source_shift)
            ),
            "replay_recovery": trajectory_tools.scalar_summary(
                ratio(replay_shift, source_shift)
            ),
        }
    pair = mask_rows["concept_pair"]
    selected = mask_rows["selected_concept"]
    unselected = mask_rows["unselected_concept"]
    query = mask_rows["query_boundary"]
    gains = {
        "pair_minus_best_constituent_mediation": (
            float(pair["mediation_fraction"]["median"])
            - max(
                float(selected["mediation_fraction"]["median"]),
                float(unselected["mediation_fraction"]["median"]),
            )
        ),
        "pair_minus_best_constituent_replay": (
            float(pair["replay_recovery"]["median"])
            - max(
                float(selected["replay_recovery"]["median"]),
                float(unselected["replay_recovery"]["median"]),
            )
        ),
        "pair_minus_query_boundary_mediation": (
            float(pair["mediation_fraction"]["median"])
            - float(query["mediation_fraction"]["median"])
        ),
    }
    gate = prereg["confirmation_gate"]
    pair_gate = (
        pair["source_shift"]["positive_rate"]
        >= gate["concept_pair_source_positive_rate_min"]
        and pair["blocked_amount"]["positive_rate"]
        >= gate["concept_pair_blocked_positive_rate_min"]
        and pair["mediation_fraction"]["median"]
        >= gate["concept_pair_mediation_fraction_median_min"]
        and pair["replay_shift"]["positive_rate"]
        >= gate["concept_pair_replay_positive_rate_min"]
        and pair["replay_recovery"]["median"]
        >= gate["concept_pair_replay_recovery_median_min"]
    )
    alliance_gate = (
        pair_gate
        and gains["pair_minus_best_constituent_mediation"]
        >= gate["pair_minus_best_constituent_mediation_min"]
        and gains["pair_minus_best_constituent_replay"]
        >= gate["pair_minus_best_constituent_replay_min"]
        and gains["pair_minus_query_boundary_mediation"]
        >= gate["pair_minus_query_boundary_mediation_min"]
    )
    selected_matches_pair = (
        pair_gate
        and gains["pair_minus_best_constituent_mediation"]
        < gate["pair_minus_best_constituent_mediation_min"]
    )
    return {
        "mask_metrics": mask_rows,
        "alliance_gains": gains,
        "concept_pair_gate_passed": bool(pair_gate),
        "concept_pair_alliance_gate_passed": bool(alliance_gate),
        "selected_or_unselected_matches_pair": bool(
            selected_matches_pair
        ),
    }


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1047 protocol audit failed")
    targets = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / "targets.jsonl"
    )
    targets_by_index = {
        int(row["target_index"]): row for row in targets
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
    receiver_depth = int(
        prereg["model_depths"][model_name]["receiver_depth"]
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

        source_cache = np.lib.format.open_memmap(
            atlas_dir / "source_states.fp16.npy",
            mode="w+",
            dtype=np.float16,
            shape=(
                len(cases_list),
                len(source_tools.SOURCE_ROLES),
                protocol.MAX_SOURCE_SPAN,
                info.d_model,
            ),
        )
        clean_logits = np.lib.format.open_memmap(
            atlas_dir / "clean_candidate_logits.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(cases_list), len(protocol.material.FAMILIES)),
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
        paired_logits = np.lib.format.open_memmap(
            atlas_dir / "paired_candidate_logits.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(
                len(targets),
                len(protocol.CONFIRMATION_MASKS),
                2,
                len(protocol.material.FAMILIES),
            ),
        )
        response_norms = np.lib.format.open_memmap(
            atlas_dir / "coalition_response_norms.fp32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(targets), 1, len(protocol.CONFIRMATION_MASKS)),
        )
        for array in (
            source_cache,
            clean_logits,
            baseline_logits,
            paired_logits,
            response_norms,
        ):
            array[:] = np.nan

        capture = source_tools.SourceStateCapture(
            layers[source_depth - 1], source_cache, case_to_local
        )
        capture.register()
        try:
            for row_batch in chunks(
                cases_list, CLEAN_BATCH_SIZE[model_name]
            ):
                (
                    input_ids,
                    attention_mask,
                    positions,
                    masks,
                    pre_positions,
                    case_indices,
                ) = source_tools.make_clean_batch(
                    row_batch,
                    pad_token_id=pad_token_id,
                    device=device,
                )
                capture.begin(positions, masks, case_indices)
                with torch.inference_mode():
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                capture.end()
                logits = output.logits
                batch = torch.arange(
                    logits.shape[0], device=logits.device
                )
                selected = logits[
                    batch, pre_positions.to(logits.device), :
                ].float().index_select(
                    -1, candidate_ids.to(logits.device)
                )
                local = np.asarray(
                    [case_to_local[int(value)] for value in case_indices],
                    dtype=np.int64,
                )
                clean_logits[local] = selected.detach().cpu().numpy()
                del output, logits, selected
        finally:
            capture.close()
        source_cache.flush()
        clean_logits.flush()

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
                        int(row["confirmation_index"])
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
                    batch, pre_positions.to(logits.device), :
                ].float().index_select(
                    -1, candidate_ids.to(logits.device)
                )
                baseline_logits[target_indices] = selected.reshape(
                    len(target_batch),
                    2,
                    len(protocol.material.FAMILIES),
                ).detach().cpu().numpy()
                del output, logits, selected

            coalition_swap = coalition_tools.CoalitionSwap(
                layers[receiver_depth - 1], response_norms
            )
            coalition_swap.register()
            try:
                for mask_slot, mask_name in enumerate(
                    protocol.CONFIRMATION_MASKS
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
                        positions, masks, full_sequence = (
                            coalition_positions(
                                target_batch, mask_name, cases
                            )
                        )
                        target_indices = np.asarray(
                            [
                                int(row["confirmation_index"])
                                for row in target_batch
                            ],
                            dtype=np.int64,
                        )
                        source_patch.begin(
                            source_positions, source_masks, payloads
                        )
                        coalition_swap.begin(
                            positions,
                            masks,
                            full_sequence,
                            target_indices,
                            0,
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
                        paired_logits[
                            target_indices, mask_slot, :, :
                        ] = selected.reshape(
                            len(target_batch),
                            2,
                            len(protocol.material.FAMILIES),
                        ).detach().cpu().numpy()
                        del output, logits, selected
            finally:
                coalition_swap.close()
        finally:
            source_patch.close()
        for array in (baseline_logits, paired_logits, response_norms):
            array.flush()

        analysis = analyze(
            paired_logits, baseline_logits, targets, prereg
        )
        summary = {
            "schema_version": "phase1047_model_summary.v1",
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
            "receiver_depth": receiver_depth,
            "source_cache_finite": trajectory_tools.finite_summary(
                source_cache
            ),
            "baseline_logits_finite": trajectory_tools.finite_summary(
                baseline_logits
            ),
            "paired_logits_finite": trajectory_tools.finite_summary(
                paired_logits
            ),
            "response_norms_finite": trajectory_tools.finite_summary(
                response_norms
            ),
            "analysis": analysis,
            "elapsed_seconds": float(time.time() - started),
        }
        protocol.write_json(atlas_dir / "summary.json", summary)
        print(json.dumps({
            "model": model_name,
            "pair_mediation": analysis["mask_metrics"][
                "concept_pair"
            ]["mediation_fraction"]["median"],
            "selected_mediation": analysis["mask_metrics"][
                "selected_concept"
            ]["mediation_fraction"]["median"],
            "unselected_mediation": analysis["mask_metrics"][
                "unselected_concept"
            ]["mediation_fraction"]["median"],
            "pair_replay": analysis["mask_metrics"][
                "concept_pair"
            ]["replay_recovery"]["median"],
            "alliance_gains": analysis["alliance_gains"],
            "pair_gate": analysis["concept_pair_gate_passed"],
            "alliance_gate": analysis[
                "concept_pair_alliance_gate_passed"
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
