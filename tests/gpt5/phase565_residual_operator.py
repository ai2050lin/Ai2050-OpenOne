#!/usr/bin/env python3
"""Validate frozen full-residual multi-position operators on Qwen3."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))
os.environ.setdefault("PROBE_TORCH_DTYPE", "bfloat16")

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase557_natural_color_source_intervention import word_scores  # noqa: E402
from phase559_binding_event_collect import semantic_positions, tensor_from_output  # noqa: E402
from phase559_causal_screen import deterministic_roll, replace_primary  # noqa: E402


MODEL = "qwen3"
PARENT_DIR = ROOT / "tests/gpt5/result/phase564_source_conditioned_edge"
OUT_DIR = ROOT / "tests/gpt5/result/phase565_residual_multiposition_operator"
PATH_ROWS = PARENT_DIR / "phase564_qwen3_edge_behavior_rows.jsonl"
CONTRACT_PATH = OUT_DIR / "phase565_residual_operator_frozen_contract.json"
CANDIDATES_PATH = OUT_DIR / "phase565_residual_operator_candidates.json"
ROWS_PATH = OUT_DIR / "phase565_residual_operator_rows.jsonl"
SUMMARY_PATH = OUT_DIR / "phase565_residual_operator_execution_summary.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_case_order(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs: dict[str, dict[int, dict[str, Any]]] = {}
    for row in cases:
        pairs.setdefault(row["pair_id"], {})[int(row["binding"])] = row
    if any(set(members) != {0, 1} for members in pairs.values()):
        raise RuntimeError("Phase565 lost a counterfactual pair")
    ordered = []
    for pair_id in sorted(pairs):
        ordered.extend((pairs[pair_id][0], pairs[pair_id][1]))
    return ordered


def run(batch_size: int, restart: bool) -> Path:
    if batch_size < 2 or batch_size % 2:
        raise ValueError("Phase565 batch size must be a positive even number")
    contract = read_json(CONTRACT_PATH)
    registry = read_json(CANDIDATES_PATH)
    if (
        contract["model"] != MODEL
        or contract["evidence_policy"]["sealed_split_read"]
        or not registry["candidate_family_frozen_before_model_execution"]
    ):
        raise RuntimeError("Phase565 frozen contract drift")
    selected = set(contract["selected_anchor_ids"])
    cases = paired_case_order([
        row for row in read_jsonl(PATH_ROWS)
        if row["split"] == contract["split"]
        and row["anchor_id"] in selected
        and row["semantic_correct"]
    ])
    if len(cases) != contract["recipient_case_count"]:
        raise RuntimeError("Phase565 behavior-qualified denominator drift")
    candidates = registry["candidates"]
    if len(candidates) != contract["candidate_count"]:
        raise RuntimeError("Phase565 candidate denominator drift")
    if restart:
        ROWS_PATH.unlink(missing_ok=True)
        SUMMARY_PATH.unlink(missing_ok=True)
    if ROWS_PATH.exists():
        raise RuntimeError("Phase565 resume is disabled; use --restart")
    donor_by_case: dict[str, dict[str, Any]] = {}
    for index in range(0, len(cases), 2):
        left, right = cases[index:index + 2]
        donor_by_case[left["case_id"]] = right
        donor_by_case[right["case_id"]] = left

    loaded = None
    started = time.monotonic()
    completed = 0
    try:
        loaded = load_probe_model(MODEL)
        loaded.tokenizer.padding_side = "left"
        layers = get_layers(loaded.model)
        run_dtype = str(next(loaded.model.parameters()).dtype)
        cuda_used = torch.cuda.is_available() and str(loaded.input_device).startswith("cuda")
        if run_dtype != "torch.bfloat16" or len(layers) != 36 or not cuda_used:
            raise RuntimeError(f"Phase565 model drift: {run_dtype}/{len(layers)}/{loaded.input_device}")
        conditions = contract["conditions"]
        capture_layers = sorted({
            int(candidate[key])
            for candidate in candidates
            for key in ("layer", "wrong_depth_control_layer")
        })
        target_layers = sorted({int(candidate["layer"]) for candidate in candidates})

        for batch_start in range(0, len(cases), batch_size):
            batch_rows = cases[batch_start:batch_start + batch_size]
            batch_case_ids = {row["case_id"] for row in batch_rows}
            if any(donor_by_case[row["case_id"]]["case_id"] not in batch_case_ids for row in batch_rows):
                raise RuntimeError("Phase565 batch split a counterfactual pair")
            local_by_case = {row["case_id"]: index for index, row in enumerate(batch_rows)}
            individual = [semantic_positions(loaded.tokenizer, row) for row in batch_rows]
            encoded = loaded.tokenizer(
                [row["prompt"] for row in batch_rows],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            )
            sequence_length = int(encoded["input_ids"].shape[1])
            position_rows_by_candidate: dict[str, list[list[int]]] = {}
            wrong_rows_by_candidate: dict[str, list[list[int]]] = {}
            for candidate in candidates:
                position_rows = []
                wrong_rows = []
                for row_index, (ids, semantic) in enumerate(individual):
                    mask = encoded["attention_mask"][row_index].bool()
                    mask_ids = encoded["input_ids"][row_index][mask].tolist()
                    if [int(value) for value in mask_ids] != ids:
                        raise RuntimeError("Phase565 tokenization drift")
                    left_padding = sequence_length - len(ids)
                    if candidate["position_block"] == "semantic7":
                        positions = sorted({
                            left_padding + int(semantic[role])
                            for role in candidate["semantic_positions"]
                        })
                    else:
                        positions = list(range(left_padding, sequence_length))
                    position_rows.append(positions)
                    wrong_rows.append([max(0, position - 1) for position in positions])
                position_rows_by_candidate[candidate["candidate_id"]] = position_rows
                wrong_rows_by_candidate[candidate["candidate_id"]] = wrong_rows
            encoded = {key: value.to(loaded.input_device) for key, value in encoded.items()}
            captures: dict[int, torch.Tensor] = {}

            def capture(layer_index: int):
                def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
                    captures[layer_index] = tensor_from_output(output).detach()
                return hook

            handles = [layers[layer_index].register_forward_hook(capture(layer_index)) for layer_index in capture_layers]
            with torch.inference_mode():
                loaded.model(**encoded, use_cache=False)
            for handle in handles:
                handle.remove()
            if set(captures) != set(capture_layers):
                raise RuntimeError("Phase565 capture hooks did not all fire")

            for layer_index in target_layers:
                layer_candidates = [
                    candidate for candidate in candidates if int(candidate["layer"]) == layer_index
                ]
                specs = [
                    (candidate, condition)
                    for candidate in layer_candidates
                    for condition in conditions
                ]
                expanded_encoded = {
                    key: value.repeat((len(specs),) + (1,) * (value.ndim - 1))
                    for key, value in encoded.items()
                }
                replacements: list[list[torch.Tensor]] = []
                target_position_rows: list[list[list[int]]] = []
                replacement_sources: list[list[str]] = []
                roll_shifts: list[list[int | None]] = []
                for candidate, condition in specs:
                    candidate_id = candidate["candidate_id"]
                    correct_rows = position_rows_by_candidate[candidate_id]
                    wrong_rows = wrong_rows_by_candidate[candidate_id]
                    spec_replacements = []
                    spec_targets = []
                    spec_sources = []
                    spec_rolls = []
                    for local, row in enumerate(batch_rows):
                        donor = donor_by_case[row["case_id"]]
                        donor_local = local_by_case[donor["case_id"]]
                        recipient_positions = correct_rows[local]
                        donor_positions = correct_rows[donor_local]
                        if len(recipient_positions) != len(donor_positions):
                            raise RuntimeError("Phase565 paired sequence width drift")
                        recipient_state = captures[layer_index][local, recipient_positions, :]
                        donor_state = captures[layer_index][donor_local, donor_positions, :]
                        target_positions = recipient_positions
                        replacement = donor_state
                        source = "paired_donor_residual_block"
                        roll = deterministic_roll(candidate_id, row["case_id"], donor_state.shape[-1])
                        if condition == "same_case_restore":
                            replacement = recipient_state
                            source = "same_case_residual_identity"
                        elif condition == "paired_contrast_neutralize":
                            replacement = (recipient_state + donor_state) / 2.0
                            source = "paired_residual_midpoint"
                        elif condition == "wrong_depth_donor_replace":
                            wrong_depth = int(candidate["wrong_depth_control_layer"])
                            replacement = captures[wrong_depth][donor_local, donor_positions, :]
                            source = "paired_donor_residual_wrong_depth"
                        elif condition == "wrong_position_donor_replace":
                            donor_wrong_positions = wrong_rows[donor_local]
                            replacement = captures[layer_index][donor_local, donor_wrong_positions, :]
                            source = "paired_donor_residual_one_token_left"
                        elif condition == "channel_roll_donor_replace":
                            replacement = torch.roll(donor_state, roll, dims=-1)
                            source = "channel_rolled_paired_donor_residual"
                        elif condition != "paired_donor_residual_replace":
                            raise RuntimeError(f"Unsupported Phase565 condition: {condition}")
                        spec_replacements.append(replacement)
                        spec_targets.append(target_positions)
                        spec_sources.append(source)
                        spec_rolls.append(roll if condition == "channel_roll_donor_replace" else None)
                    replacements.append(spec_replacements)
                    target_position_rows.append(spec_targets)
                    replacement_sources.append(spec_sources)
                    roll_shifts.append(spec_rolls)

                def intervention_hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
                    primary = tensor_from_output(output).clone()
                    local_batch = len(batch_rows)
                    for spec_index in range(len(specs)):
                        for local in range(local_batch):
                            expanded_index = spec_index * local_batch + local
                            positions = target_position_rows[spec_index][local]
                            primary[expanded_index, positions, :] = replacements[spec_index][local]
                    return replace_primary(output, primary)

                handle = layers[layer_index].register_forward_hook(intervention_hook)
                with torch.inference_mode():
                    result = loaded.model(**expanded_encoded, use_cache=False)
                handle.remove()
                logits = result.logits[:, -1, :].detach().float().cpu()
                baseline_scores: dict[str, list[dict[str, float]]] = {}
                for spec_index, (candidate, condition) in enumerate(specs):
                    if condition != "same_case_restore":
                        continue
                    baseline_scores[candidate["candidate_id"]] = [
                        word_scores(
                            logits[spec_index * len(batch_rows) + index],
                            loaded.tokenizer,
                            row["all_candidates"],
                        )
                        for index, row in enumerate(batch_rows)
                    ]
                output_rows = []
                for spec_index, (candidate, condition) in enumerate(specs):
                    candidate_id = candidate["candidate_id"]
                    for index, recipient in enumerate(batch_rows):
                        donor = donor_by_case[recipient["case_id"]]
                        scores = word_scores(
                            logits[spec_index * len(batch_rows) + index],
                            loaded.tokenizer,
                            recipient["all_candidates"],
                        )
                        baseline = baseline_scores[candidate_id][index]
                        baseline_margin = baseline[donor["target"]] - baseline[recipient["target"]]
                        margin = scores[donor["target"]] - scores[recipient["target"]]
                        positions = target_position_rows[spec_index][index]
                        output_rows.append({
                            "schema_version": "phase565_residual_operator.v1",
                            "phase_id": "Phase565",
                            "created_at": now(),
                            "model": MODEL,
                            "torch_dtype": run_dtype,
                            "split": contract["split"],
                            "candidate_id": candidate_id,
                            "layer": layer_index,
                            "component": candidate["component"],
                            "position_block": candidate["position_block"],
                            "position_count": len(positions),
                            "condition": condition,
                            "recipient_case_id": recipient["case_id"],
                            "donor_case_id": donor["case_id"],
                            "pair_id": recipient["pair_id"],
                            "anchor_id": recipient["anchor_id"],
                            "binding": recipient["binding"],
                            "query_object_index": recipient["query_object_index"],
                            "surface_id": recipient["surface_id"],
                            "fact_order": recipient["fact_order"],
                            "factorial_cell_without_binding": (
                                f"query{recipient['query_object_index']}_"
                                f"surface{recipient['surface_id']}_order{recipient['fact_order']}"
                            ),
                            "color_regime": recipient["color_regime"],
                            "recipient_target": recipient["target"],
                            "donor_target": donor["target"],
                            "baseline_scores": baseline,
                            "intervention_scores": scores,
                            "baseline_switch_margin": baseline_margin,
                            "intervention_switch_margin": margin,
                            "donor_switch_effect": margin - baseline_margin,
                            "baseline_prediction": max(baseline, key=baseline.get),
                            "intervention_prediction": max(scores, key=scores.get),
                            "intervention_donor_wins": max(scores, key=scores.get) == donor["target"],
                            "intervention_recipient_retained": max(scores, key=scores.get) == recipient["target"],
                            "roll_shift": roll_shifts[spec_index][index],
                            "replacement_source": replacement_sources[spec_index][index],
                            "distributed_state_sufficiency_only": True,
                            "compute_edge": False,
                            "sealed": False,
                        })
                append_jsonl(ROWS_PATH, output_rows)
                completed += len(output_rows)
                del result, logits, expanded_encoded, replacements, output_rows
            del captures, encoded
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if batch_start == 0 or completed == contract["expected_intervention_rows"]:
                print(
                    f"[{time.strftime('%H:%M:%S')}] Phase565 residual operator "
                    f"{completed}/{contract['expected_intervention_rows']}",
                    flush=True,
                )
        rows = read_jsonl(ROWS_PATH)
        if len(rows) != contract["expected_intervention_rows"]:
            raise RuntimeError("Phase565 output denominator drift")
        summary = {
            "schema_version": "phase565_residual_operator_execution_summary.v1",
            "phase_id": "Phase565",
            "created_at": now(),
            "status": "complete",
            "model": MODEL,
            "torch_dtype": run_dtype,
            "cuda_used": cuda_used,
            "case_count": len(cases),
            "candidate_count": len(candidates),
            "condition_count": len(conditions),
            "intervention_row_count": len(rows),
            "runtime_seconds": time.monotonic() - started,
            "rows_sha256": sha256_file(ROWS_PATH),
            "effect_baseline": contract["effect_baseline"],
            "distributed_state_sufficiency_only": True,
            "head_channel_parameter_neuron_scan_executed": False,
            "sealed_split_read": False,
        }
        write_json(SUMMARY_PATH, summary)
        print(SUMMARY_PATH)
        return SUMMARY_PATH
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    run(args.batch_size, args.restart)
