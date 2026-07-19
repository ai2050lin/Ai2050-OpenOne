#!/usr/bin/env python3
"""Validate frozen multi-position attention reader blocks on Qwen3."""

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
from phase559_causal_screen import component_module, deterministic_roll, replace_primary  # noqa: E402


MODEL = "qwen3"
PHASE559 = ROOT / "tests/gpt5/result/phase559_fixed_identity_replication"
OUT_DIR = ROOT / "tests/gpt5/result/phase561_source_to_query_trace"
CONTRACT_PATH = OUT_DIR / "phase563_multiposition_reader_frozen_contract.json"
CANDIDATES_PATH = OUT_DIR / "phase563_multiposition_reader_candidate_registry.json"
PATH_ROWS = PHASE559 / "phase559_qwen3_path_behavior_rows.jsonl"
ROWS_PATH = OUT_DIR / "phase563_multiposition_reader_rows.jsonl"
SUMMARY_PATH = OUT_DIR / "phase563_multiposition_reader_execution_summary.json"


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
        raise RuntimeError("Phase563 lost a counterfactual pair")
    ordered = []
    for pair_id in sorted(pairs):
        ordered.extend((pairs[pair_id][0], pairs[pair_id][1]))
    return ordered


def run(batch_size: int, restart: bool) -> Path:
    if batch_size < 2 or batch_size % 2:
        raise ValueError("Phase563 batch size must be a positive even number")
    contract = read_json(CONTRACT_PATH)
    registry = read_json(CANDIDATES_PATH)
    if (
        contract["model"] != MODEL
        or contract["evidence_policy"]["sealed_split_read"]
        or not registry["candidate_family_frozen_before_model_execution"]
    ):
        raise RuntimeError("Phase563 frozen contract drift")
    selected = set(contract["selected_anchor_ids"])
    cases = paired_case_order([
        row
        for row in read_jsonl(PATH_ROWS)
        if row["split"] == contract["split"] and row["anchor_id"] in selected
    ])
    if len(cases) != contract["recipient_case_count"] or any(not row["semantic_correct"] for row in cases):
        raise RuntimeError("Phase563 behavior-qualified denominator drift")
    donor_by_case: dict[str, dict[str, Any]] = {}
    for index in range(0, len(cases), 2):
        left, right = cases[index:index + 2]
        donor_by_case[left["case_id"]] = right
        donor_by_case[right["case_id"]] = left
    candidates = registry["candidates"]
    if len(candidates) != contract["candidate_count"]:
        raise RuntimeError("Phase563 candidate denominator drift")
    if restart:
        ROWS_PATH.unlink(missing_ok=True)
        SUMMARY_PATH.unlink(missing_ok=True)
    if ROWS_PATH.exists():
        raise RuntimeError("Phase563 resume is disabled; use --restart")

    loaded = None
    started = time.monotonic()
    try:
        loaded = load_probe_model(MODEL)
        loaded.tokenizer.padding_side = "left"
        layers = get_layers(loaded.model)
        run_dtype = str(next(loaded.model.parameters()).dtype)
        cuda_used = torch.cuda.is_available() and str(loaded.input_device).startswith("cuda")
        if run_dtype != "torch.bfloat16" or len(layers) != 36 or not cuda_used:
            raise RuntimeError(
                f"Phase563 model drift: dtype={run_dtype}, layers={len(layers)}, "
                f"input_device={loaded.input_device}, cuda={torch.cuda.is_available()}"
            )

        completed = 0
        conditions = contract["conditions"]
        target_layers = sorted({int(candidate["layer"]) for candidate in candidates})
        capture_layers = sorted({
            int(candidate[key])
            for candidate in candidates
            for key in ("layer", "wrong_depth_control_layer")
        })
        for batch_start in range(0, len(cases), batch_size):
            batch_rows = cases[batch_start:batch_start + batch_size]
            batch_case_ids = {row["case_id"] for row in batch_rows}
            if any(
                donor_by_case[row["case_id"]]["case_id"] not in batch_case_ids
                for row in batch_rows
            ):
                raise RuntimeError("Phase563 batch split a counterfactual pair")
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
            positions_by_candidate: dict[str, torch.Tensor] = {}
            wrong_positions_by_candidate: dict[str, torch.Tensor] = {}
            for candidate in candidates:
                candidate_id = candidate["candidate_id"]
                position_rows = []
                for row_index, (ids, semantic) in enumerate(individual):
                    mask_ids = encoded["input_ids"][row_index][
                        encoded["attention_mask"][row_index].bool()
                    ].tolist()
                    if [int(value) for value in mask_ids] != ids:
                        raise RuntimeError("Phase563 tokenization drift")
                    left_padding = sequence_length - len(ids)
                    position_rows.append([
                        left_padding + semantic[role]
                        for role in candidate["semantic_positions"]
                    ])
                position_rows_by_candidate[candidate_id] = position_rows
                positions = torch.tensor(
                    position_rows, dtype=torch.long, device=loaded.input_device
                )
                positions_by_candidate[candidate_id] = positions
                wrong_positions_by_candidate[candidate_id] = torch.clamp(positions - 1, min=0)
            encoded = {key: value.to(loaded.input_device) for key, value in encoded.items()}
            captures: dict[int, torch.Tensor] = {}

            def capture(layer_index: int):
                def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
                    captures[layer_index] = tensor_from_output(output).detach()
                return hook

            handles = [
                component_module(layers, layer_index, "attention_output").register_forward_hook(
                    capture(layer_index)
                )
                for layer_index in capture_layers
            ]
            with torch.inference_mode():
                baseline_result = loaded.model(**encoded, use_cache=False)
            for handle in handles:
                handle.remove()
            if set(captures) != set(capture_layers):
                raise RuntimeError("Phase563 capture hooks did not fire")
            for layer_index in target_layers:
                layer_candidates = [
                    candidate for candidate in candidates
                    if int(candidate["layer"]) == layer_index
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
                replacement_tensors = []
                roll_shifts_by_spec = []
                replacement_sources_by_spec = []
                for candidate, condition in specs:
                    candidate_id = candidate["candidate_id"]
                    positions = positions_by_candidate[candidate_id]
                    wrong_positions = wrong_positions_by_candidate[candidate_id]
                    replacements = []
                    roll_shifts = []
                    replacement_sources = []
                    for local, row in enumerate(batch_rows):
                        donor = donor_by_case[row["case_id"]]
                        donor_local = local_by_case[donor["case_id"]]
                        recipient_state = captures[layer_index][local, positions[local], :]
                        donor_state = captures[layer_index][
                            donor_local, positions[donor_local], :
                        ]
                        replacement = donor_state
                        replacement_source = "paired_donor_correct_multiposition_block"
                        roll = deterministic_roll(
                            candidate_id, row["case_id"], replacement.shape[-1]
                        )
                        if condition == "paired_contrast_neutralize":
                            replacement = (recipient_state + donor_state) / 2.0
                            replacement_source = "paired_multiposition_block_midpoint"
                        elif condition == "wrong_depth_donor_replace":
                            wrong_depth = int(candidate["wrong_depth_control_layer"])
                            replacement = captures[wrong_depth][
                                donor_local, positions[donor_local], :
                            ]
                            replacement_source = "paired_donor_wrong_depth_multiposition_block"
                        elif condition == "wrong_position_donor_replace":
                            replacement = captures[layer_index][
                                donor_local, wrong_positions[donor_local], :
                            ]
                            replacement_source = "paired_donor_one_token_left_block"
                        elif condition == "channel_roll_donor_replace":
                            replacement = torch.roll(replacement, roll, dims=-1)
                            replacement_source = "channel_rolled_paired_donor_block"
                        elif condition == "same_case_restore":
                            replacement = recipient_state
                            replacement_source = "same_batch_no_op"
                        replacements.append(replacement)
                        roll_shifts.append(
                            roll if condition == "channel_roll_donor_replace" else None
                        )
                        replacement_sources.append(replacement_source)
                    replacement_tensors.append(torch.stack(replacements).to(
                        device=loaded.input_device,
                        dtype=next(loaded.model.parameters()).dtype,
                    ))
                    roll_shifts_by_spec.append(roll_shifts)
                    replacement_sources_by_spec.append(replacement_sources)

                def intervention_hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
                    primary = tensor_from_output(output).clone()
                    local_batch = len(batch_rows)
                    for spec_index, (candidate, _condition) in enumerate(specs):
                        positions = positions_by_candidate[candidate["candidate_id"]]
                        batch_indices = (
                            spec_index * local_batch
                            + torch.arange(local_batch, device=primary.device)
                        )[:, None]
                        primary[batch_indices, positions, :] = replacement_tensors[spec_index]
                    return replace_primary(output, primary)

                module = component_module(layers, layer_index, "attention_output")
                handle = module.register_forward_hook(intervention_hook)
                with torch.inference_mode():
                    result = loaded.model(**expanded_encoded, use_cache=False)
                handle.remove()
                logits = result.logits[:, -1, :].detach().float().cpu()
                same_case_scores: dict[str, list[dict[str, float]]] = {}
                for spec_index, (candidate, condition) in enumerate(specs):
                    if condition != "same_case_restore":
                        continue
                    same_case_scores[candidate["candidate_id"]] = [
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
                    position_rows = position_rows_by_candidate[candidate_id]
                    for index, recipient in enumerate(batch_rows):
                        donor = donor_by_case[recipient["case_id"]]
                        result_index = spec_index * len(batch_rows) + index
                        scores = word_scores(
                            logits[result_index], loaded.tokenizer, recipient["all_candidates"]
                        )
                        baseline = same_case_scores[candidate_id][index]
                        baseline_margin = (
                            baseline[donor["target"]] - baseline[recipient["target"]]
                        )
                        margin = scores[donor["target"]] - scores[recipient["target"]]
                        output_rows.append({
                            "schema_version": "phase563_multiposition_reader.v1",
                            "phase_id": "Phase563",
                            "created_at": now(),
                            "model": MODEL,
                            "torch_dtype": run_dtype,
                            "split": contract["split"],
                            "candidate_id": candidate_id,
                            "layer": layer_index,
                            "component": candidate["component"],
                            "position_block": candidate["position_block"],
                            "semantic_positions": candidate["semantic_positions"],
                            "semantic_role_count": len(candidate["semantic_positions"]),
                            "unique_token_position_count": len(set(position_rows[index])),
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
                            "recipient_target": recipient["target"],
                            "donor_target": donor["target"],
                            "baseline_scores": baseline,
                            "intervention_scores": scores,
                            "baseline_switch_margin": baseline_margin,
                            "intervention_switch_margin": margin,
                            "donor_switch_effect": margin - baseline_margin,
                            "baseline_prediction": max(baseline, key=baseline.get),
                            "intervention_prediction": max(scores, key=scores.get),
                            "intervention_donor_wins": (
                                max(scores, key=scores.get) == donor["target"]
                            ),
                            "intervention_recipient_retained": (
                                max(scores, key=scores.get) == recipient["target"]
                            ),
                            "roll_shift": roll_shifts_by_spec[spec_index][index],
                            "replacement_source": (
                                replacement_sources_by_spec[spec_index][index]
                            ),
                            "source_recompute_intervention": True,
                            "block_sufficiency_only": True,
                            "compute_edge": False,
                            "sealed": False,
                        })
                append_jsonl(ROWS_PATH, output_rows)
                completed += len(output_rows)
                del (
                    result,
                    logits,
                    expanded_encoded,
                    replacement_tensors,
                    roll_shifts_by_spec,
                    replacement_sources_by_spec,
                    output_rows,
                    intervention_hook,
                    handle,
                )
            del baseline_result, captures, encoded, handles
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if batch_start == 0 or completed == contract["expected_intervention_rows"]:
                print(
                    f"[{time.strftime('%H:%M:%S')}] Phase563 fused conditions "
                    f"{completed}/{contract['expected_intervention_rows']}",
                    flush=True,
                )

        final_rows = read_jsonl(ROWS_PATH)
        if len(final_rows) != contract["expected_intervention_rows"]:
            raise RuntimeError("Phase563 output denominator drift")
        summary = {
            "schema_version": "phase563_multiposition_reader_execution_summary.v1",
            "phase_id": "Phase563",
            "created_at": now(),
            "status": "complete",
            "model": MODEL,
            "torch_dtype": run_dtype,
            "cuda_used": cuda_used,
            "case_count": len(cases),
            "candidate_count": len(candidates),
            "condition_count": len(contract["conditions"]),
            "intervention_row_count": len(final_rows),
            "runtime_seconds": time.monotonic() - started,
            "rows_sha256": sha256_file(ROWS_PATH),
            "effect_baseline": "same_case_restore_scores_from_same_fused_batch_shape",
            "source_recompute_intervention_executed": True,
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    run(args.batch_size, args.restart)
