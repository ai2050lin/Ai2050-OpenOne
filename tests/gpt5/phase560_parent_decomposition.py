#!/usr/bin/env python3
"""Decompose qualified Phase560 source-color edges into coarse parent components."""

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
from phase559_causal_screen import replace_primary  # noqa: E402


MODEL = "qwen3"
OUT_DIR = ROOT / "tests/gpt5/result/phase560_semantic_color_route"
PARENT_DIR = ROOT / "tests/gpt5/result/phase559_fixed_identity_replication"
CONTRACT_PATH = OUT_DIR / "phase560_parent_decomposition_frozen_contract.json"
CANDIDATES_PATH = OUT_DIR / "phase560_semantic_color_qualified_candidates.json"
PATH_ROWS = PARENT_DIR / "phase559_qwen3_path_behavior_rows.jsonl"
ROWS_PATH = OUT_DIR / "phase560_parent_decomposition_rows.jsonl"
SUMMARY_PATH = OUT_DIR / "phase560_parent_decomposition_execution_summary.json"
COMPONENTS = ("layer_input", "attention_output", "mlp_output", "layer_output")


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
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run(batch_size: int, restart: bool) -> Path:
    contract = read_json(CONTRACT_PATH)
    candidate_registry = read_json(CANDIDATES_PATH)
    candidates = [
        row for row in candidate_registry["qualified_candidates"]
        if row["candidate_id"] in set(contract["candidate_ids"])
    ]
    selected = set(contract["selected_anchor_ids"])
    cases = [
        row for row in read_jsonl(PATH_ROWS)
        if row["split"] == contract["split"] and row["anchor_id"] in selected
    ]
    if len(cases) != contract["recipient_case_count"] or len(candidates) != contract["candidate_count"]:
        raise RuntimeError("Phase560 parent denominator drift")
    if any(not row["semantic_correct"] or row["sealed"] for row in cases):
        raise RuntimeError("Phase560 parent cases are not behavior-qualified open rows")
    pair_members: dict[str, dict[int, dict[str, Any]]] = {}
    for row in cases:
        pair_members.setdefault(row["pair_id"], {})[int(row["binding"])] = row
    donor_by_case = {
        row["case_id"]: pair_members[row["pair_id"]][1 - int(row["binding"])]
        for row in cases
    }
    if restart:
        ROWS_PATH.unlink(missing_ok=True)
        SUMMARY_PATH.unlink(missing_ok=True)
    if ROWS_PATH.exists():
        raise RuntimeError("Phase560 parent resume is disabled; use --restart")

    loaded = None
    started = time.monotonic()
    try:
        loaded = load_probe_model(MODEL)
        loaded.tokenizer.padding_side = "left"
        layers = get_layers(loaded.model)
        run_dtype = str(next(loaded.model.parameters()).dtype)
        if run_dtype != "torch.bfloat16" or len(layers) != 36:
            raise RuntimeError(f"Phase560 parent model drift: {run_dtype}/{len(layers)}")
        captures: dict[tuple[str, str, str], torch.Tensor] = {}
        for batch_start in range(0, len(cases), batch_size):
            batch_rows = cases[batch_start:batch_start + batch_size]
            individual = [semantic_positions(loaded.tokenizer, row) for row in batch_rows]
            encoded = loaded.tokenizer(
                [row["prompt"] for row in batch_rows], return_tensors="pt", padding=True,
                truncation=True, max_length=256,
            )
            sequence_length = int(encoded["input_ids"].shape[1])
            positions_by_candidate = {}
            for candidate in candidates:
                indices = []
                for row_index, (ids, semantic) in enumerate(individual):
                    batch_ids = encoded["input_ids"][row_index][encoded["attention_mask"][row_index].bool()].tolist()
                    if [int(value) for value in batch_ids] != ids:
                        raise RuntimeError("Phase560 parent capture tokenization drift")
                    indices.append(sequence_length - len(ids) + semantic["source_color_end"])
                positions_by_candidate[candidate["candidate_id"]] = torch.tensor(
                    indices, dtype=torch.long, device=loaded.input_device
                )
            handles = []

            def save(candidate_id: str, component: str, value: torch.Tensor) -> None:
                indices = positions_by_candidate[candidate_id]
                batch_index = torch.arange(value.shape[0], device=value.device)
                selected_vectors = value[batch_index, indices, :].detach().float().cpu()
                for local, row in enumerate(batch_rows):
                    captures[(candidate_id, component, row["case_id"])] = selected_vectors[local]

            def make_pre(candidate_id: str):
                def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
                    save(candidate_id, "layer_input", inputs[0])
                return hook

            def make_forward(candidate_id: str, component: str):
                def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
                    save(candidate_id, component, tensor_from_output(output))
                return hook

            for candidate in candidates:
                layer = layers[int(candidate["layer"])]
                candidate_id = candidate["candidate_id"]
                handles.extend((
                    layer.register_forward_pre_hook(make_pre(candidate_id)),
                    layer.self_attn.register_forward_hook(make_forward(candidate_id, "attention_output")),
                    layer.mlp.register_forward_hook(make_forward(candidate_id, "mlp_output")),
                    layer.register_forward_hook(make_forward(candidate_id, "layer_output")),
                ))
            encoded = {key: value.to(loaded.input_device) for key, value in encoded.items()}
            with torch.inference_mode():
                result = loaded.model(**encoded, use_cache=False)
            for handle in handles:
                handle.remove()
            del result, encoded

        completed = 0
        for candidate in candidates:
            candidate_id = candidate["candidate_id"]
            layer_index = int(candidate["layer"])
            for batch_start in range(0, len(cases), batch_size):
                batch_rows = cases[batch_start:batch_start + batch_size]
                individual = [semantic_positions(loaded.tokenizer, row) for row in batch_rows]
                encoded = loaded.tokenizer(
                    [row["prompt"] for row in batch_rows], return_tensors="pt", padding=True,
                    truncation=True, max_length=256,
                )
                sequence_length = int(encoded["input_ids"].shape[1])
                positions = []
                for row_index, (ids, semantic) in enumerate(individual):
                    batch_ids = encoded["input_ids"][row_index][encoded["attention_mask"][row_index].bool()].tolist()
                    if [int(value) for value in batch_ids] != ids:
                        raise RuntimeError("Phase560 parent intervention tokenization drift")
                    positions.append(sequence_length - len(ids) + semantic["source_color_end"])
                positions_tensor = torch.tensor(positions, dtype=torch.long, device=loaded.input_device)
                encoded = {key: value.to(loaded.input_device) for key, value in encoded.items()}
                with torch.inference_mode():
                    baseline_result = loaded.model(**encoded, use_cache=False)
                baseline_logits = baseline_result.logits[:, -1, :].detach().float().cpu()
                baseline_scores = [
                    word_scores(baseline_logits[index], loaded.tokenizer, row["all_candidates"])
                    for index, row in enumerate(batch_rows)
                ]

                for condition in contract["conditions"]:
                    component = condition.removesuffix("_donor_replace")
                    if condition == "same_case_restore":
                        component = "same_case"
                        replacement_tensor = torch.stack([
                            captures[(candidate_id, "layer_output", row["case_id"])]
                            for row in batch_rows
                        ])
                    else:
                        replacement_tensor = torch.stack([
                            captures[(candidate_id, component, donor_by_case[row["case_id"]]["case_id"])]
                            for row in batch_rows
                        ])
                    replacement_tensor = replacement_tensor.to(
                        device=loaded.input_device, dtype=next(loaded.model.parameters()).dtype
                    )

                    def patch_primary(value: torch.Tensor) -> torch.Tensor:
                        primary = value.clone()
                        batch_index = torch.arange(primary.shape[0], device=primary.device)
                        primary[batch_index, positions_tensor, :] = replacement_tensor
                        return primary

                    def pre_hook(_module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
                        return (patch_primary(inputs[0]), *inputs[1:])

                    def forward_hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
                        return replace_primary(output, patch_primary(tensor_from_output(output)))

                    handle = None
                    if condition == "layer_input_donor_replace":
                        handle = layers[layer_index].register_forward_pre_hook(pre_hook)
                    elif condition == "attention_output_donor_replace":
                        handle = layers[layer_index].self_attn.register_forward_hook(forward_hook)
                    elif condition == "mlp_output_donor_replace":
                        handle = layers[layer_index].mlp.register_forward_hook(forward_hook)
                    elif condition == "layer_output_donor_replace":
                        handle = layers[layer_index].register_forward_hook(forward_hook)
                    with torch.inference_mode():
                        result = loaded.model(**encoded, use_cache=False)
                    if handle is not None:
                        handle.remove()
                    logits = result.logits[:, -1, :].detach().float().cpu()
                    output_rows = []
                    for index, recipient in enumerate(batch_rows):
                        donor = donor_by_case[recipient["case_id"]]
                        scores = word_scores(logits[index], loaded.tokenizer, recipient["all_candidates"])
                        baseline = baseline_scores[index]
                        baseline_margin = baseline[donor["target"]] - baseline[recipient["target"]]
                        margin = scores[donor["target"]] - scores[recipient["target"]]
                        output_rows.append({
                            "schema_version": "phase560_parent_decomposition.v1",
                            "phase_id": "Phase560",
                            "created_at": now(),
                            "model": MODEL,
                            "torch_dtype": run_dtype,
                            "split": contract["split"],
                            "candidate_id": candidate_id,
                            "layer": layer_index,
                            "zone": candidate["zone"],
                            "semantic_position": "source_color_end",
                            "condition": condition,
                            "intervened_component": component,
                            "recipient_case_id": recipient["case_id"],
                            "donor_case_id": donor["case_id"],
                            "anchor_id": recipient["anchor_id"],
                            "query_object_index": recipient["query_object_index"],
                            "surface_id": recipient["surface_id"],
                            "fact_order": recipient["fact_order"],
                            "recipient_target": recipient["target"],
                            "donor_target": donor["target"],
                            "baseline_switch_margin": baseline_margin,
                            "intervention_switch_margin": margin,
                            "donor_switch_effect": margin - baseline_margin,
                            "intervention_donor_wins": max(scores, key=scores.get) == donor["target"],
                            "intervention_recipient_retained": max(scores, key=scores.get) == recipient["target"],
                            "parent_component_diagnostic": True,
                            "binding_operation": False,
                            "sealed": False,
                        })
                    append_jsonl(ROWS_PATH, output_rows)
                    completed += len(output_rows)
                    del result, logits, replacement_tensor, output_rows
                del baseline_result, baseline_logits, encoded
                if batch_start == 0 or completed == contract["expected_intervention_rows"] or (batch_start // batch_size) % 24 == 23:
                    print(
                        f"[{time.strftime('%H:%M:%S')}] qwen3 Phase560 parent decomposition "
                        f"{completed}/{contract['expected_intervention_rows']}",
                        flush=True,
                    )
        final_rows = read_jsonl(ROWS_PATH)
        if len(final_rows) != contract["expected_intervention_rows"]:
            raise RuntimeError("Phase560 parent output denominator drift")
        summary = {
            "schema_version": "phase560_parent_decomposition_execution_summary.v1",
            "phase_id": "Phase560",
            "created_at": now(),
            "status": "complete",
            "model": MODEL,
            "torch_dtype": run_dtype,
            "case_count": len(cases),
            "candidate_count": len(candidates),
            "condition_count": len(contract["conditions"]),
            "intervention_row_count": len(final_rows),
            "runtime_seconds": time.monotonic() - started,
            "rows_sha256": sha256_file(ROWS_PATH),
            "parent_component_decomposition_executed": True,
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
