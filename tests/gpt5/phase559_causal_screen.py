#!/usr/bin/env python3
"""Screen frozen Phase559 source/query boundaries by legal downstream recompute."""

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


MODEL = "qwen3"
OUT_DIR = ROOT / "tests/gpt5/result/phase559_fixed_identity_replication"
CONTRACT_PATH = OUT_DIR / "phase559_causal_screen_frozen_contract.json"
CANDIDATES_PATH = OUT_DIR / "phase559_binding_candidate_registry.json"
PATH_ROWS = OUT_DIR / "phase559_qwen3_path_behavior_rows.jsonl"
ROWS_PATH = OUT_DIR / "phase559_causal_screen_rows.jsonl"
SUMMARY_PATH = OUT_DIR / "phase559_causal_screen_execution_summary.json"


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


def replace_primary(output: Any, primary: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (primary, *output[1:])
    return primary


def deterministic_roll(candidate_id: str, case_id: str, width: int) -> int:
    value = int(hashlib.sha256(f"{candidate_id}|{case_id}".encode("utf-8")).hexdigest()[:8], 16)
    return 1 + value % (width - 1)


def component_module(layers: list[Any], layer_index: int, component: str) -> Any:
    if component == "layer_output":
        return layers[layer_index]
    if component == "attention_output":
        return layers[layer_index].self_attn
    if component == "mlp_output":
        return layers[layer_index].mlp
    raise ValueError(f"Unsupported intervention component: {component}")


def run(
    batch_size: int,
    restart: bool,
    *,
    contract_path: Path = CONTRACT_PATH,
    candidates_path: Path = CANDIDATES_PATH,
    path_rows: Path = PATH_ROWS,
    rows_path: Path = ROWS_PATH,
    summary_path: Path = SUMMARY_PATH,
) -> Path:
    contract = read_json(contract_path)
    registry = read_json(candidates_path)
    if contract["model"] != MODEL or contract["evidence_policy"]["sealed_split_read"]:
        raise RuntimeError("Phase559 causal screen contract drift")
    selected = set(contract["selected_anchor_ids"])
    cases = [
        row for row in read_jsonl(path_rows)
        if row["split"] == contract["split"] and row["anchor_id"] in selected
    ]
    if len(cases) != contract["recipient_case_count"] or any(not row["semantic_correct"] for row in cases):
        raise RuntimeError("Phase559 causal screen case denominator drift")
    case_by_id = {row["case_id"]: row for row in cases}
    pair_members: dict[str, dict[int, dict[str, Any]]] = {}
    for row in cases:
        pair_members.setdefault(row["pair_id"], {})[int(row["binding"])] = row
    donor_by_case = {
        row["case_id"]: pair_members[row["pair_id"]][1 - int(row["binding"])]
        for row in cases
    }
    candidates = registry.get("candidates", registry.get("qualified_candidates", []))
    if len(candidates) != contract["candidate_count"]:
        raise RuntimeError("Phase559 causal screen candidate denominator drift")
    if restart:
        rows_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
    if rows_path.exists():
        raise RuntimeError("Phase559 causal screen resume is disabled; use --restart")

    loaded = None
    started = time.monotonic()
    try:
        loaded = load_probe_model(MODEL)
        loaded.tokenizer.padding_side = "left"
        layers = get_layers(loaded.model)
        run_dtype = str(next(loaded.model.parameters()).dtype)
        if run_dtype != "torch.bfloat16" or len(layers) != 36:
            raise RuntimeError(f"Phase559 causal screen model drift: {run_dtype}/{len(layers)}")

        captures: dict[tuple[str, str, str], torch.Tensor] = {}
        capture_requests = [
            (candidate, "correct", int(candidate["layer"]), candidate["semantic_position"])
            for candidate in candidates
        ] + [
            (
                candidate, "wrong_depth", int(candidate["wrong_depth_control_layer"]),
                candidate["semantic_position"],
            )
            for candidate in candidates
        ] + [
            (
                candidate, "wrong_position", int(candidate["layer"]),
                candidate["wrong_position_control"],
            )
            for candidate in candidates
        ]
        for batch_start in range(0, len(cases), batch_size):
            batch_rows = cases[batch_start:batch_start + batch_size]
            individual = [semantic_positions(loaded.tokenizer, row) for row in batch_rows]
            encoded = loaded.tokenizer(
                [row["prompt"] for row in batch_rows], return_tensors="pt", padding=True,
                truncation=True, max_length=256,
            )
            sequence_length = int(encoded["input_ids"].shape[1])
            request_positions: dict[tuple[str, str], torch.Tensor] = {}
            for candidate, label, _layer, semantic_position in capture_requests:
                indices = []
                for row_index, (ids, semantic) in enumerate(individual):
                    mask_ids = encoded["input_ids"][row_index][encoded["attention_mask"][row_index].bool()].tolist()
                    if [int(value) for value in mask_ids] != ids:
                        raise RuntimeError("Phase559 screen capture tokenization drift")
                    indices.append(sequence_length - len(ids) + semantic[semantic_position])
                request_positions[(candidate["candidate_id"], label)] = torch.tensor(
                    indices, dtype=torch.long, device=loaded.input_device
                )
            handles = []

            def make_capture(candidate: dict[str, Any], label: str):
                def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
                    value = tensor_from_output(output)
                    indices = request_positions[(candidate["candidate_id"], label)]
                    batch_index = torch.arange(value.shape[0], device=value.device)
                    selected_vectors = value[batch_index, indices, :].detach().float().cpu()
                    for local, row in enumerate(batch_rows):
                        captures[(candidate["candidate_id"], label, row["case_id"])] = selected_vectors[local]
                return hook

            for candidate, label, layer, _position in capture_requests:
                handles.append(
                    component_module(layers, layer, candidate["component"]).register_forward_hook(
                        make_capture(candidate, label)
                    )
                )
            encoded = {key: value.to(loaded.input_device) for key, value in encoded.items()}
            with torch.inference_mode():
                result = loaded.model(**encoded, use_cache=False)
            for handle in handles:
                handle.remove()
            del result, encoded

        completed = 0
        conditions = contract["conditions"]
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
                    mask_ids = encoded["input_ids"][row_index][encoded["attention_mask"][row_index].bool()].tolist()
                    if [int(value) for value in mask_ids] != ids:
                        raise RuntimeError("Phase559 screen intervention tokenization drift")
                    positions.append(sequence_length - len(ids) + semantic[candidate["semantic_position"]])
                positions_tensor = torch.tensor(positions, dtype=torch.long, device=loaded.input_device)
                encoded = {key: value.to(loaded.input_device) for key, value in encoded.items()}
                with torch.inference_mode():
                    baseline_result = loaded.model(**encoded, use_cache=False)
                baseline_logits = baseline_result.logits[:, -1, :].detach().float().cpu()
                baseline_scores = [
                    word_scores(baseline_logits[index], loaded.tokenizer, row["all_candidates"])
                    for index, row in enumerate(batch_rows)
                ]

                for condition in conditions:
                    replacements = []
                    roll_shifts = []
                    replacement_sources = []
                    for row in batch_rows:
                        donor = donor_by_case[row["case_id"]]
                        recipient_state = captures[(candidate_id, "correct", row["case_id"])]
                        donor_state = captures[(candidate_id, "correct", donor["case_id"])]
                        replacement = donor_state
                        replacement_source = "paired_donor_correct_coordinate"
                        roll = deterministic_roll(candidate_id, row["case_id"], replacement.numel())
                        if condition == "paired_contrast_neutralize":
                            replacement = (recipient_state + donor_state) / 2.0
                            replacement_source = "paired_state_midpoint"
                        elif condition == "wrong_depth_donor_replace":
                            replacement = captures[(candidate_id, "wrong_depth", donor["case_id"])]
                            replacement_source = "paired_donor_wrong_depth"
                        elif condition == "wrong_position_donor_replace":
                            replacement = captures[(candidate_id, "wrong_position", donor["case_id"])]
                            replacement_source = "paired_donor_wrong_position"
                        elif condition == "channel_roll_donor_replace":
                            replacement = torch.roll(replacement, roll)
                            replacement_source = "channel_rolled_paired_donor"
                        elif condition == "same_case_restore":
                            replacement = recipient_state
                            replacement_source = "same_batch_no_op"
                        replacements.append(replacement)
                        roll_shifts.append(roll if condition == "channel_roll_donor_replace" else None)
                        replacement_sources.append(replacement_source)
                    replacement_tensor = torch.stack(replacements).to(
                        device=loaded.input_device, dtype=next(loaded.model.parameters()).dtype
                    )

                    def intervention_hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
                        if condition == "same_case_restore":
                            return output
                        primary = tensor_from_output(output).clone()
                        batch_index = torch.arange(primary.shape[0], device=primary.device)
                        primary[batch_index, positions_tensor, :] = replacement_tensor
                        return replace_primary(output, primary)

                    handle = component_module(
                        layers, layer_index, candidate["component"]
                    ).register_forward_hook(intervention_hook)
                    with torch.inference_mode():
                        result = loaded.model(**encoded, use_cache=False)
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
                            "schema_version": f"{contract['phase_id'].lower()}_causal_screen.v1",
                            "phase_id": contract["phase_id"],
                            "created_at": now(),
                            "model": MODEL,
                            "torch_dtype": run_dtype,
                            "split": contract["split"],
                            "candidate_id": candidate_id,
                            "boundary": candidate["boundary"],
                            "zone": candidate["zone"],
                            "layer": layer_index,
                            "component": candidate["component"],
                            "semantic_position": candidate["semantic_position"],
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
                                f"query{recipient['query_object_index']}_surface{recipient['surface_id']}_"
                                f"order{recipient['fact_order']}"
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
                            "intervention_donor_wins": max(scores, key=scores.get) == donor["target"],
                            "intervention_recipient_retained": max(scores, key=scores.get) == recipient["target"],
                            "roll_shift": roll_shifts[index],
                            "replacement_source": replacement_sources[index],
                            "source_recompute_intervention": True,
                            "screen_sufficiency_only": True,
                            "compute_edge": False,
                            "sealed": False,
                        })
                    append_jsonl(rows_path, output_rows)
                    completed += len(output_rows)
                    del result, logits, replacement_tensor, output_rows
                del baseline_result, baseline_logits, encoded
                if batch_start == 0 or completed == contract["expected_intervention_rows"] or (batch_start // batch_size) % 24 == 23:
                    print(
                        f"[{time.strftime('%H:%M:%S')}] qwen3 {contract['phase_id']} causal screen "
                        f"{completed}/{contract['expected_intervention_rows']}",
                        flush=True,
                    )

        final_rows = read_jsonl(rows_path)
        if len(final_rows) != contract["expected_intervention_rows"]:
            raise RuntimeError("Phase559 causal screen output denominator drift")
        summary = {
            "schema_version": f"{contract['phase_id'].lower()}_causal_screen_execution_summary.v1",
            "phase_id": contract["phase_id"],
            "created_at": now(),
            "status": "complete",
            "model": MODEL,
            "torch_dtype": run_dtype,
            "case_count": len(cases),
            "candidate_count": len(candidates),
            "condition_count": len(conditions),
            "intervention_row_count": len(final_rows),
            "runtime_seconds": time.monotonic() - started,
            "rows_sha256": sha256_file(rows_path),
            "source_recompute_intervention_executed": True,
            "deletion_executed": "paired_contrast_neutralize" in conditions,
            "head_channel_parameter_neuron_scan_executed": False,
            "sealed_split_read": False,
        }
        write_json(summary_path, summary)
        print(summary_path)
        return summary_path
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
