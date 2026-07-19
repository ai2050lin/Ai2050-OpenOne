#!/usr/bin/env python3
"""Locate the earliest held-out layer-input factor-transfer boundary."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase556_fruit_encoding_protocol import CELLS  # noqa: E402
from phase556_fruit_causal_intervention import (  # noqa: E402
    finite_or_none,
    matched_pairs,
    observer_prompt,
    READOUT_CONTRACT,
    safe_fraction,
    scores_are_valid,
    word_scores,
)


OUT_DIR = ROOT / "tests/gpt5/result/phase556_fruit_encoding"
CASES_PATH = OUT_DIR / "phase556_open_cases.jsonl"
BEHAVIOR_PATH = OUT_DIR / "phase556_qwen3_behavior_rows.jsonl"
MODEL = "qwen3"
SPLIT_OFFSETS = {
    "boundary_discovery": (12, 24),
    "boundary_confirmation": (24, 36),
}
MECHANISMS = ("category_reuse", "attribute_binding")
SCENARIOS = ("matched_factor_delta", "channel_roll_delta")
LAYER_GRID = (0, 4, 8, 12, 16, 20, 24, 28, 29, 30, 31, 32, 33, 34, 35)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def output_path(split: str) -> Path:
    return OUT_DIR / "layer_input_boundary" / split / "phase556_boundary_rows.jsonl"


def summary_path(split: str) -> Path:
    return OUT_DIR / "layer_input_boundary" / split / "phase556_boundary_execution_summary.json"


def qualified_anchor_slice(split: str) -> list[str]:
    behavior = read_jsonl(BEHAVIOR_PATH)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in behavior:
        if (
            row["split"] == "independent_confirmation"
            and row["case_type"] == "controlled_factorial"
            and 48 <= int(row["world_index"]) < 96
        ):
            grouped.setdefault(row["anchor_id"], []).append(row)
    qualified = sorted(
        anchor for anchor, rows in grouped.items()
        if len(rows) == 16 and all(row["semantic_correct"] for row in rows)
    )
    start, stop = SPLIT_OFFSETS[split]
    selected = qualified[start:stop]
    if len(selected) != stop - start:
        raise RuntimeError(f"Insufficient Phase556 {MODEL} boundary anchors: {len(selected)}")
    return selected


def run(split: str, restart: bool, batch_size: int) -> Path:
    anchors = qualified_anchor_slice(split)
    cases = [
        row for row in read_jsonl(CASES_PATH)
        if row["model"] == MODEL
        and row["split"] == "independent_confirmation"
        and row["case_type"] == "controlled_factorial"
        and row["anchor_id"] in anchors
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in cases:
        grouped.setdefault(row["anchor_id"], []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: CELLS.index(row["factorial_cell"]))

    output = output_path(split)
    if restart:
        output.unlink(missing_ok=True)
        summary_path(split).unlink(missing_ok=True)
    completed = {row["anchor_id"] for row in read_jsonl(output)} if output.exists() else set()
    loaded = None
    started = time.monotonic()
    try:
        loaded = load_probe_model(MODEL)
        loaded.tokenizer.padding_side = "left"
        layers = get_layers(loaded.model)
        if max(LAYER_GRID) >= len(layers):
            raise RuntimeError(f"Phase556 boundary layer grid exceeds {len(layers)} layers")
        run_dtype = str(next(loaded.model.parameters()).dtype)
        new_anchor_count = 0
        for anchor_index, (anchor_id, rows) in enumerate(sorted(grouped.items()), 1):
            if anchor_id in completed:
                continue
            captures: dict[int, torch.Tensor] = {}
            handles = []
            for layer_index in LAYER_GRID:
                def make_hook(index: int):
                    def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
                        captures[index] = inputs[0][:, -1, :].detach().float().cpu()
                    return hook
                handles.append(layers[layer_index].register_forward_pre_hook(make_hook(layer_index)))
            encoded = loaded.tokenizer(
                [observer_prompt(MODEL, row["prompt"]) for row in rows],
                return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            )
            encoded = {key: value.to(loaded.input_device) for key, value in encoded.items()}
            with torch.inference_mode():
                natural = loaded.model(**encoded, use_cache=False)
            natural_logits = natural.logits[:, -1, :].detach().float().cpu()
            for handle in handles:
                handle.remove()
            if set(captures) != set(LAYER_GRID):
                raise RuntimeError("Phase556 boundary capture mismatch")
            case_index = {row["case_id"]: index for index, row in enumerate(rows)}
            baseline_scores = {
                row["case_id"]: word_scores(
                    natural_logits[index], loaded.tokenizer, sorted(set(row["all_candidates"]))
                )
                for index, row in enumerate(rows)
            }
            anchor_output: list[dict[str, Any]] = []
            for layer_index in LAYER_GRID:
                for mechanism in MECHANISMS:
                    tasks: list[dict[str, Any]] = []
                    for pair in matched_pairs(rows, mechanism):
                        directions = (
                            ("factor_0_to_1", pair["recipient"], pair["donor"]),
                            ("factor_1_to_0", pair["donor"], pair["recipient"]),
                        )
                        for direction, recipient, donor in directions:
                            recipient_index = case_index[recipient["case_id"]]
                            donor_index = case_index[donor["case_id"]]
                            delta = captures[layer_index][donor_index] - captures[layer_index][recipient_index]
                            shift = 1 + int(hashlib.sha256(
                                f"{layer_index}|{anchor_id}|{recipient['case_id']}".encode("utf-8")
                            ).hexdigest()[:8], 16) % max(1, delta.numel() - 1)
                            for scenario in SCENARIOS:
                                tasks.append({
                                    "factor": pair["factor"],
                                    "pair_role": pair["pair_role"],
                                    "recipient": recipient,
                                    "donor": donor,
                                    "intervention_direction": direction,
                                    "scenario": scenario,
                                    "delta": delta if scenario == "matched_factor_delta" else torch.roll(delta, shift, -1),
                                    "roll_shift": shift if scenario == "channel_roll_delta" else None,
                                })
                    for batch_start in range(0, len(tasks), batch_size):
                        batch_tasks = tasks[batch_start:batch_start + batch_size]
                        batch = loaded.tokenizer(
                            [observer_prompt(MODEL, task["recipient"]["prompt"]) for task in batch_tasks],
                            return_tensors="pt", padding=True, truncation=True, max_length=512,
                        )
                        batch = {key: value.to(loaded.input_device) for key, value in batch.items()}
                        deltas = torch.stack([task["delta"] for task in batch_tasks]).to(loaded.input_device)

                        def patch_hook(_module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
                            hidden = inputs[0].clone()
                            hidden[:, -1, :] = hidden[:, -1, :] + deltas.to(hidden.dtype)
                            return (hidden, *inputs[1:])

                        handle = layers[layer_index].register_forward_pre_hook(patch_hook)
                        with torch.inference_mode():
                            patched = loaded.model(**batch, use_cache=False)
                        handle.remove()
                        logits = patched.logits[:, -1, :].detach().float().cpu()
                        for offset, task in enumerate(batch_tasks):
                            recipient = task["recipient"]
                            donor = task["donor"]
                            words = sorted(set(recipient["all_candidates"] + donor["all_candidates"]))
                            recipient_scores = baseline_scores[recipient["case_id"]]
                            donor_scores = baseline_scores[donor["case_id"]]
                            patched_scores = word_scores(logits[offset], loaded.tokenizer, words)
                            baseline_valid = scores_are_valid(recipient_scores)
                            donor_valid = scores_are_valid(donor_scores)
                            patched_valid = scores_are_valid(patched_scores)
                            valid = baseline_valid and donor_valid and patched_valid
                            recipient_target = recipient["target"]
                            donor_target = donor["target"]
                            baseline_choice = max(recipient_scores, key=recipient_scores.get) if baseline_valid else None
                            donor_choice = max(donor_scores, key=donor_scores.get) if donor_valid else None
                            patched_choice = max(patched_scores, key=patched_scores.get) if patched_valid else None
                            if recipient_target != donor_target and valid:
                                base_margin = recipient_scores[donor_target] - recipient_scores[recipient_target]
                                donor_margin = donor_scores[donor_target] - donor_scores[recipient_target]
                                patch_margin = patched_scores[donor_target] - patched_scores[recipient_target]
                                fraction = safe_fraction(patch_margin - base_margin, donor_margin - base_margin)
                            else:
                                base_margin = donor_margin = patch_margin = fraction = None
                            anchor_output.append({
                                "schema_version": "phase556_layer_input_boundary_row.v1",
                                "phase_id": "Phase556",
                                "created_at": now(),
                                "model": MODEL,
                                "torch_dtype": run_dtype,
                                "split": split,
                                "anchor_id": anchor_id,
                                "world_index": int(recipient["world_index"]),
                                "mechanism": mechanism,
                                "component": "layer_input",
                                "layer": layer_index,
                                "layer_count": len(layers),
                                "relative_depth": layer_index / max(1, len(layers) - 1),
                                "pair_role": task["pair_role"],
                                "scenario": task["scenario"],
                                "intervention_direction": task["intervention_direction"],
                                "recipient_case_id": recipient["case_id"],
                                "donor_case_id": donor["case_id"],
                                "recipient_target": recipient_target,
                                "donor_target": donor_target,
                                "baseline_choice": baseline_choice,
                                "natural_donor_choice": donor_choice,
                                "patched_choice": patched_choice,
                                "baseline_semantic_correct_restricted": baseline_choice == recipient_target,
                                "natural_donor_semantic_correct_restricted": donor_choice == donor_target,
                                "patched_donor_selected": patched_choice == donor_target,
                                "patched_recipient_preserved": patched_choice == recipient_target,
                                "baseline_donor_margin": base_margin,
                                "natural_donor_margin": donor_margin,
                                "patched_donor_margin": patch_margin,
                                "transfer_fraction": fraction,
                                "patch_delta_norm": finite_or_none(float(task["delta"].norm().item())),
                                "channel_roll_shift": task["roll_shift"],
                                "numerical_valid": valid,
                                "candidate_selected_without_this_anchor": True,
                                "restricted_readout_contract": READOUT_CONTRACT,
                                "semantic_position": (
                                    "answer_content_boundary_after_natural_newline"
                                    if MODEL == "glm4" else "query_end"
                                ),
                                "observer_prefix": "\n" if MODEL == "glm4" else "",
                                "compute_edge": False,
                                "causal_qualified": False,
                                "sealed": False,
                            })
                        del patched, logits, batch, deltas
            append_jsonl(output, anchor_output)
            new_anchor_count += 1
            del natural, natural_logits, encoded, captures, anchor_output
            if new_anchor_count == 1 or new_anchor_count % 4 == 0 or anchor_index == len(grouped):
                print(
                    f"[{time.strftime('%H:%M:%S')}] {MODEL} Phase556 "
                    f"{split} {new_anchor_count}/{len(grouped)}",
                    flush=True,
                )
        final_rows = read_jsonl(output)
        expected = len(grouped) * len(LAYER_GRID) * len(MECHANISMS) * 16 * len(SCENARIOS)
        if len(final_rows) != expected:
            raise RuntimeError(f"Incomplete Phase556 boundary rows: {len(final_rows)}/{expected}")
        summary = {
            "schema_version": "phase556_layer_input_boundary_execution_summary.v1",
            "phase_id": "Phase556",
            "created_at": now(),
            "status": "complete",
            "model": MODEL,
            "torch_dtype": run_dtype,
            "split": split,
            "anchor_count": len(grouped),
            "anchor_offset": list(SPLIT_OFFSETS[split]),
            "layer_grid": list(LAYER_GRID),
            "row_count": len(final_rows),
            "runtime_seconds": time.monotonic() - started,
            "rows_sha256": sha256_file(output),
            "restricted_readout_contract": READOUT_CONTRACT,
            "sealed_split_read": False,
        }
        write_json(summary_path(split), summary)
        print(summary_path(split))
        return summary_path(split)
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=tuple(SPLIT_OFFSETS))
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    run(args.split, args.restart, args.batch_size)
