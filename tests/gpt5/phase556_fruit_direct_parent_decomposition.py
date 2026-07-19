#!/usr/bin/env python3
"""Decompose the direct parents of replicated Phase556 boundaries."""

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
    tensor_from_output,
    word_scores,
)


OUT_DIR = ROOT / "tests/gpt5/result/phase556_fruit_encoding"
CASES_PATH = OUT_DIR / "phase556_open_cases.jsonl"
BEHAVIOR_PATH = OUT_DIR / "phase556_qwen3_behavior_rows.jsonl"
BOUNDARY_PATH = OUT_DIR / "phase556_layer_input_boundary_analysis.json"
OUTPUT = OUT_DIR / "direct_parent_decomposition/phase556_direct_parent_rows.jsonl"
SUMMARY = OUT_DIR / "direct_parent_decomposition/phase556_direct_parent_execution_summary.json"
MODEL = "qwen3"
ANCHOR_SLICE = (36, 44)
CONDITIONS = (
    "residual_carry_delta",
    "attention_write_delta",
    "mlp_write_delta",
    "attention_mlp_joint_delta",
    "all_parent_delta",
    "channel_roll_all_parent",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def qualified_anchors() -> list[str]:
    behavior = read_jsonl(BEHAVIOR_PATH)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in behavior:
        if (
            row["split"] == "independent_confirmation"
            and row["case_type"] == "controlled_factorial"
            and 48 <= int(row["world_index"]) < 96
        ):
            grouped.setdefault(row["anchor_id"], []).append(row)
    anchors = sorted(
        anchor for anchor, rows in grouped.items()
        if len(rows) == 16 and all(row["semantic_correct"] for row in rows)
    )
    selected = anchors[slice(*ANCHOR_SLICE)]
    if len(selected) != ANCHOR_SLICE[1] - ANCHOR_SLICE[0]:
        raise RuntimeError(f"Insufficient Phase556 parent anchors: {len(selected)}")
    return selected


def run(restart: bool, batch_size: int) -> Path:
    boundary = read_json(BOUNDARY_PATH)
    target_layers = {
        mechanism: int(report["earliest_replicated_layer"])
        for mechanism, report in boundary["mechanism_reports"].items()
        if report["earliest_replicated_layer"] is not None
    }
    if not target_layers:
        raise RuntimeError("No replicated Phase556 boundary authorized for parent decomposition")
    parent_layers = {mechanism: layer - 1 for mechanism, layer in target_layers.items()}
    if min(parent_layers.values()) < 0:
        raise RuntimeError("Cannot decompose a layer-zero boundary")
    anchors = qualified_anchors()
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
    if restart:
        OUTPUT.unlink(missing_ok=True)
        SUMMARY.unlink(missing_ok=True)
    completed = {row["anchor_id"] for row in read_jsonl(OUTPUT)} if OUTPUT.exists() else set()
    loaded = None
    started = time.monotonic()
    try:
        loaded = load_probe_model(MODEL)
        loaded.tokenizer.padding_side = "left"
        layers = get_layers(loaded.model)
        run_dtype = str(next(loaded.model.parameters()).dtype)
        new_anchor_count = 0
        for anchor_id, rows in sorted(grouped.items()):
            if anchor_id in completed:
                continue
            captures: dict[tuple[int, str], torch.Tensor] = {}
            handles = []
            for parent_layer in sorted(set(parent_layers.values())):
                def make_pre(layer_index: int):
                    def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
                        captures[(layer_index, "layer_input")] = inputs[0][:, -1, :].detach().float().cpu()
                    return hook
                def make_forward(layer_index: int, component: str):
                    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
                        captures[(layer_index, component)] = tensor_from_output(output)[:, -1, :].detach().float().cpu()
                    return hook
                handles.extend([
                    layers[parent_layer].register_forward_pre_hook(make_pre(parent_layer)),
                    layers[parent_layer].self_attn.register_forward_hook(make_forward(parent_layer, "attention_output")),
                    layers[parent_layer].mlp.register_forward_hook(make_forward(parent_layer, "mlp_output")),
                ])
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
            case_index = {row["case_id"]: index for index, row in enumerate(rows)}
            baseline_scores = {
                row["case_id"]: word_scores(
                    natural_logits[index], loaded.tokenizer, sorted(set(row["all_candidates"]))
                )
                for index, row in enumerate(rows)
            }
            anchor_output: list[dict[str, Any]] = []
            for mechanism, boundary_layer in sorted(target_layers.items()):
                parent_layer = parent_layers[mechanism]
                tasks: list[dict[str, Any]] = []
                for pair in matched_pairs(rows, mechanism):
                    for direction, recipient, donor in (
                        ("factor_0_to_1", pair["recipient"], pair["donor"]),
                        ("factor_1_to_0", pair["donor"], pair["recipient"]),
                    ):
                        recipient_index = case_index[recipient["case_id"]]
                        donor_index = case_index[donor["case_id"]]
                        deltas = {
                            component: captures[(parent_layer, component)][donor_index]
                            - captures[(parent_layer, component)][recipient_index]
                            for component in ("layer_input", "attention_output", "mlp_output")
                        }
                        shift = 1 + int(hashlib.sha256(
                            f"{mechanism}|{anchor_id}|{recipient['case_id']}".encode("utf-8")
                        ).hexdigest()[:8], 16) % max(1, deltas["layer_input"].numel() - 1)
                        for condition in CONDITIONS:
                            tasks.append({
                                "factor": pair["factor"],
                                "pair_role": pair["pair_role"],
                                "recipient": recipient,
                                "donor": donor,
                                "direction": direction,
                                "condition": condition,
                                "deltas": deltas,
                                "roll_shift": shift,
                            })
                for batch_start in range(0, len(tasks), batch_size):
                    batch_tasks = tasks[batch_start:batch_start + batch_size]
                    batch = loaded.tokenizer(
                        [observer_prompt(MODEL, task["recipient"]["prompt"]) for task in batch_tasks],
                        return_tensors="pt", padding=True, truncation=True, max_length=512,
                    )
                    batch = {key: value.to(loaded.input_device) for key, value in batch.items()}
                    active_components = {
                        "residual_carry_delta": ("layer_input",),
                        "attention_write_delta": ("attention_output",),
                        "mlp_write_delta": ("mlp_output",),
                        "attention_mlp_joint_delta": ("attention_output", "mlp_output"),
                        "all_parent_delta": ("layer_input", "attention_output", "mlp_output"),
                        "channel_roll_all_parent": ("layer_input", "attention_output", "mlp_output"),
                    }
                    child_state_deltas = []
                    for task in batch_tasks:
                        delta = sum(
                            (task["deltas"][component] for component in active_components[task["condition"]]),
                            torch.zeros_like(task["deltas"]["layer_input"]),
                        )
                        if task["condition"] == "channel_roll_all_parent":
                            delta = torch.roll(delta, task["roll_shift"], -1)
                        child_state_deltas.append(delta)
                    child_state_delta = torch.stack(child_state_deltas).to(loaded.input_device)

                    def boundary_pre_hook(_module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
                        hidden = inputs[0].clone()
                        hidden[:, -1, :] += child_state_delta.to(hidden.dtype)
                        return (hidden, *inputs[1:])

                    patch_handle = layers[boundary_layer].register_forward_pre_hook(
                        boundary_pre_hook
                    )
                    with torch.inference_mode():
                        patched = loaded.model(**batch, use_cache=False)
                    patch_handle.remove()
                    logits = patched.logits[:, -1, :].detach().float().cpu()
                    for offset, task in enumerate(batch_tasks):
                        recipient, donor = task["recipient"], task["donor"]
                        words = sorted(set(recipient["all_candidates"] + donor["all_candidates"]))
                        recipient_scores = baseline_scores[recipient["case_id"]]
                        donor_scores = baseline_scores[donor["case_id"]]
                        patched_scores = word_scores(logits[offset], loaded.tokenizer, words)
                        baseline_valid = scores_are_valid(recipient_scores)
                        donor_valid = scores_are_valid(donor_scores)
                        patched_valid = scores_are_valid(patched_scores)
                        valid = baseline_valid and donor_valid and patched_valid
                        recipient_target, donor_target = recipient["target"], donor["target"]
                        baseline_choice = max(recipient_scores, key=recipient_scores.get) if baseline_valid else None
                        donor_choice = max(donor_scores, key=donor_scores.get) if donor_valid else None
                        patched_choice = max(patched_scores, key=patched_scores.get) if patched_valid else None
                        if recipient_target != donor_target and valid:
                            base_margin = recipient_scores[donor_target] - recipient_scores[recipient_target]
                            donor_margin = donor_scores[donor_target] - donor_scores[recipient_target]
                            patch_margin = patched_scores[donor_target] - patched_scores[recipient_target]
                            fraction = safe_fraction(patch_margin - base_margin, donor_margin - base_margin)
                        else:
                            fraction = None
                        anchor_output.append({
                            "schema_version": "phase556_direct_parent_decomposition_row.v2",
                            "phase_id": "Phase556",
                            "created_at": now(),
                            "model": MODEL,
                            "torch_dtype": run_dtype,
                            "split": "independent_parent_holdout",
                            "anchor_id": anchor_id,
                            "mechanism": mechanism,
                            "boundary_layer": boundary_layer,
                            "parent_layer": parent_layer,
                            "condition": task["condition"],
                            "pair_role": task["pair_role"],
                            "intervention_direction": task["direction"],
                            "recipient_target": recipient_target,
                            "donor_target": donor_target,
                            "baseline_choice": baseline_choice,
                            "natural_donor_choice": donor_choice,
                            "patched_choice": patched_choice,
                            "baseline_semantic_correct_restricted": baseline_choice == recipient_target,
                            "natural_donor_semantic_correct_restricted": donor_choice == donor_target,
                            "patched_donor_selected": patched_choice == donor_target,
                            "patched_recipient_preserved": patched_choice == recipient_target,
                            "transfer_fraction": fraction,
                            "numerical_valid": valid,
                            "parent_delta_norms": {
                                component: finite_or_none(float(task["deltas"][component].norm().item()))
                                for component in task["deltas"]
                            },
                            "child_state_patch_delta_norm": finite_or_none(
                                float(child_state_deltas[offset].norm().item())
                            ),
                            "intervention_location": "boundary_layer_input",
                            "parent_intervention_semantics": (
                                "additive_parent_component_delta_at_child_state"
                            ),
                            "additive_ledger_decomposition": True,
                            "compute_edge": False,
                            "restricted_readout_contract": READOUT_CONTRACT,
                            "semantic_position": (
                                "answer_content_boundary_after_natural_newline"
                                if MODEL == "glm4" else "query_end"
                            ),
                            "observer_prefix": "\n" if MODEL == "glm4" else "",
                            "causal_qualified": False,
                            "sealed": False,
                        })
                    del patched, logits, batch, child_state_delta, child_state_deltas
            append_jsonl(OUTPUT, anchor_output)
            new_anchor_count += 1
            del natural, natural_logits, encoded, captures, anchor_output
            if new_anchor_count == 1 or new_anchor_count % 4 == 0 or new_anchor_count == len(grouped):
                print(
                    f"[{time.strftime('%H:%M:%S')}] {MODEL} Phase556 parent "
                    f"{new_anchor_count}/{len(grouped)}",
                    flush=True,
                )
        final_rows = read_jsonl(OUTPUT)
        expected = len(grouped) * len(target_layers) * 16 * len(CONDITIONS)
        if len(final_rows) != expected:
            raise RuntimeError(f"Incomplete Phase556 parent rows: {len(final_rows)}/{expected}")
        summary = {
            "schema_version": "phase556_direct_parent_execution_summary.v2",
            "phase_id": "Phase556",
            "created_at": now(),
            "status": "complete",
            "model": MODEL,
            "torch_dtype": run_dtype,
            "anchor_count": len(grouped),
            "anchor_slice": list(ANCHOR_SLICE),
            "target_layers": target_layers,
            "parent_layers": parent_layers,
            "conditions": list(CONDITIONS),
            "intervention_location": "boundary_layer_input",
            "parent_intervention_semantics": "additive_parent_component_delta_at_child_state",
            "row_count": len(final_rows),
            "runtime_seconds": time.monotonic() - started,
            "restricted_readout_contract": READOUT_CONTRACT,
            "sealed_split_read": False,
        }
        write_json(SUMMARY, summary)
        print(SUMMARY)
        return SUMMARY
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    run(args.restart, args.batch_size)
