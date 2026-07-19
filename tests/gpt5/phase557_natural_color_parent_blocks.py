#!/usr/bin/env python3
"""Decompose replicated Phase557 object-source edges into coarse parent blocks."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))
os.environ.setdefault("PROBE_TORCH_DTYPE", "bfloat16")

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase557_natural_color_event_collect import (  # noqa: E402
    observer_prompt,
    semantic_positions,
    tensor_from_output,
)
from phase557_natural_color_source_intervention import word_scores  # noqa: E402


MODELS = ("qwen3", "glm4")
STAGES = {
    "parent_discovery": {"split": "behavior_confirmation", "surfaces": (0, 1)},
    "parent_confirmation": {"split": "unseen_recombination", "surfaces": (2, 3)},
}
COMPONENTS = ("layer_input", "attention_output", "mlp_output", "layer_output")
CONDITIONS = (
    "same_case_restore",
    "layer_output_donor_replace",
    "layer_input_donor_replace",
    "layer_input_delete",
    "layer_input_roll",
    "attention_output_donor_replace",
    "attention_output_delete",
    "attention_output_roll",
    "mlp_output_donor_replace",
    "mlp_output_delete",
    "mlp_output_roll",
)
OUT_DIR = ROOT / "tests/gpt5/result/phase557_fruit_composite"
CASES_PATH = OUT_DIR / "phase557_open_cases.jsonl"
CANDIDATE_REGISTRY = OUT_DIR / "phase557_natural_color_source_candidate_registry.json"
REPLICATED_EDGES = OUT_DIR / "phase557_replicated_natural_color_compute_edges.jsonl"


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def rows_path(model: str, stage: str) -> Path:
    return (
        OUT_DIR / "natural_color_parent_blocks" / model / stage
        / "phase557_natural_color_parent_rows.jsonl"
    )


def summary_path(model: str, stage: str) -> Path:
    return (
        OUT_DIR / "natural_color_parent_blocks" / model / stage
        / "phase557_natural_color_parent_summary.json"
    )


def replace_primary(output: Any, primary: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (primary, *output[1:])
    return primary


def condition_component(condition: str) -> str | None:
    if condition == "same_case_restore":
        return None
    if condition.startswith("layer_output"):
        return "layer_output"
    if condition.startswith("layer_input"):
        return "layer_input"
    if condition.startswith("attention_output"):
        return "attention_output"
    if condition.startswith("mlp_output"):
        return "mlp_output"
    raise ValueError(condition)


def correct_case_ids(model: str, split: str, surfaces: tuple[int, ...]) -> set[str]:
    return {
        row["case_id"] for row in read_jsonl(OUT_DIR / f"phase557_{model}_behavior_rows.jsonl")
        if row["split"] == split
        and row["case_type"] == "natural_parametric"
        and row["natural_relation"] == "color"
        and row["is_fruit"]
        and int(row["surface_id"]) in surfaces
        and row["semantic_correct"]
    }


def run(model_key: str, stage: str, batch_size: int, restart: bool) -> Path:
    spec = STAGES[stage]
    split = spec["split"]
    surfaces = tuple(spec["surfaces"])
    replicated_ids = {
        row["candidate_id"] for row in read_jsonl(REPLICATED_EDGES) if row["model"] == model_key
    }
    registry = read_json(CANDIDATE_REGISTRY)
    candidates = [
        row for row in registry["candidates"]
        if row["model"] == model_key and row["candidate_id"] in replicated_ids
    ]
    if not candidates:
        raise RuntimeError(f"No replicated Phase557 source edge for {model_key}")
    cases = [
        row for row in read_jsonl(CASES_PATH)
        if row["model"] == model_key
        and row["split"] == split
        and row["case_type"] == "natural_parametric"
        and row["natural_relation"] == "color"
        and row["is_fruit"]
        and int(row["surface_id"]) in surfaces
    ]
    if len(cases) != 24:
        raise RuntimeError(f"Expected 24 Phase557 parent-block cases, got {len(cases)}")
    correct_ids = correct_case_ids(model_key, split, surfaces)
    output = rows_path(model_key, stage)
    if restart:
        output.unlink(missing_ok=True)
        summary_path(model_key, stage).unlink(missing_ok=True)
    if output.exists():
        raise RuntimeError("Parent-block executor is non-resumable; use --restart")

    loaded = None
    started = time.monotonic()
    try:
        loaded = load_probe_model(model_key)
        loaded.tokenizer.padding_side = "left"
        layers = get_layers(loaded.model)
        run_dtype = str(next(loaded.model.parameters()).dtype)
        if run_dtype != "torch.bfloat16":
            raise RuntimeError(f"Phase557 parent blocks require BF16, got {run_dtype}")
        colors = sorted({row["target"] for row in cases})

        captures: dict[tuple[str, int, str], torch.Tensor] = {}
        for batch_start in range(0, len(cases), batch_size):
            batch_rows = cases[batch_start:batch_start + batch_size]
            individual = [semantic_positions(loaded.tokenizer, model_key, row) for row in batch_rows]
            texts = [observer_prompt(model_key, row["prompt"]) for row in batch_rows]
            encoded = loaded.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            sequence_length = int(encoded["input_ids"].shape[1])
            positions = []
            for row_index, (ids, semantic) in enumerate(individual):
                batch_ids = encoded["input_ids"][row_index][encoded["attention_mask"][row_index].bool()].tolist()
                if [int(value) for value in batch_ids] != ids:
                    raise RuntimeError("Phase557 parent capture tokenization drift")
                positions.append(sequence_length - len(ids) + semantic["object_source_end"])
            indices = torch.tensor(positions, dtype=torch.long, device=loaded.input_device)
            handles = []

            def store(component: str, layer_index: int, value: torch.Tensor) -> None:
                batch_index = torch.arange(value.shape[0], device=value.device)
                selected = value[batch_index, indices, :].detach().float().cpu()
                for index, row in enumerate(batch_rows):
                    captures[(component, layer_index, row["case_id"])] = selected[index]

            def make_pre(layer_index: int):
                def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
                    store("layer_input", layer_index, inputs[0])
                return hook

            def make_forward(component: str, layer_index: int):
                def hook(_module: Any, _inputs: tuple[Any, ...], output_value: Any) -> None:
                    store(component, layer_index, tensor_from_output(output_value))
                return hook

            for candidate in candidates:
                layer_index = int(candidate["layer"])
                layer = layers[layer_index]
                handles.append(layer.register_forward_pre_hook(make_pre(layer_index)))
                handles.append(layer.self_attn.register_forward_hook(make_forward("attention_output", layer_index)))
                handles.append(layer.mlp.register_forward_hook(make_forward("mlp_output", layer_index)))
                handles.append(layer.register_forward_hook(make_forward("layer_output", layer_index)))
            encoded = {key: value.to(loaded.input_device) for key, value in encoded.items()}
            with torch.inference_mode():
                result = loaded.model(**encoded, use_cache=False)
            for handle in handles:
                handle.remove()
            del result, encoded

        groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in cases:
            groups[(int(row["surface_id"]), int(row["fact_order"]))].append(row)
        tasks: list[dict[str, Any]] = []
        for candidate in candidates:
            layer_index = int(candidate["layer"])
            for group_key, group_rows in sorted(groups.items()):
                eligible = sorted(
                    [row for row in group_rows if row["case_id"] in correct_ids],
                    key=lambda row: row["object_label"],
                )
                if len(eligible) < 3:
                    continue
                centroids = {
                    component: torch.stack([
                        captures[(component, layer_index, row["case_id"])] for row in group_rows
                    ]).mean(dim=0)
                    for component in COMPONENTS
                }
                for recipient in eligible:
                    for donor in eligible:
                        if recipient["case_id"] == donor["case_id"] or recipient["target"] == donor["target"]:
                            continue
                        for condition in CONDITIONS:
                            component = condition_component(condition)
                            replacement = None
                            roll_shift = None
                            if component is not None:
                                donor_state = captures[(component, layer_index, donor["case_id"])]
                                if condition.endswith("_delete"):
                                    replacement = centroids[component]
                                elif condition.endswith("_roll"):
                                    width = donor_state.numel()
                                    value = int(hashlib.sha256(
                                        f"{candidate['candidate_id']}|{component}|{recipient['case_id']}".encode("utf-8")
                                    ).hexdigest()[:8], 16)
                                    roll_shift = 1 + value % max(1, width - 1)
                                    replacement = torch.roll(donor_state, roll_shift)
                                else:
                                    replacement = donor_state
                            tasks.append({
                                "candidate": candidate,
                                "recipient": recipient,
                                "donor": donor,
                                "group_key": group_key,
                                "condition": condition,
                                "component": component,
                                "replacement": replacement,
                                "roll_shift": roll_shift,
                            })
        if not tasks:
            raise RuntimeError(f"No Phase557 parent-block tasks for {model_key}/{stage}")

        completed = 0
        task_groups = defaultdict(list)
        for task in tasks:
            task_groups[(task["candidate"]["candidate_id"], task["condition"])].append(task)
        for (_candidate_id, condition), condition_tasks in sorted(task_groups.items()):
            for batch_start in range(0, len(condition_tasks), batch_size):
                batch_tasks = condition_tasks[batch_start:batch_start + batch_size]
                batch_rows = [task["recipient"] for task in batch_tasks]
                individual = [semantic_positions(loaded.tokenizer, model_key, row) for row in batch_rows]
                texts = [observer_prompt(model_key, row["prompt"]) for row in batch_rows]
                encoded = loaded.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
                sequence_length = int(encoded["input_ids"].shape[1])
                positions = []
                for row_index, (ids, semantic) in enumerate(individual):
                    batch_ids = encoded["input_ids"][row_index][encoded["attention_mask"][row_index].bool()].tolist()
                    if [int(value) for value in batch_ids] != ids:
                        raise RuntimeError("Phase557 parent intervention tokenization drift")
                    positions.append(sequence_length - len(ids) + semantic["object_source_end"])
                indices = torch.tensor(positions, dtype=torch.long, device=loaded.input_device)
                layer_index = int(batch_tasks[0]["candidate"]["layer"])
                component = batch_tasks[0]["component"]
                replacements = None
                if component is not None:
                    replacements = torch.stack([task["replacement"] for task in batch_tasks]).to(
                        device=loaded.input_device, dtype=next(loaded.model.parameters()).dtype
                    )

                def modify(value: torch.Tensor) -> torch.Tensor:
                    primary = value.clone()
                    if replacements is not None:
                        batch_index = torch.arange(primary.shape[0], device=primary.device)
                        primary[batch_index, indices, :] = replacements
                    return primary

                def pre_hook(_module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
                    return (modify(inputs[0]), *inputs[1:])

                def forward_hook(_module: Any, _inputs: tuple[Any, ...], output_value: Any) -> Any:
                    return replace_primary(output_value, modify(tensor_from_output(output_value)))

                encoded = {key: value.to(loaded.input_device) for key, value in encoded.items()}
                with torch.inference_mode():
                    baseline_result = loaded.model(**encoded, use_cache=False)
                handle = None
                if component == "layer_input":
                    handle = layers[layer_index].register_forward_pre_hook(pre_hook)
                elif component == "attention_output":
                    handle = layers[layer_index].self_attn.register_forward_hook(forward_hook)
                elif component == "mlp_output":
                    handle = layers[layer_index].mlp.register_forward_hook(forward_hook)
                elif component == "layer_output":
                    handle = layers[layer_index].register_forward_hook(forward_hook)
                with torch.inference_mode():
                    result = loaded.model(**encoded, use_cache=False)
                if handle is not None:
                    handle.remove()
                baseline_logits = baseline_result.logits[:, -1, :].detach().float().cpu()
                logits = result.logits[:, -1, :].detach().float().cpu()
                output_rows = []
                for index, task in enumerate(batch_tasks):
                    recipient = task["recipient"]
                    donor = task["donor"]
                    baseline = word_scores(baseline_logits[index], loaded.tokenizer, colors)
                    scores = word_scores(logits[index], loaded.tokenizer, colors)
                    baseline_switch = baseline[donor["target"]] - baseline[recipient["target"]]
                    intervention_switch = scores[donor["target"]] - scores[recipient["target"]]
                    prediction = max(scores, key=scores.get)
                    output_rows.append({
                        "schema_version": "phase557_natural_color_parent_block.v1",
                        "phase_id": "Phase557",
                        "created_at": now(),
                        "model": model_key,
                        "torch_dtype": run_dtype,
                        "stage": stage,
                        "split": split,
                        "surfaces": list(surfaces),
                        "candidate_id": task["candidate"]["candidate_id"],
                        "layer": layer_index,
                        "condition": condition,
                        "parent_component": component,
                        "recipient_case_id": recipient["case_id"],
                        "recipient_object": recipient["object_label"],
                        "recipient_color": recipient["target"],
                        "donor_case_id": donor["case_id"],
                        "donor_object": donor["object_label"],
                        "donor_color": donor["target"],
                        "surface_id": task["group_key"][0],
                        "fact_order": task["group_key"][1],
                        "baseline_scores": baseline,
                        "intervention_scores": scores,
                        "donor_switch_effect": intervention_switch - baseline_switch,
                        "intervention_donor_wins": prediction == donor["target"],
                        "intervention_recipient_retained": prediction == recipient["target"],
                        "roll_shift": task["roll_shift"],
                        "source_recompute_intervention": True,
                        "qualified_parent_block": False,
                        "sealed": False,
                    })
                append_jsonl(output, output_rows)
                completed += len(output_rows)
                del baseline_result, result, encoded, output_rows
                if completed == len(tasks) or completed % 440 < batch_size:
                    print(
                        f"[{time.strftime('%H:%M:%S')}] {model_key}/{stage} parent blocks "
                        f"{completed}/{len(tasks)}",
                        flush=True,
                    )

        final_rows = read_jsonl(output)
        if len(final_rows) != len(tasks):
            raise RuntimeError(f"Incomplete Phase557 parent rows: {len(final_rows)}/{len(tasks)}")
        summary = {
            "schema_version": "phase557_natural_color_parent_summary.v1",
            "phase_id": "Phase557",
            "created_at": now(),
            "status": "complete",
            "model": model_key,
            "stage": stage,
            "split": split,
            "surfaces": list(surfaces),
            "torch_dtype": run_dtype,
            "behavior_correct_case_count": len(correct_ids),
            "candidate_count": len(candidates),
            "condition_count": len(CONDITIONS),
            "row_count": len(final_rows),
            "runtime_seconds": time.monotonic() - started,
            "rows_path": str(output.relative_to(ROOT)),
            "rows_sha256": sha256_file(output),
            "source_parent_recompute_executed": True,
            "head_channel_parameter_neuron_scan_executed": False,
            "sealed_split_read": False,
        }
        write_json(summary_path(model_key, stage), summary)
        print(summary_path(model_key, stage))
        return summary_path(model_key, stage)
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    run(args.model, args.stage, args.batch_size, args.restart)


if __name__ == "__main__":
    main()
