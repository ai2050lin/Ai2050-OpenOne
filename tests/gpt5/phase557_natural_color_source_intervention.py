#!/usr/bin/env python3
"""Test legal object-source recompute edges for Phase557 natural color.

The intervention replaces the layer-output state at the fruit-name token and
lets all later layers and positions recompute naturally. It never edits the
query-end state, parameters, heads, channels, or sealed examples.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
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


PHASE = "Phase557"
MODELS = ("qwen3", "glm4")
CONFIRMATION_SPLIT = "behavior_confirmation"
UNSEEN_SPLIT = "unseen_recombination"
SPLITS = (CONFIRMATION_SPLIT, UNSEEN_SPLIT)
OUT_DIR = ROOT / "tests/gpt5/result/phase557_fruit_composite"
CASES_PATH = OUT_DIR / "phase557_open_cases.jsonl"
REGISTRY_PATH = OUT_DIR / "phase557_natural_color_source_candidate_registry.json"
CONDITIONS = (
    "same_case_restore",
    "object_specific_delete",
    "correct_donor_replace",
    "wrong_depth_donor_replace",
    "relation_position_donor_replace",
    "channel_roll_donor_replace",
)
READOUT = "restricted_first_non_whitespace_color_token"


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def rows_path(model: str, split: str) -> Path:
    if split == CONFIRMATION_SPLIT:
        return OUT_DIR / "natural_color_source" / model / "phase557_natural_color_source_rows.jsonl"
    return (
        OUT_DIR / "natural_color_source" / model / split
        / "phase557_natural_color_source_rows.jsonl"
    )


def summary_path(model: str, split: str) -> Path:
    if split == CONFIRMATION_SPLIT:
        return OUT_DIR / "natural_color_source" / model / "phase557_natural_color_source_summary.json"
    return (
        OUT_DIR / "natural_color_source" / model / split
        / "phase557_natural_color_source_summary.json"
    )


def replace_primary(output: Any, primary: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (primary, *output[1:])
    return primary


def word_token_ids(tokenizer: Any, word: str) -> list[int]:
    ids: set[int] = set()
    for text in (word, " " + word, "\n" + word):
        encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
        for token_id in encoded:
            piece = tokenizer.decode([int(token_id)], skip_special_tokens=False)
            if piece.strip():
                ids.add(int(token_id))
                break
    if not ids:
        raise ValueError(f"No content token for {word!r}")
    return sorted(ids)


def word_scores(logits: torch.Tensor, tokenizer: Any, words: list[str]) -> dict[str, float]:
    token_sets = {word: set(word_token_ids(tokenizer, word)) for word in words}
    for index, word in enumerate(words):
        for other in words[index + 1:]:
            overlap = token_sets[word] & token_sets[other]
            if overlap:
                raise ValueError(f"Color first-token collision: {word}/{other}: {sorted(overlap)}")
    return {
        word: float(logits[sorted(token_sets[word])].float().max().item())
        for word in words
    }


def deterministic_roll(candidate_id: str, recipient_id: str, width: int) -> int:
    if width <= 1:
        return 0
    value = int(hashlib.sha256(f"{candidate_id}|{recipient_id}".encode("utf-8")).hexdigest()[:8], 16)
    return 1 + value % (width - 1)


def semantic_correct_cases(model: str, split: str) -> set[str]:
    rows = read_jsonl(OUT_DIR / f"phase557_{model}_behavior_rows.jsonl")
    return {
        row["case_id"] for row in rows
        if row["split"] == split
        and row["case_type"] == "natural_parametric"
        and row["natural_relation"] == "color"
        and row["is_fruit"]
        and row["semantic_correct"]
    }


def run(model_key: str, split: str, batch_size: int, restart: bool) -> Path:
    registry = read_json(REGISTRY_PATH)
    candidates = [row for row in registry["candidates"] if row["model"] == model_key]
    if split == UNSEEN_SPLIT:
        qualified_path = OUT_DIR / "phase557_qualified_natural_color_compute_edges.jsonl"
        qualified_ids = {
            row["candidate_id"] for row in read_jsonl(qualified_path) if row["model"] == model_key
        }
        candidates = [row for row in candidates if row["candidate_id"] in qualified_ids]
    if split == CONFIRMATION_SPLIT and len(candidates) != 3:
        raise RuntimeError(f"Expected three frozen source candidates for {model_key}")
    if not candidates:
        raise RuntimeError(f"No confirmation-qualified source candidates for {model_key}/{split}")
    correct_ids = semantic_correct_cases(model_key, split)
    cases = [
        row for row in read_jsonl(CASES_PATH)
        if row["model"] == model_key
        and row["split"] == split
        and row["case_type"] == "natural_parametric"
        and row["natural_relation"] == "color"
        and row["is_fruit"]
    ]
    if len(cases) != 48:
        raise RuntimeError(f"Expected 48 natural fruit-color confirmation rows for {model_key}")
    case_by_id = {row["case_id"]: row for row in cases}
    output = rows_path(model_key, split)
    if restart:
        output.unlink(missing_ok=True)
        summary_path(model_key, split).unlink(missing_ok=True)

    loaded = None
    started = time.monotonic()
    try:
        loaded = load_probe_model(model_key)
        loaded.tokenizer.padding_side = "left"
        layers = get_layers(loaded.model)
        run_dtype = str(next(loaded.model.parameters()).dtype)
        if run_dtype != "torch.bfloat16":
            raise RuntimeError(f"Phase557 source recompute requires BF16, got {run_dtype}")
        if {int(row["layer_count"]) for row in candidates} != {len(layers)}:
            raise RuntimeError("Phase557 source candidate layer-count drift")
        colors = sorted({row["target"] for row in cases})
        coordinate_layers = sorted({
            int(value)
            for candidate in candidates
            for value in (candidate["layer"], candidate["wrong_depth_control_layer"])
        })

        captures: dict[tuple[int, str, str], torch.Tensor] = {}
        for batch_start in range(0, len(cases), batch_size):
            batch_rows = cases[batch_start:batch_start + batch_size]
            texts = [observer_prompt(model_key, row["prompt"]) for row in batch_rows]
            individual = [semantic_positions(loaded.tokenizer, model_key, row) for row in batch_rows]
            encoded = loaded.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            sequence_length = int(encoded["input_ids"].shape[1])
            positions = {"object_source_end": [], "relation_request_end": []}
            for row_index, (ids, semantic) in enumerate(individual):
                batch_ids = encoded["input_ids"][row_index][encoded["attention_mask"][row_index].bool()].tolist()
                if [int(value) for value in batch_ids] != ids:
                    raise RuntimeError("Phase557 source capture tokenization drift")
                offset = sequence_length - len(ids)
                for position in positions:
                    positions[position].append(offset + semantic[position])
            position_tensors = {
                key: torch.tensor(value, dtype=torch.long, device=loaded.input_device)
                for key, value in positions.items()
            }
            handles = []

            def make_capture(layer_index: int):
                def hook(_module: Any, _inputs: tuple[Any, ...], output_value: Any) -> None:
                    value = tensor_from_output(output_value)
                    batch_index = torch.arange(value.shape[0], device=value.device)
                    for position, indices in position_tensors.items():
                        selected = value[batch_index, indices, :].detach().float().cpu()
                        for index, row in enumerate(batch_rows):
                            captures[(layer_index, position, row["case_id"])] = selected[index]
                return hook

            for layer_index in coordinate_layers:
                handles.append(layers[layer_index].register_forward_hook(make_capture(layer_index)))
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
            wrong_layer = int(candidate["wrong_depth_control_layer"])
            for group_key, group_rows in sorted(groups.items()):
                eligible = sorted(
                    [row for row in group_rows if row["case_id"] in correct_ids],
                    key=lambda row: row["object_label"],
                )
                if len(eligible) < 3:
                    continue
                centroid = torch.stack([
                    captures[(layer_index, "object_source_end", row["case_id"])]
                    for row in group_rows
                ]).mean(dim=0)
                for recipient in eligible:
                    for donor in eligible:
                        if recipient["case_id"] == donor["case_id"] or recipient["target"] == donor["target"]:
                            continue
                        roll = deterministic_roll(
                            candidate["candidate_id"], recipient["case_id"],
                            captures[(layer_index, "object_source_end", donor["case_id"])].numel(),
                        )
                        replacements = {
                            "same_case_restore": captures[(layer_index, "object_source_end", recipient["case_id"])],
                            "object_specific_delete": centroid,
                            "correct_donor_replace": captures[(layer_index, "object_source_end", donor["case_id"])],
                            "wrong_depth_donor_replace": captures[(wrong_layer, "object_source_end", donor["case_id"])],
                            "relation_position_donor_replace": captures[(layer_index, "relation_request_end", donor["case_id"])],
                            "channel_roll_donor_replace": torch.roll(
                                captures[(layer_index, "object_source_end", donor["case_id"])], roll
                            ),
                        }
                        for condition in CONDITIONS:
                            tasks.append({
                                "candidate": candidate,
                                "recipient": recipient,
                                "donor": donor,
                                "group_key": group_key,
                                "condition": condition,
                                "replacement": replacements[condition],
                                "roll_shift": roll if condition == "channel_roll_donor_replace" else None,
                            })
        if not tasks:
            raise RuntimeError(f"No behavior-correct Phase557 source intervention pairs for {model_key}")

        if output.exists() and not restart:
            raise RuntimeError("Resume is intentionally disabled; use --restart for the frozen task matrix")
        completed_rows = 0
        task_groups = [
            [task for task in tasks if task["candidate"]["candidate_id"] == candidate["candidate_id"]]
            for candidate in candidates
        ]
        for candidate_tasks in task_groups:
          for batch_start in range(0, len(candidate_tasks), batch_size):
            batch_tasks = candidate_tasks[batch_start:batch_start + batch_size]
            batch_rows = [task["recipient"] for task in batch_tasks]
            texts = [observer_prompt(model_key, row["prompt"]) for row in batch_rows]
            individual = [semantic_positions(loaded.tokenizer, model_key, row) for row in batch_rows]
            encoded = loaded.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            sequence_length = int(encoded["input_ids"].shape[1])
            object_positions = []
            for row_index, (ids, semantic) in enumerate(individual):
                batch_ids = encoded["input_ids"][row_index][encoded["attention_mask"][row_index].bool()].tolist()
                if [int(value) for value in batch_ids] != ids:
                    raise RuntimeError("Phase557 source intervention tokenization drift")
                object_positions.append(sequence_length - len(ids) + semantic["object_source_end"])
            object_positions_tensor = torch.tensor(
                object_positions, dtype=torch.long, device=loaded.input_device
            )
            replacements = torch.stack([task["replacement"] for task in batch_tasks]).to(
                device=loaded.input_device, dtype=next(loaded.model.parameters()).dtype
            )
            candidate_layers = {int(task["candidate"]["layer"]) for task in batch_tasks}
            if len(candidate_layers) != 1:
                raise RuntimeError("Intervention batch crossed candidate layers")
            layer_index = next(iter(candidate_layers))

            def intervention_hook(_module: Any, _inputs: tuple[Any, ...], output_value: Any) -> Any:
                primary = tensor_from_output(output_value).clone()
                for index, task in enumerate(batch_tasks):
                    # A true same-batch no-op validates the hook and readout
                    # without importing BF16 differences from another batch.
                    if task["condition"] == "same_case_restore":
                        continue
                    primary[index, object_positions_tensor[index], :] = replacements[index]
                return replace_primary(output_value, primary)

            encoded = {key: value.to(loaded.input_device) for key, value in encoded.items()}
            with torch.inference_mode():
                baseline_result = loaded.model(**encoded, use_cache=False)
            baseline_logits = baseline_result.logits[:, -1, :].detach().float().cpu()
            handle = layers[layer_index].register_forward_hook(intervention_hook)
            with torch.inference_mode():
                result = loaded.model(**encoded, use_cache=False)
            handle.remove()
            logits = result.logits[:, -1, :].detach().float().cpu()
            output_rows: list[dict[str, Any]] = []
            for index, task in enumerate(batch_tasks):
                recipient = task["recipient"]
                donor = task["donor"]
                scores = word_scores(logits[index], loaded.tokenizer, colors)
                baseline = word_scores(baseline_logits[index], loaded.tokenizer, colors)
                baseline_switch = baseline[donor["target"]] - baseline[recipient["target"]]
                intervention_switch = scores[donor["target"]] - scores[recipient["target"]]
                output_rows.append({
                    "schema_version": "phase557_natural_color_source_intervention.v1",
                    "phase_id": PHASE,
                    "created_at": now(),
                    "model": model_key,
                    "torch_dtype": run_dtype,
                    "split": split,
                    "candidate_id": task["candidate"]["candidate_id"],
                    "candidate_zone": task["candidate"]["zone"],
                    "layer": layer_index,
                    "wrong_depth_control_layer": task["candidate"]["wrong_depth_control_layer"],
                    "component": "layer_output",
                    "source_position": "object_source_end",
                    "condition": task["condition"],
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
                    "baseline_switch_margin": baseline_switch,
                    "intervention_switch_margin": intervention_switch,
                    "donor_switch_effect": intervention_switch - baseline_switch,
                    "baseline_prediction": max(baseline, key=baseline.get),
                    "intervention_prediction": max(scores, key=scores.get),
                    "intervention_donor_wins": max(scores, key=scores.get) == donor["target"],
                    "intervention_recipient_retained": max(scores, key=scores.get) == recipient["target"],
                    "roll_shift": task["roll_shift"],
                    "readout_contract": READOUT,
                    "source_recompute_intervention": True,
                    "causal_intervention_executed": True,
                    "qualified_compute_edge": False,
                    "sealed": False,
                })
            append_jsonl(output, output_rows)
            completed_rows += len(output_rows)
            del baseline_result, result, encoded, output_rows
            if batch_start == 0 or completed_rows == len(tasks) or (batch_start // batch_size) % 20 == 0:
                print(
                    f"[{time.strftime('%H:%M:%S')}] {model_key} Phase557 source recompute "
                    f"{completed_rows}/{len(tasks)}",
                    flush=True,
                )

        final_rows = read_jsonl(output)
        if len(final_rows) != len(tasks):
            raise RuntimeError(f"Incomplete Phase557 source rows: {len(final_rows)}/{len(tasks)}")
        summary = {
            "schema_version": "phase557_natural_color_source_summary.v1",
            "phase_id": PHASE,
            "created_at": now(),
            "status": "complete",
            "model": model_key,
            "torch_dtype": run_dtype,
            "split": split,
            "behavior_correct_case_count": len(correct_ids),
            "candidate_count": len(candidates),
            "candidate_layers": [row["layer"] for row in candidates],
            "condition_count": len(CONDITIONS),
            "intervention_row_count": len(final_rows),
            "runtime_seconds": time.monotonic() - started,
            "rows_path": str(output.relative_to(ROOT)),
            "rows_sha256": sha256_file(output),
            "source_recompute_intervention_executed": True,
            "query_end_patch_executed": False,
            "head_channel_parameter_neuron_scan_executed": False,
            "sealed_split_read": False,
        }
        write_json(summary_path(model_key, split), summary)
        print(summary_path(model_key, split))
        return summary_path(model_key, split)
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--split", choices=SPLITS, default=CONFIRMATION_SPLIT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    run(args.model, args.split, args.batch_size, args.restart)


if __name__ == "__main__":
    main()
