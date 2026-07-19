#!/usr/bin/env python3
"""Collect full-layer component-by-role route geometry for qualified Phase551 contracts."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))
sys.path.insert(0, str(ROOT / "tests/glm5"))

from model_utils import load_model, release_model  # noqa: E402
from phase358_multiresolution_component_conservation import install_hooks  # noqa: E402
import phase545_natural_entry_physical_collect as base  # noqa: E402
from phase551_model_specific_route_protocol import (  # noqa: E402
    CELLS,
    MODELS,
    OUT_DIR,
    VALIDATION_CASES_PATH,
    VALIDATION_PROTOCOL_PATH,
)
from phase551_model_specific_behavior_analysis import QUALIFICATION_PATH  # noqa: E402


COMPONENTS = ("layer_input", "attention_output", "mlp_output", "layer_output")
ROLES = ("source", "query", "current")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
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


def output_path(model: str) -> Path:
    return OUT_DIR / f"phase551_{model}_full_layer_route_rows.jsonl"


def summary_path(model: str) -> Path:
    return OUT_DIR / f"phase551_{model}_full_layer_route_execution.json"


def mean_positions(tensor: torch.Tensor, batch_index: int, positions: list[int]) -> torch.Tensor:
    index = torch.tensor(positions, dtype=torch.long, device=tensor.device)
    return tensor[batch_index].index_select(0, index).float().mean(dim=0).detach().cpu()


def capture_anchor(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    captures: dict[tuple[str, int], torch.Tensor],
    rows: list[dict[str, Any]],
    cell_order: tuple[str, ...] = CELLS,
) -> tuple[dict[int, dict[str, dict[str, dict[str, torch.Tensor]]]], float, dict[str, int]]:
    ordered = sorted(rows, key=lambda row: cell_order.index(row["factorial_cell"]))
    prompt_ids = [
        [int(value) for value in tokenizer(row["prompt"], add_special_tokens=True)["input_ids"]]
        for row in ordered
    ]
    locations = []
    for row, ids in zip(ordered, prompt_ids):
        locations.append({
            "source": base.fragment_positions(tokenizer, ids, row["source_fragment"]),
            "query": base.fragment_positions(tokenizer, ids, row["query_fragment"]),
        })
    encoded = tokenizer(
        [row["prompt"] for row in ordered], return_tensors="pt", padding=True,
        truncation=True, max_length=512,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    captures.clear()
    with torch.inference_mode():
        result = model(**encoded, use_cache=False, return_dict=True)
    width = int(encoded["input_ids"].shape[1])
    vectors: dict[int, dict[str, dict[str, dict[str, torch.Tensor]]]] = {}
    max_ledger_error = 0.0
    source_counts = []
    query_counts = []
    for layer_index in range(len(layers)):
        vectors[layer_index] = {}
        for component in COMPONENTS:
            vectors[layer_index][component] = {}
            tensor = captures[(component, layer_index)]
            for batch_index, (row, ids, location) in enumerate(zip(ordered, prompt_ids, locations)):
                padding = width - len(ids)
                source_positions = [padding + value for value in location["source"]]
                query_positions = [padding + value for value in location["query"]]
                vectors[layer_index][component][row["factorial_cell"]] = {
                    "source": mean_positions(tensor, batch_index, source_positions),
                    "query": mean_positions(tensor, batch_index, query_positions),
                    "current": tensor[batch_index, width - 1].float().detach().cpu(),
                }
                if layer_index == 0 and component == "layer_input":
                    source_counts.append(len(source_positions))
                    query_counts.append(len(query_positions))
        for batch_index, ids in enumerate(prompt_ids):
            padding = width - len(ids)
            positions = sorted({
                width - 1,
                *[padding + value for value in locations[batch_index]["source"]],
                *[padding + value for value in locations[batch_index]["query"]],
            })
            position_index = torch.tensor(positions, dtype=torch.long, device=device)
            actual = captures[("layer_output", layer_index)][batch_index].index_select(0, position_index).float()
            reconstructed = (
                captures[("layer_input", layer_index)][batch_index].index_select(0, position_index)
                + captures[("attention_output", layer_index)][batch_index].index_select(0, position_index)
                + captures[("mlp_output", layer_index)][batch_index].index_select(0, position_index)
            ).float()
            errors = torch.linalg.vector_norm(actual - reconstructed, dim=-1) / torch.clamp(
                torch.linalg.vector_norm(actual, dim=-1), min=1e-8,
            )
            max_ledger_error = max(max_ledger_error, float(errors.max().item()))
    del result, encoded
    captures.clear()
    return vectors, max_ledger_error, {
        "minimum_source_position_count": min(source_counts),
        "maximum_source_position_count": max(source_counts),
        "minimum_query_position_count": min(query_counts),
        "maximum_query_position_count": max(query_counts),
    }


def feature_metrics(values: dict[str, torch.Tensor]) -> dict[str, float]:
    route_a = base.normalized_delta(values["route0_answer_a"], values["route1_answer_a"])
    route_b = base.normalized_delta(values["route0_answer_b"], values["route1_answer_b"])
    answer_r0 = base.normalized_delta(values["route0_answer_a"], values["route0_answer_b"])
    answer_r1 = base.normalized_delta(values["route1_answer_a"], values["route1_answer_b"])
    route_effect = (route_a + route_b) / 2.0
    answer_effect = (answer_r0 + answer_r1) / 2.0
    return {
        "route_delta_answer_a": route_a,
        "route_delta_answer_b": route_b,
        "answer_delta_route0": answer_r0,
        "answer_delta_route1": answer_r1,
        "route_effect": base.clean(route_effect),
        "answer_identity_effect": base.clean(answer_effect),
        "route_minus_answer_effect": base.clean(route_effect - answer_effect),
        "route_to_answer_ratio": base.clean(route_effect / max(answer_effect, 1e-8)),
        "route_direction_alignment": base.cosine(
            values["route0_answer_a"] - values["route1_answer_a"],
            values["route0_answer_b"] - values["route1_answer_b"],
        ),
        "answer_direction_alignment": base.cosine(
            values["route0_answer_a"] - values["route0_answer_b"],
            values["route1_answer_a"] - values["route1_answer_b"],
        ),
    }


def run(model_name: str, restart: bool) -> Path:
    protocol = read_json(VALIDATION_PROTOCOL_PATH)
    if protocol["validation_cases_sha256"] != sha256_file(VALIDATION_CASES_PATH):
        raise RuntimeError("Phase551 validation cases drift")
    qualified = {
        row["mechanism_id"] for row in read_jsonl(QUALIFICATION_PATH)
        if row["model"] == model_name and row["observer_collection_authorized"]
    }
    if not qualified:
        payload = {
            "schema_version": "phase551_full_layer_route_execution.v1",
            "phase_id": "Phase551",
            "created_at": now(),
            "model": model_name,
            "status": "skipped_by_validation_behavior_gate",
            "authorized_mechanisms": [],
            "cuda_loaded": False,
            "anchor_count": 0,
            "row_count": 0,
            "new_sealed_split_read": False,
        }
        write_json(summary_path(model_name), payload)
        return summary_path(model_name)
    source = [
        row for row in read_jsonl(VALIDATION_CASES_PATH)
        if row["model"] == model_name and row["mechanism_id"] in qualified
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source:
        grouped[row["anchor_id"]].append(row)
    output = output_path(model_name)
    if restart:
        output.unlink(missing_ok=True)
        summary_path(model_name).unlink(missing_ok=True)
    completed = {row["anchor_id"] for row in read_jsonl(output)} if output.exists() else set()
    model = None
    handles: list[Any] = []
    captures: dict[tuple[str, int], torch.Tensor] = {}
    started = time.monotonic()
    new_anchors = 0
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Phase551 full-layer observation requires CUDA")
        model, tokenizer, device = load_model(model_name)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        layers = list(base.get_layers(model))
        handles = install_hooks(layers, captures)
        for anchor_id in sorted(grouped):
            if anchor_id in completed:
                continue
            anchor_rows = grouped[anchor_id]
            vectors, ledger_error, position_counts = capture_anchor(
                model, tokenizer, device, layers, captures, anchor_rows,
            )
            reference = anchor_rows[0]
            rows_out = []
            for layer_index in range(len(layers)):
                features = {}
                for component in COMPONENTS:
                    for role in ROLES:
                        feature_key = f"{component}__{role}"
                        values = {
                            cell: vectors[layer_index][component][cell][role]
                            for cell in CELLS
                        }
                        features[feature_key] = feature_metrics(values)
                rows_out.append({
                    "schema_version": "phase551_full_layer_route_row.v1",
                    "phase_id": "Phase551",
                    "created_at": now(),
                    "model": model_name,
                    "family_id": reference["family_id"],
                    "mechanism_id": reference["mechanism_id"],
                    "selected_scaffold_id": reference["scaffold_id"],
                    "split": reference["split"],
                    "world_index": reference["world_index"],
                    "anchor_id": anchor_id,
                    "stage": "prompt_end",
                    "layer": layer_index,
                    "layer_count": len(layers),
                    "relative_depth": base.clean(layer_index / max(1, len(layers) - 1)),
                    "features": features,
                    "max_component_ledger_relative_error": base.clean(ledger_error),
                    **position_counts,
                    "physical": True,
                    "observer_only": True,
                    "predictive": False,
                    "compute_edge": False,
                    "causal": False,
                    "single_neuron": False,
                    "sealed": False,
                })
            append_jsonl(output, rows_out)
            new_anchors += 1
            del vectors, rows_out
            if new_anchors == 1 or new_anchors % 24 == 0 or len(completed) + new_anchors == len(grouped):
                print(
                    f"[{time.strftime('%H:%M:%S')}] {model_name} full-layer route "
                    f"{len(completed) + new_anchors}/{len(grouped)}",
                    flush=True,
                )
        final_rows = read_jsonl(output)
        observed = {row["anchor_id"] for row in final_rows}
        expected_rows = len(grouped) * len(layers)
        if len(observed) != len(grouped) or len(final_rows) != expected_rows:
            raise RuntimeError(
                f"Incomplete Phase551 observer: {model_name} {len(observed)}/{len(grouped)} "
                f"rows={len(final_rows)}/{expected_rows}"
            )
        payload = {
            "schema_version": "phase551_full_layer_route_execution.v1",
            "phase_id": "Phase551",
            "created_at": now(),
            "model": model_name,
            "status": "complete",
            "authorized_mechanisms": sorted(qualified),
            "cuda_loaded": True,
            "anchor_count": len(grouped),
            "row_count": len(final_rows),
            "layer_count": len(layers),
            "new_anchors_this_invocation": new_anchors,
            "runtime_seconds_this_invocation": time.monotonic() - started,
            "max_component_ledger_relative_error": max(
                row["max_component_ledger_relative_error"] for row in final_rows
            ),
            "rows_path": str(output.relative_to(ROOT)),
            "rows_sha256": sha256_file(output),
            "full_hidden_vectors_persisted": False,
            "head_channel_neuron_scan_executed": False,
            "new_sealed_split_read": False,
        }
        write_json(summary_path(model_name), payload)
        print(summary_path(model_name))
        return summary_path(model_name)
    finally:
        for handle in handles:
            handle.remove()
        captures.clear()
        if model is not None:
            release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    run(args.model, args.restart)


if __name__ == "__main__":
    main()
