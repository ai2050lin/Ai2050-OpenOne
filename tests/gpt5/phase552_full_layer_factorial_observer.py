#!/usr/bin/env python3
"""Collect full-layer Phase552 semantic-route, surface, and answer geometry."""

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
from phase551_full_layer_route_observer import COMPONENTS, ROLES, capture_anchor  # noqa: E402
from phase552_factorial_behavior_analysis import QUALIFICATION_PATH  # noqa: E402
from phase552_surface_route_answer_protocol import (  # noqa: E402
    CASES_PATH, CELLS, MODELS, OUT_DIR, PROTOCOL_PATH,
)


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
    return OUT_DIR / f"phase552_{model}_full_layer_factorial_rows.jsonl"


def summary_path(model: str) -> Path:
    return OUT_DIR / f"phase552_{model}_full_layer_factorial_execution.json"


def cell(route: int, surface: int, answer: str) -> str:
    return f"route{route}_surface{surface}_answer_{answer}"


def average(values: list[float]) -> float:
    return base.clean(sum(values) / max(len(values), 1))


def feature_metrics(values: dict[str, torch.Tensor]) -> dict[str, float]:
    route_deltas = [
        base.normalized_delta(values[cell(0, surface, answer)], values[cell(1, surface, answer)])
        for surface in (0, 1) for answer in ("a", "b")
    ]
    surface_deltas = [
        base.normalized_delta(values[cell(route, 0, answer)], values[cell(route, 1, answer)])
        for route in (0, 1) for answer in ("a", "b")
    ]
    answer_deltas = [
        base.normalized_delta(values[cell(route, surface, "a")], values[cell(route, surface, "b")])
        for route in (0, 1) for surface in (0, 1)
    ]
    route_effect = average(route_deltas)
    surface_effect = average(surface_deltas)
    answer_effect = average(answer_deltas)
    maximum_control = max(surface_effect, answer_effect)
    return {
        "semantic_route_effect": route_effect,
        "surface_form_effect": surface_effect,
        "answer_identity_effect": answer_effect,
        "maximum_control_effect": maximum_control,
        "route_minus_max_control": base.clean(route_effect - maximum_control),
        "route_to_max_control_ratio": base.clean(route_effect / max(maximum_control, 1e-8)),
        "route_effect_surface0_answer_a": route_deltas[0],
        "route_effect_surface0_answer_b": route_deltas[1],
        "route_effect_surface1_answer_a": route_deltas[2],
        "route_effect_surface1_answer_b": route_deltas[3],
    }


def run(model_name: str, restart: bool) -> Path:
    protocol = read_json(PROTOCOL_PATH)
    if protocol["registered_cases_sha256"] != sha256_file(CASES_PATH):
        raise RuntimeError("Phase552 registered cases drift")
    qualified = {
        row["mechanism_id"] for row in read_jsonl(QUALIFICATION_PATH)
        if row["model"] == model_name and row["observer_collection_authorized"]
    }
    if not qualified:
        payload = {
            "schema_version": "phase552_full_layer_factorial_execution.v1",
            "phase_id": "Phase552",
            "created_at": now(),
            "model": model_name,
            "status": "skipped_by_behavior_gate",
            "authorized_mechanisms": [],
            "cuda_loaded": False,
            "anchor_count": 0,
            "row_count": 0,
            "new_sealed_split_read": False,
        }
        write_json(summary_path(model_name), payload)
        return summary_path(model_name)
    source = [
        row for row in read_jsonl(CASES_PATH)
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
            raise RuntimeError("Phase552 full-layer observation requires CUDA")
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
                model, tokenizer, device, layers, captures, anchor_rows, cell_order=CELLS,
            )
            reference = anchor_rows[0]
            rows_out = []
            for layer_index in range(len(layers)):
                features = {}
                for component in COMPONENTS:
                    for role in ROLES:
                        key = f"{component}__{role}"
                        values = {
                            name: vectors[layer_index][component][name][role]
                            for name in CELLS
                        }
                        features[key] = feature_metrics(values)
                rows_out.append({
                    "schema_version": "phase552_full_layer_factorial_row.v1",
                    "phase_id": "Phase552",
                    "created_at": now(),
                    "model": model_name,
                    "family_id": reference["family_id"],
                    "mechanism_id": reference["mechanism_id"],
                    "surface0_scaffold_id": next(
                        row["scaffold_id"] for row in anchor_rows if row["surface_factor"] == 0
                    ),
                    "surface1_scaffold_id": next(
                        row["scaffold_id"] for row in anchor_rows if row["surface_factor"] == 1
                    ),
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
                    f"[{time.strftime('%H:%M:%S')}] {model_name} Phase552 observer "
                    f"{len(completed) + new_anchors}/{len(grouped)}",
                    flush=True,
                )
        final_rows = read_jsonl(output)
        observed = {row["anchor_id"] for row in final_rows}
        expected_rows = len(grouped) * len(layers)
        if len(observed) != len(grouped) or len(final_rows) != expected_rows:
            raise RuntimeError(f"Incomplete Phase552 observer: {model_name}")
        payload = {
            "schema_version": "phase552_full_layer_factorial_execution.v1",
            "phase_id": "Phase552",
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
