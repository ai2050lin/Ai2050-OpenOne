#!/usr/bin/env python3
"""Collect frozen-window route and answer effects for Phase549."""

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
sys.path.insert(0, str(ROOT / "tests/glm5"))

from hf_probe_env import get_layers  # noqa: E402
from model_utils import load_model, release_model  # noqa: E402
from phase548_matched_observer_collect import component_tensor, cosine, normalized_delta  # noqa: E402
from phase549_route_answer_factorial_protocol import (  # noqa: E402
    CASES_PATH, CELLS, MODELS, OUT_DIR, PROTOCOL_PATH,
)


BEHAVIOR_PATH = OUT_DIR / "phase549_behavior_qualification.jsonl"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def output_path(model: str) -> Path:
    return OUT_DIR / f"phase549_{model}_factorial_observer_rows.jsonl"


def summary_path(model: str) -> Path:
    return OUT_DIR / f"phase549_{model}_factorial_observer_execution.json"


def capture(
    model: Any, tokenizer: Any, device: torch.device, layers: list[Any],
    target_layers: list[int], rows: list[dict[str, Any]],
) -> dict[int, dict[str, torch.Tensor]]:
    ordered = sorted(rows, key=lambda row: CELLS.index(row["factorial_cell"]))
    captures: dict[int, torch.Tensor] = {}
    handles = []
    for layer_index in target_layers:
        def hook(_module: Any, _inputs: Any, output: Any, idx: int = layer_index) -> None:
            captures[idx] = component_tensor(output).detach()
        handles.append(layers[layer_index].self_attn.register_forward_hook(hook))
    encoded = tokenizer(
        [row["prompt"] for row in ordered], return_tensors="pt", padding=True,
        truncation=True, max_length=512,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    try:
        with torch.inference_mode():
            model(**encoded, use_cache=False, return_dict=True)
        return {
            layer_index: {
                row["factorial_cell"]: captures[layer_index][index, -1].float().detach().cpu()
                for index, row in enumerate(ordered)
            }
            for layer_index in target_layers
        }
    finally:
        for handle in handles:
            handle.remove()
        del encoded


def concatenated(vectors: dict[int, dict[str, torch.Tensor]], cell: str) -> torch.Tensor:
    return torch.cat([vectors[layer][cell].float() for layer in sorted(vectors)])


def metrics(vectors: dict[int, dict[str, torch.Tensor]]) -> dict[str, float]:
    values = {cell: concatenated(vectors, cell) for cell in CELLS}
    route_a = normalized_delta(values["route0_answer_a"], values["route1_answer_a"])
    route_b = normalized_delta(values["route0_answer_b"], values["route1_answer_b"])
    answer_r0 = normalized_delta(values["route0_answer_a"], values["route0_answer_b"])
    answer_r1 = normalized_delta(values["route1_answer_a"], values["route1_answer_b"])
    return {
        "route_delta_answer_a": route_a,
        "route_delta_answer_b": route_b,
        "answer_delta_route0": answer_r0,
        "answer_delta_route1": answer_r1,
        "route_effect": round((route_a + route_b) / 2.0, 9),
        "answer_identity_effect": round((answer_r0 + answer_r1) / 2.0, 9),
        "route_minus_answer_effect": round((route_a + route_b - answer_r0 - answer_r1) / 2.0, 9),
        "route_direction_alignment": cosine(
            values["route0_answer_a"] - values["route1_answer_a"],
            values["route0_answer_b"] - values["route1_answer_b"],
        ),
        "answer_direction_alignment": cosine(
            values["route0_answer_a"] - values["route0_answer_b"],
            values["route1_answer_a"] - values["route1_answer_b"],
        ),
    }


def collect(model_name: str, use_8bit: bool, restart: bool) -> Path:
    protocol = read_json(PROTOCOL_PATH)
    if protocol["registered_cases_sha256"] != sha256_file(CASES_PATH):
        raise RuntimeError("Phase549 registered cases drift")
    qualified = {
        row["mechanism_id"] for row in read_jsonl(BEHAVIOR_PATH)
        if row["model"] == model_name and row["observer_collection_authorized"]
    }
    target_layers = list(protocol["frozen_windows"][model_name]["target_layers"])
    if not qualified or not target_layers:
        write_json(summary_path(model_name), {
            "schema_version": "phase549_factorial_observer_execution.v1", "phase_id": "Phase549",
            "created_at": now(), "model": model_name,
            "status": "skipped_by_behavior_or_prior_window_gate", "cuda_loaded": False,
            "authorized_mechanisms": sorted(qualified), "target_layers": target_layers,
            "row_count": 0, "new_sealed_split_read": False,
        })
        return summary_path(model_name)
    source = [
        row for row in read_jsonl(CASES_PATH)
        if row["model"] == model_name and row["mechanism_id"] in qualified
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in source:
        grouped.setdefault(row["anchor_id"], []).append(row)
    output = output_path(model_name)
    if restart:
        output.unlink(missing_ok=True)
        summary_path(model_name).unlink(missing_ok=True)
    completed = {row["anchor_id"] for row in read_jsonl(output)} if output.exists() else set()
    pending = [(key, grouped[key]) for key in sorted(grouped) if key not in completed]
    loaded = None
    started = time.monotonic()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Phase549 observer collection requires CUDA")
        loaded, tokenizer, device = load_model(model_name, use_8bit=True if use_8bit else None)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        layers = list(get_layers(loaded))
        for index, (anchor_id, rows) in enumerate(pending, 1):
            vectors = capture(loaded, tokenizer, device, layers, target_layers, rows)
            by_cell = {row["factorial_cell"]: row for row in rows}
            common = {
                "schema_version": "phase549_factorial_observer_row.v1", "phase_id": "Phase549",
                "created_at": now(), "model": model_name, "family_id": "content_knowledge",
                "mechanism_id": by_cell["route0_answer_a"]["mechanism_id"],
                "split": by_cell["route0_answer_a"]["split"],
                "pair_index": by_cell["route0_answer_a"]["pair_index"], "anchor_id": anchor_id,
                "stage": "prompt_end", "role": "current", "component": "attention_output",
                "observer_only": True, "compute_edge": False, "causal": False,
                "single_neuron": False, "sealed": False,
            }
            rows_out = []
            for layer in target_layers:
                rows_out.append(dict(common, layer=layer, aggregation="single_layer", **metrics({layer: vectors[layer]})))
            rows_out.append(dict(common, layer=None, aggregation="frozen_three_layer_platform", **metrics(vectors)))
            append_jsonl(output, rows_out)
            del vectors
            if index == 1 or index % 32 == 0 or index == len(pending):
                print(
                    f"[{time.strftime('%H:%M:%S')}] {model_name} factorial observer "
                    f"{len(completed) + index}/{len(grouped)}", flush=True,
                )
        final_rows = read_jsonl(output)
        expected = len(grouped) * (len(target_layers) + 1)
        if len(final_rows) != expected:
            raise RuntimeError(f"Incomplete Phase549 observer rows: {len(final_rows)}/{expected}")
        write_json(summary_path(model_name), {
            "schema_version": "phase549_factorial_observer_execution.v1", "phase_id": "Phase549",
            "created_at": now(), "model": model_name, "status": "complete", "cuda_loaded": True,
            "authorized_mechanisms": sorted(qualified), "target_layers": target_layers,
            "anchor_count": len(grouped), "row_count": len(final_rows),
            "rows_path": str(output.relative_to(ROOT)), "rows_sha256": sha256_file(output),
            "runtime_seconds_this_invocation": time.monotonic() - started,
            "full_hidden_vectors_persisted": False, "head_channel_neuron_scan_executed": False,
            "new_sealed_split_read": False,
        })
        return summary_path(model_name)
    finally:
        if loaded is not None:
            release_model(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--use-8bit", action="store_true")
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    collect(args.model, args.use_8bit, args.restart)


if __name__ == "__main__":
    main()
