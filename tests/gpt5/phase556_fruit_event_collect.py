#!/usr/bin/env python3
"""Collect direction-preserving full-layer Phase556 factor events.

Only behavior-qualified, all-16-cells-correct controlled anchors are observed.
Full vectors are kept only in memory; persisted rows contain factor-effect
magnitudes, conditional factor effects, and component-ledger diagnostics.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase556_fruit_encoding_protocol import CELLS, FACTORS  # noqa: E402


MODELS = ("qwen3", "glm4")
SPLITS = ("discovery", "independent_confirmation")
OUT_DIR = ROOT / "tests/gpt5/result/phase556_fruit_encoding"
CASES_PATH = OUT_DIR / "phase556_open_cases.jsonl"
QUALIFICATION_PATH = OUT_DIR / "phase556_behavior_qualification.jsonl"
COMPONENTS = ("layer_input", "attention_output", "mlp_output", "layer_output")


def observer_prompt(model: str, prompt: str) -> str:
    # Every open GLM4 controlled generation starts with this formatting token.
    return prompt + "\n" if model == "glm4" else prompt


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


def output_path(model: str, split: str) -> Path:
    return OUT_DIR / "event_collection" / model / split / "phase556_event_rows.jsonl"


def summary_path(model: str, split: str) -> Path:
    return OUT_DIR / "event_collection" / model / split / "phase556_event_summary.json"


def tensor_from_output(output: Any) -> torch.Tensor:
    value = output[0] if isinstance(output, tuple) else output
    if not torch.is_tensor(value):
        raise TypeError(f"Unexpected hook output: {type(value).__name__}")
    return value


def factor_effect(vectors: torch.Tensor, rows: list[dict[str, Any]], subset: tuple[str, ...]) -> torch.Tensor:
    coefficients = torch.tensor(
        [
            float(
                torch.tensor([
                    1.0 if int(row["factor_values"][factor]) else -1.0
                    for factor in subset
                ]).prod().item()
            )
            for row in rows
        ],
        dtype=torch.float32,
    )
    return (vectors.float() * coefficients[:, None]).mean(dim=0)


def conditional_effect(
    vectors: torch.Tensor,
    rows: list[dict[str, Any]],
    factor: str,
    query_value: int,
) -> torch.Tensor:
    selected = [index for index, row in enumerate(rows) if int(row["factor_values"]["query"]) == query_value]
    positive = [index for index in selected if int(rows[index]["factor_values"][factor]) == 1]
    negative = [index for index in selected if int(rows[index]["factor_values"][factor]) == 0]
    return vectors[positive].float().mean(dim=0) - vectors[negative].float().mean(dim=0)


def clean(value: float) -> float:
    if not torch.isfinite(torch.tensor(value)):
        return 0.0
    return float(value)


def qualified_anchors(model: str, split: str) -> set[str]:
    reports = {row["model"]: row for row in read_jsonl(QUALIFICATION_PATH)}
    if model not in reports or not reports[model]["internal_collection_authorized"]:
        return set()
    behavior = read_jsonl(OUT_DIR / f"phase556_{model}_behavior_rows.jsonl")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in behavior:
        if row["split"] != split or row["case_type"] != "controlled_factorial":
            continue
        grouped.setdefault(row["anchor_id"], []).append(row)
    return {
        anchor for anchor, rows in grouped.items()
        if len(rows) == 16 and all(row["semantic_correct"] for row in rows)
    }


def run(model_key: str, split: str, restart: bool) -> Path:
    anchors = qualified_anchors(model_key, split)
    if not anchors:
        raise RuntimeError(f"No Phase556 authorized anchors for {model_key}/{split}")
    cases = [
        row for row in read_jsonl(CASES_PATH)
        if row["model"] == model_key
        and row["split"] == split
        and row["case_type"] == "controlled_factorial"
        and row["anchor_id"] in anchors
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in cases:
        grouped.setdefault(row["anchor_id"], []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: CELLS.index(row["factorial_cell"]))
        if [row["factorial_cell"] for row in rows] != list(CELLS):
            raise RuntimeError("Phase556 observer factorial order drift")

    output = output_path(model_key, split)
    if restart:
        output.unlink(missing_ok=True)
        summary_path(model_key, split).unlink(missing_ok=True)
    completed = {row["anchor_id"] for row in read_jsonl(output)} if output.exists() else set()
    loaded = None
    handles: list[Any] = []
    captures: dict[str, list[torch.Tensor]] = {key: [] for key in COMPONENTS}
    started = time.monotonic()
    new_anchor_count = 0
    max_ledger_error = 0.0
    try:
        loaded = load_probe_model(model_key)
        loaded.tokenizer.padding_side = "left"
        layers = get_layers(loaded.model)
        run_dtype = str(next(loaded.model.parameters()).dtype)

        def pre_hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            captures["layer_input"].append(inputs[0][:, -1, :].detach().float().cpu())

        def make_hook(component: str):
            def hook(_module: Any, _inputs: tuple[Any, ...], output_value: Any) -> None:
                value = tensor_from_output(output_value)
                captures[component].append(value[:, -1, :].detach().float().cpu())
            return hook

        for layer in layers:
            handles.append(layer.register_forward_pre_hook(pre_hook))
            handles.append(layer.self_attn.register_forward_hook(make_hook("attention_output")))
            handles.append(layer.mlp.register_forward_hook(make_hook("mlp_output")))
            handles.append(layer.register_forward_hook(make_hook("layer_output")))

        for anchor_index, (anchor_id, rows) in enumerate(sorted(grouped.items()), 1):
            if anchor_id in completed:
                continue
            for values in captures.values():
                values.clear()
            encoded = loaded.tokenizer(
                [observer_prompt(model_key, row["prompt"]) for row in rows], return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            )
            encoded = {key: value.to(loaded.input_device) for key, value in encoded.items()}
            with torch.inference_mode():
                result = loaded.model(**encoded, use_cache=False)
            if any(len(captures[key]) != len(layers) for key in COMPONENTS):
                raise RuntimeError(f"Phase556 hook count mismatch for {model_key}/{anchor_id}")
            anchor_rows: list[dict[str, Any]] = []
            for layer_index in range(len(layers)):
                layer_input = captures["layer_input"][layer_index]
                attention = captures["attention_output"][layer_index]
                mlp = captures["mlp_output"][layer_index]
                layer_output = captures["layer_output"][layer_index]
                residual_error = (layer_output - layer_input - attention - mlp).norm(dim=1)
                denominator = layer_output.norm(dim=1).clamp_min(1e-12)
                ledger_error = float((residual_error / denominator).max().item())
                max_ledger_error = max(max_ledger_error, ledger_error)
                for component in COMPONENTS:
                    vectors = captures[component][layer_index]
                    mean_state_norm = float(vectors.norm(dim=1).mean().item())
                    effects: dict[str, Any] = {}
                    for width in range(1, len(FACTORS) + 1):
                        for subset in itertools.combinations(FACTORS, width):
                            value = factor_effect(vectors, rows, subset)
                            name = "*".join(subset)
                            norm = float(value.norm().item())
                            effects[name] = {
                                "effect_norm": clean(norm),
                                "relative_effect_norm": clean(norm / max(mean_state_norm, 1e-12)),
                            }
                    conditional: dict[str, Any] = {}
                    for factor in ("category", "binding", "entity"):
                        for query_value in (0, 1):
                            value = conditional_effect(vectors, rows, factor, query_value)
                            norm = float(value.norm().item())
                            conditional[f"{factor}|query={query_value}"] = {
                                "effect_norm": clean(norm),
                                "relative_effect_norm": clean(norm / max(mean_state_norm, 1e-12)),
                            }
                    anchor_rows.append({
                        "schema_version": "phase556_factor_event_row.v1",
                        "phase_id": "Phase556",
                        "created_at": now(),
                        "model": model_key,
                        "torch_dtype": run_dtype,
                        "split": split,
                        "anchor_id": anchor_id,
                        "world_index": rows[0]["world_index"],
                        "surface_id": rows[0]["surface_id"],
                        "fact_order": rows[0]["fact_order"],
                        "attribute_relation": rows[0]["attribute_relation"],
                        "layer": layer_index,
                        "layer_count": len(layers),
                        "relative_depth": layer_index / max(1, len(layers) - 1),
                        "component": component,
                        "semantic_position": (
                            "answer_content_boundary_after_natural_newline"
                            if model_key == "glm4" else "query_end"
                        ),
                        "observer_prefix": "\n" if model_key == "glm4" else "",
                        "mean_state_norm": clean(mean_state_norm),
                        "factor_effects": effects,
                        "conditional_effects": conditional,
                        "component_ledger_relative_error": clean(ledger_error),
                        "direction_preserved_in_memory": True,
                        "full_vector_persisted": False,
                        "observer_only": True,
                        "compute_edge": False,
                        "causal": False,
                        "sealed": False,
                    })
            append_jsonl(output, anchor_rows)
            new_anchor_count += 1
            del result, encoded, anchor_rows
            if new_anchor_count == 1 or new_anchor_count % 16 == 0 or anchor_index == len(grouped):
                print(
                    f"[{time.strftime('%H:%M:%S')}] {model_key}/{split} Phase556 events "
                    f"{len(completed) + new_anchor_count}/{len(grouped)}",
                    flush=True,
                )
        final_rows = read_jsonl(output)
        observed = {row["anchor_id"] for row in final_rows}
        expected_rows = len(grouped) * len(layers) * len(COMPONENTS)
        if observed != set(grouped) or len(final_rows) != expected_rows:
            raise RuntimeError(
                f"Incomplete Phase556 events {model_key}/{split}: {len(final_rows)}/{expected_rows}"
            )
        summary = {
            "schema_version": "phase556_factor_event_summary.v1",
            "phase_id": "Phase556",
            "created_at": now(),
            "status": "complete",
            "model": model_key,
            "torch_dtype": run_dtype,
            "split": split,
            "anchor_count": len(grouped),
            "layer_count": len(layers),
            "component_count": len(COMPONENTS),
            "event_row_count": len(final_rows),
            "new_anchor_count_this_invocation": new_anchor_count,
            "runtime_seconds_this_invocation": time.monotonic() - started,
            "max_component_ledger_relative_error": max(row["component_ledger_relative_error"] for row in final_rows),
            "rows_path": str(output.relative_to(ROOT)),
            "rows_sha256": sha256_file(output),
            "full_vectors_persisted": False,
            "causal_intervention_executed": False,
            "head_channel_neuron_scan_executed": False,
            "sealed_split_read": False,
        }
        write_json(summary_path(model_key, split), summary)
        print(summary_path(model_key, split))
        return summary_path(model_key, split)
    finally:
        for handle in handles:
            handle.remove()
        for values in captures.values():
            values.clear()
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("split", choices=SPLITS)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    run(args.model, args.split, args.restart)
