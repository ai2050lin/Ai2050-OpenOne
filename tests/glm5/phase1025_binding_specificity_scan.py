#!/usr/bin/env python3
"""Run frozen-depth binding-specificity controls in three FP16 models."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, get_model_info
import phase1025_binding_specificity_protocol as protocol
from phase1023_fp16_utils import (
    MODELS,
    load_fp16,
    quantization_audit,
    release_fp16,
)


ROLES = protocol.ROLES
ROLE_INDEX = {role: index for index, role in enumerate(ROLES)}
BATCH_SIZE = {"qwen3": 32, "glm4": 8, "deepseek7b": 8}
EPS = 1e-8


def chunks(rows: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def make_batch(
    rows: list[dict[str, Any]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(len(row["input_ids"]) for row in rows)
    ids = torch.full(
        (len(rows), width),
        int(pad_token_id),
        dtype=torch.long,
    )
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    positions = torch.empty((len(rows), len(ROLES)), dtype=torch.long)
    for index, row in enumerate(rows):
        value = torch.tensor(row["input_ids"], dtype=torch.long)
        ids[index, :len(value)] = value
        mask[index, :len(value)] = 1
        for role_index, role in enumerate(ROLES):
            positions[index, role_index] = int(
                row["role_positions"][role]
            )
    return ids.to(device), mask.to(device), positions


class SelectiveCapture:
    def __init__(self, model, layers, depths: list[int]):
        self.model = model
        self.layers = layers
        self.depths = depths
        self.positions: torch.Tensor | None = None
        self.values: dict[int, torch.Tensor] = {}
        self.counts: dict[int, int] = defaultdict(int)
        self.handles = []

    def _hook(self, depth: int):
        def hook(module, args, output):
            value = output[0] if isinstance(output, tuple) else output
            if self.positions is None:
                raise RuntimeError("capture positions missing")
            positions = self.positions.to(value.device)
            batch = torch.arange(value.shape[0], device=value.device)[:, None]
            self.values[depth] = value[batch, positions, :].detach()
            self.counts[depth] += 1
            return output
        return hook

    def register(self) -> None:
        if 0 in self.depths:
            self.handles.append(
                self.model.get_input_embeddings().register_forward_hook(
                    self._hook(0)
                )
            )
        for depth in self.depths:
            if depth == 0:
                continue
            self.handles.append(
                self.layers[depth - 1].register_forward_hook(
                    self._hook(depth)
                )
            )

    def begin(self, positions: torch.Tensor) -> None:
        self.positions = positions
        self.values = {}
        self.counts = defaultdict(int)

    def stacked_cpu(self) -> torch.Tensor:
        missing = set(self.depths) - set(self.values)
        if missing:
            raise RuntimeError(f"missing frozen depths: {sorted(missing)}")
        repeated = {
            depth: self.counts[depth]
            for depth in self.depths
            if self.counts[depth] != 1
        }
        if repeated:
            raise RuntimeError(f"hook count drift: {repeated}")
        return torch.stack([
            self.values[depth].to("cpu") for depth in self.depths
        ], dim=2)

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles = []
        self.positions = None
        self.values = {}


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, EPS)


def top1(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    similarity = normalize_rows(left) @ normalize_rows(right).T
    labels = np.arange(len(left))
    same = similarity[labels, labels]
    wrong = similarity.copy()
    wrong[labels, labels] = -np.inf
    return (
        float(np.mean(np.argmax(similarity, axis=1) == labels)),
        float(np.mean(same - np.max(wrong, axis=1))),
    )


def grid(
    values: np.ndarray,
    cases: list[dict[str, Any]],
    split: str,
) -> np.ndarray:
    result = np.empty(
        (len(protocol.CONDITIONS), 4, 8) + tuple(values.shape[1:]),
        dtype=np.float32,
    )
    condition_index = {
        value: index
        for index, value in enumerate(protocol.CONDITIONS)
    }
    seen = set()
    for index, row in enumerate(cases):
        if row["split"] != split:
            continue
        key = (
            condition_index[row["condition"]],
            int(row["surface_index"]),
            int(row["target_index"]),
        )
        result[key] = values[index]
        seen.add(key)
    if len(seen) != 128:
        raise RuntimeError(f"incomplete {split} grid: {len(seen)}")
    return result


def condition_metrics(
    values: np.ndarray,
    cases: list[dict[str, Any]],
    split: str,
) -> dict[str, Any]:
    values = grid(values, cases, split)
    values = values - values.mean(axis=(1, 2), keepdims=True)
    result = {}
    for condition_index, condition in enumerate(protocol.CONDITIONS):
        accuracies = []
        margins = []
        for left_surface in range(4):
            for right_surface in range(4):
                if left_surface == right_surface:
                    continue
                accuracy, margin = top1(
                    values[condition_index, left_surface],
                    values[condition_index, right_surface],
                )
                accuracies.append(accuracy)
                margins.append(margin)
        result[condition] = {
            "target_cross_surface_top1": float(np.mean(accuracies)),
            "chance": 1.0 / 8.0,
            "true_vs_wrong_margin": float(np.mean(margins)),
        }
    bound = result["target_bound"]["target_cross_surface_top1"]
    result["bound_minus_controls"] = {
        condition: bound - result[condition][
            "target_cross_surface_top1"
        ]
        for condition in protocol.CONDITIONS
        if condition != "target_bound"
    }
    return result


def alignment_metrics(
    target: np.ndarray,
    focus: np.ndarray,
    cases: list[dict[str, Any]],
    split: str,
) -> dict[str, Any]:
    target_grid = grid(target, cases, split)
    focus_grid = grid(focus, cases, split)
    result = {}
    for condition_index, condition in enumerate(protocol.CONDITIONS):
        target_values = target_grid[condition_index]
        query_values = focus_grid[condition_index]
        target_values = (
            target_values
            - target_values.mean(axis=1, keepdims=True)
        )
        query_values = (
            query_values
            - query_values.mean(axis=1, keepdims=True)
        )
        target_prototype = target_values.mean(axis=0)
        accuracies = []
        margins = []
        for surface in range(4):
            accuracy, margin = top1(
                query_values[surface],
                target_prototype,
            )
            accuracies.append(accuracy)
            margins.append(margin)
        result[condition] = {
            "target_query_top1": float(np.mean(accuracies)),
            "chance": 1.0 / 8.0,
            "true_vs_wrong_margin": float(np.mean(margins)),
        }
    bound = result["target_bound"]["target_query_top1"]
    result["bound_minus_controls"] = {
        condition: bound - result[condition]["target_query_top1"]
        for condition in protocol.CONDITIONS
        if condition != "target_bound"
    }
    return result


def finite_audit(raw: np.ndarray) -> dict[str, Any]:
    total = int(np.prod(raw.shape))
    nonfinite = 0
    for start in range(0, raw.shape[0], 32):
        nonfinite += int(np.count_nonzero(
            ~np.isfinite(np.asarray(raw[start:start + 32]))
        ))
    return {
        "shape": list(raw.shape),
        "value_count": total,
        "nonfinite_count": nonfinite,
        "all_finite": nonfinite == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    args = parser.parse_args()

    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    cases = protocol.read_jsonl(
        protocol.OUT_ROOT / "protocol" / f"cases.{args.model}.jsonl"
    )
    frozen = sorted({
        0,
        *map(
            int,
            prereg["frozen_layers_from_phase1024"][args.model],
        ),
    })
    started = time.time()
    model = tokenizer = None
    try:
        model, tokenizer, device, placement = load_fp16(args.model)
        precision_audit = quantization_audit(model)
        if (
            precision_audit["has_quantized_modules"]
            or precision_audit["has_bf16_parameters"]
            or not precision_audit["has_fp16_parameters"]
        ):
            raise RuntimeError("FP16/no-quantization audit failed")
        layers = get_layers(model)
        info = get_model_info(model, args.model)
        raw_path = (
            protocol.OUT_ROOT / "atlas" / args.model / "states.fp16.npy"
        )
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw = np.lib.format.open_memmap(
            raw_path,
            mode="w+",
            dtype=np.float16,
            shape=(
                len(cases),
                len(ROLES),
                len(frozen),
                info.d_model,
            ),
        )
        capture = SelectiveCapture(model, layers, frozen)
        capture.register()
        base_model = model.model
        try:
            offset = 0
            for batch_index, batch in enumerate(
                chunks(cases, BATCH_SIZE[args.model]), 1
            ):
                input_ids, attention_mask, positions = make_batch(
                    batch,
                    pad_token_id=tokenizer.pad_token_id,
                    device=device,
                )
                capture.begin(positions)
                with torch.inference_mode():
                    base_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
                values = capture.stacked_cpu().numpy().astype(
                    np.float16, copy=False
                )
                raw[offset:offset + len(batch)] = values
                offset += len(batch)
                if batch_index % 5 == 0:
                    print(
                        f"[phase1025] {args.model} "
                        f"cases={offset}/{len(cases)}",
                        flush=True,
                    )
        finally:
            capture.close()
        raw.flush()

        metric_rows = []
        for role_index, role in enumerate(ROLES):
            for depth_index, depth in enumerate(frozen):
                values = np.asarray(raw[:, role_index, depth_index, :])
                row = {
                    "schema_version": "phase1025_metric.v1",
                    "model": args.model,
                    "role": role,
                    "depth": depth,
                    "relative_depth": depth / max(len(layers), 1),
                    "condition_metrics": {
                        split: condition_metrics(values, cases, split)
                        for split in protocol.SPLITS
                    },
                }
                if role == "focus_end":
                    row["target_query_alignment"] = {
                        split: alignment_metrics(
                            np.asarray(
                                raw[
                                    :,
                                    ROLE_INDEX["target_end"],
                                    depth_index,
                                    :,
                                ]
                            ),
                            values,
                            cases,
                            split,
                        )
                        for split in protocol.SPLITS
                    }
                metric_rows.append(row)
        protocol.write_jsonl(
            raw_path.parent / "metrics.jsonl", metric_rows
        )
        summary = {
            "schema_version": "phase1025_model_summary.v1",
            "phase": protocol.PHASE,
            "protocol_digest": prereg["protocol_digest"],
            "model": args.model,
            "precision": "fp16",
            "quantization": "none",
            "placement": placement,
            "runtime_precision_audit": precision_audit,
            "frozen_depths": frozen,
            "selection_source": "phase1024_only",
            "tensor_finiteness": finite_audit(raw),
            "metric_count": len(metric_rows),
            "elapsed_seconds": time.time() - started,
            "claim_limit": (
                "binding-specific observational control only; no output "
                "ability or causal mechanism claim"
            ),
        }
        protocol.write_json(raw_path.parent / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        if model is not None:
            release_fp16(model)
        del tokenizer


if __name__ == "__main__":
    main()
