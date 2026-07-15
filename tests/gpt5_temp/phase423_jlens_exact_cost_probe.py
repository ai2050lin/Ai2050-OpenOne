#!/usr/bin/env python3
"""Measure the exact Jacobian-lens estimator cost without fitting a partial lens.

The probe executes a small number of the official estimator's output-dimension
batches.  It is a resource qualification tool only: truncated Jacobian rows are
never saved or interpreted as a semantic observer.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import jlens  # noqa: E402
from hf_probe_env import load_probe_model, release_loaded  # noqa: E402
from jlens.fitting import valid_position_mask  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402


OUT = ROOT / "tests/gpt5/result/phase423_workspace_observer_qualification"
PROMPT = (
    "A careful observer records a sequence of ordinary events. "
    "The blue marker follows the green marker, the square remains beside the "
    "circle, and each statement is checked before the next statement is read. "
    "This neutral passage is used only to measure a model operation."
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: float) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite measurement: {value}")
    return round(float(value), 8)


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark(
    model_key: str,
    *,
    dim_batch: int,
    measured_passes: int,
    max_seq_len: int,
    skip_first: int,
    source_layer_count: int,
) -> dict[str, Any]:
    loaded = None
    started = time.perf_counter()
    try:
        loaded = load_probe_model(model_key)
        wrapped = jlens.from_hf(loaded.model, loaded.tokenizer, compile=False)
        target_layer = wrapped.n_layers - 1
        if source_layer_count == 1:
            source_layers = [wrapped.n_layers // 2]
        else:
            source_layers = sorted(
                {
                    round(index * (target_layer - 1) / (source_layer_count - 1))
                    for index in range(source_layer_count)
                }
            )
        input_ids = wrapped.encode(PROMPT, max_length=max_seq_len)
        seq_len = int(input_ids.shape[1])
        position_mask = valid_position_mask(seq_len, skip_first=skip_first)
        valid_positions_cpu = position_mask.nonzero(as_tuple=True)[0]
        n_valid = int(valid_positions_cpu.numel())
        total_passes = math.ceil(wrapped.d_model / dim_batch)
        n_probe_passes = min(measured_passes, total_passes)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        synchronize()
        graph_started = time.perf_counter()
        with (
            ActivationRecorder(
                wrapped.layers,
                at=[*source_layers, target_layer],
                start_graph_at=min(source_layers),
            ) as recorder,
            torch.enable_grad(),
        ):
            replicated_ids = input_ids.expand(dim_batch, -1)
            wrapped.forward(replicated_ids)
            synchronize()
            forward_seconds = time.perf_counter() - graph_started

            target_activation = recorder.activations[target_layer]
            source_activations = [recorder.activations[layer] for layer in source_layers]
            valid_positions = valid_positions_cpu.to(target_activation.device)
            batch_indices = torch.arange(dim_batch, device=target_activation.device)
            cotangent = torch.zeros_like(target_activation)
            backward_seconds: list[float] = []
            row_norms: list[float] = []
            row_checksums: list[float] = []

            for pass_idx in range(n_probe_passes):
                dim_start = pass_idx * dim_batch
                n_dims = min(dim_batch, wrapped.d_model - dim_start)
                cotangent.zero_()
                cotangent[
                    batch_indices[:n_dims, None],
                    valid_positions[None, :],
                    dim_start + batch_indices[:n_dims, None],
                ] = 1.0
                synchronize()
                pass_started = time.perf_counter()
                grads = torch.autograd.grad(
                    outputs=target_activation,
                    inputs=source_activations,
                    grad_outputs=cotangent,
                    retain_graph=pass_idx < n_probe_passes - 1,
                )
                synchronize()
                backward_seconds.append(time.perf_counter() - pass_started)
                for grad in grads:
                    positions_on_device = valid_positions.to(grad.device, non_blocking=True)
                    rows = grad[:n_dims, positions_on_device, :].float().mean(dim=1)
                    row_norms.extend(float(value) for value in rows.norm(dim=-1).cpu())
                    row_checksums.extend(float(value) for value in rows.sum(dim=-1).cpu())
                    del rows
                del grads

        median_backward = statistics.median(backward_seconds)
        mean_backward = statistics.fmean(backward_seconds)
        projected_prompt = forward_seconds + total_passes * median_backward
        matrix_bytes_fp32 = wrapped.d_model * wrapped.d_model * 4
        peak_allocated = (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        )
        peak_reserved = (
            int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0
        )
        return {
            "schema_version": "phase423_jlens_exact_cost_probe.v1",
            "phase": 423,
            "measurement_kind": "resource_qualification_only",
            "semantic_interpretation_allowed": False,
            "official_implementation_commit": "581d398613e5602a5af361e1c34d3a92ea82ba8e",
            "model": model_key,
            "adapter": repr(wrapped),
            "layout": wrapped.layout.__dict__,
            "source_layers": source_layers,
            "target_layer": target_layer,
            "relative_source_depths": [
                clean(layer / (wrapped.n_layers - 1)) for layer in source_layers
            ],
            "n_layers": wrapped.n_layers,
            "d_model": wrapped.d_model,
            "seq_len": seq_len,
            "skip_first": skip_first,
            "valid_positions": n_valid,
            "dim_batch": dim_batch,
            "measured_passes": n_probe_passes,
            "required_passes_per_prompt": total_passes,
            "forward_seconds": clean(forward_seconds),
            "backward_seconds": [clean(value) for value in backward_seconds],
            "median_backward_seconds": clean(median_backward),
            "mean_backward_seconds": clean(mean_backward),
            "projected_seconds_per_full_prompt": clean(projected_prompt),
            "projected_hours_100_prompts": clean(projected_prompt * 100 / 3600),
            "projected_hours_1000_prompts": clean(projected_prompt * 1000 / 3600),
            "matrix_bytes_fp32_per_layer": matrix_bytes_fp32,
            "matrix_bytes_fp16_per_layer": matrix_bytes_fp32 // 2,
            "matrix_bytes_fp32_all_sources": matrix_bytes_fp32 * len(source_layers),
            "matrix_bytes_fp16_all_sources": matrix_bytes_fp32 * len(source_layers) // 2,
            "peak_cuda_allocated_bytes": peak_allocated,
            "peak_cuda_reserved_bytes": peak_reserved,
            "measured_row_norm_min": clean(min(row_norms)),
            "measured_row_norm_max": clean(max(row_norms)),
            "measured_row_checksum": clean(sum(row_checksums)),
            "started_at": now(),
            "wall_seconds": clean(time.perf_counter() - started),
        }
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("qwen3", "glm4", "deepseek7b"))
    parser.add_argument("--dim-batch", type=int, default=1)
    parser.add_argument("--measured-passes", type=int, default=3)
    parser.add_argument("--max-seq-len", type=int, default=32)
    parser.add_argument("--skip-first", type=int, default=4)
    parser.add_argument("--source-layer-count", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dim_batch < 1 or args.measured_passes < 1:
        raise SystemExit("dim batch and measured passes must be positive")
    result = benchmark(
        args.model,
        dim_batch=args.dim_batch,
        measured_passes=args.measured_passes,
        max_seq_len=args.max_seq_len,
        skip_first=args.skip_first,
        source_layer_count=args.source_layer_count,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.model}_exact_cost_probe.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
