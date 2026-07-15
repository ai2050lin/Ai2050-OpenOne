#!/usr/bin/env python3
"""Fit two independent exact Jacobian lenses and merge them for one model."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
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
from hf_probe_env import load_probe_model, release_loaded, vram_gb  # noqa: E402


OUT = ROOT / "tests/gpt5/result/phase423_workspace_observer_qualification"
MODELS = ("qwen3", "glm4", "deepseek7b")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def clean(value: float) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite scalar: {value}")
    return round(float(value), 10)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_layers(n_layers: int, count: int) -> list[int]:
    target = n_layers - 1
    return sorted(
        {
            round(index * (target - 1) / (count - 1))
            for index in range(count)
        }
    )


def matrix_stability(
    lens_a: jlens.JacobianLens,
    lens_b: jlens.JacobianLens,
) -> list[dict[str, Any]]:
    rows = []
    for layer in lens_a.source_layers:
        a = lens_a.jacobians[layer].float()
        b = lens_b.jacobians[layer].float()
        a_norm = a.norm()
        b_norm = b.norm()
        cosine = torch.sum(a * b) / (a_norm * b_norm + 1e-12)
        cosine = torch.clamp(cosine, -1.0, 1.0)
        relative_difference = (a - b).norm() / ((a_norm + b_norm) / 2 + 1e-12)
        rows.append(
            {
                "layer": layer,
                "matrix_cosine": clean(float(cosine)),
                "relative_difference": clean(float(relative_difference)),
                "a_frobenius_norm": clean(float(a_norm)),
                "b_frobenius_norm": clean(float(b_norm)),
                "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
            }
        )
        del a, b
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summarize-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jlens.configure_logging()
    protocol = read_json(OUT / "phase423_protocol.json")
    fit_rows = read_jsonl(OUT / "phase423_fit_prompts.jsonl")
    model_root = OUT / "models" / args.model
    summary_path = model_root / "phase423_fit_summary.json"
    merged_path = model_root / "phase423_jacobian_lens_merged.pt"
    half_a_path = model_root / "phase423_jacobian_lens_fit_a.pt"
    half_b_path = model_root / "phase423_jacobian_lens_fit_b.pt"
    if args.summarize_existing:
        if not all(path.exists() for path in (summary_path, merged_path, half_a_path, half_b_path)):
            raise FileNotFoundError("Cannot summarize missing Phase423 lens files")
        summary = read_json(summary_path)
        lens_a = jlens.JacobianLens.load(str(half_a_path))
        lens_b = jlens.JacobianLens.load(str(half_b_path))
        stability = matrix_stability(lens_a, lens_b)
        cosines = [row["matrix_cosine"] for row in stability]
        differences = [row["relative_difference"] for row in stability]
        gates = protocol["frozen_gates"]
        finite = all(row["finite"] for row in stability)
        summary.update(
            {
                "matrix_stability": stability,
                "matrix_cosine_median": clean(statistics.median(cosines)),
                "matrix_cosine_min": clean(min(cosines)),
                "relative_difference_median": clean(statistics.median(differences)),
                "matrix_finite": finite,
                "fit_reproducibility_gate_pass": bool(
                    finite
                    and statistics.median(cosines)
                    >= float(gates["half_matrix_cosine_median_min"])
                    and min(cosines) >= float(gates["half_matrix_cosine_layer_min"])
                    and statistics.median(differences)
                    <= float(gates["half_relative_difference_median_max"])
                ),
                "summary_recomputed_at": now(),
            }
        )
        write_json(summary_path, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if (
        not args.force
        and summary_path.exists()
        and merged_path.exists()
        and half_a_path.exists()
        and half_b_path.exists()
    ):
        print(summary_path)
        return

    model_root.mkdir(parents=True, exist_ok=True)
    loaded = None
    started = time.perf_counter()
    try:
        loaded = load_probe_model(args.model)
        wrapped = jlens.from_hf(loaded.model, loaded.tokenizer, compile=False)
        fit_contract = protocol["fit_contract"]
        layers = relative_layers(wrapped.n_layers, int(fit_contract["source_layer_count"]))
        target_layer = wrapped.n_layers - 1
        dim_batch = int(fit_contract["dim_batch_by_model"][args.model])
        lenses: dict[str, jlens.JacobianLens] = {}
        split_seconds: dict[str, float] = {}
        for split, output_path in (("fit_a", half_a_path), ("fit_b", half_b_path)):
            prompts = [row["text"] for row in fit_rows if row["split"] == split]
            if len(prompts) != 50:
                raise RuntimeError(f"Expected 50 {split} prompts, got {len(prompts)}")
            checkpoint = model_root / f"phase423_{split}_checkpoint.pt"
            split_started = time.perf_counter()
            lens = jlens.fit(
                wrapped,
                prompts,
                source_layers=layers,
                target_layer=target_layer,
                dim_batch=dim_batch,
                max_seq_len=int(fit_contract["max_seq_len"]),
                skip_first=int(fit_contract["skip_first"]),
                checkpoint_path=str(checkpoint),
                checkpoint_every=int(fit_contract["checkpoint_every"]),
                resume=not args.force,
            )
            lens.save(str(output_path), dtype=torch.float16)
            lenses[split] = lens
            split_seconds[split] = time.perf_counter() - split_started
            checkpoint.unlink(missing_ok=True)

        stability = matrix_stability(lenses["fit_a"], lenses["fit_b"])
        merged = jlens.JacobianLens.merge([lenses["fit_a"], lenses["fit_b"]])
        merged.save(str(merged_path), dtype=torch.float16)
        gates = protocol["frozen_gates"]
        cosines = [row["matrix_cosine"] for row in stability]
        differences = [row["relative_difference"] for row in stability]
        finite = all(row["finite"] for row in stability)
        fit_gate = bool(
            finite
            and statistics.median(cosines)
            >= float(gates["half_matrix_cosine_median_min"])
            and min(cosines) >= float(gates["half_matrix_cosine_layer_min"])
            and statistics.median(differences)
            <= float(gates["half_relative_difference_median_max"])
        )
        allocated, reserved = vram_gb()
        summary = {
            "schema_version": "phase423_workspace_observer_fit.v1",
            "phase": 423,
            "model": args.model,
            "model_adapter": repr(wrapped),
            "dtype": str(next(loaded.model.parameters()).dtype),
            "official_implementation_commit": protocol["official_commit"],
            "source_layers": layers,
            "relative_source_depths": [
                clean(layer / (wrapped.n_layers - 1)) for layer in layers
            ],
            "target_layer": target_layer,
            "d_model": wrapped.d_model,
            "n_prompts_fit_a": lenses["fit_a"].n_prompts,
            "n_prompts_fit_b": lenses["fit_b"].n_prompts,
            "n_prompts_merged": merged.n_prompts,
            "dim_batch": dim_batch,
            "split_seconds": {key: clean(value) for key, value in split_seconds.items()},
            "wall_seconds": clean(time.perf_counter() - started),
            "vram_allocated_gb_at_end": clean(allocated),
            "vram_reserved_gb_at_end": clean(reserved),
            "matrix_stability": stability,
            "matrix_cosine_median": clean(statistics.median(cosines)),
            "matrix_cosine_min": clean(min(cosines)),
            "relative_difference_median": clean(statistics.median(differences)),
            "matrix_finite": finite,
            "fit_reproducibility_gate_pass": fit_gate,
            "lens_files": {
                "fit_a": str(half_a_path.relative_to(ROOT)),
                "fit_b": str(half_b_path.relative_to(ROOT)),
                "merged": str(merged_path.relative_to(ROOT)),
            },
            "lens_file_sizes": {
                "fit_a": half_a_path.stat().st_size,
                "fit_b": half_b_path.stat().st_size,
                "merged": merged_path.stat().st_size,
            },
            "lens_sha256": {
                "fit_a": sha256_file(half_a_path),
                "fit_b": sha256_file(half_b_path),
                "merged": sha256_file(merged_path),
            },
            "semantic_interpretation_allowed": False,
            "completed_at": now(),
        }
        write_json(summary_path, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
