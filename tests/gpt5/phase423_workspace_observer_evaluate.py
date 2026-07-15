#!/usr/bin/env python3
"""Evaluate a fitted Jacobian lens against the vanilla logit lens."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import jlens  # noqa: E402
from hf_probe_env import load_probe_model, release_loaded  # noqa: E402
from jlens.hooks import ActivationRecorder  # noqa: E402


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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def clean(value: float) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite scalar: {value}")
    return round(float(value), 10)


def candidate_token_ids(tokenizer: Any, surface: str) -> list[int]:
    variants = {
        surface,
        " " + surface,
        surface.lower(),
        " " + surface.lower(),
        surface.title(),
        " " + surface.title(),
    }
    output: set[int] = set()
    for variant in variants:
        ids = tokenizer(variant, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            output.add(int(ids[0]))
    return sorted(output)


def token_rank(logits: torch.Tensor, token_ids: list[int]) -> int | None:
    if not token_ids:
        return None
    scores = logits[token_ids]
    ranks = (logits[:, None] > scores[None, :]).sum(dim=0)
    return int(ranks.min().item()) + 1


@torch.inference_mode()
def evaluate_item(
    wrapped: Any,
    lens: jlens.JacobianLens,
    row: dict[str, Any],
) -> list[dict[str, Any]]:
    input_ids = wrapped.encode(row["prompt"], max_length=512)
    layers = lens.source_layers
    final_layer = wrapped.n_layers - 1
    with ActivationRecorder(wrapped.layers, at=[*layers, final_layer]) as recorder:
        wrapped.forward(input_ids)
        activations = {layer: recorder.activations[layer].detach() for layer in [*layers, final_layer]}

    position = int(row["readout_position"])
    raw_residuals = torch.stack(
        [activations[layer][0, position].float() for layer in layers], dim=0
    )
    transported = torch.stack(
        [
            raw_residuals[index] @ lens.jacobians[layer].T
            for index, layer in enumerate(layers)
        ],
        dim=0,
    )
    jlens_logits = wrapped.unembed(transported).float().cpu()
    logit_logits = wrapped.unembed(raw_residuals).float().cpu()
    model_logits = wrapped.unembed(activations[final_layer][0, position].float()).float().cpu()
    outputs: list[dict[str, Any]] = []
    for surface in row["intermediates"]:
        ids = candidate_token_ids(wrapped.tokenizer, str(surface))
        j_ranks = [token_rank(logits, ids) for logits in jlens_logits]
        l_ranks = [token_rank(logits, ids) for logits in logit_logits]
        eligible = bool(ids)
        j_min = min(rank for rank in j_ranks if rank is not None) if eligible else None
        l_min = min(rank for rank in l_ranks if rank is not None) if eligible else None
        j_best_index = j_ranks.index(j_min) if eligible else None
        l_best_index = l_ranks.index(l_min) if eligible else None
        outputs.append(
            {
                "schema_version": "phase423_workspace_observer_evaluation_row.v1",
                "model": wrapped._hf_model.config.name_or_path.split("/")[-1],
                "evaluation_id": row["evaluation_id"],
                "dataset": row["dataset"],
                "split": row["split"],
                "item_name": row["name"],
                "surface": surface,
                "candidate_token_ids": ids,
                "eligible_single_token": eligible,
                "source_layers": layers,
                "jlens_ranks_by_layer": j_ranks,
                "logit_lens_ranks_by_layer": l_ranks,
                "jlens_min_rank": j_min,
                "logit_lens_min_rank": l_min,
                "jlens_best_layer": layers[j_best_index] if eligible else None,
                "logit_lens_best_layer": layers[l_best_index] if eligible else None,
                "jlens_improved": bool(eligible and j_min < l_min),
                "jlens_equal": bool(eligible and j_min == l_min),
                "model_next_token_rank": token_rank(model_logits, ids),
                "observer_edge": True,
                "compute_edge": False,
                "causal": False,
            }
        )
    del activations, raw_residuals, transported, jlens_logits, logit_logits, model_logits
    return outputs


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["split"])].append(row)
    summaries: list[dict[str, Any]] = []
    for (dataset, split), values in sorted(grouped.items()):
        eligible = [row for row in values if row["eligible_single_token"]]
        total = len(values)
        count = len(eligible)
        j_ranks = [int(row["jlens_min_rank"]) for row in eligible]
        l_ranks = [int(row["logit_lens_min_rank"]) for row in eligible]
        summary: dict[str, Any] = {
            "dataset": dataset,
            "split": split,
            "total_intermediates": total,
            "eligible_intermediates": count,
            "eligible_fraction": clean(count / total if total else 0.0),
            "improved_fraction": clean(
                sum(bool(row["jlens_improved"]) for row in eligible) / count if count else 0.0
            ),
            "equal_fraction": clean(
                sum(bool(row["jlens_equal"]) for row in eligible) / count if count else 0.0
            ),
        }
        for name, ranks in (("jlens", j_ranks), ("logit_lens", l_ranks)):
            summary[f"{name}_median_rank"] = clean(statistics.median(ranks)) if ranks else None
            summary[f"{name}_mrr"] = (
                clean(statistics.fmean(1.0 / rank for rank in ranks)) if ranks else 0.0
            )
            for k in (1, 5, 10, 50):
                summary[f"{name}_pass_at_{k}"] = clean(
                    sum(rank <= k for rank in ranks) / count if count else 0.0
                )
        summary["mrr_ratio"] = clean(
            summary["jlens_mrr"] / (summary["logit_lens_mrr"] + 1e-12)
        )
        summary["pass_at_10_gain"] = clean(
            summary["jlens_pass_at_10"] - summary["logit_lens_pass_at_10"]
        )
        summaries.append(summary)
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = read_json(OUT / "phase423_protocol.json")
    eval_rows = read_jsonl(OUT / "phase423_evaluation_items.jsonl")
    model_root = OUT / "models" / args.model
    lens_path = model_root / "phase423_jacobian_lens_merged.pt"
    fit_summary = read_json(model_root / "phase423_fit_summary.json")
    loaded = None
    started = time.perf_counter()
    try:
        loaded = load_probe_model(args.model)
        wrapped = jlens.from_hf(loaded.model, loaded.tokenizer, compile=False)
        lens = jlens.JacobianLens.load(str(lens_path))
        if lens.d_model != wrapped.d_model or lens.source_layers != fit_summary["source_layers"]:
            raise RuntimeError("Lens/model contract mismatch")
        lens.jacobians = {
            layer: matrix.to(loaded.input_device, dtype=torch.float32)
            for layer, matrix in lens.jacobians.items()
        }
        rows: list[dict[str, Any]] = []
        for index, item in enumerate(eval_rows, start=1):
            rows.extend(evaluate_item(wrapped, lens, item))
            if index % 24 == 0:
                print(f"{args.model}: {index}/{len(eval_rows)} evaluation items", flush=True)
        for row in rows:
            row["model"] = args.model
        summaries = summarize(rows)
        write_jsonl(model_root / "phase423_observer_evaluation_rows.jsonl", rows)
        output = {
            "schema_version": "phase423_workspace_observer_evaluation.v1",
            "phase": 423,
            "model": args.model,
            "fit_reproducibility_gate_pass": fit_summary["fit_reproducibility_gate_pass"],
            "evaluation_item_count": len(eval_rows),
            "intermediate_row_count": len(rows),
            "summaries": summaries,
            "wall_seconds": clean(time.perf_counter() - started),
            "observer_only": True,
            "workspace_claim_allowed": False,
            "compute_edge_claim_allowed": False,
            "causal_claim_allowed": False,
            "completed_at": now(),
        }
        write_json(model_root / "phase423_observer_evaluation_summary.json", output)
        print(json.dumps(output, ensure_ascii=False, indent=2))
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
