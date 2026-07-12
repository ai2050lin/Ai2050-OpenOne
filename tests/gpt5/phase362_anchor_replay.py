#!/usr/bin/env python3
"""Replay Phase362 source edges and component ledgers without model weights."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase362_generation_time_trace/independent_generation_time"
MODELS = ("qwen3", "glm4", "deepseek7b")
MAX_ERROR = 0.01


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_error(actual: torch.Tensor, reconstructed: torch.Tensor) -> float:
    error = float(torch.linalg.vector_norm(actual.float() - reconstructed.float()).item())
    scale = float(torch.linalg.vector_norm(actual.float()).item())
    return error / max(scale, 1e-8)


def main() -> None:
    rows = []
    manifests = [read_json(OUT / "sealed_anchors" / model / "manifest.json") for model in MODELS]
    for manifest in manifests:
        for file_row in manifest["files"]:
            path = OUT / file_row["relative_path"]
            payload = torch.load(path, map_location="cpu", weights_only=True)
            projected_values = payload["projected_value_states"].float()
            probabilities = payload["selected_attention_probabilities"].float()
            edge_heads = []
            for head_index in range(projected_values.shape[0]):
                edge_heads.append(torch.einsum(
                    "bqs,bsh->bqh",
                    probabilities[:, head_index], projected_values[head_index],
                ))
            edge_heads = torch.stack(edge_heads)
            edge_error = relative_error(payload["projected_head_outputs"], edge_heads)
            attention_sum = edge_heads.sum(dim=0)
            if payload["attention_bias"] is not None:
                attention_sum += payload["attention_bias"].float()
            attention_error = relative_error(payload["attention_output"], attention_sum)
            mlp_sum = payload["mlp_shard_contributions"].float().sum(dim=0)
            if payload["mlp_bias"] is not None:
                mlp_sum += payload["mlp_bias"].float()
            mlp_error = relative_error(payload["mlp_output"], mlp_sum)
            block = (payload["layer_input"] + payload["attention_output"]) + payload["mlp_output"]
            block_error = relative_error(payload["layer_output"], block)
            input_norm_error = relative_error(payload["input_norm_actual"], payload["input_norm_replayed"])
            post_norm_error = relative_error(payload["post_norm_actual"], payload["post_norm_replayed"])
            probability_error = float((probabilities.sum(dim=-1) - 1).abs().max().item())
            values = {
                "file_hash": sha256_file(path) == file_row["sha256"],
                "edge": edge_error <= MAX_ERROR,
                "attention": attention_error <= MAX_ERROR,
                "mlp": mlp_error <= MAX_ERROR,
                "block": block_error <= MAX_ERROR,
                "input_norm": input_norm_error <= MAX_ERROR,
                "post_norm": post_norm_error <= MAX_ERROR,
                "probability": probability_error <= MAX_ERROR,
            }
            rows.append({
                "model": manifest["model"], "anchor_id": payload["anchor_id"],
                "anchor_type": payload["anchor_type"], "generation_time": payload["generation_time"],
                "layer_index": payload["layer_index"],
                "errors": {
                    "edge": edge_error, "attention": attention_error, "mlp": mlp_error,
                    "block": block_error, "input_norm": input_norm_error,
                    "post_norm": post_norm_error, "probability": probability_error,
                },
                "gates": values, "all_gates_pass": all(values.values()),
            })
            del payload, projected_values, probabilities, edge_heads
    error_names = ("edge", "attention", "mlp", "block", "input_norm", "post_norm", "probability")
    summary = {
        "schema_version": "39.0.0", "phase_id": "Phase362", "created_at": now(),
        "denominator": {
            "model_count": 3, "anchor_count": 9,
            "anchor_time_count": sum(row["anchor_time_count"] for row in manifests),
            "layer_file_count": len(rows),
            "total_byte_count": sum(row["total_byte_count"] for row in manifests),
        },
        "quality": {
            "all_online_gates_pass": all(row["all_online_gates_pass"] for row in manifests),
            "all_offline_gates_pass": all(row["all_gates_pass"] for row in rows),
            "max_errors": {name: max(row["errors"][name] for row in rows) for name in error_names},
        },
        "claim_boundary": {
            "source_edge_format_replayable": True,
            "edge_conservation_is_information_mechanism": False,
            "physical_confirmation_opened": False,
            "causal_intervention_executed": False,
        },
        "next_decision": "run_independent_calibration_generation_time_trace",
    }
    write_jsonl(OUT / "phase362_anchor_replay_rows.jsonl", rows)
    write_json(OUT / "phase362_anchor_replay_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
