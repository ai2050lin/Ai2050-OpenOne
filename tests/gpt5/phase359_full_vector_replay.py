#!/usr/bin/env python3
"""Replay sealed Phase359 tensor ledgers without loading model weights."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase359_full_vector_anchor"
MODELS = ("qwen3", "glm4", "deepseek7b")
MAX_RELATIVE_ERROR = 0.01
MAX_PROBABILITY_ERROR = 0.01


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def find_manifest(model: str) -> Path:
    matches = list((OUT / "sealed_tensors" / model).glob("*/manifest.json"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one sealed manifest for {model}, found {len(matches)}")
    return matches[0]


def replay_model(model: str) -> dict[str, Any]:
    manifest_path = find_manifest(model)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for file_row in manifest["files"]:
        path = OUT / file_row["relative_path"]
        hash_pass = sha256_file(path) == file_row["sha256"]
        payload = torch.load(path, map_location="cpu", weights_only=True)
        attention_sum = payload["projected_head_contributions"].sum(dim=0)
        if payload["attention_projection_bias"] is not None:
            attention_sum += payload["attention_projection_bias"]
        mlp_sum = payload["projected_mlp_shard_contributions"].sum(dim=0)
        if payload["mlp_down_projection_bias"] is not None:
            mlp_sum += payload["mlp_down_projection_bias"]
        block_sum = (payload["layer_input"] + payload["attention_output"]) + payload["mlp_output"]
        probability_error = float(
            (payload["attention_probabilities"].float().sum(dim=-1) - 1).abs().max().item()
        )
        rows.append({
            "model": model,
            "anchor_id": manifest["anchor_id"],
            "layer_index": file_row["layer_index"],
            "file_hash_pass": hash_pass,
            "attention_relative_reconstruction_error": relative_error(payload["attention_output"], attention_sum),
            "mlp_relative_reconstruction_error": relative_error(payload["mlp_output"], mlp_sum),
            "block_relative_reconstruction_error": relative_error(payload["layer_output"], block_sum),
            "attention_probability_sum_error": probability_error,
            "normalization_finite": bool(
                torch.isfinite(payload["input_normalized_state"]).all()
                and torch.isfinite(payload["post_attention_normalized_state"]).all()
            ),
        })
        del payload
    gates = {
        "file_integrity_pass": all(row["file_hash_pass"] for row in rows),
        "attention_reconstruction_pass": all(
            row["attention_relative_reconstruction_error"] <= MAX_RELATIVE_ERROR for row in rows
        ),
        "mlp_reconstruction_pass": all(
            row["mlp_relative_reconstruction_error"] <= MAX_RELATIVE_ERROR for row in rows
        ),
        "block_reconstruction_pass": all(
            row["block_relative_reconstruction_error"] <= MAX_RELATIVE_ERROR for row in rows
        ),
        "attention_probability_pass": all(
            row["attention_probability_sum_error"] <= MAX_PROBABILITY_ERROR for row in rows
        ),
        "normalization_finite_pass": all(row["normalization_finite"] for row in rows),
    }
    return {
        "model": model,
        "anchor_id": manifest["anchor_id"],
        "layer_count": manifest["layer_count"],
        "replayed_layer_count": len(rows),
        "total_byte_count": manifest["total_byte_count"],
        "gates": gates,
        "all_replay_gates_pass": all(gates.values()) and len(rows) == manifest["layer_count"],
        "max_errors": {
            "attention_relative_reconstruction_error": max(row["attention_relative_reconstruction_error"] for row in rows),
            "mlp_relative_reconstruction_error": max(row["mlp_relative_reconstruction_error"] for row in rows),
            "block_relative_reconstruction_error": max(row["block_relative_reconstruction_error"] for row in rows),
            "attention_probability_sum_error": max(row["attention_probability_sum_error"] for row in rows),
        },
        "rows": rows,
    }


def main() -> None:
    models = [replay_model(model) for model in MODELS]
    summary = {
        "schema_version": "35.0.0",
        "phase_id": "Phase359",
        "created_at": now(),
        "model_count": len(models),
        "anchor_count": len(models),
        "layer_file_count": sum(row["replayed_layer_count"] for row in models),
        "total_byte_count": sum(row["total_byte_count"] for row in models),
        "all_models_replay_pass": all(row["all_replay_gates_pass"] for row in models),
        "models": [{key: value for key, value in row.items() if key != "rows"} for row in models],
        "evidence_boundary": {
            "full_vector_anchor_format_replayable": True,
            "blind_motif_discovery_completed": False,
            "physical_heldout_opened": False,
            "causal_sealed_opened": False,
            "language_encoding_closed": False,
        },
        "next_decision": "start_balanced_r0_r1_blind_discovery_only_after_case_bank_denominator_is_frozen",
    }
    with (OUT / "phase359_replay_rows.jsonl").open("w", encoding="utf-8") as handle:
        for model in models:
            for row in model["rows"]:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    (OUT / "phase359_replay_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: summary[key] for key in (
        "model_count", "anchor_count", "layer_file_count", "total_byte_count", "all_models_replay_pass", "next_decision"
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
