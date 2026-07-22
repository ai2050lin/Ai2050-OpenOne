#!/usr/bin/env python3
"""Collect only answer-boundary residual states for qualified Phase602 branches."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))
os.environ.setdefault("PROBE_TORCH_DTYPE", "bfloat16")

from hf_probe_env import load_probe_model, release_loaded  # noqa: E402
from phase548_shared_attention_compute_protocol import render_chat  # noqa: E402
import phase602_three_track_protocol as phase602  # noqa: E402
from phase602_three_track_behavior import output_paths as behavior_paths  # noqa: E402
import phase603_fruit_residual_protocol as protocol  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_gzip_rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def output_paths(model: str) -> dict[str, Path]:
    stem = protocol.OUT_DIR / f"phase603_{model}_qualified_residual"
    return {
        "arrays": stem.with_suffix(".npz"),
        "metadata": stem.with_name(stem.name + "_metadata.json"),
        "summary": stem.with_name(stem.name + "_summary.json"),
    }


def extract(model: str, restart: bool = False) -> Path:
    if model not in protocol.QUALIFIED_BRANCHES:
        raise RuntimeError(f"No Phase603 qualified branches for {model}")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase603 requires CUDA")
    frozen = json.loads(protocol.PROTOCOL_PATH.read_text())
    if frozen["phase602_protocol_sha256"] != sha256_file(phase602.PROTOCOL_PATH):
        raise RuntimeError("Phase603 upstream protocol drift")
    tracks = set(protocol.QUALIFIED_BRANCHES[model])
    cases = [row for row in read_gzip_rows(phase602.CASES_PATH) if row["track"] in tracks]
    behavior = {row["case_id"]: row for row in read_gzip_rows(behavior_paths(model)["rows"])}
    if set(row["case_id"] for row in cases) - set(behavior):
        raise RuntimeError("Phase603 behavior binding is incomplete")
    paths = output_paths(model)
    if restart:
        for path in paths.values():
            path.unlink(missing_ok=True)
    loaded = None
    activation_batches: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        loaded = load_probe_model(model)
        if loaded.input_device.type != "cuda":
            raise RuntimeError(f"Phase603 requires CUDA, got {loaded.input_device}")
        loaded.tokenizer.padding_side = "left"
        if loaded.tokenizer.pad_token_id is None:
            loaded.tokenizer.pad_token_id = loaded.tokenizer.eos_token_id
        dtype = str(next(loaded.model.parameters()).dtype)
        if dtype != "torch.bfloat16":
            raise RuntimeError(f"Phase603 requires BF16, got {dtype}")
        for start in range(0, len(cases), protocol.FIXED_BATCH_SIZE):
            batch = cases[start : start + protocol.FIXED_BATCH_SIZE]
            prompts = [render_chat(loaded.tokenizer, model, row["raw_prompt"]) for row in batch]
            encoded = loaded.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=True)
            encoded = {key: value.to(loaded.input_device) for key, value in encoded.items()}
            with torch.inference_mode():
                result = loaded.model(
                    **encoded,
                    use_cache=False,
                    output_hidden_states=True,
                    output_attentions=False,
                    return_dict=True,
                )
            states = torch.stack([state[:, -1, :] for state in result.hidden_states], dim=1)
            activation_batches.append(states.to(torch.float16).cpu().numpy())
            for row in batch:
                outcome = behavior[row["case_id"]]
                metadata.append({
                    "case_id": row["case_id"],
                    "concept_id": row["concept_id"],
                    "concept_label": row["concept_label"],
                    "split": row["split"],
                    "track": row["track"],
                    "surface_id": row["surface_id"],
                    "target_letter": row["target_letter"],
                    "fruit_member": row["fruit_member"],
                    "entity_role": row["entity_role"],
                    "behavior_correct": outcome["forced_choice_correct"],
                    "behavior_margin": outcome["target_margin"],
                })
            del result, states, encoded
            if start == 0 or (start // protocol.FIXED_BATCH_SIZE + 1) % 20 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] {model} Phase603 {min(start + protocol.FIXED_BATCH_SIZE, len(cases))}/{len(cases)}", flush=True)
        activations = np.concatenate(activation_batches, axis=0)
        np.savez_compressed(paths["arrays"], activations=activations)
        paths["metadata"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        summary = {
            "schema_version": "phase603_residual_extract_summary.v1",
            "phase_id": protocol.PHASE,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "model": model,
            "tracks": sorted(tracks),
            "case_count": len(cases),
            "array_shape": list(activations.shape),
            "array_dtype": str(activations.dtype),
            "runtime_seconds": time.monotonic() - started,
            "arrays_sha256": sha256_file(paths["arrays"]),
            "metadata_sha256": sha256_file(paths["metadata"]),
            "observation_coordinate": frozen["observation_coordinate"],
            "internal_state_collected": True,
            "attention_or_mlp_collected": False,
            "causal_intervention": False,
        }
        paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
        return paths["summary"]
    finally:
        release_loaded(loaded)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=tuple(protocol.QUALIFIED_BRANCHES))
    parser.add_argument("--restart", action="store_true")
    arguments = parser.parse_args()
    extract(arguments.model, arguments.restart)
