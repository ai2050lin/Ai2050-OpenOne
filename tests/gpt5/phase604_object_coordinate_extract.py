#!/usr/bin/env python3
"""Collect full-layer residuals at the last object-label occurrence."""

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
import phase604_object_coordinate_protocol as protocol  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def output_paths(model: str) -> dict[str, Path]:
    stem = protocol.OUT_DIR / f"phase604_{model}_object_coordinate"
    return {
        "arrays": stem.with_suffix(".npz"),
        "metadata": stem.with_name(stem.name + "_metadata.json"),
        "summary": stem.with_name(stem.name + "_summary.json"),
    }


def last_occurrence_end(tokenizer: Any, input_ids: list[int], label: str) -> int:
    positions: set[int] = set()
    for text in (label, " " + label):
        needle = [int(value) for value in tokenizer.encode(text, add_special_tokens=False)]
        for start in range(len(input_ids) - len(needle) + 1):
            if input_ids[start : start + len(needle)] == needle:
                positions.add(start + len(needle) - 1)
    if not positions:
        raise RuntimeError(f"Cannot locate object label tokens: {label}")
    return max(positions)


def extract(model: str, restart: bool = False) -> Path:
    if model not in protocol.QUALIFIED_BRANCHES or not torch.cuda.is_available():
        raise RuntimeError("Phase604 requires a qualified model and CUDA")
    tracks = set(protocol.QUALIFIED_BRANCHES[model])
    cases = [row for row in read_rows(phase602.CASES_PATH) if row["track"] in tracks]
    behavior = {row["case_id"]: row for row in read_rows(behavior_paths(model)["rows"])}
    paths = output_paths(model)
    if restart:
        for path in paths.values():
            path.unlink(missing_ok=True)
    loaded = None
    batches: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    occurrence_counts: list[int] = []
    started = time.monotonic()
    try:
        loaded = load_probe_model(model)
        if loaded.input_device.type != "cuda" or str(next(loaded.model.parameters()).dtype) != "torch.bfloat16":
            raise RuntimeError("Phase604 requires CUDA BF16")
        loaded.tokenizer.padding_side = "left"
        if loaded.tokenizer.pad_token_id is None:
            loaded.tokenizer.pad_token_id = loaded.tokenizer.eos_token_id
        for start in range(0, len(cases), protocol.FIXED_BATCH_SIZE):
            batch = cases[start : start + protocol.FIXED_BATCH_SIZE]
            prompts = [render_chat(loaded.tokenizer, model, row["raw_prompt"]) for row in batch]
            encoded = loaded.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=True)
            positions = []
            for index, row in enumerate(batch):
                ids = [int(value) for value in encoded["input_ids"][index].tolist()]
                position = last_occurrence_end(loaded.tokenizer, ids, row["concept_label"])
                positions.append(position)
                occurrence_counts.append(sum(
                    ids[pos : pos + len(needle)] == needle
                    for text in (row["concept_label"], " " + row["concept_label"])
                    for needle in [[int(value) for value in loaded.tokenizer.encode(text, add_special_tokens=False)]]
                    for pos in range(len(ids) - len(needle) + 1)
                ))
            encoded = {key: value.to(loaded.input_device) for key, value in encoded.items()}
            with torch.inference_mode():
                result = loaded.model(**encoded, use_cache=False, output_hidden_states=True, output_attentions=False, return_dict=True)
            row_index = torch.arange(len(batch), device=loaded.input_device)
            position_index = torch.tensor(positions, device=loaded.input_device)
            states = torch.stack([state[row_index, position_index, :] for state in result.hidden_states], dim=1)
            batches.append(states.to(torch.float16).cpu().numpy())
            for row, position in zip(batch, positions):
                outcome = behavior[row["case_id"]]
                metadata.append({
                    "case_id": row["case_id"], "concept_id": row["concept_id"],
                    "concept_label": row["concept_label"], "split": row["split"], "track": row["track"],
                    "surface_id": row["surface_id"], "target_letter": row["target_letter"],
                    "fruit_member": row["fruit_member"], "entity_role": row["entity_role"],
                    "behavior_correct": outcome["forced_choice_correct"], "object_end_token_index": position,
                })
            del result, states, encoded
            if start == 0 or (start // protocol.FIXED_BATCH_SIZE + 1) % 20 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] {model} Phase604 {min(start + protocol.FIXED_BATCH_SIZE, len(cases))}/{len(cases)}", flush=True)
        activations = np.concatenate(batches, axis=0)
        np.savez_compressed(paths["arrays"], activations=activations)
        paths["metadata"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        summary = {
            "schema_version": "phase604_object_coordinate_extract.v1", "phase_id": protocol.PHASE,
            "created_at": datetime.now(timezone.utc).isoformat(), "status": "complete", "model": model,
            "tracks": sorted(tracks), "case_count": len(cases), "array_shape": list(activations.shape),
            "array_dtype": str(activations.dtype), "minimum_exact_occurrence_count": min(occurrence_counts),
            "maximum_exact_occurrence_count": max(occurrence_counts), "runtime_seconds": time.monotonic() - started,
            "arrays_sha256": sha256_file(paths["arrays"]), "metadata_sha256": sha256_file(paths["metadata"]),
            "internal_state_collected": True, "future_option_tokens_excluded": True,
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
