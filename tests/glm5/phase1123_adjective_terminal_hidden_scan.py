#!/usr/bin/env python3
"""Collect one model's frozen Phase1123 terminal residual event map."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16
import phase1121_wordnet_adjective_double_orthogonal_protocol as source
import phase1123_adjective_terminal_hidden_protocol as protocol


def run(model_name: str) -> dict[str, Any]:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1123 protocol audit failed")
    rows = protocol.read_jsonl(protocol.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl")
    if protocol.digest(rows) != prereg["case_digests"][model_name]:
        raise RuntimeError("Phase1123 case digest mismatch")
    projection_spec = prereg["projection_specs"][model_name]
    projection_path = protocol.OUT_ROOT / projection_spec["path"]
    if protocol.file_sha256(projection_path) != projection_spec["sha256"]:
        raise RuntimeError("Phase1123 projection digest mismatch")
    projection_np = np.load(projection_path, allow_pickle=False)
    model_spec = prereg["model_specs"][model_name]
    if projection_np.shape != (model_spec["hidden_size"], prereg["projection_dimension"]):
        raise RuntimeError("Phase1123 projection shape mismatch")

    source_detail = {
        int(row["case_index"]): row
        for row in source.read_jsonl(source.OUT_ROOT / "behavior" / model_name / "candidate_detail.jsonl")
    }
    if len(source_detail) != len(rows):
        raise RuntimeError("Phase1121 detail count mismatch")

    started = time.time()
    model = None
    try:
        model, _tokenizer, device, placement = load_fp16(model_name)
        precision = quantization_audit(model)
        if precision["has_quantized_modules"] or precision["has_bf16_parameters"] or not precision["has_fp16_parameters"]:
            raise RuntimeError("FP16/no-quantization audit failed")
        if int(model.config.hidden_size) != model_spec["hidden_size"] or int(model.config.num_hidden_layers) != model_spec["num_hidden_layers"]:
            raise RuntimeError("Phase1123 loaded model architecture mismatch")

        case_count = len(rows)
        hidden_state_count = model_spec["hidden_state_count"]
        role_count = len(protocol.ROLES)
        projection_dim = prereg["projection_dimension"]
        projected_states = np.full(
            (case_count, hidden_state_count, role_count, projection_dim),
            np.nan,
            dtype=np.float32,
        )
        output_z = np.full(case_count, np.nan, dtype=np.float32)
        source_z = np.full(case_count, np.nan, dtype=np.float32)
        projection_cache: dict[str, torch.Tensor] = {}

        by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_length[len(row["input_ids"])].append(row)

        with torch.inference_mode():
            for length in sorted(by_length):
                panel = by_length[length]
                batch_size = int(prereg["batch_sizes"][model_name])
                for start in range(0, len(panel), batch_size):
                    batch = panel[start:start + batch_size]
                    input_ids = torch.tensor([row["input_ids"] for row in batch], dtype=torch.long, device=device)
                    attention_mask = torch.ones_like(input_ids)
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    hidden_states = output.hidden_states
                    if hidden_states is None or len(hidden_states) != hidden_state_count:
                        raise RuntimeError(f"unexpected hidden state count for {model_name}")
                    logits = output.logits[:, -1, :].float()
                    for slot, row in enumerate(batch):
                        candidate_ids = row["candidate_first_token_ids"]
                        z = float((logits[slot, int(candidate_ids["true"][0])] - logits[slot, int(candidate_ids["false"][0])]).item())
                        case_index = int(row["case_index"])
                        output_z[case_index] = z
                        source_value = source_detail[case_index]["z_true_minus_false"]
                        source_z[case_index] = float(source_value) if source_value is not None else np.nan

                    for layer_index, hidden in enumerate(hidden_states):
                        hidden_device = str(hidden.device)
                        if hidden_device not in projection_cache:
                            projection_cache[hidden_device] = torch.from_numpy(projection_np).to(device=hidden.device, dtype=torch.float32)
                        role_indices = torch.tensor(
                            [[int(row["role_indices"][role]) for role in protocol.ROLES] for row in batch],
                            dtype=torch.long,
                            device=hidden.device,
                        )
                        batch_indices = torch.arange(len(batch), device=hidden.device).unsqueeze(1).expand_as(role_indices)
                        states = hidden[batch_indices, role_indices, :].float()
                        projected = torch.matmul(states, projection_cache[hidden_device])
                        projected_cpu = projected.float().cpu().numpy()
                        for slot, row in enumerate(batch):
                            projected_states[int(row["case_index"]), layer_index, :, :] = projected_cpu[slot]
                        del role_indices, batch_indices, states, projected, projected_cpu

                    del output, hidden_states, logits, input_ids, attention_mask
                print(json.dumps({"phase": protocol.PHASE, "model": model_name, "length_complete": length}), flush=True)

        finite_fraction = float(
            (np.isfinite(projected_states).sum() + np.isfinite(output_z).sum() + np.isfinite(source_z).sum())
            / (projected_states.size + output_z.size + source_z.size)
        )
        max_z_error = float(np.nanmax(np.abs(output_z - source_z)))
        output_root = protocol.OUT_ROOT / "hidden" / model_name
        output_root.mkdir(parents=True, exist_ok=True)
        artifact_path = output_root / "hidden_detail.npz"
        np.savez_compressed(
            artifact_path,
            case_indices=np.arange(case_count, dtype=np.int32),
            projected_states=projected_states,
            output_z=output_z,
            source_z=source_z,
        )
        summary_core = {
            "schema_version": "phase1123_adjective_terminal_hidden_model_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "case_digest": prereg["case_digests"][model_name],
            "precision": precision,
            "placement": placement,
            "case_count": case_count,
            "hidden_state_count": hidden_state_count,
            "hidden_size": model_spec["hidden_size"],
            "role_count": role_count,
            "projection_dimension": projection_dim,
            "finite_fraction": finite_fraction,
            "maximum_behavior_z_reproduction_error": max_z_error,
            "artifact": str(artifact_path.relative_to(protocol.OUT_ROOT)).replace("\\", "/"),
            "artifact_sha256": protocol.file_sha256(artifact_path),
            "elapsed_seconds": time.time() - started,
        }
        summary = dict(summary_core)
        summary["summary_digest"] = protocol.digest(summary_core)
        protocol.write_json(output_root / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    finally:
        if model is not None:
            release_fp16(model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=protocol.MODELS)
    args = parser.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
