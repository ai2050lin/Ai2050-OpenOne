#!/usr/bin/env python3
"""Collect frozen residual readouts for one Phase1120 checkpoint."""

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
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

from phase1023_fp16_utils import quantization_audit, release_fp16
import phase1120_pythia_hidden_formation_protocol as protocol


BATCH_SIZE = 16


def parameter_probe(model: torch.nn.Module) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        flat = parameter.detach().reshape(-1)
        if flat.numel() == 0:
            continue
        indices = sorted({0, flat.numel() // 3, (2 * flat.numel()) // 3, flat.numel() - 1})
        values = [float(flat[index].float().item()) for index in indices]
        samples.append({"name": name, "shape": list(parameter.shape), "indices": indices, "values": values})
    return {
        "method": "four deterministic FP32-read samples per named parameter",
        "parameter_count": len(samples),
        "digest": protocol.digest(samples),
    }


def selected_output_scores(
    state: torch.Tensor,
    candidate_ids: torch.Tensor,
    output_head: torch.nn.Module,
) -> torch.Tensor:
    weight = output_head.weight[candidate_ids].float()
    scores = torch.einsum("bd,bkd->bk", state.float(), weight)
    bias = getattr(output_head, "bias", None)
    if bias is not None:
        scores = scores + bias[candidate_ids].float()
    return scores


def run(checkpoint: str) -> dict[str, Any]:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    source_integrity = protocol.read_json(protocol.SOURCE_ROOT / "protocol" / "checkpoint_integrity.json")
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1120 protocol audit failed")
    if checkpoint not in prereg["checkpoints"]:
        raise RuntimeError(f"checkpoint not frozen: {checkpoint}")

    rows = list(protocol.read_jsonl(protocol.OUT_ROOT / "protocol" / "cases.jsonl"))
    if protocol.digest(rows) != prereg["case_digest"]:
        raise RuntimeError("case digest mismatch")
    projection_path = protocol.OUT_ROOT / "protocol" / "projection_matrix.npy"
    if protocol.file_sha256(projection_path) != prereg["projection"]["sha256"]:
        raise RuntimeError("projection digest mismatch")
    projection_np = np.load(projection_path, allow_pickle=False)
    if projection_np.shape != (protocol.HIDDEN_SIZE, protocol.PROJECTION_DIM):
        raise RuntimeError("projection shape mismatch")

    local_path = protocol.MODEL_ROOT / checkpoint
    started = time.time()
    model = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            local_path,
            dtype=torch.float16,
            local_files_only=True,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to("cuda")
        model.eval()
        precision = quantization_audit(model)
        if precision["has_quantized_modules"] or precision["has_bf16_parameters"] or not precision["has_fp16_parameters"]:
            raise RuntimeError("FP16/no-quantization audit failed")
        probe = parameter_probe(model)
        expected_probe = source_integrity["checkpoints"][checkpoint]["parameter_probe"]
        if probe["digest"] != expected_probe["digest"]:
            raise RuntimeError("loaded parameter probe does not match the Phase1117 integrity ledger")
        if int(model.config.hidden_size) != protocol.HIDDEN_SIZE or int(model.config.num_hidden_layers) != protocol.NUM_LAYERS:
            raise RuntimeError("model architecture mismatch")

        final_norm = model.gpt_neox.final_layer_norm
        output_head = model.get_output_embeddings()
        projection = torch.from_numpy(projection_np).to(device="cuda", dtype=torch.float32)
        case_count = len(rows)
        state_projection = np.full(
            (case_count, protocol.HIDDEN_STATE_COUNT, protocol.PROJECTION_DIM),
            np.nan,
            dtype=np.float16,
        )
        true_z = np.full((case_count, protocol.HIDDEN_STATE_COUNT), np.nan, dtype=np.float32)
        control_z = np.full((case_count, protocol.HIDDEN_STATE_COUNT), np.nan, dtype=np.float32)
        final_selected_logit_error = np.full((case_count, 4), np.nan, dtype=np.float32)

        by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_length[len(row["input_ids"])].append(row)

        with torch.inference_mode():
            for length in sorted(by_length):
                panel = by_length[length]
                for start in range(0, len(panel), BATCH_SIZE):
                    batch = panel[start : start + BATCH_SIZE]
                    input_ids = torch.tensor([row["input_ids"] for row in batch], dtype=torch.long, device="cuda")
                    attention_mask = torch.ones_like(input_ids)
                    candidate_ids = torch.tensor(
                        [[*row["true_candidate_ids"], *row["control_candidate_ids"]] for row in batch],
                        dtype=torch.long,
                        device="cuda",
                    )
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    hidden_states = output.hidden_states
                    if hidden_states is None or len(hidden_states) != protocol.HIDDEN_STATE_COUNT:
                        raise RuntimeError(f"unexpected hidden-state count: {None if hidden_states is None else len(hidden_states)}")
                    actual_scores = torch.gather(output.logits[:, -1, :].float(), 1, candidate_ids)

                    for layer_index, hidden in enumerate(hidden_states):
                        state = hidden[:, -1, :]
                        if layer_index < len(hidden_states) - 1:
                            state = final_norm(state)
                        state32 = state.float()
                        scores = selected_output_scores(state32, candidate_ids, output_head)
                        unit_state = F.normalize(state32, p=2, dim=-1, eps=1.0e-12)
                        projected = unit_state @ projection
                        for slot, row in enumerate(batch):
                            case_index = int(row["case_index"])
                            values = scores[slot]
                            true_z[case_index, layer_index] = float((values[0] - values[1]).item())
                            control_z[case_index, layer_index] = float((values[2] - values[3]).item())
                            state_projection[case_index, layer_index, :] = projected[slot].half().cpu().numpy()
                        if layer_index == len(hidden_states) - 1:
                            error = (scores - actual_scores).abs().cpu().numpy()
                            for slot, row in enumerate(batch):
                                final_selected_logit_error[int(row["case_index"]), :] = error[slot]

                    del output, hidden_states, actual_scores, input_ids, attention_mask, candidate_ids
                print(json.dumps({"phase": protocol.PHASE, "checkpoint": checkpoint, "length_complete": length}), flush=True)

        finite_mask = np.isfinite(true_z) & np.isfinite(control_z)
        projection_finite = np.isfinite(state_projection)
        error_finite = np.isfinite(final_selected_logit_error)
        finite_fraction = float(
            (finite_mask.sum() + projection_finite.sum() + error_finite.sum())
            / (finite_mask.size + projection_finite.size + error_finite.size)
        )
        max_error = float(np.nanmax(final_selected_logit_error))
        output_root = protocol.OUT_ROOT / "hidden" / checkpoint
        output_root.mkdir(parents=True, exist_ok=True)
        artifact_path = output_root / "hidden_detail.npz"
        np.savez_compressed(
            artifact_path,
            case_indices=np.arange(case_count, dtype=np.int32),
            true_z=true_z,
            control_z=control_z,
            state_projection=state_projection,
            final_selected_logit_error=final_selected_logit_error,
        )
        summary_core = {
            "schema_version": "phase1120_pythia_hidden_checkpoint_summary.v1",
            "phase": protocol.PHASE,
            "checkpoint": checkpoint,
            "protocol_digest": prereg["protocol_digest"],
            "case_digest": prereg["case_digest"],
            "parameter_probe": probe,
            "expected_parameter_probe_digest": expected_probe["digest"],
            "precision": precision,
            "placement": "full_cuda",
            "quantization": "none",
            "case_count": case_count,
            "hidden_state_count": protocol.HIDDEN_STATE_COUNT,
            "hidden_size": protocol.HIDDEN_SIZE,
            "projection_dimension": protocol.PROJECTION_DIM,
            "finite_fraction": finite_fraction,
            "maximum_final_selected_logit_error": max_error,
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
    parser.add_argument("checkpoint", choices=protocol.CHECKPOINTS)
    args = parser.parse_args()
    run(args.checkpoint)


if __name__ == "__main__":
    main()
