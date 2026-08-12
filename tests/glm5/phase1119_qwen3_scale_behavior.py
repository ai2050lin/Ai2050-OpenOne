#!/usr/bin/env python3
"""Run one Qwen3 size for the frozen Phase1119 behavior scale arm."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

from phase1023_fp16_utils import quantization_audit
import phase1119_qwen3_scale_protocol as protocol


BATCH_SIZE = 32
PHASE1118_PROTOCOL = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1118_qwen3_14b_fp16_offload_smoke"
    / "protocol"
    / "protocol.json"
)


def load_model(model_name: str) -> tuple[Any, Any, dict[str, str], str]:
    local_path = protocol.MODEL_ROOTS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(
        local_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_name == "qwen3_4b":
        model = AutoModelForCausalLM.from_pretrained(
            local_path,
            dtype=torch.float16,
            local_files_only=True,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_safetensors=True,
        ).to("cuda:0")
        device_map = {"": "0"}
        placement = "full_cuda"
    else:
        phase1118 = protocol.read_json(PHASE1118_PROTOCOL)
        if phase1118["schema_version"] != "phase1118_qwen3_14b_fp16_smoke_protocol.v5":
            raise RuntimeError("Phase1118 successful device map is not frozen revision 5")
        device_map = {
            key: (int(value) if value.isdigit() else value)
            for key, value in phase1118["device_map"].items()
        }
        offload_root = protocol.OUT_ROOT / "offload" / "qwen3_14b"
        offload_root.mkdir(parents=True, exist_ok=True)
        config = AutoConfig.from_pretrained(local_path, local_files_only=True, trust_remote_code=True)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                config,
                dtype=torch.float16,
                trust_remote_code=True,
            )
        model.tie_weights()
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=str(local_path),
            device_map=device_map,
            no_split_module_classes=list(model._no_split_modules),
            offload_folder=str(offload_root),
            offload_buffers=False,
            dtype=torch.float16,
            offload_state_dict=True,
            force_hooks=True,
            strict=True,
        )
        device_map = {str(key): str(value) for key, value in model.hf_device_map.items()}
        placement = "cuda_disk_offload"
    model.eval()
    return model, tokenizer, {str(key): str(value) for key, value in device_map.items()}, placement


def padded_batch(rows: list[dict[str, Any]], pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = max(len(row["input_ids"]) for row in rows)
    padded: list[list[int]] = []
    masks: list[list[int]] = []
    for row in rows:
        values = [int(value) for value in row["input_ids"]]
        padding = maximum - len(values)
        padded.append([pad_token_id] * padding + values)
        masks.append([0] * padding + [1] * len(values))
    return (
        torch.tensor(padded, dtype=torch.long, device="cuda:0"),
        torch.tensor(masks, dtype=torch.long, device="cuda:0"),
    )


def run(model_name: str) -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    protocol_audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    rows = list(protocol.read_jsonl(protocol.OUT_ROOT / "protocol" / "cases.jsonl"))
    if not protocol_audit["all_checks_passed"]:
        raise RuntimeError("Phase1119 protocol audit failed")
    if protocol.digest(rows) != prereg["case_digest"]:
        raise RuntimeError("Phase1119 case digest mismatch")
    if model_name not in prereg["models"]:
        raise RuntimeError("model is not frozen")

    started = time.time()
    model = None
    try:
        model, tokenizer, device_map, placement = load_model(model_name)
        precision = quantization_audit(model)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count != prereg["expected_parameter_counts"][model_name]:
            raise RuntimeError("parameter count mismatch")
        if precision["has_quantized_modules"] or precision["has_bf16_parameters"] or not precision["has_fp16_parameters"]:
            raise RuntimeError("FP16/no-quantization gate failed")

        torch.cuda.reset_peak_memory_stats()
        details: list[dict[str, Any]] = []
        batch_times: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(rows), BATCH_SIZE):
                batch = rows[start : start + BATCH_SIZE]
                input_ids, attention_mask = padded_batch(batch, int(tokenizer.pad_token_id))
                before = time.time()
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
                logits = output.logits[:, -1, :].float()
                batch_times.append(time.time() - before)
                top_ids = torch.argmax(logits, dim=-1)
                for slot, row in enumerate(batch):
                    true_scores = [float(logits[slot, token_id].item()) for token_id in row["true_candidate_ids"]]
                    control_scores = [
                        float(logits[slot, token_id].item())
                        for token_id in row["control_candidate_ids"]
                    ]
                    true_z = true_scores[0] - true_scores[1]
                    control_z = control_scores[0] - control_scores[1]
                    expected = int(row["sense"])
                    expected_margin = true_scores[expected] - true_scores[1 - expected]
                    finite = all(
                        math.isfinite(value)
                        for value in [*true_scores, *control_scores, true_z, control_z]
                    )
                    top_id = int(top_ids[slot].item())
                    details.append(
                        {
                            "schema_version": "phase1119_qwen3_scale_detail.v1",
                            "phase": protocol.PHASE,
                            "model": model_name,
                            "case_index": int(row["case_index"]),
                            "record_id": row["record_id"],
                            "pair_id": row["pair_id"],
                            "concept_id": row["concept_id"],
                            "split": row["split"],
                            "template": int(row["template"]),
                            "sense": expected,
                            "true_candidate_labels": row["true_candidate_labels"],
                            "control_concept_id": row["control_concept_id"],
                            "control_candidate_labels": row["control_candidate_labels"],
                            "true_scores": true_scores if finite else None,
                            "control_scores": control_scores if finite else None,
                            "true_z": true_z if finite else None,
                            "control_z": control_z if finite else None,
                            "expected_margin": expected_margin if finite else None,
                            "finite": finite,
                            "candidate_hit": finite and expected_margin > 0.0,
                            "top_token_id": top_id,
                            "top_token_text": tokenizer.decode([top_id]),
                            "direct_true_candidate": top_id in row["true_candidate_ids"],
                        }
                    )
                del output, logits, top_ids, input_ids, attention_mask
                print(
                    json.dumps(
                        {
                            "phase": protocol.PHASE,
                            "model": model_name,
                            "completed": min(start + BATCH_SIZE, len(rows)),
                            "total": len(rows),
                        }
                    ),
                    flush=True,
                )

        finite_rows = [row for row in details if row["finite"]]
        core = {
            "schema_version": "phase1119_qwen3_scale_behavior_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "case_digest": prereg["case_digest"],
            "model_manifest_digest": prereg["model_manifest_digests"][model_name],
            "precision": precision,
            "parameter_count": parameter_count,
            "placement": placement,
            "device_map": device_map,
            "batch_size": BATCH_SIZE,
            "batch_count": len(batch_times),
            "batch_seconds": batch_times,
            "case_count": len(details),
            "finite_fraction": len(finite_rows) / max(len(details), 1),
            "candidate_accuracy": sum(row["candidate_hit"] for row in finite_rows)
            / max(len(finite_rows), 1),
            "direct_true_candidate_rate": sum(row["direct_true_candidate"] for row in details)
            / max(len(details), 1),
            "detail_digest": protocol.digest(details),
            "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "elapsed_seconds": time.time() - started,
        }
        summary = dict(core)
        summary["summary_digest"] = protocol.digest(core)
        output_root = protocol.OUT_ROOT / "behavior" / model_name
        protocol.write_jsonl(output_root / "candidate_detail.jsonl", details)
        protocol.write_json(output_root / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=tuple(protocol.MODEL_ROOTS))
    run(parser.parse_args().model)
