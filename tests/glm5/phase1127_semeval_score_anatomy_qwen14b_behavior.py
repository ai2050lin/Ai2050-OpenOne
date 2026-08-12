#!/usr/bin/env python3
"""Run the frozen Phase1127 Qwen3-14B FP16 SemEval behavior endpoint."""

from __future__ import annotations

import gc
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

from phase1023_fp16_utils import quantization_audit
import phase1127_semeval_score_anatomy_qwen14b_protocol as protocol


def load_model(prereg: dict[str, Any]) -> tuple[Any, Any, dict[str, str]]:
    local_path = protocol.MODEL_ROOT
    tokenizer = AutoTokenizer.from_pretrained(
        local_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(local_path, local_files_only=True, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config,
            dtype=torch.float16,
            trust_remote_code=True,
        )
    model.tie_weights()
    device_map = {
        key: (int(value) if str(value).isdigit() else value)
        for key, value in prereg["model"]["device_map"].items()
    }
    offload_root = protocol.OUT_ROOT / "offload" / "qwen3_14b"
    offload_root.mkdir(parents=True, exist_ok=True)
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
    model.eval()
    return model, tokenizer, {str(key): str(value) for key, value in model.hf_device_map.items()}


def padded_batch(rows: list[dict[str, Any]], pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    maximum = max(len(row["input_ids"]) for row in rows)
    padded: list[list[int]] = []
    masks: list[list[int]] = []
    offsets: list[int] = []
    for row in rows:
        values = [int(value) for value in row["input_ids"]]
        padding = maximum - len(values)
        padded.append([pad_token_id] * padding + values)
        masks.append([0] * padding + [1] * len(values))
        offsets.append(padding)
    return (
        torch.tensor(padded, dtype=torch.long, device="cuda:0"),
        torch.tensor(masks, dtype=torch.long, device="cuda:0"),
        offsets,
    )


def score_batch(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    rows: list[dict[str, Any]],
    offsets: list[int],
) -> list[dict[str, float | None]]:
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    logits = output.logits
    selected_rows: list[int] = []
    selected_prediction_positions: list[int] = []
    selected_target_ids: list[int] = []
    selected_kinds: list[str] = []
    selected_case_offsets: list[int] = []
    for batch_index, row in enumerate(rows):
        candidate_positions = [int(value) for value in row["candidate_positions"]]
        positions = [*candidate_positions, *[int(value) for value in row["suffix_positions"]]]
        for local_index, position in enumerate(positions):
            selected_rows.append(batch_index)
            selected_prediction_positions.append(offsets[batch_index] + position - 1)
            selected_target_ids.append(int(row["input_ids"][position]))
            selected_kinds.append("candidate" if local_index < len(candidate_positions) else "suffix")
            selected_case_offsets.append(batch_index)

    row_index = torch.tensor(selected_rows, device=logits.device, dtype=torch.long)
    prediction_index = torch.tensor(selected_prediction_positions, device=logits.device, dtype=torch.long)
    target_ids = torch.tensor(selected_target_ids, device=logits.device, dtype=torch.long)
    selected_logits = logits[row_index, prediction_index, :].float()
    selected_log_probs = (
        selected_logits.gather(1, target_ids.unsqueeze(1)).squeeze(1)
        - torch.logsumexp(selected_logits, dim=-1)
    ).detach().cpu().tolist()

    per_case: list[dict[str, list[float]]] = [{"candidate": [], "suffix": []} for _ in rows]
    for case_offset, kind, value in zip(selected_case_offsets, selected_kinds, selected_log_probs):
        per_case[case_offset][kind].append(float(value))

    results: list[dict[str, float | None]] = []
    for values in per_case:
        candidate_logp = sum(values["candidate"])
        suffix_mean = sum(values["suffix"]) / len(values["suffix"]) if values["suffix"] else 0.0
        finite = math.isfinite(candidate_logp) and math.isfinite(suffix_mean)
        results.append({
            "candidate_logp": candidate_logp if finite else None,
            "suffix_mean_logp": suffix_mean if finite else None,
            "total_score": candidate_logp + suffix_mean if finite else None,
        })
    del output, logits, selected_logits
    return results


def main() -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    rows = protocol.read_jsonl(protocol.OUT_ROOT / "protocol" / "cases.qwen3_14b.jsonl")
    if not audit["passed"] or audit["protocol_digest"] != prereg["protocol_digest"]:
        raise RuntimeError("Phase1127 protocol is not authorized")
    if protocol.digest(rows) != prereg["carrier"]["case_digest"]:
        raise RuntimeError("Phase1127 case digest mismatch")

    started = time.time()
    model = None
    try:
        model, tokenizer, device_map = load_model(prereg)
        precision = quantization_audit(model)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count != prereg["model"]["expected_parameter_count"]:
            raise RuntimeError("Qwen3-14B parameter count mismatch")
        if precision["has_quantized_modules"] or precision["has_bf16_parameters"] or not precision["has_fp16_parameters"]:
            raise RuntimeError("Phase1127 FP16/no-quantization audit failed")

        torch.cuda.reset_peak_memory_stats()
        buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            bucket = ((len(row["input_ids"]) + 7) // 8) * 8
            buckets[bucket].append(row)

        details: list[dict[str, Any]] = []
        batch_times: list[float] = []
        completed = 0
        with torch.inference_mode():
            for bucket in sorted(buckets):
                panel = buckets[bucket]
                for start in range(0, len(panel), int(prereg["carrier"]["batch_size"])):
                    batch = panel[start : start + int(prereg["carrier"]["batch_size"])]
                    input_ids, attention_mask, offsets = padded_batch(batch, int(tokenizer.pad_token_id))
                    before = time.time()
                    scores = score_batch(model, input_ids, attention_mask, batch, offsets)
                    batch_times.append(time.time() - before)
                    for row, score in zip(batch, scores):
                        finite = all(score[key] is not None for key in ("candidate_logp", "suffix_mean_logp", "total_score"))
                        details.append({
                            "case_index": row["case_index"],
                            "panel_index": row["panel_index"],
                            "item": row["item"],
                            "pos": row["pos"],
                            "partition": row["partition"],
                            "replica": row["replica"],
                            "route": row["route"],
                            "route_item": row["route_item"],
                            "context_sense": row["context_sense"],
                            "candidate_side": row["candidate_side"],
                            "candidate": row["candidate"],
                            "source_instance_id": row["source_instance_id"],
                            "lexical_overlap": row["lexical_overlap"],
                            "suffix_token_count": len(row["suffix_positions"]),
                            **score,
                            "finite": finite,
                        })
                    completed += len(batch)
                    print(json.dumps({"phase": protocol.PHASE, "completed": completed, "total": len(rows)}), flush=True)
                    del input_ids, attention_mask

        details.sort(key=lambda row: row["case_index"])
        if [row["case_index"] for row in details] != list(range(len(rows))):
            raise RuntimeError("Phase1127 result case order mismatch")
        finite_count = sum(bool(row["finite"]) for row in details)
        core = {
            "schema_version": "phase1127_qwen3_14b_semeval_behavior.v1",
            "phase": protocol.PHASE,
            "model": "qwen3_14b",
            "protocol_digest": prereg["protocol_digest"],
            "case_digest": prereg["carrier"]["case_digest"],
            "model_manifest_digest": prereg["model"]["manifest_digest"],
            "case_count": len(details),
            "finite_count": finite_count,
            "finite_rate": finite_count / len(details),
            "precision": precision,
            "parameter_count": parameter_count,
            "placement": "cuda_disk_offload",
            "device_map": device_map,
            "batch_size": prereg["carrier"]["batch_size"],
            "batch_count": len(batch_times),
            "batch_seconds": batch_times,
            "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "elapsed_seconds": time.time() - started,
            "detail_digest": protocol.digest(details),
        }
        summary = dict(core)
        summary["summary_digest"] = protocol.digest(core)
        output_root = protocol.OUT_ROOT / "behavior" / "qwen3_14b"
        protocol.write_jsonl(output_root / "scores.jsonl", details)
        protocol.write_json(output_root / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
