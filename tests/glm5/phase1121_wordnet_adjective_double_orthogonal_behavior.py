#!/usr/bin/env python3
"""Run one frozen Phase1121 model in FP16 without quantization."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = ROOT / "tests" / "glm5"
sys.path.insert(0, str(TEST_ROOT))

import phase1121_wordnet_adjective_double_orthogonal_protocol as protocol
from phase1023_fp16_utils import load_fp16, quantization_audit, release_fp16


BATCH_SIZES = {"pythia": 16, "qwen3": 12, "glm4": 4, "deepseek7b": 4}


def load_model(model_name: str):
    if model_name != "pythia":
        return load_fp16(model_name)
    tokenizer = AutoTokenizer.from_pretrained(protocol.PYTHIA_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        protocol.PYTHIA_PATH,
        dtype=torch.float16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to("cuda")
    model.eval()
    return model, tokenizer, torch.device("cuda"), {
        "placement": "full_cuda",
        "max_memory": None,
        "parameter_dtypes": quantization_audit(model)["parameter_dtypes"],
        "quantization": "none",
    }


def run(model_name: str) -> None:
    prereg = protocol.read_json(protocol.OUT_ROOT / "protocol" / "preregistration.json")
    audit = protocol.read_json(protocol.OUT_ROOT / "protocol" / "audit.json")
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1121 protocol audit failed")
    rows = list(protocol.read_jsonl(protocol.OUT_ROOT / "protocol" / f"cases.{model_name}.jsonl"))
    if protocol.digest(rows) != prereg["case_digests"][model_name]:
        raise RuntimeError("Phase1121 case digest mismatch")

    started = time.time()
    model = None
    try:
        model, _tokenizer, device, placement = load_model(model_name)
        precision = quantization_audit(model)
        if precision["has_quantized_modules"] or precision["has_bf16_parameters"] or not precision["has_fp16_parameters"]:
            raise RuntimeError("FP16/no-quantization audit failed")

        by_length: dict[int, list[dict]] = defaultdict(list)
        for row in rows:
            by_length[len(row["input_ids"])].append(row)
        detail: list[dict] = []
        batch_size = BATCH_SIZES[model_name]
        with torch.inference_mode():
            for length in sorted(by_length):
                panel = by_length[length]
                for start in range(0, len(panel), batch_size):
                    batch = panel[start:start + batch_size]
                    input_ids = torch.tensor([row["input_ids"] for row in batch], dtype=torch.long, device=device)
                    attention_mask = torch.ones_like(input_ids)
                    output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
                    logits = output.logits[:, -1, :].float()
                    top_ids = torch.argmax(logits, dim=-1)
                    for slot, row in enumerate(batch):
                        candidate_ids = {key: int(values[0]) for key, values in row["candidate_first_token_ids"].items()}
                        scores = {key: float(logits[slot, token_id].item()) for key, token_id in candidate_ids.items()}
                        z = scores["true"] - scores["false"]
                        finite = all(math.isfinite(value) for value in [*scores.values(), z])
                        expected = row["expected_class"]
                        expected_margin = scores[expected] - scores["false" if expected == "true" else "true"]
                        top_id = int(top_ids[slot].item())
                        detail.append({
                            "schema_version": "phase1121_adjective_double_orthogonal_detail.v1",
                            "phase": protocol.PHASE,
                            "model": model_name,
                            "case_index": row["case_index"],
                            "record_id": row["record_id"],
                            "interaction_id": row["interaction_id"],
                            "concept_id": row["concept_id"],
                            "split": row["split"],
                            "template": row["template"],
                            "surface": row["surface"],
                            "context_sense": row["context_sense"],
                            "definition_sense": row["definition_sense"],
                            "truth": row["truth"],
                            "scores": scores if finite else None,
                            "z_true_minus_false": z if finite else None,
                            "expected_margin": expected_margin if finite else None,
                            "finite": finite,
                            "candidate_hit": finite and expected_margin > 0.0,
                            "top_token_id": top_id,
                            "direct_candidate": top_id in candidate_ids.values(),
                        })
                    del output, logits, top_ids, input_ids, attention_mask
                print(json.dumps({"phase": protocol.PHASE, "model": model_name, "length_complete": length}), flush=True)

        finite_rows = [row for row in detail if row["finite"]]
        summary_core = {
            "schema_version": "phase1121_adjective_double_orthogonal_behavior_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "case_digest": prereg["case_digests"][model_name],
            "precision": precision,
            "placement": placement,
            "case_count": len(detail),
            "detail_digest": protocol.digest(detail),
            "finite_fraction": len(finite_rows) / max(len(detail), 1),
            "candidate_accuracy": sum(bool(row["candidate_hit"]) for row in finite_rows) / max(len(finite_rows), 1),
            "direct_candidate_output_rate": sum(bool(row["direct_candidate"]) for row in detail) / max(len(detail), 1),
            "elapsed_seconds": time.time() - started,
        }
        summary = dict(summary_core)
        summary["summary_digest"] = protocol.digest(summary_core)
        output_root = protocol.OUT_ROOT / "behavior" / model_name
        protocol.write_jsonl(output_root / "candidate_detail.jsonl", detail)
        protocol.write_json(output_root / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
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
