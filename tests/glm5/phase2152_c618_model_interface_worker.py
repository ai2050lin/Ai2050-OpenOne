#!/usr/bin/env python3
"""Sequential model-specific behavior worker for C618."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))

import phase2147_c613_c620_conditional_gear_campaign as campaign


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("glm4", "deepseek7b", "qwen3_14b"), required=True)
    parser.add_argument("--material", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [r for r in campaign.read_rows(args.material) if r["cross_model_subset"]]
    model = None
    try:
        if args.model == "qwen3_14b":
            # The repository's proven disk-offload path is intentionally reused.
            import phase2145_c611_natural_qwen14_worker as q14
            from accelerate import init_empty_weights, load_checkpoint_and_dispatch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
            import torch
            tokenizer = AutoTokenizer.from_pretrained(str(q14.MODEL_ROOT), local_files_only=True, trust_remote_code=True, use_fast=False)
            if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
            compiled = campaign.compile_rows(tokenizer, rows)
            config = AutoConfig.from_pretrained(str(q14.MODEL_ROOT), local_files_only=True, trust_remote_code=True)
            with init_empty_weights(): model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=torch.float16)
            q14.OFFLOAD_ROOT.mkdir(parents=True, exist_ok=True)
            model = load_checkpoint_and_dispatch(model, checkpoint=str(q14.MODEL_ROOT), device_map=q14.device_map(),
                no_split_module_classes=list(model._no_split_modules), offload_folder=str(q14.OFFLOAD_ROOT),
                offload_buffers=False, dtype=torch.float16, offload_state_dict=True, force_hooks=True, strict=True)
            model.eval(); device = torch.device("cuda:0"); placement = "fp16_disk_offload"
        else:
            model, tokenizer, device, placement = campaign.c607.passport.previous.model_base().load_bf16(args.model)
            compiled = campaign.compile_rows(tokenizer, rows)
        scores_all = campaign.c607.batch_candidate_scores(model, device, compiled, batch_size=2 if args.model == "qwen3_14b" else 8)
        behavior = []
        for i, (item, scores) in enumerate(zip(compiled, scores_all)):
            text = campaign.c607.greedy_text(model, tokenizer, device, item["prompt_ids"], max_new_tokens=32)
            pred, gen = int(np.argmax(scores)), campaign.generated_prediction(text, item["answer_candidates"])
            behavior.append({"case_id": item["case_id"], "candidate_correct": pred == item["gold_position"],
                             "generated_text": text, "generated_correct": gen == item["gold_position"]})
            print(f"[{args.model}] {i + 1}/{len(compiled)}", flush=True)
        ca = float(np.mean([r["candidate_correct"] for r in behavior])); ga = float(np.mean([r["generated_correct"] for r in behavior]))
        save(args.output, {"status": "closed" if ca >= .75 and ga >= .75 else "behavior_unqualified",
                           "model": args.model, "rows": len(rows), "candidate_accuracy": ca, "generated_accuracy": ga,
                           "hiddenstate_ran": False, "placement": placement,
                           "strict_interpretation": "This phase qualifies an output interface only; internal comparison remains a later contract."})
        campaign.write_rows(args.output.parent.parent / f"raw/{args.model}_behavior.jsonl", behavior)
    except Exception as error:
        save(args.output, {"status": "worker_error", "model": args.model, "error_type": type(error).__name__,
                           "error": str(error), "hiddenstate_ran": False})
        raise
    finally:
        if args.model == "qwen3_14b":
            if model is not None: del model
        else:
            campaign.c607.passport.previous.model_base().release_bf16(model)
        gc.collect()


if __name__ == "__main__":
    main()
