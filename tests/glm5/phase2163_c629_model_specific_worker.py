#!/usr/bin/env python3
"""Sequential model-specific behavior and qualified HiddenState worker for C629."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))

import phase2159_c625_c629_flagship_gear_campaign as campaign


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_model(name: str):
    if name != "qwen3_14b":
        return (*campaign.previous.c607.passport.previous.model_base().load_bf16(name), "standard")
    import phase2145_c611_natural_qwen14_worker as q14
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(q14.MODEL_ROOT), local_files_only=True, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(str(q14.MODEL_ROOT), local_files_only=True, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=torch.float16)
    q14.OFFLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    model = load_checkpoint_and_dispatch(
        model, checkpoint=str(q14.MODEL_ROOT), device_map=q14.device_map(),
        no_split_module_classes=list(model._no_split_modules), offload_folder=str(q14.OFFLOAD_ROOT),
        offload_buffers=False, dtype=torch.float16, offload_state_dict=True, force_hooks=True, strict=True)
    model.eval()
    return model, tokenizer, torch.device("cuda:0"), "fp16_disk_offload", "qwen14"


def release_model(name: str, model) -> None:
    if name == "qwen3_14b":
        if model is not None:
            del model
        torch.cuda.empty_cache()
    else:
        campaign.previous.c607.passport.previous.model_base().release_bf16(model)
    gc.collect()


def capture_hidden(model, device, compiled: list[dict], output_dir: Path) -> dict:
    base = model.model
    modules = [base.embed_tokens, *list(base.layers), base.norm]
    coordinates = int(base.embed_tokens.weight.shape[1])
    selected = compiled[:6]
    target = output_dir / "role_fields.float16.npy"
    fields = np.lib.format.open_memmap(target, mode="w+", dtype=np.float16,
                                       shape=(len(selected), len(modules), len(campaign.ROLES), coordinates))
    hooks, captured = [], []
    def hook(_module, _args, output):
        captured.append(output[0] if isinstance(output, tuple) else output)
    hooks = [module.register_forward_hook(hook) for module in modules]
    try:
        for row_i, item in enumerate(selected):
            ids = torch.tensor([item["prompt_ids"]], dtype=torch.long, device=device)
            mask = torch.ones_like(ids); pos = torch.arange(ids.shape[1], device=device)[None]
            captured.clear()
            with torch.inference_mode():
                model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
            if len(captured) != len(modules):
                raise RuntimeError((len(captured), len(modules)))
            for q, tensor in enumerate(captured):
                for role_i, role in enumerate(campaign.ROLES):
                    fields[row_i, q, role_i] = tensor[0, int(item["role_positions"][role][-1])].float().cpu().numpy().astype(np.float16)
            print(f"[hidden] {row_i + 1}/{len(selected)}", flush=True)
    finally:
        for handle in hooks:
            handle.remove()
    fields.flush()
    values = np.asarray(fields, np.float32)
    role_rms = np.sqrt(np.mean(values * values, axis=(0, 3)))
    role_rms = role_rms / (np.sqrt(np.sum(role_rms * role_rms, axis=1, keepdims=True)) + 1e-12)
    topology = [{"relative_depth": q / max(1, len(modules) - 1),
                 "role_rms_normalized": role_rms[q].tolist()} for q in range(len(modules))]
    save(output_dir / "role_topology.json", topology)
    fields.flush(); del values, fields
    return {"hiddenstate_ran": True, "hidden_rows": len(selected), "checkpoints": len(modules),
            "coordinates": coordinates, "role_field": str(target.relative_to(ROOT)),
            "role_topology": str((output_dir / "role_topology.json").relative_to(ROOT))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("glm4", "deepseek7b", "qwen3_14b"), required=True)
    parser.add_argument("--material", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [row for row in campaign.read_rows(args.material) if row["cross_model_subset"]]
    model = None
    try:
        model, tokenizer, device, placement, loader = load_model(args.model)
        compiled = campaign.compile_rows(tokenizer, rows)
        scores_all = campaign.previous.c607.batch_candidate_scores(
            model, device, compiled, batch_size=2 if args.model == "qwen3_14b" else 8)
        behavior = []
        for i, (item, scores) in enumerate(zip(compiled, scores_all)):
            text = campaign.previous.c607.greedy_text(model, tokenizer, device, item["prompt_ids"], max_new_tokens=24)
            candidate = int(np.argmax(scores)); generated = campaign.generated_prediction(text, item["answer_candidates"])
            behavior.append({"case_id": item["case_id"], "candidate_correct": candidate == item["gold_position"],
                             "generated_text": text, "generated_correct": generated == item["gold_position"]})
            print(f"[{args.model}] behavior {i + 1}/{len(compiled)}", flush=True)
        campaign.write_rows(args.output.parent / "behavior.jsonl", behavior)
        candidate_accuracy = float(np.mean([row["candidate_correct"] for row in behavior]))
        generated_accuracy = float(np.mean([row["generated_correct"] for row in behavior]))
        qualified = candidate_accuracy >= campaign.BEHAVIOR_GATE and generated_accuracy >= campaign.BEHAVIOR_GATE
        hidden = {"hiddenstate_ran": False}
        if qualified:
            correct_ids = {row["case_id"] for row in behavior if row["candidate_correct"] and row["generated_correct"]}
            hidden = capture_hidden(model, device, [row for row in compiled if row["case_id"] in correct_ids], args.output.parent)
        save(args.output, {"status": "closed" if qualified else "behavior_unqualified", "model": args.model,
                           "rows": len(rows), "candidate_accuracy": candidate_accuracy,
                           "generated_accuracy": generated_accuracy, "placement": placement, "loader": loader,
                           **hidden,
                           "strict_interpretation": "A qualified topology is model-specific; physical coordinate IDs are not aligned across models."})
    except Exception as error:
        save(args.output, {"status": "worker_error", "model": args.model, "error_type": type(error).__name__,
                           "error": str(error), "hiddenstate_ran": False})
        raise
    finally:
        release_model(args.model, model)


if __name__ == "__main__":
    main()
