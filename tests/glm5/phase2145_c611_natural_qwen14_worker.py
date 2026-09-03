#!/usr/bin/env python3
"""Sequential FP16 disk-offloaded Qwen3-14B natural-output worker for C611."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
MODEL_ROOT = ROOT / "models/hf/Qwen3-14B"
OFFLOAD_ROOT = ROOT / "tests/glm5/result/phase1118_qwen3_14b_fp16_offload_smoke/disk_offload_revision5"
sys.path.insert(0, str(TESTS))

import phase2141_c607_c611_natural_output_compiler_campaign as campaign


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_rows(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def device_map() -> dict:
    mapping = {"model.embed_tokens": 0, "model.rotary_emb": 0, "lm_head": "disk", "model.norm": "disk"}
    for layer in range(40):
        mapping[f"model.layers.{layer}"] = 0 if layer < 18 else "disk"
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--material", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw_dir = args.output.parent.parent / "raw/qwen3_14b"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in campaign.read_rows(args.material)
            if r["panel"] == "atomic" and r["surface"] == "open" and r["unit"] in {0, 6}]
    model = None
    states = None
    hooks = []
    captured = []
    weights_loaded = False
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_ROOT), local_files_only=True,
                                                   trust_remote_code=True, use_fast=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        compiled = campaign.compile_rows(tokenizer, rows)
        write_rows(raw_dir / "compiled.jsonl", compiled)
        config = AutoConfig.from_pretrained(str(MODEL_ROOT), local_files_only=True, trust_remote_code=True)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=torch.float16)
        OFFLOAD_ROOT.mkdir(parents=True, exist_ok=True)
        model = load_checkpoint_and_dispatch(
            model, checkpoint=str(MODEL_ROOT), device_map=device_map(),
            no_split_module_classes=list(model._no_split_modules), offload_folder=str(OFFLOAD_ROOT),
            offload_buffers=False, dtype=torch.float16, offload_state_dict=True,
            force_hooks=True, strict=True,
        )
        model.eval()
        weights_loaded = True
        candidate_scores = campaign.batch_candidate_scores(model, torch.device("cuda:0"), compiled, batch_size=2)
        behavior = []
        for i, (item, scores) in enumerate(zip(compiled, candidate_scores)):
            text = campaign.greedy_text(model, tokenizer, torch.device("cuda:0"), item["prompt_ids"])
            generated = campaign.generated_prediction(text, item["answer_candidates"])
            behavior.append({
                "case_id": item["case_id"], "candidate_prediction": int(np.argmax(scores)),
                "candidate_correct": int(np.argmax(scores)) == item["gold_position"],
                "generated_text": text, "generated_prediction": generated,
                "generated_correct": generated == item["gold_position"],
            })
            print(f"[qwen14 behavior] {i + 1}/{len(compiled)}", flush=True)
        write_rows(raw_dir / "behavior.jsonl", behavior)
        candidate_accuracy = float(np.mean([r["candidate_correct"] for r in behavior]))
        generated_accuracy = float(np.mean([r["generated_correct"] for r in behavior]))
        if candidate_accuracy < campaign.BEHAVIOR_GATE or generated_accuracy < campaign.BEHAVIOR_GATE:
            save(args.output, {
                "status": "behavior_unqualified", "model": "Qwen3-14B", "rows": len(rows),
                "candidate_accuracy": candidate_accuracy, "generated_accuracy": generated_accuracy,
                "hiddenstate_ran": False, "functional_candidate": False, "weights_loaded": weights_loaded,
            })
            return

        base = model.model
        layers = list(base.layers)
        checkpoints = len(layers) + 2
        coordinates = int(model.get_input_embeddings().weight.shape[1])
        states = np.lib.format.open_memmap(raw_dir / "role_last.float16.npy", mode="w+", dtype=np.float16,
                                           shape=(len(compiled), checkpoints, len(campaign.ROLES), coordinates))

        def hook(_module, _args, output):
            captured.append(output[0] if isinstance(output, tuple) else output)

        hooks.append(base.embed_tokens.register_forward_hook(hook))
        hooks.extend(layer.register_forward_hook(hook) for layer in layers)
        hooks.append(base.norm.register_forward_hook(hook))
        qpoints = sorted(set((0, round((checkpoints - 1) * .25), round((checkpoints - 1) * .5),
                                      round((checkpoints - 1) * .75), checkpoints - 1)))
        index = []
        representative = None
        for i, item in enumerate(compiled):
            ids = torch.tensor([item["prompt_ids"]], dtype=torch.long, device="cuda:0")
            mask = torch.ones_like(ids)
            pos = torch.arange(ids.shape[1], device="cuda:0")[None]
            captured.clear()
            with torch.inference_mode():
                model(input_ids=ids, attention_mask=mask, position_ids=pos,
                      use_cache=False, return_dict=True)
            if len(captured) != checkpoints:
                raise RuntimeError((len(captured), checkpoints))
            for q, hidden in enumerate(captured):
                for role_i, role in enumerate(campaign.ROLES):
                    at = int(item["role_positions"][role][-1])
                    states[i, q, role_i] = hidden[0, at].float().cpu().numpy().astype(np.float16)
            if representative is None and item["partition"] == "lockbox":
                representative = {
                    "case_id": item["case_id"], "qpoints": qpoints,
                    "roles": list(campaign.ROLES), "coordinates": coordinates,
                    "role_last_states": [
                        [captured[q][0, int(item["role_positions"][role][-1])].float().cpu().numpy().astype(np.float16).tolist()
                         for role in campaign.ROLES] for q in qpoints
                    ],
                }
            index.append({
                "hidden_index": i, "case_id": item["case_id"], "panel": item["panel"],
                "family": item["family"], "operation_domain": item["operation_domain"],
                "surface": item["surface"], "unit": item["unit"], "partition": item["partition"],
                "cell": item["cell"], "generated_correct": behavior[i]["generated_correct"],
            })
            states.flush()
            print(f"[qwen14 field] {i + 1}/{len(compiled)}", flush=True)
        write_rows(raw_dir / "hidden_index.jsonl", index)
        pairs = campaign.transition_pairs(index)
        grouped = defaultdict(list)
        for pair in pairs:
            grouped[pair["operation"]].append(pair)
        metrics = {}
        for operation, values in grouped.items():
            train = [p for p in values if p["partition"] == "discovery"]
            test = [p for p in values if p["partition"] == "lockbox"]
            if not train or not test:
                continue
            for q in qpoints:
                truth = np.stack([
                    np.asarray(states[p["right"]["hidden_index"], q], np.float32)
                    - np.asarray(states[p["left"]["hidden_index"], q], np.float32) for p in test
                ])
                prototype = np.mean([
                    np.asarray(states[p["right"]["hidden_index"], q], np.float32)
                    - np.asarray(states[p["left"]["hidden_index"], q], np.float32) for p in train
                ], axis=0)
                correct = campaign.previous.metric(np.broadcast_to(prototype, truth.shape), truth)
                zero = campaign.previous.metric(np.zeros_like(truth), truth)
                metrics[f"{operation}|q{q}"] = {
                    "samples": len(test), "prototype": correct, "zero": zero,
                    "gate": correct["nrmse"] <= zero["nrmse"] - campaign.CONTROL_MARGIN,
                }
        passes = sum(v["gate"] for v in metrics.values())
        total = len(metrics)
        raw_path = raw_dir / "role_last.float16.npy"
        save(args.output, {
            "status": "closed", "model": "Qwen3-14B", "precision": "float16", "rows": len(rows),
            "candidate_accuracy": candidate_accuracy, "generated_accuracy": generated_accuracy,
            "hiddenstate_ran": True, "checkpoints": checkpoints, "coordinates": coordinates,
            "qpoints": qpoints, "shape": list(states.shape), "raw_path": str(raw_path.relative_to(ROOT)),
            "raw_bytes": raw_path.stat().st_size, "metric_passes": passes, "metric_total": total,
            "metrics": metrics, "functional_candidate": total > 0 and passes / total >= .5,
            "representative": representative, "weights_loaded": weights_loaded,
            "device_map": {str(k): str(v) for k, v in model.hf_device_map.items()},
            "strict_interpretation": "Within-model response topology only; coordinate IDs are not aligned across models.",
        })
    except Exception as error:
        save(args.output, {"status": "worker_error", "model": "Qwen3-14B", "weights_loaded": weights_loaded,
                           "error_type": type(error).__name__, "error": str(error),
                           "hiddenstate_ran": False, "functional_candidate": False})
        raise
    finally:
        for handle in hooks:
            handle.remove()
        if states is not None:
            states.flush()
            del states
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
