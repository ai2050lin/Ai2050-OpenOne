#!/usr/bin/env python3
"""Isolated Qwen3-14B FP16 CUDA-plus-disk worker for C597."""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))

import phase1797_c263_c272_state_operator_common as compiler
import phase2125_c591_c598_fresh_scope_lockbox_campaign as campaign

MODEL_ROOT = ROOT / "models/hf/Qwen3-14B"
OFFLOAD_ROOT = ROOT / "tests/glm5/result/phase1118_qwen3_14b_fp16_offload_smoke/disk_offload_revision5"
FAMILIES = ("discourse_permutation", "fact_voice_fixed_query", "evidence_paraphrase", "clause_packaging", "path_depth", "translation_language")
UNITS = tuple(range(8))


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_rows(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def metric(prediction: np.ndarray, truth: np.ndarray) -> dict:
    p = np.asarray(prediction, np.float64).reshape(-1)
    y = np.asarray(truth, np.float64).reshape(-1)
    error = p - y
    denom = math.sqrt(float(np.mean(y * y))) + 1e-12
    return {"nrmse": math.sqrt(float(np.mean(error * error))) / denom, "cosine": float(np.dot(p, y) / (np.linalg.norm(p) * np.linalg.norm(y) + 1e-12)), "sign_agreement": float(np.mean(np.sign(p) == np.sign(y)))}


def role_bundle(states: np.ndarray, row: dict, q: int) -> np.ndarray:
    values = []
    for role in campaign.ROLES:
        positions = [int(v) for v in row["role_positions"][role]]
        values.append(np.asarray(states[row["hidden_index"], q, positions], np.float32).mean(axis=0))
    return np.stack(values)


def device_map() -> dict:
    result = {"model.embed_tokens": 0}
    result.update({f"model.layers.{i}": 0 if i < 18 else "disk" for i in range(40)})
    result.update({"model.norm": "disk", "model.rotary_emb": "cpu", "lm_head": "disk"})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--material", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.parent.parent / "raw/qwen3_14b"
    out.mkdir(parents=True, exist_ok=True)
    all_rows = campaign.read_rows(args.material)
    rows = [row for row in all_rows if row["panel"] == "atomic" and row["family"] in FAMILIES and row["surface"] == "record" and row["unit"] in UNITS and row["operation_domain"] == campaign.previous.ATOMIC_SPECS[row["family"]][0]]
    model = None
    hooks = []
    captured = []
    states = None
    weights_loaded = False
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_ROOT), local_files_only=True, trust_remote_code=True, use_fast=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        compiled = compiler.compile_qwen(tokenizer, rows)
        write_rows(out / "compiled.jsonl", compiled)
        config = AutoConfig.from_pretrained(str(MODEL_ROOT), local_files_only=True, trust_remote_code=True)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, dtype=torch.float16, trust_remote_code=True)
        model.tie_weights()
        model = load_checkpoint_and_dispatch(model, checkpoint=str(MODEL_ROOT), device_map=device_map(), no_split_module_classes=list(model._no_split_modules), offload_folder=str(OFFLOAD_ROOT), offload_buffers=False, dtype=torch.float16, offload_state_dict=True, force_hooks=True, strict=True)
        weights_loaded = True
        model.eval()
        behavior = []
        pad = int(tokenizer.pad_token_id)
        for start in range(0, len(compiled), 4):
            batch = compiled[start:start + 4]
            width = max(len(r["prompt_ids"]) for r in batch)
            ids = torch.full((len(batch), width), pad, dtype=torch.long, device="cuda:0")
            mask = torch.zeros_like(ids)
            for i, row in enumerate(batch):
                seq = row["prompt_ids"]
                ids[i, :len(seq)] = torch.tensor(seq, device="cuda:0")
                mask[i, :len(seq)] = 1
            pos = mask.long().cumsum(-1) - 1
            pos.masked_fill_(mask == 0, 0)
            with torch.inference_mode():
                logits = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True).logits
            for i, row in enumerate(batch):
                length = len(row["prompt_ids"])
                scores = [float(logits[i, length - 1, candidate[0]]) for candidate in row["candidate_ids"]]
                pred = int(scores[1] > scores[0])
                behavior.append({"case_id": row["case_id"], "correct": pred == row["gold_position"], "prediction": pred, "scores": scores})
            print(f"[Qwen3-14B behavior] {min(start + len(batch), len(compiled))}/{len(compiled)}", flush=True)
        write_rows(out / "behavior.jsonl", behavior)
        accuracy = float(np.mean([r["correct"] for r in behavior]))
        if accuracy < 0.75:
            save(args.output, {"status": "behavior_unqualified", "model": "Qwen3-14B", "rows": len(rows), "behavior_accuracy": accuracy, "hiddenstate_ran": False, "functional_candidate": False, "weights_loaded": weights_loaded})
            raise SystemExit(2)

        base = model.model
        layers = list(base.layers)
        checkpoints = len(layers) + 2
        coordinates = int(model.get_input_embeddings().weight.shape[1])
        width = max(len(r["prompt_ids"]) for r in compiled)
        n = len(compiled)
        raw_path = out / "full_token_states.float16.npy"
        states = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.float16, shape=(n, checkpoints, width, coordinates))
        def hook(_module, _args, output):
            captured.append(output[0] if isinstance(output, tuple) else output)
        hooks.append(base.embed_tokens.register_forward_hook(hook))
        hooks.extend(layer.register_forward_hook(hook) for layer in layers)
        hooks.append(base.norm.register_forward_hook(hook))
        index = []
        for i, row in enumerate(compiled):
            seq = row["prompt_ids"]
            ids = torch.tensor([seq], dtype=torch.long, device="cuda:0")
            mask = torch.ones_like(ids)
            pos = torch.arange(len(seq), device="cuda:0")[None]
            captured.clear()
            with torch.inference_mode():
                model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
            if len(captured) != checkpoints:
                raise RuntimeError((len(captured), checkpoints))
            for q, state in enumerate(captured):
                states[i, q, :len(seq)] = state[0, :len(seq)].float().cpu().numpy().astype(np.float16)
            source = rows[i]
            index.append({"hidden_index": i, "case_id": row["case_id"], "family": source["family"], "operation_domain": source["operation_domain"], "unit": source["unit"], "variant": source["variant"], "partition": source["partition"], "length": len(seq), "role_positions": row["role_positions"]})
            if i % 8 == 0 or i + 1 == n:
                states.flush()
                print(f"[Qwen3-14B field] {i + 1}/{n}", flush=True)
        write_rows(out / "hidden_index.jsonl", index)
        pairs = defaultdict(dict)
        for row in index:
            pairs[(row["family"], row["unit"])][row["variant"]] = row
        qpoints = sorted(set((0, round((checkpoints - 1) * 0.25), round((checkpoints - 1) * 0.5), round((checkpoints - 1) * 0.75), checkpoints - 1)))
        metrics = {}
        gates = {}
        representatives = {}
        family_summary = {}
        for family in FAMILIES:
            train = [v for (f, unit), v in pairs.items() if f == family and unit < 4 and set(v) == {0, 1}]
            test = [v for (f, unit), v in pairs.items() if f == family and unit >= 4 and set(v) == {0, 1}]
            for q in qpoints:
                tr = np.stack([role_bundle(states, v[1], q) - role_bundle(states, v[0], q) for v in train])
                te = np.stack([role_bundle(states, v[1], q) - role_bundle(states, v[0], q) for v in test])
                proto = tr.mean(axis=0)
                value = {"samples": len(test), "correct": metric(np.broadcast_to(proto, te.shape), te), "zero": metric(np.zeros_like(te), te)}
                value["gate"] = value["correct"]["nrmse"] <= value["zero"]["nrmse"] - campaign.CONTROL_MARGIN
                key = f"{family}|q{q}"
                metrics[key] = value
                gates[key] = value["gate"]
                representatives[key] = proto.tolist()
            passed = sum(v for k, v in gates.items() if k.startswith(family + "|"))
            total = sum(k.startswith(family + "|") for k in gates)
            family_summary[family] = {"passed": passed, "total": total, "pass_rate": passed / max(total, 1), "candidate": passed / max(total, 1) >= 0.75}
        result = {"status": "closed", "model": "Qwen3-14B", "precision": "float16", "rows": n, "behavior_accuracy": accuracy, "hiddenstate_ran": True, "checkpoints": checkpoints, "coordinates": coordinates, "qpoints": qpoints, "shape": list(states.shape), "raw_path": str(raw_path.relative_to(ROOT)), "raw_bytes": raw_path.stat().st_size, "metrics": metrics, "gates": gates, "family_summary": family_summary, "representative_full_coordinates": representatives, "functional_candidate": any(v["candidate"] for v in family_summary.values()), "device_map": {str(k): str(v) for k, v in model.hf_device_map.items()}, "weights_loaded": weights_loaded, "strict_interpretation": "model-internal fresh-material topology only; no coordinate identity with smaller or different models"}
        save(args.output, result)
    except SystemExit:
        raise
    except Exception as error:
        save(args.output, {"status": "worker_error", "model": "Qwen3-14B", "weights_loaded": weights_loaded, "error_type": type(error).__name__, "error": str(error), "hiddenstate_ran": False, "functional_candidate": False})
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
