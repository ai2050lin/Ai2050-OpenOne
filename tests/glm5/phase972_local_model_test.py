#!/usr/bin/env python3
"""
Phase 972: 三模型本地调用基础测试
==================================
调用本地 deepseek7b, qwen3, glm4 进行基础测试:
Task 1: 基础生成能力 (10 prompts)
Task 2: EOS停止控制 (自然 vs b注入)
Task 3: gap测量与跨模型对比

测试模型: qwen3(最小,先测) -> deepseek7b -> glm4(最大,最后)
"""

from __future__ import annotations
import gc, json, sys, time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
from model_utils import load_model, get_layers, get_model_info, release_model, get_W_U
from phase951_protocol_atlas import ensure_dir
from phase966_natural_stop import log
from phase964_forward_diff import make_eos_inject_hook, get_head_dims, EN_PROMPTS_50

PHASE = 972
RESULT_DIR = Path("results/phase972_local_model_test")

# 测试prompts (覆盖不同类型)
TEST_PROMPTS = [
    # 事实型
    "The capital of France is",
    "The largest planet is",
    "Water boils at",
    "The speed of light is",
    # 定义型
    "The sun is a",
    "Dogs are",
    "Ice is",
    "Gold is a",
    # 因果型
    "Fire needs",
    "Rain falls because",
    # 中文
    "法国的首都是",
    "太阳是一颗",
]

EXPECTED = {
    "The capital of France is": ["paris"],
    "The largest planet is": ["jupiter"],
    "Water boils at": ["100", "212", "celsius"],
    "The speed of light is": ["299", "300", "186"],
    "The sun is a": ["star"],
    "Dogs are": ["animal", "mammal", "pet", "dog"],
    "Ice is": ["frozen", "solid", "water", "cold"],
    "Gold is a": ["metal", "element", "precious"],
    "Fire needs": ["oxygen", "air", "fuel", "heat"],
    "Rain falls because": ["water", "cloud", "gravity"],
    "法国的首都是": ["巴黎", "paris"],
    "太阳是一颗": ["星", "恒星", "star"],
}


def evaluate_answer(prompt: str, generated: str, has_eos: bool, n_tokens: int) -> Dict[str, Any]:
    """简单评估: 是否含expected关键词 + 是否停止 + 是否简短."""
    content = generated.strip()
    equiv = EXPECTED.get(prompt, [])
    has_expected = len(equiv) == 0 or any(e.lower() in content.lower() for e in equiv)
    is_short = 0 < n_tokens < 20
    return {
        "has_eos": has_eos,
        "has_expected": has_expected,
        "is_short": is_short,
        "clean": has_eos and is_short and has_expected,
        "content": content[:80],
    }


def task1_basic_generation(model, tokenizer, device, eos_id, model_name: str):
    """Task 1: 基础生成能力测试 (自然生成, 无干预)."""
    log(f"  Task 1: Basic generation (natural, no intervention)...")
    results = []
    t0 = time.time()
    for prompt in TEST_PROMPTS:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        try:
            with torch.no_grad():
                oid = model.generate(input_ids, max_new_tokens=30, do_sample=False, pad_token_id=eos_id)
            gt = oid[0][input_ids.shape[1]:]
            gen = tokenizer.decode(gt, skip_special_tokens=False)
            has_eos = gt[-1].item() == eos_id
            n_tokens = len(gt)
        except Exception as e:
            gen = f"ERROR: {e}"; has_eos = False; n_tokens = 0
        ev = evaluate_answer(prompt, gen, has_eos, n_tokens)
        results.append({"prompt": prompt, "generated": gen[:100], **ev})
        status = "OK" if ev["clean"] else ("EOS" if has_eos else "NO_EOS")
        log(f"    [{status}] '{prompt[:25]}': {gen[:50]}")
    n = len(results)
    clean = sum(r["clean"] for r in results)
    eos = sum(r["has_eos"] for r in results)
    expected = sum(r["has_expected"] for r in results)
    log(f"  Task 1 result: clean={clean}/{n}  eos={eos}/{n}  expected={expected}/{n}  ({time.time()-t0:.0f}s)")
    return {"summary": {"clean_rate": clean/n, "eos_rate": eos/n, "expected_rate": expected/n, "n": n},
            "raw_results": results}


def task2_eos_injection(model, tokenizer, device, eos_id, model_name: str):
    """Task 2: EOS注入测试 (b=0,5,10,15,20 + delay=2)."""
    log(f"  Task 2: EOS injection test (b in [0,5,10,15,20], delay=2)...")
    biases = [0, 5, 10, 15, 20]
    test_prompts = TEST_PROMPTS[:6]  # 前6个
    results = []
    t0 = time.time()
    for b in biases:
        b_results = []
        for prompt in test_prompts:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            if b > 0:
                h = model.lm_head.register_forward_hook(make_eos_inject_hook(eos_id, b, 2))
            try:
                with torch.no_grad():
                    oid = model.generate(input_ids, max_new_tokens=30, do_sample=False, pad_token_id=eos_id)
                gt = oid[0][input_ids.shape[1]:]
                gen = tokenizer.decode(gt, skip_special_tokens=False)
                has_eos = gt[-1].item() == eos_id
                n_tokens = len(gt)
            except Exception as e:
                gen = f"ERROR: {e}"; has_eos = False; n_tokens = 0
            if b > 0: h.remove()
            ev = evaluate_answer(prompt, gen, has_eos, n_tokens)
            b_results.append({"prompt": prompt, "b": b, "generated": gen[:60], **ev})
        clean = sum(r["clean"] for r in b_results)
        eos = sum(r["has_eos"] for r in b_results)
        results.append({"b": b, "clean": clean, "eos": eos, "n": len(b_results), "details": b_results})
        log(f"    b={b:2d}: clean={clean}/{len(b_results)}  eos={eos}/{len(b_results)}")
    log(f"  Task 2 done ({time.time()-t0:.0f}s)")
    return {"summary": [{"b": r["b"], "clean_rate": r["clean"]/r["n"], "eos_rate": r["eos"]/r["n"]} for r in results],
            "raw_results": results}


def task3_gap_measurement(model, tokenizer, device, eos_id, model_name: str):
    """Task 3: gap测量 (top1 logit - EOS logit)."""
    log(f"  Task 3: Gap measurement...")
    results = []
    t0 = time.time()
    for prompt in TEST_PROMPTS[:8]:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(input_ids, use_cache=False).logits[0, -1].float().cpu().numpy()
        top1_val = float(np.sort(logits)[-1])
        top1_id = int(logits.argmax())
        eos_val = float(logits[eos_id]) if eos_id else 0
        gap = top1_val - eos_val
        top1_token = tokenizer.decode([top1_id])
        results.append({"prompt": prompt, "gap": gap, "top1_val": top1_val, "eos_val": eos_val,
                        "top1_token": top1_token, "top1_id": top1_id})
        log(f"    '{prompt[:25]}': gap={gap:.2f}  top1='{top1_token}'({top1_val:.2f})  eos={eos_val:.2f}")
    mean_gap = float(np.mean([r["gap"] for r in results]))
    log(f"  Mean gap: {mean_gap:.2f}  ({time.time()-t0:.0f}s)")
    return {"mean_gap": mean_gap, "raw_results": results}


def run_model(model_name: str):
    log(f"\n{'='*60}\nPhase 972: {model_name}\n{'='*60}")
    model_dir = RESULT_DIR / model_name
    ensure_dir(model_dir)

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    eos_id = tokenizer.eos_token_id
    log(f"  {info.model_class}, {info.n_layers}L, d={info.d_model}, vocab={info.vocab_size}")
    log(f"  device={device}, eos_id={eos_id}")

    results = {"model": model_name, "model_info": {"class": info.model_class, "n_layers": info.n_layers,
               "d_model": info.d_model, "vocab_size": info.vocab_size}}
    t_start = time.time()

    # Task 1: 基础生成
    r1 = task1_basic_generation(model, tokenizer, device, eos_id, model_name)
    results["task1"] = r1
    (model_dir / "task1_basic_generation.json").write_text(
        json.dumps(r1, ensure_ascii=False, indent=2), encoding="utf-8")
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Task 2: EOS注入
    r2 = task2_eos_injection(model, tokenizer, device, eos_id, model_name)
    results["task2"] = r2
    (model_dir / "task2_eos_injection.json").write_text(
        json.dumps(r2, ensure_ascii=False, indent=2), encoding="utf-8")
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Task 3: gap测量
    r3 = task3_gap_measurement(model, tokenizer, device, eos_id, model_name)
    results["task3"] = r3
    (model_dir / "task3_gap.json").write_text(
        json.dumps(r3, ensure_ascii=False, indent=2), encoding="utf-8")

    elapsed = time.time() - t_start
    results["elapsed_seconds"] = elapsed
    log(f"\n  {model_name} total: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    release_model(model)
    save_path = RESULT_DIR / f"{model_name}_result.json"
    save_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"  Saved: {save_path}")
    return results


def main():
    ensure_dir(RESULT_DIR)
    log(f"Phase {PHASE} started: Local model test (qwen3 -> deepseek7b -> glm4)")
    model_name = sys.argv[1] if len(sys.argv) > 1 else None
    if model_name:
        run_model(model_name)
    else:
        # 按大小顺序: qwen3(小) -> deepseek7b(中) -> glm4(大)
        for m in ["qwen3", "deepseek7b", "glm4"]:
            try:
                run_model(m)
            except Exception as e:
                log(f"  {m} FAILED: {e}")
                import traceback; traceback.print_exc()
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    log(f"\nPhase {PHASE} complete!")


if __name__ == "__main__":
    main()
