#!/usr/bin/env python3
"""
Phase 970: 大规模完成检测与自然增益替代审计
============================================
Phase 969: GLM4 adaptive+dynamic=90%(10p), adaptive=75%(20p), DS7B=40%(20p).
Phase 970 要回答:
  - 90%能否在50+/65 prompt上稳定? (>80%?)
  - CompletionDetected是否优于BoundaryDetected?
  - 能否逐步减少外部b? (r=1,2,3,5)
  - 能否找到更多自然组件缩小gap?
  - qwen3/DS7B跨模型adaptive+dynamic?

Task 1: GLM4 65p大规模验证 (adaptive + boundary-dynamic) [核心验证]
Task 2: Completion检测器多变体对比 (boundary / length-aware / confidence-aware)
Task 3: b缩减曲线 (b'=gap_ablate+2/r, r∈{1,2,3,5}) [自然组件贡献上限]
Task 4: 自然组件组合搜索 (扩展层范围,找更多safe heads)
Task 5: DS7B adaptive+dynamic大规模 + 完成检测器验证
Task 6: qwen3 adaptive+dynamic (如显存允许)

评估v5: 扩展语义等价 + 单位换算 + 模板污染检测 + 垃圾前缀检测
"""

from __future__ import annotations
import gc, json, sys, time, re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
from transformers import LogitsProcessor

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
from model_utils import load_model, get_layers, get_model_info, release_model
from phase951_protocol_atlas import ensure_dir
from phase966_natural_stop import log, register_multi_head_ablation, get_boundary_ids
from phase964_forward_diff import make_head_hook, make_eos_inject_hook, get_head_dims, EN_PROMPTS_50

PHASE = 970
RESULT_DIR = Path("results/phase970_completion_audit")

# ============================================================
# SAFE HEADS (from Phase 967/968/969)
# ============================================================
SAFE_HEADS = {
    "glm4": [(38, 5), (39, 23), (38, 0), (35, 1)],
    "deepseek7b": [(27, 12), (25, 20), (26, 0), (26, 9), (26, 17), (26, 20), (25, 16), (26, 7), (26, 12), (25, 21)],
    "qwen3": [(34, 1), (33, 8)],   # weak, from Phase 966
}

# ============================================================
# EXTENDED PROMPT SET (65 prompts, more diverse)
# - 多种答案长度 (短/中/长)
# - 多种语义关系 (定义/数值/枚举/因果)
# - 多种单位 (温度/速度/数量)
# - 多种开放程度
# ============================================================
EXTRA_PROMPTS_15 = [
    # 长答案/枚举类
    "The three states of matter are",
    "The primary colors are",
    "The seasons of the year are",
    # 因果/解释类
    "Rain falls because",
    "Leaves change color in autumn because",
    "Ice melts when",
    # 数值+单位类
    "The freezing point of water is",
    "A normal human body temperature is",
    "The number of continents is",
    "The number of planets in the solar system is",
    # 定义/属性类
    "A mammal is",
    "An island is",
    "A volcano is",
    "A dictionary is",
    "The ocean is",
]
EN_PROMPTS_65 = EN_PROMPTS_50 + EXTRA_PROMPTS_15

# ============================================================
# EVALUATION v5: 扩展语义等价 + 单位换算 + 模板/垃圾检测
# ============================================================
SEMANTIC_EQUIV_V5: Dict[str, List[str]] = {
    "The capital of France is": ["paris"],
    "The largest planet is": ["jupiter"],
    "Water boils at": ["100", "212", "celsius", "fahrenheit", "boiling"],
    "The speed of light is": ["299", "300", "186", "light", "c"],
    "The sun is a": ["star", "sun"],
    "Dogs are": ["animal", "mammal", "loyal", "companion", "pet", "dog"],
    "The sky is": ["blue", "clear", "up", "atmosphere"],
    "Grass is": ["green", "plant"],
    "Fire needs": ["oxygen", "air", "fuel", "heat"],
    "Ice is": ["frozen", "solid", "water", "cold"],
    "The Earth is": ["round", "sphere", "planet", "earth"],
    "A triangle has": ["three", "3", "side", "angle"],
    "Shakespeare was": ["english", "playwright", "poet", "writer", "bard"],
    "Tokyo is the capital of": ["japan"],
    "The Pacific Ocean is": ["large", "biggest", "largest", "deep"],
    "Gold is a": ["metal", "element", "precious"],
    "Plants need": ["water", "sun", "light"],
    "Humans breathe": ["oxygen", "air"],
    "The moon is": ["natural", "satellite", "moon"],
    "Birds can": ["fly"],
    "The largest country is": ["russia"],
    "A square has": ["four", "4", "side"],
    "Mathematics is": ["study", "science", "abstract"],
    "Music is": ["sound", "art", "rhythm", "melody"],
    "The brain is": ["organ", "mind", "think"],
    "Iron is a": ["metal", "element"],
    "Trees produce": ["oxygen", "fruit", "wood"],
    "Rivers flow": ["water", "down", "sea", "ocean"],
    "Volcanoes erupt": ["lava", "magma", "fire"],
    "Stars are": ["sun", "hot", "bright", "gas", "star"],
    "The heart pumps": ["blood"],
    "DNA contains": ["gene", "code", "information"],
    "Gravity pulls": ["down", "attract"],
    "Light travels": ["fast", "speed", "wave"],
    "Sound is": ["wave", "vibration", "noise"],
    "Heat is": ["energy", "thermal"],
    "A compass points": ["north", "direction"],
    "The equator is": ["line", "middle", "earth"],
    "Antarctica is": ["cold", "ice", "continent"],
    "Diamonds are": ["hard", "carbon", "gem"],
    "Oxygen is": ["gas", "element", "breath"],
    "The kidney filters": ["blood", "waste"],
    "Whales are": ["mammal", "sea", "large"],
    "The alphabet has": ["letter", "26"],
    "A century is": ["100", "year"],
    "The constitution is": ["law", "document", "rule"],
    "Bridges connect": ["place", "side", "land"],
    "Computers process": ["data", "information"],
    "Languages evolve": ["change", "time"],
    "The internet is": ["network", "web", "online"],
    # Extra 15
    "The three states of matter are": ["solid", "liquid", "gas"],
    "The primary colors are": ["red", "blue", "yellow"],
    "The seasons of the year are": ["spring", "summer", "autumn", "winter"],
    "Rain falls because": ["water", "cloud", "condense", "gravity"],
    "Leaves change color in autumn because": ["chlorophyll", "green", "pigment"],
    "Ice melts when": ["heat", "warm", "temperature", "above"],
    "The freezing point of water is": ["0", "32", "celsius", "fahrenheit"],
    "A normal human body temperature is": ["37", "98", "celsius", "fahrenheit"],
    "The number of continents is": ["7", "seven"],
    "The number of planets in the solar system is": ["8", "eight"],
    "A mammal is": ["animal", "milk", "warm", "blood"],
    "An island is": ["land", "water", "surround"],
    "A volcano is": ["mountain", "lava", "erupt"],
    "A dictionary is": ["word", "meaning", "definition", "book"],
    "The ocean is": ["water", "salt", "sea", "large"],
}

SPECIAL_TOKENS = ["</s>", "<|im_end|>", "<｜end▁of▁sentence｜>",
                  "</s>", "<|end|>", "```", "```", "\\boxed{}"]
GARBAGE_PATTERNS = ["\\boxed", "<think", "</think", "```", "```", "Step 1", "Reasoning:"]


def evaluate_clean_v5(prompt: str, generated: str, has_eos: bool, n_tokens: int) -> Dict[str, Any]:
    """v5: 扩展语义等价 + 模板/垃圾检测 + 单位换算容忍."""
    content = generated
    for st in SPECIAL_TOKENS:
        content = content.replace(st, "")
    content = content.strip()
    is_ascii = all(ord(c) < 256 for c in content)
    # 答案长度容忍: 枚举类答案可更长
    is_enum = any(k in prompt for k in ["states of matter", "primary colors", "seasons", "three", "planets"])
    max_tok = 25 if is_enum else 15
    is_short = 0 < n_tokens < max_tok
    equiv = SEMANTIC_EQUIV_V5.get(prompt, [])
    has_expected = len(equiv) == 0 or any(e.lower() in content.lower() for e in equiv)
    has_garbage = any(g.lower() in content.lower() for g in GARBAGE_PATTERNS)
    return {
        "strict_clean": has_eos and is_short and has_expected and is_ascii and not has_garbage,
        "has_eos": has_eos, "is_short": is_short, "has_expected": has_expected,
        "is_ascii": is_ascii, "has_garbage": has_garbage,
        "content": content[:80],
    }


# ============================================================
# COMPLETION DETECTORS (3 variants)
# ============================================================
def get_completion_ids(tokenizer) -> Dict[str, set]:
    """Get token IDs for boundary + content completion signals."""
    boundary_ids = set()
    for tok_str in [".", "\n", ",", "。", "，", ";", ":", " and", " or"]:
        toks = tokenizer.encode(tok_str, add_special_tokens=False)
        if toks:
            boundary_ids.add(toks[0])
    # 单位/枚举完成信号
    enum_ids = set()
    for tok_str in [" and", " or", ":", ";"]:
        toks = tokenizer.encode(tok_str, add_special_tokens=False)
        if toks:
            enum_ids.add(toks[0])
    return {"boundary": boundary_ids, "enum": enum_ids}


class BoundaryDynamicProcessor(LogitsProcessor):
    """Phase 969 baseline: inject after boundary token (period/comma/newline)."""
    def __init__(self, eos_id, bias, min_delay=2, boundary_ids=None):
        self.eos_id = eos_id; self.bias = bias
        self.min_delay = min_delay
        self.boundary_ids = set(boundary_ids or [])
        self.step = 0; self.inject = False
    def __call__(self, input_ids, scores):
        if self.step >= self.min_delay:
            if len(input_ids[0]) > 0 and input_ids[0, -1].item() in self.boundary_ids:
                self.inject = True
            if self.inject:
                scores[:, self.eos_id] += self.bias
        self.step += 1
        return scores


class LengthAwareProcessor(LogitsProcessor):
    """v2: boundary + min content tokens (>=2 non-boundary content tokens generated).
    防止过早停在无内容输出上."""
    def __init__(self, eos_id, bias, min_delay=2, boundary_ids=None, min_content=2):
        self.eos_id = eos_id; self.bias = bias
        self.min_delay = min_delay
        self.boundary_ids = set(boundary_ids or [])
        self.min_content = min_content
        self.step = 0; self.inject = False
        self.content_count = 0
    def __call__(self, input_ids, scores):
        if self.step >= self.min_delay and len(input_ids[0]) > 0:
            last = input_ids[0, -1].item()
            if last in self.boundary_ids:
                self.inject = True
            # 统计content token (非boundary, 非eos, 非纯空格)
            if last not in self.boundary_ids and last != self.eos_id:
                self.content_count += 1
            # 只在content足够 + boundary出现后才注入
            if self.inject and self.content_count >= self.min_content:
                scores[:, self.eos_id] += self.bias
        self.step += 1
        return scores


class ConfidenceAwareProcessor(LogitsProcessor):
    """v3: boundary + gap narrowing signal (EOS logit相对top1上升).
    检测模型自身是否在"倾向停止"——更自然的完成信号."""
    def __init__(self, eos_id, bias, min_delay=2, boundary_ids=None,
                 gap_history_len=3, gap_drop_threshold=0.5):
        self.eos_id = eos_id; self.bias = bias
        self.min_delay = min_delay
        self.boundary_ids = set(boundary_ids or [])
        self.gap_history: List[float] = []
        self.gap_history_len = gap_history_len
        self.gap_drop_threshold = gap_drop_threshold
        self.step = 0; self.inject = False
        self.boundary_seen = False
    def __call__(self, input_ids, scores):
        if self.step >= self.min_delay and len(input_ids[0]) > 0:
            last = input_ids[0, -1].item()
            if last in self.boundary_ids:
                self.boundary_seen = True
            # 计算当前gap (top1 - EOS)
            sc = scores[0]
            eos_logit = float(sc[self.eos_id])
            top1_logit = float(torch.max(sc))
            gap = top1_logit - eos_logit
            self.gap_history.append(gap)
            if len(self.gap_history) > self.gap_history_len:
                self.gap_history.pop(0)
            # gap在最近N步显著下降 → 模型倾向停止
            gap_narrowing = False
            if len(self.gap_history) >= self.gap_history_len:
                gap_drop = self.gap_history[0] - self.gap_history[-1]
                gap_narrowing = gap_drop > self.gap_drop_threshold
            # 注入条件: boundary出现 + (gap在缩小 OR 已经经过额外延迟)
            if self.boundary_seen and (gap_narrowing or self.step >= self.min_delay + 4):
                self.inject = True
            if self.inject:
                scores[:, self.eos_id] += self.bias
        self.step += 1
        return scores


# ============================================================
# GAP MEASUREMENT
# ============================================================
def measure_gap(model, input_ids, eos_id, layers, safe_heads, d_head):
    """Measure base gap and ablated gap for a prompt."""
    with torch.no_grad():
        bl = model(input_ids, use_cache=False).logits[0, -1].detach().float().cpu().numpy()
    base_top1 = float(np.sort(bl)[-1])
    base_eos = float(bl[eos_id]) if eos_id else 0
    base_gap = base_top1 - base_eos
    handles = register_multi_head_ablation(layers, safe_heads, d_head)
    with torch.no_grad():
        al = model(input_ids, use_cache=False).logits[0, -1].detach().float().cpu().numpy()
    for h in handles: h.remove()
    ablate_top1 = float(np.sort(al)[-1])
    ablate_eos = float(al[eos_id]) if eos_id else 0
    ablate_gap = ablate_top1 - ablate_eos
    return base_gap, ablate_gap


def generate_with_processor(model, tokenizer, input_ids, processor, max_new=30, pad_id=None):
    """Generate with a LogitsProcessor, return (gen_text, has_eos, n_tokens)."""
    try:
        with torch.no_grad():
            oid = model.generate(input_ids, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=pad_id, logits_processor=[processor] if processor else [])
        gt = oid[0][input_ids.shape[1]:]
        gen = tokenizer.decode(gt, skip_special_tokens=False)
        he = gt[-1].item() == pad_id if pad_id is not None else False
        ng = len(gt)
        return gen, he, ng
    except Exception as e:
        return f"ERROR: {e}", False, 0


def generate_with_hook(model, tokenizer, input_ids, hook, max_new=30, pad_id=None):
    """Generate with an lm_head hook (fixed delay injection)."""
    try:
        with torch.no_grad():
            oid = model.generate(input_ids, max_new_tokens=max_new, do_sample=False, pad_token_id=pad_id)
        gt = oid[0][input_ids.shape[1]:]
        gen = tokenizer.decode(gt, skip_special_tokens=False)
        he = gt[-1].item() == pad_id if pad_id is not None else False
        ng = len(gt)
        return gen, he, ng
    except Exception as e:
        return f"ERROR: {e}", False, 0


# ============================================================
# TASK RUNNERS
# ============================================================
def task1_large_scale_validation(model, tokenizer, device, info, layers, eos_id,
                                  safe_heads, d_head, boundary_ids, model_name, n_prompts=65):
    """Task 1: GLM4 65p大规模验证 (adaptive + boundary-dynamic) [核心验证]."""
    log(f"  Task 1: Large-scale validation ({n_prompts} prompts, adaptive+boundary-dynamic)...")
    prompts = EN_PROMPTS_65[:n_prompts]
    all_r = []
    t0 = time.time()
    for pi, prompt in enumerate(prompts):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        base_gap, ablate_gap = measure_gap(model, input_ids, eos_id, layers, safe_heads, d_head)
        b_adaptive = int(ablate_gap + 2)
        handles = register_multi_head_ablation(layers, safe_heads, d_head)
        proc = BoundaryDynamicProcessor(eos_id, b_adaptive, min_delay=2, boundary_ids=boundary_ids)
        gen, he, ng = generate_with_processor(model, tokenizer, input_ids, proc, max_new=30, pad_id=eos_id)
        for h in handles: h.remove()
        ce = evaluate_clean_v5(prompt, gen, he, ng)
        all_r.append({"prompt": prompt, "base_gap": base_gap, "ablate_gap": ablate_gap,
                       "b": b_adaptive, "generated": gen[:80], "has_eos": he,
                       "n_tokens": ng, "strict_clean": ce["strict_clean"],
                       "has_expected": ce["has_expected"], "has_garbage": ce["has_garbage"]})
        if (pi+1) % 10 == 0:
            clean_so_far = sum(r["strict_clean"] for r in all_r)
            log(f"    {pi+1}/{n_prompts} clean_so_far={clean_so_far}/{pi+1} ({time.time()-t0:.0f}s)")

    n = len(all_r)
    clean = sum(r["strict_clean"] for r in all_r)
    eos = sum(r["has_eos"] for r in all_r)
    expected = sum(r["has_expected"] for r in all_r)
    garbage = sum(r["has_garbage"] for r in all_r)
    s = {"clean_rate": clean/n, "eos_rate": eos/n, "expected_rate": expected/n,
         "garbage_rate": garbage/n, "mean_b": float(np.mean([r["b"] for r in all_r])), "n": n}
    log(f"    RESULT: clean={clean}/{n}={s['clean_rate']:.3f}  eos={s['eos_rate']:.3f}  "
        f"expected={s['expected_rate']:.3f}  garbage={s['garbage_rate']:.3f}  mean_b={s['mean_b']:.1f}")
    log(f"    Successes:")
    for r in all_r:
        if r["strict_clean"]:
            log(f"      OK  '{r['prompt'][:30]}': b={r['b']} '{r['generated'][:45]}'")
    log(f"    Failures (EOS but not clean):")
    for r in all_r:
        if r["has_eos"] and not r["strict_clean"]:
            log(f"      FAIL '{r['prompt'][:30]}': b={r['b']} '{r['generated'][:45]}' "
                f"exp={r['has_expected']} gb={r['has_garbage']}")
    log(f"    Task 1 done ({time.time()-t0:.0f}s)")
    return {"summary": s, "raw_results": all_r}


def task2_completion_detector_comparison(model, tokenizer, device, info, layers, eos_id,
                                         safe_heads, d_head, boundary_ids, model_name, n_prompts=20):
    """Task 2: Completion检测器多变体对比 (boundary / length-aware / confidence-aware)."""
    log(f"  Task 2: Completion detector comparison ({n_prompts} prompts × 3 variants)...")
    prompts = EN_PROMPTS_65[:n_prompts]
    variants = ["boundary", "length_aware", "confidence_aware"]
    all_r = []
    t0 = time.time()
    for pi, prompt in enumerate(prompts):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        base_gap, ablate_gap = measure_gap(model, input_ids, eos_id, layers, safe_heads, d_head)
        b_adaptive = int(ablate_gap + 2)
        row = {"prompt": prompt, "b": b_adaptive, "base_gap": base_gap, "ablate_gap": ablate_gap}
        for vname in variants:
            handles = register_multi_head_ablation(layers, safe_heads, d_head)
            if vname == "boundary":
                proc = BoundaryDynamicProcessor(eos_id, b_adaptive, min_delay=2, boundary_ids=boundary_ids)
            elif vname == "length_aware":
                proc = LengthAwareProcessor(eos_id, b_adaptive, min_delay=2, boundary_ids=boundary_ids, min_content=2)
            elif vname == "confidence_aware":
                proc = ConfidenceAwareProcessor(eos_id, b_adaptive, min_delay=2, boundary_ids=boundary_ids)
            gen, he, ng = generate_with_processor(model, tokenizer, input_ids, proc, max_new=30, pad_id=eos_id)
            for h in handles: h.remove()
            ce = evaluate_clean_v5(prompt, gen, he, ng)
            row[f"{vname}_gen"] = gen[:60]
            row[f"{vname}_eos"] = he
            row[f"{vname}_clean"] = ce["strict_clean"]
            row[f"{vname}_ntok"] = ng
        all_r.append(row)
        if (pi+1) % 5 == 0:
            log(f"    {pi+1}/{n_prompts} ({time.time()-t0:.0f}s)")

    n = len(all_r)
    summary = {}
    for v in variants:
        clean = sum(r[f"{v}_clean"] for r in all_r)
        eos = sum(r[f"{v}_eos"] for r in all_r)
        toks = [r[f"{v}_ntok"] for r in all_r]
        summary[v] = {"clean_rate": clean/n, "eos_rate": eos/n,
                      "mean_tokens": float(np.mean(toks)) if toks else 0}
        log(f"    {v:20s}: clean={clean}/{n}={clean/n:.3f}  eos={eos/n:.3f}  mean_tok={np.mean(toks):.1f}")
    # 比较哪种变体最好
    best = max(variants, key=lambda v: summary[v]["clean_rate"])
    log(f"    BEST variant: {best} (clean={summary[best]['clean_rate']:.3f})")
    log(f"    Task 2 done ({time.time()-t0:.0f}s)")
    return {"summary": summary, "best_variant": best, "raw_results": all_r}


def task3_b_reduction_curve(model, tokenizer, device, info, layers, eos_id,
                             safe_heads, d_head, boundary_ids, model_name, n_prompts=20):
    """Task 3: b缩减曲线 (b'=gap_ablate+2/r, r∈{1,2,3,5}) [自然组件贡献上限]."""
    log(f"  Task 3: b reduction curve (r∈{{1,2,3,5}}, {n_prompts} prompts)...")
    prompts = EN_PROMPTS_65[:n_prompts]
    r_values = [1, 2, 3, 5]
    all_r = []
    t0 = time.time()
    for pi, prompt in enumerate(prompts):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        base_gap, ablate_gap = measure_gap(model, input_ids, eos_id, layers, safe_heads, d_head)
        row = {"prompt": prompt, "ablate_gap": ablate_gap}
        for r in r_values:
            b_reduced = max(1, int((ablate_gap + 2) / r))
            handles = register_multi_head_ablation(layers, safe_heads, d_head)
            proc = BoundaryDynamicProcessor(eos_id, b_reduced, min_delay=2, boundary_ids=boundary_ids)
            gen, he, ng = generate_with_processor(model, tokenizer, input_ids, proc, max_new=30, pad_id=eos_id)
            for h in handles: h.remove()
            ce = evaluate_clean_v5(prompt, gen, he, ng)
            row[f"r{r}_b"] = b_reduced
            row[f"r{r}_gen"] = gen[:50]
            row[f"r{r}_eos"] = he
            row[f"r{r}_clean"] = ce["strict_clean"]
        all_r.append(row)
        if (pi+1) % 5 == 0:
            log(f"    {pi+1}/{n_prompts} ({time.time()-t0:.0f}s)")

    n = len(all_r)
    summary = {}
    for r in r_values:
        clean = sum(row[f"r{r}_clean"] for row in all_r)
        eos = sum(row[f"r{r}_eos"] for row in all_r)
        bs = [row[f"r{r}_b"] for row in all_r]
        summary[f"r{r}"] = {"clean_rate": clean/n, "eos_rate": eos/n,
                            "mean_b": float(np.mean(bs)), "n": n}
        log(f"    r={r} (mean_b={np.mean(bs):.1f}): clean={clean}/{n}={clean/n:.3f}  eos={eos/n:.3f}")
    # 自然组件贡献上限: r=∞ (b=0)的代理 = r=5的clean率
    log(f"    自然组件贡献上限估计: r=5时clean={summary['r5']['clean_rate']:.3f}")
    log(f"    Task 3 done ({time.time()-t0:.0f}s)")
    return {"summary": summary, "raw_results": all_r}


def task4_natural_component_search(model, tokenizer, device, info, layers, eos_id,
                                    safe_heads, d_head, model_name, boundary_ids=None,
                                    n_prompts=5, search_n_layers=6):
    """Task 4: 自然组件组合搜索 (扩展层范围,找更多safe heads).
    搜索最后search_n_layers层的所有head,找Δgap<0且不破坏内容的head."""
    log(f"  Task 4: Natural component search ({search_n_layers} layers × all heads, {n_prompts} prompts)...")
    n_layers = info.n_layers
    n_heads, _ = get_head_dims(model, info)
    search_layers = list(range(max(0, n_layers - search_n_layers), n_layers))
    log(f"    Searching layers {search_layers[0]}-{search_layers[-1]} ({len(search_layers)}L × {n_heads}H)")

    prompts = EN_PROMPTS_65[:n_prompts]
    gap_data = defaultdict(lambda: {"delta_gap": [], "delta_top1": [], "delta_eos": []})
    t0 = time.time()
    for pi, prompt in enumerate(prompts):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            bl = model(input_ids, use_cache=False).logits[0, -1].detach().float().cpu().numpy()
        base_top1 = float(np.sort(bl)[-1])
        base_eos = float(bl[eos_id]) if eos_id else 0
        base_gap = base_top1 - base_eos
        for L in search_layers:
            for H in range(n_heads):
                sc = H * d_head; ec = sc + d_head
                handle = layers[L].self_attn.o_proj.register_forward_pre_hook(make_head_hook(sc, ec, 0.0))
                try:
                    with torch.no_grad():
                        pl = model(input_ids, use_cache=False).logits[0, -1].detach().float().cpu().numpy()
                except:
                    pl = bl.copy()
                handle.remove()
                patched_top1 = float(np.sort(pl)[-1])
                patched_eos = float(pl[eos_id]) if eos_id else 0
                patched_gap = patched_top1 - patched_eos
                key = f"L{L}_H{H}"
                gap_data[key]["delta_gap"].append(patched_gap - base_gap)
                gap_data[key]["delta_top1"].append(patched_top1 - base_top1)
                gap_data[key]["delta_eos"].append(patched_eos - base_eos)
        log(f"    {pi+1}/{n_prompts} prompts ({time.time()-t0:.0f}s)")

    # 聚合
    head_gap = {}
    for key, d in gap_data.items():
        head_gap[key] = {
            "delta_gap": float(np.mean(d["delta_gap"])),
            "delta_top1": float(np.mean(d["delta_top1"])),
            "delta_eos": float(np.mean(d["delta_eos"])),
        }
    # 排序: Δgap最负 = ablate后gap缩小最多 = 最适合ablate
    gap_sorted = sorted(head_gap.items(), key=lambda x: x[1]["delta_gap"])
    # safe head分类: Δgap<0 (缩小gap) 且 |Δtop1|<0.3 (不破坏内容)
    safe_candidates = []
    for k, v in gap_sorted:
        if v["delta_gap"] < -0.05 and abs(v["delta_top1"]) < 0.3:
            parts = k.split("_")
            L = int(parts[0][1:]); H = int(parts[1][1:])
            safe_candidates.append({"layer": L, "head": H, "key": k,
                                    "delta_gap": v["delta_gap"],
                                    "delta_top1": v["delta_top1"],
                                    "delta_eos": v["delta_eos"]})
    log(f"    Found {len(safe_candidates)} safe head candidates (Δgap<-0.05, |Δtop1|<0.3)")
    for sc in safe_candidates[:15]:
        log(f"      {sc['key']}: Δgap={sc['delta_gap']:.4f}  Δtop1={sc['delta_top1']:.4f}  ΔEOS={sc['delta_eos']:.4f}")

    # 测试: 现有safe heads vs 新发现的safe heads组合
    existing_safe = safe_heads
    new_safe_top = [(sc["layer"], sc["head"]) for sc in safe_candidates[:8]]
    combined_safe = list(dict.fromkeys(existing_safe + new_safe_top))[:12]

    test_prompts = EN_PROMPTS_65[:8]
    log(f"    Testing gap reduction: existing={len(existing_safe)} vs new={len(new_safe_top)} vs combined={len(combined_safe)}")
    gap_compare = {"existing": [], "new": [], "combined": []}
    for prompt in test_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            bl = model(input_ids, use_cache=False).logits[0, -1].detach().float().cpu().numpy()
        bg = float(np.sort(bl)[-1]) - (float(bl[eos_id]) if eos_id else 0)
        for name, heads_set in [("existing", existing_safe), ("new", new_safe_top), ("combined", combined_safe)]:
            handles = register_multi_head_ablation(layers, heads_set, d_head)
            with torch.no_grad():
                pl = model(input_ids, use_cache=False).logits[0, -1].detach().float().cpu().numpy()
            for h in handles: h.remove()
            ag = float(np.sort(pl)[-1]) - (float(pl[eos_id]) if eos_id else 0)
            gap_compare[name].append({"prompt": prompt, "base_gap": bg, "ablate_gap": ag, "reduction": bg - ag})

    head_counts = {"existing": len(existing_safe), "new": len(new_safe_top), "combined": len(combined_safe)}
    for name, data in gap_compare.items():
        mean_red = float(np.mean([d["reduction"] for d in data])) if data else 0
        log(f"      {name:10s} ({head_counts[name]} heads): mean_gap_reduction={mean_red:.3f}")

    # 测试combined + reduced b 的clean率
    log(f"    Testing combined_safe + adaptive b (8 prompts)...")
    combined_clean_results = []
    for prompt in test_prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        _, ablate_gap = measure_gap(model, input_ids, eos_id, layers, combined_safe, d_head)
        b_adaptive = int(ablate_gap + 2)
        handles = register_multi_head_ablation(layers, combined_safe, d_head)
        proc = BoundaryDynamicProcessor(eos_id, b_adaptive, min_delay=2, boundary_ids=boundary_ids)
        gen, he, ng = generate_with_processor(model, tokenizer, input_ids, proc, max_new=30, pad_id=eos_id)
        for h in handles: h.remove()
        ce = evaluate_clean_v5(prompt, gen, he, ng)
        combined_clean_results.append({"prompt": prompt, "b": b_adaptive, "ablate_gap": ablate_gap,
                                       "generated": gen[:60], "has_eos": he,
                                       "strict_clean": ce["strict_clean"]})
    cc_clean = sum(r["strict_clean"] for r in combined_clean_results)
    log(f"    Combined safe + adaptive b: clean={cc_clean}/{len(combined_clean_results)}")
    log(f"    Task 4 done ({time.time()-t0:.0f}s)")
    return {"safe_candidates": safe_candidates[:20], "gap_compare": gap_compare,
            "combined_clean_results": combined_clean_results,
            "existing_safe": existing_safe, "new_safe_top": new_safe_top,
            "combined_safe": combined_safe}


def task5_ds7b_validation(model, tokenizer, device, info, layers, eos_id,
                           safe_heads, d_head, boundary_ids, model_name, n_prompts=30):
    """Task 5: DS7B adaptive+dynamic大规模 + 完成检测器验证."""
    log(f"  Task 5: DS7B adaptive+dynamic + completion detector ({n_prompts} prompts)...")
    prompts = EN_PROMPTS_65[:n_prompts]
    all_r = []
    t0 = time.time()
    for pi, prompt in enumerate(prompts):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        base_gap, ablate_gap = measure_gap(model, input_ids, eos_id, layers, safe_heads, d_head)
        b_adaptive = int(ablate_gap + 2)
        row = {"prompt": prompt, "b": b_adaptive, "ablate_gap": ablate_gap}
        # boundary-dynamic
        handles = register_multi_head_ablation(layers, safe_heads, d_head)
        proc = BoundaryDynamicProcessor(eos_id, b_adaptive, min_delay=2, boundary_ids=boundary_ids)
        gen, he, ng = generate_with_processor(model, tokenizer, input_ids, proc, max_new=30, pad_id=eos_id)
        for h in handles: h.remove()
        ce = evaluate_clean_v5(prompt, gen, he, ng)
        row["boundary_gen"] = gen[:60]; row["boundary_eos"] = he
        row["boundary_clean"] = ce["strict_clean"]; row["boundary_ntok"] = ng
        all_r.append(row)
        if (pi+1) % 10 == 0:
            cs = sum(r["boundary_clean"] for r in all_r)
            log(f"    {pi+1}/{n_prompts} clean_so_far={cs}/{pi+1} ({time.time()-t0:.0f}s)")

    n = len(all_r)
    clean = sum(r["boundary_clean"] for r in all_r)
    eos = sum(r["boundary_eos"] for r in all_r)
    s = {"clean_rate": clean/n, "eos_rate": eos/n, "mean_b": float(np.mean([r["b"] for r in all_r])), "n": n}
    log(f"    DS7B adaptive+dynamic: clean={clean}/{n}={s['clean_rate']:.3f}  eos={s['eos_rate']:.3f}")
    log(f"    Task 5 done ({time.time()-t0:.0f}s)")
    return {"summary": s, "raw_results": all_r}


# ============================================================
# MAIN RUNNER
# ============================================================
def run_model(model_name: str):
    log(f"\n{'='*60}\nPhase 970: {model_name}\n{'='*60}")
    model_dir = RESULT_DIR / model_name
    ensure_dir(model_dir)
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_heads, d_head = get_head_dims(model, info)
    layers = get_layers(model)
    eos_id = tokenizer.eos_token_id
    safe_heads = SAFE_HEADS.get(model_name, [])
    boundary_ids = get_boundary_ids(tokenizer)
    log(f"  {info.model_class}, {info.n_layers}L, {n_heads}H, d_head={d_head}, safe_heads={len(safe_heads)}")

    results = {"model": model_name, "phase": PHASE}
    t_start = time.time()

    if model_name == "glm4":
        # GLM4: 全部4个任务 (核心验证)
        r1 = task1_large_scale_validation(model, tokenizer, device, info, layers, eos_id,
                                          safe_heads, d_head, boundary_ids, model_name, n_prompts=65)
        results["task1"] = r1
        (model_dir / "task1_large_scale_65p.json").write_text(
            json.dumps(r1, ensure_ascii=False, indent=2), encoding="utf-8")
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        r2 = task2_completion_detector_comparison(model, tokenizer, device, info, layers, eos_id,
                                                   safe_heads, d_head, boundary_ids, model_name, n_prompts=20)
        results["task2"] = r2
        (model_dir / "task2_completion_comparison.json").write_text(
            json.dumps(r2, ensure_ascii=False, indent=2), encoding="utf-8")
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        r3 = task3_b_reduction_curve(model, tokenizer, device, info, layers, eos_id,
                                      safe_heads, d_head, boundary_ids, model_name, n_prompts=20)
        results["task3"] = r3
        (model_dir / "task3_b_reduction.json").write_text(
            json.dumps(r3, ensure_ascii=False, indent=2), encoding="utf-8")
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        r4 = task4_natural_component_search(model, tokenizer, device, info, layers, eos_id,
                                            safe_heads, d_head, model_name, n_prompts=5, search_n_layers=6)
        results["task4"] = r4
        (model_dir / "task4_natural_search.json").write_text(
            json.dumps(r4, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    elif model_name == "deepseek7b":
        # DS7B: Task 5 (adaptive+dynamic大规模)
        r5 = task5_ds7b_validation(model, tokenizer, device, info, layers, eos_id,
                                    safe_heads, d_head, boundary_ids, model_name, n_prompts=30)
        results["task5"] = r5
        (model_dir / "task5_ds7b_adaptive_dynamic.json").write_text(
            json.dumps(r5, ensure_ascii=False, indent=2), encoding="utf-8")
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        # DS7B也跑完成检测器对比
        r2 = task2_completion_detector_comparison(model, tokenizer, device, info, layers, eos_id,
                                                   safe_heads, d_head, boundary_ids, model_name, n_prompts=20)
        results["task2"] = r2
        (model_dir / "task2_completion_comparison.json").write_text(
            json.dumps(r2, ensure_ascii=False, indent=2), encoding="utf-8")

    elif model_name == "qwen3":
        # qwen3: adaptive+dynamic (safe heads弱,但测试效果)
        r5 = task5_ds7b_validation(model, tokenizer, device, info, layers, eos_id,
                                    safe_heads, d_head, boundary_ids, model_name, n_prompts=20)
        results["task5_qwen3"] = r5
        (model_dir / "task5_qwen3_adaptive_dynamic.json").write_text(
            json.dumps(r5, ensure_ascii=False, indent=2), encoding="utf-8")

    elapsed = time.time() - t_start
    results["elapsed_seconds"] = elapsed
    log(f"\n  Total: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    release_model(model)
    save_path = RESULT_DIR / f"{model_name}_result.json"
    save_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"  {model_name} complete. Saved: {save_path}")
    return results


def main():
    ensure_dir(RESULT_DIR)
    log(f"Phase {PHASE} started")
    log(f"Tasks: 1=GLM4_65p_validation, 2=completion_detector_compare, "
        f"3=b_reduction_curve, 4=natural_search, 5=DS7B/qwen3_adaptive_dynamic")
    model_name = sys.argv[1] if len(sys.argv) > 1 else None
    if model_name:
        run_model(model_name)
    else:
        for m in ["glm4", "deepseek7b", "qwen3"]:
            try:
                run_model(m)
            except Exception as e:
                log(f"  {m} FAILED: {e}"); import traceback; traceback.print_exc()
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    log(f"\nPhase {PHASE} complete!")


if __name__ == "__main__":
    main()
