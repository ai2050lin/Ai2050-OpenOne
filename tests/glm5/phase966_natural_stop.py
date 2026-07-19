#!/usr/bin/env python3
"""
Phase 966: 自然停止控制组件搜索与动态完成检测
================================================
Phase 965证明delayed注入可达strict-clean。本阶段寻找自然等价物：
Task 1: 大规模验证(修正评估: 语义等价+特殊token)
Task 2: Δgap搜索——找ablate后gap缩小的head (Δgap=gap_ablate-gap_normal<0)
Task 3: 多head联合ablate + 测新gap + 尝试自然停止
Task 4: 联合ablate + 减小注入(hybrid: 自然+直接)
"""

from __future__ import annotations
import gc, json, sys, time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import LogitsProcessor

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
from model_utils import load_model, get_layers, get_model_info, release_model
from phase951_protocol_atlas import ensure_dir
from phase965_stop_control import DynamicEOSProcessor, get_boundary_ids
from phase964_forward_diff import make_head_hook, make_eos_inject_hook, get_head_dims, EN_PROMPTS_50, MAX_TOKENS

PHASE = 966
RESULT_DIR = Path("results/phase966_natural_stop")

MODEL_CFG = {
    "qwen3":      {"b": 30, "d": 2, "n_val": 30, "n_diff": 3, "n_multi": 10, "search_layers": 5},
    "glm4":       {"b": 20, "d": 2, "n_val": 10, "n_diff": 2, "n_multi": 3,  "search_layers": 3},
    "deepseek7b": {"b": 20, "d": 2, "n_val": 20, "n_diff": 3, "n_multi": 5,  "search_layers": 3},
}

# Semantic equivalence for answer checking
SEMANTIC_EQUIV = {
    "The capital of France is": ["paris"],
    "The largest planet is": ["jupiter"],
    "Water boils at": ["100", "212", "celsius", "fahrenheit"],
    "The speed of light is": ["299", "300", "186", "light"],
    "The sun is a": ["star"],
    "Dogs are": ["animal", "mammal"],
    "The sky is": ["blue"],
    "Grass is": ["green"],
    "Fire needs": ["oxygen", "air"],
    "Ice is": ["frozen", "solid", "water"],
    "The Earth is": ["round", "sphere", "planet"],
    "A triangle has": ["three", "3"],
    "Shakespeare was": ["english", "playwright", "poet", "writer"],
    "Tokyo is the capital of": ["japan"],
    "The Pacific Ocean is": ["large", "biggest", "largest", "deep"],
    "Gold is a": ["metal", "element", "precious"],
    "Plants need": ["water", "sun", "light"],
    "Humans breathe": ["oxygen", "air"],
    "The moon is": ["natural", "satellite"],
    "Birds can": ["fly"],
    "The largest country is": ["russia"],
    "A square has": ["four", "4"],
    "Mathematics is": ["study", "science", "abstract"],
    "Iron is a": ["metal", "element"],
    "Trees produce": ["oxygen", "fruit", "wood"],
    "Stars are": ["sun", "hot", "bright", "gas"],
}

SPECIAL_TOKENS = ["<|endoftext|>", "<|im_end|>", "<｜end▁of▁sentence｜>",
                  "</s>", "<|end|>", "</think>", "<think>", "\\boxed{}"]


def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def evaluate_clean_v3(prompt, generated, has_eos, n_tokens):
    """v3: semantic equivalence + special token stripping."""
    content = generated
    for st in SPECIAL_TOKENS:
        content = content.replace(st, "")
    content = content.strip()
    is_ascii = all(ord(c) < 256 for c in content)
    is_short = 0 < n_tokens < 15
    equiv = SEMANTIC_EQUIV.get(prompt, [])
    has_expected = len(equiv) == 0 or any(e.lower() in content.lower() for e in equiv)
    strict_clean = has_eos and is_short and has_expected and is_ascii
    return {"strict_clean": strict_clean, "has_eos": has_eos, "is_short": is_short,
            "has_expected": has_expected, "is_ascii": is_ascii, "content": content[:80]}


def register_multi_head_ablation(layers, head_specs, d_head):
    """Register hooks to ablate multiple heads. Returns handles list."""
    by_layer = defaultdict(list)
    for L, H in head_specs:
        by_layer[L].append((H * d_head, (H + 1) * d_head))
    handles = []
    for L, slices in by_layer.items():
        def make_hook(slices_tuple):
            def hook(module, args):
                inp = args[0] if isinstance(args, tuple) else args
                patched = inp.clone()
                for sc, ec in slices_tuple:
                    patched[:, :, sc:ec] = 0
                return (patched,)
            return hook
        handles.append(layers[L].self_attn.o_proj.register_forward_pre_hook(make_hook(tuple(slices))))
    return handles


def run_model(model_name):
    log(f"\n{'='*60}\nPhase 966: {model_name}\n{'='*60}")
    model_dir = RESULT_DIR / model_name
    ensure_dir(model_dir)
    cfg = MODEL_CFG[model_name]

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_heads, d_head = get_head_dims(model, info)
    layers = get_layers(model)
    n_layers = info.n_layers
    eos_id = tokenizer.eos_token_id
    boundary_ids = get_boundary_ids(tokenizer)
    log(f"  {info.model_class}, {info.n_layers}L, d={info.d_model}")

    results = {"model": model_name}
    t_start = time.time()

    # ============================================================
    # Task 1: Large-scale validation with fixed evaluation
    # ============================================================
    log(f"  Task 1: Large-scale validation ({cfg['n_val']} prompts, v3 eval)...")
    prompts = EN_PROMPTS_50[:cfg["n_val"]]
    all_r1 = []
    for pi, prompt in enumerate(prompts):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        # Normal
        with torch.no_grad():
            oid = model.generate(input_ids, max_new_tokens=30, do_sample=False, pad_token_id=eos_id)
        gt = oid[0][input_ids.shape[1]:]
        gen_n = tokenizer.decode(gt, skip_special_tokens=False)
        ce_n = evaluate_clean_v3(prompt, gen_n, gt[-1].item()==eos_id, len(gt))

        # Fixed delay injection
        h = model.lm_head.register_forward_hook(make_eos_inject_hook(eos_id, cfg["b"], cfg["d"]))
        with torch.no_grad():
            oid = model.generate(input_ids, max_new_tokens=30, do_sample=False, pad_token_id=eos_id)
        gt = oid[0][input_ids.shape[1]:]
        gen_f = tokenizer.decode(gt, skip_special_tokens=False)
        ce_f = evaluate_clean_v3(prompt, gen_f, gt[-1].item()==eos_id, len(gt))
        h.remove()

        # Dynamic delay
        proc = DynamicEOSProcessor(eos_id, cfg["b"], min_delay=2, boundary_ids=boundary_ids)
        with torch.no_grad():
            oid = model.generate(input_ids, max_new_tokens=30, do_sample=False, pad_token_id=eos_id, logits_processor=[proc])
        gt = oid[0][input_ids.shape[1]:]
        gen_d = tokenizer.decode(gt, skip_special_tokens=False)
        ce_d = evaluate_clean_v3(prompt, gen_d, gt[-1].item()==eos_id, len(gt))

        all_r1.append({"prompt": prompt,
                       "normal_clean": ce_n["strict_clean"],
                       "fixed_clean": ce_f["strict_clean"],
                       "dynamic_clean": ce_d["strict_clean"],
                       "fixed_gen": gen_f[:80], "dynamic_gen": gen_d[:80]})
        if (pi+1) % 10 == 0: log(f"    {pi+1}/{len(prompts)} ({time.time()-t_start:.0f}s)")

    n = len(all_r1)
    s1 = {"n": n,
          "normal_clean_rate": sum(r["normal_clean"] for r in all_r1)/n,
          "fixed_clean_rate": sum(r["fixed_clean"] for r in all_r1)/n,
          "dynamic_clean_rate": sum(r["dynamic_clean"] for r in all_r1)/n}
    results["task1"] = {"summary": s1, "raw_results": all_r1}
    (model_dir / "task1_validation.json").write_text(json.dumps(results["task1"], ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Normal: {s1['normal_clean_rate']:.2f}  Fixed: {s1['fixed_clean_rate']:.2f}  Dynamic: {s1['dynamic_clean_rate']:.2f}")
    for r in all_r1[:3]:
        log(f"    '{r['prompt'][:25]}': fixed='{r['fixed_gen'][:40]}' dyn='{r['dynamic_gen'][:40]}'")
    log(f"    Task 1 done ({time.time()-t_start:.0f}s)")

    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ============================================================
    # Task 2: Δgap search (find heads where ablation narrows gap)
    # ============================================================
    log(f"  Task 2: Δgap search ({cfg['search_layers']} layers, {cfg['n_diff']} prompts)...")
    search_layers = list(range(max(0, n_layers - cfg["search_layers"]), n_layers))
    proto_ids = {"EOS": eos_id}
    for ts in [".", " "]:
        toks = tokenizer.encode(ts, add_special_tokens=False)
        if toks: proto_ids[ts] = toks[0]

    gap_data = defaultdict(lambda: {"delta_gap": [], "delta_top1": [], "delta_eos": []})
    for pi, prompt in enumerate(EN_PROMPTS_50[:cfg["n_diff"]]):
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
                except: pl = bl.copy()
                handle.remove()

                patched_top1 = float(np.sort(pl)[-1])
                patched_eos = float(pl[eos_id]) if eos_id else 0
                patched_gap = patched_top1 - patched_eos

                key = f"L{L}_H{H}"
                gap_data[key]["delta_gap"].append(patched_gap - base_gap)  # <0 means ablation narrows gap
                gap_data[key]["delta_top1"].append(patched_top1 - base_top1)
                gap_data[key]["delta_eos"].append(patched_eos - base_eos)
        log(f"    {pi+1}/{cfg['n_diff']} prompts ({time.time()-t_start:.0f}s)")

    # Aggregate and find top gap-narrowing heads (Δgap < 0 from ablation)
    head_gap = {}
    for key, d in gap_data.items():
        head_gap[key] = {
            "delta_gap": float(np.mean(d["delta_gap"])),
            "delta_top1": float(np.mean(d["delta_top1"])),
            "delta_eos": float(np.mean(d["delta_eos"])),
        }

    # Sort by delta_gap (most negative = ablation narrows gap most = best to ablate)
    gap_sorted = sorted(head_gap.items(), key=lambda x: x[1]["delta_gap"])
    top_ablate = [(k, v["delta_gap"]) for k, v in gap_sorted if v["delta_gap"] < -0.01][:20]

    results["task2"] = {"search_layers": search_layers,
        "base_gap": base_gap,
        "top_gap_narrowing_heads": top_ablate,
        "all_heads": head_gap}
    (model_dir / "task2_gap_search.json").write_text(json.dumps(results["task2"], ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Base gap: {base_gap:.3f}")
    log(f"    Top 10 gap-narrowing heads (ablate these to shrink gap):")
    for k, dg in top_ablate[:10]:
        v = head_gap[k]
        log(f"      {k}: Δgap={dg:.4f}  Δtop1={v['delta_top1']:.4f}  ΔEOS={v['delta_eos']:.4f}")
    log(f"    Task 2 done ({time.time()-t_start:.0f}s)")

    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ============================================================
    # Task 3: Multi-head ablation + measure new gap + try generation
    # ============================================================
    log(f"  Task 3: Multi-head ablation ({cfg['n_multi']} prompts)...")
    # Parse top heads
    top_heads = []
    for k, _ in top_ablate[:10]:  # Top 10 gap-narrowing heads
        parts = k.split("_")
        top_heads.append((int(parts[0][1:]), int(parts[1][1:])))

    # Measure gap with multi-head ablation
    prompts_multi = EN_PROMPTS_50[:cfg["n_multi"]]
    gap_reduction_data = []
    for pi, prompt in enumerate(prompts_multi[:3]):  # Just 3 for gap measurement
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            bl = model(input_ids, use_cache=False).logits[0, -1].detach().float().cpu().numpy()
        base_gap = float(np.sort(bl)[-1]) - (float(bl[eos_id]) if eos_id else 0)

        handles = register_multi_head_ablation(layers, top_heads, d_head)
        with torch.no_grad():
            pl = model(input_ids, use_cache=False).logits[0, -1].detach().float().cpu().numpy()
        for h in handles: h.remove()
        ablate_gap = float(np.sort(pl)[-1]) - (float(pl[eos_id]) if eos_id else 0)
        gap_reduction_data.append({"prompt": prompt, "base_gap": base_gap,
                                    "ablate_gap": ablate_gap, "reduction": base_gap - ablate_gap})
        log(f"    p{pi}: gap {base_gap:.3f} → {ablate_gap:.3f} (reduction={base_gap-ablate_gap:.3f})")

    # Try generation with multi-head ablation (no injection)
    log(f"    Trying generation with multi-head ablation only (no injection)...")
    ablate_results = []
    for pi, prompt in enumerate(prompts_multi):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        handles = register_multi_head_ablation(layers, top_heads, d_head)
        try:
            with torch.no_grad():
                oid = model.generate(input_ids, max_new_tokens=30, do_sample=False, pad_token_id=eos_id)
            gt = oid[0][input_ids.shape[1]:]
            gen = tokenizer.decode(gt, skip_special_tokens=False)
            he = gt[-1].item() == eos_id; ng = len(gt)
        except Exception as e:
            gen = f"ERROR: {e}"; he = False; ng = 0
        for h in handles: h.remove()
        ce = evaluate_clean_v3(prompt, gen, he, ng)
        ablate_results.append({"prompt": prompt, "generated": gen[:100], "has_eos": he,
                                "n_tokens": ng, "strict_clean": ce["strict_clean"]})

    ablate_clean = sum(r["strict_clean"] for r in ablate_results)
    log(f"    Ablation-only clean: {ablate_clean}/{len(ablate_results)}")

    # Try hybrid: multi-head ablation + reduced injection (b/2)
    log(f"    Trying hybrid: ablation + reduced injection (b={cfg['b']//2})...")
    hybrid_results = []
    for pi, prompt in enumerate(prompts_multi):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        handles = register_multi_head_ablation(layers, top_heads, d_head)
        h_inject = model.lm_head.register_forward_hook(make_eos_inject_hook(eos_id, cfg["b"]//2, cfg["d"]))
        try:
            with torch.no_grad():
                oid = model.generate(input_ids, max_new_tokens=30, do_sample=False, pad_token_id=eos_id)
            gt = oid[0][input_ids.shape[1]:]
            gen = tokenizer.decode(gt, skip_special_tokens=False)
            he = gt[-1].item() == eos_id; ng = len(gt)
        except Exception as e:
            gen = f"ERROR: {e}"; he = False; ng = 0
        for h in handles: h.remove()
        h_inject.remove()
        ce = evaluate_clean_v3(prompt, gen, he, ng)
        hybrid_results.append({"prompt": prompt, "generated": gen[:100], "has_eos": he,
                                "n_tokens": ng, "strict_clean": ce["strict_clean"]})

    hybrid_clean = sum(r["strict_clean"] for r in hybrid_results)
    log(f"    Hybrid clean: {hybrid_clean}/{len(hybrid_results)}")

    s3 = {"ablate_only_clean_rate": ablate_clean/len(ablate_results),
          "hybrid_clean_rate": hybrid_clean/len(hybrid_results),
          "gap_reductions": gap_reduction_data,
          "top_heads": top_heads}
    results["task3"] = {"summary": s3,
                        "ablate_results": ablate_results,
                        "hybrid_results": hybrid_results}
    (model_dir / "task3_multi_ablate.json").write_text(json.dumps(results["task3"], ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"    Sample (p0):")
    for r in ablate_results[:1]:
        log(f"      ablate: eos={r['has_eos']} clean={r['strict_clean']} text={r['generated'][:60]}")
    for r in hybrid_results[:1]:
        log(f"      hybrid: eos={r['has_eos']} clean={r['strict_clean']} text={r['generated'][:60]}")
    log(f"    Task 3 done ({time.time()-t_start:.0f}s)")

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
    log(f"Tasks: 1=large_scale_v3, 2=gap_search, 3=multi_ablate+hybrid")
    model_name = sys.argv[1] if len(sys.argv) > 1 else None
    if model_name:
        run_model(model_name)
    else:
        for m in ["qwen3", "glm4", "deepseek7b"]:
            try: run_model(m)
            except Exception as e:
                log(f"  {m} FAILED: {e}"); import traceback; traceback.print_exc()
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    log(f"\nPhase {PHASE} complete!")


if __name__ == "__main__":
    main()
