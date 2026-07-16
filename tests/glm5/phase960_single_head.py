#!/usr/bin/env python3
"""
Phase 960: 单注意力头拆分、放大与严格生成闭合审计
==================================================
Task1: 单head拆分 — 逐个零化每个token-specific head, 测rollout
Task2: head boost — 放大head(1.5x/2.0x), 测是否促进停止
Task3: strict-clean评价 — 定义并测量干净输出率
"""

from __future__ import annotations
import gc, json, sys, time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
from model_utils import load_model, get_layers, get_model_info, release_model
from phase951_protocol_atlas import ensure_dir

PHASE = 960
RESULT_DIR = Path("results/phase960_single_head")

# Per-model head configs
MODEL_HEADS = {
    "qwen3": [
        {"layer": 35, "head": 0, "role": "period", "d_head": 80},
        {"layer": 33, "head": 8, "role": "period", "d_head": 80},
    ],
    "glm4": [
        {"layer": 39, "head": 21, "role": "EOS", "d_head": 128},
        {"layer": 38, "head": 0, "role": "EOS", "d_head": 128},
        {"layer": 38, "head": 7, "role": "EOS", "d_head": 128},
    ],
    "deepseek7b": [
        {"layer": 26, "head": 19, "role": "space", "d_head": 112},
        {"layer": 26, "head": 25, "role": "space", "d_head": 112},
    ],
}

PROMPTS = [
    "The capital of France is",
    "What is water?",
    "The sky is blue.",
    "Explain gravity:",
    "How many continents?",
    "Name two planets:",
    "Classify: dog is a",
    "The largest ocean is",
]

MAX_TOKENS = 40

# Expected clean answers for strict-clean evaluation
EXPECTED_ANSWERS = {
    "The capital of France is": "Paris",
    "What is water?": "",
    "The sky is blue.": "",
    "Explain gravity:": "",
    "How many continents?": "seven",
    "Name two planets:": "",
    "Classify: dog is a": "animal",
    "The largest ocean is": "Pacific",
}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def make_head_hook(sc, ec, scale=0.0):
    """Hook to zero (scale=0) or boost (scale=1.5/2.0) a specific head."""
    def hook(module, args):
        inp = args[0] if isinstance(args, tuple) else args
        patched = inp.clone()
        if patched.ndim >= 3:
            if scale == 0.0:
                patched[:, :, sc:ec] = 0
            else:
                patched[:, :, sc:ec] = patched[:, :, sc:ec] * scale
        return (patched,)
    return hook


def generate_with_hook(model, tokenizer, device, prompt, layers, head_info, scale=0.0):
    """Generate text with a head hook applied."""
    handles = []
    if head_info is not None:
        li = head_info["layer"]
        hi = head_info["head"]
        dh = head_info["d_head"]
        sc = hi * dh
        ec = sc + dh
        h = layers[li].self_attn.o_proj.register_forward_pre_hook(
            make_head_hook(sc, ec, scale)
        )
        handles.append(h)

    try:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=MAX_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_tokens = output_ids[0][input_ids.shape[1]:]
        generated = tokenizer.decode(gen_tokens, skip_special_tokens=False)
        has_eos = gen_tokens[-1].item() == tokenizer.eos_token_id if tokenizer.eos_token_id else False
        n_gen = len(gen_tokens)
    except Exception as e:
        generated = f"ERROR: {e}"
        has_eos = False
        n_gen = 0

    for h in handles:
        h.remove()

    return generated, has_eos, n_gen


def evaluate_strict_clean(prompt, generated, has_eos, n_tokens):
    """Evaluate if output is strict-clean."""
    expected = EXPECTED_ANSWERS.get(prompt, "")
    # strict-clean criteria:
    # 1. EOS terminated
    # 2. Short output (n_tokens < 15)
    # 3. Contains expected answer (if defined)
    # 4. No language switch (ASCII only for English prompts)
    is_ascii = all(ord(c) < 256 for c in generated)
    is_short = n_tokens < 15 and n_tokens > 0
    has_expected = (expected == "") or (expected.lower() in generated.lower())

    strict_clean = has_eos and is_short and has_expected and is_ascii
    return {
        "strict_clean": strict_clean,
        "has_eos": has_eos,
        "is_short": is_short,
        "has_expected": has_expected,
        "is_ascii": is_ascii,
        "n_tokens": n_tokens,
    }


# ============================================================
# TASK 1+2+3: Single head ablation + boost + strict-clean
# ============================================================

def run_all_tasks(model, tokenizer, device, info, model_name: str,
                   save_dir: Path) -> dict:
    """Run all tasks: single head ablate, boost, and evaluate."""
    log("  Task 1-3: Single head ablation + boost + strict-clean...")

    layers = get_layers(model)
    heads = MODEL_HEADS.get(model_name, [])

    if not heads:
        return {"model": model_name, "skipped": True}

    # Conditions: normal, ablate each head, boost each head (1.5x, 2.0x)
    conditions = [("normal", None, 0.0)]

    for h in heads:
        label = f"ablate_L{h['layer']}_H{h['head']}"
        conditions.append((label, h, 0.0))

    for h in heads:
        label = f"boost1.5_L{h['layer']}_H{h['head']}"
        conditions.append((label, h, 1.5))

    for h in heads:
        label = f"boost2.0_L{h['layer']}_H{h['head']}"
        conditions.append((label, h, 2.0))

    # Also test ablate_all and boost_all
    conditions.append(("ablate_all", "all", 0.0))
    conditions.append(("boost1.5_all", "all", 1.5))

    log(f"    {len(conditions)} conditions × {len(PROMPTS)} prompts")

    all_results = []

    for pi, prompt in enumerate(PROMPTS):
        for cond_name, head_info, scale in conditions:
            if head_info == "all":
                # Apply to all heads simultaneously
                handles = []
                for h in heads:
                    sc = h["head"] * h["d_head"]
                    ec = sc + h["d_head"]
                    handles.append(
                        layers[h["layer"]].self_attn.o_proj.register_forward_pre_hook(
                            make_head_hook(sc, ec, scale)
                        )
                )
                try:
                    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                    with torch.no_grad():
                        output_ids = model.generate(input_ids, max_new_tokens=MAX_TOKENS,
                                                   do_sample=False, pad_token_id=tokenizer.eos_token_id)
                    gen_tokens = output_ids[0][input_ids.shape[1]:]
                    generated = tokenizer.decode(gen_tokens, skip_special_tokens=False)
                    has_eos = gen_tokens[-1].item() == tokenizer.eos_token_id if tokenizer.eos_token_id else False
                    n_gen = len(gen_tokens)
                except Exception as e:
                    generated = f"ERROR: {e}"
                    has_eos = False
                    n_gen = 0
                for h in handles:
                    h.remove()
            else:
                generated, has_eos, n_gen = generate_with_hook(
                    model, tokenizer, device, prompt, layers, head_info, scale
                )

            # Evaluate strict-clean
            clean_eval = evaluate_strict_clean(prompt, generated, has_eos, n_gen)

            all_results.append({
                "prompt": prompt,
                "condition": cond_name,
                "generated": generated[:200],
                "has_eos": has_eos,
                "n_tokens": n_gen,
                "strict_clean": clean_eval["strict_clean"],
                "is_ascii": clean_eval["is_ascii"],
                "is_short": clean_eval["is_short"],
                "has_expected": clean_eval["has_expected"],
                "period_count": generated.count("."),
                "language_switched": not clean_eval["is_ascii"],
            })

        if (pi + 1) % 2 == 0:
            log(f"    {pi+1}/{len(PROMPTS)} prompts")

    # Aggregate
    cond_agg = defaultdict(lambda: {
        "eos_count": 0, "strict_clean_count": 0, "lang_switch_count": 0,
        "n": 0, "tokens": [], "periods": []
    })

    for r in all_results:
        c = r["condition"]
        cond_agg[c]["eos_count"] += int(r["has_eos"])
        cond_agg[c]["strict_clean_count"] += int(r["strict_clean"])
        cond_agg[c]["lang_switch_count"] += int(r["language_switched"])
        cond_agg[c]["n"] += 1
        cond_agg[c]["tokens"].append(r["n_tokens"])
        cond_agg[c]["periods"].append(r["period_count"])

    summary = {}
    for c, d in cond_agg.items():
        summary[c] = {
            "eos_rate": d["eos_count"] / max(d["n"], 1),
            "strict_clean_rate": d["strict_clean_count"] / max(d["n"], 1),
            "lang_switch_rate": d["lang_switch_count"] / max(d["n"], 1),
            "mean_tokens": float(np.mean(d["tokens"])) if d["tokens"] else 0,
            "mean_periods": float(np.mean(d["periods"])) if d["periods"] else 0,
            "n": d["n"],
        }

    output = {
        "task": "task1_2_3_single_head",
        "model": model_name,
        "heads": [(h["layer"], h["head"], h["role"]) for h in heads],
        "max_tokens": MAX_TOKENS,
        "n_prompts": len(PROMPTS),
        "conditions": [c[0] for c in conditions],
        "summary": summary,
        "raw_results": all_results,
    }

    save_path = save_dir / "single_head_results.json"
    save_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Saved: {save_path}")

    # Print summary
    log(f"    Summary (eos_rate / strict_clean / lang_switch):")
    for c in [c[0] for c in conditions]:
        if c in summary:
            s = summary[c]
            log(f"      {c:25s}: eos={s['eos_rate']:.2f}  clean={s['strict_clean_rate']:.2f}  "
                f"switch={s['lang_switch_rate']:.2f}  tokens={s['mean_tokens']:.1f}")

    # Print sample for first prompt
    log(f"    Sample (prompt 0: '{PROMPTS[0][:30]}'):")
    for r in all_results:
        if r["prompt"] == PROMPTS[0] and r["condition"] in ["normal", "ablate_all", "boost2.0_all"]:
            log(f"      {r['condition']:25s}: eos={r['has_eos']}  clean={r['strict_clean']}  "
                f"text={r['generated'][:80]}")

    return output


# ============================================================
# MAIN
# ============================================================

def run_model(model_name: str) -> None:
    log(f"\n{'='*60}")
    log(f"Phase 960: {model_name}")
    log(f"{'='*60}")

    model_dir = RESULT_DIR / model_name
    ensure_dir(model_dir)

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    log(f"  {info.model_class}, {info.n_layers}L, d={info.d_model}")

    t_start = time.time()

    try:
        run_all_tasks(model, tokenizer, device, info, model_name, model_dir)
    except Exception as e:
        log(f"  FAILED: {e}")
        import traceback; traceback.print_exc()

    elapsed = time.time() - t_start
    log(f"\n  Total: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    release_model(model)
    log(f"  {model_name} complete")


def main():
    ensure_dir(RESULT_DIR)
    log(f"Phase {PHASE} started")

    # Run GLM4 first (most interesting), then qwen3, then DS7B
    for m in ["glm4", "qwen3", "deepseek7b"]:
        run_model(m)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log(f"\nPhase {PHASE} complete!")


if __name__ == "__main__":
    main()
