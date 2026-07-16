#!/usr/bin/env python3
"""
Phase 959: Head级rollout干预 + Logit margin分析
================================================
Task1: Head级rollout — 零化token-specific heads后生成文本
Task2: Logit margin分析 — 为什么通道干预不改变argmax
Task3: 扩大规模rollout — 10 prompts × 50 tokens
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
from model_utils import load_model, get_layers, get_model_info, release_model, get_W_U
from phase951_protocol_atlas import ensure_dir

PHASE = 959
RESULT_DIR = Path("results/phase959_head_rollout")

# Token-specific heads from Phase 958
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

# MLP super channels from Phase 953
SUPER_CHANNELS = {
    "qwen3": [935, 36, 284, 153, 188],
    "glm4": [12274, 7968, 5155, 5902, 1106],
    "deepseek7b": [15791, 15305, 1106, 4985, 14464],
}

ROLLOUT_PROMPTS = [
    "The capital of France is",
    "List three colors:\n1.",
    "What is water?",
    "Category: apple is a",
    "The sky is blue.",
    "Explain gravity:",
    "Name two planets:\n-",
    "Classify: dog is a",
    "The largest ocean is",
    "How many continents?",
]

ROLLOUT_MAX_TOKENS = 50


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def get_proto_ids(tokenizer):
    proto_ids = {}
    for pt in [".", " ", "is", " a", "Solution", "Step", "1"]:
        ids = tokenizer.encode(pt, add_special_tokens=False)
        if ids:
            proto_ids[pt] = ids[0]
    if tokenizer.eos_token_id is not None:
        proto_ids["<EOS>"] = tokenizer.eos_token_id
    return proto_ids


# ============================================================
# TASK 1: Head-Level Rollout Intervention
# ============================================================

def task1_head_rollout(model, tokenizer, device, info, model_name: str,
                        save_dir: Path) -> dict:
    """Ablate token-specific heads during text generation."""
    log("  Task 1: Head-level rollout intervention...")

    layers = get_layers(model)
    heads_to_ablate = MODEL_HEADS.get(model_name, [])
    channels = SUPER_CHANNELS.get(model_name, [])[:5]

    if not heads_to_ablate:
        return {"task": "task1", "model": model_name, "skipped": True}

    log(f"    Heads to ablate: {[(h['layer'], h['head'], h['role']) for h in heads_to_ablate]}")

    conditions = [
        ("normal", []),
        ("ablate_heads", [("head", h) for h in heads_to_ablate]),
        ("ablate_channels", [("channel", c) for c in channels]),
        ("ablate_both", [("head", h) for h in heads_to_ablate] + [("channel", c) for c in channels]),
    ]

    results = []

    for pi, prompt in enumerate(ROLLOUT_PROMPTS):
        for cond_name, interventions in conditions:
            # Register hooks
            handles = []

            for interv_type, interv_data in interventions:
                if interv_type == "head":
                    li = interv_data["layer"]
                    head_idx = interv_data["head"]
                    d_head = interv_data["d_head"]
                    start_col = head_idx * d_head
                    end_col = start_col + d_head

                    def make_head_hook(sc, ec):
                        def hook(module, args):
                            inp = args[0] if isinstance(args, tuple) else args
                            patched = inp.clone()
                            if patched.ndim >= 3:
                                patched[:, :, sc:ec] = 0  # Zero ALL positions, not just last
                            return (patched,)
                        return hook

                    h = layers[li].self_attn.o_proj.register_forward_pre_hook(
                        make_head_hook(start_col, end_col)
                    )
                    handles.append(h)

                elif interv_type == "channel":
                    ch_id = interv_data
                    last_layer = info.n_layers - 1

                    def make_ch_hook(channel_id):
                        def hook(module, args):
                            inp = args[0] if isinstance(args, tuple) else args
                            patched = inp.clone()
                            if patched.ndim >= 3:
                                patched[:, -1, channel_id] = 0
                            return (patched,)
                        return hook

                    h = layers[last_layer].mlp.down_proj.register_forward_pre_hook(
                        make_ch_hook(ch_id)
                    )
                    handles.append(h)

            # Generate
            try:
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids,
                        max_new_tokens=ROLLOUT_MAX_TOKENS,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                gen_tokens = output_ids[0][input_ids.shape[1]:]
                generated = tokenizer.decode(gen_tokens, skip_special_tokens=False)

                # Analyze
                period_count = generated.count(".")
                space_count = generated.count(" ")
                newline_count = generated.count("\n")
                word_count = len(generated.split())
                has_eos = gen_tokens[-1].item() == tokenizer.eos_token_id if tokenizer.eos_token_id else False
                n_gen = len(gen_tokens)

                results.append({
                    "prompt": prompt,
                    "condition": cond_name,
                    "generated": generated[:300],
                    "period_count": period_count,
                    "space_count": space_count,
                    "newline_count": newline_count,
                    "word_count": word_count,
                    "n_tokens": n_gen,
                    "has_eos": has_eos,
                })
            except Exception as e:
                results.append({"prompt": prompt, "condition": cond_name, "error": str(e)})

            for h in handles:
                h.remove()

        if (pi + 1) % 2 == 0:
            log(f"    {pi+1}/{len(ROLLOUT_PROMPTS)} prompts")

    # Aggregate
    cond_agg = defaultdict(lambda: {
        "periods": [], "spaces": [], "words": [], "tokens": [],
        "eos_count": 0, "n": 0
    })
    for r in results:
        if "error" in r:
            continue
        c = r["condition"]
        cond_agg[c]["periods"].append(r["period_count"])
        cond_agg[c]["spaces"].append(r["space_count"])
        cond_agg[c]["words"].append(r["word_count"])
        cond_agg[c]["tokens"].append(r["n_tokens"])
        cond_agg[c]["eos_count"] += int(r["has_eos"])
        cond_agg[c]["n"] += 1

    summary = {}
    for c, d in cond_agg.items():
        summary[c] = {
            "mean_periods": float(np.mean(d["periods"])) if d["periods"] else 0,
            "mean_spaces": float(np.mean(d["spaces"])) if d["spaces"] else 0,
            "mean_words": float(np.mean(d["words"])) if d["words"] else 0,
            "mean_tokens": float(np.mean(d["tokens"])) if d["tokens"] else 0,
            "eos_rate": d["eos_count"] / max(d["n"], 1),
            "n": d["n"],
        }

    output = {
        "task": "task1_head_rollout",
        "model": model_name,
        "heads": [(h["layer"], h["head"], h["role"]) for h in heads_to_ablate],
        "channels": channels,
        "max_tokens": ROLLOUT_MAX_TOKENS,
        "n_prompts": len(ROLLOUT_PROMPTS),
        "summary": summary,
        "raw_results": results,
    }

    save_path = save_dir / "task1_head_rollout.json"
    save_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Saved: {save_path}")

    log(f"    Rollout summary:")
    for c in ["normal", "ablate_heads", "ablate_channels", "ablate_both"]:
        if c in summary:
            s = summary[c]
            log(f"      {c:18s}: periods={s['mean_periods']:.1f}  spaces={s['mean_spaces']:.1f}  "
                f"words={s['mean_words']:.1f}  tokens={s['mean_tokens']:.1f}  eos={s['eos_rate']:.2f}")

    # Print sample outputs for first prompt
    log(f"    Sample (prompt 0: '{ROLLOUT_PROMPTS[0][:30]}'):")
    for r in results[:4]:
        if "generated" in r and r["prompt"] == ROLLOUT_PROMPTS[0]:
            log(f"      {r['condition']:18s}: {r['generated'][:100]}")

    return output


# ============================================================
# TASK 2: Logit Margin Analysis
# ============================================================

def task2_logit_margin(model, tokenizer, device, info, model_name: str,
                        save_dir: Path) -> dict:
    """Measure argmax margin to explain why intervention doesn't change output."""
    log("  Task 2: Logit margin analysis...")

    layers = get_layers(model)
    heads_to_ablate = MODEL_HEADS.get(model_name, [])
    channels = SUPER_CHANNELS.get(model_name, [])[:5]
    proto_ids = get_proto_ids(tokenizer)

    prompts = ROLLOUT_PROMPTS[:6]
    conditions = ["normal", "ablate_heads", "ablate_channels"]

    results = []

    for pi, prompt in enumerate(prompts):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        for cond_name in conditions:
            handles = []

            if cond_name == "ablate_heads":
                for h_info in heads_to_ablate:
                    li = h_info["layer"]
                    head_idx = h_info["head"]
                    d_head = h_info["d_head"]
                    sc = head_idx * d_head
                    ec = sc + d_head

                    def make_hook(s, e):
                        def hook(module, args):
                            inp = args[0] if isinstance(args, tuple) else args
                            patched = inp.clone()
                            if patched.ndim >= 3:
                                patched[:, -1, s:e] = 0
                            return (patched,)
                        return hook

                    handles.append(layers[li].self_attn.o_proj.register_forward_pre_hook(make_hook(sc, ec)))

            elif cond_name == "ablate_channels":
                last_layer = info.n_layers - 1
                for ch in channels:
                    def make_ch_hook(c):
                        def hook(module, args):
                            inp = args[0] if isinstance(args, tuple) else args
                            patched = inp.clone()
                            if patched.ndim >= 3:
                                patched[:, -1, c] = 0
                            return (patched,)
                        return hook
                    handles.append(layers[last_layer].mlp.down_proj.register_forward_pre_hook(make_ch_hook(ch)))

            with torch.no_grad():
                out = model(input_ids=input_ids, use_cache=False)
                logits = out.logits[0, -1].detach().float().cpu().numpy()

            for h in handles:
                h.remove()

            # Top-5 tokens
            top5_idx = np.argsort(logits)[-5:][::-1]
            top5_vals = logits[top5_idx]
            top5_tokens = [tokenizer.decode([int(i)]) for i in top5_idx]

            # Margin
            margin = float(top5_vals[0] - top5_vals[1]) if len(top5_vals) > 1 else 0

            # Protocol token logits
            proto_logits = {pt: float(logits[tid]) for pt, tid in proto_ids.items() if tid < len(logits)}

            # Protocol token ranks
            proto_ranks = {}
            for pt, tid in proto_ids.items():
                if tid < len(logits):
                    proto_ranks[pt] = int(np.sum(logits > logits[tid]) + 1)

            results.append({
                "prompt": prompt,
                "condition": cond_name,
                "top1_token": top5_tokens[0],
                "top1_logit": float(top5_vals[0]),
                "top2_token": top5_tokens[1] if len(top5_tokens) > 1 else "",
                "top2_logit": float(top5_vals[1]) if len(top5_vals) > 1 else 0,
                "margin": margin,
                "proto_logits": proto_logits,
                "proto_ranks": proto_ranks,
                "argmax_changed": False,  # Will compare below
            })

        # Check if argmax changed
        if len(results) >= 3:
            normal_top1 = results[-3]["top1_token"]
            for i in range(1, 3):
                results[-3 + i]["argmax_changed"] = (results[-3 + i]["top1_token"] != normal_top1)

        if (pi + 1) % 3 == 0:
            log(f"    {pi+1}/{len(prompts)} prompts")

    # Aggregate
    cond_stats = defaultdict(lambda: {"margins": [], "argmax_changed": 0, "n": 0})
    for r in results:
        c = r["condition"]
        cond_stats[c]["margins"].append(r["margin"])
        cond_stats[c]["argmax_changed"] += int(r.get("argmax_changed", False))
        cond_stats[c]["n"] += 1

    summary = {}
    for c, d in cond_stats.items():
        summary[c] = {
            "mean_margin": float(np.mean(d["margins"])) if d["margins"] else 0,
            "std_margin": float(np.std(d["margins"])) if d["margins"] else 0,
            "argmax_change_rate": d["argmax_changed"] / max(d["n"], 1),
            "n": d["n"],
        }

    output = {
        "task": "task2_margin",
        "model": model_name,
        "n_prompts": len(prompts),
        "summary": summary,
        "raw_results": results[:18],
    }

    save_path = save_dir / "task2_margin.json"
    save_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Saved: {save_path}")

    log(f"    Margin analysis:")
    for c in conditions:
        if c in summary:
            s = summary[c]
            log(f"      {c:18s}: margin={s['mean_margin']:.3f}±{s['std_margin']:.3f}  "
                f"argmax_change={s['argmax_change_rate']:.2f}")

    return output


# ============================================================
# MAIN
# ============================================================

def run_model(model_name: str) -> None:
    log(f"\n{'='*60}")
    log(f"Phase 959: {model_name}")
    log(f"{'='*60}")

    model_dir = RESULT_DIR / model_name
    ensure_dir(model_dir)

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    log(f"  {info.model_class}, {info.n_layers}L, d={info.d_model}")

    t_start = time.time()

    try:
        task1_head_rollout(model, tokenizer, device, info, model_name, model_dir)
    except Exception as e:
        log(f"  Task 1 FAILED: {e}")
        import traceback; traceback.print_exc()

    try:
        task2_logit_margin(model, tokenizer, device, info, model_name, model_dir)
    except Exception as e:
        log(f"  Task 2 FAILED: {e}")
        import traceback; traceback.print_exc()

    elapsed = time.time() - t_start
    log(f"\n  Total: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    release_model(model)
    log(f"  {model_name} complete")


def main():
    ensure_dir(RESULT_DIR)
    log(f"Phase {PHASE} started")

    for m in ["qwen3", "deepseek7b", "glm4"]:
        run_model(m)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log(f"\nPhase {PHASE} complete!")


if __name__ == "__main__":
    main()
