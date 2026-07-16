#!/usr/bin/env python3
"""
Phase 958: Attention头级与MLP通道级协议因果审计 + 自然rollout验证
================================================================
Task1: GLM4 attention head级消融(逐头零化,测protocol logit)
Task2: 自然rollout验证(ablate/boost protocol通道后生成文本)
Task3: qwen3/DS7B MLP通道单通道per-token效果细化
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

PHASE = 958
RESULT_DIR = Path("results/phase958_head_channel_rollout")

PROTO_TOKENS = [".", " ", "is", " a", "Solution", "Step", "1"]
ROLLOUT_PROMPTS = [
    "The capital of France is",
    "List three colors:\n1.",
    "What is water?",
    "Category: apple is a",
    "The sky is blue.",
]
ROLLOUT_MAX_TOKENS = 30

SUPER_CHANNELS = {
    "qwen3": [935, 36, 284, 153, 188],
    "glm4": [12274, 7968, 5155, 5902, 1106],
    "deepseek7b": [15791, 15305, 1106, 4985, 14464],
}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def get_proto_ids(tokenizer):
    proto_ids = {}
    for pt in PROTO_TOKENS:
        ids = tokenizer.encode(pt, add_special_tokens=False)
        if ids:
            proto_ids[pt] = ids[0]
    if tokenizer.eos_token_id is not None:
        proto_ids["<EOS>"] = tokenizer.eos_token_id
    return proto_ids


# ============================================================
# TASK 1: Attention Head-Level Ablation (GLM4 focus)
# ============================================================

def task1_head_ablation(model, tokenizer, device, info, model_name: str,
                         save_dir: Path) -> dict:
    """Zero each attention head at last 3 layers, measure protocol logit change."""
    log("  Task 1: Attention head-level ablation...")

    n_layers = info.n_layers
    layers = get_layers(model)

    # Get attention config
    layer0_attn = layers[0].self_attn
    n_heads = getattr(model.config, "num_attention_heads", 32)
    d_head = info.d_model // n_heads

    log(f"    n_heads={n_heads}, d_head={d_head}, d_model={info.d_model}")

    # Test last 3 layers
    test_layers = list(range(n_layers - 3, n_layers))

    proto_ids = get_proto_ids(tokenizer)
    prompts = ["The apple is", "The sky is", "The grass is", "The sun is"]

    # Results: {layer_head: {token: [deltas]}}
    results = defaultdict(lambda: defaultdict(list))

    for pi, prompt in enumerate(prompts):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Baseline
        with torch.no_grad():
            base_out = model(input_ids=input_ids, use_cache=False)
            base_logits = base_out.logits[0, -1].detach().float().cpu().numpy()

        for li in test_layers:
            attn = layers[li].self_attn

            for head_idx in range(n_heads):
                # Hook o_proj input to zero out specific head
                start_col = head_idx * d_head
                end_col = start_col + d_head

                def make_head_hook(sc, ec):
                    def hook(module, args):
                        inp = args[0] if isinstance(args, tuple) else args
                        patched = inp.clone()
                        if patched.ndim >= 3:
                            patched[:, -1, sc:ec] = 0
                        return (patched,)
                    return hook

                handle = attn.o_proj.register_forward_pre_hook(
                    make_head_hook(start_col, end_col)
                )

                with torch.no_grad():
                    try:
                        out = model(input_ids=input_ids, use_cache=False)
                        patched_logits = out.logits[0, -1].detach().float().cpu().numpy()
                    except:
                        patched_logits = base_logits.copy()

                handle.remove()

                key = f"L{li}_H{head_idx}"
                for pt, tid in proto_ids.items():
                    if tid < len(patched_logits):
                        delta = float(patched_logits[tid] - base_logits[tid])
                        results[key][pt].append(delta)

        if (pi + 1) % 2 == 0:
            log(f"    {pi+1}/{len(prompts)} prompts")

    # Aggregate: find heads with largest effect
    head_stats = {}
    for key, tokens in results.items():
        token_means = {}
        for pt, deltas in tokens.items():
            token_means[pt] = {
                "mean": float(np.mean(deltas)),
                "abs_mean": float(np.mean(np.abs(deltas))),
                "std": float(np.std(deltas)),
            }

        # Overall effect size (mean abs across tokens)
        overall_abs = float(np.mean([v["abs_mean"] for v in token_means.values()]))

        # Classify: does it affect period more than others?
        period_effect = token_means.get(".", {}).get("abs_mean", 0)
        eos_effect = token_means.get("<EOS>", {}).get("abs_mean", 0)
        all_effects = [v["abs_mean"] for v in token_means.values()]
        effect_variance = float(np.std(all_effects))

        # Role classification
        max_token = max(token_means.items(), key=lambda x: x[1]["abs_mean"])
        min_token = min(token_means.items(), key=lambda x: x[1]["abs_mean"])

        if effect_variance < 0.1 * overall_abs:
            role = "general_support"
        elif max_token[1]["abs_mean"] > 3 * min_token[1]["abs_mean"]:
            role = f"token_specific({max_token[0]})"
        elif any(v["mean"] > 0 for v in token_means.values()) and any(v["mean"] < 0 for v in token_means.values()):
            role = "mixed_opposite"
        else:
            role = "mixed"

        head_stats[key] = {
            "overall_abs_effect": overall_abs,
            "effect_variance": effect_variance,
            "role": role,
            "token_effects": token_means,
            "max_token": max_token[0],
            "min_token": min_token[0],
        }

    # Sort by overall effect
    sorted_heads = sorted(head_stats.items(), key=lambda x: -x[1]["overall_abs_effect"])

    output = {
        "task": "task1_head_ablation",
        "model": model_name,
        "n_heads": n_heads,
        "d_head": d_head,
        "test_layers": test_layers,
        "n_prompts": len(prompts),
        "head_stats": dict(sorted_heads[:50]),  # Top 50
        "n_heads_tested": len(head_stats),
    }

    save_path = save_dir / "task1_heads.json"
    save_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Saved: {save_path}")

    # Print top 10 heads
    log(f"    Top 10 heads by effect:")
    for key, hs in sorted_heads[:10]:
        log(f"      {key}: effect={hs['overall_abs_effect']:.4f}  role={hs['role']}  "
            f"max={hs['max_token']}  period={hs['token_effects'].get('.', {}).get('abs_mean', 0):.4f}")

    return output


# ============================================================
# TASK 2: Natural Rollout with Protocol Channel Intervention
# ============================================================

def task2_rollout(model, tokenizer, device, info, model_name: str,
                   save_dir: Path) -> dict:
    """Generate text with ablated/boosted protocol channels."""
    log("  Task 2: Natural rollout with channel intervention...")

    last_layer = info.n_layers - 1
    layers_list = get_layers(model)
    target_layer = layers_list[last_layer]
    channels = SUPER_CHANNELS.get(model_name, [])[:5]

    if not channels:
        return {"task": "task2", "model": model_name, "skipped": True}

    conditions = [
        ("normal", 1.0, []),
        ("ablate_K5", 0.0, channels[:5]),
        ("boost_K5", 1.5, channels[:5]),
    ]

    results = []

    for pi, prompt in enumerate(ROLLOUT_PROMPTS):
        for cond_name, scale, chs in conditions:
            # Register hook if needed
            handle = None
            if chs:
                def make_pre_hook(ch_list, scl):
                    ch_set = set(int(c) for c in ch_list)
                    def pre_hook(module, args):
                        inp = args[0] if isinstance(args, tuple) else args
                        patched = inp.clone()
                        if patched.ndim >= 3:
                            for c in ch_set:
                                if scl == 0:
                                    patched[:, -1, c] = 0
                                else:
                                    patched[:, -1, c] = patched[:, -1, c] * scl
                        return (patched,)
                    return pre_hook

                handle = target_layer.mlp.down_proj.register_forward_pre_hook(
                    make_pre_hook(chs, scale)
                )

            # Generate
            try:
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids,
                        max_new_tokens=ROLLOUT_MAX_TOKENS,
                        do_sample=False,
                        temperature=1.0,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                generated = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=False)

                # Analyze output
                period_count = generated.count(".")
                space_count = generated.count(" ")
                newline_count = generated.count("\n")
                eos_in_output = tokenizer.eos_token in generated if tokenizer.eos_token else False
                word_count = len(generated.split())
                output_length = len(generated)

                # Check if EOS was generated
                has_eos = output_ids[0][-1].item() == tokenizer.eos_token_id if tokenizer.eos_token_id else False

                results.append({
                    "prompt": prompt,
                    "condition": cond_name,
                    "generated_text": generated[:200],
                    "period_count": period_count,
                    "space_count": space_count,
                    "newline_count": newline_count,
                    "word_count": word_count,
                    "output_length": output_length,
                    "has_eos": has_eos,
                    "n_tokens_generated": output_ids.shape[1] - input_ids.shape[1],
                })
            except Exception as e:
                results.append({
                    "prompt": prompt,
                    "condition": cond_name,
                    "error": str(e),
                })

            if handle:
                handle.remove()

        if (pi + 1) % 2 == 0:
            log(f"    {pi+1}/{len(ROLLOUT_PROMPTS)} prompts")

    # Aggregate
    cond_agg = defaultdict(lambda: {
        "period_counts": [], "space_counts": [], "word_counts": [],
        "n_tokens": [], "has_eos_count": 0, "n_prompts": 0
    })

    for r in results:
        if "error" in r:
            continue
        c = r["condition"]
        cond_agg[c]["period_counts"].append(r["period_count"])
        cond_agg[c]["space_counts"].append(r["space_count"])
        cond_agg[c]["word_counts"].append(r["word_count"])
        cond_agg[c]["n_tokens"].append(r["n_tokens_generated"])
        cond_agg[c]["has_eos_count"] += int(r["has_eos"])
        cond_agg[c]["n_prompts"] += 1

    summary = {}
    for cond, data in cond_agg.items():
        summary[cond] = {
            "mean_period_count": float(np.mean(data["period_counts"])) if data["period_counts"] else 0,
            "mean_space_count": float(np.mean(data["space_counts"])) if data["space_counts"] else 0,
            "mean_word_count": float(np.mean(data["word_counts"])) if data["word_counts"] else 0,
            "mean_n_tokens": float(np.mean(data["n_tokens"])) if data["n_tokens"] else 0,
            "eos_termination_rate": data["has_eos_count"] / max(data["n_prompts"], 1),
            "n_prompts": data["n_prompts"],
        }

    output = {
        "task": "task2_rollout",
        "model": model_name,
        "max_new_tokens": ROLLOUT_MAX_TOKENS,
        "n_prompts": len(ROLLOUT_PROMPTS),
        "channels": channels,
        "summary": summary,
        "raw_results": results,
    }

    save_path = save_dir / "task2_rollout.json"
    save_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Saved: {save_path}")

    # Print summary
    log(f"    Rollout summary:")
    for cond in ["normal", "ablate_K5", "boost_K5"]:
        if cond in summary:
            s = summary[cond]
            log(f"      {cond:15s}: periods={s['mean_period_count']:.1f}  words={s['mean_word_count']:.1f}  "
                f"tokens={s['mean_n_tokens']:.1f}  eos_rate={s['eos_termination_rate']:.2f}")

    # Print sample outputs
    log(f"    Sample outputs (prompt 0):")
    for r in results[:3]:
        if "generated_text" in r and r["prompt"] == ROLLOUT_PROMPTS[0]:
            log(f"      {r['condition']:15s}: {r['generated_text'][:100]}")

    return output


# ============================================================
# MAIN
# ============================================================

def run_model(model_name: str) -> None:
    log(f"\n{'='*60}")
    log(f"Phase 958: {model_name}")
    log(f"{'='*60}")

    model_dir = RESULT_DIR / model_name
    ensure_dir(model_dir)

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    log(f"  {info.model_class}, {info.n_layers}L, d={info.d_model}")

    t_start = time.time()

    # Task 1: Head ablation (all models, but most interesting for GLM4)
    try:
        task1_head_ablation(model, tokenizer, device, info, model_name, model_dir)
    except Exception as e:
        log(f"  Task 1 FAILED: {e}")
        import traceback; traceback.print_exc()

    # Task 2: Rollout (all models)
    try:
        task2_rollout(model, tokenizer, device, info, model_name, model_dir)
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
