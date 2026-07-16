#!/usr/bin/env python3
"""
Phase 957: Attention非线性与逐词元Even来源审计
==============================================
Task1: Attention消融 — 跳过attention vs 跳过MLP, 测Even变化
Task2: 逐token Even热图 — 7 token × 采样层 × 3模型
Task3: 二阶曲率 (= 2*Even, 从Task1数据直接计算)
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

PHASE = 957
RESULT_DIR = Path("results/phase957_attention_even")
BETA = 4.0
N_RANDOM = 4

# 7 protocol tokens for heatmap
HEATMAP_TOKENS = [".", " ", "is", " a", "Solution", "Step", "1"]

PROMPTS = [
    "The apple is", "The sky is", "The grass is",
    "The sun is", "The night is", "The snow is",
]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def get_proto_ids(tokenizer, tokens=None):
    toks = tokens or HEATMAP_TOKENS
    proto_ids = {}
    for pt in toks:
        ids = tokenizer.encode(pt, add_special_tokens=False)
        if ids:
            proto_ids[pt] = ids[0]
    if tokenizer.eos_token_id is not None:
        proto_ids["<EOS>"] = tokenizer.eos_token_id
    return proto_ids


def zero_module_output(output):
    """Zero out the last token position of a module's output."""
    if isinstance(output, tuple):
        if not output or not torch.is_tensor(output[0]):
            return output
        patched = output[0].clone()
        if patched.ndim >= 3:
            patched[:, -1, :] = 0
        return (patched, *output[1:])
    if torch.is_tensor(output):
        patched = output.clone()
        if patched.ndim >= 3:
            patched[:, -1, :] = 0
        return patched
    return output


# ============================================================
# TASK 1: Attention vs MLP Ablation (Even source)
# ============================================================

def task1_attn_vs_mlp(model, tokenizer, device, info, model_name: str,
                       save_dir: Path) -> dict:
    """Skip attention or MLP at last K layers, measure Even change."""
    log("  Task 1: Attention vs MLP ablation...")

    n_layers = info.n_layers
    layers = get_layers(model)
    embed = model.get_input_embeddings()
    proto_ids = get_proto_ids(tokenizer)
    d_model = info.d_model

    rng = np.random.RandomState(42)
    random_dirs = [rng.randn(d_model).astype(np.float32) for _ in range(N_RANDOM)]
    random_dirs = [d / np.linalg.norm(d) for d in random_dirs]

    # Skip last 3 layers' attention or MLP
    K_SKIP = 3
    skip_layers = list(range(n_layers - K_SKIP, n_layers))

    # Conditions: normal, skip_attn, skip_mlp
    conditions = ["normal", "skip_attn", "skip_mlp"]

    # Results: {condition: {token: {plus_deltas: [], minus_deltas: []}}}
    results = defaultdict(lambda: defaultdict(lambda: {"plus_deltas": [], "minus_deltas": []}))

    for pi, prompt in enumerate(PROMPTS):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        for condition in conditions:
            # Register hooks to skip attention or MLP
            hooks = []
            if condition == "skip_attn":
                for li in skip_layers:
                    h = layers[li].self_attn.register_forward_hook(
                        lambda mod, inp, out: zero_module_output(out)
                    )
                    hooks.append(h)
            elif condition == "skip_mlp":
                for li in skip_layers:
                    h = layers[li].mlp.register_forward_hook(
                        lambda mod, inp, out: zero_module_output(out)
                    )
                    hooks.append(h)

            # Baseline for this condition
            with torch.no_grad():
                base_out = model(input_ids=input_ids, use_cache=False)
                base_logits = base_out.logits[0, -1].detach().float().cpu().numpy()

            for ri, rdir in enumerate(random_dirs):
                dir_tensor = torch.tensor(rdir, dtype=torch.float32, device=device)

                for sign_label, sign in [("+", 1.0), ("-", -1.0)]:
                    inputs_embeds = embed(input_ids).detach().clone()
                    inputs_embeds[0, -1, :] += (sign * BETA * dir_tensor).to(inputs_embeds.dtype)

                    with torch.no_grad():
                        try:
                            out = model(inputs_embeds=inputs_embeds, use_cache=False)
                            inj_logits = out.logits[0, -1].detach().float().cpu().numpy()
                        except:
                            inj_logits = base_logits.copy()

                    for pt, tid in proto_ids.items():
                        if tid < len(inj_logits):
                            delta = float(inj_logits[tid] - base_logits[tid])
                            if sign_label == "+":
                                results[condition][pt]["plus_deltas"].append(delta)
                            else:
                                results[condition][pt]["minus_deltas"].append(delta)

            for h in hooks:
                h.remove()

        if (pi + 1) % 2 == 0:
            log(f"    {pi+1}/{len(PROMPTS)} prompts")

    # Compute Odd/Even per condition
    cond_stats = {}
    for condition, tokens in results.items():
        cond_stats[condition] = {}
        for pt, data in tokens.items():
            plus = data.get("plus_deltas", [])
            minus = data.get("minus_deltas", [])
            if not plus or not minus:
                continue

            odds = [(p - m) / 2.0 for p, m in zip(plus, minus)]
            evens = [(p + m) / 2.0 for p, m in zip(plus, minus)]

            cond_stats[condition][pt] = {
                "mean_abs_odd": float(np.mean(np.abs(odds))),
                "mean_abs_even": float(np.mean(np.abs(evens))),
                "odd_even_ratio": float(np.mean(np.abs(odds)) / max(np.mean(np.abs(evens)), 0.001)),
                "even_signed": float(np.mean(evens)),
                "curvature": float(np.mean(evens) * 2),  # Curv = 2*Even
            }

    output = {
        "task": "task1_attn_vs_mlp",
        "model": model_name,
        "beta": BETA,
        "n_prompts": len(PROMPTS),
        "n_random": N_RANDOM,
        "skip_layers": skip_layers,
        "conditions": conditions,
        "cond_stats": cond_stats,
    }

    save_path = save_dir / "task1_attn_mlp.json"
    save_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Saved: {save_path}")

    # Print comparison
    log(f"    Even magnitude: normal vs skip_attn vs skip_mlp")
    for pt in [".", " ", "is", " a", "<EOS>"]:
        normal = cond_stats.get("normal", {}).get(pt, {})
        skip_a = cond_stats.get("skip_attn", {}).get(pt, {})
        skip_m = cond_stats.get("skip_mlp", {}).get(pt, {})
        if normal and skip_a and skip_m:
            n_even = normal["mean_abs_even"]
            a_even = skip_a["mean_abs_even"]
            m_even = skip_m["mean_abs_even"]
            log(f"      {pt:12s}: normal={n_even:.4f}  skip_attn={a_even:.4f}({a_even/n_even:.1f}x)  "
                f"skip_mlp={m_even:.4f}({m_even/n_even:.1f}x)")

    return output


# ============================================================
# TASK 2: Per-token Even Heatmap (layer-by-layer)
# ============================================================

def task2_token_heatmap(model, tokenizer, device, info, model_name: str,
                         save_dir: Path) -> dict:
    """Build token × layer Even magnitude heatmap."""
    log("  Task 2: Per-token Even heatmap...")

    n_layers = info.n_layers
    d_model = info.d_model
    layers = get_layers(model)
    embed = model.get_input_embeddings()
    proto_ids = get_proto_ids(tokenizer)

    # Sample layers
    sample_layers = list(range(0, n_layers, max(1, n_layers // 6)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    sample_layers = sorted(set(sample_layers))

    injection_points = ["embedding"] + [f"L{li}" for li in sample_layers]

    rng = np.random.RandomState(42)
    random_dirs = [rng.randn(d_model).astype(np.float32) for _ in range(N_RANDOM)]
    random_dirs = [d / np.linalg.norm(d) for d in random_dirs]

    # Results: {inj_point: {token: {plus_deltas: [], minus_deltas: []}}}
    results = defaultdict(lambda: defaultdict(lambda: {"plus_deltas": [], "minus_deltas": []}))

    for pi, prompt in enumerate(PROMPTS):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        for inj_point in injection_points:
            # Baseline
            with torch.no_grad():
                base_out = model(input_ids=input_ids, use_cache=False)
                base_logits = base_out.logits[0, -1].detach().float().cpu().numpy()

            for ri, rdir in enumerate(random_dirs):
                dir_tensor = torch.tensor(rdir, dtype=torch.float32, device=device)

                for sign_label, sign in [("+", 1.0), ("-", -1.0)]:
                    if inj_point == "embedding":
                        inputs_embeds = embed(input_ids).detach().clone()
                        inputs_embeds[0, -1, :] += (sign * BETA * dir_tensor).to(inputs_embeds.dtype)
                        with torch.no_grad():
                            out = model(inputs_embeds=inputs_embeds, use_cache=False)
                            inj_logits = out.logits[0, -1].detach().float().cpu().numpy()
                    else:
                        li = int(inj_point[1:])
                        hook_data = {"active": False}

                        def make_hook(direction, beta_val, sign_val):
                            def hook(module, input, output):
                                if hook_data["active"]:
                                    if isinstance(output, tuple):
                                        patched = output[0].clone()
                                        patched[:, -1, :] += (sign_val * beta_val * direction).to(patched.dtype)
                                        return (patched, *output[1:])
                                    elif torch.is_tensor(output):
                                        patched = output.clone()
                                        patched[:, -1, :] += (sign_val * beta_val * direction).to(patched.dtype)
                                        return patched
                                return output
                            return hook

                        h = layers[li].register_forward_hook(make_hook(dir_tensor, BETA, sign))
                        hook_data["active"] = True
                        with torch.no_grad():
                            try:
                                out = model(input_ids=input_ids, use_cache=False)
                                inj_logits = out.logits[0, -1].detach().float().cpu().numpy()
                            except:
                                inj_logits = base_logits.copy()
                        hook_data["active"] = False
                        h.remove()

                    for pt, tid in proto_ids.items():
                        if tid < len(inj_logits):
                            delta = float(inj_logits[tid] - base_logits[tid])
                            if sign_label == "+":
                                results[inj_point][pt]["plus_deltas"].append(delta)
                            else:
                                results[inj_point][pt]["minus_deltas"].append(delta)

        if (pi + 1) % 2 == 0:
            log(f"    {pi+1}/{len(PROMPTS)} prompts")

    # Build heatmap: {inj_point: {token: {even_abs, odd_abs, ratio, curvature}}}
    heatmap = {}
    for inj_point, tokens in results.items():
        heatmap[inj_point] = {}
        for pt, data in tokens.items():
            plus = data.get("plus_deltas", [])
            minus = data.get("minus_deltas", [])
            if not plus or not minus:
                continue

            odds = [(p - m) / 2.0 for p, m in zip(plus, minus)]
            evens = [(p + m) / 2.0 for p, m in zip(plus, minus)]

            heatmap[inj_point][pt] = {
                "even_abs": float(np.mean(np.abs(evens))),
                "odd_abs": float(np.mean(np.abs(odds))),
                "ratio": float(np.mean(np.abs(odds)) / max(np.mean(np.abs(evens)), 0.001)),
                "even_signed": float(np.mean(evens)),
                "curvature": float(np.mean(evens) * 2),
            }

    output = {
        "task": "task2_heatmap",
        "model": model_name,
        "beta": BETA,
        "n_prompts": len(PROMPTS),
        "n_random": N_RANDOM,
        "injection_points": injection_points,
        "tokens": list(proto_ids.keys()),
        "heatmap": heatmap,
    }

    save_path = save_dir / "task2_heatmap.json"
    save_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Saved: {save_path}")

    # Print heatmap for Even magnitude
    log(f"    Even magnitude heatmap:")
    header = "Layer".ljust(12)
    for pt in [".", " ", "is", " a", "<EOS>", "Solution"]:
        header += f"  {pt:>10s}"
    log(f"    {header}")
    for inj_point in injection_points:
        row = inj_point.ljust(12)
        for pt in [".", " ", "is", " a", "<EOS>", "Solution"]:
            val = heatmap.get(inj_point, {}).get(pt, {}).get("even_abs", 0)
            row += f"  {val:10.4f}"
        log(f"    {row}")

    return output


# ============================================================
# MAIN
# ============================================================

def run_model(model_name: str) -> None:
    log(f"\n{'='*60}")
    log(f"Phase 957: {model_name}")
    log(f"{'='*60}")

    model_dir = RESULT_DIR / model_name
    ensure_dir(model_dir)

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    log(f"  {info.model_class}, {info.n_layers}L, d={info.d_model}")

    t_start = time.time()

    try:
        task1_attn_vs_mlp(model, tokenizer, device, info, model_name, model_dir)
    except Exception as e:
        log(f"  Task 1 FAILED: {e}")
        import traceback; traceback.print_exc()

    try:
        task2_token_heatmap(model, tokenizer, device, info, model_name, model_dir)
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

    for m in ["qwen3", "glm4", "deepseek7b"]:
        run_model(m)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log(f"\nPhase {PHASE} complete!")


if __name__ == "__main__":
    main()
