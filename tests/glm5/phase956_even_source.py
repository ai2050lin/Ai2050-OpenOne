#!/usr/bin/env python3
"""
Phase 956: Even效应来源定位与协议通道族机制审计
================================================
Task1: 逐层Odd/Even分解 — 在不同层注入±u，找Even效应产生层
Task2: LayerNorm跳过实验 — 跳过最后层LN，测Even是否消失
Task3: MLP饱和度分析 — gate/SiLU激活分布对比±u
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

PHASE = 956
RESULT_DIR = Path("results/phase956_even_source")
BETA = 4.0
N_RANDOM = 4  # random directions per layer

PROTO_TOKENS = [".", " ", "is", " a"]
PROMPTS = [
    "The apple is", "The sky is", "The grass is",
    "The sun is", "The night is", "The snow is",
]


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


def extract_logits(model, input_ids, proto_ids):
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)
        logits = out.logits[0, -1].detach().float().cpu().numpy()
    return {pt: float(logits[tid]) for pt, tid in proto_ids.items() if tid < len(logits)}


# ============================================================
# TASK 1: Layer-by-layer Odd/Even decomposition
# ============================================================

def task1_layer_odd_even(model, tokenizer, device, info, model_name: str,
                          save_dir: Path) -> dict:
    """Inject ±u at each layer, measure Odd/Even at final logits."""
    log("  Task 1: Layer-by-layer Odd/Even decomposition...")

    n_layers = info.n_layers
    d_model = info.d_model
    layers = get_layers(model)
    embed = model.get_input_embeddings()
    proto_ids = get_proto_ids(tokenizer)

    # Sample layers (every 3-4 layers + first + last)
    sample_layers = list(range(0, n_layers, max(1, n_layers // 8)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    sample_layers = sorted(set(sample_layers))

    # Also test embedding-level injection for comparison
    injection_points = ["embedding"] + [f"L{li}" for li in sample_layers]

    log(f"    Injection points: {injection_points}")

    rng = np.random.RandomState(42)
    random_dirs = [rng.randn(d_model).astype(np.float32) for _ in range(N_RANDOM)]
    random_dirs = [d / np.linalg.norm(d) for d in random_dirs]

    # Results: {injection_point: {token: {"odd": [], "even": []}}}
    results = defaultdict(lambda: defaultdict(lambda: {"odd": [], "even": [], "total": []}))

    for pi, prompt in enumerate(PROMPTS):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        for inj_point in injection_points:
            # Baseline for this prompt
            base_logits = extract_logits(model, input_ids, proto_ids)

            for ri, rdir in enumerate(random_dirs):
                dir_tensor = torch.tensor(rdir, dtype=torch.float32, device=device)

                for sign_label, sign in [("+", 1.0), ("-", -1.0)]:
                    if inj_point == "embedding":
                        # Inject at embedding level
                        inputs_embeds = embed(input_ids).detach().clone()
                        inputs_embeds[0, -1, :] += (sign * BETA * dir_tensor).to(inputs_embeds.dtype)
                        with torch.no_grad():
                            out = model(inputs_embeds=inputs_embeds, use_cache=False)
                            inj_logits = out.logits[0, -1].detach().float().cpu().numpy()
                    else:
                        # Inject at layer output via hook
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
                            except Exception:
                                inj_logits = np.zeros(len(proto_ids))
                        hook_data["active"] = False
                        h.remove()

                    # Compute deltas
                    for pt, tid in proto_ids.items():
                        if tid < len(inj_logits):
                            delta = float(inj_logits[tid] - base_logits.get(pt, 0))
                            results[inj_point][pt][sign_label + "_delta"] = results[inj_point][pt].get(sign_label + "_delta", [])
                            # Store in a flat structure
                            if sign_label == "+":
                                results[inj_point][pt]["plus_deltas"] = results[inj_point][pt].get("plus_deltas", [])
                                results[inj_point][pt]["plus_deltas"].append(delta)
                            else:
                                results[inj_point][pt]["minus_deltas"] = results[inj_point][pt].get("minus_deltas", [])
                                results[inj_point][pt]["minus_deltas"].append(delta)

        if (pi + 1) % 2 == 0:
            log(f"    {pi+1}/{len(PROMPTS)} prompts")

    # Compute Odd/Even per injection point
    layer_stats = {}
    for inj_point, tokens in results.items():
        layer_stats[inj_point] = {}
        for pt, data in tokens.items():
            plus = data.get("plus_deltas", [])
            minus = data.get("minus_deltas", [])
            if not plus or not minus:
                continue

            odds = [(p - m) / 2.0 for p, m in zip(plus, minus)]
            evens = [(p + m) / 2.0 for p, m in zip(plus, minus)]

            mean_odd = float(np.mean(np.abs(odds)))
            mean_even = float(np.mean(np.abs(evens)))
            mean_even_signed = float(np.mean(evens))

            layer_stats[inj_point][pt] = {
                "mean_abs_odd": mean_odd,
                "mean_abs_even": mean_even,
                "odd_even_ratio": mean_odd / max(mean_even, 0.001),
                "even_signed": mean_even_signed,
                "n": len(odds),
            }

    output = {
        "task": "task1_layer_odd_even",
        "model": model_name,
        "beta": BETA,
        "n_prompts": len(PROMPTS),
        "n_random": N_RANDOM,
        "injection_points": injection_points,
        "layer_stats": layer_stats,
    }

    save_path = save_dir / "task1_layer_oddeven.json"
    save_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Saved: {save_path}")

    # Print key results: Even magnitude per layer for period
    log(f"    Even magnitude per layer (period token):")
    for inj_point in injection_points:
        if inj_point in layer_stats and "." in layer_stats[inj_point]:
            ls = layer_stats[inj_point]["."]
            log(f"      {inj_point:12s}: |Odd|={ls['mean_abs_odd']:.4f}  |Even|={ls['mean_abs_even']:.4f}  "
                f"ratio={ls['odd_even_ratio']:.2f}  even_signed={ls['even_signed']:+.4f}")

    return output


# ============================================================
# TASK 2: LayerNorm skip experiment
# ============================================================

def task2_layernorm_skip(model, tokenizer, device, info, model_name: str,
                          save_dir: Path) -> dict:
    """Skip final LayerNorm, test if Even effect disappears."""
    log("  Task 2: LayerNorm skip experiment...")

    d_model = info.d_model
    embed = model.get_input_embeddings()
    proto_ids = get_proto_ids(tokenizer)

    rng = np.random.RandomState(42)
    random_dirs = [rng.randn(d_model).astype(np.float32) for _ in range(N_RANDOM)]
    random_dirs = [d / np.linalg.norm(d) for d in random_dirs]

    # Find the final LayerNorm
    final_ln = None
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        final_ln = model.model.norm
    elif hasattr(model, "model") and hasattr(model.model, "final_layernorm"):
        final_ln = model.model.final_layernorm

    if final_ln is None:
        log("    Cannot find final LayerNorm, skipping")
        return {"task": "task2", "model": model_name, "skipped": True, "reason": "no_final_ln"}

    log(f"    Final LN: {type(final_ln).__name__}")

    # Results: {condition: {token: {odd: [], even: []}}}
    results = defaultdict(lambda: defaultdict(lambda: {"plus_deltas": [], "minus_deltas": []}))

    for pi, prompt in enumerate(PROMPTS):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        for condition in ["normal", "skip_ln"]:
            # Baseline
            hook_handle = None
            if condition == "skip_ln":
                def skip_hook(module, input, output):
                    # Return input unchanged (skip the normalization)
                    if isinstance(input, tuple):
                        return input[0]
                    return input
                hook_handle = final_ln.register_forward_hook(skip_hook)

            base_logits = extract_logits(model, input_ids, proto_ids)

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
                            inj_logits = np.zeros(10)

                    for pt, tid in proto_ids.items():
                        if tid < len(inj_logits):
                            delta = float(inj_logits[tid] - base_logits.get(pt, 0))
                            results[condition][pt][sign_label + "_deltas"] = results[condition][pt].get(sign_label + "_deltas", [])
                            if sign_label == "+":
                                results[condition][pt]["plus_deltas"].append(delta)
                            else:
                                results[condition][pt]["minus_deltas"].append(delta)

            if hook_handle:
                hook_handle.remove()

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
            }

    output = {
        "task": "task2_layernorm_skip",
        "model": model_name,
        "n_prompts": len(PROMPTS),
        "conditions": ["normal", "skip_ln"],
        "cond_stats": cond_stats,
    }

    save_path = save_dir / "task2_ln_skip.json"
    save_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Saved: {save_path}")

    # Print comparison
    log(f"    Even magnitude: normal vs skip_ln")
    for pt in [".", " ", "is", " a", "<EOS>"]:
        normal = cond_stats.get("normal", {}).get(pt, {})
        skip = cond_stats.get("skip_ln", {}).get(pt, {})
        if normal and skip:
            log(f"      {pt:12s}: normal |Even|={normal['mean_abs_even']:.4f} ratio={normal['odd_even_ratio']:.2f}  "
                f"skip |Even|={skip['mean_abs_even']:.4f} ratio={skip['odd_even_ratio']:.2f}  "
                f"change={skip['mean_abs_even'] - normal['mean_abs_even']:+.4f}")

    return output


# ============================================================
# TASK 3: MLP saturation analysis
# ============================================================

def task3_mlp_saturation(model, tokenizer, device, info, model_name: str,
                          save_dir: Path) -> dict:
    """Measure MLP gate/SiLU saturation with and without perturbation."""
    log("  Task 3: MLP saturation analysis...")

    last_layer = info.n_layers - 1
    d_model = info.d_model
    embed = model.get_input_embeddings()
    layers = get_layers(model)

    rng = np.random.RandomState(42)
    rdir = rng.randn(d_model).astype(np.float32)
    rdir = rdir / np.linalg.norm(rdir)

    results = []

    for pi, prompt in enumerate(PROMPTS[:4]):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        for condition, label in [("baseline", "baseline"), ("inject_plus", "+u"), ("inject_minus", "-u")]:
            captured = {}

            def make_gate_hook():
                def hook(module, input, output):
                    captured["gate_raw"] = output[0, -1, :].detach().float().cpu().numpy()
                return hook

            # Hook gate_proj (or gate_up_proj for GLM4)
            mlp = layers[last_layer].mlp
            if hasattr(mlp, "gate_proj"):
                h = mlp.gate_proj.register_forward_hook(make_gate_hook())
            elif hasattr(mlp, "gate_up_proj"):
                # For merged gate_up, we need to split
                inter = info.intermediate_size
                def make_merged_hook():
                    def hook(module, input, output):
                        out = output[0, -1, :].detach().float().cpu().numpy()
                        captured["gate_raw"] = out[:inter]
                    return hook
                h = mlp.gate_up_proj.register_forward_hook(make_merged_hook())
            else:
                continue

            if condition == "baseline":
                with torch.no_grad():
                    _ = model(input_ids=input_ids, use_cache=False)
            else:
                sign = 1.0 if condition == "inject_plus" else -1.0
                dir_tensor = torch.tensor(sign * rdir, dtype=torch.float32, device=device)
                inputs_embeds = embed(input_ids).detach().clone()
                inputs_embeds[0, -1, :] += (BETA * dir_tensor).to(inputs_embeds.dtype)
                with torch.no_grad():
                    _ = model(inputs_embeds=inputs_embeds, use_cache=False)

            h.remove()

            gate_raw = captured.get("gate_raw")
            if gate_raw is not None:
                # SiLU activation: silu(x) = x * sigmoid(x)
                gate_act = gate_raw * (1.0 / (1.0 + np.exp(-gate_raw)))

                # Saturation: |silu(x)| > threshold means highly activated
                saturation_count = int(np.sum(np.abs(gate_act) > 5.0))
                near_zero_count = int(np.sum(np.abs(gate_act) < 0.1))
                mean_activation = float(np.mean(np.abs(gate_act)))
                max_activation = float(np.max(np.abs(gate_act)))
                activation_norm = float(np.linalg.norm(gate_act))

                results.append({
                    "prompt": prompt,
                    "condition": label,
                    "saturation_count": saturation_count,
                    "near_zero_count": near_zero_count,
                    "mean_abs_activation": mean_activation,
                    "max_abs_activation": max_activation,
                    "activation_norm": activation_norm,
                    "n_intermediate": len(gate_raw),
                    "saturation_ratio": saturation_count / len(gate_raw),
                })

        if (pi + 1) % 2 == 0:
            log(f"    {pi+1}/4 prompts")

    # Aggregate
    cond_agg = defaultdict(lambda: {"sat_ratios": [], "act_norms": [], "mean_acts": []})
    for r in results:
        c = r["condition"]
        cond_agg[c]["sat_ratios"].append(r["saturation_ratio"])
        cond_agg[c]["act_norms"].append(r["activation_norm"])
        cond_agg[c]["mean_acts"].append(r["mean_abs_activation"])

    summary = {}
    for cond, data in cond_agg.items():
        summary[cond] = {
            "mean_saturation_ratio": float(np.mean(data["sat_ratios"])),
            "mean_activation_norm": float(np.mean(data["act_norms"])),
            "mean_abs_activation": float(np.mean(data["mean_acts"])),
            "n": len(data["sat_ratios"]),
        }

    output = {
        "task": "task3_mlp_saturation",
        "model": model_name,
        "layer": last_layer,
        "beta": BETA,
        "summary": summary,
        "raw_results": results[:12],
    }

    save_path = save_dir / "task3_saturation.json"
    save_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Saved: {save_path}")

    for cond in ["baseline", "+u", "-u"]:
        if cond in summary:
            s = summary[cond]
            log(f"    {cond:10s}: sat_ratio={s['mean_saturation_ratio']:.4f}  "
                f"act_norm={s['mean_activation_norm']:.2f}  "
                f"mean_act={s['mean_abs_activation']:.4f}")

    return output


# ============================================================
# MAIN
# ============================================================

def run_model(model_name: str) -> None:
    log(f"\n{'='*60}")
    log(f"Phase 956: {model_name}")
    log(f"{'='*60}")

    model_dir = RESULT_DIR / model_name
    ensure_dir(model_dir)

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    log(f"  {info.model_class}, {info.n_layers}L, d={info.d_model}")

    t_start = time.time()

    try:
        task1_layer_odd_even(model, tokenizer, device, info, model_name, model_dir)
    except Exception as e:
        log(f"  Task 1 FAILED: {e}")
        import traceback; traceback.print_exc()

    try:
        task2_layernorm_skip(model, tokenizer, device, info, model_name, model_dir)
    except Exception as e:
        log(f"  Task 2 FAILED: {e}")
        import traceback; traceback.print_exc()

    try:
        task3_mlp_saturation(model, tokenizer, device, info, model_name, model_dir)
    except Exception as e:
        log(f"  Task 3 FAILED: {e}")
        import traceback; traceback.print_exc()

    elapsed = time.time() - t_start
    log(f"\n  Total: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    release_model(model)
    log(f"  {model_name} complete")


def main():
    ensure_dir(RESULT_DIR)
    log(f"Phase {PHASE} started")

    for m in ["qwen3", "deepseek7b"]:
        run_model(m)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log(f"\nPhase {PHASE} complete!")


if __name__ == "__main__":
    main()
