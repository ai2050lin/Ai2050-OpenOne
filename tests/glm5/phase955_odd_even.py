#!/usr/bin/env python3
"""
Phase 955: 随机基线分解与协议通道族角色审计
=============================================
Task1-3: Odd/Even分解 + 语义残差稳定性
  - 16随机方向 × ±1 (共32次per prompt)
  - Odd = (Δz(+u) - Δz(-u)) / 2  → 方向投影效应
  - Even = (Δz(+u) + Δz(-u)) / 2  → 范数/曲率效应
Task4:   协议通道族角色分解
  - 单通道ablate/boost + leave-one-out
  - 分类: general/specific/suppressor
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
from phase951_protocol_atlas import _get_mlp_activation_hook, ensure_dir

PHASE = 955
RESULT_DIR = Path("results/phase955_odd_even")
BETA = 4.0
N_RANDOM = 16  # 16 random directions, each tested ±1

SEMANTIC_PAIRS = [
    ("color", "red", "blue"),
    ("size", "big", "small"),
    ("shape", "round", "square"),
    ("emotion", "happy", "sad"),
    ("speed", "fast", "slow"),
    ("function", "tool", "food"),
    ("category", "animal", "plant"),
]

PROTO_TOKENS = [".", " ", "\n", "the", " The", "is", " a", "Solution", "Step", "1"]

PROMPTS = [
    "The apple is", "The sky is", "The grass is", "The sun is",
    "The night is", "The snow is", "The fire is", "The ocean is",
    "The lemon is", "The rose is", "The coal is", "The cloud is",
    "The wood is", "The wine is", "The metal is",
]

SUPER_CHANNELS = {
    "qwen3": [935, 36, 284, 153, 188],
    "glm4": [12274, 7968, 5155, 5902, 1106],
    "deepseek7b": [15791, 15305, 1106, 4985, 14464],
}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
# TASK 1-3: Odd/Even Decomposition + Semantic Residual
# ============================================================

def task1_3_odd_even(model, tokenizer, device, info, model_name: str,
                      save_dir: Path) -> dict:
    """Odd/Even decomposition of random baseline + semantic residual."""
    log("  Task 1-3: Odd/Even decomposition...")

    W_U = get_W_U(model, model_name)
    embed = model.get_input_embeddings()

    # Build semantic directions
    sem_dirs = {}
    for cat, w1, w2 in SEMANTIC_PAIRS:
        ids1 = tokenizer.encode(w1, add_special_tokens=False)
        ids2 = tokenizer.encode(w2, add_special_tokens=False)
        if ids1 and ids2 and ids1[0] < W_U.shape[0] and ids2[0] < W_U.shape[0]:
            d = W_U[ids1[0]] - W_U[ids2[0]]
            norm = np.linalg.norm(d)
            if norm > 1e-10:
                sem_dirs[f"{cat}_{w1}-{w2}"] = d / norm

    # Protocol token IDs
    proto_ids = {}
    for pt in PROTO_TOKENS:
        ids = tokenizer.encode(pt, add_special_tokens=False)
        if ids:
            proto_ids[pt] = ids[0]
    if tokenizer.eos_token_id is not None:
        proto_ids["<EOS>"] = tokenizer.eos_token_id

    # Generate random directions
    rng = np.random.RandomState(42)
    d_model = W_U.shape[1]
    random_dirs = [rng.randn(d_model).astype(np.float32) for _ in range(N_RANDOM)]
    random_dirs = [d / np.linalg.norm(d) for d in random_dirs]

    log(f"    {len(PROMPTS)} prompts, {len(sem_dirs)} sem dirs, {N_RANDOM} random dirs x 2")

    all_results = []

    for pi, prompt in enumerate(PROMPTS):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Baseline
        with torch.no_grad():
            base_out = model(input_ids=input_ids, use_cache=False)
            base_logits = base_out.logits[0, -1].detach().float().cpu().numpy()

        prompt_data = {"prompt": prompt, "semantic": {}, "random": []}

        # Semantic injections
        for dname, sdir in sem_dirs.items():
            dir_t = torch.tensor(sdir, dtype=torch.float32, device=device)
            inputs_embeds = embed(input_ids).detach().clone()
            inputs_embeds[0, -1, :] += (BETA * dir_t).to(inputs_embeds.dtype)
            with torch.no_grad():
                inj_out = model(inputs_embeds=inputs_embeds, use_cache=False)
                inj_logits = inj_out.logits[0, -1].detach().float().cpu().numpy()

            deltas = {}
            for pt, tid in proto_ids.items():
                if tid < len(inj_logits):
                    deltas[pt] = float(inj_logits[tid] - base_logits[tid])
            prompt_data["semantic"][dname] = deltas

        # Random direction injections (+u and -u)
        for ri, rdir in enumerate(random_dirs):
            rand_entry = {"idx": ri}
            for sign_label, sign in [("+", 1.0), ("-", -1.0)]:
                dir_t = torch.tensor(sign * rdir, dtype=torch.float32, device=device)
                inputs_embeds = embed(input_ids).detach().clone()
                inputs_embeds[0, -1, :] += (BETA * dir_t).to(inputs_embeds.dtype)
                with torch.no_grad():
                    rand_out = model(inputs_embeds=inputs_embeds, use_cache=False)
                    rand_logits = rand_out.logits[0, -1].detach().float().cpu().numpy()

                deltas = {}
                for pt, tid in proto_ids.items():
                    if tid < len(rand_logits):
                        deltas[pt] = float(rand_logits[tid] - base_logits[tid])
                rand_entry[sign_label] = deltas

            # Compute Odd and Even
            odd = {}
            even = {}
            for pt in proto_ids:
                dp = rand_entry["+"].get(pt, 0)
                dn = rand_entry["-"].get(pt, 0)
                odd[pt] = (dp - dn) / 2.0
                even[pt] = (dp + dn) / 2.0
            rand_entry["odd"] = odd
            rand_entry["even"] = even
            prompt_data["random"].append(rand_entry)

        all_results.append(prompt_data)

        if (pi + 1) % 5 == 0:
            log(f"    {pi+1}/{len(PROMPTS)} prompts")

    # Aggregate: Odd/Even statistics + semantic residual
    # For each token: compute mean |Odd|, mean |Even|, and Odd/Even ratio
    token_stats = {}
    for pt in proto_ids:
        all_odds = []
        all_evens = []
        all_sem_deltas = defaultdict(list)  # by direction
        all_rand_deltas = []

        for pd in all_results:
            for re in pd["random"]:
                all_odds.append(abs(re["odd"].get(pt, 0)))
                all_evens.append(abs(re["even"].get(pt, 0)))
                all_rand_deltas.append(re["+"].get(pt, 0))
                all_rand_deltas.append(re["-"].get(pt, 0))

            for dname, deltas in pd["semantic"].items():
                all_sem_deltas[dname].append(deltas.get(pt, 0))

        mean_odd = float(np.mean(all_odds)) if all_odds else 0
        mean_even = float(np.mean(all_evens)) if all_evens else 0
        rand_mean = float(np.mean(all_rand_deltas)) if all_rand_deltas else 0

        # Semantic residual per direction
        residuals = {}
        for dname, vals in all_sem_deltas.items():
            sem_mean = float(np.mean(vals))
            residuals[dname] = {
                "sem_mean": sem_mean,
                "rand_baseline": rand_mean,
                "residual": sem_mean - rand_mean,
                "residual_std": float(np.std(vals)),
                "n": len(vals),
            }

        token_stats[pt] = {
            "mean_abs_odd": mean_odd,
            "mean_abs_even": mean_even,
            "odd_even_ratio": mean_odd / max(mean_even, 0.001),
            "rand_baseline_mean": rand_mean,
            "residuals": residuals,
            "n_positive_residuals": sum(1 for r in residuals.values() if r["residual"] > 0),
            "n_total_directions": len(residuals),
        }

    output = {
        "task": "task1_3_odd_even",
        "model": model_name,
        "beta": BETA,
        "n_prompts": len(PROMPTS),
        "n_semantic_dirs": len(sem_dirs),
        "n_random_dirs": N_RANDOM,
        "n_random_total": N_RANDOM * 2,  # ±1
        "token_stats": token_stats,
    }

    save_path = save_dir / "task1_3_odd_even.json"
    save_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Saved: {save_path}")

    # Print key results
    log(f"    Odd/Even decomposition (mean |Odd| vs mean |Even|):")
    for pt in [".", " ", "<EOS>", "is", " a"]:
        if pt in token_stats:
            ts = token_stats[pt]
            log(f"      {pt:12s}: |Odd|={ts['mean_abs_odd']:.4f}  |Even|={ts['mean_abs_even']:.4f}  "
                f"ratio={ts['odd_even_ratio']:.2f}  rand_base={ts['rand_baseline_mean']:+.3f}  "
                f"resid_pos={ts['n_positive_residuals']}/{ts['n_total_directions']}")

    return output


# ============================================================
# TASK 4: Channel Role Decomposition
# ============================================================

def task4_channel_roles(model, tokenizer, device, info, model_name: str,
                         save_dir: Path) -> dict:
    """Decompose protocol channels into roles: general/specific/suppressor."""
    log("  Task 4: Channel role decomposition...")

    last_layer = info.n_layers - 1
    layers_list = get_layers(model)
    target_layer = layers_list[last_layer]
    channels = SUPER_CHANNELS.get(model_name, [])[:5]

    if not channels:
        return {"task": "task4", "model": model_name, "skipped": True}

    proto_ids = {}
    for pt in PROTO_TOKENS:
        ids = tokenizer.encode(pt, add_special_tokens=False)
        if ids:
            proto_ids[pt] = ids[0]
    if tokenizer.eos_token_id is not None:
        proto_ids["<EOS>"] = tokenizer.eos_token_id

    # Conditions: single ablate, single boost, leave-one-out ablate
    conditions = []
    for ch in channels:
        conditions.append((f"ablate_ch{ch}", 0.0, [ch]))
        conditions.append((f"boost_ch{ch}", 1.5, [ch]))
    # Leave-one-out: ablate all except ch
    for ch in channels:
        loo_channels = [c for c in channels if c != ch]
        conditions.append((f"loo_ch{ch}", 0.0, loo_channels))

    results = []

    for pi, prompt in enumerate(PROMPTS[:10]):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Baseline
        with torch.no_grad():
            base_out = model(input_ids=input_ids, use_cache=False)
            base_logits = base_out.logits[0, -1].detach().float().cpu().numpy()

        for cond_name, scale, chs in conditions:
            def make_pre_hook(ch_list, scl):
                ch_set = set(int(c) for c in ch_list)
                def pre_hook(module, args):
                    inp = args[0]
                    if isinstance(inp, tuple):
                        inp = inp[0]
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
            with torch.no_grad():
                try:
                    patched_out = model(input_ids=input_ids, use_cache=False)
                    patched_logits = patched_out.logits[0, -1].detach().float().cpu().numpy()
                except:
                    patched_logits = base_logits.copy()
            handle.remove()

            token_deltas = {}
            for pt, tid in proto_ids.items():
                if tid < len(patched_logits):
                    token_deltas[pt] = float(patched_logits[tid] - base_logits[tid])

            results.append({
                "prompt": prompt,
                "condition": cond_name,
                "channels": chs,
                "scale": scale,
                "token_deltas": token_deltas,
            })

        if (pi + 1) % 3 == 0:
            log(f"    {pi+1}/10 prompts")

    # Aggregate per condition
    cond_agg = {}
    for cond_name, _, _ in conditions:
        cond_results = [r for r in results if r["condition"] == cond_name]
        token_means = {}
        for pt in proto_ids:
            vals = [r["token_deltas"].get(pt, 0) for r in cond_results]
            token_means[pt] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        cond_agg[cond_name] = token_means

    # Classify channels
    channel_roles = {}
    for ch in channels:
        ablate_key = f"ablate_ch{ch}"
        boost_key = f"boost_ch{ch}"
        loo_key = f"loo_ch{ch}"

        ablate_effects = cond_agg.get(ablate_key, {})
        boost_effects = cond_agg.get(boost_key, {})
        loo_effects = cond_agg.get(loo_key, {})

        # Compute statistics
        ablate_means = {pt: v["mean"] for pt, v in ablate_effects.items()}
        boost_means = {pt: v["mean"] for pt, v in boost_effects.items()}

        # All tokens decrease → general support
        all_negative = all(v < 0 for v in ablate_means.values())
        # Some tokens increase → suppressor
        any_positive = any(v > 0 for v in ablate_means.values())
        # Variance across tokens → token-specific
        ablate_vals = list(ablate_means.values())
        token_variance = float(np.std(ablate_vals)) if ablate_vals else 0

        # Mean effect
        mean_ablate = float(np.mean(ablate_vals)) if ablate_vals else 0
        mean_boost = float(np.mean(list(boost_means.values()))) if boost_means else 0
        mean_loo = float(np.mean([loo_effects.get(pt, {}).get("mean", 0) for pt in proto_ids]))

        # Classify
        if all_negative and token_variance < 0.3:
            role = "general_support"
        elif any_positive and token_variance > 0.3:
            role = "suppressor"
        elif token_variance > 0.5:
            role = "token_specific"
        else:
            role = "mixed"

        channel_roles[str(ch)] = {
            "role": role,
            "mean_ablate": mean_ablate,
            "mean_boost": mean_boost,
            "mean_loo": mean_loo,
            "token_variance": token_variance,
            "ablate_effects": ablate_means,
            "boost_effects": boost_means,
        }

    output = {
        "task": "task4_channel_roles",
        "model": model_name,
        "n_prompts": 10,
        "channels": channels,
        "channel_roles": channel_roles,
        "condition_results": cond_agg,
    }

    save_path = save_dir / "task4_roles.json"
    save_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"    Saved: {save_path}")

    # Print roles
    for ch, role_data in channel_roles.items():
        log(f"    Channel {ch}: role={role_data['role']}  "
            f"mean_ablate={role_data['mean_ablate']:+.3f}  "
            f"variance={role_data['token_variance']:.3f}")

    return output


# ============================================================
# MAIN
# ============================================================

def run_model(model_name: str) -> None:
    log(f"\n{'='*60}")
    log(f"Phase 955: {model_name}")
    log(f"{'='*60}")

    model_dir = RESULT_DIR / model_name
    ensure_dir(model_dir)

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    log(f"  {info.model_class}, {info.n_layers}L, d={info.d_model}")

    t_start = time.time()

    try:
        task1_3_odd_even(model, tokenizer, device, info, model_name, model_dir)
    except Exception as e:
        log(f"  Task 1-3 FAILED: {e}")
        import traceback; traceback.print_exc()

    try:
        task4_channel_roles(model, tokenizer, device, info, model_name, model_dir)
    except Exception as e:
        log(f"  Task 4 FAILED: {e}")
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
