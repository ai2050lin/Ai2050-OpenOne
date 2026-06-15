"""
Phase 497 R3: RMSNorm翻转机制的数学分析
=========================================
核心问题: 为什么MLP在pre-norm空间减少D，但经过RMSNorm后在post-norm空间增加D?

假设: RMSNorm是一个非线性映射，它改变了不同方向对D的相对贡献。
当MLP改变hidden state时:
1. MLP输出增加hidden state的范数
2. RMSNorm除以更大的范数
3. 这压缩了所有方向的贡献
4. 但不同方向的压缩率不同(非线性效应)
5. 最终导致D_post与D_pre方向相反

验证方法:
1. 计算RMSNorm对D的Jacobian(雅可比矩阵)
2. 检查MLP输出在不同方向上的范数和D贡献
3. 计算MLP对RMSNorm scale的贡献
4. 分解: MLP→范数变化→RMSNorm重缩放→D变化

简化实验:
- 只分析Qwen3 (有RMSNorm weight)
- 5个类别各5个样本
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc
import json
import time
import numpy as np
import torch
from pathlib import Path
from model_utils import (load_model, get_layers, get_model_info,
                          release_model, get_W_U, MODEL_CONFIGS)
from datetime import datetime

CATEGORIES = {
    "fruit": {"objects": ["apple", "banana", "orange", "grape", "pear"],
              "relation": "is a type of fruit", "target_tokens": ["fruit"]},
    "clothing": {"objects": ["shirt", "dress", "jacket", "pants", "coat"],
                 "relation": "is a type of clothing", "target_tokens": ["clothing"]},
    "emotion": {"objects": ["joy", "anger", "fear", "sadness", "surprise"],
                "relation": "is a type of emotion", "target_tokens": ["emotion"]},
    "action": {"objects": ["run", "eat", "build", "throw", "buy"],
               "relation": "is a type of action", "target_tokens": ["action"]},
    "animal": {"objects": ["dog", "cat", "horse", "elephant", "tiger"],
               "relation": "is a type of animal", "target_tokens": ["animal"]},
}

OUTPUT_DIR = Path("results/glm5")


def compute_D(h, W_U, target_ids, comp_ids):
    logits = h @ W_U.T
    t = np.mean([logits[i] for i in target_ids if i < len(logits)])
    c = np.mean([logits[i] for i in comp_ids if i < len(logits)])
    return float(t - c)


def rmsnorm_numpy(x, weight, eps=1e-5):
    rms = np.sqrt(np.mean(x ** 2) + eps)
    return x / rms * weight


def run_r3(model_name: str = "qwen3"):
    print(f"\n{'='*70}")
    print(f"Phase 497 R3: RMSNorm Flip Mechanism Analysis - {model_name}")
    print(f"{'='*70}")

    t0 = time.time()
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    W_U = get_W_U(model, model_name)
    n_layer = info.n_layers
    d_model = info.d_model
    last_layer = n_layer - 1

    # Get RMSNorm weight
    rmsnorm_w = None
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        try:
            w = model.model.norm.weight
            if w.device.type != 'meta':
                rmsnorm_w = w.detach().float().cpu().numpy()
        except NotImplementedError:
            pass

    if rmsnorm_w is None:
        print("  ERROR: RMSNorm weight not available, cannot proceed")
        release_model(model)
        return

    print(f"  Model: {info.model_class}, {n_layer}L, d={d_model}, RMSNorm norm={np.linalg.norm(rmsnorm_w):.2f}")

    # Token IDs
    all_tokens = ["fruit", "clothing", "emotion", "action", "animal",
                  "vehicle", "container", "plant", "number", "color"]
    cat_token_ids = {}
    for tok in all_tokens:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        if ids:
            cat_token_ids[tok] = ids[0]

    results = {}

    for cat_name, cat_data in CATEGORIES.items():
        target_ids = [cat_token_ids[t] for t in cat_data["target_tokens"] if t in cat_token_ids]
        if not target_ids:
            continue
        comp_ids = [v for k, v in cat_token_ids.items() if k != cat_name]

        cat_results = []

        for obj in cat_data["objects"]:
            prompt = f"A {obj} {cat_data['relation']}"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            # ===== Capture all intermediate states =====
            captured = {}

            def make_hook(key):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        captured[key] = output[0].detach().float().cpu().numpy()
                return hook

            hook_a = layers[last_layer].self_attn.register_forward_hook(make_hook("attn"))
            hook_m = layers[last_layer].mlp.register_forward_hook(make_hook("mlp"))

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True)

            hook_a.remove()
            hook_m.remove()

            # Key states
            h_Ln2 = out.hidden_states[last_layer][0, -1, :].float().cpu().numpy()
            h_post_norm = out.hidden_states[last_layer + 1][0, -1, :].float().cpu().numpy()

            attn_out = captured.get("attn", np.zeros((1, 1, d_model)))[0, -1, :]
            mlp_out = captured.get("mlp", np.zeros((1, 1, d_model)))[0, -1, :]

            # h_pre_norm = h_Ln2 + attn_out + mlp_out
            h_pre_norm = h_Ln2 + attn_out + mlp_out
            h_no_mlp = h_Ln2 + attn_out

            # ===== Core Analysis =====
            # 1. D in different spaces
            D_pre_full = compute_D(h_pre_norm, W_U, target_ids, comp_ids)
            D_pre_no_mlp = compute_D(h_no_mlp, W_U, target_ids, comp_ids)
            D_post_full = compute_D(rmsnorm_numpy(h_pre_norm, rmsnorm_w), W_U, target_ids, comp_ids)
            D_post_no_mlp = compute_D(rmsnorm_numpy(h_no_mlp, rmsnorm_w), W_U, target_ids, comp_ids)

            # 2. RMS values
            rms_full = np.sqrt(np.mean(h_pre_norm ** 2) + 1e-5)
            rms_no_mlp = np.sqrt(np.mean(h_no_mlp ** 2) + 1e-5)
            mlp_norm = np.linalg.norm(mlp_out)

            # 3. RMSNorm's Jacobian effect on D
            # D = <h_post, W_U[target]> - <h_post, W_U[comp]>
            # h_post = h_pre / rms * rmsnorm_w
            # dD/dh_pre = (1/rms) * rmsnorm_w * W_U[target]^T - (1/rms) * rmsnorm_w * W_U[comp]^T
            # But also dD/d(rms) is important!

            # The key: D_post depends on both the DIRECTION and the RMS of h_pre
            # D_post(h) = <h/rms(h) * w, W_U[target]-W_U[comp]>

            # W_target - W_comp (the "D readout direction")
            W_target = np.mean([W_U[tid] for tid in target_ids], axis=0)  # [d_model]
            W_comp = np.mean([W_U[cid] for cid in comp_ids if cid < W_U.shape[0]], axis=0)
            W_D = W_target - W_comp  # [d_model]

            # With RMSNorm weight: effective readout is rmsnorm_w * W_D
            effective_readout = rmsnorm_w * W_D  # [d_model]

            # D_pre = <h_pre, W_D> (without RMSNorm)
            # D_post = <h_pre/rms * rmsnorm_w, W_D> = <h_pre, rmsnorm_w * W_D> / rms
            #        = <h_pre, effective_readout> / rms

            # Verify
            D_pre_verify = float(np.dot(h_pre_norm, W_D))
            D_post_verify = float(np.dot(h_pre_norm, effective_readout) / rms_full)

            # D_pre_no_mlp verify
            D_pre_no_mlp_verify = float(np.dot(h_no_mlp, W_D))
            D_post_no_mlp_verify = float(np.dot(h_no_mlp, effective_readout) / rms_no_mlp)

            # 4. The critical decomposition
            # D_post_full = <h_pre, eff_readout> / rms_full
            # D_post_no_mlp = <h_no_mlp, eff_readout> / rms_no_mlp
            # = <h_pre - mlp_out, eff_readout> / rms_no_mlp

            # Effect of MLP on D_post:
            # ΔD_post = D_post_full - D_post_no_mlp
            #         = <h_pre, eff_readout>/rms_full - <h_no_mlp, eff_readout>/rms_no_mlp

            # Decompose into:
            # (a) Direction effect: change in <h, eff_readout> due to MLP
            # (b) Scale effect: change in 1/rms due to MLP

            # Let a_full = <h_pre, eff_readout>, a_no_mlp = <h_no_mlp, eff_readout>
            a_full = float(np.dot(h_pre_norm, effective_readout))
            a_no_mlp = float(np.dot(h_no_mlp, effective_readout))
            delta_a = a_full - a_no_mlp  # = <mlp_out, effective_readout>

            # Scale effect: if only rms changed but a stayed the same
            scale_only_D = a_no_mlp / rms_full  # what D would be if only rms changed
            delta_scale = scale_only_D - D_post_no_mlp  # scale effect

            # Direction effect: if only a changed but rms stayed the same
            dir_only_D = a_full / rms_no_mlp  # what D would be if only direction changed
            delta_dir = dir_only_D - D_post_no_mlp  # direction effect

            # Total should be approximately:
            # ΔD_post ≈ delta_scale + delta_dir (not exact due to interaction)

            delta_D_post = D_post_full - D_post_no_mlp
            interaction = delta_D_post - delta_scale - delta_dir

            # 5. MLP's contribution to effective readout direction
            mlp_readout = float(np.dot(mlp_out, effective_readout))
            mlp_readout_per_norm = mlp_readout / (mlp_norm + 1e-10)

            # 6. Key ratio: how much does MLP change the RMS?
            rms_ratio = rms_full / rms_no_mlp

            sample_data = {
                "obj": obj,
                "D_pre_full": D_pre_full,
                "D_pre_no_mlp": D_pre_no_mlp,
                "D_post_full": D_post_full,
                "D_post_no_mlp": D_post_no_mlp,
                "delta_D_pre": D_pre_full - D_pre_no_mlp,  # MLP pre-norm effect
                "delta_D_post": D_post_full - D_post_no_mlp,  # MLP post-norm effect
                "rms_full": rms_full,
                "rms_no_mlp": rms_no_mlp,
                "mlp_norm": mlp_norm,
                "rms_ratio": rms_ratio,
                "a_full": a_full,  # <h_pre, effective_readout>
                "a_no_mlp": a_no_mlp,  # <h_no_mlp, effective_readout>
                "delta_a": delta_a,  # <mlp_out, effective_readout>
                "scale_effect": delta_scale,
                "dir_effect": delta_dir,
                "interaction": interaction,
                "mlp_readout": mlp_readout,
                "mlp_readout_per_norm": mlp_readout_per_norm,
            }
            cat_results.append(sample_data)

            print(f"    {cat_name}/{obj}: ΔD_pre={D_pre_full-D_pre_no_mlp:.2f}, "
                  f"ΔD_post={D_post_full-D_post_no_mlp:.2f}, "
                  f"scale={delta_scale:.2f}, dir={delta_dir:.2f}, "
                  f"rms_ratio={rms_ratio:.4f}, mlp_readout={mlp_readout:.2f}")

        # Summary
        results[cat_name] = {
            "n_samples": len(cat_results),
            "mean_delta_D_pre": float(np.mean([s["delta_D_pre"] for s in cat_results])),
            "mean_delta_D_post": float(np.mean([s["delta_D_post"] for s in cat_results])),
            "mean_scale_effect": float(np.mean([s["scale_effect"] for s in cat_results])),
            "mean_dir_effect": float(np.mean([s["dir_effect"] for s in cat_results])),
            "mean_interaction": float(np.mean([s["interaction"] for s in cat_results])),
            "mean_rms_ratio": float(np.mean([s["rms_ratio"] for s in cat_results])),
            "mean_mlp_readout": float(np.mean([s["mlp_readout"] for s in cat_results])),
            "mean_delta_a": float(np.mean([s["delta_a"] for s in cat_results])),
            "sample_details": cat_results,
        }

        torch.cuda.empty_cache()

    # Save
    out_data = {
        "phase": 497,
        "round": 3,
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "analysis_type": "rmsnorm_flip_mechanism",
        "results": results,
    }

    out_path = OUTPUT_DIR / f"phase497_{model_name}_r3.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {out_path}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"Phase 497 R3 核心结果: {model_name}")
    print(f"{'='*70}")

    print(f"\n--- RMSNorm翻转机制分解 ---")
    print(f"{'Cat':>12} | {'ΔD_pre':>8} | {'ΔD_post':>8} | {'scale':>8} | {'dir':>8} | "
          f"{'inter':>8} | {'rms_r':>6} | {'mlp_rdot':>8}")
    print("-" * 85)
    for cat, data in results.items():
        print(f"{cat:>12} | {data['mean_delta_D_pre']:>8.2f} | "
              f"{data['mean_delta_D_post']:>8.2f} | "
              f"{data['mean_scale_effect']:>8.2f} | "
              f"{data['mean_dir_effect']:>8.2f} | "
              f"{data['mean_interaction']:>8.2f} | "
              f"{data['mean_rms_ratio']:>6.4f} | "
              f"{data['mean_mlp_readout']:>8.2f}")

    # Key insight
    print(f"\n--- 关键洞察 ---")
    for cat, data in results.items():
        pre = data['mean_delta_D_pre']
        post = data['mean_delta_D_post']
        scale = data['mean_scale_effect']
        dir_eff = data['mean_dir_effect']
        flip = "YES" if (pre > 0) != (post > 0) else "NO"
        dominant = "SCALE" if abs(scale) > abs(dir_eff) else "DIR"
        print(f"  {cat}: pre={pre:+.2f} → post={post:+.2f} flip={flip} "
              f"dominant={dominant} scale={scale:+.2f} dir={dir_eff:+.2f}")

    release_model(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_r3(model_name)
