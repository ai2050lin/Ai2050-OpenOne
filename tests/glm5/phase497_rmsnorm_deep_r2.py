"""
Phase 497 R2: RMSNorm效应深入 + 正确shared方向MLP因果干预
============================================================
修复R1的问题:
1. Exp2 shared方向用W_U SVD top component是错的，改用PCA
2. 加入D_pre和D_post的各组件贡献分解
3. 检查RMSNorm的方向vs范数分离效应

核心实验:
Exp1: 各组件(residual/attn/mlp)的D贡献分别在pre-norm和post-norm下测量
Exp2: 用PCA shared方向做MLP因果干预(ablate/double/reverse)
Exp3: RMSNorm的方向效应vs范数效应分离
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
    "fruit": {
        "objects": ["apple", "banana", "orange", "grape", "pear",
                    "peach", "mango", "plum", "cherry", "lemon"],
        "relation": "is a type of fruit",
        "target_tokens": ["fruit"],
    },
    "clothing": {
        "objects": ["shirt", "dress", "jacket", "pants", "coat",
                    "skirt", "sweater", "blouse", "scarf", "vest"],
        "relation": "is a type of clothing",
        "target_tokens": ["clothing"],
    },
    "emotion": {
        "objects": ["joy", "anger", "fear", "sadness", "surprise",
                    "disgust", "pride", "shame", "guilt", "envy"],
        "relation": "is a type of emotion",
        "target_tokens": ["emotion"],
    },
    "action": {
        "objects": ["run", "eat", "build", "throw", "buy",
                    "learn", "measure", "communicate", "swim", "write"],
        "relation": "is a type of action",
        "target_tokens": ["action"],
    },
    "animal": {
        "objects": ["dog", "cat", "horse", "elephant", "tiger",
                    "dolphin", "eagle", "snake", "rabbit", "whale"],
        "relation": "is a type of animal",
        "target_tokens": ["animal"],
    },
}

OUTPUT_DIR = Path("results/glm5")


def rmsnorm_numpy(x, weight=None, eps=1e-5):
    """RMSNorm in numpy"""
    rms = np.sqrt(np.mean(x ** 2) + eps)
    normed = x / rms
    if weight is not None:
        normed = normed * weight
    return normed


def compute_D(hidden_np, W_U, target_ids, comp_ids):
    """从hidden state计算DCF"""
    logits = hidden_np @ W_U.T
    target_logit = np.mean([logits[tid] for tid in target_ids if tid < len(logits)])
    comp_logits = [logits[cid] for cid in comp_ids if cid < len(logits)]
    if len(comp_logits) == 0:
        return 0.0
    return float(target_logit - np.mean(comp_logits))


def get_pca_shared_direction(model, tokenizer, device, layers, W_U, n_layer, categories):
    """
    用PCA从实际hidden states获取shared_semantic方向
    方法: 收集所有类别的hidden states，找跨类别共享的主方向
    """
    d_model = W_U.shape[1]
    all_hiddens = []

    for cat_name, cat_data in categories.items():
        for obj in cat_data["objects"][:5]:  # 每类5个对象
            prompt = f"A {obj} {cat_data['relation']}"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True)

            # 使用L(n-2)的hidden state (倒数第二层)
            h = out.hidden_states[n_layer - 1][0, -1, :].float().cpu().numpy()
            all_hiddens.append(h)

    all_hiddens = np.array(all_hiddens)  # [n_samples, d_model]
    # 中心化
    mean_h = np.mean(all_hiddens, axis=0)
    centered = all_hiddens - mean_h
    # PCA: 取第一个主成分 = 跨类别shared方向
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    shared_dir = Vt[0]  # [d_model]
    shared_dir = shared_dir / (np.linalg.norm(shared_dir) + 1e-10)

    # 验证: shared方向应该是跨类别一致的
    projs = all_hiddens @ shared_dir
    print(f"  PCA shared direction: samples={len(all_hiddens)}, "
          f"proj range=[{projs.min():.2f}, {projs.max():.2f}], "
          f"proj std={projs.std():.2f}")

    return shared_dir, mean_h


def run_phase497_r2(model_name: str, round_num: int = 2):
    """Phase 497 R2主实验"""
    print(f"\n{'='*70}")
    print(f"Phase 497 R{round_num}: RMSNorm Deep Analysis - {model_name}")
    print(f"{'='*70}")

    t0 = time.time()
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    W_U = get_W_U(model, model_name)
    n_layer = info.n_layers
    d_model = info.d_model
    last_layer = n_layer - 1

    print(f"  Model: {info.model_class}, {n_layer} layers, d={d_model}, t={time.time()-t0:.1f}s")

    # ===== 获取PCA shared方向 =====
    print(f"\n  Computing PCA shared direction...")
    shared_dir, mean_hidden = get_pca_shared_direction(
        model, tokenizer, device, layers, W_U, n_layer, CATEGORIES)

    # ===== 获取RMSNorm weight =====
    rmsnorm_w = None
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        try:
            w = model.model.norm.weight
            if w.device.type != 'meta':
                rmsnorm_w = w.detach().float().cpu().numpy()
                print(f"  RMSNorm weight: norm={np.linalg.norm(rmsnorm_w):.2f}")
        except NotImplementedError:
            print(f"  RMSNorm weight: meta device, unavailable")

    # ===== 准备token IDs =====
    all_cat_tokens = ["fruit", "clothing", "emotion", "action", "animal",
                      "vehicle", "container", "plant", "number", "color"]
    cat_token_ids = {}
    for tok in all_cat_tokens:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        if ids:
            cat_token_ids[tok] = ids[0]

    n_samples = 10
    exp1_results = {}  # 组件D贡献分解
    exp2_results = {}  # MLP shared因果
    exp3_results = {}  # RMSNorm方向vs范数

    for cat_name, cat_data in CATEGORIES.items():
        objects = cat_data["objects"][:n_samples]
        relation = cat_data["relation"]
        target_tokens = cat_data["target_tokens"]

        target_ids = [cat_token_ids[t] for t in target_tokens if t in cat_token_ids]
        if not target_ids:
            continue
        comp_ids = [v for k, v in cat_token_ids.items() if k != cat_name]

        cat_exp1 = []
        cat_exp2 = []
        cat_exp3 = []

        for obj in objects:
            prompt = f"A {obj} {relation}"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            # ===== 捕获所有中间状态 =====
            captured = {}

            def make_capture_hook(key, sub_module_name):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        captured[key] = output[0].detach().float().cpu().numpy()
                    else:
                        captured[key] = output.detach().float().cpu().numpy()
                return hook

            # Hook Attn和MLP输出
            hook_a = layers[last_layer].self_attn.register_forward_hook(
                make_capture_hook("attn_out", "attn"))
            hook_m = layers[last_layer].mlp.register_forward_hook(
                make_capture_hook("mlp_out", "mlp"))
            hook_l = layers[last_layer].register_forward_hook(
                make_capture_hook("layer_out", "layer"))

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True)

            hook_a.remove()
            hook_m.remove()
            hook_l.remove()

            # 提取关键hidden states
            # h_Ln2 = L(n-2)输出 = L(n-1)输入 (last token position)
            h_Ln2 = out.hidden_states[last_layer][0, -1, :].float().cpu().numpy()

            # h_Ln1_pre_norm = L(n-1)输出 (last token) - 这就是pre-RMSNorm
            if "layer_out" in captured:
                h_pre_norm = captured["layer_out"][0, -1, :]
            else:
                h_pre_norm = out.hidden_states[last_layer][0, -1, :].float().cpu().numpy()

            # h_Ln1_post_norm = 经过RMSNorm后
            h_post_norm = out.hidden_states[last_layer + 1][0, -1, :].float().cpu().numpy()

            # Attn和MLP输出
            attn_out = captured.get("attn_out", np.zeros((1, 1, d_model)))[0, -1, :]
            mlp_out = captured.get("mlp_out", np.zeros((1, 1, d_model)))[0, -1, :]

            # ===== Exp1: 各组件D贡献分解 =====
            # 在pre-norm空间
            D_full_pre = compute_D(h_pre_norm, W_U, target_ids, comp_ids)
            D_residual_pre = compute_D(h_Ln2, W_U, target_ids, comp_ids)
            D_attn_pre = compute_D(attn_out, W_U, target_ids, comp_ids)
            D_mlp_pre = compute_D(mlp_out, W_U, target_ids, comp_ids)

            # 在post-norm空间 (手动RMSNorm)
            if rmsnorm_w is not None:
                h_post_manual = rmsnorm_numpy(h_pre_norm, rmsnorm_w)
                D_full_post = compute_D(h_post_manual, W_U, target_ids, comp_ids)
            else:
                D_full_post = compute_D(h_post_norm, W_U, target_ids, comp_ids)

            # 验证: h_pre_norm ≈ h_Ln2 + attn_out + mlp_out
            residual_check = h_pre_norm - (h_Ln2 + attn_out + mlp_out)
            residual_error = float(np.linalg.norm(residual_check))

            # ===== Exp2: MLP Shared方向因果干预 =====
            # 分解MLP输出为shared和nonshared
            proj_shared = float(np.dot(mlp_out, shared_dir))
            mlp_shared = proj_shared * shared_dir
            mlp_nonshared = mlp_out - mlp_shared

            # 干预: 修改MLP输出的shared分量，然后手动构建完整的pre-norm hidden
            # 方法: h_modified = h_Ln2 + attn_out + mlp_modified
            # 然后手动RMSNorm

            # Baseline pre-norm with full MLP
            # h_baseline = h_Ln2 + attn_out + mlp_out = h_pre_norm

            # Ablate shared: 移除MLP中的shared分量
            mlp_ablate = mlp_nonshared  # mlp_out - mlp_shared
            h_ablate = h_Ln2 + attn_out + mlp_ablate
            D_ablate_pre = compute_D(h_ablate, W_U, target_ids, comp_ids)
            if rmsnorm_w is not None:
                D_ablate_post = compute_D(rmsnorm_numpy(h_ablate, rmsnorm_w), W_U, target_ids, comp_ids)
            else:
                D_ablate_post = D_ablate_pre  # fallback

            # Double shared: 加倍MLP中的shared分量
            mlp_double = mlp_out + mlp_shared
            h_double = h_Ln2 + attn_out + mlp_double
            D_double_pre = compute_D(h_double, W_U, target_ids, comp_ids)
            if rmsnorm_w is not None:
                D_double_post = compute_D(rmsnorm_numpy(h_double, rmsnorm_w), W_U, target_ids, comp_ids)
            else:
                D_double_post = D_double_pre

            # Reverse shared: 反转MLP中的shared分量
            mlp_reverse = mlp_out - 2 * mlp_shared
            h_reverse = h_Ln2 + attn_out + mlp_reverse
            D_reverse_pre = compute_D(h_reverse, W_U, target_ids, comp_ids)
            if rmsnorm_w is not None:
                D_reverse_post = compute_D(rmsnorm_numpy(h_reverse, rmsnorm_w), W_U, target_ids, comp_ids)
            else:
                D_reverse_post = D_reverse_pre

            # Zero MLP
            h_zeromlp = h_Ln2 + attn_out
            D_zeromlp_pre = compute_D(h_zeromlp, W_U, target_ids, comp_ids)
            if rmsnorm_w is not None:
                D_zeromlp_post = compute_D(rmsnorm_numpy(h_zeromlp, rmsnorm_w), W_U, target_ids, comp_ids)
            else:
                D_zeromlp_post = D_zeromlp_pre

            # ===== Exp3: RMSNorm方向vs范数分离 =====
            # 固定RMSNorm scale (用baseline的scale)
            if rmsnorm_w is not None:
                baseline_rms = np.sqrt(np.mean(h_pre_norm ** 2) + 1e-5)
                zeromlp_rms = np.sqrt(np.mean(h_zeromlp ** 2) + 1e-5)

                # 只改变方向，固定范数 (用baseline RMSNorm scale)
                h_fixed_scale = h_zeromlp / (np.sqrt(np.mean(h_zeromlp ** 2) + 1e-5)) * baseline_rms
                h_fixed_scale_normed = h_fixed_scale / baseline_rms * rmsnorm_w  # 等效: h_zeromlp方向 + baseline scale
                # 更简单: 用baseline的RMS值归一化zeromlp hidden
                D_fixed_scale = compute_D(
                    (h_zeromlp / zeromlp_rms) * rmsnorm_w, W_U, target_ids, comp_ids)

                # 只改变范数，固定方向 (用baseline方向)
                h_fixed_dir = h_pre_norm / np.sqrt(np.mean(h_pre_norm ** 2) + 1e-5)
                # 但用zeromlp的范数
                h_fixed_dir_scaled = h_fixed_dir * zeromlp_rms
                D_fixed_dir = compute_D(
                    rmsnorm_numpy(h_fixed_dir_scaled, rmsnorm_w), W_U, target_ids, comp_ids)

                exp3_data = {
                    "obj": obj,
                    "baseline_rms": float(baseline_rms),
                    "zeromlp_rms": float(zeromlp_rms),
                    "rms_change_ratio": float(zeromlp_rms / baseline_rms),
                    "D_full_post": D_full_post,
                    "D_zeromlp_post": D_zeromlp_post,
                    "D_fixed_scale_post": D_fixed_scale,
                    "D_fixed_dir_post": D_fixed_dir,
                    "scale_effect": D_fixed_scale - D_full_post,  # 只改scale的效应
                    "dir_effect": D_fixed_dir - D_full_post,  # 只改方向的效应
                }
                cat_exp3.append(exp3_data)

            # 记录Exp1
            cat_exp1.append({
                "obj": obj,
                "D_full_pre": D_full_pre,
                "D_full_post": D_full_post if 'D_full_post' in dir() else 0,
                "D_residual_pre": D_residual_pre,
                "D_attn_pre": D_attn_pre,
                "D_mlp_pre": D_mlp_pre,
                "D_zeromlp_pre": D_zeromlp_pre,
                "D_zeromlp_post": D_zeromlp_post,
                "residual_error": residual_error,
                "proj_shared_on_mlp": proj_shared,
                "mlp_norm": float(np.linalg.norm(mlp_out)),
                "mlp_shared_norm": float(np.linalg.norm(mlp_shared)),
                "mlp_nonshared_norm": float(np.linalg.norm(mlp_nonshared)),
            })

            # 记录Exp2
            cat_exp2.append({
                "obj": obj,
                "D_baseline_pre": D_full_pre,
                "D_baseline_post": D_full_post if 'D_full_post' in dir() else 0,
                "D_ablate_pre": D_ablate_pre,
                "D_ablate_post": D_ablate_post,
                "D_double_pre": D_double_pre,
                "D_double_post": D_double_post,
                "D_reverse_pre": D_reverse_pre,
                "D_reverse_post": D_reverse_post,
                "D_zeromlp_pre": D_zeromlp_pre,
                "D_zeromlp_post": D_zeromlp_post,
                "delta_ablate_pre": D_ablate_pre - D_full_pre,
                "delta_ablate_post": D_ablate_post - (D_full_post if 'D_full_post' in dir() else 0),
                "delta_double_pre": D_double_pre - D_full_pre,
                "delta_double_post": D_double_post - (D_full_post if 'D_full_post' in dir() else 0),
                "delta_reverse_pre": D_reverse_pre - D_full_pre,
                "delta_reverse_post": D_reverse_post - (D_full_post if 'D_full_post' in dir() else 0),
            })

            print(f"    {cat_name}/{obj}: D_pre={D_full_pre:.2f}, D_post={D_full_post:.2f}, "
                  f"Δ(ablate)_pre={D_ablate_pre-D_full_pre:.2f}, "
                  f"Δ(ablate)_post={D_ablate_post-(D_full_post):.2f}")

        # 汇总
        if cat_exp1:
            exp1_results[cat_name] = {
                "n_samples": len(cat_exp1),
                "mean_D_full_pre": float(np.mean([s["D_full_pre"] for s in cat_exp1])),
                "mean_D_residual_pre": float(np.mean([s["D_residual_pre"] for s in cat_exp1])),
                "mean_D_attn_pre": float(np.mean([s["D_attn_pre"] for s in cat_exp1])),
                "mean_D_mlp_pre": float(np.mean([s["D_mlp_pre"] for s in cat_exp1])),
                "mean_D_zeromlp_pre": float(np.mean([s["D_zeromlp_pre"] for s in cat_exp1])),
                "mean_D_zeromlp_post": float(np.mean([s["D_zeromlp_post"] for s in cat_exp1])),
                "mean_residual_error": float(np.mean([s["residual_error"] for s in cat_exp1])),
                "sample_details": cat_exp1,
            }

        if cat_exp2:
            exp2_results[cat_name] = {
                "n_samples": len(cat_exp2),
                "mean_delta_ablate_pre": float(np.nanmean([s["delta_ablate_pre"] for s in cat_exp2])),
                "mean_delta_ablate_post": float(np.nanmean([s["delta_ablate_post"] for s in cat_exp2])),
                "mean_delta_double_pre": float(np.nanmean([s["delta_double_pre"] for s in cat_exp2])),
                "mean_delta_double_post": float(np.nanmean([s["delta_double_post"] for s in cat_exp2])),
                "mean_delta_reverse_pre": float(np.nanmean([s["delta_reverse_pre"] for s in cat_exp2])),
                "mean_delta_reverse_post": float(np.nanmean([s["delta_reverse_post"] for s in cat_exp2])),
                "sample_details": cat_exp2,
            }

        if cat_exp3:
            exp3_results[cat_name] = {
                "n_samples": len(cat_exp3),
                "mean_rms_change_ratio": float(np.nanmean([s["rms_change_ratio"] for s in cat_exp3])),
                "mean_scale_effect": float(np.nanmean([s["scale_effect"] for s in cat_exp3])),
                "mean_dir_effect": float(np.nanmean([s["dir_effect"] for s in cat_exp3])),
                "sample_details": cat_exp3,
            }

        torch.cuda.empty_cache()

    # ===== 保存结果 =====
    results = {
        "phase": 497,
        "round": round_num,
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_info": {
            "class": info.model_class,
            "n_layers": n_layer,
            "d_model": d_model,
        },
        "shared_direction_method": "PCA_on_Ln2_hidden_states",
        "exp1_component_D_decomposition": exp1_results,
        "exp2_mlp_shared_causal": exp2_results,
        "exp3_rmsnorm_direction_vs_norm": exp3_results,
    }

    suffix = "deepseek7b" if model_name == "deepseek7b" else model_name
    out_path = OUTPUT_DIR / f"phase497_{suffix}_r{round_num}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {out_path}")

    # ===== 打印核心结果 =====
    print(f"\n{'='*70}")
    print(f"Phase 497 R{round_num} 核心结果: {model_name}")
    print(f"{'='*70}")

    print(f"\n--- Exp1: 组件D贡献(pre-norm) ---")
    print(f"{'Cat':>12} | {'D_res':>8} | {'D_attn':>8} | {'D_mlp':>8} | {'D_full':>8} | {'err':>8}")
    print("-" * 65)
    for cat, data in exp1_results.items():
        print(f"{cat:>12} | {data['mean_D_residual_pre']:>8.2f} | {data['mean_D_attn_pre']:>8.2f} | "
              f"{data['mean_D_mlp_pre']:>8.2f} | {data['mean_D_full_pre']:>8.2f} | "
              f"{data['mean_residual_error']:>8.2f}")

    if exp2_results:
        print(f"\n--- Exp2: MLP Shared因果干预 (pre-norm vs post-norm) ---")
        print(f"{'Cat':>12} | {'Δ(abl)_pre':>10} | {'Δ(abl)_post':>10} | "
              f"{'Δ(dbl)_pre':>10} | {'Δ(dbl)_post':>10} | {'Δ(rev)_pre':>10} | {'Δ(rev)_post':>10}")
        print("-" * 85)
        for cat, data in exp2_results.items():
            print(f"{cat:>12} | {data['mean_delta_ablate_pre']:>10.2f} | "
                  f"{data['mean_delta_ablate_post']:>10.2f} | "
                  f"{data['mean_delta_double_pre']:>10.2f} | "
                  f"{data['mean_delta_double_post']:>10.2f} | "
                  f"{data['mean_delta_reverse_pre']:>10.2f} | "
                  f"{data['mean_delta_reverse_post']:>10.2f}")

    if exp3_results:
        print(f"\n--- Exp3: RMSNorm方向vs范数效应 ---")
        print(f"{'Cat':>12} | {'rms_ratio':>10} | {'scale_eff':>10} | {'dir_eff':>10}")
        print("-" * 50)
        for cat, data in exp3_results.items():
            print(f"{cat:>12} | {data['mean_rms_change_ratio']:>10.3f} | "
                  f"{data['mean_scale_effect']:>10.2f} | {data['mean_dir_effect']:>10.2f}")

    # ===== 释放模型 =====
    release_model(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\n  Model released.")

    return results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    if model_name == "all":
        for mn in ["qwen3", "glm4", "deepseek7b"]:
            try:
                run_phase497_r2(mn, round_num)
            except Exception as e:
                print(f"!!! {mn} failed: {e}")
                import traceback; traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(3)
    else:
        run_phase497_r2(model_name, round_num)

    print("\nPhase 497 R2 complete!")
