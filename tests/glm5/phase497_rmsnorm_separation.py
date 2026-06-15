"""
Phase 497: Final RMSNorm Separation & MLP Shared Causal Closure
================================================================
核心目标:
1. 分离final RMSNorm的直接效应和间接放大效应
2. Qwen3 MLP shared方向因果闭环验证
3. GLM4保守机制定位 (Attn vs 残差流 vs RMSNorm)
4. DS7B可靠性验证 (比R1更细粒度)

Exp1: Final RMSNorm前后D对比
  - 在final RMSNorm前读raw hidden → 计算D_raw
  - 在final RMSNorm后读normed hidden → 计算D_normed
  - 零化MLP/Attn后分别测D_raw和D_normed
  - 计算RMSNorm放大因子

Exp2: Qwen3 MLP Shared因果闭环
  - 只消融MLP输出的shared分量 → D变化?
  - 只加倍MLP输出的shared分量 → D变化?
  - 只反转MLP输出的shared分量 → D变化?
  - 对比: 消融/加倍/反转后的D_raw vs D_normed

Exp3: GLM4保守机制来源
  - L(n-1)各组件对shared方向的贡献:
    residual_input, attn_output, mlp_output
  - 零化Attn后D_raw和D_normed
  - 零化MLP后D_raw和D_normed
  - 残差流自身的shared方向是否为刹车

Exp4: DS7B RMSNorm效应验证
  - 与Exp1同方法，验证DS7B的RMSNorm放大是否合理
  - 检查CPU offload是否导致hook数据不一致

Usage:
  python tests/glm5/phase497_rmsnorm_separation.py qwen3 1
  python tests/glm5/phase497_rmsnorm_separation.py glm4 1
  python tests/glm5/phase497_rmsnorm_separation.py deepseek7b 1
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

# ===== 语义类别定义 =====
CATEGORIES = {
    "fruit": {
        "objects": ["apple", "banana", "orange", "grape", "pear",
                    "peach", "mango", "plum"],
        "relation": "is a type of fruit",
        "target_tokens": ["fruit"],
    },
    "clothing": {
        "objects": ["shirt", "dress", "jacket", "pants", "coat",
                    "skirt", "sweater", "blouse"],
        "relation": "is a type of clothing",
        "target_tokens": ["clothing"],
    },
    "emotion": {
        "objects": ["joy", "anger", "fear", "sadness", "surprise",
                    "disgust", "pride", "shame"],
        "relation": "is a type of emotion",
        "target_tokens": ["emotion"],
    },
    "action": {
        "objects": ["run", "eat", "build", "throw", "buy",
                    "learn", "measure", "communicate"],
        "relation": "is a type of action",
        "target_tokens": ["action"],
    },
    "animal": {
        "objects": ["dog", "cat", "horse", "elephant", "tiger",
                    "dolphin", "eagle", "snake"],
        "relation": "is a type of animal",
        "target_tokens": ["animal"],
    },
}

OUTPUT_DIR = Path("results/glm5")


def get_shared_direction(model, model_name, layers, W_U_np, n_layer):
    """获取末层shared_semantic方向"""
    info = get_model_info(model, model_name)
    d_model = info.d_model

    # 使用最后3层中每层的所有token的hidden state均值方向
    # 这里用W_U的top singular vector作为更稳定的shared方向
    # 因为shared_semantic是跨类别共享的读出基底
    U, S, Vt = np.linalg.svd(W_U_np, full_matrices=False)
    # 第一个右奇异向量是最强shared方向
    shared_dir = Vt[0]  # [d_model]
    shared_dir = shared_dir / (np.linalg.norm(shared_dir) + 1e-10)
    return shared_dir


def get_target_token_id(tokenizer, token_str):
    """获取目标token的ID"""
    ids = tokenizer.encode(token_str, add_special_tokens=False)
    if len(ids) > 0:
        return ids[0]
    return None


def compute_D_from_hidden(hidden_np, W_U_np, target_ids, comp_ids):
    """
    从hidden state直接计算DCF (不经过RMSNorm)
    D = log P(target) - log P(mean_competitor)
    """
    logits = hidden_np @ W_U_np.T  # [vocab]

    target_logit = np.mean([logits[tid] for tid in target_ids if tid < len(logits)])
    comp_logits = [logits[cid] for cid in comp_ids if cid < len(logits)]
    if len(comp_logits) == 0:
        return 0.0
    comp_logit = np.mean(comp_logits)

    return float(target_logit - comp_logit)


def rmsnorm_numpy(x, eps=1e-5):
    """RMSNorm in numpy"""
    rms = np.sqrt(np.mean(x ** 2) + eps)
    return x / rms


def get_final_rmsnorm(model, model_name):
    """获取final RMSNorm模块"""
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        return model.model.norm
    if hasattr(model, 'model') and hasattr(model.model, 'final_layernorm'):
        return model.model.final_layernorm
    return None


def get_rmsnorm_weight(model, model_name):
    """获取final RMSNorm的weight"""
    norm_mod = get_final_rmsnorm(model, model_name)
    if norm_mod is not None and hasattr(norm_mod, 'weight'):
        try:
            w = norm_mod.weight
            if w.device.type != 'meta':
                return w.detach().float().cpu().numpy()
        except NotImplementedError:
            pass
    return None


def run_experiment(model_name: str, round_num: int):
    """Phase 497主实验"""
    print(f"\n{'='*70}")
    print(f"Phase 497 R{round_num}: Final RMSNorm Separation - {model_name}")
    print(f"{'='*70}")

    # ===== 加载模型 =====
    t0 = time.time()
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    W_U = get_W_U(model, model_name)
    print(f"  Model loaded: {info.model_class}, {info.n_layers} layers, "
          f"d_model={info.d_model}, load_time={time.time()-t0:.1f}s")

    n_layer = info.n_layers
    d_model = info.d_model
    last_layer = n_layer - 1

    # ===== 获取shared方向 =====
    shared_dir = get_shared_direction(model, model_name, layers, W_U, n_layer)
    print(f"  Shared direction: norm={np.linalg.norm(shared_dir):.4f}")

    # ===== 获取final RMSNorm weight =====
    rmsnorm_w = get_rmsnorm_weight(model, model_name)
    if rmsnorm_w is not None:
        print(f"  Final RMSNorm weight: shape={rmsnorm_w.shape}, norm={np.linalg.norm(rmsnorm_w):.4f}")
    else:
        print(f"  Final RMSNorm weight: not accessible (meta device)")

    # ===== 准备竞争类别token IDs =====
    all_category_tokens = ["fruit", "clothing", "emotion", "action", "animal",
                           "vehicle", "container", "plant", "number", "color"]
    cat_token_ids = {}
    for tok in all_category_tokens:
        tid = get_target_token_id(tokenizer, tok)
        if tid is not None:
            cat_token_ids[tok] = tid

    # ===== Exp1: Final RMSNorm前后D对比 =====
    print(f"\n--- Exp1: Final RMSNorm Separation ---")

    # 我们需要在L(n-1)结束后、final RMSNorm前/后分别捕获hidden state
    # 方法: hook捕获L(n-1)输出 (pre-final-norm), 然后手动应用RMSNorm

    n_samples_per_cat = 10 if round_num == 1 else 15

    exp1_results = {}
    exp2_results = {}
    exp3_results = {}

    for cat_name, cat_data in CATEGORIES.items():
        objects = cat_data["objects"][:n_samples_per_cat]
        relation = cat_data["relation"]
        target_tokens = cat_data["target_tokens"]

        # 目标token IDs
        target_ids = [cat_token_ids[t] for t in target_tokens if t in cat_token_ids]
        if not target_ids:
            print(f"  Skip {cat_name}: no target token IDs")
            continue

        # 竞争token IDs (其他类别)
        comp_ids = [v for k, v in cat_token_ids.items() if k != cat_name]

        cat_exp1 = []
        cat_exp2 = []
        cat_exp3 = []

        for obj in objects:
            prompt = f"A {obj} {relation}"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            n_tokens = input_ids.shape[1]

            # ===== 基线: 正常forward, 捕获多位置hidden states =====
            # Hook: 捕获L(n-1)输出 (pre-final-norm) 和 final RMSNorm后
            captured_raw = {}  # pre-RMSNorm hidden states
            captured_normed = {}  # post-RMSNorm hidden states (from model output)

            def make_pre_norm_hook(key):
                def hook(module, input, output):
                    # L(n-1)的输出 = pre-RMSNorm的输入
                    if isinstance(output, tuple):
                        captured_raw[key] = output[0].detach().float().cpu().numpy()
                    else:
                        captured_raw[key] = output.detach().float().cpu().numpy()
                return hook

            # 注册hook在最后一层
            hook_last = layers[last_layer].register_forward_hook(
                make_pre_norm_hook("last_layer_out"))

            # 正常forward
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True)
            hook_last.remove()

            # post-RMSNorm: 从model output的hidden_states获取
            hs_last = out.hidden_states[last_layer + 1]  # +1因为包含了embedding层
            captured_normed["last_layer_out"] = hs_last[0, -1, :].float().cpu().numpy()

            # pre-RMSNorm: L(n-1)的输出 (最后一个token)
            if "last_layer_out" in captured_raw:
                h_pre_norm = captured_raw["last_layer_out"][0, -1, :]  # [d_model]
            else:
                print(f"  Warning: no raw capture for {obj}, using hidden_states")
                h_pre_norm = out.hidden_states[last_layer][0, -1, :].float().cpu().numpy()

            h_post_norm = captured_normed["last_layer_out"]

            # 手动RMSNorm验证
            if rmsnorm_w is not None:
                h_manual_norm = rmsnorm_numpy(h_pre_norm) * rmsnorm_w
                norm_diff = np.linalg.norm(h_manual_norm - h_post_norm)
            else:
                h_manual_norm = None
                norm_diff = None

            # 计算D: pre-RMSNorm vs post-RMSNorm
            D_pre_norm = compute_D_from_hidden(h_pre_norm, W_U, target_ids, comp_ids)
            D_post_norm = compute_D_from_hidden(h_post_norm, W_U, target_ids, comp_ids)

            # RMSNorm放大因子
            rmsnorm_gain = D_post_norm - D_pre_norm
            if abs(D_pre_norm) > 0.1:
                rmsnorm_ratio = D_post_norm / D_pre_norm
            else:
                rmsnorm_ratio = float('inf')

            # Shared方向分析
            proj_shared_pre = float(np.dot(h_pre_norm, shared_dir))
            proj_shared_post = float(np.dot(h_post_norm, shared_dir))

            # ===== 零化MLP后 =====
            # Hook: 零化MLP输出
            captured_raw_zeromlp = {}

            def make_zero_mlp_hook():
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        return (torch.zeros_like(output[0]),) + output[1:]
                    return torch.zeros_like(output)
                return hook

            def make_pre_norm_hook2(key):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        captured_raw_zeromlp[key] = output[0].detach().float().cpu().numpy()
                    else:
                        captured_raw_zeromlp[key] = output.detach().float().cpu().numpy()
                return hook

            hook_zero_mlp = layers[last_layer].mlp.register_forward_hook(make_zero_mlp_hook())
            hook_last2 = layers[last_layer].register_forward_hook(make_pre_norm_hook2("out"))

            with torch.no_grad():
                out_zeromlp = model(input_ids=input_ids, attention_mask=attention_mask,
                                   output_hidden_states=True)
            hook_zero_mlp.remove()
            hook_last2.remove()

            hs_zeromlp = out_zeromlp.hidden_states[last_layer + 1]
            h_zeromlp_post = hs_zeromlp[0, -1, :].float().cpu().numpy()

            if "out" in captured_raw_zeromlp:
                h_zeromlp_pre = captured_raw_zeromlp["out"][0, -1, :]
            else:
                h_zeromlp_pre = out_zeromlp.hidden_states[last_layer][0, -1, :].float().cpu().numpy()

            D_zeromlp_pre = compute_D_from_hidden(h_zeromlp_pre, W_U, target_ids, comp_ids)
            D_zeromlp_post = compute_D_from_hidden(h_zeromlp_post, W_U, target_ids, comp_ids)

            # ===== 零化Attn后 =====
            captured_raw_zeroattn = {}

            def make_zero_attn_hook():
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        return (torch.zeros_like(output[0]),) + output[1:]
                    return torch.zeros_like(output)
                return hook

            def make_pre_norm_hook3(key):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        captured_raw_zeroattn[key] = output[0].detach().float().cpu().numpy()
                    else:
                        captured_raw_zeroattn[key] = output.detach().float().cpu().numpy()
                return hook

            hook_zero_attn = layers[last_layer].self_attn.register_forward_hook(make_zero_attn_hook())
            hook_last3 = layers[last_layer].register_forward_hook(make_pre_norm_hook3("out"))

            with torch.no_grad():
                out_zeroattn = model(input_ids=input_ids, attention_mask=attention_mask,
                                    output_hidden_states=True)
            hook_zero_attn.remove()
            hook_last3.remove()

            hs_zeroattn = out_zeroattn.hidden_states[last_layer + 1]
            h_zeroattn_post = hs_zeroattn[0, -1, :].float().cpu().numpy()

            if "out" in captured_raw_zeroattn:
                h_zeroattn_pre = captured_raw_zeroattn["out"][0, -1, :]
            else:
                h_zeroattn_pre = out_zeroattn.hidden_states[last_layer][0, -1, :].float().cpu().numpy()

            D_zeroattn_pre = compute_D_from_hidden(h_zeroattn_pre, W_U, target_ids, comp_ids)
            D_zeroattn_post = compute_D_from_hidden(h_zeroattn_post, W_U, target_ids, comp_ids)

            # ===== Exp2: MLP Shared分量因果干预 (仅Qwen3/DS7B) =====
            # 在L(n-1)的MLP输出上分解shared/nonshared，然后干预shared分量
            exp2_data = None
            if model_name in ("qwen3", "deepseek7b"):
                # 获取MLP输出
                mlp_out_captured = {}

                def make_mlp_capture_hook():
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            mlp_out_captured["mlp_out"] = output[0].detach().float().cpu()
                        else:
                            mlp_out_captured["mlp_out"] = output.detach().float().cpu()
                    return hook

                hook_mlp_cap = layers[last_layer].mlp.register_forward_hook(make_mlp_capture_hook())

                with torch.no_grad():
                    out_base2 = model(input_ids=input_ids, attention_mask=attention_mask,
                                     output_hidden_states=True)
                hook_mlp_cap.remove()

                if "mlp_out" in mlp_out_captured:
                    mlp_out = mlp_out_captured["mlp_out"]  # [1, n_tokens, d_model]
                    mlp_last = mlp_out[0, -1, :].numpy()  # [d_model]

                    # 分解shared/nonshared
                    proj_s = np.dot(mlp_last, shared_dir)
                    mlp_shared = proj_s * shared_dir  # shared分量
                    mlp_nonshared = mlp_last - mlp_shared  # nonshared分量

                    # 干预: 消融shared分量
                    def make_ablate_shared_hook():
                        def hook(module, input, output):
                            if isinstance(output, tuple):
                                o = output[0]
                                proj = torch.matmul(o, torch.tensor(shared_dir, dtype=o.dtype, device=o.device))
                                shared_comp = proj.unsqueeze(-1) * torch.tensor(shared_dir, dtype=o.dtype, device=o.device)
                                return (o - shared_comp,) + output[1:]
                            return output
                        return hook

                    # 消融shared
                    hook_as = layers[last_layer].mlp.register_forward_hook(make_ablate_shared_hook())
                    with torch.no_grad():
                        out_ablate = model(input_ids=input_ids, attention_mask=attention_mask,
                                          output_hidden_states=True)
                    hook_as.remove()
                    D_ablate_shared = compute_D_from_hidden(
                        out_ablate.hidden_states[last_layer+1][0,-1,:].float().cpu().numpy(),
                        W_U, target_ids, comp_ids)

                    # 加倍shared
                    def make_double_shared_hook():
                        def hook(module, input, output):
                            if isinstance(output, tuple):
                                o = output[0]
                                proj = torch.matmul(o, torch.tensor(shared_dir, dtype=o.dtype, device=o.device))
                                shared_comp = proj.unsqueeze(-1) * torch.tensor(shared_dir, dtype=o.dtype, device=o.device)
                                return (o + shared_comp,) + output[1:]
                            return output
                        return hook

                    hook_ds = layers[last_layer].mlp.register_forward_hook(make_double_shared_hook())
                    with torch.no_grad():
                        out_double = model(input_ids=input_ids, attention_mask=attention_mask,
                                          output_hidden_states=True)
                    hook_ds.remove()
                    D_double_shared = compute_D_from_hidden(
                        out_double.hidden_states[last_layer+1][0,-1,:].float().cpu().numpy(),
                        W_U, target_ids, comp_ids)

                    # 反转shared
                    def make_reverse_shared_hook():
                        def hook(module, input, output):
                            if isinstance(output, tuple):
                                o = output[0]
                                proj = torch.matmul(o, torch.tensor(shared_dir, dtype=o.dtype, device=o.device))
                                shared_comp = proj.unsqueeze(-1) * torch.tensor(shared_dir, dtype=o.dtype, device=o.device)
                                return (o - 2*shared_comp,) + output[1:]
                            return output
                        return hook

                    hook_rs = layers[last_layer].mlp.register_forward_hook(make_reverse_shared_hook())
                    with torch.no_grad():
                        out_reverse = model(input_ids=input_ids, attention_mask=attention_mask,
                                           output_hidden_states=True)
                    hook_rs.remove()
                    D_reverse_shared = compute_D_from_hidden(
                        out_reverse.hidden_states[last_layer+1][0,-1,:].float().cpu().numpy(),
                        W_U, target_ids, comp_ids)

                    exp2_data = {
                        "proj_shared_on_mlp": float(proj_s),
                        "mlp_shared_norm": float(np.linalg.norm(mlp_shared)),
                        "mlp_nonshared_norm": float(np.linalg.norm(mlp_nonshared)),
                        "D_baseline": D_post_norm,
                        "D_ablate_shared": D_ablate_shared,
                        "D_double_shared": D_double_shared,
                        "D_reverse_shared": D_reverse_shared,
                        "delta_ablate": D_ablate_shared - D_post_norm,
                        "delta_double": D_double_shared - D_post_norm,
                        "delta_reverse": D_reverse_shared - D_post_norm,
                    }

            # ===== Exp3: 组件shared方向贡献分解 =====
            # residual_input = h_{L-2} (last token)
            # attn_output = captured from L(n-1) attn
            # mlp_output = captured from L(n-1) mlp

            # 获取L(n-2)的输出 (即L(n-1)的输入)
            h_Ln2 = out.hidden_states[last_layer][0, -1, :].float().cpu().numpy()

            # 捕获Attn和MLP输出
            attn_out_cap = {}
            mlp_out_cap2 = {}

            def make_attn_cap_hook():
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        attn_out_cap["attn"] = output[0].detach().float().cpu().numpy()
                return hook

            def make_mlp_cap_hook2():
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        mlp_out_cap2["mlp"] = output[0].detach().float().cpu().numpy()
                return hook

            hook_a = layers[last_layer].self_attn.register_forward_hook(make_attn_cap_hook())
            hook_m = layers[last_layer].mlp.register_forward_hook(make_mlp_cap_hook2())

            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            hook_a.remove()
            hook_m.remove()

            # 各组件对shared方向的贡献
            residual_shared = float(np.dot(h_Ln2, shared_dir))
            attn_shared = float(np.dot(attn_out_cap.get("attn", np.zeros((1,1,d_model)))[0,-1,:], shared_dir)) if "attn" in attn_out_cap else 0.0
            mlp_shared = float(np.dot(mlp_out_cap2.get("mlp", np.zeros((1,1,d_model)))[0,-1,:], shared_dir)) if "mlp" in mlp_out_cap2 else 0.0

            total_shared = residual_shared + attn_shared + mlp_shared
            post_norm_shared = float(np.dot(h_post_norm, shared_dir))

            exp3_data = {
                "residual_shared_proj": residual_shared,
                "attn_shared_proj": attn_shared,
                "mlp_shared_proj": mlp_shared,
                "sum_shared_proj": total_shared,
                "post_norm_shared_proj": post_norm_shared,
                "residual_frac": residual_shared / (abs(total_shared) + 1e-10),
                "attn_frac": attn_shared / (abs(total_shared) + 1e-10),
                "mlp_frac": mlp_shared / (abs(total_shared) + 1e-10),
            }

            # 记录Exp1数据
            sample_exp1 = {
                "obj": obj,
                "D_pre_norm": D_pre_norm,
                "D_post_norm": D_post_norm,
                "rmsnorm_gain": rmsnorm_gain,
                "rmsnorm_ratio": rmsnorm_ratio,
                "proj_shared_pre": proj_shared_pre,
                "proj_shared_post": proj_shared_post,
                "D_zeromlp_pre": D_zeromlp_pre,
                "D_zeromlp_post": D_zeromlp_post,
                "delta_D_zeromlp_pre": D_zeromlp_pre - D_pre_norm,
                "delta_D_zeromlp_post": D_zeromlp_post - D_post_norm,
                "D_zeroattn_pre": D_zeroattn_pre,
                "D_zeroattn_post": D_zeroattn_post,
                "delta_D_zeroattn_pre": D_zeroattn_pre - D_pre_norm,
                "delta_D_zeroattn_post": D_zeroattn_post - D_post_norm,
                "pre_norm_diff": D_zeromlp_pre - D_pre_norm,  # MLP直接效应(pre-norm)
                "post_norm_diff": D_zeromlp_post - D_post_norm,  # MLP总效应(post-norm)
                "rmsnorm_amplification": (D_zeromlp_post - D_post_norm) / (D_zeromlp_pre - D_pre_norm + 1e-10),
            }

            if norm_diff is not None:
                sample_exp1["rmsnorm_verify_diff"] = float(norm_diff)

            cat_exp1.append(sample_exp1)
            if exp2_data:
                exp2_data["obj"] = obj
                cat_exp2.append(exp2_data)

            exp3_data["obj"] = obj
            cat_exp3.append(exp3_data)

            print(f"    {cat_name}/{obj}: D_pre={D_pre_norm:.2f}, D_post={D_post_norm:.2f}, "
                  f"gain={rmsnorm_gain:.2f}, ΔD(zeromlp)_pre={D_zeromlp_pre-D_pre_norm:.2f}, "
                  f"ΔD(zeromlp)_post={D_zeromlp_post-D_post_norm:.2f}")

        # 汇总
        if cat_exp1:
            exp1_results[cat_name] = {
                "n_samples": len(cat_exp1),
                "mean_D_pre_norm": float(np.mean([s["D_pre_norm"] for s in cat_exp1])),
                "mean_D_post_norm": float(np.mean([s["D_post_norm"] for s in cat_exp1])),
                "mean_rmsnorm_gain": float(np.mean([s["rmsnorm_gain"] for s in cat_exp1])),
                "mean_rmsnorm_ratio": float(np.mean([s["rmsnorm_ratio"] for s in cat_exp1 if s["rmsnorm_ratio"] != float('inf')])),
                "mean_delta_zeromlp_pre": float(np.mean([s["delta_D_zeromlp_pre"] for s in cat_exp1])),
                "mean_delta_zeromlp_post": float(np.mean([s["delta_D_zeromlp_post"] for s in cat_exp1])),
                "mean_delta_zeroattn_pre": float(np.mean([s["delta_D_zeroattn_pre"] for s in cat_exp1])),
                "mean_delta_zeroattn_post": float(np.mean([s["delta_D_zeroattn_post"] for s in cat_exp1])),
                "mean_rmsnorm_amplification": float(np.mean([s["rmsnorm_amplification"] for s in cat_exp1
                                                              if abs(s["pre_norm_diff"]) > 0.01])),
                "sample_details": cat_exp1,
            }

        if cat_exp2:
            exp2_results[cat_name] = {
                "n_samples": len(cat_exp2),
                "mean_delta_ablate": float(np.mean([s["delta_ablate"] for s in cat_exp2])),
                "mean_delta_double": float(np.mean([s["delta_double"] for s in cat_exp2])),
                "mean_delta_reverse": float(np.mean([s["delta_reverse"] for s in cat_exp2])),
                "mean_proj_shared_on_mlp": float(np.mean([s["proj_shared_on_mlp"] for s in cat_exp2])),
                "sample_details": cat_exp2,
            }

        if cat_exp3:
            exp3_results[cat_name] = {
                "n_samples": len(cat_exp3),
                "mean_residual_shared": float(np.mean([s["residual_shared_proj"] for s in cat_exp3])),
                "mean_attn_shared": float(np.mean([s["attn_shared_proj"] for s in cat_exp3])),
                "mean_mlp_shared": float(np.mean([s["mlp_shared_proj"] for s in cat_exp3])),
                "mean_residual_frac": float(np.mean([s["residual_frac"] for s in cat_exp3])),
                "mean_attn_frac": float(np.mean([s["attn_frac"] for s in cat_exp3])),
                "mean_mlp_frac": float(np.mean([s["mlp_frac"] for s in cat_exp3])),
                "sample_details": cat_exp3,
            }

        # 清理GPU
        torch.cuda.empty_cache()

    # ===== 保存结果 =====
    results = {
        "phase": 497,
        "round": round_num,
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_info": {
            "class": info.model_class,
            "n_layers": info.n_layers,
            "d_model": info.d_model,
        },
        "exp1_rmsnorm_separation": exp1_results,
        "exp2_mlp_shared_causal": exp2_results,
        "exp3_component_shared_decomposition": exp3_results,
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

    print(f"\n--- Exp1: RMSNorm分离 ---")
    print(f"{'Cat':>12} | {'D_pre':>8} | {'D_post':>8} | {'gain':>8} | "
          f"{'Δ(0MLP)_pre':>12} | {'Δ(0MLP)_post':>12} | {'amp':>6}")
    print("-" * 80)
    for cat, data in exp1_results.items():
        amp = data.get("mean_rmsnorm_amplification", 0)
        print(f"{cat:>12} | {data['mean_D_pre_norm']:>8.2f} | {data['mean_D_post_norm']:>8.2f} | "
              f"{data['mean_rmsnorm_gain']:>8.2f} | {data['mean_delta_zeromlp_pre']:>12.2f} | "
              f"{data['mean_delta_zeromlp_post']:>12.2f} | {amp:>6.1f}")

    if exp2_results:
        print(f"\n--- Exp2: MLP Shared因果闭环 ---")
        print(f"{'Cat':>12} | {'Δ(ablate)':>10} | {'Δ(double)':>10} | {'Δ(reverse)':>10} | {'proj_s':>8}")
        print("-" * 60)
        for cat, data in exp2_results.items():
            print(f"{cat:>12} | {data['mean_delta_ablate']:>10.2f} | "
                  f"{data['mean_delta_double']:>10.2f} | "
                  f"{data['mean_delta_reverse']:>10.2f} | "
                  f"{data['mean_proj_shared_on_mlp']:>8.2f}")

    if exp3_results:
        print(f"\n--- Exp3: 组件shared方向贡献 ---")
        print(f"{'Cat':>12} | {'residual%':>10} | {'attn%':>10} | {'mlp%':>10}")
        print("-" * 50)
        for cat, data in exp3_results.items():
            print(f"{cat:>12} | {data['mean_residual_frac']:>10.2f} | "
                  f"{data['mean_attn_frac']:>10.2f} | "
                  f"{data['mean_mlp_frac']:>10.2f}")

    # ===== 释放模型 =====
    release_model(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\n  Model released, GPU cleared.")

    return results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    if model_name == "all":
        for mn in ["qwen3", "glm4", "deepseek7b"]:
            try:
                run_experiment(mn, round_num)
            except Exception as e:
                print(f"!!! {mn} failed: {e}")
                import traceback; traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(3)
    else:
        run_experiment(model_name, round_num)

    print("\nPhase 497 R1 complete!")
