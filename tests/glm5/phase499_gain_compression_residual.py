"""
Phase 499: Gain门控维度结构、目标-竞争重排与残差语义主轴闭环
================================================================
核心目标:
1. Exp1: Gain维度结构 - gain_eff类别差异来自哪些维度?
2. Exp2: 目标-竞争项不等比例压缩 - 证明DCF翻转来自target/competitor相对压缩
3. Exp3: 残差流语义主载体验证 - 消融/加倍/反转残差验证
4. Exp4: MLP抑制方向机制 - MLP输出在w_D/g⊙w_D/residual上的投影
5. Exp5: 动作类专项 - 扩大动作集合验证竞争项压缩机制

关键数学:
  D_post = <h_pre, g⊙w_D> / rms(h_pre)
  gain_eff(c) = D_with_gain(c) - D_no_gain(c)
  compression_rate = log(|post| / |pre|)

Usage:
  python tests/glm5/phase499_gain_compression_residual.py qwen3 1
  python tests/glm5/phase499_gain_compression_residual.py glm4 1
  python tests/glm5/phase499_gain_compression_residual.py deepseek7b 1
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
from datetime import datetime
from model_utils import (get_layers, get_model_info, get_W_U, MODEL_CONFIGS)


CATEGORIES = {
    "fruit": {
        "objects": ["apple", "banana", "orange", "grape", "pear",
                    "peach", "mango", "plum", "cherry", "lemon"],
        "relation": "is a type of fruit",
        "target_tokens": ["fruit"],
        "competitor_tokens": ["food", "plant", "vegetable", "crop", "thing"],
    },
    "clothing": {
        "objects": ["shirt", "dress", "jacket", "pants", "coat",
                    "skirt", "sweater", "blouse", "scarf", "vest"],
        "relation": "is a type of clothing",
        "target_tokens": ["clothing"],
        "competitor_tokens": ["garment", "fabric", "apparel", "outfit", "attire"],
    },
    "emotion": {
        "objects": ["joy", "anger", "fear", "sadness", "surprise",
                    "disgust", "pride", "shame", "guilt", "envy"],
        "relation": "is a type of emotion",
        "target_tokens": ["emotion"],
        "competitor_tokens": ["feeling", "sentiment", "mood", "sensation", "thought"],
    },
    "action": {
        "objects": ["run", "eat", "build", "throw", "buy",
                    "learn", "speak", "write", "move", "swim"],
        "relation": "is a type of action",
        "target_tokens": ["action"],
        "competitor_tokens": ["activity", "movement", "process", "task", "event"],
    },
    "animal": {
        "objects": ["dog", "cat", "horse", "elephant", "tiger",
                    "dolphin", "eagle", "snake", "rabbit", "whale"],
        "relation": "is a type of animal",
        "target_tokens": ["animal"],
        "competitor_tokens": ["creature", "beast", "mammal", "species", "pet"],
    },
}

# Exp5: 扩展动作集合
ACTION_EXTENDED = {
    "action_ext": {
        "objects": ["run", "eat", "build", "throw", "buy",
                    "learn", "speak", "write", "move", "swim",
                    "read", "cook", "drive", "jump", "sing",
                    "draw", "teach", "fight", "grow", "fly"],
        "relation": "is a type of action",
        "target_tokens": ["action"],
    }
}

OUTPUT_DIR = Path("results/glm5")


def load_rmsnorm_weight(model, model_name):
    """加载final RMSNorm weight, 支持meta device和safetensors"""
    # 方法1: 直接从模型读取
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        norm = model.model.norm
        if hasattr(norm, 'weight') and not norm.weight.is_meta:
            return norm.weight.detach().cpu().float().numpy()

    # 方法2: 从safetensors读取
    cfg = MODEL_CONFIGS[model_name]
    model_path = Path(cfg["path"])
    safetensors_files = sorted(model_path.glob("*.safetensors"))
    if safetensors_files:
        from safetensors import safe_open
        for sf_file in safetensors_files:
            try:
                with safe_open(sf_file, framework='pt', device='cpu') as sf:
                    for key in sf.keys():
                        if 'norm' in key and key.endswith('.weight') and 'layers' not in key:
                            w = sf.get_tensor(key)
                            if len(w.shape) == 1:
                                print(f"  [RMSNorm] Loaded from {sf_file.name}: {key}, shape={w.shape}")
                                return w.float().numpy()
            except Exception as e:
                continue

    print(f"  [RMSNorm] WARNING: Could not load RMSNorm weight for {model_name}")
    return None


def rmsnorm_numpy(x, weight=None, eps=1e-5):
    """RMSNorm in numpy: x / rms(x) * weight"""
    rms = np.sqrt(np.mean(x ** 2) + eps)
    normed = x / rms
    if weight is not None:
        normed = normed * weight
    return normed


def compute_dcf(h, W_U, target_ids, competitor_ids):
    """计算DCF: D = mean(target_logits) - mean(competitor_logits)"""
    target_logits = [float(np.dot(h, W_U[tid])) for tid in target_ids if tid < W_U.shape[0]]
    comp_logits = [float(np.dot(h, W_U[cid])) for cid in competitor_ids if cid < W_U.shape[0]]
    if not target_logits or not comp_logits:
        return 0.0, 0.0, 0.0, [], []
    D = float(np.mean(target_logits) - np.mean(comp_logits))
    return D, float(np.mean(target_logits)), float(np.mean(comp_logits)), target_logits, comp_logits


def find_top_competitor_ids(W_U, target_ids, h_pre, n_comp=5):
    """找到对给定h_pre最强的n_comp个非target token ids"""
    all_logits = W_U @ h_pre  # [vocab_size]
    # 排除target ids
    for tid in target_ids:
        if tid < len(all_logits):
            all_logits[tid] = -1e10
    top_ids = np.argsort(all_logits)[-n_comp:][::-1]
    return [int(i) for i in top_ids]


def load_model_bf16(model_name):
    """BF16 + device_map=auto 加载模型"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = MODEL_CONFIGS[model_name]
    print(f"[Phase499] Loading {model_name} (bf16 + device_map=auto)...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 尝试flash_attention_2, 失败则回退sdpa
    for attn_impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"],
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
                attn_implementation=attn_impl,
            )
            print(f"[Phase499] {model_name} loaded with attn_impl={attn_impl}")
            break
        except Exception as e:
            print(f"[Phase499] attn_impl={attn_impl} failed: {e}, trying next...")
            continue
    else:
        raise RuntimeError(f"Failed to load {model_name} with any attn_impl")

    model.eval()
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"[Phase499] {model_name}: device={device}, class={type(model).__name__}, GPU={gpu_mem:.2f}GB")
    return model, tokenizer, device


def get_last_layer_outputs(model, tokenizer, device, prompt, n_layers):
    """用hook捕获末层的residual_input, attn_output, mlp_output"""
    layers = get_layers(model)
    last_layer = layers[n_layers - 1]

    captured = {}

    def hook_fn(key):
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured[key] = output[0].detach().float().cpu().numpy()
            else:
                captured[key] = output.detach().float().cpu().numpy()
        return hook

    # Hook末层self_attn和mlp
    hooks = []
    hooks.append(last_layer.self_attn.register_forward_hook(hook_fn("attn_output")))
    hooks.append(last_layer.mlp.register_forward_hook(hook_fn("mlp_output")))

    # Hook末层自身获取residual_input
    def residual_hook(module, input, output):
        if isinstance(input, tuple) and len(input) > 0:
            captured["residual_input"] = input[0].detach().float().cpu().numpy()
    hooks.append(last_layer.register_forward_hook(residual_hook))

    # 前向推理
    input_device = next(model.parameters()).device
    toks = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = toks["input_ids"].to(input_device)
    attention_mask = toks["attention_mask"].to(input_device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True)

    for h in hooks:
        h.remove()

    # 获取末层pre-norm和post-norm hidden state
    # Qwen3源码: all_hidden_states在每层前添加, 最后添加norm后的状态
    # hs[0]=embedding, hs[1]=L0_out, ..., hs[35]=L34_out, hs[36]=norm(L35_out)
    # 所以: h_pre(=L35输出, pre-final-norm) 需要从hook重建
    #       h_post(=norm后的输出) = hs[-1]
    hs = out.hidden_states
    hs_len = len(hs)
    # h_post: 最终RMSNorm输出 (从hidden_states获取)
    h_post = hs[hs_len - 1][0, -1].float().cpu().numpy()

    residual_input = captured.get("residual_input")
    attn_output = captured.get("attn_output")
    mlp_output = captured.get("mlp_output")

    # h_pre: 最后一层输出 = residual_input + attn_output + mlp_output
    # captured中已是numpy数组 [1, seq, d_model], 需要取last token
    ri = residual_input[0, -1] if residual_input is not None else None
    ao = attn_output[0, -1] if attn_output is not None else None
    mo = mlp_output[0, -1] if mlp_output is not None else None

    h_pre_from_hooks = None
    if ri is not None:
        h_pre_from_hooks = ri.astype(np.float64)
        if ao is not None:
            h_pre_from_hooks = h_pre_from_hooks + ao.astype(np.float64)
        if mo is not None:
            h_pre_from_hooks = h_pre_from_hooks + mo.astype(np.float64)

    # 验证: h_pre_from_hooks 应该 ≈ hs[n_layers-1] (pre-norm for last layer input)
    # 实际: h_pre = output of last layer = ri + ao + mo
    # h_post = hs[-1] = RMSNorm(h_pre)

    # 手动计算h_normed_no_gain和rms (从h_pre_from_hooks)
    if h_pre_from_hooks is not None:
        rms_val = float(np.sqrt(np.mean(h_pre_from_hooks ** 2) + 1e-5))
        h_normed_no_gain = h_pre_from_hooks / rms_val
    else:
        rms_val = 0.0
        h_normed_no_gain = None

    return {
        "h_pre": h_pre_from_hooks,
        "h_post": h_post,
        "h_normed_no_gain": h_normed_no_gain,
        "rms": rms_val,
        "residual_input": ri,
        "attn_output": ao,
        "mlp_output": mo,
    }


# ============== Exp1: Gain维度结构分析 ==============
def exp1_gain_dimension_structure(h_pre, g, W_U, target_ids, competitor_ids):
    """
    分解gain效应的维度来源:
    1. 高/中/低gain维度对D_no_gain和D_with_gain的贡献
    2. hidden×gain交互: gain放大了h_pre中哪些维度的信号?
    """
    d = len(h_pre)
    if g is None:
        return {"error": "no gain weight available"}

    # 按gain值分三档
    p33 = np.percentile(g, 33)
    p66 = np.percentile(g, 66)
    high_dims = np.where(g > p66)[0]
    mid_dims = np.where((g > p33) & (g <= p66))[0]
    low_dims = np.where(g <= p33)[0]

    rms = np.sqrt(np.mean(h_pre ** 2) + 1e-5)
    h_normed = h_pre / rms  # 无gain的归一化

    results = {}

    for dim_group, dims, label in [
        (high_dims, high_dims, "high_gain"),
        (mid_dims, mid_dims, "mid_gain"),
        (low_dims, low_dims, "low_gain"),
    ]:
        # h_normed在这些维度上的范数
        h_slice_norm = float(np.linalg.norm(h_normed[dims]))

        # 这些维度对D_no_gain的贡献
        d_no_gain_parts = []
        d_with_gain_parts = []
        for tid in target_ids:
            if tid < W_U.shape[0]:
                w_tid = W_U[tid]
                # no gain: <h_normed[dims], w_tid[dims]>
                d_no_gain_parts.append(float(np.dot(h_normed[dims], w_tid[dims])))
                # with gain: <h_normed[dims]*g[dims], w_tid[dims]>
                d_with_gain_parts.append(float(np.dot(h_normed[dims] * g[dims], w_tid[dims])))

        c_no_gain_parts = []
        c_with_gain_parts = []
        for cid in competitor_ids:
            if cid < W_U.shape[0]:
                w_cid = W_U[cid]
                c_no_gain_parts.append(float(np.dot(h_normed[dims], w_cid[dims])))
                c_with_gain_parts.append(float(np.dot(h_normed[dims] * g[dims], w_cid[dims])))

        t_no = float(np.mean(d_no_gain_parts)) if d_no_gain_parts else 0
        t_with = float(np.mean(d_with_gain_parts)) if d_with_gain_parts else 0
        c_no = float(np.mean(c_no_gain_parts)) if c_no_gain_parts else 0
        c_with = float(np.mean(c_with_gain_parts)) if c_with_gain_parts else 0

        results[label] = {
            "n_dims": len(dims),
            "mean_g": float(np.mean(g[dims])),
            "h_norm_in_dims": h_slice_norm,
            "target_contrib_no_gain": t_no,
            "target_contrib_with_gain": t_with,
            "competitor_contrib_no_gain": c_no,
            "competitor_contrib_with_gain": c_with,
            "D_no_gain": t_no - c_no,
            "D_with_gain": t_with - c_with,
            "gain_effect_D": (t_with - c_with) - (t_no - c_no),
        }

    # 全维度汇总
    D_no_gain, t_no, c_no, _, _ = compute_dcf(h_normed, W_U, target_ids, competitor_ids)
    h_with_gain = h_normed * g
    D_with_gain, t_with, c_with, _, _ = compute_dcf(h_with_gain, W_U, target_ids, competitor_ids)

    results["full"] = {
        "D_no_gain": D_no_gain,
        "D_with_gain": D_with_gain,
        "gain_effect": D_with_gain - D_no_gain,
        "target_no_gain": t_no,
        "target_with_gain": t_with,
        "competitor_no_gain": c_no,
        "competitor_with_gain": c_with,
    }

    return results


# ============== Exp2: 目标-竞争项不等比例压缩 ==============
def exp2_target_competitor_compression(h_pre, h_post, W_U, target_ids, competitor_ids):
    """
    分别测量target和competitor的pre/post压缩率
    """
    # Pre-norm logits
    target_logits_pre = [float(np.dot(h_pre, W_U[tid])) for tid in target_ids if tid < W_U.shape[0]]
    comp_logits_pre = [float(np.dot(h_pre, W_U[cid])) for cid in competitor_ids if cid < W_U.shape[0]]

    # Post-norm logits
    target_logits_post = [float(np.dot(h_post, W_U[tid])) for tid in target_ids if tid < W_U.shape[0]]
    comp_logits_post = [float(np.dot(h_post, W_U[cid])) for cid in competitor_ids if cid < W_U.shape[0]]

    if not target_logits_pre or not comp_logits_pre:
        return {"error": "no logits computed"}

    t_pre = float(np.mean(target_logits_pre))
    t_post = float(np.mean(target_logits_post))
    c_pre = float(np.mean(comp_logits_pre))
    c_post = float(np.mean(comp_logits_post))

    # 压缩率: post/pre (log尺度更稳定)
    t_compression = float(np.log(abs(t_post) / max(abs(t_pre), 1e-10)))
    c_compression = float(np.log(abs(c_post) / max(abs(c_pre), 1e-10)))

    # 绝对值压缩
    t_abs_ratio = float(abs(t_post) / max(abs(t_pre), 1e-10))
    c_abs_ratio = float(abs(c_post) / max(abs(c_pre), 1e-10))

    # 符号翻转检测
    t_sign_flipped = (t_pre * t_post < 0)
    c_sign_flipped = (c_pre * c_post < 0)

    return {
        "target_logit_pre": t_pre,
        "target_logit_post": t_post,
        "competitor_logit_pre": c_pre,
        "competitor_logit_post": c_post,
        "D_pre": t_pre - c_pre,
        "D_post": t_post - c_post,
        "target_compression_log": t_compression,
        "competitor_compression_log": c_compression,
        "target_abs_ratio": t_abs_ratio,
        "competitor_abs_ratio": c_abs_ratio,
        "target_sign_flipped": t_sign_flipped,
        "competitor_sign_flipped": c_sign_flipped,
        "compression_diff_log": t_compression - c_compression,
        "target_retained_pct": float(t_post / max(abs(t_pre), 1e-10) * 100) if t_pre > 0 else 0,
        "competitor_retained_pct": float(c_post / max(abs(c_pre), 1e-10) * 100) if c_pre > 0 else 0,
    }


# ============== Exp3: 残差流语义主载体验证 ==============
def exp3_residual_semantic_verification(h_pre, residual_input, attn_output, mlp_output,
                                         g, W_U, target_ids, competitor_ids):
    """
    通过消融/加倍/反转residual验证其语义主体地位
    """
    if residual_input is None or attn_output is None or mlp_output is None:
        return {"error": "hook capture failed"}

    # h_pre = residual_input[0,-1] + attn_output[0,-1] + mlp_output[0,-1]
    # 但residual_input是层输入, attn/mlp输出是增量
    # 实际: h_pre = residual_input + attn_output + mlp_output
    # (取决于hook位置, residual_input是进入末层前的状态)

    # 验证: 检查是否 residual_input + attn_output + mlp_output ≈ h_pre
    reconstructed = residual_input + attn_output + mlp_output
    recon_error = float(np.linalg.norm(reconstructed - h_pre))
    recon_rel_error = recon_error / max(float(np.linalg.norm(h_pre)), 1e-10)

    results = {
        "reconstruction_error": recon_error,
        "reconstruction_rel_error": recon_rel_error,
    }

    # 各组件在h_pre中的比例
    r_norm = float(np.linalg.norm(residual_input))
    a_norm = float(np.linalg.norm(attn_output))
    m_norm = float(np.linalg.norm(mlp_output))
    h_norm = float(np.linalg.norm(h_pre))
    results["component_norms"] = {
        "residual": r_norm,
        "attn": a_norm,
        "mlp": m_norm,
        "total": h_norm,
    }

    # 干预实验: 修改residual的权重
    interventions = {
        "full": {"residual_weight": 1.0, "attn_weight": 1.0, "mlp_weight": 1.0},
        "no_residual": {"residual_weight": 0.0, "attn_weight": 1.0, "mlp_weight": 1.0},
        "double_residual": {"residual_weight": 2.0, "attn_weight": 1.0, "mlp_weight": 1.0},
        "half_residual": {"residual_weight": 0.5, "attn_weight": 1.0, "mlp_weight": 1.0},
        "reverse_residual": {"residual_weight": -1.0, "attn_weight": 1.0, "mlp_weight": 1.0},
    }

    for name, weights in interventions.items():
        rw = weights["residual_weight"]
        aw = weights["attn_weight"]
        mw = weights["mlp_weight"]

        h_mod = rw * residual_input + aw * attn_output + mw * mlp_output

        # Pre-norm D
        D_pre, t_pre, c_pre, _, _ = compute_dcf(h_mod, W_U, target_ids, competitor_ids)

        # Post-norm D (apply RMSNorm)
        if g is not None:
            h_post_mod = rmsnorm_numpy(h_mod, weight=g)
        else:
            h_post_mod = rmsnorm_numpy(h_mod, weight=None)
        D_post, t_post, c_post, _, _ = compute_dcf(h_post_mod, W_U, target_ids, competitor_ids)

        results[name] = {
            "D_pre": D_pre,
            "D_post": D_post,
            "target_pre": t_pre,
            "target_post": t_post,
            "competitor_pre": c_pre,
            "competitor_post": c_post,
            "h_norm": float(np.linalg.norm(h_mod)),
            "rms": float(np.sqrt(np.mean(h_mod ** 2) + 1e-5)),
        }

    return results


# ============== Exp4: MLP抑制方向机制 ==============
def exp4_mlp_suppression_mechanism(h_pre, mlp_output, g, W_U, target_ids, competitor_ids):
    """
    分析MLP输出在w_D, g⊙w_D, residual方向上的投影
    """
    if mlp_output is None:
        return {"error": "no mlp_output"}

    results = {}

    for token_type, ids, label in [
        ("target", target_ids, "target"),
        ("competitor", competitor_ids, "competitor"),
    ]:
        for tid in ids:
            if tid >= W_U.shape[0]:
                continue
            w_tid = W_U[tid]  # [d_model]
            w_tid_norm = w_tid / max(np.linalg.norm(w_tid), 1e-10)

            # MLP在w_D方向的投影
            proj_mlp_on_wD = float(np.dot(mlp_output, w_tid_norm))

            # MLP在g⊙w_D方向的投影
            if g is not None:
                gw_D = g * w_tid
                gw_D_norm = gw_D / max(np.linalg.norm(gw_D), 1e-10)
                proj_mlp_on_gwD = float(np.dot(mlp_output, gw_D_norm))
            else:
                proj_mlp_on_gwD = 0.0

            # h_pre在w_D方向的投影
            proj_h_on_wD = float(np.dot(h_pre, w_tid_norm))

            # MLP与h_pre在w_D上的对齐度
            if abs(proj_h_on_wD) > 1e-10:
                mlp_alignment = proj_mlp_on_wD / proj_h_on_wD
            else:
                mlp_alignment = 0.0

            if label not in results:
                results[label] = {
                    "proj_mlp_on_wD": [],
                    "proj_mlp_on_gwD": [],
                    "proj_h_on_wD": [],
                    "mlp_alignment": [],
                }

            results[label]["proj_mlp_on_wD"].append(proj_mlp_on_wD)
            results[label]["proj_mlp_on_gwD"].append(proj_mlp_on_gwD)
            results[label]["proj_h_on_wD"].append(proj_h_on_wD)
            results[label]["mlp_alignment"].append(mlp_alignment)

    # 汇总 (先收集keys避免遍历时修改dict)
    for label in list(results.keys()):
        orig_keys = list(results[label].keys())
        for key in orig_keys:
            vals = results[label][key]
            if isinstance(vals, list):
                results[label][key + "_mean"] = float(np.mean(vals)) if vals else 0.0
                results[label][key + "_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0

    # MLP对D的直接贡献
    D_mlp_direct, _, _, _, _ = compute_dcf(mlp_output, W_U, target_ids, competitor_ids)
    results["D_mlp_direct"] = D_mlp_direct
    results["mlp_norm"] = float(np.linalg.norm(mlp_output))
    results["h_pre_norm"] = float(np.linalg.norm(h_pre))

    return results


# ============== Exp5: 动作类专项 ==============
def exp5_action_specific(h_pre, h_post, g, W_U, target_ids, competitor_ids, tokenizer):
    """
    扩展动作集合, 详细分析每个动作的target/competitor压缩
    """
    # 对每个target token, 记录其单独的压缩率
    results = {}

    for tid in target_ids:
        if tid >= W_U.shape[0]:
            continue
        tok_name = tokenizer.decode([tid]).strip()

        logit_pre = float(np.dot(h_pre, W_U[tid]))
        logit_post = float(np.dot(h_post, W_U[tid]))

        results[f"target_{tok_name}_id{tid}"] = {
            "logit_pre": logit_pre,
            "logit_post": logit_post,
            "compression_log": float(np.log(abs(logit_post) / max(abs(logit_pre), 1e-10))),
            "abs_ratio": float(abs(logit_post) / max(abs(logit_pre), 1e-10)),
            "sign_flipped": (logit_pre * logit_post < 0),
        }

    for cid in competitor_ids[:3]:  # 只取top3竞争项
        if cid >= W_U.shape[0]:
            continue
        tok_name = tokenizer.decode([cid]).strip()

        logit_pre = float(np.dot(h_pre, W_U[cid]))
        logit_post = float(np.dot(h_post, W_U[cid]))

        results[f"competitor_{tok_name}_id{cid}"] = {
            "logit_pre": logit_pre,
            "logit_post": logit_post,
            "compression_log": float(np.log(abs(logit_post) / max(abs(logit_pre), 1e-10))),
            "abs_ratio": float(abs(logit_post) / max(abs(logit_pre), 1e-10)),
            "sign_flipped": (logit_pre * logit_post < 0),
        }

    # 整体D变化
    D_pre, t_pre, c_pre, _, _ = compute_dcf(h_pre, W_U, target_ids, competitor_ids)
    D_post, t_post, c_post, _, _ = compute_dcf(h_post, W_U, target_ids, competitor_ids)

    results["D_pre"] = D_pre
    results["D_post"] = D_post
    results["target_mean_pre"] = t_pre
    results["target_mean_post"] = t_post
    results["competitor_mean_pre"] = c_pre
    results["competitor_mean_post"] = c_post
    results["target_compression_pct"] = float(t_post / max(abs(t_pre), 1e-10) * 100) if t_pre != 0 else 0
    results["competitor_compression_pct"] = float(c_post / max(abs(c_pre), 1e-10) * 100) if c_pre != 0 else 0

    return results


# ============== 主测试流程 ==============
def run_phase499(model_name, round_num):
    """运行Phase 499所有实验"""
    t_start = time.time()
    print(f"\n{'='*60}")
    print(f"Phase 499: {model_name} Round {round_num}")
    print(f"{'='*60}")

    # 加载模型
    model, tokenizer, device = load_model_bf16(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    print(f"  Model: {model_info.model_class}, L={n_layers}, d={d_model}")

    # 加载关键权重
    print(f"  Loading W_U and RMSNorm weight...")
    W_U = get_W_U(model, model_name)
    g = load_rmsnorm_weight(model, model_name)
    if g is not None:
        print(f"  RMSNorm weight: mean={np.mean(g):.4f}, pct>1={np.mean(g>1)*100:.1f}%")

    # 选择类别
    categories = CATEGORIES
    if round_num >= 2:
        # R2加入扩展动作集合
        categories = {**CATEGORIES, **ACTION_EXTENDED}

    all_results = {
        "phase": 499,
        "round": round_num,
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_info": {
            "class": model_info.model_class,
            "n_layers": n_layers,
            "d_model": d_model,
        },
        "rmsnorm_weight_available": g is not None,
        "exp1_gain_dimension": {},
        "exp2_target_competitor_compression": {},
        "exp3_residual_semantic": {},
        "exp4_mlp_suppression": {},
        "exp5_action_specific": {},
    }

    total_samples = sum(len(cat["objects"]) for cat in categories.values())
    sample_count = 0

    for cat_name, cat_data in categories.items():
        print(f"\n--- Category: {cat_name} ---")
        exp1_samples = []
        exp2_samples = []
        exp3_samples = []
        exp4_samples = []
        exp5_samples = []

        for obj in cat_data["objects"]:
            sample_count += 1
            prompt = f"The {obj} {cat_data['relation']}"
            if model_name == "glm4":
                prompt = f"[gMASK]<sop>{prompt}"  # GLM4特殊前缀

            try:
                # 获取末层输出
                outputs = get_last_layer_outputs(model, tokenizer, device, prompt, n_layers)
                h_pre = outputs["h_pre"]        # hook重建: ri + ao + mo (pre-RMSNorm)
                h_post = outputs["h_post"]      # hs[-1] (post-RMSNorm)
                h_normed_no_gain = outputs["h_normed_no_gain"]
                rms_val = outputs["rms"]
                residual_input = outputs["residual_input"]  # 已经是[d]形状
                attn_output = outputs["attn_output"]        # 已经是[d]形状
                mlp_output = outputs["mlp_output"]          # 已经是[d]形状

                # 获取target和competitor token ids
                target_ids = []
                for tok in cat_data["target_tokens"]:
                    ids = tokenizer.encode(tok, add_special_tokens=False)
                    target_ids.extend(ids)

                # 使用固定语义竞争词 (比top-k更有意义)
                competitor_ids = []
                if "competitor_tokens" in cat_data:
                    for tok in cat_data["competitor_tokens"]:
                        ids = tokenizer.encode(tok, add_special_tokens=False)
                        competitor_ids.extend(ids)
                else:
                    # 回退到top-k
                    competitor_ids = find_top_competitor_ids(W_U, target_ids, h_post, n_comp=5)

                # ---- Exp1: Gain维度结构 ----
                exp1_r = exp1_gain_dimension_structure(h_pre, g, W_U, target_ids, competitor_ids)
                exp1_samples.append(exp1_r)

                # ---- Exp2: 目标-竞争压缩 ----
                exp2_r = exp2_target_competitor_compression(h_pre, h_post, W_U, target_ids, competitor_ids)
                exp2_samples.append(exp2_r)

                # ---- Exp3: 残差语义验证 ----
                exp3_r = exp3_residual_semantic_verification(
                    h_pre, residual_input, attn_output, mlp_output,
                    g, W_U, target_ids, competitor_ids)
                exp3_samples.append(exp3_r)

                # ---- Exp4: MLP抑制方向 ----
                exp4_r = exp4_mlp_suppression_mechanism(
                    h_pre, mlp_output, g, W_U, target_ids, competitor_ids)
                exp4_samples.append(exp4_r)

                # ---- Exp5: 动作类专项 ----
                if "action" in cat_name:
                    exp5_r = exp5_action_specific(
                        h_pre, h_post, g, W_U, target_ids, competitor_ids, tokenizer)
                    exp5_samples.append(exp5_r)

                # 定时日志
                if sample_count % 5 == 0:
                    elapsed = time.time() - t_start
                    print(f"  [{sample_count}/{total_samples}] {obj}: "
                          f"D_pre={exp2_r.get('D_pre', 0):.2f}, "
                          f"D_post={exp2_r.get('D_post', 0):.2f}, "
                          f"elapsed={elapsed:.1f}s")

                # 检查NaN
                if np.any(np.isnan(h_pre)) or np.any(np.isnan(h_post)):
                    print(f"  WARNING: NaN detected for {obj}, skipping...")
                    continue

            except Exception as e:
                print(f"  ERROR processing {obj}: {e}")
                continue

        # 汇总每类别的平均结果
        def safe_mean(samples, key, default=0):
            vals = [s.get(key, default) for s in samples if isinstance(s, dict) and key in s and not isinstance(s[key], dict)]
            return float(np.mean(vals)) if vals else default

        # Exp1 汇总
        if exp1_samples:
            full_data = [s.get("full", {}) for s in exp1_samples if "full" in s]
            all_results["exp1_gain_dimension"][cat_name] = {
                "mean_D_no_gain": safe_mean(full_data, "D_no_gain"),
                "mean_D_with_gain": safe_mean(full_data, "D_with_gain"),
                "mean_gain_effect": safe_mean(full_data, "gain_effect"),
                "mean_target_no_gain": safe_mean(full_data, "target_no_gain"),
                "mean_target_with_gain": safe_mean(full_data, "target_with_gain"),
                "mean_competitor_no_gain": safe_mean(full_data, "competitor_no_gain"),
                "mean_competitor_with_gain": safe_mean(full_data, "competitor_with_gain"),
                "n_samples": len(exp1_samples),
                # 高/低gain维度贡献
                "high_gain_D_no_gain_mean": safe_mean(
                    [s.get("high_gain", {}) for s in exp1_samples if "high_gain" in s], "D_no_gain"),
                "high_gain_D_with_gain_mean": safe_mean(
                    [s.get("high_gain", {}) for s in exp1_samples if "high_gain" in s], "D_with_gain"),
                "high_gain_gain_effect_mean": safe_mean(
                    [s.get("high_gain", {}) for s in exp1_samples if "high_gain" in s], "gain_effect_D"),
                "low_gain_D_no_gain_mean": safe_mean(
                    [s.get("low_gain", {}) for s in exp1_samples if "low_gain" in s], "D_no_gain"),
                "low_gain_D_with_gain_mean": safe_mean(
                    [s.get("low_gain", {}) for s in exp1_samples if "low_gain" in s], "D_with_gain"),
                "low_gain_gain_effect_mean": safe_mean(
                    [s.get("low_gain", {}) for s in exp1_samples if "low_gain" in s], "gain_effect_D"),
            }

        # Exp2 汇总
        if exp2_samples:
            all_results["exp2_target_competitor_compression"][cat_name] = {
                "mean_D_pre": safe_mean(exp2_samples, "D_pre"),
                "mean_D_post": safe_mean(exp2_samples, "D_post"),
                "mean_target_logit_pre": safe_mean(exp2_samples, "target_logit_pre"),
                "mean_target_logit_post": safe_mean(exp2_samples, "target_logit_post"),
                "mean_competitor_logit_pre": safe_mean(exp2_samples, "competitor_logit_pre"),
                "mean_competitor_logit_post": safe_mean(exp2_samples, "competitor_logit_post"),
                "mean_target_compression_log": safe_mean(exp2_samples, "target_compression_log"),
                "mean_competitor_compression_log": safe_mean(exp2_samples, "competitor_compression_log"),
                "mean_target_abs_ratio": safe_mean(exp2_samples, "target_abs_ratio"),
                "mean_competitor_abs_ratio": safe_mean(exp2_samples, "competitor_abs_ratio"),
                "mean_compression_diff_log": safe_mean(exp2_samples, "compression_diff_log"),
                "n_target_sign_flipped": sum(1 for s in exp2_samples if s.get("target_sign_flipped", False)),
                "n_competitor_sign_flipped": sum(1 for s in exp2_samples if s.get("competitor_sign_flipped", False)),
                "n_samples": len(exp2_samples),
            }

        # Exp3 汇总
        if exp3_samples:
            valid_exp3 = [s for s in exp3_samples if "error" not in s]
            if valid_exp3:
                all_results["exp3_residual_semantic"][cat_name] = {
                    "mean_reconstruction_rel_error": safe_mean(valid_exp3, "reconstruction_rel_error"),
                }
                for intervention in ["full", "no_residual", "double_residual", "half_residual", "reverse_residual"]:
                    intv_data = [s.get(intervention, {}) for s in valid_exp3 if intervention in s]
                    all_results["exp3_residual_semantic"][cat_name][f"mean_{intervention}_D_pre"] = safe_mean(intv_data, "D_pre")
                    all_results["exp3_residual_semantic"][cat_name][f"mean_{intervention}_D_post"] = safe_mean(intv_data, "D_post")
                    all_results["exp3_residual_semantic"][cat_name][f"mean_{intervention}_target_pre"] = safe_mean(intv_data, "target_pre")
                    all_results["exp3_residual_semantic"][cat_name][f"mean_{intervention}_target_post"] = safe_mean(intv_data, "target_post")
                    all_results["exp3_residual_semantic"][cat_name][f"mean_{intervention}_competitor_pre"] = safe_mean(intv_data, "competitor_pre")
                    all_results["exp3_residual_semantic"][cat_name][f"mean_{intervention}_competitor_post"] = safe_mean(intv_data, "competitor_post")
                all_results["exp3_residual_semantic"][cat_name]["n_samples"] = len(valid_exp3)

        # Exp4 汇总
        if exp4_samples:
            valid_exp4 = [s for s in exp4_samples if "error" not in s]
            if valid_exp4:
                all_results["exp4_mlp_suppression"][cat_name] = {
                    "mean_D_mlp_direct": safe_mean(valid_exp4, "D_mlp_direct"),
                    "mean_mlp_norm": safe_mean(valid_exp4, "mlp_norm"),
                    "mean_h_pre_norm": safe_mean(valid_exp4, "h_pre_norm"),
                    "mean_target_proj_mlp_on_wD": safe_mean(
                        [s.get("target", {}) for s in valid_exp4], "proj_mlp_on_wD_mean"),
                    "mean_target_proj_mlp_on_gwD": safe_mean(
                        [s.get("target", {}) for s in valid_exp4], "proj_mlp_on_gwD_mean"),
                    "mean_target_proj_h_on_wD": safe_mean(
                        [s.get("target", {}) for s in valid_exp4], "proj_h_on_wD_mean"),
                    "mean_target_mlp_alignment": safe_mean(
                        [s.get("target", {}) for s in valid_exp4], "mlp_alignment_mean"),
                    "mean_competitor_proj_mlp_on_wD": safe_mean(
                        [s.get("competitor", {}) for s in valid_exp4], "proj_mlp_on_wD_mean"),
                    "mean_competitor_proj_mlp_on_gwD": safe_mean(
                        [s.get("competitor", {}) for s in valid_exp4], "proj_mlp_on_gwD_mean"),
                    "mean_competitor_proj_h_on_wD": safe_mean(
                        [s.get("competitor", {}) for s in valid_exp4], "proj_h_on_wD_mean"),
                    "mean_competitor_mlp_alignment": safe_mean(
                        [s.get("competitor", {}) for s in valid_exp4], "mlp_alignment_mean"),
                    "n_samples": len(valid_exp4),
                }

        # Exp5 汇总 (仅action类别)
        if exp5_samples:
            all_results["exp5_action_specific"][cat_name] = {
                "n_samples": len(exp5_samples),
                "sample_details": exp5_samples,
            }

    # 释放模型
    gpu_mem_before = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    del model
    gc.collect()
    torch.cuda.empty_cache()
    gpu_mem_after = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"\n  GPU memory: {gpu_mem_before:.2f}GB -> {gpu_mem_after:.2f}GB")

    # 保存结果
    output_file = OUTPUT_DIR / f"phase499_{model_name}_r{round_num}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Results saved to {output_file}")

    elapsed = time.time() - t_start
    print(f"  Total time: {elapsed:.1f}s")

    return all_results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    valid_models = ["qwen3", "glm4", "deepseek7b"]
    if model_name not in valid_models:
        print(f"Invalid model: {model_name}. Use one of: {valid_models}")
        sys.exit(1)

    result = run_phase499(model_name, round_num)

    # 打印关键摘要
    print(f"\n{'='*60}")
    print(f"Phase 499 Summary: {model_name} R{round_num}")
    print(f"{'='*60}")

    for exp_name in ["exp1_gain_dimension", "exp2_target_competitor_compression",
                     "exp3_residual_semantic", "exp4_mlp_suppression", "exp5_action_specific"]:
        print(f"\n{exp_name}:")
        exp_data = result.get(exp_name, {})
        for cat, vals in exp_data.items():
            if isinstance(vals, dict):
                # 选择关键指标打印
                key_vals = {k: f"{v:.3f}" if isinstance(v, float) else v
                           for k, v in vals.items() if not k.startswith("sample_") and not isinstance(v, dict)}
                print(f"  {cat}: {key_vals}")
