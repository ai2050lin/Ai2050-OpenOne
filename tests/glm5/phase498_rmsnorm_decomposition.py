"""
Phase 498: RMSNorm Readout Geometry & Norm Channel Closure
==========================================================
核心目标:
1. 精确分解D_post = numerator / denominator的三个效应:
   - 分子效应(numerator): <h, g⊙w_D> 的变化
   - 分母效应(denominator): rms(h) 的变化
   - 增益向量效应(gain): g⊙w_D vs w_D 方向差异
2. 固定RMSNorm对照: 分离RMS分母/RMSNorm weight/残差方向效应
3. MLP范数通道闭环: 保持方向只改范数 vs 保持范数只改方向
4. Action类符号翻转专项分析

关键数学公式:
  D = <h, g⊙w_D> / rms(h) = numerator / denominator
  D_post = numerator_post / rms_post
  
  干预MLP后:
  δD_post ≈ δnumerator/rms - D·δrms/rms  (一阶展开)
  第一项=方向/分子效应, 第二项=范数/分母效应

Usage:
  python tests/glm5/phase498_rmsnorm_decomposition.py qwen3 1
  python tests/glm5/phase498_rmsnorm_decomposition.py glm4 1
  python tests/glm5/phase498_rmsnorm_decomposition.py deepseek7b 1
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
    """RMSNorm in numpy: x / rms(x) * weight"""
    rms = np.sqrt(np.mean(x ** 2) + eps)
    normed = x / rms
    if weight is not None:
        normed = normed * weight
    return normed, rms


def compute_D_from_hidden(hidden_np, W_U, target_ids, comp_ids):
    """从hidden state计算DCF"""
    logits = hidden_np @ W_U.T
    target_logit = np.mean([logits[tid] for tid in target_ids if tid < len(logits)])
    comp_logits = [logits[cid] for cid in comp_ids if cid < len(logits)]
    if len(comp_logits) == 0:
        return 0.0
    return float(target_logit - np.mean(comp_logits))


def compute_numerator(hidden_np, effective_w, target_ids, comp_ids):
    """计算分子 <h, effective_w> 对D的贡献 (对每个token)"""
    projections = hidden_np @ effective_w.T  # [vocab]
    target_proj = np.mean([projections[tid] for tid in target_ids if tid < len(projections)])
    comp_proj = [projections[cid] for cid in comp_ids if cid < len(projections)]
    if len(comp_proj) == 0:
        return 0.0
    return float(target_proj - np.mean(comp_proj))


def get_rmsnorm_weight(model, model_name):
    """获取final RMSNorm的weight向量"""
    # 尝试多种路径
    for attr in ['model.norm', 'model.final_layernorm', 'model.decoder.final_layer_norm']:
        parts = attr.split('.')
        obj = model
        for p in parts:
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                obj = None
                break
        if obj is not None and hasattr(obj, 'weight'):
            w = obj.weight.detach()
            # 检查是否在meta device上
            if str(w.device) == 'meta':
                print(f"  [WARN] RMSNorm weight on meta device, trying safetensors...")
                return None
            return w.float().cpu().numpy()
    
    print(f"  [WARN] Cannot find RMSNorm weight for {model_name}")
    return None


def get_lm_head_weight(model, model_name):
    """获取lm_head权重"""
    if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
        w = model.lm_head.weight.detach()
        if str(w.device) == 'meta':
            return None
        return w.float().cpu().numpy()
    return None


def make_prompt(obj, relation):
    return f"The {obj} {relation}"


def get_target_and_comp_ids(tokenizer, target_tokens, comp_categories_tokens):
    """获取目标token和竞争token的ID"""
    target_ids = []
    for t in target_tokens:
        ids = tokenizer.encode(t, add_special_tokens=False)
        target_ids.extend(ids)
    
    comp_ids = []
    for cat_tokens in comp_categories_tokens:
        for t in cat_tokens:
            ids = tokenizer.encode(t, add_special_tokens=False)
            comp_ids.extend(ids)
    return target_ids, comp_ids


def run_exp1_rmsnorm_math_decomposition(model, tokenizer, model_name, n_samples=8):
    """
    Exp1: RMSNorm数学精确分解
    D_post = numerator / rms(h)
    
    对每个样本记录:
    - numerator_full, numerator_no_mlp
    - rms_full, rms_no_mlp
    - D_post_full, D_post_no_mlp
    - 分子效应: Δnumerator / rms_full
    - 分母效应: -D_full * Δrms / rms_full
    - 残差(高阶交互)
    """
    print("\n" + "="*60)
    print("Exp1: RMSNorm数学精确分解")
    print("="*60)
    
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    last_layer = layers[-1]
    W_U = get_W_U(model, model_name)
    rmsnorm_w = get_rmsnorm_weight(model, model_name)
    
    # 获取所有竞争类别的target tokens
    all_cat_tokens = {k: v["target_tokens"] for k, v in CATEGORIES.items()}
    
    results = {}
    
    for cat_name, cat_data in CATEGORIES.items():
        comp_cats = [k for k in CATEGORIES if k != cat_name]
        comp_tokens = [CATEGORIES[k]["target_tokens"] for k in comp_cats]
        
        cat_results = []
        objects = cat_data["objects"][:n_samples]
        
        for obj in objects:
            prompt = make_prompt(obj, cat_data["relation"])
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"]
            attn_mask = inputs["attention_mask"]
            
            # 找到输入设备
            input_device = next(model.parameters()).device
            input_ids = input_ids.to(input_device)
            attn_mask = attn_mask.to(input_device)
            
            target_ids, comp_ids = get_target_and_comp_ids(
                tokenizer, cat_data["target_tokens"], comp_tokens)
            
            # === 基线: 正常前向 ===
            captured = {}
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    captured['last_hidden'] = output[0].detach()
                else:
                    captured['last_hidden'] = output.detach()
            
            h = last_layer.register_forward_hook(hook_fn)
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn_mask,
                           output_hidden_states=True)
            h.remove()
            
            last_hidden_full = out.hidden_states[-1][0, -1].float().cpu().numpy()
            if 'last_hidden' in captured:
                layer_output = captured['last_hidden'][0, -1].float().cpu().numpy()
            else:
                layer_output = last_hidden_full
            
            # 计算各种量
            h_pre = last_hidden_full  # pre-norm hidden state (最后一层输出后的残差)
            
            # 实际上hidden_states[-1]是post-norm的, 我们需要pre-norm
            # hidden_states[-2]是最后一层输入 = pre-last-layer
            h_pre_last_layer_input = out.hidden_states[-2][0, -1].float().cpu().numpy()
            
            # 尝试从模型结构获取final norm前的hidden
            # 方法: 从hidden_states[-1]反推，或直接用模型计算
            # 最可靠: 获取model.model.norm之前的残差
            # 我们用hook来捕获
            
            # 重新用hook获取pre-norm和post-norm
            captured2 = {}
            def hook_pre_norm(module, input, output):
                # input[0]是pre-norm的hidden
                captured2['pre_norm'] = input[0].detach()
                if isinstance(output, tuple):
                    captured2['post_norm'] = output[0].detach()
                else:
                    captured2['post_norm'] = output.detach()
            
            # 找到final norm层
            final_norm = None
            if hasattr(model, 'model') and hasattr(model.model, 'norm'):
                final_norm = model.model.norm
            elif hasattr(model, 'model') and hasattr(model.model, 'final_layernorm'):
                final_norm = model.model.final_layernorm
            
            if final_norm is not None:
                h2 = final_norm.register_forward_hook(hook_pre_norm)
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attn_mask)
                h2.remove()
                
                h_pre_norm = captured2['pre_norm'][0, -1].float().cpu().numpy()
                h_post_norm = captured2['post_norm'][0, -1].float().cpu().numpy()
            else:
                print(f"  [WARN] Cannot find final norm layer")
                h_pre_norm = h_pre_last_layer_input
                h_post_norm = last_hidden_full
            
            # === 零化MLP ===
            mlp = last_layer.mlp
            captured3 = {}
            def hook_pre_norm2(module, input, output):
                captured3['pre_norm'] = input[0].detach()
                if isinstance(output, tuple):
                    captured3['post_norm'] = output[0].detach()
                else:
                    captured3['post_norm'] = output.detach()
            
            def hook_zero_mlp(module, input, output):
                if isinstance(output, tuple):
                    return (torch.zeros_like(output[0]),) + output[1:]
                return torch.zeros_like(output)
            
            h_mlp = mlp.register_forward_hook(hook_zero_mlp)
            h_fn2 = final_norm.register_forward_hook(hook_pre_norm2) if final_norm else None
            
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attn_mask)
            
            h_mlp.remove()
            if h_fn2:
                h_fn2.remove()
            
            if 'pre_norm' in captured3:
                h_pre_norm_no_mlp = captured3['pre_norm'][0, -1].float().cpu().numpy()
                h_post_norm_no_mlp = captured3['post_norm'][0, -1].float().cpu().numpy()
            else:
                h_pre_norm_no_mlp = h_pre_norm
                h_post_norm_no_mlp = h_post_norm
            
            # === 计算各种D和分解 ===
            D_pre_full = compute_D_from_hidden(h_pre_norm, W_U, target_ids, comp_ids)
            D_pre_no_mlp = compute_D_from_hidden(h_pre_norm_no_mlp, W_U, target_ids, comp_ids)
            D_post_full = compute_D_from_hidden(h_post_norm, W_U, target_ids, comp_ids)
            D_post_no_mlp = compute_D_from_hidden(h_post_norm_no_mlp, W_U, target_ids, comp_ids)
            
            # RMS值
            rms_pre_full = float(np.sqrt(np.mean(h_pre_norm ** 2) + 1e-5))
            rms_pre_no_mlp = float(np.sqrt(np.mean(h_pre_norm_no_mlp ** 2) + 1e-5))
            
            # 如果有RMSNorm weight, 计算有效读出方向
            if rmsnorm_w is not None:
                effective_w_full = W_U * rmsnorm_w[np.newaxis, :]  # [vocab, d_model]
                
                # numerator = <h, g⊙w_D> 对D的贡献
                num_full = compute_numerator(h_pre_norm, effective_w_full, target_ids, comp_ids)
                num_no_mlp = compute_numerator(h_pre_norm_no_mlp, effective_w_full, target_ids, comp_ids)
                
                # D_post精确分解
                # D_post = numerator / rms(h)
                D_post_verify_full = num_full / rms_pre_full
                D_post_verify_no_mlp = num_no_mlp / rms_pre_no_mlp
                
                # ΔD_post的分解
                delta_D_post = D_post_no_mlp - D_post_full
                delta_num = num_no_mlp - num_full
                delta_rms = rms_pre_no_mlp - rms_pre_full
                
                # 一阶近似:
                # δD ≈ δnum/rms - D·δrms/rms
                numerator_effect = delta_num / rms_pre_full
                denominator_effect = -D_post_full * delta_rms / rms_pre_full
                interaction = delta_D_post - numerator_effect - denominator_effect
                
                # 检查gain向量效应: 用g⊙w_D vs w_D
                num_no_gain_full = compute_numerator(h_pre_norm, W_U, target_ids, comp_ids)
                num_no_gain_no_mlp = compute_numerator(h_pre_norm_no_mlp, W_U, target_ids, comp_ids)
                
                D_no_gain_full = num_no_gain_full / rms_pre_full
                D_no_gain_no_mlp = num_no_gain_no_mlp / rms_pre_no_mlp
                
                gain_effect = D_post_full - D_no_gain_full  # gain向量对D的贡献
            else:
                num_full = num_no_mlp = 0.0
                D_post_verify_full = D_post_verify_no_mlp = 0.0
                delta_D_post = D_post_no_mlp - D_post_full
                delta_num = delta_rms = 0.0
                numerator_effect = denominator_effect = interaction = 0.0
                D_no_gain_full = D_no_gain_no_mlp = 0.0
                gain_effect = 0.0
            
            cat_results.append({
                "obj": obj,
                "D_pre_full": D_pre_full,
                "D_pre_no_mlp": D_pre_no_mlp,
                "D_post_full": D_post_full,
                "D_post_no_mlp": D_post_no_mlp,
                "delta_D_pre": D_pre_no_mlp - D_pre_full,
                "delta_D_post": D_post_no_mlp - D_post_full,
                "rms_pre_full": rms_pre_full,
                "rms_pre_no_mlp": rms_pre_no_mlp,
                "rms_ratio": rms_pre_no_mlp / rms_pre_full if rms_pre_full != 0 else 0,
                "numerator_full": num_full,
                "numerator_no_mlp": num_no_mlp,
                "delta_numerator": delta_num,
                "numerator_effect": numerator_effect,
                "denominator_effect": denominator_effect,
                "interaction": interaction,
                "gain_effect": gain_effect,
                "D_no_gain_full": D_no_gain_full,
                "D_no_gain_no_mlp": D_no_gain_no_mlp,
                "D_post_verify_full": D_post_verify_full,
                "D_post_verify_no_mlp": D_post_verify_no_mlp,
            })
            
            print(f"  {cat_name}/{obj}: D_pre={D_pre_full:.2f}/{D_pre_no_mlp:.2f}, "
                  f"D_post={D_post_full:.2f}/{D_post_no_mlp:.2f}, "
                  f"num_eff={numerator_effect:.4f}, den_eff={denominator_effect:.4f}, "
                  f"interact={interaction:.4f}, gain_eff={gain_effect:.4f}")
        
        # 汇总
        means = {}
        for key in ["D_pre_full", "D_pre_no_mlp", "D_post_full", "D_post_no_mlp",
                     "delta_D_pre", "delta_D_post", "rms_pre_full", "rms_pre_no_mlp",
                     "rms_ratio", "numerator_full", "numerator_no_mlp", "delta_numerator",
                     "numerator_effect", "denominator_effect", "interaction", "gain_effect",
                     "D_no_gain_full", "D_no_gain_no_mlp"]:
            means[f"mean_{key}"] = float(np.mean([r[key] for r in cat_results]))
        
        means["n_samples"] = len(cat_results)
        means["sample_details"] = cat_results
        results[cat_name] = means
    
    return results


def run_exp2_fixed_rmsnorm_control(model, tokenizer, model_name, n_samples=8):
    """
    Exp2: 固定RMSNorm对照实验
    四种读出模式:
    1. normal: 正常RMSNorm(h) → logits
    2. fixed_denom: RMSNorm(h)但固定分母为baseline的rms
    3. no_gain: h/rms(h) → logits (去掉gain weight)
    4. no_norm: h → logits (直接pre-norm读出)
    """
    print("\n" + "="*60)
    print("Exp2: 固定RMSNorm对照实验")
    print("="*60)
    
    info = get_model_info(model, model_name)
    W_U = get_W_U(model, model_name)
    rmsnorm_w = get_rmsnorm_weight(model, model_name)
    
    # 找到final norm层
    final_norm = None
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        final_norm = model.model.norm
    elif hasattr(model, 'model') and hasattr(model.model, 'final_layernorm'):
        final_norm = model.model.final_layernorm
    
    all_cat_tokens = {k: v["target_tokens"] for k, v in CATEGORIES.items()}
    
    results = {}
    
    for cat_name, cat_data in CATEGORIES.items():
        comp_cats = [k for k in CATEGORIES if k != cat_name]
        comp_tokens = [CATEGORIES[k]["target_tokens"] for k in comp_cats]
        
        cat_results = []
        objects = cat_data["objects"][:n_samples]
        
        for obj in objects:
            prompt = make_prompt(obj, cat_data["relation"])
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"]
            attn_mask = inputs["attention_mask"]
            
            input_device = next(model.parameters()).device
            input_ids = input_ids.to(input_device)
            attn_mask = attn_mask.to(input_device)
            
            target_ids, comp_ids = get_target_and_comp_ids(
                tokenizer, cat_data["target_tokens"], comp_tokens)
            
            # 获取pre-norm和post-norm hidden
            captured = {}
            def hook_pre_norm(module, input, output):
                captured['pre_norm'] = input[0].detach()
                captured['post_norm'] = (output[0] if isinstance(output, tuple) else output).detach()
            
            if final_norm is not None:
                h = final_norm.register_forward_hook(hook_pre_norm)
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attn_mask)
                h.remove()
                
                h_pre = captured['pre_norm'][0, -1].float().cpu().numpy()
                h_post = captured['post_norm'][0, -1].float().cpu().numpy()
            else:
                with torch.no_grad():
                    out = model(input_ids=input_ids, attention_mask=attn_mask,
                               output_hidden_states=True)
                h_post = out.hidden_states[-1][0, -1].float().cpu().numpy()
                h_pre = out.hidden_states[-2][0, -1].float().cpu().numpy()
            
            # 四种读出模式的D
            # Mode 1: normal (D_post_full)
            D_normal = compute_D_from_hidden(h_post, W_U, target_ids, comp_ids)
            
            # Mode 2: fixed_denom (用baseline rms, 不重新归一化)
            # 先获取baseline rms (全类别平均)
            rms_baseline = float(np.sqrt(np.mean(h_pre ** 2) + 1e-5))
            if rmsnorm_w is not None:
                h_fixed_denom = h_pre / rms_baseline * rmsnorm_w
            else:
                h_fixed_denom = h_pre / rms_baseline
            D_fixed_denom = compute_D_from_hidden(h_fixed_denom, W_U, target_ids, comp_ids)
            
            # Mode 3: no_gain (去掉gain weight, 只做h/rms)
            h_no_gain = h_pre / rms_baseline
            D_no_gain = compute_D_from_hidden(h_no_gain, W_U, target_ids, comp_ids)
            
            # Mode 4: no_norm (直接pre-norm读出)
            D_no_norm = compute_D_from_hidden(h_pre, W_U, target_ids, comp_ids)
            
            # 分离效应
            rms_denom_effect = D_normal - D_fixed_denom  # 动态分母vs固定分母
            gain_weight_effect = D_fixed_denom - D_no_gain  # gain向量效应
            norm_scale_effect = D_no_gain - D_no_norm  # 归一化缩放效应 (h/rms vs h)
            
            cat_results.append({
                "obj": obj,
                "D_normal": D_normal,
                "D_fixed_denom": D_fixed_denom,
                "D_no_gain": D_no_gain,
                "D_no_norm": D_no_norm,
                "rms_baseline": rms_baseline,
                "rms_denom_effect": rms_denom_effect,
                "gain_weight_effect": gain_weight_effect,
                "norm_scale_effect": norm_scale_effect,
            })
            
            print(f"  {cat_name}/{obj}: D_normal={D_normal:.2f}, D_fixed_denom={D_fixed_denom:.2f}, "
                  f"D_no_gain={D_no_gain:.2f}, D_no_norm={D_no_norm:.2f}, "
                  f"denom_eff={rms_denom_effect:.4f}, gain_eff={gain_weight_effect:.4f}, "
                  f"scale_eff={norm_scale_effect:.4f}")
        
        means = {}
        for key in ["D_normal", "D_fixed_denom", "D_no_gain", "D_no_norm",
                     "rms_baseline", "rms_denom_effect", "gain_weight_effect", "norm_scale_effect"]:
            means[f"mean_{key}"] = float(np.mean([r[key] for r in cat_results]))
        
        means["n_samples"] = len(cat_results)
        means["sample_details"] = cat_results
        results[cat_name] = means
    
    return results


def run_exp3_mlp_norm_channel(model, tokenizer, model_name, n_samples=8):
    """
    Exp3: MLP范数通道闭环
    测试MLP影响D的主要通道:
    - direction_only: 保持MLP方向但归零范数 → D变化?
    - norm_only: 保持MLP范数但随机方向 → D变化?
    - scale_mlp: 缩放MLP输出(0.5x, 2x) → D变化?
    - orthogonal_mlp: 用正交于原MLP的等范数向量替换 → D变化?
    """
    print("\n" + "="*60)
    print("Exp3: MLP范数通道闭环")
    print("="*60)
    
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    last_layer = layers[-1]
    mlp = last_layer.mlp
    W_U = get_W_U(model, model_name)
    
    # 找到final norm层
    final_norm = None
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        final_norm = model.model.norm
    elif hasattr(model, 'model') and hasattr(model.model, 'final_layernorm'):
        final_norm = model.model.final_layernorm
    
    all_cat_tokens = {k: v["target_tokens"] for k, v in CATEGORIES.items()}
    
    results = {}
    
    for cat_name, cat_data in CATEGORIES.items():
        comp_cats = [k for k in CATEGORIES if k != cat_name]
        comp_tokens = [CATEGORIES[k]["target_tokens"] for k in comp_cats]
        
        cat_results = []
        objects = cat_data["objects"][:n_samples]
        
        for obj in objects:
            prompt = make_prompt(obj, cat_data["relation"])
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"]
            attn_mask = inputs["attention_mask"]
            
            input_device = next(model.parameters()).device
            input_ids = input_ids.to(input_device)
            attn_mask = attn_mask.to(input_device)
            
            target_ids, comp_ids = get_target_and_comp_ids(
                tokenizer, cat_data["target_tokens"], comp_tokens)
            
            # 先获取baseline D和MLP输出
            captured_baseline = {}
            captured_mlp = {}
            
            def hook_norm(module, input, output):
                captured_baseline['post_norm'] = (output[0] if isinstance(output, tuple) else output).detach()
            
            def hook_mlp_out(module, input, output):
                captured_mlp['mlp_output'] = (output[0] if isinstance(output, tuple) else output).detach()
            
            h_norm = final_norm.register_forward_hook(hook_norm) if final_norm else None
            h_mlp = mlp.register_forward_hook(hook_mlp_out)
            
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn_mask,
                           output_hidden_states=True)
            
            if h_norm:
                h_norm.remove()
            h_mlp.remove()
            
            D_baseline = compute_D_from_hidden(
                out.hidden_states[-1][0, -1].float().cpu().numpy(),
                W_U, target_ids, comp_ids)
            
            mlp_output = captured_mlp['mlp_output']  # [batch, seq, d_model]
            mlp_vec = mlp_output[0, -1].float().cpu().numpy()  # [d_model]
            mlp_norm = float(np.linalg.norm(mlp_vec))
            mlp_dir = mlp_vec / (mlp_norm + 1e-10)
            
            # === 零化MLP ===
            def hook_zero_mlp(module, input, output):
                if isinstance(output, tuple):
                    return (torch.zeros_like(output[0]),) + output[1:]
                return torch.zeros_like(output)
            
            h_z = mlp.register_forward_hook(hook_zero_mlp)
            with torch.no_grad():
                out_z = model(input_ids=input_ids, attention_mask=attn_mask,
                             output_hidden_states=True)
            h_z.remove()
            
            D_zero_mlp = compute_D_from_hidden(
                out_z.hidden_states[-1][0, -1].float().cpu().numpy(),
                W_U, target_ids, comp_ids)
            
            # === 缩放MLP: 0.5x ===
            def hook_scale_mlp_05(module, input, output):
                if isinstance(output, tuple):
                    return (output[0] * 0.5,) + output[1:]
                return output * 0.5
            
            h_s05 = mlp.register_forward_hook(hook_scale_mlp_05)
            with torch.no_grad():
                out_s05 = model(input_ids=input_ids, attention_mask=attn_mask,
                               output_hidden_states=True)
            h_s05.remove()
            
            D_scale_05 = compute_D_from_hidden(
                out_s05.hidden_states[-1][0, -1].float().cpu().numpy(),
                W_U, target_ids, comp_ids)
            
            # === 缩放MLP: 2x ===
            def hook_scale_mlp_2(module, input, output):
                if isinstance(output, tuple):
                    return (output[0] * 2.0,) + output[1:]
                return output * 2.0
            
            h_s2 = mlp.register_forward_hook(hook_scale_mlp_2)
            with torch.no_grad():
                out_s2 = model(input_ids=input_ids, attention_mask=attn_mask,
                              output_hidden_states=True)
            h_s2.remove()
            
            D_scale_2 = compute_D_from_hidden(
                out_s2.hidden_states[-1][0, -1].float().cpu().numpy(),
                W_U, target_ids, comp_ids)
            
            # === 正交MLP: 保持范数但用正交方向 ===
            rng = np.random.RandomState(42)
            rand_dir = rng.randn(len(mlp_dir))
            rand_dir = rand_dir / np.linalg.norm(rand_dir)
            # 移除mlp_dir方向的分量
            ortho_dir = rand_dir - np.dot(rand_dir, mlp_dir) * mlp_dir
            ortho_norm = np.linalg.norm(ortho_dir)
            if ortho_norm > 1e-10:
                ortho_dir = ortho_dir / ortho_norm
            else:
                ortho_dir = rand_dir
            ortho_vec = ortho_dir * mlp_norm  # 保持范数
            
            ortho_tensor = torch.tensor(ortho_vec, dtype=mlp_output.dtype, device=mlp_output.device)
            
            def hook_ortho_mlp(module, input, output):
                if isinstance(output, tuple):
                    new_out = output[0].clone()
                    new_out[0, -1] = ortho_tensor
                    return (new_out,) + output[1:]
                new_out = output.clone()
                new_out[0, -1] = ortho_tensor
                return new_out
            
            h_o = mlp.register_forward_hook(hook_ortho_mlp)
            with torch.no_grad():
                out_o = model(input_ids=input_ids, attention_mask=attn_mask,
                             output_hidden_states=True)
            h_o.remove()
            
            D_ortho_mlp = compute_D_from_hidden(
                out_o.hidden_states[-1][0, -1].float().cpu().numpy(),
                W_U, target_ids, comp_ids)
            
            # === 方向归零但保持范数贡献: 把MLP方向对齐到residual方向 ===
            # 获取residual (L(n-2) output)
            h_residual = out.hidden_states[-2][0, -1].float().cpu().numpy()
            res_norm = np.linalg.norm(h_residual)
            res_dir = h_residual / (res_norm + 1e-10)
            # MLP aligned to residual direction with same norm
            aligned_vec = res_dir * mlp_norm
            
            aligned_tensor = torch.tensor(aligned_vec, dtype=mlp_output.dtype, device=mlp_output.device)
            
            def hook_aligned_mlp(module, input, output):
                if isinstance(output, tuple):
                    new_out = output[0].clone()
                    new_out[0, -1] = aligned_tensor
                    return (new_out,) + output[1:]
                new_out = output.clone()
                new_out[0, -1] = aligned_tensor
                return new_out
            
            h_a = mlp.register_forward_hook(hook_aligned_mlp)
            with torch.no_grad():
                out_a = model(input_ids=input_ids, attention_mask=attn_mask,
                             output_hidden_states=True)
            h_a.remove()
            
            D_aligned_mlp = compute_D_from_hidden(
                out_a.hidden_states[-1][0, -1].float().cpu().numpy(),
                W_U, target_ids, comp_ids)
            
            cat_results.append({
                "obj": obj,
                "mlp_norm": mlp_norm,
                "D_baseline": D_baseline,
                "D_zero_mlp": D_zero_mlp,
                "D_scale_05": D_scale_05,
                "D_scale_2": D_scale_2,
                "D_ortho_mlp": D_ortho_mlp,
                "D_aligned_mlp": D_aligned_mlp,
                "delta_zero": D_zero_mlp - D_baseline,
                "delta_scale_05": D_scale_05 - D_baseline,
                "delta_scale_2": D_scale_2 - D_baseline,
                "delta_ortho": D_ortho_mlp - D_baseline,
                "delta_aligned": D_aligned_mlp - D_baseline,
            })
            
            print(f"  {cat_name}/{obj}: mlp_norm={mlp_norm:.2f}, "
                  f"Δzero={D_zero_mlp-D_baseline:.3f}, Δ0.5x={D_scale_05-D_baseline:.3f}, "
                  f"Δ2x={D_scale_2-D_baseline:.3f}, Δortho={D_ortho_mlp-D_baseline:.3f}, "
                  f"Δaligned={D_aligned_mlp-D_baseline:.3f}")
        
        means = {}
        for key in ["mlp_norm", "D_baseline", "D_zero_mlp", "D_scale_05", "D_scale_2",
                     "D_ortho_mlp", "D_aligned_mlp",
                     "delta_zero", "delta_scale_05", "delta_scale_2",
                     "delta_ortho", "delta_aligned"]:
            means[f"mean_{key}"] = float(np.mean([r[key] for r in cat_results]))
        
        means["n_samples"] = len(cat_results)
        means["sample_details"] = cat_results
        results[cat_name] = means
    
    return results


def run_exp4_action_sign_flip(model, tokenizer, model_name, n_samples=8):
    """
    Exp4: Action类符号翻转专项
    分析action为什么在RMSNorm前后符号不同
    """
    print("\n" + "="*60)
    print("Exp4: Action类符号翻转专项")
    print("="*60)
    
    info = get_model_info(model, model_name)
    W_U = get_W_U(model, model_name)
    rmsnorm_w = get_rmsnorm_weight(model, model_name)
    
    final_norm = None
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        final_norm = model.model.norm
    elif hasattr(model, 'model') and hasattr(model.model, 'final_layernorm'):
        final_norm = model.model.final_layernorm
    
    cat_data = CATEGORIES["action"]
    comp_cats = [k for k in CATEGORIES if k != "action"]
    comp_tokens = [CATEGORIES[k]["target_tokens"] for k in comp_cats]
    
    results = []
    objects = cat_data["objects"][:n_samples]
    
    for obj in objects:
        prompt = make_prompt(obj, cat_data["relation"])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"]
        attn_mask = inputs["attention_mask"]
        
        input_device = next(model.parameters()).device
        input_ids = input_ids.to(input_device)
        attn_mask = attn_mask.to(input_device)
        
        target_ids, comp_ids = get_target_and_comp_ids(
            tokenizer, cat_data["target_tokens"], comp_tokens)
        
        # 获取pre/post norm hidden
        captured = {}
        def hook_pre_norm(module, input, output):
            captured['pre_norm'] = input[0].detach()
            captured['post_norm'] = (output[0] if isinstance(output, tuple) else output).detach()
        
        if final_norm is not None:
            h = final_norm.register_forward_hook(hook_pre_norm)
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attn_mask)
            h.remove()
            
            h_pre = captured['pre_norm'][0, -1].float().cpu().numpy()
            h_post = captured['post_norm'][0, -1].float().cpu().numpy()
        else:
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn_mask,
                           output_hidden_states=True)
            h_post = out.hidden_states[-1][0, -1].float().cpu().numpy()
            h_pre = out.hidden_states[-2][0, -1].float().cpu().numpy()
        
        # 计算D
        D_pre = compute_D_from_hidden(h_pre, W_U, target_ids, comp_ids)
        D_post = compute_D_from_hidden(h_post, W_U, target_ids, comp_ids)
        
        # 逐token读出分析: 看action token vs comp tokens的logit
        logits_pre = h_pre @ W_U.T
        logits_post = h_post @ W_U.T
        
        target_logits_pre = [float(logits_pre[tid]) for tid in target_ids if tid < len(logits_pre)]
        target_logits_post = [float(logits_post[tid]) for tid in target_ids if tid < len(logits_post)]
        comp_logits_pre = [float(logits_pre[cid]) for cid in comp_ids if cid < len(logits_pre)]
        comp_logits_post = [float(logits_post[cid]) for cid in comp_ids if cid < len(logits_post)]
        
        # RMSNorm对action方向的影响
        rms_val = float(np.sqrt(np.mean(h_pre ** 2) + 1e-5))
        
        # 对比实体类别
        entity_results = {}
        for ecat in ["fruit", "animal"]:
            e_data = CATEGORIES[ecat]
            e_target_ids, _ = get_target_and_comp_ids(
                tokenizer, e_data["target_tokens"], comp_tokens)
            e_D_pre = compute_D_from_hidden(h_pre, W_U, e_target_ids, comp_ids)
            e_D_post = compute_D_from_hidden(h_post, W_U, e_target_ids, comp_ids)
            entity_results[ecat] = {"D_pre": e_D_pre, "D_post": e_D_post}
        
        # 分析pre-norm到post-norm的D翻转条件
        if rmsnorm_w is not None:
            # 用gain-weighted方向分析
            h_normed = h_pre / rms_val
            h_gained = h_normed * rmsnorm_w
            
            # 各方向对D_pre的贡献
            d = len(h_pre)
            # 计算各维度的贡献
            dim_contrib_pre = h_pre * W_U[target_ids[0]] if len(target_ids) > 0 and target_ids[0] < W_U.shape[0] else np.zeros(d)
            dim_contrib_post = h_gained * W_U[target_ids[0]] if len(target_ids) > 0 and target_ids[0] < W_U.shape[0] else np.zeros(d)
            
            # 高贡献维度分析
            top_dims_pre = np.argsort(np.abs(dim_contrib_pre))[-20:][::-1]
            top_dims_post = np.argsort(np.abs(dim_contrib_post))[-20:][::-1]
            
            gain_flip_dims = np.sum((dim_contrib_pre > 0) != (dim_contrib_post > 0))
        else:
            dim_contrib_pre = dim_contrib_post = np.zeros(1)
            top_dims_pre = top_dims_post = []
            gain_flip_dims = 0
        
        results.append({
            "obj": obj,
            "D_pre": D_pre,
            "D_post": D_post,
            "sign_flipped": (D_pre * D_post < 0),
            "rms": rms_val,
            "target_logit_pre": float(np.mean(target_logits_pre)) if target_logits_pre else 0,
            "target_logit_post": float(np.mean(target_logits_post)) if target_logits_post else 0,
            "comp_logit_pre": float(np.mean(comp_logits_pre)) if comp_logits_pre else 0,
            "comp_logit_post": float(np.mean(comp_logits_post)) if comp_logits_post else 0,
            "entity_comparison": entity_results,
            "gain_flip_dims": int(gain_flip_dims),
        })
        
        print(f"  action/{obj}: D_pre={D_pre:.2f}, D_post={D_post:.2f}, "
              f"flipped={D_pre*D_post<0}, rms={rms_val:.2f}, "
              f"target_logit: {float(np.mean(target_logits_pre)):.2f}→{float(np.mean(target_logits_post)):.2f}, "
              f"comp_logit: {float(np.mean(comp_logits_pre)):.2f}→{float(np.mean(comp_logits_post)):.2f}")
    
    return results


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    n_samples = 8 if round_num == 1 else 12
    
    print(f"\n{'#'*60}")
    print(f"Phase 498 R{round_num}: RMSNorm Readout Geometry")
    print(f"Model: {model_name}, n_samples={n_samples}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}\n")
    
    # 加载模型
    t0 = time.time()
    model, tokenizer, device = load_model(model_name)
    print(f"Model loaded in {time.time()-t0:.1f}s")
    
    all_results = {
        "phase": 498,
        "round": round_num,
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_info": {
            "class": type(model).__name__,
            "n_layers": get_model_info(model, model_name).n_layers,
            "d_model": get_model_info(model, model_name).d_model,
        },
    }
    
    # Exp1
    try:
        exp1 = run_exp1_rmsnorm_math_decomposition(model, tokenizer, model_name, n_samples)
        all_results["exp1_rmsnorm_math"] = exp1
    except Exception as e:
        print(f"Exp1 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["exp1_error"] = str(e)
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Exp2
    try:
        exp2 = run_exp2_fixed_rmsnorm_control(model, tokenizer, model_name, n_samples)
        all_results["exp2_fixed_rmsnorm"] = exp2
    except Exception as e:
        print(f"Exp2 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["exp2_error"] = str(e)
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Exp3
    try:
        exp3 = run_exp3_mlp_norm_channel(model, tokenizer, model_name, n_samples)
        all_results["exp3_mlp_norm_channel"] = exp3
    except Exception as e:
        print(f"Exp3 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["exp3_error"] = str(e)
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Exp4
    try:
        exp4 = run_exp4_action_sign_flip(model, tokenizer, model_name, n_samples)
        all_results["exp4_action_sign_flip"] = exp4
    except Exception as e:
        print(f"Exp4 FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["exp4_error"] = str(e)
    
    # 保存结果
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / f"phase498_{model_name}_r{round_num}.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_file}")
    
    # 释放模型
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Phase 498 R{round_num} complete for {model_name}")


if __name__ == "__main__":
    main()
