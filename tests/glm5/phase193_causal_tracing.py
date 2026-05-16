"""
Phase 193: Causal Tracing via Activation Patching (Resample Ablation)
====================================================================
核心转向: 从"相关性"(head输出变化) 到 "因果性"(ablation后语义功能是否消失)

用户关键洞察:
- attention pattern ≠ computation (attention只决定"从哪读", OV circuit决定"算什么")
- 真正需要的是activation patching (因果干预), 不是统计差异
- 语义不是向量偏移, 而是"分布式程序"

方法: Resample Ablation
- Clean input: 有语义功能的句子 → P_clean
- Corrupted input: 没有语义功能的句子 → P_corrupt  
- Patch: 在clean run中, 将某组件输出替换为corrupted版本 → P_patched
- Causal Effect = (||P_clean - P_corrupt|| - ||P_patched - P_corrupt||) / ||P_clean - P_corrupt||
  = "这个组件对语义功能的因果贡献有多大"

实验:
Exp1: Layer-level Attention Resample Ablation — 每层attention的因果贡献
Exp2: Layer-level MLP Resample Ablation — 每层MLP的因果贡献
Exp3: Attention vs MLP 因果分解 — 哪类组件对哪种语义功能更关键
Exp4: Causal Layer Progression — 语义信息在哪些层被写入
Exp5: 跨功能因果重叠 — 哪些层是共享的vs专用的

15句对/功能, 共75句对, 数据量加大确保统计可靠性
"""

import sys
import os
# 强制stdout无缓冲
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

import gc
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from pathlib import Path

from model_utils import (load_model, get_model_info, get_layers, 
                          release_model, MODEL_CONFIGS)


# ===== 语义功能句对 (15 pairs/function, 比Phase 192增加50%) =====
SEMANTIC_PAIRS = {
    "negation": [
        ("The cat sleeps peacefully", "The cat does not sleep peacefully"),
        ("Birds fly south in winter", "Birds do not fly south in winter"),
        ("The door was closed", "The door was not closed"),
        ("She accepted the offer", "She did not accept the offer"),
        ("The machine works perfectly", "The machine does not work perfectly"),
        ("He finished the project", "He did not finish the project"),
        ("The system is stable", "The system is not stable"),
        ("Water boils at high temperature", "Water does not boil at high temperature"),
        ("They found the solution", "They did not find the solution"),
        ("The light was on", "The light was not on"),
        ("Money solves problems", "Money does not solve problems"),
        ("She believed the story", "She did not believe the story"),
        ("The car stopped quickly", "The car did not stop quickly"),
        ("He understood the message", "He did not understand the message"),
        ("The plan succeeded", "The plan did not succeed"),
    ],
    "tense": [
        ("The cat sleeps on the mat", "The cat slept on the mat"),
        ("She walks to the store", "She walked to the store"),
        ("He reads the book carefully", "He read the book carefully"),
        ("The train arrives at noon", "The train arrived at noon"),
        ("They build houses here", "They built houses here"),
        ("The river flows through town", "The river flowed through town"),
        ("She writes letters home", "She wrote letters home"),
        ("The bird sings every morning", "The bird sang every morning"),
        ("He drives to work daily", "He drove to work daily"),
        ("The children play outside", "The children played outside"),
        ("Water freezes in winter", "Water froze in winter"),
        ("She teaches mathematics", "She taught mathematics"),
        ("The wind blows from the north", "The wind blew from the north"),
        ("He knows the answer", "He knew the answer"),
        ("The sun rises early", "The sun rose early"),
    ],
    "role_binding": [
        ("The dog chased the cat", "The cat chased the dog"),
        ("The teacher praised the student", "The student praised the teacher"),
        ("The manager fired the employee", "The employee fired the manager"),
        ("The doctor examined the patient", "The patient examined the doctor"),
        ("The police arrested the suspect", "The suspect arrested the police"),
        ("The mother hugged the child", "The child hugged the mother"),
        ("The judge sentenced the criminal", "The criminal sentenced the judge"),
        ("The boss promoted the worker", "The worker promoted the boss"),
        ("The cat watched the bird", "The bird watched the cat"),
        ("The king rewarded the knight", "The knight rewarded the king"),
        ("The hunter tracked the deer", "The deer tracked the hunter"),
        ("The chef served the customer", "The customer served the chef"),
        ("The author thanked the editor", "The editor thanked the author"),
        ("The driver helped the passenger", "The passenger helped the driver"),
        ("The captain led the soldier", "The soldier led the captain"),
    ],
    "question": [
        ("The cat sleeps on the mat", "Does the cat sleep on the mat"),
        ("She walks to the store", "Does she walk to the store"),
        ("He reads the book", "Does he read the book"),
        ("The train arrives at noon", "Does the train arrive at noon"),
        ("They build houses here", "Do they build houses here"),
        ("The river flows through town", "Does the river flow through town"),
        ("She writes letters home", "Does she write letters home"),
        ("The bird sings every morning", "Does the bird sing every morning"),
        ("He drives to work daily", "Does he drive to work daily"),
        ("The children play outside", "Do the children play outside"),
        ("Water freezes in winter", "Does water freeze in winter"),
        ("She teaches mathematics", "Does she teach mathematics"),
        ("The wind blows from the north", "Does the wind blow from the north"),
        ("He knows the answer", "Does he know the answer"),
        ("The sun rises early", "Does the sun rise early"),
    ],
    "conditional": [
        ("The cat sleeps on the mat", "If the cat sleeps on the mat"),
        ("She walks to the store", "If she walks to the store"),
        ("He reads the book", "If he reads the book"),
        ("The train arrives at noon", "If the train arrives at noon"),
        ("They build houses here", "If they build houses here"),
        ("The river flows through town", "If the river flows through town"),
        ("She writes letters home", "If she writes letters home"),
        ("The bird sings every morning", "If the bird sings every morning"),
        ("He drives to work daily", "If he drives to work daily"),
        ("The children play outside", "If the children play outside"),
        ("Water freezes in winter", "If water freezes in winter"),
        ("She teaches mathematics", "If she teaches mathematics"),
        ("The wind blows from the north", "If the wind blows from the north"),
        ("He knows the answer", "If he knows the answer"),
        ("The sun rises early", "If the sun rises early"),
    ],
}


def load_model_bf16(model_name: str):
    """BF16 + device_map=auto 加载模型 (参考model_demo_bf16.py)"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    cfg = MODEL_CONFIGS[model_name]
    print(f"[bf16] Loading {model_name} (bfloat16 + device_map=auto)...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager",
    )
    model.eval()
    
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"[bf16] {model_name} loaded: device={device}, class={type(model).__name__}, "
          f"GPU={gpu_mem:.2f}GB")
    return model, tokenizer, device


def get_input_device(model):
    """获取输入tensor应放的设备"""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def record_component_outputs(model, input_ids, attention_mask, layers):
    """
    前向传播并记录所有attention和MLP的输出
    返回: {attn: {layer_idx: tensor}, mlp: {layer_idx: tensor}}
    """
    recorded = {'attn': {}, 'mlp': {}}
    hooks = []
    
    for li, layer in enumerate(layers):
        def make_attn_hook(idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    recorded['attn'][idx] = output[0].detach().clone()
                else:
                    recorded['attn'][idx] = output.detach().clone()
            return hook_fn
        
        def make_mlp_hook(idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    recorded['mlp'][idx] = output[0].detach().clone()
                else:
                    recorded['mlp'][idx] = output.detach().clone()
            return hook_fn
        
        h1 = layer.self_attn.register_forward_hook(make_attn_hook(li))
        h2 = layer.mlp.register_forward_hook(make_mlp_hook(li))
        hooks.extend([h1, h2])
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    for h in hooks:
        h.remove()
    
    logits = outputs.logits[0, -1, :]
    probs = F.softmax(logits.float(), dim=-1)
    
    return recorded, probs


def run_with_patch(model, input_ids, attention_mask, layers, 
                   patch_type, patch_layer, corrupted_output, 
                   patch_last_pos_only=False):
    """
    运行模型, 在指定层替换组件输出为corrupted版本 (Resample Ablation)
    
    Args:
        patch_type: 'attn' 或 'mlp'
        patch_layer: 要patch的层索引
        corrupted_output: 要注入的corrupted输出tensor
        patch_last_pos_only: 如果True, 只替换最后一个token位置(用于序列长度不匹配时)
    """
    device = input_ids.device
    
    if patch_last_pos_only:
        # 只替换最后一个位置的输出 — 适用于clean/corrupted序列长度不同的情况
        last_pos = attention_mask.sum().item() - 1  # 最后一个有效token的位置
        
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                out = output[0].clone()
            else:
                out = output.clone()
            
            # 只替换最后一个位置
            if corrupted_output.shape[1] > last_pos:
                out[:, last_pos, :] = corrupted_output[:, last_pos, :]
            
            if isinstance(output, tuple):
                return (out,) + output[1:]
            return out
    else:
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                return (corrupted_output.to(device),) + output[1:]
            return corrupted_output.to(device)
    
    if patch_type == 'attn':
        hook = layers[patch_layer].self_attn.register_forward_hook(hook_fn)
    elif patch_type == 'mlp':
        hook = layers[patch_layer].mlp.register_forward_hook(hook_fn)
    else:
        raise ValueError(f"Unknown patch_type: {patch_type}")
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    hook.remove()
    
    logits = outputs.logits[0, -1, :]
    probs = F.softmax(logits.float(), dim=-1)
    return probs


def compute_semantic_diff(probs_clean, probs_corrupt, top_k=100):
    """
    计算两个概率分布之间的语义差异 (L1距离, 基于top-k tokens)
    使用top-k避免logit噪声
    """
    # 找到两个分布中概率最高的k个token
    combined_top = torch.topk(torch.max(probs_clean, probs_corrupt), k=top_k).indices
    
    diff = (probs_clean[combined_top] - probs_corrupt[combined_top]).abs().sum().item()
    return diff


def run_exp1_exp2(model, tokenizer, model_name):
    """
    Exp1+2: Layer-level Resample Ablation (Attention + MLP)
    
    对每个语义功能, 每个句对:
    1. 运行clean → P_clean, 记录所有组件输出
    2. 运行corrupted → P_corrupt, 记录所有组件输出
    3. 对每层(attn/mlp): 在clean run中替换为corrupted版本 → P_patched
    4. Causal Effect = (diff_clean_corrupt - diff_patched_corrupt) / diff_clean_corrupt
    """
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    layers = get_layers(model)
    input_device = get_input_device(model)
    
    print(f"\n{'='*70}")
    print(f"Exp1+2: Layer-level Resample Ablation — {model_name}")
    print(f"  n_layers={n_layers}, d_model={info.d_model}")
    print(f"{'='*70}")
    
    results = {}
    
    for func_type, pairs in SEMANTIC_PAIRS.items():
        print(f"\n  --- {func_type}: {len(pairs)} pairs ---")
        t0_func = time.time()
        
        # 存储每个句对的因果效应
        attn_effects = defaultdict(list)  # {layer_idx: [effects across pairs]}
        mlp_effects = defaultdict(list)
        baseline_diffs = []
        
        for pi, (s_clean, s_corrupt) in enumerate(pairs):
            if pi % 3 == 0:
                print(f"    [{func_type}] {pi}/{len(pairs)}: {s_clean[:40]}...")
                sys.stdout.flush()
            
            # Tokenize
            inputs_c = tokenizer(s_clean, return_tensors="pt", truncation=True, max_length=64)
            inputs_r = tokenizer(s_corrupt, return_tensors="pt", truncation=True, max_length=64)
            ids_c = inputs_c["input_ids"].to(input_device)
            ids_r = inputs_r["input_ids"].to(input_device)
            mask_c = inputs_c["attention_mask"].to(input_device)
            mask_r = inputs_r["attention_mask"].to(input_device)
            
            # Step 1: 运行clean和corrupted, 记录所有组件输出
            try:
                clean_acts, probs_clean = record_component_outputs(model, ids_c, mask_c, layers)
                corrupt_acts, probs_corrupt = record_component_outputs(model, ids_r, mask_r, layers)
            except Exception as e:
                print(f"    ⚠ Recording failed for pair {pi}: {e}")
                continue
            
            # 计算baseline差异
            baseline_diff = compute_semantic_diff(probs_clean, probs_corrupt)
            if baseline_diff < 1e-6:
                # 两个分布几乎相同, 跳过
                continue
            baseline_diffs.append(baseline_diff)
            
            # Step 2: 对每层做resample ablation
            for li in range(n_layers):
                # Patch attention at layer li
                if li in clean_acts['attn'] and li in corrupt_acts['attn']:
                    try:
                        # 确保形状匹配
                        c_shape = clean_acts['attn'][li].shape
                        r_shape = corrupt_acts['attn'][li].shape
                        if c_shape == r_shape:
                            probs_patched = run_with_patch(
                                model, ids_c, mask_c, layers, 
                                'attn', li, corrupt_acts['attn'][li]
                            )
                        else:
                            # 序列长度不同时, 只替换最后一个token位置
                            probs_patched = run_with_patch(
                                model, ids_c, mask_c, layers,
                                'attn', li, corrupt_acts['attn'][li],
                                patch_last_pos_only=True
                            )
                        patched_diff = compute_semantic_diff(probs_patched, probs_corrupt)
                        effect = max(0, (baseline_diff - patched_diff) / baseline_diff)
                        attn_effects[li].append(effect)
                    except Exception as e:
                        if li == 0:
                            print(f"    ⚠ Attn patch failed at L{li}: {e}")
                
                # Patch MLP at layer li
                if li in clean_acts['mlp'] and li in corrupt_acts['mlp']:
                    try:
                        c_shape = clean_acts['mlp'][li].shape
                        r_shape = corrupt_acts['mlp'][li].shape
                        if c_shape == r_shape:
                            probs_patched = run_with_patch(
                                model, ids_c, mask_c, layers,
                                'mlp', li, corrupt_acts['mlp'][li]
                            )
                        else:
                            # 序列长度不同时, 只替换最后一个token位置
                            probs_patched = run_with_patch(
                                model, ids_c, mask_c, layers,
                                'mlp', li, corrupt_acts['mlp'][li],
                                patch_last_pos_only=True
                            )
                        patched_diff = compute_semantic_diff(probs_patched, probs_corrupt)
                        effect = max(0, (baseline_diff - patched_diff) / baseline_diff)
                        mlp_effects[li].append(effect)
                    except Exception as e:
                        if li == 0:
                            print(f"    ⚠ MLP patch failed at L{li}: {e}")
            
            # 释放GPU内存
            del clean_acts, corrupt_acts, probs_clean, probs_corrupt
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # 计算每层的平均因果效应
        attn_mean = {li: float(np.mean(effects)) for li, effects in attn_effects.items() if effects}
        mlp_mean = {li: float(np.mean(effects)) for li, effects in mlp_effects.items() if effects}
        
        results[func_type] = {
            'attn_causal': attn_mean,
            'mlp_causal': mlp_mean,
            'baseline_diffs': [float(x) for x in baseline_diffs],
            'mean_baseline_diff': float(np.mean(baseline_diffs)) if baseline_diffs else 0,
            'n_pairs': len(baseline_diffs),
        }
        
        # 打印Top-5最关键的层
        attn_top5 = sorted(attn_mean.items(), key=lambda x: x[1], reverse=True)[:5]
        mlp_top5 = sorted(mlp_mean.items(), key=lambda x: x[1], reverse=True)[:5]
        
        print(f"\n  [{func_type}] baseline_diff={np.mean(baseline_diffs):.4f} (n={len(baseline_diffs)})")
        print(f"  Attention Top-5 causal layers: {[(f'L{li}', f'{v:.3f}') for li, v in attn_top5]}")
        print(f"  MLP Top-5 causal layers: {[(f'L{li}', f'{v:.3f}') for li, v in mlp_top5]}")
        
        elapsed = time.time() - t0_func
        print(f"  Time: {elapsed:.1f}s")
    
    return results


def run_exp3(results_exp12, model_name):
    """Exp3: Attention vs MLP 因果分解"""
    print(f"\n{'='*70}")
    print(f"Exp3: Attention vs MLP 因果分解 — {model_name}")
    print(f"{'='*70}")
    
    for func_type, data in results_exp12.items():
        attn_total = sum(data['attn_causal'].values())
        mlp_total = sum(data['mlp_causal'].values())
        total = attn_total + mlp_total
        
        if total > 0:
            attn_pct = attn_total / total * 100
            mlp_pct = mlp_total / total * 100
        else:
            attn_pct = mlp_pct = 50.0
        
        # 找到关键层
        attn_key = sorted(data['attn_causal'].items(), key=lambda x: x[1], reverse=True)[:3]
        mlp_key = sorted(data['mlp_causal'].items(), key=lambda x: x[1], reverse=True)[:3]
        
        print(f"\n  {func_type}:")
        print(f"    Attention总贡献: {attn_pct:.1f}% | MLP总贡献: {mlp_pct:.1f}%")
        print(f"    Attention关键层: {[(f'L{li}', f'{v:.4f}') for li, v in attn_key]}")
        print(f"    MLP关键层: {[(f'L{li}', f'{v:.4f}') for li, v in mlp_key]}")
        
        results_exp12[func_type]['attn_pct'] = attn_pct
        results_exp12[func_type]['mlp_pct'] = mlp_pct


def run_exp4(results_exp12, model_name):
    """Exp4: Causal Layer Progression — 语义信息在哪些层被写入"""
    print(f"\n{'='*70}")
    print(f"Exp4: Causal Layer Progression — {model_name}")
    print(f"{'='*70}")
    
    for func_type, data in results_exp12.items():
        attn_c = data['attn_causal']
        mlp_c = data['mlp_causal']
        
        # 合并attention和MLP的因果贡献
        combined = defaultdict(float)
        for li, v in attn_c.items():
            combined[li] += v
        for li, v in mlp_c.items():
            combined[li] += v
        
        # 计算累积贡献
        sorted_layers = sorted(combined.keys())
        cumulative = []
        running = 0
        for li in sorted_layers:
            running += combined[li]
            cumulative.append((li, running))
        
        total_causal = running
        
        # 找到50%, 80%, 95%的层
        layers_50 = layers_80 = layers_95 = -1
        for li, cum in cumulative:
            if cum >= 0.5 * total_causal and layers_50 == -1:
                layers_50 = li
            if cum >= 0.8 * total_causal and layers_80 == -1:
                layers_80 = li
            if cum >= 0.95 * total_causal and layers_95 == -1:
                layers_95 = li
        
        print(f"\n  {func_type}:")
        print(f"    总因果贡献: {total_causal:.4f}")
        print(f"    50%因果在L{layers_50} | 80%在L{layers_80} | 95%在L{layers_95}")
        
        # 前5层 vs 中5层 vs 后5层的贡献
        n = max(sorted_layers) + 1 if sorted_layers else 0
        if n >= 15:
            early = sum(combined[li] for li in range(0, n//3))
            mid = sum(combined[li] for li in range(n//3, 2*n//3))
            late = sum(combined[li] for li in range(2*n//3, n))
            print(f"    浅层(0-{n//3-1}): {early:.4f} | 中层({n//3}-{2*n//3-1}): {mid:.4f} | "
                  f"深层({2*n//3}-{n-1}): {late:.4f}")
        
        results_exp12[func_type]['progression'] = {
            'total_causal': total_causal,
            'layers_50': layers_50,
            'layers_80': layers_80,
            'layers_95': layers_95,
        }


def run_exp5(results_exp12, model_name):
    """Exp5: 跨功能因果重叠 — 哪些层是共享的vs专用的"""
    print(f"\n{'='*70}")
    print(f"Exp5: 跨功能因果重叠 — {model_name}")
    print(f"{'='*70}")
    
    # 对每个功能, 找到top-30%因果层
    func_key_layers = {}
    for func_type, data in results_exp12.items():
        combined = defaultdict(float)
        for li, v in data['attn_causal'].items():
            combined[li] += v
        for li, v in data['mlp_causal'].items():
            combined[li] += v
        
        total = sum(combined.values())
        if total == 0:
            continue
        
        # 按贡献排序, 取top 30%的层
        sorted_layers = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        n_key = max(1, len(sorted_layers) // 3)
        key_set = set(li for li, v in sorted_layers[:n_key])
        func_key_layers[func_type] = key_set
    
    # 计算功能对之间的重叠
    func_types = list(func_key_layers.keys())
    print("\n  关键层重叠 (Top-30% causal layers):")
    
    overlap_matrix = {}
    for i, ft1 in enumerate(func_types):
        for j, ft2 in enumerate(func_types):
            if i >= j:
                continue
            s1 = func_key_layers[ft1]
            s2 = func_key_layers[ft2]
            if len(s1) == 0 or len(s2) == 0:
                jaccard = 0
            else:
                jaccard = len(s1 & s2) / len(s1 | s2)
            
            overlap_matrix[f"{ft1}_vs_{ft2}"] = jaccard
            print(f"    {ft1} vs {ft2}: Jaccard={jaccard:.3f} "
                  f"(shared={len(s1 & s2)}, {ft1}={len(s1)}, {ft2}={len(s2)})")
    
    # 找到专用层 (只在1个功能中关键的层)
    all_key_layers = set()
    for s in func_key_layers.values():
        all_key_layers.update(s)
    
    layer_owners = defaultdict(set)
    for ft, layers_set in func_key_layers.items():
        for li in layers_set:
            layer_owners[li].add(ft)
    
    dedicated = {ft: sum(1 for li in s if len(layer_owners[li]) == 1) 
                 for ft, s in func_key_layers.items()}
    universal = sum(1 for li in all_key_layers if len(layer_owners[li]) >= 3)
    
    print(f"\n  专用关键层数: {dedicated}")
    print(f"  通用关键层数 (3+功能): {universal}")
    
    results_exp12['exp5'] = {
        'overlap_jaccard': overlap_matrix,
        'dedicated_layers': dedicated,
        'universal_layers': universal,
    }


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    
    print(f"\n{'='*70}")
    print(f"Phase 193: Causal Tracing via Activation Patching (Resample Ablation)")
    print(f"Model: {model_name}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}")
    
    # ===== 1. 加载模型 =====
    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    t_load = time.time() - t0
    print(f"  Model loaded in {t_load:.1f}s")
    
    info = get_model_info(model, model_name)
    print(f"  class={info.model_class}, n_layers={info.n_layers}, d_model={info.d_model}")
    
    # ===== 2. 运行Exp1+2: Layer-level Resample Ablation =====
    t0 = time.time()
    results = run_exp1_exp2(model, tokenizer, model_name)
    t_exp12 = time.time() - t0
    print(f"\n  Exp1+2 completed in {t_exp12:.1f}s")
    
    # ===== 3. 运行Exp3: Attention vs MLP分解 =====
    run_exp3(results, model_name)
    
    # ===== 4. 运行Exp4: Causal Layer Progression =====
    run_exp4(results, model_name)
    
    # ===== 5. 运行Exp5: 跨功能因果重叠 =====
    run_exp5(results, model_name)
    
    # ===== 6. 打印综合结果 =====
    print(f"\n\n{'='*70}")
    print(f"Phase 193 综合结果 — {model_name}")
    print(f"{'='*70}")
    
    for func_type, data in results.items():
        if func_type == 'exp5':
            continue
        
        print(f"\n  === {func_type} ===")
        
        # Top-5 attention causal layers
        attn_top5 = sorted(data['attn_causal'].items(), key=lambda x: x[1], reverse=True)[:5]
        mlp_top5 = sorted(data['mlp_causal'].items(), key=lambda x: x[1], reverse=True)[:5]
        
        print(f"  Attention Top-5: {[(f'L{li}', f'{v:.4f}') for li, v in attn_top5]}")
        print(f"  MLP Top-5: {[(f'L{li}', f'{v:.4f}') for li, v in mlp_top5]}")
        
        if 'attn_pct' in data:
            print(f"  Attention%: {data['attn_pct']:.1f}% | MLP%: {data['mlp_pct']:.1f}%")
        
        if 'progression' in data:
            p = data['progression']
            print(f"  50% causal by L{p['layers_50']} | 80% by L{p['layers_80']} | "
                  f"95% by L{p['layers_95']}")
    
    # ===== 7. 保存结果 =====
    output_dir = Path("tests/glm5_temp")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M")
    output_file = output_dir / f"phase193_{model_name}_{timestamp}.json"
    
    # 转换为可序列化格式
    serializable = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serializable[k] = {}
            for k2, v2 in v.items():
                if isinstance(v2, dict):
                    serializable[k][k2] = {str(k3): v3 for k3, v3 in v2.items()}
                else:
                    serializable[k][k2] = v2
        else:
            serializable[k] = v
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    
    print(f"\n  Results saved to {output_file}")
    
    # ===== 8. 释放模型 =====
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"\nPhase 193 COMPLETE for {model_name}")
    print(f"Total time: {time.time() - t0:.1f}s (load={t_load:.1f}s, exp1+2={t_exp12:.1f}s)")


if __name__ == "__main__":
    main()
