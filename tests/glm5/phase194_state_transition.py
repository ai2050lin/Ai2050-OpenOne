"""
Phase 194: State Transition Analysis — ΔS_l = S_{l+1} - S_l
============================================================

核心转向: 从"状态几何"(S_l) 到 "状态转移"(ΔS_l)

用户关键洞察:
- hidden state可能只是"缓存", 不是"语义本体"
- 真正语义存在于ΔS_l (变化量), 不是S_l本身
- 语义 = 对状态施加的条件变换, 不是状态的几何位置
- Transformer更像迭代状态精化, 不是几何平移

5个实验:
Exp1: ΔS Decomposition — 每层的ΔS有多少来自attention vs MLP
      ΔS_l = S_{l+1} - S_l = (attn_write + mlp_write + ln_correction)
Exp2: Semantic ΔS — 语义变体的ΔS是否有特异性方向?
      ΔS_clean vs ΔS_corrupt: 哪些层的ΔS在语义变体间有显著差异?
Exp3: ΔS Non-Commutativity — 不同语义功能的ΔS是否非交换?
      F_i ∘ F_j vs F_j ∘ F_i: ΔS的顺序是否影响结果?
Exp4: ΔS Operator Algebra — ΔS向量是否形成可识别的代数结构?
      交换子 [F_i, F_j] 的范数, ΔS的封闭性
Exp5: Cross-Model ΔS Equivalence — 不同模型的ΔS结构是否等价?

20句对/功能 (加大数据量), 对device_map=auto模型用10句对+层采样
"""

import sys
import os
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

from model_utils import (get_model_info, get_layers, release_model, MODEL_CONFIGS)


# ===== 语义功能句对 (20 pairs/function, 加大数据量) =====
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
        ("The bridge collapsed", "The bridge did not collapse"),
        ("The student passed the exam", "The student did not pass the exam"),
        ("The company grew rapidly", "The company did not grow rapidly"),
        ("The experiment worked", "The experiment did not work"),
        ("The weather improved", "The weather did not improve"),
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
        ("She runs five miles daily", "She ran five miles daily"),
        ("The boat sails across the lake", "The boat sailed across the lake"),
        ("He paints beautiful pictures", "He painted beautiful pictures"),
        ("The bell rings at midnight", "The bell rang at midnight"),
        ("She grows roses in spring", "She grew roses in spring"),
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
        ("The fox chased the rabbit", "The rabbit chased the fox"),
        ("The guard protected the prisoner", "The prisoner protected the guard"),
        ("The coach trained the athlete", "The athlete trained the coach"),
        ("The officer questioned the witness", "The witness questioned the officer"),
        ("The nurse cared for the patient", "The patient cared for the nurse"),
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
        ("She runs five miles daily", "Does she run five miles daily"),
        ("The boat sails across the lake", "Does the boat sail across the lake"),
        ("He paints beautiful pictures", "Does he paint beautiful pictures"),
        ("The bell rings at midnight", "Does the bell ring at midnight"),
        ("She grows roses in spring", "Does she grow roses in spring"),
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
        ("She runs five miles daily", "If she runs five miles daily"),
        ("The boat sails across the lake", "If the boat sails across the lake"),
        ("He paints beautiful pictures", "If he paints beautiful pictures"),
        ("The bell rings at midnight", "If the bell rings at midnight"),
        ("She grows roses in spring", "If she grows roses in spring"),
    ],
}


def load_model_bf16(model_name: str):
    """BF16 + device_map=auto 加载模型 (参考model_demo_bf16.py)"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    cfg = MODEL_CONFIGS[model_name]
    print(f"[bf16] Loading {model_name} (bfloat16 + device_map=auto)...")
    sys.stdout.flush()
    
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
    sys.stdout.flush()
    return model, tokenizer, device


def get_input_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def record_residual_and_components(model, input_ids, attention_mask, layers):
    """
    记录每层的residual stream状态和组件写入
    
    返回:
        residual_stream: {layer_idx: tensor[1, seq_len, d_model]} — 每层输出的residual
        attn_writes: {layer_idx: tensor[1, seq_len, d_model]} — 每层attention的写入
        mlp_writes: {layer_idx: tensor[1, seq_len, d_model]} — 每层MLP的写入
        probs: tensor[vocab_size] — 最终概率分布
    """
    n_layers = len(layers)
    residual_stream = {}
    attn_writes = {}
    mlp_writes = {}
    hooks = []
    
    for li, layer in enumerate(layers):
        def make_residual_hook(idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    residual_stream[idx] = output[0].detach().float().cpu()
                else:
                    residual_stream[idx] = output.detach().float().cpu()
            return hook_fn
        
        def make_attn_hook(idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    attn_writes[idx] = output[0].detach().float().cpu()
                else:
                    attn_writes[idx] = output.detach().float().cpu()
            return hook_fn
        
        def make_mlp_hook(idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    mlp_writes[idx] = output[0].detach().float().cpu()
                else:
                    mlp_writes[idx] = output.detach().float().cpu()
            return hook_fn
        
        # Hook在层输出、attention输出、MLP输出上
        h1 = layer.register_forward_hook(make_residual_hook(li))
        h2 = layer.self_attn.register_forward_hook(make_attn_hook(li))
        h3 = layer.mlp.register_forward_hook(make_mlp_hook(li))
        hooks.extend([h1, h2, h3])
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    for h in hooks:
        h.remove()
    
    torch.cuda.empty_cache()
    
    logits = outputs.logits[0, -1, :]
    probs = F.softmax(logits.float(), dim=-1)
    
    return residual_stream, attn_writes, mlp_writes, probs


def compute_delta_S(residual_stream, last_token_only=True):
    """
    计算状态转移向量: ΔS_l = S_{l+1} - S_l
    
    Args:
        residual_stream: {layer_idx: tensor[1, seq_len, d_model]}
        last_token_only: 如果True, 只取最后一个token位置
    
    Returns:
        delta_S: {layer_idx: tensor[d_model]} — 每层的ΔS (相对前一层)
    """
    delta_S = {}
    sorted_layers = sorted(residual_stream.keys())
    
    for i in range(len(sorted_layers) - 1):
        li = sorted_layers[i]
        li_next = sorted_layers[i + 1]
        
        s_curr = residual_stream[li]
        s_next = residual_stream[li_next]
        
        if last_token_only:
            # 取最后一个有效token位置
            delta = (s_next[0, -1, :] - s_curr[0, -1, :]).numpy()
        else:
            delta = (s_next - s_curr).numpy()
        
        delta_S[li] = delta
    
    return delta_S


def run_exp12(model, tokenizer, model_name):
    """
    Exp1: ΔS Decomposition — 每层ΔS有多少来自Attention vs MLP
    Exp2: Semantic ΔS — 语义变体的ΔS是否有特异性方向?
    """
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    layers = get_layers(model)
    input_device = get_input_device(model)
    
    # 对device_map=auto模型用10句对+层采样
    max_pairs = 20 if model_name == 'qwen3' else 10
    
    print(f"\n{'='*70}")
    print(f"Exp1+2: State Transition Analysis — {model_name}")
    print(f"  n_layers={n_layers}, d_model={info.d_model}, max_pairs={max_pairs}")
    print(f"{'='*70}")
    sys.stdout.flush()
    
    results = {}
    
    for func_type, pairs in SEMANTIC_PAIRS.items():
        print(f"\n  --- {func_type}: {len(pairs)} pairs (using {max_pairs}) ---")
        sys.stdout.flush()
        t0_func = time.time()
        
        # 收集所有句对的ΔS
        all_clean_deltaS = defaultdict(list)   # {layer: [delta_S vectors]}
        all_corrupt_deltaS = defaultdict(list)
        all_attn_norms = defaultdict(list)     # {layer: [||attn_write||]}
        all_mlp_norms = defaultdict(list)      # {layer: [||mlp_write||]}
        all_delta_norms = defaultdict(list)    # {layer: [||ΔS||]}
        
        # 用于计算ΔS在语义变体间的差异 (Exp2)
        all_clean_deltaS_stack = defaultdict(list)
        all_corrupt_deltaS_stack = defaultdict(list)
        
        for pi, (s_clean, s_corrupt) in enumerate(pairs):
            if pi >= max_pairs:
                break
            if pi % 2 == 0:
                elapsed = time.time() - t0_func
                print(f"    [{func_type}] {pi}/{max_pairs} ({elapsed:.0f}s): {s_clean[:35]}...")
                sys.stdout.flush()
            
            # Tokenize
            inputs_c = tokenizer(s_clean, return_tensors="pt", truncation=True, max_length=64)
            inputs_r = tokenizer(s_corrupt, return_tensors="pt", truncation=True, max_length=64)
            ids_c = inputs_c["input_ids"].to(input_device)
            ids_r = inputs_r["input_ids"].to(input_device)
            mask_c = inputs_c["attention_mask"].to(input_device)
            mask_r = inputs_r["attention_mask"].to(input_device)
            
            try:
                # 记录clean和corrupted的residual stream + 组件写入
                rs_clean, attn_clean, mlp_clean, probs_clean = \
                    record_residual_and_components(model, ids_c, mask_c, layers)
                rs_corrupt, attn_corrupt, mlp_corrupt, probs_corrupt = \
                    record_residual_and_components(model, ids_r, mask_r, layers)
            except Exception as e:
                print(f"    ⚠ Recording failed for pair {pi}: {e}")
                sys.stdout.flush()
                continue
            
            # 计算ΔS
            deltaS_clean = compute_delta_S(rs_clean, last_token_only=True)
            deltaS_corrupt = compute_delta_S(rs_corrupt, last_token_only=True)
            
            # Exp1: 分解ΔS的来源 (attention vs MLP)
            for li in deltaS_clean.keys():
                ds = deltaS_clean[li]
                ds_norm = float(np.linalg.norm(ds))
                all_delta_norms[li].append(ds_norm)
                
                # Attention写入的范数
                if li in attn_clean:
                    attn_norm = float(np.linalg.norm(attn_clean[li][0, -1, :].numpy()))
                    all_attn_norms[li].append(attn_norm)
                
                # MLP写入的范数
                if li in mlp_clean:
                    mlp_norm = float(np.linalg.norm(mlp_clean[li][0, -1, :].numpy()))
                    all_mlp_norms[li].append(mlp_norm)
                
                # 存储ΔS向量
                all_clean_deltaS_stack[li].append(ds)
                all_corrupt_deltaS_stack[li].append(deltaS_corrupt[li])
            
            # 释放内存
            del rs_clean, rs_corrupt, attn_clean, attn_corrupt, mlp_clean, mlp_corrupt
            del probs_clean, probs_corrupt
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
        
        # 汇总Exp1结果: 每层的ΔS分解
        delta_decomp = {}
        for li in sorted(all_delta_norms.keys()):
            mean_delta = np.mean(all_delta_norms[li])
            mean_attn = np.mean(all_attn_norms[li]) if li in all_attn_norms else 0
            mean_mlp = np.mean(all_mlp_norms[li]) if li in all_mlp_norms else 0
            
            total_write = mean_attn + mean_mlp
            attn_frac = mean_attn / total_write if total_write > 0 else 0
            mlp_frac = mean_mlp / total_write if total_write > 0 else 0
            
            delta_decomp[li] = {
                'delta_norm': float(mean_delta),
                'attn_norm': float(mean_attn),
                'mlp_norm': float(mean_mlp),
                'attn_frac': float(attn_frac),
                'mlp_frac': float(mlp_frac),
            }
        
        # Exp2: 语义ΔS特异性 — ΔS在clean vs corrupt间的方向差异
        semantic_delta_specificity = {}
        for li in sorted(all_clean_deltaS_stack.keys()):
            clean_deltas = np.array(all_clean_deltaS_stack[li])  # [n_pairs, d_model]
            corrupt_deltas = np.array(all_corrupt_deltaS_stack[li])
            
            if clean_deltas.shape[0] < 3:
                continue
            
            # 计算clean ΔS和corrupt ΔS的平均方向
            clean_mean = np.mean(clean_deltas, axis=0)
            corrupt_mean = np.mean(corrupt_deltas, axis=0)
            
            clean_mean_norm = np.linalg.norm(clean_mean)
            corrupt_mean_norm = np.linalg.norm(corrupt_mean)
            
            if clean_mean_norm > 1e-6 and corrupt_mean_norm > 1e-6:
                # 余弦相似度: clean和corrupt的ΔS方向是否不同?
                cos_similarity = float(np.dot(clean_mean, corrupt_mean) / 
                                       (clean_mean_norm * corrupt_mean_norm))
                # ΔS差异的范数
                delta_diff = np.linalg.norm(clean_mean - corrupt_mean)
                
                # ΔS特异性分数: 方向差异 / 平均范数
                specificity = delta_diff / ((clean_mean_norm + corrupt_mean_norm) / 2 + 1e-10)
            else:
                cos_similarity = 0
                specificity = 0
            
            semantic_delta_specificity[li] = {
                'cos_sim': float(cos_similarity),
                'specificity': float(specificity),
                'clean_delta_norm': float(clean_mean_norm),
                'corrupt_delta_norm': float(corrupt_mean_norm),
            }
        
        # 存储ΔS向量用于后续实验
        results[func_type] = {
            'delta_decomp': delta_decomp,
            'semantic_specificity': semantic_delta_specificity,
            'n_pairs': min(len(pairs), max_pairs),
        }
        
        # 打印Exp1结果: ΔS分解
        print(f"\n  [{func_type}] Exp1: ΔS Decomposition (Top-5 layers by ΔS norm):")
        top5_layers = sorted(delta_decomp.items(), key=lambda x: x[1]['delta_norm'], reverse=True)[:5]
        for li, d in top5_layers:
            print(f"    L{li}: ΔS={d['delta_norm']:.2f}, "
                  f"attn={d['attn_frac']:.1%}, mlp={d['mlp_frac']:.1%}")
        
        # 打印Exp2结果: 语义特异性
        print(f"  [{func_type}] Exp2: Semantic ΔS Specificity (Top-5):")
        top5_spec = sorted(semantic_delta_specificity.items(), 
                          key=lambda x: x[1]['specificity'], reverse=True)[:5]
        for li, d in top5_spec:
            print(f"    L{li}: specificity={d['specificity']:.3f}, "
                  f"cos_sim={d['cos_sim']:.3f}")
        
        elapsed = time.time() - t0_func
        print(f"  Time: {elapsed:.1f}s")
        sys.stdout.flush()
    
    return results


def run_exp3(results_exp12, model_name):
    """
    Exp3: ΔS Non-Commutativity — 不同语义功能的ΔS是否非交换?
    
    测量: [F_neg, F_tense] = F_neg ∘ F_tense - F_tense ∘ F_neg
    即: ΔS_neg在tense的ΔS方向上的投影 vs ΔS_tense在neg的ΔS方向上的投影
    """
    print(f"\n{'='*70}")
    print(f"Exp3: ΔS Non-Commutativity — {model_name}")
    print(f"{'='*70}")
    sys.stdout.flush()
    
    # 从Exp1+2的结果中无法直接获取ΔS向量, 需要重新计算
    # 但我们可以用Exp1+2中存储的方向信息来近似
    # 这里用一个更直接的方法: 比较不同功能的ΔS方向之间的余弦
    
    func_types = list(results_exp12.keys())
    n_funcs = len(func_types)
    
    # 收集每个功能在每个层的semantic specificity
    # 用它来量化"这个功能在这一层的ΔS有多独特"
    layer_specificities = defaultdict(dict)
    for ft in func_types:
        spec = results_exp12[ft]['semantic_specificity']
        for li, d in spec.items():
            layer_specificities[li][ft] = d['specificity']
    
    # 非交换性近似: 如果功能A在层l有高特异性, 功能B在层l也有高特异性,
    # 但A在l+1的特异性和B在l+1的特异性不同 → 非交换
    # 更好的方法: 比较ΔS方向的差异
    
    # 简化: 用ΔS范数比来近似非交换性
    # 如果功能A的ΔS范数在层l很大但B很小, 而在层m相反 → 操作非交换
    delta_norms = defaultdict(dict)
    for ft in func_types:
        decomp = results_exp12[ft]['delta_decomp']
        for li, d in decomp.items():
            delta_norms[li][ft] = d['delta_norm']
    
    # 计算跨功能的ΔS范数相关矩阵
    print(f"\n  Cross-function ΔS norm correlation:")
    all_layers = sorted(delta_norms.keys())
    
    for i, ft1 in enumerate(func_types):
        for j, ft2 in enumerate(func_types):
            if j <= i:
                continue
            # 收集两个功能在所有层的ΔS范数
            norms1 = []
            norms2 = []
            for li in all_layers:
                if ft1 in delta_norms[li] and ft2 in delta_norms[li]:
                    norms1.append(delta_norms[li][ft1])
                    norms2.append(delta_norms[li][ft2])
            
            if len(norms1) > 5:
                corr = float(np.corrcoef(norms1, norms2)[0, 1])
                print(f"    {ft1} vs {ft2}: ΔS norm corr={corr:.3f}")
    
    # 计算ΔS特异性在不同层之间的排序差异 → 非交换性的间接证据
    print(f"\n  ΔS specificity ranking per function (top-3 layers):")
    for ft in func_types:
        spec = results_exp12[ft]['semantic_specificity']
        top3 = sorted(spec.items(), key=lambda x: x[1]['specificity'], reverse=True)[:3]
        top3_str = ', '.join(f'L{li}: {d["specificity"]:.3f}' for li, d in top3)
        print(f"    {ft}: {top3_str}")
    
    sys.stdout.flush()
    return {'note': 'Exp3 needs ΔS vectors for full non-commutativity analysis'}


def run_exp4(model, tokenizer, model_name):
    """
    Exp4: ΔS Operator Algebra — ΔS向量是否形成可识别的代数结构?
    
    核心测试: 
    1. ΔS的封闭性: F_i(ΔS_j) 是否仍然在ΔS空间中?
    2. 交换子: ||ΔS_i ∘ ΔS_j - ΔS_j ∘ ΔS_i|| 
    3. ΔS的线性独立性: 不同功能的ΔS是否张成不同的子空间?
    """
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    layers = get_layers(model)
    input_device = get_input_device(model)
    max_pairs = 20 if model_name == 'qwen3' else 10
    
    print(f"\n{'='*70}")
    print(f"Exp4: ΔS Operator Algebra — {model_name}")
    print(f"{'='*70}")
    sys.stdout.flush()
    
    # 收集每个功能在每个层的平均ΔS向量
    func_delta_means = {}  # {func_type: {layer: mean_ΔS_vector}}
    
    for func_type, pairs in SEMANTIC_PAIRS.items():
        print(f"\n  [{func_type}] Computing mean ΔS vectors...")
        sys.stdout.flush()
        
        delta_S_all = defaultdict(list)
        
        for pi, (s_clean, s_corrupt) in enumerate(pairs):
            if pi >= max_pairs:
                break
            
            inputs_c = tokenizer(s_clean, return_tensors="pt", truncation=True, max_length=64)
            inputs_r = tokenizer(s_corrupt, return_tensors="pt", truncation=True, max_length=64)
            ids_c = inputs_c["input_ids"].to(input_device)
            ids_r = inputs_r["attention_mask"].to(input_device)
            mask_c = inputs_c["attention_mask"].to(input_device)
            ids_r_ids = inputs_r["input_ids"].to(input_device)
            
            try:
                rs_clean, _, _, _ = record_residual_and_components(model, ids_c, mask_c, layers)
            except Exception as e:
                continue
            
            deltaS = compute_delta_S(rs_clean, last_token_only=True)
            for li, ds in deltaS.items():
                delta_S_all[li].append(ds)
            
            del rs_clean
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # 计算平均ΔS
        mean_delta = {}
        for li, deltas in delta_S_all.items():
            mean_delta[li] = np.mean(deltas, axis=0)
        
        func_delta_means[func_type] = mean_delta
    
    # 测试1: ΔS的线性独立性
    print(f"\n  --- Test 1: ΔS Linear Independence ---")
    sys.stdout.flush()
    
    # 选择中间层(信息最丰富的层)
    mid_layer = n_layers // 2
    test_layers = sorted(set([0, mid_layer // 2, mid_layer, mid_layer + mid_layer // 2, n_layers - 2]))
    test_layers = [li for li in test_layers if li in func_delta_means.get('negation', {})]
    
    for li in test_layers[:3]:
        # 收集所有功能在这一层的ΔS向量
        vecs = []
        func_names = []
        for ft in SEMANTIC_PAIRS.keys():
            if li in func_delta_means[ft]:
                vecs.append(func_delta_means[ft][li])
                func_names.append(ft)
        
        if len(vecs) < 2:
            continue
        
        mat = np.array(vecs)  # [n_funcs, d_model]
        
        # 计算两两余弦相似度
        print(f"    Layer {li}: Pairwise cosine similarity of ΔS vectors:")
        for i in range(len(func_names)):
            for j in range(i + 1, len(func_names)):
                cos = float(np.dot(vecs[i], vecs[j]) / 
                           (np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[j]) + 1e-10))
                print(f"      {func_names[i]} vs {func_names[j]}: cos={cos:.4f}")
        
        # SVD分析: ΔS张成的子空间维度
        mat_centered = mat - np.mean(mat, axis=0, keepdims=True)
        if mat_centered.shape[0] > 1:
            U, S, Vt = np.linalg.svd(mat_centered, full_matrices=False)
            # 有效维度: 贡献>5%的奇异值数量
            S_norm = S / (np.sum(S) + 1e-10)
            eff_dim = np.sum(S_norm > 0.05)
            print(f"    Layer {li}: ΔS effective dimensionality={eff_dim} "
                  f"(of {len(func_names)} functions)")
            print(f"    Singular values: {S[:5].round(3)}")
    
    # 测试2: 交换子范数 [F_i, F_j] = ΔS_i - ΔS_j ≠ 0
    print(f"\n  --- Test 2: ΔS Commutator Norms ---")
    sys.stdout.flush()
    
    func_list = list(SEMANTIC_PAIRS.keys())
    commutator_results = {}
    
    for li in test_layers[:3]:
        print(f"    Layer {li}:")
        for i in range(len(func_list)):
            for j in range(i + 1, len(func_list)):
                ft1, ft2 = func_list[i], func_list[j]
                if li in func_delta_means[ft1] and li in func_delta_means[ft2]:
                    ds1 = func_delta_means[ft1][li]
                    ds2 = func_delta_means[ft2][li]
                    
                    # "交换子"范数: ||ΔS_1 - ΔS_2||
                    comm_norm = float(np.linalg.norm(ds1 - ds2))
                    # 归一化: 除以平均范数
                    avg_norm = (np.linalg.norm(ds1) + np.linalg.norm(ds2)) / 2
                    norm_comm = comm_norm / (avg_norm + 1e-10)
                    
                    commutator_results[(ft1, ft2, li)] = {
                        'raw': comm_norm,
                        'normalized': norm_comm,
                    }
                    print(f"      [{ft1}, {ft2}]: ||comm||={comm_norm:.3f}, "
                          f"normalized={norm_comm:.3f}")
    
    sys.stdout.flush()
    
    # 测试3: ΔS在层间的传播 — F_i(L_k) vs F_i(L_{k+1}) 的相似度
    print(f"\n  --- Test 3: ΔS Cross-Layer Consistency ---")
    sys.stdout.flush()
    
    for ft in SEMANTIC_PAIRS.keys():
        delta_means = func_delta_means[ft]
        sorted_layers = sorted(delta_means.keys())
        
        cos_vals = []
        for idx in range(len(sorted_layers) - 1):
            li1, li2 = sorted_layers[idx], sorted_layers[idx + 1]
            if li1 in delta_means and li2 in delta_means:
                v1 = delta_means[li1]
                v2 = delta_means[li2]
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 > 1e-6 and n2 > 1e-6:
                    cos_vals.append(float(np.dot(v1, v2) / (n1 * n2)))
        
        if cos_vals:
            print(f"    {ft}: mean cross-layer cos={np.mean(cos_vals):.3f}, "
                  f"std={np.std(cos_vals):.3f}")
    
    sys.stdout.flush()
    return {
        'commutator_norms': {f"{k[0]}_{k[1]}_L{k[2]}": v 
                           for k, v in commutator_results.items()},
        'func_delta_means_layers': {ft: list(delta_means.keys()) 
                                   for ft, delta_means in func_delta_means.items()},
    }


def run_exp5(results_exp12_all_models):
    """
    Exp5: Cross-Model ΔS Equivalence — 不同模型的ΔS结构是否等价?
    """
    print(f"\n{'='*70}")
    print(f"Exp5: Cross-Model ΔS Equivalence")
    print(f"{'='*70}")
    sys.stdout.flush()
    
    # 比较不同模型的ΔS分解模式
    models = list(results_exp12_all_models.keys())
    
    if len(models) < 2:
        print("  Need at least 2 models for comparison")
        return {}
    
    print(f"\n  Cross-model Attention fraction in ΔS:")
    print(f"  {'Function':<15} " + " ".join(f"{m:>10}" for m in models))
    print(f"  {'-'*15} " + " ".join(f"{'-'*10}" for m in models))
    
    for ft in SEMANTIC_PAIRS.keys():
        attn_fracs = []
        for m in models:
            if ft in results_exp12_all_models[m]:
                decomp = results_exp12_all_models[m][ft]['delta_decomp']
                # 平均所有层的attention分数
                attn_fs = [d['attn_frac'] for d in decomp.values()]
                avg_attn = np.mean(attn_fs) if attn_fs else 0
                attn_fracs.append(f"{avg_attn:.1%}")
            else:
                attn_fracs.append("N/A")
        print(f"  {ft:<15} " + " ".join(f"{v:>10}" for v in attn_fracs))
    
    # 比较MLP分数
    print(f"\n  Cross-model MLP fraction in ΔS:")
    print(f"  {'Function':<15} " + " ".join(f"{m:>10}" for m in models))
    print(f"  {'-'*15} " + " ".join(f"{'-'*10}" for m in models))
    
    for ft in SEMANTIC_PAIRS.keys():
        mlp_fracs = []
        for m in models:
            if ft in results_exp12_all_models[m]:
                decomp = results_exp12_all_models[m][ft]['delta_decomp']
                mlp_fs = [d['mlp_frac'] for d in decomp.values()]
                avg_mlp = np.mean(mlp_fs) if mlp_fs else 0
                mlp_fracs.append(f"{avg_mlp:.1%}")
            else:
                mlp_fracs.append("N/A")
        print(f"  {ft:<15} " + " ".join(f"{v:>10}" for v in mlp_fracs))
    
    # 比较semantic specificity
    print(f"\n  Cross-model ΔS semantic specificity (avg top-3 layers):")
    print(f"  {'Function':<15} " + " ".join(f"{m:>10}" for m in models))
    print(f"  {'-'*15} " + " ".join(f"{'-'*10}" for m in models))
    
    for ft in SEMANTIC_PAIRS.keys():
        specs = []
        for m in models:
            if ft in results_exp12_all_models[m]:
                spec = results_exp12_all_models[m][ft]['semantic_specificity']
                top3 = sorted(spec.items(), key=lambda x: x[1]['specificity'], reverse=True)[:3]
                avg_spec = np.mean([d['specificity'] for _, d in top3]) if top3 else 0
                specs.append(f"{avg_spec:.3f}")
            else:
                specs.append("N/A")
        print(f"  {ft:<15} " + " ".join(f"{v:>10}" for v in specs))
    
    sys.stdout.flush()
    return {'note': 'Cross-model comparison complete'}


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}. Choose from: {list(MODEL_CONFIGS.keys())}")
        return
    
    t_start = time.time()
    
    print(f"\n{'='*70}")
    print(f"Phase 194: State Transition Analysis — ΔS_l = S_{{l+1}} - S_l")
    print(f"Model: {model_name}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}")
    sys.stdout.flush()
    
    # Load model
    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    t_load = time.time() - t0
    
    info = get_model_info(model, model_name)
    print(f"  Model loaded in {t_load:.1f}s")
    print(f"  class={info.model_class}, n_layers={info.n_layers}, d_model={info.d_model}")
    sys.stdout.flush()
    
    # Exp1+2: ΔS Decomposition + Semantic ΔS
    t0 = time.time()
    results_exp12 = run_exp12(model, tokenizer, model_name)
    t_exp12 = time.time() - t0
    
    # Exp3: ΔS Non-Commutativity (simplified)
    results_exp3 = run_exp3(results_exp12, model_name)
    
    # Exp4: ΔS Operator Algebra
    t0 = time.time()
    results_exp4 = run_exp4(model, tokenizer, model_name)
    t_exp4 = time.time() - t0
    
    # Save results
    all_results = {
        'exp12': results_exp12,
        'exp3': results_exp3,
        'exp4': results_exp4,
        'meta': {
            'model': model_name,
            'n_layers': info.n_layers,
            'd_model': info.d_model,
            't_load': round(t_load, 1),
            't_exp12': round(t_exp12, 1),
            't_exp4': round(t_exp4, 1),
        }
    }
    
    # JSON序列化辅助
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    all_results = make_serializable(all_results)
    
    out_dir = Path("tests/glm5_temp")
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M')
    out_file = out_dir / f"phase194_{model_name}_{timestamp}.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n  Results saved to {out_file}")
    
    # ===== 综合结果 =====
    print(f"\n{'='*70}")
    print(f"Phase 194 综合结果 — {model_name}")
    print(f"{'='*70}")
    
    for ft in SEMANTIC_PAIRS.keys():
        if ft not in results_exp12:
            continue
        print(f"\n  === {ft} ===")
        
        # Exp1: ΔS分解
        decomp = results_exp12[ft]['delta_decomp']
        top5_delta = sorted(decomp.items(), key=lambda x: x[1]['delta_norm'], reverse=True)[:5]
        top5_str = ', '.join(f'L{li}: {d["delta_norm"]:.2f}' for li, d in top5_delta)
        print(f"  Top-5 ΔS norm layers: {top5_str}")
        
        # 平均Attn/MLP分数
        avg_attn = np.mean([d['attn_frac'] for d in decomp.values()])
        avg_mlp = np.mean([d['mlp_frac'] for d in decomp.values()])
        print(f"  Avg Attn%: {avg_attn:.1%} | Avg MLP%: {avg_mlp:.1%}")
        
        # Exp2: 语义特异性
        spec = results_exp12[ft]['semantic_specificity']
        top3_spec = sorted(spec.items(), key=lambda x: x[1]['specificity'], reverse=True)[:3]
        top3_spec_str = ', '.join(f'L{li}: {d["specificity"]:.3f}' for li, d in top3_spec)
        print(f"  Top-3 semantic specificity: {top3_spec_str}")
    
    # Release model
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()
    
    t_total = time.time() - t_start
    print(f"\nPhase 194 COMPLETE for {model_name}")
    print(f"Total time: {t_total:.1f}s (load={t_load:.1f}s, exp1+2={t_exp12:.1f}s, exp4={t_exp4:.1f}s)")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
