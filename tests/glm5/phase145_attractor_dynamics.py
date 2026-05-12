"""
Phase 145: 吸引子动力学与稳定/不稳定模态分析
============================================

直接检验用户第三次批评中的核心论点:
1. Transformer是"约束稳定传播系统"还是"纯前馈映射"?
2. 吸引子恢复: 扰动后系统是否回归原轨道?
3. 稳定/不稳定模态谱: 哪些方向被压制,哪些方向持续存在?
4. 约束修复时间: 不同约束类型在哪层被修复?
5. 扰动恢复的语义特异性: 语义扰动vs随机扰动的恢复差异

四大实验:
  Exp A: 吸引子恢复实验 (最高优先级)
    - 对hidden state加扰动,观察后续层是否回归原轨道
    - 测量 d(h_clean, h_perturbed) 随层变化
    - 区分: 吸引子 vs 纯前馈

  Exp B: 稳定/不稳定模态谱
    - 计算Jacobian的奇异值谱
    - 找出: 快速收缩方向 vs 持续方向
    - 与语义方向对齐度分析

  Exp C: 约束修复动力学 (精细版)
    - 多种约束类型的修复时间
    - 修复速度对比: SVA vs TENSE vs SCOPE vs LOGIC vs SEMANTIC

  Exp D: 语义vs随机扰动的恢复差异
    - 语义方向扰动 vs 随机方向扰动
    - 恢复速度对比
    - 直接测试"约束稳定传播系统"假说

用法:
  python tests/glm5/phase145_attractor_dynamics.py qwen3
  python tests/glm5/phase145_attractor_dynamics.py glm4
  python tests/glm5/phase145_attractor_dynamics.py deepseek7b
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc
import json
import time
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from model_utils import (load_model, get_layers, get_model_info, get_W_U,
                          release_model, get_layer_weights, MODEL_CONFIGS)

TEMP_DIR = Path("tests/glm5_temp")
TEMP_DIR.mkdir(exist_ok=True)


def get_device_for_input(model):
    """获取输入tensor应放的设备"""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_hidden_states(model, input_ids, attention_mask, model_info):
    """获取所有层的hidden states"""
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True)
    hs = out.hidden_states  # tuple of (n_layers+1,)
    # 转为CPU float tensor list
    return [h.float().cpu() for h in hs]


def compute_jacobian_at_layer(model, input_ids, attention_mask, model_info,
                               target_layer, target_pos=-1, eps=1.0):
    """
    计算指定层的Jacobian (数值方法)
    J[i,j] = (h_{l+1}[i] when h_l[j]+=eps - h_{l+1}[i] when h_l[j]-=eps) / (2*eps)
    
    用hook注入扰动到target_layer, 观察target_layer+1的变化
    
    返回: J [d_model, d_model] (可能采样以降低内存)
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)
    
    # 采样维度以控制内存和速度 (最多测80维,否则太慢)
    max_dim = min(d_model, 80)
    sample_indices = np.random.choice(d_model, max_dim, replace=False)
    sample_indices = np.sort(sample_indices)
    
    # 先获取clean hidden states
    clean_hs = get_hidden_states(model, input_ids, attention_mask, model_info)
    h_clean_next = clean_hs[target_layer + 1][0, target_pos].clone()  # [d_model]
    
    J_cols = []  # 每列是一个d_model维向量(采样后的)
    
    # 对每个采样维度注入扰动
    for idx, j in enumerate(sample_indices):
        # 正扰动
        captured = {}
        def make_hook_pos(key):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    h = output[0].clone()
                    h[0, target_pos, j] += eps
                    captured[key] = h
                    return (h,) + output[1:]
                else:
                    h = output.clone()
                    h[0, target_pos, j] += eps
                    captured[key] = h
                    return h
            return hook
        
        hook = layers[target_layer].register_forward_hook(make_hook_pos(f"pos_{j}"))
        with torch.no_grad():
            out_pos = model(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True)
        hook.remove()
        h_pos_next = out_pos.hidden_states[target_layer + 1][0, target_pos].float().cpu()
        
        # 负扰动
        captured = {}
        def make_hook_neg(key):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    h = output[0].clone()
                    h[0, target_pos, j] -= eps
                    captured[key] = h
                    return (h,) + output[1:]
                else:
                    h = output.clone()
                    h[0, target_pos, j] -= eps
                    captured[key] = h
                    return h
            return hook
        
        hook = layers[target_layer].register_forward_hook(make_hook_neg(f"neg_{j}"))
        with torch.no_grad():
            out_neg = model(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True)
        hook.remove()
        h_neg_next = out_neg.hidden_states[target_layer + 1][0, target_pos].float().cpu()
        
        # 中心差分
        col = (h_pos_next - h_neg_next) / (2 * eps)
        J_cols.append(col.numpy())
        
        if idx % 50 == 0:
            print(f"    Jacobian列 {idx}/{max_dim}, j={j}")
    
    J_sampled = np.stack(J_cols, axis=1)  # [d_model, max_dim]
    return J_sampled, sample_indices


# ============================================================
# Exp A: 吸引子恢复实验
# ============================================================
def exp_a_attractor_recovery(model, tokenizer, model_info, model_name):
    """
    核心实验: 扰动hidden state后,观察后续层是否回归原轨道
    
    如果系统有吸引子结构: d(h_clean, h_perturbed) 随层衰减
    如果系统是纯前馈: d(h_clean, h_perturbed) 保持或增长
    
    测试条件:
    - 20个不同句子
    - 3种扰动: 随机方向, 语义方向(同义词替换), 约束违背方向
    - 扰动注入层: L0, L_n/4, L_n/2, L_3n/4
    - 测量: 注入后每层的恢复度
    """
    print("\n" + "="*60)
    print("Exp A: 吸引子恢复实验")
    print("="*60)
    
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)
    input_device = get_device_for_input(model)
    
    # 测试句子 (10个,覆盖多种语法结构)
    test_sentences = [
        "The scientist discovered a new element yesterday.",
        "Every student in the class passed the final exam.",
        "Not all the birds can fly in the winter.",
        "She has been working on this project for three years.",
        "The quick brown fox jumps over the lazy dog.",
        "The company decided to expand its operations.",
        "He carefully read the instructions before starting.",
        "The weather has been unusually cold this winter.",
        "The old building was demolished last month.",
        "Researchers found evidence of ancient civilizations.",
    ]
    
    # 注入层
    inject_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4]
    
    # 扰动幅度
    eps_values = [0.5, 1.0, 2.0, 5.0]
    
    results = {}
    
    for sent_idx, sentence in enumerate(test_sentences):
        print(f"\n  句子 {sent_idx+1}/{len(test_sentences)}: {sentence[:50]}...")
        
        # Tokenize
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        seq_len = input_ids.shape[1]
        target_pos = seq_len - 1  # 最后一个token
        
        # 获取clean hidden states
        clean_hs = get_hidden_states(model, input_ids, attention_mask, model_info)
        # clean_hs[l][0, target_pos] 是第l层target_pos位置的hidden state
        
        for inject_l in inject_layers:
            if inject_l >= n_layers:
                continue
            
            for eps in eps_values:
                key = f"sent{sent_idx}_L{inject_l}_eps{eps}"
                
                # === 随机扰动 ===
                h_clean = clean_hs[inject_l][0, target_pos].numpy()  # [d_model]
                
                # 生成随机方向
                np.random.seed(42 + sent_idx)
                v_random = np.random.randn(d_model)
                v_random = v_random / np.linalg.norm(v_random) * eps
                
                # 注入扰动
                perturbed_random = {}
                def make_hook_random(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0].clone()
                        h[0, target_pos] += torch.tensor(v_random, dtype=h.dtype, device=h.device)
                        return (h,) + output[1:]
                    else:
                        h = output.clone()
                        h[0, target_pos] += torch.tensor(v_random, dtype=h.dtype, device=h.device)
                        return h
                
                hook = layers[inject_l].register_forward_hook(make_hook_random)
                perturbed_hs_random = get_hidden_states(model, input_ids, attention_mask, model_info)
                hook.remove()
                
                # 计算每层的恢复度
                recovery_random = []
                for l in range(inject_l, n_layers + 1):
                    h_c = clean_hs[l][0, target_pos].numpy()
                    h_p = perturbed_hs_random[l][0, target_pos].numpy()
                    dist = np.linalg.norm(h_p - h_c)
                    recovery_random.append(float(dist))
                
                # === 语义方向扰动 (用LM head的对数概率空间中, 同义词方向) ===
                # 用clean hidden state的LM head投影方向
                W_U = get_W_U(model)  # [vocab, d_model]
                # 取clean_hs最后一层的logits top-1方向作为"语义方向"
                h_last = clean_hs[n_layers][0, target_pos].numpy()
                logits = W_U @ h_last
                top_token_idx = np.argmax(logits)
                v_semantic = W_U[top_token_idx]  # 词汇表中最高概率token的方向
                v_semantic = v_semantic / (np.linalg.norm(v_semantic) + 1e-8) * eps
                
                perturbed_semantic = {}
                def make_hook_semantic(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0].clone()
                        h[0, target_pos] += torch.tensor(v_semantic, dtype=h.dtype, device=h.device)
                        return (h,) + output[1:]
                    else:
                        h = output.clone()
                        h[0, target_pos] += torch.tensor(v_semantic, dtype=h.dtype, device=h.device)
                        return h
                
                hook = layers[inject_l].register_forward_hook(make_hook_semantic)
                perturbed_hs_semantic = get_hidden_states(model, input_ids, attention_mask, model_info)
                hook.remove()
                
                recovery_semantic = []
                for l in range(inject_l, n_layers + 1):
                    h_c = clean_hs[l][0, target_pos].numpy()
                    h_p = perturbed_hs_semantic[l][0, target_pos].numpy()
                    dist = np.linalg.norm(h_p - h_c)
                    recovery_semantic.append(float(dist))
                
                # === 约束违背方向扰动 ===
                # 用两个hidden state的差异方向(正确vs错误约束)
                # 简化: 用W_U的第2高概率token与第1高概率token的差异方向
                top2_indices = np.argsort(logits)[-2:][::-1]
                v_constraint = W_U[top2_indices[0]] - W_U[top2_indices[1]]
                v_constraint = v_constraint / (np.linalg.norm(v_constraint) + 1e-8) * eps
                
                perturbed_constraint = {}
                def make_hook_constraint(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0].clone()
                        h[0, target_pos] += torch.tensor(v_constraint, dtype=h.dtype, device=h.device)
                        return (h,) + output[1:]
                    else:
                        h = output.clone()
                        h[0, target_pos] += torch.tensor(v_constraint, dtype=h.dtype, device=h.device)
                        return h
                
                hook = layers[inject_l].register_forward_hook(make_hook_constraint)
                perturbed_hs_constraint = get_hidden_states(model, input_ids, attention_mask, model_info)
                hook.remove()
                
                recovery_constraint = []
                for l in range(inject_l, n_layers + 1):
                    h_c = clean_hs[l][0, target_pos].numpy()
                    h_p = perturbed_hs_constraint[l][0, target_pos].numpy()
                    dist = np.linalg.norm(h_p - h_c)
                    recovery_constraint.append(float(dist))
                
                results[key] = {
                    "inject_layer": inject_l,
                    "eps": eps,
                    "sentence_idx": sent_idx,
                    "recovery_random": recovery_random,
                    "recovery_semantic": recovery_semantic,
                    "recovery_constraint": recovery_constraint,
                    "initial_dist_random": recovery_random[0],
                    "initial_dist_semantic": recovery_semantic[0],
                    "initial_dist_constraint": recovery_constraint[0],
                }
                
                del perturbed_hs_random, perturbed_hs_semantic, perturbed_hs_constraint
                torch.cuda.empty_cache()
        
        # 每个句子后清理
        if sent_idx % 5 == 4:
            print(f"    完成 {sent_idx+1}/{len(test_sentences)} 句子")
    
    return results


# ============================================================
# Exp B: 稳定/不稳定模态谱
# ============================================================
def exp_b_stable_unstable_spectrum(model, tokenizer, model_info, model_name):
    """
    计算Jacobian的奇异值谱,分析稳定/不稳定模态
    
    核心问题:
    - 哪些方向快速收缩 (稳定模态, 被系统"纠正")
    - 哪些方向持续存在 (不稳定/中性模态, 承载信息)
    - 稳定方向与语义方向的关系
    
    方法:
    - 对5个不同句子,在4个不同层计算Jacobian
    - 分析奇异值谱
    - 分析左/右奇异向量与W_U的对齐度
    """
    print("\n" + "="*60)
    print("Exp B: 稳定/不稳定模态谱")
    print("="*60)
    
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    input_device = get_device_for_input(model)
    
    test_sentences = [
        "The scientist discovered a new element yesterday.",
        "Every student in the class passed the final exam.",
        "Not all the birds can fly in the winter.",
    ]
    
    sample_layers = [0, n_layers//2, n_layers-1]
    
    results = {}
    
    # 获取W_U用于对齐分析
    W_U = get_W_U(model)  # [vocab, d_model]
    # 对W_U做SVD,取前20个方向作为"语义方向"
    U_wu, S_wu, Vt_wu = np.linalg.svd(W_U, full_matrices=False)
    semantic_dirs = Vt_wu[:20]  # [20, d_model]
    
    for sent_idx, sentence in enumerate(test_sentences):
        print(f"\n  句子 {sent_idx+1}/{len(test_sentences)}: {sentence[:50]}...")
        
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        
        for l_idx, layer in enumerate(sample_layers):
            if layer >= n_layers:
                continue
            
            print(f"    计算层 {layer} 的Jacobian谱...")
            
            # 计算Jacobian (采样版)
            J_sampled, sample_indices = compute_jacobian_at_layer(
                model, input_ids, attention_mask, model_info,
                target_layer=layer, target_pos=-1, eps=1.0
            )
            # J_sampled: [d_model, max_dim], 列对应sample_indices维度
            
            # SVD
            U_j, S_j, Vt_j = np.linalg.svd(J_sampled, full_matrices=False)
            
            # 奇异值统计
            total_energy = np.sum(S_j**2)
            cum_energy = np.cumsum(S_j**2) / total_energy
            
            # 前10个奇异值
            top_sv = S_j[:min(20, len(S_j))].tolist()
            
            # 有效秩 (参与率)
            pr = (np.sum(S_j)**2) / (np.sum(S_j**2) + 1e-10)
            
            # 奇异值分布统计
            n_contract = np.sum(S_j < 0.5)  # 收缩模态 (sigma < 0.5)
            n_expand = np.sum(S_j > 1.5)    # 扩张模态 (sigma > 1.5)
            n_neutral = np.sum((S_j >= 0.5) & (S_j <= 1.5))  # 中性模态
            
            # 左奇异向量与语义方向的对齐度
            alignment_with_semantic = []
            for k in range(min(10, U_j.shape[1])):
                u_k = U_j[:, k]  # [d_model]
                alignments = [abs(np.dot(u_k, sd)) for sd in semantic_dirs]
                max_align = max(alignments)
                alignment_with_semantic.append(float(max_align))
            
            # 扩张方向 vs 收缩方向 与语义的对齐
            if n_expand > 0:
                expand_aligns = [alignment_with_semantic[k] for k in range(min(n_expand, len(alignment_with_semantic)))]
                mean_expand_align = np.mean(expand_aligns)
            else:
                mean_expand_align = 0.0
            
            if n_contract > 0:
                contract_start = max(0, len(alignment_with_semantic) - min(n_contract, len(alignment_with_semantic)))
                contract_aligns = alignment_with_semantic[contract_start:]
                mean_contract_align = np.mean(contract_aligns) if contract_aligns else 0.0
            else:
                mean_contract_align = 0.0
            
            key = f"sent{sent_idx}_L{layer}"
            results[key] = {
                "layer": layer,
                "sentence_idx": sent_idx,
                "top_singular_values": top_sv,
                "participation_ratio": float(pr),
                "n_contract": int(n_contract),
                "n_expand": int(n_expand),
                "n_neutral": int(n_neutral),
                "total_dim": len(S_j),
                "cum_energy_50": int(np.searchsorted(cum_energy, 0.5) + 1),
                "cum_energy_90": int(np.searchsorted(cum_energy, 0.9) + 1),
                "alignment_with_semantic": alignment_with_semantic,
                "mean_expand_semantic_align": float(mean_expand_align),
                "mean_contract_semantic_align": float(mean_contract_align),
            }
            
            del J_sampled, U_j, S_j, Vt_j
            torch.cuda.empty_cache()
    
    return results


# ============================================================
# Exp C: 约束修复动力学 (精细版)
# ============================================================
def exp_c_constraint_repair(model, tokenizer, model_info, model_name):
    """
    精细测量约束修复时间
    
    对比5种约束类型:
    1. SVA (主谓一致): "The cat walks" vs "The cat walk"
    2. TENSE (时态): "She walked" vs "She walk"
    3. SCOPE (辖域): "All birds can fly" vs "Not all birds can fly"
    4. LOGIC (逻辑): "A is B and B is C, so A is C" vs "A is B and B is C, so A is D"
    5. SEMANTIC (语义): "The scientist discovered" vs "The scientist destroyed"
    
    测量:
    - 约束违背信号在哪层首次变得可观测
    - 在哪层被最大程度修复
    - 修复速度(每层减少的delta比例)
    """
    print("\n" + "="*60)
    print("Exp C: 约束修复动力学")
    print("="*60)
    
    n_layers = model_info.n_layers
    input_device = get_device_for_input(model)
    
    constraint_pairs = {
        "SVA": [
            ("The cat walks slowly.", "The cat walk slowly."),
            ("The dogs run fast.", "The dogs runs fast."),
            ("She writes every day.", "She write every day."),
            ("The birds fly south.", "The birds flies south."),
            ("He reads the book.", "He read the book."),
            ("They play together.", "They plays together."),
            ("The river flows east.", "The river flow east."),
            ("We study mathematics.", "We studies mathematics."),
            ("The children laugh loudly.", "The children laughs loudly."),
            ("My friends travel often.", "My friends travels often."),
        ],
        "TENSE": [
            ("She walked to school.", "She walk to school."),
            ("He has finished the work.", "He has finish the work."),
            ("They went to the park.", "They go to the park."),
            ("The train arrived late.", "The train arrive late."),
            ("She wrote a letter.", "She write a letter."),
            ("He discovered the truth.", "He discover the truth."),
            ("We visited the museum.", "We visit the museum."),
            ("The storm destroyed everything.", "The storm destroy everything."),
            ("They built a house.", "They build a house."),
            ("She completed the task.", "She complete the task."),
        ],
        "SCOPE": [
            ("All birds can fly.", "Not all birds can fly."),
            ("Every student passed.", "Not every student passed."),
            ("All doors were open.", "Not all doors were open."),
            ("Every child received a gift.", "Not every child received a gift."),
            ("All members agreed.", "Not all members agreed."),
            ("Every player scored.", "Not every player scored."),
            ("All tickets were sold.", "Not all tickets were sold."),
            ("Every citizen voted.", "Not every citizen voted."),
            ("All rooms were booked.", "Not all rooms were booked."),
            ("Every employee attended.", "Not every employee attended."),
        ],
        "LOGIC": [
            ("If it rains, the ground gets wet.", "If it rains, the ground stays dry."),
            ("All cats are animals, so cats breathe.", "All cats are animals, so cats fly."),
            ("She studied hard and passed.", "She studied hard and failed."),
            ("The sun rises in the east.", "The sun rises in the west."),
            ("Water freezes at zero degrees.", "Water freezes at one hundred degrees."),
            ("Birds have wings and can fly.", "Birds have wings and can swim."),
            ("Two plus two equals four.", "Two plus two equals five."),
            ("Humans need oxygen to survive.", "Humans need carbon dioxide to survive."),
            ("Ice is cold and solid.", "Ice is hot and liquid."),
            ("Gravity pulls objects downward.", "Gravity pushes objects upward."),
        ],
        "SEMANTIC": [
            ("The scientist discovered a new element.", "The scientist destroyed a new element."),
            ("She carefully opened the door.", "She carefully broke the door."),
            ("The teacher explained the theory.", "The teacher hid the theory."),
            ("He gently placed the vase.", "He gently smashed the vase."),
            ("The doctor healed the patient.", "The doctor harmed the patient."),
            ("The artist painted a landscape.", "The artist burned a landscape."),
            ("She planted flowers in spring.", "She pulled flowers in spring."),
            ("The builder constructed the wall.", "The builder demolished the wall."),
            ("He rescued the drowning child.", "He ignored the drowning child."),
            ("The chef prepared a delicious meal.", "The chef spoiled a delicious meal."),
        ],
    }
    
    results = {}
    
    for constraint_type, pairs in constraint_pairs.items():
        print(f"\n  约束类型: {constraint_type} ({len(pairs)} 对)")
        
        type_results = {
            "first_observable_layers": [],
            "max_repair_layers": [],
            "repair_speeds": [],
            "delta_trajectories": [],
        }
        
        for pair_idx, (correct, wrong) in enumerate(pairs):
            # Tokenize
            inputs_c = tokenizer(correct, return_tensors="pt", truncation=True, max_length=64)
            inputs_w = tokenizer(wrong, return_tensors="pt", truncation=True, max_length=64)
            
            input_ids_c = inputs_c["input_ids"].to(input_device)
            attn_c = inputs_c["attention_mask"].to(input_device)
            input_ids_w = inputs_w["input_ids"].to(input_device)
            attn_w = inputs_w["attention_mask"].to(input_device)
            
            # 获取hidden states
            hs_c = get_hidden_states(model, input_ids_c, attn_c, model_info)
            hs_w = get_hidden_states(model, input_ids_w, attn_w, model_info)
            
            # 计算每层的delta (使用最后一个token)
            # 注意: 两个句子长度可能不同,用较短的长度
            min_len = min(hs_c[0].shape[1], hs_w[0].shape[1])
            target_pos = min_len - 1
            
            deltas = []
            for l in range(n_layers + 1):
                h_c = hs_c[l][0, target_pos].numpy()
                h_w = hs_w[l][0, target_pos].numpy()
                delta = np.linalg.norm(h_c - h_w)
                deltas.append(float(delta))
            
            # 首次可观测层 (delta超过baseline 2倍)
            # baseline = embedding层的delta
            baseline_delta = deltas[0]
            first_obs = 0
            for l in range(1, n_layers + 1):
                if deltas[l] > baseline_delta * 2:
                    first_obs = l
                    break
            
            # 最大修复层 (delta下降最多的层)
            max_delta = max(deltas)
            if max_delta > 0:
                # 从最大delta的层开始,找到delta下降最大的位置
                max_delta_layer = deltas.index(max_delta)
                repair_ratios = []
                for l in range(max_delta_layer, n_layers + 1):
                    if l > max_delta_layer:
                        ratio = (deltas[l-1] - deltas[l]) / (deltas[l-1] + 1e-10)
                        repair_ratios.append((l, ratio))
                
                if repair_ratios:
                    max_repair_layer = max(repair_ratios, key=lambda x: x[1])[0]
                    max_repair_ratio = max(repair_ratios, key=lambda x: x[1])[1]
                else:
                    max_repair_layer = n_layers
                    max_repair_ratio = 0.0
            else:
                max_repair_layer = 0
                max_repair_ratio = 0.0
            
            # 修复速度 (delta从最大到末层的总衰减比例)
            if max_delta > 0:
                repair_speed = (max_delta - deltas[-1]) / (max_delta + 1e-10)
            else:
                repair_speed = 0.0
            
            type_results["first_observable_layers"].append(first_obs)
            type_results["max_repair_layers"].append(max_repair_layer)
            type_results["repair_speeds"].append(float(repair_speed))
            type_results["delta_trajectories"].append(deltas)
            
            del hs_c, hs_w
            torch.cuda.empty_cache()
        
        # 汇总
        results[constraint_type] = {
            "n_pairs": len(pairs),
            "mean_first_observable": float(np.mean(type_results["first_observable_layers"])),
            "mean_max_repair_layer": float(np.mean(type_results["max_repair_layers"])),
            "mean_repair_speed": float(np.mean(type_results["repair_speeds"])),
            "mean_delta_trajectory": [float(x) for x in np.mean(type_results["delta_trajectories"], axis=0)],
            "std_repair_speed": float(np.std(type_results["repair_speeds"])),
        }
        
        print(f"    首次可观测层: {results[constraint_type]['mean_first_observable']:.1f}")
        print(f"    最大修复层: {results[constraint_type]['mean_max_repair_layer']:.1f}")
        print(f"    修复速度: {results[constraint_type]['mean_repair_speed']:.3f}")
    
    return results


# ============================================================
# Exp D: 语义vs随机扰动的恢复差异
# ============================================================
def exp_d_semantic_vs_random_recovery(model, tokenizer, model_info, model_name):
    """
    直接测试"约束稳定传播系统"假说的核心预测:
    
    如果系统是约束稳定的:
    - 随机方向扰动应该被快速纠正(偏离可行域)
    - 语义方向扰动应该更稳定(沿可行域方向)
    
    方法:
    - 对15个句子,在中间层注入扰动
    - 3种扰动方向:
      a) 随机方向
      b) W_U主成分方向(语义方向)
      c) W_U正交方向(非语义方向)
    - 测量: 扰动在不同层后的剩余比例
    """
    print("\n" + "="*60)
    print("Exp D: 语义vs随机扰动的恢复差异")
    print("="*60)
    
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)
    input_device = get_device_for_input(model)
    
    # 获取W_U并计算主成分
    W_U = get_W_U(model)  # [vocab, d_model]
    U_wu, S_wu, Vt_wu = np.linalg.svd(W_U, full_matrices=False)
    # Vt_wu[:k] 是W_U的前k个右奇异向量(语义方向)
    
    test_sentences = [
        "The scientist discovered a new element yesterday.",
        "Every student in the class passed the final exam.",
        "Not all the birds can fly in the winter.",
        "She has been working on this project for three years.",
        "The quick brown fox jumps over the lazy dog.",
        "The company decided to expand its operations.",
        "He carefully read the instructions before starting.",
        "The weather has been unusually cold this winter.",
        "The old building was demolished last month.",
        "The library contains thousands of rare manuscripts.",
        "She was reading when the phone rang suddenly.",
        "The experiment produced unexpected but interesting results.",
        "The river flows through the valley into the sea.",
        "He has never visited that country before.",
        "The concert was cancelled due to bad weather.",
    ]
    
    inject_layer = n_layers // 2  # 在中间层注入
    eps = 2.0
    
    results = {
        "random": {"remaining_ratios": [], "trajectory_means": None},
        "semantic_pc1": {"remaining_ratios": [], "trajectory_means": None},
        "semantic_pc5": {"remaining_ratios": [], "trajectory_means": None},
        "semantic_pc20": {"remaining_ratios": [], "trajectory_means": None},
        "orthogonal": {"remaining_ratios": [], "trajectory_means": None},
    }
    
    all_trajectories = {k: [] for k in results.keys()}
    
    for sent_idx, sentence in enumerate(test_sentences):
        print(f"\n  句子 {sent_idx+1}/{len(test_sentences)}: {sentence[:50]}...")
        
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        seq_len = input_ids.shape[1]
        target_pos = seq_len - 1
        
        # Clean hidden states
        clean_hs = get_hidden_states(model, input_ids, attention_mask, model_info)
        h_clean_inject = clean_hs[inject_layer][0, target_pos].numpy()
        
        # 定义扰动方向
        perturbation_dirs = {}
        
        # a) 随机方向
        np.random.seed(42 + sent_idx)
        v_rand = np.random.randn(d_model)
        v_rand = v_rand / np.linalg.norm(v_rand)
        perturbation_dirs["random"] = v_rand
        
        # b) W_U PC1方向
        perturbation_dirs["semantic_pc1"] = Vt_wu[0]
        
        # c) W_U PC5方向
        if Vt_wu.shape[0] > 4:
            perturbation_dirs["semantic_pc5"] = Vt_wu[4]
        
        # d) W_U PC20方向
        if Vt_wu.shape[0] > 19:
            perturbation_dirs["semantic_pc20"] = Vt_wu[19]
        
        # e) W_U 正交方向 (第100个奇异向量之后)
        if Vt_wu.shape[0] > 100:
            perturbation_dirs["orthogonal"] = Vt_wu[100]
        else:
            # 构造正交方向
            v_orth = np.random.randn(d_model)
            for k in range(min(20, Vt_wu.shape[0])):
                v_orth -= np.dot(v_orth, Vt_wu[k]) * Vt_wu[k]
            if np.linalg.norm(v_orth) > 1e-8:
                v_orth = v_orth / np.linalg.norm(v_orth)
            perturbation_dirs["orthogonal"] = v_orth
        
        for dir_name, v_dir in perturbation_dirs.items():
            if dir_name not in results:
                continue
            
            # 注入扰动
            v_perturb = v_dir * eps
            
            def make_hook_perturb(module, input, output, vp=v_perturb):
                if isinstance(output, tuple):
                    h = output[0].clone()
                    h[0, target_pos] += torch.tensor(vp, dtype=h.dtype, device=h.device)
                    return (h,) + output[1:]
                else:
                    h = output.clone()
                    h[0, target_pos] += torch.tensor(vp, dtype=h.dtype, device=h.device)
                    return h
            
            hook = layers[inject_layer].register_forward_hook(make_hook_perturb)
            perturbed_hs = get_hidden_states(model, input_ids, attention_mask, model_info)
            hook.remove()
            
            # 计算每层的剩余扰动比例
            trajectory = []
            for l in range(inject_layer, n_layers + 1):
                h_c = clean_hs[l][0, target_pos].numpy()
                h_p = perturbed_hs[l][0, target_pos].numpy()
                dist = np.linalg.norm(h_p - h_c)
                trajectory.append(float(dist))
            
            # 初始扰动距离
            initial_dist = trajectory[0]
            
            # 剩余比例 (相对于初始扰动)
            remaining = [t / (initial_dist + 1e-10) for t in trajectory]
            
            # 末层剩余比例
            final_remaining = remaining[-1] if remaining else 1.0
            
            results[dir_name]["remaining_ratios"].append(float(final_remaining))
            all_trajectories[dir_name].append(remaining)
            
            del perturbed_hs
            torch.cuda.empty_cache()
    
    # 汇总轨迹
    for dir_name in all_trajectories:
        if all_trajectories[dir_name]:
            # 对齐长度 (取最短的)
            min_len = min(len(t) for t in all_trajectories[dir_name])
            trimmed = [t[:min_len] for t in all_trajectories[dir_name]]
            mean_traj = np.mean(trimmed, axis=0).tolist()
            results[dir_name]["trajectory_means"] = mean_traj
    
    # 计算统计
    summary = {}
    for dir_name in results:
        ratios = results[dir_name]["remaining_ratios"]
        if ratios:
            summary[dir_name] = {
                "mean_remaining": float(np.mean(ratios)),
                "std_remaining": float(np.std(ratios)),
                "n_sentences": len(ratios),
            }
    
    return {"per_direction": results, "summary": summary}


# ============================================================
# 主函数
# ============================================================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    assert model_name in MODEL_CONFIGS, f"Unknown model: {model_name}"
    
    print(f"\nPhase 145: 吸引子动力学与稳定/不稳定模态分析")
    print(f"模型: {model_name}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # 加载模型
    cfg = MODEL_CONFIGS[model_name]
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    use_8bit = model_name in ("deepseek7b", "glm4") and gpu_mem_gb < 16
    
    print(f"加载方式: {'8bit' if use_8bit else 'bfloat16'}")
    model, tokenizer, device = load_model(model_name, use_8bit=use_8bit)
    model_info = get_model_info(model, model_name)
    print(f"模型: {model_info.model_class}, {model_info.n_layers}层, d={model_info.d_model}")
    
    all_results = {
        "model": model_name,
        "model_info": {
            "n_layers": model_info.n_layers,
            "d_model": model_info.d_model,
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    
    # Exp A: 吸引子恢复
    print("\n\n" + "#"*60)
    print("# Exp A: 吸引子恢复实验 (最高优先级)")
    print("#"*60)
    try:
        exp_a = exp_a_attractor_recovery(model, tokenizer, model_info, model_name)
        all_results["exp_a"] = exp_a
        print("Exp A 完成!")
    except Exception as e:
        print(f"Exp A 失败: {e}")
        import traceback; traceback.print_exc()
        all_results["exp_a_error"] = str(e)
    
    gc.collect(); torch.cuda.empty_cache()
    
    # Exp B: 稳定/不稳定模态谱
    print("\n\n" + "#"*60)
    print("# Exp B: 稳定/不稳定模态谱")
    print("#"*60)
    try:
        exp_b = exp_b_stable_unstable_spectrum(model, tokenizer, model_info, model_name)
        all_results["exp_b"] = exp_b
        print("Exp B 完成!")
    except Exception as e:
        print(f"Exp B 失败: {e}")
        import traceback; traceback.print_exc()
        all_results["exp_b_error"] = str(e)
    
    gc.collect(); torch.cuda.empty_cache()
    
    # Exp C: 约束修复动力学
    print("\n\n" + "#"*60)
    print("# Exp C: 约束修复动力学")
    print("#"*60)
    try:
        exp_c = exp_c_constraint_repair(model, tokenizer, model_info, model_name)
        all_results["exp_c"] = exp_c
        print("Exp C 完成!")
    except Exception as e:
        print(f"Exp C 失败: {e}")
        import traceback; traceback.print_exc()
        all_results["exp_c_error"] = str(e)
    
    gc.collect(); torch.cuda.empty_cache()
    
    # Exp D: 语义vs随机扰动
    print("\n\n" + "#"*60)
    print("# Exp D: 语义vs随机扰动恢复差异")
    print("#"*60)
    try:
        exp_d = exp_d_semantic_vs_random_recovery(model, tokenizer, model_info, model_name)
        all_results["exp_d"] = exp_d
        print("Exp D 完成!")
    except Exception as e:
        print(f"Exp D 失败: {e}")
        import traceback; traceback.print_exc()
        all_results["exp_d_error"] = str(e)
    
    # 保存结果
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = TEMP_DIR / f"phase145_{model_name}_attractor_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    print(f"\n结果保存到: {out_path}")
    
    # 打印关键结果摘要
    print("\n" + "="*60)
    print("关键结果摘要")
    print("="*60)
    
    # Exp A摘要
    if "exp_a" in all_results:
        exp_a = all_results["exp_a"]
        print("\nExp A: 吸引子恢复")
        # 按注入层和扰动类型汇总
        for inject_l_str in ["0", str(model_info.n_layers//4), str(model_info.n_layers//2), str(3*model_info.n_layers//4)]:
            for eps_str in ["1.0", "2.0"]:
                # 收集所有匹配的key
                random_final = []
                semantic_final = []
                constraint_final = []
                for key, val in exp_a.items():
                    if val["inject_layer"] == int(inject_l_str) and val["eps"] == float(eps_str):
                        random_final.append(val["recovery_random"][-1] / (val["initial_dist_random"] + 1e-10))
                        semantic_final.append(val["recovery_semantic"][-1] / (val["initial_dist_semantic"] + 1e-10))
                        constraint_final.append(val["recovery_constraint"][-1] / (val["initial_dist_constraint"] + 1e-10))
                
                if random_final:
                    print(f"  L{inject_l_str}, eps={eps_str}: "
                          f"random={np.mean(random_final):.3f}, "
                          f"semantic={np.mean(semantic_final):.3f}, "
                          f"constraint={np.mean(constraint_final):.3f}")
    
    # Exp C摘要
    if "exp_c" in all_results:
        print("\nExp C: 约束修复")
        for c_type, c_data in all_results["exp_c"].items():
            print(f"  {c_type}: 首现层={c_data['mean_first_observable']:.1f}, "
                  f"最大修复层={c_data['mean_max_repair_layer']:.1f}, "
                  f"修复速度={c_data['mean_repair_speed']:.3f}")
    
    # Exp D摘要
    if "exp_d" in all_results and "summary" in all_results["exp_d"]:
        print("\nExp D: 语义vs随机扰动恢复")
        for dir_name, s in all_results["exp_d"]["summary"].items():
            print(f"  {dir_name}: 末层剩余={s['mean_remaining']:.3f} +/- {s['std_remaining']:.3f}")
    
    # 释放模型
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()
    print("\nPhase 145 完成!")


if __name__ == "__main__":
    main()
