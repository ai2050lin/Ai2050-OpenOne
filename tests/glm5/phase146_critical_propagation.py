"""Phase 146: 临界稳定传播系统 — 跨模型验证 + 输出等价稳定性
================================================================

基于Phase 145用户深度分析的关键修正和新实验:

核心修正:
1. "非吸引子" ≠ "无约束" — Transformer可能是轨道约束系统
2. Jacobian≈I ≠ "没有结构" — 近保距传播本身就是高级结构
3. 语言结构编码在方向几何,不是能量几何
4. 需要区分: 状态稳定 vs 输出稳定

关键新实验:
  Exp A+: 吸引子恢复 + 输出等价稳定性 (最重要!)
    - 扰动后hidden state是否偏离? (已知: 是)
    - 但top-k logits是否稳定? (关键新测!)
    - 这直接测试 "输出等价类稳定" vs "状态稳定"
    
  Exp B+: 轨道方向保持 (trajectory direction preservation)
    - 测量 cos(h_clean, h_perturbed) 随层变化
    - 如果方向保持(cos>0.9)即使幅度偏离, 输出可能仍正确
    
  Exp C+: 扰动方向演化
    - 注入方向v0, 在后续层追踪v0方向的投影
    - cos(v0, delta_h_l) 随层变化
    - 测试扰动方向是否旋转 (方向惯性)
    
  Exp D: 约束修复动力学 (沿用Phase 145)
    - 正确/错误句的delta轨迹
    
  Exp E: W_U投影下的等价类
    - 扰动在W_U null space vs row space中的分量
    - 如果null space中的扰动被放大, 不影响输出

用法:
  python tests/glm5/phase146_critical_propagation.py qwen3
  python tests/glm5/phase146_critical_propagation.py glm4
  python tests/glm5/phase146_critical_propagation.py deepseek7b
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
                          release_model, MODEL_CONFIGS)

TEMP_DIR = Path("tests/glm5_temp")
TEMP_DIR.mkdir(exist_ok=True)


def get_device_for_input(model):
    """获取输入tensor应放的设备"""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_clean_hidden_states(model, input_ids, attention_mask):
    """获取所有层的clean hidden states"""
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True)
    return [h.float().cpu() for h in out.hidden_states]


def compute_logits(W_U, h):
    """计算logits, 自动处理float16 W_U"""
    if W_U is not None:
        if W_U.dtype == np.float16:
            return W_U.astype(np.float32) @ h.astype(np.float32)
        return W_U @ h
    return None  # 需要通过模型计算


def compute_logits_via_model(model, hidden_state_tensor, position):
    """通过模型的lm_head计算logits (用于8bit模型W_U加载失败时)"""
    with torch.no_grad():
        # hidden_state_tensor: [1, seq_len, d_model] on cpu
        # 需要放到模型设备上
        device = next(model.parameters()).device
        hs = hidden_state_tensor.to(device).to(model.dtype)
        logits = model.lm_head(hs)  # [1, seq_len, vocab]
        return logits[0, position].float().cpu().numpy()


def perturb_with_hook(model, input_ids, attention_mask, model_info,
                      inject_layer, target_pos, perturbation_vector):
    """
    在inject_layer注入扰动, 获取后续层的hidden states + 最终logits
    
    返回: (perturbed_hidden_states, perturbed_logits) 或 None
    """
    n_layers = model_info.n_layers
    layers = get_layers(model)
    d_model = model_info.d_model
    
    perturbed_hs = None
    
    try:
        v_tensor = torch.tensor(perturbation_vector, dtype=torch.float32)
        
        def make_hook(vp):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    h = output[0].clone().float()
                    h[0, target_pos] += vp.to(h.device)
                    return (h.to(output[0].dtype),) + output[1:]
                else:
                    h = output.clone().float()
                    h[0, target_pos] += vp.to(h.device)
                    return h.to(output.dtype)
            return hook
        
        hook = layers[inject_layer].register_forward_hook(make_hook(v_tensor))
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                       output_hidden_states=True)
        hook.remove()
        
        perturbed_hs = [h.float().cpu() for h in out.hidden_states]
        # logits from output (直接用模型计算,不依赖W_U矩阵)
        perturbed_logits = out.logits[0, target_pos].float().cpu().numpy()
        
        return perturbed_hs, perturbed_logits
        
    except Exception as e:
        try:
            hook.remove()
        except:
            pass
        print(f"    Hook方法失败: {e}")
        return None


# ============================================================
# Exp A+: 吸引子恢复 + 输出等价稳定性
# ============================================================
def exp_a_output_equivalence(model, tokenizer, model_info, model_name, W_U):
    """
    核心实验: 扰动后的输出等价稳定性
    
    测试:
    1. hidden state偏离是否回归? (Phase 145已测: 不回归)
    2. 但top-k logits是否稳定? (新测!)
    3. 轨道方向cos(h_clean, h_perturbed)是否保持? (新测!)
    
    这直接区分: 状态稳定 vs 输出稳定
    """
    print("\n" + "="*60)
    print("Exp A+: 吸引子恢复 + 输出等价稳定性")
    print("="*60)
    
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    input_device = get_device_for_input(model)
    
    # 根据模型调整数据量: bfloat16用15句, 8bit用10句
    use_8bit_mode = model_name in ("deepseek7b", "glm4")
    n_sentences = 10 if use_8bit_mode else 15
    
    test_sentences = [
        "The scientist discovered a new element yesterday.",
        "Every student in the class passed the final exam.",
        "Not all the birds can fly in the winter.",
        "She has been working on this project for three years.",
        "The quick brown fox jumps over the lazy dog.",
        "The ancient castle stood on top of the mountain.",
        "He carefully placed the fragile vase on the shelf.",
        "The children played happily in the garden all afternoon.",
        "Despite the rain, the team continued their practice.",
        "The professor explained the complex theory to students.",
        "A small bird sang beautifully in the morning light.",
        "The company launched a revolutionary new product today.",
        "She always drinks coffee before starting her work.",
        "The river flows gently through the green valley.",
        "His latest novel became an instant bestseller worldwide.",
    ][:n_sentences]
    
    inject_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-2]
    eps_values = [0.5, 1.0, 2.0, 5.0]
    
    results = {}
    
    for sent_idx, sentence in enumerate(test_sentences):
        t0 = time.time()
        print(f"\n  句子 {sent_idx+1}/{len(test_sentences)}: {sentence[:50]}...")
        
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        seq_len = input_ids.shape[1]
        target_pos = seq_len - 1
        
        # Clean trajectory
        clean_hs = get_clean_hidden_states(model, input_ids, attention_mask)
        
        # Clean logits (from final hidden state)
        h_clean_final = clean_hs[n_layers][0, target_pos].numpy()
        if W_U is not None:
            clean_logits = compute_logits(W_U, h_clean_final)
        else:
            # 通过模型lm_head计算
            clean_logits = compute_logits_via_model(model, clean_hs[n_layers], target_pos)
        clean_top5 = np.argsort(clean_logits)[-5:][::-1]
        clean_top10 = np.argsort(clean_logits)[-10:][::-1]
        
        for inject_l in inject_layers:
            if inject_l >= n_layers:
                continue
            
            for eps in eps_values:
                key = f"sent{sent_idx}_L{inject_l}_eps{eps}"
                
                # === 随机扰动 ===
                np.random.seed(42 + sent_idx)
                v_random = np.random.randn(d_model)
                v_random = v_random / np.linalg.norm(v_random) * eps
                
                result = perturb_with_hook(
                    model, input_ids, attention_mask, model_info,
                    inject_l, target_pos, v_random
                )
                
                if result is None:
                    print(f"    L{inject_l} eps={eps} 失败,跳过")
                    continue
                
                perturbed_hs, perturbed_logits = result
                
                # --- 计算3类稳定性指标 ---
                
                # 1. 状态距离: ||h_perturbed - h_clean|| (归一化到初始扰动)
                state_distances = []
                for l in range(inject_l, n_layers + 1):
                    h_c = clean_hs[l][0, target_pos].numpy()
                    h_p = perturbed_hs[l][0, target_pos].numpy()
                    dist = np.linalg.norm(h_p - h_c)
                    state_distances.append(float(dist))
                
                initial_dist = state_distances[0] if state_distances[0] > 1e-10 else eps
                normalized_state_dists = [d / initial_dist for d in state_distances]
                
                # 2. 方向保持: cos(h_clean, h_perturbed) 随层变化
                direction_cosines = []
                for l in range(inject_l, n_layers + 1):
                    h_c = clean_hs[l][0, target_pos].numpy()
                    h_p = perturbed_hs[l][0, target_pos].numpy()
                    norm_c = np.linalg.norm(h_c)
                    norm_p = np.linalg.norm(h_p)
                    if norm_c > 1e-10 and norm_p > 1e-10:
                        cos_val = float(np.dot(h_c, h_p) / (norm_c * norm_p))
                    else:
                        cos_val = 0.0
                    direction_cosines.append(cos_val)
                
                # 3. 输出等价稳定性
                perturbed_top5 = np.argsort(perturbed_logits)[-5:][::-1]
                perturbed_top10 = np.argsort(perturbed_logits)[-10:][::-1]
                
                top5_overlap = len(set(clean_top5) & set(perturbed_top5)) / 5.0
                top10_overlap = len(set(clean_top10) & set(perturbed_top10)) / 10.0
                top1_match = 1 if clean_top5[0] == perturbed_top5[0] else 0
                
                # Logits correlation
                logits_corr = float(np.corrcoef(clean_logits, perturbed_logits)[0, 1])
                
                # KL divergence (approximate)
                clean_probs = np.exp(clean_logits - np.max(clean_logits))
                clean_probs = clean_probs / clean_probs.sum()
                perturbed_probs = np.exp(perturbed_logits - np.max(perturbed_logits))
                perturbed_probs = perturbed_probs / perturbed_probs.sum()
                kl_div = float(np.sum(clean_probs * (np.log(clean_probs + 1e-10) - np.log(perturbed_probs + 1e-10))))
                
                results[key] = {
                    "inject_layer": inject_l,
                    "eps": eps,
                    "sentence_idx": sent_idx,
                    "perturbation_type": "random",
                    "state_distances": state_distances,
                    "normalized_state_dists": normalized_state_dists,
                    "direction_cosines": direction_cosines,
                    "top1_match": top1_match,
                    "top5_overlap": top5_overlap,
                    "top10_overlap": top10_overlap,
                    "logits_correlation": logits_corr,
                    "kl_divergence": kl_div,
                    "initial_dist": float(initial_dist),
                }
                
                del perturbed_hs, perturbed_logits
                torch.cuda.empty_cache()
        
        # 只释放clean_hs, 保留results
        del clean_hs
        torch.cuda.empty_cache()
        
        elapsed = time.time() - t0
        print(f"    句子{sent_idx+1}完成, 耗时{elapsed:.1f}s")
    
    return results


# ============================================================
# Exp B+: 扰动方向演化 (方向惯性测试)
# ============================================================
def exp_b_direction_evolution(model, tokenizer, model_info, model_name, W_U):
    """
    测试注入的扰动方向在后续层如何演化
    
    核心问题: 扰动方向是否旋转? (方向惯性)
    - cos(v0, delta_h_l) 随层变化
    - 如果cos保持高值: 扰动方向被保持 → 方向惯性
    - 如果cos下降: 扰动方向被旋转 → 子空间混合
    """
    print("\n" + "="*60)
    print("Exp B+: 扰动方向演化")
    print("="*60)
    
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    input_device = get_device_for_input(model)
    
    test_sentences = [
        "The scientist discovered a new element yesterday.",
        "Every student in the class passed the final exam.",
        "She has been working on this project for three years.",
        "The ancient castle stood on top of the mountain.",
        "The professor explained the complex theory to students.",
        "A small bird sang beautifully in the morning light.",
        "The company launched a revolutionary new product today.",
        "She always drinks coffee before starting her work.",
        "The river flows gently through the green valley.",
        "His latest novel became an instant bestseller worldwide.",
    ]
    
    inject_layers = [0, n_layers//2, n_layers-2]
    eps_values = [2.0]  # 固定eps=2.0
    
    results = {}
    
    for sent_idx, sentence in enumerate(test_sentences):
        t0 = time.time()
        print(f"\n  句子 {sent_idx+1}/{len(test_sentences)}: {sentence[:50]}...")
        
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        seq_len = input_ids.shape[1]
        target_pos = seq_len - 1
        
        clean_hs = get_clean_hidden_states(model, input_ids, attention_mask)
        
        for inject_l in inject_layers:
            if inject_l >= n_layers:
                continue
            
            eps = 2.0
            key = f"sent{sent_idx}_L{inject_l}"
            
            # 注入随机扰动
            np.random.seed(42 + sent_idx)
            v0 = np.random.randn(d_model)
            v0_norm = np.linalg.norm(v0)
            v0_normalized = v0 / v0_norm
            v_random = v0_normalized * eps
            
            result = perturb_with_hook(
                model, input_ids, attention_mask, model_info,
                inject_l, target_pos, v_random
            )
            
            if result is None:
                continue
            
            perturbed_hs, perturbed_logits = result
            
            # 追踪扰动方向演化
            direction_alignment = []  # cos(v0, delta_h_l)
            delta_norms = []  # ||delta_h_l||
            
            for l in range(inject_l, n_layers + 1):
                h_c = clean_hs[l][0, target_pos].numpy()
                h_p = perturbed_hs[l][0, target_pos].numpy()
                delta = h_p - h_c
                delta_norm = np.linalg.norm(delta)
                
                if delta_norm > 1e-10:
                    cos_v0 = float(np.dot(v0_normalized, delta / delta_norm))
                else:
                    cos_v0 = 0.0
                
                direction_alignment.append(cos_v0)
                delta_norms.append(float(delta_norm))
            
            # 同时测试 W_U row space 方向扰动
            # 找出W_U的top-1 token方向
            h_clean_final = clean_hs[n_layers][0, target_pos].numpy()
            if W_U is not None:
                clean_logits_vec = compute_logits(W_U, h_clean_final)
            else:
                clean_logits_vec = compute_logits_via_model(model, clean_hs[n_layers], target_pos)
            top_token = np.argmax(clean_logits_vec)
            
            # W_U方向扰动 (仅当W_U可用时)
            wu_alignment = []
            wu_delta_norms = []
            wu_output_effect = None
            
            if W_U is not None:
                wu_dir = W_U[top_token].copy()
                wu_dir_norm = np.linalg.norm(wu_dir)
                if wu_dir_norm > 1e-10:
                    wu_dir_normalized = wu_dir / wu_dir_norm
                else:
                    wu_dir_normalized = np.zeros(d_model)
                
                v_wu = wu_dir_normalized * eps
                
                result_wu = perturb_with_hook(
                    model, input_ids, attention_mask, model_info,
                    inject_l, target_pos, v_wu
                )
                
                if result_wu is not None:
                    perturbed_hs_wu, perturbed_logits_wu = result_wu
                    
                    for l in range(inject_l, n_layers + 1):
                        h_c = clean_hs[l][0, target_pos].numpy()
                        h_p = perturbed_hs_wu[l][0, target_pos].numpy()
                        delta = h_p - h_c
                        delta_norm = np.linalg.norm(delta)
                        
                        if delta_norm > 1e-10:
                            cos_wu = float(np.dot(wu_dir_normalized, delta / delta_norm))
                        else:
                            cos_wu = 0.0
                        
                        wu_alignment.append(cos_wu)
                        wu_delta_norms.append(float(delta_norm))
                    
                    # 输出影响: W_U方向扰动 vs 随机方向扰动的logits变化
                    clean_top5_b = np.argsort(compute_logits(W_U, h_clean_final))[-5:][::-1]
                    perturbed_top5_wu = np.argsort(perturbed_logits_wu)[-5:][::-1]
                    perturbed_top5_rand = np.argsort(perturbed_logits)[-5:][::-1]
                    
                    wu_top1_match = 1 if clean_top5_b[0] == perturbed_top5_wu[0] else 0
                    rand_top1_match = 1 if clean_top5_b[0] == perturbed_top5_rand[0] else 0
                    wu_top5_overlap = len(set(clean_top5_b.tolist()) & set(perturbed_top5_wu.tolist())) / 5.0
                    rand_top5_overlap = len(set(clean_top5_b.tolist()) & set(perturbed_top5_rand.tolist())) / 5.0
                    
                    wu_output_effect = {
                        "wu_top1_match": wu_top1_match,
                        "rand_top1_match": rand_top1_match,
                        "wu_top5_overlap": wu_top5_overlap,
                        "rand_top5_overlap": rand_top5_overlap,
                    }
                    
                    del perturbed_hs_wu, perturbed_logits_wu
            
            results[key] = {
                "inject_layer": inject_l,
                "eps": eps,
                "sentence_idx": sent_idx,
                "direction_alignment": direction_alignment,  # cos(v0, delta)
                "delta_norms": delta_norms,
                "wu_alignment": wu_alignment,  # cos(W_U方向, delta)
                "wu_delta_norms": wu_delta_norms,
                "wu_output_effect": wu_output_effect,
            }
            
            del perturbed_hs, perturbed_logits
            torch.cuda.empty_cache()
        
        del clean_hs
        torch.cuda.empty_cache()
        
        elapsed = time.time() - t0
        print(f"    句子{sent_idx+1}完成, 耗时{elapsed:.1f}s")
    
    return results


# ============================================================
# Exp D: 约束修复动力学
# ============================================================
def exp_d_constraint_dynamics(model, tokenizer, model_info, model_name, W_U):
    """
    约束修复动力学 — 加大样本量
    
    核心: 不仅测delta轨迹, 还测:
    1. 正确/错误句的top-k logits差异
    2. 正确/错误句的cos(h_correct, h_wrong)随层变化
    """
    print("\n" + "="*60)
    print("Exp D: 约束修复动力学 + 输出差异")
    print("="*60)
    
    n_layers = model_info.n_layers
    input_device = get_device_for_input(model)
    
    # 8bit模型减少约束对数以节省内存和时间
    n_pairs = 5 if model_name in ("deepseek7b", "glm4") else 10
    
    constraint_pairs = {
        "SVA": [
            ("The cat walks slowly.", "The cat walk slowly."),
            ("The dogs run fast.", "The dogs runs fast."),
            ("She writes every day.", "She write every day."),
            ("The birds fly south.", "The birds flies south."),
            ("He reads the book.", "He read the book."),
            ("The child plays outside.", "The child play outside."),
            ("They sing beautifully.", "They sings beautifully."),
            ("Water flows downhill.", "Water flow downhill."),
            ("The sun rises early.", "The sun rise early."),
            ("My friends help often.", "My friends helps often."),
        ][:n_pairs],
        "TENSE": [
            ("She walked to school.", "She walk to school."),
            ("He has finished the work.", "He has finish the work."),
            ("They went to the park.", "They go to the park."),
            ("The train arrived late.", "The train arrive late."),
            ("She wrote a letter.", "She write a letter."),
            ("The dog barked loudly.", "The dog bark loudly."),
            ("We enjoyed the concert.", "We enjoy the concert."),
            ("He discovered the truth.", "He discover the truth."),
            ("The storm destroyed the house.", "The storm destroy the house."),
            ("She prepared the dinner.", "She prepare the dinner."),
        ][:n_pairs],
        "SCOPE": [
            ("All birds can fly.", "Not all birds can fly."),
            ("Every student passed.", "Not every student passed."),
            ("All doors were open.", "Not all doors were open."),
            ("Every child received a gift.", "Not every child received a gift."),
            ("All members agreed.", "Not all members agreed."),
            ("Every player scored a goal.", "Not every player scored a goal."),
            ("All flowers bloomed.", "Not all flowers bloomed."),
            ("Every candidate passed the test.", "Not every candidate passed the test."),
            ("All machines worked perfectly.", "Not all machines worked perfectly."),
            ("Every student understood the lesson.", "Not every student understood the lesson."),
        ][:n_pairs],
        "LOGIC": [
            ("If it rains, the ground gets wet.", "If it rains, the ground stays dry."),
            ("All cats are animals, so cats breathe.", "All cats are animals, so cats fly."),
            ("She studied hard and passed.", "She studied hard and failed."),
            ("The sun rises in the east.", "The sun rises in the west."),
            ("Water freezes at zero degrees.", "Water freezes at one hundred degrees."),
            ("Heavy objects fall downward.", "Heavy objects fall upward."),
            ("Fire is hot and burns.", "Fire is cold and freezes."),
            ("Day follows night consistently.", "Day follows night randomly."),
            ("Plants need water to grow.", "Plants need oil to grow."),
            ("Birds have wings and can fly.", "Birds have wings and can swim."),
        ][:n_pairs],
        "SEMANTIC": [
            ("The scientist discovered a new element.", "The scientist destroyed a new element."),
            ("She carefully opened the door.", "She carefully broke the door."),
            ("The teacher explained the theory.", "The teacher hid the theory."),
            ("He gently placed the vase.", "He gently smashed the vase."),
            ("The doctor healed the patient.", "The doctor harmed the patient."),
            ("The artist created a masterpiece.", "The artist destroyed a masterpiece."),
            ("She warmly welcomed the guests.", "She warmly rejected the guests."),
            ("The builder constructed the bridge.", "The builder demolished the bridge."),
            ("He carefully repaired the engine.", "He carefully broke the engine."),
            ("The gardener planted new flowers.", "The gardener pulled new flowers."),
        ][:n_pairs],
    }
    
    results = {}
    
    for constraint_type, pairs in constraint_pairs.items():
        t0 = time.time()
        print(f"\n  约束类型: {constraint_type} ({len(pairs)} 对)")
        
        type_results = {
            "delta_trajectories": [],
            "direction_cosines": [],
            "output_divergences": [],
        }
        
        for pair_idx, (correct, wrong) in enumerate(pairs):
            inputs_c = tokenizer(correct, return_tensors="pt", truncation=True, max_length=64)
            inputs_w = tokenizer(wrong, return_tensors="pt", truncation=True, max_length=64)
            
            input_ids_c = inputs_c["input_ids"].to(input_device)
            attn_c = inputs_c["attention_mask"].to(input_device)
            input_ids_w = inputs_w["input_ids"].to(input_device)
            attn_w = inputs_w["attention_mask"].to(input_device)
            
            try:
                hs_c = get_clean_hidden_states(model, input_ids_c, attn_c)
                hs_w = get_clean_hidden_states(model, input_ids_w, attn_w)
                
                min_len = min(hs_c[0].shape[1], hs_w[0].shape[1])
                target_pos = min_len - 1
                
                deltas = []
                dir_cos = []
                for l in range(n_layers + 1):
                    h_c = hs_c[l][0, target_pos].numpy()
                    h_w = hs_w[l][0, target_pos].numpy()
                    delta = np.linalg.norm(h_c - h_w)
                    deltas.append(float(delta))
                    
                    # 方向保持
                    norm_c = np.linalg.norm(h_c)
                    norm_w = np.linalg.norm(h_w)
                    if norm_c > 1e-10 and norm_w > 1e-10:
                        cos_val = float(np.dot(h_c, h_w) / (norm_c * norm_w))
                    else:
                        cos_val = 0.0
                    dir_cos.append(cos_val)
                
                # 输出差异
                h_c_final = hs_c[n_layers][0, target_pos].numpy()
                h_w_final = hs_w[n_layers][0, target_pos].numpy()
                logits_c = compute_logits(W_U, h_c_final) if W_U is not None else compute_logits_via_model(model, hs_c[n_layers], target_pos)
                logits_w = compute_logits(W_U, h_w_final) if W_U is not None else compute_logits_via_model(model, hs_w[n_layers], target_pos)
                
                top5_c = set(np.argsort(logits_c)[-5:][::-1].tolist())
                top5_w = set(np.argsort(logits_w)[-5:][::-1].tolist())
                top5_overlap = len(top5_c & top5_w) / 5.0
                top1_match = 1 if np.argmax(logits_c) == np.argmax(logits_w) else 0
                
                type_results["delta_trajectories"].append(deltas)
                type_results["direction_cosines"].append(dir_cos)
                type_results["output_divergences"].append({
                    "top5_overlap": top5_overlap,
                    "top1_match": top1_match,
                })
                
            except Exception as e:
                print(f"    对{pair_idx}失败: {e}")
                continue
            
            del hs_c, hs_w
            torch.cuda.empty_cache()
        
        if type_results["delta_trajectories"]:
            mean_deltas = np.mean(type_results["delta_trajectories"], axis=0)
            mean_cos = np.mean(type_results["direction_cosines"], axis=0)
            mean_top5 = np.mean([d["top5_overlap"] for d in type_results["output_divergences"]])
            mean_top1 = np.mean([d["top1_match"] for d in type_results["output_divergences"]])
            
            peak_idx = int(np.argmax(mean_deltas))
            peak_val = mean_deltas[peak_idx]
            final_val = mean_deltas[-1]
            ratio = final_val / peak_val if peak_val > 0 else 0
            
            results[constraint_type] = {
                "n_pairs": len(type_results["delta_trajectories"]),
                "mean_delta_trajectory": [float(x) for x in mean_deltas],
                "mean_direction_cosines": [float(x) for x in mean_cos],
                "peak_layer": peak_idx,
                "peak_delta": float(peak_val),
                "final_delta": float(final_val),
                "final_peak_ratio": float(ratio),
                "mean_top5_overlap": float(mean_top5),
                "mean_top1_match": float(mean_top1),
            }
        
        elapsed = time.time() - t0
        print(f"    {constraint_type}完成, 耗时{elapsed:.1f}s")
    
    return results


# ============================================================
# Exp E: W_U投影下的等价类
# ============================================================
def exp_e_wu_subspace(model, tokenizer, model_info, model_name, W_U):
    """
    W_U投影下的等价类分析
    
    核心问题: 扰动在W_U row space vs null space中的分量如何演化?
    如果null space中的扰动被放大但不影响输出 → 输出等价类稳定
    """
    print("\n" + "="*60)
    print("Exp E: W_U投影下的等价类")
    print("="*60)
    
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    input_device = get_device_for_input(model)
    
    # 预计算W_U的row space和null space投影
    if W_U is None:
        print("  W_U不可用, 跳过Exp E")
        return {}
    
    from scipy.sparse.linalg import svds
    # 8bit模型减小k值以节省内存
    max_k = 100 if model_name in ("deepseek7b", "glm4") else 200
    k = min(max_k, min(W_U.shape[0], W_U.shape[1]) - 2)
    W_U_f32 = W_U.astype(np.float32) if W_U.dtype == np.float16 else W_U
    U, S, Vt = svds(W_U_f32.T.astype(np.float32), k=k)
    del W_U_f32  # 释放临时float32副本
    U = U.astype(np.float32)  # 确保U是float32
    # U: [d_model, k] — W_U row space基
    # 扰动在row space中的投影 = U @ (U^T @ delta)
    # 扰动在null space中的分量 = delta - row_space_projection
    
    test_sentences = [
        "The scientist discovered a new element yesterday.",
        "Every student in the class passed the final exam.",
        "She has been working on this project for three years.",
        "The ancient castle stood on top of the mountain.",
        "The professor explained the complex theory to students.",
        "A small bird sang beautifully in the morning light.",
        "The company launched a revolutionary new product today.",
        "She always drinks coffee before starting her work.",
        "The river flows gently through the green valley.",
        "His latest novel became an instant bestseller worldwide.",
    ]
    
    inject_layers = [0, n_layers//2, n_layers-2]
    eps = 2.0
    
    results = {}
    
    for sent_idx, sentence in enumerate(test_sentences):
        t0 = time.time()
        print(f"\n  句子 {sent_idx+1}/{len(test_sentences)}: {sentence[:50]}...")
        
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        seq_len = input_ids.shape[1]
        target_pos = seq_len - 1
        
        clean_hs = get_clean_hidden_states(model, input_ids, attention_mask)
        
        for inject_l in inject_layers:
            if inject_l >= n_layers:
                continue
            
            key = f"sent{sent_idx}_L{inject_l}"
            
            # 随机扰动
            np.random.seed(42 + sent_idx)
            v_random = np.random.randn(d_model)
            v_random = v_random / np.linalg.norm(v_random) * eps
            
            result = perturb_with_hook(
                model, input_ids, attention_mask, model_info,
                inject_l, target_pos, v_random
            )
            
            if result is None:
                continue
            
            perturbed_hs, perturbed_logits = result
            
            # 计算每层扰动在W_U row space和null space中的分量
            row_space_energy = []
            null_space_energy = []
            total_energy = []
            
            for l in range(inject_l, n_layers + 1):
                h_c = clean_hs[l][0, target_pos].numpy()
                h_p = perturbed_hs[l][0, target_pos].numpy()
                delta = h_p - h_c
                
                # Row space projection
                proj_coeffs = U.T @ delta  # [k]
                row_proj = U @ proj_coeffs  # [d_model]
                null_proj = delta - row_proj
                
                row_energy = float(np.linalg.norm(row_proj) ** 2)
                null_energy = float(np.linalg.norm(null_proj) ** 2)
                total = row_energy + null_energy
                
                row_space_energy.append(row_energy)
                null_space_energy.append(null_energy)
                total_energy.append(total)
            
            # 计算输出影响
            h_clean_final = clean_hs[n_layers][0, target_pos].numpy()
            if W_U is not None:
                clean_logits = compute_logits(W_U, h_clean_final)
            else:
                clean_logits = compute_logits_via_model(model, clean_hs[n_layers], target_pos)
            clean_top5 = np.argsort(clean_logits)[-5:][::-1]
            perturbed_top5 = np.argsort(perturbed_logits)[-5:][::-1]
            
            top1_match = 1 if clean_top5[0] == perturbed_top5[0] else 0
            top5_overlap = len(set(clean_top5.tolist()) & set(perturbed_top5.tolist())) / 5.0
            
            results[key] = {
                "inject_layer": inject_l,
                "eps": eps,
                "sentence_idx": sent_idx,
                "row_space_energy": row_space_energy,
                "null_space_energy": null_space_energy,
                "total_energy": total_energy,
                "top1_match": top1_match,
                "top5_overlap": top5_overlap,
            }
            
            del perturbed_hs, perturbed_logits
            torch.cuda.empty_cache()
        
        del clean_hs
        torch.cuda.empty_cache()
        
        elapsed = time.time() - t0
        print(f"    句子{sent_idx+1}完成, 耗时{elapsed:.1f}s")
    
    return results


# ============================================================
# 主函数
# ============================================================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    assert model_name in MODEL_CONFIGS, f"Unknown model: {model_name}"
    
    print(f"\nPhase 146: 临界稳定传播系统 — 跨模型验证")
    print(f"模型: {model_name}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # 加载模型
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    use_8bit = model_name in ("deepseek7b", "glm4") and gpu_mem_gb < 16
    
    print(f"加载方式: {'8bit' if use_8bit else 'bfloat16'}")
    model, tokenizer, device = load_model(model_name, use_8bit=use_8bit)
    model_info = get_model_info(model, model_name)
    print(f"模型: {model_info.model_class}, {model_info.n_layers}层, d={model_info.d_model}")
    
    # 预加载W_U (8bit模型跳过, 改用模型lm_head计算logits以节省内存)
    # 8bit模型加载W_U from safetensors会导致OOM崩溃
    print("\n加载W_U...")
    use_model_logits = False
    W_U = None
    if use_8bit:
        print("8bit模型: 跳过W_U加载, 使用模型lm_head计算logits")
        use_model_logits = True
    else:
        try:
            W_U = get_W_U(model, model_name)
            print(f"W_U: shape={W_U.shape}, norm={np.linalg.norm(W_U):.1f}")
        except Exception as e:
            print(f"W_U加载失败({e}), 使用模型lm_head计算logits")
            W_U = None
            use_model_logits = True
    
    # 测试hook兼容性
    print("\n--- 测试hook兼容性 ---")
    layers = get_layers(model)
    input_device = get_device_for_input(model)
    test_inputs = tokenizer("Hello world", return_tensors="pt", truncation=True, max_length=32)
    test_ids = test_inputs["input_ids"].to(input_device)
    test_mask = test_inputs["attention_mask"].to(input_device)
    
    hook_works = False
    try:
        test_perturb = np.ones(model_info.d_model) * 0.01
        result = perturb_with_hook(model, test_ids, test_mask, model_info, 0, -1, test_perturb)
        if result is not None:
            hook_works = True
            print("  Hook测试通过!")
            del result
        else:
            print("  Hook测试失败, 将使用fallback")
    except Exception as e:
        print(f"  Hook测试失败: {e}")
    
    del test_ids, test_mask
    torch.cuda.empty_cache()
    
    all_results = {
        "model": model_name,
        "model_info": {
            "n_layers": model_info.n_layers,
            "d_model": model_info.d_model,
            "model_class": model_info.model_class,
        },
        "use_8bit": use_8bit,
        "hook_works": hook_works,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    
    # Exp A+: 输出等价稳定性
    print("\n\n" + "#"*60)
    print("# Exp A+: 输出等价稳定性")
    print("#"*60)
    try:
        exp_a = exp_a_output_equivalence(model, tokenizer, model_info, model_name, W_U)
        all_results["exp_a"] = exp_a
        print("Exp A+ 完成!")
    except Exception as e:
        print(f"Exp A+ 失败: {e}")
        import traceback; traceback.print_exc()
        all_results["exp_a_error"] = str(e)
    
    gc.collect(); torch.cuda.empty_cache()
    
    # Exp B+: 扰动方向演化
    print("\n\n" + "#"*60)
    print("# Exp B+: 扰动方向演化")
    print("#"*60)
    try:
        exp_b = exp_b_direction_evolution(model, tokenizer, model_info, model_name, W_U)
        all_results["exp_b"] = exp_b
        print("Exp B+ 完成!")
    except Exception as e:
        print(f"Exp B+ 失败: {e}")
        import traceback; traceback.print_exc()
        all_results["exp_b_error"] = str(e)
    
    gc.collect(); torch.cuda.empty_cache()
    
    # Exp D: 约束修复动力学
    print("\n\n" + "#"*60)
    print("# Exp D: 约束修复动力学")
    print("#"*60)
    try:
        exp_d = exp_d_constraint_dynamics(model, tokenizer, model_info, model_name, W_U)
        all_results["exp_d"] = exp_d
        print("Exp D 完成!")
    except Exception as e:
        print(f"Exp D 失败: {e}")
        import traceback; traceback.print_exc()
        all_results["exp_d_error"] = str(e)
    
    gc.collect(); torch.cuda.empty_cache()
    
    # Exp E: W_U投影下的等价类
    print("\n\n" + "#"*60)
    print("# Exp E: W_U投影下的等价类")
    print("#"*60)
    try:
        exp_e = exp_e_wu_subspace(model, tokenizer, model_info, model_name, W_U)
        all_results["exp_e"] = exp_e
        print("Exp E 完成!")
    except Exception as e:
        print(f"Exp E 失败: {e}")
        import traceback; traceback.print_exc()
        all_results["exp_e_error"] = str(e)
    
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
    out_path = TEMP_DIR / f"phase146_{model_name}_critical_propagation_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    print(f"\n结果保存到: {out_path}")
    
    # ===== 打印关键摘要 =====
    print("\n" + "="*60)
    print("关键结果摘要")
    print("="*60)
    
    if "exp_a" in all_results:
        exp_a = all_results["exp_a"]
        print("\nExp A+: 输出等价稳定性")
        for inject_l in [0, model_info.n_layers//2, model_info.n_layers-2]:
            for eps in [0.5, 1.0, 2.0, 5.0]:
                top1_matches = []
                top5_overlaps = []
                logits_corrs = []
                final_state_dists = []
                final_dir_cos = []
                for key, val in exp_a.items():
                    if val["inject_layer"] == inject_l and abs(val["eps"] - eps) < 0.01:
                        top1_matches.append(val["top1_match"])
                        top5_overlaps.append(val["top5_overlap"])
                        logits_corrs.append(val["logits_correlation"])
                        if val["normalized_state_dists"]:
                            final_state_dists.append(val["normalized_state_dists"][-1])
                        if val["direction_cosines"]:
                            final_dir_cos.append(val["direction_cosines"][-1])
                
                if top1_matches:
                    print(f"  L{inject_l}, eps={eps}: "
                          f"top1_match={np.mean(top1_matches):.3f}, "
                          f"top5_overlap={np.mean(top5_overlaps):.3f}, "
                          f"logits_corr={np.mean(logits_corrs):.4f}, "
                          f"state_dist={np.mean(final_state_dists):.2f}x, "
                          f"dir_cos={np.mean(final_dir_cos):.4f}")
    
    if "exp_b" in all_results:
        exp_b = all_results["exp_b"]
        print("\nExp B+: 扰动方向演化")
        for inject_l in [0, model_info.n_layers//2, model_info.n_layers-2]:
            alignments = []
            for key, val in exp_b.items():
                if val["inject_layer"] == inject_l:
                    if val["direction_alignment"]:
                        alignments.append(val["direction_alignment"])
            if alignments:
                mean_align = np.mean(alignments, axis=0)
                print(f"  L{inject_l}: 初始cos={mean_align[0]:.4f}, "
                      f"中间cos={mean_align[len(mean_align)//2]:.4f}, "
                      f"最终cos={mean_align[-1]:.4f}")
    
    if "exp_d" in all_results:
        print("\nExp D: 约束修复动力学")
        for c_type, c_data in all_results["exp_d"].items():
            print(f"  {c_type}: peak@L{c_data['peak_layer']}={c_data['peak_delta']:.1f}, "
                  f"ratio={c_data['final_peak_ratio']:.3f}, "
                  f"top5_overlap={c_data['mean_top5_overlap']:.3f}, "
                  f"top1_match={c_data['mean_top1_match']:.3f}")
    
    if "exp_e" in all_results:
        exp_e = all_results["exp_e"]
        print("\nExp E: W_U投影下的等价类")
        for inject_l in [0, model_info.n_layers//2, model_info.n_layers-2]:
            row_ratios = []
            null_ratios = []
            for key, val in exp_e.items():
                if val["inject_layer"] == inject_l:
                    total = np.array(val["total_energy"])
                    row = np.array(val["row_space_energy"])
                    null = np.array(val["null_space_energy"])
                    if total[-1] > 0:
                        row_ratios.append(row[-1] / total[-1])
                        null_ratios.append(null[-1] / total[-1])
            if row_ratios:
                print(f"  L{inject_l}: row_space_ratio={np.mean(row_ratios):.4f}, "
                      f"null_space_ratio={np.mean(null_ratios):.4f}")
    
    # 释放模型
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()
    print("\nPhase 146 完成!")


if __name__ == "__main__":
    main()
