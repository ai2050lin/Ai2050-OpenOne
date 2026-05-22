"""
Phase 58: 骨干子空间语义解码 + 扩大概念对 + 神经元级归属
=====================================================
核心目标: 破解编码机制 — 回答3个根本问题:
  1. 骨干子空间编码了什么? (语义解码)
  2. 共享比例与语义相关性的函数关系? (shared_ratio = f(semantic_similarity))
  3. 哪些神经元属于骨干? 哪些属于特异? (神经元级归属)

实验设计:
  Part 1: 20+概念对 — 5类语义关系
    - 上下位 (apple/fruit, dog/animal, red/color, Paris/city, piano/instrument)
    - 同义 (big/large, happy/glad, fast/quick, begin/start, beautiful/pretty)
    - 反义 (hot/cold, up/down, love/hate, light/dark, young/old)
    - 相关 (doctor/hospital, chef/kitchen, teacher/school, bird/nest, fish/water)
    - 无关 (apple/planet, dog/math, red/democracy, piano/bacteria, city/poem)

  Part 2: 骨干子空间语义解码
    - 收集所有概念的激活, 提取骨干方向
    - 投影骨干方向到W_U, 解码top-50词
    - 投影特异方向到W_U, 对比解码内容

  Part 3: 神经元级归属
    - 计算每个神经元在骨干vs特异子空间中的投影能量
    - 识别"骨干神经元"vs"特异神经元"
    - 分析骨干神经元的层分布和功能

  Part 4: shared_ratio = f(semantic_similarity) 函数拟合
    - 用WordNet/人工标注的语义距离
    - 拟合shared_ratio与语义距离的函数关系

跨模型: Qwen3, DS7B, GLM4 (依次运行)
"""

import sys
import os
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

PROJECT = Path("d:/Ai2050/TransformerLens-Project")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tests" / "glm5"))

from model_utils import (
    load_model, get_layers, get_model_info, get_layer_weights,
    get_W_U, release_model, safe_decode, MODEL_CONFIGS
)

def log_time(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    safe_msg = msg.encode('ascii', errors='replace').decode('ascii')
    print(f"[{ts}] {safe_msg}", flush=True)

# ===== 20+概念对: 5类语义关系 =====
# 每对10个模板句子, 控制语法一致

SEMANTIC_PAIRS = {
    # === 上下位 (Hyponym-Hypernym) === 语义距离=1
    "hyponym_1": {"w_a": "apple", "w_b": "fruit", "relation": "hyponym",
                  "distance": 1, "desc": "apple/fruit"},
    "hyponym_2": {"w_a": "dog", "w_b": "animal", "relation": "hyponym",
                  "distance": 1, "desc": "dog/animal"},
    "hyponym_3": {"w_a": "red", "w_b": "color", "relation": "hyponym",
                  "distance": 1, "desc": "red/color"},
    "hyponym_4": {"w_a": "Paris", "w_b": "city", "relation": "hyponym",
                  "distance": 1, "desc": "Paris/city"},
    "hyponym_5": {"w_a": "piano", "w_b": "instrument", "relation": "hyponym",
                  "distance": 1, "desc": "piano/instrument"},

    # === 同义 (Synonym) === 语义距离=2
    "synonym_1": {"w_a": "big", "w_b": "large", "relation": "synonym",
                  "distance": 2, "desc": "big/large"},
    "synonym_2": {"w_a": "happy", "w_b": "glad", "relation": "synonym",
                  "distance": 2, "desc": "happy/glad"},
    "synonym_3": {"w_a": "fast", "w_b": "quick", "relation": "synonym",
                  "distance": 2, "desc": "fast/quick"},
    "synonym_4": {"w_a": "begin", "w_b": "start", "relation": "synonym",
                  "distance": 2, "desc": "begin/start"},
    "synonym_5": {"w_a": "beautiful", "w_b": "pretty", "relation": "synonym",
                  "distance": 2, "desc": "beautiful/pretty"},

    # === 反义 (Antonym) === 语义距离=3
    "antonym_1": {"w_a": "hot", "w_b": "cold", "relation": "antonym",
                  "distance": 3, "desc": "hot/cold"},
    "antonym_2": {"w_a": "up", "w_b": "down", "relation": "antonym",
                  "distance": 3, "desc": "up/down"},
    "antonym_3": {"w_a": "love", "w_b": "hate", "relation": "antonym",
                  "distance": 3, "desc": "love/hate"},
    "antonym_4": {"w_a": "light", "w_b": "dark", "relation": "antonym",
                  "distance": 3, "desc": "light/dark"},
    "antonym_5": {"w_a": "young", "w_b": "old", "relation": "antonym",
                  "distance": 3, "desc": "young/old"},

    # === 相关 (Associated) === 语义距离=4
    "associated_1": {"w_a": "doctor", "w_b": "hospital", "relation": "associated",
                     "distance": 4, "desc": "doctor/hospital"},
    "associated_2": {"w_a": "chef", "w_b": "kitchen", "relation": "associated",
                     "distance": 4, "desc": "chef/kitchen"},
    "associated_3": {"w_a": "teacher", "w_b": "school", "relation": "associated",
                     "distance": 4, "desc": "teacher/school"},
    "associated_4": {"w_a": "bird", "w_b": "nest", "relation": "associated",
                     "distance": 4, "desc": "bird/nest"},
    "associated_5": {"w_a": "fish", "w_b": "water", "relation": "associated",
                     "distance": 4, "desc": "fish/water"},

    # === 无关 (Unrelated) === 语义距离=5
    "unrelated_1": {"w_a": "apple", "w_b": "planet", "relation": "unrelated",
                    "distance": 5, "desc": "apple/planet"},
    "unrelated_2": {"w_a": "dog", "w_b": "math", "relation": "unrelated",
                    "distance": 5, "desc": "dog/math"},
    "unrelated_3": {"w_a": "red", "w_b": "democracy", "relation": "unrelated",
                    "distance": 5, "desc": "red/democracy"},
    "unrelated_4": {"w_a": "piano", "w_b": "bacteria", "relation": "unrelated",
                    "distance": 5, "desc": "piano/bacteria"},
    "unrelated_5": {"w_a": "city", "w_b": "poem", "relation": "unrelated",
                    "distance": 5, "desc": "city/poem"},
}

# 控制语法的统一模板 — 所有概念对使用相同语法结构
CONTROLLED_TEMPLATES = [
    "The {w} is very interesting",
    "I think about the {w} often",
    "She mentioned the {w} yesterday",
    "We discussed the {w} carefully",
    "They found the {w} nearby",
    "He described the {w} clearly",
    "The book mentions the {w}",
    "Everyone knows about the {w}",
    "I remember the {w} well",
    "The {w} caught my attention",
    "Please consider the {w} again",
    "Someone brought up the {w}",
    "The {w} appeared suddenly",
    "We observed the {w} closely",
    "The {w} was quite remarkable",
]

# ===== 核心函数 =====

def find_target_pos_in_full(tokenizer, input_ids, target_word):
    """在完整token序列中找目标词位置"""
    tokens_list = input_ids[0].tolist()
    
    # 策略1: 逐token解码精确匹配
    for i in range(len(tokens_list)):
        for j in range(i+1, min(i+5, len(tokens_list)+1)):
            decoded = tokenizer.decode(tokens_list[i:j])
            stripped = decoded.strip().lower()
            if stripped == target_word.lower():
                return i, j - i
    
    # 策略2: 宽松匹配
    for i in range(len(tokens_list)):
        for j in range(i+1, min(i+5, len(tokens_list)+1)):
            decoded = tokenizer.decode(tokens_list[i:j])
            stripped = decoded.strip().lower()
            if stripped and target_word.lower() in stripped and len(stripped) <= len(target_word) + 3:
                return i, j - i
    
    return None, None


def collect_word_activations(model, tokenizer, device, word, templates,
                              target_layers, n_layers):
    """收集一个词在所有模板中的激活"""
    activations = {li: [] for li in target_layers}
    found = 0
    
    with torch.no_grad():
        for tmpl in templates:
            sentence = tmpl.replace("{w}", word)
            inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
            input_ids = inputs.input_ids.to(device)
            seq_len = input_ids.shape[1]
            
            pos, tlen = find_target_pos_in_full(tokenizer, input_ids, word)
            if pos is None or pos >= seq_len:
                continue
            
            actual_pos = pos + (tlen // 2)
            actual_pos = min(actual_pos, seq_len - 1)
            found += 1
            
            outputs = model(input_ids, output_hidden_states=True)
            hidden = outputs.hidden_states
            
            for li in target_layers:
                layer_act = hidden[li + 1][0, actual_pos].detach().cpu().float().numpy()
                activations[li].append(layer_act)
    
    return activations, found


def extract_subspace(vectors, n_dims=15):
    """PCA提取子空间"""
    if len(vectors) < 2:
        return None, None, None
    
    X = np.array(vectors)
    mean = X.mean(axis=0)
    X_centered = X - mean
    
    cov = X_centered.T @ X_centered / len(X_centered)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    n = min(n_dims, len(eigenvalues))
    return eigenvectors[:, :n], eigenvalues[:n], mean


def compute_subspace_overlap(basis_a, basis_b):
    """计算两个子空间的重叠度"""
    if basis_a is None or basis_b is None:
        return 0.0
    proj = basis_b.T @ basis_a @ basis_a.T @ basis_b
    return np.trace(proj) / min(basis_a.shape[1], basis_b.shape[1])


def compute_shared_specific_subspace(activations_a, activations_b, n_dims=15):
    """提取共享子空间和独特子空间"""
    all_acts = activations_a + activations_b
    X_all = np.array(all_acts)
    mean_all = X_all.mean(axis=0)
    
    # 整体PCA
    X_centered = X_all - mean_all
    cov = X_centered.T @ X_centered / len(X_centered)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    n = min(n_dims, len(eigenvalues))
    shared_basis = eigenvectors[:, :n]
    
    # 投影到共享子空间
    X_a = np.array(activations_a) - mean_all
    X_b = np.array(activations_b) - mean_all
    
    proj_a = X_a @ shared_basis @ shared_basis.T
    proj_b = X_b @ shared_basis @ shared_basis.T
    
    # 残差(独特部分)
    res_a = X_a - proj_a
    res_b = X_b - proj_b
    
    # 方差分解
    var_total_a = np.sum(X_a ** 2)
    var_shared_a = np.sum(proj_a ** 2)
    var_unique_a = np.sum(res_a ** 2)
    
    var_total_b = np.sum(X_b ** 2)
    var_shared_b = np.sum(proj_b ** 2)
    var_unique_b = np.sum(res_b ** 2)
    
    shared_ratio_a = var_shared_a / max(var_total_a, 1e-10)
    shared_ratio_b = var_shared_b / max(var_total_b, 1e-10)
    
    # 均值差异分析
    mean_a = np.array(activations_a).mean(axis=0) - mean_all
    mean_b = np.array(activations_b).mean(axis=0) - mean_all
    delta = mean_a - mean_b
    
    # delta在共享vs独特子空间中的能量
    delta_shared = shared_basis.T @ delta
    delta_shared_energy = np.sum(delta_shared ** 2)
    delta_total_energy = np.sum(delta ** 2)
    
    delta_unique_ratio = 1.0 - delta_shared_energy / max(delta_total_energy, 1e-10)
    
    # cos相似度
    cos = np.dot(mean_a, mean_b) / (np.linalg.norm(mean_a) * np.linalg.norm(mean_b) + 1e-10)
    
    return {
        "shared_ratio_A": float(shared_ratio_a),
        "shared_ratio_B": float(shared_ratio_b),
        "avg_shared_ratio": float((shared_ratio_a + shared_ratio_b) / 2),
        "delta_unique_ratio": float(delta_unique_ratio),
        "cos_mean": float(cos),
        "subspace_overlap": float(compute_subspace_overlap(
            extract_subspace(activations_a, n_dims)[0],
            extract_subspace(activations_b, n_dims)[0]
        )) if len(activations_a) >= 2 and len(activations_b) >= 2 else 0.0,
        "n_samples_a": len(activations_a),
        "n_samples_b": len(activations_b),
    }, shared_basis, mean_all, eigenvalues[:n]


def decode_direction_to_vocab(direction, W_U, tokenizer, top_k=50):
    """将一个方向投影到W_U, 解码top-k词"""
    logits = W_U @ direction  # [vocab_size]
    
    # softmax for probabilities
    exp_logits = np.exp(logits - logits.max())
    probs = exp_logits / exp_logits.sum()
    
    top_indices = np.argsort(probs)[::-1][:top_k]
    
    decoded = []
    for idx in top_indices:
        token_str = safe_decode(tokenizer, idx)
        decoded.append({
            "token": token_str,
            "index": int(idx),
            "logit": float(logits[idx]),
            "prob": float(probs[idx]),
        })
    return decoded


def compute_neuron_attribution(activations_dict, shared_basis, mean_all, d_model):
    """
    计算每个神经元的骨干/特异归属
    
    activations_dict: {pair_key: {"a": [vec, ...], "b": [vec, ...]}}
    shared_basis: 骨干子空间基 [d_model, n_dims]
    mean_all: 均值向量 [d_model]
    """
    # 收集所有激活 (已经是该层的激活列表)
    all_acts = []
    for pair_key, pair_data in activations_dict.items():
        for word_key in ["a", "b"]:
            acts = pair_data[word_key]
            if isinstance(acts, list):
                all_acts.extend(acts)
    
    if len(all_acts) < 2:
        return None
    
    X = np.array(all_acts)
    X_centered = X - mean_all
    
    # 每个神经元的总方差
    total_var = np.var(X_centered, axis=0)  # [d_model]
    
    # 投影到共享子空间
    proj = X_centered @ shared_basis  # [N, n_dims]
    recon = proj @ shared_basis.T  # [N, d_model]
    res = X_centered - recon
    
    # 每个神经元在共享子空间中的方差
    shared_var = np.var(recon, axis=0)  # [d_model]
    unique_var = np.var(res, axis=0)  # [d_model]
    
    # 骨干归属度
    backbone_score = shared_var / (total_var + 1e-10)
    
    return {
        "backbone_score": backbone_score.tolist(),  # [d_model]
        "total_var": total_var.tolist(),
        "shared_var": shared_var.tolist(),
        "unique_var": unique_var.tolist(),
        "top_backbone_neurons": np.argsort(backbone_score)[::-1][:50].tolist(),
        "top_specific_neurons": np.argsort(backbone_score)[:50].tolist(),
        "mean_backbone_score": float(backbone_score.mean()),
        "median_backbone_score": float(np.median(backbone_score)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen3", "glm4", "deepseek7b"], required=True)
    parser.add_argument("--n_dims", type=int, default=20, help="子空间维度")
    args = parser.parse_args()
    
    model_name = args.model
    n_dims = args.n_dims
    
    # 加载模型
    log_time(f"Loading {model_name}...")
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    W_U = get_W_U(model, model_name)
    d_model = info.d_model
    
    # 采样层
    n_layers = info.n_layers
    target_layers = sorted(set([0, 1] + list(range(0, n_layers, max(1, n_layers // 10)))))
    log_time(f"{model_name}: n_layers={n_layers}, d_model={d_model}, target_layers={target_layers}")
    
    # ===== Part 1: 25对概念对的子空间分析 =====
    log_time(f"Part 1: Collecting activations for {len(SEMANTIC_PAIRS)} pairs...")
    
    all_pair_results = {}
    all_activations = {}  # {pair_key: {"a": {layer: [vecs]}, "b": {layer: [vecs]}}}
    
    for pair_key, pair_info in SEMANTIC_PAIRS.items():
        w_a = pair_info["w_a"]
        w_b = pair_info["w_b"]
        log_time(f"  Pair {pair_key}: {w_a}/{w_b} ({pair_info['relation']})")
        
        acts_a, found_a = collect_word_activations(
            model, tokenizer, device, w_a, CONTROLLED_TEMPLATES, target_layers, n_layers)
        acts_b, found_b = collect_word_activations(
            model, tokenizer, device, w_b, CONTROLLED_TEMPLATES, target_layers, n_layers)
        
        log_time(f"    Found: {w_a}={found_a}/15, {w_b}={found_b}/15")
        
        all_activations[pair_key] = {"a": acts_a, "b": acts_b}
        
        layer_results = {}
        for li in target_layers:
            if len(acts_a[li]) >= 2 and len(acts_b[li]) >= 2:
                metrics, shared_basis, mean_all, eigvals = compute_shared_specific_subspace(
                    acts_a[li], acts_b[li], n_dims)
                layer_results[str(li)] = metrics
            else:
                layer_results[str(li)] = {"error": "insufficient samples"}
        
        all_pair_results[pair_key] = {
            "pair_info": pair_info,
            "layers": layer_results,
        }
    
    # ===== Part 2: 骨干子空间提取与语义解码 =====
    log_time("Part 2: Extracting backbone subspace and decoding...")
    
    backbone_decode_results = {}
    
    for li in target_layers:
        log_time(f"  Layer {li}: collecting all concept activations...")
        
        # 收集所有概念的所有激活
        all_concept_acts = []
        concept_labels = []  # (pair_key, word_key)
        
        for pair_key, pair_data in all_activations.items():
            for word_key in ["a", "b"]:
                acts = pair_data[word_key].get(li, [])
                if len(acts) >= 1:
                    all_concept_acts.extend(acts)
                    concept_labels.extend([(pair_key, word_key)] * len(acts))
        
        if len(all_concept_acts) < 10:
            log_time(f"    Skipping layer {li}: only {len(all_concept_acts)} samples")
            continue
        
        X = np.array(all_concept_acts)
        mean_all = X.mean(axis=0)
        X_centered = X - mean_all
        
        # 整体PCA — 提取骨干方向
        cov = X_centered.T @ X_centered / len(X_centered)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 骨干 = 前20个主成分
        n_backbone = min(20, len(eigenvalues))
        backbone_basis = eigenvectors[:, :n_backbone]
        backbone_eigvals = eigenvalues[:n_backbone]
        
        # 计算方差解释比
        total_var = eigenvalues.sum()
        backbone_var_ratio = backbone_eigvals.sum() / total_var
        
        # 解码骨干方向
        log_time(f"    Decoding backbone directions (var_ratio={backbone_var_ratio:.3f})...")
        backbone_decoded = []
        for d in range(min(n_backbone, 5)):  # 解码前5个骨干方向
            direction = backbone_basis[:, d]
            top_words = decode_direction_to_vocab(direction, W_U, tokenizer, top_k=30)
            backbone_decoded.append({
                "direction_idx": d,
                "eigenvalue": float(backbone_eigvals[d]),
                "var_explained": float(backbone_eigvals[d] / total_var),
                "top_words": top_words[:15],
            })
        
        # 解码特异方向 (PC 20-40)
        specific_decoded = []
        for d in range(n_backbone, min(n_backbone + 5, len(eigenvalues))):
            direction = eigenvectors[:, d]
            top_words = decode_direction_to_vocab(direction, W_U, tokenizer, top_k=30)
            specific_decoded.append({
                "direction_idx": d,
                "eigenvalue": float(eigenvalues[d]),
                "var_explained": float(eigenvalues[d] / total_var),
                "top_words": top_words[:15],
            })
        
        # 神经元级归属
        log_time(f"    Computing neuron attribution...")
        # 收集该层所有概念的激活
        layer_acts_for_attr = {}
        for pk in all_activations:
            layer_acts_for_attr[pk] = {
                "a": all_activations[pk]["a"].get(li, []),
                "b": all_activations[pk]["b"].get(li, []),
            }
        neuron_attr = compute_neuron_attribution(
            layer_acts_for_attr, backbone_basis, mean_all, d_model
        )
        
        # 骨干方向之间的配对重叠度
        pair_overlaps = {}
        for pk1 in list(SEMANTIC_PAIRS.keys())[:3]:  # 只取3对避免计算量太大
            for pk2 in list(SEMANTIC_PAIRS.keys())[:3]:
                if pk1 >= pk2:
                    continue
                acts_a1 = all_activations[pk1]["a"].get(li, [])
                acts_b1 = all_activations[pk1]["b"].get(li, [])
                acts_a2 = all_activations[pk2]["a"].get(li, [])
                acts_b2 = all_activations[pk2]["b"].get(li, [])
                
                if len(acts_a1) >= 2 and len(acts_b1) >= 2 and len(acts_a2) >= 2 and len(acts_b2) >= 2:
                    basis1_a, _, _ = extract_subspace(acts_a1, n_dims=10)
                    basis1_b, _, _ = extract_subspace(acts_b1, n_dims=10)
                    basis2_a, _, _ = extract_subspace(acts_a2, n_dims=10)
                    basis2_b, _, _ = extract_subspace(acts_b2, n_dims=10)
                    
                    if basis1_a is not None and basis2_a is not None:
                        pair_overlaps[f"{pk1}_vs_{pk2}"] = {
                            "a_a_overlap": float(compute_subspace_overlap(basis1_a, basis2_a)),
                            "b_b_overlap": float(compute_subspace_overlap(basis1_b, basis2_b)),
                        }
        
        backbone_decode_results[str(li)] = {
            "backbone_var_ratio": float(backbone_var_ratio),
            "backbone_decoded": backbone_decoded,
            "specific_decoded": specific_decoded,
            "neuron_attribution": neuron_attr,
            "pair_overlaps": pair_overlaps,
            "n_total_samples": len(all_concept_acts),
            "eigenvalue_spectrum": [float(e) for e in eigenvalues[:30]],
        }
    
    # ===== Part 3: shared_ratio = f(semantic_similarity) =====
    log_time("Part 3: Fitting shared_ratio = f(semantic_similarity)...")
    
    # 中间层结果
    mid_layer = target_layers[len(target_layers) // 2]
    
    similarity_data = []
    for pair_key, pair_result in all_pair_results.items():
        layer_data = pair_result["layers"].get(str(mid_layer), {})
        if "error" not in layer_data:
            similarity_data.append({
                "pair_key": pair_key,
                "relation": pair_result["pair_info"]["relation"],
                "semantic_distance": pair_result["pair_info"]["distance"],
                "shared_ratio": layer_data.get("avg_shared_ratio", 0),
                "cos_mean": layer_data.get("cos_mean", 0),
                "delta_unique_ratio": layer_data.get("delta_unique_ratio", 0),
            })
    
    # 按relation分组统计
    relation_stats = defaultdict(lambda: {"shared_ratios": [], "cos_means": [], "delta_uniques": []})
    for sd in similarity_data:
        rel = sd["relation"]
        relation_stats[rel]["shared_ratios"].append(sd["shared_ratio"])
        relation_stats[rel]["cos_means"].append(sd["cos_mean"])
        relation_stats[rel]["delta_uniques"].append(sd["delta_unique_ratio"])
    
    relation_summary = {}
    for rel, stats in relation_stats.items():
        relation_summary[rel] = {
            "mean_shared_ratio": float(np.mean(stats["shared_ratios"])),
            "std_shared_ratio": float(np.std(stats["shared_ratios"])),
            "mean_cos": float(np.mean(stats["cos_means"])),
            "std_cos": float(np.std(stats["cos_means"])),
            "mean_delta_unique": float(np.mean(stats["delta_uniques"])),
            "n_pairs": len(stats["shared_ratios"]),
        }
    
    # ===== 保存结果 =====
    output = {
        "model": model_name,
        "n_dims": n_dims,
        "target_layers": target_layers,
        "n_layers": n_layers,
        "d_model": d_model,
        "n_concept_pairs": len(SEMANTIC_PAIRS),
        "pair_results": all_pair_results,
        "backbone_decode": backbone_decode_results,
        "similarity_function": {
            "mid_layer": mid_layer,
            "per_pair_data": similarity_data,
            "relation_summary": relation_summary,
        },
        "timestamp": datetime.now().isoformat(),
    }
    
    out_dir = PROJECT / "results" / "subspace_topology"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"exp4_backbone_decode_{model_name}.json"
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    log_time(f"Results saved to {out_file}")
    
    # ===== 摘要输出 =====
    log_time("")
    log_time("=" * 60)
    log_time(f"PHASE 58 SUMMARY - {model_name}")
    log_time("=" * 60)
    
    log_time("")
    log_time("--- Part 1: Concept Pair Shared Ratios (mid layer) ---")
    for sd in sorted(similarity_data, key=lambda x: x["semantic_distance"]):
        log_time(f"  {sd['pair_key']:20s} dist={sd['semantic_distance']} "
                 f"shared={sd['shared_ratio']:.3f} cos={sd['cos_mean']:.3f} "
                 f"delta_unique={sd['delta_unique_ratio']:.3f}")
    
    log_time("")
    log_time("--- Part 3: Relation Summary ---")
    for rel in ["hyponym", "synonym", "antonym", "associated", "unrelated"]:
        if rel in relation_summary:
            rs = relation_summary[rel]
            log_time(f"  {rel:12s}: shared={rs['mean_shared_ratio']:.3f}+-{rs['std_shared_ratio']:.3f} "
                     f"cos={rs['mean_cos']:.3f}+-{rs['std_cos']:.3f} "
                     f"delta_unique={rs['mean_delta_unique']:.3f} n={rs['n_pairs']}")
    
    log_time("")
    log_time("--- Part 2: Backbone Decode (key layers) ---")
    for li_str in sorted(backbone_decode_results.keys(), key=int):
        bd = backbone_decode_results[li_str]
        log_time(f"  L{li_str}: backbone_var={bd['backbone_var_ratio']:.3f}, "
                 f"samples={bd['n_total_samples']}")
        # 解码top-5骨干方向
        for d_info in bd["backbone_decoded"][:3]:
            top5 = " ".join([t["token"].strip() for t in d_info["top_words"][:5]])
            log_time(f"    PC{d_info['direction_idx']}: var={d_info['var_explained']:.4f} "
                     f"top=[{top5}]")
    
    if backbone_decode_results:
        # 神经元归属摘要
        mid_key = str(mid_layer)
        if mid_key in backbone_decode_results:
            na = backbone_decode_results[mid_key].get("neuron_attribution")
            if na:
                log_time("")
                log_time("--- Part 2b: Neuron Attribution (mid layer) ---")
                log_time(f"  Mean backbone score: {na['mean_backbone_score']:.4f}")
                log_time(f"  Median backbone score: {na['median_backbone_score']:.4f}")
                top_bn = na["top_backbone_neurons"][:10]
                log_time(f"  Top backbone neurons: {top_bn}")
                # 解码骨干神经元
                for ni in top_bn[:5]:
                    direction = np.zeros(d_model)
                    direction[ni] = 1.0
                    top_words = decode_direction_to_vocab(direction, W_U, tokenizer, top_k=5)
                    tw = " ".join([t["token"].strip() for t in top_words[:3]])
                    log_time(f"    Neuron {ni}: backbone_score={na['backbone_score'][ni]:.4f} top=[{tw}]")
    
    # 释放模型
    release_model(model)
    log_time("Done!")


if __name__ == "__main__":
    import torch
    main()
