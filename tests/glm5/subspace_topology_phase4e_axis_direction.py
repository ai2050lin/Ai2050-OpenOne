"""
Phase 58e: 语义轴方向分析 — 反义词在同一轴的方向
=================================================
核心问题: hot/cold共享高维度(overlap=0.30), 但它们如何区分?
  - 是否在同一语义轴的不同方向?
  - 方向编码了什么? (极性? 程度? 语义特征?)

方法:
  1. 提取hot和cold的各自子空间
  2. 计算共享子空间(两个子空间的交集)
  3. 在共享子空间中, 分析hot vs cold的均值偏移方向
  4. 解码偏移方向到W_U, 看它编码了什么
  5. 对比同义词(big/large)在同一分析中的结果
  6. 扩展到所有5类关系

关键对比:
  - 同义词: 共享轴 + 方向一致 → 偏移方向应编码"语体差异"
  - 反义词: 共享轴 + 方向相反 → 偏移方向应编码"极性"
  - 上下位: 部分共享 + 方向一致 → 偏移方向应编码"具体性/抽象性"
"""

import sys, json, numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

PROJECT = Path("d:/Ai2050/TransformerLens-Project")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tests" / "glm5"))

from model_utils import load_model, get_model_info, get_W_U, release_model, safe_decode
from subspace_topology_phase4b_backbone_decode import WORD_TEMPLATES, SEMANTIC_PAIRS
import torch

def log_time(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    safe_msg = msg.encode('ascii', errors='replace').decode('ascii')
    print(f"[{ts}] {safe_msg}", flush=True)


def find_target_pos_in_full(tokenizer, input_ids, target_word):
    tokens_list = input_ids[0].tolist()
    for i in range(len(tokens_list)):
        for j in range(i+1, min(i+5, len(tokens_list)+1)):
            decoded = tokenizer.decode(tokens_list[i:j])
            stripped = decoded.strip().lower()
            if stripped == target_word.lower():
                return i, j - i
    for i in range(len(tokens_list)):
        for j in range(i+1, min(i+5, len(tokens_list)+1)):
            decoded = tokenizer.decode(tokens_list[i:j])
            stripped = decoded.strip().lower()
            if stripped and target_word.lower() in stripped and len(stripped) <= len(target_word) + 3:
                return i, j - i
    return None, None


def collect_word_activations(model, tokenizer, device, word, templates, target_layers, n_layers):
    activations = {li: [] for li in target_layers}
    found = 0
    with torch.no_grad():
        for tmpl in templates:
            inputs = tokenizer(tmpl, return_tensors="pt", add_special_tokens=True)
            input_ids = inputs.input_ids.to(device)
            seq_len = input_ids.shape[1]
            pos, tlen = find_target_pos_in_full(tokenizer, input_ids, word)
            if pos is None or pos >= seq_len:
                continue
            actual_pos = min(pos + (tlen // 2), seq_len - 1)
            found += 1
            outputs = model(input_ids, output_hidden_states=True)
            hidden = outputs.hidden_states
            for li in target_layers:
                activations[li].append(hidden[li + 1][0, actual_pos].detach().cpu().float().numpy())
    return activations, found


def pca_subspace(vectors, n_dims=10):
    X = np.array(vectors)
    mean = X.mean(axis=0)
    X_c = X - mean
    U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    n = min(n_dims, Vt.shape[0])
    return Vt[:n].T, (S ** 2) / len(X_c), mean


def decode_direction(direction, W_U, tokenizer, top_k=20):
    logits = W_U @ direction
    exp_logits = np.exp(logits - logits.max())
    probs = exp_logits / exp_logits.sum()
    top_indices = np.argsort(probs)[::-1][:top_k]
    return [{"token": safe_decode(tokenizer, idx), "prob": float(probs[idx]),
             "logit": float(logits[idx])} for idx in top_indices]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen3", "glm4", "deepseek7b"], required=True)
    args = parser.parse_args()
    
    model_name = args.model
    n_dims = 10
    
    log_time(f"Loading {model_name}...")
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    W_U = get_W_U(model, model_name)
    d_model = info.d_model
    n_layers = info.n_layers
    
    # 只分析关键层
    target_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    target_layers = sorted(set(target_layers))
    log_time(f"{model_name}: layers={target_layers}")
    
    # 选择代表对
    test_pairs = [
        ("synonym_1", "big", "large", "synonym"),
        ("synonym_4", "begin", "start", "synonym"),
        ("antonym_1", "hot", "cold", "antonym"),
        ("antonym_3", "love", "hate", "antonym"),
        ("hyponym_1", "apple", "fruit", "hyponym"),
        ("hyponym_2", "dog", "animal", "hyponym"),
        ("associated_1", "doctor", "hospital", "associated"),
        ("unrelated_1", "apple", "planet", "unrelated"),
    ]
    
    results = {}
    
    for pair_key, w_a, w_b, relation in test_pairs:
        log_time(f"Analyzing {pair_key}: {w_a}/{w_b} ({relation})")
        
        if w_a not in WORD_TEMPLATES or w_b not in WORD_TEMPLATES:
            log_time(f"  SKIP: templates missing")
            continue
        
        acts_a, found_a = collect_word_activations(
            model, tokenizer, device, w_a, WORD_TEMPLATES[w_a], target_layers, n_layers)
        acts_b, found_b = collect_word_activations(
            model, tokenizer, device, w_b, WORD_TEMPLATES[w_b], target_layers, n_layers)
        
        log_time(f"  Found: {w_a}={found_a}, {w_b}={found_b}")
        
        layer_data = {}
        for li in target_layers:
            if len(acts_a[li]) < 2 or len(acts_b[li]) < 2:
                continue
            
            # 1. 各自PCA
            basis_a, eigvals_a, mean_a = pca_subspace(acts_a[li], n_dims)
            basis_b, eigvals_b, mean_b = pca_subspace(acts_b[li], n_dims)
            
            # 2. 均值偏移方向
            delta = mean_a - mean_b  # A相对B的偏移
            delta_norm = np.linalg.norm(delta)
            delta_unit = delta / (delta_norm + 1e-10)
            
            # 3. 偏移方向在各自子空间中的投影
            proj_delta_to_a = basis_a @ basis_a.T @ delta_unit
            proj_delta_to_b = basis_b @ basis_b.T @ delta_unit
            
            # 偏移方向在A子空间中的能量比
            energy_in_a = np.sum(proj_delta_to_a ** 2)
            energy_in_b = np.sum(proj_delta_to_b ** 2)
            
            # 4. 解码偏移方向
            delta_decoded = decode_direction(delta_unit, W_U, tokenizer, top_k=15)
            
            # 5. 反方向解码 (-delta = B相对A的方向)
            neg_delta_decoded = decode_direction(-delta_unit, W_U, tokenizer, top_k=15)
            
            # 6. 在共享子空间中的投影方向
            # 共享子空间 = 两个子空间的"交集"方向
            # 用A的基在B子空间中的投影来找共享方向
            shared_proj = basis_b.T @ basis_a  # [n_b, n_a]
            # SVD找最大共享方向
            U_s, s_s, Vt_s = np.linalg.svd(shared_proj)
            # 共享方向 = A空间中的Vt_s[0] 和 B空间中的U_s[0]
            
            # 7. 在共享方向上, A和B的均值偏移
            if len(s_s) > 0 and s_s[0] > 0.01:
                shared_dir_in_a = basis_a @ Vt_s[0]  # 在原空间中的共享方向1
                shared_dir_in_b = basis_b @ U_s[:, 0]
                # 用A方向的共享轴
                shared_axis = shared_dir_in_a / (np.linalg.norm(shared_dir_in_a) + 1e-10)
                
                # A和B在共享轴上的位置
                pos_a_on_axis = np.dot(mean_a, shared_axis)
                pos_b_on_axis = np.dot(mean_b, shared_axis)
                axis_separation = pos_a_on_axis - pos_b_on_axis
                
                # 解码共享轴
                shared_decoded = decode_direction(shared_axis, W_U, tokenizer, top_k=10)
                neg_shared_decoded = decode_direction(-shared_axis, W_U, tokenizer, top_k=10)
            else:
                shared_axis = None
                pos_a_on_axis = 0
                pos_b_on_axis = 0
                axis_separation = 0
                shared_decoded = []
                neg_shared_decoded = []
            
            layer_data[str(li)] = {
                "delta_energy_in_a": float(energy_in_a),
                "delta_energy_in_b": float(energy_in_b),
                "delta_norm": float(delta_norm),
                "pos_a_on_axis": float(pos_a_on_axis),
                "pos_b_on_axis": float(pos_b_on_axis),
                "axis_separation": float(axis_separation),
                "shared_singular_value": float(s_s[0]) if len(s_s) > 0 else 0,
                "delta_decoded_top5": [{"token": d["token"], "prob": d["prob"]} for d in delta_decoded[:5]],
                "neg_delta_decoded_top5": [{"token": d["token"], "prob": d["prob"]} for d in neg_delta_decoded[:5]],
                "shared_decoded_top5": [{"token": d["token"], "prob": d["prob"]} for d in shared_decoded[:5]],
                "neg_shared_decoded_top5": [{"token": d["token"], "prob": d["prob"]} for d in neg_shared_decoded[:5]],
            }
        
        results[pair_key] = {"relation": relation, "w_a": w_a, "w_b": w_b, "layers": layer_data}
    
    # 保存
    output = {
        "model": model_name, "n_dims": n_dims,
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }
    
    out_dir = PROJECT / "results" / "subspace_topology"
    out_file = out_dir / f"exp4e_axis_direction_{model_name}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    log_time(f"Saved to {out_file}")
    
    # 摘要
    log_time("")
    log_time("=" * 70)
    log_time(f"PHASE 58e: Axis Direction Analysis - {model_name}")
    log_time("=" * 70)
    
    for pk, pd in results.items():
        log_time(f"\n--- {pk}: {pd['w_a']}/{pd['w_b']} ({pd['relation']}) ---")
        for lk in sorted(pd["layers"].keys(), key=int):
            ld = pd["layers"][lk]
            delta_top = [t["token"].strip()[:10] for t in ld["delta_decoded_top5"][:3]]
            neg_top = [t["token"].strip()[:10] for t in ld["neg_delta_decoded_top5"][:3]]
            log_time(f"  L{lk}: axis_sep={ld['axis_separation']:.3f} sv={ld['shared_singular_value']:.3f}")
            log_time(f"    delta({pd['w_a']}->{pd['w_b']}): {delta_top}")
            log_time(f"    -delta({pd['w_b']}->{pd['w_a']}): {neg_top}")
            if ld["shared_decoded_top5"]:
                shared_top = [t["token"].strip()[:10] for t in ld["shared_decoded_top5"][:3]]
                neg_shared_top = [t["token"].strip()[:10] for t in ld["neg_shared_decoded_top5"][:3]]
                log_time(f"    shared_axis+: {shared_top}")
                log_time(f"    shared_axis-: {neg_shared_top}")
    
    release_model(model)
    log_time("Done!")


if __name__ == "__main__":
    main()
