"""
Phase 58c: 正确的概念复用/差异化分析
=====================================
Phase 58b问题诊断:
  - shared_ratio ≈ 1.0: 方法论假象 (20维PCA捕获99%方差, 所以"共享"=全部)
  - cos = -1.0: 均值方向几乎平行但反向 (两个词在同一维度但不同方向)
  - 唯一有效指标: subspace_overlap (子空间重叠度)

正确方法:
  1. 分别对每个词做PCA, 提取各自的子空间
  2. 计算子空间重叠度 (subspace overlap) 
  3. 计算CCA (Canonical Correlation Analysis) — 更精确的子空间对齐度量
  4. 分析投影到对方子空间的能量比
  5. 逐层分析子空间重叠度的演化

核心指标:
  - overlap(A,B): 子空间重叠度 (0=完全独立, 1=完全相同)
  - proj_energy(A→B): A在B子空间中的投影能量比
  - cca_corr: CCA典型相关系数 (更鲁棒的子空间对齐度量)
"""

import sys, os, json, argparse, numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

PROJECT = Path("d:/Ai2050/TransformerLens-Project")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tests" / "glm5"))

from model_utils import load_model, get_model_info, get_W_U, release_model, safe_decode
import torch

def log_time(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    safe_msg = msg.encode('ascii', errors='replace').decode('ascii')
    print(f"[{ts}] {safe_msg}", flush=True)

# 导入Phase 58b的模板和配置
from subspace_topology_phase4b_backbone_decode import WORD_TEMPLATES, SEMANTIC_PAIRS

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


def collect_word_activations(model, tokenizer, device, word, templates,
                              target_layers, n_layers):
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
    """PCA提取子空间, 返回 (basis, eigenvalues, mean)"""
    X = np.array(vectors)
    mean = X.mean(axis=0)
    X_c = X - mean
    # 使用SVD提高数值稳定性
    U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    # Vt[:n] 是前n个主成分方向
    n = min(n_dims, Vt.shape[0])
    eigenvalues = (S ** 2) / len(X_c)
    return Vt[:n].T, eigenvalues[:n], mean  # [d_model, n_dims]


def subspace_overlap(basis_a, basis_b):
    """子空间重叠度: tr(B^T A A^T B) / k"""
    if basis_a is None or basis_b is None:
        return 0.0
    proj = basis_b.T @ basis_a @ basis_a.T @ basis_b
    k = min(basis_a.shape[1], basis_b.shape[1])
    return float(np.trace(proj) / k)


def projection_energy_ratio(vectors, basis):
    """计算向量集在子空间中的投影能量比"""
    X = np.array(vectors)
    if len(X) == 0 or basis is None:
        return 0.0
    mean = X.mean(axis=0)
    X_c = X - mean
    proj = X_c @ basis @ basis.T
    return float(np.sum(proj ** 2) / (np.sum(X_c ** 2) + 1e-10))


def cca_correlation(acts_a, acts_b, n_dims=10):
    """CCA典型相关分析 — 子空间对齐的更精确度量"""
    X_a = np.array(acts_a)
    X_b = np.array(acts_b)
    
    if len(X_a) < 2 or len(X_b) < 2:
        return [], 0.0
    
    # 使用min样本数
    n_samples = min(len(X_a), len(X_b))
    X_a_c = X_a[:n_samples] - X_a[:n_samples].mean(axis=0)
    X_b_c = X_b[:n_samples] - X_b[:n_samples].mean(axis=0)
    
    d_a = X_a_c.shape[1]
    d_b = X_b_c.shape[1]
    n = min(n_dims, n_samples - 1, d_a, d_b)
    if n < 1:
        return [], 0.0
    
    cov_ab = X_a_c.T @ X_b_c / n_samples
    cov_aa = X_a_c.T @ X_a_c / n_samples
    cov_bb = X_b_c.T @ X_b_c / n_samples
    
    # Whitening + SVD approach
    try:
        # Add regularization
        reg = 1e-6
        L_aa = np.linalg.cholesky(cov_aa + reg * np.eye(cov_aa.shape[0]))
        L_bb = np.linalg.cholesky(cov_bb + reg * np.eye(cov_bb.shape[0]))
        
        L_aa_inv = np.linalg.inv(L_aa)
        L_bb_inv = np.linalg.inv(L_bb)
        
        M = L_aa_inv.T @ cov_ab @ L_bb_inv.T
        U, s, Vt = np.linalg.svd(M)
        
        corrs = s[:n]
        mean_corr = float(np.mean(corrs))
        return corrs.tolist(), mean_corr
    except:
        return [], 0.0


def decode_direction(direction, W_U, tokenizer, top_k=20):
    logits = W_U @ direction
    exp_logits = np.exp(logits - logits.max())
    probs = exp_logits / exp_logits.sum()
    top_indices = np.argsort(probs)[::-1][:top_k]
    return [{"token": safe_decode(tokenizer, idx), "prob": float(probs[idx])}
            for idx in top_indices]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen3", "glm4", "deepseek7b"], required=True)
    parser.add_argument("--n_dims", type=int, default=10, help="子空间维度(用小值避免trivial overlap)")
    args = parser.parse_args()
    
    model_name = args.model
    n_dims = args.n_dims
    
    log_time(f"Loading {model_name}...")
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    W_U = get_W_U(model, model_name)
    d_model = info.d_model
    n_layers = info.n_layers
    target_layers = sorted(set([0, 1] + list(range(0, n_layers, max(1, n_layers // 10)))))
    log_time(f"{model_name}: n_layers={n_layers}, d_model={d_model}, layers={target_layers}")
    
    # ===== Part 1: 收集所有激活 =====
    log_time(f"Collecting activations for {len(SEMANTIC_PAIRS)} pairs...")
    
    all_activations = {}
    
    for pair_key, pair_info in SEMANTIC_PAIRS.items():
        w_a = pair_info["w_a"]
        w_b = pair_info["w_b"]
        
        if w_a not in WORD_TEMPLATES or w_b not in WORD_TEMPLATES:
            continue
        
        log_time(f"  {pair_key}: {w_a}/{w_b} ({pair_info['relation']})")
        
        acts_a, found_a = collect_word_activations(
            model, tokenizer, device, w_a, WORD_TEMPLATES[w_a], target_layers, n_layers)
        acts_b, found_b = collect_word_activations(
            model, tokenizer, device, w_b, WORD_TEMPLATES[w_b], target_layers, n_layers)
        
        log_time(f"    Found: {w_a}={found_a}/15, {w_b}={found_b}/15")
        all_activations[pair_key] = {"a": acts_a, "b": acts_b, "info": pair_info}
    
    # ===== Part 2: 正确的子空间分析 =====
    log_time("Computing subspace analysis with CORRECT metrics...")
    
    pair_results = {}
    
    for pair_key, pair_data in all_activations.items():
        layer_data = {}
        for li in target_layers:
            acts_a = pair_data["a"].get(li, [])
            acts_b = pair_data["b"].get(li, [])
            
            if len(acts_a) < 2 or len(acts_b) < 2:
                layer_data[str(li)] = {"error": "insufficient"}
                continue
            
            # 1. 各自PCA
            basis_a, eigvals_a, mean_a = pca_subspace(acts_a, n_dims)
            basis_b, eigvals_b, mean_b = pca_subspace(acts_b, n_dims)
            
            # 2. 子空间重叠度
            overlap = subspace_overlap(basis_a, basis_b)
            
            # 3. 投影能量比: A在B子空间中的能量
            proj_a_to_b = projection_energy_ratio(acts_a, basis_b)
            proj_b_to_a = projection_energy_ratio(acts_b, basis_a)
            
            # 4. CCA
            cca_corrs, cca_mean = cca_correlation(acts_a, acts_b, n_dims)
            
            # 5. 均值方向cos
            delta_a = np.array(acts_a).mean(axis=0) - np.array(acts_a + acts_b).mean(axis=0)
            delta_b = np.array(acts_b).mean(axis=0) - np.array(acts_a + acts_b).mean(axis=0)
            cos_mean = np.dot(delta_a, delta_b) / (np.linalg.norm(delta_a) * np.linalg.norm(delta_b) + 1e-10)
            
            # 6. 方差解释谱
            total_var_a = eigvals_a.sum()
            total_var_b = eigvals_b.sum()
            top1_ratio_a = float(eigvals_a[0] / total_var_a) if total_var_a > 0 else 0
            top1_ratio_b = float(eigvals_b[0] / total_var_b) if total_var_b > 0 else 0
            
            layer_data[str(li)] = {
                "overlap": float(overlap),
                "proj_a_to_b": float(proj_a_to_b),
                "proj_b_to_a": float(proj_b_to_a),
                "avg_proj": float((proj_a_to_b + proj_b_to_a) / 2),
                "cca_mean": float(cca_mean),
                "cca_top5": cca_corrs[:5] if len(cca_corrs) >= 5 else cca_corrs,
                "cos_mean": float(cos_mean),
                "top1_var_ratio_a": float(top1_ratio_a),
                "top1_var_ratio_b": float(top1_ratio_b),
                "n_a": len(acts_a),
                "n_b": len(acts_b),
            }
        
        pair_results[pair_key] = {"pair_info": pair_data["info"], "layers": layer_data}
    
    # ===== Part 3: 骨干子空间提取 =====
    log_time("Extracting backbone subspace...")
    
    backbone_results = {}
    
    for li in target_layers:
        # 收集所有概念的激活
        all_acts = []
        for pk, pd in all_activations.items():
            all_acts.extend(pd["a"].get(li, []))
            all_acts.extend(pd["b"].get(li, []))
        
        if len(all_acts) < 20:
            continue
        
        X = np.array(all_acts)
        mean_all = X.mean(axis=0)
        X_c = X - mean_all
        
        # SVD
        U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
        eigvals = (S ** 2) / len(X_c)
        total_var = eigvals.sum()
        
        # 骨干 = 前20主成分
        n_backbone = min(20, len(eigvals))
        backbone_basis = Vt[:n_backbone].T  # [d_model, 20]
        backbone_var_ratio = float(eigvals[:n_backbone].sum() / total_var)
        
        # 解码骨干方向
        backbone_decoded = []
        for d in range(min(5, n_backbone)):
            tw = decode_direction(backbone_basis[:, d], W_U, tokenizer, top_k=15)
            backbone_decoded.append({
                "direction": d,
                "var_explained": float(eigvals[d] / total_var),
                "top_words": tw[:8],
            })
        
        # 计算每个概念对在骨干中的共享能量
        pair_backbone_energy = {}
        for pk, pd in all_activations.items():
            for wk in ["a", "b"]:
                acts = pd[wk].get(li, [])
                if len(acts) >= 2:
                    energy = projection_energy_ratio(acts, backbone_basis)
                    if pk not in pair_backbone_energy:
                        pair_backbone_energy[pk] = {}
                    pair_backbone_energy[pk][wk] = float(energy)
        
        # 神经元归属
        proj = X_c @ backbone_basis
        recon = proj @ backbone_basis.T
        res = X_c - recon
        total_var_neuron = np.var(X_c, axis=0)
        shared_var_neuron = np.var(recon, axis=0)
        backbone_score = shared_var_neuron / (total_var_neuron + 1e-10)
        
        backbone_results[str(li)] = {
            "backbone_var_ratio": backbone_var_ratio,
            "n_samples": len(all_acts),
            "backbone_decoded": backbone_decoded,
            "mean_backbone_score": float(backbone_score.mean()),
            "top_backbone_neurons": np.argsort(backbone_score)[::-1][:10].tolist(),
            "pair_backbone_energy": pair_backbone_energy,
        }
        
        log_time(f"  L{li}: backbone_var={backbone_var_ratio:.3f} mean_score={backbone_score.mean():.3f}")
    
    # ===== Part 4: 语义距离函数 =====
    mid_layer = target_layers[len(target_layers) // 2]
    
    sim_data = []
    for pk, pr in pair_results.items():
        ld = pr["layers"].get(str(mid_layer), {})
        if "error" not in ld:
            sim_data.append({
                "pair_key": pk,
                "relation": pr["pair_info"]["relation"],
                "distance": pr["pair_info"]["distance"],
                "overlap": ld.get("overlap", 0),
                "avg_proj": ld.get("avg_proj", 0),
                "cca_mean": ld.get("cca_mean", 0),
                "cos_mean": ld.get("cos_mean", 0),
            })
    
    relation_stats = defaultdict(lambda: {"overlap": [], "proj": [], "cca": [], "cos": []})
    for sd in sim_data:
        rel = sd["relation"]
        relation_stats[rel]["overlap"].append(sd["overlap"])
        relation_stats[rel]["proj"].append(sd["avg_proj"])
        relation_stats[rel]["cca"].append(sd["cca_mean"])
        relation_stats[rel]["cos"].append(sd["cos_mean"])
    
    relation_summary = {}
    for rel, stats in relation_stats.items():
        relation_summary[rel] = {
            "mean_overlap": float(np.mean(stats["overlap"])),
            "std_overlap": float(np.std(stats["overlap"])),
            "mean_proj": float(np.mean(stats["proj"])),
            "mean_cca": float(np.mean(stats["cca"])),
            "mean_cos": float(np.mean(stats["cos"])),
            "n": len(stats["overlap"]),
        }
    
    # ===== 保存 =====
    output = {
        "model": model_name, "n_dims": n_dims, "d_model": d_model,
        "n_layers": n_layers, "target_layers": target_layers,
        "pair_results": pair_results,
        "backbone_results": backbone_results,
        "similarity_function": {
            "mid_layer": mid_layer,
            "per_pair": sim_data,
            "relation_summary": relation_summary,
        },
        "timestamp": datetime.now().isoformat(),
    }
    
    out_dir = PROJECT / "results" / "subspace_topology"
    out_file = out_dir / f"exp4c_correct_subspace_{model_name}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    log_time(f"Saved to {out_file}")
    
    # ===== 摘要 =====
    log_time("")
    log_time("=" * 70)
    log_time(f"PHASE 58c SUMMARY - {model_name} (CORRECT metrics)")
    log_time("=" * 70)
    
    log_time(f"\n--- Subspace Overlap by Relation (L{mid_layer}, n_dims={n_dims}) ---")
    for rel in ["hyponym", "synonym", "antonym", "associated", "unrelated"]:
        if rel in relation_summary:
            rs = relation_summary[rel]
            log_time("  {:12s}: overlap={:.3f}+-{:.3f} proj={:.3f} cca={:.3f} cos={:.3f} n={}".format(
                rel, rs['mean_overlap'], rs['std_overlap'],
                rs['mean_proj'], rs['mean_cca'], rs['mean_cos'], rs['n']))
    
    log_time("\n--- Per-Pair Overlap (sorted) ---")
    for sd in sorted(sim_data, key=lambda x: -x['overlap']):
        log_time("  {:20s} {} overlap={:.3f} proj={:.3f} cca={:.3f} cos={:.3f}".format(
            sd['pair_key'], sd['relation'],
            sd['overlap'], sd['avg_proj'], sd['cca_mean'], sd['cos_mean']))
    
    log_time("\n--- Layer Evolution: overlap by relation ---")
    layers_data = {}
    for pk, pr in pair_results.items():
        rel = pr["pair_info"]["relation"]
        for lk, ld in pr["layers"].items():
            if 'error' not in ld:
                if lk not in layers_data:
                    layers_data[lk] = {}
                if rel not in layers_data[lk]:
                    layers_data[lk][rel] = []
                layers_data[lk][rel].append(ld.get('overlap', 0))
    
    for lk in sorted(layers_data.keys(), key=int):
        parts = []
        for rel in ["hyponym", "synonym", "antonym", "associated", "unrelated"]:
            if rel in layers_data[lk]:
                m = sum(layers_data[lk][rel]) / len(layers_data[lk][rel])
                parts.append("{}={:.3f}".format(rel[:4], m))
        log_time("  L{}: {}".format(lk, " | ".join(parts)))
    
    log_time("\n--- Backbone Decode (key layers) ---")
    for lk in ['9', '15', '27', '33']:
        if lk in backbone_results:
            bd = backbone_results[lk]
            log_time("  L{} backbone_var={:.3f}:".format(lk, bd['backbone_var_ratio']))
            for d_info in bd["backbone_decoded"][:3]:
                tw = [t['token'].strip()[:10] for t in d_info['top_words'][:5]]
                log_time("    PC{} var={:.4f}: {}".format(d_info['direction'], d_info['var_explained'], tw))
    
    release_model(model)
    log_time("Done!")


if __name__ == "__main__":
    main()
