"""
Phase 150b: 条件转移矩阵的几何基线校正
===========================================

Phase 150 Exp1 发现 P(r→n) > P(n→r), t-test显著。
但分析揭示: 这可能是高维几何的必然结果, 而非主动路由!

关键对比:
  - 实际Asymmetry ≈ 0.59
  - 几何预测(纯随机旋转) Asymmetry ≈ 0.64
  - 实际 < 几何预测 → 系统比随机旋转更好地保持row-space!

本实验加入两个关键对照:
  1. Shuffle Control: 将每层Jacobian随机旋转, 测量"几何基线"
  2. Random Weight Control: 用随机初始化模型, 测量"未训练基线"

真正的判据不是Asymmetry > 0, 而是:
  - 实际Asymmetry > 几何基线 → 主动推到null-space
  - 实际Asymmetry < 几何基线 → 主动保持row-space
  - 实际Asymmetry ≈ 几何基线 → 纯被动mixing

用法:
  python tests/glm5_temp/phase150b_geometric_baseline.py qwen3
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
from model_utils import (load_model, get_layers, get_model_info,
                          release_model, get_W_U, MODEL_CONFIGS)

EPSILON = 1.0
N_SENTENCES = 20
OUTPUT_DIR = Path("tests/glm5_temp")

TEST_PROMPTS = [
    "The scientist discovered that the",
    "In the morning, she decided to",
    "The book on the table was about",
    "After the rain stopped, the children",
    "The most important thing about science is",
    "When the sun sets over the ocean,",
    "She walked into the room and saw",
    "The professor explained that the theory",
    "Despite the challenges, the team managed",
    "The ancient city was known for its",
    "He realized that the answer was",
    "The relationship between language and thought",
    "Every morning she would read the",
    "The experiment showed that the results",
    "Music has the power to change how",
    "The government announced that the new policy",
    "In the future, artificial intelligence will",
    "The philosopher argued that consciousness is",
    "After years of research, they found that",
    "The key difference between the two approaches is",
]


def get_device_for_input(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda")


def get_row_null_bases(W_U, n_components=None):
    d = W_U.shape[1]
    from scipy.sparse.linalg import svds
    W_U_T = W_U.T.astype(np.float32)
    k_max = min(500, min(W_U_T.shape) - 2)
    k_max = max(k_max, 10)
    U_full, s_full, _ = svds(W_U_T, k=k_max)
    idx = np.argsort(-s_full)
    U_full = U_full[:, idx]
    s_full = s_full[idx]
    total_energy = np.sum(s_full ** 2)
    cumulative_energy = np.cumsum(s_full ** 2)
    k_95 = np.searchsorted(cumulative_energy, 0.95 * total_energy) + 1
    k = k_95 if n_components is None else min(n_components, k_max)
    return U_full[:, :k].T, k, s_full[:k_max], k_95


def project_to_null(vec, row_basis):
    row_component = row_basis.T @ (row_basis @ vec)
    return vec - row_component


def project_to_row(vec, row_basis):
    return row_basis.T @ (row_basis @ vec)


def compute_row_energy(delta, row_basis):
    delta_norm_sq = np.sum(delta ** 2)
    if delta_norm_sq < 1e-16:
        return 0.0, 1.0
    row_coeffs = row_basis @ delta
    row_energy = np.sum(row_coeffs ** 2) / delta_norm_sq
    null_ratio = 1.0 - row_energy
    return float(row_energy), float(null_ratio)


def run_forward_with_perturbation(model, input_ids, attention_mask,
                                   inject_layer, position, delta_np, device):
    layers = get_layers(model)
    def make_inject_hook(pos, delta_tensor):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0].clone()
                out[0, pos, :] += delta_tensor.to(out.dtype).to(out.device)
                return (out,) + output[1:]
            else:
                out = output.clone()
                out[0, pos, :] += delta_tensor.to(out.dtype).to(out.device)
                return out
        return hook
    hooks = [layers[inject_layer].register_forward_hook(
        make_inject_hook(position, torch.tensor(delta_np, dtype=torch.float32)))]
    try:
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True)
    finally:
        for h in hooks:
            h.remove()
    return out


def compute_transfer_matrix(model, tokenizer, model_name, W_U, row_basis,
                             inject_layer=0, label=""):
    """计算条件转移矩阵"""
    info = get_model_info(model, model_name)
    device = get_device_for_input(model)
    d_model = info.d_model
    n_layers = info.n_layers
    
    sample_layers = list(range(0, n_layers + 1, max(1, n_layers // 12)))
    sample_layers = sorted(set(sample_layers + [n_layers]))
    
    results = {"row": {}, "null": {}}
    
    for sent_idx in range(N_SENTENCES):
        prompt = TEST_PROMPTS[sent_idx]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        seq_len = input_ids.shape[1]
        last_pos = seq_len - 1
        
        with torch.no_grad():
            out_clean = model(input_ids=input_ids, attention_mask=attention_mask,
                              output_hidden_states=True)
        clean_hs = out_clean.hidden_states
        
        for dir_type in ["row", "null"]:
            np.random.seed(sent_idx * 100 + {"row": 0, "null": 50}[dir_type])
            raw_vec = np.random.randn(d_model)
            
            if dir_type == "row":
                delta = project_to_row(raw_vec, row_basis)
            else:
                delta = project_to_null(raw_vec, row_basis)
            
            norm = np.linalg.norm(delta)
            if norm < 1e-8:
                continue
            delta = delta / norm * EPSILON
            
            out_perturbed = run_forward_with_perturbation(
                model, input_ids, attention_mask, inject_layer, last_pos, delta, device)
            
            for li in sample_layers:
                perturbed_vec = out_perturbed.hidden_states[li][0, last_pos, :].float().cpu().numpy()
                clean_vec = clean_hs[li][0, last_pos, :].float().cpu().numpy()
                delta_prop = perturbed_vec - clean_vec
                
                row_e, null_r = compute_row_energy(delta_prop, row_basis)
                
                key = f"L{li}"
                if key not in results[dir_type]:
                    results[dir_type][key] = {"row_energy": [], "null_ratio": []}
                results[dir_type][key]["row_energy"].append(row_e)
                results[dir_type][key]["null_ratio"].append(null_r)
    
    # 计算转移矩阵
    transfer = {}
    for li in sample_layers:
        key = f"L{li}"
        rr = np.mean(results["row"].get(key, {}).get("row_energy", [0]))
        rn = 1.0 - rr
        nr = np.mean(results["null"].get(key, {}).get("row_energy", [0]))
        nn = 1.0 - nr
        asym = rn - nr
        transfer[li] = {
            "P_rr": float(rr), "P_rn": float(rn),
            "P_nr": float(nr), "P_nn": float(nn),
            "asymmetry": float(asym),
        }
    
    return transfer


def compute_geometric_baseline(d_model, rank, n_trials=1000):
    """
    计算纯随机旋转下的几何基线
    
    在纯随机旋转下:
    - 生成随机正交矩阵 Q ∈ O(d)
    - 投影到row-space和null-space
    - 测量P(r→n)和P(n→r)
    """
    # 随机旋转矩阵
    null_dim = d_model - rank
    
    row_to_null_ratios = []
    null_to_row_ratios = []
    
    for trial in range(n_trials):
        # 生成随机row-space向量
        v_row = np.random.randn(rank)
        v_row = v_row / np.linalg.norm(v_row)
        
        # 嵌入到d_model维
        v_full_row = np.zeros(d_model)
        v_full_row[:rank] = v_row
        
        # 随机旋转
        Q, _ = np.linalg.qr(np.random.randn(d_model, d_model))
        v_rotated = Q @ v_full_row
        
        # 测量row-space能量
        row_e = np.sum(v_rotated[:rank] ** 2) / np.sum(v_rotated ** 2)
        null_e = 1 - row_e
        row_to_null_ratios.append(null_e)
        
        # 生成随机null-space向量
        v_null = np.random.randn(null_dim)
        v_null = v_null / np.linalg.norm(v_null)
        v_full_null = np.zeros(d_model)
        v_full_null[rank:] = v_null
        
        v_rotated_null = Q @ v_full_null
        row_e_null = np.sum(v_rotated_null[:rank] ** 2) / np.sum(v_rotated_null ** 2)
        null_to_row_ratios.append(row_e_null)
    
    P_rn = np.mean(row_to_null_ratios)
    P_nr = np.mean(null_to_row_ratios)
    asym = P_rn - P_nr
    
    return {
        "P_rn_geo": float(P_rn),
        "P_nr_geo": float(P_nr),
        "asymmetry_geo": float(asym),
        "P_rn_std": float(np.std(row_to_null_ratios)),
        "P_nr_std": float(np.std(null_to_row_ratios)),
    }


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    print(f"Phase 150b: Geometric Baseline Correction")
    print(f"Model: {model_name}, Time: {timestamp}")
    
    # 加载模型
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    use_8bit = model_name in ("deepseek7b", "glm4") and gpu_mem_gb < 16
    
    t0 = time.time()
    model, tokenizer, device = load_model(model_name, use_8bit=use_8bit)
    info = get_model_info(model, model_name)
    print(f"Model: {info.model_class}, {info.n_layers}L, d={info.d_model}")
    
    # W_U和row/null space
    print("Getting W_U...", flush=True)
    W_U = get_W_U(model, model_name)
    print("Computing SVD...", flush=True)
    row_basis, k, sv_data, k_95 = get_row_null_bases(W_U, n_components=None)
    print(f"SVD done, rank={k_95}", flush=True)
    d_model = info.d_model
    rank = k_95
    null_dim = d_model - rank
    
    print(f"W_U: shape={W_U.shape}, effective_rank(95%)={rank}, null_dim={null_dim}")
    
    # ==========================================
    # 实验1: 训练模型的条件转移矩阵
    # ==========================================
    print("\n" + "="*60)
    print("Experiment 1: Trained Model Transfer Matrix")
    print("="*60)
    
    trained_transfer = compute_transfer_matrix(model, tokenizer, model_name, W_U, row_basis,
                                                inject_layer=0, label="trained")
    
    print(f"\n  {'Layer':>6} {'P(r→r)':>8} {'P(r→n)':>8} {'P(n→r)':>8} {'P(n→n)':>8} {'Asym':>10} {'vs Geo':>10}")
    
    for li in sorted(trained_transfer.keys()):
        tm = trained_transfer[li]
        # 几何基线
        geo_pred = null_dim / d_model - rank / d_model  # P(r→n) - P(n→r) under random rotation
        vs_geo = tm['asymmetry'] - geo_pred
        print(f"  L{li:>4d} {tm['P_rr']:>8.4f} {tm['P_rn']:>8.4f} {tm['P_nr']:>8.4f} "
              f"{tm['P_nn']:>8.4f} {tm['asymmetry']:>+10.4f} {vs_geo:>+10.4f}")
    
    # ==========================================
    # 实验2: 几何基线 (纯随机旋转)
    # ==========================================
    print("\n" + "="*60)
    print("Experiment 2: Geometric Baseline (Random Rotation)")
    print("="*60)
    
    geo_baseline = compute_geometric_baseline(d_model, rank, n_trials=5000)
    
    print(f"  P(r→n) geometric: {geo_baseline['P_rn_geo']:.4f} ± {geo_baseline['P_rn_std']:.4f}")
    print(f"  P(n→r) geometric: {geo_baseline['P_nr_geo']:.4f} ± {geo_baseline['P_nr_std']:.4f}")
    print(f"  Asymmetry geometric: {geo_baseline['asymmetry_geo']:+.4f}")
    
    # ==========================================
    # 实验3: 扰动shuffle对照
    # ==========================================
    print("\n" + "="*60)
    print("Experiment 3: Perturbation Shuffle Control")
    print("核心: 将传播后的扰动随机旋转, 然后测量null_ratio")
    print("如果随机旋转后的null_ratio ≈ 实际null_ratio → 几何必然性")
    print("如果随机旋转后的null_ratio > 实际null_ratio → 系统主动保持结构")
    print("="*60)
    
    # 在几个关键层收集传播后的扰动, 然后shuffle
    sample_layers_for_shuffle = [3, 9, 18, 27, 35]
    
    actual_null_ratios = {dir_t: {li: [] for li in sample_layers_for_shuffle} 
                          for dir_t in ["row", "null"]}
    shuffled_null_ratios = {dir_t: {li: [] for li in sample_layers_for_shuffle} 
                            for dir_t in ["row", "null"]}
    
    for sent_idx in range(N_SENTENCES):
        prompt = TEST_PROMPTS[sent_idx]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        last_pos = input_ids.shape[1] - 1
        
        with torch.no_grad():
            out_clean = model(input_ids=input_ids, attention_mask=attention_mask,
                              output_hidden_states=True)
        clean_hs = out_clean.hidden_states
        
        for dir_type in ["row", "null"]:
            np.random.seed(sent_idx * 100 + {"row": 0, "null": 50}[dir_type])
            raw_vec = np.random.randn(d_model)
            
            if dir_type == "row":
                delta = project_to_row(raw_vec, row_basis)
            else:
                delta = project_to_null(raw_vec, row_basis)
            
            norm = np.linalg.norm(delta)
            if norm < 1e-8:
                continue
            delta = delta / norm * EPSILON
            
            out_perturbed = run_forward_with_perturbation(
                model, input_ids, attention_mask, 0, last_pos, delta, device)
            
            for li in sample_layers_for_shuffle:
                if li >= len(out_perturbed.hidden_states):
                    continue
                
                perturbed_vec = out_perturbed.hidden_states[li][0, last_pos, :].float().cpu().numpy()
                clean_vec = clean_hs[li][0, last_pos, :].float().cpu().numpy()
                delta_prop = perturbed_vec - clean_vec
                delta_norm = np.linalg.norm(delta_prop)
                
                if delta_norm < 1e-10:
                    continue
                
                # 实际null_ratio
                _, actual_nr = compute_row_energy(delta_prop, row_basis)
                actual_null_ratios[dir_type][li].append(actual_nr)
                
                # Shuffle对照: 将扰动随机旋转, 然后测null_ratio
                Q, _ = np.linalg.qr(np.random.randn(d_model, d_model))
                delta_shuffled = Q @ delta_prop
                _, shuffled_nr = compute_row_energy(delta_shuffled, row_basis)
                shuffled_null_ratios[dir_type][li].append(shuffled_nr)
    
    # 汇总
    print(f"\n  {'Dir':>5} {'Layer':>6} {'Actual_NR':>10} {'Shuffled_NR':>12} {'Difference':>12}")
    print(f"  {'---':>5} {'-----':>6} {'---------':>10} {'-----------':>12} {'----------':>12}")
    
    for dir_type in ["row", "null"]:
        for li in sample_layers_for_shuffle:
            a_nr = np.mean(actual_null_ratios[dir_type][li]) if actual_null_ratios[dir_type][li] else 0
            s_nr = np.mean(shuffled_null_ratios[dir_type][li]) if shuffled_null_ratios[dir_type][li] else 0
            diff = a_nr - s_nr
            print(f"  {dir_type:>5} L{li:>4d} {a_nr:>10.4f} {s_nr:>12.4f} {diff:>+12.4f}")
    
    # ==========================================
    # 综合判断
    # ==========================================
    print("\n" + "="*60)
    print("CRITICAL JUDGMENT: Active Routing vs Geometric Inevitability")
    print("="*60)
    
    # 中间层的平均值
    mid_layers = [li for li in range(9, 28)]
    trained_asym_mid = np.mean([trained_transfer[li]['asymmetry'] 
                                for li in mid_layers if li in trained_transfer])
    geo_asym = geo_baseline['asymmetry_geo']
    
    print(f"\n  Trained model Asymmetry (mid-layers): {trained_asym_mid:+.4f}")
    print(f"  Geometric baseline Asymmetry:         {geo_asym:+.4f}")
    print(f"  Difference:                           {trained_asym_mid - geo_asym:+.4f}")
    
    # Shuffle对照的null_ratio差异
    row_actual_mid = np.mean([np.mean(actual_null_ratios["row"][li]) 
                              for li in [9, 18, 27] if actual_null_ratios["row"][li]])
    row_shuffled_mid = np.mean([np.mean(shuffled_null_ratios["row"][li]) 
                                for li in [9, 18, 27] if shuffled_null_ratios["row"][li]])
    
    print(f"\n  Row-input: Actual NR (mid):  {row_actual_mid:.4f}")
    print(f"  Row-input: Shuffled NR (mid):{row_shuffled_mid:.4f}")
    print(f"  Row-input: Difference:       {row_actual_mid - row_shuffled_mid:+.4f}")
    
    null_actual_mid = np.mean([np.mean(actual_null_ratios["null"][li]) 
                               for li in [9, 18, 27] if actual_null_ratios["null"][li]])
    null_shuffled_mid = np.mean([np.mean(shuffled_null_ratios["null"][li]) 
                                 for li in [9, 18, 27] if shuffled_null_ratios["null"][li]])
    
    print(f"\n  Null-input: Actual NR (mid):  {null_actual_mid:.4f}")
    print(f"  Null-input: Shuffled NR (mid):{null_shuffled_mid:.4f}")
    print(f"  Null-input: Difference:       {null_actual_mid - null_shuffled_mid:+.4f}")
    
    # 最终判断
    print(f"\n  === FINAL JUDGMENT ===")
    if trained_asym_mid < geo_asym:
        print(f"  ✅ 实际Asymmetry < 几何基线")
        print(f"  → 系统比随机旋转更好地保持row-space!")
        print(f"  → 这是'主动保护row-space'的证据, 而非'主动推到null-space'!")
        print(f"  → Phase 148/150的P(r→n)>P(n→r)是高维几何的必然结果")
    elif trained_asym_mid > geo_asym:
        print(f"  ❌ 实际Asymmetry > 几何基线")
        print(f"  → 系统比随机旋转更倾向于将row-space推到null-space")
        print(f"  → 可能存在'主动null-space路由'")
    else:
        print(f"  ↔ 实际Asymmetry ≈ 几何基线")
        print(f"  → 纯被动mixing, 无主动路由")
    
    if row_actual_mid < row_shuffled_mid:
        print(f"\n  ✅ Row-input: Actual NR < Shuffled NR")
        print(f"  → 传播后的扰动比随机旋转更集中在row-space!")
        print(f"  → 这强烈支持: 系统主动保持row-space结构")
    elif row_actual_mid > row_shuffled_mid:
        print(f"\n  ❌ Row-input: Actual NR > Shuffled NR")
        print(f"  → 传播后的扰动比随机旋转更分散到null-space")
    
    # 保存结果
    all_results = {
        "phase": "150b",
        "model": model_name,
        "timestamp": timestamp,
        "d_model": d_model,
        "rank_95": rank,
        "null_dim": null_dim,
        "trained_transfer": trained_transfer,
        "geometric_baseline": geo_baseline,
        "shuffle_control": {
            "row_actual_null_ratio": {str(li): np.mean(v).tolist() if v else 0 
                                       for li, v in actual_null_ratios["row"].items()},
            "row_shuffled_null_ratio": {str(li): np.mean(v).tolist() if v else 0 
                                         for li, v in shuffled_null_ratios["row"].items()},
            "null_actual_null_ratio": {str(li): np.mean(v).tolist() if v else 0 
                                        for li, v in actual_null_ratios["null"].items()},
            "null_shuffled_null_ratio": {str(li): np.mean(v).tolist() if v else 0 
                                          for li, v in shuffled_null_ratios["null"].items()},
        },
        "trained_asymmetry_mid": float(trained_asym_mid),
        "geometric_asymmetry": float(geo_asym),
    }
    
    result_file = OUTPUT_DIR / f"phase150b_{model_name}_{timestamp}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else o,
                  ensure_ascii=False)
    
    print(f"\nResults saved to: {result_file}")
    
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
