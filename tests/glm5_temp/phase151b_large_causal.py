"""
Phase 151b: 加大Exp2数据量 — 因果灵敏度是核心实验
====================================================

Exp2揭示: dist=0时top1_change_rate=6.7%
但: 只有15个样本, 远远不够!
需要: 大样本(每个距离>=100个样本)来可靠估计因果灵敏度

另外: 需要更系统地测量:
1. 不同扰动强度下的因果灵敏度曲线
2. 不同语义类型token的因果灵敏度差异
3. 注意力头级别的因果灵敏度(哪些头对因果传播贡献最大)
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
N_SENTENCES = 30  # 加大
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
    "The cat sat on the windowsill and",
    "She opened the door and found that",
    "The river flowed through the valley and",
    "He picked up the phone and called",
    "The children played in the park until",
    "The teacher told the students that they",
    "After dinner, the family went to the",
    "The car stopped at the red light and",
    "She looked at the painting and felt",
    "The old man walked slowly through the",
]


def get_device_for_input(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda")


def run_forward_with_perturbation_at_position(model, input_ids, attention_mask,
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
                        output_hidden_states=True, output_attentions=True)
    except Exception as e:
        out = None
    finally:
        for h in hooks:
            h.remove()
    return out


def get_row_null_bases(W_U, n_components=None):
    from scipy.sparse.linalg import svds
    W_U_T = W_U.T.astype(np.float32)
    k_max = min(500, min(W_U_T.shape) - 2)
    k_max = max(k_max, 10)
    
    print(f"  Computing SVD of W_U^T (shape={W_U_T.shape}), k_max={k_max}...")
    t0 = time.time()
    U_full, s_full, Vt_full = svds(W_U_T, k=k_max)
    idx = np.argsort(-s_full)
    U_full = U_full[:, idx]
    s_full = s_full[idx]
    print(f"  SVD done in {time.time()-t0:.1f}s")
    
    total_energy = np.sum(s_full ** 2)
    cumulative_energy = np.cumsum(s_full ** 2)
    k_90 = np.searchsorted(cumulative_energy, 0.90 * total_energy) + 1
    k_95 = np.searchsorted(cumulative_energy, 0.95 * total_energy) + 1
    
    if n_components is None:
        k = k_95
    else:
        k = min(n_components, k_max)
    
    row_basis = U_full[:, :k].T
    return row_basis, k, s_full[:k_max], k_90, k_95


def project_to_row(vec, row_basis):
    return row_basis.T @ (row_basis @ vec)


def project_to_null(vec, row_basis):
    return vec - row_basis.T @ (row_basis @ vec)


def compute_row_energy(delta, row_basis):
    delta_norm_sq = np.sum(delta ** 2)
    if delta_norm_sq < 1e-16:
        return 0.0, 1.0
    row_coeffs = row_basis @ delta
    row_energy = np.sum(row_coeffs ** 2) / delta_norm_sq
    return float(row_energy), float(1.0 - row_energy)


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    print(f"Phase 151b: Large-Sample Causal Sensitivity")
    print(f"Model: {model_name}, Time: {timestamp}")
    
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    use_8bit = model_name in ("deepseek7b", "glm4") and gpu_mem_gb < 16
    
    t0 = time.time()
    model, tokenizer, device = load_model(model_name, use_8bit=use_8bit)
    info = get_model_info(model, model_name)
    print(f"Model: {info.model_class}, {info.n_layers}L, d={info.d_model}")
    print(f"Load time: {time.time()-t0:.1f}s")
    
    n_layers = info.n_layers
    d_model = info.d_model
    
    # 获取W_U (用于row/null-space方向)
    W_U = get_W_U(model, model_name)
    row_basis, k, sv_data, k_90, k_95 = get_row_null_bases(W_U, n_components=None)
    print(f"W_U: rank(95%)={k}")
    
    # === 大样本因果灵敏度实验 ===
    print("\n" + "="*60)
    print("Large-Sample Causal Token Sensitivity")
    print("="*60)
    
    # 三个子实验:
    # A: 不同距离的因果灵敏度 (大量样本)
    # B: 不同扰动方向的因果灵敏度 (row-space vs null-space vs semantic)
    # C: 扰动强度依赖性
    
    all_results = []
    
    # --- Part A: 距离依赖性 (大样本) ---
    print("\n--- Part A: Distance Dependence (large sample) ---")
    
    distance_results = {}  # distance -> list of {top1_changed, kl, prob_change, ...}
    
    for sent_idx in range(N_SENTENCES):
        prompt = TEST_PROMPTS[sent_idx]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        seq_len = input_ids.shape[1]
        last_pos = seq_len - 1
        
        # Clean forward
        with torch.no_grad():
            out_clean = model(input_ids=input_ids, attention_mask=attention_mask,
                              output_hidden_states=True)
        clean_logits = out_clean.logits[0, -1, :].float().cpu().numpy()
        clean_top1 = int(np.argmax(clean_logits))
        clean_probs = np.exp(clean_logits - np.max(clean_logits))
        clean_probs = clean_probs / np.sum(clean_probs)
        
        # 在每个position注入随机扰动(在L0), 测量last token的响应
        # 每个position做3次随机扰动
        for inject_pos in range(seq_len):
            for trial in range(3):
                np.random.seed(sent_idx * 10000 + inject_pos * 100 + trial)
                delta = np.random.randn(d_model)
                norm = np.linalg.norm(delta)
                if norm > 1e-8:
                    delta = delta / norm * EPSILON
                
                out_perturbed = run_forward_with_perturbation_at_position(
                    model, input_ids, attention_mask, 0, inject_pos, delta, device)
                
                if out_perturbed is None:
                    continue
                
                perturbed_logits = out_perturbed.logits[0, -1, :].float().cpu().numpy()
                perturbed_top1 = int(np.argmax(perturbed_logits))
                perturbed_probs = np.exp(perturbed_logits - np.max(perturbed_logits))
                perturbed_probs = perturbed_probs / np.sum(perturbed_probs)
                
                # 因果灵敏度指标
                top1_changed = int(perturbed_top1 != clean_top1)
                kl_div = np.sum(clean_probs * np.log((clean_probs + 1e-10) / (perturbed_probs + 1e-10)))
                prob_change = np.max(np.abs(perturbed_probs - clean_probs))
                top1_prob_drop = clean_probs[clean_top1] - perturbed_probs[clean_top1]
                
                distance = last_pos - inject_pos
                
                if distance not in distance_results:
                    distance_results[distance] = []
                
                distance_results[distance].append({
                    'top1_changed': top1_changed,
                    'kl_div': float(kl_div),
                    'prob_change': float(prob_change),
                    'top1_prob_drop': float(top1_prob_drop),
                    'distance': distance,
                })
                
                all_results.append({
                    'type': 'distance',
                    'sent_idx': sent_idx,
                    'inject_pos': inject_pos,
                    'distance': distance,
                    'top1_changed': top1_changed,
                    'kl_div': float(kl_div),
                    'prob_change': float(prob_change),
                })
    
    # --- Part B: 方向依赖性 (row-space vs null-space) ---
    print("\n--- Part B: Direction Dependence ---")
    
    direction_results = {'row': [], 'null': [], 'random': []}
    
    for sent_idx in range(min(N_SENTENCES, 20)):
        prompt = TEST_PROMPTS[sent_idx]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        seq_len = input_ids.shape[1]
        last_pos = seq_len - 1
        
        # Clean forward
        with torch.no_grad():
            out_clean = model(input_ids=input_ids, attention_mask=attention_mask,
                              output_hidden_states=True)
        clean_logits = out_clean.logits[0, -1, :].float().cpu().numpy()
        clean_top1 = int(np.argmax(clean_logits))
        clean_probs = np.exp(clean_logits - np.max(clean_logits))
        clean_probs = clean_probs / np.sum(clean_probs)
        
        for dir_type in ['row', 'null', 'random']:
            for trial in range(3):
                np.random.seed(sent_idx * 1000 + trial + {'row': 0, 'null': 50, 'random': 100}[dir_type])
                raw = np.random.randn(d_model)
                
                if dir_type == 'row':
                    delta = project_to_row(raw, row_basis)
                elif dir_type == 'null':
                    delta = project_to_null(raw, row_basis)
                else:
                    delta = raw
                
                norm = np.linalg.norm(delta)
                if norm < 1e-8:
                    continue
                delta = delta / norm * EPSILON
                
                out_perturbed = run_forward_with_perturbation_at_position(
                    model, input_ids, attention_mask, 0, last_pos, delta, device)
                
                if out_perturbed is None:
                    continue
                
                perturbed_logits = out_perturbed.logits[0, -1, :].float().cpu().numpy()
                perturbed_top1 = int(np.argmax(perturbed_logits))
                perturbed_probs = np.exp(perturbed_logits - np.max(perturbed_logits))
                perturbed_probs = perturbed_probs / np.sum(perturbed_probs)
                
                top1_changed = int(perturbed_top1 != clean_top1)
                kl_div = np.sum(clean_probs * np.log((clean_probs + 1e-10) / (perturbed_probs + 1e-10)))
                prob_change = np.max(np.abs(perturbed_probs - clean_probs))
                
                direction_results[dir_type].append({
                    'top1_changed': top1_changed,
                    'kl_div': float(kl_div),
                    'prob_change': float(prob_change),
                })
    
    # --- Part C: 扰动强度依赖性 ---
    print("\n--- Part C: Perturbation Scale Dependence ---")
    
    scale_results = {}
    epsilons = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    for eps in epsilons:
        scale_results[eps] = []
        
        for sent_idx in range(min(N_SENTENCES, 15)):
            prompt = TEST_PROMPTS[sent_idx]
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            seq_len = input_ids.shape[1]
            last_pos = seq_len - 1
            
            # Clean forward
            with torch.no_grad():
                out_clean = model(input_ids=input_ids, attention_mask=attention_mask,
                                  output_hidden_states=True)
            clean_logits = out_clean.logits[0, -1, :].float().cpu().numpy()
            clean_top1 = int(np.argmax(clean_logits))
            clean_probs = np.exp(clean_logits - np.max(clean_logits))
            clean_probs = clean_probs / np.sum(clean_probs)
            
            for trial in range(3):
                np.random.seed(sent_idx * 1000 + int(eps * 100) + trial)
                delta = np.random.randn(d_model)
                norm = np.linalg.norm(delta)
                if norm > 1e-8:
                    delta = delta / norm * eps
                
                out_perturbed = run_forward_with_perturbation_at_position(
                    model, input_ids, attention_mask, 0, last_pos, delta, device)
                
                if out_perturbed is None:
                    continue
                
                perturbed_logits = out_perturbed.logits[0, -1, :].float().cpu().numpy()
                perturbed_top1 = int(np.argmax(perturbed_logits))
                perturbed_probs = np.exp(perturbed_logits - np.max(perturbed_logits))
                perturbed_probs = perturbed_probs / np.sum(perturbed_probs)
                
                top1_changed = int(perturbed_top1 != clean_top1)
                kl_div = np.sum(clean_probs * np.log((clean_probs + 1e-10) / (perturbed_probs + 1e-10)))
                prob_change = np.max(np.abs(perturbed_probs - clean_probs))
                
                scale_results[eps].append({
                    'top1_changed': top1_changed,
                    'kl_div': float(kl_div),
                    'prob_change': float(prob_change),
                })
    
    # === 汇总 ===
    print("\n" + "="*60)
    print("LARGE-SAMPLE SUMMARY")
    print("="*60)
    
    # Part A: 距离依赖性
    print("\n=== Part A: Distance Dependence ===")
    print(f"{'Dist':>5} {'N':>5} {'top1%':>8} {'KL':>10} {'prob_chg':>10} {'Interpretation':>20}")
    
    for dist in sorted(distance_results.keys()):
        data = distance_results[dist]
        n = len(data)
        top1_rate = np.mean([d['top1_changed'] for d in data]) * 100
        kl_mean = np.mean([d['kl_div'] for d in data])
        prob_mean = np.mean([d['prob_change'] for d in data])
        
        if top1_rate > 5:
            interp = "STRONG causal"
        elif top1_rate > 1:
            interp = "Weak causal"
        else:
            interp = "~No causal"
        
        print(f"{dist:>5d} {n:>5d} {top1_rate:>7.2f}% {kl_mean:>10.6f} {prob_mean:>10.6f} {interp:>20}")
    
    # Part B: 方向依赖性
    print("\n=== Part B: Direction Dependence (same position, L0, last_pos) ===")
    for dir_type in ['row', 'null', 'random']:
        data = direction_results[dir_type]
        if data:
            top1_rate = np.mean([d['top1_changed'] for d in data]) * 100
            kl_mean = np.mean([d['kl_div'] for d in data])
            prob_mean = np.mean([d['prob_change'] for d in data])
            print(f"  {dir_type:>8}: top1_change={top1_rate:.2f}%, KL={kl_mean:.6f}, prob_change={prob_mean:.6f}")
    
    # Part C: 扰动强度依赖性
    print("\n=== Part C: Perturbation Scale Dependence ===")
    print(f"{'EPS':>8} {'N':>5} {'top1%':>8} {'KL':>10} {'prob_chg':>10}")
    for eps in sorted(scale_results.keys()):
        data = scale_results[eps]
        if data:
            n = len(data)
            top1_rate = np.mean([d['top1_changed'] for d in data]) * 100
            kl_mean = np.mean([d['kl_div'] for d in data])
            prob_mean = np.mean([d['prob_change'] for d in data])
            print(f"{eps:>8.2f} {n:>5d} {top1_rate:>7.2f}% {kl_mean:>10.6f} {prob_mean:>10.6f}")
    
    # 关键对比: Euclidean coupling ≈ 0 vs Causal sensitivity > 0
    print("\n=== KEY CONTRAST: Euclidean vs Causal ===")
    total_top1 = sum(1 for d in all_results if d['top1_changed'])
    total_n = len(all_results)
    overall_rate = total_top1 / total_n * 100 if total_n > 0 else 0
    print(f"Overall: {total_top1}/{total_n} = {overall_rate:.2f}% top1 changes")
    print(f"Phase 150 found: cross-coupling (Euclidean) ≈ 0")
    print(f"Phase 151b finds: causal sensitivity (top1 change) = {overall_rate:.1f}%")
    if overall_rate > 1:
        print(f">>> Euclidean coupling ≈ 0 BUT causal sensitivity > 0!")
        print(f">>> This CONFIRMS: small coupling can change discrete decisions!")
        print(f">>> Language constraint propagation IS causal, not Euclidean!")
    
    # 保存
    save_data = {
        "phase": "151b",
        "model": model_name,
        "timestamp": timestamp,
        "part_a_distance": {str(k): {"n": len(v), 
                                      "top1_rate": float(np.mean([d['top1_changed'] for d in v])),
                                      "kl_mean": float(np.mean([d['kl_div'] for d in v])),
                                      "prob_change_mean": float(np.mean([d['prob_change'] for d in v]))}
                           for k, v in distance_results.items()},
        "part_b_direction": {k: {"n": len(v),
                                  "top1_rate": float(np.mean([d['top1_changed'] for d in v])),
                                  "kl_mean": float(np.mean([d['kl_div'] for d in v])),
                                  "prob_change_mean": float(np.mean([d['prob_change'] for d in v]))}
                            for k, v in direction_results.items()},
        "part_c_scale": {str(k): {"n": len(v),
                                   "top1_rate": float(np.mean([d['top1_changed'] for d in v])),
                                   "kl_mean": float(np.mean([d['kl_div'] for d in v])),
                                   "prob_change_mean": float(np.mean([d['prob_change'] for d in v]))}
                         for k, v in scale_results.items()},
    }
    
    result_file = OUTPUT_DIR / f"phase151b_{model_name}_{timestamp}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {result_file}")
    
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    print("Model released.")


if __name__ == "__main__":
    main()
