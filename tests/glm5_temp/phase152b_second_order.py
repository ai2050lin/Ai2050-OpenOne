"""
Phase 152b: 修正版 Second-Order Propagation Experiment
======================================================

修正: hidden_states[0] = embedding, hidden_states[1] = L0 output
→ 用 hs[1] 的差作为 δ_ref, 不是 hs[0]
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
from model_utils import (load_model, get_layers, get_model_info, release_model)

OUTPUT_DIR = Path("tests/glm5_temp")

TEST_PROMPT = "The scientist discovered that the"
N_PERTURBATIONS = 200


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    print(f"Phase 152b: Second-Order Propagation (FIXED)")
    print(f"Model: {model_name}, Time: {timestamp}")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    
    prompt = TEST_PROMPT
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    last_pos = input_ids.shape[1] - 1
    
    with torch.no_grad():
        out_clean = model(input_ids=input_ids, attention_mask=attention_mask,
                          output_hidden_states=True)
    clean_hs = out_clean.hidden_states
    # hidden_states: [0]=embedding, [1]=L0 output, ..., [n_layers]=L(n-1) output
    
    sample_layers = [1, 2, 4, 6, 8, 12, 18, 24, 30, 36]  # 从hs[1]开始!
    
    # 收集各层的扰动响应
    delta_at_layer = {}
    for li in sample_layers:
        delta_at_layer[li] = []
    
    # 同时记录注入的原始delta方向
    input_deltas = []
    
    layers = get_layers(model)
    
    for p_idx in range(N_PERTURBATIONS):
        np.random.seed(200 + p_idx)
        delta = np.random.randn(d_model)
        delta = delta / np.linalg.norm(delta) * 1.0
        input_deltas.append(delta.copy())
        
        delta_tensor = torch.tensor(delta, dtype=torch.float32)
        
        def make_hook(pos, delta_t):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    out = output[0].clone()
                    out[0, pos, :] += delta_t.to(out.dtype).to(out.device)
                    return (out,) + output[1:]
                else:
                    out = output.clone()
                    out[0, pos, :] += delta_t.to(out.dtype).to(out.device)
                    return out
            return hook
        
        hooks = [layers[0].register_forward_hook(make_hook(last_pos, delta_tensor))]
        
        try:
            with torch.no_grad():
                out_p = model(input_ids=input_ids, attention_mask=attention_mask,
                              output_hidden_states=True)
            
            for li in sample_layers:
                p_vec = out_p.hidden_states[li][0, last_pos, :].float().cpu().numpy()
                c_vec = clean_hs[li][0, last_pos, :].float().cpu().numpy()
                delta_at_layer[li].append(p_vec - c_vec)
        except:
            pass
        
        for h in hooks:
            h.remove()
        
        if p_idx % 50 == 0:
            print(f"  Progress: {p_idx}/{N_PERTURBATIONS}")
    
    print(f"  Progress: {N_PERTURBATIONS}/{N_PERTURBATIONS}")
    
    # === 分析 ===
    print("\n" + "="*60)
    print("Results: Second-Order Propagation (FIXED)")
    print("="*60)
    
    # 参考层: hs[1] = L0 output (第一个有意义的层)
    ref_layer = 1
    delta_ref = np.array(delta_at_layer[ref_layer])  # [N, d_model]
    input_deltas_arr = np.array(input_deltas)  # [N, d_model]
    
    # 1. 一阶cos: cos(δ^(ℓ), δ_input)
    print("\n  --- First-Order: cos(δ_ℓ, δ_input) ---")
    first_order_cos = {}
    for li in sample_layers:
        cos_values = []
        delta_l = np.array(delta_at_layer[li])
        for p in range(min(100, delta_l.shape[0])):
            nl = np.linalg.norm(delta_l[p])
            ni = np.linalg.norm(input_deltas_arr[p])
            if nl > 1e-10 and ni > 1e-10:
                cos_values.append(float(np.dot(delta_l[p], input_deltas_arr[p]) / (nl * ni)))
        avg_cos = np.mean(cos_values) if cos_values else 0
        first_order_cos[li] = avg_cos
        print(f"    L{li:>3d}(hs[{li}]): cos(δ_ℓ, δ_input)={avg_cos:.6f}")
    
    # 2. 一阶cos: cos(δ^(ℓ), δ^(ref))
    print(f"\n  --- First-Order: cos(δ_ℓ, δ_ref) [ref=hs[{ref_layer}]] ---")
    first_order_cos_ref = {}
    for li in sample_layers:
        cos_values = []
        delta_l = np.array(delta_at_layer[li])
        for p in range(min(100, min(delta_l.shape[0], delta_ref.shape[0]))):
            nl = np.linalg.norm(delta_l[p])
            nr = np.linalg.norm(delta_ref[p])
            if nl > 1e-10 and nr > 1e-10:
                cos_values.append(float(np.dot(delta_l[p], delta_ref[p]) / (nl * nr)))
        avg_cos = np.mean(cos_values) if cos_values else 0
        first_order_cos_ref[li] = avg_cos
        print(f"    L{li:>3d}(hs[{li}]): cos(δ_ℓ, δ_ref)={avg_cos:.6f}")
    
    # 3. 二阶: PCA + subspace overlap
    print(f"\n  --- Second-Order: PCA Subspace Overlap [ref=hs[{ref_layer}]] ---")
    
    # PCA of delta_ref
    delta_ref_centered = delta_ref - delta_ref.mean(axis=0)
    U_ref, s_ref, Vt_ref = np.linalg.svd(delta_ref_centered, full_matrices=False)
    pcs_ref = Vt_ref[:min(10, Vt_ref.shape[0]), :]  # [10, d_model]
    
    second_order_overlap = {}
    pc1_corr = {}
    
    for li in sample_layers:
        delta_l = np.array(delta_at_layer[li])
        if delta_l.shape[0] < 10:
            continue
        
        delta_l_centered = delta_l - delta_l.mean(axis=0)
        try:
            U_l, s_l, Vt_l = np.linalg.svd(delta_l_centered, full_matrices=False)
            pcs_l = Vt_l[:min(10, Vt_l.shape[0]), :]
            
            # PC1 correlation
            pc1_c = abs(float(np.dot(pcs_l[0], pcs_ref[0])))
            pc1_corr[li] = pc1_c
            
            # Subspace overlap (top 5 PCs)
            n_sub = min(5, pcs_ref.shape[0], pcs_l.shape[0])
            if n_sub > 0:
                Q_ref = pcs_ref[:n_sub].T @ pcs_ref[:n_sub]
                Q_l = pcs_l[:n_sub].T @ pcs_l[:n_sub]
                overlap = np.trace(Q_ref @ Q_l) / n_sub
            else:
                overlap = 0
            second_order_overlap[li] = overlap
            
            # Effective rank
            total_e = np.sum(s_l ** 2)
            cumul = np.cumsum(s_l ** 2)
            k90 = np.searchsorted(cumul, 0.90 * total_e) + 1
            top_ratio = s_l[0] ** 2 / total_e if total_e > 0 else 0
            
            print(f"    L{li:>3d}(hs[{li}]): PC1_corr={pc1_c:.4f}, overlap={overlap:.4f}, "
                  f"rank(90%)={k90}, top_ratio={top_ratio:.4f}")
        except:
            pass
    
    # 4. 关键对比: 一阶 vs 二阶
    print(f"\n  === CRITICAL: First-Order vs Second-Order Decay ===")
    print(f"  {'Layer':>6} | {'cos(1st-order)':>14} | {'overlap(2nd-order)':>18} | {'Diagnosis'}")
    print(f"  {'-'*6}-+-{'-'*14}-+-{'-'*18}-+-{'-'*20}")
    
    for li in sample_layers:
        cos_1st = first_order_cos.get(li, 0)
        overlap_2nd = second_order_overlap.get(li, 0)
        pc1 = pc1_corr.get(li, 0)
        
        if cos_1st < 0.1 and overlap_2nd > 0.2:
            diagnosis = "2ND-ORDER PRESERVED ★"
        elif cos_1st < 0.1 and overlap_2nd < 0.05:
            diagnosis = "1st&2nd both decayed"
        elif cos_1st >= 0.1:
            diagnosis = "1st-order still present"
        else:
            diagnosis = "mixed/weak 2nd-order"
        
        print(f"  L{li:>4d} | {cos_1st:>14.6f} | {overlap_2nd:>18.4f} | {diagnosis}")
    
    # 保存结果
    results = {
        "phase": "152b",
        "model": model_name,
        "timestamp": timestamp,
        "first_order_cos_with_input": first_order_cos,
        "first_order_cos_with_ref": first_order_cos_ref,
        "second_order_overlap": second_order_overlap,
        "pc1_correlation": pc1_corr,
    }
    
    result_file = OUTPUT_DIR / f"phase152b_{model_name}_{timestamp}.json"
    
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        raise TypeError(f"Cannot serialize {type(obj)}")
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=convert, ensure_ascii=False)
    
    print(f"\nResults saved to: {result_file}")
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    print("Done!")


if __name__ == "__main__":
    main()
