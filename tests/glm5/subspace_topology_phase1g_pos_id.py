"""
Subspace Topology Phase 1g: 逐Token位置ID Profile
==================================================

核心问题: "2D语义空间"是全token混合的假象吗？
验证: 对同一组句子，分别计算每个token位置的ID

假设: 预测位置(last token)的ID远高于非预测位置

Run:
  python tests/glm5/subspace_topology_phase1g_pos_id.py --model qwen3
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import argparse
import gc
import json
from pathlib import Path

from model_utils import load_model, get_layers, get_model_info, release_model

OUTPUT_DIR = Path("results/subspace_topology")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 所有句子长度接近(~7-10 tokens)以便按位置分组
SENTENCES = [
    "The apple is red and sweet.",
    "Paris is the capital of France.",
    "Water boils at 100 degrees.",
    "The apple is NOT red at all.",
    "Is the apple really red now?",
    "Justice is a fundamental concept.",
    "John gave Mary a book today.",
    "The sky is blue and clear.",
    "The sky is NOT blue today.",
    "Is the sky blue and clear?",
    "All cats are animals, clearly.",
    "Write code to sort a list.",
    "What is 49 plus 2 equal?",
    "If A=B and B=C, then A=",
    "The earth revolves around us.",
    "Ice melts when heated slowly.",
    "Books are made of paper too.",
    "Birds can fly in the sky.",
    "Freedom means different things.",
    "Time passes differently always.",
    "The physical color is green.",
    "It is not true that red.",
    "5, 6, 7, and 8 follow on.",
    "What is 15 times 7 equal?",
    "Translate this to French now.",
    "Create a class for linked list.",
    "The shape of a ball is round.",
    "Alice told Bob a big secret.",
    "Dogs are loyal and friendly.",
    "饕餮是一种传说中的神兽。",
]


def compute_participation_ratio(eigenvalues):
    lam = np.array(eigenvalues, dtype=np.float64)
    lam = lam[lam > 1e-12]
    if len(lam) == 0:
        return 0.0
    return float((np.sum(lam))**2 / np.sum(lam**2))


def robust_svd(matrix, k=None):
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
        return U, S, Vt
    except np.linalg.LinAlgError:
        from sklearn.decomposition import TruncatedSVD
        k = k or min(100, matrix.shape[1] - 1, matrix.shape[0] - 1)
        svd_obj = TruncatedSVD(n_components=k, random_state=42)
        svd_obj.fit(matrix.astype(np.float32))
        return None, svd_obj.singular_values_, svd_obj.components_


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3")
    args = parser.parse_args()
    
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)
    
    print(f"\n模型: {model_info.name}, {n_layers}层, d_model={d_model}")
    
    # ========================================
    # 收集激活: 按位置分组
    # ========================================
    print(f"\n收集 {len(SENTENCES)} 个句子的残差流(按位置分组)...")
    
    # 确定最大序列长度
    max_len = 0
    tokenized = []
    for sent in SENTENCES:
        toks = tokenizer(sent, return_tensors="pt")
        seq_len = toks.input_ids.shape[1]
        tokenized.append((sent, toks, seq_len))
        max_len = max(max_len, seq_len)
    
    print(f"最大序列长度: {max_len}")
    
    # 按位置和层存储激活
    # pos_layer_acts[pos][layer] = list of activations at that position
    pos_layer_acts = {pos: {li: [] for li in range(n_layers)} for pos in range(max_len)}
    
    for si, (sent, toks, seq_len) in enumerate(tokenized):
        toks = toks.to(device)
        embed_layer = model.get_input_embeddings()
        inputs_embeds = embed_layer(toks.input_ids).detach().clone().to(model.dtype)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        
        captured = {}
        hooks = []
        for li in range(n_layers):
            layer = layers[li]
            def make_hook(key):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        captured[key] = output[0].detach().float()
                    else:
                        captured[key] = output.detach().float()
                return hook
            hooks.append(layer.register_forward_hook(make_hook(f"L{li}")))
        
        with torch.no_grad():
            try:
                _ = model(inputs_embeds=inputs_embeds, position_ids=position_ids)
            except Exception:
                pass
        
        for h in hooks:
            h.remove()
        
        for li in range(n_layers):
            key = f"L{li}"
            if key in captured:
                acts = captured[key][0, :, :].cpu().numpy()  # [seq_len, d_model]
                for pos in range(seq_len):
                    pos_layer_acts[pos][li].append(acts[pos, :])
        
        del captured
        gc.collect()
    
    # ========================================
    # 按位置计算ID profile
    # ========================================
    target_layers = sorted(set(
        [0, 1, 5, 6] + 
        list(range(0, n_layers, max(1, n_layers//8))) + 
        [n_layers-3, n_layers-2, n_layers-1]
    ))
    target_layers = sorted(set([l for l in target_layers if l < n_layers]))
    
    # 只分析有足够样本的位置(至少10个)
    valid_positions = [p for p in range(max_len) 
                       if len(pos_layer_acts[p][0]) >= 10]
    
    print(f"\n有效位置: {valid_positions} (需要≥10个样本)")
    print(f"目标层: {target_layers}")
    
    results = {}
    
    for li in target_layers:
        layer_result = {"layer": li}
        
        # 全位置混合ID (baseline)
        all_acts = []
        for pos in valid_positions:
            all_acts.extend(pos_layer_acts[pos][li])
        all_acts = np.array(all_acts)
        mean = all_acts.mean(axis=0)
        centered = all_acts - mean
        _, S_all, _ = robust_svd(centered)
        id_all = compute_participation_ratio(S_all**2 / (len(all_acts) - 1)) if S_all is not None else 0
        
        # 去偏置ID
        if S_all is not None and len(S_all) > 1:
            U_all, _, Vt_all = robust_svd(centered)
            if U_all is not None:
                debiased = centered - U_all[:, :1] @ np.diag(S_all[:1]) @ Vt_all[:1, :]
            else:
                proj = centered @ Vt_all[:1, :].T
                debiased = centered - proj @ Vt_all[:1, :]
            _, S_db, _ = robust_svd(debiased)
            id_debiased = compute_participation_ratio(S_db**2 / (len(all_acts) - 1)) if S_db is not None else 0
        else:
            id_debiased = 0
        
        layer_result["all_pos"] = {"total_ID": id_all, "debiased_ID": id_debiased, "n_samples": len(all_acts)}
        
        # 逐位置ID
        pos_ids = {}
        for pos in valid_positions:
            acts = np.array(pos_layer_acts[pos][li])
            if len(acts) < 5:
                continue
            mean_p = acts.mean(axis=0)
            centered_p = acts - mean_p
            _, S_p, _ = robust_svd(centered_p)
            id_p = compute_participation_ratio(S_p**2 / (len(acts) - 1)) if S_p is not None else 0
            
            # 去偏置ID
            if S_p is not None and len(S_p) > 1:
                U_p, _, Vt_p = robust_svd(centered_p)
                if U_p is not None:
                    debiased_p = centered_p - U_p[:, :1] @ np.diag(S_p[:1]) @ Vt_p[:1, :]
                else:
                    proj = centered_p @ Vt_p[:1, :].T
                    debiased_p = centered_p - proj @ Vt_p[:1, :]
                _, S_dp, _ = robust_svd(debiased_p)
                id_dp = compute_participation_ratio(S_dp**2 / (len(acts) - 1)) if S_dp is not None else 0
            else:
                id_dp = 0
            
            pos_ids[pos] = {
                "total_ID": id_p, 
                "debiased_ID": id_dp,
                "n_samples": len(acts),
                "is_last": pos == max(valid_positions)
            }
        
        layer_result["per_position"] = pos_ids
        results[f"L{li}"] = layer_result
        
        # 打印
        print(f"\n  L{li:2d}: all_pos ID={id_all:.1f}, debiased={id_debiased:.1f}")
        for pos in valid_positions:
            if pos in pos_ids:
                pid = pos_ids[pos]
                marker = " ← LAST" if pid["is_last"] else ""
                print(f"    pos{pos:2d}: ID={pid['total_ID']:7.2f}  debiased={pid['debiased_ID']:7.2f}  n={pid['n_samples']}{marker}")
    
    # ========================================
    # 核心对比总结
    # ========================================
    print(f"\n{'='*80}")
    print("核心对比: 预测位置(last) vs 非预测位置(其他) 的语义维度")
    print(f"{'='*80}")
    
    for li in target_layers:
        r = results[f"L{li}"]
        pos_data = r["per_position"]
        last_pos = max(valid_positions)
        
        # 非最后位置的平均ID
        non_last_ids = [pos_data[p]["debiased_ID"] for p in valid_positions 
                       if p in pos_data and p != last_pos]
        last_id = pos_data.get(last_pos, {}).get("debiased_ID", 0)
        
        avg_non_last = np.mean(non_last_ids) if non_last_ids else 0
        ratio = last_id / avg_non_last if avg_non_last > 0 else float('inf')
        
        print(f"  L{li:2d}: last_pos debiased_ID={last_id:.1f}, avg_other={avg_non_last:.1f}, ratio={ratio:.1f}x")
    
    # 保存
    out_path = OUTPUT_DIR / f"exp1g_position_id_{model_info.name}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {out_path}")
    
    release_model(model)
    print("Done!")


if __name__ == "__main__":
    main()
