"""
Subspace Topology Phase 2c: 控制位置距离的偏相关分析
====================================================

核心问题: attn↔Δ(QK)的正相关是否由位置距离驱动?
- 长距离位置对: 原始重叠低 + attention稀疏 → 可能产生假性正相关
- 需要偏除位置距离效应, 检验纯attn↔Δ(QK)关系

方法: 在相同位置距离区间内, 对比高attn和低attn的Δ(QK)

Run:
  python tests/glm5/subspace_topology_phase2c_partial_corr.py --model qwen3
"""
import sys, os, time, gc, json
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path("results/subspace_topology")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def log_time(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}")


def load_model_bf16(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from model_utils import MODEL_CONFIGS
    cfg = MODEL_CONFIGS[model_name]
    t0 = time.time()
    log_time(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    attn_impl = "eager" if model_name == "deepseek7b" else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, local_files_only=True, attn_implementation=attn_impl,
    )
    model.eval()
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log_time(f"{model_name} loaded, GPU={gpu_mem:.2f}GB, {time.time()-t0:.1f}s")
    return model, tokenizer, next(model.parameters()).device


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


def get_debiased_subspace(acts, n_dims=15):
    acts = np.nan_to_num(acts, nan=0.0, posinf=0.0, neginf=0.0)
    mean = acts.mean(axis=0)
    centered = acts - mean
    try:
        _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        from sklearn.decomposition import TruncatedSVD
        k = min(n_dims + 10, centered.shape[1] - 1, centered.shape[0] - 1)
        svd_obj = TruncatedSVD(n_components=k, random_state=42)
        svd_obj.fit(centered.astype(np.float32))
        S = svd_obj.singular_values_
        Vt = svd_obj.components_
    n_return = min(n_dims, len(S) - 1)
    return Vt[1:1+n_return, :].T, S


def subspace_overlap(V1, V2, k=None):
    if k:
        V1 = V1[:, :k]
    Q1 = np.linalg.qr(V1)[0]
    Q2 = np.linalg.qr(V2)[0]
    P2 = Q2 @ Q2.T
    proj = Q1.T @ P2 @ Q1
    return float(np.mean(np.diag(proj)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3")
    args = parser.parse_args()
    t_start = time.time()
    
    model, tokenizer, device = load_model_bf16(args.model)
    from model_utils import get_layers, get_model_info, release_model, get_layer_weights
    model_info = get_model_info(model, args.model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)
    
    try:
        n_heads = model.config.num_attention_heads
    except:
        n_heads = d_model
    head_dim = d_model // n_heads if d_model % n_heads == 0 else 128
    try:
        n_kv_heads = model.config.num_key_value_heads
    except:
        n_kv_heads = n_heads
    n_rep = n_heads // n_kv_heads
    
    log_time(f"n_heads={n_heads}, n_kv={n_kv_heads}, head_dim={head_dim}")
    
    # 采样层 (只分析中间和深层, 因为浅层相关性低)
    target_layers = sorted(set(
        list(range(0, n_layers, max(1, n_layers//6))) + 
        [n_layers//2, n_layers-2, n_layers-1]
    ))
    target_layers = sorted(set([l for l in target_layers if l < n_layers]))
    
    # 收集数据
    log_time(f"收集 {len(SENTENCES)} 句的attention weights和残差流...")
    
    max_len = 0
    tokenized = []
    for sent in SENTENCES:
        toks = tokenizer(sent, return_tensors="pt")
        seq_len = toks.input_ids.shape[1]
        tokenized.append((sent, toks, seq_len))
        max_len = max(max_len, seq_len)
    
    pos_layer_acts = {pos: {li: [] for li in range(n_layers)} for pos in range(max_len)}
    attn_data = {li: {} for li in target_layers}
    
    for si, (sent, toks, seq_len) in enumerate(tokenized):
        input_device = next(model.parameters()).device
        toks = toks.to(input_device)
        embed_layer = model.get_input_embeddings()
        inputs_embeds = embed_layer(toks.input_ids).detach().clone().to(model.dtype)
        position_ids = torch.arange(seq_len, device=input_device).unsqueeze(0)
        
        captured = {}
        hooks = []
        for li in range(n_layers):
            def make_hook(key):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        captured[key] = output[0].detach().float()
                    else:
                        captured[key] = output.detach().float()
                return hook
            hooks.append(layers[li].register_forward_hook(make_hook(f"L{li}")))
        
        with torch.no_grad():
            try:
                out = model(inputs_embeds=inputs_embeds, position_ids=position_ids, output_attentions=True)
            except:
                out = None
        
        for h in hooks:
            h.remove()
        
        if out is not None and out.attentions is not None:
            for ti, li in enumerate(target_layers):
                if li < len(out.attentions):
                    attn_data[li][si] = out.attentions[li][0].float().cpu().numpy()
        
        for li in range(n_layers):
            key = f"L{li}"
            if key in captured:
                acts = captured[key][0, :, :].cpu().numpy()
                for pos in range(seq_len):
                    pos_layer_acts[pos][li].append(acts[pos, :])
        
        del captured, out
        if (si + 1) % 10 == 0:
            log_time(f"  {si+1}/{len(SENTENCES)} 句")
    
    log_time(f"数据收集完成")
    
    # 提取权重
    def safe_get_weight(w):
        if w.is_meta:
            return None
        return w.detach().cpu().float().numpy()
    
    layer_wqk = {}
    for li in target_layers:
        try:
            sa = layers[li].self_attn
            W_q = safe_get_weight(sa.q_proj.weight)
            W_k = safe_get_weight(sa.k_proj.weight)
            if W_q is not None and W_k is not None:
                layer_wqk[li] = (W_q, W_k)
        except:
            pass
    target_layers = [l for l in target_layers if l in layer_wqk]
    
    # 核心分析: 按位置距离分组的偏相关
    n_dims = 15
    valid_positions = [p for p in range(max_len) if len(pos_layer_acts[p][0]) >= 10]
    
    results = {}
    
    for li in target_layers:
        W_q, W_k = layer_wqk[li]
        
        pos_subspaces = {}
        for pos in valid_positions:
            acts = np.array(pos_layer_acts[pos][li])
            if len(acts) < 5:
                continue
            V_sem, _ = get_debiased_subspace(acts, n_dims=n_dims)
            pos_subspaces[pos] = V_sem
        
        positions_with_data = sorted(pos_subspaces.keys())
        if len(positions_with_data) < 3:
            continue
        
        # 原始子空间重叠度
        original_overlaps = {}
        for i, p1 in enumerate(positions_with_data):
            for j, p2 in enumerate(positions_with_data):
                if j <= i:
                    continue
                original_overlaps[(p1, p2)] = subspace_overlap(pos_subspaces[p1], pos_subspaces[p2], k=n_dims)
        
        # Q-K子空间重叠度 (取所有head平均)
        qk_avg_overlaps = {}
        for i, p1 in enumerate(positions_with_data):
            Q_p1_all = W_q @ pos_subspaces[p1]  # [n_heads*hd, 15]
            for j, p2 in enumerate(positions_with_data):
                if j <= i:
                    continue
                K_p2_all = W_k @ pos_subspaces[p2]  # [n_kv*hd, 15]
                
                # 逐head计算, 取平均
                head_overlaps = []
                for h in range(min(n_heads, 32)):
                    W_q_h = W_q[h*head_dim:(h+1)*head_dim, :]
                    if n_kv_heads < n_heads:
                        kv_h = h // n_rep
                        W_k_h = W_k[kv_h*head_dim:(kv_h+1)*head_dim, :]
                    else:
                        W_k_h = W_k[h*head_dim:(h+1)*head_dim, :]
                    
                    Q_h = W_q_h @ pos_subspaces[p1]  # [head_dim, 15]
                    K_h = W_k_h @ pos_subspaces[p2]   # [head_dim, 15]
                    
                    if head_dim >= n_dims:
                        ov = subspace_overlap(Q_h, K_h, k=n_dims)
                        head_overlaps.append(ov)
                
                qk_avg_overlaps[(p1, p2)] = np.mean(head_overlaps) if head_overlaps else 0
        
        # 收集(attn, ΔQK, distance)三元组
        triplets = []  # (attn_weight, delta_qk, distance)
        
        for si in attn_data[li]:
            attn_w = attn_data[li][si]  # [n_heads, seq_len, seq_len]
            seq_len = attn_w.shape[1]
            
            for qi in range(min(seq_len, max_len)):
                if qi not in pos_subspaces:
                    continue
                for kj in range(min(qi, max_len)):  # causal
                    if kj not in pos_subspaces:
                        continue
                    
                    # 平均所有head的attention
                    avg_attn = float(attn_w[:, qi, kj].mean())
                    pair = (kj, qi) if kj < qi else (qi, kj)
                    delta_qk = qk_avg_overlaps.get(pair, 0) - original_overlaps.get(pair, 0)
                    distance = abs(qi - kj)
                    
                    triplets.append((avg_attn, delta_qk, distance))
        
        if len(triplets) < 50:
            continue
        
        attn_arr = np.array([t[0] for t in triplets])
        delta_arr = np.array([t[1] for t in triplets])
        dist_arr = np.array([t[2] for t in triplets])
        
        layer_result = {"layer": li, "n_samples": len(triplets)}
        
        # 1. 全局相关性
        corr_global = np.corrcoef(attn_arr, delta_arr)[0, 1]
        corr_attn_dist = np.corrcoef(attn_arr, dist_arr)[0, 1]
        corr_delta_dist = np.corrcoef(delta_arr, dist_arr)[0, 1]
        
        log_time(f"\n--- Layer {li} ---")
        log_time(f"  样本数: {len(triplets)}")
        log_time(f"  全局: corr(attn,ΔQK)={corr_global:.3f}, corr(attn,dist)={corr_attn_dist:.3f}, corr(ΔQK,dist)={corr_delta_dist:.3f}")
        
        layer_result["corr_global"] = float(corr_global)
        layer_result["corr_attn_dist"] = float(corr_attn_dist)
        layer_result["corr_delta_dist"] = float(corr_delta_dist)
        
        # 2. 偏相关: 控制distance后attn↔ΔQK的关系
        # 使用分距离区间的方法
        unique_dists = sorted(set(dist_arr))
        if len(unique_dists) < 2:
            continue
        
        # 按距离分组: adjacent(1), short(2-3), medium(4-5), long(6+)
        dist_groups = {
            "adjacent(1)": dist_arr == 1,
            "short(2-3)": (dist_arr >= 2) & (dist_arr <= 3),
            "medium(4-5)": (dist_arr >= 4) & (dist_arr <= 5),
            "long(6+)": dist_arr >= 6,
        }
        
        log_time(f"  按距离分组的偏相关:")
        for gname, mask in dist_groups.items():
            if mask.sum() < 20:
                continue
            g_attn = attn_arr[mask]
            g_delta = delta_arr[mask]
            g_corr = np.corrcoef(g_attn, g_delta)[0, 1] if len(g_attn) > 5 else 0
            
            # 高attn vs 低attn的ΔQK对比
            p75 = np.percentile(g_attn, 75)
            p25 = np.percentile(g_attn, 25)
            high_mask = mask & (attn_arr >= p75)
            low_mask = mask & (attn_arr <= p25)
            
            high_delta = delta_arr[high_mask].mean() if high_mask.sum() > 0 else 0
            low_delta = delta_arr[low_mask].mean() if low_mask.sum() > 0 else 0
            
            log_time(f"    {gname}: n={mask.sum()}, corr={g_corr:.3f}, "
                     f"high_attn_Δ={high_delta:+.4f}, low_attn_Δ={low_delta:+.4f}, diff={high_delta-low_delta:+.4f}")
            
            layer_result[f"dist_{gname}_corr"] = float(g_corr)
            layer_result[f"dist_{gname}_high_delta"] = float(high_delta)
            layer_result[f"dist_{gname}_low_delta"] = float(low_delta)
        
        # 3. Semi-partial correlation (Fisher's z变换)
        # 计算偏除distance后的attn↔ΔQK相关性
        if len(triplets) > 50:
            from scipy import stats
            # 简单偏相关: r(attn,ΔQK|dist) = (r_attn_delta - r_attn_dist*r_delta_dist) / sqrt(...)
            r_ad = corr_global
            r_ax = corr_attn_dist
            r_dx = corr_delta_dist
            
            denom = np.sqrt(max((1 - r_ax**2) * (1 - r_dx**2), 1e-10))
            partial_corr = (r_ad - r_ax * r_dx) / denom
            
            log_time(f"  偏相关(控制距离): r(attn,ΔQK|dist)={partial_corr:.3f}")
            layer_result["partial_corr"] = float(partial_corr)
        
        results[f"L{li}"] = layer_result
    
    # 综合总结
    log_time(f"\n{'='*80}")
    log_time(f"核心总结: 控制位置距离后的attn↔Δ(QK)关系")
    log_time(f"{'='*80}")
    
    for li in target_layers:
        if f"L{li}" not in results:
            continue
        r = results[f"L{li}"]
        gc = r.get("corr_global", 0)
        pc = r.get("partial_corr", 0)
        
        # 各距离组的corr
        dist_corrs = []
        for gname in ["adjacent(1)", "short(2-3)", "medium(4-5)", "long(6+)"]:
            k = f"dist_{gname}_corr"
            if k in r:
                dist_corrs.append(f"{gname}={r[k]:.3f}")
        
        log_time(f"  L{li:2d}: global={gc:.3f}, partial={pc:.3f} | {', '.join(dist_corrs)}")
    
    # 保存
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj
    
    out_path = OUTPUT_DIR / f"exp2c_partial_corr_{model_info.name}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(convert(results), f, indent=2, ensure_ascii=False)
    log_time(f"结果已保存到 {out_path}")
    
    release_model(model)
    log_time(f"总耗时: {(time.time()-t_start)/60:.1f}分钟")
    log_time("Done!")


if __name__ == "__main__":
    main()
