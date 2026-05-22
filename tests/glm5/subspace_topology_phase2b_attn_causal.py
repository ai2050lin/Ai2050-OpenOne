"""
Subspace Topology Phase 2b: Attention Pattern 与子空间翻译的因果关系
====================================================================

核心问题: 子空间翻译效应(ΔK最大)是否真正驱动了attention的信息传递?

实验设计:
1. 收集实际的attention weights (softmax后的注意力分布)
2. 对比"高注意力权重位置对"和"低注意力权重位置对"的子空间对齐度
3. 如果翻译效应是因果的: 高注意力位置对应该有更高的Q-K子空间对齐度
4. 计算attention entropy与翻译效应的相关性

Run:
  python tests/glm5/subspace_topology_phase2b_attn_causal.py --model qwen3
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


# ===== 模型加载 (BF16 + device_map="auto") =====
def load_model_bf16(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from model_utils import MODEL_CONFIGS
    
    cfg = MODEL_CONFIGS[model_name]
    t0 = time.time()
    print(f"[{datetime.now():%H:%M:%S}] Loading {model_name} (bf16 + auto)...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    attn_impl = "eager" if model_name == "deepseek7b" else "sdpa"
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation=attn_impl,
    )
    model.eval()
    
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"[{datetime.now():%H:%M:%S}] {model_name} loaded: GPU={gpu_mem:.2f}GB, {time.time()-t0:.1f}s")
    
    return model, tokenizer, device


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
    
    # ===== 1. 加载模型 =====
    model, tokenizer, device = load_model_bf16(args.model)
    from model_utils import get_layers, get_model_info, release_model, get_layer_weights
    model_info = get_model_info(model, args.model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)
    
    log_time(f"模型: {model_info.name}, {n_layers}层, d_model={d_model}")
    
    # 获取head信息
    try:
        n_heads = model.config.num_attention_heads
    except:
        n_heads = d_model
    head_dim = d_model // n_heads if d_model % n_heads == 0 else 128
    
    # GQA
    try:
        n_kv_heads = model.config.num_key_value_heads
    except:
        n_kv_heads = n_heads
    n_rep = n_heads // n_kv_heads
    
    log_time(f"n_heads={n_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}, n_rep={n_rep}")
    
    # ===== 2. 收集attention patterns和残差流 =====
    log_time(f"收集 {len(SENTENCES)} 句的attention weights和残差流...")
    
    # 采样层
    target_layers = sorted(set(
        [0, 1] + list(range(0, n_layers, max(1, n_layers//6))) + 
        [n_layers//2, n_layers-2, n_layers-1]
    ))
    target_layers = sorted(set([l for l in target_layers if l < n_layers]))
    
    # 按层和句子存储attention weights: {li: {si: attn_weights}}
    # attn_weights shape: [n_heads, seq_len, seq_len]
    attn_data = {li: {} for li in target_layers}
    
    # 按位置和层存储残差流
    max_len = 0
    tokenized = []
    for sent in SENTENCES:
        toks = tokenizer(sent, return_tensors="pt")
        seq_len = toks.input_ids.shape[1]
        tokenized.append((sent, toks, seq_len))
        max_len = max(max_len, seq_len)
    
    pos_layer_acts = {pos: {li: [] for li in range(n_layers)} for pos in range(max_len)}
    
    for si, (sent, toks, seq_len) in enumerate(tokenized):
        input_device = next(model.parameters()).device
        toks = toks.to(input_device)
        embed_layer = model.get_input_embeddings()
        inputs_embeds = embed_layer(toks.input_ids).detach().clone().to(model.dtype)
        position_ids = torch.arange(seq_len, device=input_device).unsqueeze(0)
        
        # 收集residual stream
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
        
        # 收集attention weights (需要output_attentions=True)
        with torch.no_grad():
            try:
                out = model(inputs_embeds=inputs_embeds, position_ids=position_ids,
                           output_attentions=True)
            except Exception as e:
                log_time(f"  句{si} 前向失败: {e}")
                out = None
        
        for h in hooks:
            h.remove()
        
        # 提取attention weights
        if out is not None and out.attentions is not None:
            for ti, li in enumerate(target_layers):
                if li < len(out.attentions):
                    # out.attentions[li]: [1, n_heads, seq_len, seq_len]
                    attn_w = out.attentions[li][0].float().cpu().numpy()  # [n_heads, seq_len, seq_len]
                    attn_data[li][si] = attn_w
        
        # 提取residual stream
        for li in range(n_layers):
            key = f"L{li}"
            if key in captured:
                acts = captured[key][0, :, :].cpu().numpy()
                for pos in range(seq_len):
                    pos_layer_acts[pos][li].append(acts[pos, :])
        
        del captured, out
        if (si + 1) % 10 == 0:
            log_time(f"  已处理 {si+1}/{len(SENTENCES)} 句")
            torch.cuda.empty_cache()
    
    log_time(f"数据收集完成")
    
    # ===== 3. 提取权重 =====
    log_time(f"提取W_q, W_k权重...")
    
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
    log_time(f"可分析层: {target_layers}")
    
    # ===== 4. 核心分析: attention pattern vs 子空间对齐 =====
    log_time(f"核心分析: attention权重 vs Q-K子空间对齐度...")
    
    n_dims = 15
    valid_positions = [p for p in range(max_len) if len(pos_layer_acts[p][0]) >= 10]
    
    results = {}
    
    for li in target_layers:
        W_q, W_k = layer_wqk[li]
        
        # 获取各位置的语义子空间
        pos_subspaces = {}
        for pos in valid_positions:
            acts = np.array(pos_layer_acts[pos][li])
            if len(acts) < 5:
                continue
            V_sem, _ = get_debiased_subspace(acts, n_dims=n_dims)
            pos_subspaces[pos] = V_sem
        
        positions_with_data = sorted(pos_subspaces.keys())
        if len(positions_with_data) < 2:
            continue
        
        # 原始子空间重叠度
        original_overlaps = {}
        for i, p1 in enumerate(positions_with_data):
            for j, p2 in enumerate(positions_with_data):
                if j <= i:
                    continue
                original_overlaps[(p1, p2)] = subspace_overlap(pos_subspaces[p1], pos_subspaces[p2], k=n_dims)
        
        # Q-K子空间重叠度 (逐head)
        qk_head_overlaps = {}
        for h in range(min(n_heads, 32)):
            W_q_h = W_q[h*head_dim:(h+1)*head_dim, :]
            if n_kv_heads < n_heads:
                kv_h = h // n_rep
                W_k_h = W_k[kv_h*head_dim:(kv_h+1)*head_dim, :]
            else:
                W_k_h = W_k[h*head_dim:(h+1)*head_dim, :]
            
            qk_overlaps = {}
            for i, p1 in enumerate(positions_with_data):
                Q_h_p1 = W_q_h @ pos_subspaces[p1]  # [head_dim, 15]
                for j, p2 in enumerate(positions_with_data):
                    if j <= i:
                        continue
                    K_h_p2 = W_k_h @ pos_subspaces[p2]  # [head_dim, 15]
                    if head_dim >= n_dims:
                        ov = subspace_overlap(Q_h_p1, K_h_p2, k=n_dims)
                        qk_overlaps[(p1, p2)] = ov
            
            if qk_overlaps:
                qk_head_overlaps[h] = qk_overlaps
        
        # 对比: 高attention vs 低attention位置对的子空间对齐度
        # 收集所有(attention_weight, qk_overlap, original_overlap)三元组
        attn_qk_pairs = []  # (attn_weight, qk_overlap, original_overlap, layer, head)
        
        for si in attn_data[li]:
            attn_w = attn_data[li][si]  # [n_heads, seq_len, seq_len]
            seq_len = attn_w.shape[1]
            
            for h in range(min(n_heads, 32)):
                if h not in qk_head_overlaps:
                    continue
                
                for qi in range(min(seq_len, max_len)):
                    if qi not in pos_subspaces:
                        continue
                    for kj in range(min(seq_len, max_len)):
                        if kj not in pos_subspaces or kj >= qi:
                            continue  # causal: 只看kj < qi
                        
                        attn_weight = float(attn_w[h, qi, kj])
                        pair = (kj, qi) if kj < qi else (qi, kj)
                        
                        if pair in qk_head_overlaps[h]:
                            qk_ov = qk_head_overlaps[h][pair]
                            orig_ov = original_overlaps.get(pair, 0)
                            attn_qk_pairs.append((attn_weight, qk_ov, orig_ov, li, h))
        
        if len(attn_qk_pairs) < 10:
            log_time(f"  L{li}: 数据不足({len(attn_qk_pairs)}对), 跳过")
            continue
        
        # 分析
        attn_weights = np.array([p[0] for p in attn_qk_pairs])
        qk_overlaps_arr = np.array([p[1] for p in attn_qk_pairs])
        orig_overlaps_arr = np.array([p[2] for p in attn_qk_pairs])
        delta_arr = qk_overlaps_arr - orig_overlaps_arr  # Δ(QK)
        
        # 按attention权重分组
        n_groups = 4
        attn_percentiles = np.percentile(attn_weights, [25, 50, 75])
        
        groups = {
            "low": attn_weights <= attn_percentiles[0],
            "med_low": (attn_weights > attn_percentiles[0]) & (attn_weights <= attn_percentiles[1]),
            "med_high": (attn_weights > attn_percentiles[1]) & (attn_weights <= attn_percentiles[2]),
            "high": attn_weights > attn_percentiles[2],
        }
        
        layer_result = {"layer": li}
        log_time(f"\n--- Layer {li} ---")
        log_time(f"  样本数: {len(attn_qk_pairs)}")
        log_time(f"  Attention分布: mean={attn_weights.mean():.4f}, std={attn_weights.std():.4f}")
        
        for gname, mask in groups.items():
            if mask.sum() < 5:
                continue
            g_attn = attn_weights[mask].mean()
            g_qk = qk_overlaps_arr[mask].mean()
            g_orig = orig_overlaps_arr[mask].mean()
            g_delta = delta_arr[mask].mean()
            
            log_time(f"  {gname:10s}: attn={g_attn:.4f}, QK_overlap={g_qk:.4f}, "
                     f"orig={g_orig:.4f}, Δ(QK)={g_delta:+.4f}")
            
            layer_result[f"{gname}_attn"] = float(g_attn)
            layer_result[f"{gname}_qk_overlap"] = float(g_qk)
            layer_result[f"{gname}_orig_overlap"] = float(g_orig)
            layer_result[f"{gname}_delta"] = float(g_delta)
        
        # 相关性分析
        if len(attn_qk_pairs) > 20:
            corr_qk = np.corrcoef(attn_weights, qk_overlaps_arr)[0, 1]
            corr_delta = np.corrcoef(attn_weights, delta_arr)[0, 1]
            log_time(f"  相关性: attn↔QK_overlap={corr_qk:.3f}, attn↔Δ(QK)={corr_delta:.3f}")
            layer_result["corr_attn_qk"] = float(corr_qk)
            layer_result["corr_attn_delta"] = float(corr_delta)
        
        # Attention entropy vs 翻译效应
        # 高entropy = 分散注意力 = 弱翻译; 低entropy = 集中注意力 = 强翻译?
        entropies = []
        mean_deltas = []
        for h in range(min(n_heads, 32)):
            if h not in qk_head_overlaps:
                continue
            head_attn = []
            head_deltas = []
            for si in attn_data[li]:
                attn_w = attn_data[li][si]
                for qi in range(attn_w.shape[1]):
                    row = attn_w[h, qi, :qi+1]
                    row = row / max(row.sum(), 1e-10)  # normalize
                    if len(row) > 1:
                        ent = -np.sum(row * np.log(row + 1e-10))
                        head_attn.append(ent)
            
            if head_attn:
                entropies.append(np.mean(head_attn))
                # 该head的平均Δ(QK)
                head_delta_vals = list(qk_head_overlaps[h].values())
                head_orig_vals = [original_overlaps.get(pair, 0) for pair in qk_head_overlaps[h]]
                head_delta_mean = np.mean(np.array(head_delta_vals) - np.array(head_orig_vals))
                mean_deltas.append(head_delta_mean)
        
        if len(entropies) > 3:
            corr_ent_delta = np.corrcoef(entropies, mean_deltas)[0, 1]
            log_time(f"  Entropy↔Δ(QK)相关性: {corr_ent_delta:.3f}")
            layer_result["corr_entropy_delta"] = float(corr_ent_delta)
            layer_result["mean_entropy"] = float(np.mean(entropies))
        
        results[f"L{li}"] = layer_result
    
    # ===== 5. 综合总结 =====
    log_time(f"\n{'='*80}")
    log_time(f"核心总结: Attention Pattern vs 子空间翻译")
    log_time(f"{'='*80}")
    
    for li in target_layers:
        if f"L{li}" not in results:
            continue
        r = results[f"L{li}"]
        
        high_qk = r.get("high_qk_overlap", 0)
        low_qk = r.get("low_qk_overlap", 0)
        corr = r.get("corr_attn_qk", 0)
        
        log_time(f"  L{li:2d}: high_attn QK={high_qk:.3f}, low_attn QK={low_qk:.3f}, "
                 f"diff={high_qk-low_qk:+.3f}, corr={corr:.3f}")
    
    # ===== 6. 保存 =====
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
    
    out_path = OUTPUT_DIR / f"exp2b_attn_causal_{model_info.name}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(convert(results), f, indent=2, ensure_ascii=False)
    log_time(f"结果已保存到 {out_path}")
    
    release_model(model)
    log_time(f"总耗时: {(time.time()-t_start)/60:.1f}分钟")
    log_time("Done!")


if __name__ == "__main__":
    main()
