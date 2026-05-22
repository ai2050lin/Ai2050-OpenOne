"""
Subspace Topology Phase 2a: 注意力子空间翻译分析
=================================================

核心问题: attention heads如何在正交的位置子空间之间"翻译"信息?

实验设计:
1. 收集每个token位置的语义子空间 (debiased PCA, ~15维)
2. 对每个子空间应用 W_q, W_k, W_v 投影
3. 测量投影后的子空间对齐度变化 (Δ_alignment)
4. 分析共享核心经过 attention 后的能量保持率
5. 逐 head 分析: 不同 head 的"翻译"模式

关键指标:
- Δ_alignment_Q = alignment(W_q@S_i, W_q@S_j) - alignment(S_i, S_j)
  > 0 → Q投影"对齐"了子空间 (翻译)
  ≈ 0 → Q投影保持正交
  < 0 → Q投影进一步分离
- shared_core_preservation: 共享核心经过Q/K/V后的能量保持比
- per_head_specialization: 不同head的翻译模式差异

Run:
  python tests/glm5/subspace_topology_phase2a_attn_translation.py --model qwen3
  python tests/glm5/subspace_topology_phase2a_attn_translation.py --model glm4
  python tests/glm5/subspace_topology_phase2a_attn_translation.py --model deepseek7b
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

# ===== 模型加载 (BF16 + device_map="auto", 兼容GLM4/DS7B) =====
def load_model_bf16(model_name: str):
    """BF16 + device_map="auto" 加载, 兼容所有模型"""
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
    
    # DS7B使用sliding window attention, 需要eager模式
    attn_impl = "eager" if model_name == "deepseek7b" else "sdpa"
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation=attn_impl,  # SDPA: 内存高效, DS7B用eager
    )
    model.eval()
    
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    
    if hasattr(model, 'hf_device_map'):
        dmap = model.hf_device_map
        gpu_c = sum(1 for v in dmap.values() if 'cuda' in str(v))
        cpu_c = sum(1 for v in dmap.values() if 'cpu' in str(v))
        print(f"[{datetime.now():%H:%M:%S}] {model_name} loaded: GPU={gpu_c} comp, CPU={cpu_c} comp, "
              f"GPU mem={gpu_mem:.2f}GB, {time.time()-t0:.1f}s")
    else:
        print(f"[{datetime.now():%H:%M:%S}] {model_name} loaded: device={device}, GPU={gpu_mem:.2f}GB, "
              f"{time.time()-t0:.1f}s")
    
    return model, tokenizer, device


# ===== 句子集 (30句, 覆盖多样语义) =====
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


def log_time(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}")


def get_debiased_subspace(acts, n_dims=15):
    """获取去偏置后的语义子空间基"""
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
    V_semantic = Vt[1:1+n_return, :].T  # [d_model, n_dims], 跳过PC1(偏置)
    return V_semantic, S


def subspace_overlap(V1, V2, k=None):
    """计算子空间重叠度: V1在V2上的投影能量比"""
    if k:
        V1 = V1[:, :k]
    Q1 = np.linalg.qr(V1)[0]
    Q2 = np.linalg.qr(V2)[0]
    P2 = Q2 @ Q2.T
    proj = Q1.T @ P2 @ Q1
    return float(np.mean(np.diag(proj)))


def project_subspace(V_subspace, W_proj):
    """
    将子空间基通过线性投影矩阵映射
    
    Args:
        V_subspace: [d_model, k] 子空间基
        W_proj: [d_out, d_model] 投影矩阵
    
    Returns:
        V_projected: [d_out, k] 投影后的子空间基
    """
    return W_proj @ V_subspace  # [d_out, k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3")
    args = parser.parse_args()
    
    t_start = time.time()
    
    # ===== 1. 加载模型 =====
    model, tokenizer, device = load_model_bf16(args.model)
    from model_utils import get_layers, get_model_info, release_model, get_layer_weights, get_W_U
    model_info = get_model_info(model, args.model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)
    
    log_time(f"模型: {model_info.name}, {n_layers}层, d_model={d_model}, class={model_info.model_class}")
    
    # ===== 2. 收集残差流激活 =====
    log_time(f"收集 {len(SENTENCES)} 个句子的残差流激活...")
    
    max_len = 0
    tokenized = []
    for sent in SENTENCES:
        toks = tokenizer(sent, return_tensors="pt")
        seq_len = toks.input_ids.shape[1]
        tokenized.append((sent, toks, seq_len))
        max_len = max(max_len, seq_len)
    
    # 按位置和层存储
    pos_layer_acts = {pos: {li: [] for li in range(n_layers)} for pos in range(max_len)}
    
    for si, (sent, toks, seq_len) in enumerate(tokenized):
        input_device = next(model.parameters()).device
        toks = toks.to(input_device)
        embed_layer = model.get_input_embeddings()
        inputs_embeds = embed_layer(toks.input_ids).detach().clone().to(model.dtype)
        position_ids = torch.arange(seq_len, device=input_device).unsqueeze(0)
        
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
            except Exception as e:
                log_time(f"  句{si} 前向失败: {e}")
        
        for h in hooks:
            h.remove()
        
        for li in range(n_layers):
            key = f"L{li}"
            if key in captured:
                acts = captured[key][0, :, :].cpu().numpy()
                for pos in range(seq_len):
                    pos_layer_acts[pos][li].append(acts[pos, :])
        
        del captured
        if (si + 1) % 10 == 0:
            log_time(f"  已处理 {si+1}/{len(SENTENCES)} 句")
    
    log_time(f"激活收集完成")
    
    # ===== 3. 提取权重矩阵 =====
    log_time(f"提取各层的 W_q, W_k, W_v, W_o 权重...")
    
    # 采样层
    target_layers = sorted(set(
        [0, 1] + 
        list(range(0, n_layers, max(1, n_layers//8))) + 
        [n_layers//2, n_layers-2, n_layers-1]
    ))
    target_layers = sorted(set([l for l in target_layers if l < n_layers]))
    
    def safe_get_weight(weight_tensor):
        """安全获取权重, 处理meta device"""
        if weight_tensor.is_meta:
            return None
        return weight_tensor.detach().cpu().float().numpy()
    
    layer_weights = {}
    skipped_layers = []
    for li in target_layers:
        try:
            sa = layers[li].self_attn
            W_q = safe_get_weight(sa.q_proj.weight)
            W_k = safe_get_weight(sa.k_proj.weight)
            W_v = safe_get_weight(sa.v_proj.weight)
            W_o = safe_get_weight(sa.o_proj.weight)
            
            if any(w is None for w in [W_q, W_k, W_v, W_o]):
                log_time(f"  L{li}: 跳过 (权重在meta device)")
                skipped_layers.append(li)
                continue
            
            lw = get_layer_weights(layers[li], d_model, model_info.mlp_type)
            layer_weights[li] = lw
            log_time(f"  L{li}: W_q={lw.W_q.shape}, W_k={lw.W_k.shape}, W_v={lw.W_v.shape}, W_o={lw.W_o.shape}")
        except Exception as e:
            log_time(f"  L{li}: 跳过 (错误: {e})")
            skipped_layers.append(li)
    
    target_layers = [l for l in target_layers if l not in skipped_layers]
    log_time(f"可分析层: {target_layers} ({len(target_layers)}/{len(target_layers)+len(skipped_layers)}")
    
    # ===== 4. 获取各位置的语义子空间 =====
    log_time(f"计算各位置的语义子空间 (debiased, 15维)...")
    
    n_dims = 15
    valid_positions = [p for p in range(max_len) 
                       if len(pos_layer_acts[p][0]) >= 10]
    log_time(f"有效位置: {valid_positions}")
    
    # 获取每个位置在每层的语义子空间
    pos_subspaces = {}  # {li: {pos: V_semantic}}
    for li in target_layers:
        pos_subspaces[li] = {}
        for pos in valid_positions:
            acts = np.array(pos_layer_acts[pos][li])
            if len(acts) < 5:
                continue
            V_sem, _ = get_debiased_subspace(acts, n_dims=n_dims)
            pos_subspaces[li][pos] = V_sem  # [d_model, 15]
    
    # ===== 5. 核心分析: 子空间翻译 =====
    log_time(f"核心分析: W_q/W_k/W_v 的子空间翻译效应...")
    
    results = {}
    
    for li in target_layers:
        log_time(f"\n--- Layer {li} ---")
        lw = layer_weights[li]
        W_q = lw.W_q  # [d_model, d_model]
        W_k = lw.W_k
        W_v = lw.W_v
        W_o = lw.W_o
        
        positions_with_data = sorted(pos_subspaces[li].keys())
        if len(positions_with_data) < 2:
            log_time(f"  位置数不足, 跳过")
            continue
        
        layer_result = {"layer": li}
        
        # 5a. 原始子空间对齐度 (基线)
        original_overlaps = {}
        for i, p1 in enumerate(positions_with_data):
            for j, p2 in enumerate(positions_with_data):
                if j <= i:
                    continue
                ov = subspace_overlap(pos_subspaces[li][p1], pos_subspaces[li][p2], k=n_dims)
                original_overlaps[f"{p1}-{p2}"] = ov
        
        avg_original = np.mean(list(original_overlaps.values()))
        log_time(f"  原始子空间平均重叠度: {avg_original:.4f}")
        
        # 5b. Q投影后的子空间对齐度
        q_subspaces = {}
        for pos in positions_with_data:
            V_proj = project_subspace(pos_subspaces[li][pos], W_q)  # [d_model, 15]
            q_subspaces[pos] = V_proj
        
        q_overlaps = {}
        for i, p1 in enumerate(positions_with_data):
            for j, p2 in enumerate(positions_with_data):
                if j <= i:
                    continue
                ov = subspace_overlap(q_subspaces[p1], q_subspaces[p2], k=n_dims)
                q_overlaps[f"{p1}-{p2}"] = ov
        
        avg_q = np.mean(list(q_overlaps.values()))
        delta_q = avg_q - avg_original
        log_time(f"  Q投影后平均重叠度: {avg_q:.4f} (Δ={delta_q:+.4f})")
        
        # 5c. K投影后的子空间对齐度
        k_subspaces = {}
        for pos in positions_with_data:
            V_proj = project_subspace(pos_subspaces[li][pos], W_k)
            k_subspaces[pos] = V_proj
        
        k_overlaps = {}
        for i, p1 in enumerate(positions_with_data):
            for j, p2 in enumerate(positions_with_data):
                if j <= i:
                    continue
                ov = subspace_overlap(k_subspaces[p1], k_subspaces[p2], k=n_dims)
                k_overlaps[f"{p1}-{p2}"] = ov
        
        avg_k = np.mean(list(k_overlaps.values()))
        delta_k = avg_k - avg_original
        log_time(f"  K投影后平均重叠度: {avg_k:.4f} (Δ={delta_k:+.4f})")
        
        # 5d. V投影后的子空间对齐度
        v_subspaces = {}
        for pos in positions_with_data:
            V_proj = project_subspace(pos_subspaces[li][pos], W_v)
            v_subspaces[pos] = V_proj
        
        v_overlaps = {}
        for i, p1 in enumerate(positions_with_data):
            for j, p2 in enumerate(positions_with_data):
                if j <= i:
                    continue
                ov = subspace_overlap(v_subspaces[p1], v_subspaces[p2], k=n_dims)
                v_overlaps[f"{p1}-{p2}"] = ov
        
        avg_v = np.mean(list(v_overlaps.values()))
        delta_v = avg_v - avg_original
        log_time(f"  V投影后平均重叠度: {avg_v:.4f} (Δ={delta_v:+.4f})")
        
        # 5e. V→O投影后的对齐度 (完整attention输出路径)
        # GQA处理: W_v [n_kv*hd, d] → W_o [d, n_heads*hd]
        # 需要将V输出扩展到n_heads再乘W_o
        n_heads_q = W_q.shape[0]  # n_heads * head_dim (Q的输出维度)
        n_kv_dim = W_v.shape[0]    # n_kv_heads * head_dim
        
        # 从模型config获取准确的head信息
        try:
            nh_conf = model.config.num_attention_heads
        except:
            nh_conf = n_heads_q // d_model if n_heads_q >= d_model else n_heads_q
        
        # head_dim推断
        if n_heads_q == d_model:
            hd_conf = d_model // nh_conf
        else:
            hd_conf = n_heads_q // nh_conf
        
        # 构建组合W_ov矩阵 (d_model → d_model)
        if n_kv_dim == n_heads_q:
            # 无GQA: 直接乘
            W_ov = W_o @ W_v  # [d_model, d_model]
        else:
            # GQA: 需要扩展V输出
            n_kv_h = n_kv_dim // hd_conf
            n_rep = nh_conf // n_kv_h  # 每个KV head被多少Q head共享
            
            # 扩展W_v: [n_kv*hd, d_model] → [n_heads*hd, d_model]
            W_v_3d = W_v.reshape(n_kv_h, hd_conf, d_model)
            W_v_expanded = np.repeat(W_v_3d, n_rep, axis=0).reshape(nh_conf * hd_conf, d_model)
            W_ov = W_o @ W_v_expanded  # [d_model, d_model]
            
            log_time(f"  GQA: n_heads={nh_conf}, n_kv_heads={n_kv_h}, head_dim={hd_conf}, n_rep={n_rep}")
        
        vo_overlaps = {}
        for i, p1 in enumerate(positions_with_data):
            for j, p2 in enumerate(positions_with_data):
                if j <= i:
                    continue
                # V→O: 直接用组合W_ov矩阵
                V_p1_vo = project_subspace(pos_subspaces[li][p1], W_ov)
                V_p2_vo = project_subspace(pos_subspaces[li][p2], W_ov)
                ov = subspace_overlap(V_p1_vo, V_p2_vo, k=n_dims)
                vo_overlaps[f"{p1}-{p2}"] = ov
        
        avg_vo = np.mean(list(vo_overlaps.values()))
        delta_vo = avg_vo - avg_original
        log_time(f"  V→O投影后平均重叠度: {avg_vo:.4f} (Δ={delta_vo:+.4f})")
        
        # 5f. 共享核心信息流
        # 提取共享核心: 所有位置的中心化数据的PCA
        all_acts_centered = []
        for pos in positions_with_data:
            acts = np.array(pos_layer_acts[pos][li])
            mean = acts.mean(axis=0)
            centered = acts - mean
            # 去除PC1 (偏置)
            try:
                _, S_tmp, Vt_tmp = np.linalg.svd(centered, full_matrices=False)
                centered_debiased = centered - centered @ Vt_tmp[0:1, :].T @ Vt_tmp[0:1, :]
            except:
                centered_debiased = centered
            all_acts_centered.append(centered_debiased)
        
        all_acts_centered = np.vstack(all_acts_centered)
        try:
            _, S_shared, Vt_shared = np.linalg.svd(all_acts_centered, full_matrices=False)
        except:
            from sklearn.decomposition import TruncatedSVD
            svd_obj = TruncatedSVD(n_components=min(50, all_acts_centered.shape[0]-1, all_acts_centered.shape[1]-1), random_state=42)
            svd_obj.fit(all_acts_centered.astype(np.float32))
            Vt_shared = svd_obj.components_
            S_shared = svd_obj.singular_values_
        
        # 共享核心 = 前14个主成分 (跳过PC1偏置)
        n_core = min(14, Vt_shared.shape[0] - 1)
        V_core = Vt_shared[1:1+n_core, :].T  # [d_model, 14]
        
        # 共享核心经过Q/K/V后的能量保持率
        core_norm_sq = np.sum(V_core ** 2)
        
        Q_core = W_q @ V_core  # [n_heads*hd, 14]
        K_core = W_k @ V_core  # [n_kv*hd, 14]
        V_core_proj = W_v @ V_core  # [n_kv*hd, 14]
        
        q_preservation = float(np.sum(Q_core ** 2) / core_norm_sq)
        k_preservation = float(np.sum(K_core ** 2) / core_norm_sq)
        v_preservation = float(np.sum(V_core_proj ** 2) / core_norm_sq)
        
        # V→O路径: 使用组合W_ov
        VO_core = W_ov @ V_core  # [d_model, 14]
        vo_preservation = float(np.sum(VO_core ** 2) / core_norm_sq)
        
        log_time(f"  共享核心能量保持: Q={q_preservation:.3f}, K={k_preservation:.3f}, "
                 f"V={v_preservation:.3f}, V→O={vo_preservation:.3f}")
        
        # 5g. 逐 head 分析
        # 从模型config获取准确的head信息
        try:
            n_heads = model.config.num_attention_heads
        except:
            n_heads = W_q.shape[0] // max(1, W_q.shape[0] // d_model)
        
        # head_dim: 从W_q的输出维度和n_heads推断
        if W_q.shape[0] == d_model and d_model % n_heads == 0:
            head_dim = d_model // n_heads
        else:
            head_dim = W_q.shape[0] // n_heads
        
        # W_q可能shape = [n_heads * head_dim, d_model]
        q_out_dim = W_q.shape[0]  # n_heads * head_dim
        
        log_time(f"  n_heads={n_heads}, head_dim={head_dim}, q_out_dim={q_out_dim}")
        
        # 逐head的Q投影对齐度
        per_head_delta_q = []
        per_head_delta_k = []
        
        # n_kv_heads for GQA
        n_kv_heads = W_k.shape[0] // head_dim
        n_rep = n_heads // n_kv_heads if n_kv_heads < n_heads else 1
        
        for h in range(min(n_heads, 32)):  # 最多分析32个head
            # 提取单head的W_q: [head_dim, d_model]
            if q_out_dim == n_heads * head_dim:
                W_q_h = W_q[h*head_dim:(h+1)*head_dim, :]
            else:
                # fallback: W_q可能shape=[d_model, d_model]
                W_q_h = W_q[h*head_dim:(h+1)*head_dim, :]
            
            # W_k可能使用GQA
            if n_kv_heads < n_heads:
                kv_h = h // n_rep
                W_k_h = W_k[kv_h*head_dim:(kv_h+1)*head_dim, :]
            else:
                W_k_h = W_k[h*head_dim:(h+1)*head_dim, :]
            
            # 单head的Q子空间对齐
            q_h_subspaces = {}
            for pos in positions_with_data:
                V_proj = project_subspace(pos_subspaces[li][pos], W_q_h)  # [head_dim, 15]
                q_h_subspaces[pos] = V_proj
            
            # 只在head_dim >= n_dims时计算
            if head_dim >= n_dims:
                q_h_overlaps = []
                for i, p1 in enumerate(positions_with_data):
                    for j, p2 in enumerate(positions_with_data):
                        if j <= i:
                            continue
                        ov = subspace_overlap(q_h_subspaces[p1], q_h_subspaces[p2], k=n_dims)
                        q_h_overlaps.append(ov)
                avg_q_h = np.mean(q_h_overlaps) if q_h_overlaps else 0
                per_head_delta_q.append(avg_q_h - avg_original)
            
            # 单head的K子空间对齐
            k_h_subspaces = {}
            for pos in positions_with_data:
                V_proj = project_subspace(pos_subspaces[li][pos], W_k_h)
                k_h_subspaces[pos] = V_proj
            
            if head_dim >= n_dims:
                k_h_overlaps = []
                for i, p1 in enumerate(positions_with_data):
                    for j, p2 in enumerate(positions_with_data):
                        if j <= i:
                            continue
                        ov = subspace_overlap(k_h_subspaces[p1], k_h_subspaces[p2], k=n_dims)
                        k_h_overlaps.append(ov)
                avg_k_h = np.mean(k_h_overlaps) if k_h_overlaps else 0
                per_head_delta_k.append(avg_k_h - avg_original)
        
        # head统计
        if per_head_delta_q:
            log_time(f"  Head Q投影: mean_Δ={np.mean(per_head_delta_q):+.4f}, "
                     f"max_Δ={np.max(per_head_delta_q):+.4f}, min_Δ={np.min(per_head_delta_q):+.4f}")
        if per_head_delta_k:
            log_time(f"  Head K投影: mean_Δ={np.mean(per_head_delta_k):+.4f}, "
                     f"max_Δ={np.max(per_head_delta_k):+.4f}, min_Δ={np.min(per_head_delta_k):+.4f}")
        
        # 5h. Q-K跨位置相似度 (attention的核心)
        # Q和K在不同维度空间(GQA), 所以必须逐head计算
        # 对于每个head h: Q_h(pos_i) @ K_h(pos_j)^T 的子空间结构
        qk_cross_overlaps_per_head = []
        
        for h in range(min(n_heads, 32)):
            # Q head h: W_q_h [head_dim, d_model]
            W_q_h = W_q[h*head_dim:(h+1)*head_dim, :]
            # K head (GQA): W_k_h [head_dim, d_model]
            if n_kv_heads < n_heads:
                kv_h = h // n_rep
                W_k_h = W_k[kv_h*head_dim:(kv_h+1)*head_dim, :]
            else:
                W_k_h = W_k[h*head_dim:(h+1)*head_dim, :]
            
            # Q_h和K_h都在head_dim空间, 可以比较
            qk_overlaps = []
            for i, p1 in enumerate(positions_with_data):
                Q_h_p1 = project_subspace(pos_subspaces[li][p1], W_q_h)  # [head_dim, 15]
                for j, p2 in enumerate(positions_with_data):
                    if j <= i:
                        continue
                    K_h_p2 = project_subspace(pos_subspaces[li][p2], W_k_h)  # [head_dim, 15]
                    # 只在head_dim >= n_dims时计算
                    if head_dim >= n_dims:
                        ov = subspace_overlap(Q_h_p1, K_h_p2, k=n_dims)
                        qk_overlaps.append(ov)
            
            if qk_overlaps:
                qk_cross_overlaps_per_head.append(np.mean(qk_overlaps))
        
        avg_qk_cross = float(np.mean(qk_cross_overlaps_per_head)) if qk_cross_overlaps_per_head else 0
        log_time(f"  Q(pos_i)-K(pos_j)跨位置重叠度(逐head平均): {avg_qk_cross:.4f}")
        
        # 5i. 相邻 vs 远距对比
        adj_orig, dist_orig = [], []
        adj_q, dist_q = [], []
        adj_k, dist_k = [], []
        adj_vo, dist_vo = [], []
        
        for key in original_overlaps:
            p1, p2 = map(int, key.split('-'))
            if abs(p2 - p1) <= 1:
                adj_orig.append(original_overlaps[key])
                adj_q.append(q_overlaps[key])
                adj_k.append(k_overlaps[key])
                adj_vo.append(vo_overlaps[key])
            elif abs(p2 - p1) >= 3:
                dist_orig.append(original_overlaps[key])
                dist_q.append(q_overlaps[key])
                dist_k.append(k_overlaps[key])
                dist_vo.append(vo_overlaps[key])
        
        layer_result.update({
            "original_overlap": {
                "mean": float(avg_original),
                "adjacent_mean": float(np.mean(adj_orig)) if adj_orig else 0,
                "distant_mean": float(np.mean(dist_orig)) if dist_orig else 0,
            },
            "q_overlap": {
                "mean": float(avg_q),
                "delta": float(delta_q),
                "adjacent_mean": float(np.mean(adj_q)) if adj_q else 0,
                "distant_mean": float(np.mean(dist_q)) if dist_q else 0,
            },
            "k_overlap": {
                "mean": float(avg_k),
                "delta": float(delta_k),
                "adjacent_mean": float(np.mean(adj_k)) if adj_k else 0,
                "distant_mean": float(np.mean(dist_k)) if dist_k else 0,
            },
            "v_overlap": {
                "mean": float(avg_v),
                "delta": float(delta_v),
            },
            "vo_overlap": {
                "mean": float(avg_vo),
                "delta": float(delta_vo),
                "adjacent_mean": float(np.mean(adj_vo)) if adj_vo else 0,
                "distant_mean": float(np.mean(dist_vo)) if dist_vo else 0,
            },
            "shared_core_flow": {
                "q_preservation": float(q_preservation),
                "k_preservation": float(k_preservation),
                "v_preservation": float(v_preservation),
                "vo_preservation": float(vo_preservation),
            },
            "qk_cross_overlap": float(avg_qk_cross),
            "per_head": {
                "n_heads": int(n_heads),
                "head_dim": int(head_dim),
                "delta_q_mean": float(np.mean(per_head_delta_q)) if per_head_delta_q else 0,
                "delta_q_max": float(np.max(per_head_delta_q)) if per_head_delta_q else 0,
                "delta_q_min": float(np.min(per_head_delta_q)) if per_head_delta_q else 0,
                "delta_k_mean": float(np.mean(per_head_delta_k)) if per_head_delta_k else 0,
                "delta_k_max": float(np.max(per_head_delta_k)) if per_head_delta_k else 0,
                "delta_k_min": float(np.min(per_head_delta_k)) if per_head_delta_k else 0,
            },
            "positions_used": positions_with_data,
        })
        
        results[f"L{li}"] = layer_result
    
    # ===== 6. 综合总结 =====
    log_time(f"\n{'='*80}")
    log_time(f"核心总结: 注意力子空间翻译效应")
    log_time(f"{'='*80}")
    
    for li in target_layers:
        r = results[f"L{li}"]
        orig = r["original_overlap"]["mean"]
        dq = r["q_overlap"]["delta"]
        dk = r["k_overlap"]["delta"]
        dv = r["v_overlap"]["delta"]
        dvo = r["vo_overlap"]["delta"]
        core_q = r["shared_core_flow"]["q_preservation"]
        core_k = r["shared_core_flow"]["k_preservation"]
        core_v = r["shared_core_flow"]["v_preservation"]
        core_vo = r["shared_core_flow"]["vo_preservation"]
        
        log_time(f"  L{li:2d}: orig={orig:.3f} | ΔQ={dq:+.3f} ΔK={dk:+.3f} ΔV={dv:+.3f} ΔVO={dvo:+.3f} | "
                 f"core: Q={core_q:.2f} K={core_k:.2f} V={core_v:.2f} VO={core_vo:.2f}")
    
    # 翻译模式分类
    log_time(f"\n翻译模式分析:")
    for li in target_layers:
        r = results[f"L{li}"]
        dq = r["q_overlap"]["delta"]
        dk = r["k_overlap"]["delta"]
        dvo = r["vo_overlap"]["delta"]
        
        if dvo > 0.05:
            mode = "ALIGN (对齐翻译)"
        elif dvo < -0.05:
            mode = "SEPARATE (进一步分离)"
        else:
            mode = "PRESERVE (保持正交)"
        
        log_time(f"  L{li:2d}: VO_Δ={dvo:+.3f} → {mode}")
    
    # ===== 7. 保存结果 =====
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
    
    OUTPUT_DIR = Path("results/subspace_topology")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"exp2a_attn_translation_{model_info.name}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(convert(results), f, indent=2, ensure_ascii=False)
    log_time(f"结果已保存到 {out_path}")
    
    # ===== 8. 释放模型 =====
    release_model(model)
    
    t_total = time.time() - t_start
    log_time(f"总耗时: {t_total/60:.1f}分钟")
    log_time("Done!")


if __name__ == "__main__":
    main()
