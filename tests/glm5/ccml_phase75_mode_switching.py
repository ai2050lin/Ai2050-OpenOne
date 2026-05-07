"""
Phase 75: 模式切换 + 注意力路由动力学
========================================

核心转变: 从"状态几何"到"状态转化+信息路由"

实验A: 模式切换 — 不同任务指令如何改变残差流轨迹?
  - 5种模式: chat, translate, summarize, reason, code
  - 测量: 轨迹分离度, 模式切换层, 轨迹聚类

实验B: 注意力路由拓扑 — "Translate:"改变了哪些heads?
  - 对比不同模式下的attention pattern
  - 找出: 哪些heads的routing发生显著变化
  - 定位: 模式切换的层

实验C: 生成轨迹稳定性 — 推理/翻译/聊天是否形成不同trajectory?
  - 记录token-by-token生成的hidden state
  - 分析轨迹的几何结构

Usage:
  python ccml_phase75_mode_switching.py --exp a   # 模式切换
  python ccml_phase75_mode_switching.py --exp b   # 注意力路由
  python ccml_phase75_mode_switching.py --exp c   # 生成轨迹
"""

import torch
import numpy as np
import argparse
from pathlib import Path
from transformer_lens import HookedTransformer

# ============================================================
# 辅助函数
# ============================================================

def get_model():
    """加载GPT-2 Small, fp32, CPU"""
    model = HookedTransformer.from_pretrained(
        "gpt2-small",
        center_unembed=False,
        center_writing_weights=False,
        fold_ln=False,
        device="cpu",
    )
    model.eval()
    return model

def get_mode_prompts():
    """5种任务模式的prompts"""
    base_text = "The weather was beautiful and the children played in the park"
    
    prompts = {
        "chat": f"Hello! {base_text}. How are you today?",
        "translate": f"Translate to French: {base_text}",
        "summarize": f"Summarize: {base_text}",
        "reason": f"Think step by step: Why did the children play in the park?",
        "code": f"Write Python code: A function that checks if weather is beautiful",
    }
    return prompts

def get_diverse_prompts():
    """多组不同内容的prompts，用于验证模式效应的泛化性"""
    texts = [
        "The scientist discovered a new species of butterfly in the rainforest",
        "The economy grew by three percent last quarter according to reports",
        "The artist painted a stunning portrait of the mountain landscape",
    ]
    
    prompt_sets = []
    for text in texts:
        prompts = {
            "chat": f"Hello! {text}. What do you think?",
            "translate": f"Translate to French: {text}",
            "summarize": f"Summarize: {text}",
            "reason": f"Think step by step: What are the implications of this?",
            "code": f"Write Python code: Process this information",
        }
        prompt_sets.append(prompts)
    return prompt_sets

# ============================================================
# 实验A: 模式切换 — 残差流轨迹分离
# ============================================================

def exp_a_mode_switching(model):
    """不同任务模式的残差流轨迹分离度"""
    print("=" * 70)
    print("实验A: 模式切换 — 残差流轨迹分离度")
    print("=" * 70)
    
    prompts = get_mode_prompts()
    mode_names = list(prompts.keys())
    
    # 收集各模式的残差流轨迹
    mode_trajectories = {}
    
    with torch.no_grad():
        for mode, prompt in prompts.items():
            tokens = model.to_tokens(prompt)
            _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
            
            # 收集每层最后一个token的残差流
            trajectory = []
            for layer in range(model.cfg.n_layers):
                h = cache["resid_post", layer][-1]  # last token, last layer
                trajectory.append(h.detach().numpy())
            
            # 也收集resid_pre (层输入)
            mode_trajectories[mode] = np.array(trajectory)
            print(f"  {mode}: prompt length={tokens.shape[-1]}, trajectory shape={np.array(trajectory).shape}")
    
    # ---- 分析1: 模式间轨迹的余弦距离 ----
    print("\n--- 模式间轨迹余弦距离 (逐层) ---")
    print(f"{'Layer':>6}", end="")
    for m in mode_names:
        print(f"  {m:>10}", end="")
    print()
    
    # 以chat模式为参考
    ref = mode_trajectories["chat"]
    
    for layer in range(model.cfg.n_layers):
        print(f"L{layer:>4}", end="")
        h_ref = ref[layer]
        h_ref_norm = h_ref / (np.linalg.norm(h_ref) + 1e-10)
        
        for mode in mode_names:
            h = mode_trajectories[mode][layer]
            h_norm = h / (np.linalg.norm(h) + 1e-10)
            cos_sim = np.dot(h_ref_norm, h_norm)
            cos_dist = 1 - cos_sim
            print(f"  {cos_dist:>10.4f}", end="")
        print()
    
    # ---- 分析2: 模式轨迹的成对距离矩阵 ----
    print("\n--- 成对余弦距离矩阵 (Layer 6) ---")
    layer_idx = 6
    print(f"{'':>12}", end="")
    for m in mode_names:
        print(f"  {m:>10}", end="")
    print()
    
    for m1 in mode_names:
        print(f"{m1:>12}", end="")
        h1 = mode_trajectories[m1][layer_idx]
        h1_norm = h1 / (np.linalg.norm(h1) + 1e-10)
        for m2 in mode_names:
            h2 = mode_trajectories[m2][layer_idx]
            h2_norm = h2 / (np.linalg.norm(h2) + 1e-10)
            cos_dist = 1 - np.dot(h1_norm, h2_norm)
            print(f"  {cos_dist:>10.4f}", end="")
        print()
    
    # ---- 分析3: 轨迹"模式签名" — 每层不同模式的方差 ----
    print("\n--- 模式间方差 (5个模式在每层的分散度) ---")
    print(f"{'Layer':>6}  {'Var':>10}  {'MaxCosDist':>12}  {'RandBaseline':>12}  {'Z-score':>8}")
    
    for layer in range(model.cfg.n_layers):
        # 5个模式的h
        hs = np.array([mode_trajectories[m][layer] for m in mode_names])
        mean_h = hs.mean(axis=0)
        
        # 余弦距离方差
        cos_dists = []
        for h in hs:
            h_norm = h / (np.linalg.norm(h) + 1e-10)
            m_norm = mean_h / (np.linalg.norm(mean_h) + 1e-10)
            cos_dists.append(1 - np.dot(h_norm, m_norm))
        var = np.var(cos_dists)
        max_dist = max(cos_dists)
        
        # 随机基线: 5个随机高斯向量
        rand_dists = []
        for _ in range(100):
            rand_hs = np.random.randn(5, 768)
            rand_mean = rand_hs.mean(axis=0)
            for rh in rand_hs:
                rh_norm = rh / (np.linalg.norm(rh) + 1e-10)
                rm_norm = rand_mean / (np.linalg.norm(rand_mean) + 1e-10)
                rand_dists.append(1 - np.dot(rh_norm, rm_norm))
        rand_mean_dist = np.mean(rand_dists)
        rand_std = np.std(rand_dists)
        z_score = (max_dist - rand_mean_dist) / (rand_std + 1e-10)
        
        print(f"L{layer:>4}  {var:>10.6f}  {max_dist:>12.6f}  {rand_mean_dist:>12.6f}  {z_score:>8.1f}σ")
    
    # ---- 分析4: 多组prompt验证泛化性 ----
    print("\n--- 多组prompt的模式分离度 (Layer 6) ---")
    diverse_sets = get_diverse_prompts()
    
    layer_idx = 6
    all_mode_h = {m: [] for m in mode_names}
    
    with torch.no_grad():
        for pset in diverse_sets:
            for mode, prompt in pset.items():
                tokens = model.to_tokens(prompt)
                _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
                h = cache["resid_post", layer_idx][-1].detach().numpy()
                all_mode_h[mode].append(h)
    
    # 计算模式间的余弦距离
    print(f"{'Pair':>30}  {'CosDist':>10}  {'RandMean':>10}  {'Z':>6}")
    for i, m1 in enumerate(mode_names):
        for m2 in mode_names[i+1:]:
            dists = []
            for h1, h2 in zip(all_mode_h[m1], all_mode_h[m2]):
                h1n = h1 / (np.linalg.norm(h1) + 1e-10)
                h2n = h2 / (np.linalg.norm(h2) + 1e-10)
                dists.append(1 - np.dot(h1n, h2n))
            mean_dist = np.mean(dists)
            
            # 随机基线
            rand_dists = []
            for _ in range(200):
                rh1 = np.random.randn(768)
                rh2 = np.random.randn(768)
                rh1n = rh1 / np.linalg.norm(rh1)
                rh2n = rh2 / np.linalg.norm(rh2)
                rand_dists.append(1 - np.dot(rh1n, rh2n))
            rand_mean = np.mean(rand_dists)
            rand_std = np.std(rand_dists)
            z = (mean_dist - rand_mean) / (rand_std + 1e-10)
            
            print(f"{m1:>15} vs {m2:<12}  {mean_dist:>10.4f}  {rand_mean:>10.4f}  {z:>5.1f}σ")
    
    # ---- 分析5: 模式内一致性 vs 模式间分离 ----
    print("\n--- 模式内一致性(3个不同prompt) vs 模式间分离 ---")
    
    # 用3组prompt，计算同模式不同prompt的距离 vs 不同模式的距离
    within_dists = {m: [] for m in mode_names}
    between_dists = []
    
    for m in mode_names:
        hs = all_mode_h[m]
        for i in range(len(hs)):
            for j in range(i+1, len(hs)):
                h1n = hs[i] / (np.linalg.norm(hs[i]) + 1e-10)
                h2n = hs[j] / (np.linalg.norm(hs[j]) + 1e-10)
                within_dists[m].append(1 - np.dot(h1n, h2n))
    
    for i, m1 in enumerate(mode_names):
        for m2 in mode_names[i+1:]:
            for h1 in all_mode_h[m1]:
                for h2 in all_mode_h[m2]:
                    h1n = h1 / (np.linalg.norm(h1) + 1e-10)
                    h2n = h2 / (np.linalg.norm(h2) + 1e-10)
                    between_dists.append(1 - np.dot(h1n, h2n))
    
    print(f"{'Mode':>12}  {'WithinDist':>12}")
    for m in mode_names:
        print(f"{m:>12}  {np.mean(within_dists[m]):>12.4f}")
    print(f"{'BETWEEN':>12}  {np.mean(between_dists):>12.4f}")
    print(f"\nRatio (between/within): {np.mean(between_dists) / np.mean([np.mean(within_dists[m]) for m in mode_names]):.3f}")
    
    # ---- 分析6: 模式特异方向 ----
    print("\n--- 模式特异方向 (PCA on 5 modes × 3 prompts) ---")
    
    all_hs = []
    labels = []
    for m in mode_names:
        for h in all_mode_h[m]:
            all_hs.append(h)
            labels.append(m)
    all_hs = np.array(all_hs)
    
    # PCA
    mean_h = all_hs.mean(axis=0)
    centered = all_hs - mean_h
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    
    # 各模式在PC1-5上的投影
    print(f"{'Mode':>12}", end="")
    for pc in range(5):
        print(f"  {'PC'+str(pc+1):>8}", end="")
    print()
    
    for m in mode_names:
        idx = [i for i, l in enumerate(labels) if l == m]
        proj = U[idx, :5]
        mean_proj = proj.mean(axis=0)
        print(f"{m:>12}", end="")
        for pc in range(5):
            print(f"  {mean_proj[pc]:>8.3f}", end="")
        print()
    
    # PC中模式可分性
    print(f"\n{'PC':>4}  {'Variance%':>10}  {'ModeSep(Sil)':>13}")
    from sklearn.metrics import silhouette_score
    
    for pc in range(min(10, U.shape[1])):
        var_pct = S[pc]**2 / (S**2).sum() * 100
        
        # 只用这一个PC做silhouette
        proj_1d = U[:, pc].reshape(-1, 1)
        try:
            sil = silhouette_score(proj_1d, labels)
        except:
            sil = 0
        print(f"PC{pc+1:>2}  {var_pct:>10.2f}%  {sil:>13.4f}")


# ============================================================
# 实验B: 注意力路由拓扑
# ============================================================

def exp_b_attention_routing(model):
    """不同模式下的注意力路由变化"""
    print("=" * 70)
    print("实验B: 注意力路由拓扑 — 'Translate:'改变了哪些heads?")
    print("=" * 70)
    
    prompts = get_mode_prompts()
    mode_names = list(prompts.keys())
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    
    # 收集各模式的attention patterns
    mode_attns = {}
    
    with torch.no_grad():
        for mode, prompt in prompts.items():
            tokens = model.to_tokens(prompt)
            _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
            
            attns = []
            for layer in range(n_layers):
                # pattern: [n_heads, seq_len, seq_len]
                pattern = cache["pattern", layer].detach().numpy()
                attns.append(pattern)
            
            mode_attns[mode] = attns
            print(f"  {mode}: collected attention for {n_layers} layers")
    
    # ---- 分析1: 各head的attention entropy变化 ----
    print("\n--- Attention Entropy变化 (chat vs 其他模式) ---")
    print("高entropy = 分散attention; 低entropy = 集中attention")
    
    chat_attns = mode_attns["chat"]
    
    # 计算每个head的attention entropy
    def attn_entropy(pattern):
        """pattern: [n_heads, seq_q, seq_k] → 每个head的平均entropy"""
        entropies = []
        for h in range(pattern.shape[0]):
            # 每个query position的entropy
            head_ent = 0
            for q in range(pattern.shape[1]):
                p = pattern[h, q] + 1e-10
                ent = -np.sum(p * np.log(p))
                head_ent += ent
            entropies.append(head_ent / pattern.shape[1])
        return np.array(entropies)
    
    chat_entropies = [attn_entropy(a) for a in chat_attns]
    
    print(f"\n{'Layer':>6}  {'Head':>6}", end="")
    for mode in mode_names[1:]:
        print(f"  {mode:>12}_Δent", end="")
    print()
    
    significant_heads = []
    
    for layer in range(n_layers):
        for head in range(n_heads):
            chat_ent = chat_entropies[layer][head]
            deltas = {}
            significant = False
            for mode in mode_names[1:]:
                mode_ent = attn_entropy(mode_attns[mode][layer])[head]
                delta = mode_ent - chat_ent
                deltas[mode] = delta
                if abs(delta) > 0.5:  # 阈值
                    significant = True
            
            if significant:
                print(f"L{layer:>4}  H{head:>4}", end="")
                for mode in mode_names[1:]:
                    print(f"  {deltas[mode]:>12.3f}", end="")
                print()
                significant_heads.append((layer, head, deltas))
    
    print(f"\n共 {len(significant_heads)} 个head在不同模式下entropy变化>0.5")
    
    # ---- 分析2: Attention分布的KL散度 ----
    print("\n--- Attention分布KL散度 (vs chat) ---")
    print("KL>0: 该模式attention偏离chat模式")
    
    # 对每个layer, 取各自最后一个query token的attention分布
    # 由于不同prompt长度不同，需要取各自最后一个query position
    def last_token_attn_kl(p1, p2):
        """p1, p2: [n_heads, seq_q, seq_k] → 各head的KL散度
        取各自最后一个query token的attention分布"""
        kls = []
        n_heads = p1.shape[0]
        last_q1 = p1.shape[1] - 1
        last_q2 = p2.shape[1] - 1
        min_k = min(p1.shape[2], p2.shape[2])
        
        for h in range(n_heads):
            q1 = p1[h, last_q1, :min_k] + 1e-10
            q2 = p2[h, last_q2, :min_k] + 1e-10
            kl = np.sum(q1 * np.log(q1 / q2))
            kls.append(kl)
        return np.array(kls)
    
    print(f"\n{'Layer':>6}", end="")
    for mode in mode_names[1:]:
        print(f"  {mode:>10}_maxKL", end="")
        print(f"  {mode:>10}_meanKL", end="")
    print()
    
    for layer in range(n_layers):
        print(f"L{layer:>4}", end="")
        for mode in mode_names[1:]:
            kls = last_token_attn_kl(chat_attns[layer], mode_attns[mode][layer])
            print(f"  {np.max(kls):>10.3f}", end="")
            print(f"  {np.mean(kls):>10.3f}", end="")
        print()
    
    # ---- 分析3: 哪些token被更多地/更少地attend ----
    print("\n--- Attention重定向: 哪些token被更多/更少attend ---")
    
    # 找translate模式的显著变化
    for mode in ["translate", "reason", "code"]:
        print(f"\n  Mode: {mode} (vs chat)")
        print(f"  {'Layer':>6}  {'Head':>6}  {'TopAttended_Change':>40}")
        
        for layer in [0, 3, 6, 9, 11]:
            for head in range(n_heads):
                chat_pat = chat_attns[layer][head]  # [seq_q, seq_k]
                mode_pat = mode_attns[mode][layer][head]
                
                # 各自最后一个query token的attention差异
                last_q_chat = chat_pat.shape[0] - 1
                last_q_mode = mode_pat.shape[0] - 1
                min_k = min(chat_pat.shape[1], mode_pat.shape[1])
                diff = mode_pat[last_q_mode, :min_k] - chat_pat[last_q_chat, :min_k]
                
                # 找变化最大的position
                top_pos = np.argmax(np.abs(diff))
                top_diff = diff[top_pos]
                
                if abs(top_diff) > 0.05:
                    print(f"  L{layer:>4}  H{head:>4}  pos{top_pos:>3}: {top_diff:>+7.3f}")
    
    # ---- 分析4: Head的重要性排序 (基于模式间差异) ----
    print("\n--- 模式敏感度最高的heads (按KL散度总和排序) ---")
    
    head_sensitivity = {}
    for layer in range(n_layers):
        for head in range(n_heads):
            total_kl = 0
            for mode in mode_names[1:]:
                kls = last_token_attn_kl(chat_attns[layer], mode_attns[mode][layer])
                total_kl += kls[head]
            head_sensitivity[(layer, head)] = total_kl
    
    sorted_heads = sorted(head_sensitivity.items(), key=lambda x: x[1], reverse=True)
    print(f"{'Rank':>6}  {'Layer':>6}  {'Head':>6}  {'TotalKL':>10}")
    for rank, ((layer, head), kl) in enumerate(sorted_heads[:20]):
        print(f"{rank+1:>6}  L{layer:>4}  H{head:>4}  {kl:>10.3f}")
    
    # ---- 分析5: 层级分布 ----
    print("\n--- 模式敏感heads的层级分布 ---")
    top_20_layers = [layer for (layer, head), _ in sorted_heads[:20]]
    for layer in range(n_layers):
        count = top_20_layers.count(layer)
        if count > 0:
            bar = "█" * count
            print(f"  L{layer:>2}: {bar} ({count})")


# ============================================================
# 实验C: 生成轨迹稳定性
# ============================================================

def exp_c_generation_trajectory(model):
    """token-by-token生成轨迹"""
    print("=" * 70)
    print("实验C: 生成轨迹 — 推理/翻译/聊天是否形成不同trajectory?")
    print("=" * 70)
    
    # 短prompts，生成5个token
    gen_prompts = {
        "chat": "Hello, I would like to",
        "translate": "Translate to French: The cat",
        "reason": "Think step by step: If it rains",
        "summarize": "Summarize: The quick brown fox",
        "code": "Write Python: def hello",
    }
    
    mode_names = list(gen_prompts.keys())
    n_gen_tokens = 8
    
    # 收集生成轨迹
    mode_gen_trajectories = {}
    
    for mode, prompt in gen_prompts.items():
        print(f"\n  Generating for mode: {mode}")
        print(f"  Prompt: {prompt}")
        
        tokens = model.to_tokens(prompt)
        trajectory = []  # 每步的resid_post at last token
        
        with torch.no_grad():
            current_tokens = tokens.clone()
            
            for step in range(n_gen_tokens):
                # Run model - don't remove batch dim for easier token manipulation
                logits = model(current_tokens)
                
                # logits: [batch, seq_len, vocab] or [seq_len, vocab]
                if logits.dim() == 3:
                    last_logits = logits[0, -1]  # [vocab]
                else:
                    last_logits = logits[-1]  # [vocab]
                
                # Greedy decode
                next_token_id = last_logits.argmax().item()
                
                # Now run with cache to get hidden states
                _, cache = model.run_with_cache(current_tokens, remove_batch_dim=True)
                
                # 收集各层的resid_post at last position
                layer_states = []
                for layer in range(model.cfg.n_layers):
                    h = cache["resid_post", layer][-1].detach().numpy()
                    layer_states.append(h)
                trajectory.append(np.array(layer_states))
                
                # Append next token
                current_tokens = torch.cat([current_tokens, torch.tensor([[next_token_id]])], dim=-1)
                
                # Print generated token
                gen_token_str = model.tokenizer.decode([next_token_id])
                print(f"    Step {step}: +{gen_token_str}")
        
        mode_gen_trajectories[mode] = np.array(trajectory)  # [n_steps, n_layers, 768]
    
    # ---- 分析1: 生成轨迹的层间演化 ----
    print("\n--- 生成轨迹: 连续step间的余弦距离 ---")
    
    for mode in mode_names:
        traj = mode_gen_trajectories[mode]  # [steps, layers, 768]
        print(f"\n  Mode: {mode}")
        print(f"  {'Step':>6}", end="")
        for layer in [0, 3, 6, 9, 11]:
            print(f"  {'L'+str(layer):>8}", end="")
        print()
        
        for step in range(1, min(n_gen_tokens, traj.shape[0])):
            print(f"  {step:>6}", end="")
            for layer in [0, 3, 6, 9, 11]:
                h_prev = traj[step-1, layer]
                h_curr = traj[step, layer]
                h_prev_n = h_prev / (np.linalg.norm(h_prev) + 1e-10)
                h_curr_n = h_curr / (np.linalg.norm(h_curr) + 1e-10)
                cos_dist = 1 - np.dot(h_prev_n, h_curr_n)
                print(f"  {cos_dist:>8.4f}", end="")
            print()
    
    # ---- 分析2: 生成轨迹的模式可分性 ----
    print("\n--- 生成轨迹: 模式可分性 (Layer 6, 各step) ---")
    
    for step in range(min(n_gen_tokens, 5)):
        print(f"\n  Generation step {step}:")
        hs = [mode_gen_trajectories[m][step, 6] for m in mode_names]
        
        print(f"  {'':>12}", end="")
        for m in mode_names:
            print(f"  {m:>10}", end="")
        print()
        
        for m1 in mode_names:
            print(f"  {m1:>12}", end="")
            h1 = hs[mode_names.index(m1)]
            h1n = h1 / (np.linalg.norm(h1) + 1e-10)
            for m2 in mode_names:
                h2 = hs[mode_names.index(m2)]
                h2n = h2 / (np.linalg.norm(h2) + 1e-10)
                print(f"  {1 - np.dot(h1n, h2n):>10.4f}", end="")
            print()
    
    # ---- 分析3: 轨迹的"模式签名"随生成步骤的演化 ----
    print("\n--- 模式签名演化: PC投影随生成步骤的变化 ---")
    
    # 所有模式×步骤的h at Layer 6
    all_hs = []
    labels = []
    step_labels = []
    for mode in mode_names:
        for step in range(min(n_gen_tokens, mode_gen_trajectories[mode].shape[0])):
            all_hs.append(mode_gen_trajectories[mode][step, 6])
            labels.append(mode)
            step_labels.append(step)
    
    all_hs = np.array(all_hs)
    mean_h = all_hs.mean(axis=0)
    centered = all_hs - mean_h
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    
    print(f"  {'Mode':>12}  {'Step':>6}  {'PC1':>8}  {'PC2':>8}")
    for i, (mode, step) in enumerate(zip(labels, step_labels)):
        print(f"  {mode:>12}  {step:>6}  {U[i,0]:>8.3f}  {U[i,1]:>8.3f}")
    
    # ---- 分析4: 轨迹是否收敛到模式特异的attractor ----
    print("\n--- 轨迹收敛性: 后期步骤的within-mode距离 vs between-mode距离 ---")
    
    # 用最后3个步骤
    late_steps = list(range(max(0, n_gen_tokens-3), n_gen_tokens))
    
    within_dists = {m: [] for m in mode_names}
    between_dists = []
    
    # Each mode has one trajectory, so we compare layer-wise distances at late steps
    # Within: distance between consecutive steps for same mode
    # Between: distance between same step across modes
    
    for step in late_steps:
        for m in mode_names:
            if step > 0 and step < mode_gen_trajectories[m].shape[0]:
                h_prev = mode_gen_trajectories[m][step-1, 6]
                h_curr = mode_gen_trajectories[m][step, 6]
                h_prev_n = h_prev / (np.linalg.norm(h_prev) + 1e-10)
                h_curr_n = h_curr / (np.linalg.norm(h_curr) + 1e-10)
                within_dists[m].append(1 - np.dot(h_prev_n, h_curr_n))
        
        for i, m1 in enumerate(mode_names):
            for m2 in mode_names[i+1:]:
                if step < mode_gen_trajectories[m1].shape[0] and step < mode_gen_trajectories[m2].shape[0]:
                    h1 = mode_gen_trajectories[m1][step, 6]
                    h2 = mode_gen_trajectories[m2][step, 6]
                    h1n = h1 / (np.linalg.norm(h1) + 1e-10)
                    h2n = h2 / (np.linalg.norm(h2) + 1e-10)
                    between_dists.append(1 - np.dot(h1n, h2n))
    
    print(f"  Within-mode step distance: {np.mean([np.mean(v) for v in within_dists.values()]):.4f}")
    print(f"  Between-mode distance:     {np.mean(between_dists):.4f}")
    print(f"  Ratio (between/within):    {np.mean(between_dists) / (np.mean([np.mean(v) for v in within_dists.values()]) + 1e-10):.2f}")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, choices=["a", "b", "c"])
    args = parser.parse_args()
    
    print("Loading GPT-2 Small (fp32)...")
    model = get_model()
    print(f"Model loaded: {model.cfg.n_layers} layers, {model.cfg.n_heads} heads, d_model={model.cfg.d_model}")
    
    if args.exp == "a":
        exp_a_mode_switching(model)
    elif args.exp == "b":
        exp_b_attention_routing(model)
    elif args.exp == "c":
        exp_c_generation_trajectory(model)

if __name__ == "__main__":
    main()
