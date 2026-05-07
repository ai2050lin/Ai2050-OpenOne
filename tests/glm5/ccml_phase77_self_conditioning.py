"""
Phase 77: 自回归自条件 (Autoregressive Self-Conditioning)
==========================================================

核心范式转变:
  Phase 76的因果干预虽然方向对，但犯了4个硬伤:
  1. L0H10 "最重要" → 实际是早层扰动的级联放大，不是功能重要性
  2. L6-L8 "模式写入点" → patching阈值 ≠ 信息起源
  3. BOS "信息瓶颈" → 注意力汇聚点 ≠ 语义承载
  4. 单head ablation弱 → 完全分布式? → 需要多head联合干预

真正的核心问题:
  模式切换不是hidden-state attractor跳变，
  而是 autoregressive self-conditioning:
  
  h_t → token_t → h_{t+1} → token_{t+1} → ...
  
  初始prompt只提供微弱偏置，
  生成的第一个token反过来条件化后续生成，
  形成自稳定轨迹。

三个实验:
  A: Long-horizon rollout divergence
     - 不同模式prompt下，逐步生成，测量每步的KL散度
     - 核心假设: 初始KL小，随生成步数递增放大
     
  B: Multi-head ablation / path knockout
     - 不做单head，做head组联合ablation
     - 按层组(早/中/晚)和按功能组(ablation KL最高的top-k)
     - 核心假设: 存在冗余backup，多head联合knockout才有效
     
  C: Teacher-forced trajectory analysis
     - 同一token序列，不同mode prompt，hidden轨迹如何偏移
     - 核心假设: 模式差异是叠加在内容轨迹上的微弱偏置
     - 这是最关键的实验：分离模式效应和内容效应

Usage:
  python ccml_phase77_self_conditioning.py --exp a
  python ccml_phase77_self_conditioning.py --exp b
  python ccml_phase77_self_conditioning.py --exp c
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

def kl_divergence(p_logits, q_logits):
    """KL(p || q) from logits"""
    p_logprobs = torch.log_softmax(p_logits, dim=-1)
    q_probs = torch.softmax(q_logits, dim=-1)
    return torch.nn.functional.kl_div(p_logprobs, q_probs, reduction='sum').item()

def get_mode_prompts():
    """返回多组模式prompt，用于rollout实验"""
    return {
        "chat": [
            "Hello, I would like to talk about",
            "Let's have a conversation about",
            "Hi there! Can we discuss",
        ],
        "translate": [
            "Translate to French:",
            "Translate to German:",
            "Translate to Spanish:",
        ],
        "code": [
            "Write Python code:",
            "Write a function that",
            "Implement the following:",
        ],
        "reason": [
            "Think step by step:",
            "Let's reason about this:",
            "Consider the following logic:",
        ],
    }

# ============================================================
# 实验A: Long-horizon rollout divergence
# ============================================================

def exp_a_rollout_divergence(model):
    """不同模式prompt下，生成长序列，测量模式间KL随步数的演化"""
    print("=" * 70)
    print("实验A: Long-horizon Rollout Divergence")
    print("核心问题: 模式间KL是否随生成步数递增放大?")
    print("=" * 70)
    
    mode_prompts = get_mode_prompts()
    mode_names = list(mode_prompts.keys())
    n_steps = 30  # 生成30步
    
    # ---- 1. 对每对模式，测量逐步KL ----
    print(f"\n--- 逐步生成 {n_steps} tokens, 测量模式间KL演化 ---")
    
    # 为每个模式收集生成序列
    mode_sequences = {}
    mode_logit_traces = {}  # 每步的logit分布
    
    for mode, prompts in mode_prompts.items():
        all_logit_traces = []
        all_sequences = []
        
        for prompt in prompts:
            tokens = model.to_tokens(prompt)
            seq = tokens[0].tolist()
            logit_trace = []
            
            with torch.no_grad():
                for step in range(n_steps):
                    logits = model(torch.tensor([seq]))
                    last_logits = logits[0, -1]
                    logit_trace.append(last_logits.clone())
                    
                    # greedy decoding
                    next_token = last_logits.argmax().item()
                    seq.append(next_token)
            
            all_logit_traces.append(logit_trace)
            all_sequences.append(seq)
        
        mode_logit_traces[mode] = all_logit_traces
        mode_sequences[mode] = all_sequences
    
    # ---- 2. 计算每对模式在每个step的KL ----
    print("\n--- 模式对间KL随步数演化 ---")
    
    mode_pairs = [
        ("chat", "translate"),
        ("chat", "code"),
        ("chat", "reason"),
        ("translate", "code"),
    ]
    
    for m1, m2 in mode_pairs:
        print(f"\n  === {m1} vs {m2} ===")
        print(f"  {'Step':>4}  {'MeanKL':>10}  {'StdKL':>10}  {'Trend':>10}")
        
        step_kls = []
        for step in range(n_steps):
            pair_kls = []
            # 对每对prompt组合计算KL
            for trace1 in mode_logit_traces[m1]:
                for trace2 in mode_logit_traces[m2]:
                    if step < len(trace1) and step < len(trace2):
                        kl = kl_divergence(trace1[step], trace2[step])
                        pair_kls.append(kl)
            
            mean_kl = np.mean(pair_kls) if pair_kls else 0
            std_kl = np.std(pair_kls) if pair_kls else 0
            step_kls.append(mean_kl)
            print(f"  {step:>4}  {mean_kl:>10.4f}  {std_kl:>10.4f}")
        
        # 分析趋势: 前5步 vs 后5步的KL增长率
        if len(step_kls) >= 10:
            early_mean = np.mean(step_kls[:5])
            late_mean = np.mean(step_kls[-5:])
            growth = late_mean / (early_mean + 1e-10)
            print(f"  → 前5步均值KL: {early_mean:.4f}, 后5步均值KL: {late_mean:.4f}, 增长比: {growth:.2f}x")
    
    # ---- 3. 生成示例: 看不同模式实际生成了什么 ----
    print("\n\n--- 生成示例 (前3个prompt，前20步) ---")
    for mode, prompts in mode_prompts.items():
        print(f"\n  [{mode}]")
        for i, prompt in enumerate(prompts[:2]):
            tokens = model.to_tokens(prompt)
            seq = tokens[0].tolist()
            with torch.no_grad():
                for step in range(20):
                    logits = model(torch.tensor([seq]))
                    next_token = logits[0, -1].argmax().item()
                    seq.append(next_token)
            generated = model.tokenizer.decode(seq)
            print(f"    Prompt {i}: {generated[:120]}...")
    
    # ---- 4. 同模式内的KL (baseline: 纯粹由内容差异导致) ----
    print("\n\n--- 同模式内不同prompt间的KL (内容差异baseline) ---")
    for mode, prompts in mode_prompts.items():
        if len(prompts) < 2:
            continue
        internal_kls = []
        for step in range(min(n_steps, 20)):
            for i in range(len(prompts)):
                for j in range(i+1, len(prompts)):
                    if step < len(mode_logit_traces[mode][i]) and step < len(mode_logit_traces[mode][j]):
                        kl = kl_divergence(
                            mode_logit_traces[mode][i][step],
                            mode_logit_traces[mode][j][step]
                        )
                        internal_kls.append(kl)
        
        if internal_kls:
            print(f"  {mode}: 内部KL均值={np.mean(internal_kls):.4f}, "
                  f"步0-4均值={np.mean([kl_divergence(mode_logit_traces[mode][0][s], mode_logit_traces[mode][1][s]) for s in range(min(5, n_steps))]):.4f}")
    
    # ---- 5. 关键指标: 跨模式KL / 同模式KL 的比值 ----
    print("\n\n--- 关键指标: 跨模式KL / 同模式KL 比值 ---")
    print("如果比值≈1: 模式差异≈内容差异(模式指令无实质效果)")
    print("如果比值>1: 模式差异超出内容差异(存在模式效应)")
    print("如果比值随步数增长: 自条件放大效应存在")
    
    for m1, m2 in mode_pairs[:3]:
        cross_kls = []
        for step in range(min(n_steps, 20)):
            for trace1 in mode_logit_traces[m1][:1]:
                for trace2 in mode_logit_traces[m2][:1]:
                    if step < len(trace1) and step < len(trace2):
                        cross_kls.append(kl_divergence(trace1[step], trace2[step]))
        
        # 同模式内KL
        m1_internal = []
        if len(mode_logit_traces[m1]) >= 2:
            for step in range(min(n_steps, 20)):
                if step < len(mode_logit_traces[m1][0]) and step < len(mode_logit_traces[m1][1]):
                    m1_internal.append(kl_divergence(
                        mode_logit_traces[m1][0][step],
                        mode_logit_traces[m1][1][step]
                    ))
        
        if m1_internal and cross_kls:
            ratio_early = (np.mean(cross_kls[:5]) / (np.mean(m1_internal[:5]) + 1e-10))
            ratio_late = (np.mean(cross_kls[-5:]) / (np.mean(m1_internal[-5:]) + 1e-10))
            print(f"  {m1} vs {m2}: 前5步比值={ratio_early:.2f}x, 后5步比值={ratio_late:.2f}x")


# ============================================================
# 实验B: Multi-head ablation / path knockout
# ============================================================

def exp_b_multi_head_ablation(model):
    """多head联合ablation，测试冗余和协同效应"""
    print("=" * 70)
    print("实验B: Multi-head Ablation / Path Knockout")
    print("核心问题: 是否存在冗余backup? 多head联合删除是否崩溃?")
    print("=" * 70)
    
    prompts = {
        "chat": "Hello, I would like to talk about the weather today.",
        "translate": "Translate to French: The cat sat on the mat.",
        "code": "Write Python code: def factorial(n):",
    }
    
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    
    # ---- 1. 基线 ----
    print("\n--- 收集基线logits ---")
    baseline_logits = {}
    with torch.no_grad():
        for mode, prompt in prompts.items():
            tokens = model.to_tokens(prompt)
            baseline_logits[mode] = model(tokens)[0, -1]
    
    # ---- 2. 按层组ablation ----
    print("\n--- 按层组ablation: 删除整层的所有heads ---")
    print("测试: 整层knockout的效应 vs 单head之和")
    
    for layer in range(n_layers):
        # 删除该层所有heads
        def ablate_full_layer(activation, hook):
            activation[:, :, :, :] = 0
            return activation
        
        with torch.no_grad():
            mode_kls = {}
            for mode, prompt in prompts.items():
                tokens = model.to_tokens(prompt)
                hook_name = f"blocks.{layer}.attn.hook_z"
                ablated_logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(hook_name, ablate_full_layer)],
                )
                kl = kl_divergence(ablated_logits[0, -1], baseline_logits[mode])
                mode_kls[mode] = kl
            
            mean_kl = np.mean(list(mode_kls.values()))
            print(f"  L{layer:>2}: mean_KL={mean_kl:.4f}, "
                  f"per-mode=[{', '.join(f'{m}={kl:.4f}' for m, kl in mode_kls.items())}]")
    
    # ---- 3. Top-k heads联合ablation ----
    print("\n\n--- Top-k heads联合ablation ---")
    print("Phase 76发现: L0H10单head KL最大(0.085)")
    print("问题: 删除top-1, top-3, top-5, top-10 heads的效应是否超线性?")
    
    # 先重新扫描获取所有head的KL排名
    head_kl_ranking = []
    for layer in range(n_layers):
        for head in range(n_heads):
            cur_layer = layer
            cur_head = head
            
            def make_ablate_fn(h_idx):
                def hook_fn(z, hook):
                    z[:, :, h_idx, :] = 0
                    return z
                return hook_fn
            
            with torch.no_grad():
                kls = []
                for mode, prompt in prompts.items():
                    tokens = model.to_tokens(prompt)
                    hook_name = f"blocks.{cur_layer}.attn.hook_z"
                    ablated = model.run_with_hooks(
                        tokens,
                        fwd_hooks=[(hook_name, make_ablate_fn(cur_head))],
                    )
                    kls.append(kl_divergence(ablated[0, -1], baseline_logits[mode]))
                head_kl_ranking.append((cur_layer, cur_head, np.mean(kls)))
    
    head_kl_ranking.sort(key=lambda x: x[2], reverse=True)
    
    print("\n  Top-15 heads by single-ablation KL:")
    for rank, (l, h, kl) in enumerate(head_kl_ranking[:15]):
        print(f"    #{rank+1}: L{l}H{h}, KL={kl:.4f}")
    
    # 联合ablation: top-1, top-3, top-5, top-10
    for k in [1, 3, 5, 10]:
        top_k_heads = [(l, h) for l, h, _ in head_kl_ranking[:k]]
        
        def make_multi_ablate(heads_to_ablate):
            def hook_fn(z, hook):
                layer_id = int(hook.name.split('.')[1])
                for (l, h) in heads_to_ablate:
                    if l == layer_id:
                        z[:, :, h, :] = 0
                return z
            return hook_fn
        
        # 需要为每层创建hook
        with torch.no_grad():
            mode_kls = {}
            for mode, prompt in prompts.items():
                tokens = model.to_tokens(prompt)
                
                hooks = []
                affected_layers = set(l for l, h in top_k_heads)
                for l in affected_layers:
                    hook_name = f"blocks.{l}.attn.hook_z"
                    heads_in_layer = [h for (ll, h) in top_k_heads if ll == l]
                    
                    def make_layer_ablate(heads_list):
                        def fn(z, hook):
                            for h in heads_list:
                                z[:, :, h, :] = 0
                            return z
                        return fn
                    
                    hooks.append((hook_name, make_layer_ablate(heads_in_layer)))
                
                ablated = model.run_with_hooks(tokens, fwd_hooks=hooks)
                kl = kl_divergence(ablated[0, -1], baseline_logits[mode])
                mode_kls[mode] = kl
            
            mean_kl = np.mean(list(mode_kls.values()))
            # 期望的线性叠加
            expected_linear = sum(kl for _, _, kl in head_kl_ranking[:k])
            
            print(f"\n  Top-{k} heads联合ablation:")
            print(f"    实际KL={mean_kl:.4f}, 线性期望={expected_linear:.4f}, "
                  f"比率={mean_kl/(expected_linear+1e-10):.2f}")
            print(f"    per-mode=[{', '.join(f'{m}={kl:.4f}' for m, kl in mode_kls.items())}]")
    
    # ---- 4. 按层组联合ablation (早/中/晚) ----
    print("\n\n--- 按层组联合ablation ---")
    layer_groups = {
        "early (L0-3)": list(range(0, 4)),
        "middle (L4-7)": list(range(4, 8)),
        "late (L8-11)": list(range(8, 12)),
    }
    
    for group_name, layers in layer_groups.items():
        with torch.no_grad():
            mode_kls = {}
            for mode, prompt in prompts.items():
                tokens = model.to_tokens(prompt)
                
                hooks = []
                for l in layers:
                    hook_name = f"blocks.{l}.attn.hook_z"
                    def ablate_all(z, hook):
                        z[:, :, :, :] = 0
                        return z
                    hooks.append((hook_name, ablate_all))
                
                ablated = model.run_with_hooks(tokens, fwd_hooks=hooks)
                kl = kl_divergence(ablated[0, -1], baseline_logits[mode])
                mode_kls[mode] = kl
            
            mean_kl = np.mean(list(mode_kls.values()))
            print(f"  {group_name}: mean_KL={mean_kl:.4f}, "
                  f"per-mode=[{', '.join(f'{m}={kl:.4f}' for m, kl in mode_kls.items())}]")
    
    # ---- 5. L0全部heads ablation (修正Phase 76的误读) ----
    print("\n\n--- L0全部heads ablation (修正Phase 76误读) ---")
    print("Phase 76结论: L0H10是最重要的head")
    print("修正: L0KL大可能只是早层扰动级联放大，不是功能重要性")
    print("如果L0全部ablate后KL与单head差不多 → 说明不是L0的功能，而是扰动传播")
    
    with torch.no_grad():
        # L0单head最大KL
        l0_single_max = max(kl for l, h, kl in head_kl_ranking if l == 0)
        l0_single_sum = sum(kl for l, h, kl in head_kl_ranking if l == 0)
        
        # L0全部heads ablate
        mode_kls = {}
        for mode, prompt in prompts.items():
            tokens = model.to_tokens(prompt)
            hook_name = "blocks.0.attn.hook_z"
            def ablate_all(z, hook):
                z[:, :, :, :] = 0
                return z
            ablated = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, ablate_all)])
            kl = kl_divergence(ablated[0, -1], baseline_logits[mode])
            mode_kls[mode] = kl
        
        l0_full_kl = np.mean(list(mode_kls.values()))
        print(f"  L0单head最大KL: {l0_single_max:.4f}")
        print(f"  L0全部heads KL: {l0_full_kl:.4f}")
        print(f"  L0单head KL之和: {l0_single_sum:.4f}")
        print(f"  比率(全部/单最大): {l0_full_kl/l0_single_max:.2f}")
        print(f"  比率(全部/总和): {l0_full_kl/l0_single_sum:.2f}")
        print(f"  → 如果全部≈单最大: 说明L0内部冗余，L0H10只是最大扰动点")
        print(f"  → 如果全部>>单最大: 说明L0各head有独立贡献")


# ============================================================
# 实验C: Teacher-forced trajectory analysis
# ============================================================

def exp_c_teacher_forced_trajectory(model):
    """同一token序列，不同mode prompt，hidden轨迹如何偏移"""
    print("=" * 70)
    print("实验C: Teacher-forced Trajectory Analysis")
    print("核心问题: 模式差异是叠加在内容轨迹上的微弱偏置?")
    print("=" * 70)
    
    # ---- 设计思路 ----
    # 关键: 分离模式效应和内容效应
    # 方法: 对同一个"内容"，加上不同的mode prefix
    # 然后对共享的内容部分，比较hidden state的偏移
    
    # 例如:
    #   序列A: "Translate to French:" + "The cat sat on the mat"
    #   序列B: "Write about:" + "The cat sat on the mat"  
    #   序列C: "Explain why:" + "The cat sat on the mat"
    # 在共享部分 "The cat sat on the mat" 的hidden states，
    # 模式差异是什么?
    
    shared_content = "The cat sat on the mat and the dog ran in the park"
    mode_prefixes = {
        "translate_fr": "Translate to French:",
        "translate_de": "Translate to German:",
        "summarize": "Summarize:",
        "explain": "Explain:",
        "continue": "Continue the text:",
        "neutral": "",  # 无prefix，纯内容
    }
    
    # ---- 1. 构造teacher-forced序列 ----
    print("\n--- 构造teacher-forced序列 ---")
    
    full_sequences = {}
    prefix_lengths = {}
    
    for mode, prefix in mode_prefixes.items():
        full_text = prefix + " " + shared_content if prefix else shared_content
        tokens = model.to_tokens(full_text)
        full_sequences[mode] = tokens
        # prefix长度
        prefix_tokens = model.to_tokens(prefix + " ") if prefix else model.to_tokens("")
        prefix_lengths[mode] = prefix_tokens.shape[-1]
        decoded = [model.tokenizer.decode([t.item()]) for t in tokens[0]]
        print(f"  {mode}: {len(decoded)} tokens, prefix={prefix_lengths[mode]} tokens")
        print(f"    Full: {full_text[:80]}...")
    
    # ---- 2. 收集各模式的hidden states ----
    print("\n--- 收集各模式的residual stream轨迹 ---")
    
    mode_hidden = {}  # mode -> {layer: tensor of hidden states}
    
    with torch.no_grad():
        for mode, tokens in full_sequences.items():
            _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
            mode_hidden[mode] = {}
            for layer in range(model.cfg.n_layers):
                resid = cache["resid_post", layer]  # [seq, d_model]
                mode_hidden[mode][layer] = resid
    
    # ---- 3. 共享内容部分的hidden state偏移 ----
    # 只比较shared content对应的token位置
    print("\n--- 共享内容部分的hidden state偏移 ---")
    
    # 找到最短的prefix长度作为对齐起点
    # 实际上不同mode的prefix长度不同，所以shared content的起始位置不同
    # 但shared content本身是相同的token序列
    
    # 更好的方法: 直接用neutral(无prefix)的token序列作为参照
    # 然后对其他mode，只看与shared content相同的token位置
    
    neutral_tokens = full_sequences["neutral"]
    neutral_seq_len = neutral_tokens.shape[-1]
    neutral_token_ids = neutral_tokens[0].tolist()
    
    print(f"\n  Neutral序列长度: {neutral_seq_len} tokens")
    print(f"  Neutral tokens: {[model.tokenizer.decode([t]) for t in neutral_token_ids[:10]]}...")
    
    # 对每个mode，找到shared content部分
    # 简化: 对齐最后N个token（shared content在末尾）
    shared_token_ids = model.to_tokens(shared_content)[0].tolist()
    shared_len = len(shared_token_ids)
    
    print(f"  Shared content长度: {shared_len} tokens")
    
    # 计算每对mode在shared content部分的cosine distance
    print("\n--- 共享内容位置的cosine distance (各层) ---")
    
    compare_modes = ["translate_fr", "summarize", "explain", "continue", "neutral"]
    reference = "neutral"
    
    for layer in [0, 2, 4, 6, 8, 10, 11]:
        print(f"\n  Layer {layer}:")
        
        for mode in compare_modes:
            if mode == reference:
                continue
            
            # 取shared content部分
            mode_seq_len = full_sequences[mode].shape[-1]
            mode_shared_start = mode_seq_len - shared_len
            ref_shared_start = 0  # neutral没有prefix
            
            mode_resid = mode_hidden[mode][layer][mode_shared_start:]  # [shared_len, d_model]
            ref_resid = mode_hidden[reference][layer][ref_shared_start:ref_shared_start+shared_len]
            
            if mode_resid.shape[0] < shared_len or ref_resid.shape[0] < shared_len:
                min_len = min(mode_resid.shape[0], ref_resid.shape[0])
                mode_resid = mode_resid[:min_len]
                ref_resid = ref_resid[:min_len]
            
            # 对每个shared token位置计算cosine distance
            cos_dists = []
            for pos in range(mode_resid.shape[0]):
                v1 = mode_resid[pos].float()
                v2 = ref_resid[pos].float()
                cos_sim = torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
                cos_dists.append(1 - cos_sim)
            
            mean_cos_dist = np.mean(cos_dists)
            # 位置0 vs 最后位置的cosine distance
            first_cd = cos_dists[0] if cos_dists else 0
            last_cd = cos_dists[-1] if cos_dists else 0
            
            print(f"    {mode:>15} vs {reference}: mean_cos_dist={mean_cos_dist:.6f}, "
                  f"first={first_cd:.6f}, last={last_cd:.6f}")
    
    # ---- 4. 逐token偏移向量分析 ----
    print("\n\n--- 偏移向量分析: 模式偏移是否方向一致? ---")
    print("如果translate_fr和translate_de的偏移向量方向相似 → 模式效应是系统性的")
    print("如果各mode偏移向量正交 → 模式效应是随机的")
    
    for layer in [4, 6, 8, 11]:
        print(f"\n  Layer {layer}:")
        
        # 收集各mode相对于neutral的偏移向量
        offset_vectors = {}
        for mode in ["translate_fr", "translate_de", "summarize", "explain", "continue"]:
            mode_seq_len = full_sequences[mode].shape[-1]
            mode_shared_start = mode_seq_len - shared_len
            
            mode_resid = mode_hidden[mode][layer][mode_shared_start:]
            ref_resid = mode_hidden[reference][layer][:shared_len]
            
            min_len = min(mode_resid.shape[0], ref_resid.shape[0])
            offset = mode_resid[:min_len] - ref_resid[:min_len]  # [shared_len, d_model]
            offset_vectors[mode] = offset
        
        # 计算偏移向量之间的cosine similarity
        mode_list = list(offset_vectors.keys())
        print(f"  偏移向量cosine similarity (取最后token位置):")
        
        for i, m1 in enumerate(mode_list):
            for m2 in mode_list[i+1:]:
                v1 = offset_vectors[m1][-1].float()
                v2 = offset_vectors[m2][-1].float()
                cos_sim = torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
                print(f"    {m1:>15} vs {m2:>15}: cos_sim={cos_sim:.4f}")
    
    # ---- 5. 偏移向量的norm vs 位置 ----
    print("\n\n--- 偏移向量norm随token位置的变化 ---")
    print("核心: 模式偏置是仅在prefix处，还是传播到后续content?")
    
    for mode in ["translate_fr", "summarize", "code_mode_placeholder"]:
        if mode not in offset_vectors and mode != "code_mode_placeholder":
            continue
        if mode == "code_mode_placeholder":
            mode = "continue"  # fallback
    
    for mode in ["translate_fr", "summarize", "continue"]:
        mode_seq_len = full_sequences[mode].shape[-1]
        mode_shared_start = mode_seq_len - shared_len
        
        print(f"\n  {mode}:")
        for layer in [0, 4, 8, 11]:
            mode_resid = mode_hidden[mode][layer][mode_shared_start:]
            ref_resid = mode_hidden[reference][layer][:shared_len]
            
            min_len = min(mode_resid.shape[0], ref_resid.shape[0])
            offset = mode_resid[:min_len] - ref_resid[:min_len]
            
            # 每5个token打印一次offset norm
            norms = [offset[p].float().norm().item() for p in range(min_len)]
            sampled = [(p, norms[p]) for p in range(0, min_len, max(1, min_len//8))]
            print(f"    L{layer:>2}: " + 
                  ", ".join(f"pos{p}={n:.2f}" for p, n in sampled))
    
    # ---- 6. 关键指标: 模式偏移在深层是否被放大 ----
    print("\n\n--- 模式偏移的层间放大因子 ---")
    print("如果深层偏移norm > 浅层 → 存在层间放大（支持self-conditioning假说）")
    print("如果深层偏移norm ≈ 浅层 → 无放大（模式信息不在residual传播）")
    
    for mode in ["translate_fr", "summarize", "continue"]:
        mode_seq_len = full_sequences[mode].shape[-1]
        mode_shared_start = mode_seq_len - shared_len
        
        print(f"\n  {mode} (最后shared token位置的偏移norm):")
        for layer in range(model.cfg.n_layers):
            mode_resid = mode_hidden[mode][layer][mode_shared_start:]
            ref_resid = mode_hidden[reference][layer][:shared_len]
            
            min_len = min(mode_resid.shape[0], ref_resid.shape[0])
            if min_len > 0:
                offset = (mode_resid[min_len-1] - ref_resid[min_len-1]).float()
                norm = offset.norm().item()
                print(f"    L{layer:>2}: offset_norm={norm:.2f}")


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
        exp_a_rollout_divergence(model)
    elif args.exp == "b":
        exp_b_multi_head_ablation(model)
    elif args.exp == "c":
        exp_c_teacher_forced_trajectory(model)

if __name__ == "__main__":
    main()
