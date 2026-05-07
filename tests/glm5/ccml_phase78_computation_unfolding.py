"""
Phase 78: Computation Unfolding — 语言计算动力学
================================================

范式转变: 从"找feature"到"computation unfolding"

8条路线分析:
  路线1 统计几何派 → 高维幻觉, correlation ≠ mechanism
  路线2 Mechanistic Interpretability → 局部circuit无法解释全局
  路线3 动力系统派 → 我们正在进入
  路线4 信息论派 → 压缩结构
  路线5 程序归纳派 → 隐式程序执行
  路线6 生成递归派 → self-conditioning (核心!)
  路线7 神经符号派 → 变量绑定与组合
  路线8 生物脑派 → 离散近似

统一方向: 动力系统 + 信息论 + 程序归纳 + generation rollout

四个核心实验:
  A: 相对偏移分析 — Phase 77遗留: offset_norm / residual_norm
     如果增长 → 真实放大; 如果恒定 → norm伴生假象

  B: CoT计算展开 ★★★★★ — 最关键实验
     同一问题, CoT vs 直答, hidden轨迹如何不同?
     CoT是否 = 隐式计算外显化?
     每步生成是否累积了更多信息?

  C: 自稳定递归动力学
     从不同prompt开始生成, 轨迹是否收敛到同一模式?
     这是否像attractor dynamics?

  D: 信息累积率 — 每步生成增加了多少新信息?
     用互信息/entropy变化量衡量
     如果CoT的每步累积率 > 直答 → CoT确实是"计算展开"

Usage:
  python ccml_phase78_computation_unfolding.py --exp a
  python ccml_phase78_computation_unfolding.py --exp b
  python ccml_phase78_computation_unfolding.py --exp c
  python ccml_phase78_computation_unfolding.py --exp d
"""

import torch
import numpy as np
import argparse
from transformer_lens import HookedTransformer

def get_model():
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
    p_logprobs = torch.log_softmax(p_logits, dim=-1)
    q_probs = torch.softmax(q_logits, dim=-1)
    return torch.nn.functional.kl_div(p_logprobs, q_probs, reduction='sum').item()

# ============================================================
# 实验A: 相对偏移分析 (Phase 77遗留关键问题)
# ============================================================

def exp_a_relative_offset(model):
    """offset_norm / residual_norm 随层的演化"""
    print("=" * 70)
    print("实验A: 相对偏移分析 — 模式偏移是真实放大还是norm伴生?")
    print("=" * 70)
    
    shared_content = "The cat sat on the mat and the dog ran in the park"
    mode_prefixes = {
        "translate_fr": "Translate to French:",
        "translate_de": "Translate to German:",
        "summarize": "Summarize:",
        "explain": "Explain:",
        "continue": "Continue the text:",
        "neutral": "",
    }
    
    # 收集各模式的hidden states
    mode_hidden = {}
    mode_tokens = {}
    shared_token_ids = model.to_tokens(shared_content)[0].tolist()
    shared_len = len(shared_token_ids)
    
    print(f"\nShared content: {shared_len} tokens")
    
    with torch.no_grad():
        for mode, prefix in mode_prefixes.items():
            full_text = (prefix + " " + shared_content) if prefix else shared_content
            tokens = model.to_tokens(full_text)
            mode_tokens[mode] = tokens
            _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
            mode_hidden[mode] = {}
            for layer in range(model.cfg.n_layers):
                mode_hidden[mode][layer] = cache["resid_post", layer]
    
    reference = "neutral"
    
    # 核心分析: 对每个mode, 每层, 每个shared content位置
    # 计算 offset_norm 和 residual_norm
    print("\n--- 相对偏移 = offset_norm / residual_norm ---")
    print("如果随层增长 → 真实信号放大")
    print("如果恒定 → 只是norm自然增长的伴生现象")
    
    for mode in ["translate_fr", "summarize", "continue"]:
        mode_seq_len = mode_tokens[mode].shape[-1]
        mode_shared_start = mode_seq_len - shared_len
        
        print(f"\n  === {mode} ===")
        print(f"  {'Layer':>6}  {'OffsetNorm':>12}  {'ResidNorm':>12}  {'RelativeOffset':>15}  {'Trend':>8}")
        
        relative_offsets = []
        
        for layer in range(model.cfg.n_layers):
            # 取最后一个shared content token位置
            mode_resid = mode_hidden[mode][layer]
            ref_resid = mode_hidden[reference][layer]
            
            # 最后shared位置
            mode_last = mode_resid[-1].float()
            ref_last = ref_resid[-1].float()
            
            offset = mode_last - ref_last
            offset_norm = offset.norm().item()
            resid_norm = ref_last.norm().item()
            relative = offset_norm / (resid_norm + 1e-10)
            relative_offsets.append(relative)
            
            trend = ""
            if layer > 0 and relative_offsets[-1] > relative_offsets[-2] * 1.05:
                trend = "UP"
            elif layer > 0 and relative_offsets[-1] < relative_offsets[-2] * 0.95:
                trend = "DOWN"
            elif layer > 0:
                trend = "flat"
            
            print(f"  L{layer:>4}  {offset_norm:>12.2f}  {resid_norm:>12.2f}  {relative:>15.6f}  {trend:>8}")
        
        # 汇总: 前4层均值 vs 后4层均值
        early = np.mean(relative_offsets[:4])
        late = np.mean(relative_offsets[-4:])
        growth = late / (early + 1e-10)
        print(f"  → 早期(L0-3)均值: {early:.6f}, 晚期(L8-11)均值: {late:.6f}, 增长比: {growth:.2f}x")
    
    # 额外: 所有shared content位置的相对偏移(不只是最后位置)
    print("\n\n--- 各shared content位置的相对偏移 (层汇总) ---")
    for mode in ["translate_fr", "summarize"]:
        mode_seq_len = mode_tokens[mode].shape[-1]
        mode_shared_start = mode_seq_len - shared_len
        
        print(f"\n  {mode}:")
        for layer in [0, 4, 8, 11]:
            mode_resid = mode_hidden[mode][layer]
            ref_resid = mode_hidden[reference][layer]
            
            min_len = min(mode_resid.shape[0], ref_resid.shape[0])
            # 只看shared content部分
            mode_shared = mode_resid[-min(min_len, shared_len):]
            ref_shared = ref_resid[-min(min_len, shared_len):]
            
            actual_len = min(mode_shared.shape[0], ref_shared.shape[0])
            
            pos_relatives = []
            for pos in range(actual_len):
                offset = (mode_shared[pos] - ref_shared[pos]).float()
                offset_norm = offset.norm().item()
                resid_norm = ref_shared[pos].float().norm().item()
                pos_relatives.append(offset_norm / (resid_norm + 1e-10))
            
            # 位置0 vs 位置中 vs 位置末
            first_r = pos_relatives[0] if pos_relatives else 0
            mid_r = pos_relatives[len(pos_relatives)//2] if pos_relatives else 0
            last_r = pos_relatives[-1] if pos_relatives else 0
            mean_r = np.mean(pos_relatives)
            
            print(f"    L{layer:>2}: first={first_r:.6f}, mid={mid_r:.6f}, last={last_r:.6f}, mean={mean_r:.6f}")


# ============================================================
# 实验B: CoT计算展开 ★★★★★ (最关键实验)
# ============================================================

def exp_b_cot_unfolding(model):
    """CoT vs 直答: hidden轨迹如何不同?"""
    print("=" * 70)
    print("实验B: CoT计算展开 — 隐式计算外显化?")
    print("=" * 70)
    
    # 设计CoT vs 直答对比实验
    # 用GPT-2能处理的简单任务
    
    pairs = [
        # (direct_prompt, cot_prompt, task_name)
        (
            "The capital of France is",
            "Let me think. The capital of France is",
            "geography"
        ),
        (
            "2 + 3 =",
            "Let me add these numbers. 2 + 3 =",
            "addition"
        ),
        (
            "The opposite of hot is",
            "Let me think about opposites. The opposite of hot is",
            "antonym"
        ),
        (
            "If it rains, the ground gets",
            "Let me reason about this. If it rains, the ground gets",
            "causal"
        ),
        (
            "A cat is a type of",
            "Let me think about categories. A cat is a type of",
            "category"
        ),
        (
            "After winter comes",
            "Let me think about seasons. After winter comes",
            "sequence"
        ),
        (
            "The past tense of walk is",
            "Let me think about grammar. The past tense of walk is",
            "grammar"
        ),
        (
            "Water freezes at",
            "Let me think about science. Water freezes at",
            "science"
        ),
    ]
    
    n_gen_steps = 20
    
    print(f"\n--- 生成 {n_gen_steps} 步, 比较CoT vs 直答的轨迹 ---")
    
    for direct_prompt, cot_prompt, task_name in pairs:
        print(f"\n{'='*60}")
        print(f"  Task: {task_name}")
        print(f"  Direct: {direct_prompt}")
        print(f"  CoT:    {cot_prompt}")
        
        # ---- 1. 生成序列 ----
        with torch.no_grad():
            # Direct
            d_tokens = model.to_tokens(direct_prompt)
            d_seq = d_tokens[0].tolist()
            d_logit_trace = []
            
            for step in range(n_gen_steps):
                logits = model(torch.tensor([d_seq]))
                last_logits = logits[0, -1]
                d_logit_trace.append(last_logits.clone())
                next_token = last_logits.argmax().item()
                d_seq.append(next_token)
            
            # CoT
            c_tokens = model.to_tokens(cot_prompt)
            c_seq = c_tokens[0].tolist()
            c_logit_trace = []
            
            for step in range(n_gen_steps):
                logits = model(torch.tensor([c_seq]))
                last_logits = logits[0, -1]
                c_logit_trace.append(last_logits.clone())
                next_token = last_logits.argmax().item()
                c_seq.append(next_token)
        
        # ---- 2. 生成文本 ----
        d_text = model.tokenizer.decode(d_seq[:len(d_seq)])
        c_text = model.tokenizer.decode(c_seq[:len(c_seq)])
        print(f"\n  Direct生成: {d_text[:100]}...")
        print(f"  CoT生成:    {c_text[:100]}...")
        
        # ---- 3. 轨迹分析: 逐步KL与基准的差异 ----
        # 用neutral prompt ("The")作为参考基线
        neutral_prompt = "The"
        with torch.no_grad():
            n_tokens = model.to_tokens(neutral_prompt)
            n_seq = n_tokens[0].tolist()
            n_logit_trace = []
            for step in range(n_gen_steps):
                logits = model(torch.tensor([n_seq]))
                last_logits = logits[0, -1]
                n_logit_trace.append(last_logits.clone())
                next_token = last_logits.argmax().item()
                n_seq.append(next_token)
        
        # 逐步计算: direct vs neutral, CoT vs neutral
        print(f"\n  逐步KL (vs neutral baseline):")
        print(f"  {'Step':>4}  {'DirectKL':>10}  {'CoTKL':>10}  {'CoT/Direct':>12}")
        
        d_kls = []
        c_kls = []
        for step in range(min(len(d_logit_trace), len(c_logit_trace), len(n_logit_trace))):
            d_kl = kl_divergence(d_logit_trace[step], n_logit_trace[step])
            c_kl = kl_divergence(c_logit_trace[step], n_logit_trace[step])
            d_kls.append(d_kl)
            c_kls.append(c_kl)
            ratio = c_kl / (d_kl + 1e-10)
            print(f"  {step:>4}  {d_kl:>10.4f}  {c_kl:>10.4f}  {ratio:>12.2f}")
        
        # 汇总
        early_d = np.mean(d_kls[:5]) if len(d_kls) >= 5 else np.mean(d_kls)
        early_c = np.mean(c_kls[:5]) if len(c_kls) >= 5 else np.mean(c_kls)
        late_d = np.mean(d_kls[-5:]) if len(d_kls) >= 5 else np.mean(d_kls)
        late_c = np.mean(c_kls[-5:]) if len(c_kls) >= 5 else np.mean(c_kls)
        
        print(f"\n  Direct:  早期KL={early_d:.4f}, 晚期KL={late_d:.4f}")
        print(f"  CoT:     早期KL={early_c:.4f}, 晚期KL={late_c:.4f}")
        print(f"  CoT/Direct比值: 早期={early_c/(early_d+1e-10):.2f}x, 晚期={late_c/(late_d+1e-10):.2f}x")
    
    # ---- 4. 核心对比: CoT vs Direct的hidden轨迹结构 ----
    print("\n\n" + "=" * 70)
    print("核心分析: CoT vs Direct的hidden state轨迹")
    print("=" * 70)
    
    # 对每个task, 在最后生成步, 收集每层的hidden state
    for direct_prompt, cot_prompt, task_name in pairs[:4]:
        print(f"\n--- {task_name} ---")
        
        with torch.no_grad():
            # 生成序列
            d_tokens = model.to_tokens(direct_prompt)
            d_seq = d_tokens[0].tolist()
            for step in range(10):
                logits = model(torch.tensor([d_seq]))
                next_token = logits[0, -1].argmax().item()
                d_seq.append(next_token)
            
            c_tokens = model.to_tokens(cot_prompt)
            c_seq = c_tokens[0].tolist()
            for step in range(10):
                logits = model(torch.tensor([c_seq]))
                next_token = logits[0, -1].argmax().item()
                c_seq.append(next_token)
        
        # 收集每层的hidden state (在最后token位置)
        with torch.no_grad():
            _, d_cache = model.run_with_cache(torch.tensor([d_seq]), remove_batch_dim=True)
            _, c_cache = model.run_with_cache(torch.tensor([c_seq]), remove_batch_dim=True)
        
        print(f"  {'Layer':>6}  {'CosSim':>8}  {'D_norm':>10}  {'C_norm':>10}  {'Offset_norm':>12}")
        
        for layer in range(model.cfg.n_layers):
            d_h = d_cache["resid_post", layer][-1].float()  # 最后token
            c_h = c_cache["resid_post", layer][-1].float()
            
            cos_sim = torch.nn.functional.cosine_similarity(d_h.unsqueeze(0), c_h.unsqueeze(0)).item()
            d_norm = d_h.norm().item()
            c_norm = c_h.norm().item()
            offset = (c_h - d_h).norm().item()
            
            print(f"  L{layer:>4}  {cos_sim:>8.4f}  {d_norm:>10.2f}  {c_norm:>10.2f}  {offset:>12.2f}")


# ============================================================
# 实验C: 自稳定递归动力学
# ============================================================

def exp_c_self_stabilization(model):
    """从不同初始条件出发, 生成轨迹是否收敛到同一模式?"""
    print("=" * 70)
    print("实验C: 自稳定递归动力学 — 轨迹是否收敛?")
    print("=" * 70)
    
    # 设计: 3个不同开头, 但暗示同一模式
    # 看生成轨迹是否收敛到相似的hidden state
    
    mode_groups = {
        "french": [
            "Translate to French: The",
            "En français: The",  
            "French translation of",
        ],
        "code": [
            "Write Python code: def",
            "Implement a function: def",
            "Code this in Python: def",
        ],
        "reasoning": [
            "Think step by step: If",
            "Let's reason about: If",
            "Consider the logic: If",
        ],
    }
    
    n_steps = 25
    
    for mode, prompts in mode_groups.items():
        print(f"\n{'='*60}")
        print(f"  Mode: {mode}")
        print(f"  Prompts: {str(prompts).encode('ascii', 'replace').decode()}")
        
        trajectories = []  # 每个prompt的hidden轨迹
        generated_texts = []
        
        for prompt in prompts:
            with torch.no_grad():
                tokens = model.to_tokens(prompt)
                seq = tokens[0].tolist()
                
                # 收集每步最后token的hidden state (L6, 一个中间层)
                h_traces = {layer: [] for layer in [4, 6, 8, 11]}
                
                for step in range(n_steps):
                    logits, cache = model.run_with_cache(torch.tensor([seq]), remove_batch_dim=True)
                    
                    for layer in h_traces:
                        h = cache["resid_post", layer][-1].float()  # 最后token
                        h_traces[layer].append(h.clone())
                    
                    next_token = logits[0, -1].argmax().item()
                    seq.append(next_token)
                
                trajectories.append(h_traces)
                generated_texts.append(model.tokenizer.decode(seq[:50]))
        
        # 打印生成文本
        for i, text in enumerate(generated_texts):
            safe_text = text[:120].encode('ascii', 'replace').decode()
            print(f"\n  Prompt {i}: {safe_text}...")
        
        # 分析轨迹收敛: 逐步计算不同prompt间的cosine similarity
        print(f"\n  轨迹收敛分析 (prompt对间cosine similarity):")
        
        for layer in [6, 11]:
            print(f"\n  Layer {layer}:")
            
            # 3个prompt两两比较
            pairs = [(0, 1), (0, 2), (1, 2)]
            
            for i, j in pairs:
                h_i = trajectories[i][layer]
                h_j = trajectories[j][layer]
                
                min_len = min(len(h_i), len(h_j))
                cos_sims = []
                for step in range(min_len):
                    cs = torch.nn.functional.cosine_similarity(
                        h_i[step].unsqueeze(0), h_j[step].unsqueeze(0)
                    ).item()
                    cos_sims.append(cs)
                
                early = np.mean(cos_sims[:5]) if cos_sims else 0
                late = np.mean(cos_sims[-5:]) if cos_sims else 0
                
                print(f"    Prompt{i} vs Prompt{j}: early_cos={early:.4f}, late_cos={late:.4f}, "
                      f"change={'UP' if late > early else 'DOWN'} ({late-early:+.4f})")
        
        # 额外: 轨迹的"收缩率" — 相邻步的hidden state变化
        print(f"\n  轨迹收缩率 (相邻步的cosine similarity — 越高=变化越小):")
        for layer in [6, 11]:
            for i in range(len(trajectories)):
                h_trace = trajectories[i][layer]
                step_cos = []
                for step in range(1, len(h_trace)):
                    cs = torch.nn.functional.cosine_similarity(
                        h_trace[step-1].unsqueeze(0), h_trace[step].unsqueeze(0)
                    ).item()
                    step_cos.append(cs)
                
                early_move = np.mean(step_cos[:5]) if step_cos else 0
                late_move = np.mean(step_cos[-5:]) if step_cos else 0
                print(f"    L{layer} Prompt{i}: early_step_cos={early_move:.4f}, late_step_cos={late_move:.4f}")


# ============================================================
# 实验D: 信息累积率 — 每步生成增加多少新信息?
# ============================================================

def exp_d_information_accumulation(model):
    """每步生成中, logit分布的信息增量"""
    print("=" * 70)
    print("实验D: 信息累积率 — 每步生成增加多少新信息?")
    print("=" * 70)
    
    # 用entropy变化和top-1 token变化来衡量信息累积
    
    prompts = {
        "direct_add": "2 + 3 =",
        "cot_add": "Let me add. 2 + 3 =",
        "direct_capital": "The capital of France is",
        "cot_capital": "Let me think. The capital of France is",
        "direct_opposite": "The opposite of hot is",
        "cot_opposite": "Think about opposites. The opposite of hot is",
    }
    
    n_steps = 15
    
    for name, prompt in prompts.items():
        is_cot = name.startswith("cot_")
        task = name.split("_")[1] if "_" in name else name
        
        print(f"\n{'='*50}")
        print(f"  {name}: {prompt}")
        
        with torch.no_grad():
            tokens = model.to_tokens(prompt)
            seq = tokens[0].tolist()
            
            entropies = []
            top1_probs = []
            top1_tokens = []
            
            for step in range(n_steps):
                logits = model(torch.tensor([seq]))
                last_logits = logits[0, -1]
                
                probs = torch.softmax(last_logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
                top1_prob, top1_idx = probs.max(dim=0)
                top1_token = model.tokenizer.decode([top1_idx.item()])
                
                entropies.append(entropy)
                top1_probs.append(top1_prob.item())
                top1_tokens.append(top1_token)
                
                next_token = top1_idx.item()
                seq.append(next_token)
        
        generated = model.tokenizer.decode(seq)
        print(f"  生成: {generated[:100]}...")
        
        # 信息指标
        print(f"\n  {'Step':>4}  {'Entropy':>10}  {'Top1Prob':>10}  {'Top1Token':>12}  {'EntropyDelta':>12}")
        
        for step in range(len(entropies)):
            delta = entropies[step] - entropies[step-1] if step > 0 else 0
            print(f"  {step:>4}  {entropies[step]:>10.4f}  {top1_probs[step]:>10.4f}  "
                  f"{top1_tokens[step]:>12}  {delta:>+12.4f}")
        
        # 汇总
        early_e = np.mean(entropies[:5]) if entropies else 0
        late_e = np.mean(entropies[-5:]) if entropies else 0
        early_p = np.mean(top1_probs[:5]) if top1_probs else 0
        late_p = np.mean(top1_probs[-5:]) if top1_probs else 0
        
        print(f"\n  早期entropy={early_e:.4f}, 晚期entropy={late_e:.4f}")
        print(f"  早期top1_prob={early_p:.4f}, 晚期top1_prob={late_p:.4f}")
    
    # ---- CoT vs Direct 信息累积对比 ----
    print("\n\n" + "=" * 70)
    print("CoT vs Direct 信息累积对比")
    print("=" * 70)
    
    task_pairs = [
        ("direct_add", "cot_add"),
        ("direct_capital", "cot_capital"),
        ("direct_opposite", "cot_opposite"),
    ]
    
    for d_name, c_name in task_pairs:
        d_prompt = prompts[d_name]
        c_prompt = prompts[c_name]
        task = d_name.split("_")[1]
        
        with torch.no_grad():
            # Direct
            d_tokens = model.to_tokens(d_prompt)
            d_seq = d_tokens[0].tolist()
            d_entropies = []
            d_confidences = []  # top-1 probability
            
            for step in range(n_steps):
                logits = model(torch.tensor([d_seq]))
                probs = torch.softmax(logits[0, -1], dim=-1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
                top1_prob = probs.max().item()
                d_entropies.append(entropy)
                d_confidences.append(top1_prob)
                next_token = logits[0, -1].argmax().item()
                d_seq.append(next_token)
            
            # CoT
            c_tokens = model.to_tokens(c_prompt)
            c_seq = c_tokens[0].tolist()
            c_entropies = []
            c_confidences = []
            
            for step in range(n_steps):
                logits = model(torch.tensor([c_seq]))
                probs = torch.softmax(logits[0, -1], dim=-1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
                top1_prob = probs.max().item()
                c_entropies.append(entropy)
                c_confidences.append(top1_prob)
                next_token = logits[0, -1].argmax().item()
                c_seq.append(next_token)
        
        # 对比
        d_mean_e = np.mean(d_entropies)
        c_mean_e = np.mean(c_entropies)
        d_mean_c = np.mean(d_confidences)
        c_mean_c = np.mean(c_confidences)
        
        # 信息累积: 逐步confidence是否增长
        d_conf_trend = np.polyfit(range(len(d_confidences)), d_confidences, 1)[0] if len(d_confidences) > 1 else 0
        c_conf_trend = np.polyfit(range(len(c_confidences)), c_confidences, 1)[0] if len(c_confidences) > 1 else 0
        
        print(f"\n  {task}:")
        print(f"    Direct: mean_entropy={d_mean_e:.4f}, mean_conf={d_mean_c:.4f}, conf_trend={d_conf_trend:+.6f}")
        print(f"    CoT:    mean_entropy={c_mean_e:.4f}, mean_conf={c_mean_c:.4f}, conf_trend={c_conf_trend:+.6f}")
        print(f"    → CoT的confidence趋势{'>' if c_conf_trend > d_conf_trend else '<='}Direct")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, choices=["a", "b", "c", "d"])
    args = parser.parse_args()
    
    print("Loading GPT-2 Small (fp32)...")
    model = get_model()
    print(f"Model loaded: {model.cfg.n_layers} layers, {model.cfg.n_heads} heads, d_model={model.cfg.d_model}")
    
    if args.exp == "a":
        exp_a_relative_offset(model)
    elif args.exp == "b":
        exp_b_cot_unfolding(model)
    elif args.exp == "c":
        exp_c_self_stabilization(model)
    elif args.exp == "d":
        exp_d_information_accumulation(model)

if __name__ == "__main__":
    main()
