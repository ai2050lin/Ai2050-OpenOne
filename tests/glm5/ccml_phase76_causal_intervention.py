"""
Phase 76: 因果干预 + 计算图分析
=================================

核心转变: 从"观察统计"到"因果干预"

Phase 75发现L5H7可能是"模式路由head"，但只有相关性，没有因果性。
Phase 76通过干预实验验证因果性，并分析计算图结构。

实验A: Head Ablation — 关闭单个head，测量行为变化
  - 关闭L5H7，翻译/推理/代码的logit分布如何变化?
  - 关闭其他top-heads，对比效应
  - 关闭随机head作为基线

实验B: Activation Patching — 跨模式patching
  - 将chat模式的h patch到translate模式
  - 将translate模式的attention patch到chat模式
  - 测量: 哪些层/heads的patch能改变行为?

实验C: 计算图分析 — Token-to-token influence
  - 对每个token position，哪些其他position对它影响最大?
  - 不同模式下，influence graph如何变化?
  - 是否存在"信息瓶颈"token?

Usage:
  python ccml_phase76_causal_intervention.py --exp a   # Head ablation
  python ccml_phase76_causal_intervention.py --exp b   # Activation patching
  python ccml_phase76_causal_intervention.py --exp c   # Computation graph
"""

import torch
import numpy as np
import argparse
from pathlib import Path
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

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

def get_test_prompts():
    """测试用prompts"""
    return {
        "chat": "Hello, I would like to talk about the weather today.",
        "translate": "Translate to French: The cat sat on the mat.",
        "reason": "Think step by step: If it rains, then the ground gets wet.",
        "code": "Write Python code: def factorial(n):",
        "summarize": "Summarize: The quick brown fox jumped over the lazy dog.",
    }

def get_top_k_tokens(logits, k=10):
    """获取top-k token及其概率"""
    probs = torch.softmax(logits, dim=-1)
    top_k = torch.topk(probs, k)
    return top_k.indices.tolist(), top_k.values.tolist()

# ============================================================
# 实验A: Head Ablation
# ============================================================

def exp_a_head_ablation(model):
    """关闭单个head，测量行为变化"""
    print("=" * 70)
    print("实验A: Head Ablation — 关闭单个head的因果效应")
    print("=" * 70)
    
    prompts = get_test_prompts()
    mode_names = list(prompts.keys())
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    
    # Phase 75发现的模式敏感heads
    sensitive_heads = [
        (5, 7),   # 最敏感
        (6, 9),   # 第2
        (9, 10),  # 第3
        (5, 0),   # 第4
        (11, 8),  # 第5
    ]
    
    # 随机对照heads
    random_heads = [(1, 5), (3, 2), (7, 8), (10, 1), (4, 6)]
    
    # ---- 1. 基线: 无ablation ----
    print("\n--- 基线logit分布 (无ablation) ---")
    
    baseline_logits = {}
    baseline_top_tokens = {}
    
    with torch.no_grad():
        for mode, prompt in prompts.items():
            tokens = model.to_tokens(prompt)
            logits = model(tokens)  # [batch, seq, vocab]
            last_logits = logits[0, -1]  # [vocab]
            baseline_logits[mode] = last_logits
            
            top_indices, top_probs = get_top_k_tokens(last_logits)
            top_strs = [model.tokenizer.decode([t]) for t in top_indices]
            print(f"\n  {mode}: top-5 = {list(zip(top_strs[:5], [f'{p:.3f}' for p in top_probs[:5]]))}")
    
    # ---- 2. Ablation函数 ----
    def ablate_head(layer, head):
        """返回一个hook function，将指定head的输出置零
        使用hook_z: [batch, seq, n_heads, d_head]"""
        def hook_fn(z, hook):
            z[:, :, head, :] = 0
            return z
        return hook_fn
    
    # ---- 3. Ablation实验: 敏感heads ----
    print("\n--- 敏感Head Ablation (Phase 75发现的模式路由heads) ---")
    
    for layer, head in sensitive_heads:
        print(f"\n  === Ablating L{layer}H{head} ===")
        
        for mode, prompt in prompts.items():
            tokens = model.to_tokens(prompt)
            
            with torch.no_grad():
                # 运行model，ablate指定head
                hook_name = f"blocks.{layer}.attn.hook_z"
                logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(hook_name, ablate_head(layer, head))],
                )
                last_logits = logits[0, -1]
                
                # 计算与基线的KL散度
                kl_div = torch.nn.functional.kl_div(
                    torch.log_softmax(last_logits, dim=-1),
                    torch.softmax(baseline_logits[mode], dim=-1),
                    reduction='sum'
                ).item()
                
                # 计算top-1 token变化
                top_indices, top_probs = get_top_k_tokens(last_logits)
                baseline_top = get_top_k_tokens(baseline_logits[mode])[0][0]
                top_strs = [model.tokenizer.decode([t]) for t in top_indices]
                
                changed = "CHANGED" if top_indices[0] != baseline_top else "same"
                print(f"    {mode}: KL={kl_div:.3f}, top-1={top_strs[0]}({top_probs[0]:.3f}) [{changed}]")
    
    # ---- 4. Ablation实验: 随机heads (对照) ----
    print("\n--- 随机Head Ablation (对照组) ---")
    
    for layer, head in random_heads:
        print(f"\n  === Ablating L{layer}H{head} ===")
        
        for mode, prompt in prompts.items():
            tokens = model.to_tokens(prompt)
            
            with torch.no_grad():
                hook_name = f"blocks.{layer}.attn.hook_z"
                logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(hook_name, ablate_head(layer, head))],
                )
                last_logits = logits[0, -1]
                
                kl_div = torch.nn.functional.kl_div(
                    torch.log_softmax(last_logits, dim=-1),
                    torch.softmax(baseline_logits[mode], dim=-1),
                    reduction='sum'
                ).item()
                
                top_indices, top_probs = get_top_k_tokens(last_logits)
                baseline_top = get_top_k_tokens(baseline_logits[mode])[0][0]
                top_strs = [model.tokenizer.decode([t]) for t in top_indices]
                
                changed = "CHANGED" if top_indices[0] != baseline_top else "same"
                print(f"    {mode}: KL={kl_div:.3f}, top-1={top_strs[0]}({top_probs[0]:.3f}) [{changed}]")
    
    # ---- 5. 汇总: 敏感vs随机heads的KL散度对比 ----
    print("\n\n--- 汇总: 敏感heads vs 随机heads的平均KL散度 ---")
    
    all_heads = sensitive_heads + random_heads
    all_kls = {}
    
    for layer, head in all_heads:
        head_kls = []
        for mode, prompt in prompts.items():
            tokens = model.to_tokens(prompt)
            
            with torch.no_grad():
                hook_name = f"blocks.{layer}.attn.hook_z"
                logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(hook_name, ablate_head(layer, head))],
                )
                last_logits = logits[0, -1]
                
                kl_div = torch.nn.functional.kl_div(
                    torch.log_softmax(last_logits, dim=-1),
                    torch.softmax(baseline_logits[mode], dim=-1),
                    reduction='sum'
                ).item()
                head_kls.append(kl_div)
        
        label = "SENSITIVE" if (layer, head) in sensitive_heads else "random"
        all_kls[(layer, head)] = (np.mean(head_kls), label)
    
    print(f"{'Head':>10}  {'Type':>10}  {'MeanKL':>10}")
    for (l, h), (kl, label) in sorted(all_kls.items(), key=lambda x: x[1][0], reverse=True):
        print(f"L{l}H{h:>3}  {label:>10}  {kl:>10.4f}")
    
    # ---- 6. 全head扫描: 每个head的ablation效应 ----
    print("\n--- 全head扫描: 每个head ablation后的平均KL散度 ---")
    
    head_effects = {}
    
    for layer in range(n_layers):
        for head in range(n_heads):
            head_kls = []
            for mode, prompt in prompts.items():
                tokens = model.to_tokens(prompt)
                
                with torch.no_grad():
                    hook_name = f"blocks.{layer}.attn.hook_z"
                    logits = model.run_with_hooks(
                        tokens,
                        fwd_hooks=[(hook_name, ablate_head(layer, head))],
                    )
                    last_logits = logits[0, -1]
                    
                    kl_div = torch.nn.functional.kl_div(
                        torch.log_softmax(last_logits, dim=-1),
                        torch.softmax(baseline_logits[mode], dim=-1),
                        reduction='sum'
                    ).item()
                    head_kls.append(kl_div)
            
            head_effects[(layer, head)] = np.mean(head_kls)
        
        # 打印该层最强的head
        layer_effects = [(h, head_effects[(layer, h)]) for h in range(n_heads)]
        layer_effects.sort(key=lambda x: x[1], reverse=True)
        top_h, top_kl = layer_effects[0]
        print(f"  L{layer:>2}: strongest H{top_h} (KL={top_kl:.4f}), "
              f"top3=[{', '.join(f'H{h}(KL={kl:.3f})' for h, kl in layer_effects[:3])}]")
    
    # ---- 7. 按层汇总 ----
    print("\n--- 按层的平均ablation效应 ---")
    for layer in range(n_layers):
        layer_kls = [head_effects[(layer, h)] for h in range(n_heads)]
        print(f"  L{layer:>2}: mean={np.mean(layer_kls):.4f}, max={np.max(layer_kls):.4f}, "
              f"std={np.std(layer_kls):.4f}")


# ============================================================
# 实验B: Activation Patching
# ============================================================

def exp_b_activation_patching(model):
    """跨模式activation patching"""
    print("=" * 70)
    print("实验B: Activation Patching — 跨模式干预")
    print("=" * 70)
    
    prompts = get_test_prompts()
    n_layers = model.cfg.n_layers
    
    # ---- 1. 收集各模式的cache ----
    print("\n--- 收集各模式的activation cache ---")
    
    mode_caches = {}
    mode_tokens = {}
    
    with torch.no_grad():
        for mode, prompt in prompts.items():
            tokens = model.to_tokens(prompt)
            mode_tokens[mode] = tokens
            _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
            mode_caches[mode] = cache
            print(f"  {mode}: {tokens.shape[-1]} tokens")
    
    # ---- 2. Residual stream patching ----
    print("\n--- Residual Stream Patching: 将translate的h patch到chat ---")
    print("测量: patch后chat的logit分布是否向translate偏移?")
    
    source_mode = "translate"
    target_mode = "chat"
    
    # 基线logits
    with torch.no_grad():
        chat_tokens = mode_tokens[target_mode]
        chat_baseline = model(chat_tokens)[0, -1]
        translate_baseline = model(mode_tokens[source_mode])[0, -1]
    
    # 基线KL: chat vs translate
    baseline_kl = torch.nn.functional.kl_div(
        torch.log_softmax(chat_baseline, dim=-1),
        torch.softmax(translate_baseline, dim=-1),
        reduction='sum'
    ).item()
    print(f"  Baseline KL(chat, translate) = {baseline_kl:.4f}")
    
    # 逐层patch: 将translate的resid_post替换到chat
    print(f"\n  Patching resid_post from {source_mode} to {target_mode}:")
    print(f"  {'Layer':>6}  {'KL(chat_patched, translate)':>30}  {'KL_reduction%':>15}")
    
    for layer in range(n_layers):
        # translate的resid_post at this layer
        translate_resid = mode_caches[source_mode]["resid_post", layer]
        
        def make_patch_fn(cached_activation):
            def patch_fn(activation, hook):
                # Replace with cached activation
                # 但seq length可能不同，只patch最后一个token
                activation[0, -1, :] = cached_activation[-1]
                return activation
            return patch_fn
        
        with torch.no_grad():
            hook_name = f"blocks.{layer}.hook_resid_post"
            patched_logits = model.run_with_hooks(
                chat_tokens,
                fwd_hooks=[(hook_name, make_patch_fn(translate_resid))],
            )
            patched_last = patched_logits[0, -1]
            
            patched_kl = torch.nn.functional.kl_div(
                torch.log_softmax(patched_last, dim=-1),
                torch.softmax(translate_baseline, dim=-1),
                reduction='sum'
            ).item()
            
            reduction = (baseline_kl - patched_kl) / (baseline_kl + 1e-10) * 100
            print(f"  L{layer:>4}  {patched_kl:>30.4f}  {reduction:>14.1f}%")
    
    # ---- 3. Attention output patching (逐层逐head) ----
    print("\n--- Attention Output Patching: 将translate的attn output patch到chat ---")
    print("找出: 哪些层+heads的attn output最影响模式行为")
    
    # 简化: 只测几个关键层
    key_layers = [0, 3, 5, 6, 7, 9, 11]
    
    for layer in key_layers:
        # translate的attn output
        translate_attn_out = mode_caches[source_mode]["attn_out", layer]
        
        def make_attn_patch_fn(cached_act):
            def patch_fn(activation, hook):
                activation[0, -1, :] = cached_act[-1]
                return activation
            return patch_fn
        
        with torch.no_grad():
            hook_name = f"blocks.{layer}.hook_attn_out"
            patched_logits = model.run_with_hooks(
                chat_tokens,
                fwd_hooks=[(hook_name, make_attn_patch_fn(translate_attn_out))],
            )
            patched_last = patched_logits[0, -1]
            
            patched_kl = torch.nn.functional.kl_div(
                torch.log_softmax(patched_last, dim=-1),
                torch.softmax(translate_baseline, dim=-1),
                reduction='sum'
            ).item()
            
            reduction = (baseline_kl - patched_kl) / (baseline_kl + 1e-10) * 100
            print(f"  L{layer}: KL={patched_kl:.4f}, reduction={reduction:.1f}%")
    
    # ---- 4. 反向patching: chat → translate ----
    print("\n--- 反向Patching: 将chat的h patch到translate ---")
    
    for layer in range(n_layers):
        chat_resid = mode_caches[target_mode]["resid_post", layer]
        
        def make_patch_fn2(cached_act):
            def patch_fn(activation, hook):
                activation[0, -1, :] = cached_act[-1]
                return activation
            return patch_fn
        
        with torch.no_grad():
            hook_name = f"blocks.{layer}.hook_resid_post"
            patched_logits = model.run_with_hooks(
                mode_tokens[source_mode],
                fwd_hooks=[(hook_name, make_patch_fn2(chat_resid))],
            )
            patched_last = patched_logits[0, -1]
            
            patched_kl = torch.nn.functional.kl_div(
                torch.log_softmax(patched_last, dim=-1),
                torch.softmax(chat_baseline, dim=-1),
                reduction='sum'
            ).item()
            
            reverse_baseline_kl = torch.nn.functional.kl_div(
                torch.log_softmax(model(mode_tokens[source_mode])[0, -1], dim=-1),
                torch.softmax(chat_baseline, dim=-1),
                reduction='sum'
            ).item()
            
            reduction = (reverse_baseline_kl - patched_kl) / (reverse_baseline_kl + 1e-10) * 100
            print(f"  L{layer}: KL={patched_kl:.4f}, reduction={reduction:.1f}%")


# ============================================================
# 实验C: 计算图分析
# ============================================================

def exp_c_computation_graph(model):
    """Token-to-token influence structure"""
    print("=" * 70)
    print("实验C: 计算图分析 — Token间影响结构")
    print("=" * 70)
    
    prompts = get_test_prompts()
    mode_names = list(prompts.keys())
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    
    # ---- 1. OV path影响力分析 ----
    print("\n--- OV Path影响力: 各head对最终logit的贡献 ---")
    print("使用activation patching: 将某head输出置零，测量logit变化")
    
    # 选择translate模式作为分析对象
    target_mode = "translate"
    prompt = prompts[target_mode]
    tokens = model.to_tokens(prompt)
    token_strs = [model.tokenizer.decode([t.item()]) for t in tokens[0]]
    
    print(f"  Prompt: {prompt}")
    print(f"  Tokens: {token_strs}")
    
    # 基线
    with torch.no_grad():
        baseline_logits = model(tokens)[0, -1]
    
    # 对每个layer+head，测量ablation效应
    print(f"\n  各head对最终logit的影响力 (KL散度):")
    print(f"  {'Layer':>6}  {'Head':>6}  {'KL':>10}  {'top3_influenced_tokens':>40}")
    
    head_importance = {}
    
    for layer in [0, 3, 5, 6, 7, 9, 11]:
        for head in range(n_heads):
            with torch.no_grad():
                def ablate_hook(activation, hook):
                    activation[:, :, head, :] = 0
                    return activation
                
                hook_name = f"blocks.{layer}.attn.hook_z"
                ablated_logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(hook_name, ablate_hook)],
                )
                ablated_last = ablated_logits[0, -1]
                
                kl = torch.nn.functional.kl_div(
                    torch.log_softmax(ablated_last, dim=-1),
                    torch.softmax(baseline_logits, dim=-1),
                    reduction='sum'
                ).item()
                
                head_importance[(layer, head)] = kl
    
    # 排序打印top heads
    sorted_heads = sorted(head_importance.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Top-20 最重要heads (对translate模式):")
    print(f"  {'Rank':>6}  {'Layer':>6}  {'Head':>6}  {'KL':>10}")
    for rank, ((layer, head), kl) in enumerate(sorted_heads[:20]):
        print(f"  {rank+1:>6}  L{layer:>4}  H{head:>4}  {kl:>10.4f}")
    
    # ---- 2. Token-to-token influence via attention ----
    print("\n--- Token间Influence Graph (via attention) ---")
    print("各token位置被其他位置attend的强度 (translate模式)")
    
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
    
    # 汇总各层各head的attention到influence matrix
    seq_len = tokens.shape[-1]
    total_influence = np.zeros((seq_len, seq_len))
    
    for layer in range(n_layers):
        pattern = cache["pattern", layer].detach().numpy()  # [n_heads, seq_q, seq_k]
        # 平均各head
        mean_pattern = pattern.mean(axis=0)  # [seq_q, seq_k]
        total_influence += mean_pattern
    
    # 归一化
    total_influence /= n_layers
    
    # 打印: 每个token被哪些其他token最多attend
    print(f"\n  各token位置的被attend强度 (top-3 source positions):")
    for pos in range(seq_len):
        token_str = token_strs[pos]
        # 哪些位置最attend到这个位置?
        attended_by = total_influence[:, pos]  # 每个query position对pos的attention
        top_sources = np.argsort(attended_by)[::-1][:3]
        top_vals = attended_by[top_sources]
        source_strs = [f"pos{s}({token_strs[s]})={v:.3f}" for s, v in zip(top_sources, top_vals)]
        print(f"  pos{pos:>2}({token_str:>12}): ← {', '.join(source_strs)}")
    
    # ---- 3. 跨模式influence graph差异 ----
    print("\n--- 跨模式Influence Graph差异 ---")
    print("比较chat vs translate vs code的influence pattern")
    
    mode_influences = {}
    
    for mode in ["chat", "translate", "code"]:
        prompt = prompts[mode]
        tokens = model.to_tokens(prompt)
        
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
        
        seq_len = tokens.shape[-1]
        total_influence = np.zeros((seq_len, seq_len))
        
        for layer in range(n_layers):
            pattern = cache["pattern", layer].detach().numpy()
            mean_pattern = pattern.mean(axis=0)
            total_influence += mean_pattern
        
        total_influence /= n_layers
        mode_influences[mode] = total_influence
    
    # 比较: 每个模式的"指令token influence" vs "内容token influence"
    for mode in ["chat", "translate", "code"]:
        prompt = prompts[mode]
        tokens = model.to_tokens(prompt)
        seq_len = tokens.shape[-1]
        
        influence = mode_influences[mode]
        
        # 最后一个token被其他token attend的总和
        last_token_attended = influence[-1, :].sum()
        
        # 被第一个token (通常是模式指令)的influence
        first_token_influence = influence[-1, 0] if seq_len > 1 else 0
        
        # 被最近3个token的influence
        recent_influence = influence[-1, -4:-1].sum() if seq_len > 3 else 0
        
        print(f"\n  {mode}:")
        print(f"    Last token total received attention: {last_token_attended:.4f}")
        print(f"    From first token (mode instruction): {first_token_influence:.4f} ({first_token_influence/last_token_attended*100:.1f}%)")
        print(f"    From recent 3 tokens:               {recent_influence:.4f} ({recent_influence/last_token_attended*100:.1f}%)")
    
    # ---- 4. 信息瓶颈分析 ----
    print("\n--- 信息瓶颈: 哪些token位置是关键路由节点? ---")
    
    for mode in ["translate", "code"]:
        prompt = prompts[mode]
        tokens = model.to_tokens(prompt)
        token_strs_local = [model.tokenizer.decode([t.item()]) for t in tokens[0]]
        seq_len = tokens.shape[-1]
        
        influence = mode_influences[mode]
        
        # 每个位置的"路由强度" = 作为source被attend的总和
        routing_strength = influence.sum(axis=0)  # 每个位置被所有query attend的总和
        
        print(f"\n  {mode} — 各位置的routing strength:")
        for pos in range(seq_len):
            bar = "█" * int(routing_strength[pos] * 5)
            print(f"    pos{pos:>2}({token_strs_local[pos]:>15}): {routing_strength[pos]:.3f} {bar}")


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
        exp_a_head_ablation(model)
    elif args.exp == "b":
        exp_b_activation_patching(model)
    elif args.exp == "c":
        exp_c_computation_graph(model)

if __name__ == "__main__":
    main()
