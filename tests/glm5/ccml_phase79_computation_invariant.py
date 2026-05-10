"""
Phase 79: Computation Invariants — 在content变化中寻找不变的计算策略
=====================================================================

核心范式升级:
  Phase 78的错误: 把"统计轨迹"解释成"计算轨迹"
  → trajectory不同 ≠ computation不同
  
  三个层次:
    1. content trajectory — token内容 (最表层, 最容易发散)
    2. representation trajectory — h_t (中间层, 被content驱动)
    3. computation policy — routing/update/retrieval strategy (最深层)
  
  真正的问题: 在content改变时, 什么保持不变?

四个核心实验:
  A: Routing Topology Invariant ★★★★★
     不同数学题是否调用同一种head routing pattern?
     不同翻译句子是否调用同一种attention graph?
     → 核心方法: 比较attention pattern的graph结构, 而不是hidden point
  
  B: Residual Transition Jacobian Invariant
     不同输入通过同一层时, 局部线性映射是否共享结构?
     → 核心方法: 计算Jacobian的SVD, 比较top singular vectors方向
  
  C: Boundary-Conditioned Computation ★★★★★ (最关键)
     prefix如何改变computation policy?
     → 核心方法: 同一content, 不同prefix, 看attention routing如何变化
     → 这是Phase 78 boundary effect的深化: 不看hidden偏移, 看computation policy偏移
  
  D: CoT Policy Invariant (需要Qwen3)
     不同CoT链是否共享同一种step expansion policy?
     → 只在有推理能力的模型上做

关键方法论:
  - 不比较hidden point, 而比较computation operator
  - 不看"在哪里", 而看"如何计算"
  - 不看convergence, 而看invariance

Usage:
  python ccml_phase79_computation_invariant.py --exp a
  python ccml_phase79_computation_invariant.py --exp b
  python ccml_phase79_computation_invariant.py --exp c
  python ccml_phase79_computation_invariant.py --exp d
  python ccml_phase79_computation_invariant.py --exp all
"""

import torch
import numpy as np
import argparse
from collections import defaultdict
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

# ============================================================
# 工具函数
# ============================================================

def get_attention_patterns(model, text):
    """获取所有层所有head的attention pattern"""
    tokens = model.to_tokens(text)
    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
    
    patterns = {}
    for layer in range(model.cfg.n_layers):
        # attn_pattern shape: [n_heads, seq_len, seq_len]
        patterns[layer] = cache[f'blocks.{layer}.attn.hook_pattern'].detach().cpu()
    
    return patterns, cache

def compute_graph_similarity(pat1, pat2, method='cosine'):
    """
    比较两个attention pattern的graph结构相似性
    关键: 比较的是routing topology, 不是具体数值
    
    方法: 
      1. 将attention pattern视为有向加权图
      2. 比较图的拓扑性质
    """
    # pat1, pat2: [n_heads, seq_len, seq_len]
    n_heads = pat1.shape[0]
    
    # 对齐到最小序列长度
    min_seq = min(pat1.shape[1], pat2.shape[1])
    
    similarities = []
    for h in range(n_heads):
        # 方法1: 截取到相同长度后展平比较cosine similarity
        sub1 = pat1[h, :min_seq, :min_seq].flatten().float()
        sub2 = pat2[h, :min_seq, :min_seq].flatten().float()
        
        norm1 = sub1.norm()
        norm2 = sub2.norm()
        
        if norm1 > 1e-8 and norm2 > 1e-8:
            cos_sim = torch.nn.functional.cosine_similarity(sub1.unsqueeze(0), sub2.unsqueeze(0)).item()
        else:
            cos_sim = 0.0
        
        similarities.append(cos_sim)
    
    return similarities

def compute_topk_routing_overlap(pat1, pat2, k=5):
    """
    比较两个attention pattern的top-k routing overlap
    核心: 不看attention weight数值, 而看"哪些token被关注"这个routing决策
    
    如果不同输入的同一head关注相同位置 → routing policy不变
    """
    n_heads = pat1.shape[0]
    seq_len = pat1.shape[1]
    
    # 只比较shared的token位置 (取min seq_len)
    min_len = min(pat1.shape[2], pat2.shape[2])
    
    overlaps = []
    for h in range(n_heads):
        head_overlap = 0
        count = 0
        for pos in range(min_len):
            # 每个位置, 找top-k被关注的source位置
            topk1 = torch.topk(pat1[h, pos, :min_len], min(k, min_len)).indices
            topk2 = torch.topk(pat2[h, pos, :min_len], min(k, min_len)).indices
            
            # Jaccard overlap
            set1 = set(topk1.tolist())
            set2 = set(topk2.tolist())
            if len(set1 | set2) > 0:
                head_overlap += len(set1 & set2) / len(set1 | set2)
            count += 1
        
        overlaps.append(head_overlap / max(count, 1))
    
    return overlaps

def compute_row_rank_correlation(pat1, pat2):
    """
    Spearman rank correlation of attention distributions
    核心: 排序不变 → routing priority不变
    
    即使具体权重值不同, 如果排序相同 → computation policy相同
    """
    n_heads = pat1.shape[0]
    min_len = min(pat1.shape[2], pat2.shape[2])
    
    correlations = []
    for h in range(n_heads):
        head_corr = 0
        count = 0
        for pos in range(min_len):
            row1 = pat1[h, pos, :min_len].float()
            row2 = pat2[h, pos, :min_len].float()
            
            # Rank correlation (Spearman)
            rank1 = row1.argsort().argsort().float()
            rank2 = row2.argsort().argsort().float()
            
            rank1_centered = rank1 - rank1.mean()
            rank2_centered = rank2 - rank2.mean()
            
            denom = rank1_centered.norm() * rank2_centered.norm()
            if denom > 1e-8:
                corr = (rank1_centered @ rank2_centered) / denom
                head_corr += corr.item()
                count += 1
        
        correlations.append(head_corr / max(count, 1))
    
    return correlations

# ============================================================
# 实验A: Routing Topology Invariant ★★★★★
# ============================================================

def exp_a_routing_topology(model):
    """
    核心问题: 不同content是否共享同一种attention routing topology?
    
    三个维度比较:
      1. 同类任务不同内容 (如: 不同加法题) → routing应该高度相似
      2. 不同类任务 (如: 加法 vs 翻译) → routing应该不同
      3. 同句不同prefix → routing如何被prefix改变
    
    如果computation policy存在 → 同类任务的routing拓扑应当不变
    """
    print("=" * 70)
    print("实验A: Routing Topology Invariant")
    print("核心问题: content改变时, attention routing是否保持不变?")
    print("=" * 70)
    
    # ---- 第一组: 同类数学任务 ----
    math_tasks = {
        "add_1": "2 + 3 =",
        "add_2": "7 + 4 =",
        "add_3": "15 + 23 =",
        "add_4": "9 + 6 =",
        "add_5": "11 + 8 =",
        "add_6": "3 + 5 =",
        "add_7": "12 + 7 =",
        "add_8": "4 + 9 =",
        "add_9": "21 + 14 =",
        "add_10": "6 + 2 =",
    }
    
    # ---- 第二组: 同类翻译任务 ----
    translate_tasks = {
        "trans_1": "Translate to French: The cat is on the mat",
        "trans_2": "Translate to French: The dog runs in the park",
        "trans_3": "Translate to French: The bird sings a song",
        "trans_4": "Translate to French: The sun shines bright",
        "trans_5": "Translate to French: The water flows down",
        "trans_6": "Translate to French: The child plays outside",
        "trans_7": "Translate to French: The tree grows tall",
        "trans_8": "Translate to French: The rain falls softly",
        "trans_9": "Translate to French: The wind blows hard",
        "trans_10": "Translate to French: The moon rises slowly",
    }
    
    # ---- 第三组: 同类补全任务 ----
    continue_tasks = {
        "cont_1": "The capital of France is",
        "cont_2": "The capital of Germany is",
        "cont_3": "The capital of Japan is",
        "cont_4": "The capital of Italy is",
        "cont_5": "The capital of Spain is",
        "cont_6": "The capital of China is",
        "cont_7": "The capital of Brazil is",
        "cont_8": "The capital of India is",
        "cont_9": "The capital of Russia is",
        "cont_10": "The capital of Egypt is",
    }
    
    # ---- 第四组: 反义词任务 ----
    antonym_tasks = {
        "ant_1": "The opposite of hot is",
        "ant_2": "The opposite of big is",
        "ant_3": "The opposite of fast is",
        "ant_4": "The opposite of happy is",
        "ant_5": "The opposite of light is",
        "ant_6": "The opposite of strong is",
        "ant_7": "The opposite of loud is",
        "ant_8": "The opposite of rough is",
        "ant_9": "The opposite of wide is",
        "ant_10": "The opposite of tall is",
    }
    
    task_groups = {
        "addition": math_tasks,
        "translate_fr": translate_tasks,
        "capital": continue_tasks,
        "antonym": antonym_tasks,
    }
    
    # 收集所有attention patterns
    all_patterns = {}
    for group_name, tasks in task_groups.items():
        print(f"\n  处理 {group_name} 组...")
        all_patterns[group_name] = {}
        for task_name, text in tasks.items():
            patterns, _ = get_attention_patterns(model, text)
            all_patterns[group_name][task_name] = patterns
    
    # ---- 核心分析1: 同组内routing相似度 ----
    print("\n" + "=" * 50)
    print("分析1: 同组内routing topology相似度")
    print("如果computation policy存在 → 同组内routing应当高度一致")
    print("=" * 50)
    
    for group_name, tasks in task_groups.items():
        task_names = list(tasks.keys())
        n_tasks = len(task_names)
        
        # 两两比较
        within_cos_sims = []
        within_routing_overlaps = []
        within_rank_corrs = []
        
        for i in range(n_tasks):
            for j in range(i+1, n_tasks):
                name1, name2 = task_names[i], task_names[j]
                
                for layer in range(model.cfg.n_layers):
                    pat1 = all_patterns[group_name][name1][layer]
                    pat2 = all_patterns[group_name][name2][layer]
                    
                    # Cosine similarity
                    cos_sims = compute_graph_similarity(pat1, pat2)
                    within_cos_sims.append((layer, np.mean(cos_sims)))
                    
                    # Top-k routing overlap
                    routing_overlaps = compute_topk_routing_overlap(pat1, pat2, k=3)
                    within_routing_overlaps.append((layer, np.mean(routing_overlaps)))
                    
                    # Rank correlation
                    rank_corrs = compute_row_rank_correlation(pat1, pat2)
                    within_rank_corrs.append((layer, np.mean(rank_corrs)))
        
        # 按层聚合
        for metric_name, metric_data in [
            ("Cosine Sim", within_cos_sims),
            ("Top-3 Routing Overlap", within_routing_overlaps),
            ("Rank Correlation", within_rank_corrs),
        ]:
            layer_vals = defaultdict(list)
            for layer, val in metric_data:
                layer_vals[layer].append(val)
            
            print(f"\n  {group_name} — {metric_name}:")
            print(f"  {'Layer':<8} {'Mean':<10} {'Std':<10}")
            for layer in sorted(layer_vals.keys()):
                vals = layer_vals[layer]
                print(f"  L{layer:<7} {np.mean(vals):<10.4f} {np.std(vals):<10.4f}")
    
    # ---- 核心分析2: 跨组routing相似度 ----
    print("\n" + "=" * 50)
    print("分析2: 跨组routing topology相似度")
    print("如果computation policy是task-specific → 跨组routing应当低于组内")
    print("=" * 50)
    
    group_names = list(task_groups.keys())
    
    for metric_name, metric_fn in [
        ("Cosine Sim", compute_graph_similarity),
        ("Top-3 Routing Overlap", lambda p1, p2: compute_topk_routing_overlap(p1, p2, k=3)),
        ("Rank Correlation", compute_row_rank_correlation),
    ]:
        print(f"\n  {metric_name} — 跨组比较 (L6, 中层关键层):")
        
        # 选L6作为代表层
        target_layer = 6
        
        # 组内均值
        within_means = {}
        for g1 in group_names:
            task_names = list(task_groups[g1].keys())
            vals = []
            for i in range(len(task_names)):
                for j in range(i+1, len(task_names)):
                    result = metric_fn(
                        all_patterns[g1][task_names[i]][target_layer],
                        all_patterns[g1][task_names[j]][target_layer],
                    )
                    vals.append(np.mean(result))
            within_means[g1] = np.mean(vals) if vals else 0
        
        # 跨组均值
        cross_means = {}
        for i, g1 in enumerate(group_names):
            for j, g2 in enumerate(group_names):
                if i >= j:
                    continue
                t1 = list(task_groups[g1].keys())[0]
                t2 = list(task_groups[g2].keys())[0]
                result = metric_fn(
                    all_patterns[g1][t1][target_layer],
                    all_patterns[g2][t2][target_layer],
                )
                cross_means[f"{g1} vs {g2}"] = np.mean(result)
        
        print(f"  组内均值:")
        for g, v in within_means.items():
            print(f"    {g}: {v:.4f}")
        print(f"  跨组均值:")
        for pair, v in cross_means.items():
            print(f"    {pair}: {v:.4f}")
        
        # 关键判据: 组内 > 跨组 → computation policy是task-specific
        within_avg = np.mean(list(within_means.values()))
        cross_avg = np.mean(list(cross_means.values()))
        print(f"\n  ★ 组内均值: {within_avg:.4f}, 跨组均值: {cross_avg:.4f}")
        if within_avg > cross_avg:
            print(f"  ★ 组内 > 跨组 → computation policy有task-specific结构!")
            print(f"  ★ 差距: {within_avg - cross_avg:.4f}")
        else:
            print(f"  ★ 组内 ≈ 跨组 → routing可能只是position-driven, 没有task-specific policy")
    
    # ---- 核心分析3: Position-driven vs Task-driven 分离 ----
    print("\n" + "=" * 50)
    print("分析3: Position-driven vs Task-driven 分离")
    print("如果routing主要由位置决定 → 不同任务在相同位置应有相同routing")
    print("如果routing主要由任务决定 → 不同任务在相同位置应有不同routing")
    print("=" * 50)
    
    # 构造等长输入: "The [X] is [Y]" 格式
    position_controlled = {
        "capital_1": "The capital of France is",
        "capital_2": "The capital of Germany is",
        "capital_3": "The capital of Japan is",
        "capital_4": "The capital of Italy is",
        "capital_5": "The capital of Spain is",
    }
    
    # 同等长度但完全不同内容
    different_content = {
        "antonym_1": "The opposite of hot is",
        "antonym_2": "The opposite of big is",
        "antonym_3": "The opposite of fast is",
        "antonym_4": "The opposite of happy is",
        "antonym_5": "The opposite of light is",
    }
    
    pos_patterns = {}
    diff_patterns = {}
    
    print("  收集position-controlled patterns...")
    for name, text in position_controlled.items():
        patterns, _ = get_attention_patterns(model, text)
        pos_patterns[name] = patterns
    
    print("  收集different-content patterns...")
    for name, text in different_content.items():
        patterns, _ = get_attention_patterns(model, text)
        diff_patterns[name] = patterns
    
    # 在每个位置, 比较同组内和跨组的attention distribution
    for layer in [0, 3, 6, 9, 11]:
        # 同组内相似
        pos_sims = []
        pos_names = list(position_controlled.keys())
        for i in range(len(pos_names)):
            for j in range(i+1, len(pos_names)):
                cos = compute_graph_similarity(
                    pos_patterns[pos_names[i]][layer],
                    pos_patterns[pos_names[j]][layer],
                )
                pos_sims.append(np.mean(cos))
        
        # 跨组相似
        cross_sims = []
        diff_names = list(different_content.keys())
        for pn in pos_names[:3]:
            for dn in diff_names[:3]:
                cos = compute_graph_similarity(
                    pos_patterns[pn][layer],
                    diff_patterns[dn][layer],
                )
                cross_sims.append(np.mean(cos))
        
        print(f"\n  Layer {layer}:")
        print(f"    同类任务(position-controlled): {np.mean(pos_sims):.4f} ± {np.std(pos_sims):.4f}")
        print(f"    跨类任务(different-content):   {np.mean(cross_sims):.4f} ± {np.std(cross_sims):.4f}")
        
        diff = np.mean(pos_sims) - np.mean(cross_sims)
        print(f"    差距: {diff:.4f} ({'task-driven' if diff > 0.05 else 'position-driven'})")
    
    print("\n" + "=" * 50)
    print("实验A核心结论:")
    print("如果组内routing >> 跨组routing → computation policy是task-specific invariant")
    print("如果组内routing ≈ 跨组routing → routing是position/content-driven, 不是policy")
    print("=" * 50)


# ============================================================
# 实验B: Residual Transition Operator Invariant
# ============================================================

def exp_b_transition_operator(model):
    """
    核心问题: 不同输入通过同一层时, residual transition operator是否共享结构?
    
    方法:
      对每一层, 计算两个不同输入的Jacobian
      比较Jacobian的SVD结构 (top singular vectors方向)
      
    如果computation policy不变 → Jacobian的top singular vectors应当对齐
    即使具体hidden state完全不同
    """
    print("=" * 70)
    print("实验B: Residual Transition Operator Invariant")
    print("核心问题: 不同输入的层间transition operator是否共享结构?")
    print("=" * 70)
    
    # 准备多组输入
    test_pairs = {
        "same_task_diff_content": [
            ("2 + 3 =", "7 + 4 ="),
            ("The opposite of hot is", "The opposite of big is"),
            ("The capital of France is", "The capital of Germany is"),
        ],
        "diff_task": [
            ("2 + 3 =", "The opposite of hot is"),
            ("The capital of France is", "Translate to French: hello"),
            ("The opposite of hot is", "The cat sat on the"),
        ],
    }
    
    def get_residual_stream(model, text):
        """获取每层的residual stream"""
        tokens = model.to_tokens(text)
        _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
        
        residuals = {}
        for layer in range(model.cfg.n_layers):
            residuals[layer] = cache[f'blocks.{layer}.hook_resid_mid'].detach().cpu()
        
        # 最后位置的residual
        last_pos_residuals = {}
        for layer in range(model.cfg.n_layers):
            last_pos_residuals[layer] = residuals[layer][-1]  # [d_model]
        
        return last_pos_residuals, residuals
    
    def compute_layer_operator_structure(model, text1, text2, layer):
        """
        通过有限差分近似Jacobian, 比较其SVD结构
        
        方法: 对hidden state施加小扰动, 观察输出的变化方向
        两个输入的Jacobian如果共享top singular directions → computation policy相同
        """
        tokens1 = model.to_tokens(text1)
        tokens2 = model.to_tokens(text2)
        
        # 获取层输入
        _, cache1 = model.run_with_cache(tokens1, remove_batch_dim=True)
        _, cache2 = model.run_with_cache(tokens2, remove_batch_dim=True)
        
        h1 = cache1[f'blocks.{layer}.hook_resid_pre'][-1].clone()  # last position
        h2 = cache2[f'blocks.{layer}.hook_resid_pre'][-1].clone()
        
        # 获取层输出
        h1_out = cache1[f'blocks.{layer}.hook_resid_post'][-1].clone()
        h2_out = cache2[f'blocks.{layer}.hook_resid_post'][-1].clone()
        
        # 有限差分Jacobian近似: 用多个随机方向探测
        n_probes = 30
        eps = 0.01
        
        d_model = h1.shape[0]
        
        # 随机探测方向
        probe_dirs = torch.randn(n_probes, d_model)
        probe_dirs = probe_dirs / probe_dirs.norm(dim=1, keepdim=True)
        
        # 对text1计算Jacobian的近似列空间
        delta_outputs_1 = []
        for i in range(n_probes):
            h_perturbed = h1 + eps * probe_dirs[i]
            # 直接用hook注入perturbed input
            def make_hook(perturbed_input, results_list):
                def hook_fn(module, input, output):
                    # 替换resid_pre
                    pass
                return hook_fn
            
            # 简化: 用线性近似
            # J @ probe_dir ≈ (f(x + eps*dir) - f(x)) / eps
            # 但f(x)已经知道, 直接用residual = h_out - h_in
            residual1 = h1_out - h1
            # 线性近似: J ≈ (residual_direction)，但这不完全对
            # 更好的方法: 用input-output差分的投影
            
            # 实际上, 对于残差连接: h_out = h_in + attn(h_in) + mlp(h_in)
            # transition = attn(h_in) + mlp(h_in)
            # 我们比较的是: 不同输入的transition operator的"主方向"
            pass
        
        # 简化方法: 直接比较transition direction (h_out - h_in) 的方向
        transition1 = h1_out - h1
        transition2 = h2_out - h2
        
        # 归一化
        t1_norm = transition1 / (transition1.norm() + 1e-8)
        t2_norm = transition2 / (transition2.norm() + 1e-8)
        
        # 方向相似度
        direction_sim = torch.nn.functional.cosine_similarity(
            t1_norm.unsqueeze(0), t2_norm.unsqueeze(0)
        ).item()
        
        # Norm比
        norm_ratio = transition1.norm().item() / (transition2.norm().item() + 1e-8)
        
        return direction_sim, norm_ratio, transition1.norm().item(), transition2.norm().item()
    
    print("\n--- Transition Direction Analysis ---")
    print("比较: 不同输入通过同一层时, transition (h_out - h_in) 的方向是否对齐?")
    print("如果对齐 -> computation policy invariant (不同输入走同一条计算路径)")
    print("如果不对齐 -> 每个输入有独立的transition (无共享policy)")
    
    for pair_type, pairs in test_pairs.items():
        print(f"\n  {pair_type}:")
        print(f"  {'Pair':<40} {'Layer':<8} {'DirSim':<10} {'Norm1':<10} {'Norm2':<10} {'NormRatio':<10}")
        
        for text1, text2 in pairs:
            pair_label = f"{text1[:20]:>20} vs {text2[:20]:<20}"
            
            for layer in [0, 3, 6, 9, 11]:
                dir_sim, norm_ratio, n1, n2 = compute_layer_operator_structure(
                    model, text1, text2, layer
                )
                print(f"  {pair_label:<40} L{layer:<7} {dir_sim:<10.4f} {n1:<10.4f} {n2:<10.4f} {norm_ratio:<10.4f}")
    
    # ---- 更深入: 用多输入计算transition的主方向 ----
    print("\n--- Transition PCA Analysis ---")
    print("对同类任务收集所有transition vectors, PCA看是否落在低维子空间")
    print("如果低维 → computation policy有强约束 → invariant存在")
    
    for task_type, tasks in [
        ("addition", [f"{a} + {b} =" for a, b in 
                      [(2,3),(7,4),(15,23),(9,6),(11,8),(3,5),(12,7),(4,9),(21,14),(6,2),
                       (8,1),(5,5),(10,3),(13,6),(1,7),(2,8),(4,3),(6,4),(9,2),(3,7)]]),
        ("antonym", [f"The opposite of {w} is" for w in 
                      ["hot","big","fast","happy","light","strong","loud","rough","wide","tall",
                       "cold","small","slow","sad","dark","weak","quiet","smooth","narrow","short"]]),
        ("capital", [f"The capital of {c} is" for c in 
                      ["France","Germany","Japan","Italy","Spain","China","Brazil","India","Russia","Egypt",
                       "UK","Canada","Mexico","Korea","Turkey","Norway","Sweden","Poland","Greece","Portugal"]]),
    ]:
        for layer in [0, 3, 6, 9, 11]:
            transitions = []
            for text in tasks:
                tokens = model.to_tokens(text)
                _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
                h_in = cache[f'blocks.{layer}.hook_resid_pre'][-1]
                h_out = cache[f'blocks.{layer}.hook_resid_post'][-1]
                transition = (h_out - h_in).detach().cpu().numpy()
                transitions.append(transition)
            
            transitions = np.array(transitions)  # [n_tasks, d_model]
            
            # PCA
            from sklearn.decomposition import PCA
            pca = PCA()
            pca.fit(transitions)
            
            # 前3个PC解释的方差比
            var_explained = pca.explained_variance_ratio_[:3]
            cumulative = np.cumsum(var_explained)
            
            print(f"  {task_type} L{layer}: PC1={var_explained[0]:.4f}, PC1-2={cumulative[1]:.4f}, PC1-3={cumulative[2]:.4f}")
    
    print("\n" + "=" * 50)
    print("实验B核心结论:")
    print("如果transition vectors高度低维(PC1>0.8) → 强computation policy约束")
    print("如果transition vectors高维分散 → 无共享policy, 每个输入独立映射")
    print("=" * 50)


# ============================================================
# 实验C: Boundary-Conditioned Computation ★★★★★
# ============================================================

def exp_c_boundary_conditioned(model):
    """
    核心问题: prefix如何改变computation policy (不是hidden state)?
    
    Phase 78发现: boundary处hidden偏移最大
    但这可能是content-driven, 不一定说明computation policy变化
    
    关键升级: 不看hidden偏移, 看computation operator偏移
    → prefix是否改变了后续层的attention routing?
    → prefix是否改变了后续层的transition operator?
    
    方法: 
      同一content, 不同prefix
      比较: 
        1. prefix后的attention routing是否不同?
        2. 前缀是否只影响"参与哪些token", 还是影响"如何计算"?
    """
    print("=" * 70)
    print("实验C: Boundary-Conditioned Computation")
    print("核心问题: prefix改变的是content还是computation policy?")
    print("=" * 70)
    
    # ---- 设计1: 同一content, 不同prefix ----
    shared_content = "the cat sat on the mat"
    
    prefixes = {
        "translate_fr": "Translate to French:",
        "translate_de": "Translate to German:",
        "summarize": "Summarize:",
        "explain": "Explain:",
        "continue": "Continue the text:",
        "question": "Question about:",
        "reverse": "Reverse the letters of:",
        "rhyme": "Find a rhyme for:",
    }
    
    print("\n--- 分析1: Prefix对content部分attention routing的影响 ---")
    print("如果prefix改变了content部分的routing → prefix确实改变了computation policy")
    print("如果prefix只改变了自己的attention → prefix只是添加了context, 没改变policy")
    
    # 收集patterns
    all_patterns = {}
    all_caches = {}
    for pname, prefix in prefixes.items():
        text = f"{prefix} {shared_content}"
        patterns, cache = get_attention_patterns(model, text)
        all_patterns[pname] = patterns
        all_caches[pname] = cache
    
    # 比较content token位置的attention
    # 假设prefix长度不同, 需要对齐content位置
    prefix_lengths = {}
    for pname, prefix in prefixes.items():
        text = f"{prefix} {shared_content}"
        tokens = model.to_tokens(text)
        prefix_lengths[pname] = tokens.shape[1] - len(model.to_tokens(shared_content)[0])
    
    content_len = model.to_tokens(shared_content).shape[1]
    
    # 对每个层, 比较不同prefix下content位置的attention pattern
    print(f"\n  Content长度: {content_len} tokens")
    print(f"  Prefix长度: {prefix_lengths}")
    
    for layer in [0, 3, 6, 9, 11]:
        print(f"\n  Layer {layer} — content位置的attention routing比较:")
        
        # 取content部分每个位置, 比较不同prefix下的attention分布
        # 只看content的最后一个位置 (这是"决策点")
        
        # 各prefix在content最后位置的attention vector
        last_content_attns = {}
        for pname in prefixes:
            pat = all_patterns[pname][layer]  # [n_heads, seq_len, seq_len]
            pfx_len = prefix_lengths[pname]
            # 最后content位置的attention
            last_pos_attn = pat[:, -1, :]  # [n_heads, seq_len]
            last_content_attns[pname] = last_pos_attn
        
        # 两两比较不同prefix在content最后位置的routing
        pnames = list(prefixes.keys())
        pair_sims = []
        for i in range(len(pnames)):
            for j in range(i+1, len(pnames)):
                p1, p2 = pnames[i], pnames[j]
                attn1 = last_content_attns[p1]
                attn2 = last_content_attns[p2]
                
                # 对齐: 只比较content部分
                min_len = min(attn1.shape[1], attn2.shape[1])
                
                # 每个head的routing rank correlation
                head_corrs = []
                for h in range(attn1.shape[0]):
                    row1 = attn1[h, :min_len].float()
                    row2 = attn2[h, :min_len].float()
                    
                    rank1 = row1.argsort().argsort().float()
                    rank2 = row2.argsort().argsort().float()
                    
                    r1c = rank1 - rank1.mean()
                    r2c = rank2 - rank2.mean()
                    
                    denom = r1c.norm() * r2c.norm()
                    if denom > 1e-8:
                        corr = (r1c @ r2c) / denom
                        head_corrs.append(corr.item())
                
                pair_sims.append(np.mean(head_corrs))
        
        print(f"    不同prefix间routing rank correlation: {np.mean(pair_sims):.4f} ± {np.std(pair_sims):.4f}")
    
    # ---- 设计2: Prefix对transition operator的影响 ----
    print("\n--- 分析2: Prefix对content部分transition operator的影响 ---")
    
    for layer in [0, 3, 6, 9, 11]:
        transition_directions = {}
        
        for pname, prefix in prefixes.items():
            text = f"{prefix} {shared_content}"
            tokens = model.to_tokens(text)
            _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
            
            # Content最后位置的transition
            pfx_len = prefix_lengths[pname]
            h_in = cache[f'blocks.{layer}.hook_resid_pre'][-1]
            h_out = cache[f'blocks.{layer}.hook_resid_post'][-1]
            transition = h_out - h_in
            
            transition_directions[pname] = transition / (transition.norm() + 1e-8)
        
        # 两两比较transition方向
        pnames = list(prefixes.keys())
        dir_sims = []
        for i in range(len(pnames)):
            for j in range(i+1, len(pnames)):
                sim = torch.nn.functional.cosine_similarity(
                    transition_directions[pnames[i]].unsqueeze(0),
                    transition_directions[pnames[j]].unsqueeze(0),
                ).item()
                dir_sims.append(sim)
        
        print(f"  Layer {layer}: transition direction similarity = {np.mean(dir_sims):.4f} ± {np.std(dir_sims):.4f}")
    
    # ---- 设计3: 同类prefix vs 不同类prefix ----
    print("\n--- 分析3: 同类prefix vs 异类prefix的computation policy差异 ---")
    
    # 同类: translate_fr vs translate_de (都是翻译)
    # 异类: translate_fr vs summarize (不同任务)
    
    same_category_pairs = [
        ("translate_fr", "translate_de"),  # 都是翻译
    ]
    
    diff_category_pairs = [
        ("translate_fr", "summarize"),  # 翻译 vs 总结
        ("translate_fr", "continue"),   # 翻译 vs 续写
        ("translate_fr", "reverse"),    # 翻译 vs 反转
        ("summarize", "continue"),      # 总结 vs 续写
        ("translate_fr", "question"),   # 翻译 vs 提问
    ]
    
    for layer in [0, 3, 6, 9, 11]:
        print(f"\n  Layer {layer}:")
        
        # 同类prefix的routing similarity
        same_sims = []
        for p1, p2 in same_category_pairs:
            attn1 = all_patterns[p1][layer][:, -1, :]
            attn2 = all_patterns[p2][layer][:, -1, :]
            min_len = min(attn1.shape[1], attn2.shape[1])
            
            head_corrs = []
            for h in range(attn1.shape[0]):
                row1 = attn1[h, :min_len].float()
                row2 = attn2[h, :min_len].float()
                rank1 = row1.argsort().argsort().float()
                rank2 = row2.argsort().argsort().float()
                r1c = rank1 - rank1.mean()
                r2c = rank2 - rank2.mean()
                denom = r1c.norm() * r2c.norm()
                if denom > 1e-8:
                    head_corrs.append((r1c @ r2c / denom).item())
            same_sims.append(np.mean(head_corrs))
        
        # 异类prefix的routing similarity
        diff_sims = []
        for p1, p2 in diff_category_pairs:
            attn1 = all_patterns[p1][layer][:, -1, :]
            attn2 = all_patterns[p2][layer][:, -1, :]
            min_len = min(attn1.shape[1], attn2.shape[1])
            
            head_corrs = []
            for h in range(attn1.shape[0]):
                row1 = attn1[h, :min_len].float()
                row2 = attn2[h, :min_len].float()
                rank1 = row1.argsort().argsort().float()
                rank2 = row2.argsort().argsort().float()
                r1c = rank1 - rank1.mean()
                r2c = rank2 - rank2.mean()
                denom = r1c.norm() * r2c.norm()
                if denom > 1e-8:
                    head_corrs.append((r1c @ r2c / denom).item())
            diff_sims.append(np.mean(head_corrs))
        
        # Transition direction similarity
        same_dir_sims = []
        for p1, p2 in same_category_pairs:
            sim = torch.nn.functional.cosine_similarity(
                transition_directions.get(p1, torch.zeros(1)).unsqueeze(0) if p1 in transition_directions else torch.zeros(1).unsqueeze(0),
                transition_directions.get(p2, torch.zeros(1)).unsqueeze(0) if p2 in transition_directions else torch.zeros(1).unsqueeze(0),
            ).item() if p1 in transition_directions and p2 in transition_directions else 0
            same_dir_sims.append(sim)
        
        diff_dir_sims = []
        for p1, p2 in diff_category_pairs:
            sim = torch.nn.functional.cosine_similarity(
                transition_directions.get(p1, torch.zeros(1)).unsqueeze(0) if p1 in transition_directions else torch.zeros(1).unsqueeze(0),
                transition_directions.get(p2, torch.zeros(1)).unsqueeze(0) if p2 in transition_directions else torch.zeros(1).unsqueeze(0),
            ).item() if p1 in transition_directions and p2 in transition_directions else 0
            diff_dir_sims.append(sim)
        
        print(f"    同类routing sim: {np.mean(same_sims):.4f}, 异类: {np.mean(diff_sims):.4f}, 差距: {np.mean(same_sims)-np.mean(diff_sims):.4f}")
        if same_dir_sims and diff_dir_sims:
            print(f"    同类transition sim: {np.mean(same_dir_sims):.4f}, 异类: {np.mean(diff_dir_sims):.4f}, 差距: {np.mean(same_dir_sims)-np.mean(diff_dir_sims):.4f}")
    
    # ---- 设计4: Head-level analysis: 哪些heads的routing被prefix改变? ----
    print("\n--- 分析4: Head-level routing sensitivity to prefix ---")
    
    for layer in [3, 6, 9]:
        print(f"\n  Layer {layer} — 各head的prefix sensitivity:")
        
        pnames = list(prefixes.keys())
        head_sensitivity = []
        
        for h in range(model.cfg.n_heads):
            # 收集该head在不同prefix下的attention (content最后位置)
            attn_vectors = []
            for pname in pnames:
                pat = all_patterns[pname][layer]
                attn_vectors.append(pat[h, -1, :].float())
            
            # 计算这些向量间的平均两两cosine similarity
            pair_sims = []
            for i in range(len(attn_vectors)):
                for j in range(i+1, len(attn_vectors)):
                    min_len = min(attn_vectors[i].shape[0], attn_vectors[j].shape[0])
                    sim = torch.nn.functional.cosine_similarity(
                        attn_vectors[i][:min_len].unsqueeze(0),
                        attn_vectors[j][:min_len].unsqueeze(0),
                    ).item()
                    pair_sims.append(sim)
            
            avg_sim = np.mean(pair_sims)
            head_sensitivity.append((h, avg_sim))
        
        # 按sensitivity排序 (低sim = 高sensitivity)
        head_sensitivity.sort(key=lambda x: x[1])
        
        print(f"    最prefix-sensitive heads (routing被prefix大幅改变):")
        for h, sim in head_sensitivity[:4]:
            print(f"      Head {h}: avg cross-prefix sim = {sim:.4f}")
        print(f"    最prefix-insensitive heads (routing不受prefix影响):")
        for h, sim in head_sensitivity[-4:]:
            print(f"      Head {h}: avg cross-prefix sim = {sim:.4f}")
    
    print("\n" + "=" * 50)
    print("实验C核心结论:")
    print("如果同类prefix的routing >> 异类prefix → prefix确实conditioning computation policy")
    print("如果所有prefix的routing都相似 → prefix只影响content, 不影响policy")
    print("如果部分heads sensitive, 部分insensitive → 存在'policy control heads'")
    print("=" * 50)


# ============================================================
# 实验D: Head Interaction Graph Invariant
# ============================================================

def exp_d_head_interaction(model):
    """
    核心问题: 不同任务中, head之间的交互图是否保持不变?
    
    方法: 
      对每个任务, 计算每对head的"功能重叠度"
      (如果两个head对相同token位置贡献最大 → 功能重叠)
      
    如果head interaction graph在不同任务间保持 → computation policy invariant
    """
    print("=" * 70)
    print("实验D: Head Interaction Graph Invariant")
    print("核心问题: head之间的功能交互图是否在不同内容间保持不变?")
    print("=" * 70)
    
    # 多个任务
    tasks = {
        "add_1": "2 + 3 =",
        "add_2": "7 + 4 =",
        "add_3": "15 + 23 =",
        "add_4": "9 + 6 =",
        "add_5": "11 + 8 =",
        "antonym_1": "The opposite of hot is",
        "antonym_2": "The opposite of big is",
        "antonym_3": "The opposite of fast is",
        "capital_1": "The capital of France is",
        "capital_2": "The capital of Germany is",
        "capital_3": "The capital of Japan is",
    }
    
    # 收集每个head的contribution pattern
    # 使用attn.hook_result: 每个head对residual stream的贡献
    
    head_contributions = {}
    for task_name, text in tasks.items():
        tokens = model.to_tokens(text)
        _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
        
        # 对每层每head, 获取其输出向量
        contributions = {}
        for layer in range(model.cfg.n_layers):
            # 使用hook_attn_out获取attention输出, 再用z和W_O手动计算
            # hook_z shape: [seq_len, n_heads, d_head]
            z = cache[f'blocks.{layer}.attn.hook_z']  # [seq_len, n_heads, d_head]
            W_O = model.blocks[layer].attn.W_O  # [n_heads, d_head, d_model]
            
            # 计算每个head的输出: z @ W_O
            # z[-1] = [n_heads, d_head], W_O = [n_heads, d_head, d_model]
            z_last = z[-1].detach().cpu()  # [n_heads, d_head]
            W_O_cpu = W_O.detach().cpu()  # [n_heads, d_head, d_model]
            
            head_outputs = torch.zeros(z_last.shape[0], W_O_cpu.shape[2])
            for h in range(z_last.shape[0]):
                head_outputs[h] = z_last[h] @ W_O_cpu[h]  # [d_model]
            
            contributions[layer] = head_outputs  # [n_heads, d_model]
        
        head_contributions[task_name] = contributions
    
    # 计算head interaction graph
    # 定义: 两个head的"交互"= 它们贡献向量的cosine similarity
    # 如果cos > 0 → 功能协同; cos < 0 → 功能对抗; cos ≈ 0 → 功能独立
    
    for layer in [3, 6, 9]:
        print(f"\n  Layer {layer} — Head Interaction Graph:")
        
        # 对每个任务, 计算head interaction matrix
        task_graphs = {}
        for task_name in tasks:
            contribs = head_contributions[task_name][layer]  # [n_heads, d_model]
            n_heads = contribs.shape[0]
            
            # Head interaction matrix
            interaction = torch.zeros(n_heads, n_heads)
            for i in range(n_heads):
                for j in range(n_heads):
                    sim = torch.nn.functional.cosine_similarity(
                        contribs[i].unsqueeze(0), contribs[j].unsqueeze(0)
                    ).item()
                    interaction[i, j] = sim
            
            task_graphs[task_name] = interaction
        
        # 比较不同任务的interaction graph
        task_names = list(tasks.keys())
        
        # 同组内graph similarity
        add_graph_sims = []
        for i in range(5):
            for j in range(i+1, 5):
                sim = torch.nn.functional.cosine_similarity(
                    task_graphs[task_names[i]].flatten().unsqueeze(0),
                    task_graphs[task_names[j]].flatten().unsqueeze(0),
                ).item()
                add_graph_sims.append(sim)
        
        antonym_graph_sims = []
        for i in [5,6,7]:
            for j in range(i+1, 8):
                sim = torch.nn.functional.cosine_similarity(
                    task_graphs[task_names[i]].flatten().unsqueeze(0),
                    task_graphs[task_names[j]].flatten().unsqueeze(0),
                ).item()
                antonym_graph_sims.append(sim)
        
        capital_graph_sims = []
        for i in [8,9]:
            for j in range(i+1, 11):
                sim = torch.nn.functional.cosine_similarity(
                    task_graphs[task_names[i]].flatten().unsqueeze(0),
                    task_graphs[task_names[j]].flatten().unsqueeze(0),
                ).item()
                capital_graph_sims.append(sim)
        
        # 跨组graph similarity
        cross_graph_sims = []
        add_indices = [0,1,2,3,4]
        antonym_indices = [5,6,7]
        capital_indices = [8,9,10]
        
        for ai in add_indices[:2]:
            for bi in antonym_indices[:2]:
                sim = torch.nn.functional.cosine_similarity(
                    task_graphs[task_names[ai]].flatten().unsqueeze(0),
                    task_graphs[task_names[bi]].flatten().unsqueeze(0),
                ).item()
                cross_graph_sims.append(sim)
        
        for ai in add_indices[:2]:
            for ci in capital_indices[:2]:
                sim = torch.nn.functional.cosine_similarity(
                    task_graphs[task_names[ai]].flatten().unsqueeze(0),
                    task_graphs[task_names[ci]].flatten().unsqueeze(0),
                ).item()
                cross_graph_sims.append(sim)
        
        within_avg = np.mean(add_graph_sims + antonym_graph_sims + capital_graph_sims)
        cross_avg = np.mean(cross_graph_sims)
        
        print(f"    同组interaction graph sim: {within_avg:.4f}")
        print(f"    跨组interaction graph sim: {cross_avg:.4f}")
        print(f"    差距: {within_avg - cross_avg:.4f}")
        
        # 检查: 是否某些head pair的交互在所有任务中都保持?
        print(f"\n    稳定交互head pairs (在所有任务中交互符号一致):")
        n_heads = model.cfg.n_heads
        
        stable_count = 0
        total_pairs = 0
        for i in range(n_heads):
            for j in range(i+1, n_heads):
                signs = []
                for task_name in tasks:
                    sign = torch.sign(task_graphs[task_name][i, j]).item()
                    signs.append(sign)
                
                # 所有任务中符号一致
                if all(s == signs[0] for s in signs) and signs[0] != 0:
                    stable_count += 1
                total_pairs += 1
        
        print(f"      稳定pair数: {stable_count}/{total_pairs} ({stable_count/total_pairs*100:.1f}%)")
    
    print("\n" + "=" * 50)
    print("实验D核心结论:")
    print("如果head interaction graph在同组内稳定, 跨组不同 → computation policy是task-specific")
    print("如果head interaction graph在所有任务中都相似 → policy是universal")
    print("如果部分head pairs始终稳定 → 存在'computation backbone'")
    print("=" * 50)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="all", choices=["a", "b", "c", "d", "all"])
    args = parser.parse_args()
    
    print("Phase 79: Computation Invariants — 在content变化中寻找不变的计算策略")
    print("=" * 70)
    print("核心范式: content trajectory ≠ computation policy")
    print("关键问题: content改变时, 什么保持不变?")
    print("=" * 70)
    
    model = get_model()
    
    if args.exp in ["a", "all"]:
        exp_a_routing_topology(model)
    
    if args.exp in ["b", "all"]:
        exp_b_transition_operator(model)
    
    if args.exp in ["c", "all"]:
        exp_c_boundary_conditioned(model)
    
    if args.exp in ["d", "all"]:
        exp_d_head_interaction(model)
    
    print("\n\n" + "=" * 70)
    print("Phase 79 完成总结")
    print("=" * 70)
    print("""
三层分离:
  1. content trajectory — 最表层, 最容易发散
  2. representation trajectory — 中间层, 被content驱动
  3. computation policy — 最深层, 可能跨content稳定

关键判据:
  - routing topology invariant → task-specific computation policy存在
  - transition operator invariant → 层间计算结构跨内容共享
  - boundary-conditioned computation → prefix改变computation policy
  - head interaction graph invariant → 存在computation backbone

如果GPT-2的computation policy也是position-driven而非task-driven:
  → 说明真正的computation policy可能只存在于有能力的模型中
  → 下一步必须在Qwen3上验证
    """)
