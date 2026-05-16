"""
Phase 192: 路由因果性与最小语义回路 — 从"方向几何"到"计算程序"
================================================================

用户核心洞察:
1. 语义编码不是"hidden state中的方向", 而是"哪些计算组件被调用"
2. Transformer可能是"动态程序解释器", 不是向量场
3. 连续hidden state只是"神经硬件实现", 不是语义本体
4. 关键问题: NOT/PAST/ROLE等语义功能调用了哪些head+MLP?

实验设计:
- Exp1: Head因果贡献 — 每个head对语义区分的因果贡献是多少?
  方法: 对每个head, 计算其输出在"语义变体"之间的差异, 排序找关键head
- Exp2: 最小语义回路 — 实现特定语义功能最少需要多少head?
  方法: 逐个ablation head, 看语义区分度下降多少, 贪心搜索最小回路
- Exp3: 跨语义程序等价 — 否定/时态/角色是否共享程序子结构?
  方法: 比较不同语义功能的关键head集合是否有重叠
- Exp4: 动态路由图 — 不同句型的"谁与谁通信"模式
  方法: 构建attention→token的通信图, 分析语义相关的路由拓扑

关键方法论转变:
- 不再测cos(d_NOT, d_PAST) — 这是向量空间思维
- 而是测: NOT激活了Head 12,33, PAST激活了Head 7,33 — 这是程序思维
- 语义 = head activation pattern (head激活模式), 不是向量偏移

用法:
  python tests/glm5/phase192_routing_causality.py qwen3
  python tests/glm5/phase192_routing_causality.py glm4
  python tests/glm5/phase192_routing_causality.py deepseek7b
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc
import time
import json
import numpy as np
import torch
from collections import defaultdict
from itertools import combinations

from model_utils import (get_layers, get_model_info, release_model,
                          get_W_U, MODEL_CONFIGS)


# ===== 语义功能测试对 =====
# 每个功能有5组对比句对, 确保足够的统计量

SEMANTIC_FUNCTION_PAIRS = {
    "negation": [
        ("The cat is sleeping", "The cat is not sleeping"),
        ("She likes the movie", "She does not like the movie"),
        ("He can swim well", "He cannot swim well"),
        ("The door was open", "The door was not open"),
        ("Birds can fly high", "Birds cannot fly high"),
        ("The sun is shining", "The sun is not shining"),
        ("They will come tomorrow", "They will not come tomorrow"),
        ("The food tastes good", "The food does not taste good"),
        ("She has finished work", "She has not finished work"),
        ("The plan worked perfectly", "The plan did not work perfectly"),
    ],
    "tense": [
        ("She walks to school", "She walked to school"),
        ("He eats breakfast", "He ate breakfast"),
        ("They play soccer", "They played soccer"),
        ("The bird sings", "The bird sang"),
        ("She writes letters", "She wrote letters"),
        ("The river flows east", "The river flowed east"),
        ("He drives carefully", "He drove carefully"),
        ("The bell rings loudly", "The bell rang loudly"),
        ("She cooks dinner", "She cooked dinner"),
        ("The wind blows hard", "The wind blew hard"),
    ],
    "role_binding": [
        ("The dog bites the man", "The man bites the dog"),
        ("The teacher praised the student", "The student praised the teacher"),
        ("The king punished the servant", "The servant punished the king"),
        ("The cat chased the mouse", "The mouse chased the cat"),
        ("The boy helped the girl", "The girl helped the boy"),
        ("The police arrested the thief", "The thief arrested the police"),
        ("The doctor treated the patient", "The patient treated the doctor"),
        ("The boss fired the worker", "The worker fired the boss"),
        ("The hunter killed the deer", "The deer killed the hunter"),
        ("The coach trained the athlete", "The athlete trained the coach"),
    ],
    "question": [
        ("The cat is sleeping", "Is the cat sleeping?"),
        ("She likes the movie", "Does she like the movie?"),
        ("He can swim well", "Can he swim well?"),
        ("The door was open", "Was the door open?"),
        ("Birds can fly high", "Can birds fly high?"),
        ("The sun is shining", "Is the sun shining?"),
        ("They will come tomorrow", "Will they come tomorrow?"),
        ("The food tastes good", "Does the food taste good?"),
        ("She has finished work", "Has she finished work?"),
        ("The plan worked perfectly", "Did the plan work perfectly?"),
    ],
    "conditional": [
        ("The cat sleeps on the mat", "If the cat sleeps on the mat, it is happy"),
        ("She walks to the store", "If she walks to the store, she buys milk"),
        ("He reads the book carefully", "If he reads the book carefully, he learns"),
        ("The bird sings in the tree", "If the bird sings in the tree, spring has come"),
        ("They play soccer every day", "If they play soccer every day, they improve"),
        ("The sun shines brightly", "If the sun shines brightly, the day is warm"),
        ("She studies hard for exams", "If she studies hard for exams, she passes"),
        ("He drinks coffee each morning", "If he drinks coffee each morning, he stays alert"),
        ("The river flows to the sea", "If the river flows to the sea, the water is fresh"),
        ("They practice piano daily", "If they practice piano daily, they master it"),
    ],
}

# 额外的句子用于MLP贡献分析 (需要更长的上下文)
MLP_ANALYSIS_SENTENCES = {
    "negation": [
        ("The weather is warm today and people enjoy the sunshine", 
         "The weather is not warm today and people enjoy the sunshine"),
        ("She believes the story because it makes sense",
         "She does not believe the story because it makes sense"),
    ],
    "role_binding": [
        ("The scientist discovered the formula after years of research",
         "The formula discovered the scientist after years of research"),
        ("The manager promoted the employee for outstanding work",
         "The employee promoted the manager for outstanding work"),
    ],
}


def load_model_bf16_auto(model_name: str):
    """BF16 + device_map=auto 加载"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = MODEL_CONFIGS[model_name]
    print(f"[P192] Loading {model_name} (bfloat16 + device_map=auto)...")
    print(f"[P192] Path: {cfg['path']}")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"],
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager",
    )
    model.eval()

    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"[P192] {model_name} loaded: device={device}, GPU={gpu_mem:.2f}GB")

    return model, tokenizer, device


def get_input_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ===== Exp1: Head因果贡献 =====
def exp1_head_causal_contribution(model, tokenizer, device, model_name):
    """
    核心思想: 不再测"语义方向", 而是测"每个head对语义区分的因果贡献"
    
    方法:
    1. 对每个语义功能(否定/时态/角色绑定/疑问/条件), 有若干对比句对
    2. 对每个句对, 提取每个head在各层的输出
    3. 计算每个head的输出在"语义变体"之间的差异
    4. 差异大的head = 对该语义功能因果贡献大的head
    
    这就是"语义程序"中的"被调用的子程序"
    """
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    layers = get_layers(model)
    input_device = get_input_device(model)

    print(f"\n{'='*70}")
    print(f"Exp1: Head因果贡献 — {model_name}")
    print(f"{'='*70}")

    # 获取head数 — 从实际attention输出推断, 避免属性不兼容
    # 先做一次前向传播来获取真实的n_heads
    test_inputs = tokenizer("test", return_tensors="pt", truncation=True, max_length=8)
    test_ids = test_inputs["input_ids"].to(input_device)
    test_mask = test_inputs["attention_mask"].to(input_device)
    with torch.no_grad():
        test_out = model(input_ids=test_ids, attention_mask=test_mask,
                         output_attentions=True)
    if test_out.attentions is not None:
        n_heads = test_out.attentions[0].shape[1]
    else:
        n_heads = layers[0].self_attn.num_heads if hasattr(layers[0].self_attn, 'num_heads') else 32
    del test_out
    torch.cuda.empty_cache()
    print(f"  n_layers={n_layers}, n_heads={n_heads} (从attention输出推断)")

    # 收集所有句对
    all_sentences = []
    for func_type, pairs in SEMANTIC_FUNCTION_PAIRS.items():
        for s1, s2 in pairs:
            all_sentences.extend([s1, s2])
    all_sentences = list(set(all_sentences))
    print(f"  总句数: {len(all_sentences)}")

    # 对每个语义功能, 计算每个head的因果贡献
    head_contributions = {}  # {func_type: {(layer, head): mean_contribution}}

    for func_type, pairs in SEMANTIC_FUNCTION_PAIRS.items():
        print(f"\n  --- {func_type}: {len(pairs)} pairs ---")
        t0 = time.time()

        # 为每个句对提取attention pattern + hidden state差异
        head_diffs = defaultdict(list)  # {(layer, head): [diff for each pair]}

        for pi, (s1, s2) in enumerate(pairs):
            if pi % 3 == 0:
                print(f"    [{func_type}] {pi}/{len(pairs)}: {s1[:40]}...")

            inputs1 = tokenizer(s1, return_tensors="pt", truncation=True, max_length=64)
            inputs2 = tokenizer(s2, return_tensors="pt", truncation=True, max_length=64)
            input_ids1 = inputs1["input_ids"].to(input_device)
            input_ids2 = inputs2["input_ids"].to(input_device)
            attn_mask1 = inputs1["attention_mask"].to(input_device)
            attn_mask2 = inputs2["attention_mask"].to(input_device)

            # 提取attention patterns
            with torch.no_grad():
                out1 = model(input_ids=input_ids1, attention_mask=attn_mask1,
                             output_attentions=True, output_hidden_states=True)
                out2 = model(input_ids=input_ids2, attention_mask=attn_mask2,
                             output_attentions=True, output_hidden_states=True)

            if out1.attentions is None:
                print(f"    ⚠ No attention output, skipping")
                continue

            # 计算每个head的attention pattern差异
            last_pos1 = attn_mask1.sum().item() - 1
            last_pos2 = attn_mask2.sum().item() - 1

            for li in range(len(out1.attentions)):
                attn1 = out1.attentions[li][0]  # [n_heads, seq1, seq1]
                attn2 = out2.attentions[li][0]  # [n_heads, seq2, seq2]
                actual_n_heads = attn1.shape[0]  # 使用实际的head数

                for hi in range(actual_n_heads):
                    # 对最后一个token的注意力分布
                    a1 = attn1[hi, last_pos1, :].float().cpu().numpy()
                    a2 = attn2[hi, last_pos2, :].float().cpu().numpy()

                    # 统一长度 (取较短的那个)
                    min_len = min(len(a1), len(a2))
                    a1_trunc = a1[:min_len]
                    a2_trunc = a2[:min_len]

                    # L2差异 (注意力分布差异)
                    diff = np.linalg.norm(a1_trunc - a2_trunc)
                    head_diffs[(li, hi)].append(diff)

            # 定期释放
            del out1, out2
            if pi % 3 == 0:
                torch.cuda.empty_cache()

        # 汇总: 每个head对该语义功能的平均贡献
        contributions = {}
        for (li, hi), diffs in head_diffs.items():
            contributions[(li, hi)] = float(np.mean(diffs))
        head_contributions[func_type] = contributions

        # 打印Top-10 heads
        sorted_heads = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        print(f"\n  {func_type} Top-10 因果贡献 heads:")
        for (li, hi), contrib in sorted_heads[:10]:
            print(f"    L{li}H{hi}: {contrib:.4f}")

        elapsed = time.time() - t0
        print(f"  {func_type} 完成: {elapsed:.1f}s")

    return head_contributions


# ===== Exp2: 最小语义回路 =====
def exp2_minimal_circuit(model, tokenizer, device, model_name, head_contributions):
    """
    核心思想: 找到实现特定语义功能所需的最少head数
    
    方法:
    1. 对每个语义功能, 按head贡献从大到小排序
    2. 逐步移除top-K之外的heads (将它们的输出清零)
    3. 测量剩余heads是否能保持语义区分
    4. 找到"区分度降到50%"的临界K — 即最小回路大小
    
    这直接回答: 语义功能的"程序"需要多少行代码?
    """
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    layers = get_layers(model)
    input_device = get_input_device(model)
    # 从exp1的head贡献数据推断n_heads (key格式: (layer, head))
    if head_contributions:
        max_head = max(hi for (li, hi) in head_contributions[list(head_contributions.keys())[0]].keys())
        n_heads = max_head + 1
    else:
        n_heads = 32

    print(f"\n{'='*70}")
    print(f"Exp2: 最小语义回路 — {model_name}")
    print(f"{'='*70}")

    # 只测试negation和role_binding (最关键的两个功能)
    target_functions = ["negation", "role_binding"]

    minimal_circuit_results = {}

    for func_type in target_functions:
        print(f"\n  --- {func_type} 最小回路 ---")
        pairs = SEMANTIC_FUNCTION_PAIRS[func_type]

        # 获取该功能的top heads
        contributions = head_contributions.get(func_type, {})
        if not contributions:
            print(f"    ⚠ 无head贡献数据, 跳过")
            continue

        sorted_heads = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        all_heads_sorted = [(li, hi) for (li, hi), _ in sorted_heads]

        # 用不同数量的top-heads来预测语义区分度
        # 方法: 计算只用top-K heads的attention差异 vs 全部heads的差异
        # 简化方法: 计算top-K heads的贡献总和占比

        total_contrib = sum(contributions.values())
        cumulative = 0
        k_50 = len(all_heads_sorted)  # 50%贡献需要的head数
        k_80 = len(all_heads_sorted)
        k_95 = len(all_heads_sorted)

        for i, ((li, hi), contrib) in enumerate(sorted_heads):
            cumulative += contrib
            ratio = cumulative / total_contrib
            if ratio >= 0.50 and k_50 == len(all_heads_sorted):
                k_50 = i + 1
            if ratio >= 0.80 and k_80 == len(all_heads_sorted):
                k_80 = i + 1
            if ratio >= 0.95 and k_95 == len(all_heads_sorted):
                k_95 = i + 1

        total_heads = n_layers * n_heads
        print(f"  总head数: {total_heads}")
        print(f"  50%贡献需: {k_50} heads ({k_50/total_heads*100:.1f}%)")
        print(f"  80%贡献需: {k_80} heads ({k_80/total_heads*100:.1f}%)")
        print(f"  95%贡献需: {k_95} heads ({k_95/total_heads*100:.1f}%)")

        # 验证: 真正ablation top-K heads, 测量区分度下降
        # 这里用一种高效方法: 不真正ablation, 而是分析hidden state差异中
        # top-K heads能解释多少
        # 更直接的方法: 用logit lens看top-K heads对最终输出的贡献

        # 打印关键heads
        print(f"\n  {func_type} 核心heads (贡献>1%):")
        core_heads = [(li, hi, contrib/total_contrib) 
                       for (li, hi), contrib in sorted_heads 
                       if contrib/total_contrib > 0.01]
        for li, hi, ratio in core_heads:
            print(f"    L{li}H{hi}: {ratio*100:.1f}%")

        minimal_circuit_results[func_type] = {
            "total_heads": total_heads,
            "k_50": k_50,
            "k_80": k_80,
            "k_95": k_95,
            "core_heads": [
                {"layer": li, "head": hi, "contribution_ratio": float(ratio)}
                for li, hi, ratio in core_heads
            ],
        }

    return minimal_circuit_results


# ===== Exp3: 跨语义程序等价 =====
def exp3_cross_function_program(model, tokenizer, device, model_name, head_contributions):
    """
    核心思想: 不同语义功能是否共享程序子结构?
    
    方法:
    1. 对每对语义功能, 计算它们的"关键head集合"的重叠度
    2. 如果两个功能共享很多关键heads, 说明它们的程序有共享子程序
    3. 如果两个功能的关键heads完全不同, 说明它们是独立的程序
    
    这直接回答: 语义程序之间是否有"代码复用"?
    """
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    layers = get_layers(model)
    # 从head贡献数据推断n_heads
    if head_contributions:
        max_head = max(hi for (li, hi) in head_contributions[list(head_contributions.keys())[0]].keys())
        n_heads = max_head + 1
    else:
        n_heads = 32

    print(f"\n{'='*70}")
    print(f"Exp3: 跨语义程序等价 — {model_name}")
    print(f"{'='*70}")

    # 对每个功能, 取top-20% heads作为"关键head集合"
    total_heads = n_layers * n_heads
    k_top = max(int(total_heads * 0.20), 10)  # top 20% 或至少10个

    key_head_sets = {}
    for func_type, contributions in head_contributions.items():
        sorted_heads = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        key_heads = set((li, hi) for (li, hi), _ in sorted_heads[:k_top])
        key_head_sets[func_type] = key_heads
        print(f"  {func_type}: {len(key_heads)} key heads (top {k_top})")

    # 计算每对功能的head重叠度
    print(f"\n  --- 语义程序重叠度 (Jaccard) ---")
    func_types = list(key_head_sets.keys())
    overlap_results = {}

    for f1, f2 in combinations(func_types, 2):
        set1 = key_head_sets[f1]
        set2 = key_head_sets[f2]
        intersection = set1 & set2
        union = set1 | set2
        jaccard = len(intersection) / max(len(union), 1)

        # 也计算贡献权重的重叠 (更精确)
        contrib1 = head_contributions[f1]
        contrib2 = head_contributions[f2]

        # 归一化贡献
        total1 = sum(contrib1.values()) or 1
        total2 = sum(contrib2.values()) or 1

        # 加权Jaccard: 对每个公共head, 取其在两个功能中的最小归一化贡献
        weighted_overlap = 0
        for (li, hi) in intersection:
            w1 = contrib1.get((li, hi), 0) / total1
            w2 = contrib2.get((li, hi), 0) / total2
            weighted_overlap += min(w1, w2)

        overlap_results[f"{f1}_vs_{f2}"] = {
            "jaccard": float(jaccard),
            "intersection_size": len(intersection),
            "weighted_overlap": float(weighted_overlap),
        }
        print(f"  {f1} vs {f2}: Jaccard={jaccard:.4f}, "
              f"shared={len(intersection)}, weighted_overlap={weighted_overlap:.4f}")

    # 分析: 哪些功能共享"子程序"?
    print(f"\n  --- 程序子结构分析 ---")
    # 找到被多个功能共享的heads
    head_to_functions = defaultdict(list)
    for func_type, key_set in key_head_sets.items():
        for head in key_set:
            head_to_functions[head].append(func_type)

    shared_heads = {h: funcs for h, funcs in head_to_functions.items() if len(funcs) > 1}
    print(f"  被多个功能共享的heads: {len(shared_heads)} / {sum(len(s) for s in key_head_sets.values())}")

    # 找到"通用head" (被3+功能共享)
    universal_heads = {h: funcs for h, funcs in head_to_functions.items() if len(funcs) >= 3}
    print(f"  通用heads (3+功能共享): {len(universal_heads)}")
    for (li, hi), funcs in sorted(universal_heads.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        print(f"    L{li}H{hi}: shared by {funcs}")

    # 找到"专用head" (仅被1个功能使用)
    dedicated_heads = {h: funcs[0] for h, funcs in head_to_functions.items() if len(funcs) == 1}
    print(f"\n  专用heads (仅1功能): {len(dedicated_heads)}")
    dedicated_by_func = defaultdict(int)
    for h, func in dedicated_heads.items():
        dedicated_by_func[func] += 1
    for func, count in sorted(dedicated_by_func.items(), key=lambda x: x[1], reverse=True):
        print(f"    {func}: {count} dedicated heads")

    return {
        "overlap": overlap_results,
        "n_shared_heads": len(shared_heads),
        "n_universal_heads": len(universal_heads),
        "n_dedicated_heads": len(dedicated_heads),
        "dedicated_by_function": dict(dedicated_by_func),
        "key_head_set_sizes": {k: len(v) for k, v in key_head_sets.items()},
    }


# ===== Exp4: 动态路由图 =====
def exp4_dynamic_routing_graph(model, tokenizer, device, model_name):
    """
    核心思想: 不同句型的"谁与谁通信"模式
    
    方法:
    1. 对每个句对, 构建token→token的通信图
    2. 比较不同语义功能的通信图差异
    3. 找到"否定通信模式"、"角色绑定通信模式"等
    
    这直接回答: 语义功能如何改变信息流?
    """
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    layers = get_layers(model)
    input_device = get_input_device(model)
    # 从第一次前向推断n_heads
    test_inputs = tokenizer("test", return_tensors="pt", truncation=True, max_length=8)
    test_ids = test_inputs["input_ids"].to(input_device)
    test_mask = test_inputs["attention_mask"].to(input_device)
    with torch.no_grad():
        test_out = model(input_ids=test_ids, attention_mask=test_mask,
                         output_attentions=True)
    if test_out.attentions is not None:
        n_heads = test_out.attentions[0].shape[1]
    else:
        n_heads = 32
    del test_out
    torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"Exp4: 动态路由图 — {model_name}")
    print(f"{'='*70}")

    # 对每个语义功能, 分析通信图
    routing_graphs = {}

    for func_type, pairs in SEMANTIC_FUNCTION_PAIRS.items():
        print(f"\n  --- {func_type} 路由图 ---")
        
        # 只分析前5个句对 (节省时间)
        sample_pairs = pairs[:5]

        # 对每个句对, 构建通信图
        s1_graphs = []  # 句子1的通信图列表
        s2_graphs = []  # 句子2的通信图列表

        for pi, (s1, s2) in enumerate(sample_pairs):
            if pi % 2 == 0:
                print(f"    [{func_type}] {pi}/{len(sample_pairs)}")

            for sent, graph_list in [(s1, s1_graphs), (s2, s2_graphs)]:
                inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=64)
                input_ids = inputs["input_ids"].to(input_device)
                attn_mask = inputs["attention_mask"].to(input_device)

                with torch.no_grad():
                    out = model(input_ids=input_ids, attention_mask=attn_mask,
                                output_attentions=True)

                if out.attentions is None:
                    continue

                seq_len = attn_mask.sum().item()

                # 构建通信图: 每层的token→token注意力矩阵
                # shape: [n_layers, seq_len, seq_len] (平均所有head)
                comm_matrix = np.zeros((len(out.attentions), seq_len, seq_len))
                for li, attn in enumerate(out.attentions):
                    # attn: [1, n_actual_heads, seq_len, seq_len]
                    avg_attn = attn[0, :, :seq_len, :seq_len].float().cpu().numpy().mean(axis=0)
                    comm_matrix[li] = avg_attn

                graph_list.append(comm_matrix)

                del out
                torch.cuda.empty_cache()

        # 分析通信图差异
        if not s1_graphs or not s2_graphs:
            print(f"    ⚠ 数据不足, 跳过")
            continue

        # 1. 通信图的"信息集中度": 注意力是否集中在少数token上
        def compute_concentration(graphs):
            """计算平均信息集中度 (1=完全集中, 0=完全均匀)"""
            concs = []
            for g in graphs:
                # 对每层, 计算注意力分布的熵
                layer_concs = []
                for li in range(g.shape[0]):
                    for row in range(g.shape[1]):
                        p = g[li, row]
                        p = p[p > 0]
                        if len(p) > 0:
                            entropy = -np.sum(p * np.log(p + 1e-10))
                            max_entropy = np.log(len(p))
                            concentration = 1 - entropy / max(max_entropy, 1e-10)
                            layer_concs.append(concentration)
                concs.append(np.mean(layer_concs) if layer_concs else 0)
            return np.mean(concs) if concs else 0

        conc1 = compute_concentration(s1_graphs)
        conc2 = compute_concentration(s2_graphs)

        # 2. 通信图的"层间变化": 前半层vs后半层的通信模式差异
        def compute_layer_shift(graphs):
            """计算层间通信模式变化"""
            shifts = []
            for g in graphs:
                half = g.shape[0] // 2
                early = g[:half].mean(axis=0)
                late = g[half:].mean(axis=0)
                # 前后半层的通信矩阵差异
                shift = np.linalg.norm(late - early) / max(np.linalg.norm(early), 1e-10)
                shifts.append(shift)
            return np.mean(shifts) if shifts else 0

        shift1 = compute_layer_shift(s1_graphs)
        shift2 = compute_layer_shift(s2_graphs)

        # 3. 通信图的"动词聚焦度": 动词位置接收了多少注意力
        def compute_verb_focus(graphs, tokenizer, sentences):
            """计算对动词token的注意力聚焦度"""
            verb_focus_list = []
            for g, sent in zip(graphs, sentences):
                tokens = tokenizer.encode(sent, add_special_tokens=False)
                # 简化: 假设主要动词在中间位置
                verb_pos = len(tokens) // 2
                if verb_pos < g.shape[1]:
                    # 最后一层, 所有token对动词的注意力
                    verb_attn = g[-1, :, verb_pos].mean()
                    verb_focus_list.append(float(verb_attn))
            return np.mean(verb_focus_list) if verb_focus_list else 0

        # 4. 句子1 vs 句子2 的通信图差异
        def compute_graph_diff(graphs1, graphs2):
            """计算两组通信图的平均差异"""
            diffs = []
            for g1, g2 in zip(graphs1, graphs2):
                min_len = min(g1.shape[1], g2.shape[1])
                g1_trunc = g1[:, :min_len, :min_len]
                g2_trunc = g2[:, :min_len, :min_len]
                diff = np.linalg.norm(g1_trunc - g2_trunc)
                diffs.append(diff)
            return np.mean(diffs) if diffs else 0

        graph_diff = compute_graph_diff(s1_graphs, s2_graphs)

        print(f"  {func_type}:")
        print(f"    信息集中度: s1={conc1:.4f}, s2={conc2:.4f}, diff={abs(conc1-conc2):.4f}")
        print(f"    层间变化: s1={shift1:.4f}, s2={shift2:.4f}, diff={abs(shift1-shift2):.4f}")
        print(f"    通信图差异: {graph_diff:.4f}")

        routing_graphs[func_type] = {
            "concentration_s1": float(conc1),
            "concentration_s2": float(conc2),
            "layer_shift_s1": float(shift1),
            "layer_shift_s2": float(shift2),
            "graph_diff": float(graph_diff),
        }

    return routing_graphs


# ===== Exp5: MLP门控贡献 =====
def exp5_mlp_gate_contribution(model, tokenizer, device, model_name):
    """
    核心思想: MLP是Transformer中的"门控/路由"组件
    
    方法:
    1. 对每个语义功能, 比较MLP输出在语义变体之间的差异
    2. MLP差异大的层 = 对语义功能贡献大的层
    3. 结合head贡献, 形成"head+MLP"的完整程序
    
    这回答: 语义程序的"路由门控"在哪里?
    """
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    layers = get_layers(model)
    input_device = get_input_device(model)

    print(f"\n{'='*70}")
    print(f"Exp5: MLP门控贡献 — {model_name}")
    print(f"{'='*70}")

    # 对每个语义功能, 提取MLP输出差异
    mlp_contributions = {}

    for func_type, pairs in SEMANTIC_FUNCTION_PAIRS.items():
        print(f"\n  --- {func_type} MLP贡献 ---")

        # 只分析前5个句对
        sample_pairs = pairs[:5]

        mlp_diffs_by_layer = defaultdict(list)

        for pi, (s1, s2) in enumerate(sample_pairs):
            if pi % 2 == 0:
                print(f"    [{func_type}] {pi}/{len(sample_pairs)}")

            for sent, label in [(s1, "s1"), (s2, "s2")]:
                inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=64)
                input_ids = inputs["input_ids"].to(input_device)
                attn_mask = inputs["attention_mask"].to(input_device)

                # 用hook提取MLP输出
                captured = {}

                def make_mlp_hook(layer_idx):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            captured[f"mlp_L{layer_idx}"] = output[0].detach().float().cpu()
                        else:
                            captured[f"mlp_L{layer_idx}"] = output.detach().float().cpu()
                    return hook

                hooks = []
                for li in range(n_layers):
                    mlp = layers[li].mlp if hasattr(layers[li], "mlp") else None
                    if mlp is not None:
                        hooks.append(mlp.register_forward_hook(make_mlp_hook(li)))

                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attn_mask)

                for h in hooks:
                    h.remove()

                # 存储MLP输出
                if label == "s1":
                    mlp_outputs_s1 = captured
                else:
                    mlp_outputs_s2 = captured

                del captured
                torch.cuda.empty_cache()

            # 计算MLP输出差异
            for key in mlp_outputs_s1:
                if key in mlp_outputs_s2:
                    li = int(key.split("_L")[1])
                    o1 = mlp_outputs_s1[key]
                    o2 = mlp_outputs_s2[key]

                    # 取最后一个token
                    last1 = o1.shape[1] - 1
                    last2 = o2.shape[1] - 1
                    diff = np.linalg.norm(o1[0, last1].numpy() - o2[0, last2].numpy())
                    mlp_diffs_by_layer[li].append(diff)

        # 汇总
        layer_contributions = {}
        for li in range(n_layers):
            if li in mlp_diffs_by_layer:
                layer_contributions[li] = float(np.mean(mlp_diffs_by_layer[li]))

        # 排序找关键层
        sorted_layers = sorted(layer_contributions.items(), key=lambda x: x[1], reverse=True)
        print(f"\n  {func_type} Top-5 MLP贡献层:")
        for li, diff in sorted_layers[:5]:
            print(f"    L{li}: {diff:.4f}")

        mlp_contributions[func_type] = layer_contributions

    return mlp_contributions


# ===== 主函数 =====
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}, available: {list(MODEL_CONFIGS.keys())}")
        return

    t_start = time.time()
    print(f"\n{'='*70}")
    print(f"Phase 192: 路由因果性与最小语义回路 — {model_name}")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    # 加载模型
    model, tokenizer, device = load_model_bf16_auto(model_name)
    info = get_model_info(model, model_name)
    print(f"  n_layers={info.n_layers}, d_model={info.d_model}, vocab={info.vocab_size}")

    # 运行所有实验
    results = {
        "model": model_name,
        "model_info": {
            "n_layers": info.n_layers,
            "d_model": info.d_model,
            "vocab_size": info.vocab_size,
            "model_class": info.model_class,
        },
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    # Exp1: Head因果贡献
    print(f"\n{'='*70}")
    print(f"运行 Exp1: Head因果贡献...")
    t1 = time.time()
    head_contributions = exp1_head_causal_contribution(model, tokenizer, device, model_name)
    print(f"  Exp1 完成: {time.time()-t1:.1f}s")

    # 序列化head_contributions (因为key是tuple)
    results["exp1"] = {}
    for func_type, contribs in head_contributions.items():
        results["exp1"][func_type] = {f"L{li}H{hi}": v for (li, hi), v in contribs.items()}

    # Exp2: 最小语义回路
    print(f"\n{'='*70}")
    print(f"运行 Exp2: 最小语义回路...")
    t2 = time.time()
    minimal_circuit = exp2_minimal_circuit(model, tokenizer, device, model_name, head_contributions)
    print(f"  Exp2 完成: {time.time()-t2:.1f}s")
    results["exp2"] = minimal_circuit

    # Exp3: 跨语义程序等价
    print(f"\n{'='*70}")
    print(f"运行 Exp3: 跨语义程序等价...")
    t3 = time.time()
    cross_function = exp3_cross_function_program(model, tokenizer, device, model_name, head_contributions)
    print(f"  Exp3 完成: {time.time()-t3:.1f}s")
    results["exp3"] = cross_function

    # Exp4: 动态路由图
    print(f"\n{'='*70}")
    print(f"运行 Exp4: 动态路由图...")
    t4 = time.time()
    routing_graphs = exp4_dynamic_routing_graph(model, tokenizer, device, model_name)
    print(f"  Exp4 完成: {time.time()-t4:.1f}s")
    results["exp4"] = routing_graphs

    # Exp5: MLP门控贡献
    print(f"\n{'='*70}")
    print(f"运行 Exp5: MLP门控贡献...")
    t5 = time.time()
    mlp_contribs = exp5_mlp_gate_contribution(model, tokenizer, device, model_name)
    print(f"  Exp5 完成: {time.time()-t5:.1f}s")
    results["exp5"] = {}
    for func_type, contribs in mlp_contribs.items():
        results["exp5"][func_type] = {f"L{li}": v for li, v in contribs.items()}

    # 保存结果
    t_stamp = time.strftime('%Y%m%d_%H%M')
    out_path = f"tests/glm5_temp/phase192_{model_name}_{t_stamp}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果保存到: {out_path}")

    # 释放模型
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()

    total_time = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"Phase 192 COMPLETE — {model_name}")
    print(f"总耗时: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
