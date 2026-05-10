"""
Phase 110: 计算拓扑分析 — 激活路由、注意力分叉与稀疏激活拓扑
=================================================================

Phase 109的硬伤 (用户批判):
  1. "方向主义" — 一直在找"方向", 但真正稳定的可能是"激活拓扑"
     翻译信号不是某个向量方向在传播, 而是某些neuron/head被条件化激活
     应该从 "哪个方向在传播" 转向 "哪些计算路径被激活"

  2. "hidden state是计算残留, 不是计算本身"
     h_l只是attention + MLP混合后的结果
     真正计算发生在: attention routing, MLP gating, neuron sparsity
     应该分析: 哪些head/neuron参与, 而不是向量朝哪个方向

  3. "假设全局连续流" — Transformer不是平滑流
     attention本质是竞争门控(softmax(QK))
     微小输入差异 → attention pattern突变 → 后续计算图改变
     更像"分叉计算图"而非"平滑轨迹"

  4. "表示空间 ≠ 计算空间"
     cosine, SVD, PR, CKA, angle可能都只是"计算阴影"
     应该转向: computation graph extraction

Phase 110核心升级:
  从"向量几何"到"计算拓扑"
  不再问"哪个方向在传播", 而是"哪些计算路径被激活"

关键实验:
  Exp 1: Attention Head Routing — 翻译vs中文的head激活差异
    核心: 翻译prompt和中文prompt在每层激活哪些attention head?
    问题: 是否存在"翻译专用head"或"翻译敏感head"?
    方法: 对每个head, 比较翻译prompt和中文prompt的attention pattern差异

  Exp 2: Attention Pattern Bifurcation — 注意力模式分叉
    核心: 微小prompt差异(猫的英文是 vs 猫在英文中叫)在哪层导致
          attention pattern突然分叉?
    这是真正的"分叉点" — 不是向量空间的分叉, 而是计算图的分叉
    方法: 逐步layer attention pattern相似度, 找到突然下降的层

  Exp 3: MLP Neuron Activation Routing — MLP神经元激活路由
    核心: SwiGLU的gate_proj产生稀疏激活
    哪些MLP neuron被翻译prompt激活, 而不被中文prompt激活?
    方法: 对gate_proj输出应用SiLU后的激活值, 找差分激活neuron

  Exp 4: Computation Path Classification — 计算路径分类
    核心: 综合attention + MLP的激活模式, 对不同prompt做路径分类
    不同语义域(动物/自然/颜色)是否走不同计算路径?
    翻译vs中文是否走不同计算路径?

Run:
  python tests/glm5/ccml_phase110_computation_topology.py --model qwen3 --exp 1
  python tests/glm5/ccml_phase110_computation_topology.py --model qwen3 --exp 2
  python tests/glm5/ccml_phase110_computation_topology.py --model qwen3 --exp 3
  python tests/glm5/ccml_phase110_computation_topology.py --model qwen3 --exp 4
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import gc
import json
import time
from collections import defaultdict
from scipy.linalg import subspace_angles

from model_utils import load_model, get_layers, get_model_info, release_model, get_W_U


# ============================================================
# 测试数据
# ============================================================
ANIMAL_PAIRS = [
    ("猫", "cat"), ("狗", "dog"), ("鱼", "fish"), ("鸟", "bird"),
    ("马", "horse"), ("牛", "cow"), ("羊", "sheep"), ("猪", "pig"),
    ("鸡", "chicken"), ("鸭", "duck"),
]

NATURE_PAIRS = [
    ("水", "water"), ("火", "fire"), ("风", "wind"), ("雨", "rain"),
    ("雪", "snow"), ("冰", "ice"), ("雷", "thunder"), ("雾", "fog"),
    ("霜", "frost"), ("云", "cloud"),
]

OBJECT_PAIRS = [
    ("花", "flower"), ("树", "tree"), ("石", "stone"), ("铁", "iron"),
    ("金", "gold"), ("茶", "tea"), ("沙", "sand"), ("草", "grass"),
    ("血", "blood"), ("光", "light"),
]

CELESTIAL_PAIRS = [
    ("月", "moon"), ("日", "sun"), ("星", "star"), ("河", "river"),
    ("山", "mountain"), ("海", "sea"), ("天", "sky"), ("地", "earth"),
    ("夜", "night"), ("昼", "day"),
]

ALL_PAIRS = ANIMAL_PAIRS + NATURE_PAIRS + OBJECT_PAIRS + CELESTIAL_PAIRS  # 40词对

COLOR_PAIRS = [
    ("红", "red"), ("蓝", "blue"), ("绿", "green"), ("白", "white"),
    ("黑", "black"),
]


def get_token_id(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[0] if ids else None


# ============================================================
# 核心工具: 收集attention pattern和MLP activation
# ============================================================
def collect_attention_and_mlp(model, tokenizer, device, prompt, n_layers, n_heads=32, head_dim=128):
    """收集单个prompt在每层的attention pattern和MLP gate activation
    
    Returns:
        attention_patterns: dict[l] = (n_heads, seq_len, seq_len) 的attention weights
        mlp_gate_activations: dict[l] = (intermediate_size,) 的gate activation值
        hidden_states: dict[l] = (d_model,) 的hidden state
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    seq_len = inputs["input_ids"].shape[1]
    
    attention_patterns = {}
    mlp_gate_activations = {}
    hidden_states = {}
    
    layers = get_layers(model)
    
    # 注册hook来捕获attention pattern和MLP gate activation
    hooks = []
    
    def make_attn_hook(l):
        def hook_fn(module, input, output):
            # output通常是 (hidden_states, attn_weights, past_key_value)
            # 但不同模型可能不同
            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]  # (batch, n_heads, seq_len, seq_len)
                if attn_weights is not None:
                    attention_patterns[l] = attn_weights[0].detach().float().cpu().numpy()
        return hook_fn
    
    def make_mlp_hook(l):
        def hook_fn(module, input, output):
            # 对于SwiGLU: gate = SiLU(gate_proj(x)), up = up_proj(x), out = gate * up
            # 我们需要gate activation (即SiLU之后)
            # 但这里只能获取MLP的最终输出
            # 需要在gate_proj的输出上加hook
            pass
        return hook_fn
    
    # 在self_attn上注册hook
    for l, layer in enumerate(layers):
        h = layer.self_attn.register_forward_hook(make_attn_hook(l))
        hooks.append(h)
    
    with torch.no_grad():
        outputs = model(inputs["input_ids"], output_hidden_states=True, output_attentions=True)
    
    # 移除hook
    for h in hooks:
        h.remove()
    
    # 收集hidden states
    for l in range(n_layers + 1):
        hidden_states[l] = outputs.hidden_states[l][0, -1, :].float().cpu().numpy()
    
    # 收集attentions (如果模型支持output_attentions)
    if outputs.attentions is not None:
        for l in range(len(outputs.attentions)):
            attention_patterns[l] = outputs.attentions[l][0].detach().float().cpu().numpy()
    
    return attention_patterns, mlp_gate_activations, hidden_states


def compute_mlp_gate_activations(model, tokenizer, device, prompt, n_layers):
    """单独计算MLP gate activation (用forward hook在gate_proj之后)
    
    SwiGLU: output = down_proj(SiLU(gate_proj(x)) * up_proj(x))
    我们要收集: SiLU(gate_proj(x)) 的值 → 这是稀疏激活模式
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    layers = get_layers(model)
    
    gate_activations = {}
    hooks = []
    
    def make_gate_hook(l):
        def hook_fn(module, input, output):
            # gate_proj的输出 → 应用SiLU → 这就是gate activation
            gate_act = torch.nn.functional.silu(output)
            # 取最后一个token的activation
            gate_activations[l] = gate_act[0, -1, :].detach().float().cpu().numpy()
        return hook_fn
    
    for l, layer in enumerate(layers):
        if hasattr(layer.mlp, 'gate_proj'):
            h = layer.mlp.gate_proj.register_forward_hook(make_gate_hook(l))
            hooks.append(h)
    
    with torch.no_grad():
        outputs = model(inputs["input_ids"])
    
    for h in hooks:
        h.remove()
    
    return gate_activations


def compute_up_activations(model, tokenizer, device, prompt, n_layers):
    """计算MLP up_proj的activation"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    layers = get_layers(model)
    
    up_activations = {}
    hooks = []
    
    def make_up_hook(l):
        def hook_fn(module, input, output):
            up_activations[l] = output[0, -1, :].detach().float().cpu().numpy()
        return hook_fn
    
    for l, layer in enumerate(layers):
        if hasattr(layer.mlp, 'up_proj'):
            h = layer.mlp.up_proj.register_forward_hook(make_up_hook(l))
            hooks.append(h)
    
    with torch.no_grad():
        outputs = model(inputs["input_ids"])
    
    for h in hooks:
        h.remove()
    
    return up_activations


# ============================================================
# Exp 1: Attention Head Routing — 翻译vs中文的head激活差异
# ============================================================
def exp1_attention_routing(args):
    model_name = args.model
    print(f"\n{'='*60}")
    print("Exp 1: Attention Head Routing — 翻译vs中文的head激活差异")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    n_heads = 32  # Qwen3-4B
    head_dim = 128

    # ========================================
    # A. 收集attention patterns — 翻译 vs 中文
    # ========================================
    print(f"\n  收集attention patterns...")
    
    test_pairs = ALL_PAIRS[:20]  # 用20个词对, 避免太慢
    
    # 存储每层每个head的attention pattern
    # key: (layer, head), value: list of attention vectors (last token attending to all tokens)
    zh_attn = defaultdict(list)   # 中文prompt的attention
    trans_attn = defaultdict(list)  # 翻译prompt的attention
    
    # 按语义域分类
    domain_attn = defaultdict(lambda: defaultdict(list))  # domain → (l, head) → patterns

    for i, (zh, en) in enumerate(test_pairs):
        # 确定语义域
        if (zh, en) in ANIMAL_PAIRS:
            domain = "animal"
        elif (zh, en) in NATURE_PAIRS:
            domain = "nature"
        elif (zh, en) in OBJECT_PAIRS:
            domain = "object"
        else:
            domain = "celestial"
        
        # 中文prompt
        zh_prompt = f"{zh}是一种"
        attn_zh, _, _ = collect_attention_and_mlp(model, tokenizer, device, zh_prompt, n_layers)
        
        # 翻译prompt
        trans_prompt = f'"{zh}"的英文是'
        attn_trans, _, _ = collect_attention_and_mlp(model, tokenizer, device, trans_prompt, n_layers)
        
        for l in range(n_layers):
            if l in attn_zh and l in attn_trans:
                # attn_zh[l]: (n_heads, seq_len_q, seq_len_k)
                # 取最后一个token对前面所有token的attention (最后一行)
                for h in range(n_heads):
                    # 中文prompt
                    zh_attn_weight = attn_zh[l][h, -1, :]  # (seq_len,)
                    zh_attn[(l, h)].append(zh_attn_weight)
                    domain_attn[domain][(l, h)].append(("zh", zh_attn_weight))
                    
                    # 翻译prompt
                    trans_attn_weight = attn_trans[l][h, -1, :]
                    trans_attn[(l, h)].append(trans_attn_weight)
                    domain_attn[domain][(l, h)].append(("trans", trans_attn_weight))
        
        if (i + 1) % 5 == 0:
            print(f"    已处理 {i+1}/{len(test_pairs)} 个词对")

    # ========================================
    # B. 分析: 翻译vs中文的head差异
    # ========================================
    print(f"\n  === 翻译vs中文的Attention Head差异 ===")
    print(f"  (按层统计: 差异最大的heads)")
    
    head_diff_by_layer = {}
    
    for l in range(n_layers):
        layer_diffs = []
        for h in range(n_heads):
            if (l, h) not in zh_attn or (l, h) not in trans_attn:
                continue
            
            zh_patterns = np.array(zh_attn[(l, h)])    # (n_samples, seq_len)
            trans_patterns = np.array(trans_attn[(l, h)])  # (n_samples, seq_len)
            
            # 平均attention pattern
            zh_mean = np.mean(zh_patterns, axis=0)
            trans_mean = np.mean(trans_patterns, axis=0)
            
            # 用JS散度衡量差异 (attention是概率分布)
            # 注意: 不同prompt长度不同, 不能直接比较
            # 改用: cosine similarity (与长度无关) 或 只比较公共长度部分
            def attn_similarity(p, q):
                """计算两个attention pattern的1-cosine距离"""
                n1 = np.linalg.norm(p)
                n2 = np.linalg.norm(q)
                if n1 < 1e-10 or n2 < 1e-10:
                    return 0.0
                return float(np.dot(p, q) / (n1 * n2))
            
            # 方法1: 用cosine similarity — 但不同prompt长度不同!
            # 解决: 只比较公共前缀的attention, 或用归一化的统计量
            
            # 统计量方法: 比较attention的熵和集中度
            def attn_entropy(p):
                """attention分布的熵"""
                p = np.clip(p, 1e-10, 1.0)
                p = p / np.sum(p)
                return -np.sum(p * np.log(p))
            
            def attn_max(p):
                """attention的最大值 (集中度)"""
                return np.max(p)
            
            def attn_sparsity(p, threshold=0.01):
                """attention的稀疏度 (>threshold的比例)"""
                return np.mean(p > threshold)
            
            zh_entropy = attn_entropy(zh_mean)
            trans_entropy = attn_entropy(trans_mean)
            entropy_diff = abs(zh_entropy - trans_entropy)
            
            zh_max = attn_max(zh_mean)
            trans_max = attn_max(trans_mean)
            max_diff = abs(zh_max - trans_max)
            
            zh_sparse = attn_sparsity(zh_mean)
            trans_sparse = attn_sparsity(trans_mean)
            sparse_diff = abs(zh_sparse - trans_sparse)
            
            # 综合差异度
            attn_diff = entropy_diff + max_diff + sparse_diff
            layer_diffs.append((h, attn_diff, zh_mean, trans_mean))
        
        # 按差异排序
        layer_diffs.sort(key=lambda x: x[1], reverse=True)
        head_diff_by_layer[l] = layer_diffs
        
        # 打印top-3 heads
        if l % 6 == 0 or l >= n_layers - 3:
            top3 = layer_diffs[:3]
            print(f"    L{l}: top-3 heads = ", end="")
            for h, ad, _, _ in top3:
                print(f"H{h}(diff={ad:.4f}) ", end="")
            print()

    # ========================================
    # C. 关键问题: 是否存在"翻译专用head"?
    # ========================================
    print(f"\n  === '翻译专用head'分析 ===")
    print(f"  如果某head的attn_diff始终很高 → 该head在翻译vs中文中角色不同")
    
    # 对每个head, 统计在多少层中attn_diff排top-5
    head_rank_count = defaultdict(int)
    for l in range(n_layers):
        if l in head_diff_by_layer:
            top5_heads = [h for h, _, _, _ in head_diff_by_layer[l][:5]]
            for h in top5_heads:
                head_rank_count[h] += 1
    
    # 排序
    head_rank_sorted = sorted(head_rank_count.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Head ID → 在多少层中排名top-5 (attn_diff差异):")
    for h, count in head_rank_sorted[:10]:
        print(f"    Head {h}: {count}次")

    # ========================================
    # D. 逐层attn_diff差异的层间变化
    # ========================================
    print(f"\n  === 逐层attn_diff差异(所有head平均) ===")
    mean_diff_by_layer = []
    for l in range(n_layers):
        if l in head_diff_by_layer:
            mean_d = np.mean([d for _, d, _, _ in head_diff_by_layer[l]])
            max_d = np.max([d for _, d, _, _ in head_diff_by_layer[l]])
            mean_diff_by_layer.append((l, mean_d, max_d))
            if l % 3 == 0 or l >= n_layers - 3:
                print(f"    L{l}: mean_diff={mean_d:.4f}, max_diff={max_d:.4f}")
    
    # 找差异突然增大的层 (可能是"分叉点")
    print(f"\n  === attn_diff差异的层间变化率 ===")
    for i in range(1, len(mean_diff_by_layer)):
        l_prev, mean_prev, _ = mean_diff_by_layer[i-1]
        l_curr, mean_curr, max_curr = mean_diff_by_layer[i]
        delta = mean_curr - mean_prev
        if abs(delta) > 0.005:
            direction = "↑" if delta > 0 else "↓"
            print(f"    L{l_prev}→L{l_curr}: Δmean_diff={delta:+.4f} {direction}")

    # ========================================
    # E. 语义域之间的attention差异
    # ========================================
    print(f"\n  === 语义域间attention差异 ===")
    domain_names = ["animal", "nature", "object", "celestial"]
    
    for l in [0, 6, 12, 18, 24, 30, n_layers - 1]:
        print(f"\n    L{l}:")
        for h in range(min(4, n_heads)):  # 只看前4个head
            domain_means = {}
            for domain in domain_names:
                patterns = [p for ptype, p in domain_attn[domain].get((l, h), []) if ptype == "zh"]
                if patterns:
                    domain_means[domain] = np.mean(patterns, axis=0)
            
            if len(domain_means) >= 2:
                # 计算域间cosine distance
                pairs_dist = []
                for d1 in domain_names:
                    for d2 in domain_names:
                        if d1 < d2 and d1 in domain_means and d2 in domain_means:
                            p = domain_means[d1]
                            q = domain_means[d2]
                            n1 = np.linalg.norm(p)
                            n2 = np.linalg.norm(q)
                            if n1 > 1e-10 and n2 > 1e-10:
                                cos = np.dot(p, q) / (n1 * n2)
                                pairs_dist.append(1 - cos)
                if pairs_dist:
                    print(f"      H{h}: mean inter-domain dist={np.mean(pairs_dist):.4f}")

    results = {
        "mean_diff_by_layer": [(int(l), float(m), float(mx)) for l, m, mx in mean_diff_by_layer],
        "head_rank_count": {str(k): int(v) for k, v in head_rank_sorted[:20]},
    }

    out_path = f"tests/glm5_temp/phase110_exp1_{model_name}_attn_routing.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  保存到 {out_path}")

    release_model(model)
    return results


# ============================================================
# Exp 2: Attention Pattern Bifurcation — 注意力分叉点
# ============================================================
def exp2_attention_bifurcation(args):
    model_name = args.model
    print(f"\n{'='*60}")
    print("Exp 2: Attention Pattern Bifurcation — 注意力模式分叉点")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    n_heads = 32

    # ========================================
    # A. 设计分叉测试组
    # ========================================
    print(f"\n  定义分叉测试组...")

    # 组1: 完全相同的语义, 不同的表述方式
    # "猫"的英文是 vs 猫在英文中叫 vs 请把猫翻译成英文
    bifurcation_groups = [
        {
            "name": "same_translation_different_phrasing",
            "prompts": [
                '"猫"的英文是',
                '猫在英文中叫',
                '请把猫翻译成英文',
            ],
            "description": "同一翻译任务的不同表述",
        },
        {
            "name": "different_words_same_task",
            "prompts": [
                '"猫"的英文是',
                '"狗"的英文是',
                '"鱼"的英文是',
            ],
            "description": "不同词的翻译任务",
        },
        {
            "name": "same_word_different_task",
            "prompts": [
                "猫是一种",        # 中文续写
                '"猫"的英文是',   # 翻译
                "猫的拼音是",      # 拼音
            ],
            "description": "同一词的不同任务",
        },
    ]

    # ========================================
    # B. 收集每组的attention patterns
    # ========================================
    print(f"\n  收集attention patterns...")

    group_attn = {}  # group_name → prompt_idx → layer → (n_heads, seq_len)

    for group in bifurcation_groups:
        gname = group["name"]
        group_attn[gname] = {}
        
        for pidx, prompt in enumerate(group["prompts"]):
            attn, _, _ = collect_attention_and_mlp(model, tokenizer, device, prompt, n_layers)
            group_attn[gname][pidx] = attn
            
            # 只打印层0和最后一层
            if 0 in attn:
                print(f"    {gname}[{pidx}] '{prompt}': L0 attn shape={attn[0].shape}")

    # ========================================
    # C. 分析: attention pattern在哪层开始分叉?
    # ========================================
    print(f"\n  === Attention Pattern分叉分析 ===")
    
    for group in bifurcation_groups:
        gname = group["name"]
        print(f"\n  --- {gname}: {group['description']} ---")
        
        # 计算每层中, 不同prompt之间的attention pattern相似度
        attns = group_attn[gname]
        n_prompts = len(group["prompts"])
        
        for l in range(n_layers):
            if l not in attns[0] or l not in attns[1]:
                continue
            
            # 每个prompt在该层的所有head的平均attention
            # attns[pidx][l]: (n_heads, seq_len_q, seq_len_k)
            # 取最后一个token的attention (最后一行)
            
            pairwise_diffs = []
            for p1 in range(n_prompts):
                for p2 in range(p1 + 1, n_prompts):
                    if l not in attns[p1] or l not in attns[p2]:
                        continue
                    
                    # 每个head的attention差异 (用统计量, 避免长度不匹配)
                    head_diffs = []
                    for h in range(n_heads):
                        a1 = attns[p1][l][h, -1, :]  # (seq_len,)
                        a2 = attns[p2][l][h, -1, :]
                        
                        # 用熵和集中度的差异 (与长度无关)
                        def attn_entropy(p):
                            p = np.clip(p, 1e-10, 1.0)
                            p = p / np.sum(p)
                            return -np.sum(p * np.log(p))
                        
                        def attn_max(p):
                            return np.max(p)
                        
                        e1, e2 = attn_entropy(a1), attn_entropy(a2)
                        m1, m2 = attn_max(a1), attn_max(a2)
                        
                        diff = abs(e1 - e2) + abs(m1 - m2)
                        head_diffs.append(diff)
                    
                    if head_diffs:
                        pairwise_diffs.append(np.mean(head_diffs))
            
            if pairwise_diffs and (l % 3 == 0 or l >= n_layers - 3):
                mean_diff = np.mean(pairwise_diffs)
                print(f"    L{l}: mean_attn_diff={mean_diff:.4f}")

    # ========================================
    # D. 更精细: 逐head分析分叉
    # ========================================
    print(f"\n  === 逐Head分叉分析 (same_word_different_task组) ===")
    print(f"  中文续写 vs 翻译, 每个head的attention统计量差异")
    
    gname = "same_word_different_task"
    attns = group_attn[gname]
    
    # prompt 0 = 中文续写, prompt 1 = 翻译
    for l in [0, 3, 6, 9, 12, 18, 24, 30, 33, 35]:
        if l not in attns[0] or l not in attns[1]:
            continue
        
        head_diffs = []
        for h in range(n_heads):
            a_zh = attns[0][l][h, -1, :]
            a_trans = attns[1][l][h, -1, :]
            
            # 用统计量差异 (与长度无关)
            def attn_entropy(p):
                p = np.clip(p, 1e-10, 1.0)
                p = p / np.sum(p)
                return -np.sum(p * np.log(p))
            def attn_max(p):
                return np.max(p)
            
            e_zh, e_trans = attn_entropy(a_zh), attn_entropy(a_trans)
            m_zh, m_trans = attn_max(a_zh), attn_max(a_trans)
            diff = abs(e_zh - e_trans) + abs(m_zh - m_trans)
            head_diffs.append((h, diff, e_zh, e_trans))
        
        # 找差异最大的heads
        head_diffs.sort(key=lambda x: x[1], reverse=True)
        
        if head_diffs:
            top5 = head_diffs[:5]
            print(f"    L{l}: mean_diff={np.mean([d for _, d, _, _ in head_diffs]):.4f}, "
                  f"最不同heads={[f'H{h}={d:.2f}' for h, d, _, _ in top5[:3]]}")

    results = {
        "bifurcation_groups": [g["name"] for g in bifurcation_groups],
    }

    out_path = f"tests/glm5_temp/phase110_exp2_{model_name}_attn_bifurcation.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  保存到 {out_path}")

    release_model(model)
    return results


# ============================================================
# Exp 3: MLP Neuron Activation Routing
# ============================================================
def exp3_mlp_activation_routing(args):
    model_name = args.model
    print(f"\n{'='*60}")
    print("Exp 3: MLP Neuron Activation Routing — MLP神经元激活路由")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    intermediate_size = model_info.intermediate_size  # 9728

    # ========================================
    # A. 收集MLP gate activations
    # ========================================
    print(f"\n  收集MLP gate activations (SwiGLU gate_proj → SiLU)...")
    print(f"  intermediate_size = {intermediate_size}")
    
    test_pairs = ALL_PAIRS[:20]  # 20个词对
    
    # 存储每层的gate activation
    zh_gate = defaultdict(list)    # 中文prompt
    trans_gate = defaultdict(list)  # 翻译prompt
    
    for i, (zh, en) in enumerate(test_pairs):
        # 中文prompt
        zh_prompt = f"{zh}是一种"
        gate_zh = compute_mlp_gate_activations(model, tokenizer, device, zh_prompt, n_layers)
        
        # 翻译prompt
        trans_prompt = f'"{zh}"的英文是'
        gate_trans = compute_mlp_gate_activations(model, tokenizer, device, trans_prompt, n_layers)
        
        for l in range(n_layers):
            if l in gate_zh:
                zh_gate[l].append(gate_zh[l])
            if l in gate_trans:
                trans_gate[l].append(gate_trans[l])
        
        if (i + 1) % 5 == 0:
            print(f"    已处理 {i+1}/{len(test_pairs)} 个词对")

    # ========================================
    # B. 分析: 翻译vs中文的neuron激活差异
    # ========================================
    print(f"\n  === MLP Neuron激活差异 (翻译vs中文) ===")
    
    neuron_diff_by_layer = {}
    
    for l in range(n_layers):
        if l not in zh_gate or l not in trans_gate:
            continue
        
        zh_data = np.array(zh_gate[l])    # (n_samples, intermediate_size)
        trans_data = np.array(trans_gate[l])  # (n_samples, intermediate_size)
        
        # 每个neuron的平均激活值
        zh_mean = np.mean(zh_data, axis=0)    # (intermediate_size,)
        trans_mean = np.mean(trans_data, axis=0)
        
        # 差分激活
        diff = trans_mean - zh_mean  # 正=翻译更激活, 负=中文更激活
        
        # 激活稀疏性: 有多少neuron被激活 (>0.1)
        zh_active = np.mean(zh_mean > 0.1) * 100
        trans_active = np.mean(trans_mean > 0.1) * 100
        
        # 差分激活的统计
        diff_positive = np.sum(diff > 0.01)   # 翻译更激活的neuron数
        diff_negative = np.sum(diff < -0.01)  # 中文更激活的neuron数
        diff_top10 = np.argsort(np.abs(diff))[-10:]  # 差异最大的10个neuron
        
        # 差分幅度
        diff_magnitude = np.mean(np.abs(diff))
        diff_max = np.max(np.abs(diff))
        
        neuron_diff_by_layer[l] = {
            "zh_active_pct": float(zh_active),
            "trans_active_pct": float(trans_active),
            "diff_positive_count": int(diff_positive),
            "diff_negative_count": int(diff_negative),
            "diff_magnitude": float(diff_magnitude),
            "diff_max": float(diff_max),
            "top10_diff_neurons": diff_top10.tolist(),
            "top10_diff_values": diff[diff_top10].tolist(),
        }
        
        if l % 6 == 0 or l >= n_layers - 3:
            print(f"    L{l}: zh_active={zh_active:.1f}%, trans_active={trans_active:.1f}%, "
                  f"diff_pos={diff_positive}, diff_neg={diff_negative}, "
                  f"diff_mag={diff_magnitude:.4f}, diff_max={diff_max:.4f}")

    # ========================================
    # C. 关键分析: "翻译专用neuron"跨层一致性
    # ========================================
    print(f"\n  === '翻译专用neuron'跨层一致性 ===")
    print(f"  如果某些neuron在多层中都对翻译更激活 → 这些neuron形成计算路径")
    
    # 找每层top-1%翻译差分neuron
    top_pct = 0.01
    top_n = max(1, int(intermediate_size * top_pct))
    
    layer_top_neurons = {}
    for l in range(n_layers):
        if l not in zh_gate or l not in trans_gate:
            continue
        zh_data = np.array(zh_gate[l])
        trans_data = np.array(trans_gate[l])
        zh_mean = np.mean(zh_data, axis=0)
        trans_mean = np.mean(trans_data, axis=0)
        diff = trans_mean - zh_mean
        top_neurons = set(np.argsort(diff)[-top_n:])
        layer_top_neurons[l] = top_neurons
    
    # 跨层overlap
    print(f"\n  相邻层top-{top_pct*100}%翻译差分neuron的overlap:")
    for l in range(0, n_layers - 1, 6):
        if l in layer_top_neurons and l + 1 in layer_top_neurons:
            overlap = len(layer_top_neurons[l] & layer_top_neurons[l + 1]) / top_n
            print(f"    L{l}→L{l+1}: overlap={overlap:.2%}")
    
    # L0 vs L36的overlap
    if 0 in layer_top_neurons and n_layers - 1 in layer_top_neurons:
        overlap = len(layer_top_neurons[0] & layer_top_neurons[n_layers - 1]) / top_n
        print(f"    L0→L{n_layers-1}: overlap={overlap:.2%}")

    # ========================================
    # D. 稀疏激活分析: 有多少neuron是"任务特异的"?
    # ========================================
    print(f"\n  === 任务特异neuron分析 ===")
    print(f"  定义: 在翻译prompt中激活(zh_mean>0.1), 但在中文prompt中不激活(zh_mean<0.05)")
    print(f"  或者反过来")
    
    for l in [0, 6, 12, 18, 24, 30, n_layers - 1]:
        if l not in zh_gate or l not in trans_gate:
            continue
        
        zh_data = np.array(zh_gate[l])
        trans_data = np.array(trans_gate[l])
        zh_mean = np.mean(zh_data, axis=0)
        trans_mean = np.mean(trans_data, axis=0)
        
        # 翻译特异: 翻译中激活, 中文中不激活
        trans_specific = np.sum((trans_mean > 0.1) & (zh_mean < 0.05))
        # 中文特异: 中文中激活, 翻译中不激活
        zh_specific = np.sum((zh_mean > 0.1) & (trans_mean < 0.05))
        # 共享: 都激活
        shared = np.sum((trans_mean > 0.1) & (zh_mean > 0.1))
        # 都不激活
        inactive = np.sum((trans_mean < 0.05) & (zh_mean < 0.05))
        
        total = len(zh_mean)
        print(f"    L{l}: trans_specific={trans_specific}({trans_specific/total*100:.1f}%), "
              f"zh_specific={zh_specific}({zh_specific/total*100:.1f}%), "
              f"shared={shared}({shared/total*100:.1f}%), "
              f"inactive={inactive}({inactive/total*100:.1f}%)")

    results = {
        "neuron_diff_by_layer": {str(k): v for k, v in neuron_diff_by_layer.items()},
    }

    out_path = f"tests/glm5_temp/phase110_exp3_{model_name}_mlp_routing.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  保存到 {out_path}")

    release_model(model)
    return results


# ============================================================
# Exp 4: Computation Path Classification
# ============================================================
def exp4_computation_path(args):
    model_name = args.model
    print(f"\n{'='*60}")
    print("Exp 4: Computation Path Classification — 计算路径分类")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    n_heads = 32
    intermediate_size = model_info.intermediate_size

    # ========================================
    # A. 收集完整计算指纹
    # ========================================
    print(f"\n  收集完整计算指纹 (attention + MLP gate)...")
    
    # 每个prompt的计算指纹 = 每层的(head attention pattern, gate activation)
    # 用这个指纹来分类不同类型的prompt
    
    test_data = {
        "animal_zh": [("猫是一种", "animal"), ("狗是一种", "animal"), ("鱼是一种", "animal"), ("鸟是一种", "animal")],
        "animal_trans": [('"猫"的英文是', "animal"), ('"狗"的英文是', "animal"), 
                        ('"鱼"的英文是', "animal"), ('"鸟"的英文是', "animal")],
        "nature_zh": [("水是一种", "nature"), ("火是一种", "nature"), ("风是一种", "nature")],
        "nature_trans": [('"水"的英文是', "nature"), ('"火"的英文是', "nature"), 
                        ('"风"的英文是', "nature")],
        "color_zh": [("红是一种", "color"), ("蓝是一种", "color"), ("绿是一种", "color")],
        "color_trans": [('"红"的英文是', "color"), ('"蓝"的英文是', "color"), 
                        ('"绿"的英文是', "color")],
    }
    
    # 收集计算指纹
    fingerprints = {}  # prompt_idx → layer → {"attn": array, "gate": array}
    labels = {}  # prompt_idx → (task_type, domain)
    
    idx = 0
    for group_name, items in test_data.items():
        task_type = "trans" if "trans" in group_name else "zh"
        domain = group_name.replace("_zh", "").replace("_trans", "")
        
        for prompt, _ in items:
            attn, _, _ = collect_attention_and_mlp(model, tokenizer, device, prompt, n_layers)
            gate = compute_mlp_gate_activations(model, tokenizer, device, prompt, n_layers)
            
            fingerprints[idx] = {"attn": attn, "gate": gate}
            labels[idx] = (task_type, domain)
            idx += 1
        
        print(f"    {group_name}: {len(items)} prompts")

    # ========================================
    # B. 在每层用计算指纹分类: 翻译vs中文
    # ========================================
    print(f"\n  === 各层计算指纹的翻译vs中文分类 ===")
    print(f"  用每层的attention pattern + MLP gate做分类")
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    
    for l in range(0, n_layers, 3):
        # 构建特征向量
        X_attn = []
        X_gate = []
        X_combined = []
        y_task = []  # 0=中文, 1=翻译
        y_domain = []  # 0=animal, 1=nature, 2=color
        
        for idx in fingerprints:
            fp = fingerprints[idx]
            task, domain = labels[idx]
            
            # Attention特征: 所有head的attention pattern拼起来
            if l in fp["attn"]:
                attn_flat = fp["attn"][l][:, -1, :].flatten()  # (n_heads * seq_len,)
                X_attn.append(attn_flat)
            else:
                X_attn.append(np.zeros(1))
            
            # MLP gate特征: top-100激活值
            if l in fp["gate"]:
                gate = fp["gate"][l]
                top100_idx = np.argsort(np.abs(gate))[-100:]
                gate_feat = np.zeros(100)
                gate_feat[:len(top100_idx)] = gate[top100_idx]
                X_gate.append(gate_feat)
            else:
                X_gate.append(np.zeros(100))
            
            # 合并
            if l in fp["attn"] and l in fp["gate"]:
                X_combined.append(np.concatenate([attn_flat, gate_feat]))
            else:
                X_combined.append(np.zeros(101))
            
            y_task.append(0 if task == "zh" else 1)
            domain_map = {"animal": 0, "nature": 1, "color": 2}
            y_domain.append(domain_map.get(domain, 0))
        
        X_attn = np.array(X_attn)
        X_gate = np.array(X_gate)
        X_combined = np.array(X_combined)
        y_task = np.array(y_task)
        y_domain = np.array(y_domain)
        
        # 分类: 翻译vs中文
        if len(set(y_task)) > 1 and X_combined.shape[1] > 1:
            try:
                lr = LogisticRegression(max_iter=1000, C=1.0)
                lr.fit(X_combined, y_task)
                acc_task = accuracy_score(y_task, lr.predict(X_combined))
            except:
                acc_task = -1
            
            try:
                lr_attn = LogisticRegression(max_iter=1000, C=1.0)
                lr_attn.fit(X_attn, y_task)
                acc_attn = accuracy_score(y_task, lr_attn.predict(X_attn))
            except:
                acc_attn = -1
            
            try:
                lr_gate = LogisticRegression(max_iter=1000, C=1.0)
                lr_gate.fit(X_gate, y_task)
                acc_gate = accuracy_score(y_task, lr_gate.predict(X_gate))
            except:
                acc_gate = -1
            
            if l % 6 == 0 or l >= n_layers - 3:
                print(f"    L{l}: task分类 — combined={acc_task:.2f}, attn={acc_attn:.2f}, gate={acc_gate:.2f}")

    # ========================================
    # C. 语义域分类 (只用中文prompt, 避免任务混淆)
    # ========================================
    print(f"\n  === 语义域分类 (只用中文prompt) ===")
    
    for l in range(0, n_layers, 6):
        X_gate = []
        y_dom = []
        
        for idx in fingerprints:
            task, domain = labels[idx]
            if task != "zh":
                continue
            
            if l in fingerprints[idx]["gate"]:
                gate = fingerprints[idx]["gate"][l]
                top100_idx = np.argsort(np.abs(gate))[-100:]
                gate_feat = np.zeros(100)
                gate_feat[:len(top100_idx)] = gate[top100_idx]
                X_gate.append(gate_feat)
                domain_map = {"animal": 0, "nature": 1, "color": 2}
                y_dom.append(domain_map.get(domain, 0))
        
        if len(set(y_dom)) > 1:
            X_gate = np.array(X_gate)
            y_dom = np.array(y_dom)
            try:
                lr = LogisticRegression(max_iter=1000, C=1.0)
                lr.fit(X_gate, y_dom)
                acc = accuracy_score(y_dom, lr.predict(X_gate))
                print(f"    L{l}: domain分类(gate) = {acc:.2f}")
            except:
                print(f"    L{l}: domain分类失败")

    results = {
        "n_prompts": len(fingerprints),
        "groups": list(test_data.keys()),
    }

    out_path = f"tests/glm5_temp/phase110_exp4_{model_name}_computation_path.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  保存到 {out_path}")

    release_model(model)
    return results


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3")
    parser.add_argument("--exp", type=int, default=1)
    args = parser.parse_args()

    if args.exp == 1:
        exp1_attention_routing(args)
    elif args.exp == 2:
        exp2_attention_bifurcation(args)
    elif args.exp == 3:
        exp3_mlp_activation_routing(args)
    elif args.exp == 4:
        exp4_computation_path(args)
    else:
        print(f"Unknown exp: {args.exp}")
