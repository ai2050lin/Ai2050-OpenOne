"""
Phase 201: Circuit Dynamics & Latent Program Detection
========================================================

理论框架 (基于Phase 200修正):

  Phase 200的核心发现: 语言模式 = 内部路由结构的改变
  但解释有误:
  - A组(qa/cot/conditional)不是"句法编译", 而是"任务态切换"(task-state switching)
    → 它们共同要求模型改变未来生成策略(建立"未来轨迹控制")
  - B组(translation/coding/negation)不是"语义附加", 而是"token级变换/局部约束注入"
    → 它们不需要改变整个trajectory policy, 只做局部变换
  - 深层收敛不只是LayerNorm效应, 更是"输出协议收敛"
    → 后层正在"从潜在计算翻译成语言发射", 天然收敛

  核心理论修正:
  - LLM = 条件路由计算系统 (conditional routing computation system)
  - 不同prompt → 激活不同子计算流 (sub-computation flow)
  - 语言不是静态向量编码, 而是**动态回路编码**
  - "语义" = 激活了哪些计算路径 (prompt → route selection)
  - 内部存在"程序态": QA程序/推理程序/代码程序, 共享参数但激活不同子网络
  - 接近 mixture-of-computation 而非 mixture-of-experts

  ★关键警告★: 不要把传统语言学结构直接投影到神经结构
  - "conditional"在模型里可能是"未来分支控制器", 不是语法
  - "question"在模型里可能是"答案检索模式", 不是句法

Phase 201实验 (4个子实验):

Exp1: Head Co-Activation Graph (头共激活图)
  核心问题: 注意力头是否形成稳定的共激活团簇? 是否存在"程序特定"的头集成?
  方法: 提取各头的attention pattern, 构建共激活矩阵, 识别路由开关头
  如果存在离散的头集成 → 支持"潜在程序"假说

Exp2: Routing State Space Graph (路由状态空间图)
  核心问题: 路由空间的结构是什么? 向量空间? 离散图? 层次结构?
  方法: 计算所有mode对的层间路由距离, 构建距离矩阵, 分析图结构
  如果是层次结构 → 支持"任务态→子任务"的组织

Exp3: Circuit Reuse Algebra (回路复用代数) ★最关键★
  核心问题: R_{trans+code} ≈ R_{trans} ⊕ R_{code}?
  方法: 测试组合prompt的路由, 与单独prompt路由的和比较
  如果线性可加 → 路由是向量空间结构
  如果非线性 → 路由有离散程序交互

Exp4: Task-State vs First-Token Causal Test (任务态vs首token因果测试)
  核心问题: 路由由首token决定还是由任务需求决定?
  方法:
  - 条件A: 相同首token("The"), 不同任务需求(结尾添加指令)
  - 条件B: 不同首token, 相同任务需求(翻译/CoT的不同表达)
  如果任务决定路由 → 支持任务态切换理论
  如果首token决定路由 → 支持首token决定论

数据量 (加大!):
  - Qwen3: 30句 × 多条件
  - GLM4/DS7B: 15句 × 多条件

模型加载: bf16 + device_map="auto" + eager attention (需要output_attentions)
"""

import sys, os
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent.parent / "tests"))

import gc, time, json, math, warnings
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from model_demo_bf16 import load_model_bf16
from model_utils import get_model_info, release_model, get_layers

warnings.filterwarnings('ignore')


# ========================================================================
# 基础句子 — 30句, 比Phase 200增加50%
# ========================================================================
BASE_SENTENCES = [
    "The cat chases the dog",
    "The teacher helps the student",
    "The leader guides the team",
    "The doctor treats the patient",
    "The chef cooks the meal",
    "The writer drafts the letter",
    "The farmer plants the seed",
    "The artist paints the portrait",
    "The scientist discovers the element",
    "The engineer designs the bridge",
    "The judge delivers the verdict",
    "The soldier defends the fortress",
    "The musician composes the symphony",
    "The pilot flies the airplane",
    "The author writes the novel",
    "The builder constructs the house",
    "The driver operates the vehicle",
    "The guard protects the treasure",
    "The merchant trades the goods",
    "The hunter tracks the prey",
    "The baker prepares the bread",
    "The sailor navigates the ship",
    "The programmer writes the code",
    "The mechanic repairs the engine",
    "The librarian organizes the books",
    "The photographer captures the image",
    "The detective solves the mystery",
    "The architect plans the building",
    "The translator converts the text",
    "The analyst evaluates the data",
]

LITE_SENTENCES = BASE_SENTENCES[:15]


# ========================================================================
# Mode Prompt Templates — 修正版
# ========================================================================
# 注意: 不再用"句法/语义/语用"标签, 而用"任务态/token级"标签
MODE_PROMPTS = {
    # 默认态: 基础陈述句
    "normal": [
        lambda b: b,
    ],

    # ★任务态切换★ — 要求模型建立"未来轨迹控制"
    "qa": [  # 答案检索模式
        lambda b: "Does " + b[0].lower() + b[1:] + "?",
        lambda b: "What does " + b.split()[1] + " " + b.split()[0].lower() + " do?",
        lambda b: "Is it true that " + b[0].lower() + b[1:] + "?",
    ],
    "cot": [  # 多步潜在规划模式
        lambda b: "Problem: " + b[0].lower() + b[1:] + ". Think step by step.",
        lambda b: "Let's think step by step about " + b[0].lower() + b[1:] + ".",
        lambda b: b + " Please reason step by step.",
    ],
    "conditional": [  # 假设分支维持模式
        lambda b: "If " + b[0].lower() + b[1:] + ", then",
        lambda b: "Suppose that " + b[0].lower() + b[1:],
        lambda b: "Assuming " + b[0].lower() + b[1:] + ", what follows?",
    ],

    # ★Token级变换★ — 局部约束注入, 不改变整体轨迹策略
    "translation": [
        lambda b: "Translate to Chinese: " + b,
        lambda b: b + " in Chinese:",
        lambda b: "How do you say '" + b + "' in Chinese?",
    ],
    "coding": [
        lambda b: "Write code for: " + b[0].lower() + b[1:],
        lambda b: "def " + b.split()[1].lower() + "():",
        lambda b: "Implement a function that " + b[0].lower() + b[1:],
    ],
    "negation": [
        lambda b: b.replace(b.split()[1], "does not " + b.split()[1].lower(), 1) if len(b.split()) >= 2 else b,
        lambda b: "It is false that " + b[0].lower() + b[1:],
    ],

    # ★微扰★ — 最小计算改变
    "narrative": [
        lambda b: "Once upon a time, " + b[0].lower() + b[1:],
        lambda b: "In the story, " + b[0].lower() + b[1:],
    ],
}

# 用于Exp3的组合模式prompt
COMBINED_PROMPTS = {
    # B+B: token级+token级 → 预期最可加
    "trans_code": lambda b: "Translate to Chinese and write code for: " + b[0].lower() + b[1:],
    "trans_negation": lambda b: "Translate to Chinese: It is false that " + b[0].lower() + b[1:],

    # A+A: 任务态+任务态 → 预期最竞争(一个主导)
    "qa_cot": lambda b: "Does " + b[0].lower() + b[1:] + "? Think step by step.",
    "conditional_cot": lambda b: "If " + b[0].lower() + b[1:] + ", think step by step.",

    # A+B: 任务态+token级 → 预期层次性(任务态先, token级后)
    "coding_qa": lambda b: "Write a function that checks: does " + b[0].lower() + b[1:] + "?",
    "qa_translation": lambda b: "Translate to Chinese: Does " + b[0].lower() + b[1:] + "?",
}

# 用于Exp4的因果测试prompt
CAUSAL_CONDITION_A = {
    # 相同首token("The"), 不同任务在结尾
    "A1_normal": lambda b: b,
    "A2_trans_end": lambda b: b + ". Translate to Chinese.",
    "A3_cot_end": lambda b: b + ". Think step by step about this.",
    "A4_code_end": lambda b: b + ". Write code for this.",
}

CAUSAL_CONDITION_B = {
    # 不同首token, 相同任务(翻译)
    "B1_trans_start": lambda b: "Translate to Chinese: " + b,
    "B2_trans_end": lambda b: b + ". Translate to Chinese.",
    "B3_trans_question": lambda b: "How do you say '" + b + "' in Chinese?",
}

CAUSAL_CONDITION_C = {
    # 不同首token, 相同任务(CoT)
    "C1_cot_problem": lambda b: "Problem: " + b[0].lower() + b[1:] + ". Think step by step.",
    "C2_cot_end": lambda b: b + ". Think step by step about this.",
    "C3_cot_lets": lambda b: "Let's think step by step: " + b[0].lower() + b[1:] + ".",
}


# ========================================================================
# 工具函数
# ========================================================================

def compute_cosine_sim(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def compute_cosine_dist(v1, v2):
    return 1.0 - compute_cosine_sim(v1, v2)


def compute_euclidean_dist(v1, v2):
    return float(np.linalg.norm(v1 - v2))


def log_progress(exp_name, current, total, t_start, extra=""):
    elapsed = time.time() - t_start
    gpu = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"  [{exp_name}] {current}/{total} ({elapsed:.0f}s) GPU={gpu:.2f}GB {extra}")


def extract_hidden_states(model, tokenizer, device, text, n_layers, max_len=96):
    """提取所有层的hidden state (last token position)"""
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(device)
    with torch.no_grad():
        try:
            out = model(input_ids=ids, output_hidden_states=True)
        except Exception:
            attn_mask = torch.ones_like(ids)
            out = model(input_ids=ids, attention_mask=attn_mask, output_hidden_states=True)
    hs = out.hidden_states
    states = {}
    for li, h in enumerate(hs):
        if li > n_layers:
            break
        states[li] = h[0, -1, :].detach().float().cpu().numpy()
    return states


def extract_attention_patterns(model, tokenizer, device, text, n_layers, max_len=64):
    """提取各层各head的attention pattern (last token → all previous), 统一长度"""
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(device)
    with torch.no_grad():
        try:
            out = model(input_ids=ids, output_attentions=True)
        except Exception:
            attn_mask = torch.ones_like(ids)
            out = model(input_ids=ids, attention_mask=attn_mask, output_attentions=True)

    attns = out.attentions
    patterns = {}
    if attns is None:
        return patterns

    for li, attn in enumerate(attns):
        if li >= n_layers:
            break
        # last query position → all keys: (n_heads, seq_len)
        raw = attn[0, :, -1, :].detach().float().cpu().numpy()
        n_heads = raw.shape[0]
        actual_len = raw.shape[1]
        # Pad/truncate to max_len for consistent shape
        if actual_len >= max_len:
            patterns[li] = raw[:, -max_len:]
        else:
            padded = np.zeros((n_heads, max_len), dtype=raw.dtype)
            padded[:, -actual_len:] = raw
            patterns[li] = padded
    return patterns


def compute_routing_vector(states, sample_layers=None):
    """
    计算路由向量: 每层的残差贡献 r_l = h_l - h_{l-1}
    返回: {layer: routing_vector}
    """
    routing = {}
    layers = sorted(states.keys())
    for i in range(1, len(layers)):
        li = layers[i]
        li_prev = layers[i-1]
        if li - li_prev == 1:  # 只用相邻层
            routing[li] = states[li] - states[li_prev]
    return routing


# ========================================================================
# EXP1: Head Co-Activation Graph
# ========================================================================
def exp1_head_coactivation(model, tokenizer, device, info, sentences):
    """
    核心实验: 注意力头是否形成稳定的共激活团簇?

    方法:
    1. 提取各mode的attention pattern
    2. 对每个head, 计算其跨mode的"路由变化向量"
    3. 构建head间共激活矩阵: 哪些head一起变化?
    4. 识别"路由开关头" (routing switch heads): 跨mode变化最大的头
    5. 识别"不变头" (invariant heads): 跨mode变化最小的头
    """
    print("\n" + "="*60)
    print("Exp1: Head Co-Activation Graph")
    print("="*60)

    n_layers = info.n_layers
    modes = ["normal", "qa", "cot", "translation", "coding", "negation", "conditional"]

    # 采样层: 早/中/晚各取2层
    sample_layers = sorted(set([
        min(2, n_layers-1), min(5, n_layers-1),
        n_layers // 3, n_layers // 2,
        2 * n_layers // 3,
        max(0, n_layers - 4), max(0, n_layers - 1),
    ]))
    print(f"  Sample layers: {sample_layers}")
    print(f"  Modes: {modes}")

    # 收集: {mode: {layer: [attn_patterns]}}
    # 每个attn_pattern shape: (n_heads, seq_len)
    attn_data = {mode: {li: [] for li in sample_layers} for mode in modes}

    n_sents = min(len(sentences), 8)  # attention提取较慢, 限制句数
    t_start = time.time()
    total = n_sents * len(modes)

    for si, base in enumerate(sentences[:n_sents]):
        for mode in modes:
            template = MODE_PROMPTS[mode][0]
            text = template(base)
            patterns = extract_attention_patterns(model, tokenizer, device, text, n_layers)

            for li in sample_layers:
                if li in patterns:
                    attn_data[mode][li].append(patterns[li])

        log_progress("Exp1", si+1, n_sents, t_start)

    # === 分析 ===
    print("\n--- Exp1A: Head Routing Change (vs normal) ---")

    # 对每个层和每个head, 计算它跨mode的路由变化
    # routing_change[li][hi] = {mode: distance_from_normal}
    routing_changes = {}
    head_specialization = {}  # 每个head的总路由变化

    for li in sample_layers:
        normal_patterns = attn_data['normal'].get(li, [])
        if not normal_patterns:
            continue

        # 取normal的平均pattern
        normal_avg = np.mean(normal_patterns, axis=0)  # (n_heads, seq_len)
        n_heads = normal_avg.shape[0]

        routing_changes[li] = {}
        head_specialization[li] = []

        for hi in range(n_heads):
            changes = {}
            total_change = 0
            for mode in modes:
                if mode == 'normal':
                    changes[mode] = 0.0
                    continue
                mode_patterns = attn_data[mode].get(li, [])
                if not mode_patterns:
                    continue
                mode_avg = np.mean(mode_patterns, axis=0)
                if hi < mode_avg.shape[0]:
                    dist = compute_cosine_dist(normal_avg[hi], mode_avg[hi])
                    changes[mode] = dist
                    total_change += dist

            routing_changes[li][hi] = changes
            head_specialization[li].append((hi, total_change))

    # 输出: 每层最specialized的5个头
    for li in sample_layers:
        if li not in head_specialization:
            continue
        specs = sorted(head_specialization[li], key=lambda x: x[1], reverse=True)
        print(f"\n  Layer {li} — Top-5 routing switch heads (most changed across modes):")
        for hi, tc in specs[:5]:
            changes = routing_changes[li][hi]
            change_str = " ".join(f"{m}={changes.get(m,0):.3f}" for m in modes if m != 'normal')
            print(f"    Head {hi}: total_change={tc:.3f} | {change_str}")

        # 最不变的5个头
        print(f"  Layer {li} — Top-5 invariant heads (least changed across modes):")
        for hi, tc in specs[-5:]:
            changes = routing_changes[li][hi]
            change_str = " ".join(f"{m}={changes.get(m,0):.3f}" for m in modes if m != 'normal')
            print(f"    Head {hi}: total_change={tc:.3f} | {change_str}")

    # === Exp1B: Head Co-Activation (同一层内) ===
    print("\n--- Exp1B: Head Co-Activation Matrix ---")

    # 对每个层, 计算head间的路由变化相关性
    # 如果head_i和head_j跨mode的变化模式相似 → 它们共激活
    coactivation_results = {}

    for li in sample_layers:
        if li not in routing_changes:
            continue
        n_heads = len(routing_changes[li])
        if n_heads < 2:
            continue

        # 构建head × mode的路由变化矩阵
        mode_list = [m for m in modes if m != 'normal']
        change_matrix = np.zeros((n_heads, len(mode_list)))
        for hi in range(n_heads):
            for mi, mode in enumerate(mode_list):
                change_matrix[hi, mi] = routing_changes[li][hi].get(mode, 0)

        # 计算head间的相关矩阵
        if change_matrix.shape[0] > 1 and change_matrix.shape[1] > 0:
            # 标准化
            cm = change_matrix - change_matrix.mean(axis=1, keepdims=True)
            std = cm.std(axis=1, keepdims=True)
            std[std < 1e-10] = 1
            cm_norm = cm / std

            # 相关系数矩阵
            corr = np.corrcoef(change_matrix)

            # 找到最强共激活对
            n = corr.shape[0]
            pairs = []
            for i in range(n):
                for j in range(i+1, n):
                    if not np.isnan(corr[i, j]):
                        pairs.append((i, j, corr[i, j]))

            pairs.sort(key=lambda x: abs(x[2]), reverse=True)

            print(f"\n  Layer {li} — Top-10 co-activating head pairs:")
            for hi, hj, c in pairs[:10]:
                print(f"    Head {hi} ↔ Head {hj}: correlation={c:.4f}")

            # 找到最强反相关对(一个激活另一个抑制)
            anti_pairs = [p for p in pairs if p[2] < 0]
            if anti_pairs:
                print(f"  Layer {li} — Top-5 anti-correlated head pairs:")
                for hi, hj, c in sorted(anti_pairs, key=lambda x: x[2])[:5]:
                    print(f"    Head {hi} ↔ Head {hj}: correlation={c:.4f}")

            coactivation_results[li] = {
                'top_coactivated': [(hi, hj, float(c)) for hi, hj, c in pairs[:10]],
                'n_heads': n_heads,
                'avg_abs_corr': float(np.mean(np.abs(corr[np.triu_indices(n, k=1)]))),
            }

    # === Exp1C: 跨层路由头一致性 ===
    print("\n--- Exp1C: Cross-Layer Routing Head Consistency ---")

    # 在不同层, 同一个head_idx是否都是"路由开关头"?
    if len(sample_layers) >= 2:
        # 取每层top-5 specialized heads
        top_heads_per_layer = {}
        for li in sample_layers:
            if li in head_specialization:
                specs = sorted(head_specialization[li], key=lambda x: x[1], reverse=True)
                top_heads_per_layer[li] = set(hi for hi, _ in specs[:5])

        # 检查跨层重叠
        all_layers = sorted(top_heads_per_layer.keys())
        print(f"  Top-5 specialized heads per layer:")
        for li in all_layers:
            print(f"    L{li}: {sorted(top_heads_per_layer[li])}")

        # Jaccard相似度
        if len(all_layers) >= 2:
            for i in range(len(all_layers)-1):
                l1, l2 = all_layers[i], all_layers[i+1]
                s1, s2 = top_heads_per_layer[l1], top_heads_per_layer[l2]
                if s1 and s2:
                    jaccard = len(s1 & s2) / len(s1 | s2)
                    print(f"  Jaccard(L{l1}, L{l2}) = {jaccard:.3f}")

    return {
        'routing_changes': {str(li): {str(hi): changes for hi, changes in rc.items()}
                           for li, rc in routing_changes.items()},
        'coactivation': {str(li): v for li, v in coactivation_results.items()},
    }


# ========================================================================
# EXP2: Routing State Space Graph
# ========================================================================
def exp2_routing_state_space(model, tokenizer, device, info, sentences):
    """
    核心实验: 路由空间的结构是什么?

    方法:
    1. 对每种mode, 提取中间层和晚层的hidden state
    2. 计算所有mode对的层间路由距离
    3. 构建距离矩阵, 分析图结构
    4. 使用欧氏距离和余弦距离双重验证, 避免余弦距离的数学伪象
    """
    print("\n" + "="*60)
    print("Exp2: Routing State Space Graph")
    print("="*60)

    n_layers = info.n_layers
    modes = list(MODE_PROMPTS.keys())

    mid_layer = n_layers // 2
    late_layer = max(0, n_layers - 3)
    sample_layers = [mid_layer, late_layer]

    print(f"  Mid layer: {mid_layer}, Late layer: {late_layer}")

    # 收集: {mode: {layer: [hidden_states]}}
    activations = {mode: {li: [] for li in sample_layers} for mode in modes}

    t_start = time.time()
    total = len(sentences) * len(modes)

    for si, base in enumerate(sentences):
        for mode, templates in MODE_PROMPTS.items():
            text = templates[0](base)
            states = extract_hidden_states(model, tokenizer, device, text, n_layers)
            for li in sample_layers:
                if li in states:
                    activations[mode][li].append(states[li])

        if (si + 1) % 5 == 0:
            log_progress("Exp2", si+1, len(sentences), t_start)

    # === 分析: Mode-Mode距离矩阵 ===
    results = {}

    for li in sample_layers:
        print(f"\n--- Routing Distance Matrix at Layer {li} ---")

        # 计算mode centroids
        centroids = {}
        for mode in modes:
            acts = activations[mode][li]
            if acts:
                centroids[mode] = np.mean(acts, axis=0)

        # 余弦距离矩阵
        print(f"\n  Cosine distance (1-cos):")
        print(f"  {'':>12}", end="")
        for m in modes:
            print(f" {m[:7]:>7}", end="")
        print()

        cos_dist_matrix = {}
        for m1 in modes:
            print(f"  {m1[:12]:>12}", end="")
            for m2 in modes:
                if m1 in centroids and m2 in centroids:
                    d = compute_cosine_dist(centroids[m1], centroids[m2])
                else:
                    d = 0.0
                print(f" {d:7.3f}", end="")
                cos_dist_matrix[f"{m1}_vs_{m2}_L{li}"] = float(d)
            print()

        # 欧氏距离矩阵 (避免余弦距离伪象)
        print(f"\n  Euclidean distance (×0.01 for readability):")
        print(f"  {'':>12}", end="")
        for m in modes:
            print(f" {m[:7]:>7}", end="")
        print()

        euc_dist_matrix = {}
        for m1 in modes:
            print(f"  {m1[:12]:>12}", end="")
            for m2 in modes:
                if m1 in centroids and m2 in centroids:
                    d = compute_euclidean_dist(centroids[m1], centroids[m2])
                else:
                    d = 0.0
                print(f" {d*0.01:7.3f}", end="")
                euc_dist_matrix[f"{m1}_vs_{m2}_L{li}"] = float(d)
            print()

        # 范数分析 (检查余弦距离是否被范数差异扭曲)
        print(f"\n  Norm analysis:")
        for mode in modes:
            if mode in centroids:
                n = float(np.linalg.norm(centroids[mode]))
                acts = activations[mode][li]
                if len(acts) > 1:
                    std_n = float(np.std([np.linalg.norm(a) for a in acts]))
                else:
                    std_n = 0
                print(f"    {mode:<15} norm={n:.2f} ± {std_n:.2f}")

        results[f'cos_dist_L{li}'] = cos_dist_matrix
        results[f'euc_dist_L{li}'] = euc_dist_matrix

    # === 分析: 任务态 vs token级 的路由距离比较 ===
    print("\n--- Task-State vs Token-Level Routing Distance ---")

    task_state_modes = ["qa", "cot", "conditional"]
    token_level_modes = ["translation", "coding", "negation"]

    for li in sample_layers:
        # 任务态内部的平均距离
        ts_dists = []
        for i, m1 in enumerate(task_state_modes):
            for m2 in task_state_modes[i+1:]:
                k = f"{m1}_vs_{m2}_L{li}"
                if k in cos_dist_matrix:
                    ts_dists.append(cos_dist_matrix[k])

        # token级内部的平均距离
        tl_dists = []
        for i, m1 in enumerate(token_level_modes):
            for m2 in token_level_modes[i+1:]:
                k = f"{m1}_vs_{m2}_L{li}"
                if k in cos_dist_matrix:
                    tl_dists.append(cos_dist_matrix[k])

        # 任务态 vs token级之间的平均距离
        cross_dists = []
        for m1 in task_state_modes:
            for m2 in token_level_modes:
                k = f"{m1}_vs_{m2}_L{li}"
                if k in cos_dist_matrix:
                    cross_dists.append(cos_dist_matrix[k])

        print(f"\n  Layer {li}:")
        print(f"    Task-state internal avg:  {np.mean(ts_dists):.4f}" if ts_dists else "    N/A")
        print(f"    Token-level internal avg:  {np.mean(tl_dists):.4f}" if tl_dists else "    N/A")
        print(f"    Cross-group avg:          {np.mean(cross_dists):.4f}" if cross_dists else "    N/A")

        if ts_dists and tl_dists and cross_dists:
            # 如果cross > max(ts, tl) → 两组是分离的
            # 如果cross ≈ avg(ts, tl) → 两组是混合的
            ratio = np.mean(cross_dists) / max(np.mean(ts_dists), np.mean(tl_dists), 0.001)
            print(f"    Separation ratio:         {ratio:.2f}")
            if ratio > 1.3:
                print(f"    >>> GROUPS ARE SEPARATED in routing space")
            else:
                print(f"    >>> Groups overlap in routing space")

    return results


# ========================================================================
# EXP3: Circuit Reuse Algebra (★最关键★)
# ========================================================================
def exp3_circuit_reuse_algebra(model, tokenizer, device, info, sentences):
    """
    核心实验: R_{trans+code} ≈ R_{trans} ⊕ R_{code}?

    这是最关键的实验, 直接测试路由系统的数学结构:
    - 如果线性可加 → 路由是向量空间 → "路由方向"可以组合
    - 如果非线性 → 路由是离散程序 → 组合产生新程序

    方法:
    1. 对每个句子, 提取以下mode的hidden states:
       - normal, translation, coding, qa, cot
       - trans+code, qa+cot, trans+negation, conditional+cot, coding+qa, qa+translation
    2. 计算路由向量: r_l = h_l - h_{l-1}
    3. 对每个组合mode, 比较:
       - R_actual (组合prompt的路由)
       - R_predicted = R_A + R_B - R_normal (线性叠加预测)
       - R_A (单独A的路由)
       - R_B (单独B的路由)
    4. 如果 sim(R_actual, R_predicted) > max(sim(R_actual, R_A), sim(R_actual, R_B)):
       → 线性叠加比任一单独路由更接近 → 支持可加性
    """
    print("\n" + "="*60)
    print("Exp3: Circuit Reuse Algebra (CRITICAL)")
    print("="*60)

    n_layers = info.n_layers
    # 全层分析 (每隔几层采样)
    sample_step = max(1, n_layers // 12)
    sample_layers = sorted(set(list(range(0, n_layers+1, sample_step)) + [n_layers-1]))
    if 0 not in sample_layers:
        sample_layers = [0] + sample_layers

    print(f"  Sample layers: {sample_layers}")

    # 需要的单独mode和组合mode
    individual_modes = ["normal", "translation", "coding", "qa", "cot", "negation", "conditional"]
    combined_modes = list(COMBINED_PROMPTS.keys())

    # 组合定义: (combined_name, mode_A, mode_B)
    combinations = [
        ("trans_code", "translation", "coding"),       # B+B
        ("trans_negation", "translation", "negation"),  # B+B
        ("qa_cot", "qa", "cot"),                        # A+A
        ("conditional_cot", "conditional", "cot"),       # A+A
        ("coding_qa", "coding", "qa"),                  # B+A
        ("qa_translation", "qa", "translation"),        # A+B
    ]

    # 收集所有mode的hidden states
    # {sentence_idx: {mode: {layer: hidden_state}}}
    all_states = {}

    t_start = time.time()
    n_sents = len(sentences)
    total = n_sents * (len(individual_modes) + len(combined_modes))

    for si, base in enumerate(sentences):
        all_states[si] = {}

        # 单独mode
        for mode in individual_modes:
            text = MODE_PROMPTS[mode][0](base)
            states = extract_hidden_states(model, tokenizer, device, text, n_layers)
            all_states[si][mode] = states

        # 组合mode
        for comb_name, comb_fn in COMBINED_PROMPTS.items():
            text = comb_fn(base)
            states = extract_hidden_states(model, tokenizer, device, text, n_layers)
            all_states[si][comb_name] = states

        if (si + 1) % 3 == 0:
            log_progress("Exp3", si+1, n_sents, t_start,
                         f"({(si+1)*(len(individual_modes)+len(combined_modes))}/{total})")

    # === 分析: 线性叠加测试 ===
    print("\n--- Exp3A: Linearity Test (Layer-by-Layer) ---")

    linearity_results = {}

    for comb_name, mode_a, mode_b in combinations:
        print(f"\n  Combination: {comb_name} = {mode_a} + {mode_b}")

        # 在每个层, 测试线性叠加
        layer_linearity = []

        for li in sample_layers:
            if li == 0:
                continue  # L0没有残差贡献

            sims = {
                'predicted': [],
                'mode_a': [],
                'mode_b': [],
                'normal_a': [],
                'normal_b': [],
            }

            for si in all_states:
                s = all_states[si]
                # 需要所有相关mode在这一层都有数据
                needed = [comb_name, mode_a, mode_b, 'normal']
                if not all(m in s and li in s[m] and (li-1) in s[m] for m in needed):
                    continue

                # 计算路由向量
                r_combined = s[comb_name][li] - s[comb_name][li-1]
                r_a = s[mode_a][li] - s[mode_a][li-1]
                r_b = s[mode_b][li] - s[mode_b][li-1]
                r_normal = s['normal'][li] - s['normal'][li-1]

                # 线性叠加预测: R_combined_pred = R_a + R_b - R_normal
                r_predicted = r_a + r_b - r_normal

                # 余弦相似度
                sims['predicted'].append(compute_cosine_sim(r_combined, r_predicted))
                sims['mode_a'].append(compute_cosine_sim(r_combined, r_a))
                sims['mode_b'].append(compute_cosine_sim(r_combined, r_b))
                sims['normal_a'].append(compute_cosine_sim(r_a, r_normal))
                sims['normal_b'].append(compute_cosine_sim(r_b, r_normal))

            if sims['predicted']:
                avg = {k: float(np.mean(v)) for k, v in sims.items()}
                layer_linearity.append((li, avg))

        # 输出各层的线性度
        print(f"    {'Layer':>6} {'Predicted':>10} {'Mode_A':>10} {'Mode_B':>10} {'Winner':>10}")
        for li, avg in layer_linearity:
            pred = avg['predicted']
            a_sim = avg['mode_a']
            b_sim = avg['mode_b']
            best_single = max(a_sim, b_sim)
            winner = "LINEAR" if pred > best_single + 0.02 else \
                     mode_a.upper() if a_sim > b_sim + 0.02 else \
                     mode_b.upper() if b_sim > a_sim + 0.02 else "TIE"
            print(f"    {li:6d} {pred:10.4f} {a_sim:10.4f} {b_sim:10.4f} {winner:>10}")

        # 汇总
        if layer_linearity:
            avg_pred = np.mean([avg['predicted'] for _, avg in layer_linearity])
            avg_a = np.mean([avg['mode_a'] for _, avg in layer_linearity])
            avg_b = np.mean([avg['mode_b'] for _, avg in layer_linearity])

            print(f"\n    Summary for {comb_name}:")
            print(f"      Avg sim to predicted: {avg_pred:.4f}")
            print(f"      Avg sim to {mode_a}: {avg_a:.4f}")
            print(f"      Avg sim to {mode_b}: {avg_b:.4f}")

            if avg_pred > max(avg_a, avg_b) + 0.02:
                print(f"      >>> LINEAR SUPERPOSITION: routing is additive")
            elif avg_pred > max(avg_a, avg_b):
                print(f"      >>> WEAK LINEARITY: slightly better than single mode")
            else:
                dominant = mode_a if avg_a > avg_b else mode_b
                print(f"      >>> PROGRAM DOMINANCE: {dominant} program dominates")
                print(f"      >>> Combined mode follows {dominant} routing, not additive")

            linearity_results[comb_name] = {
                'avg_sim_predicted': float(avg_pred),
                'avg_sim_a': float(avg_a),
                'avg_sim_b': float(avg_b),
                'mode_a': mode_a,
                'mode_b': mode_b,
                'is_linear': bool(avg_pred > max(avg_a, avg_b) + 0.02),
                'dominant': mode_a if avg_a > avg_b else mode_b,
            }

    # === Exp3B: 组类型分析 (B+B vs A+A vs A+B) ===
    print("\n--- Exp3B: Combination Type Analysis ---")

    type_results = {"B+B": [], "A+A": [], "A+B": []}
    type_map = {
        "trans_code": "B+B", "trans_negation": "B+B",
        "qa_cot": "A+A", "conditional_cot": "A+A",
        "coding_qa": "A+B", "qa_translation": "A+B",
    }

    for comb_name, comb_type in type_map.items():
        if comb_name in linearity_results:
            r = linearity_results[comb_name]
            type_results[comb_type].append(r['avg_sim_predicted'])

    for ctype, vals in type_results.items():
        if vals:
            print(f"  {ctype}: avg linearity = {np.mean(vals):.4f} (n={len(vals)})")

    if type_results["B+B"] and type_results["A+A"]:
        bb = np.mean(type_results["B+B"])
        aa = np.mean(type_results["A+A"])
        print(f"\n  B+B vs A+A linearity: {bb:.4f} vs {aa:.4f}")
        if bb > aa + 0.03:
            print(f"  >>> Token-level combinations are MORE LINEAR than task-state combinations")
            print(f"  >>> This supports: token-level = additive direction, task-state = discrete program")
        elif aa > bb + 0.03:
            print(f"  >>> Task-state combinations are MORE LINEAR (unexpected!)")
        else:
            print(f"  >>> No significant difference between combination types")

    return linearity_results


# ========================================================================
# EXP4: Task-State vs First-Token Causal Test
# ========================================================================
def exp4_task_state_causal(model, tokenizer, device, info, sentences):
    """
    核心实验: 路由由首token决定还是由任务需求决定?

    这是Phase 200最大硬伤的直接解决:
    - Phase 200发现A组(qa/cot/conditional)首token不同, B组首token相同
    - 但无法确定: 分离是因为首token不同, 还是因为任务需求不同?

    条件A: 相同首token("The"), 不同任务在结尾
    → 如果路由不同 → 任务需求能改变路由, 不是首token决定论
    → 如果路由相同 → 首token决定了路由

    条件B: 不同首token, 相同任务(翻译)
    → 如果路由相同 → 任务需求决定路由, 首token不重要
    → 如果路由不同 → 首token影响了路由

    条件C: 不同首token, 相同任务(CoT)
    → 同上, 但针对任务态切换模式
    """
    print("\n" + "="*60)
    print("Exp4: Task-State vs First-Token Causal Test")
    print("="*60)

    n_layers = info.n_layers
    mid_layer = n_layers // 2

    # 收集所有条件的hidden states
    # {sentence_idx: {condition: {layer: hidden_state}}}
    all_states = {}

    # 所有条件
    conditions_a = list(CAUSAL_CONDITION_A.keys())
    conditions_b = list(CAUSAL_CONDITION_B.keys())
    conditions_c = list(CAUSAL_CONDITION_C.keys())
    all_conditions = conditions_a + conditions_b + conditions_c

    print(f"  Conditions A (same start, diff task): {conditions_a}")
    print(f"  Conditions B (diff start, same task=trans): {conditions_b}")
    print(f"  Conditions C (diff start, same task=cot): {conditions_c}")

    t_start = time.time()
    n_sents = len(sentences)

    for si, base in enumerate(sentences):
        all_states[si] = {}

        # 条件A
        for cond, fn in CAUSAL_CONDITION_A.items():
            text = fn(base)
            states = extract_hidden_states(model, tokenizer, device, text, n_layers)
            all_states[si][cond] = states

        # 条件B
        for cond, fn in CAUSAL_CONDITION_B.items():
            text = fn(base)
            states = extract_hidden_states(model, tokenizer, device, text, n_layers)
            all_states[si][cond] = states

        # 条件C
        for cond, fn in CAUSAL_CONDITION_C.items():
            text = fn(base)
            states = extract_hidden_states(model, tokenizer, device, text, n_layers)
            all_states[si][cond] = states

        if (si + 1) % 5 == 0:
            log_progress("Exp4", si+1, n_sents, t_start)

    # === 分析 ===

    # 条件A分析: 相同首token, 不同任务
    print("\n--- Condition A: Same first token ('The'), different task at end ---")
    print(f"  Mid layer = {mid_layer}")

    # A1(normal) vs A2(trans_end) vs A3(cot_end) vs A4(code_end)
    # 如果A2/A3/A4与A1不同 → 任务在结尾也能改变路由
    for cond_b in ['A2_trans_end', 'A3_cot_end', 'A4_code_end']:
        sims = []
        for si in all_states:
            if mid_layer in all_states[si].get('A1_normal', {}) and mid_layer in all_states[si].get(cond_b, {}):
                sim = compute_cosine_sim(
                    all_states[si]['A1_normal'][mid_layer],
                    all_states[si][cond_b][mid_layer]
                )
                sims.append(sim)
        if sims:
            avg = np.mean(sims)
            dist = 1 - avg
            task = cond_b.split('_', 1)[1]
            print(f"  A1(normal) vs {cond_b}: sim={avg:.4f}, dist={dist:.4f}")
            if dist > 0.05:
                print(f"    >>> Task '{task}' at end CHANGES routing (same first token!)")
            else:
                print(f"    >>> Task '{task}' at end does NOT change routing (first token dominates)")

    # 条件A内部距离矩阵
    print(f"\n  Condition A internal distance matrix (L{mid_layer}):")
    print(f"  {'':>18}", end="")
    for c in conditions_a:
        print(f" {c[:7]:>7}", end="")
    print()
    for c1 in conditions_a:
        print(f"  {c1[:18]:>18}", end="")
        for c2 in conditions_a:
            sims = []
            for si in all_states:
                if mid_layer in all_states[si].get(c1, {}) and mid_layer in all_states[si].get(c2, {}):
                    sims.append(compute_cosine_sim(
                        all_states[si][c1][mid_layer],
                        all_states[si][c2][mid_layer]
                    ))
            d = 1 - np.mean(sims) if sims else 0
            print(f" {d:7.3f}", end="")
        print()

    # 条件B分析: 不同首token, 相同任务(翻译)
    print("\n--- Condition B: Different first token, same task (translation) ---")

    # B1, B2, B3都是翻译, 但首token不同
    # 如果它们路由相似 → 任务决定路由
    # 如果它们路由不同 → 首token影响路由
    b_sims = {}
    for c1 in conditions_b:
        for c2 in conditions_b:
            if c1 >= c2:
                continue
            sims = []
            for si in all_states:
                if mid_layer in all_states[si].get(c1, {}) and mid_layer in all_states[si].get(c2, {}):
                    sims.append(compute_cosine_sim(
                        all_states[si][c1][mid_layer],
                        all_states[si][c2][mid_layer]
                    ))
            if sims:
                b_sims[f"{c1}_vs_{c2}"] = float(np.mean(sims))

    print(f"  Cross-start translation similarity (L{mid_layer}):")
    for k, v in b_sims.items():
        print(f"    {k}: sim={v:.4f}, dist={1-v:.4f}")

    avg_b_sim = np.mean(list(b_sims.values())) if b_sims else 0
    print(f"  Average cross-start similarity: {avg_b_sim:.4f}")
    if avg_b_sim > 0.9:
        print(f"  >>> Translation routing is CONSISTENT across different first tokens")
        print(f"  >>> TASK DEMAND determines routing, not first token")
    elif avg_b_sim > 0.7:
        print(f"  >>> Translation routing is PARTIALLY affected by first token")
        print(f"  >>> Both task demand and first token contribute to routing")
    else:
        print(f"  >>> Translation routing STRONGLY depends on first token")
        print(f"  >>> FIRST TOKEN dominates routing")

    # 条件C分析: 不同首token, 相同任务(CoT)
    print("\n--- Condition C: Different first token, same task (CoT) ---")

    c_sims = {}
    for c1 in conditions_c:
        for c2 in conditions_c:
            if c1 >= c2:
                continue
            sims = []
            for si in all_states:
                if mid_layer in all_states[si].get(c1, {}) and mid_layer in all_states[si].get(c2, {}):
                    sims.append(compute_cosine_sim(
                        all_states[si][c1][mid_layer],
                        all_states[si][c2][mid_layer]
                    ))
            if sims:
                c_sims[f"{c1}_vs_{c2}"] = float(np.mean(sims))

    print(f"  Cross-start CoT similarity (L{mid_layer}):")
    for k, v in c_sims.items():
        print(f"    {k}: sim={v:.4f}, dist={1-v:.4f}")

    avg_c_sim = np.mean(list(c_sims.values())) if c_sims else 0
    print(f"  Average cross-start similarity: {avg_c_sim:.4f}")
    if avg_c_sim > 0.9:
        print(f"  >>> CoT routing is CONSISTENT across different first tokens")
        print(f"  >>> TASK DEMAND determines routing for task-state modes too")
    elif avg_c_sim > 0.7:
        print(f"  >>> CoT routing is PARTIALLY affected by first token")
    else:
        print(f"  >>> CoT routing STRONGLY depends on first token")

    # === 关键综合判断 ===
    print("\n--- Exp4 Overall Judgment ---")

    # 从条件A提取"任务在结尾是否改变路由"
    a_dists = {}
    for cond_b in ['A2_trans_end', 'A3_cot_end', 'A4_code_end']:
        sims = []
        for si in all_states:
            if mid_layer in all_states[si].get('A1_normal', {}) and mid_layer in all_states[si].get(cond_b, {}):
                sims.append(1 - compute_cosine_sim(
                    all_states[si]['A1_normal'][mid_layer],
                    all_states[si][cond_b][mid_layer]
                ))
        if sims:
            a_dists[cond_b] = np.mean(sims)

    avg_a_dist = np.mean(list(a_dists.values())) if a_dists else 0

    print(f"\n  Condition A (same start, task at end): avg routing change = {avg_a_dist:.4f}")
    print(f"  Condition B (diff start, same task=trans): avg cross-start sim = {avg_b_sim:.4f}")
    print(f"  Condition C (diff start, same task=cot): avg cross-start sim = {avg_c_sim:.4f}")

    # 判断
    task_shifts_routing = avg_a_dist > 0.03  # 任务在结尾能改变路由
    task_determines_routing = avg_b_sim > 0.85 and avg_c_sim > 0.85  # 不同首token但路由相似

    print(f"\n  Verdict:")
    if task_shifts_routing and task_determines_routing:
        print(f"  >>> TASK DEMAND determines routing")
        print(f"  >>> First token has SOME effect but task demand is primary")
        print(f"  >>> Supports 'task-state switching' theory, NOT 'first-token determinism'")
    elif task_shifts_routing and not task_determines_routing:
        print(f"  >>> BOTH task demand and first token affect routing")
        print(f"  >>> Task demand can override first-token effects")
        print(f"  >>> Partial support for both theories")
    elif not task_shifts_routing and task_determines_routing:
        print(f"  >>> CONTRADICTION: task at end doesn't change routing, but diff start has same routing?")
        print(f"  >>> Need more data to resolve")
    else:
        print(f"  >>> FIRST TOKEN is the primary determinant of routing")
        print(f"  >>> Task demand at end has minimal effect")
        print(f"  >>> Supports 'first-token determinism'")

    return {
        'condition_a_dists': {k: float(v) for k, v in a_dists.items()},
        'condition_b_cross_sim': float(avg_b_sim),
        'condition_c_cross_sim': float(avg_c_sim),
        'task_shifts_routing': bool(task_shifts_routing),
        'task_determines_routing': bool(task_determines_routing),
    }


# ========================================================================
# MAIN
# ========================================================================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    is_lite = model_name != "qwen3"

    t0 = time.time()
    print(f"[Phase 201] Circuit Dynamics & Latent Program Detection — {model_name}")
    print(f"[Phase 201] Time: {datetime.now()}")
    print(f"[Phase 201] Lite mode: {is_lite}")
    print(f"[Phase 201] Theory: LLM = conditional routing computation system")
    print(f"[Phase 201] Key test: R_trans+code ≈ R_trans ⊕ R_code?")

    # Load model
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    print(f"[load] {model_name}: {info.model_class}, {info.n_layers}L, d={info.d_model}")

    # Config
    sentences = LITE_SENTENCES if is_lite else BASE_SENTENCES
    print(f"  Sentences: {len(sentences)}")

    # Run experiments
    print(f"\n{'='*60}")
    print("Starting Phase 201 experiments...")
    print(f"{'='*60}")

    exp1_results = exp1_head_coactivation(model, tokenizer, device, info, sentences)
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    exp2_results = exp2_routing_state_space(model, tokenizer, device, info, sentences)
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    exp3_results = exp3_circuit_reuse_algebra(model, tokenizer, device, info, sentences)
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    exp4_results = exp4_task_state_causal(model, tokenizer, device, info, sentences)

    # ===== Final Summary =====
    print("\n" + "="*60)
    print(f"PHASE 201 SUMMARY — {model_name}")
    print("="*60)

    print("\n1. Head Co-Activation (Exp1):")
    print("   (see detailed output above)")

    print("\n2. Routing State Space (Exp2):")
    print("   (see distance matrices above)")

    print("\n3. Circuit Reuse Algebra (Exp3):")
    for comb_name, r in exp3_results.items():
        verdict = "LINEAR" if r['is_linear'] else f"DOMINATED by {r['dominant']}"
        print(f"   {comb_name} ({r['mode_a']}+{r['mode_b']}): {verdict}")
        print(f"     sim_to_predicted={r['avg_sim_predicted']:.4f}, "
              f"sim_to_{r['mode_a']}={r['avg_sim_a']:.4f}, "
              f"sim_to_{r['mode_b']}={r['avg_sim_b']:.4f}")

    print("\n4. Task-State vs First-Token (Exp4):")
    if exp4_results:
        print(f"   Task shifts routing: {exp4_results.get('task_shifts_routing', 'N/A')}")
        print(f"   Task determines routing: {exp4_results.get('task_determines_routing', 'N/A')}")
        print(f"   Condition A avg dist: {exp4_results.get('condition_a_dists', {})}")
        print(f"   Condition B cross-start sim: {exp4_results.get('condition_b_cross_sim', 'N/A')}")
        print(f"   Condition C cross-start sim: {exp4_results.get('condition_c_cross_sim', 'N/A')}")

    # Save results
    out_path = Path(f"tests/glm5_temp/phase201_{model_name}_results.json")

    def convert(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    save_data = {
        'model': model_name,
        'n_sentences': len(sentences),
        'exp1_coactivation': {k: convert(v) for k, v in exp1_results.items()},
        'exp3_linearity': {k: convert(v) for k, v in exp3_results.items()},
        'exp4_causal': {k: convert(v) for k, v in exp4_results.items()},
        'timestamp': datetime.now().isoformat(),
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

    # Release
    elapsed = time.time() - t0
    print(f"\n[Phase 201] COMPLETE in {elapsed:.1f}s ({model_name})")
    release_model(model)


if __name__ == "__main__":
    main()
