"""
Phase 200: Internal Computational Regime Analysis
==================================================

理论框架 (基于Phase 198-199纠错后的升级):

  核心转向: 从"输出KL分析"转向"内部计算状态分析"
  
  Phase 199的瓶颈:
  - KL/entropy/basin 都是输出统计, 无法区分"语义约束"和"token级扰动"
  - "约束"概念本身可能有误(negation=扰动, question=模式切换, conditional=弱扰动)
  - 需要进入模型内部, 研究计算组织结构

  Pattern Compiler Theory:
  - 语言模型不是词→词映射, 而是模式编译器
  - 输入prompt → 识别模式 → 激活程序 → 组织计算回路 → 建立状态空间 → 稳定生成
  - "step by step"不是"更多推理", 而是"切换到分解式程序"

Phase 200实验 (4个子实验):

Exp1: Activation Regime Clustering (激活状态聚类)
  核心问题: 不同mode是否进入不同稳定激活区域?
  方法: 提取不同prompt下各层hidden state, 计算层间激活距离矩阵
  如果QA/CoT/Coding形成分离的激活聚类 → mode是真实的计算相

Exp2: Routing Topology (路由拓扑)
  核心问题: 不同mode激活哪些head和MLP? head如何协同?
  方法: 提取各head的attention pattern, 计算head-level的激活相似度
  如果不同mode激活不同的head ensemble → 存在模式特定的路由

Exp3: Phase Transition Boundary (相变边界精确定位)
  核心问题: CoT的离散相变边界在哪里?
  方法: 在"think"→"think carefully"→"think step by step"之间插入更细粒度的中间态
  逐步添加关键词, 精确定位相变发生的确切位置

Exp4: Representation Reuse (表征复用)
  核心问题: 同一个circuit是否被QA/CoT/Coding共同复用?
  方法: 计算不同mode在相同层的激活相似度
  如果mid-layer的激活跨mode高度相似 → 存在可复用的中间回路

数据量:
  - Qwen3: 20句 × 36层 × 多条件 (Exp1+2+4)
  - GLM4/DS7B: 8句 (lite模式)

模型加载: bf16 + device_map="auto" + flash attention + 定时日志
"""

import sys, os
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

import gc, time, json, math, random, warnings
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# bf16加载
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tests"))
from model_demo_bf16 import load_model_bf16
from model_utils import get_model_info, release_model, get_layers

warnings.filterwarnings('ignore')

# ===== 基础句 =====
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
]

LITE_SENTENCES = BASE_SENTENCES[:8]

# ===== Mode Prompt Templates =====
# 每种mode有多个prompt变体, 用于测量mode内部的稳定性和mode之间的分离度
MODE_PROMPTS = {
    # Normal mode: 基础陈述句
    "normal": [
        lambda b: b,
    ],
    
    # QA mode: 疑问句
    "qa": [
        lambda b: "Does " + b[0].lower() + b[1:] + "?",
        lambda b: "What does " + b.split()[1] + " " + b.split()[0].lower() + " do?",
        lambda b: "Is it true that " + b[0].lower() + b[1:] + "?",
    ],
    
    # CoT mode: 链式思维
    "cot": [
        lambda b: "Problem: " + b[0].lower() + b[1:] + ". Think step by step.",
        lambda b: "Let's think step by step about " + b[0].lower() + b[1:] + ".",
        lambda b: b + " Please reason step by step.",
    ],
    
    # Translation mode
    "translation": [
        lambda b: "Translate to Chinese: " + b,
        lambda b: b + " in Chinese:",
        lambda b: "How do you say '" + b + "' in Chinese?",
    ],
    
    # Coding mode
    "coding": [
        lambda b: "Write code for: " + b[0].lower() + b[1:],
        lambda b: "def " + b.split()[1].lower() + "():",
        lambda b: "Implement a function that " + b[0].lower() + b[1:],
    ],
    
    # Negation mode
    "negation": [
        lambda b: b.replace(b.split()[1], "does not " + b.split()[1].lower(), 1) if len(b.split()) >= 2 else b,
        lambda b: "It is false that " + b[0].lower() + b[1:],
    ],
    
    # Conditional mode
    "conditional": [
        lambda b: "If " + b[0].lower() + b[1:] + ", then",
        lambda b: "Suppose that " + b[0].lower() + b[1:],
    ],
    
    # Narrative mode
    "narrative": [
        lambda b: "Once upon a time, " + b[0].lower() + b[1:],
        lambda b: "In the story, " + b[0].lower() + b[1:],
    ],
}

# ===== CoT渐变系列 (精细版, 用于Exp3相变边界) =====
COT_FINE_GRADIENT = [
    ("base", "Problem: 2+3=?"),
    ("think", "Problem: 2+3=? Think"),
    ("think_more", "Problem: 2+3=? Think more"),
    ("think_carefully", "Problem: 2+3=? Think carefully"),
    ("think_deeply", "Problem: 2+3=? Think deeply"),
    ("think_thoroughly", "Problem: 2+3=? Think thoroughly"),
    ("think_step", "Problem: 2+3=? Think step"),
    ("think_step_by", "Problem: 2+3=? Think step by"),
    ("think_step_by_step", "Problem: 2+3=? Think step by step"),
    ("lets_think", "Problem: 2+3=? Let's think step by step"),
    ("cot_format", "Problem: 2+3=?\nLet's think step by step:\n"),
]

# 编程渐变系列
CODING_FINE_GRADIENT = [
    ("base", "Add two numbers"),
    ("using", "Add two numbers using"),
    ("using_code", "Add two numbers using code"),
    ("write_code", "Write code to add two numbers"),
    ("write_function", "Write a function to add two numbers"),
    ("implement", "Implement add two numbers"),
    ("def_start", "def add"),
    ("def_full", "def add_two_numbers"),
    ("def_paren", "def add_two_numbers("),
    ("def_param", "def add_two_numbers(a, b):"),
]


# ===== 工具函数 =====

def compute_cosine_sim(v1, v2):
    """余弦相似度"""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def compute_kl(p, q):
    """KL(p||q)"""
    p = p.float().clamp(min=1e-10)
    q = q.float().clamp(min=1e-10)
    kl = (p * (p.log() - q.log())).sum().item()
    if math.isnan(kl) or math.isinf(kl):
        return 0.0
    return max(kl, 0.0)


def compute_entropy(probs):
    """Shannon entropy"""
    p = probs.float().clamp(min=1e-10)
    h = -(p * p.log()).sum().item()
    return h if not math.isnan(h) else 0.0


def extract_hidden_states(model, tokenizer, device, text, n_layers):
    """
    提取所有层的hidden state (last token position)
    
    Returns:
        dict: {layer_idx: numpy_array[d_model]} — 每层最后一个token的hidden state
    """
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).input_ids.to(device)
    
    with torch.no_grad():
        try:
            out = model(input_ids=ids, output_hidden_states=True)
        except Exception:
            # 有些模型需要attention_mask
            attn_mask = torch.ones_like(ids)
            out = model(input_ids=ids, attention_mask=attn_mask, output_hidden_states=True)
    
    hs = out.hidden_states  # tuple of (1, seq_len, d_model)
    
    states = {}
    for li, h in enumerate(hs):
        if li > n_layers:  # 跳过embedding layer(0)之后的层
            break
        # 取最后一个token的hidden state
        states[li] = h[0, -1, :].detach().float().cpu().numpy()
    
    return states


def extract_attention_patterns(model, tokenizer, device, text, n_layers, max_len=64):
    """
    提取各层各head的attention pattern (last token → all previous tokens)
    
    Returns:
        dict: {layer_idx: numpy_array[n_heads, max_len]} — 各head对last token的attention weights (padded/truncated)
    """
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(device)
    seq_len = ids.shape[1]
    
    with torch.no_grad():
        try:
            out = model(input_ids=ids, output_attentions=True)
        except Exception:
            attn_mask = torch.ones_like(ids)
            out = model(input_ids=ids, attention_mask=attn_mask, output_attentions=True)
    
    attns = out.attentions  # tuple of (1, n_heads, seq_len, seq_len)
    
    patterns = {}
    if attns is None:
        return patterns
    
    for li, attn in enumerate(attns):
        if li >= n_layers:
            break
        # 取最后一个query position的attention → all keys
        # shape: (1, n_heads, seq_len, seq_len)
        # last_query → all keys: attn[0, :, -1, :] shape=(n_heads, seq_len)
        raw = attn[0, :, -1, :].detach().float().cpu().numpy()  # (n_heads, seq_len)
        
        # Pad or truncate to max_len for consistent shape across different inputs
        n_heads = raw.shape[0]
        actual_len = raw.shape[1]
        if actual_len >= max_len:
            patterns[li] = raw[:, -max_len:]  # 取最后max_len个
        else:
            padded = np.zeros((n_heads, max_len), dtype=raw.dtype)
            padded[:, -actual_len:] = raw
            patterns[li] = padded
    
    return patterns


def run_autoregressive(model, tokenizer, device, text, n_steps):
    """自回归生成n步"""
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    step_data = []
    
    for step in range(n_steps):
        with torch.no_grad():
            out = model(ids)
            logits = out.logits[:, -1, :].to(device).float()
            probs = F.softmax(logits, dim=-1)
        
        h = compute_entropy(probs[0])
        next_tok = probs.argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_tok], dim=-1)
        step_data.append({'entropy': h, 'probs': probs[0]})
    
    return step_data


# ========================================================================
# EXP1: Activation Regime Clustering
# ========================================================================
def exp1_activation_regime(model, tokenizer, device, info, sentences, n_samples=3):
    """
    核心实验: 不同mode是否形成不同的激活区域?
    
    方法:
    1. 对每个mode的prompt, 提取所有层的hidden state
    2. 计算层间激活的余弦距离矩阵
    3. 对mid-layer和last-layer做mode间距离分析
    4. 如果mode形成分离聚类 → mode是真实的计算相
    
    关键指标:
    - Mode separation: 不同mode在mid/late layer的激活余弦距离
    - Mode coherence: 同一mode内部不同prompt的激活相似度
    - Layer trajectory: 从早层到晚层, mode分离度如何变化
    """
    print("\n" + "="*60)
    print("Exp1: Activation Regime Clustering")
    print("="*60)
    
    n_layers = info.n_layers
    modes = list(MODE_PROMPTS.keys())
    
    # 采样层: 早(5), 中(n/2), 晚(n-3)
    sample_layers = sorted(set([
        min(5, n_layers-1),
        n_layers // 3,
        n_layers // 2,
        2 * n_layers // 3,
        max(0, n_layers - 3),
    ]))
    
    print(f"  Modes: {modes}")
    print(f"  Sample layers: {sample_layers}")
    
    # 收集激活: {mode: {layer: [hidden_states]}}
    activations = {mode: {li: [] for li in sample_layers} for mode in modes}
    
    t_start = time.time()
    total = len(sentences) * len(modes) * n_samples
    done = 0
    
    for si, base in enumerate(sentences):
        for mode, templates in MODE_PROMPTS.items():
            for ti, template in enumerate(templates[:n_samples]):
                text = template(base)
                states = extract_hidden_states(model, tokenizer, device, text, n_layers)
                
                for li in sample_layers:
                    if li in states:
                        activations[mode][li].append(states[li])
                
                done += 1
        
        # 定时日志
        if (si + 1) % 4 == 0 or si == 0:
            elapsed = time.time() - t_start
            gpu = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            print(f"  Sentence {si+1}/{len(sentences)}, done={done}/{total}, "
                  f"elapsed={elapsed:.0f}s, GPU={gpu:.2f}GB")
    
    # === Analysis ===
    print("\n--- Exp1A: Mode Separation by Layer ---")
    
    # 计算每个mode在每个层的平均激活
    mode_centroids = {}
    for mode in modes:
        mode_centroids[mode] = {}
        for li in sample_layers:
            acts = activations[mode][li]
            if acts:
                mode_centroids[mode][li] = np.mean(acts, axis=0)
            else:
                mode_centroids[mode][li] = np.zeros(info.d_model)
    
    # 计算mode间的余弦距离矩阵
    print(f"\n  Mode cosine similarity at each layer:")
    print(f"  {'Layer':>6}", end="")
    for m in modes:
        print(f" {m[:7]:>7}", end="")
    print()
    
    for li in sample_layers:
        # 计算所有mode对的余弦相似度
        print(f"  {li:6d}", end="")
        for m in modes:
            # 该mode与normal的相似度
            sim = compute_cosine_sim(mode_centroids[m][li], mode_centroids['normal'][li])
            print(f" {sim:7.3f}", end="")
        print()
    
    # Mode separation matrix (last sample layer)
    print(f"\n--- Exp1B: Mode-Mode Distance Matrix (Layer {sample_layers[-1]}) ---")
    li = sample_layers[-1]
    print(f"  {'':>12}", end="")
    for m in modes:
        print(f" {m[:7]:>7}", end="")
    print()
    
    separation_matrix = {}
    for m1 in modes:
        print(f"  {m1[:12]:>12}", end="")
        for m2 in modes:
            sim = compute_cosine_sim(mode_centroids[m1][li], mode_centroids[m2][li])
            dist = 1.0 - sim
            print(f" {dist:7.3f}", end="")
            separation_matrix[f"{m1}_vs_{m2}"] = float(dist)
        print()
    
    # Mode coherence: 同一mode内部的不同prompt相似度
    print(f"\n--- Exp1C: Mode Coherence (intra-mode similarity) ---")
    coherence = {}
    for mode in modes:
        for li in sample_layers:
            acts = activations[mode][li]
            if len(acts) >= 2:
                # 计算所有对之间的平均余弦相似度
                sims = []
                for i in range(min(len(acts), 20)):
                    for j in range(i+1, min(len(acts), 20)):
                        sims.append(compute_cosine_sim(acts[i], acts[j]))
                coherence[f"{mode}_L{li}"] = float(np.mean(sims))
    
    for mode in modes:
        vals = {k: v for k, v in coherence.items() if k.startswith(mode)}
        if vals:
            avg = np.mean(list(vals.values()))
            print(f"  {mode:<15} avg_coherence={avg:.4f}")
    
    # Layer trajectory: mode分离度如何随层变化
    print(f"\n--- Exp1D: Mode Separation Trajectory (vs normal) ---")
    print(f"  {'Layer':>6}", end="")
    for m in modes:
        if m != 'normal':
            print(f" {m[:10]:>10}", end="")
    print()
    
    trajectory = {}
    for li in sample_layers:
        print(f"  {li:6d}", end="")
        for m in modes:
            if m != 'normal':
                dist = 1.0 - compute_cosine_sim(mode_centroids[m][li], mode_centroids['normal'][li])
                print(f" {dist:10.4f}", end="")
                trajectory[f"{m}_L{li}"] = float(dist)
        print()
    
    return {
        'separation_matrix': separation_matrix,
        'coherence': coherence,
        'trajectory': trajectory,
        'sample_layers': sample_layers,
    }


# ========================================================================
# EXP2: Routing Topology
# ========================================================================
def exp2_routing_topology(model, tokenizer, device, info, sentences, n_samples=2):
    """
    核心实验: 不同mode激活哪些head? head如何协同?
    
    方法:
    1. 提取各层的attention pattern
    2. 计算不同mode的head激活差异
    3. 分析head ensemble的协同模式
    
    注意: device_map="auto"下output_attentions可能较慢,
          所以只采样少量句子
    """
    print("\n" + "="*60)
    print("Exp2: Routing Topology")
    print("="*60)
    
    n_layers = info.n_layers
    modes = ["normal", "qa", "cot", "translation", "coding", "negation", "conditional"]
    
    # 采样层
    sample_layers = sorted(set([
        min(3, n_layers-1),
        n_layers // 4,
        n_layers // 2,
        3 * n_layers // 4,
        max(0, n_layers - 2),
    ]))
    
    print(f"  Sample layers: {sample_layers}")
    
    # 收集attention patterns: {mode: {layer: [attn_patterns]}}
    attn_data = {mode: {li: [] for li in sample_layers} for mode in modes}
    
    t_start = time.time()
    
    for si, base in enumerate(sentences[:5]):  # 只用5句, attention提取较慢
        for mode in modes:
            templates = MODE_PROMPTS[mode][:n_samples]
            for template in templates:
                text = template(base)
                patterns = extract_attention_patterns(model, tokenizer, device, text, n_layers)
                
                for li in sample_layers:
                    if li in patterns:
                        attn_data[mode][li].append(patterns[li])  # (n_heads, seq_len)
        
        if (si + 1) % 2 == 0:
            elapsed = time.time() - t_start
            gpu = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            print(f"  Sentence {si+1}/5, elapsed={elapsed:.0f}s, GPU={gpu:.2f}GB")
    
    # === Analysis ===
    print("\n--- Exp2A: Head Activation Pattern Similarity ---")
    
    # 对每个mode在每个层, 计算head attention的平均模式
    mode_attn_centroids = {}
    for mode in modes:
        mode_attn_centroids[mode] = {}
        for li in sample_layers:
            patterns = attn_data[mode][li]
            if patterns:
                # 确保所有pattern shape一致
                valid_patterns = [p for p in patterns if p.shape == patterns[0].shape]
                if valid_patterns:
                    avg_pattern = np.mean(valid_patterns, axis=0)  # (n_heads, max_len)
                    mode_attn_centroids[mode][li] = avg_pattern
    
    # 计算mode间的head-level路由差异
    print(f"\n  Head routing distance (vs normal) at each layer:")
    print(f"  {'Layer':>6} {'n_heads':>8}", end="")
    for m in modes:
        if m != 'normal':
            print(f" {m[:10]:>10}", end="")
    print()
    
    routing_results = {}
    for li in sample_layers:
        normal_pattern = mode_attn_centroids.get('normal', {}).get(li)
        if normal_pattern is None:
            continue
        
        n_heads = normal_pattern.shape[0]
        print(f"  {li:6d} {n_heads:8d}", end="")
        
        for m in modes:
            if m == 'normal':
                continue
            m_pattern = mode_attn_centroids.get(m, {}).get(li)
            if m_pattern is None:
                print(f" {'N/A':>10}", end="")
                continue
            
            # 计算head-level的平均余弦距离
            # 对每个head, 计算attention pattern的余弦距离
            head_dists = []
            for h in range(min(n_heads, m_pattern.shape[0])):
                try:
                    sim = compute_cosine_sim(normal_pattern[h], m_pattern[h])
                    head_dists.append(1.0 - sim)
                except:
                    pass
            
            avg_dist = np.mean(head_dists) if head_dists else 0
            print(f" {avg_dist:10.4f}", end="")
            routing_results[f"{m}_L{li}"] = float(avg_dist)
        
        print()
    
    # Head specialization: 哪些head在不同mode间差异最大?
    print(f"\n--- Exp2B: Head Specialization (Layer {sample_layers[-1]}) ---")
    li = sample_layers[-1]
    normal_pattern = mode_attn_centroids.get('normal', {}).get(li)
    
    if normal_pattern is not None:
        n_heads = normal_pattern.shape[0]
        # 对每个head, 计算它跨mode的变异度
        head_specialization = []
        for h in range(n_heads):
            h_patterns = []
            for mode in modes:
                mp = mode_attn_centroids.get(mode, {}).get(li)
                if mp is not None and h < mp.shape[0]:
                    h_patterns.append(mp[h])
            
            if len(h_patterns) >= 2:
                # 计算head pattern跨mode的方差
                h_stack = np.array(h_patterns)
                variance = np.var(h_stack, axis=0).mean()
                head_specialization.append((h, variance))
        
        # 排序: 方差最大的head = 最specialized
        head_specialization.sort(key=lambda x: x[1], reverse=True)
        print(f"  Top-10 most specialized heads:")
        for h, var in head_specialization[:10]:
            print(f"    Head {h}: variance={var:.6f}")
    
    return routing_results


# ========================================================================
# EXP3: Phase Transition Boundary (Fine-grained)
# ========================================================================
def exp3_phase_boundary(model, tokenizer, device, info, n_repeats=3):
    """
    核心实验: CoT相变边界在哪里?
    
    方法: 在"think"→"think step by step"之间插入更细粒度的中间态
    逐步添加关键词, 精确定位相变发生的确切位置
    
    同时也测试coding的相变边界
    """
    print("\n" + "="*60)
    print("Exp3: Phase Transition Boundary (Fine-grained)")
    print("="*60)
    
    n_layers = info.n_layers
    mid_layer = n_layers // 2
    late_layer = max(0, n_layers - 3)
    
    for gradient_name, gradient in [("CoT", COT_FINE_GRADIENT), ("Coding", CODING_FINE_GRADIENT)]:
        print(f"\n--- {gradient_name} Phase Boundary ---")
        
        # 参考点: base prompt
        ref_name, ref_text = gradient[0]
        ref_states = extract_hidden_states(model, tokenizer, device, ref_text, n_layers)
        
        results = []
        
        for gi, (gname, gtext) in enumerate(gradient):
            # 1. Hidden state分析
            states = extract_hidden_states(model, tokenizer, device, gtext, n_layers)
            
            # 与reference的余弦距离
            mid_dist = 0
            late_dist = 0
            if mid_layer in states and mid_layer in ref_states:
                mid_dist = 1.0 - compute_cosine_sim(states[mid_layer], ref_states[mid_layer])
            if late_layer in states and late_layer in ref_states:
                late_dist = 1.0 - compute_cosine_sim(states[late_layer], ref_states[late_layer])
            
            # 2. 输出KL (参考Phase 199的方法)
            ref_out = run_autoregressive(model, tokenizer, device, ref_text, 1)
            cur_out = run_autoregressive(model, tokenizer, device, gtext, 1)
            kl0 = compute_kl(cur_out[0]['probs'], ref_out[0]['probs'])
            entropy0 = cur_out[0]['entropy']
            
            results.append({
                'name': gname,
                'text': gtext[:50],
                'mid_dist': mid_dist,
                'late_dist': late_dist,
                'kl0': kl0,
                'entropy': entropy0,
            })
            
            print(f"  [{gi:2d}] {gname:<20} mid_dist={mid_dist:.4f} late_dist={late_dist:.4f} "
                  f"KL[0]={kl0:.3f} H={entropy0:.3f}")
        
        # 检测相变点: 最大的跳跃
        print(f"\n  Phase transition analysis:")
        
        for metric in ['mid_dist', 'late_dist', 'kl0', 'entropy']:
            vals = [r[metric] for r in results]
            jumps = []
            for i in range(1, len(vals)):
                jumps.append(abs(vals[i] - vals[i-1]))
            
            max_jump = max(jumps) if jumps else 0
            max_jump_idx = jumps.index(max_jump) + 1 if jumps else -1
            
            # 平均跳跃 vs 最大跳跃
            avg_jump = np.mean(jumps) if jumps else 0
            ratio = max_jump / avg_jump if avg_jump > 0.001 else 0
            
            print(f"    {metric}: max_jump={max_jump:.4f} at {gradient[max_jump_idx][0]}, "
                  f"avg_jump={avg_jump:.4f}, ratio={ratio:.1f}x")
            
            if ratio > 3.0:
                print(f"    >>> PHASE TRANSITION DETECTED at '{gradient[max_jump_idx][0]}' "
                      f"(jump is {ratio:.1f}x larger than average)")
    
    return {}


# ========================================================================
# EXP4: Representation Reuse
# ========================================================================
def exp4_representation_reuse(model, tokenizer, device, info, sentences):
    """
    核心实验: 同一个circuit是否被不同mode复用?
    
    方法:
    1. 对每个mode, 提取所有层的hidden state
    2. 计算不同mode在同一层的激活相似度
    3. 如果mid-layer跨mode高度相似 → 存在可复用的中间回路
    4. 如果late-layer跨mode差异大 → 模式分化在晚期发生
    
    这直接检验: "语言模式论" vs "语义向量论"
    - 如果语义向量论正确: 不同mode应该在所有层都不同(因为语义不同)
    - 如果模式编译论正确: 早/中层应该共享(因为基础语法相同), 晚层分化(因为模式不同)
    """
    print("\n" + "="*60)
    print("Exp4: Representation Reuse")
    print("="*60)
    
    n_layers = info.n_layers
    modes = ["normal", "qa", "cot", "translation", "coding", "negation", "conditional"]
    
    # 全层分析
    all_layer_sample = sorted(set(list(range(0, n_layers, max(1, n_layers // 10))) + [n_layers - 1]))
    
    print(f"  Layer sample points: {all_layer_sample}")
    
    # 对每个句子, 收集所有mode在所有层的hidden state
    # {sentence_idx: {mode: {layer: hidden_state}}}
    all_states = {}
    
    t_start = time.time()
    
    for si, base in enumerate(sentences[:10]):  # 10句足够
        all_states[si] = {}
        
        for mode in modes:
            template = MODE_PROMPTS[mode][0]  # 每个mode只用第一个prompt
            text = template(base)
            states = extract_hidden_states(model, tokenizer, device, text, n_layers)
            all_states[si][mode] = states
        
        if (si + 1) % 3 == 0:
            elapsed = time.time() - t_start
            gpu = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            print(f"  Sentence {si+1}/10, elapsed={elapsed:.0f}s, GPU={gpu:.2f}GB")
    
    # === Analysis ===
    print("\n--- Exp4A: Cross-Mode Similarity by Layer (vs normal) ---")
    print(f"  {'Layer':>6}", end="")
    for m in modes:
        if m != 'normal':
            print(f" {m[:7]:>7}", end="")
    print()
    
    reuse_data = {}
    for li in all_layer_sample:
        print(f"  {li:6d}", end="")
        
        for m in modes:
            if m == 'normal':
                continue
            
            # 计算跨句子的平均余弦相似度
            sims = []
            for si in all_states:
                if li in all_states[si].get('normal', {}) and li in all_states[si].get(m, {}):
                    sim = compute_cosine_sim(
                        all_states[si]['normal'][li],
                        all_states[si][m][li]
                    )
                    sims.append(sim)
            
            avg_sim = np.mean(sims) if sims else 0
            print(f" {avg_sim:7.4f}", end="")
            reuse_data[f"{m}_L{li}"] = float(avg_sim)
        
        print()
    
    # 分析: 早层vs晚层的分化
    print(f"\n--- Exp4B: Mode Differentiation Profile ---")
    
    early_layers = [li for li in all_layer_sample if li < n_layers // 3]
    mid_layers = [li for li in all_layer_sample if n_layers // 3 <= li < 2 * n_layers // 3]
    late_layers = [li for li in all_layer_sample if li >= 2 * n_layers // 3]
    
    for stage, layer_set in [("Early", early_layers), ("Mid", mid_layers), ("Late", late_layers)]:
        if not layer_set:
            continue
        print(f"\n  {stage} layers ({layer_set}):")
        for m in modes:
            if m == 'normal':
                continue
            vals = [reuse_data.get(f"{m}_L{li}", 0) for li in layer_set]
            avg = np.mean(vals) if vals else 0
            print(f"    {m:<15} avg_similarity_to_normal={avg:.4f}")
    
    # 关键问题: 是否存在"共享中间层"?
    print(f"\n--- Exp4C: Shared Intermediate Representation Test ---")
    
    # 假设: 如果模式编译论正确, mid-layer应该跨mode高相似
    # 如果语义向量论正确, 所有层都应该跨mode低相似
    
    mid_li = n_layers // 2
    late_li = max(0, n_layers - 3)
    
    # 计算mode-pair相似度在mid vs late
    print(f"  Mode-pair similarity at mid(L{mid_li}) vs late(L{late_li}):")
    
    for m1 in modes:
        for m2 in modes:
            if m1 >= m2:
                continue
            
            # Mid-layer similarity
            mid_sims = []
            late_sims = []
            
            for si in all_states:
                if mid_li in all_states[si].get(m1, {}) and mid_li in all_states[si].get(m2, {}):
                    mid_sims.append(compute_cosine_sim(
                        all_states[si][m1][mid_li],
                        all_states[si][m2][mid_li]
                    ))
                if late_li in all_states[si].get(m1, {}) and late_li in all_states[si].get(m2, {}):
                    late_sims.append(compute_cosine_sim(
                        all_states[si][m1][late_li],
                        all_states[si][m2][late_li]
                    ))
            
            mid_avg = np.mean(mid_sims) if mid_sims else 0
            late_avg = np.mean(late_sims) if late_sims else 0
            diff = late_avg - mid_avg
            
            if abs(diff) > 0.02 or mid_avg < 0.95:
                print(f"    {m1[:7]:>7} vs {m2[:7]:>7}: mid={mid_avg:.4f}, late={late_avg:.4f}, "
                      f"diff={diff:+.4f} {'>>> DIVERGENT' if diff < -0.05 else ''}")
    
    return reuse_data


# ========================================================================
# MAIN
# ========================================================================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    is_lite = model_name != "qwen3"
    
    t0 = time.time()
    print(f"[Phase 200] Internal Computational Regime Analysis — {model_name}")
    print(f"[Phase 200] Time: {datetime.now()}")
    print(f"[Phase 200] Lite mode: {is_lite}")
    
    # Load model
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    print(f"[load] {model_name}: {info.model_class}, {info.n_layers}L, d={info.d_model}")
    
    # Config
    if is_lite:
        sentences = LITE_SENTENCES
        n_samples = 2
    else:
        sentences = BASE_SENTENCES[:20]
        n_samples = 3
    
    print(f"\nConfig: {len(sentences)} sentences, n_samples={n_samples}")
    
    # Run experiments
    exp1_results = exp1_activation_regime(model, tokenizer, device, info, sentences, n_samples)
    exp2_results = exp2_routing_topology(model, tokenizer, device, info, sentences, n_samples)
    exp3_results = exp3_phase_boundary(model, tokenizer, device, info)
    exp4_results = exp4_representation_reuse(model, tokenizer, device, info, sentences)
    
    # ===== Final Summary =====
    print("\n" + "="*60)
    print(f"PHASE 200 SUMMARY — {model_name}")
    print("="*60)
    
    print("\n1. Activation Regime Clustering:")
    if 'trajectory' in exp1_results:
        for k, v in sorted(exp1_results['trajectory'].items()):
            if v > 0.01:
                print(f"   {k}: distance_from_normal={v:.4f}")
    
    print("\n2. Routing Topology:")
    for k, v in sorted(exp2_results.items()):
        if v > 0.001:
            print(f"   {k}: routing_distance={v:.4f}")
    
    print("\n3. Phase Boundary: (see detailed output above)")
    
    print("\n4. Representation Reuse:")
    if exp4_results:
        # 早层vs晚层的关键差异
        early_vals = [v for k, v in exp4_results.items() if '_L0' in k or '_L3' in k]
        late_vals = [v for k, v in exp4_results.items() if f'_L{info.n_layers-1}' in k or f'_L{info.n_layers-3}' in k]
        if early_vals:
            print(f"   Early layer avg cross-mode similarity: {np.mean(early_vals):.4f}")
        if late_vals:
            print(f"   Late layer avg cross-mode similarity: {np.mean(late_vals):.4f}")
        if early_vals and late_vals:
            diff = np.mean(late_vals) - np.mean(early_vals)
            print(f"   Differentiation (late - early): {diff:+.4f}")
            if diff < -0.05:
                print(f"   >>> LATE LAYER DIVERGENCE: Mode specialization happens in late layers")
            elif diff > 0.05:
                print(f"   >>> EARLY LAYER DIVERGENCE: Mode specialization starts early")
            else:
                print(f"   >>> UNIFORM: Mode similarity is roughly constant across layers")
    
    # Save results
    out_path = Path(f"tests/glm5_temp/phase200_{model_name}_results.json")
    
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
        'exp1_regime': {k: convert(v) for k, v in exp1_results.get('separation_matrix', {}).items()},
        'exp1_coherence': {k: convert(v) for k, v in exp1_results.get('coherence', {}).items()},
        'exp1_trajectory': {k: convert(v) for k, v in exp1_results.get('trajectory', {}).items()},
        'exp2_routing': {k: convert(v) for k, v in exp2_results.items()},
        'exp4_reuse': {k: convert(v) for k, v in exp4_results.items()},
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")
    
    # Release
    elapsed = time.time() - t0
    print(f"\n[Phase 200] COMPLETE in {elapsed:.1f}s ({model_name})")
    release_model(model)


if __name__ == "__main__":
    main()
