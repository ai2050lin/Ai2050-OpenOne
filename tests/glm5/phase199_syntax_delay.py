"""
Phase 199: Syntax-Controlled Semantic Perturbation + Delay Spectrum Mapping
=============================================================================

理论框架升级 (基于Phase 198纠错):
  Level 1: Mode — 全局动力学机制 (CoT, translation, coding)
  Level 2: Constraint — 局部轨迹塑形 (negation, question, conditional)
  Level 3: Autoregressive chaos — 自回归发散基线

Phase 198核心发现:
  - KL slope ≈ tautology (semantic/random ratio ≈ 1.0x)
  - Conditional delayed effect: KL[0]小但slope大 → 真正的时间非对称性
  - CoT: slope=0.09但entropy shift=-1.27 → 模式效应而非轨迹效应
  - Question KL[0]高可能部分来自syntactic restructuring

Phase 199目标 (按优先级):

Exp1: Syntax-Controlled Semantic Perturbation [最关键!]
  核心问题: 语义效应 vs 句法混淆
  设计:
  - semantic_negation: "The cat does not chase the dog" (语义+句法都变)
  - syntax_negation: "The cat XYZ chase the dog" (只变句法，不变语义，插入无意义token)
  - semantic_question: "Does the cat chase the dog?" (语义+句法都变)
  - syntax_question: "The cat chase the dog, however" (句法结构变化但不是疑问)
  - semantic_conditional: "If the cat chases the dog" (语义+句法都变)
  - syntax_conditional: "When the cat chases the dog" (句法结构类似但语义不同)
  
  关键对照:
  - 如果syntax_negation和semantic_negation的KL[0]相似 → question效应主要是句法
  - 如果syntax_negation和semantic_negation的KL[0]不同 → 语义有真实效应

Exp2: Delay Spectrum Deep Mapping [最深方向]
  核心问题: 不同语言结构影响未来的时间尺度不同
  设计:
  - 即时约束: negation, question, role_binding
  - 延迟约束: conditional ("If X"), future tense ("will X")
  - 长程约束: narrative setup ("In a world where X")
  - 超长程: CoT trigger ("Let's think step by step")
  
  测量: KL[0], KL[1], KL[2]... 的完整profile, 不仅看slope
  关键指标: delay_k = argmin_k(KL[k] > threshold) — 约束何时真正开始生效

Exp3: Mode Transition Continuity [相变 vs 连续变化]
  核心问题: 模式切换是离散相变还是连续变化?
  设计:
  - 用渐变prompt: "Think" → "Think carefully" → "Think step by step" → "Let's think step by step"
  - 测量entropy/KL在渐变路径上是否连续变化
  - 如果连续 → mode是吸引子区域，不是离散态
  - 如果突变 → mode是真正相变

数据量:
  - Qwen3: 20句 × 12步 × 30采样 (Exp1+2)
  - GLM4/DS7B: 8句 × 8步 × 15采样

模型加载: bf16 + device_map="auto" + flash attention + 定时日志
"""

import sys, os
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

import gc, time, json, math, random
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# bf16加载
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tests"))
from model_demo_bf16 import load_model_bf16
from model_utils import get_model_info, release_model

# ===== 基础句 (20句, 覆盖多种主谓宾结构) =====
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

LITE_SENTENCES = BASE_SENTENCES[:8]  # GLM4/DS7B用8句

# ===== Exp1: Syntax-Controlled Semantic Perturbation =====

def make_semantic_negation(base):
    """语义否定: 改变语义+句法"""
    words = base.split()
    if len(words) >= 2:
        return words[0] + " " + words[1] + " does not " + " ".join(words[2:])
    return base

def make_syntax_negation(base):
    """句法否定: 只改变句法结构(插入无意义token), 不改变语义
    在动词前插入一个无意义但占位的token, 模仿'does not'的句法插入效应"""
    words = base.split()
    if len(words) >= 3:
        # 在第2和第3个词之间插入无意义占位
        return words[0] + " " + words[1] + " zzz " + " ".join(words[2:])
    return base

def make_syntax_negation_v2(base):
    """句法否定v2: 插入一个真正的语法功能词(如'then'), 不改变语义真值"""
    words = base.split()
    if len(words) >= 3:
        return words[0] + " " + words[1] + " then " + " ".join(words[2:])
    return base

def make_semantic_question(base):
    """语义疑问: 改变语义+句法"""
    return "Does " + base[0].lower() + base[1:] + "?"

def make_syntax_question(base):
    """句法疑问: 句法结构变化(倒装)但不形成真正疑问
    使用'Therefore,'前缀, 造成类似的句法重构但不是疑问"""
    return "Therefore, " + base[0].lower() + base[1:]

def make_syntax_question_v2(base):
    """句法疑问v2: 在句尾加问号但不改变语序, 模拟标点效应"""
    return base + "?"

def make_semantic_conditional(base):
    """语义条件: 'If'建立假设世界"""
    return "If " + base[0].lower() + base[1:]

def make_syntax_conditional(base):
    """句法条件: 'When'也是从句结构但不建立假设世界"""
    return "When " + base[0].lower() + base[1:]

def make_syntax_conditional_v2(base):
    """句法条件v2: 'After'也是从句但语义不同"""
    return "After " + base[0].lower() + base[1:]

def make_random_control(base):
    """随机控制: 加随机token"""
    return base + " xyz"

def make_length_control(base):
    """长度控制: 加一个常见但无语义影响的前缀"""
    return "Well, " + base[0].lower() + base[1:]


# Exp1 条件映射
EXP1_CONDITIONS = {
    # 语义扰动 (语义+句法都变)
    "sem_negation": make_semantic_negation,
    "sem_question": make_semantic_question,
    "sem_conditional": make_semantic_conditional,
    
    # 句法控制 (句法变化, 语义尽量不变)
    "syn_negation": make_syntax_negation_v2,      # 插入'then'而非'does not'
    "syn_question": make_syntax_question,           # 'Therefore,'而非'Does...?'
    "syn_question_v2": make_syntax_question_v2,     # 只加问号
    "syn_conditional": make_syntax_conditional,     # 'When'而非'If'
    "syn_conditional_v2": make_syntax_conditional_v2, # 'After'而非'If'
    
    # 随机/长度控制
    "rand_token": make_random_control,
    "length_ctrl": make_length_control,
    
    # 纯句法插入 (无意义token)
    "syn_insertion": make_syntax_negation,          # 插入'zzz'
}

# ===== Exp2: Delay Spectrum Deep Mapping =====

# 延迟谱条件: 从即时到超长程
DELAY_CONDITIONS = {
    # 即时约束
    "negation": lambda b: make_semantic_negation(b),
    "question": lambda b: make_semantic_question(b),
    
    # 延迟约束
    "conditional_if": lambda b: "If " + b[0].lower() + b[1:],
    "conditional_unless": lambda b: "Unless " + b[0].lower() + b[1:],
    
    # 时态约束 (未来)
    "future_will": lambda b: b.replace(b.split()[1], b.split()[1] + " will", 1) if len(b.split()) >= 2 else b,
    "past_tense": lambda b: b.replace(b.split()[1], "had " + b.split()[1], 1) if len(b.split()) >= 2 else b,
    
    # 长程约束
    "narrative_setup": lambda b: "In a world where " + b[0].lower() + b[1:] + ",",
    "suppose_that": lambda b: "Suppose that " + b[0].lower() + b[1:],
    
    # 随机基线
    "rand_token": lambda b: b + " xyz",
    "rand_period": lambda b: b + ".",
}

# ===== Exp3: Mode Transition Continuity =====

# 渐变prompt: 从无模式到强模式
COT_GRADIENT = [
    "Problem: 2+3=? ",                    # 无模式
    "Problem: 2+3=? Think ",             # 弱提示
    "Problem: 2+3=? Think carefully ",    # 中等提示
    "Problem: 2+3=? Think step by step", # 强提示(但非完整CoT)
    "Problem: 2+3=? Let's think step by step", # 完整CoT触发
]

TRANSLATION_GRADIENT = [
    "The cat sleeps. ",
    "The cat sleeps in Chinese: ",
    "Translate the cat sleeps: ",
    "Translate to Chinese: The cat sleeps. ",
]

CODING_GRADIENT = [
    "Add two numbers ",
    "Add two numbers using code: ",
    "Write code to add two numbers: ",
    "def add_two_numbers",
]


# ===== 工具函数 =====

def compute_kl(p, q):
    """KL(p||q) with NaN protection"""
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

def effective_branching(probs, threshold=0.01):
    """有效分支因子"""
    return (probs > threshold).sum().item()

def run_autoregressive(model, tokenizer, device, text, n_steps):
    """自回归生成n步, 返回每步的probs和统计"""
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    step_data = []
    
    for step in range(n_steps):
        with torch.no_grad():
            out = model(ids)
            logits = out.logits[:, -1, :].to(device).float()
            probs = F.softmax(logits, dim=-1)
        
        h = compute_entropy(probs[0])
        ebf = effective_branching(probs[0])
        top5 = torch.topk(probs[0], 5)
        
        step_data.append({
            'entropy': h,
            'ebf': ebf,
            'top5_ids': top5.indices.tolist(),
            'top5_probs': top5.values.tolist(),
            'top5_tokens': [tokenizer.decode([t]).strip() for t in top5.indices.tolist()],
            'probs': probs[0],
        })
        
        # Greedy next token
        next_tok = probs.argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_tok], dim=-1)
    
    return step_data

def sample_first_tokens(model, tokenizer, device, text, n_samples, temperature=0.8):
    """采样n_samples次, 返回首token分布"""
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    first_tokens = []
    for _ in range(n_samples):
        with torch.no_grad():
            out = model(ids)
            logits = out.logits[:, -1, :].to(device).float()
            next_id = torch.multinomial(F.softmax(logits / temperature, dim=-1), 1)
            tok = tokenizer.decode(next_id[0]).strip()
            first_tokens.append(tok)
    return first_tokens

def find_delay_step(kl_list, threshold=1.0):
    """找到约束何时开始生效: KL首次超过threshold的步数"""
    for i, kl in enumerate(kl_list):
        if kl > threshold:
            return i
    return len(kl_list)  # 从未超过threshold


# ========================================================================
# EXP1: Syntax-Controlled Semantic Perturbation
# ========================================================================
def exp1_syntax_semantic(model, tokenizer, device, sentences, n_steps, n_samples):
    """
    核心实验: 区分语义效应与句法混淆
    
    关键对照:
    - sem_negation vs syn_negation (语义否定 vs 句法插入)
    - sem_question vs syn_question (语义疑问 vs 句法重构)  
    - sem_conditional vs syn_conditional (If vs When/After)
    
    如果语义效应是真实的:
    - sem_* 的 KL[0] 应该与 syn_* 有系统性差异
    - sem_conditional 的delay pattern应该与 syn_conditional 不同
    """
    print("\n" + "="*60)
    print("Exp1: Syntax-Controlled Semantic Perturbation")
    print("="*60)
    
    conditions = list(EXP1_CONDITIONS.keys())
    kl_profiles = {c: [] for c in conditions}
    entropy_profiles = {c: [] for c in conditions}
    basin_data = {c: [] for c in conditions}
    
    t_start = time.time()
    
    for si, base in enumerate(sentences):
        # Base trajectory
        base_steps = run_autoregressive(model, tokenizer, device, base, n_steps)
        
        for cond, fn in EXP1_CONDITIONS.items():
            text = fn(base)
            pert_steps = run_autoregressive(model, tokenizer, device, text, n_steps)
            
            # KL at each step
            kl_list = []
            for step in range(n_steps):
                kl = compute_kl(pert_steps[step]['probs'], base_steps[step]['probs'])
                kl_list.append(kl)
            kl_profiles[cond].append(kl_list)
            entropy_profiles[cond].append([s['entropy'] for s in pert_steps])
            
            # Basin analysis for first 3 sentences
            if si < 3:
                tokens = sample_first_tokens(model, tokenizer, device, text, n_samples)
                basin_data[cond].append(Counter(tokens))
        
        # 定时日志
        if (si + 1) % 2 == 0 or si == 0:
            elapsed = time.time() - t_start
            gpu = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            print(f"  Sentence {si+1}/{len(sentences)}, elapsed={elapsed:.0f}s, GPU={gpu:.2f}GB")
    
    # --- Analysis ---
    print("\n--- Exp1A: KL Profile (mean across sentences) ---")
    mean_kls = {}
    for cond in conditions:
        mean_kls[cond] = [np.mean([kl[step] for kl in kl_profiles[cond]]) for step in range(n_steps)]
    
    # Print header
    print(f"{'Step':>4} |", end="")
    for c in conditions:
        short = c[:10]
        print(f" {short:>10}", end="")
    print()
    
    for step in range(n_steps):
        print(f"{step:4d} |", end="")
        for c in conditions:
            print(f" {mean_kls[c][step]:10.3f}", end="")
        print()
    
    # KL slope
    print("\n--- Exp1B: KL[0] and Slope ---")
    print(f"{'Condition':<20} {'KL[0]':>8} {'KL[-1]':>8} {'Slope':>8} {'Type':>20}")
    slopes = {}
    for cond in conditions:
        arr = np.array(mean_kls[cond])
        slope = np.polyfit(range(n_steps), arr, 1)[0]
        slopes[cond] = slope
        kl0 = arr[0]
        kl_last = arr[-1]
        
        if kl0 > 5.0 and slope < 1.0:
            dtype = "IMMEDIATE (syntactic?)"
        elif kl0 < 1.0 and slope > 0.5:
            dtype = "DELAYED (semantic?)"
        elif kl0 < 1.0 and slope < 0.3:
            dtype = "MINIMAL"
        else:
            dtype = "MODERATE"
        
        print(f"{cond:<20} {kl0:8.3f} {kl_last:8.3f} {slope:8.4f} {dtype:>20}")
    
    # --- CRITICAL: Semantic vs Syntax comparison ---
    print("\n--- Exp1C: Semantic vs Syntax KL[0] Comparison ---")
    print(f"{'Pair':<35} {'Sem KL[0]':>10} {'Syn KL[0]':>10} {'Delta':>10} {'Semantic?':>10}")
    
    comparisons = [
        ("negation", "sem_negation", "syn_negation"),
        ("negation", "sem_negation", "syn_insertion"),
        ("question", "sem_question", "syn_question"),
        ("question", "sem_question", "syn_question_v2"),
        ("conditional", "sem_conditional", "syn_conditional"),
        ("conditional", "sem_conditional", "syn_conditional_v2"),
    ]
    
    semantic_signals = {}
    for name, sem_cond, syn_cond in comparisons:
        sem_kl0 = np.mean(mean_kls[sem_cond][0])
        syn_kl0 = np.mean(mean_kls[syn_cond][0])
        delta = sem_kl0 - syn_kl0
        is_semantic = "YES" if abs(delta) > 1.0 else ("MARGINAL" if abs(delta) > 0.3 else "NO")
        print(f"{name}: {sem_cond} vs {syn_cond:<15} {sem_kl0:10.3f} {syn_kl0:10.3f} {delta:10.3f} {is_semantic:>10}")
        semantic_signals[f"{sem_cond}_vs_{syn_cond}"] = {
            'sem_kl0': sem_kl0,
            'syn_kl0': syn_kl0,
            'delta': delta,
            'is_semantic': is_semantic,
        }
    
    # --- Delay comparison: semantic vs syntax conditional ---
    print("\n--- Exp1D: Conditional Delay: Semantic vs Syntax ---")
    sem_cond_kl = mean_kls["sem_conditional"]
    syn_cond_kl = mean_kls["syn_conditional"]
    syn_cond_v2_kl = mean_kls["syn_conditional_v2"]
    
    print(f"{'Step':>4} | {'If (sem)':>10} {'When (syn)':>10} {'After (syn)':>12} {'If-When':>8} {'If-After':>9}")
    for step in range(n_steps):
        diff1 = sem_cond_kl[step] - syn_cond_kl[step]
        diff2 = sem_cond_kl[step] - syn_cond_v2_kl[step]
        print(f"{step:4d} | {sem_cond_kl[step]:10.3f} {syn_cond_kl[step]:10.3f} {syn_cond_v2_kl[step]:12.3f} {diff1:8.3f} {diff2:9.3f}")
    
    # Basin comparison
    print("\n--- Exp1E: Attractor Basin Structure ---")
    for cond in conditions:
        if basin_data[cond]:
            all_tokens = []
            for bc in basin_data[cond]:
                all_tokens.extend(bc.elements())
            counts = Counter(all_tokens)
            n_unique = len(counts)
            total = sum(counts.values())
            top1 = counts.most_common(1)[0][1] / total if counts else 0
            probs_arr = np.array([v/total for v in counts.values()])
            ent = -(probs_arr * np.log(probs_arr + 1e-12)).sum()
            print(f"  {cond:<20} basins={n_unique:3d}, top1={top1:.3f}, entropy={ent:.2f}")
    
    return {
        'kl_profiles': {c: mean_kls[c] for c in conditions},
        'slopes': slopes,
        'semantic_signals': semantic_signals,
    }


# ========================================================================
# EXP2: Delay Spectrum Deep Mapping
# ========================================================================
def exp2_delay_spectrum(model, tokenizer, device, sentences, n_steps):
    """
    延迟谱深度映射: 不同语言结构影响未来的时间尺度
    
    核心指标:
    - delay_step: KL首次超过阈值的步数 (约束何时开始生效)
    - KL profile shape: 不仅是slope, 而是完整曲线形态
    - delay_ratio: KL[0]/max(KL) — 初始爆发 vs 渐进增长
    """
    print("\n" + "="*60)
    print("Exp2: Delay Spectrum Deep Mapping")
    print("="*60)
    
    conditions = list(DELAY_CONDITIONS.keys())
    kl_data = {c: [] for c in conditions}
    
    t_start = time.time()
    
    for si, base in enumerate(sentences):
        base_steps = run_autoregressive(model, tokenizer, device, base, n_steps)
        
        for cond, fn in DELAY_CONDITIONS.items():
            text = fn(base)
            pert_steps = run_autoregressive(model, tokenizer, device, text, n_steps)
            kl_list = [compute_kl(pert_steps[s]['probs'], base_steps[s]['probs']) for s in range(n_steps)]
            kl_data[cond].append(kl_list)
        
        if (si + 1) % 5 == 0 or si == 0:
            elapsed = time.time() - t_start
            print(f"  Sentence {si+1}/{len(sentences)}, elapsed={elapsed:.0f}s")
    
    # --- Analysis ---
    print("\n--- Exp2A: Delay Spectrum Full Profile ---")
    mean_kls = {}
    for cond in conditions:
        mean_kls[cond] = [np.mean([kl[step] for kl in kl_data[cond]]) for step in range(n_steps)]
    
    print(f"{'Step':>4} |", end="")
    for c in conditions:
        print(f" {c[:8]:>8}", end="")
    print()
    
    for step in range(n_steps):
        print(f"{step:4d} |", end="")
        for c in conditions:
            print(f" {mean_kls[c][step]:8.3f}", end="")
        print()
    
    # --- Delay metrics ---
    print("\n--- Exp2B: Delay Metrics ---")
    print(f"{'Condition':<20} {'KL[0]':>8} {'KL[1]':>8} {'KL[2]':>8} {'Slope':>8} "
          f"{'Delay_1':>8} {'Delay_2':>8} {'K0/max':>8} {'Type':>20}")
    
    delay_results = {}
    for cond in conditions:
        arr = np.array(mean_kls[cond])
        slope = np.polyfit(range(n_steps), arr, 1)[0]
        kl0 = arr[0]
        kl1 = arr[1] if len(arr) > 1 else kl0
        kl2 = arr[2] if len(arr) > 2 else kl0
        kl_max = max(arr)
        
        delay_1 = find_delay_step(arr, threshold=1.0)
        delay_2 = find_delay_step(arr, threshold=2.0)
        ratio_k0_max = kl0 / kl_max if kl_max > 0 else 0
        
        # Classify delay type
        if delay_1 == 0:
            dtype = "IMMEDIATE"
        elif delay_1 <= 2:
            dtype = "FAST_DELAY"
        elif delay_1 <= 4:
            dtype = "MEDIUM_DELAY"
        else:
            dtype = "SLOW_DELAY"
        
        print(f"{cond:<20} {kl0:8.3f} {kl1:8.3f} {kl2:8.3f} {slope:8.4f} "
              f"{delay_1:8d} {delay_2:8d} {ratio_k0_max:8.3f} {dtype:>20}")
        
        delay_results[cond] = {
            'kl0': float(kl0), 'kl1': float(kl1), 'kl2': float(kl2),
            'slope': float(slope),
            'delay_1': int(delay_1), 'delay_2': int(delay_2),
            'k0_max_ratio': float(ratio_k0_max),
            'type': dtype,
        }
    
    # --- Semantic vs Random delay comparison ---
    print("\n--- Exp2C: Semantic Delay vs Random Delay ---")
    sem_conds = ["negation", "question", "conditional_if", "conditional_unless",
                 "future_will", "narrative_setup", "suppose_that"]
    rand_conds = ["rand_token", "rand_period"]
    
    for sc in sem_conds:
        if sc not in mean_kls:
            continue
        sem_delay = delay_results[sc]['delay_1']
        sem_kl0 = delay_results[sc]['kl0']
        for rc in rand_conds:
            rand_delay = delay_results[rc]['delay_1']
            rand_kl0 = delay_results[rc]['kl0']
            delta_delay = sem_delay - rand_delay
            delta_kl0 = sem_kl0 - rand_kl0
            print(f"  {sc} vs {rc}: delay_diff={delta_delay:+d}, KL[0]_diff={delta_kl0:+.3f}")
    
    return delay_results


# ========================================================================
# EXP3: Mode Transition Continuity
# ========================================================================
def exp3_mode_continuity(model, tokenizer, device, n_steps, n_samples):
    """
    模式跃迁连续性: 模式切换是离散相变还是连续变化?
    
    方法: 在渐变prompt上测量entropy/KL是否连续变化
    """
    print("\n" + "="*60)
    print("Exp3: Mode Transition Continuity")
    print("="*60)
    
    results = {}
    
    for mode_name, gradient in [("cot", COT_GRADIENT), ("translation", TRANSLATION_GRADIENT), ("coding", CODING_GRADIENT)]:
        print(f"\n--- Mode: {mode_name} ---")
        
        # 参考轨迹: 第一个prompt(无模式)
        ref_text = gradient[0]
        ref_steps = run_autoregressive(model, tokenizer, device, ref_text, n_steps)
        
        mode_data = []
        for gi, text in enumerate(gradient):
            steps = run_autoregressive(model, tokenizer, device, text, n_steps)
            
            # KL vs reference
            kl_list = [compute_kl(steps[s]['probs'], ref_steps[s]['probs']) for s in range(n_steps)]
            
            # Entropy profile
            entropy_list = [s['entropy'] for s in steps]
            
            # First token basin
            tokens = sample_first_tokens(model, tokenizer, device, text, n_samples)
            token_counts = Counter(tokens)
            n_unique = len(token_counts)
            total = sum(token_counts.values())
            top1 = token_counts.most_common(1)[0][1] / total if token_counts else 0
            
            mode_data.append({
                'text': text,
                'kl_vs_ref': kl_list,
                'entropy': entropy_list,
                'n_basins': n_unique,
                'top1_prob': top1,
                'top3_tokens': token_counts.most_common(3),
            })
            
            print(f"  [{gi}] '{text[:50]}...'")
            print(f"       KL[0]={kl_list[0]:.3f}, KL[-1]={kl_list[-1]:.3f}, "
                  f"H[0]={entropy_list[0]:.3f}, basins={n_unique}, top1={top1:.3f}")
        
        # Check continuity: is the transition smooth or discontinuous?
        kl0_values = [d['kl_vs_ref'][0] for d in mode_data]
        entropy0_values = [d['entropy'][0] for d in mode_data]
        basins_values = [d['n_basins'] for d in mode_data]
        
        # Maximum jump in KL[0] between adjacent gradient levels
        max_kl_jump = 0
        max_jump_idx = -1
        for i in range(1, len(kl0_values)):
            jump = abs(kl0_values[i] - kl0_values[i-1])
            if jump > max_kl_jump:
                max_kl_jump = jump
                max_jump_idx = i
        
        # Maximum jump in entropy
        max_ent_jump = 0
        for i in range(1, len(entropy0_values)):
            jump = abs(entropy0_values[i] - entropy0_values[i-1])
            if jump > max_ent_jump:
                max_ent_jump = jump
        
        is_continuous = max_kl_jump < 3.0  # 如果跳跃<3, 认为是连续的
        
        print(f"\n  Continuity analysis for {mode_name}:")
        print(f"    KL[0] values: {[f'{v:.3f}' for v in kl0_values]}")
        print(f"    Entropy[0] values: {[f'{v:.3f}' for v in entropy0_values]}")
        print(f"    Basins values: {basins_values}")
        print(f"    Max KL jump: {max_kl_jump:.3f} (at step {max_jump_idx})")
        print(f"    Max entropy jump: {max_ent_jump:.3f}")
        print(f"    >>> Mode transition: {'CONTINUOUS' if is_continuous else 'DISCONTINUOUS (phase transition)'}")
        
        results[mode_name] = {
            'kl0_values': kl0_values,
            'entropy0_values': entropy0_values,
            'basins_values': basins_values,
            'max_kl_jump': max_kl_jump,
            'max_ent_jump': max_ent_jump,
            'is_continuous': is_continuous,
        }
    
    return results


# ========================================================================
# MAIN
# ========================================================================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    is_lite = model_name != "qwen3"
    
    t0 = time.time()
    print(f"[Phase 199] Syntax-Controlled Semantic Perturbation + Delay Spectrum — {model_name}")
    print(f"[Phase 199] Time: {datetime.now()}")
    print(f"[Phase 199] Lite mode: {is_lite}")
    
    # Load model
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    print(f"[load] {model_name}: {info.model_class}, {info.n_layers}L, d={info.d_model}")
    
    # Config
    if is_lite:
        n_steps = 8
        n_samples = 15
        n_sentences = 8
        sentences = LITE_SENTENCES
    else:
        n_steps = 12
        n_samples = 30
        n_sentences = 20
        sentences = BASE_SENTENCES[:n_sentences]
    
    print(f"\nConfig: {n_sentences} sentences, {n_steps} steps, {n_samples} samples")
    
    # Run experiments
    exp1_results = exp1_syntax_semantic(model, tokenizer, device, sentences, n_steps, n_samples)
    exp2_results = exp2_delay_spectrum(model, tokenizer, device, sentences, n_steps)
    exp3_results = exp3_mode_continuity(model, tokenizer, device, n_steps, n_samples)
    
    # ===== Final Summary =====
    print("\n" + "="*60)
    print(f"PHASE 199 SUMMARY — {model_name}")
    print("="*60)
    
    print("\n1. Syntax vs Semantic Signals:")
    for key, val in exp1_results['semantic_signals'].items():
        print(f"   {key}: sem_KL[0]={val['sem_kl0']:.3f}, syn_KL[0]={val['syn_kl0']:.3f}, "
              f"delta={val['delta']:+.3f}, semantic={val['is_semantic']}")
    
    print("\n2. Delay Spectrum:")
    for cond, data in exp2_results.items():
        print(f"   {cond}: KL[0]={data['kl0']:.3f}, delay_1={data['delay_1']}, "
              f"slope={data['slope']:.4f}, type={data['type']}")
    
    print("\n3. Mode Continuity:")
    for mode, data in exp3_results.items():
        print(f"   {mode}: max_kl_jump={data['max_kl_jump']:.3f}, "
              f"continuous={data['is_continuous']}")
    
    # Save results
    out_path = Path(f"tests/glm5_temp/phase199_{model_name}_results.json")
    
    # Convert numpy types for JSON
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
        'exp1_semantic_signals': exp1_results['semantic_signals'],
        'exp1_slopes': {k: convert(v) for k, v in exp1_results['slopes'].items()},
        'exp2_delay': exp2_results,
        'exp3_continuity': {k: {kk: convert(vv) for kk, vv in v.items()} for k, v in exp3_results.items()},
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")
    
    # Release
    elapsed = time.time() - t0
    print(f"\n[Phase 199] COMPLETE in {elapsed:.1f}s ({model_name})")
    release_model(model)


if __name__ == "__main__":
    main()
