"""
Phase 198: Mode Transition & Random Perturbation Control
========================================================

核心理论升级: 约束 → 模式+约束的三层结构
  Level 1: Mode (全局模式) — 决定运行什么程序
  Level 2: Constraint (约束) — 决定允许什么续写
  Level 3: Neural Realization (神经实现) — Attention/MLP/Residual

两大关键实验:

Exp1: Random Perturbation Control (最关键!)
  区分"语义传播" vs "自回归混沌"
  - semantic perturbation: negation/question/conditional
  - random perturbation: 加随机token "xyz"/"and"/标点
  - paraphrase perturbation: 同义改写 (语义保持但token不同)
  如果random的KL也递增 → KL递增是tautology
  如果semantic的KL递增显著不同 → 语义传播真实

Exp2: Mode Transition Analysis
  观察模式触发token是否导致"相变"
  - CoT trigger: "Let's think step by step"
  - Translation trigger: "Translate to Chinese:"
  - Contrast trigger: "However,"
  - QA trigger: "Question:"
  - Narrative trigger: "Once upon a time"
  - Coding trigger: "def "
  测量: 熵动力学/分支因子/attractor basin在trigger前后的突变

数据量: Qwen3=25句×12步×40采样, GLM4/DS7B=10句×8步×15采样
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

# 用bf16方式加载(不用8bit)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tests"))
from model_demo_bf16 import load_model_bf16, get_device_for_input
from model_utils import get_model_info, release_model

# ===== 25个基础句 (与Phase 197相同) =====
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
    "The scholar translates the text",
    "The captain commands the ship",
    "The manager oversees the project",
    "The programmer writes the code",
    "The analyst reviews the data",
]

# ===== Exp1: 扰动类型 =====
def make_perturbations(base):
    """生成4类扰动"""
    return {
        # 语义扰动 (之前的约束)
        "negation": f"{base.replace(' ', ' does not ', 1).replace(base.split()[1]+' does not', base.split()[1]+' does not')}",
        "question": f"Does {base[0].lower() + base[1:]}?",

        # 随机扰动
        "rand_token": f"{base} xyz",
        "rand_and": f"{base} and",
        "rand_period": f"{base}.",

        # 同义改写 (语义保持)
        "paraphrase_cat": base.replace("The cat chases the dog", "The feline pursues the hound")
                             .replace("The teacher helps the student", "The instructor assists the pupil")
                             .replace("The doctor treats the patient", "The physician heals the sick"),
    }

def make_negation(base):
    words = base.split()
    if len(words) >= 2:
        return words[0] + " " + words[1] + " does not " + " ".join(words[2:])
    return base

def make_question(base):
    return "Does " + base[0].lower() + base[1:] + "?"

# ===== Exp2: 模式触发句 =====
MODE_TRIGGERS = {
    "cot": [
        ("Problem: 2+3=? ", "Problem: 2+3=? Let's think step by step"),
        ("Question: What is 15*4? ", "Question: What is 15*4? Let's think step by step"),
        ("How many apples? ", "How many apples? Let's think step by step"),
        ("Calculate 7*8. ", "Calculate 7*8. Let's think step by step"),
        ("Solve for x: 2x=10. ", "Solve for x: 2x=10. Let's think step by step"),
    ],
    "translation": [
        ("The cat sleeps. ", "Translate to Chinese: The cat sleeps. "),
        ("He is happy. ", "Translate to Chinese: He is happy. "),
        ("The sky is blue. ", "Translate to Chinese: The sky is blue. "),
        ("Water boils at 100C. ", "Translate to Chinese: Water boils at 100C. "),
        ("The book is on the table. ", "Translate to Chinese: The book is on the table. "),
    ],
    "contrast": [
        ("He is poor", "He is poor, but"),
        ("She failed the exam", "She failed the exam, however"),
        ("The weather was terrible", "The weather was terrible, nevertheless"),
        ("The plan seemed impossible", "The plan seemed impossible, yet"),
        ("He was tired", "He was tired, but"),
    ],
    "qa": [
        ("The cat sleeps on the mat", "Question: The cat sleeps on the mat"),
        ("Paris is the capital of France", "Question: Paris is the capital of France"),
        ("Water freezes at 0 degrees", "Question: Water freezes at 0 degrees"),
        ("Einstein developed relativity", "Question: Einstein developed relativity"),
        ("The earth orbits the sun", "Question: The earth orbits the sun"),
    ],
    "narrative": [
        ("There was a village", "Once upon a time, there was a village"),
        ("A man walked alone", "Once upon a time, a man walked alone"),
        ("In the forest lived a wolf", "Once upon a time, in the forest lived a wolf"),
        ("She opened the door", "Once upon a time, she opened the door"),
        ("The kingdom prospered", "Once upon a time, the kingdom prospered"),
    ],
    "coding": [
        ("Add two numbers", "def add_two_numbers"),
        ("Sort a list", "def sort_list"),
        ("Find the maximum", "def find_max"),
        ("Calculate factorial", "def factorial"),
        ("Reverse a string", "def reverse_string"),
    ],
}

N_STEPS = 12
N_SAMPLES = 40  # Qwen3
N_STEPS_LITE = 8
N_SAMPLES_LITE = 15  # GLM4/DS7B


def safe_softmax(logits, device):
    """安全的softmax, 处理跨device的logits"""
    return F.softmax(logits.to(device).float(), dim=-1)


def compute_kl(p, q):
    """KL(p||q) with NaN protection"""
    p = p.float().clamp(min=1e-10)
    q = q.float().clamp(min=1e-10)
    kl = (p * (p.log() - q.log())).sum().item()
    if math.isnan(kl) or math.isinf(kl):
        return 0.0
    return max(kl, 0.0)


def compute_js(p, q):
    """Jensen-Shannon divergence"""
    m = 0.5 * (p + q)
    return 0.5 * compute_kl(p, m) + 0.5 * compute_kl(q, m)


def compute_entropy(probs):
    """Shannon entropy"""
    p = probs.float().clamp(min=1e-10)
    h = -(p * p.log()).sum().item()
    if math.isnan(h):
        return 0.0
    return h


def effective_branching(probs, threshold=0.01):
    """有效分支因子: 概率>threshold的token数"""
    return (probs > threshold).sum().item()


def run_autoregressive(model, tokenizer, device, text, n_steps):
    """自回归生成n步, 返回每步的probs"""
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
            'probs': probs[0],  # keep for KL computation
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


# ========================================================================
# EXP1: Random Perturbation Control
# ========================================================================
def exp1_perturbation_control(model, tokenizer, device, sentences, n_steps, n_samples):
    """
    核心对照实验: 语义扰动 vs 随机扰动 vs 同义改写

    关键指标:
    - KL递增速率 (semantic vs random)
    - 约束延迟谱 (constraint delay spectrum)
    - Attractor basin结构差异
    """
    print("\n" + "="*60)
    print("Exp1: Random Perturbation Control")
    print("="*60)

    conditions = ["negation", "question", "rand_token", "rand_and", "rand_period", "paraphrase"]
    kl_profiles = {c: [] for c in conditions}  # 每句每步的KL
    entropy_profiles = {c: [] for c in conditions}
    basin_data = {c: [] for c in conditions}

    for si, base in enumerate(sentences):
        # Base trajectory
        base_steps = run_autoregressive(model, tokenizer, device, base, n_steps)

        for cond in conditions:
            # Construct perturbed text
            if cond == "negation":
                text = make_negation(base)
            elif cond == "question":
                text = make_question(base)
            elif cond == "rand_token":
                text = base + " xyz"
            elif cond == "rand_and":
                text = base + " and"
            elif cond == "rand_period":
                text = base + "."
            elif cond == "paraphrase":
                # Simple paraphrase: swap "The" with "A"
                text = base.replace("The ", "A ", 1) if base.startswith("The ") else base + " too"
            else:
                continue

            pert_steps = run_autoregressive(model, tokenizer, device, text, n_steps)

            # Compute KL at each step
            kl_list = []
            for step in range(n_steps):
                kl = compute_kl(pert_steps[step]['probs'], base_steps[step]['probs'])
                kl_list.append(kl)
            kl_profiles[cond].append(kl_list)
            entropy_profiles[cond].append([s['entropy'] for s in pert_steps])

            # Sample first tokens for basin analysis (only 3 representative sentences)
            if si < 3:
                tokens = sample_first_tokens(model, tokenizer, device, text, n_samples)
                basin_data[cond].append(Counter(tokens))

        if (si + 1) % 5 == 0:
            print(f"  Sentence {si+1}/{len(sentences)} done, GPU={torch.cuda.memory_allocated()/1e9:.2f}GB")

    # --- Analysis ---
    print("\n--- Exp1A: KL Profile Comparison ---")
    print(f"{'Step':>4} |", end="")
    for c in conditions:
        print(f" {c[:8]:>8}", end="")
    print()

    mean_kls = {}
    for step in range(n_steps):
        print(f"{step:4d} |", end="")
        for c in conditions:
            mean_kl = np.mean([kl[step] for kl in kl_profiles[c]])
            if step == 0:
                mean_kls[c] = []
            mean_kls[c].append(mean_kl)
            print(f" {mean_kl:8.3f}", end="")
        print()

    # KL slope comparison
    print("\n--- Exp1B: KL Slope Comparison ---")
    print(f"{'Condition':<15} {'Slope':>8} {'Trend':>12} {'KL[0]':>8} {'KL[-1]':>8}")
    slopes = {}
    for c in conditions:
        mean_kl_arr = np.array(mean_kls[c])
        slope = np.polyfit(range(n_steps), mean_kl_arr, 1)[0]
        slopes[c] = slope
        trend = "INCREASING" if slope > 0.3 else ("DECREASING" if slope < -0.3 else "STABLE")
        print(f"{c:<15} {slope:8.4f} {trend:>12} {mean_kl_arr[0]:8.3f} {mean_kl_arr[-1]:8.3f}")

    # Critical comparison: semantic vs random
    print("\n--- Exp1C: Semantic vs Random KL Slope Ratio ---")
    sem_slopes = [slopes["negation"], slopes["question"]]
    rand_slopes = [slopes["rand_token"], slopes["rand_and"], slopes["rand_period"]]
    mean_sem = np.mean(sem_slopes)
    mean_rand = np.mean(rand_slopes)
    ratio = mean_sem / mean_rand if mean_rand > 0 else float('inf')
    print(f"  Semantic mean slope: {mean_sem:.4f}")
    print(f"  Random mean slope:   {mean_rand:.4f}")
    print(f"  Ratio (sem/rand):    {ratio:.2f}")
    if ratio > 2.0:
        print(f"  >>> Semantic propagation significantly stronger than random ({ratio:.1f}x)")
    elif ratio > 1.3:
        print(f"  >>> Semantic propagation moderately stronger ({ratio:.1f}x)")
    else:
        print(f"  >>> WARNING: KL increase may be autoregressive tautology (ratio={ratio:.1f})")

    # Basin comparison
    print("\n--- Exp1D: Attractor Basin Structure ---")
    for c in conditions:
        if c in basin_data and basin_data[c]:
            all_tokens = []
            for bc in basin_data[c]:
                all_tokens.extend(bc.elements())
            counts = Counter(all_tokens)
            n_unique = len(counts)
            total = sum(counts.values())
            top1 = counts.most_common(1)[0][1] / total if counts else 0
            probs_arr = np.array([v/total for v in counts.values()])
            ent = -(probs_arr * np.log(probs_arr + 1e-12)).sum()
            print(f"  {c:<15} basins={n_unique:3d}, top1={top1:.3f}, entropy={ent:.2f}")

    return {
        'kl_profiles': {c: mean_kls[c] for c in conditions},
        'slopes': slopes,
        'ratio': ratio,
    }


# ========================================================================
# EXP2: Mode Transition Analysis
# ========================================================================
def exp2_mode_transition(model, tokenizer, device, n_steps, n_samples):
    """
    模式跃迁分析: 模式触发token是否导致"相变"

    对每种mode, 比较:
    - base文本 vs 触发后文本的熵动力学
    - 触发点的熵/分支因子突变
    - Attractor basin结构变化
    """
    print("\n" + "="*60)
    print("Exp2: Mode Transition Analysis")
    print("="*60)

    results = {}

    for mode_name, pairs in MODE_TRIGGERS.items():
        print(f"\n--- Mode: {mode_name} ---")

        for pi, (base_text, trigger_text) in enumerate(pairs):
            # Run both trajectories
            base_steps = run_autoregressive(model, tokenizer, device, base_text, n_steps)
            trigger_steps = run_autoregressive(model, tokenizer, device, trigger_text, n_steps)

            # Compute KL at each step
            kl_list = []
            for step in range(n_steps):
                kl = compute_kl(trigger_steps[step]['probs'], base_steps[step]['probs'])
                kl_list.append(kl)

            # Entropy difference
            base_entropies = [s['entropy'] for s in base_steps]
            trigger_entropies = [s['entropy'] for s in trigger_steps]
            delta_h = [t - b for t, b in zip(trigger_entropies, base_entropies)]

            # Phase transition detection: maximum entropy change
            max_delta_h = max(delta_h, key=abs) if delta_h else 0
            max_delta_step = delta_h.index(max_delta_h) if delta_h else 0

            # First token basins
            base_tokens = sample_first_tokens(model, tokenizer, device, base_text, n_samples)
            trigger_tokens = sample_first_tokens(model, tokenizer, device, trigger_text, n_samples)

            base_counts = Counter(base_tokens)
            trigger_counts = Counter(trigger_tokens)

            if pi == 0:  # Only print detailed for first pair
                print(f"  Pair {pi+1}: '{base_text[:40]}...' / '{trigger_text[:40]}...'")
                print(f"  KL profile: {kl_list[0]:.2f} -> {kl_list[-1]:.2f}")
                print(f"  Max entropy change: step {max_delta_step}, delta_H={max_delta_h:.3f}")
                print(f"  Base basin:   top3={base_counts.most_common(3)}")
                print(f"  Trigger basin: top3={trigger_counts.most_common(3)}")

                # Print step-by-step entropy for first pair
                print(f"  {'Step':>4} | {'H(base)':>8} {'H(trig)':>8} {'dH':>8} {'KL':>8}")
                for step in range(min(n_steps, 8)):
                    print(f"  {step:4d} | {base_entropies[step]:8.3f} {trigger_entropies[step]:8.3f} "
                          f"{delta_h[step]:8.3f} {kl_list[step]:8.3f}")

        # Aggregate analysis for this mode
        # Run all pairs and collect metrics
        mode_kl_slopes = []
        mode_delta_h_max = []
        for base_text, trigger_text in pairs:
            bs = run_autoregressive(model, tokenizer, device, base_text, n_steps)
            ts = run_autoregressive(model, tokenizer, device, trigger_text, n_steps)
            kl_arr = [compute_kl(ts[s]['probs'], bs[s]['probs']) for s in range(n_steps)]
            if len(kl_arr) > 1:
                slope = np.polyfit(range(n_steps), kl_arr, 1)[0]
                mode_kl_slopes.append(slope)
            dh = [ts[s]['entropy'] - bs[s]['entropy'] for s in range(n_steps)]
            mode_delta_h_max.append(max(dh, key=abs) if dh else 0)

        mean_slope = np.mean(mode_kl_slopes) if mode_kl_slopes else 0
        mean_dh = np.mean(mode_delta_h_max)
        print(f"\n  Mode {mode_name} summary:")
        print(f"    Mean KL slope: {mean_slope:.4f}")
        print(f"    Mean max entropy change: {mean_dh:.3f}")

        results[mode_name] = {
            'kl_slope': mean_slope,
            'mean_delta_h': mean_dh,
        }

    # Mode comparison table
    print("\n--- Exp2 Summary: Mode Transition Effects ---")
    print(f"{'Mode':<15} {'KL slope':>10} {'Mean dH':>10} {'Type':>20}")
    for mode, data in results.items():
        slope = data['kl_slope']
        dh = data['mean_delta_h']
        # Classify mode type
        if abs(dh) > 1.0:
            mtype = "ENTROPY SHIFT"
        elif slope > 1.0:
            mtype = "TRAJECTORY DIVERGE"
        elif slope > 0.3:
            mtype = "CONSTRAINT PROP"
        else:
            mtype = "MINIMAL EFFECT"
        print(f"{mode:<15} {slope:10.4f} {dh:10.3f} {mtype:>20}")

    return results


# ========================================================================
# EXP3: Constraint Delay Spectrum
# ========================================================================
def exp3_delay_spectrum(model, tokenizer, device, sentences, n_steps):
    """
    约束延迟谱: 不同语义操作的时间传播结构

    Hypothesis:
    - negation: 即时生效 (KL[0]大, slope平缓)
    - question: 即时模式切换 (KL[0]最大)
    - conditional: 延迟约束 (KL[0]小, slope大)
    - role_binding: 渐进传播 (KL[0]中等, slope中等)
    """
    print("\n" + "="*60)
    print("Exp3: Constraint Delay Spectrum")
    print("="*60)

    conditions_map = {
        "negation": lambda b: make_negation(b),
        "question": lambda b: make_question(b),
        "conditional": lambda b: f"If {b[0].lower() + b[1:]}",
        "role_binding": lambda b: " ".join(reversed(b.split())),
    }

    # Also add random controls
    conditions_map["rand_token"] = lambda b: b + " xyz"
    conditions_map["rand_period"] = lambda b: b + "."

    kl_data = {c: [] for c in conditions_map}

    for si, base in enumerate(sentences):
        base_steps = run_autoregressive(model, tokenizer, device, base, n_steps)

        for cond, fn in conditions_map.items():
            text = fn(base)
            pert_steps = run_autoregressive(model, tokenizer, device, text, n_steps)

            kl_list = [compute_kl(pert_steps[s]['probs'], base_steps[s]['probs']) for s in range(n_steps)]
            kl_data[cond].append(kl_list)

        if (si + 1) % 5 == 0:
            print(f"  Sentence {si+1}/{len(sentences)} done")

    # Analysis
    print("\n--- Delay Spectrum: KL[0] vs KL slope ---")
    print(f"{'Condition':<15} {'KL[0]':>8} {'KL[-1]':>8} {'Slope':>8} {'Delay Type':>20}")

    delay_results = {}
    for cond in conditions_map:
        mean_kls = [np.mean([kl[step] for kl in kl_data[cond]]) for step in range(n_steps)]
        slope = np.polyfit(range(n_steps), mean_kls, 1)[0]
        kl0 = mean_kls[0]
        kl_last = mean_kls[-1]

        # Classify delay type
        if kl0 > 5.0 and slope < 1.0:
            delay_type = "IMMEDIATE (instant)"
        elif kl0 > 2.0 and slope > 0.5:
            delay_type = "IMMEDIATE+PROPAGATING"
        elif kl0 < 1.0 and slope > 0.5:
            delay_type = "DELAYED (deferred)"
        elif kl0 < 0.5 and slope < 0.3:
            delay_type = "MINIMAL"
        else:
            delay_type = "MODERATE"

        print(f"{cond:<15} {kl0:8.3f} {kl_last:8.3f} {slope:8.4f} {delay_type:>20}")
        delay_results[cond] = {'kl0': kl0, 'kl_last': kl_last, 'slope': slope, 'type': delay_type}

    return delay_results


# ========================================================================
# MAIN
# ========================================================================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    is_lite = model_name != "qwen3"  # GLM4/DS7B use lite version

    t0 = time.time()
    print(f"[Phase 198] Mode Transition & Random Perturbation Control — {model_name}")
    print(f"[Phase 198] Time: {datetime.now()}")
    print(f"[Phase 198] Lite mode: {is_lite}")

    # Load model (bf16 + device_map="auto", NOT 8bit)
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    print(f"[load] {model_name}: {info.model_class}, {info.n_layers} layers, d={info.d_model}")

    n_steps = N_STEPS_LITE if is_lite else N_STEPS
    n_samples = N_SAMPLES_LITE if is_lite else N_SAMPLES
    n_sentences = 10 if is_lite else len(BASE_SENTENCES)
    sentences = BASE_SENTENCES[:n_sentences]

    print(f"\nConfig: {n_sentences} sentences, {n_steps} steps, {n_samples} samples")

    # Run experiments
    exp1_results = exp1_perturbation_control(model, tokenizer, device, sentences, n_steps, n_samples)
    exp2_results = exp2_mode_transition(model, tokenizer, device, n_steps, n_samples)
    exp3_results = exp3_delay_spectrum(model, tokenizer, device, sentences, n_steps)

    # ===== Final Summary =====
    print("\n" + "="*60)
    print(f"PHASE 198 SUMMARY — {model_name}")
    print("="*60)

    print("\n1. Semantic vs Random Perturbation:")
    print(f"   Semantic mean KL slope: {np.mean([exp1_results['slopes']['negation'], exp1_results['slopes']['question']]):.4f}")
    print(f"   Random mean KL slope:   {np.mean([exp1_results['slopes']['rand_token'], exp1_results['slopes']['rand_and'], exp1_results['slopes']['rand_period']]):.4f}")
    print(f"   Ratio: {exp1_results['ratio']:.2f}x")
    if exp1_results['ratio'] > 1.5:
        print("   >>> KL increase reflects REAL semantic propagation (not just chaos)")
    else:
        print("   >>> WARNING: KL increase may be autoregressive tautology")

    print("\n2. Mode Transition Effects:")
    for mode, data in exp2_results.items():
        print(f"   {mode}: KL slope={data['kl_slope']:.4f}, mean dH={data['mean_delta_h']:.3f}")

    print("\n3. Constraint Delay Spectrum:")
    for cond, data in exp3_results.items():
        print(f"   {cond}: KL[0]={data['kl0']:.3f}, slope={data['slope']:.4f} ({data['type']})")

    # Save results
    out_path = Path(f"tests/glm5_temp/phase198_{model_name}_results.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'model': model_name,
            'exp1_ratio': exp1_results['ratio'],
            'exp1_slopes': exp1_results['slopes'],
            'exp2_modes': exp2_results,
            'exp3_delay': exp3_results,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

    # Release
    elapsed = time.time() - t0
    print(f"\n[Phase 198] COMPLETE in {elapsed:.1f}s ({model_name})")
    release_model(model)


if __name__ == "__main__":
    main()
