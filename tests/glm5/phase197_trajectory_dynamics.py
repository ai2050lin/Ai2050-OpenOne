"""
Phase 197: Trajectory-Level Constraint Dynamics
================================================

核心转向: 从"单步KL"到"多步生成轨迹"分析

用户关键批评:
1. KL不是语义量 — KL测量distributional divergence, 不是semantic transformation
2. 把"输出约束"误认为"内部计算" — 语义通过约束体现, 不等于约束
3. KL加性可能是数学假象 — 小扰动下KL≈二次型, 自然近似加性
4. 忽略自回归耦合 — 应研究P(x_{t:T}), 不是P(x_{t+1})
5. 语义=对未来生成轨迹的动态可达性控制

关键实验:
1. Multi-step Distribution Evolution: 每步的熵/分支因子/top-k tokens
2. Trajectory Divergence Rate: base vs constraint在每步的top-k overlap
3. Attractor Basin Structure: 采样续写→首token分析→盆地结构
4. Conditional Delayed Effect: conditional是否随步数增加KL递增?
5. Reachable Continuation Space: 每步可达token数/概率质量分布

数据量: 30句对 (Qwen3), 20句对 (GLM4/DS7B), 40个采样续写
"""

import sys
import os
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

import gc
import time
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime

from model_utils import get_model_info, release_model, MODEL_CONFIGS

# ===== 30个基础句 (与Phase 196相同) =====
SENTENCE_SEXTETS = [
    ("The cat chases the dog",
     "The cat does not chase the dog",
     "Does the cat chase the dog?",
     "The dog chases the cat",
     "Does the cat not chase the dog?",
     "Does the dog chase the cat?",
     "The dog does not chase the cat"),
    ("The teacher helps the student",
     "The teacher does not help the student",
     "Does the teacher help the student?",
     "The student helps the teacher",
     "Does the teacher not help the student?",
     "Does the student help the teacher?",
     "The student does not help the teacher"),
    ("The leader guides the team",
     "The leader does not guide the team",
     "Does the leader guide the team?",
     "The team guides the leader",
     "Does the leader not guide the team?",
     "Does the team guide the leader?",
     "The team does not guide the leader"),
    ("The doctor treats the patient",
     "The doctor does not treat the patient",
     "Does the doctor treat the patient?",
     "The patient treats the doctor",
     "Does the doctor not treat the patient?",
     "Does the patient treat the doctor?",
     "The patient does not treat the doctor"),
    ("The chef cooks the meal",
     "The chef does not cook the meal",
     "Does the chef cook the meal?",
     "The meal cooks the chef",
     "Does the chef not cook the meal?",
     "Does the meal cook the chef?",
     "The meal does not cook the chef"),
    ("The writer drafts the letter",
     "The writer does not draft the letter",
     "Does the writer draft the letter?",
     "The letter drafts the writer",
     "Does the writer not draft the letter?",
     "Does the letter draft the writer?",
     "The letter does not draft the writer"),
    ("The farmer plants the seed",
     "The farmer does not plant the seed",
     "Does the farmer plant the seed?",
     "The seed plants the farmer",
     "Does the farmer not plant the seed?",
     "Does the seed plant the farmer?",
     "The seed does not plant the farmer"),
    ("The artist paints the portrait",
     "The artist does not paint the portrait",
     "Does the artist paint the portrait?",
     "The portrait paints the artist",
     "Does the artist not paint the portrait?",
     "Does the portrait paint the artist?",
     "The portrait does not paint the artist"),
    ("The singer performs the song",
     "The singer does not perform the song",
     "Does the singer perform the song?",
     "The song performs the singer",
     "Does the singer not perform the song?",
     "Does the song perform the singer?",
     "The song does not perform the singer"),
    ("The driver steers the car",
     "The driver does not steer the car",
     "Does the driver steer the car?",
     "The car steers the driver",
     "Does the driver not steer the car?",
     "Does the car steer the driver?",
     "The car does not steer the driver"),
    ("The scientist discovers the element",
     "The scientist does not discover the element",
     "Does the scientist discover the element?",
     "The element discovers the scientist",
     "Does the scientist not discover the element?",
     "Does the element discover the scientist?",
     "The element does not discover the scientist"),
    ("The judge rules the court",
     "The judge does not rule the court",
     "Does the judge rule the court?",
     "The court rules the judge",
     "Does the judge not rule the court?",
     "Does the court rule the judge?",
     "The court does not rule the judge"),
    ("The child reads the book",
     "The child does not read the book",
     "Does the child read the book?",
     "The book reads the child",
     "Does the child not read the book?",
     "Does the book read the child?",
     "The book does not read the child"),
    ("The guard protects the palace",
     "The guard does not protect the palace",
     "Does the guard protect the palace?",
     "The palace protects the guard",
     "Does the guard not protect the palace?",
     "Does the palace protect the guard?",
     "The palace does not protect the guard"),
    ("The baker bakes the bread",
     "The baker does not bake the bread",
     "Does the baker bake the bread?",
     "The bread bakes the baker",
     "Does the baker not bake the bread?",
     "Does the bread bake the baker?",
     "The bread does not bake the baker"),
    ("The pilot flies the plane",
     "The pilot does not fly the plane",
     "Does the pilot fly the plane?",
     "The plane flies the pilot",
     "Does the pilot not fly the plane?",
     "Does the plane fly the pilot?",
     "The plane does not fly the pilot"),
    ("The nurse cares for the patient",
     "The nurse does not care for the patient",
     "Does the nurse care for the patient?",
     "The patient cares for the nurse",
     "Does the nurse not care for the patient?",
     "Does the patient care for the nurse?",
     "The patient does not care for the nurse"),
    ("The soldier defends the city",
     "The soldier does not defend the city",
     "Does the soldier defend the city?",
     "The city defends the soldier",
     "Does the soldier not defend the city?",
     "Does the city defend the soldier?",
     "The city does not defend the soldier"),
    ("The student solves the problem",
     "The student does not solve the problem",
     "Does the student solve the problem?",
     "The problem solves the student",
     "Does the student not solve the problem?",
     "Does the problem solve the student?",
     "The problem does not solve the student"),
    ("The builder constructs the house",
     "The builder does not construct the house",
     "Does the builder construct the house?",
     "The house constructs the builder",
     "Does the builder not construct the house?",
     "Does the house construct the builder?",
     "The house does not construct the builder"),
    ("The cat sleeps on the mat",
     "The cat does not sleep on the mat",
     "Does the cat sleep on the mat?",
     "The mat sleeps on the cat",
     "Does the cat not sleep on the mat?",
     "Does the mat sleep on the cat?",
     "The mat does not sleep on the cat"),
    ("The dog runs through the park",
     "The dog does not run through the park",
     "Does the dog run through the park?",
     "The park runs through the dog",
     "Does the dog not run through the park?",
     "Does the park run through the dog?",
     "The park does not run through the dog"),
    ("The bird sings in the tree",
     "The bird does not sing in the tree",
     "Does the bird sing in the tree?",
     "The tree sings in the bird",
     "Does the bird not sing in the tree?",
     "Does the tree sing in the bird?",
     "The tree does not sing in the bird"),
    ("The fish swims in the river",
     "The fish does not swim in the river",
     "Does the fish swim in the river?",
     "The river swims in the fish",
     "Does the fish not swim in the river?",
     "Does the river swim in the fish?",
     "The river does not swim in the fish"),
    ("The wind blows across the field",
     "The wind does not blow across the field",
     "Does the wind blow across the field?",
     "The field blows across the wind",
     "Does the wind not blow across the field?",
     "Does the field blow across the wind?",
     "The field does not blow across the wind"),
]

# ===== 条件句测试 (检测延迟约束效应) =====
# 关键设计: conditional的约束效应可能在后续token中才显现
CONDITIONAL_PAIRS = [
    {"base": "The cat chases the dog",
     "conditional": "If the cat chases the dog",
     "question": "Does the cat chase the dog?"},
    {"base": "The ground is dry",
     "conditional": "If it rains, the ground",
     "question": "Is the ground dry?"},
    {"base": "The student studies hard",
     "conditional": "If the student studies hard, the student",
     "question": "Does the student study hard?"},
    {"base": "The temperature drops below zero",
     "conditional": "If the temperature drops below zero, the water",
     "question": "Does the temperature drop below zero?"},
    {"base": "The sun rises in the east",
     "conditional": "If the sun rises in the east, the sky",
     "question": "Does the sun rise in the east?"},
]

CONDITION_NAMES = ["base", "negation", "question", "role_binding", "neg+q", "q+rb", "neg+rb"]
CONDITIONAL_NAMES = ["base", "conditional", "question"]


def load_model_bf16(model_name: str):
    """BF16 + device_map=auto + flash attention"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = MODEL_CONFIGS[model_name]
    print(f"[load] Loading {model_name} (bf16 + auto + flash)...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try flash_attention_2 first, fall back to sdpa, then eager
    for attn_impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"],
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
                attn_implementation=attn_impl,
            )
            print(f"[load] {model_name} loaded with attn_impl={attn_impl}")
            break
        except Exception as e:
            print(f"[load] attn_impl={attn_impl} failed: {e}")
            if attn_impl == "eager":
                raise
            continue

    model.eval()
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"[load] device={device}, GPU={gpu_mem:.2f}GB, class={type(model).__name__}")
    return model, tokenizer, device


def get_input_device(model):
    """获取输入tensor应放的设备"""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===== 信息论工具函数 =====

def compute_entropy(probs_np):
    """Shannon entropy from probability array"""
    probs_np = probs_np[probs_np > 0]
    return -float(np.sum(probs_np * np.log2(probs_np)))


def compute_renyi_entropy(probs_np, alpha):
    """Renyi entropy of order alpha"""
    if alpha == 1.0:
        return compute_entropy(probs_np)
    if alpha == 0.0:
        return float(np.log2(np.sum(probs_np > 0)))
    if alpha == float('inf'):
        return -float(np.log2(np.max(probs_np)))
    probs_pos = probs_np[probs_np > 0]
    return float(1.0 / (1.0 - alpha) * np.log2(np.sum(probs_pos ** alpha)))


def compute_kl(p_np, q_np, eps=1e-10):
    """KL(p || q) — p is reference, q is comparison"""
    p_safe = np.clip(p_np, eps, None)
    q_safe = np.clip(q_np, eps, None)
    return float(np.sum(p_safe * np.log2(p_safe / q_safe)))


def compute_js(p_np, q_np, eps=1e-10):
    """Jensen-Shannon divergence (symmetric, bounded [0,1])"""
    m = 0.5 * (p_np + q_np)
    return 0.5 * compute_kl(p_np, m, eps) + 0.5 * compute_kl(q_np, m, eps)


def compute_effective_branching(probs_np):
    """Effective branching factor = exp(H_2)"""
    h2 = compute_renyi_entropy(probs_np, 2.0)
    return float(2.0 ** h2)


def topk_overlap(set_a, set_b, k=20):
    """Jaccard overlap between top-k token sets"""
    return len(set_a & set_b) / max(len(set_a | set_b), 1)


# ===== Experiment 1: Multi-step Distribution Evolution =====

def multi_step_profile(model, tokenizer, device, sentence, n_steps=12):
    """
    贪婪生成n_steps步, 每步记录:
    - 概率分布 → 熵, 分支因子, top-k tokens
    - 与base的KL/JS/top-k overlap

    返回: list of dicts, 每步一个
    """
    input_device = get_input_device(model)

    # Encode
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(input_device)
    attention_mask = inputs["attention_mask"].to(input_device)

    step_data = []
    past_kv = None

    with torch.no_grad():
        for step in range(n_steps):
            if past_kv is None:
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                            use_cache=True, past_key_values=None)
            else:
                # Only process the last token
                out = model(input_ids=next_input_ids,
                            attention_mask=next_attention_mask,
                            use_cache=True, past_key_values=past_kv)

            logits = out.logits[0, -1].float()  # [vocab]
            past_kv = out.past_key_values

            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            probs = np.clip(probs, 0, None)
            probs = probs / probs.sum()  # renormalize

            # Metrics
            entropy = compute_entropy(probs)
            h0 = compute_renyi_entropy(probs, 0.0)
            h2 = compute_renyi_entropy(probs, 2.0)
            h_inf = compute_renyi_entropy(probs, float('inf'))
            ebf = compute_effective_branching(probs)

            top10_ids = np.argsort(probs)[-10:][::-1]
            top10 = [(tokenizer.decode([int(i)]).strip(), float(probs[i])) for i in top10_ids]
            top20_set = set(int(x) for x in np.argsort(probs)[-20:])

            # Reachable tokens: probability > 0.01
            n_reachable = int(np.sum(probs > 0.01))
            # Probability mass in top-1, top-5, top-10
            sorted_probs = np.sort(probs)[::-1]
            mass_top1 = float(sorted_probs[0]) if len(sorted_probs) > 0 else 0
            mass_top5 = float(np.sum(sorted_probs[:5])) if len(sorted_probs) >= 5 else float(np.sum(sorted_probs))
            mass_top10 = float(np.sum(sorted_probs[:10])) if len(sorted_probs) >= 10 else float(np.sum(sorted_probs))

            step_data.append({
                "step": step,
                "entropy": entropy,
                "h0": h0, "h2": h2, "h_inf": h_inf,
                "ebf": ebf,
                "top10": top10,
                "top20_set": top20_set,
                "n_reachable": n_reachable,
                "mass_top1": mass_top1,
                "mass_top5": mass_top5,
                "mass_top10": mass_top10,
                "probs": probs,  # full distribution for KL/JS computation
            })

            # Greedy next token
            next_token_id = torch.tensor([[int(np.argmax(probs))]], device=input_device)
            next_input_ids = next_token_id
            next_attention_mask = torch.cat([
                attention_mask,
                torch.ones(1, 1, device=input_device, dtype=attention_mask.dtype)
            ], dim=1)
            attention_mask = next_attention_mask

            # Periodic GPU memory check
            if step == 0:
                gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                print(f"    Step 0 done, GPU={gpu_mem:.2f}GB")

    return step_data


# ===== Experiment 2: Attractor Basin Structure =====

def sample_continuations(model, tokenizer, device, sentence, n_samples=40, n_tokens=8, temperature=1.0):
    """
    采样n_samples个续写, 分析首token分布和盆地结构

    返回: {
        "continuations": [list of strings],
        "first_tokens": [list of (token_str, prob)],
        "basin_count": int,
        "basin_concentration": float,
        "basin_entropy": float,
        "first_token_dist": {token: count},
    }
    """
    input_device = get_input_device(model)

    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
    input_ids_base = inputs["input_ids"].to(input_device)
    attention_mask_base = inputs["attention_mask"].to(input_device)

    continuations = []
    first_tokens = []
    first_token_ids = []

    for i in range(n_samples):
        with torch.no_grad():
            gen_ids = model.generate(
                input_ids_base,
                attention_mask=attention_mask_base,
                max_new_tokens=n_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=50,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        # Get the generated part
        gen_part = gen_ids[0, input_ids_base.shape[1]:]
        gen_text = tokenizer.decode(gen_part, skip_special_tokens=True)

        # First token
        if len(gen_part) > 0:
            first_id = int(gen_part[0])
            first_tok = tokenizer.decode([first_id]).strip()
            first_tokens.append(first_tok)
            first_token_ids.append(first_id)

        continuations.append(gen_text)

        if (i + 1) % 10 == 0:
            print(f"    Sampled {i+1}/{n_samples}")

    # Basin analysis
    first_token_counts = Counter(first_tokens)
    total = len(first_tokens)

    # Basin count: tokens with >5% of samples
    basin_tokens = [(tok, cnt) for tok, cnt in first_token_counts.most_common()
                    if cnt / total > 0.05]
    basin_count = len(basin_tokens)
    basin_concentration = basin_tokens[0][1] / total if basin_tokens else 0

    # Basin entropy
    first_token_probs = np.array([cnt / total for cnt in first_token_counts.values()])
    basin_entropy = compute_entropy(first_token_probs) if len(first_token_probs) > 0 else 0

    return {
        "continuations": continuations,
        "first_tokens": first_tokens,
        "first_token_ids": first_token_ids,
        "basin_count": basin_count,
        "basin_concentration": basin_concentration,
        "basin_entropy": basin_entropy,
        "first_token_dist": dict(first_token_counts.most_common(10)),
        "basin_tokens": basin_tokens,
    }


# ===== Experiment 3: Conditional Delayed Effect =====

def conditional_trajectory_analysis(model, tokenizer, device, cond_pairs, n_steps=12):
    """
    专门测试条件句的延迟约束效应

    对每个conditional pair, 计算:
    - conditional vs base的KL at each step
    - question vs base的KL at each step
    - 是否conditional的KL随步数递增?
    """
    results = []

    for pair in cond_pairs:
        base = pair["base"]
        conditional = pair["conditional"]
        question = pair["question"]

        print(f"  Cond pair: '{base[:30]}...' / '{conditional[:30]}...'")

        # Get multi-step profiles
        base_profile = multi_step_profile(model, tokenizer, device, base, n_steps=n_steps)
        cond_profile = multi_step_profile(model, tokenizer, device, conditional, n_steps=n_steps)
        ques_profile = multi_step_profile(model, tokenizer, device, question, n_steps=n_steps)

        # Compute KL/JS at each step
        step_kls = []
        for step in range(min(len(base_profile), len(cond_profile))):
            p_base = base_profile[step]["probs"]
            p_cond = cond_profile[step]["probs"]
            kl = compute_kl(p_cond, p_base)
            js = compute_js(p_cond, p_base)
            overlap = topk_overlap(base_profile[step]["top20_set"],
                                   cond_profile[step]["top20_set"])
            step_kls.append({
                "step": step,
                "kl_cond_base": kl,
                "js_cond_base": js,
                "overlap_cond_base": overlap,
            })

        # Also question vs base
        for step in range(min(len(base_profile), len(ques_profile))):
            if step < len(step_kls):
                p_ques = ques_profile[step]["probs"]
                kl_q = compute_kl(p_ques, base_profile[step]["probs"])
                js_q = compute_js(p_ques, base_profile[step]["probs"])
                overlap_q = topk_overlap(base_profile[step]["top20_set"],
                                         ques_profile[step]["top20_set"])
                step_kls[step]["kl_ques_base"] = kl_q
                step_kls[step]["js_ques_base"] = js_q
                step_kls[step]["overlap_ques_base"] = overlap_q

        results.append({
            "base": base,
            "conditional": conditional,
            "question": question,
            "step_kls": step_kls,
        })

    return results


# ===== Main Analysis =====

def analyze_trajectory_dynamics(model, tokenizer, device, sextets, cond_pairs,
                                 model_name, n_steps=12, n_samples=40):
    """完整的轨迹动力学分析"""

    n_sentences = len(sextets)
    print(f"\n{'='*70}")
    print(f"Phase 197: Trajectory-Level Constraint Dynamics — {model_name}")
    print(f"Sentences: {n_sentences} main + {len(cond_pairs)} conditional")
    print(f"Steps: {n_steps}, Samples: {n_samples}")
    print(f"{'='*70}\n")

    # === Exp1: Multi-step Distribution Evolution ===
    print("=" * 50)
    print("Exp1: Multi-step Distribution Evolution")
    print("=" * 50)

    all_profiles = {}  # condition_name -> list of step_data (per sentence)

    for cond_idx, cond_name in enumerate(CONDITION_NAMES):
        print(f"\n--- Condition: {cond_name} ---")
        cond_profiles = []

        for sent_idx, sextet in enumerate(sextets):
            sentence = sextet[cond_idx]
            if sent_idx % 5 == 0:
                print(f"  Sentence {sent_idx+1}/{n_sentences}: '{sentence[:40]}...'")

            profile = multi_step_profile(model, tokenizer, device, sentence, n_steps=n_steps)
            cond_profiles.append(profile)

        all_profiles[cond_name] = cond_profiles

    # Aggregate: mean over sentences at each step
    print("\n--- Exp1 Results: Mean distribution metrics ---")
    agg = {}
    for cond_name in CONDITION_NAMES:
        profiles = all_profiles[cond_name]
        n_s = len(profiles)
        agg[cond_name] = []
        for step in range(n_steps):
            entropies = [profiles[s][step]["entropy"] for s in range(n_s) if step < len(profiles[s])]
            ebfs = [profiles[s][step]["ebf"] for s in range(n_s) if step < len(profiles[s])]
            h_infs = [profiles[s][step]["h_inf"] for s in range(n_s) if step < len(profiles[s])]
            n_reach = [profiles[s][step]["n_reachable"] for s in range(n_s) if step < len(profiles[s])]
            m_top1 = [profiles[s][step]["mass_top1"] for s in range(n_s) if step < len(profiles[s])]

            agg[cond_name].append({
                "step": step,
                "entropy": float(np.mean(entropies)),
                "ebf": float(np.mean(ebfs)),
                "h_inf": float(np.mean(h_infs)),
                "n_reachable": float(np.mean(n_reach)),
                "mass_top1": float(np.mean(m_top1)),
            })

    # Print summary table
    print(f"\n{'Step':>4} | ", end="")
    for cn in CONDITION_NAMES:
        print(f"{'H('+cn[:4]+')':>8} {'EBF':>6}", end=" ")
    print()
    print("-" * 100)
    for step in range(n_steps):
        print(f"{step:>4} | ", end="")
        for cn in CONDITION_NAMES:
            d = agg[cn][step]
            print(f"{d['entropy']:>8.2f} {d['ebf']:>6.1f}", end=" ")
        print()

    # === Exp2: Trajectory Divergence Rate ===
    print("\n" + "=" * 50)
    print("Exp2: Trajectory Divergence Rate (base vs constraints)")
    print("=" * 50)

    divergence = {}
    for cond_name in ["negation", "question", "role_binding", "neg+q", "q+rb", "neg+rb"]:
        base_profiles = all_profiles["base"]
        cond_profiles = all_profiles[cond_name]
        n_s = min(len(base_profiles), len(cond_profiles))

        step_divs = []
        for step in range(n_steps):
            kls = []
            jss = []
            overlaps = []
            for s in range(n_s):
                if step >= len(base_profiles[s]) or step >= len(cond_profiles[s]):
                    continue
                p_base = base_profiles[s][step]["probs"]
                p_cond = cond_profiles[s][step]["probs"]
                kls.append(compute_kl(p_cond, p_base))
                jss.append(compute_js(p_cond, p_base))
                overlaps.append(topk_overlap(
                    base_profiles[s][step]["top20_set"],
                    cond_profiles[s][step]["top20_set"]
                ))

            step_divs.append({
                "step": step,
                "kl": float(np.mean(kls)) if kls else 0,
                "js": float(np.mean(jss)) if jss else 0,
                "overlap": float(np.mean(overlaps)) if overlaps else 0,
            })
        divergence[cond_name] = step_divs

    # Print divergence table
    print(f"\n{'Step':>4} | ", end="")
    for cn in ["negation", "question", "role_binding"]:
        print(f"{'KL('+cn[:4]+')':>8} {'JS':>6} {'Ovlp':>5}", end=" ")
    print()
    print("-" * 80)
    for step in range(n_steps):
        print(f"{step:>4} | ", end="")
        for cn in ["negation", "question", "role_binding"]:
            d = divergence[cn][step]
            print(f"{d['kl']:>8.3f} {d['js']:>6.3f} {d['overlap']:>5.2f}", end=" ")
        print()

    # Check divergence rate: is KL increasing or decreasing?
    print("\n--- KL Trend Analysis ---")
    for cn in ["negation", "question", "role_binding"]:
        kls = [d["kl"] for d in divergence[cn]]
        if len(kls) >= 2:
            # Linear regression slope
            x = np.arange(len(kls))
            slope = float(np.polyfit(x, kls, 1)[0])
            kl_first = kls[0]
            kl_last = kls[-1]
            trend = "INCREASING" if slope > 0.05 else "DECREASING" if slope < -0.05 else "STABLE"
            print(f"  {cn}: KL slope={slope:.4f} ({trend}), KL[0]={kl_first:.3f}, KL[{len(kls)-1}]={kl_last:.3f}")

    # === Exp3: Attractor Basin Structure ===
    print("\n" + "=" * 50)
    print("Exp3: Attractor Basin Structure")
    print("=" * 50)

    basin_results = {}
    for cond_idx, cond_name in enumerate(CONDITION_NAMES):
        print(f"\n--- Condition: {cond_name} ---")
        all_basins = []

        # Sample from subset of sentences for efficiency
        sample_indices = list(range(0, min(n_sentences, 15)))
        for sent_idx in sample_indices:
            sentence = sextets[sent_idx][cond_idx]
            if sent_idx % 5 == 0:
                print(f"  Sampling from sentence {sent_idx+1}/{len(sample_indices)}: '{sentence[:30]}...'")

            basins = sample_continuations(model, tokenizer, device, sentence,
                                          n_samples=n_samples, n_tokens=8, temperature=1.0)
            all_basins.append(basins)

        # Aggregate basin metrics
        mean_basin_count = float(np.mean([b["basin_count"] for b in all_basins]))
        mean_basin_conc = float(np.mean([b["basin_concentration"] for b in all_basins]))
        mean_basin_ent = float(np.mean([b["basin_entropy"] for b in all_basins]))

        # Aggregate first-token distribution
        all_first_tokens = Counter()
        for b in all_basins:
            all_first_tokens.update(b["first_token_dist"])

        basin_results[cond_name] = {
            "mean_basin_count": mean_basin_count,
            "mean_basin_concentration": mean_basin_conc,
            "mean_basin_entropy": mean_basin_ent,
            "top_first_tokens": dict(all_first_tokens.most_common(15)),
            "per_sentence": all_basins,
        }

        print(f"  Basin count: {mean_basin_count:.1f}")
        print(f"  Basin concentration (top-1): {mean_basin_conc:.3f}")
        print(f"  Basin entropy: {mean_basin_ent:.2f}")
        print(f"  Top first tokens: {dict(all_first_tokens.most_common(8))}")

    # === Exp4: Conditional Delayed Effect ===
    print("\n" + "=" * 50)
    print("Exp4: Conditional Delayed Constraint Effect")
    print("=" * 50)

    cond_results = conditional_trajectory_analysis(model, tokenizer, device, cond_pairs, n_steps=n_steps)

    # Aggregate over conditional pairs
    print("\n--- Mean KL/JS Profile: Conditional vs Base ---")
    for step in range(n_steps):
        kls_c = [r["step_kls"][step]["kl_cond_base"] for r in cond_results if step < len(r["step_kls"])]
        kls_q = [r["step_kls"][step].get("kl_ques_base", 0) for r in cond_results if step < len(r["step_kls"])]
        jss_c = [r["step_kls"][step]["js_cond_base"] for r in cond_results if step < len(r["step_kls"])]
        overlaps_c = [r["step_kls"][step]["overlap_cond_base"] for r in cond_results if step < len(r["step_kls"])]

        kl_c_mean = float(np.mean(kls_c)) if kls_c else 0
        kl_q_mean = float(np.mean(kls_q)) if kls_q else 0
        js_c_mean = float(np.mean(jss_c)) if jss_c else 0
        ovlp_mean = float(np.mean(overlaps_c)) if overlaps_c else 0

        print(f"  Step {step:>2}: KL(cond)={kl_c_mean:.3f}, KL(ques)={kl_q_mean:.3f}, "
              f"JS(cond)={js_c_mean:.4f}, overlap={ovlp_mean:.3f}")

    # Test: Does conditional KL increase over steps?
    cond_kls = []
    ques_kls = []
    for step in range(n_steps):
        kls_c = [r["step_kls"][step]["kl_cond_base"] for r in cond_results if step < len(r["step_kls"])]
        kls_q = [r["step_kls"][step].get("kl_ques_base", 0) for r in cond_results if step < len(r["step_kls"])]
        cond_kls.append(float(np.mean(kls_c)) if kls_c else 0)
        ques_kls.append(float(np.mean(kls_q)) if kls_q else 0)

    x = np.arange(len(cond_kls))
    if len(cond_kls) >= 2:
        slope_cond = float(np.polyfit(x, cond_kls, 1)[0])
        slope_ques = float(np.polyfit(x, ques_kls, 1)[0])
        print(f"\n  KL slope: conditional={slope_cond:.4f}, question={slope_ques:.4f}")
        if slope_cond > 0.05:
            print(f"  >>> CONDITIONAL shows INCREASING KL (delayed constraint effect!)")
        elif slope_cond < -0.05:
            print(f"  >>> CONDITIONAL shows DECREASING KL (constraint fades)")
        else:
            print(f"  >>> CONDITIONAL shows STABLE KL (no clear delayed effect)")

    # === Exp5: Reachable Continuation Space ===
    print("\n" + "=" * 50)
    print("Exp5: Reachable Continuation Space")
    print("=" * 50)

    print(f"\n{'Step':>4} | ", end="")
    for cn in CONDITION_NAMES:
        print(f"{'Reach('+cn[:4]+')':>8} {'mTop1':>6}", end=" ")
    print()
    print("-" * 120)
    for step in range(n_steps):
        print(f"{step:>4} | ", end="")
        for cn in CONDITION_NAMES:
            d = agg[cn][step]
            print(f"{d['n_reachable']:>8.1f} {d['mass_top1']:>6.3f}", end=" ")
        print()

    # === Final Summary ===
    print("\n" + "=" * 70)
    print("PHASE 197 SUMMARY")
    print("=" * 70)

    # 1. Divergence rate at step 0 (immediate) vs step 11 (trajectory)
    print("\n1. Trajectory Divergence: Step 0 vs Step 11")
    for cn in ["negation", "question", "role_binding"]:
        kl_0 = divergence[cn][0]["kl"]
        kl_last = divergence[cn][-1]["kl"]
        ovlp_0 = divergence[cn][0]["overlap"]
        ovlp_last = divergence[cn][-1]["overlap"]
        print(f"  {cn}: KL {kl_0:.3f}→{kl_last:.3f} (Δ={kl_last-kl_0:+.3f}), "
              f"overlap {ovlp_0:.3f}→{ovlp_last:.3f} (Δ={ovlp_last-ovlp_0:+.3f})")

    # 2. Attractor basin comparison
    print("\n2. Attractor Basin Structure")
    for cn in CONDITION_NAMES:
        r = basin_results[cn]
        print(f"  {cn}: count={r['mean_basin_count']:.1f}, "
              f"concentration={r['mean_basin_concentration']:.3f}, "
              f"entropy={r['mean_basin_entropy']:.2f}")

    # 3. Question creates more basins?
    base_basins = basin_results["base"]["mean_basin_count"]
    ques_basins = basin_results["question"]["mean_basin_count"]
    print(f"\n  Question vs Base basins: {ques_basins:.1f} vs {base_basins:.1f}")
    if ques_basins > base_basins:
        print(f"  >>> QUESTION opens more attractor basins (+{ques_basins-base_basins:.1f})")
    else:
        print(f"  >>> QUESTION does NOT open more basins (surprising!)")

    # 4. Conditional delayed effect
    if len(cond_kls) >= 2:
        slope = float(np.polyfit(np.arange(len(cond_kls)), cond_kls, 1)[0])
        print(f"\n4. Conditional KL slope: {slope:.4f}")
        if slope > 0.05:
            print(f"  >>> DELAYED CONSTRAINT confirmed for conditional")
        else:
            print(f"  >>> No clear delayed effect for conditional")

    # 5. Entropy trajectory: which constraints increase/maintain/decrease entropy?
    print("\n5. Entropy Trajectory (step 0 → step 11)")
    for cn in CONDITION_NAMES:
        h0 = agg[cn][0]["entropy"]
        h_last = agg[cn][-1]["entropy"]
        delta = h_last - h0
        trend = "↑" if delta > 0.5 else "↓" if delta < -0.5 else "→"
        print(f"  {cn}: {h0:.2f}→{h_last:.2f} (Δ={delta:+.2f}) {trend}")

    # Save results
    save_data = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "n_sentences": n_sentences,
        "n_steps": n_steps,
        "n_samples": n_samples,
        "agg_profiles": {cn: agg[cn] for cn in CONDITION_NAMES},
        "divergence": {cn: divergence.get(cn, []) for cn in ["negation", "question", "role_binding", "neg+q", "q+rb", "neg+rb"]},
        "basin_results": {cn: {
            "mean_basin_count": basin_results[cn]["mean_basin_count"],
            "mean_basin_concentration": basin_results[cn]["mean_basin_concentration"],
            "mean_basin_entropy": basin_results[cn]["mean_basin_entropy"],
            "top_first_tokens": basin_results[cn]["top_first_tokens"],
        } for cn in CONDITION_NAMES},
        "conditional_results": [{
            "base": r["base"],
            "conditional": r["conditional"],
            "step_kls": r["step_kls"],
        } for r in cond_results],
    }

    out_path = Path(f"tests/glm5_temp/phase197_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved to {out_path}")

    return save_data


# ===== Main =====

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
        return

    t_start = time.time()
    print(f"[Phase 197] Starting trajectory dynamics analysis for {model_name}")
    print(f"[Phase 197] Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Adjust data sizes per model
    if model_name == "qwen3":
        n_main = 25
        n_samples = 40
    else:
        n_main = 15
        n_samples = 30

    n_steps = 12
    sextets = SENTENCE_SEXTETS[:n_main]
    cond_pairs = CONDITIONAL_PAIRS[:min(5, len(CONDITIONAL_PAIRS))]

    # Load model
    model, tokenizer, device = load_model_bf16(model_name)

    # Run analysis
    results = analyze_trajectory_dynamics(
        model, tokenizer, device,
        sextets, cond_pairs,
        model_name, n_steps=n_steps, n_samples=n_samples
    )

    # Release model
    release_model(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    t_total = time.time() - t_start
    print(f"\n[Phase 197] COMPLETE in {t_total:.1f}s ({model_name})")
    print(f"[Phase 197] Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
