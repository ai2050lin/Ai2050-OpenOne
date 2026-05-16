"""
Phase 198b: Lite Mode Transition (GLM4 & DS7B)
==============================================
极简版: 只验证3个核心问题

1. semantic vs random KL slope是否不同?
   - 5句 × 3条件(neg/question/rand_token) × 6步
   
2. Conditional延迟效应是否确认?
   - 3个条件句对 × 6步
   
3. 模式切换的attractor basin
   - 2种模式(cot/translation) × 10采样
"""
import sys, os
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

import gc, time, math
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tests"))
from model_demo_bf16 import load_model_bf16
from model_utils import get_model_info, release_model

# 精简5句
SENTENCES = [
    "The cat chases the dog",
    "The teacher helps the student",
    "The doctor treats the patient",
    "The writer drafts the letter",
    "The scientist discovers the element",
]

COND_PAIRS = [
    ("The cat chases the dog", "If the cat chases the dog"),
    ("The ground is dry", "If it rains, the ground"),
    ("The student studies hard", "If the student studies hard"),
]

MODE_PAIRS = {
    "cot": ("Problem: 2+3=? ", "Problem: 2+3=? Let's think step by step"),
    "translation": ("The cat sleeps. ", "Translate to Chinese: The cat sleeps. "),
}

N_STEPS = 6
N_SAMPLES = 10

def compute_kl(p, q):
    p = p.float().clamp(min=1e-10)
    q = q.float().clamp(min=1e-10)
    kl = (p * (p.log() - q.log())).sum().item()
    return max(kl, 0) if not (math.isnan(kl) or math.isinf(kl)) else 0.0

def compute_entropy(probs):
    p = probs.float().clamp(min=1e-10)
    h = -(p * p.log()).sum().item()
    return h if not math.isnan(h) else 0.0

def run_autoregressive(model, tokenizer, device, text, n_steps):
    """自回归生成, 返回每步probs"""
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    step_data = []
    for step in range(n_steps):
        with torch.no_grad():
            out = model(ids)
            logits = out.logits[:, -1, :].to(device).float()
            probs = F.softmax(logits, dim=-1)
        h = compute_entropy(probs[0])
        step_data.append({'entropy': h, 'probs': probs[0]})
        next_tok = probs.argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_tok], dim=-1)
    return step_data

def sample_first_tokens(model, tokenizer, device, text, n_samples, temperature=0.8):
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    tokens = []
    for _ in range(n_samples):
        with torch.no_grad():
            out = model(ids)
            logits = out.logits[:, -1, :].to(device).float()
            next_id = torch.multinomial(F.softmax(logits/temperature, dim=-1), 1)
            tokens.append(tokenizer.decode(next_id[0]).strip())
    return tokens

def make_negation(base):
    words = base.split()
    if len(words) >= 2:
        return words[0] + " " + words[1] + " does not " + " ".join(words[2:])
    return base

def make_question(base):
    return "Does " + base[0].lower() + base[1:] + "?"

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "glm4"
    t0 = time.time()
    print(f"[Phase 198b] Lite mode transition — {model_name}")
    print(f"[Phase 198b] Time: {datetime.now()}")

    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    print(f"[load] {model_name}: {info.model_class}, {info.n_layers}L, d={info.d_model}")
    print(f"Config: {len(SENTENCES)} sent, {N_STEPS} steps, {N_SAMPLES} samples")

    # ===== Exp1: Semantic vs Random KL =====
    print("\n====== Exp1: Semantic vs Random KL =====")
    conditions = {
        "negation": lambda b: make_negation(b),
        "question": lambda b: make_question(b),
        "rand_token": lambda b: b + " xyz",
    }
    kl_data = {c: [] for c in conditions}

    for si, base in enumerate(SENTENCES):
        base_steps = run_autoregressive(model, tokenizer, device, base, N_STEPS)
        for cond, fn in conditions.items():
            text = fn(base)
            pert_steps = run_autoregressive(model, tokenizer, device, text, N_STEPS)
            kl_list = [compute_kl(pert_steps[s]['probs'], base_steps[s]['probs']) for s in range(N_STEPS)]
            kl_data[cond].append(kl_list)
        print(f"  S{si+1}/{len(SENTENCES)} done, GPU={torch.cuda.memory_allocated()/1e9:.2f}GB, t={time.time()-t0:.0f}s")

    print("\n--- KL Profile ---")
    print(f"{'Step':>4} |", end="")
    for c in conditions:
        print(f" {c[:8]:>8}", end="")
    print()
    mean_kls = {}
    for step in range(N_STEPS):
        print(f"{step:4d} |", end="")
        for c in conditions:
            mk = np.mean([kl[step] for kl in kl_data[c]])
            mean_kls.setdefault(c, []).append(mk)
            print(f" {mk:8.3f}", end="")
        print()

    print("\n--- KL Slope ---")
    slopes = {}
    for c in conditions:
        arr = np.array(mean_kls[c])
        slope = np.polyfit(range(N_STEPS), arr, 1)[0]
        slopes[c] = slope
        trend = "INC" if slope > 0.2 else ("DEC" if slope < -0.2 else "STABLE")
        print(f"  {c:<12} slope={slope:.4f} ({trend}), KL[0]={arr[0]:.3f}, KL[-1]={arr[-1]:.3f}")

    sem_slope = np.mean([slopes["negation"], slopes["question"]])
    rand_slope = slopes["rand_token"]
    ratio = sem_slope / rand_slope if rand_slope > 0 else float('inf')
    print(f"\n  Semantic mean slope: {sem_slope:.4f}")
    print(f"  Random slope:       {rand_slope:.4f}")
    print(f"  Ratio: {ratio:.2f}x")
    if ratio > 1.5:
        print("  >>> Semantic propagation REAL")
    else:
        print("  >>> WARNING: KL increase may be autoregressive tautology")

    # ===== Exp2: Conditional Delay =====
    print("\n====== Exp2: Conditional Delay =====")
    for base_t, cond_t in COND_PAIRS:
        bs = run_autoregressive(model, tokenizer, device, base_t, N_STEPS)
        cs = run_autoregressive(model, tokenizer, device, cond_t, N_STEPS)
        kl_list = [compute_kl(cs[s]['probs'], bs[s]['probs']) for s in range(N_STEPS)]
        slope = np.polyfit(range(N_STEPS), kl_list, 1)[0] if len(kl_list) > 1 else 0
        print(f"  '{base_t[:30]}' vs '{cond_t[:30]}'")
        print(f"    KL: {kl_list[0]:.3f} -> {kl_list[-1]:.3f}, slope={slope:.4f}")

    # ===== Exp3: Mode Basin =====
    print("\n====== Exp3: Mode Attractor Basin =====")
    for mode, (base_t, trigger_t) in MODE_PAIRS.items():
        base_tokens = sample_first_tokens(model, tokenizer, device, base_t, N_SAMPLES)
        trigger_tokens = sample_first_tokens(model, tokenizer, device, trigger_t, N_SAMPLES)
        bc = Counter(base_tokens)
        tc = Counter(trigger_tokens)
        print(f"  {mode}:")
        print(f"    Base:    {bc.most_common(5)}")
        print(f"    Trigger: {tc.most_common(5)}")

    # ===== Exp4: Delay Spectrum =====
    print("\n====== Exp4: Constraint Delay Spectrum =====")
    delay_conds = {
        "negation": lambda b: make_negation(b),
        "question": lambda b: make_question(b),
        "conditional": lambda b: f"If {b[0].lower() + b[1:]}",
        "rand_token": lambda b: b + " xyz",
    }
    delay_results = {}
    for cond, fn in delay_conds.items():
        kl_lists = []
        for base in SENTENCES[:3]:
            bs = run_autoregressive(model, tokenizer, device, base, N_STEPS)
            text = fn(base)
            ps = run_autoregressive(model, tokenizer, device, text, N_STEPS)
            kl_lists.append([compute_kl(ps[s]['probs'], bs[s]['probs']) for s in range(N_STEPS)])
        mean_kl = np.mean(kl_lists, axis=0)
        slope = np.polyfit(range(N_STEPS), mean_kl, 1)[0]
        kl0 = mean_kl[0]
        delay_type = "IMMEDIATE" if kl0 > 3.0 else ("DELAYED" if kl0 < 1.0 and slope > 0.3 else "MODERATE")
        delay_results[cond] = {'kl0': kl0, 'slope': slope, 'type': delay_type}
        print(f"  {cond:<12} KL[0]={kl0:.3f}, slope={slope:.4f} ({delay_type})")

    # Summary
    elapsed = time.time() - t0
    print(f"\n[Phase 198b] COMPLETE in {elapsed:.1f}s ({model_name})")
    print(f"\n=== SUMMARY ===")
    print(f"1. Sem/Rand KL slope ratio: {ratio:.2f}x")
    print(f"2. Delay spectrum:")
    for c, d in delay_results.items():
        print(f"   {c}: {d['type']} (KL[0]={d['kl0']:.2f}, slope={d['slope']:.4f})")

    release_model(model)

if __name__ == "__main__":
    main()
