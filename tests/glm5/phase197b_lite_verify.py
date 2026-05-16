"""
Phase 197b: Lite Verification (GLM4 & DS7B)
============================================

精简版: 以Qwen3结果为主, GLM4/DS7B为辅验证核心结论

Qwen3核心发现(待验证):
1. 所有约束的KL随步数递增 (INCREASING)
2. Question打开更多attractor basins (5.1 vs 4.3)
3. Negation减少basins并提高concentration (2.9, 0.490)
4. Conditional KL slope=0.76 → 延迟约束效应
5. 采样续写首token: question→What/Also/Or, base→逗号/in/of

精简参数: 8句×4条件×8步 + 15采样 + 3条件句对
"""
import sys, os
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

import gc, time, json, math
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter, defaultdict
from datetime import datetime
from model_utils import load_model, get_model_info, release_model

# ===== 精简8句 =====
SENTENCES = [
    ("The cat chases the dog",
     "The cat does not chase the dog",
     "Does the cat chase the dog?",
     "The dog chases the cat"),
    ("The teacher helps the student",
     "The teacher does not help the student",
     "Does the teacher help the student?",
     "The student helps the teacher"),
    ("The doctor treats the patient",
     "The doctor does not treat the patient",
     "Does the doctor treat the patient?",
     "The patient treats the doctor"),
    ("The writer drafts the letter",
     "The writer does not draft the letter",
     "Does the writer draft the letter?",
     "The letter drafts the writer"),
    ("The scientist discovers the element",
     "The scientist does not discover the element",
     "Does the scientist discover the element?",
     "The element discovers the scientist"),
    ("The chef cooks the meal",
     "The chef does not cook the meal",
     "Does the chef cook the meal?",
     "The meal cooks the chef"),
    ("The farmer plants the seed",
     "The farmer does not plant the seed",
     "Does the farmer plant the seed?",
     "The seed plants the farmer"),
    ("The artist paints the portrait",
     "The artist does not paint the portrait",
     "Does the artist paint the portrait?",
     "The portrait paints the artist"),
]

# 3个条件句对
COND_PAIRS = [
    ("The cat chases the dog", "If the cat chases the dog"),
    ("The ground is dry", "If it rains, the ground"),
    ("The student studies hard", "If the student studies hard"),
]

CONDITIONS = ["base", "negation", "question", "role_binding"]
N_STEPS = 8
N_SAMPLES = 15

def get_cond_text(sextet, cond):
    base, neg, ques, role = sextet[0], sextet[1], sextet[2], sextet[3]
    return {"base": base, "negation": neg, "question": ques, "role_binding": role}[cond]

def compute_kl_js(p, q):
    # Ensure same device and add NaN protection
    p = p.float().clamp(min=1e-10)
    q = q.float().clamp(min=1e-10)
    kl = (p * (p.log() - q.log())).sum().item()
    js = 0.5 * (F.kl_div(q.log(), p, reduction='sum') + F.kl_div(p.log(), q, reduction='sum')).item()
    overlap = (p * q).sum().sqrt().item()
    if math.isnan(kl) or math.isinf(kl):
        kl = 0.0
    if math.isnan(js) or math.isinf(js):
        js = 0.0
    if math.isnan(overlap) or math.isinf(overlap):
        overlap = 0.0
    return max(kl, 0), max(js, 0), overlap

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "glm4"
    t0 = time.time()
    print(f"[Phase 197b] Lite verify for {model_name}")
    print(f"[Phase 197b] Time: {datetime.now()}")
    print(f"[Phase 197b] Sentences={len(SENTENCES)}, Steps={N_STEPS}, Samples={N_SAMPLES}")

    # Load model
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    print(f"[load] {model_name} loaded, GPU={torch.cuda.memory_allocated()/1e9:.2f}GB")

    # ===== Exp1: Multi-step Distribution Evolution =====
    print("\n====== Exp1: Multi-step Distribution Evolution ======")
    entropy_data = {c: np.zeros((len(SENTENCES), N_STEPS)) for c in CONDITIONS}

    for ci, cond in enumerate(CONDITIONS):
        print(f"\n--- {cond} ---")
        for si, sext in enumerate(SENTENCES):
            text = get_cond_text(sext, cond)
            ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
            current = ids
            for step in range(N_STEPS):
                with torch.no_grad():
                    out = model(current)
                    logits = out.logits[:, -1, :].to(device)
                    probs = F.softmax(logits.float(), dim=-1)
                entropy_data[cond][si, step] = -(probs * probs.log()).sum().item()
                # Sample top-1 token
                next_tok = probs.argmax(dim=-1, keepdim=True)
                current = torch.cat([current, next_tok], dim=-1)
                if step % 4 == 0:
                    print(f"  S{si} step{step} H={entropy_data[cond][si,step]:.2f}")

    # Print mean entropy per step
    print("\n--- Mean Entropy ---")
    print("Step |", "  ".join(f"{c[:4]:>6}" for c in CONDITIONS))
    for step in range(N_STEPS):
        vals = "  ".join(f"{entropy_data[c][:,step].mean():6.2f}" for c in CONDITIONS)
        print(f"  {step:2d} | {vals}")

    # ===== Exp2: Trajectory Divergence (base vs constraint) =====
    print("\n====== Exp2: Trajectory Divergence ======")
    kl_data = {c: [] for c in ["negation", "question", "role_binding"]}

    for si, sext in enumerate(SENTENCES):
        base_ids = tokenizer(sext[0], return_tensors="pt").input_ids.to(device)
        for cidx, cond in enumerate(["negation", "question", "role_binding"]):
            text = get_cond_text(sext, cond)
            alt_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
            base_cur = base_ids.clone()
            alt_cur = alt_ids.clone()
            kl_steps = []
            for step in range(N_STEPS):
                with torch.no_grad():
                    base_out = model(base_cur)
                    alt_out = model(alt_cur)
                    bp = F.softmax(base_out.logits[:, -1, :].to(device).float(), dim=-1)
                    ap = F.softmax(alt_out.logits[:, -1, :].to(device).float(), dim=-1)
                kl, js, ovlp = compute_kl_js(ap, bp)
                kl_steps.append((kl, js, ovlp))
                bn = bp.argmax(dim=-1, keepdim=True)
                an = ap.argmax(dim=-1, keepdim=True)
                base_cur = torch.cat([base_cur, bn], dim=-1)
                alt_cur = torch.cat([alt_cur, an], dim=-1)
            kl_data[cond].append(kl_steps)
        if si % 2 == 0:
            print(f"  Sentence {si+1}/{len(SENTENCES)} done")

    print("\n--- Mean KL Profile ---")
    print("Step |", "  ".join(f"KL({c[:4]})" for c in ["negation","question","role_binding"]))
    for step in range(N_STEPS):
        vals = []
        for c in ["negation","question","role_binding"]:
            mean_kl = np.mean([s[step][0] for s in kl_data[c]])
            vals.append(f"{mean_kl:6.2f}")
        print(f"  {step:2d} | " + "  ".join(vals))

    # KL slope
    print("\n--- KL Trend ---")
    for c in ["negation","question","role_binding"]:
        mean_kls = [np.mean([s[step][0] for s in kl_data[c]]) for step in range(N_STEPS)]
        slope = np.polyfit(range(N_STEPS), mean_kls, 1)[0]
        trend = "INCREASING" if slope > 0.2 else ("DECREASING" if slope < -0.2 else "STABLE")
        print(f"  {c}: slope={slope:.4f} ({trend}), KL[0]={mean_kls[0]:.3f}, KL[{N_STEPS-1}]={mean_kls[-1]:.3f}")

    # ===== Exp3: Attractor Basin Structure =====
    print("\n====== Exp3: Attractor Basin Structure ======")
    for cond in CONDITIONS:
        first_tokens = []
        for si, sext in enumerate(SENTENCES):
            text = get_cond_text(sext, cond)
            ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
            for _ in range(N_SAMPLES):
                with torch.no_grad():
                    out = model(ids)
                    logits = out.logits[:, -1, :].to(device).float()
                    next_id = torch.multinomial(F.softmax(logits/0.8, dim=-1), 1)
                    tok = tokenizer.decode(next_id[0]).strip()
                    first_tokens.append(tok)
        counts = Counter(first_tokens)
        n_unique = len(counts)
        top1_frac = counts.most_common(1)[0][1] / len(first_tokens) if counts else 0
        total = len(first_tokens)
        probs_arr = np.array([c/total for c in counts.values()])
        ent = -(probs_arr * np.log(probs_arr + 1e-12)).sum()
        top5 = counts.most_common(5)
        print(f"  {cond}: basins={n_unique}, top1_frac={top1_frac:.3f}, entropy={ent:.2f}")
        print(f"    Top5: {top5}")

    # ===== Exp4: Conditional Delayed Effect =====
    print("\n====== Exp4: Conditional Delayed Effect ======")
    for base_t, cond_t in COND_PAIRS:
        print(f"  Pair: '{base_t[:30]}...' / '{cond_t[:30]}...'")
        base_ids = tokenizer(base_t, return_tensors="pt").input_ids.to(device)
        cond_ids = tokenizer(cond_t, return_tensors="pt").input_ids.to(device)
        base_cur = base_ids.clone()
        cond_cur = cond_ids.clone()
        kl_list = []
        for step in range(N_STEPS):
            with torch.no_grad():
                bp = F.softmax(model(base_cur).logits[:, -1, :].to(device).float(), dim=-1)
                cp = F.softmax(model(cond_cur).logits[:, -1, :].to(device).float(), dim=-1)
            kl = (cp * (cp.log() - bp.log())).sum().item()
            kl = max(kl, 0)
            kl_list.append(kl)
            bn = bp.argmax(dim=-1, keepdim=True)
            cn = cp.argmax(dim=-1, keepdim=True)
            base_cur = torch.cat([base_cur, bn], dim=-1)
            cond_cur = torch.cat([cond_cur, cn], dim=-1)
        slope = np.polyfit(range(N_STEPS), kl_list, 1)[0]
        print(f"    KL: {kl_list[0]:.2f} → {kl_list[-1]:.2f}, slope={slope:.4f}")

    # Summary
    elapsed = time.time() - t0
    print(f"\n[Phase 197b] COMPLETE in {elapsed:.1f}s ({model_name})")

    # Release
    del model; gc.collect(); torch.cuda.empty_cache()
    print("[model_utils] GPU released")

if __name__ == "__main__":
    main()
