"""
Phase 165: Constraint Transport Theory — 约束输运理论
=====================================================

Critical pivot from Phase 164:

Phase 164 discovered:
  - Concepts induce stable deformations on P(future)
  - Deformations are approximately additive (cos≈0.86)
  - Deformations are non-commutative (KL>0)
  - 1-step equivalence breaks down at 5-step (40%)

BUT user's critical correction:
  - We measured T_{c,context}, not T_c (context-independent operator)
  - KL is not the right distance (should use Wasserstein/JS)
  - We need Future Path Equivalence, not just P(next_1)
  - We need Constraint Jacobian: ∂P(Future)/∂c
  - We need Operator Closure: T_A ∘ T_B in same deformation family?

Core theory upgrade:
  Language = "self-conditioned constraint propagation system"
  - token = constraint injector
  - concept = stable reusable constraint deformation
  - hidden state = constraint cache
  - logits = future path boundary conditions
  - generation = constraint propagation

Four experiments:

  Exp 1: ★★★ Future Path Equivalence (最关键!)
    - Find context pairs with similar P(next_1)
    - Fix COMMON continuation sequence (eliminate autoregressive drift!)
    - Compare P(next|c1+prefix) vs P(next|c2+prefix) at each step
    - KEY: Path equivalence breakdown with shared prefix

  Exp 2: ★★★ Constraint Jacobian (概念敏感度)
    - For each concept, measure deformation STABILITY across diverse contexts
    - If deformation is context-independent → true operator (Jacobian rank 1)
    - If deformation depends heavily on context → not a true operator
    - KEY: Which concepts have low-rank deformation matrices?

  Exp 3: ★★★ Concept Fixed Point (概念不动点)
    - Find concepts whose deformation pattern is STABLE across very diverse contexts
    - Use contexts from different domains (poetry, legal, dialogue, technical)
    - KEY: These are the "true operators" — context-independent constraints

  Exp 4: ★★★ Operator Closure (算子闭包)
    - Test if T_A ∘ T_B produces deformations in the same family as T_A and T_B
    - Test associativity: T_A ∘ (T_B ∘ T_C) vs (T_A ∘ T_B) ∘ T_C
    - Test identity operator and inverse
    - KEY: Is the deformation family algebraically closed?

Usage: python tests/glm5/phase165_constraint_transport.py <model_name>
  model_name: qwen3, glm4, deepseek7b
"""

import sys
import os
import time
import json
import gc
import numpy as np
from datetime import datetime
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'glm5'))

from model_utils import load_model, get_model_info, release_model, MODEL_CONFIGS


# ===== CONCEPT SETS (expanded from Phase 164) =====

CONCEPT_FAMILIES = {
    "animal": ["cat", "dog", "tiger", "lion", "horse", "elephant", "eagle", "whale"],
    "vehicle": ["car", "truck", "bus", "train", "bicycle", "airplane", "boat", "motorcycle"],
    "color": ["red", "blue", "green", "white", "black", "yellow", "purple", "orange"],
    "abstract": ["democracy", "freedom", "justice", "power", "truth", "beauty", "wisdom", "courage"],
    "emotion": ["love", "anger", "fear", "joy", "sadness", "hope", "pride", "shame"],
}

# All concepts
ALL_CONCEPTS = []
CONCEPT_TO_FAMILY = {}
for family, concepts in CONCEPT_FAMILIES.items():
    for c in concepts:
        ALL_CONCEPTS.append(c)
        CONCEPT_TO_FAMILY[c] = family


# ===== CONTEXT TEMPLATES (expanded, more diverse) =====

CONTEXT_TEMPLATES = [
    # Simple narrative
    "I saw a ___ in the park yesterday",
    "The ___ was standing near the old building",
    "She described the ___ as something quite remarkable because",
    # Scientific
    "The researcher studied the ___ and discovered that",
    "According to the experiment, the ___ was found to be",
    "The laboratory analysis of the ___ revealed that",
    # Philosophical
    "The philosopher contemplated the ___ and concluded that",
    "In the context of ethics, the ___ represents",
    "The nature of ___ remains one of the deepest questions because",
    # Dialogue
    '"I think the ___ is very important," she said, because',
    '"Have you considered the ___?" he asked, explaining that',
    '"The ___ changed everything," the professor stated, pointing out that',
    # Descriptive
    "The ancient ___ stood in the center of the village, its",
    "When the ___ appeared on the horizon, everyone could see that",
    "The artist painted the ___ in a way that captured",
    # Factual
    "Historically, the ___ has been associated with",
    "The report showed that the ___ was responsible for",
    "In many cultures, the ___ symbolizes",
    # Abstract reasoning
    "If we consider the ___ carefully, we realize that",
    "The existence of ___ raises the fundamental question of whether",
    "Understanding ___ requires us to think deeply about",
    # Contrastive
    "Unlike other things, the ___ has the unique property that",
    "While most ___ are common, this one was special because",
    "The difference between this ___ and others is that",
    # Conditional
    "If the ___ were to disappear, the consequences would be",
    "Without the ___, the system would collapse because",
    "Given the ___, we can predict that",
    # Temporal
    "Before the ___ existed, people believed that",
    "After the ___ was introduced, everything changed because",
    "During the ___, the most remarkable thing was",
]


# ===== DIVERSE PROMPTS (for Exp 1 — very different domains) =====

DIVERSE_PROMPTS = {
    "narrative": [
        "The old man walked slowly through the quiet village, his steps",
        "She opened the mysterious letter and discovered that",
        "The detective examined the crime scene carefully and noticed",
        "In the distant kingdom, a young princess decided to",
        "The storm raged outside while inside the cabin",
        "After years of wandering, the traveler finally found",
        "The cat jumped onto the windowsill and watched as",
        "The children played in the garden until suddenly",
    ],
    "technical": [
        "The algorithm processes the input data by first computing",
        "In quantum mechanics, the wave function collapses when",
        "The neural network learns to minimize the loss function by",
        "The chemical reaction produces a catalyst that",
        "The software architecture uses a microservices pattern where",
        "The mathematical proof relies on the assumption that",
        "The database query optimization reduces latency by",
        "The electrical circuit consists of resistors connected in",
    ],
    "dialogue": [
        '"I think we should reconsider the plan," she said, because',
        '"Have you ever wondered why," he asked, looking at',
        '"That is not what I meant," the professor replied, explaining that',
        '"We need to leave now," the captain shouted, warning that',
        '"I disagree completely," said the scientist, pointing out that',
        '"The answer is obvious," she smiled, and then added that',
        '"Please listen carefully," the teacher said, because',
        '"What if we are wrong?" she whispered, afraid that',
    ],
    "descriptive": [
        "The sunset painted the sky in shades of amber and crimson as",
        "The old cathedral stood majestically against the skyline, its",
        "The garden was filled with roses, lilies, and dahlias that",
        "The mountain lake reflected the surrounding peaks like a mirror, and",
        "The bustling market street was lined with vendors selling",
        "The quiet library smelled of old books and polished wood, where",
        "The winter landscape was covered in a blanket of snow that",
        "The ancient ruins contained crumbling columns and arches that",
    ],
    "philosophical": [
        "The nature of consciousness remains one of the deepest mysteries because",
        "Free will and determinism appear to conflict when we consider that",
        "The meaning of truth depends on whether we define it as",
        "Ethical relativism suggests that moral judgments are",
        "The mind-body problem arises from the observation that",
        "Knowledge requires both justification and belief because",
        "The problem of induction questions whether we can",
        "Identity over time seems paradoxical when we consider that",
    ],
    "poetry": [
        "Like silver threads upon the loom of time, the",
        "Beneath the veil of starlight, ancient and eternal, the",
        "Whispers of forgotten dreams echo through the",
        "The moonlit path winds through shadows where",
        "In gardens where the silence blooms like flowers,",
        "The sea remembers all the songs that",
        "Between the lines of morning light, there lies",
        "Beyond the edge of words, where meaning dissolves into",
    ],
    "legal": [
        "Pursuant to Section 4 of the Agreement, the defendant must",
        "The court finds that the plaintiff has standing because",
        "Under the applicable statute of limitations, the claim",
        "The contractual obligation requires that the party shall",
        "The regulatory framework provides that any violation of",
        "The fiduciary duty imposes upon the trustee the obligation to",
        "In accordance with established precedent, the ruling holds that",
        "The burden of proof requires the prosecution to demonstrate that",
    ],
}


# ===== COMPOSITION TRIPLES (for Exp 4 — associativity test) =====

COMPOSITION_TRIPLES = [
    # (A, B, C) — test T_A ∘ (T_B ∘ T_C) vs (T_A ∘ T_B) ∘ T_C
    ("red", "big", "car"),
    ("fast", "dangerous", "train"),
    ("beautiful", "ancient", "city"),
    ("small", "white", "cat"),
    ("powerful", "new", "engine"),
]


# ===== UTILITY FUNCTIONS =====

def get_device_for_input(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_softmax(logits_np):
    logits_clean = np.nan_to_num(logits_np, nan=0.0, posinf=1e4, neginf=-1e4)
    logits_max = np.max(logits_clean)
    exp_logits = np.exp(logits_clean - logits_max)
    probs = exp_logits / np.sum(exp_logits)
    if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
        probs = np.ones(len(logits_clean)) / len(logits_clean)
    return probs


def js_divergence(p, q, eps=1e-10):
    """JS(p, q) — symmetric, bounded [0, ln2]"""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def kl_divergence(p, q, eps=1e-10):
    """KL(p || q)"""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def wasserstein_1d_sort(p, q):
    """
    1-Wasserstein distance on sorted probability distributions.
    W_1(P, Q) = integral |F_P(x) - F_Q(x)| dx
    For discrete distributions sorted by value: = sum |CDF_P(i) - CDF_Q(i)| * step
    
    This is a practical approximation that treats the sorted probability mass
    as a 1D distribution. Not the full Wasserstein with embedding ground metric,
    but captures the "mass transport" aspect better than KL.
    """
    # Sort both distributions by probability value (descending)
    p_sorted = np.sort(p)[::-1]
    q_sorted = np.sort(q)[::-1]
    
    # CDF
    n = len(p_sorted)
    cdf_p = np.cumsum(p_sorted)
    cdf_q = np.cumsum(q_sorted)
    
    # W_1 = sum of |CDF_P - CDF_Q| * (1/n)
    return float(np.sum(np.abs(cdf_p - cdf_q)) / n)


def entropy(p, eps=1e-10):
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def top_k_overlap(p1, p2, k=10):
    top1 = set(np.argsort(p1)[-k:])
    top2 = set(np.argsort(p2)[-k:])
    return len(top1 & top2) / k


def get_next_dist(model, tokenizer, input_device, prompt):
    """Get P(next_token | prompt) as logits and probs."""
    import torch
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(input_device)
    attention_mask = inputs["attention_mask"].to(input_device)
    
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    
    logits = out.logits[0, -1, :].float().cpu().numpy()
    probs = safe_softmax(logits)
    
    del input_ids, attention_mask
    return logits, probs


def get_next_dist_with_prefix(model, tokenizer, input_device, base_prompt, token_ids_prefix):
    """
    Get P(next | base_prompt + token_ids_prefix).
    
    This is key for Exp 1: we fix a common continuation and compare
    conditional distributions, eliminating autoregressive drift.
    """
    import torch
    
    base_inputs = tokenizer(base_prompt, return_tensors="pt", truncation=True, max_length=64)
    base_ids = base_inputs["input_ids"].to(input_device)
    base_mask = base_inputs["attention_mask"].to(input_device)
    
    # Append the common prefix tokens
    prefix_tensor = torch.tensor([token_ids_prefix], device=input_device)
    input_ids = torch.cat([base_ids, prefix_tensor], dim=1)
    attention_mask = torch.cat([base_mask, torch.ones_like(prefix_tensor)], dim=1)
    
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    
    logits = out.logits[0, -1, :].float().cpu().numpy()
    probs = safe_softmax(logits)
    
    del input_ids, attention_mask
    return logits, probs


# ===== EXP 1: FUTURE PATH EQUIVALENCE =====

def exp1_future_path_equivalence(model, tokenizer, device, model_name,
                                   n_pairs=200, n_steps=10):
    """
    ★★★ KEY EXPERIMENT: Future Path Equivalence ★★★
    
    Find context pairs with similar P(next_1).
    Fix a COMMON continuation sequence (from one context's greedy rollout).
    Compare P(next|c1+prefix) vs P(next|c2+prefix) at each step.
    
    This eliminates the autoregressive drift problem from Phase 164 Exp D!
    
    If 1-step equivalence breaks down under shared prefix → 
    true state requires P(future_path), not just P(next_1)
    """
    import torch
    
    print("\n" + "="*60)
    print("EXP 1: Future Path Equivalence (Shared Prefix Method)")
    print("="*60)
    
    input_device = get_device_for_input(model)
    
    # Collect all prompts
    all_prompts = []
    all_domains = []
    for domain, prompts in DIVERSE_PROMPTS.items():
        for p in prompts:
            all_prompts.append(p)
            all_domains.append(domain)
    
    print(f"  Collected {len(all_prompts)} diverse prompts from {len(DIVERSE_PROMPTS)} domains")
    
    # Step 1: Get P(next_1) for all prompts
    print(f"  Step 1: Computing P(next_1) for {len(all_prompts)} prompts...")
    
    prompt_probs = []  # P(next_1) for each prompt
    prompt_logits = []
    
    for pidx, prompt in enumerate(all_prompts):
        if pidx % 20 == 0:
            print(f"    Prompt {pidx}/{len(all_prompts)}...")
        logits, probs = get_next_dist(model, tokenizer, input_device, prompt)
        prompt_probs.append(probs)
        prompt_logits.append(logits)
    
    # Step 2: Find pairs with similar P(next_1)
    print(f"  Step 2: Finding {n_pairs} pairs with similar P(next_1)...")
    
    # Compute JS divergence for all pairs
    n_p = len(all_prompts)
    pair_js = []
    
    for i in range(n_p):
        for j in range(i+1, n_p):
            js_val = js_divergence(prompt_probs[i], prompt_probs[j])
            pair_js.append((js_val, i, j))
    
    # Sort by JS divergence (ascending) — similar pairs first
    pair_js.sort(key=lambda x: x[0])
    
    # Select pairs from different similarity ranges
    # Bottom 25%: very similar (JS < Q25)
    # Middle 25-75%: moderately different
    # Top 25%: very different (JS > Q75)
    n_total_pairs = len(pair_js)
    
    # Select evenly across the range
    indices = np.linspace(0, n_total_pairs - 1, n_pairs, dtype=int)
    selected_pairs = [pair_js[idx] for idx in indices]
    
    print(f"  Selected {len(selected_pairs)} pairs, JS range: "
          f"[{selected_pairs[0][0]:.4f}, {selected_pairs[-1][0]:.4f}]")
    
    # Step 3: For each pair, generate shared prefix from c1's greedy rollout
    # Then compare P(next|c1+prefix) vs P(next|c2+prefix) at each step
    
    print(f"  Step 3: Computing path equivalence with shared prefix...")
    
    path_results = []  # List of per-pair results
    
    for pair_idx, (js_1step, i, j) in enumerate(selected_pairs):
        if pair_idx % 50 == 0:
            print(f"    Pair {pair_idx}/{len(selected_pairs)}...")
        
        prompt_i = all_prompts[i]
        prompt_j = all_prompts[j]
        
        # Generate shared prefix from c_i's greedy rollout
        inputs_i = tokenizer(prompt_i, return_tensors="pt", truncation=True, max_length=64)
        input_ids_i = inputs_i["input_ids"].to(input_device)
        attention_mask_i = inputs_i["attention_mask"].to(input_device)
        
        # Collect greedy tokens from c_i
        prefix_token_ids = []
        step_divergences = []
        
        current_ids_i = input_ids_i.clone()
        current_mask_i = attention_mask_i.clone()
        
        # Also track c_j's state
        inputs_j = tokenizer(prompt_j, return_tensors="pt", truncation=True, max_length=64)
        current_ids_j = inputs_j["input_ids"].to(input_device)
        current_mask_j = inputs_j["attention_mask"].to(input_device)
        
        for step in range(n_steps):
            try:
                # Get P(next) from both contexts with their respective prefixes
                with torch.no_grad():
                    out_i = model(input_ids=current_ids_i, attention_mask=current_mask_i)
                    out_j = model(input_ids=current_ids_j, attention_mask=current_mask_j)
                
                logits_i = out_i.logits[0, -1, :].float().cpu().numpy()
                logits_j = out_j.logits[0, -1, :].float().cpu().numpy()
                
                probs_i = safe_softmax(logits_i)
                probs_j = safe_softmax(logits_j)
                
                # Compute divergences at this step
                js_step = js_divergence(probs_i, probs_j)
                kl_i_j = kl_divergence(probs_i, probs_j)
                kl_j_i = kl_divergence(probs_j, probs_i)
                w1_step = wasserstein_1d_sort(probs_i, probs_j)
                top10_step = top_k_overlap(probs_i, probs_j, k=10)
                top5_step = top_k_overlap(probs_i, probs_j, k=5)
                top1_match = 1 if np.argmax(probs_i) == np.argmax(probs_j) else 0
                
                # Entropy of each
                ent_i = entropy(probs_i)
                ent_j = entropy(probs_j)
                
                step_divergences.append({
                    "step": step,
                    "js": js_step,
                    "kl_sym": 0.5 * (kl_i_j + kl_j_i),  # Symmetrized KL
                    "w1": w1_step,
                    "top10_overlap": top10_step,
                    "top5_overlap": top5_step,
                    "top1_match": top1_match,
                    "ent_i": ent_i,
                    "ent_j": ent_j,
                    "delta_entropy": abs(ent_i - ent_j),
                })
                
                # Get the NEXT token from c_i (greedy) — this is the shared prefix
                next_token_id = int(np.argmax(probs_i))
                prefix_token_ids.append(next_token_id)
                
                # Append to BOTH sequences (shared prefix!)
                next_tok = torch.tensor([[next_token_id]], device=input_device)
                current_ids_i = torch.cat([current_ids_i, next_tok], dim=1)
                current_mask_i = torch.cat([current_mask_i, torch.ones_like(next_tok)], dim=1)
                current_ids_j = torch.cat([current_ids_j, next_tok], dim=1)
                current_mask_j = torch.cat([current_mask_j, torch.ones_like(next_tok)], dim=1)
            
            except Exception as e:
                print(f"    Error at step {step}: {e}")
                break
        
        # Clean up tensors
        del current_ids_i, current_mask_i, current_ids_j, current_mask_j
        
        path_results.append({
            "pair_idx": pair_idx,
            "prompt_i_idx": i,
            "prompt_j_idx": j,
            "domain_i": all_domains[i],
            "domain_j": all_domains[j],
            "js_1step_initial": round(js_1step, 6),
            "same_domain": all_domains[i] == all_domains[j],
            "steps": step_divergences,
            "n_steps_completed": len(step_divergences),
        })
    
    # ===== Aggregate results =====
    
    # Per-step divergence stats
    step_js = defaultdict(list)
    step_kl = defaultdict(list)
    step_w1 = defaultdict(list)
    step_top10 = defaultdict(list)
    step_top5 = defaultdict(list)
    step_top1_match = defaultdict(list)
    
    for pr in path_results:
        for s in pr["steps"]:
            step_js[s["step"]].append(s["js"])
            step_kl[s["step"]].append(s["kl_sym"])
            step_w1[s["step"]].append(s["w1"])
            step_top10[s["step"]].append(s["top10_overlap"])
            step_top5[s["step"]].append(s["top5_overlap"])
            step_top1_match[s["step"]].append(s["top1_match"])
    
    step_stats = {}
    for step in range(n_steps):
        if step in step_js:
            step_stats[step] = {
                "js_mean": round(float(np.mean(step_js[step])), 6),
                "kl_mean": round(float(np.mean(step_kl[step])), 6),
                "w1_mean": round(float(np.mean(step_w1[step])), 6),
                "top10_mean": round(float(np.mean(step_top10[step])), 4),
                "top5_mean": round(float(np.mean(step_top5[step])), 4),
                "top1_match_rate": round(float(np.mean(step_top1_match[step])), 4),
            }
    
    # KEY ANALYSIS: Pairs that start similar (JS < Q25) — how quickly do they diverge?
    initial_js_values = [pr["js_1step_initial"] for pr in path_results]
    js_q25 = np.percentile(initial_js_values, 25)
    js_q50 = np.percentile(initial_js_values, 50)
    js_q75 = np.percentile(initial_js_values, 75)
    
    similar_pairs = [pr for pr in path_results if pr["js_1step_initial"] < js_q25]
    medium_pairs = [pr for pr in path_results if js_q25 <= pr["js_1step_initial"] < js_q75]
    different_pairs = [pr for pr in path_results if pr["js_1step_initial"] >= js_q75]
    
    def compute_divergence_growth(pair_list, label):
        """How fast does JS grow for this group?"""
        growth = {}
        for step in range(n_steps):
            js_vals = [pr["steps"][step]["js"] for pr in pair_list if step < len(pr["steps"])]
            if js_vals:
                growth[step] = {
                    "js_mean": round(float(np.mean(js_vals)), 6),
                    "js_std": round(float(np.std(js_vals)), 6),
                }
        return growth
    
    similar_growth = compute_divergence_growth(similar_pairs, "similar")
    medium_growth = compute_divergence_growth(medium_pairs, "medium")
    different_growth = compute_divergence_growth(different_pairs, "different")
    
    # Same-domain vs different-domain
    same_domain_pairs = [pr for pr in path_results if pr["same_domain"]]
    diff_domain_pairs = [pr for pr in path_results if not pr["same_domain"]]
    
    same_domain_growth = compute_divergence_growth(same_domain_pairs, "same_domain")
    diff_domain_growth = compute_divergence_growth(diff_domain_pairs, "diff_domain")
    
    # Correlation: initial JS vs final JS
    if path_results and all(len(pr["steps"]) >= n_steps for pr in path_results):
        initial_js = [pr["js_1step_initial"] for pr in path_results]
        final_js = [pr["steps"][n_steps-1]["js"] for pr in path_results]
        corr_initial_final = float(np.corrcoef(initial_js, final_js)[0, 1])
    else:
        corr_initial_final = 0.0
    
    # Path breakdown rate: pairs with small initial JS but large final JS
    if similar_pairs:
        breakdown_rates = {}
        for step in range(n_steps):
            js_at_step = [pr["steps"][step]["js"] for pr in similar_pairs if step < len(pr["steps"])]
            js_at_0 = [pr["steps"][0]["js"] for pr in similar_pairs if len(pr["steps"]) > 0]
            if js_at_step and js_at_0:
                # How many pairs have JS(step) > 2 * JS(0)?
                ratio_exploded = sum(1 for js_s in js_at_step 
                                    if js_s > 2 * np.mean(js_at_0)) / len(js_at_step)
                breakdown_rates[step] = round(ratio_exploded, 4)
    
    results = {
        "n_prompts": len(all_prompts),
        "n_pairs": len(selected_pairs),
        "n_steps": n_steps,
        "step_stats": step_stats,
        "js_quantiles": {
            "q25": round(float(js_q25), 6),
            "q50": round(float(js_q50), 6),
            "q75": round(float(js_q75), 6),
        },
        "n_similar_pairs": len(similar_pairs),
        "n_medium_pairs": len(medium_pairs),
        "n_different_pairs": len(different_pairs),
        "similar_growth": similar_growth,
        "medium_growth": medium_growth,
        "different_growth": different_growth,
        "n_same_domain": len(same_domain_pairs),
        "n_diff_domain": len(diff_domain_pairs),
        "same_domain_growth": same_domain_growth,
        "diff_domain_growth": diff_domain_growth,
        "corr_initial_final_js": round(corr_initial_final, 4),
        "breakdown_rates": breakdown_rates if 'breakdown_rates' in dir() else {},
    }
    
    print(f"\n  === Path Equivalence Results ===")
    print(f"  JS quantiles: Q25={js_q25:.4f}, Q50={js_q50:.4f}, Q75={js_q75:.4f}")
    for step in range(min(5, n_steps)):
        if step in step_stats:
            print(f"  Step {step}: JS={step_stats[step]['js_mean']:.4f}, "
                  f"W1={step_stats[step]['w1_mean']:.4f}, "
                  f"top10={step_stats[step]['top10_mean']:.3f}, "
                  f"top1_match={step_stats[step]['top1_match_rate']:.3f}")
    
    print(f"\n  Correlation(initial_JS, final_JS) = {corr_initial_final:.4f}")
    print(f"  Similar pairs: {len(similar_pairs)}, Medium: {len(medium_pairs)}, Different: {len(different_pairs)}")
    print(f"  Same domain: {len(same_domain_pairs)}, Different domain: {len(diff_domain_pairs)}")
    
    return results


# ===== EXP 2: CONSTRAINT JACOBIAN =====

def exp2_constraint_jacobian(model, tokenizer, device, model_name,
                              n_concepts=40, n_templates=27, n_top_dims=200):
    """
    ★★★ KEY EXPERIMENT: Constraint Jacobian ★★★
    
    For each concept c, measure how P(future) changes across different contexts.
    
    If deformation is context-independent → true operator (Jacobian rank 1)
    If deformation depends heavily on context → not a true operator
    
    The "Constraint Jacobian" is approximated by:
    - Build a matrix D_c where each row is the deformation vector for a different context
    - SVD of D_c reveals the "constraint channels" through which c acts
    - If rank(D_c) = 1 → c acts through a SINGLE constraint channel → TRUE OPERATOR
    - If rank(D_c) = k → c acts through k constraint channels
    """
    import torch
    
    print("\n" + "="*60)
    print("EXP 2: Constraint Jacobian (Concept Deformation Rank)")
    print("="*60)
    
    input_device = get_device_for_input(model)
    
    # Select subset of concepts
    concepts_to_test = ALL_CONCEPTS[:n_concepts]
    templates = CONTEXT_TEMPLATES[:n_templates]
    
    print(f"  Testing {len(concepts_to_test)} concepts × {len(templates)} templates")
    
    # Step 1: Compute baselines (template with "the")
    print(f"  Step 1: Computing baselines...")
    baseline_logits = {}
    baseline_probs = {}
    
    for tidx, template in enumerate(templates):
        baseline_prompt = template.replace("___", "the")
        bl, bp = get_next_dist(model, tokenizer, input_device, baseline_prompt)
        baseline_logits[template] = bl
        baseline_probs[template] = bp
    
    # Step 2: For each concept, build deformation matrix across contexts
    print(f"  Step 2: Building deformation matrices...")
    
    concept_jacobian_info = {}
    
    for cidx, concept in enumerate(concepts_to_test):
        if cidx % 10 == 0:
            print(f"    Concept {cidx}/{len(concepts_to_test)}: {concept}")
        
        family = CONCEPT_TO_FAMILY[concept]
        
        # Collect deformation vectors across templates
        deform_vectors = []  # Each row = Δlogits for one template
        js_values = []
        kl_values = []
        w1_values = []
        top10_overlaps = []
        
        for template in templates:
            intervened_prompt = template.replace("___", concept)
            int_logits, int_probs = get_next_dist(model, tokenizer, input_device, intervened_prompt)
            
            bl_probs = baseline_probs[template]
            bl_logits = baseline_logits[template]
            
            # Deformation vector = logit shift (top dimensions by absolute shift)
            delta_logits = int_logits - bl_logits
            
            # For SVD, take the top-n_top_dims dimensions by absolute shift
            top_dims = np.argsort(np.abs(delta_logits))[-n_top_dims:]
            deform_vec = delta_logits[top_dims]
            deform_vectors.append(deform_vec)
            
            # Also compute divergences
            js_val = js_divergence(int_probs, bl_probs)
            kl_val = kl_divergence(int_probs, bl_probs)
            w1_val = wasserstein_1d_sort(int_probs, bl_probs)
            t10 = top_k_overlap(int_probs, bl_probs, k=10)
            
            js_values.append(js_val)
            kl_values.append(kl_val)
            w1_values.append(w1_val)
            top10_overlaps.append(t10)
        
        # Build deformation matrix D_c: [n_templates, n_top_dims]
        D_c = np.array(deform_vectors)
        
        # SVD to find rank
        try:
            U, S, Vt = np.linalg.svd(D_c, full_matrices=False)
            
            # Effective rank (where cumulative energy reaches 90%, 95%, 99%)
            energy = S ** 2
            total_energy = np.sum(energy)
            cum_energy = np.cumsum(energy) / total_energy
            
            rank_90 = int(np.searchsorted(cum_energy, 0.90) + 1)
            rank_95 = int(np.searchsorted(cum_energy, 0.95) + 1)
            rank_99 = int(np.searchsorted(cum_energy, 0.99) + 1)
            
            # Top singular value ratio (how much of the deformation is along one direction?)
            top1_ratio = float(S[0] ** 2 / total_energy)
            top3_ratio = float(np.sum(S[:3] ** 2) / total_energy)
            top5_ratio = float(np.sum(S[:5] ** 2) / total_energy)
            
            # Singular value spectrum (first 10)
            sv_spectrum = [round(float(s), 4) for s in S[:10]]
            
        except Exception:
            rank_90, rank_95, rank_99 = 0, 0, 0
            top1_ratio, top3_ratio, top5_ratio = 0, 0, 0
            sv_spectrum = []
        
        # Deformation stability metrics
        js_cv = float(np.std(js_values) / max(np.mean(js_values), 1e-6))  # Coefficient of variation
        kl_cv = float(np.std(kl_values) / max(np.mean(kl_values), 1e-6))
        
        concept_jacobian_info[concept] = {
            "family": family,
            "js_mean": round(float(np.mean(js_values)), 4),
            "js_std": round(float(np.std(js_values)), 4),
            "js_cv": round(js_cv, 4),
            "kl_mean": round(float(np.mean(kl_values)), 4),
            "kl_std": round(float(np.std(kl_values)), 4),
            "w1_mean": round(float(np.mean(w1_values)), 4),
            "w1_std": round(float(np.std(w1_values)), 4),
            "top10_overlap_mean": round(float(np.mean(top10_overlaps)), 4),
            "deformation_rank_90": rank_90,
            "deformation_rank_95": rank_95,
            "deformation_rank_99": rank_99,
            "top1_singular_ratio": round(top1_ratio, 4),
            "top3_singular_ratio": round(top3_ratio, 4),
            "top5_singular_ratio": round(top5_ratio, 4),
            "sv_spectrum": sv_spectrum,
        }
    
    # ===== Aggregate by family =====
    family_stats = {}
    for family in CONCEPT_FAMILIES:
        concepts_in_family = [c for c in concepts_to_test if CONCEPT_TO_FAMILY[c] == family]
        if not concepts_in_family:
            continue
        
        family_stats[family] = {
            "n_concepts": len(concepts_in_family),
            "rank_90_mean": round(float(np.mean([concept_jacobian_info[c]["deformation_rank_90"] 
                                                  for c in concepts_in_family])), 2),
            "rank_95_mean": round(float(np.mean([concept_jacobian_info[c]["deformation_rank_95"] 
                                                  for c in concepts_in_family])), 2),
            "top1_ratio_mean": round(float(np.mean([concept_jacobian_info[c]["top1_singular_ratio"] 
                                                      for c in concepts_in_family])), 4),
            "top3_ratio_mean": round(float(np.mean([concept_jacobian_info[c]["top3_singular_ratio"] 
                                                      for c in concepts_in_family])), 4),
            "js_cv_mean": round(float(np.mean([concept_jacobian_info[c]["js_cv"] 
                                                for c in concepts_in_family])), 4),
        }
    
    # Find "true operators" — concepts with rank≈1 (top1_ratio > 0.8)
    true_operators = [c for c in concepts_to_test 
                      if concept_jacobian_info[c]["top1_singular_ratio"] > 0.8]
    weak_operators = [c for c in concepts_to_test 
                      if concept_jacobian_info[c]["top1_singular_ratio"] < 0.5]
    
    results = {
        "n_concepts": len(concepts_to_test),
        "n_templates": len(templates),
        "n_top_dims": n_top_dims,
        "concept_jacobian_info": concept_jacobian_info,
        "family_stats": family_stats,
        "true_operators_top1_gt_0.8": true_operators,
        "n_true_operators": len(true_operators),
        "weak_operators_top1_lt_0.5": weak_operators,
        "n_weak_operators": len(weak_operators),
        "overall_top1_ratio_mean": round(float(np.mean([concept_jacobian_info[c]["top1_singular_ratio"] 
                                                          for c in concepts_to_test])), 4),
        "overall_rank_90_mean": round(float(np.mean([concept_jacobian_info[c]["deformation_rank_90"] 
                                                      for c in concepts_to_test])), 2),
    }
    
    print(f"\n  === Constraint Jacobian Results ===")
    print(f"  Overall: top1_singular_ratio = {results['overall_top1_ratio_mean']:.4f}")
    print(f"  Overall: rank_90 = {results['overall_rank_90_mean']:.1f}")
    print(f"  True operators (top1>0.8): {true_operators}")
    print(f"  Weak operators (top1<0.5): {weak_operators}")
    
    for family, fs in family_stats.items():
        print(f"  {family}: rank_90={fs['rank_90_mean']:.1f}, "
              f"top1_ratio={fs['top1_ratio_mean']:.3f}, "
              f"js_cv={fs['js_cv_mean']:.3f}")
    
    return results


# ===== EXP 3: CONCEPT FIXED POINT =====

def exp3_concept_fixed_point(model, tokenizer, device, model_name,
                              n_concepts=20, n_domains=7):
    """
    ★★★ KEY EXPERIMENT: Concept Fixed Point ★★★
    
    Find concepts whose deformation pattern is STABLE across very diverse contexts.
    Use contexts from very different domains (poetry, legal, dialogue, technical).
    
    A "fixed point" concept is one whose deformation vector is nearly the same
    regardless of the context. This is the closest to a "true operator" T_c.
    
    We measure:
    - Cross-domain deformation correlation (how similar is ΔP in poetry vs legal?)
    - Deformation vector cosine similarity across domains
    - JS divergence of deformation-induced distributions across domains
    """
    import torch
    
    print("\n" + "="*60)
    print("EXP 3: Concept Fixed Point (Cross-Domain Stability)")
    print("="*60)
    
    input_device = get_device_for_input(model)
    
    # Select concepts from each family
    concepts_to_test = []
    for family, concepts in CONCEPT_FAMILIES.items():
        concepts_to_test.extend(concepts[:4])  # 4 per family
    
    # Use one representative template from each domain
    domain_templates = {}
    for domain, prompts in DIVERSE_PROMPTS.items():
        if domain in ["narrative", "technical", "dialogue", "descriptive", "philosophical", "poetry", "legal"]:
            # Use a modified version with ___ slot
            domain_templates[domain] = prompts[0]
    
    # Create "insertable" versions of each domain prompt
    # We'll insert the concept at the end of the prompt
    domain_prompts = {}
    for domain, prompt in domain_templates.items():
        domain_prompts[domain] = prompt
    
    print(f"  Testing {len(concepts_to_test)} concepts × {len(domain_prompts)} domains")
    
    # Step 1: Get baseline P(next) for each domain
    print(f"  Step 1: Computing domain baselines...")
    
    domain_baseline_probs = {}
    domain_baseline_logits = {}
    
    for domain, prompt in domain_prompts.items():
        bl, bp = get_next_dist(model, tokenizer, input_device, prompt)
        domain_baseline_probs[domain] = bp
        domain_baseline_logits[domain] = bl
    
    # Step 2: For each concept, compute deformation in each domain
    print(f"  Step 2: Computing cross-domain deformations...")
    
    concept_domain_deform = {}  # concept -> domain -> deformation info
    
    for concept in concepts_to_test:
        concept_domain_deform[concept] = {}
        family = CONCEPT_TO_FAMILY[concept]
        
        for domain, prompt in domain_prompts.items():
            # Insert concept at end of prompt
            intervened = prompt + " " + concept
            
            int_logits, int_probs = get_next_dist(model, tokenizer, input_device, intervened)
            bl_probs = domain_baseline_probs[domain]
            bl_logits = domain_baseline_logits[domain]
            
            # Deformation metrics
            js_val = js_divergence(int_probs, bl_probs)
            kl_val = kl_divergence(int_probs, bl_probs)
            w1_val = wasserstein_1d_sort(int_probs, bl_probs)
            t10 = top_k_overlap(int_probs, bl_probs, k=10)
            t5 = top_k_overlap(int_probs, bl_probs, k=5)
            
            # Deformation vector (logit shift) — top 200 dims
            delta_logits = int_logits - bl_logits
            top_dims = np.argsort(np.abs(delta_logits))[-200:]
            deform_vec = delta_logits[top_dims]
            
            concept_domain_deform[concept][domain] = {
                "js": round(js_val, 6),
                "kl": round(kl_val, 6),
                "w1": round(w1_val, 6),
                "top10_overlap": round(t10, 4),
                "top5_overlap": round(t5, 4),
                "deform_vec_norm": round(float(np.linalg.norm(deform_vec)), 4),
                "deform_vec": deform_vec,  # Keep for cross-domain comparison
            }
    
    # Step 3: Cross-domain deformation similarity
    print(f"  Step 3: Computing cross-domain deformation stability...")
    
    domains_list = list(domain_prompts.keys())
    n_domains = len(domains_list)
    
    concept_stability = {}
    
    for concept in concepts_to_test:
        family = CONCEPT_TO_FAMILY[concept]
        
        # JS values across domains
        js_values = [concept_domain_deform[concept][d]["js"] for d in domains_list]
        kl_values = [concept_domain_deform[concept][d]["kl"] for d in domains_list]
        w1_values = [concept_domain_deform[concept][d]["w1"] for d in domains_list]
        t10_values = [concept_domain_deform[concept][d]["top10_overlap"] for d in domains_list]
        
        # Cross-domain deformation vector correlation
        deform_vecs = [concept_domain_deform[concept][d]["deform_vec"] for d in domains_list]
        
        # Pairwise cosine similarity of deformation vectors
        cross_domain_cos = []
        for i in range(n_domains):
            for j in range(i+1, n_domains):
                v1, v2 = deform_vecs[i], deform_vecs[j]
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 > 1e-10 and n2 > 1e-10:
                    cos_val = float(np.dot(v1, v2) / (n1 * n2))
                    cross_domain_cos.append(cos_val)
                else:
                    cross_domain_cos.append(0.0)
        
        # JS stability (CV = std/mean)
        js_cv = float(np.std(js_values) / max(np.mean(js_values), 1e-6))
        kl_cv = float(np.std(kl_values) / max(np.mean(kl_values), 1e-6))
        
        concept_stability[concept] = {
            "family": family,
            "js_mean": round(float(np.mean(js_values)), 6),
            "js_std": round(float(np.std(js_values)), 6),
            "js_cv": round(js_cv, 4),
            "kl_mean": round(float(np.mean(kl_values)), 6),
            "kl_cv": round(kl_cv, 4),
            "w1_mean": round(float(np.mean(w1_values)), 6),
            "top10_mean": round(float(np.mean(t10_values)), 4),
            "cross_domain_cos_mean": round(float(np.mean(cross_domain_cos)), 4) if cross_domain_cos else 0,
            "cross_domain_cos_std": round(float(np.std(cross_domain_cos)), 4) if cross_domain_cos else 0,
            "cross_domain_cos_min": round(float(np.min(cross_domain_cos)), 4) if cross_domain_cos else 0,
        }
    
    # Step 4: Find "fixed point" concepts
    # Fixed point = high cross-domain cosine (deformation pattern is similar regardless of context)
    
    # Sort by cross_domain_cos_mean (descending) — most stable first
    sorted_concepts = sorted(concept_stability.keys(), 
                             key=lambda c: concept_stability[c]["cross_domain_cos_mean"],
                             reverse=True)
    
    fixed_points = [c for c in sorted_concepts if concept_stability[c]["cross_domain_cos_mean"] > 0.7]
    context_dependent = [c for c in sorted_concepts if concept_stability[c]["cross_domain_cos_mean"] < 0.3]
    
    # Per-family aggregate
    family_stability = {}
    for family in CONCEPT_FAMILIES:
        concepts_in_family = [c for c in concepts_to_test if CONCEPT_TO_FAMILY[c] == family]
        if not concepts_in_family:
            continue
        
        family_stability[family] = {
            "cross_domain_cos_mean": round(float(np.mean([concept_stability[c]["cross_domain_cos_mean"] 
                                                           for c in concepts_in_family])), 4),
            "js_cv_mean": round(float(np.mean([concept_stability[c]["js_cv"] 
                                               for c in concepts_in_family])), 4),
        }
    
    # Remove deform_vec from output (too large for JSON)
    for concept in concepts_to_test:
        for domain in domains_list:
            del concept_domain_deform[concept][domain]["deform_vec"]
    
    results = {
        "n_concepts": len(concepts_to_test),
        "n_domains": len(domains_list),
        "domains": domains_list,
        "concept_stability": concept_stability,
        "family_stability": family_stability,
        "fixed_points_cos_gt_0.7": fixed_points,
        "n_fixed_points": len(fixed_points),
        "context_dependent_cos_lt_0.3": context_dependent,
        "n_context_dependent": len(context_dependent),
        "overall_cross_domain_cos": round(float(np.mean([concept_stability[c]["cross_domain_cos_mean"] 
                                                           for c in concepts_to_test])), 4),
        "sorted_by_stability": [(c, round(concept_stability[c]["cross_domain_cos_mean"], 4)) 
                                for c in sorted_concepts],
    }
    
    print(f"\n  === Concept Fixed Point Results ===")
    print(f"  Overall cross-domain cosine: {results['overall_cross_domain_cos']:.4f}")
    print(f"  Fixed points (cos>0.7): {fixed_points}")
    print(f"  Context-dependent (cos<0.3): {context_dependent}")
    
    print(f"\n  === Top 10 most stable concepts ===")
    for c, cos_val in results['sorted_by_stability'][:10]:
        print(f"    {c} ({CONCEPT_TO_FAMILY[c]}): cross_domain_cos={cos_val:.4f}")
    
    print(f"\n  === Bottom 5 least stable concepts ===")
    for c, cos_val in results['sorted_by_stability'][-5:]:
        print(f"    {c} ({CONCEPT_TO_FAMILY[c]}): cross_domain_cos={cos_val:.4f}")
    
    return results


# ===== EXP 4: OPERATOR CLOSURE =====

def exp4_operator_closure(model, tokenizer, device, model_name, n_templates=10):
    """
    ★★★ KEY EXPERIMENT: Operator Closure ★★★
    
    Test if the deformation family is algebraically closed under composition.
    
    1. Associativity: T_A ∘ (T_B ∘ T_C) ≈ (T_A ∘ T_B) ∘ T_C?
    2. Closure: T_A ∘ T_B produces deformation in the SAME family as T_A or T_B?
    3. Identity: Is there a concept that acts like the identity operator?
    4. Inverse: Is there T_c' such that T_c ∘ T_c' ≈ I?
    
    This tests whether language has a genuine algebraic structure.
    """
    import torch
    
    print("\n" + "="*60)
    print("EXP 4: Operator Closure (Algebraic Structure Test)")
    print("="*60)
    
    input_device = get_device_for_input(model)
    templates = CONTEXT_TEMPLATES[:n_templates]
    
    # ===== Test 1: Associativity =====
    print(f"\n  === Test 1: Associativity ===")
    print(f"  Testing T_A ∘ (T_B ∘ T_C) vs (T_A ∘ T_B) ∘ T_C")
    
    associativity_results = []
    
    for a, b, c in COMPOSITION_TRIPLES:
        print(f"  Testing: {a} + {b} + {c}")
        
        for template in templates[:5]:  # Use 5 templates for speed
            baseline_prompt = template.replace("___", "the")
            bl_logits, bl_probs = get_next_dist(model, tokenizer, input_device, baseline_prompt)
            
            # T_A: template with a
            a_prompt = template.replace("___", a)
            a_logits, a_probs = get_next_dist(model, tokenizer, input_device, a_prompt)
            
            # T_B: template with b
            b_prompt = template.replace("___", b)
            b_logits, b_probs = get_next_dist(model, tokenizer, input_device, b_prompt)
            
            # T_C: template with c
            c_prompt = template.replace("___", c)
            c_logits, c_probs = get_next_dist(model, tokenizer, input_device, c_prompt)
            
            # T_{AB}: template with "a b"
            ab_prompt = template.replace("___", f"{a} {b}")
            ab_logits, ab_probs = get_next_dist(model, tokenizer, input_device, ab_prompt)
            
            # T_{BC}: template with "b c"
            bc_prompt = template.replace("___", f"{b} {c}")
            bc_logits, bc_probs = get_next_dist(model, tokenizer, input_device, bc_prompt)
            
            # T_{ABC}: template with "a b c"
            abc_prompt = template.replace("___", f"{a} {b} {c}")
            abc_logits, abc_probs = get_next_dist(model, tokenizer, input_device, abc_prompt)
            
            # T_{(AB)C}: template with "a b c" (same as T_{ABC} for us)
            # For true associativity test, we need:
            # "a b" composed with "c" vs "a" composed with "b c"
            # We approximate this by:
            # Left-associative: (T_A + T_B) + T_C
            # Right-associative: T_A + (T_B + T_C)
            
            delta_a = a_logits - bl_logits
            delta_b = b_logits - bl_logits
            delta_c = c_logits - bl_logits
            
            # Left-associative: (Δ_A + Δ_B) + Δ_C
            delta_left = (delta_a + delta_b) + delta_c
            
            # Right-associative: Δ_A + (Δ_B + Δ_C)
            delta_right = delta_a + (delta_b + delta_c)
            
            # They're mathematically identical! Vector addition is associative!
            # So we need a DIFFERENT test of associativity.
            
            # Better test: Compare T_{ABC} with:
            # 1. T_{AB} + T_C (apply AB composition, then add C)
            # 2. T_A + T_{BC} (apply A, then add BC composition)
            
            delta_ab = ab_logits - bl_logits
            delta_bc = bc_logits - bl_logits
            delta_abc = abc_logits - bl_logits
            
            # Left: Δ_AB + Δ_C
            delta_left_comp = delta_ab + delta_c
            
            # Right: Δ_A + Δ_BC
            delta_right_comp = delta_a + delta_bc
            
            # Cosine similarity with actual Δ_ABC
            def cosine_sim(v1, v2):
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 < 1e-10 or n2 < 1e-10:
                    return 0.0
                return float(np.dot(v1, v2) / (n1 * n2))
            
            cos_left = cosine_sim(delta_abc, delta_left_comp)
            cos_right = cosine_sim(delta_abc, delta_right_comp)
            
            # JS divergence between T_{ABC} and the two approximations
            left_approx_probs = safe_softmax(bl_logits + delta_left_comp)
            right_approx_probs = safe_softmax(bl_logits + delta_right_comp)
            
            js_left = js_divergence(abc_probs, left_approx_probs)
            js_right = js_divergence(abc_probs, right_approx_probs)
            
            # Also: does T_{ABC} look more like T_A, T_B, or T_C?
            js_abc_a = js_divergence(abc_probs, a_probs)
            js_abc_b = js_divergence(abc_probs, b_probs)
            js_abc_c = js_divergence(abc_probs, c_probs)
            
            associativity_results.append({
                "triple": f"{a}+{b}+{c}",
                "cos_left_assoc": round(cos_left, 4),
                "cos_right_assoc": round(cos_right, 4),
                "js_left_assoc": round(js_left, 6),
                "js_right_assoc": round(js_right, 6),
                "js_abc_vs_a": round(js_abc_a, 6),
                "js_abc_vs_b": round(js_abc_b, 6),
                "js_abc_vs_c": round(js_abc_c, 6),
                "dominant_concept": min([("A", js_abc_a), ("B", js_abc_b), ("C", js_abc_c)], 
                                        key=lambda x: x[1])[0],
            })
    
    # ===== Test 2: Closure — T_A ∘ T_B in same family? =====
    print(f"\n  === Test 2: Operator Closure ===")
    print(f"  Testing if T_A ∘ T_B produces deformation in same family")
    
    # Use pairs from different families
    cross_family_pairs = [
        ("cat", "red", "animal", "color"),
        ("car", "fast", "vehicle", "abstract"),  # fast is in abstract? no, emotion/abstract overlap
        ("dog", "blue", "animal", "color"),
        ("train", "green", "vehicle", "color"),
        ("justice", "red", "abstract", "color"),
        ("love", "car", "emotion", "vehicle"),
        ("freedom", "cat", "abstract", "animal"),
        ("anger", "train", "emotion", "vehicle"),
    ]
    
    closure_results = []
    
    for c1, c2, f1, f2 in cross_family_pairs:
        for template in templates[:3]:
            baseline_prompt = template.replace("___", "the")
            bl_logits, bl_probs = get_next_dist(model, tokenizer, input_device, baseline_prompt)
            
            c1_prompt = template.replace("___", c1)
            c1_logits, c1_probs = get_next_dist(model, tokenizer, input_device, c1_prompt)
            
            c2_prompt = template.replace("___", c2)
            c2_logits, c2_probs = get_next_dist(model, tokenizer, input_device, c2_prompt)
            
            composed_prompt = template.replace("___", f"{c1} {c2}")
            composed_logits, composed_probs = get_next_dist(model, tokenizer, input_device, composed_prompt)
            
            # Is the composed deformation closer to c1's family or c2's family?
            js_comp_c1 = js_divergence(composed_probs, c1_probs)
            js_comp_c2 = js_divergence(composed_probs, c2_probs)
            
            # Also: cosine of deformation vectors
            delta_c1 = c1_logits - bl_logits
            delta_c2 = c2_logits - bl_logits
            delta_comp = composed_logits - bl_logits
            
            cos_comp_c1 = cosine_sim(delta_comp, delta_c1)
            cos_comp_c2 = cosine_sim(delta_comp, delta_c2)
            
            # Dominant family
            if js_comp_c1 < js_comp_c2:
                dominant = f1
            else:
                dominant = f2
            
            closure_results.append({
                "c1": c1, "c2": c2, "f1": f1, "f2": f2,
                "js_comp_c1": round(js_comp_c1, 6),
                "js_comp_c2": round(js_comp_c2, 6),
                "cos_comp_c1": round(cos_comp_c1, 4),
                "cos_comp_c2": round(cos_comp_c2, 4),
                "dominant_family": dominant,
            })
    
    # ===== Test 3: Identity operator =====
    print(f"\n  === Test 3: Identity Operator ===")
    print(f"  Testing if any concept acts like the identity")
    
    # Test concepts that might be "neutral" — "the", "a", "thing", "something"
    identity_candidates = ["the", "a", "thing", "something", "one", "it", "that", "this"]
    
    identity_results = []
    
    for candidate in identity_candidates:
        js_values = []
        
        for template in templates[:5]:
            baseline_prompt = template.replace("___", "the")  # "the" is our default baseline
            cand_prompt = template.replace("___", candidate)
            
            _, bl_probs = get_next_dist(model, tokenizer, input_device, baseline_prompt)
            _, cand_probs = get_next_dist(model, tokenizer, input_device, cand_prompt)
            
            js_val = js_divergence(cand_probs, bl_probs)
            js_values.append(js_val)
        
        identity_results.append({
            "candidate": candidate,
            "js_mean": round(float(np.mean(js_values)), 6),
            "js_std": round(float(np.std(js_values)), 6),
        })
    
    # Sort by JS (ascending) — most identity-like first
    identity_results.sort(key=lambda x: x["js_mean"])
    
    # ===== Test 4: Inverse operator =====
    print(f"\n  === Test 4: Inverse Operator ===")
    print(f"  Testing if T_c' ∘ T_c ≈ I for any c', c")
    
    # Test antonym pairs: "hot" + "cold", "big" + "small", "good" + "bad"
    antonym_pairs = [
        ("hot", "cold"),
        ("big", "small"),
        ("good", "bad"),
        ("happy", "sad"),
        ("light", "dark"),
        ("fast", "slow"),
        ("young", "old"),
        ("rich", "poor"),
    ]
    
    inverse_results = []
    
    for word, antonym in antonym_pairs:
        js_values = []
        
        for template in templates[:5]:
            # P(baseline + word + antonym) vs P(baseline + "the")
            baseline_prompt = template.replace("___", "the")
            composed_prompt = template.replace("___", f"{word} {antonym}")
            
            _, bl_probs = get_next_dist(model, tokenizer, input_device, baseline_prompt)
            _, comp_probs = get_next_dist(model, tokenizer, input_device, composed_prompt)
            
            js_val = js_divergence(comp_probs, bl_probs)
            js_values.append(js_val)
        
        # Also: word alone vs antonym alone
        js_word_alone = []
        js_antonym_alone = []
        
        for template in templates[:5]:
            baseline_prompt = template.replace("___", "the")
            
            word_prompt = template.replace("___", word)
            _, word_probs = get_next_dist(model, tokenizer, input_device, word_prompt)
            
            antonym_prompt = template.replace("___", antonym)
            _, ant_probs = get_next_dist(model, tokenizer, input_device, antonym_prompt)
            
            _, bl_probs = get_next_dist(model, tokenizer, input_device, baseline_prompt)
            
            js_word_alone.append(js_divergence(word_probs, bl_probs))
            js_antonym_alone.append(js_divergence(ant_probs, bl_probs))
        
        inverse_results.append({
            "word": word,
            "antonym": antonym,
            "js_composed_vs_baseline_mean": round(float(np.mean(js_values)), 6),
            "js_word_alone_mean": round(float(np.mean(js_word_alone)), 6),
            "js_antonym_alone_mean": round(float(np.mean(js_antonym_alone)), 6),
            "cancellation_ratio": round(float(np.mean(js_values)) / 
                                        max(float(np.mean(js_word_alone)), 1e-6), 4),
        })
    
    # Sort by cancellation_ratio (ascending) — best cancellation first
    inverse_results.sort(key=lambda x: x["cancellation_ratio"])
    
    # ===== Aggregate =====
    
    # Associativity summary
    assoc_cos_left = [r["cos_left_assoc"] for r in associativity_results]
    assoc_cos_right = [r["cos_right_assoc"] for r in associativity_results]
    assoc_js_left = [r["js_left_assoc"] for r in associativity_results]
    assoc_js_right = [r["js_right_assoc"] for r in associativity_results]
    dominant_concepts = [r["dominant_concept"] for r in associativity_results]
    
    # Closure summary
    dominant_family_count = defaultdict(int)
    for r in closure_results:
        dominant_family_count[r["dominant_family"]] += 1
    
    # Identity summary
    best_identity = identity_results[0] if identity_results else None
    
    # Inverse summary
    best_inverse = inverse_results[0] if inverse_results else None
    
    results = {
        "n_templates": n_templates,
        
        # Associativity
        "associativity": {
            "cos_left_mean": round(float(np.mean(assoc_cos_left)), 4) if assoc_cos_left else 0,
            "cos_right_mean": round(float(np.mean(assoc_cos_right)), 4) if assoc_cos_right else 0,
            "js_left_mean": round(float(np.mean(assoc_js_left)), 6) if assoc_js_left else 0,
            "js_right_mean": round(float(np.mean(assoc_js_right)), 6) if assoc_js_right else 0,
            "dominant_concept_distribution": {k: v for k, v in defaultdict(int, 
                [(d, dominant_concepts.count(d)) for d in set(dominant_concepts)]).items()},
        },
        "associativity_details": associativity_results,
        
        # Closure
        "closure": {
            "dominant_family_distribution": dict(dominant_family_count),
            "closure_details": closure_results,
        },
        
        # Identity
        "identity": {
            "best_identity_candidate": best_identity,
            "identity_details": identity_results,
        },
        
        # Inverse
        "inverse": {
            "best_inverse_pair": best_inverse,
            "inverse_details": inverse_results,
        },
    }
    
    print(f"\n  === Operator Closure Results ===")
    print(f"  Associativity: cos_left={results['associativity']['cos_left_mean']:.4f}, "
          f"cos_right={results['associativity']['cos_right_mean']:.4f}")
    print(f"  Dominant concept in ABC: {results['associativity']['dominant_concept_distribution']}")
    print(f"  Closure dominant families: {results['closure']['dominant_family_distribution']}")
    print(f"  Best identity candidate: {best_identity}")
    print(f"  Best inverse pair: {best_inverse}")
    
    return results


# ===== MAIN =====

def main():
    import torch
    
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    print(f"\n{'='*60}")
    print(f"Phase 165: Constraint Transport Theory")
    print(f"Model: {model_name}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")
    
    # Load model
    cfg = MODEL_CONFIGS[model_name]
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    use_8bit = model_name in ("deepseek7b", "glm4") and gpu_mem_gb < 16
    
    print(f"[0] Loading {model_name} (8bit={use_8bit})...")
    model, tokenizer, device = load_model(model_name, use_8bit=use_8bit)
    info = get_model_info(model, model_name)
    print(f"[0] Model: {info.model_class}, L={info.n_layers}, d={info.d_model}, V={info.vocab_size}")
    
    all_results = {
        "model": model_name,
        "model_class": info.model_class,
        "n_layers": info.n_layers,
        "d_model": info.d_model,
        "vocab_size": info.vocab_size,
        "use_8bit": use_8bit,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    
    # ===== Exp 1: Future Path Equivalence =====
    print(f"\n[1] Exp 1: Future Path Equivalence")
    t0 = time.time()
    all_results["exp1_path_equivalence"] = exp1_future_path_equivalence(
        model, tokenizer, device, model_name, n_pairs=200, n_steps=10)
    print(f"  Time: {time.time()-t0:.1f}s")
    
    # ===== Exp 2: Constraint Jacobian =====
    print(f"\n[2] Exp 2: Constraint Jacobian")
    t0 = time.time()
    all_results["exp2_constraint_jacobian"] = exp2_constraint_jacobian(
        model, tokenizer, device, model_name, n_concepts=40, n_templates=27)
    print(f"  Time: {time.time()-t0:.1f}s")
    
    # ===== Exp 3: Concept Fixed Point =====
    print(f"\n[3] Exp 3: Concept Fixed Point")
    t0 = time.time()
    all_results["exp3_concept_fixed_point"] = exp3_concept_fixed_point(
        model, tokenizer, device, model_name, n_concepts=20, n_domains=7)
    print(f"  Time: {time.time()-t0:.1f}s")
    
    # ===== Exp 4: Operator Closure =====
    print(f"\n[4] Exp 4: Operator Closure")
    t0 = time.time()
    all_results["exp4_operator_closure"] = exp4_operator_closure(
        model, tokenizer, device, model_name, n_templates=10)
    print(f"  Time: {time.time()-t0:.1f}s")
    
    # ===== Save Results =====
    os.makedirs("tests/glm5_temp", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = f"tests/glm5_temp/phase165_{model_name}_{timestamp}.json"
    
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=convert, ensure_ascii=False)
    
    print(f"\n[5] Results saved to: {out_path}")
    
    # Release model
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"\nPhase 165 complete for {model_name}!")


if __name__ == "__main__":
    main()
