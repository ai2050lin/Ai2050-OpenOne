"""
Phase 163: Predictive States & Constraint Propagation
======================================================

THE critical theoretical pivot from Phase 162:

User's deepest insight: Language is NOT a "state machine" but a
"constraint propagation system". The true "state" is not h_t (hidden vector)
but the CONSTRAINT SET over futures: P(future | past).

Key shift: From "static information geometry" → "dynamic constraint propagation"

Core experiments:

  Exp 1: Future Equivalence Class Stability
    - Collect many diverse prompts → compute P(next | prompt)
    - Find pairs with similar P(future) but different L2 in hidden space
    - Find pairs with similar L2 but different P(future)
    - KEY QUESTION: Is h_t the true state, or is P(future|h_t)?

  Exp 2: Constraint Propagation Dynamics
    - Autoregressively generate tokens, track P(next) at each step
    - Measure how the "constraint set" evolves: entropy reduction, KL steps
    - KEY QUESTION: Is constraint propagation smooth or discontinuous?
    - Does it show attractor-like convergence?

  Exp 3: Self-conditioning vs Forced-conditioning
    - Generate autoregressively (model's own choices) → track P(future)
    - Then force DIFFERENT tokens at certain positions → track P(future)
    - KEY QUESTION: Does self-conditioning lead to qualitatively different
      constraint sets than external forcing?

  Exp 4: De-biased Predictive Compression
    - Remove mean logit (de-bias) from all logit vectors
    - Re-run Fisher/PCA analysis on de-biased logits
    - KEY QUESTION: What is the TRUE predictive dimension after removing bias?

CRITICAL: Increase data volume significantly for reliable conclusions.

Usage: python tests/glm5/phase163_predictive_states.py <model_name>
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


# ===== DIVERSE PROMPT SET =====
# 6 domains × ~25 prompts each = ~150 prompts

PROMPTS = {
    "narrative": [
        "The old man walked slowly through the quiet village, his steps",
        "She opened the mysterious letter and discovered that",
        "The detective examined the crime scene carefully and noticed",
        "In the distant kingdom, a young princess decided to",
        "The storm raged outside while inside the cabin",
        "After years of wandering, the traveler finally found",
        "The cat jumped onto the windowsill and watched as",
        "The children played in the garden until suddenly",
        "At midnight, the clock stopped and everything became",
        "The fisherman cast his net into the dark waters and",
        "The ancient tree in the center of the forest had",
        "She ran through the rain-soaked streets, searching for",
        "The musician played a haunting melody that made everyone",
        "In the small coastal town, the lighthouse keeper noticed",
        "The dragon descended from the mountain and",
        "A stranger appeared at the door carrying a package that",
        "The last train of the night arrived at the station and",
        "The artist mixed colors on the palette until she achieved",
        "High above the city, the eagle soared and",
        "The forgotten diary contained secrets about",
        "The bridge collapsed just as the carriage was",
        "Deep in the ocean, the submarine detected an unusual",
        "The flowers in the garden bloomed earlier than usual because",
        "The clockmaker repaired the antique timepiece and discovered",
    ],
    "technical": [
        "The algorithm processes the input data by first computing",
        "In quantum mechanics, the wave function collapses when",
        "The neural network learns to minimize the loss function by",
        "The chemical reaction produces a catalyst that",
        "The software architecture uses a microservices pattern where",
        "The mathematical proof relies on the assumption that",
        "The database query optimization reduces latency by",
        "The encryption algorithm transforms plaintext into ciphertext using",
        "The electrical circuit consists of resistors connected in",
        "The operating system schedules processes using a",
        "The compiler optimizes the code by eliminating",
        "The particle accelerator increases energy by",
        "The data structure allows efficient insertion through",
        "The blockchain validates transactions by requiring",
        "The genetic algorithm evolves solutions by",
        "The machine learning model achieves state-of-the-art results by",
        "The physical simulation approximates fluid dynamics using",
        "The sensor array detects electromagnetic signals from",
        "The network protocol ensures reliable delivery by implementing",
        "The mathematical series converges when the parameter satisfies",
        "The robotics system perceives its environment through",
        "The semiconductor device operates by controlling the flow of",
        "The statistical analysis reveals a significant correlation between",
        "The theorem states that for any continuous function on a closed interval",
    ],
    "dialogue": [
        '"I think we should reconsider the plan," she said, because',
        '"Have you ever wondered why," he asked, looking at',
        '"That is not what I meant," the professor replied, explaining that',
        '"We need to leave now," the captain shouted, warning that',
        '"I disagree completely," said the scientist, pointing out that',
        '"The answer is obvious," she smiled, and then added that',
        '"Please listen carefully," the teacher said, because',
        '"I have been thinking about this for a long time," he confessed, revealing',
        '"What if we are wrong?" she whispered, afraid that',
        '"The data clearly shows," the analyst stated, indicating that',
        '"Let me explain my reasoning," the philosopher began by arguing that',
        '"This changes everything," the researcher realized when she found',
        '"I do not trust that source," the journalist said, noting that',
        '"We must act quickly," the doctor insisted, explaining that',
        '"That is a fascinating observation," the colleague remarked, adding that',
        '"Can you prove this claim?" the skeptic challenged, demanding',
        '"The situation is more complex than it appears," the diplomat warned, because',
        '"I remember something similar," the old man recalled, describing how',
        '"The results are consistent with our hypothesis," the researcher concluded, showing that',
        '"There must be another explanation," the detective reasoned, suspecting that',
        '"Let us consider the alternative," she proposed, suggesting that',
        '"I never expected this outcome," the engineer admitted, acknowledging that',
        '"The timing could not be worse," the manager complained, because',
        '"We are running out of options," the leader declared, stating that',
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
        "The tropical beach stretched for miles with palm trees that",
        "The narrow alley between the buildings was dark and",
        "The concert hall had excellent acoustics that made",
        "The vintage automobile was carefully restored with",
        "The dense fog made it impossible to see beyond",
        "The vast desert stretched endlessly under the scorching sun, where",
        "The abandoned factory had broken windows and rusted machinery that",
        "The crystal chandelier hung from the ceiling, casting light that",
        "The cobblestone streets of the old town were lined with",
        "The enormous oak tree had branches that spread over",
        "The crystal-clear stream wound through the meadow, creating",
        "The modern skyscraper reflected the clouds in its glass facade, while",
        "The tiny cottage had a thatched roof and a garden filled with",
        "The volcanic island emerged from the sea with",
        "The frozen lake crackled underfoot as the temperature",
        "The underground cave contained stalactites that had formed over",
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
        "The existence of abstract objects raises the question of whether",
        "Moral responsibility presupposes that agents have the ability to",
        "The hard problem of consciousness asks why physical processes",
        "Reality may be fundamentally different from our perception because",
        "The relationship between language and thought suggests that",
        "Justice requires fairness in distribution of resources according to",
        "The limits of knowledge become apparent when we try to",
        "Art creates meaning through form and content by",
        "Time seems to flow in one direction even though the laws of physics",
        "The self might be an illusion constructed by",
        "Beauty may be objective or subjective depending on whether",
        "The existence of other minds cannot be directly verified, so we must",
        "Rationality requires that our beliefs be consistent with",
        "The problem of evil challenges the idea that the world is governed by",
        "Truth and meaning are connected through the way language",
        "The foundations of mathematics rest on axioms that are",
    ],
    "factual": [
        "The capital of Australia is Canberra, which was chosen because",
        "Photosynthesis converts sunlight into chemical energy by",
        "The speed of light in vacuum is approximately 299,792 km/s, which means",
        "The human genome contains approximately 3 billion base pairs that",
        "The Great Wall of China was built over many centuries to",
        "Water boils at 100 degrees Celsius at standard atmospheric pressure because",
        "The Roman Empire reached its greatest extent under Emperor Trajan, who",
        "DNA replication occurs during the S phase of the cell cycle when",
        "The Amazon Rainforest produces approximately 20 percent of the world's oxygen by",
        "The periodic table organizes elements by their atomic number, which represents",
        "Gravity causes objects to accelerate at approximately 9.8 meters per second squared, meaning",
        "The French Revolution began in 1789 when the people of Paris",
        "Einstein's theory of relativity predicts that time passes more slowly for",
        "The Industrial Revolution transformed manufacturing by introducing",
        "The internet was originally developed as a military communication network called",
        "Plate tectonics explains earthquakes and volcanoes through the mechanism of",
        "The immune system protects the body from pathogens by producing",
        "The Renaissance began in Italy during the 14th century as a revival of",
        "Black holes are regions of spacetime where gravity is so strong that",
        "The nitrogen cycle converts atmospheric nitrogen into forms that plants can",
        "The Declaration of Independence was signed in 1776, asserting that",
        "Electromagnetic waves travel at the speed of light and include",
        "The Pyramids of Giza were built as tombs for the pharaohs, who believed",
        "The greenhouse effect warms the Earth's surface by trapping",
    ],
}


def get_device_for_input(model):
    """Get the device for input tensors"""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_softmax(logits_np):
    """Numerically stable softmax"""
    logits_clean = np.nan_to_num(logits_np, nan=0.0, posinf=1e4, neginf=-1e4)
    logits_max = np.max(logits_clean)
    exp_logits = np.exp(logits_clean - logits_max)
    probs = exp_logits / np.sum(exp_logits)
    # Check for NaN
    if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
        # Fallback: uniform
        probs = np.ones(len(logits_clean)) / len(logits_clean)
    return probs


def kl_divergence(p, q, eps=1e-10):
    """KL(p || q) with numerical stability"""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def js_divergence(p, q, eps=1e-10):
    """JS(p, q) — symmetric version of KL"""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def entropy(p, eps=1e-10):
    """Shannon entropy of a distribution"""
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def top_k_accuracy(p1, p2, k=5):
    """How many of top-k in p1 are also in top-k of p2?"""
    top1 = set(np.argsort(p1)[-k:])
    top2 = set(np.argsort(p2)[-k:])
    return len(top1 & top2) / k


# ===== DATA COLLECTION =====

def collect_predictive_data(model, tokenizer, device, model_name, max_prompts=150):
    """Collect P(next|context) for many diverse prompts"""
    import torch
    
    info = get_model_info(model, model_name)
    input_device = get_device_for_input(model)
    
    # Get model components for logit computation
    final_norm = model.model.norm if hasattr(model.model, 'norm') else None
    lm_head = model.lm_head
    
    # Collect all prompts
    all_prompts = []
    all_domains = []
    for domain, prompts in PROMPTS.items():
        for p in prompts:
            all_prompts.append(p)
            all_domains.append(domain)
    
    # Limit
    if len(all_prompts) > max_prompts:
        indices = np.random.choice(len(all_prompts), max_prompts, replace=False)
        all_prompts = [all_prompts[i] for i in indices]
        all_domains = [all_domains[i] for i in indices]
    
    results = []
    
    for idx, (prompt, domain) in enumerate(zip(all_prompts, all_domains)):
        if idx % 20 == 0:
            print(f"  Collecting prompt {idx}/{len(all_prompts)}...")
        
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)
            
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True)
            
            hs = out.hidden_states  # tuple of (n_layers+1,) each [1, seq_len, d_model]
            logits_all_pos = out.logits  # [1, seq_len, vocab]
            
            # Process each position
            for pos in range(1, min(hs[-1].shape[1], 40)):
                try:
                    h = hs[-1][0, pos, :].detach().float().cpu().numpy()
                    logits_vec = logits_all_pos[0, pos, :].float().cpu().numpy()
                    
                    # Skip if NaN/Inf
                    if np.any(np.isnan(logits_vec)) or np.any(np.isinf(logits_vec)):
                        continue
                    if np.any(np.isnan(h)):
                        continue
                    
                    probs = safe_softmax(logits_vec)
                    
                    # Compute key statistics
                    ent = entropy(probs)
                    top1_id = int(np.argmax(probs))
                    top1_prob = float(probs[top1_id])
                    top5_ids = np.argsort(probs)[-5:][::-1]
                    top5_probs = [float(probs[i]) for i in top5_ids]
                    
                    # Concentration: probability mass in top-k
                    top10_mass = float(np.sum(np.sort(probs)[-10:]))
                    top50_mass = float(np.sum(np.sort(probs)[-50:]))
                    
                    results.append({
                        "prompt_idx": idx,
                        "domain": domain,
                        "pos": pos,
                        "hidden": h,
                        "logits": logits_vec,
                        "probs": probs,
                        "entropy": ent,
                        "top1_id": top1_id,
                        "top1_prob": top1_prob,
                        "top5_ids": top5_ids.tolist(),
                        "top5_probs": top5_probs,
                        "top10_mass": top10_mass,
                        "top50_mass": top50_mass,
                    })
                except Exception:
                    continue
        
        except Exception:
            continue
    
    return results


# ===== EXP 1: FUTURE EQUIVALENCE CLASS STABILITY =====

def exp1_future_equivalence(data):
    """
    Core test: Is h_t the true state, or is P(future|h_t)?
    
    If P(future) is the true state, then:
    - Pairs with similar P(future) should be "same state"
    - But they might be far apart in L2(hidden space)
    
    We test:
    1. Pairs with small KL but large L2 → "predictive equivalence without geometric equivalence"
    2. Pairs with small L2 but large KL → "geometric equivalence without predictive equivalence"
    """
    print("\n" + "="*60)
    print("EXP 1: Future Equivalence Class Stability")
    print("="*60)
    
    n = len(data)
    if n < 50:
        print(f"  WARNING: Only {n} data points, results may be unreliable")
    
    # Sample pairs (up to 5000 for efficiency)
    np.random.seed(42)
    max_pairs = min(5000, n * (n - 1) // 2)
    
    # Generate random pairs
    pairs_i = np.random.randint(0, n, size=max_pairs * 2)
    pairs_j = np.random.randint(0, n, size=max_pairs * 2)
    
    # Filter: i != j
    valid = pairs_i != pairs_j
    pairs_i = pairs_i[valid][:max_pairs]
    pairs_j = pairs_j[valid][:max_pairs]
    
    kl_vals = []
    l2_hidden_vals = []
    l2_logits_vals = []
    same_domain = []
    top1_match = []
    top5_overlap = []
    
    for idx in range(len(pairs_i)):
        i, j = pairs_i[idx], pairs_j[idx]
        di, dj = data[i], data[j]
        
        kl = kl_divergence(di["probs"], dj["probs"])
        l2_h = float(np.linalg.norm(di["hidden"] - dj["hidden"]))
        l2_l = float(np.linalg.norm(di["logits"] - dj["logits"]))
        
        kl_vals.append(kl)
        l2_hidden_vals.append(l2_h)
        l2_logits_vals.append(l2_l)
        same_domain.append(1 if di["domain"] == dj["domain"] else 0)
        top1_match.append(1 if di["top1_id"] == dj["top1_id"] else 0)
        top5_overlap.append(top_k_accuracy(di["probs"], dj["probs"], k=5))
    
    kl_vals = np.array(kl_vals)
    l2_hidden_vals = np.array(l2_hidden_vals)
    l2_logits_vals = np.array(l2_logits_vals)
    same_domain = np.array(same_domain)
    top1_match = np.array(top1_match)
    top5_overlap = np.array(top5_overlap)
    
    # Overall correlations
    corr_kl_l2h = float(np.corrcoef(kl_vals, l2_hidden_vals)[0, 1]) if len(kl_vals) > 2 else 0
    corr_kl_l2l = float(np.corrcoef(kl_vals, l2_logits_vals)[0, 1]) if len(kl_vals) > 2 else 0
    
    # KEY ANALYSIS: Predictive equivalence vs geometric equivalence
    
    # Thresholds: use percentiles
    kl_low = np.percentile(kl_vals, 10)  # bottom 10% = very similar P(future)
    kl_high = np.percentile(kl_vals, 90)  # top 10% = very different P(future)
    l2h_low = np.percentile(l2_hidden_vals, 10)
    l2h_high = np.percentile(l2_hidden_vals, 90)
    
    # CASE 1: Small KL but Large L2 (predictive equivalence, geometric divergence)
    pred_equiv_geo_div = np.sum((kl_vals < kl_low) & (l2_hidden_vals > np.percentile(l2_hidden_vals, 50)))
    pred_equiv_total = np.sum(kl_vals < kl_low)
    
    # CASE 2: Small L2 but Large KL (geometric equivalence, predictive divergence)
    geo_equiv_pred_div = np.sum((l2_hidden_vals < l2h_low) & (kl_vals > np.percentile(kl_vals, 50)))
    geo_equiv_total = np.sum(l2_hidden_vals < l2h_low)
    
    # CASE 3: Both small (true equivalence)
    both_equiv = np.sum((kl_vals < kl_low) & (l2_hidden_vals < l2h_low))
    
    # CASE 4: Both large (true divergence)
    both_div = np.sum((kl_vals > kl_high) & (l2_hidden_vals > l2h_high))
    
    # Conditional statistics
    # Given small KL, what is the distribution of L2?
    small_kl_mask = kl_vals < kl_low
    large_kl_mask = kl_vals > kl_high
    small_l2_mask = l2_hidden_vals < l2h_low
    large_l2_mask = l2_hidden_vals > l2h_high
    
    l2_given_small_kl = l2_hidden_vals[small_kl_mask] if np.any(small_kl_mask) else np.array([0])
    l2_given_large_kl = l2_hidden_vals[large_kl_mask] if np.any(large_kl_mask) else np.array([0])
    kl_given_small_l2 = kl_vals[small_l2_mask] if np.any(small_l2_mask) else np.array([0])
    kl_given_large_l2 = kl_vals[large_l2_mask] if np.any(large_l2_mask) else np.array([0])
    
    # Top-1 agreement rate by KL bucket
    n_buckets = 5
    kl_percentiles = [np.percentile(kl_vals, p) for p in np.linspace(0, 100, n_buckets + 1)]
    top1_by_kl = []
    for b in range(n_buckets):
        mask = (kl_vals >= kl_percentiles[b]) & (kl_vals < kl_percentiles[b + 1])
        if np.any(mask):
            top1_by_kl.append(float(np.mean(top1_match[mask])))
        else:
            top1_by_kl.append(0.0)
    
    # Top-5 overlap by KL bucket
    top5_by_kl = []
    for b in range(n_buckets):
        mask = (kl_vals >= kl_percentiles[b]) & (kl_vals < kl_percentiles[b + 1])
        if np.any(mask):
            top5_by_kl.append(float(np.mean(top5_overlap[mask])))
        else:
            top5_by_kl.append(0.0)
    
    # Same-domain analysis
    if np.any(same_domain == 1) and np.any(same_domain == 0):
        kl_same = kl_vals[same_domain == 1]
        kl_diff = kl_vals[same_domain == 0]
        l2_same = l2_hidden_vals[same_domain == 1]
        l2_diff = l2_hidden_vals[same_domain == 0]
    else:
        kl_same = kl_diff = l2_same = l2_diff = np.array([0])
    
    # Entropy analysis: pairs with similar entropy but different predictions
    ent_vals = np.array([d["entropy"] for d in data])
    ent_diff = np.abs(ent_vals[pairs_i] - ent_vals[pairs_j])
    corr_entdiff_kl = float(np.corrcoef(ent_diff, kl_vals)[0, 1]) if len(kl_vals) > 2 else 0
    
    results = {
        "n_pairs": int(len(kl_vals)),
        "corr_kl_l2hidden": round(corr_kl_l2h, 4),
        "corr_kl_l2logits": round(corr_kl_l2l, 4),
        "corr_entdiff_kl": round(corr_entdiff_kl, 4),
        
        # Key: Predictive equivalence vs geometric equivalence
        "kl_low_threshold": round(float(kl_low), 4),
        "l2h_low_threshold": round(float(l2h_low), 4),
        "pred_equiv_geo_div_rate": round(float(pred_equiv_geo_div / max(pred_equiv_total, 1)), 4),
        "geo_equiv_pred_div_rate": round(float(geo_equiv_pred_div / max(geo_equiv_total, 1)), 4),
        "both_equiv_count": int(both_equiv),
        "both_div_count": int(both_div),
        
        # Conditional distributions
        "l2_mean_given_small_kl": round(float(np.mean(l2_given_small_kl)), 4),
        "l2_mean_given_large_kl": round(float(np.mean(l2_given_large_kl)), 4),
        "l2_ratio_largeKL_vs_smallKL": round(float(np.mean(l2_given_large_kl) / max(np.mean(l2_given_small_kl), 1e-6)), 2),
        "kl_mean_given_small_l2": round(float(np.mean(kl_given_small_l2)), 4),
        "kl_mean_given_large_l2": round(float(np.mean(kl_given_large_l2)), 4),
        "kl_ratio_largeL2_vs_smallL2": round(float(np.mean(kl_given_large_l2) / max(np.mean(kl_given_small_l2), 1e-6)), 2),
        
        # Top-1 agreement by KL bucket
        "top1_agreement_by_kl_bucket": [round(x, 4) for x in top1_by_kl],
        "top5_overlap_by_kl_bucket": [round(x, 4) for x in top5_by_kl],
        
        # Same-domain vs cross-domain
        "kl_mean_same_domain": round(float(np.mean(kl_same)), 4),
        "kl_mean_cross_domain": round(float(np.mean(kl_diff)), 4),
        "l2_mean_same_domain": round(float(np.mean(l2_same)), 4),
        "l2_mean_cross_domain": round(float(np.mean(l2_diff)), 4),
        
        # Overall statistics
        "kl_mean": round(float(np.mean(kl_vals)), 4),
        "kl_median": round(float(np.median(kl_vals)), 4),
        "l2h_mean": round(float(np.mean(l2_hidden_vals)), 4),
        "l2h_median": round(float(np.median(l2_hidden_vals)), 4),
    }
    
    print(f"  N pairs: {results['n_pairs']}")
    print(f"  corr(KL, L2_hidden): {results['corr_kl_l2hidden']}")
    print(f"  corr(KL, L2_logits): {results['corr_kl_l2logits']}")
    print(f"  corr(|Δentropy|, KL): {results['corr_entdiff_kl']}")
    print(f"  ---")
    print(f"  Predictive-equiv-geo-div rate: {results['pred_equiv_geo_div_rate']} "
          f"({pred_equiv_geo_div}/{pred_equiv_total})")
    print(f"  Geo-equiv-pred-div rate: {results['geo_equiv_pred_div_rate']} "
          f"({geo_equiv_pred_div}/{geo_equiv_total})")
    print(f"  L2 given small KL: {results['l2_mean_given_small_kl']} vs "
          f"L2 given large KL: {results['l2_mean_given_large_kl']} "
          f"(ratio: {results['l2_ratio_largeKL_vs_smallKL']})")
    print(f"  KL given small L2: {results['kl_mean_given_small_l2']} vs "
          f"KL given large L2: {results['kl_mean_given_large_l2']} "
          f"(ratio: {results['kl_ratio_largeL2_vs_smallL2']})")
    print(f"  Top-1 agreement by KL: {results['top1_agreement_by_kl_bucket']}")
    print(f"  Top-5 overlap by KL: {results['top5_overlap_by_kl_bucket']}")
    print(f"  Same-domain KL: {results['kl_mean_same_domain']} vs "
          f"Cross-domain KL: {results['kl_mean_cross_domain']}")
    
    return results


# ===== EXP 2: CONSTRAINT PROPAGATION DYNAMICS =====

def exp2_constraint_propagation(model, tokenizer, device, model_name, n_prompts=80, n_steps=15):
    """
    Track how P(next) evolves as tokens are added autoregressively.
    
    Key questions:
    1. Does the constraint set monotonically narrow? (entropy decrease)
    2. Is constraint propagation smooth or discontinuous? (KL jumps)
    3. Are there "constraint collapse" events? (sudden entropy drop)
    """
    import torch
    
    print("\n" + "="*60)
    print("EXP 2: Constraint Propagation Dynamics")
    print("="*60)
    
    input_device = get_device_for_input(model)
    
    # Collect seed prompts from different domains
    seed_prompts = []
    seed_domains = []
    for domain, prompts in PROMPTS.items():
        for p in prompts[:14]:  # ~14 per domain = 84 total
            seed_prompts.append(p)
            seed_domains.append(domain)
    
    if len(seed_prompts) > n_prompts:
        indices = np.random.choice(len(seed_prompts), n_prompts, replace=False)
        seed_prompts = [seed_prompts[i] for i in indices]
        seed_domains = [seed_domains[i] for i in indices]
    
    trajectories = []  # List of {prompt_idx, domain, steps: [{pos, entropy, top1_prob, kl_from_prev, ...}]}
    
    for pidx, (prompt, domain) in enumerate(zip(seed_prompts, seed_domains)):
        if pidx % 15 == 0:
            print(f"  Processing prompt {pidx}/{len(seed_prompts)}...")
        
        try:
            # Initial context
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)
            
            traj = {"prompt_idx": pidx, "domain": domain, "steps": []}
            
            prev_probs = None
            
            for step in range(n_steps):
                try:
                    with torch.no_grad():
                        out = model(input_ids=input_ids, attention_mask=attention_mask)
                    
                    logits = out.logits[0, -1, :].float().cpu().numpy()
                    
                    if np.any(np.isnan(logits)) or np.any(np.isinf(logits)):
                        break
                    
                    probs = safe_softmax(logits)
                    ent = entropy(probs)
                    top1_id = int(np.argmax(probs))
                    top1_prob = float(probs[top1_id])
                    top5_ids = np.argsort(probs)[-5:][::-1]
                    
                    # KL from previous step
                    kl_from_prev = kl_divergence(prev_probs, probs) if prev_probs is not None else 0.0
                    js_from_prev = js_divergence(prev_probs, probs) if prev_probs is not None else 0.0
                    
                    # Top-1 stability: does top-1 change from previous?
                    top1_changed = 0
                    if prev_probs is not None:
                        prev_top1 = int(np.argmax(prev_probs))
                        top1_changed = 1 if top1_id != prev_top1 else 0
                    
                    traj["steps"].append({
                        "step": step,
                        "seq_len": int(input_ids.shape[1]),
                        "entropy": ent,
                        "top1_prob": top1_prob,
                        "top1_id": top1_id,
                        "kl_from_prev": kl_from_prev,
                        "js_from_prev": js_from_prev,
                        "top1_changed": top1_changed,
                    })
                    
                    prev_probs = probs.copy()
                    
                    # Generate next token (greedy for determinism)
                    next_token = torch.tensor([[top1_id]], device=input_device)
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                    attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
                    
                except Exception:
                    break
            
            if len(traj["steps"]) > 0:
                trajectories.append(traj)
        
        except Exception:
            continue
    
    # Analyze trajectories
    all_entropies = []
    all_kl_steps = []
    all_top1_probs = []
    all_top1_changed = []
    
    entropies_by_step = defaultdict(list)
    kl_by_step = defaultdict(list)
    top1_prob_by_step = defaultdict(list)
    top1_changed_by_step = defaultdict(list)
    
    entropies_by_domain = defaultdict(list)
    kl_by_domain = defaultdict(list)
    
    for traj in trajectories:
        domain = traj["domain"]
        for step_data in traj["steps"]:
            step = step_data["step"]
            all_entropies.append(step_data["entropy"])
            all_kl_steps.append(step_data["kl_from_prev"])
            all_top1_probs.append(step_data["top1_prob"])
            all_top1_changed.append(step_data["top1_changed"])
            
            entropies_by_step[step].append(step_data["entropy"])
            kl_by_step[step].append(step_data["kl_from_prev"])
            top1_prob_by_step[step].append(step_data["top1_prob"])
            top1_changed_by_step[step].append(step_data["top1_changed"])
            
            entropies_by_domain[domain].append(step_data["entropy"])
            kl_by_domain[domain].append(step_data["kl_from_prev"])
    
    # Entropy evolution by step
    entropy_evolution = {}
    for step in sorted(entropies_by_step.keys()):
        vals = entropies_by_step[step]
        entropy_evolution[int(step)] = {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "median": round(float(np.median(vals)), 4),
        }
    
    # KL evolution by step
    kl_evolution = {}
    for step in sorted(kl_by_step.keys()):
        vals = kl_by_step[step]
        kl_evolution[int(step)] = {
            "mean": round(float(np.mean(vals)), 4),
            "median": round(float(np.median(vals)), 4),
            "p90": round(float(np.percentile(vals, 90)), 4),
        }
    
    # Top-1 probability evolution
    top1_prob_evolution = {}
    for step in sorted(top1_prob_by_step.keys()):
        vals = top1_prob_by_step[step]
        top1_prob_evolution[int(step)] = {
            "mean": round(float(np.mean(vals)), 4),
            "median": round(float(np.median(vals)), 4),
        }
    
    # Top-1 change rate by step
    top1_change_rate = {}
    for step in sorted(top1_changed_by_step.keys()):
        vals = top1_changed_by_step[step]
        top1_change_rate[int(step)] = round(float(np.mean(vals)), 4)
    
    # Domain-specific entropy
    domain_entropy = {}
    for domain in sorted(entropies_by_domain.keys()):
        vals = entropies_by_domain[domain]
        domain_entropy[domain] = {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
        }
    
    # Constraint collapse events: entropy drops > threshold
    n_collapses = 0
    n_gradual = 0
    for traj in trajectories:
        for i in range(1, len(traj["steps"])):
            ent_drop = traj["steps"][i-1]["entropy"] - traj["steps"][i]["entropy"]
            if ent_drop > 1.0:  # Large entropy drop
                n_collapses += 1
            elif ent_drop > 0.01:
                n_gradual += 1
    
    results = {
        "n_trajectories": len(trajectories),
        "n_steps_total": len(all_entropies),
        "entropy_evolution": entropy_evolution,
        "kl_evolution": kl_evolution,
        "top1_prob_evolution": top1_prob_evolution,
        "top1_change_rate": top1_change_rate,
        "domain_entropy": domain_entropy,
        "constraint_collapse_events": n_collapses,
        "gradual_narrowing_events": n_gradual,
        "collapse_vs_gradual_ratio": round(n_collapses / max(n_gradual, 1), 4),
        "entropy_mean": round(float(np.mean(all_entropies)), 4) if all_entropies else 0,
        "kl_mean": round(float(np.mean(all_kl_steps)), 4) if all_kl_steps else 0,
        "kl_median": round(float(np.median(all_kl_steps)), 4) if all_kl_steps else 0,
    }
    
    print(f"  N trajectories: {results['n_trajectories']}")
    print(f"  Entropy mean: {results['entropy_mean']}")
    print(f"  KL step mean: {results['kl_mean']}, median: {results['kl_median']}")
    print(f"  Constraint collapses: {n_collapses}, Gradual: {n_gradual}")
    print(f"  Entropy evolution (step 0→14): " + 
          " → ".join([str(entropy_evolution.get(s, {}).get("mean", "?")) 
                       for s in range(min(15, len(entropy_evolution)))]))
    
    return results


# ===== EXP 3: SELF-CONDITIONING VS FORCED-CONDITIONING =====

def exp3_self_vs_forced_conditioning(model, tokenizer, device, model_name, n_prompts=40, n_steps=10):
    """
    Compare how P(future) evolves under:
    a) Self-conditioning: model generates its own tokens
    b) Forced 2nd-choice: use 2nd most likely token instead
    c) Forced random: use a random token from top-20
    
    Key question: Does self-conditioning lead to qualitatively different constraint sets?
    """
    import torch
    
    print("\n" + "="*60)
    print("EXP 3: Self-conditioning vs Forced-conditioning")
    print("="*60)
    
    input_device = get_device_for_input(model)
    
    # Collect seed prompts
    seed_prompts = []
    for domain, prompts in PROMPTS.items():
        for p in prompts[:7]:  # ~7 per domain = 42 total
            seed_prompts.append(p)
    
    if len(seed_prompts) > n_prompts:
        seed_prompts = seed_prompts[:n_prompts]
    
    trajectories = {
        "self": [],      # model's own choices
        "2nd_choice": [],  # 2nd most likely
        "random_top20": [],  # random from top-20
    }
    
    for pidx, prompt in enumerate(seed_prompts):
        if pidx % 10 == 0:
            print(f"  Processing prompt {pidx}/{len(seed_prompts)}...")
        
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            base_input_ids = inputs["input_ids"].to(input_device)
            base_attention_mask = inputs["attention_mask"].to(input_device)
            
            # First pass: self-conditioning (greedy)
            input_ids = base_input_ids.clone()
            attention_mask = base_attention_mask.clone()
            self_traj = []
            prev_probs = None
            
            for step in range(n_steps):
                try:
                    with torch.no_grad():
                        out = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = out.logits[0, -1, :].float().cpu().numpy()
                    if np.any(np.isnan(logits)):
                        break
                    probs = safe_softmax(logits)
                    top5_ids = np.argsort(probs)[-5:][::-1]
                    
                    self_traj.append({
                        "step": step,
                        "entropy": entropy(probs),
                        "top1_prob": float(probs[top5_ids[0]]),
                        "top1_id": int(top5_ids[0]),
                        "top2_id": int(top5_ids[1]) if len(top5_ids) > 1 else 0,
                        "top2_prob": float(probs[top5_ids[1]]) if len(top5_ids) > 1 else 0,
                        "js_from_prev": js_divergence(prev_probs, probs) if prev_probs is not None else 0,
                        "chosen_token": int(top5_ids[0]),  # greedy choice
                    })
                    prev_probs = probs.copy()
                    
                    # Greedy next
                    next_token = torch.tensor([[top5_ids[0]]], device=input_device)
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                    attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
                except Exception:
                    break
            
            trajectories["self"].append(self_traj)
            
            # Second pass: 2nd choice conditioning
            if len(self_traj) > 2:
                input_ids = base_input_ids.clone()
                attention_mask = base_attention_mask.clone()
                
                # Get the 2nd choice token at step 0
                step0_top2_id = self_traj[0]["top2_id"]
                step0_top2_prob = self_traj[0]["top2_prob"]
                
                # Inject 2nd choice at step 0
                next_token = torch.tensor([[step0_top2_id]], device=input_device)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
                
                forced2_traj = [{
                    "step": 0,
                    "entropy": self_traj[0]["entropy"],
                    "top1_prob": step0_top2_prob,  # Now the 2nd choice is "chosen"
                    "chosen_token": step0_top2_id,
                    "js_from_prev": 0,
                    "forced": True,
                }]
                
                prev_probs = None
                
                for step in range(1, n_steps):
                    try:
                        with torch.no_grad():
                            out = model(input_ids=input_ids, attention_mask=attention_mask)
                        logits = out.logits[0, -1, :].float().cpu().numpy()
                        if np.any(np.isnan(logits)):
                            break
                        probs = safe_softmax(logits)
                        top1_id = int(np.argmax(probs))
                        
                        forced2_traj.append({
                            "step": step,
                            "entropy": entropy(probs),
                            "top1_prob": float(probs[top1_id]),
                            "chosen_token": top1_id,
                            "js_from_prev": js_divergence(prev_probs, probs) if prev_probs is not None else 0,
                            "forced": False,
                        })
                        prev_probs = probs.copy()
                        
                        next_token = torch.tensor([[top1_id]], device=input_device)
                        input_ids = torch.cat([input_ids, next_token], dim=1)
                        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
                    except Exception:
                        break
                
                trajectories["2nd_choice"].append(forced2_traj)
            
            # Third pass: random top-20 conditioning
            if len(self_traj) > 2:
                input_ids = base_input_ids.clone()
                attention_mask = base_attention_mask.clone()
                
                # Get top-20 at step 0
                with torch.no_grad():
                    out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits[0, -1, :].float().cpu().numpy()
                if not np.any(np.isnan(logits)):
                    probs = safe_softmax(logits)
                    top20_ids = np.argsort(probs)[-20:][::-1]
                    
                    # Pick a random token from positions 2-20 (not top-1 or top-2)
                    if len(top20_ids) > 2:
                        random_idx = np.random.randint(2, min(20, len(top20_ids)))
                        random_token_id = int(top20_ids[random_idx])
                    else:
                        random_token_id = int(top20_ids[-1])
                    
                    next_token = torch.tensor([[random_token_id]], device=input_device)
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                    attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
                    
                    random_traj = [{
                        "step": 0,
                        "entropy": entropy(probs),
                        "top1_prob": float(probs[random_token_id]),
                        "chosen_token": random_token_id,
                        "js_from_prev": 0,
                        "forced": True,
                    }]
                    
                    prev_probs = None
                    
                    for step in range(1, n_steps):
                        try:
                            with torch.no_grad():
                                out = model(input_ids=input_ids, attention_mask=attention_mask)
                            logits = out.logits[0, -1, :].float().cpu().numpy()
                            if np.any(np.isnan(logits)):
                                break
                            probs = safe_softmax(logits)
                            top1_id = int(np.argmax(probs))
                            
                            random_traj.append({
                                "step": step,
                                "entropy": entropy(probs),
                                "top1_prob": float(probs[top1_id]),
                                "chosen_token": top1_id,
                                "js_from_prev": js_divergence(prev_probs, probs) if prev_probs is not None else 0,
                                "forced": False,
                            })
                            prev_probs = probs.copy()
                            
                            next_token = torch.tensor([[top1_id]], device=input_device)
                            input_ids = torch.cat([input_ids, next_token], dim=1)
                            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
                        except Exception:
                            break
                    
                    trajectories["random_top20"].append(random_traj)
        
        except Exception:
            continue
    
    # Analyze divergence between self and forced trajectories
    def analyze_trajectory_set(trajs, label):
        all_ent = []
        all_js = []
        ent_by_step = defaultdict(list)
        js_by_step = defaultdict(list)
        
        for traj in trajs:
            for step_data in traj:
                s = step_data["step"]
                all_ent.append(step_data["entropy"])
                ent_by_step[s].append(step_data["entropy"])
                if "js_from_prev" in step_data:
                    all_js.append(step_data["js_from_prev"])
                    js_by_step[s].append(step_data["js_from_prev"])
        
        result = {
            "n_trajectories": len(trajs),
            "entropy_mean": round(float(np.mean(all_ent)), 4) if all_ent else 0,
            "js_mean": round(float(np.mean(all_js)), 4) if all_js else 0,
            "entropy_by_step": {s: round(float(np.mean(v)), 4) for s, v in sorted(ent_by_step.items())},
        }
        return result
    
    self_analysis = analyze_trajectory_set(trajectories["self"], "self")
    forced2_analysis = analyze_trajectory_set(trajectories["2nd_choice"], "2nd_choice")
    random_analysis = analyze_trajectory_set(trajectories["random_top20"], "random_top20")
    
    # Compute cross-trajectory JS divergence (self vs forced) at each step
    cross_js_self_vs_2nd = defaultdict(list)
    cross_js_self_vs_random = defaultdict(list)
    
    min_len_2nd = min(len(trajectories["self"]), len(trajectories["2nd_choice"]))
    for i in range(min_len_2nd):
        self_traj = trajectories["self"][i]
        forced_traj = trajectories["2nd_choice"][i]
        for s in range(min(len(self_traj), len(forced_traj))):
            # Compare entropy at each step
            ent_self = self_traj[s]["entropy"]
            ent_forced = forced_traj[s]["entropy"]
            cross_js_self_vs_2nd[s].append(abs(ent_self - ent_forced))
    
    min_len_random = min(len(trajectories["self"]), len(trajectories["random_top20"]))
    for i in range(min_len_random):
        self_traj = trajectories["self"][i]
        random_traj = trajectories["random_top20"][i]
        for s in range(min(len(self_traj), len(random_traj))):
            ent_self = self_traj[s]["entropy"]
            ent_random = random_traj[s]["entropy"]
            cross_js_self_vs_random[s].append(abs(ent_self - ent_random))
    
    # Entropy divergence by step
    ent_div_self_vs_2nd = {s: round(float(np.mean(v)), 4) for s, v in sorted(cross_js_self_vs_2nd.items())}
    ent_div_self_vs_random = {s: round(float(np.mean(v)), 4) for s, v in sorted(cross_js_self_vs_random.items())}
    
    # Cumulative divergence
    cum_div_2nd = [0]
    cum_div_random = [0]
    for s in sorted(ent_div_self_vs_2nd.keys()):
        cum_div_2nd.append(cum_div_2nd[-1] + ent_div_self_vs_2nd[s])
    for s in sorted(ent_div_self_vs_random.keys()):
        cum_div_random.append(cum_div_random[-1] + ent_div_self_vs_random[s])
    
    results = {
        "self_conditioning": self_analysis,
        "2nd_choice_conditioning": forced2_analysis,
        "random_top20_conditioning": random_analysis,
        "entropy_divergence_self_vs_2nd_by_step": ent_div_self_vs_2nd,
        "entropy_divergence_self_vs_random_by_step": ent_div_self_vs_random,
        "cumulative_divergence_2nd": [round(x, 4) for x in cum_div_2nd],
        "cumulative_divergence_random": [round(x, 4) for x in cum_div_random],
    }
    
    print(f"  Self-conditioning: {self_analysis['n_trajectories']} trajectories, "
          f"entropy_mean={self_analysis['entropy_mean']}")
    print(f"  2nd-choice: {forced2_analysis['n_trajectories']} trajectories, "
          f"entropy_mean={forced2_analysis['entropy_mean']}")
    print(f"  Random-top20: {random_analysis['n_trajectories']} trajectories, "
          f"entropy_mean={random_analysis['entropy_mean']}")
    print(f"  Entropy divergence (self vs 2nd): {ent_div_self_vs_2nd}")
    print(f"  Entropy divergence (self vs random): {ent_div_self_vs_random}")
    
    return results


# ===== EXP 4: DE-BIASED PREDICTIVE COMPRESSION =====

def exp4_debiased_predictive_compression(data):
    """
    Remove mean logit (de-bias) and re-analyze.
    
    Phase 162 showed PC1 accounts for 87-97% of logit variance.
    This is likely a global bias direction. After removing it,
    what is the TRUE predictive dimension?
    """
    print("\n" + "="*60)
    print("EXP 4: De-biased Predictive Compression")
    print("="*60)
    
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    
    logits = np.nan_to_num(np.array([d["logits"] for d in data]), nan=0.0, posinf=0.0, neginf=0.0)
    hidden = np.nan_to_num(np.array([d["hidden"] for d in data]), nan=0.0, posinf=0.0, neginf=0.0)
    probs = np.array([d["probs"] for d in data])
    
    n_sample = len(logits)
    if n_sample < 100:
        print(f"  WARNING: Only {n_sample} samples, results limited")
    
    # Step 1: Original logit PCA
    n_pca = min(100, n_sample - 1, logits.shape[1])
    pca_orig = PCA(n_components=n_pca)
    pca_orig.fit(logits)
    orig_var_explained = np.cumsum(pca_orig.explained_variance_ratio_)
    
    # Step 2: De-biased logit PCA (subtract mean)
    mean_logit = np.mean(logits, axis=0)
    logits_debiased = logits - mean_logit
    
    pca_debiased = PCA(n_components=n_pca)
    pca_debiased.fit(logits_debiased)
    debiased_var_explained = np.cumsum(pca_debiased.explained_variance_ratio_)
    
    # Step 3: Logit PCA after softmax (probability-centered)
    # Subtract mean log-probability (log of geometric mean)
    log_probs = np.log(np.clip(probs, 1e-10, 1.0))
    mean_log_prob = np.mean(log_probs, axis=0)
    log_probs_centered = log_probs - mean_log_prob
    
    pca_logprob = PCA(n_components=min(n_pca, log_probs_centered.shape[1]))
    pca_logprob.fit(log_probs_centered)
    logprob_var_explained = np.cumsum(pca_logprob.explained_variance_ratio_)
    
    # Step 4: Fisher Information on de-biased logits
    # Use top-50 variance dimensions of de-biased logits
    logit_var_debiased = np.var(logits_debiased, axis=0)
    top_k = min(50, n_sample - 1)
    top_dims = np.argsort(logit_var_debiased)[-top_k:]
    
    L_reduced = logits_debiased[:, top_dims]
    
    # Compute Fisher matrix at sampled points
    fisher_ranks = []
    fisher_top5_concentration = []
    
    n_fisher_sample = min(200, n_sample)
    fisher_indices = np.random.choice(n_sample, n_fisher_sample, replace=False)
    
    for idx in fisher_indices:
        p = probs[idx]
        p_top = p[top_dims]
        p_top = np.clip(p_top, 1e-10, 1.0)
        p_top = p_top / p_top.sum()
        
        if np.any(np.isnan(p_top)):
            continue
        
        G = np.diag(p_top) - np.outer(p_top, p_top)
        G = 0.5 * (G + G.T)
        reg = max(np.trace(np.abs(G)) * 1e-10, 1e-14)
        G += np.eye(len(G)) * reg
        
        try:
            eigvals = np.linalg.eigvalsh(G)
            eigvals = np.sort(np.real(eigvals))[::-1]
            if eigvals[0] > 0:
                cumvar = np.cumsum(eigvals) / np.sum(eigvals)
                rank95 = int(np.searchsorted(cumvar, 0.95)) + 1
                fisher_ranks.append(rank95)
                fisher_top5_concentration.append(float(np.sum(eigvals[:5]) / np.sum(eigvals)))
        except Exception:
            continue
    
    # Step 5: Predictive compression — how many de-biased logit dimensions suffice?
    # Use next-step prediction as target: predict logits[i+1] from hidden[i]
    # Only for consecutive positions in same prompt
    X_hidden = []
    Y_logits_debiased = []
    
    for i in range(len(data) - 1):
        if data[i]["prompt_idx"] == data[i+1]["prompt_idx"]:
            if data[i]["pos"] + 1 == data[i+1]["pos"]:
                X_hidden.append(hidden[i])
                Y_logits_debiased.append(logits_debiased[i+1])
    
    if len(X_hidden) < 50:
        # Fallback: use all pairs
        X_hidden = hidden[:min(200, n_sample)]
        Y_logits_debiased = logits_debiased[:min(200, n_sample)]
    
    X_hidden = np.array(X_hidden)
    Y_logits_debiased = np.array(Y_logits_debiased)
    
    # Compress hidden states, then predict de-biased logits
    compression_results = {}
    for d in [2, 5, 10, 20, 30, 50]:
        pca_h = PCA(n_components=min(d, X_hidden.shape[1]))
        Z = pca_h.fit_transform(X_hidden)
        
        # Ridge regression: Z → Y
        kf = KFold(n_splits=min(3, len(X_hidden) // 10), shuffle=True, random_state=42)
        r2_scores = []
        
        for train_idx, test_idx in kf.split(Z):
            try:
                ridge = Ridge(alpha=1.0)
                ridge.fit(Z[train_idx], Y_logits_debiased[train_idx])
                r2 = ridge.score(Z[test_idx], Y_logits_debiased[test_idx])
                r2_scores.append(max(r2, 0))
            except Exception:
                continue
        
        if r2_scores:
            compression_results[d] = round(float(np.mean(r2_scores)), 4)
    
    # Summary
    dims_for_90_orig = int(np.searchsorted(orig_var_explained, 0.90)) + 1
    dims_for_90_debiased = int(np.searchsorted(debiased_var_explained, 0.90)) + 1
    dims_for_95_orig = int(np.searchsorted(orig_var_explained, 0.95)) + 1
    dims_for_95_debiased = int(np.searchsorted(debiased_var_explained, 0.95)) + 1
    
    results = {
        "n_samples": n_sample,
        
        # Dimension comparison
        "orig_dims_for_90": dims_for_90_orig,
        "orig_dims_for_95": dims_for_95_orig,
        "debiased_dims_for_90": dims_for_90_debiased,
        "debiased_dims_for_95": dims_for_95_debiased,
        "logprob_dims_for_90": int(np.searchsorted(logprob_var_explained, 0.90)) + 1,
        "logprob_dims_for_95": int(np.searchsorted(logprob_var_explained, 0.95)) + 1,
        
        # Variance explained at key dimensions
        "orig_var_at_1": round(float(orig_var_explained[0]), 4),
        "orig_var_at_5": round(float(orig_var_explained[min(4, len(orig_var_explained)-1)]), 4),
        "debiased_var_at_1": round(float(debiased_var_explained[0]), 4),
        "debiased_var_at_5": round(float(debiased_var_explained[min(4, len(debiased_var_explained)-1)]), 4),
        "logprob_var_at_1": round(float(logprob_var_explained[0]), 4),
        "logprob_var_at_5": round(float(logprob_var_explained[min(4, len(logprob_var_explained)-1)]), 4),
        
        # Fisher analysis on de-biased logits
        "fisher_rank95_mean": round(float(np.mean(fisher_ranks)), 2) if fisher_ranks else 0,
        "fisher_rank95_median": round(float(np.median(fisher_ranks)), 2) if fisher_ranks else 0,
        "fisher_top5_concentration_mean": round(float(np.mean(fisher_top5_concentration)), 4) if fisher_top5_concentration else 0,
        
        # Predictive compression
        "hidden_to_debiased_logit_r2": compression_results,
    }
    
    print(f"  Original: PC1={results['orig_var_at_1']}, dim90={dims_for_90_orig}, dim95={dims_for_95_orig}")
    print(f"  Debiased: PC1={results['debiased_var_at_1']}, dim90={dims_for_90_debiased}, dim95={dims_for_95_debiased}")
    print(f"  Log-prob: PC1={results['logprob_var_at_1']}, dim90={results['logprob_dims_for_90']}")
    print(f"  Fisher rank95 (debiased): mean={results['fisher_rank95_mean']}, "
          f"median={results['fisher_rank95_median']}")
    print(f"  Predictive compression R²: {compression_results}")
    
    return results


# ===== MAIN =====

def main():
    import torch
    
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    print(f"\n{'='*60}")
    print(f"Phase 163: Predictive States & Constraint Propagation")
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
    
    # ===== Data Collection =====
    print(f"\n[1] Collecting predictive data...")
    t0 = time.time()
    data = collect_predictive_data(model, tokenizer, device, model_name, max_prompts=150)
    t_collect = time.time() - t0
    print(f"[1] Collected {len(data)} (position, P(next)) pairs in {t_collect:.1f}s")
    
    if len(data) < 50:
        print(f"ERROR: Too few data points ({len(data)}). Aborting.")
        release_model(model)
        return
    
    # ===== Run Experiments =====
    all_results = {
        "model": model_name,
        "model_class": info.model_class,
        "n_layers": info.n_layers,
        "d_model": info.d_model,
        "vocab_size": info.vocab_size,
        "use_8bit": use_8bit,
        "n_data_points": len(data),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    
    # Exp 1: Future equivalence class stability
    print(f"\n[2] Exp 1: Future Equivalence Class Stability")
    t0 = time.time()
    all_results["exp1_future_equivalence"] = exp1_future_equivalence(data)
    print(f"  Time: {time.time()-t0:.1f}s")
    
    # Exp 2: Constraint propagation
    print(f"\n[3] Exp 2: Constraint Propagation Dynamics")
    t0 = time.time()
    all_results["exp2_constraint_propagation"] = exp2_constraint_propagation(
        model, tokenizer, device, model_name, n_prompts=80, n_steps=15)
    print(f"  Time: {time.time()-t0:.1f}s")
    
    # Exp 3: Self vs forced conditioning
    print(f"\n[4] Exp 3: Self-conditioning vs Forced-conditioning")
    t0 = time.time()
    all_results["exp3_self_vs_forced"] = exp3_self_vs_forced_conditioning(
        model, tokenizer, device, model_name, n_prompts=40, n_steps=10)
    print(f"  Time: {time.time()-t0:.1f}s")
    
    # Exp 4: De-biased predictive compression
    print(f"\n[5] Exp 4: De-biased Predictive Compression")
    t0 = time.time()
    all_results["exp4_debiased_compression"] = exp4_debiased_predictive_compression(data)
    print(f"  Time: {time.time()-t0:.1f}s")
    
    # ===== Save Results =====
    os.makedirs("tests/glm5_temp", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = f"tests/glm5_temp/phase163_{model_name}_{timestamp}.json"
    
    # Convert numpy types for JSON serialization
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
    
    print(f"\n[6] Results saved to: {out_path}")
    
    # Release model
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"\nPhase 163 complete for {model_name}!")


if __name__ == "__main__":
    main()
