"""
Phase 164: Concept Predictive Deformation Atlas
================================================

THE critical pivot from Phase 163:

Phase 163 proved: hidden ≠ state, constraint propagation ≠ state evolution,
self-conditioning produces tightest constraints.

Now the KEY question: What IS a "concept" in the constraint propagation system?

User's deepest insight: concept = operator on future distributions
  Not: concept = vector (embedding)
  Not: concept = neuron/feature
  But: concept = stable constraint transformation that:
    1. Is transferable across contexts
    2. Is composable with other concepts
    3. Imposes stable constraints on futures
    4. Is multi-step stable (not just 1-step)

Core experiments:

  Exp A: Concept Predictive Deformation
    - Fix context templates, insert concepts, rollout future 30 tokens
    - Measure ΔP(future) = D(P(future|context), P(future|context+concept))
    - Build Concept Action Tensor: concept → future deformation map
    - KEY QUESTION: Do concepts induce stable deformations across contexts?

  Exp B: Concept Family Emergence
    - Cluster concepts by shared deformation patterns
    - KEY QUESTION: Do "animal", "vehicle", "abstract" families emerge
      from deformation similarity (not embedding similarity)?

  Exp C: Operator Composition Test
    - T_adj ∘ T_noun vs T_{adj+noun}: e.g. T_red ∘ T_apple ≈ T_{red apple}?
    - Test commutativity: T_A ∘ T_B vs T_B ∘ T_A
    - KEY QUESTION: Does language have algebraic structure?

  Exp D: Multi-step Future Equivalence
    - Test if 1-step predictive equivalence implies k-step equivalence
    - KEY QUESTION: Is P(next|c1) ≈ P(next|c2) sufficient for P(future|c1) ≈ P(future|c2)?

Usage: python tests/glm5/phase164_concept_deformation.py <model_name>
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


# ===== CONCEPT SETS =====
# 4 concept families + 1 abstract family, 5 concepts each = 25 concepts

CONCEPT_FAMILIES = {
    "animal": ["cat", "dog", "tiger", "lion", "horse"],
    "vehicle": ["car", "truck", "bus", "train", "bicycle"],
    "color": ["red", "blue", "green", "white", "black"],
    "abstract": ["democracy", "freedom", "justice", "power", "truth"],
}

# Adjective concepts for composition test
ADJ_CONCEPTS = ["red", "big", "fast", "dangerous", "beautiful"]
NOUN_CONCEPTS = ["cat", "car", "city", "mountain", "river"]

# Composition pairs: (adj, noun, composed)
COMPOSITION_PAIRS = [
    ("red", "cat", "red cat"),
    ("big", "car", "big car"),
    ("fast", "train", "fast train"),
    ("dangerous", "mountain", "dangerous mountain"),
    ("beautiful", "city", "beautiful city"),
]


# ===== CONTEXT TEMPLATES =====
# 15 diverse templates with ___ slot for concept insertion

CONTEXT_TEMPLATES = [
    "I saw a ___ in the park yesterday",
    "The ___ was standing near the old building",
    "People often think about the ___ when they consider",
    "She described the ___ as something quite remarkable because",
    "After studying the ___ carefully, the researcher found that",
    "The children were fascinated by the ___ and wanted to learn",
    "In the story, the ___ played an important role by",
    "The scientist explained that the ___ could be understood as",
    "Many believe the ___ represents something fundamental about",
    "When the ___ appeared, everyone was surprised because",
    "The documentary showed how the ___ had changed over time, revealing",
    "Historically, the ___ has been associated with",
    "The artist painted the ___ in a way that captured",
    "According to the report, the ___ was responsible for",
    "The teacher asked the students to describe the ___ and they said",
]


# ===== UTILITY FUNCTIONS =====

def get_device_for_input(model):
    """Get the device for input tensors"""
    try:
        return next(model.parameters()).device
    except StopIteration:
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_softmax(logits_np):
    """Numerically stable softmax"""
    logits_clean = np.nan_to_num(logits_np, nan=0.0, posinf=1e4, neginf=-1e4)
    logits_max = np.max(logits_clean)
    exp_logits = np.exp(logits_clean - logits_max)
    probs = exp_logits / np.sum(exp_logits)
    if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
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
    """JS(p, q) — symmetric"""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def entropy(p, eps=1e-10):
    """Shannon entropy"""
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def top_k_overlap(p1, p2, k=10):
    """How many of top-k in p1 are also in top-k of p2?"""
    top1 = set(np.argsort(p1)[-k:])
    top2 = set(np.argsort(p2)[-k:])
    return len(top1 & top2) / k


def rollout_future(model, tokenizer, input_device, prompt, n_steps=30):
    """
    Autoregressively rollout future tokens, collecting P(next) at each step.
    Returns list of dicts with entropy, top probs, etc.
    """
    import torch
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(input_device)
    attention_mask = inputs["attention_mask"].to(input_device)
    
    steps = []
    
    for step in range(n_steps):
        try:
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
            
            logits = out.logits[0, -1, :].float().cpu().numpy()
            if np.any(np.isnan(logits)) or np.any(np.isinf(logits)):
                break
            
            probs = safe_softmax(logits)
            top5_ids = np.argsort(probs)[-5:][::-1]
            
            steps.append({
                "step": step,
                "entropy": entropy(probs),
                "top1_id": int(top5_ids[0]),
                "top1_prob": float(probs[top5_ids[0]]),
                "top5_ids": top5_ids.tolist(),
                "top5_probs": [float(probs[i]) for i in top5_ids],
                "probs": probs,  # Keep for later comparison
            })
            
            # Greedy next token
            next_token = torch.tensor([[top5_ids[0]]], device=input_device)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
        
        except Exception:
            break
    
    # Free memory
    del input_ids, attention_mask
    return steps


def get_next_dist(model, tokenizer, input_device, prompt):
    """Get P(next_token | prompt) — single step prediction distribution."""
    import torch
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(input_device)
    attention_mask = inputs["attention_mask"].to(input_device)
    
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    
    logits = out.logits[0, -1, :].float().cpu().numpy()
    
    del input_ids, attention_mask
    return logits


# ===== EXP A: CONCEPT PREDICTIVE DEFORMATION =====

def expA_concept_deformation(model, tokenizer, device, model_name,
                              n_templates=15, n_rollout=30):
    """
    Core experiment: How does inserting a concept change the future distribution?
    
    For each (template, concept):
      1. Compute P(future | template with ___) — baseline
      2. Compute P(future | template with concept) — intervened
      3. Measure deformation: KL, JS, entropy change, top-k shift
    
    Build: Concept Action Tensor [n_concepts × n_templates × n_metrics]
    """
    import torch
    
    print("\n" + "="*60)
    print("EXP A: Concept Predictive Deformation")
    print("="*60)
    
    input_device = get_device_for_input(model)
    
    # Collect all concepts
    all_concepts = []
    concept_families = {}
    for family, concepts in CONCEPT_FAMILIES.items():
        for c in concepts:
            all_concepts.append(c)
            concept_families[c] = family
    
    # Use subset of templates if needed
    templates = CONTEXT_TEMPLATES[:n_templates]
    
    # Results structure
    deformation_matrix = {}  # concept -> template -> metrics
    baseline_cache = {}      # template -> rollout data
    
    # Step 1: Compute baselines (template without concept insertion)
    print(f"  Computing baselines for {len(templates)} templates...")
    for tidx, template in enumerate(templates):
        if tidx % 5 == 0:
            print(f"    Baseline {tidx}/{len(templates)}...")
        
        # Create baseline prompt (replace ___ with "the" as neutral filler)
        baseline_prompt = template.replace("___", "the")
        
        # Get single-step P(next) for baseline
        baseline_logits = get_next_dist(model, tokenizer, input_device, baseline_prompt)
        baseline_probs = safe_softmax(baseline_logits)
        
        # Also rollout future for multi-step analysis
        baseline_steps = rollout_future(model, tokenizer, input_device,
                                         baseline_prompt, n_steps=n_rollout)
        
        baseline_cache[template] = {
            "prompt": baseline_prompt,
            "probs_1step": baseline_probs,
            "logits_1step": baseline_logits,
            "rollout_steps": len(baseline_steps),
            "rollout_entropy": [s["entropy"] for s in baseline_steps],
        }
    
    # Step 2: For each (concept, template), compute deformation
    print(f"  Computing deformations for {len(all_concepts)} concepts × {len(templates)} templates...")
    
    total = len(all_concepts) * len(templates)
    count = 0
    
    for concept in all_concepts:
        deformation_matrix[concept] = {}
        family = concept_families[concept]
        
        for template in templates:
            count += 1
            if count % 50 == 0:
                print(f"    Progress: {count}/{total}...")
            
            # Create intervened prompt
            intervened_prompt = template.replace("___", concept)
            
            # Get single-step P(next)
            intervened_logits = get_next_dist(model, tokenizer, input_device, intervened_prompt)
            intervened_probs = safe_softmax(intervened_logits)
            
            # Get baseline
            baseline_probs = baseline_cache[template]["probs_1step"]
            baseline_logits = baseline_cache[template]["logits_1step"]
            
            # Compute deformation metrics
            kl_1step = kl_divergence(intervened_probs, baseline_probs)
            js_1step = js_divergence(intervened_probs, baseline_probs)
            
            ent_intervened = entropy(intervened_probs)
            ent_baseline = entropy(baseline_probs)
            delta_entropy = ent_intervened - ent_baseline
            
            top10_overlap = top_k_overlap(intervened_probs, baseline_probs, k=10)
            top5_overlap = top_k_overlap(intervened_probs, baseline_probs, k=5)
            top1_match = 1 if np.argmax(intervened_probs) == np.argmax(baseline_probs) else 0
            
            # Top-5 shift: which tokens gained/lost probability most?
            prob_shift = intervened_probs - baseline_probs
            top5_gainers = np.argsort(prob_shift)[-5:][::-1].tolist()
            top5_losers = np.argsort(prob_shift)[:5].tolist()
            
            # Rollout future for multi-step deformation
            intervened_steps = rollout_future(model, tokenizer, input_device,
                                              intervened_prompt, n_steps=min(15, n_rollout))
            
            baseline_ent = baseline_cache[template]["rollout_entropy"]
            intervened_ent = [s["entropy"] for s in intervened_steps]
            
            # Multi-step entropy divergence
            n_compare = min(len(baseline_ent), len(intervened_ent))
            if n_compare > 0:
                ent_divergence = [abs(baseline_ent[i] - intervened_ent[i])
                                  for i in range(n_compare)]
                mean_ent_div = float(np.mean(ent_divergence))
                cum_ent_div = float(np.sum(ent_divergence))
            else:
                mean_ent_div = 0
                cum_ent_div = 0
            
            # Top-1 stability across rollout steps
            top1_stability = 0
            if len(intervened_steps) > 1:
                same_count = sum(1 for i in range(1, len(intervened_steps))
                                 if intervened_steps[i]["top1_id"] == intervened_steps[i-1]["top1_id"])
                top1_stability = same_count / (len(intervened_steps) - 1)
            
            deformation_matrix[concept][template] = {
                "kl_1step": round(kl_1step, 4),
                "js_1step": round(js_1step, 4),
                "delta_entropy": round(delta_entropy, 4),
                "top10_overlap": round(top10_overlap, 4),
                "top5_overlap": round(top5_overlap, 4),
                "top1_match": top1_match,
                "mean_ent_divergence_multistep": round(mean_ent_div, 4),
                "cum_ent_divergence_multistep": round(cum_ent_div, 4),
                "top1_stability": round(top1_stability, 4),
                "rollout_steps": len(intervened_steps),
                "family": family,
            }
    
    # ===== Aggregate: Per-concept deformation profile =====
    concept_profiles = {}
    for concept in all_concepts:
        family = concept_families[concept]
        metrics = defaultdict(list)
        
        for template, vals in deformation_matrix[concept].items():
            for k, v in vals.items():
                if isinstance(v, (int, float)) and k != "rollout_steps":
                    metrics[k].append(v)
        
        concept_profiles[concept] = {
            "family": family,
            "kl_mean": round(float(np.mean(metrics["kl_1step"])), 4),
            "kl_std": round(float(np.std(metrics["kl_1step"])), 4),
            "js_mean": round(float(np.mean(metrics["js_1step"])), 4),
            "delta_entropy_mean": round(float(np.mean(metrics["delta_entropy"])), 4),
            "top10_overlap_mean": round(float(np.mean(metrics["top10_overlap"])), 4),
            "top5_overlap_mean": round(float(np.mean(metrics["top5_overlap"])), 4),
            "top1_match_rate": round(float(np.mean(metrics["top1_match"])), 4),
            "multistep_div_mean": round(float(np.mean(metrics["mean_ent_divergence_multistep"])), 4),
            "multistep_div_cum": round(float(np.mean(metrics["cum_ent_divergence_multistep"])), 4),
            "top1_stability_mean": round(float(np.mean(metrics["top1_stability"])), 4),
            # KEY: deformation stability (CV = std/mean) — lower = more stable across contexts
            "kl_cv": round(float(np.std(metrics["kl_1step"]) / max(np.mean(metrics["kl_1step"]), 1e-6)), 4),
        }
    
    # ===== Cross-concept deformation similarity =====
    # For each pair of concepts, compute how similar their deformation patterns are
    print(f"  Computing cross-concept deformation similarity...")
    
    concept_list = sorted(all_concepts)
    n_c = len(concept_list)
    
    # Build deformation vectors: each concept has a vector of KL values across templates
    deformation_vectors = {}
    for concept in concept_list:
        vec = []
        for template in templates:
            vec.append(deformation_matrix[concept][template]["kl_1step"])
        deformation_vectors[concept] = np.array(vec)
    
    # Pairwise correlation of deformation patterns
    deformation_corr = np.zeros((n_c, n_c))
    for i in range(n_c):
        for j in range(n_c):
            c1, c2 = concept_list[i], concept_list[j]
            if np.std(deformation_vectors[c1]) > 1e-10 and np.std(deformation_vectors[c2]) > 1e-10:
                deformation_corr[i, j] = np.corrcoef(deformation_vectors[c1],
                                                       deformation_vectors[c2])[0, 1]
            else:
                deformation_corr[i, j] = 0.0
    
    # Intra-family vs inter-family correlation
    intra_family_corr = []
    inter_family_corr = []
    for i in range(n_c):
        for j in range(i+1, n_c):
            f1 = concept_families[concept_list[i]]
            f2 = concept_families[concept_list[j]]
            corr_val = deformation_corr[i, j]
            if f1 == f2:
                intra_family_corr.append(corr_val)
            else:
                inter_family_corr.append(corr_val)
    
    # Per-family average deformation profile
    family_profiles = {}
    for family in CONCEPT_FAMILIES:
        concepts_in_family = [c for c in all_concepts if concept_families[c] == family]
        family_kl = [concept_profiles[c]["kl_mean"] for c in concepts_in_family]
        family_js = [concept_profiles[c]["js_mean"] for c in concepts_in_family]
        family_top10 = [concept_profiles[c]["top10_overlap_mean"] for c in concepts_in_family]
        family_multistep = [concept_profiles[c]["multistep_div_mean"] for c in concepts_in_family]
        
        family_profiles[family] = {
            "n_concepts": len(concepts_in_family),
            "kl_mean": round(float(np.mean(family_kl)), 4),
            "js_mean": round(float(np.mean(family_js)), 4),
            "top10_overlap": round(float(np.mean(family_top10)), 4),
            "multistep_div": round(float(np.mean(family_multistep)), 4),
        }
    
    results = {
        "n_concepts": len(all_concepts),
        "n_templates": len(templates),
        "concept_profiles": concept_profiles,
        "family_profiles": family_profiles,
        "intra_family_corr_mean": round(float(np.mean(intra_family_corr)), 4) if intra_family_corr else 0,
        "intra_family_corr_std": round(float(np.std(intra_family_corr)), 4) if intra_family_corr else 0,
        "inter_family_corr_mean": round(float(np.mean(inter_family_corr)), 4) if inter_family_corr else 0,
        "inter_family_corr_std": round(float(np.std(inter_family_corr)), 4) if inter_family_corr else 0,
        "family_emergence_ratio": round(float(np.mean(intra_family_corr)) / max(float(np.mean(inter_family_corr)), 1e-6), 4)
            if intra_family_corr and inter_family_corr else 0,
        "deformation_corr_matrix": {concept_list[i]: {concept_list[j]: round(float(deformation_corr[i,j]), 4)
                                                        for j in range(n_c)}
                                     for i in range(n_c)},
    }
    
    print(f"\n  === Concept Profiles ===")
    for concept in all_concepts:
        p = concept_profiles[concept]
        print(f"    {concept} ({concept_families[concept]}): "
              f"KL={p['kl_mean']:.3f}±{p['kl_std']:.3f}, "
              f"top10_overlap={p['top10_overlap_mean']:.3f}, "
              f"CV={p['kl_cv']:.3f}, "
              f"multistep_div={p['multistep_div_mean']:.3f}")
    
    print(f"\n  === Family Emergence ===")
    print(f"    Intra-family corr: {results['intra_family_corr_mean']:.4f} ± {results['intra_family_corr_std']:.4f}")
    print(f"    Inter-family corr: {results['inter_family_corr_mean']:.4f} ± {results['inter_family_corr_std']:.4f}")
    print(f"    Emergence ratio (intra/inter): {results['family_emergence_ratio']:.2f}")
    
    return results


# ===== EXP B: CONCEPT FAMILY EMERGENCE =====

def expB_family_emergence(model, tokenizer, device, model_name):
    """
    Cluster concepts by their deformation patterns.
    Test if deformation-based clustering recovers semantic families.
    
    Compare 3 types of similarity:
    1. Deformation similarity (KL pattern correlation)
    2. Embedding similarity (W_U row cosine)
    3. Logit similarity (P(next) correlation)
    """
    import torch
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    
    print("\n" + "="*60)
    print("EXP B: Concept Family Emergence")
    print("="*60)
    
    input_device = get_device_for_input(model)
    
    all_concepts = []
    concept_families = {}
    for family, concepts in CONCEPT_FAMILIES.items():
        for c in concepts:
            all_concepts.append(c)
            concept_families[c] = family
    
    true_labels = [concept_families[c] for c in all_concepts]
    n_concepts = len(all_concepts)
    
    # Step 1: Collect logit distributions for each concept in a fixed context
    fixed_template = "I saw a ___ in the park yesterday"
    
    print(f"  Collecting logit distributions for {n_concepts} concepts...")
    
    concept_logits = {}
    concept_probs = {}
    
    for concept in all_concepts:
        prompt = fixed_template.replace("___", concept)
        logits = get_next_dist(model, tokenizer, input_device, prompt)
        probs = safe_softmax(logits)
        concept_logits[concept] = logits
        concept_probs[concept] = probs
    
    # Step 2: Get W_U embedding for each concept
    from model_utils import get_W_U
    W_U = get_W_U(model, model_name)
    
    concept_embeddings = {}
    for concept in all_concepts:
        tok_ids = tokenizer.encode(concept, add_special_tokens=False)
        if len(tok_ids) > 0:
            concept_embeddings[concept] = W_U[tok_ids[0]]
        else:
            concept_embeddings[concept] = np.zeros(W_U.shape[1])
    
    # Step 3: Compute pairwise similarities
    # 3a. Deformation similarity (from Exp A — use KL correlation)
    # We'll use the logit-based deformation here for simplicity
    
    # Deformation: for each pair of contexts, how different is the deformation?
    templates = CONTEXT_TEMPLATES[:10]
    
    deformation_vectors = {}
    for concept in all_concepts:
        vec = []
        # Get baseline logits for each template
        for template in templates:
            baseline_prompt = template.replace("___", "the")
            baseline_logits = get_next_dist(model, tokenizer, input_device, baseline_prompt)
            baseline_probs = safe_softmax(baseline_logits)
            
            intervened_prompt = template.replace("___", concept)
            intervened_logits = get_next_dist(model, tokenizer, input_device, intervened_prompt)
            intervened_probs = safe_softmax(intervened_logits)
            
            # Deformation vector = log-prob shift in top-100 dimensions
            prob_shift = intervened_probs - baseline_probs
            # Take top-100 by absolute shift for efficiency
            top_dims = np.argsort(np.abs(prob_shift))[-100:]
            vec.extend(prob_shift[top_dims].tolist())
        
        deformation_vectors[concept] = np.array(vec)
    
    # Compute pairwise cosine similarity for deformation vectors
    def cosine_sim(v1, v2):
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-10 or n2 < 1e-10:
            return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))
    
    # Deformation similarity matrix
    deform_sim = np.zeros((n_concepts, n_concepts))
    for i in range(n_concepts):
        for j in range(n_concepts):
            deform_sim[i, j] = cosine_sim(deformation_vectors[all_concepts[i]],
                                            deformation_vectors[all_concepts[j]])
    
    # Embedding similarity matrix
    embed_sim = np.zeros((n_concepts, n_concepts))
    for i in range(n_concepts):
        for j in range(n_concepts):
            embed_sim[i, j] = cosine_sim(concept_embeddings[all_concepts[i]],
                                          concept_embeddings[all_concepts[j]])
    
    # Logit distribution similarity (P(next) correlation)
    logit_sim = np.zeros((n_concepts, n_concepts))
    for i in range(n_concepts):
        for j in range(n_concepts):
            p1 = concept_probs[all_concepts[i]]
            p2 = concept_probs[all_concepts[j]]
            # Correlation of log-probabilities
            lp1 = np.log(np.clip(p1, 1e-10, 1.0))
            lp2 = np.log(np.clip(p2, 1e-10, 1.0))
            if np.std(lp1) > 1e-10 and np.std(lp2) > 1e-10:
                logit_sim[i, j] = np.corrcoef(lp1, lp2)[0, 1]
            else:
                logit_sim[i, j] = 0.0
    
    # Clean NaN values
    deform_sim = np.nan_to_num(deform_sim, nan=0.0)
    embed_sim = np.nan_to_num(embed_sim, nan=0.0)
    logit_sim = np.nan_to_num(logit_sim, nan=0.0)
    
    # Step 4: Clustering comparison
    n_clusters = len(CONCEPT_FAMILIES)
    
    # Cluster by deformation similarity
    deform_dist = 1 - deform_sim
    deform_dist = np.clip(deform_dist, 0, 2)
    deform_dist = np.nan_to_num(deform_dist, nan=1.0)  # Replace NaN with max distance
    clustering_deform = AgglomerativeClustering(n_clusters=n_clusters,
                                                  metric="precomputed",
                                                  linkage="average")
    labels_deform = clustering_deform.fit_predict(deform_dist)
    
    # Cluster by embedding similarity
    embed_dist = 1 - embed_sim
    embed_dist = np.clip(embed_dist, 0, 2)
    embed_dist = np.nan_to_num(embed_dist, nan=1.0)
    clustering_embed = AgglomerativeClustering(n_clusters=n_clusters,
                                                metric="precomputed",
                                                linkage="average")
    labels_embed = clustering_embed.fit_predict(embed_dist)
    
    # Cluster by logit similarity
    logit_dist = 1 - logit_sim
    logit_dist = np.clip(logit_dist, 0, 2)
    logit_dist = np.nan_to_num(logit_dist, nan=1.0)
    clustering_logit = AgglomerativeClustering(n_clusters=n_clusters,
                                                metric="precomputed",
                                                linkage="average")
    labels_logit = clustering_logit.fit_predict(logit_dist)
    
    # ARI (Adjusted Rand Index) against true family labels
    ari_deform = adjusted_rand_score(true_labels, labels_deform)
    ari_embed = adjusted_rand_score(true_labels, labels_embed)
    ari_logit = adjusted_rand_score(true_labels, labels_logit)
    
    # Silhouette scores
    try:
        sil_deform = silhouette_score(deform_dist, labels_deform, metric="precomputed")
    except Exception:
        sil_deform = 0
    try:
        sil_embed = silhouette_score(embed_dist, labels_embed, metric="precomputed")
    except Exception:
        sil_embed = 0
    try:
        sil_logit = silhouette_score(logit_dist, labels_logit, metric="precomputed")
    except Exception:
        sil_logit = 0
    
    # Intra-family vs inter-family similarity for each method
    def intra_inter_ratio(sim_matrix, labels_list, true_families):
        intra = []
        inter = []
        for i in range(len(labels_list)):
            for j in range(i+1, len(labels_list)):
                if true_families[i] == true_families[j]:
                    intra.append(sim_matrix[i, j])
                else:
                    inter.append(sim_matrix[i, j])
        return (round(float(np.mean(intra)), 4) if intra else 0,
                round(float(np.mean(inter)), 4) if inter else 0)
    
    deform_intra, deform_inter = intra_inter_ratio(deform_sim, all_concepts, true_labels)
    embed_intra, embed_inter = intra_inter_ratio(embed_sim, all_concepts, true_labels)
    logit_intra, logit_inter = intra_inter_ratio(logit_sim, all_concepts, true_labels)
    
    results = {
        "n_concepts": n_concepts,
        "n_families": n_clusters,
        "n_templates": len(templates),
        
        # Clustering quality
        "ari_deformation": round(float(ari_deform), 4),
        "ari_embedding": round(float(ari_embed), 4),
        "ari_logit": round(float(ari_logit), 4),
        "silhouette_deformation": round(float(sil_deform), 4),
        "silhouette_embedding": round(float(sil_embed), 4),
        "silhouette_logit": round(float(sil_logit), 4),
        
        # Intra/inter family similarity
        "deform_intra_mean": deform_intra,
        "deform_inter_mean": deform_inter,
        "embed_intra_mean": embed_intra,
        "embed_inter_mean": embed_inter,
        "logit_intra_mean": logit_intra,
        "logit_inter_mean": logit_inter,
        
        # Cluster assignments
        "deform_clusters": {all_concepts[i]: int(labels_deform[i]) for i in range(n_concepts)},
        "embed_clusters": {all_concepts[i]: int(labels_embed[i]) for i in range(n_concepts)},
        "logit_clusters": {all_concepts[i]: int(labels_logit[i]) for i in range(n_concepts)},
        "true_families": {c: concept_families[c] for c in all_concepts},
    }
    
    print(f"\n  === Clustering Quality (ARI) ===")
    print(f"    Deformation-based: ARI={results['ari_deformation']:.4f}")
    print(f"    Embedding-based:   ARI={results['ari_embedding']:.4f}")
    print(f"    Logit-based:       ARI={results['ari_logit']:.4f}")
    
    print(f"\n  === Intra/Inter Family Similarity ===")
    print(f"    Deformation: intra={deform_intra:.4f}, inter={deform_inter:.4f}, ratio={deform_intra/max(deform_inter,1e-6):.2f}")
    print(f"    Embedding:   intra={embed_intra:.4f}, inter={embed_inter:.4f}, ratio={embed_intra/max(embed_inter,1e-6):.2f}")
    print(f"    Logit:       intra={logit_intra:.4f}, inter={logit_inter:.4f}, ratio={logit_intra/max(logit_inter,1e-6):.2f}")
    
    print(f"\n  === Cluster Assignments (deformation) ===")
    for concept in all_concepts:
        print(f"    {concept} ({concept_families[concept]}): cluster={results['deform_clusters'][concept]}")
    
    return results


# ===== EXP C: OPERATOR COMPOSITION TEST =====

def expC_operator_composition(model, tokenizer, device, model_name, n_templates=10):
    """
    Test if concept operators compose algebraically.
    
    T_adj ∘ T_noun ≈ T_{adj+noun}?
    
    E.g.: Does "red" deformation + "cat" deformation ≈ "red cat" deformation?
    """
    import torch
    
    print("\n" + "="*60)
    print("EXP C: Operator Composition Test")
    print("="*60)
    
    input_device = get_device_for_input(model)
    templates = CONTEXT_TEMPLATES[:n_templates]
    
    composition_results = []
    
    for adj, noun, composed in COMPOSITION_PAIRS:
        print(f"  Testing: T_{adj} ∘ T_{noun} vs T_{{{composed}}}")
        
        adj_deformations = []  # KL(P(context+adj), P(context))
        noun_deformations = []
        composed_deformations = []
        
        # Also get the "combined" deformation: apply adj then noun
        for template in templates:
            # Baseline: template with "the"
            baseline_prompt = template.replace("___", "the")
            baseline_logits = get_next_dist(model, tokenizer, input_device, baseline_prompt)
            baseline_probs = safe_softmax(baseline_logits)
            
            # Adj only: template with adj
            adj_prompt = template.replace("___", adj)
            adj_logits = get_next_dist(model, tokenizer, input_device, adj_prompt)
            adj_probs = safe_softmax(adj_logits)
            
            # Noun only: template with noun
            noun_prompt = template.replace("___", noun)
            noun_logits = get_next_dist(model, tokenizer, input_device, noun_prompt)
            noun_probs = safe_softmax(noun_logits)
            
            # Composed: template with "adj noun"
            composed_prompt = template.replace("___", composed)
            composed_logits = get_next_dist(model, tokenizer, input_device, composed_prompt)
            composed_probs = safe_softmax(composed_logits)
            
            # Deformations
            kl_adj = kl_divergence(adj_probs, baseline_probs)
            kl_noun = kl_divergence(noun_probs, baseline_probs)
            kl_composed = kl_divergence(composed_probs, baseline_probs)
            
            # KEY: Does T_{adj+noun} ≈ T_adj + T_noun?
            # Compare the logit shifts
            delta_adj = adj_logits - baseline_logits  # T_adj effect
            delta_noun = noun_logits - baseline_logits  # T_noun effect
            delta_composed = composed_logits - baseline_logits  # T_{adj+noun} effect
            delta_sum = delta_adj + delta_noun  # T_adj + T_noun
            
            # Cosine similarity between delta_composed and delta_sum
            norm_comp = np.linalg.norm(delta_composed)
            norm_sum = np.linalg.norm(delta_sum)
            
            if norm_comp > 1e-10 and norm_sum > 1e-10:
                cos_composed_vs_sum = float(np.dot(delta_composed, delta_sum) / (norm_comp * norm_sum))
            else:
                cos_composed_vs_sum = 0.0
            
            # Also test: does T_adj applied to the noun-context give the same as T_{adj+noun}?
            # This is: KL(P(context+adj+noun), P(context+noun)) vs KL(P(context+adj+noun), P(context+adj))
            kl_composed_given_noun = kl_divergence(composed_probs, noun_probs)
            kl_composed_given_adj = kl_divergence(composed_probs, adj_probs)
            
            # Commutativity: T_adj ∘ T_noun vs T_noun ∘ T_adj
            # We can't truly compose in a single context, but we can test:
            # Does "adj noun" (canonical order) ≈ "noun adj" (reversed order)?
            reversed_prompt = template.replace("___", f"{noun} {adj}")
            reversed_logits = get_next_dist(model, tokenizer, input_device, reversed_prompt)
            reversed_probs = safe_softmax(reversed_logits)
            
            kl_canonical_vs_reversed = kl_divergence(composed_probs, reversed_probs)
            js_canonical_vs_reversed = js_divergence(composed_probs, reversed_probs)
            
            adj_deformations.append(kl_adj)
            noun_deformations.append(kl_noun)
            composed_deformations.append(kl_composed)
            
            composition_results.append({
                "adj": adj,
                "noun": noun,
                "composed": composed,
                "template": template,
                "kl_adj": round(kl_adj, 4),
                "kl_noun": round(kl_noun, 4),
                "kl_composed": round(kl_composed, 4),
                "cos_composed_vs_sum": round(cos_composed_vs_sum, 4),
                "kl_composed_given_noun": round(kl_composed_given_noun, 4),
                "kl_composed_given_adj": round(kl_composed_given_adj, 4),
                "kl_canonical_vs_reversed": round(kl_canonical_vs_reversed, 4),
                "js_canonical_vs_reversed": round(js_canonical_vs_reversed, 4),
            })
    
    # Aggregate per composition pair
    pair_summaries = {}
    for adj, noun, composed in COMPOSITION_PAIRS:
        pair_data = [r for r in composition_results if r["adj"] == adj and r["noun"] == noun]
        if not pair_data:
            continue
        
        pair_summaries[f"{adj}+{noun}"] = {
            "kl_adj_mean": round(float(np.mean([r["kl_adj"] for r in pair_data])), 4),
            "kl_noun_mean": round(float(np.mean([r["kl_noun"] for r in pair_data])), 4),
            "kl_composed_mean": round(float(np.mean([r["kl_composed"] for r in pair_data])), 4),
            "cos_composed_vs_sum_mean": round(float(np.mean([r["cos_composed_vs_sum"] for r in pair_data])), 4),
            "kl_composed_given_noun_mean": round(float(np.mean([r["kl_composed_given_noun"] for r in pair_data])), 4),
            "kl_composed_given_adj_mean": round(float(np.mean([r["kl_composed_given_adj"] for r in pair_data])), 4),
            "kl_canonical_vs_reversed_mean": round(float(np.mean([r["kl_canonical_vs_reversed"] for r in pair_data])), 4),
            "js_canonical_vs_reversed_mean": round(float(np.mean([r["js_canonical_vs_reversed"] for r in pair_data])), 4),
        }
    
    # Global summary
    all_cos = [r["cos_composed_vs_sum"] for r in composition_results]
    all_kl_comm = [r["kl_canonical_vs_reversed"] for r in composition_results]
    
    results = {
        "n_pairs": len(COMPOSITION_PAIRS),
        "n_templates": n_templates,
        "pair_summaries": pair_summaries,
        "global_cos_composed_vs_sum_mean": round(float(np.mean(all_cos)), 4),
        "global_cos_composed_vs_sum_std": round(float(np.std(all_cos)), 4),
        "global_kl_canonical_vs_reversed_mean": round(float(np.mean(all_kl_comm)), 4),
        "global_kl_canonical_vs_reversed_std": round(float(np.std(all_kl_comm)), 4),
    }
    
    print(f"\n  === Composition Quality ===")
    print(f"    cos(Δ_composed, Δ_adj+Δ_noun): {results['global_cos_composed_vs_sum_mean']:.4f} ± {results['global_cos_composed_vs_sum_std']:.4f}")
    print(f"    KL(canonical, reversed): {results['global_kl_canonical_vs_reversed_mean']:.4f} ± {results['global_kl_canonical_vs_reversed_std']:.4f}")
    
    for pair_name, ps in pair_summaries.items():
        print(f"    {pair_name}: cos={ps['cos_composed_vs_sum_mean']:.4f}, "
              f"KL(comm)={ps['kl_canonical_vs_reversed_mean']:.4f}")
    
    return results


# ===== EXP D: MULTI-STEP FUTURE EQUIVALENCE =====

def expD_multistep_equivalence(model, tokenizer, device, model_name,
                                n_prompts=60, n_steps=5):
    """
    Test if 1-step predictive equivalence implies k-step equivalence.
    
    For pairs of contexts with similar P(next_1):
      - Is P(next_1, next_2, ..., next_k) also similar?
      - Or does 1-step equivalence break down at 2+ steps?
    
    This tests whether the "true state" requires multi-step predictive equivalence.
    """
    import torch
    
    print("\n" + "="*60)
    print("EXP D: Multi-step Future Equivalence")
    print("="*60)
    
    input_device = get_device_for_input(model)
    
    # Collect diverse prompts (inline to avoid import issues)
    PROMPTS_163 = {
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
            "The Amazon Rainforest produces approximately 20 percent of the world oxygen by",
            "The periodic table organizes elements by their atomic number, which represents",
            "Gravity causes objects to accelerate at approximately 9.8 meters per second squared, meaning",
            "The French Revolution began in 1789 when the people of Paris",
        ],
    }
    
    all_prompts = []
    all_domains = []
    for domain, prompts in PROMPTS_163.items():
        for p in prompts[:12]:
            all_prompts.append(p)
            all_domains.append(domain)
    
    if len(all_prompts) > n_prompts:
        indices = np.random.choice(len(all_prompts), n_prompts, replace=False)
        all_prompts = [all_prompts[i] for i in indices]
        all_domains = [all_domains[i] for i in indices]
    
    # For each prompt, rollout n_steps tokens, collecting P(next) at each step
    print(f"  Collecting {n_steps}-step rollouts for {len(all_prompts)} prompts...")
    
    prompt_rollouts = []
    
    for pidx, (prompt, domain) in enumerate(zip(all_prompts, all_domains)):
        if pidx % 15 == 0:
            print(f"    Prompt {pidx}/{len(all_prompts)}...")
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        
        step_probs = []
        
        for step in range(n_steps):
            try:
                with torch.no_grad():
                    out = model(input_ids=input_ids, attention_mask=attention_mask)
                
                logits = out.logits[0, -1, :].float().cpu().numpy()
                if np.any(np.isnan(logits)):
                    break
                
                probs = safe_softmax(logits)
                step_probs.append(probs)
                
                # Greedy next
                top1_id = int(np.argmax(probs))
                next_token = torch.tensor([[top1_id]], device=input_device)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
            
            except Exception:
                break
        
        if len(step_probs) == n_steps:
            prompt_rollouts.append({
                "prompt_idx": pidx,
                "domain": domain,
                "step_probs": step_probs,  # list of prob arrays, one per step
            })
    
    print(f"  Collected {len(prompt_rollouts)} complete rollouts")
    
    if len(prompt_rollouts) < 20:
        print(f"  WARNING: Too few rollouts ({len(prompt_rollouts)}), results unreliable")
        return {"error": "too_few_rollouts", "n_rollouts": len(prompt_rollouts)}
    
    # Step 2: Find pairs with similar 1-step predictions
    # Then compare their k-step predictions
    
    n_r = len(prompt_rollouts)
    max_pairs = min(3000, n_r * (n_r - 1) // 2)
    
    # Sample pairs
    np.random.seed(42)
    pairs_i = np.random.randint(0, n_r, size=max_pairs * 2)
    pairs_j = np.random.randint(0, n_r, size=max_pairs * 2)
    valid = pairs_i != pairs_j
    pairs_i = pairs_i[valid][:max_pairs]
    pairs_j = pairs_j[valid][:max_pairs]
    
    # Compute KL at each step for each pair
    kl_by_step = defaultdict(list)  # step -> list of KL values
    
    for idx in range(len(pairs_i)):
        i, j = pairs_i[idx], pairs_j[idx]
        ri, rj = prompt_rollouts[i], prompt_rollouts[j]
        
        for step in range(n_steps):
            kl = kl_divergence(ri["step_probs"][step], rj["step_probs"][step])
            kl_by_step[step].append(kl)
    
    # Convert to arrays
    kl_arrays = {step: np.array(vals) for step, vals in kl_by_step.items()}
    
    # Correlation between 1-step KL and k-step KL
    corr_1k = {}
    for step in range(1, n_steps):
        if len(kl_arrays[0]) > 10:
            corr = float(np.corrcoef(kl_arrays[0], kl_arrays[step])[0, 1])
        else:
            corr = 0.0
        corr_1k[step] = round(corr, 4)
    
    # KEY ANALYSIS: Pairs with small 1-step KL but large k-step KL
    # → 1-step equivalence does NOT imply k-step equivalence
    kl1_threshold = np.percentile(kl_arrays[0], 20)  # bottom 20%
    
    equiv_breakdown = {}
    for step in range(1, n_steps):
        small_kl1_mask = kl_arrays[0] < kl1_threshold
        if np.any(small_kl1_mask):
            kl_k_given_small_kl1 = kl_arrays[step][small_kl1_mask]
            # What fraction of 1-step-equivalent pairs have k-step KL > median?
            kl_k_median = np.median(kl_arrays[step])
            breakdown_rate = float(np.mean(kl_k_given_small_kl1 > kl_k_median))
            equiv_breakdown[step] = {
                "breakdown_rate": round(breakdown_rate, 4),
                "kl_k_mean_given_small_kl1": round(float(np.mean(kl_k_given_small_kl1)), 4),
                "kl_k_median_overall": round(float(kl_k_median), 4),
            }
    
    # Also: conditioned on same domain vs different domain
    same_domain = []
    diff_domain = []
    for idx in range(len(pairs_i)):
        i, j = pairs_i[idx], pairs_j[idx]
        if prompt_rollouts[i]["domain"] == prompt_rollouts[j]["domain"]:
            same_domain.append(idx)
        else:
            diff_domain.append(idx)
    
    domain_effect = {}
    for step in range(n_steps):
        if same_domain and diff_domain:
            kl_same = kl_arrays[step][same_domain]
            kl_diff = kl_arrays[step][diff_domain]
            domain_effect[step] = {
                "kl_mean_same_domain": round(float(np.mean(kl_same)), 4),
                "kl_mean_diff_domain": round(float(np.mean(kl_diff)), 4),
            }
    
    results = {
        "n_rollouts": len(prompt_rollouts),
        "n_pairs": len(pairs_i),
        "n_steps": n_steps,
        "kl_mean_by_step": {step: round(float(np.mean(vals)), 4) for step, vals in kl_arrays.items()},
        "kl_median_by_step": {step: round(float(np.median(vals)), 4) for step, vals in kl_arrays.items()},
        "corr_1step_vs_kstep": corr_1k,
        "equiv_breakdown": equiv_breakdown,
        "domain_effect": domain_effect,
    }
    
    print(f"\n  === Multi-step KL ===")
    for step in range(n_steps):
        print(f"    Step {step}: KL_mean={results['kl_mean_by_step'][step]:.4f}, "
              f"KL_median={results['kl_median_by_step'][step]:.4f}")
    
    print(f"\n  === 1-step vs k-step Correlation ===")
    for step, corr in corr_1k.items():
        print(f"    corr(KL_1, KL_{step+1}) = {corr:.4f}")
    
    print(f"\n  === Equivalence Breakdown ===")
    for step, vals in equiv_breakdown.items():
        print(f"    Step {step+1}: breakdown_rate={vals['breakdown_rate']:.4f} "
              f"(KL_{step+1}|small_KL1={vals['kl_k_mean_given_small_kl1']:.4f} vs median={vals['kl_k_median_overall']:.4f})")
    
    return results


# ===== MAIN =====

def main():
    import torch
    
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    print(f"\n{'='*60}")
    print(f"Phase 164: Concept Predictive Deformation Atlas")
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
    
    # ===== Exp A: Concept Predictive Deformation =====
    print(f"\n[1] Exp A: Concept Predictive Deformation")
    t0 = time.time()
    all_results["expA_concept_deformation"] = expA_concept_deformation(
        model, tokenizer, device, model_name, n_templates=15, n_rollout=30)
    print(f"  Time: {time.time()-t0:.1f}s")
    
    # ===== Exp B: Concept Family Emergence =====
    print(f"\n[2] Exp B: Concept Family Emergence")
    t0 = time.time()
    all_results["expB_family_emergence"] = expB_family_emergence(
        model, tokenizer, device, model_name)
    print(f"  Time: {time.time()-t0:.1f}s")
    
    # ===== Exp C: Operator Composition Test =====
    print(f"\n[3] Exp C: Operator Composition Test")
    t0 = time.time()
    all_results["expC_operator_composition"] = expC_operator_composition(
        model, tokenizer, device, model_name, n_templates=10)
    print(f"  Time: {time.time()-t0:.1f}s")
    
    # ===== Exp D: Multi-step Future Equivalence =====
    print(f"\n[4] Exp D: Multi-step Future Equivalence")
    t0 = time.time()
    all_results["expD_multistep_equivalence"] = expD_multistep_equivalence(
        model, tokenizer, device, model_name, n_prompts=60, n_steps=5)
    print(f"  Time: {time.time()-t0:.1f}s")
    
    # ===== Save Results =====
    os.makedirs("tests/glm5_temp", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = f"tests/glm5_temp/phase164_{model_name}_{timestamp}.json"
    
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
    
    print(f"\nPhase 164 complete for {model_name}!")


if __name__ == "__main__":
    main()
