"""
Phase 167: Causal Constraint Direction Test — 因果约束方向测试
=============================================================

★★★ THE CRITICAL PIVOT: Statistical observation → Causal intervention! ★★★

Phase 166 found:
  - 40 concepts → rank_90 ≈ 15 (low-dimensional constraint skeleton)
  - PC0 = "concrete vs abstract" explains 50% variance
  - Same-family concepts share 1.2-1.3× more neurons

BUT: ALL of this is STATISTICAL! Are these directions CAUSAL?

User's key criticism:
  "把统计低维误认为真实数学结构 — 极易混淆!"
  "你需要因果实验(causal experiments)，不要继续只做KL和PCA"
  "概念不是算子，是条件约束变换族"
  "真正应该研究: 约束是如何形成稳定流形的"

Phase 167: The FIRST causal test of language constraint structure!

Core question: Are the PC directions "constraint attractors" or statistical artifacts?

Four experiments:

  Exp 1: ★★★ Feature Direction Transport — 特征方向输运(因果核心!)
    - Take PC0-PC4 directions from SVD at mid-layer
    - Inject into DIFFERENT contexts at the SAME layer
    - Measure: Transport Stability Score (TSS) = cos(Δlogits_i, Δlogits_j)
    - KEY: If TSS >> random → PC directions are CAUSAL constraint directions
    - KEY: If TSS ≈ random → PC directions are statistical artifacts

  Exp 2: ★★★ Activation Patching — 激活替换(因果追踪)
    - For concept pairs (cat vs freedom), swap hidden states at each layer
    - Measure: how much of the concept constraint is recovered at each layer?
    - KEY: At which layer is the concept "identity" causally written?

  Exp 3: ★★★ Constraint Jacobian Field — 约束雅可比场
    - For each PC direction, compute ∂logits/∂(PC_i) at each layer
    - Test: does the Jacobian have CONSISTENT structure across contexts?
    - KEY: What is the "causal topology" of the constraint field?

  Exp 4: ★★★ Sparse vs Orthogonal Basis (ICA vs PCA) — 稀疏基 vs 正交基
    - Use FastICA to find independent components
    - Compare causal effectiveness (TSS) of ICA vs PCA directions
    - KEY: Are real features sparse/independent or orthogonal?

Usage: python tests/glm5/phase167_causal_constraint.py <model_name>
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

from model_utils import load_model, get_model_info, get_layers, get_W_U, release_model, MODEL_CONFIGS


# ===== EXPANDED CONCEPT SETS (50 concepts for larger data) =====

CONCEPT_FAMILIES = {
    "animal": ["cat", "dog", "tiger", "lion", "horse", "elephant", "eagle", "whale", "bear", "snake"],
    "vehicle": ["car", "truck", "bus", "train", "bicycle", "airplane", "boat", "motorcycle", "ship", "helicopter"],
    "color": ["red", "blue", "green", "white", "black", "yellow", "purple", "orange", "pink", "brown"],
    "abstract": ["democracy", "freedom", "justice", "power", "truth", "beauty", "wisdom", "courage", "equality", "peace"],
    "emotion": ["love", "anger", "fear", "joy", "sadness", "hope", "pride", "shame", "guilt", "surprise"],
}

ALL_CONCEPTS = []
CONCEPT_TO_FAMILY = {}
for family, concepts in CONCEPT_FAMILIES.items():
    for c in concepts:
        ALL_CONCEPTS.append(c)
        CONCEPT_TO_FAMILY[c] = family


# ===== CONTEXT TEMPLATES (12 for robustness) =====

CONTEXT_TEMPLATES = [
    "I saw a ___ in the park yesterday",
    "The researcher studied the ___ and discovered that",
    "The philosopher contemplated the ___ and concluded that",
    '"I think the ___ is very important," she said, because',
    "The ancient ___ stood in the center of the village, its",
    "If we consider the ___ carefully, we realize that",
    "Unlike other things, the ___ has the unique property that",
    "Before the ___ existed, people believed that",
    "The ___ appeared suddenly, and everyone noticed that",
    "When the ___ was mentioned, the crowd reacted by",
    "According to experts, the ___ can be defined as",
    "The story about the ___ revealed that",
]


# ===== UTILITY FUNCTIONS =====

def get_device_for_input(model):
    import torch
    try:
        return next(model.parameters()).device
    except StopIteration:
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
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def kl_divergence(p, q, eps=1e-10):
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def cosine_sim(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def run_with_direction_injection(model, tokenizer, prompt, target_layer_idx,
                                  direction_np, alpha, input_device):
    """
    Run forward pass with a direction injected at a specific layer.
    
    Returns: (baseline_logits, modified_logits) — both numpy arrays [vocab_size]
    """
    import torch
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(input_device)
    attention_mask = inputs["attention_mask"].to(input_device)
    
    layers = get_layers(model)
    
    # Run baseline first
    with torch.no_grad():
        baseline_out = model(input_ids=input_ids, attention_mask=attention_mask)
    baseline_logits = baseline_out.logits[0, -1, :].float().cpu().numpy()
    
    # Run with injection
    direction_tensor = torch.tensor(direction_np, dtype=torch.float32)
    
    def make_injection_hook(direction, alpha, device):
        def hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0].clone()
                d = direction.to(h.dtype).to(h.device)
                h[0, -1, :] += alpha * d
                return (h,) + output[1:]
            else:
                h = output.clone()
                d = direction.to(h.dtype).to(h.device)
                h[0, -1, :] += alpha * d
                return h
        return hook
    
    hook = layers[target_layer_idx].register_forward_hook(
        make_injection_hook(direction_tensor, alpha, input_device))
    
    with torch.no_grad():
        modified_out = model(input_ids=input_ids, attention_mask=attention_mask)
    
    hook.remove()
    modified_logits = modified_out.logits[0, -1, :].float().cpu().numpy()
    
    del input_ids, attention_mask, baseline_out, modified_out, direction_tensor
    
    return baseline_logits, modified_logits


def compute_feature_directions(model, tokenizer, device, model_name,
                                n_concepts=50, n_templates=12, target_layer_frac=0.67):
    """
    Compute PC0-PC4 feature directions from MLP activation delta matrix.
    
    Returns:
        pc_directions: dict {i: np.array [d_model]} — unit-norm PC directions
        sv_info: dict with singular values and variance ratios
        concept_projections: dict {concept: np.array [n_pcs]} — projections
    """
    import torch
    
    print("\n  Computing feature directions (SVD of MLP activation delta matrix)...")
    
    input_device = get_device_for_input(model)
    layers = get_layers(model)
    n_layers = len(layers)
    info = get_model_info(model, model_name)
    
    target_layer = int(n_layers * target_layer_frac)
    print(f"  Target layer: {target_layer} (of {n_layers})")
    
    concepts = ALL_CONCEPTS[:n_concepts]
    templates = CONTEXT_TEMPLATES[:n_templates]
    
    # Step 1: Collect baseline MLP output
    print(f"  Step 1: Collecting baseline MLP output...")
    baseline_mlp_vector = None
    
    for tidx, template in enumerate(templates):
        baseline_prompt = template.replace("___", "the")
        inputs = tokenizer(baseline_prompt, return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                       output_hidden_states=True)
        
        # hidden_states[target_layer+1] = output of layer target_layer
        h_target = out.hidden_states[target_layer + 1][0, -1, :].float().cpu().numpy()
        
        if baseline_mlp_vector is None:
            baseline_mlp_vector = h_target
        else:
            baseline_mlp_vector += h_target
        
        del input_ids, attention_mask, out
    
    baseline_mlp_vector /= len(templates)
    
    # Step 2: Collect concept MLP outputs
    print(f"  Step 2: Collecting concept MLP outputs ({n_concepts} concepts × {n_templates} templates)...")
    
    concept_vectors = {}
    for cidx, concept in enumerate(concepts):
        if cidx % 10 == 0:
            print(f"    Concept {cidx}/{n_concepts}: {concept}")
        
        concept_vec = None
        
        for template in templates:
            prompt = template.replace("___", concept)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)
            
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True)
            
            h_target = out.hidden_states[target_layer + 1][0, -1, :].float().cpu().numpy()
            
            if concept_vec is None:
                concept_vec = h_target
            else:
                concept_vec += h_target
            
            del input_ids, attention_mask, out
        
        concept_vec /= len(templates)
        concept_vectors[concept] = concept_vec - baseline_mlp_vector  # Delta vector
    
    # Step 3: SVD
    print(f"  Step 3: SVD decomposition...")
    
    concept_list = sorted(concept_vectors.keys())
    M = np.array([concept_vectors[c] for c in concept_list])  # [n_concepts, d_model]
    print(f"  Activation delta matrix: {M.shape}")
    
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    
    energy = S ** 2
    total_energy = np.sum(energy)
    cum_energy = np.cumsum(energy) / total_energy
    
    rank_50 = int(np.searchsorted(cum_energy, 0.50) + 1)
    rank_90 = int(np.searchsorted(cum_energy, 0.90) + 1)
    
    top1_ratio = float(S[0] ** 2 / total_energy)
    top3_ratio = float(np.sum(S[:3] ** 2) / total_energy)
    top5_ratio = float(np.sum(S[:5] ** 2) / total_energy)
    
    # PC directions: rows of Vt (unit-norm directions in d_model space)
    n_pcs = min(5, len(S))
    pc_directions = {}
    for i in range(n_pcs):
        pc_directions[i] = Vt[i] / np.linalg.norm(Vt[i])  # Already unit-norm, but ensure
    
    # Concept projections on PCs
    concept_projections = {}
    for cidx, concept in enumerate(concept_list):
        proj = U[cidx, :n_pcs] * S[:n_pcs]  # [n_pcs]
        concept_projections[concept] = proj
    
    # PC semantic interpretation
    pc_semantics = {}
    for pc_idx in range(n_pcs):
        projections = U[:, pc_idx] * S[pc_idx]
        sorted_idx = np.argsort(projections)
        
        top5 = [(concept_list[i], round(float(projections[i]), 4)) for i in sorted_idx[-5:]][::-1]
        bottom5 = [(concept_list[i], round(float(projections[i]), 4)) for i in sorted_idx[:5]]
        
        top_families = [CONCEPT_TO_FAMILY[concept_list[i]] for i in sorted_idx[-len(concept_list)//2:]]
        family_counts = defaultdict(int)
        for f in top_families:
            family_counts[f] += 1
        
        pc_semantics[pc_idx] = {
            "singular_value": round(float(S[pc_idx]), 4),
            "variance_ratio": round(float(S[pc_idx] ** 2 / total_energy), 4),
            "top5": top5,
            "bottom5": bottom5,
            "dominant_family": max(family_counts, key=family_counts.get),
        }
    
    sv_info = {
        "target_layer": target_layer,
        "rank_50": rank_50,
        "rank_90": rank_90,
        "top1_ratio": round(top1_ratio, 4),
        "top3_ratio": round(top3_ratio, 4),
        "top5_ratio": round(top5_ratio, 4),
        "sv_top10": [round(float(s), 4) for s in S[:10]],
        "pc_semantics": pc_semantics,
    }
    
    print(f"  SVD: rank_50={rank_50}, rank_90={rank_90}, top1_ratio={top1_ratio:.4f}")
    for pc_idx in range(n_pcs):
        ps = pc_semantics[pc_idx]
        print(f"    PC{pc_idx}: var={ps['variance_ratio']:.4f}, "
              f"dominant={ps['dominant_family']}, top3={ps['top5'][:3]}")
    
    return pc_directions, sv_info, concept_projections, target_layer


# ===== EXP 1: FEATURE DIRECTION TRANSPORT TEST (因果核心!) =====

def exp1_feature_transport(model, tokenizer, device, model_name,
                           pc_directions, target_layer, n_concepts=50,
                           alphas=[1.0, 3.0, 5.0], n_pcs=5):
    """
    ★★★ Feature Direction Transport — 特征方向输运 ★★★
    
    THE core causal test:
    - Take PC0-PC4 directions computed at target_layer
    - Inject into DIFFERENT contexts at the SAME layer
    - Measure Transport Stability Score (TSS):
      TSS = average cos(Δlogits_i, Δlogits_j) across context pairs
      
    If TSS >> random → PC directions are CAUSAL constraint directions!
    If TSS ≈ random → PC directions are statistical artifacts!
    
    Also computes:
    - Dose-response: TSS vs α
    - Semantic consistency: does PC0+ always shift toward "concrete"?
    - Cross-family transport: is TSS higher within-family or across-family?
    """
    import torch
    
    print("\n" + "="*60)
    print("EXP 1: Feature Direction Transport Test (CAUSAL CORE)")
    print("="*60)
    
    input_device = get_device_for_input(model)
    layers = get_layers(model)
    n_layers = len(layers)
    
    concepts = ALL_CONCEPTS[:n_concepts]
    # Use a neutral reference template for computing reference effect
    ref_template = "The ___ is"
    
    # Also compute injection at different layers for layer-wise analysis
    sample_layers = list(range(0, n_layers, max(1, n_layers // 6))) + [target_layer, n_layers - 1]
    sample_layers = sorted(set(sample_layers))
    
    print(f"  Testing {n_pcs} PCs × {len(concepts)} concepts × {len(alphas)} alphas")
    print(f"  Injection layers: {sample_layers}")
    
    # ===== Part A: Compute reference effect for each PC at target layer =====
    print("\n  Part A: Computing reference effects...")
    
    # For each PC, compute Δlogits when injected at reference context
    # Use "the" as reference concept in ref_template
    ref_prompt = ref_template.replace("___", "the")
    
    pc_reference_effects = {}  # pc_idx -> Δlogits [vocab_size]
    ref_baseline = None
    
    # Get baseline logits for reference prompt
    inputs = tokenizer(ref_prompt, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(input_device)
    attention_mask = inputs["attention_mask"].to(input_device)
    
    with torch.no_grad():
        ref_out = model(input_ids=input_ids, attention_mask=attention_mask)
    ref_baseline = ref_out.logits[0, -1, :].float().cpu().numpy()
    
    del input_ids, attention_mask, ref_out
    
    alpha_ref = alphas[1] if len(alphas) > 1 else alphas[0]  # Use middle alpha
    
    for pc_idx in range(n_pcs):
        direction = pc_directions[pc_idx]
        
        bl, mod = run_with_direction_injection(
            model, tokenizer, ref_prompt, target_layer, direction, alpha_ref, input_device)
        
        delta_logits = mod - bl  # [vocab_size]
        pc_reference_effects[pc_idx] = delta_logits
        
        # Top-10 tokens shifted by this PC
        top10_increase = np.argsort(delta_logits)[-10:][::-1]
        top10_decrease = np.argsort(delta_logits)[:10]
        
        top10_inc_tokens = [tokenizer.decode([t]).strip() for t in top10_increase]
        top10_dec_tokens = [tokenizer.decode([t]).strip() for t in top10_decrease]
        
        print(f"    PC{pc_idx}: ref_effect norm={np.linalg.norm(delta_logits):.2f}, "
              f"top↑={top10_inc_tokens[:3]}, top↓={top10_dec_tokens[:3]}")
    
    # ===== Part B: Compute transport stability across concepts =====
    print("\n  Part B: Computing transport stability across concepts...")
    
    # For each concept, compute Δlogits for each PC
    # Then measure TSS = cos(Δlogits_concept, Δlogits_reference)
    
    alpha = alpha_ref
    pc_transport_results = defaultdict(lambda: defaultdict(dict))
    
    # Sample concepts for transport test (all 50, but process in batches)
    sample_concepts = concepts[:min(n_concepts, 30)]  # Cap at 30 for time
    
    for cidx, concept in enumerate(sample_concepts):
        if cidx % 5 == 0:
            print(f"    Concept {cidx}/{len(sample_concepts)}: {concept}")
        
        for pc_idx in range(n_pcs):
            direction = pc_directions[pc_idx]
            prompt = ref_template.replace("___", concept)
            
            bl, mod = run_with_direction_injection(
                model, tokenizer, prompt, target_layer, direction, alpha, input_device)
            
            delta_logits = mod - bl
            pc_transport_results[concept][pc_idx] = {
                "delta_logits": delta_logits,
                "baseline_probs": safe_softmax(bl),
                "modified_probs": safe_softmax(mod),
                "kl": kl_divergence(safe_softmax(mod), safe_softmax(bl)),
                "js": js_divergence(safe_softmax(mod), safe_softmax(bl)),
            }
    
    # ===== Part C: Compute TSS =====
    print("\n  Part C: Computing Transport Stability Scores...")
    
    # TSS for each PC: average cosine similarity between Δlogits across concepts
    tss_per_pc = {}
    for pc_idx in range(n_pcs):
        # Collect all Δlogits for this PC
        all_deltas = [pc_transport_results[c][pc_idx]["delta_logits"] 
                      for c in sample_concepts if c in pc_transport_results]
        
        if len(all_deltas) < 2:
            tss_per_pc[pc_idx] = 0.0
            continue
        
        # Pairwise cosine similarity
        cos_values = []
        for i in range(len(all_deltas)):
            for j in range(i + 1, len(all_deltas)):
                cos_values.append(cosine_sim(all_deltas[i], all_deltas[j]))
        
        tss_per_pc[pc_idx] = round(float(np.mean(cos_values)), 4)
        
        # Also compute TSS relative to reference effect
        ref_delta = pc_reference_effects[pc_idx]
        cos_to_ref = [cosine_sim(d, ref_delta) for d in all_deltas]
        tss_ref = round(float(np.mean(cos_to_ref)), 4)
        
        tss_per_pc[f"{pc_idx}_to_ref"] = tss_ref
    
    # ===== Part D: Random direction control =====
    print("\n  Part D: Random direction control...")
    
    d_model = len(pc_directions[0])
    n_random = 5
    random_tss_values = []
    
    for ridx in range(n_random):
        random_dir = np.random.randn(d_model)
        random_dir = random_dir / np.linalg.norm(random_dir)
        
        # Compute Δlogits for random direction across a few concepts
        random_deltas = []
        test_concepts = sample_concepts[:10]  # Use 10 concepts for random control
        
        for concept in test_concepts:
            prompt = ref_template.replace("___", concept)
            bl, mod = run_with_direction_injection(
                model, tokenizer, prompt, target_layer, random_dir, alpha, input_device)
            random_deltas.append(mod - bl)
        
        # Pairwise cosine
        for i in range(len(random_deltas)):
            for j in range(i + 1, len(random_deltas)):
                random_tss_values.append(cosine_sim(random_deltas[i], random_deltas[j]))
    
    random_tss = round(float(np.mean(random_tss_values)), 4) if random_tss_values else 0.0
    
    # ===== Part E: Dose-response (TSS vs α) =====
    print("\n  Part E: Dose-response analysis...")
    
    dose_response = {}
    test_concepts_dose = sample_concepts[:10]
    
    for alpha_val in alphas:
        dose_deltas = defaultdict(list)
        
        for concept in test_concepts_dose:
            prompt = ref_template.replace("___", concept)
            
            for pc_idx in range(min(3, n_pcs)):  # Only PC0-PC2 for dose-response
                direction = pc_directions[pc_idx]
                bl, mod = run_with_direction_injection(
                    model, tokenizer, prompt, target_layer, direction, alpha_val, input_device)
                dose_deltas[pc_idx].append(mod - bl)
        
        # Compute TSS for each PC at this alpha
        for pc_idx in dose_deltas:
            deltas = dose_deltas[pc_idx]
            cos_vals = []
            for i in range(len(deltas)):
                for j in range(i + 1, len(deltas)):
                    cos_vals.append(cosine_sim(deltas[i], deltas[j]))
            dose_response[f"PC{pc_idx}_alpha{alpha_val}"] = round(float(np.mean(cos_vals)), 4) if cos_vals else 0.0
    
    # ===== Part F: Cross-family TSS =====
    print("\n  Part F: Cross-family transport stability...")
    
    cross_family_tss = {}
    families = list(CONCEPT_FAMILIES.keys())
    
    for pc_idx in range(min(3, n_pcs)):
        same_family_cos = []
        diff_family_cos = []
        
        concepts_with_data = [c for c in sample_concepts if c in pc_transport_results 
                              and pc_idx in pc_transport_results[c]]
        
        for i, c1 in enumerate(concepts_with_data):
            for j, c2 in enumerate(concepts_with_data):
                if i >= j:
                    continue
                d1 = pc_transport_results[c1][pc_idx]["delta_logits"]
                d2 = pc_transport_results[c2][pc_idx]["delta_logits"]
                cos = cosine_sim(d1, d2)
                
                if CONCEPT_TO_FAMILY[c1] == CONCEPT_TO_FAMILY[c2]:
                    same_family_cos.append(cos)
                else:
                    diff_family_cos.append(cos)
        
        cross_family_tss[f"PC{pc_idx}_same_family"] = round(float(np.mean(same_family_cos)), 4) if same_family_cos else 0.0
        cross_family_tss[f"PC{pc_idx}_diff_family"] = round(float(np.mean(diff_family_cos)), 4) if diff_family_cos else 0.0
    
    # ===== Part G: Semantic consistency analysis =====
    print("\n  Part G: Semantic consistency of PC effects...")
    
    pc_semantic_effects = {}
    
    for pc_idx in range(n_pcs):
        ref_delta = pc_reference_effects[pc_idx]
        
        # Top-20 tokens that increase/decrease
        top20_increase = np.argsort(ref_delta)[-20:][::-1]
        top20_decrease = np.argsort(ref_delta)[:20]
        
        inc_tokens = [(tokenizer.decode([t]).strip(), round(float(ref_delta[t]), 4)) for t in top20_increase]
        dec_tokens = [(tokenizer.decode([t]).strip(), round(float(ref_delta[t]), 4)) for t in top20_decrease]
        
        # KL and JS for reference injection
        ref_probs_base = safe_softmax(ref_baseline)
        ref_probs_mod = safe_softmax(ref_baseline + ref_delta)
        ref_kl = kl_divergence(ref_probs_mod, ref_probs_base)
        ref_js = js_divergence(ref_probs_mod, ref_probs_base)
        
        pc_semantic_effects[pc_idx] = {
            "ref_kl": round(ref_kl, 6),
            "ref_js": round(ref_js, 6),
            "top10_increase": inc_tokens[:10],
            "top10_decrease": dec_tokens[:10],
            "variance_ratio": round(float(pc_reference_effects[pc_idx].dot(pc_reference_effects[pc_idx])) / 
                                     max(sum(pc_reference_effects[p].dot(pc_reference_effects[p]) 
                                              for p in range(n_pcs)), 1e-10), 4),
        }
    
    # ===== Compile results =====
    results = {
        "n_concepts": len(sample_concepts),
        "n_pcs": n_pcs,
        "target_layer": target_layer,
        "alpha_ref": alpha_ref,
        "alphas": alphas,
        "tss_per_pc": tss_per_pc,
        "random_tss": random_tss,
        "tss_advantage": {f"PC{i}": round(tss_per_pc.get(i, 0) / max(random_tss, 1e-6), 2) 
                          for i in range(n_pcs)},
        "dose_response": dose_response,
        "cross_family_tss": cross_family_tss,
        "pc_semantic_effects": pc_semantic_effects,
        "per_concept_kl": {c: {f"PC{p}": round(pc_transport_results[c][p]["kl"], 6) 
                                for p in range(n_pcs) if p in pc_transport_results.get(c, {})}
                           for c in sample_concepts if c in pc_transport_results},
        "per_concept_js": {c: {f"PC{p}": round(pc_transport_results[c][p]["js"], 6) 
                                for p in range(n_pcs) if p in pc_transport_results.get(c, {})}
                           for c in sample_concepts if c in pc_transport_results},
    }
    
    # Print summary
    print(f"\n  === Feature Direction Transport Results ===")
    print(f"  Random direction TSS: {random_tss:.4f}")
    for pc_idx in range(n_pcs):
        tss = tss_per_pc.get(pc_idx, 0)
        tss_ref = tss_per_pc.get(f"{pc_idx}_to_ref", 0)
        advantage = tss / max(random_tss, 1e-6)
        print(f"  PC{pc_idx}: TSS={tss:.4f}, TSS_to_ref={tss_ref:.4f}, "
              f"advantage={advantage:.1f}×, "
              f"KL={pc_semantic_effects[pc_idx]['ref_kl']:.4f}")
    
    print(f"\n  Cross-family TSS:")
    for pc_idx in range(min(3, n_pcs)):
        same = cross_family_tss.get(f"PC{pc_idx}_same_family", 0)
        diff = cross_family_tss.get(f"PC{pc_idx}_diff_family", 0)
        print(f"  PC{pc_idx}: same_family={same:.4f}, diff_family={diff:.4f}, ratio={same/max(diff,1e-6):.2f}")
    
    return results


# ===== EXP 2: ACTIVATION PATCHING =====

def exp2_activation_patching(model, tokenizer, device, model_name,
                              n_concepts=10, target_layer=None):
    """
    ★★★ Activation Patching — 激活替换(因果追踪) ★★★
    
    For concept pairs, swap hidden states at each layer:
    1. "Clean" run = concept prompt (e.g., "I saw a cat in the park")
    2. "Corrupted" run = baseline prompt (e.g., "I saw the in the park")
    3. At each layer l, patch corrupted with clean hidden state
    4. Measure: how much of the concept-specific logit change is recovered?
    
    KEY: At which layer is the concept "identity" causally written?
    """
    import torch
    
    print("\n" + "="*60)
    print("EXP 2: Activation Patching (Causal Tracing)")
    print("="*60)
    
    input_device = get_device_for_input(model)
    layers = get_layers(model)
    n_layers = len(layers)
    info = get_model_info(model, model_name)
    
    # Select key concepts (2 per family)
    key_concepts = []
    for family in CONCEPT_FAMILIES:
        fam_concepts = [c for c in ALL_CONCEPTS if CONCEPT_TO_FAMILY[c] == family]
        key_concepts.extend(fam_concepts[:2])
    key_concepts = key_concepts[:n_concepts]
    
    template = CONTEXT_TEMPLATES[0]  # "I saw a ___ in the park yesterday"
    
    # Sample layers for patching
    patch_layers = list(range(0, n_layers, max(1, n_layers // 8))) + [n_layers - 1]
    patch_layers = sorted(set(patch_layers))
    
    print(f"  Testing {len(key_concepts)} concepts × {len(patch_layers)} layers")
    print(f"  Concepts: {key_concepts}")
    
    concept_patching_results = {}
    
    for cidx, concept in enumerate(key_concepts):
        print(f"  Concept {cidx+1}/{len(key_concepts)}: {concept}")
        
        family = CONCEPT_TO_FAMILY[concept]
        
        # Clean prompt (with concept)
        clean_prompt = template.replace("___", concept)
        # Corrupted prompt (with baseline "the")
        corrupted_prompt = template.replace("___", "the")
        
        # Run both and get hidden states
        clean_inputs = tokenizer(clean_prompt, return_tensors="pt", truncation=True, max_length=128)
        clean_ids = clean_inputs["input_ids"].to(input_device)
        clean_mask = clean_inputs["attention_mask"].to(input_device)
        
        with torch.no_grad():
            clean_out = model(input_ids=clean_ids, attention_mask=clean_mask,
                             output_hidden_states=True)
        
        corrupted_inputs = tokenizer(corrupted_prompt, return_tensors="pt", truncation=True, max_length=128)
        corrupted_ids = corrupted_inputs["input_ids"].to(input_device)
        corrupted_mask = corrupted_inputs["attention_mask"].to(input_device)
        
        with torch.no_grad():
            corrupted_out = model(input_ids=corrupted_ids, attention_mask=corrupted_mask,
                                  output_hidden_states=True)
        
        # Get logits
        clean_logits = clean_out.logits[0, -1, :].float().cpu().numpy()
        corrupted_logits = corrupted_out.logits[0, -1, :].float().cpu().numpy()
        
        clean_probs = safe_softmax(clean_logits)
        corrupted_probs = safe_softmax(corrupted_logits)
        
        # Total logit change due to concept
        total_logit_change = np.linalg.norm(clean_logits - corrupted_logits)
        total_js = js_divergence(clean_probs, corrupted_probs)
        
        # Get all hidden states
        clean_hs = [hs[0, -1, :].float().cpu().numpy() for hs in clean_out.hidden_states]
        corrupted_hs = [hs[0, -1, :].float().cpu().numpy() for hs in corrupted_out.hidden_states]
        
        del clean_out, corrupted_out
        
        # Patch at each layer
        layer_recovery = {}
        
        for patch_layer in patch_layers:
            # Patch: replace corrupted hidden state at patch_layer with clean hidden state
            # Then continue forward pass from that layer
            
            # We need to manually run the forward pass with the patched hidden state
            # Strategy: use hooks to replace the hidden state at patch_layer
            
            # Get the clean hidden state at the output of patch_layer
            # hidden_states[patch_layer+1] = output of layer patch_layer
            clean_h_at_layer = torch.tensor(clean_hs[patch_layer + 1], 
                                             dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            
            # Run corrupted with patch
            def make_patch_hook(clean_h, layer_idx):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0].clone()
                        # Replace the last token position with clean hidden state
                        h[0, -1, :] = clean_h[0, 0, :].to(h.dtype).to(h.device)
                        return (h,) + output[1:]
                    else:
                        h = output.clone()
                        h[0, -1, :] = clean_h[0, 0, :].to(h.dtype).to(h.device)
                        return h
                return hook
            
            hook = layers[patch_layer].register_forward_hook(
                make_patch_hook(clean_h_at_layer, patch_layer))
            
            with torch.no_grad():
                patched_out = model(input_ids=corrupted_ids, attention_mask=corrupted_mask)
            
            hook.remove()
            
            patched_logits = patched_out.logits[0, -1, :].float().cpu().numpy()
            patched_probs = safe_softmax(patched_logits)
            
            # Recovery metrics
            # How much of the clean-corrupted difference is recovered by the patch?
            logit_recovery = np.linalg.norm(patched_logits - corrupted_logits) / max(total_logit_change, 1e-10)
            
            # JS recovery: how much closer to clean distribution?
            js_patched_corrupted = js_divergence(patched_probs, corrupted_probs)
            js_patched_clean = js_divergence(patched_probs, clean_probs)
            js_recovery = 1.0 - (js_patched_clean / max(total_js, 1e-10))
            
            # Top-1 token agreement
            clean_top1 = np.argmax(clean_probs)
            patched_top1 = np.argmax(patched_probs)
            corrupted_top1 = np.argmax(corrupted_probs)
            top1_match = 1 if patched_top1 == clean_top1 else 0
            
            layer_recovery[patch_layer] = {
                "logit_recovery": round(float(logit_recovery), 4),
                "js_recovery": round(float(js_recovery), 4),
                "js_patched_clean": round(float(js_patched_clean), 6),
                "js_patched_corrupted": round(float(js_patched_corrupted), 6),
                "top1_match": top1_match,
                "clean_top1": tokenizer.decode([int(clean_top1)]).strip(),
                "patched_top1": tokenizer.decode([int(patched_top1)]).strip(),
                "corrupted_top1": tokenizer.decode([int(corrupted_top1)]).strip(),
            }
            
            del patched_out
        
        # Find the "identity layer" — where patching recovers the most
        best_layer = max(patch_layers, key=lambda l: layer_recovery[l]["js_recovery"])
        
        concept_patching_results[concept] = {
            "family": family,
            "total_logit_change": round(float(total_logit_change), 4),
            "total_js": round(float(total_js), 6),
            "best_layer": best_layer,
            "best_js_recovery": layer_recovery[best_layer]["js_recovery"],
            "layer_recovery": layer_recovery,
        }
        
        del clean_ids, clean_mask, corrupted_ids, corrupted_mask
        torch.cuda.empty_cache()
    
    # ===== Aggregate results =====
    # Find the "identity layer" across all concepts
    all_best_layers = [concept_patching_results[c]["best_layer"] for c in key_concepts]
    avg_best_layer = round(float(np.mean(all_best_layers)), 1)
    
    # Per-layer average recovery
    layer_avg_recovery = {}
    for layer in patch_layers:
        recoveries = [concept_patching_results[c]["layer_recovery"][layer]["js_recovery"]
                       for c in key_concepts]
        layer_avg_recovery[layer] = round(float(np.mean(recoveries)), 4)
    
    # Early/mid/late recovery
    n_early = n_layers // 3
    n_mid = 2 * n_layers // 3
    early_layers = [l for l in patch_layers if l < n_early]
    mid_layers = [l for l in patch_layers if n_early <= l < n_mid]
    late_layers = [l for l in patch_layers if l >= n_mid]
    
    early_recovery = np.mean([layer_avg_recovery[l] for l in early_layers]) if early_layers else 0
    mid_recovery = np.mean([layer_avg_recovery[l] for l in mid_layers]) if mid_layers else 0
    late_recovery = np.mean([layer_avg_recovery[l] for l in late_layers]) if late_layers else 0
    
    results = {
        "n_concepts": len(key_concepts),
        "concepts": key_concepts,
        "patch_layers": patch_layers,
        "avg_best_layer": avg_best_layer,
        "layer_avg_recovery": layer_avg_recovery,
        "early_recovery": round(float(early_recovery), 4),
        "mid_recovery": round(float(mid_recovery), 4),
        "late_recovery": round(float(late_recovery), 4),
        "concept_patching_results": concept_patching_results,
    }
    
    print(f"\n  === Activation Patching Results ===")
    print(f"  Average best layer: {avg_best_layer}")
    print(f"  Recovery: early={early_recovery:.4f}, mid={mid_recovery:.4f}, late={late_recovery:.4f}")
    for concept, cr in concept_patching_results.items():
        print(f"  {concept} ({cr['family']}): best_layer={cr['best_layer']}, "
              f"recovery={cr['best_js_recovery']:.4f}, total_js={cr['total_js']:.6f}")
    
    return results


# ===== EXP 3: CONSTRAINT JACOBIAN FIELD =====

def exp3_constraint_jacobian(model, tokenizer, device, model_name,
                              pc_directions, n_concepts=20, alpha=3.0, n_pcs=5):
    """
    ★★★ Constraint Jacobian Field — 约束雅可比场 ★★★
    
    For each PC direction, compute the causal effect at EACH layer:
    ∂logits/∂(PC_i) at layer l = Δlogits when injecting PC_i at layer l
    
    This maps the "causal topology" of the constraint field:
    - Which layers are most sensitive to each PC direction?
    - Does the causal effect have consistent structure across contexts?
    - Is there a "causal flow" from early to late layers?
    """
    import torch
    
    print("\n" + "="*60)
    print("EXP 3: Constraint Jacobian Field")
    print("="*60)
    
    input_device = get_device_for_input(model)
    layers = get_layers(model)
    n_layers = len(layers)
    
    concepts = ALL_CONCEPTS[:n_concepts]
    template = CONTEXT_TEMPLATES[0]
    
    # Sample layers for Jacobian computation
    jacobian_layers = list(range(0, n_layers, max(1, n_layers // 6))) + [n_layers - 1]
    jacobian_layers = sorted(set(jacobian_layers))
    
    print(f"  Testing {n_pcs} PCs × {len(concepts)} concepts × {len(jacobian_layers)} layers")
    
    # For each concept and PC, compute Δlogits at each layer
    # jacobian[concept][pc_idx][layer] = Δlogits
    jacobian = defaultdict(lambda: defaultdict(dict))
    
    for cidx, concept in enumerate(concepts):
        if cidx % 5 == 0:
            print(f"  Concept {cidx}/{len(concepts)}: {concept}")
        
        prompt = template.replace("___", concept)
        
        for pc_idx in range(min(3, n_pcs)):  # Only PC0-PC2 for speed
            direction = pc_directions[pc_idx]
            
            for layer in jacobian_layers:
                bl, mod = run_with_direction_injection(
                    model, tokenizer, prompt, layer, direction, alpha, input_device)
                
                delta_logits = mod - bl
                delta_probs = safe_softmax(mod) - safe_softmax(bl)
                
                jacobian[concept][pc_idx][layer] = {
                    "logit_norm": round(float(np.linalg.norm(delta_logits)), 4),
                    "prob_norm": round(float(np.linalg.norm(delta_probs)), 4),
                    "kl": round(kl_divergence(safe_softmax(mod), safe_softmax(bl)), 6),
                    "js": round(js_divergence(safe_softmax(mod), safe_softmax(bl)), 6),
                }
    
    # ===== Layer-wise causal sensitivity =====
    print("\n  Computing layer-wise causal sensitivity...")
    
    # For each PC, average the causal effect across concepts at each layer
    layer_sensitivity = defaultdict(lambda: defaultdict(list))
    
    for concept in concepts:
        for pc_idx in range(min(3, n_pcs)):
            for layer in jacobian_layers:
                if layer in jacobian[concept][pc_idx]:
                    layer_sensitivity[pc_idx][layer].append(
                        jacobian[concept][pc_idx][layer]["js"])
    
    # Average across concepts
    avg_layer_sensitivity = {}
    for pc_idx in range(min(3, n_pcs)):
        avg_layer_sensitivity[pc_idx] = {}
        for layer in jacobian_layers:
            vals = layer_sensitivity[pc_idx].get(layer, [])
            avg_layer_sensitivity[pc_idx][layer] = round(float(np.mean(vals)), 6) if vals else 0.0
    
    # ===== Cross-layer consistency =====
    # For each PC, is the causal effect at different layers correlated?
    # This tells us if the constraint "propagates" through layers
    
    cross_layer_consistency = {}
    for pc_idx in range(min(3, n_pcs)):
        # For each pair of concepts, compute cos(Δlogits_at_layer1, Δlogits_at_layer2)
        # This requires the full Δlogits vectors, which we didn't store
        # Instead, use the JS values as a proxy
        
        # Compute correlation of JS values across concepts between layers
        js_per_layer = {}
        for layer in jacobian_layers:
            js_vals = [jacobian[c][pc_idx][layer]["js"] 
                       for c in concepts 
                       if layer in jacobian.get(c, {}).get(pc_idx, {})]
            if js_vals:
                js_per_layer[layer] = js_vals
        
        # Correlation between adjacent layers
        consistency_vals = []
        layer_list = sorted(js_per_layer.keys())
        for i in range(len(layer_list) - 1):
            l1, l2 = layer_list[i], layer_list[i + 1]
            min_len = min(len(js_per_layer[l1]), len(js_per_layer[l2]))
            if min_len >= 3:
                corr = np.corrcoef(js_per_layer[l1][:min_len], 
                                    js_per_layer[l2][:min_len])[0, 1]
                consistency_vals.append(round(float(corr), 4))
        
        cross_layer_consistency[pc_idx] = consistency_vals
    
    results = {
        "n_concepts": len(concepts),
        "n_pcs": min(3, n_pcs),
        "jacobian_layers": jacobian_layers,
        "alpha": alpha,
        "avg_layer_sensitivity": {str(k): v for k, v in avg_layer_sensitivity.items()},
        "cross_layer_consistency": {str(k): v for k, v in cross_layer_consistency.items()},
        "per_concept_jacobian": {c: {str(p): v for p, v in pcs.items()} 
                                  for c, pcs in jacobian.items()},
    }
    
    print(f"\n  === Constraint Jacobian Field Results ===")
    for pc_idx in range(min(3, n_pcs)):
        print(f"  PC{pc_idx} layer sensitivity:")
        for layer in jacobian_layers:
            sens = avg_layer_sensitivity[pc_idx].get(layer, 0)
            print(f"    L{layer}: JS={sens:.6f}")
    
    return results


# ===== EXP 4: ICA VS PCA =====

def exp4_ica_vs_pca(model, tokenizer, device, model_name,
                     pc_directions, target_layer, n_concepts=50,
                     n_templates=12, alpha=3.0, n_components=5):
    """
    ★★★ Sparse vs Orthogonal Basis (ICA vs PCA) ★★★
    
    PCA gives orthogonal directions. But real neural features might be:
    - Sparse (most dimensions inactive)
    - Non-orthogonal (correlated)
    - Overcomplete (more features than dimensions)
    
    ICA finds statistically independent components, which might better
    correspond to "real features" than PCA's orthogonal directions.
    
    Test: Compare transport stability (TSS) of ICA vs PCA directions.
    If ICA directions have higher TSS → real features are sparse/independent.
    If PCA directions have higher TSS → real features are orthogonal.
    """
    import torch
    
    print("\n" + "="*60)
    print("EXP 4: ICA vs PCA Feature Basis Comparison")
    print("="*60)
    
    input_device = get_device_for_input(model)
    layers = get_layers(model)
    n_layers = len(layers)
    info = get_model_info(model, model_name)
    
    concepts = ALL_CONCEPTS[:n_concepts]
    templates = CONTEXT_TEMPLATES[:n_templates]
    
    # Step 1: Collect MLP delta matrix (reuse from feature direction computation)
    print(f"  Step 1: Collecting MLP activation delta matrix...")
    
    # Get baseline
    baseline_vec = None
    for template in templates:
        prompt = template.replace("___", "the")
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                       output_hidden_states=True)
        
        h = out.hidden_states[target_layer + 1][0, -1, :].float().cpu().numpy()
        if baseline_vec is None:
            baseline_vec = h
        else:
            baseline_vec += h
        del input_ids, attention_mask, out
    
    baseline_vec /= len(templates)
    
    # Get concept vectors
    concept_vectors = {}
    for cidx, concept in enumerate(concepts):
        if cidx % 10 == 0:
            print(f"    Concept {cidx}/{len(concepts)}")
        
        cvec = None
        for template in templates:
            prompt = template.replace("___", concept)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)
            
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True)
            
            h = out.hidden_states[target_layer + 1][0, -1, :].float().cpu().numpy()
            if cvec is None:
                cvec = h
            else:
                cvec += h
            del input_ids, attention_mask, out
        
        cvec /= len(templates)
        concept_vectors[concept] = cvec - baseline_vec
    
    # Build matrix
    concept_list = sorted(concept_vectors.keys())
    M = np.array([concept_vectors[c] for c in concept_list])
    print(f"  Delta matrix: {M.shape}")
    
    # Step 2: ICA
    print(f"  Step 2: Computing ICA directions...")
    
    try:
        from sklearn.decomposition import FastICA
        
        ica = FastICA(n_components=n_components, random_state=42, max_iter=500)
        S_ica = ica.fit_transform(M)  # [n_concepts, n_components]
        A_ica = ica.mixing_  # [n_components, d_model] — mixing matrix
        
        # ICA components (columns of mixing matrix are the "feature directions")
        # mixing_ has shape [n_features, n_components], so A_ica[:, i] gives direction i
        ica_directions = {}
        ica_sparsity = {}
        
        for i in range(n_components):
            direction = A_ica[:, i]  # [d_model] — column i of mixing matrix
            direction = direction / max(np.linalg.norm(direction), 1e-10)
            ica_directions[i] = direction
            
            # Sparsity: what fraction of dimensions are "near zero"?
            abs_dir = np.abs(direction)
            threshold = 0.1 * np.max(abs_dir)
            sparsity = 1.0 - np.mean(abs_dir > threshold)
            ica_sparsity[i] = round(float(sparsity), 4)
        
        print(f"  ICA sparsity: {ica_sparsity}")
        
    except ImportError:
        print("  WARNING: sklearn not available, skipping ICA")
        ica_directions = {}
        ica_sparsity = {}
        S_ica = None
        A_ica = None
    
    # Step 3: Compare TSS for ICA vs PCA
    print(f"  Step 3: Comparing transport stability (ICA vs PCA)...")
    
    template_ref = "The ___ is"
    test_concepts = concepts[:15]
    
    # PCA TSS (already computed in Exp 1, but recompute for fair comparison)
    pca_tss = {}
    for pc_idx in range(min(n_components, len(pc_directions))):
        deltas = []
        for concept in test_concepts:
            prompt = template_ref.replace("___", concept)
            bl, mod = run_with_direction_injection(
                model, tokenizer, prompt, target_layer, 
                pc_directions[pc_idx], alpha, input_device)
            deltas.append(mod - bl)
        
        cos_vals = []
        for i in range(len(deltas)):
            for j in range(i + 1, len(deltas)):
                cos_vals.append(cosine_sim(deltas[i], deltas[j]))
        pca_tss[pc_idx] = round(float(np.mean(cos_vals)), 4) if cos_vals else 0.0
    
    # ICA TSS
    ica_tss = {}
    if ica_directions:
        for ica_idx in range(min(n_components, len(ica_directions))):
            deltas = []
            for concept in test_concepts:
                prompt = template_ref.replace("___", concept)
                bl, mod = run_with_direction_injection(
                    model, tokenizer, prompt, target_layer,
                    ica_directions[ica_idx], alpha, input_device)
                deltas.append(mod - bl)
            
            cos_vals = []
            for i in range(len(deltas)):
                for j in range(i + 1, len(deltas)):
                    cos_vals.append(cosine_sim(deltas[i], deltas[j]))
            ica_tss[ica_idx] = round(float(np.mean(cos_vals)), 4) if cos_vals else 0.0
    
    # PCA sparsity (for comparison)
    pca_sparsity = {}
    for pc_idx in range(min(n_components, len(pc_directions))):
        abs_dir = np.abs(pc_directions[pc_idx])
        threshold = 0.1 * np.max(abs_dir)
        sparsity = 1.0 - np.mean(abs_dir > threshold)
        pca_sparsity[pc_idx] = round(float(sparsity), 4)
    
    # Step 4: ICA semantic interpretation
    ica_semantics = {}
    if S_ica is not None:
        for ica_idx in range(n_components):
            projections = S_ica[:, ica_idx]
            sorted_idx = np.argsort(projections)
            top5 = [(concept_list[i], round(float(projections[i]), 4)) for i in sorted_idx[-5:]][::-1]
            bottom5 = [(concept_list[i], round(float(projections[i]), 4)) for i in sorted_idx[:5]]
            
            top_families = [CONCEPT_TO_FAMILY[concept_list[i]] for i in sorted_idx[-len(concept_list)//2:]]
            family_counts = defaultdict(int)
            for f in top_families:
                family_counts[f] += 1
            
            ica_semantics[ica_idx] = {
                "top5": top5,
                "bottom5": bottom5,
                "dominant_family": max(family_counts, key=family_counts.get) if family_counts else "unknown",
                "sparsity": ica_sparsity.get(ica_idx, 0),
                "tss": ica_tss.get(ica_idx, 0),
            }
    
    results = {
        "n_concepts": n_concepts,
        "n_components": n_components,
        "target_layer": target_layer,
        "alpha": alpha,
        "pca_tss": pca_tss,
        "pca_sparsity": pca_sparsity,
        "ica_tss": ica_tss,
        "ica_sparsity": ica_sparsity,
        "ica_semantics": ica_semantics,
        "pca_tss_avg": round(float(np.mean(list(pca_tss.values()))), 4) if pca_tss else 0,
        "ica_tss_avg": round(float(np.mean(list(ica_tss.values()))), 4) if ica_tss else 0,
        "pca_sparsity_avg": round(float(np.mean(list(pca_sparsity.values()))), 4) if pca_sparsity else 0,
        "ica_sparsity_avg": round(float(np.mean(list(ica_sparsity.values()))), 4) if ica_sparsity else 0,
    }
    
    print(f"\n  === ICA vs PCA Results ===")
    print(f"  PCA avg TSS: {results['pca_tss_avg']:.4f}, avg sparsity: {results['pca_sparsity_avg']:.4f}")
    print(f"  ICA avg TSS: {results['ica_tss_avg']:.4f}, avg sparsity: {results['ica_sparsity_avg']:.4f}")
    
    for pc_idx in range(min(n_components, len(pc_directions))):
        print(f"  PC{pc_idx}: TSS={pca_tss.get(pc_idx, 0):.4f}, sparsity={pca_sparsity.get(pc_idx, 0):.4f}")
    for ica_idx in range(min(n_components, len(ica_directions))):
        print(f"  ICA{ica_idx}: TSS={ica_tss.get(ica_idx, 0):.4f}, sparsity={ica_sparsity.get(ica_idx, 0):.4f}")
    
    return results


# ===== MAIN =====

def main():
    import torch
    
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    print(f"\n{'='*60}")
    print(f"Phase 167: Causal Constraint Direction Test")
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
        "n_concepts_total": len(ALL_CONCEPTS),
    }
    
    # ===== Step 0: Compute feature directions =====
    print(f"\n[0] Computing feature directions...")
    t0 = time.time()
    pc_directions, sv_info, concept_projections, target_layer = compute_feature_directions(
        model, tokenizer, device, model_name, n_concepts=50, n_templates=12)
    all_results["sv_info"] = sv_info
    print(f"  Time: {time.time()-t0:.1f}s")
    
    # ===== Exp 1: Feature Direction Transport =====
    print(f"\n[1] Exp 1: Feature Direction Transport")
    t0 = time.time()
    all_results["exp1_transport"] = exp1_feature_transport(
        model, tokenizer, device, model_name,
        pc_directions, target_layer, n_concepts=30,
        alphas=[1.0, 3.0, 5.0], n_pcs=5)
    print(f"  Time: {time.time()-t0:.1f}s")
    
    # ===== Exp 2: Activation Patching =====
    print(f"\n[2] Exp 2: Activation Patching")
    t0 = time.time()
    all_results["exp2_patching"] = exp2_activation_patching(
        model, tokenizer, device, model_name, n_concepts=10)
    print(f"  Time: {time.time()-t0:.1f}s")
    
    # ===== Exp 3: Constraint Jacobian Field =====
    print(f"\n[3] Exp 3: Constraint Jacobian Field")
    t0 = time.time()
    all_results["exp3_jacobian"] = exp3_constraint_jacobian(
        model, tokenizer, device, model_name,
        pc_directions, n_concepts=20, alpha=3.0, n_pcs=5)
    print(f"  Time: {time.time()-t0:.1f}s")
    
    # ===== Exp 4: ICA vs PCA =====
    print(f"\n[4] Exp 4: ICA vs PCA")
    t0 = time.time()
    all_results["exp4_ica_vs_pca"] = exp4_ica_vs_pca(
        model, tokenizer, device, model_name,
        pc_directions, target_layer, n_concepts=50,
        n_templates=12, alpha=3.0, n_components=5)
    print(f"  Time: {time.time()-t0:.1f}s")
    
    # ===== Save Results =====
    os.makedirs("tests/glm5_temp", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = f"tests/glm5_temp/phase167_{model_name}_{timestamp}.json"
    
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, tuple):
            return list(obj)
        return obj
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=convert, ensure_ascii=False)
    
    print(f"\n[5] Results saved to: {out_path}")
    
    # Release model
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"\nPhase 167 complete for {model_name}!")


if __name__ == "__main__":
    main()
