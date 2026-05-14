"""
Phase 169: Transport Topology and Local Dynamics — 输运拓扑与局部动力学
=========================================================================

★★★ THE CRITICAL TRANSITION: From "statistical low-dimensional" → "dynamical law"! ★★★

User's key directive: 
  "不要把统计低维结构直接等同于语言本体结构!
   真正的突破将来自: 训练动力学、激活拓扑、约束传播场、局部曲率、稀疏输运图"

Phase 169: Four experiments that probe the DYNAMICAL LAW of language processing!

  Exp 1: ★★★ Propagation Field (Jacobian-Vector Product) — 传播场分析
    - At each layer, inject perturbation along PC0-PC4
    - Measure Δlogits (causal influence on output)
    - Compute: propagation strength, transport stability, direction change
    - KEY: Which directions are AMPLIFIED vs COMPRESSED at each layer?
    - KEY: Is PC0 really the "easiest propagation direction" or a genuine semantic axis?

  Exp 2: ★★★ Neuron Transport Graph — 神经元输运图(真正的拓扑!)
    - For each concept, track which neurons carry concept-specific info
    - Build transport graph: which neurons at layer l feed into which at l+1?
    - Measure: path reuse, family-specific transport, transport continuity
    - KEY: Do same-family concepts share transport PATHS (not just activation)?

  Exp 3: ★★★ Local Curvature at Concept Points — 局部曲率场
    - At concept points, measure ∂²logits/∂h² (deviation from linearity)
    - Compare: concrete concepts vs abstract concepts
    - KEY: Are abstract concepts genuinely "high-curvature regions"?

  Exp 4: ★★★ Layer-wise PC Rotation (Training Dynamics Proxy) — 层间PC旋转
    - Compute PC0 at each layer independently
    - Track how PC0 direction rotates across layers
    - Find "phase transition" layers where PC0 suddenly changes
    - KEY: Is PC0 stable across layers (formed early) or does it rotate?

Usage: python tests/glm5/phase169_transport_topology.py <model_name>
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


# ===== CONCEPT SETS (expanded for robustness) =====

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

# Stratified concept sampling: ensure all families are represented
def get_stratified_concepts(n_per_family=6):
    """Get concepts with equal representation from each family."""
    result = []
    for family, concepts in CONCEPT_FAMILIES.items():
        result.extend(concepts[:n_per_family])
    return result

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


# ===== ROBUST MODEL LOADING =====

def load_model_auto_bf16(model_name):
    """Load model with bfloat16 + device_map='auto' (no 8-bit, avoids NaN).
    
    Falls back to 8-bit if bfloat16+auto fails.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    cfg = MODEL_CONFIGS[model_name]
    
    print(f"[load] Loading {model_name} with bfloat16 + device_map='auto'...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Try bfloat16 + device_map="auto" first (user requested, avoids NaN)
    try:
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[load] GPU memory: {gpu_mem:.1f}GB")
        
        # For Qwen3: fits entirely on GPU
        if model_name == "qwen3":
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"],
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                trust_remote_code=True,
                local_files_only=True,
                low_cpu_mem_usage=True,
                attn_implementation="eager",
            )
            model = model.to("cuda")
        else:
            # For GLM4/DS7B: bfloat16 + device_map="auto" with memory limits
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"],
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_memory={0: "10GiB", "cpu": "30GiB"},
                trust_remote_code=True,
                local_files_only=True,
                low_cpu_mem_usage=True,
                attn_implementation="eager",
            )
        
        model.eval()
        device = next(model.parameters()).device
        gpu_alloc = torch.cuda.memory_allocated() / 1e9
        print(f"[load] {model_name} loaded in bfloat16, device={device}, "
              f"class={type(model).__name__}, GPU={gpu_alloc:.2f}GB")
        use_8bit = False
        
    except Exception as e:
        print(f"[load] bfloat16+auto failed: {e}")
        print(f"[load] Falling back to 8-bit quantization...")
        
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="eager",
        )
        model.eval()
        device = next(model.parameters()).device
        gpu_alloc = torch.cuda.memory_allocated() / 1e9
        print(f"[load] {model_name} loaded in 8-bit, device={device}, "
              f"class={type(model).__name__}, GPU={gpu_alloc:.2f}GB")
        use_8bit = True
    
    return model, tokenizer, device, use_8bit


# ===== UTILITY FUNCTIONS =====

def get_device_for_input(model):
    import torch
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_h(hidden_tensor):
    """Extract hidden state vector, handling NaN from 8bit models."""
    h = hidden_tensor[0, -1, :].float().cpu().numpy()
    return np.nan_to_num(h, nan=0.0, posinf=1e4, neginf=-1e4)


def safe_softmax(logits_np):
    logits_clean = np.nan_to_num(logits_np, nan=0.0, posinf=1e4, neginf=-1e4)
    logits_max = np.max(logits_clean)
    exp_logits = np.exp(logits_clean - logits_max)
    probs = exp_logits / np.sum(exp_logits)
    if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
        probs = np.ones(len(logits_clean)) / len(logits_clean)
    return probs


def cosine_sim(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def js_divergence(p, q, eps=1e-10):
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def run_with_injection(model, tokenizer, prompt, target_layer_idx,
                       direction_np, alpha, input_device, 
                       collect_hidden_states=False, n_layers=None):
    """Run forward pass with direction injection at a specific layer.
    
    Returns:
        baseline_logits, modified_logits
        If collect_hidden_states: also returns baseline and modified hidden states
    """
    import torch
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(input_device)
    attention_mask = inputs["attention_mask"].to(input_device)
    
    layers = get_layers(model)
    
    if collect_hidden_states:
        # Baseline
        with torch.no_grad():
            baseline_out = model(input_ids=input_ids, attention_mask=attention_mask,
                               output_hidden_states=True)
        baseline_logits = baseline_out.logits[0, -1, :].float().cpu().numpy()
        baseline_hs = [safe_h(hs) for hs in baseline_out.hidden_states]
        del baseline_out
        
        # Modified
        direction_tensor = torch.tensor(direction_np, dtype=torch.float32)
        
        def make_inject_hook(direction, alpha_val, device):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    h = output[0].clone()
                    d = direction.to(h.dtype).to(h.device)
                    h[0, -1, :] += alpha_val * d
                    return (h,) + output[1:]
                else:
                    h = output.clone()
                    d = direction.to(h.dtype).to(h.device)
                    h[0, -1, :] += alpha_val * d
                    return h
            return hook
        
        hook = layers[target_layer_idx].register_forward_hook(
            make_inject_hook(direction_tensor, alpha, input_device))
        
        with torch.no_grad():
            modified_out = model(input_ids=input_ids, attention_mask=attention_mask,
                               output_hidden_states=True)
        
        hook.remove()
        modified_logits = modified_out.logits[0, -1, :].float().cpu().numpy()
        modified_hs = [safe_h(hs) for hs in modified_out.hidden_states]
        del modified_out, direction_tensor
        
        del input_ids, attention_mask
        return baseline_logits, modified_logits, baseline_hs, modified_hs
    else:
        # Baseline (no hidden states)
        with torch.no_grad():
            baseline_out = model(input_ids=input_ids, attention_mask=attention_mask)
        baseline_logits = baseline_out.logits[0, -1, :].float().cpu().numpy()
        del baseline_out
        
        # Modified
        direction_tensor = torch.tensor(direction_np, dtype=torch.float32)
        
        def make_inject_hook(direction, alpha_val, device):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    h = output[0].clone()
                    d = direction.to(h.dtype).to(h.device)
                    h[0, -1, :] += alpha_val * d
                    return (h,) + output[1:]
                else:
                    h = output.clone()
                    d = direction.to(h.dtype).to(h.device)
                    h[0, -1, :] += alpha_val * d
                    return h
            return hook
        
        hook = layers[target_layer_idx].register_forward_hook(
            make_inject_hook(direction_tensor, alpha, input_device))
        
        with torch.no_grad():
            modified_out = model(input_ids=input_ids, attention_mask=attention_mask)
        
        hook.remove()
        modified_logits = modified_out.logits[0, -1, :].float().cpu().numpy()
        
        del input_ids, attention_mask, modified_out, direction_tensor
        return baseline_logits, modified_logits


# ===== EXP 1: PROPAGATION FIELD (JACOBIAN-VECTOR PRODUCT) =====

def exp1_propagation_field(model, tokenizer, device, model_name, use_8bit,
                            n_pcs=5, alpha=3.0, n_test_concepts=10):
    """
    ★★★ Propagation Field Analysis — 传播场分析 ★★★
    
    KEY QUESTION: Is PC0 the "easiest propagation direction" or a genuine semantic axis?
    
    Method:
    - At each sampled layer l, inject perturbation α * PC_i
    - Measure Δlogits (effect on final output)
    - Compute:
      1. Propagation strength: ‖Δlogits‖ / α → how strongly does PC_i at layer l affect output?
      2. Transport stability: pairwise cosine of Δlogits across test concepts
      3. Direction change: cosine between Δlogits(PC_i, layer_l) and Δlogits(PC_i, layer_{l+1})
    
    KEY PREDICTIONS:
    - PC0 should have highest propagation strength and transport stability
    - Random directions should have weak propagation and low stability
    - Middle layers (L12-L30) should have strongest PC0 propagation
    """
    import torch
    
    print("\n" + "="*60)
    print("EXP 1: Propagation Field (Jacobian-Vector Product)")
    print("="*60)
    
    input_device = get_device_for_input(model)
    layers = get_layers(model)
    n_layers = len(layers)
    info = get_model_info(model, model_name)
    d_model = info.d_model
    
    # Sample layers for injection
    n_sample = min(8, n_layers)
    sample_layers = list(range(0, n_layers, max(1, n_layers // n_sample))) + [n_layers - 1]
    sample_layers = sorted(set(sample_layers))
    
    print(f"  Sample layers: {sample_layers}")
    print(f"  Testing {n_pcs} PCs × {len(sample_layers)} layers × {n_test_concepts} concepts")
    
    # Step 1: Compute PC directions at 2/3 layer
    target_layer = int(n_layers * 0.67)
    templates = CONTEXT_TEMPLATES[:6]
    
    print(f"  Step 1: Computing PC directions at layer {target_layer}...")
    
    # Baseline
    baseline_vec = None
    for template in templates:
        prompt = template.replace("___", "the")
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                       output_hidden_states=True)
        h = safe_h(out.hidden_states[target_layer + 1])
        if baseline_vec is None:
            baseline_vec = h
        else:
            baseline_vec += h
        del input_ids, attention_mask, out
    baseline_vec /= len(templates)
    
    # Concept vectors
    test_concepts = get_stratified_concepts(n_per_family=2)  # 10 concepts from all families
    concept_vectors = {}
    for cidx, concept in enumerate(test_concepts):
        cvec = None
        for template in templates:
            prompt = template.replace("___", concept)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True)
            h = safe_h(out.hidden_states[target_layer + 1])
            if cvec is None:
                cvec = h
            else:
                cvec += h
            del input_ids, attention_mask, out
        cvec /= len(templates)
        concept_vectors[concept] = cvec - baseline_vec
    
    # PCA
    concept_list = sorted(concept_vectors.keys())
    M = np.array([concept_vectors[c] for c in concept_list])
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    
    pc_dirs = {}
    for i in range(min(n_pcs, len(S))):
        pc_dirs[i] = Vt[i] / max(np.linalg.norm(Vt[i]), 1e-10)
    
    # Also create random directions as control
    n_random = 5
    random_dirs = {}
    for i in range(n_random):
        rd = np.random.randn(d_model)
        rd = rd / np.linalg.norm(rd)
        random_dirs[i] = rd
    
    print(f"  PC directions computed: top1_ratio={S[0]**2/np.sum(S**2):.4f}")
    
    # Step 2: Propagation field measurement
    print(f"  Step 2: Measuring propagation field...")
    
    ref_prompt = "The ___ is".replace("___", "cat")
    
    # For each PC direction at each layer, measure Δlogits
    propagation_results = {}
    
    all_directions = {f"PC{i}": pc_dirs[i] for i in range(min(n_pcs, len(pc_dirs)))}
    all_directions.update({f"random{i}": random_dirs[i] for i in range(n_random)})
    
    for dir_name, direction in all_directions.items():
        dir_results = {}
        
        for lidx, layer_l in enumerate(sample_layers):
            if lidx % 2 == 0:
                print(f"    {dir_name}, layer {layer_l} ({lidx+1}/{len(sample_layers)})")
            
            # Test on multiple concepts for transport stability
            concept_deltas = []
            for concept in test_concepts[:5]:  # Use 5 concepts for speed
                prompt = f"The {concept} is"
                bl, mod = run_with_injection(
                    model, tokenizer, prompt, layer_l, direction, alpha, input_device)
                concept_deltas.append(mod - bl)
            
            # Propagation strength: average ‖Δlogits‖ across concepts
            delta_norms = [np.linalg.norm(d) for d in concept_deltas]
            avg_strength = float(np.mean(delta_norms))
            
            # Transport stability: pairwise cosine of Δlogits across concepts
            cos_pairs = []
            for i in range(len(concept_deltas)):
                for j in range(i + 1, len(concept_deltas)):
                    cos_pairs.append(cosine_sim(concept_deltas[i], concept_deltas[j]))
            avg_tss = float(np.mean(cos_pairs)) if cos_pairs else 0.0
            
            dir_results[layer_l] = {
                "strength": round(avg_strength, 4),
                "tss": round(avg_tss, 4),
                "delta_norms": [round(float(n), 4) for n in delta_norms],
            }
        
        propagation_results[dir_name] = dir_results
    
    # Step 3: Direction change analysis
    print(f"  Step 3: Computing direction change between consecutive layers...")
    
    direction_changes = {}
    for dir_name in all_directions:
        changes = []
        for i in range(len(sample_layers) - 1):
            l1 = sample_layers[i]
            l2 = sample_layers[i + 1]
            
            # Get Δlogits at both layers for a reference concept
            prompt = "The cat is"
            bl1, mod1 = run_with_injection(
                model, tokenizer, prompt, l1, all_directions[dir_name], alpha, input_device)
            bl2, mod2 = run_with_injection(
                model, tokenizer, prompt, l2, all_directions[dir_name], alpha, input_device)
            
            delta1 = mod1 - bl1
            delta2 = mod2 - bl2
            
            # Direction change: cosine between effects at consecutive layers
            dir_change = cosine_sim(delta1, delta2)
            changes.append(round(dir_change, 4))
        
        direction_changes[dir_name] = changes
    
    # Step 4: Summary
    print(f"\n  === Propagation Field Results ===")
    
    # Compare PC0 vs random
    pc0_strengths = [propagation_results["PC0"][l]["strength"] for l in sample_layers]
    random_strengths = []
    for i in range(n_random):
        rs = [propagation_results[f"random{i}"][l]["strength"] for l in sample_layers]
        random_strengths.append(float(np.mean(rs)))
    avg_random_strength = float(np.mean(random_strengths))
    
    pc0_tss = [propagation_results["PC0"][l]["tss"] for l in sample_layers]
    random_tss = []
    for i in range(n_random):
        rt = [propagation_results[f"random{i}"][l]["tss"] for l in sample_layers]
        random_tss.append(float(np.mean(rt)))
    avg_random_tss = float(np.mean(random_tss))
    
    # PC0 vs other PCs
    pc_strengths = {}
    pc_tss = {}
    for i in range(min(n_pcs, len(pc_dirs))):
        s = [propagation_results[f"PC{i}"][l]["strength"] for l in sample_layers]
        t = [propagation_results[f"PC{i}"][l]["tss"] for l in sample_layers]
        pc_strengths[f"PC{i}"] = round(float(np.mean(s)), 4)
        pc_tss[f"PC{i}"] = round(float(np.mean(t)), 4)
    
    print(f"  PC0 avg strength: {np.mean(pc0_strengths):.4f}")
    print(f"  Random avg strength: {avg_random_strength:.4f}")
    print(f"  PC0 / Random strength ratio: {np.mean(pc0_strengths)/max(avg_random_strength,1e-6):.2f}×")
    print(f"  PC0 avg TSS: {np.mean(pc0_tss):.4f}")
    print(f"  Random avg TSS: {avg_random_tss:.4f}")
    print(f"  PC0 / Random TSS ratio: {np.mean(pc0_tss)/max(avg_random_tss,1e-6):.2f}×")
    
    # Layer-by-layer analysis
    print(f"\n  Layer-by-layer propagation strength:")
    for l in sample_layers:
        pc0_s = propagation_results["PC0"][l]["strength"]
        pc0_t = propagation_results["PC0"][l]["tss"]
        rs = float(np.mean([propagation_results[f"random{i}"][l]["strength"] for i in range(n_random)]))
        rt = float(np.mean([propagation_results[f"random{i}"][l]["tss"] for i in range(n_random)]))
        print(f"    L{l}: PC0_str={pc0_s:.4f}, rand_str={rs:.4f}, "
              f"ratio={pc0_s/max(rs,1e-6):.2f}×, "
              f"PC0_tss={pc0_t:.4f}, rand_tss={rt:.4f}, "
              f"tss_ratio={pc0_t/max(rt,1e-6):.2f}×")
    
    results = {
        "pc_vs_random": {
            "pc0_avg_strength": round(float(np.mean(pc0_strengths)), 4),
            "random_avg_strength": round(avg_random_strength, 4),
            "strength_ratio": round(float(np.mean(pc0_strengths)) / max(avg_random_strength, 1e-6), 2),
            "pc0_avg_tss": round(float(np.mean(pc0_tss)), 4),
            "random_avg_tss": round(avg_random_tss, 4),
            "tss_ratio": round(float(np.mean(pc0_tss)) / max(avg_random_tss, 1e-6), 2),
        },
        "pc_comparison": {
            "strengths": pc_strengths,
            "tss": pc_tss,
        },
        "propagation_by_layer": {
            l: {
                "PC0_strength": propagation_results["PC0"][l]["strength"],
                "PC0_tss": propagation_results["PC0"][l]["tss"],
                "random_strength": round(float(np.mean([propagation_results[f"random{i}"][l]["strength"] for i in range(n_random)])), 4),
                "random_tss": round(float(np.mean([propagation_results[f"random{i}"][l]["tss"] for i in range(n_random)])), 4),
            }
            for l in sample_layers
        },
        "direction_changes": direction_changes,
        "sample_layers": sample_layers,
        "n_test_concepts": n_test_concepts,
    }
    
    return results


# ===== EXP 2: NEURON TRANSPORT GRAPH =====

def exp2_transport_graph(model, tokenizer, device, model_name, use_8bit,
                          n_concepts=50, n_templates=6, top_k=50):
    """
    ★★★ Neuron Transport Graph — 神经元输运图 ★★★
    
    KEY QUESTION: Do same-family concepts share transport PATHS (not just activation)?
    
    Method:
    - For each concept, compute concept-specific delta (h_concept - h_baseline) at each layer
    - Find top-k neurons carrying concept-specific information at each layer
    - Build "transport edge": (layer l, neuron i) → (layer l+1, neuron j) 
      if both are in top-k for the same concept
    - Measure: transport continuity, family-specific transport overlap
    - KEY: Do same-family concepts share MORE transport edges than cross-family?
    
    This is the REAL topology: which neurons FORM PROPAGATION PATHS together!
    """
    import torch
    
    print("\n" + "="*60)
    print("EXP 2: Neuron Transport Graph")
    print("="*60)
    
    input_device = get_device_for_input(model)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    
    concepts = ALL_CONCEPTS[:n_concepts]
    templates = CONTEXT_TEMPLATES[:n_templates]
    
    # Sample layers (every 2-3 layers for efficiency)
    sample_layers = list(range(0, n_layers + 1, max(1, n_layers // 8)))
    sample_layers = sorted(set(sample_layers))
    if n_layers not in sample_layers:
        sample_layers.append(n_layers)
    
    print(f"  Testing {len(concepts)} concepts × {len(templates)} templates, top_k={top_k}")
    print(f"  Sample layers: {sample_layers}")
    
    # Step 1: Collect concept-specific deltas at each layer
    print(f"  Step 1: Collecting concept-specific deltas...")
    
    # First, collect baseline hidden states
    print(f"    Collecting baseline...")
    baseline_h = {l: None for l in sample_layers}
    for template in templates:
        prompt = template.replace("___", "the")
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                       output_hidden_states=True)
        
        for l in sample_layers:
            h = safe_h(out.hidden_states[l])
            if baseline_h[l] is None:
                baseline_h[l] = h
            else:
                baseline_h[l] += h
        del input_ids, attention_mask, out
    
    for l in sample_layers:
        baseline_h[l] /= len(templates)
    
    # Then, collect concept hidden states
    print(f"    Collecting concepts...")
    concept_deltas = {}  # {concept: {layer: delta_vector}}
    
    for cidx, concept in enumerate(concepts):
        if cidx % 10 == 0:
            print(f"      Concept {cidx}/{len(concepts)}: {concept}")
        
        concept_h = {l: None for l in sample_layers}
        for template in templates:
            prompt = template.replace("___", concept)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True)
            
            for l in sample_layers:
                h = safe_h(out.hidden_states[l])
                if concept_h[l] is None:
                    concept_h[l] = h
                else:
                    concept_h[l] += h
            del input_ids, attention_mask, out
        
        for l in sample_layers:
            concept_h[l] /= len(templates)
        
        # Delta = concept_h - baseline_h
        concept_deltas[concept] = {l: concept_h[l] - baseline_h[l] for l in sample_layers}
    
    # Step 2: Find top-k neurons per concept per layer
    print(f"  Step 2: Finding top-{top_k} neurons per concept per layer...")
    
    concept_top_neurons = {}  # {concept: {layer: set of top neuron indices}}
    for concept in concepts:
        concept_top_neurons[concept] = {}
        for l in sample_layers:
            delta = concept_deltas[concept][l]
            # Top-k by absolute value (these neurons carry concept-specific info)
            top_indices = set(np.argsort(np.abs(delta))[-top_k:])
            concept_top_neurons[concept][l] = top_indices
    
    # Step 3: Transport continuity analysis
    print(f"  Step 3: Computing transport continuity between consecutive layers...")
    
    # For each concept, measure overlap of top-k neurons between consecutive layers
    transport_continuity = {}  # {layer_pair: list of Jaccard indices}
    
    for li in range(len(sample_layers) - 1):
        l1 = sample_layers[li]
        l2 = sample_layers[li + 1]
        key = f"L{l1}_L{l2}"
        
        jaccards = []
        for concept in concepts:
            n1 = concept_top_neurons[concept].get(l1, set())
            n2 = concept_top_neurons[concept].get(l2, set())
            if n1 and n2:
                j = len(n1 & n2) / max(len(n1 | n2), 1)
                jaccards.append(j)
        
        transport_continuity[key] = {
            "avg_jaccard": round(float(np.mean(jaccards)), 4) if jaccards else 0,
            "layers": (l1, l2),
        }
    
    # Step 4: Same-family vs cross-family transport overlap
    print(f"  Step 4: Computing family-specific transport overlap...")
    
    same_family_transport = defaultdict(list)
    diff_family_transport = defaultdict(list)
    
    for li in range(len(sample_layers) - 1):
        l1 = sample_layers[li]
        l2 = sample_layers[li + 1]
        key = f"L{l1}_L{l2}"
        
        for i, c1 in enumerate(concepts):
            n1_l1 = concept_top_neurons[c1].get(l1, set())
            n1_l2 = concept_top_neurons[c1].get(l2, set())
            if not n1_l1 or not n1_l2:
                continue
                
            for j, c2 in enumerate(concepts):
                if i >= j:
                    continue
                n2_l1 = concept_top_neurons[c2].get(l1, set())
                n2_l2 = concept_top_neurons[c2].get(l2, set())
                if not n2_l1 or not n2_l2:
                    continue
                
                # Transport overlap: overlap of concept-specific paths
                # Path for c1: n1_l1 → n1_l2
                # Path for c2: n2_l1 → n2_l2
                # Transport overlap = overlap at both layers
                overlap_at_l1 = len(n1_l1 & n2_l1) / max(len(n1_l1 | n2_l1), 1)
                overlap_at_l2 = len(n1_l2 & n2_l2) / max(len(n1_l2 | n2_l2), 1)
                transport_overlap = (overlap_at_l1 + overlap_at_l2) / 2
                
                if CONCEPT_TO_FAMILY[c1] == CONCEPT_TO_FAMILY[c2]:
                    same_family_transport[key].append(transport_overlap)
                else:
                    diff_family_transport[key].append(transport_overlap)
    
    # Aggregate
    family_transport_results = {}
    for li in range(len(sample_layers) - 1):
        l1 = sample_layers[li]
        l2 = sample_layers[li + 1]
        key = f"L{l1}_L{l2}"
        
        same_avg = float(np.mean(same_family_transport[key])) if same_family_transport[key] else 0
        diff_avg = float(np.mean(diff_family_transport[key])) if diff_family_transport[key] else 0
        
        family_transport_results[key] = {
            "same_family_overlap": round(same_avg, 4),
            "diff_family_overlap": round(diff_avg, 4),
            "advantage": round(same_avg / max(diff_avg, 1e-6), 2),
            "layers": (l1, l2),
        }
    
    # Step 5: Neuron reusability analysis
    print(f"  Step 5: Computing neuron reusability...")
    
    # How many unique neurons are used across all concepts at each layer?
    neuron_reusability = {}
    for l in sample_layers:
        all_neurons = set()
        per_concept_neurons = []
        
        for concept in concepts:
            top_n = concept_top_neurons[concept].get(l, set())
            all_neurons |= top_n
            per_concept_neurons.append(top_n)
        
        n_unique = len(all_neurons)
        max_possible = top_k * len(concepts)
        
        # "Reuse rate" = 1 - (n_unique / max_possible)
        # High reuse → neurons are shared across concepts
        # Low reuse → each concept uses mostly unique neurons
        reuse_rate = 1.0 - (n_unique / max(max_possible, 1))
        
        # Neuron "importance concentration": what fraction of concepts use each neuron?
        neuron_usage_count = defaultdict(int)
        for top_n in per_concept_neurons:
            for n in top_n:
                neuron_usage_count[n] += 1
        
        # Top-10 most reused neurons
        most_reused = sorted(neuron_usage_count.items(), key=lambda x: -x[1])[:10]
        avg_reuse_count = float(np.mean(list(neuron_usage_count.values()))) if neuron_usage_count else 0
        
        neuron_reusability[l] = {
            "n_unique_neurons": n_unique,
            "reuse_rate": round(reuse_rate, 4),
            "avg_reuse_count": round(avg_reuse_count, 4),
            "top10_neuron_usage": [(int(n), int(c)) for n, c in most_reused],
        }
    
    # Print summary
    print(f"\n  === Transport Graph Results ===")
    print(f"  Transport continuity (Jaccard between consecutive layers):")
    for key, tc in transport_continuity.items():
        print(f"    {key}: avg_jaccard={tc['avg_jaccard']:.4f}")
    
    print(f"\n  Family transport overlap:")
    for key, ft in family_transport_results.items():
        print(f"    {key}: same={ft['same_family_overlap']:.4f}, "
              f"diff={ft['diff_family_overlap']:.4f}, "
              f"advantage={ft['advantage']:.2f}×")
    
    print(f"\n  Neuron reusability:")
    for l in sample_layers:
        nr = neuron_reusability[l]
        print(f"    L{l}: reuse_rate={nr['reuse_rate']:.4f}, "
              f"unique={nr['n_unique_neurons']}, "
              f"avg_reuse={nr['avg_reuse_count']:.2f}")
    
    results = {
        "transport_continuity": transport_continuity,
        "family_transport_overlap": family_transport_results,
        "neuron_reusability": neuron_reusability,
        "sample_layers": sample_layers,
        "top_k": top_k,
    }
    
    return results


# ===== EXP 3: LOCAL CURVATURE AT CONCEPT POINTS =====

def exp3_local_curvature(model, tokenizer, device, model_name, use_8bit,
                          n_concepts=30, alphas=None, target_layer_frac=0.67):
    """
    ★★★ Local Curvature at Concept Points — 局部曲率场 ★★★
    
    KEY QUESTION: Are abstract concepts genuinely "high-curvature regions"?
    
    Method:
    - At concept points, inject perturbation at multiple scales: ε, 2ε, 3ε
    - Measure Δlogits at each scale
    - If perfectly linear: Δlogits(2ε) = 2 × Δlogits(ε)
    - Curvature = ‖Δlogits(nε) - n × Δlogits(ε)‖ / ‖n × Δlogits(ε)‖
    - High curvature → strong nonlinearity → complex local geometry
    
    KEY TEST: Compare curvature for concrete (animal/vehicle/color) vs 
              abstract (abstract/emotion) concepts
    """
    import torch
    
    print("\n" + "="*60)
    print("EXP 3: Local Curvature at Concept Points")
    print("="*60)
    
    if alphas is None:
        alphas = [1.0, 2.0, 3.0, 5.0]
    
    input_device = get_device_for_input(model)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    target_layer = int(n_layers * target_layer_frac)
    
    # Use STRATIFIED sampling to ensure ALL families are represented!
    concepts = get_stratified_concepts(n_per_family=6)  # 30 concepts from all 5 families
    templates = CONTEXT_TEMPLATES[:4]  # Fewer templates for speed
    
    print(f"  Testing {len(concepts)} concepts × {len(alphas)} alphas at layer {target_layer}")
    
    # Step 1: Compute PC0 direction
    print(f"  Step 1: Computing PC0 direction...")
    
    baseline_vec = None
    for template in templates:
        prompt = template.replace("___", "the")
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                       output_hidden_states=True)
        h = safe_h(out.hidden_states[target_layer + 1])
        if baseline_vec is None:
            baseline_vec = h
        else:
            baseline_vec += h
        del input_ids, attention_mask, out
    baseline_vec /= len(templates)
    
    concept_vectors = {}
    for cidx, concept in enumerate(concepts):
        cvec = None
        for template in templates:
            prompt = template.replace("___", concept)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True)
            h = safe_h(out.hidden_states[target_layer + 1])
            if cvec is None:
                cvec = h
            else:
                cvec += h
            del input_ids, attention_mask, out
        cvec /= len(templates)
        concept_vectors[concept] = cvec - baseline_vec
    
    M = np.array([concept_vectors[c] for c in sorted(concept_vectors.keys())])
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    pc0 = Vt[0] / max(np.linalg.norm(Vt[0]), 1e-10)
    
    print(f"  PC0 computed: top1_ratio={S[0]**2/np.sum(S**2):.4f}")
    
    # Step 2: Dose-response for each concept
    print(f"  Step 2: Computing dose-response for each concept...")
    
    concept_curvature = {}
    
    for cidx, concept in enumerate(concepts):
        if cidx % 10 == 0:
            print(f"    Concept {cidx}/{len(concepts)}: {concept}")
        
        prompt = f"The {concept} is"
        
        # Get Δlogits at each alpha
        delta_logits = {}
        for alpha_val in alphas:
            bl, mod = run_with_injection(
                model, tokenizer, prompt, target_layer, pc0, alpha_val, input_device)
            delta_logits[alpha_val] = mod - bl
        
        # Compute curvature: deviation from linearity
        # If linear: Δlogits(α) ∝ α, so Δlogits(2ε) / Δlogits(ε) = 2
        # Curvature = ‖Δlogits(2ε) - 2×Δlogits(ε)‖ / ‖2×Δlogits(ε)‖
        
        # Reference: Δlogits at α=1
        ref_delta = delta_logits[alphas[0]]
        ref_norm = np.linalg.norm(ref_delta)
        
        curvatures = {}
        for alpha_val in alphas[1:]:
            actual = delta_logits[alpha_val]
            expected_linear = alpha_val * ref_delta
            deviation = actual - expected_linear
            
            # Relative curvature = ‖deviation‖ / ‖expected_linear‖
            expected_norm = np.linalg.norm(expected_linear)
            relative_curvature = np.linalg.norm(deviation) / max(expected_norm, 1e-10)
            
            curvatures[alpha_val] = round(relative_curvature, 4)
        
        # Also compute: does Δlogits point in the same direction at different α?
        direction_changes = {}
        for alpha_val in alphas[1:]:
            cos_change = cosine_sim(delta_logits[alphas[0]], delta_logits[alpha_val])
            direction_changes[alpha_val] = round(cos_change, 4)
        
        concept_curvature[concept] = {
            "family": CONCEPT_TO_FAMILY[concept],
            "curvatures": curvatures,
            "direction_changes": direction_changes,
            "ref_delta_norm": round(float(ref_norm), 4),
        }
    
    # Step 3: Compare concrete vs abstract
    print(f"  Step 3: Comparing concrete vs abstract concepts...")
    
    concrete_families = {"animal", "vehicle", "color"}
    abstract_families = {"abstract", "emotion"}
    
    concrete_curvatures = defaultdict(list)
    abstract_curvatures = defaultdict(list)
    
    for concept, cc in concept_curvature.items():
        family = cc["family"]
        for alpha_val in alphas[1:]:
            curv = cc["curvatures"].get(alpha_val, 0)
            if family in concrete_families:
                concrete_curvatures[alpha_val].append(curv)
            else:
                abstract_curvatures[alpha_val].append(curv)
    
    concrete_vs_abstract = {}
    for alpha_val in alphas[1:]:
        c_avg = float(np.mean(concrete_curvatures[alpha_val])) if concrete_curvatures[alpha_val] else 0
        a_avg = float(np.mean(abstract_curvatures[alpha_val])) if abstract_curvatures[alpha_val] else 0
        ratio = a_avg / max(c_avg, 1e-6)
        
        concrete_vs_abstract[alpha_val] = {
            "concrete_avg_curvature": round(c_avg, 4),
            "abstract_avg_curvature": round(a_avg, 4),
            "abstract_concrete_ratio": round(ratio, 2),
        }
        
        print(f"    α={alpha_val}: concrete={c_avg:.4f}, abstract={a_avg:.4f}, "
              f"ratio={ratio:.2f}×")
    
    # Step 4: JS divergence of Δlogits (how different is the effect?)
    print(f"  Step 4: Computing JS divergence of perturbation effects...")
    
    concept_js = {}
    ref_alpha = alphas[1]  # Use α=2
    
    for cidx, concept in enumerate(concepts):
        prompt = f"The {concept} is"
        bl, mod = run_with_injection(
            model, tokenizer, prompt, target_layer, pc0, ref_alpha, input_device)
        delta_probs = safe_softmax(mod) - safe_softmax(bl)
        delta_probs = np.abs(delta_probs)
        delta_probs = delta_probs / max(delta_probs.sum(), 1e-10)
        
        concept_js[concept] = {
            "family": CONCEPT_TO_FAMILY[concept],
            "delta_entropy": round(float(-np.sum(delta_probs * np.log(delta_probs + 1e-10))), 4),
        }
    
    # Compare entropy
    concrete_entropies = [concept_js[c]["delta_entropy"] for c in concepts 
                          if CONCEPT_TO_FAMILY[c] in concrete_families]
    abstract_entropies = [concept_js[c]["delta_entropy"] for c in concepts 
                          if CONCEPT_TO_FAMILY[c] in abstract_families]
    
    print(f"  Concrete avg delta entropy: {np.mean(concrete_entropies):.4f}")
    print(f"  Abstract avg delta entropy: {np.mean(abstract_entropies):.4f}")
    
    results = {
        "concept_curvature": {
            c: {
                "family": cc["family"],
                "curvatures": cc["curvatures"],
                "direction_changes": cc["direction_changes"],
                "ref_delta_norm": cc["ref_delta_norm"],
            }
            for c, cc in concept_curvature.items()
        },
        "concrete_vs_abstract": concrete_vs_abstract,
        "delta_entropy": {
            "concrete_avg": round(float(np.mean(concrete_entropies)), 4) if concrete_entropies else 0,
            "abstract_avg": round(float(np.mean(abstract_entropies)), 4) if abstract_entropies else 0,
            "ratio": round(float(np.mean(abstract_entropies)) / max(float(np.mean(concrete_entropies)), 1e-6), 2) if concrete_entropies and abstract_entropies else 0,
        },
        "alphas": alphas,
        "target_layer": target_layer,
    }
    
    return results


# ===== EXP 4: LAYER-WISE PC ROTATION (TRAINING DYNAMICS PROXY) =====

def exp4_pc_rotation(model, tokenizer, device, model_name, use_8bit,
                     n_concepts=50, n_templates=6):
    """
    ★★★ Layer-wise PC Rotation — 层间PC旋转(训练动力学代理) ★★★
    
    KEY QUESTION: Is PC0 stable across layers (formed early in training) 
                  or does it rotate (formed differently at different layers)?
    
    Method:
    - Compute concept deltas at EACH layer
    - PCA at each layer → get PC0 direction per layer
    - Measure rotation angle between PC0 at consecutive layers
    - Find "phase transition" layers where PC0 suddenly rotates
    
    INTERPRETATION:
    - If PC0 is stable (small rotation) → "PC0 is a fundamental axis, formed early"
    - If PC0 rotates significantly → "different layers use different coordinate systems"
    - Phase transition layers → layers where the network undergoes structural change
    
    This is a PROXY for training dynamics because:
    - Early layers ≈ early training (basic feature extraction)
    - Late layers ≈ late training (high-level semantic organization)
    """
    import torch
    
    print("\n" + "="*60)
    print("EXP 4: Layer-wise PC Rotation (Training Dynamics Proxy)")
    print("="*60)
    
    input_device = get_device_for_input(model)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    
    concepts = ALL_CONCEPTS[:n_concepts]
    templates = CONTEXT_TEMPLATES[:n_templates]
    
    # Sample layers
    sample_layers = list(range(0, n_layers + 1, max(1, n_layers // 8)))
    sample_layers = sorted(set(sample_layers))
    if n_layers not in sample_layers:
        sample_layers.append(n_layers)
    
    print(f"  Testing {len(concepts)} concepts × {len(sample_layers)} layers")
    print(f"  Sample layers: {sample_layers}")
    
    # Step 1: Collect concept deltas at each layer
    print(f"  Step 1: Collecting concept deltas at each layer...")
    
    # Baseline
    print(f"    Baseline...")
    baseline_h = {l: None for l in sample_layers}
    for template in templates:
        prompt = template.replace("___", "the")
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                       output_hidden_states=True)
        
        for l in sample_layers:
            h = safe_h(out.hidden_states[l])
            if baseline_h[l] is None:
                baseline_h[l] = h
            else:
                baseline_h[l] += h
        del input_ids, attention_mask, out
    
    for l in sample_layers:
        baseline_h[l] /= len(templates)
    
    # Concepts
    print(f"    Concepts...")
    concept_deltas_at_layer = {l: {} for l in sample_layers}
    
    for cidx, concept in enumerate(concepts):
        if cidx % 10 == 0:
            print(f"      Concept {cidx}/{len(concepts)}: {concept}")
        
        concept_h = {l: None for l in sample_layers}
        for template in templates:
            prompt = template.replace("___", concept)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True)
            
            for l in sample_layers:
                h = safe_h(out.hidden_states[l])
                if concept_h[l] is None:
                    concept_h[l] = h
                else:
                    concept_h[l] += h
            del input_ids, attention_mask, out
        
        for l in sample_layers:
            concept_h[l] /= len(templates)
            concept_deltas_at_layer[l][concept] = concept_h[l] - baseline_h[l]
    
    # Step 2: PCA at each layer
    print(f"  Step 2: Computing PCA at each layer...")
    
    layer_pcs = {}  # {layer: {pc0_direction, top1_ratio, rank_90, ...}}
    
    for l in sample_layers:
        M = np.array([concept_deltas_at_layer[l][c] for c in concepts])
        M = np.nan_to_num(M, nan=0.0, posinf=1e4, neginf=-1e4)
        
        try:
            U, S, Vt = np.linalg.svd(M, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        
        total = np.sum(S ** 2)
        if total < 1e-10:
            continue
        
        cumvar = np.cumsum(S ** 2) / total
        rank_90 = int(np.searchsorted(cumvar, 0.90) + 1)
        top1_ratio = float(S[0] ** 2 / total)
        
        pc0_dir = Vt[0] / max(np.linalg.norm(Vt[0]), 1e-10)
        pc1_dir = Vt[1] / max(np.linalg.norm(Vt[1]), 1e-10) if len(S) > 1 else np.zeros_like(pc0_dir)
        
        layer_pcs[l] = {
            "pc0": pc0_dir,
            "pc1": pc1_dir,
            "top1_ratio": round(top1_ratio, 4),
            "rank_90": rank_90,
            "top3_ratio": round(float(np.sum(S[:3] ** 2) / total), 4) if len(S) >= 3 else 0,
        }
        
        print(f"    L{l}: top1={top1_ratio:.4f}, rank_90={rank_90}, "
              f"top3={layer_pcs[l]['top3_ratio']:.4f}")
    
    # Step 3: PC0 rotation between consecutive layers
    print(f"  Step 3: Computing PC0 rotation angles...")
    
    rotation_angles = []
    phase_transitions = []
    
    sorted_layers = sorted(layer_pcs.keys())
    
    for i in range(len(sorted_layers) - 1):
        l1 = sorted_layers[i]
        l2 = sorted_layers[i + 1]
        
        cos_angle = cosine_sim(layer_pcs[l1]["pc0"], layer_pcs[l2]["pc0"])
        angle_rad = np.arccos(np.clip(cos_angle, -1, 1))
        angle_deg = np.degrees(angle_rad)
        
        rotation_angles.append({
            "from_layer": l1,
            "to_layer": l2,
            "cos_sim": round(cos_angle, 4),
            "angle_rad": round(float(angle_rad), 4),
            "angle_deg": round(float(angle_deg), 2),
        })
        
        # Phase transition: rotation > 30°
        if angle_deg > 30:
            phase_transitions.append({
                "from_layer": l1,
                "to_layer": l2,
                "angle_deg": round(float(angle_deg), 2),
            })
        
        print(f"    L{l1}→L{l2}: cos={cos_angle:.4f}, angle={angle_deg:.1f}°"
              + (" ← PHASE TRANSITION!" if angle_deg > 30 else ""))
    
    # Step 4: PC1 rotation (for comparison)
    print(f"  Step 4: Computing PC1 rotation angles...")
    
    pc1_rotation = []
    for i in range(len(sorted_layers) - 1):
        l1 = sorted_layers[i]
        l2 = sorted_layers[i + 1]
        
        cos_angle = cosine_sim(layer_pcs[l1]["pc1"], layer_pcs[l2]["pc1"])
        angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        
        pc1_rotation.append({
            "from_layer": l1,
            "to_layer": l2,
            "cos_sim": round(cos_angle, 4),
            "angle_deg": round(float(angle_deg), 2),
        })
    
    # Step 5: Evolution of top1_ratio across layers
    print(f"  Step 5: Evolution of top1_ratio across layers...")
    
    top1_evolution = {l: layer_pcs[l]["top1_ratio"] for l in sorted_layers}
    rank90_evolution = {l: layer_pcs[l]["rank_90"] for l in sorted_layers}
    
    # Find the layer where top1_ratio is maximum
    max_top1_layer = max(top1_evolution, key=top1_evolution.get)
    min_rank90_layer = min(rank90_evolution, key=rank90_evolution.get)
    
    print(f"  Max top1_ratio at L{max_top1_layer}: {top1_evolution[max_top1_layer]:.4f}")
    print(f"  Min rank_90 at L{min_rank90_layer}: {rank90_evolution[min_rank90_layer]}")
    
    results = {
        "layer_pcs_summary": {
            l: {
                "top1_ratio": layer_pcs[l]["top1_ratio"],
                "rank_90": layer_pcs[l]["rank_90"],
                "top3_ratio": layer_pcs[l]["top3_ratio"],
            }
            for l in sorted_layers
        },
        "pc0_rotation": rotation_angles,
        "pc1_rotation": pc1_rotation,
        "phase_transitions": phase_transitions,
        "top1_evolution": top1_evolution,
        "rank90_evolution": rank90_evolution,
        "max_top1_layer": max_top1_layer,
        "min_rank90_layer": min_rank90_layer,
    }
    
    return results


# ===== MAIN =====

def main():
    import torch
    
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    print(f"\n{'='*60}")
    print(f"Phase 169: Transport Topology and Local Dynamics")
    print(f"Model: {model_name}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")
    
    # Load model (bfloat16 + device_map="auto", fallback to 8-bit)
    model, tokenizer, device, use_8bit = load_model_auto_bf16(model_name)
    info = get_model_info(model, model_name)
    print(f"[0] Model: {info.model_class}, L={info.n_layers}, d={info.d_model}, "
          f"V={info.vocab_size}, 8bit={use_8bit}")
    
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
    
    # ===== Exp 1: Propagation Field =====
    print(f"\n[1] Exp 1: Propagation Field (Jacobian-Vector Product)")
    t0 = time.time()
    all_results["exp1_propagation"] = exp1_propagation_field(
        model, tokenizer, device, model_name, use_8bit,
        n_pcs=5, alpha=3.0, n_test_concepts=10)
    print(f"  Time: {time.time()-t0:.1f}s")
    
    # ===== Exp 2: Transport Graph =====
    print(f"\n[2] Exp 2: Neuron Transport Graph")
    t0 = time.time()
    all_results["exp2_transport"] = exp2_transport_graph(
        model, tokenizer, device, model_name, use_8bit,
        n_concepts=50, n_templates=6, top_k=50)
    print(f"  Time: {time.time()-t0:.1f}s")
    
    # ===== Exp 3: Local Curvature =====
    print(f"\n[3] Exp 3: Local Curvature at Concept Points")
    t0 = time.time()
    all_results["exp3_curvature"] = exp3_local_curvature(
        model, tokenizer, device, model_name, use_8bit,
        n_concepts=30, alphas=[1.0, 2.0, 3.0, 5.0])
    print(f"  Time: {time.time()-t0:.1f}s")
    
    # ===== Exp 4: PC Rotation =====
    print(f"\n[4] Exp 4: Layer-wise PC Rotation")
    t0 = time.time()
    all_results["exp4_rotation"] = exp4_pc_rotation(
        model, tokenizer, device, model_name, use_8bit,
        n_concepts=50, n_templates=6)
    print(f"  Time: {time.time()-t0:.1f}s")
    
    # ===== Save Results =====
    os.makedirs("tests/glm5_temp", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = f"tests/glm5_temp/phase169_{model_name}_{timestamp}.json"
    
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
    
    print(f"\nPhase 169 complete for {model_name}!")


if __name__ == "__main__":
    main()
