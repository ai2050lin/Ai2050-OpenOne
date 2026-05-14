"""
Phase 170: Dynamical Invariants — 动力不变量
=============================================

★★★ THE CRITICAL TRANSITION: From "rotating coordinates" → "rotation-invariant structure"! ★★★

User's key insight:
  "PC0 rotates 40-80° per layer, but CCA≈1 → information preserved
   → Language structure exists in ROTATION-INVARIANT constraints, not in specific directions!"

  "Language ≠ feature extraction. Language = constrained transport on a manifold."

  "The real breakthrough: flow, connection, transport, curvature, invariance, holonomy"

Four critical experiments:

  Exp 1: ★★★ Parallel Transport (JVP) — 平行输运测试 [MOST IMPORTANT]
    - At layer l, take concept direction v_l
    - Compute J_l @ v_l (Jacobian-Vector Product via perturbation)
    - Compare with "natural" direction δ_{l+1} at layer l+1
    - Transport fidelity = cos(J_l v_l, δ_{l+1})
    - KEY: Does the Jacobian preserve semantic directions?
    - This reveals: the CONNECTION of the fiber bundle

  Exp 2: ★★★ Jacobian Spectrum — 雅可比谱
    - Compute top-k singular values of J_l via randomized SVD
    - Singular values: amplification, compression, mixing
    - Isometric ratio = σ_min / σ_max → how close to rotation?
    - KEY: Is the system contractive, expansive, or isometric?
    - This reveals: the LOCAL DYNAMICAL LAW

  Exp 3: ★★★ Metric Preservation & Holonomy — 度量保持与完整旋绕
    - Compute Gram matrix G_l = Δ_l @ Δ_l^T at each layer
    - If G_l ≈ G_{l+1} → isometric → no curvature
    - If G_l ≠ G_{l+1} → find WHAT is preserved (gauge invariants)
    - Compute angle preservation between concept pairs across layers
    - KEY: What distance/angle relationships survive coordinate rotation?
    - This reveals: the CURVATURE and HOLONOMY of the constraint manifold

  Exp 4: ★★★ Semantic Gauge Invariants — 语义规范不变量
    - At each layer, compute concept rankings by distance/angle
    - Find relations preserved across ALL layers
    - These "gauge-invariant" structures = the TRUE mathematical structure of language
    - KEY: What semantic structure survives ALL coordinate rotations?
    - This reveals: the INVARIANT CORE of language

Usage: python tests/glm5/phase170_dynamical_invariants.py <model_name>
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

from model_utils import get_layers, get_model_info, release_model, MODEL_CONFIGS


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

def get_stratified_concepts(n_per_family=10):
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
]


# ===== ROBUST MODEL LOADING (BF16 + device_map="auto") =====

def load_model_auto_bf16(model_name):
    """Load model with bfloat16 + device_map='auto' (no 8-bit, avoids NaN)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = MODEL_CONFIGS[model_name]
    print(f"[load] Loading {model_name} with bfloat16 + device_map='auto'...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[load] GPU memory: {gpu_mem:.1f}GB")

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
    """Extract hidden state vector, handling NaN."""
    h = hidden_tensor[0, -1, :].float().cpu().numpy()
    return np.nan_to_num(h, nan=0.0, posinf=1e4, neginf=-1e4)


def cosine_sim(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


# ===== DATA COLLECTION: Concept deltas at all layers =====

def collect_concept_deltas(model, tokenizer, device, model_name, use_8bit,
                           concepts, templates, sample_layers):
    """
    Collect concept deltas (h_concept - h_baseline) at all sampled layers.

    Returns:
        baseline_h: {layer: numpy array (d_model,)}
        concept_h: {concept: {layer: numpy array (d_model,)}}
        concept_deltas: {concept: {layer: numpy array (d_model,)}}
    """
    import torch

    input_device = get_device_for_input(model)
    info = get_model_info(model, model_name)

    # Step 1: Collect baseline hidden states
    print(f"  Collecting baseline hidden states ({len(templates)} templates)...")
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

    # Step 2: Collect concept hidden states
    print(f"  Collecting concept hidden states ({len(concepts)} concepts × {len(templates)} templates)...")
    concept_h = {}

    for cidx, concept in enumerate(concepts):
        if cidx % 10 == 0:
            print(f"    Concept {cidx}/{len(concepts)}: {concept}")

        ch = {l: None for l in sample_layers}
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
                if ch[l] is None:
                    ch[l] = h
                else:
                    ch[l] += h
            del input_ids, attention_mask, out

        for l in sample_layers:
            ch[l] /= len(templates)
        concept_h[concept] = ch

    # Step 3: Compute concept deltas
    concept_deltas = {}
    for concept in concepts:
        concept_deltas[concept] = {
            l: concept_h[concept][l] - baseline_h[l] for l in sample_layers
        }

    return baseline_h, concept_h, concept_deltas


# ===== EXP 1: PARALLEL TRANSPORT (JACOBIAN-VECTOR PRODUCT) =====

def exp1_parallel_transport(model, tokenizer, device, model_name, use_8bit,
                            concepts, concept_deltas, sample_layers, epsilon=1.0):
    """
    ★★★ Parallel Transport Test — 平行输运测试 ★★★

    KEY QUESTION: Does the Jacobian preserve semantic directions?

    Method:
    - At layer l, inject concept direction v_l = Δ_c(l) / ‖Δ_c(l)‖
    - Compute Jacobian-Vector Product: J_l @ v_l ≈ (h_{l+1}^perturbed - h_{l+1}^baseline) / ε
    - Compare with "natural" direction at l+1: δ_{l+1} = Δ_c(l+1) / ‖Δ_c(l+1)‖
    - Transport fidelity = cos(J_l @ v_l, δ_{l+1})

    INTERPRETATION:
    - If fidelity ≈ 1 → Jacobian correctly predicts concept direction evolution → "flat connection"
    - If fidelity < 1 → Concept direction is NOT simply transported → "curved connection"
    - The difference reveals the CONNECTION of the fiber bundle
    """
    import torch

    print("\n" + "="*60)
    print("EXP 1: Parallel Transport (Jacobian-Vector Product)")
    print("="*60)

    input_device = get_device_for_input(model)
    layers = get_layers(model)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers

    # Select test concepts (2 per family = 10)
    test_concepts = get_stratified_concepts(n_per_family=2)

    # Select injection layers (every 4 layers, excluding first and last)
    inject_layers = [l for l in sample_layers if 1 <= l <= n_layers - 2]
    inject_layers = inject_layers[::2]  # Every other sampled layer
    if len(inject_layers) > 10:
        inject_layers = inject_layers[::len(inject_layers)//10 + 1]

    print(f"  Test concepts: {test_concepts}")
    print(f"  Injection layers: {inject_layers}")
    print(f"  Epsilon: {epsilon}")

    # For each injection layer, compute baseline hidden state at NEXT layer
    print(f"  Step 1: Computing baseline hidden states...")

    # First, get baseline hidden states at all layers
    baseline_at_layer = {}
    ref_prompt = "The cat is"
    inputs = tokenizer(ref_prompt, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(input_device)
    attention_mask = inputs["attention_mask"].to(input_device)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                   output_hidden_states=True)
    for l in sample_layers:
        baseline_at_layer[l] = safe_h(out.hidden_states[l])
    del input_ids, attention_mask, out

    # Step 2: For each injection layer, compute JVP for concept directions
    print(f"  Step 2: Computing Jacobian-Vector Products...")

    transport_results = {}

    for lidx, layer_l in enumerate(inject_layers):
        # Find the next sampled layer
        next_layers = [l for l in sample_layers if l > layer_l]
        if not next_layers:
            continue
        layer_next = next_layers[0]

        print(f"    Layer {layer_l} → {layer_next} ({lidx+1}/{len(inject_layers)})")

        layer_results = {
            "concept_fidelity": {},
            "random_fidelity": {},
        }

        # Test concept directions
        for concept in test_concepts:
            # Get concept direction at layer_l (normalized)
            v_l = concept_deltas[concept].get(layer_l)
            if v_l is None:
                continue
            v_l_norm = np.linalg.norm(v_l)
            if v_l_norm < 1e-10:
                continue
            v_l_unit = v_l / v_l_norm

            # Natural direction at layer_next (normalized)
            v_next = concept_deltas[concept].get(layer_next)
            if v_next is None:
                continue
            v_next_norm = np.linalg.norm(v_next)
            if v_next_norm < 1e-10:
                continue
            v_next_unit = v_next / v_next_norm

            # Compute JVP: inject ε*v_l_unit at layer_l, measure Δh at layer_next
            # We need to map layer_l to transformer layer index
            # hidden_states[layer_l] is output of transformer layer (layer_l - 1)
            # So injecting at "layer_l" means hooking transformer layer (layer_l - 1)
            # But wait: sample_layers index into hidden_states directly
            # hidden_states[k] = output after processing through k layers (0=embedding)
            # To inject into the residual stream at position layer_l:
            #   - We hook the transformer layer that outputs hidden_states[layer_l]
            #   - That's transformer layer (layer_l - 1) if layer_l > 0
            #   - For layer_l = 0 (embedding), we can't inject via hook

            if layer_l == 0:
                # Can't inject at embedding via hook, skip
                continue

            transformer_layer_idx = layer_l - 1  # Transformer layer index
            if transformer_layer_idx >= len(layers):
                continue

            # Inject perturbation
            direction_tensor = torch.tensor(v_l_unit, dtype=torch.float32)

            def make_inject_hook(direction, eps, dev):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0].clone()
                        d = direction.to(h.dtype).to(h.device)
                        h[0, -1, :] += eps * d
                        return (h,) + output[1:]
                    else:
                        h = output.clone()
                        d = direction.to(h.dtype).to(h.device)
                        h[0, -1, :] += eps * d
                        return h
                return hook

            hook = layers[transformer_layer_idx].register_forward_hook(
                make_inject_hook(direction_tensor, epsilon, input_device))

            # Run perturbed forward pass
            inputs = tokenizer(ref_prompt, return_tensors="pt", truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)
            with torch.no_grad():
                out_perturbed = model(input_ids=input_ids, attention_mask=attention_mask,
                                     output_hidden_states=True)

            hook.remove()

            # Get perturbed hidden state at layer_next
            h_next_perturbed = safe_h(out_perturbed.hidden_states[layer_next])
            del input_ids, attention_mask, out_perturbed, direction_tensor

            # Jacobian-Vector Product: J_l @ v_l ≈ (h_perturbed - h_baseline) / ε
            jvp = (h_next_perturbed - baseline_at_layer[layer_next]) / epsilon
            jvp_norm = np.linalg.norm(jvp)
            if jvp_norm < 1e-10:
                continue
            jvp_unit = jvp / jvp_norm

            # Transport fidelity
            fidelity = cosine_sim(jvp_unit, v_next_unit)

            # Also measure: how much does the direction change?
            # Direction change = angle between injected direction and its Jacobian image
            direction_change = cosine_sim(v_l_unit, jvp_unit)

            # Norm ratio: how much is the direction amplified/compressed?
            norm_ratio = jvp_norm / v_l_norm if v_l_norm > 1e-10 else 0

            layer_results["concept_fidelity"][concept] = {
                "fidelity": round(fidelity, 4),
                "direction_change": round(direction_change, 4),
                "norm_ratio": round(norm_ratio, 4),
                "family": CONCEPT_TO_FAMILY[concept],
            }

        # Test random directions for baseline
        n_random = 5
        random_fidelities = []
        for r_idx in range(n_random):
            rd = np.random.randn(info.d_model)
            rd = rd / np.linalg.norm(rd)

            direction_tensor = torch.tensor(rd, dtype=torch.float32)

            def make_inject_hook_r(direction, eps, dev):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0].clone()
                        d = direction.to(h.dtype).to(h.device)
                        h[0, -1, :] += eps * d
                        return (h,) + output[1:]
                    else:
                        h = output.clone()
                        d = direction.to(h.dtype).to(h.device)
                        h[0, -1, :] += eps * d
                        return h
                return hook

            hook = layers[transformer_layer_idx].register_forward_hook(
                make_inject_hook_r(direction_tensor, epsilon, input_device))

            inputs = tokenizer(ref_prompt, return_tensors="pt", truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)
            with torch.no_grad():
                out_perturbed = model(input_ids=input_ids, attention_mask=attention_mask,
                                     output_hidden_states=True)

            hook.remove()

            h_next_perturbed = safe_h(out_perturbed.hidden_states[layer_next])
            del input_ids, attention_mask, out_perturbed, direction_tensor

            jvp_r = (h_next_perturbed - baseline_at_layer[layer_next]) / epsilon
            jvp_r_norm = np.linalg.norm(jvp_r)
            if jvp_r_norm > 1e-10:
                jvp_r_unit = jvp_r / jvp_r_norm
                # For random direction, measure norm ratio (amplification)
                random_fidelities.append({
                    "norm_ratio": round(jvp_r_norm, 4),
                    "jvp_norm": round(jvp_r_norm, 4),
                })

        layer_results["random_fidelity"] = random_fidelities

        # Summary for this layer
        concept_fids = [v["fidelity"] for v in layer_results["concept_fidelity"].values()]
        concept_dir_changes = [v["direction_change"] for v in layer_results["concept_fidelity"].values()]
        concept_norm_ratios = [v["norm_ratio"] for v in layer_results["concept_fidelity"].values()]

        avg_fidelity = float(np.mean(concept_fids)) if concept_fids else 0
        avg_dir_change = float(np.mean(concept_dir_changes)) if concept_dir_changes else 0
        avg_norm_ratio = float(np.mean(concept_norm_ratios)) if concept_norm_ratios else 0

        layer_results["summary"] = {
            "avg_fidelity": round(avg_fidelity, 4),
            "avg_direction_change": round(avg_dir_change, 4),
            "avg_norm_ratio": round(avg_norm_ratio, 4),
            "n_concepts": len(concept_fids),
        }

        print(f"      avg_fidelity={avg_fidelity:.4f}, avg_dir_change={avg_dir_change:.4f}, "
              f"avg_norm_ratio={avg_norm_ratio:.4f}")

        transport_results[f"L{layer_l}_L{layer_next}"] = layer_results

    # Overall summary
    all_fidelities = []
    concrete_fidelities = []
    abstract_fidelities = []
    for key, lr in transport_results.items():
        for concept, cf in lr["concept_fidelity"].items():
            all_fidelities.append(cf["fidelity"])
            if cf["family"] in {"animal", "vehicle", "color"}:
                concrete_fidelities.append(cf["fidelity"])
            else:
                abstract_fidelities.append(cf["fidelity"])

    print(f"\n  === Parallel Transport Results ===")
    print(f"  Overall avg fidelity: {np.mean(all_fidelities):.4f}")
    print(f"  Concrete avg fidelity: {np.mean(concrete_fidelities):.4f}")
    print(f"  Abstract avg fidelity: {np.mean(abstract_fidelities):.4f}")
    print(f"  Abstract/Concrete ratio: {np.mean(abstract_fidelities)/max(np.mean(concrete_fidelities),1e-6):.2f}×")

    results = {
        "transport_by_layer": transport_results,
        "overall": {
            "avg_fidelity": round(float(np.mean(all_fidelities)), 4) if all_fidelities else 0,
            "concrete_avg_fidelity": round(float(np.mean(concrete_fidelities)), 4) if concrete_fidelities else 0,
            "abstract_avg_fidelity": round(float(np.mean(abstract_fidelities)), 4) if abstract_fidelities else 0,
            "abstract_concrete_ratio": round(float(np.mean(abstract_fidelities)) / max(float(np.mean(concrete_fidelities)), 1e-6), 2) if concrete_fidelities and abstract_fidelities else 0,
        },
        "epsilon": epsilon,
        "test_concepts": test_concepts,
    }

    return results


# ===== EXP 2: JACOBIAN SPECTRUM =====

def exp2_jacobian_spectrum(model, tokenizer, device, model_name, use_8bit,
                           sample_layers, n_random=20, epsilon=1.0):
    """
    ★★★ Jacobian Spectrum — 雅可比谱 ★★★

    KEY QUESTION: Is the system contractive, expansive, or isometric?

    Method:
    - At each layer, compute J @ r_i for n_random random directions r_i
    - Use randomized SVD: Y = [J@r_1, ..., J@r_k], then SVD(Y) gives top singular values
    - Singular values σ_i tell us:
      - σ_i > 1 → direction i is AMPLIFIED
      - σ_i < 1 → direction i is COMPRESSED
      - σ_i ≈ 1 → direction i is PRESERVED (isometric)
    - Isometric ratio = σ_min / σ_max → how close to rotation?
    - Condition number = σ_max / σ_min → how "distorted" is the transformation?

    INTERPRETATION:
    - If σ_i ≈ 1 for all i → isometric (near-rotation) → information-preserving transport
    - If σ_max >> 1 → some directions amplified → potential attractor
    - If σ_min << 1 → some directions compressed → information loss
    - If condition_number >> 1 → anisotropic → strong direction-dependent processing
    """
    import torch

    print("\n" + "="*60)
    print("EXP 2: Jacobian Spectrum (via Randomized SVD)")
    print("="*60)

    input_device = get_device_for_input(model)
    layers = get_layers(model)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model

    # Select injection layers
    inject_layers = [l for l in sample_layers if 1 <= l <= n_layers - 2]
    inject_layers = inject_layers[::2]  # Every other sampled layer
    if len(inject_layers) > 8:
        inject_layers = inject_layers[::len(inject_layers)//8 + 1]

    print(f"  Injection layers: {inject_layers}")
    print(f"  Random directions: {n_random}")
    print(f"  Epsilon: {epsilon}")

    spectrum_results = {}

    for lidx, layer_l in enumerate(inject_layers):
        # Find the next sampled layer
        next_layers = [l for l in sample_layers if l > layer_l]
        if not next_layers:
            continue
        layer_next = next_layers[0]

        if layer_l == 0:
            continue

        transformer_layer_idx = layer_l - 1
        if transformer_layer_idx >= len(layers):
            continue

        print(f"  Layer {layer_l} → {layer_next} ({lidx+1}/{len(inject_layers)})")

        # Step 1: Compute baseline hidden state at layer_next
        ref_prompt = "The cat is"
        inputs = tokenizer(ref_prompt, return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        with torch.no_grad():
            out_baseline = model(input_ids=input_ids, attention_mask=attention_mask,
                                output_hidden_states=True)
        h_baseline = safe_h(out_baseline.hidden_states[layer_next])
        del input_ids, attention_mask, out_baseline

        # Step 2: Compute J@r_i for random directions
        jvp_matrix = np.zeros((d_model, n_random))  # Each column is J@r_i

        for r_idx in range(n_random):
            r = np.random.randn(d_model)
            r = r / np.linalg.norm(r)

            direction_tensor = torch.tensor(r, dtype=torch.float32)

            def make_inject_hook(direction, eps, dev):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0].clone()
                        d = direction.to(h.dtype).to(h.device)
                        h[0, -1, :] += eps * d
                        return (h,) + output[1:]
                    else:
                        h = output.clone()
                        d = direction.to(h.dtype).to(h.device)
                        h[0, -1, :] += eps * d
                        return h
                return hook

            hook = layers[transformer_layer_idx].register_forward_hook(
                make_inject_hook(direction_tensor, epsilon, input_device))

            inputs = tokenizer(ref_prompt, return_tensors="pt", truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)
            with torch.no_grad():
                out_perturbed = model(input_ids=input_ids, attention_mask=attention_mask,
                                     output_hidden_states=True)

            hook.remove()

            h_perturbed = safe_h(out_perturbed.hidden_states[layer_next])
            del input_ids, attention_mask, out_perturbed, direction_tensor

            jvp = (h_perturbed - h_baseline) / epsilon
            jvp_matrix[:, r_idx] = jvp

        # Step 3: Randomized SVD of jvp_matrix
        # jvp_matrix shape: (d_model, n_random)
        # SVD gives: U (d_model, n_random), S (n_random,), Vt (n_random, n_random)
        try:
            U, S, Vt = np.linalg.svd(jvp_matrix, full_matrices=False)
        except np.linalg.LinAlgError:
            print(f"    SVD failed for layer {layer_l}")
            continue

        # S are the singular values of the Jacobian restricted to the random subspace
        # These approximate the top singular values of the full Jacobian

        # Compute key metrics
        s_max = float(S[0]) if len(S) > 0 else 0
        s_min = float(S[-1]) if len(S) > 0 else 0
        s_median = float(np.median(S)) if len(S) > 0 else 0
        condition_number = s_max / max(s_min, 1e-10)

        # Effective rank (number of singular values > 0.1 * s_max)
        eff_rank = int(np.sum(S > 0.1 * s_max)) if s_max > 0 else 0

        # Isometric ratio: what fraction of singular values are close to 1?
        isometric_count = int(np.sum(np.abs(S - 1.0) < 0.3)) if len(S) > 0 else 0
        isometric_ratio = isometric_count / max(len(S), 1)

        # Amplified vs compressed
        amplified = int(np.sum(S > 1.3))
        compressed = int(np.sum(S < 0.7))
        preserved = len(S) - amplified - compressed

        # Transport type classification
        if isometric_ratio > 0.7:
            transport_type = "ISOMETRIC (near-rotation)"
        elif condition_number > 10:
            transport_type = "ANISOTROPIC (strong direction dependence)"
        elif s_median > 1.5:
            transport_type = "EXPANSIVE"
        elif s_median < 0.5:
            transport_type = "CONTRACTIVE"
        else:
            transport_type = "MIXED"

        print(f"    σ_max={s_max:.2f}, σ_min={s_min:.2f}, cond={condition_number:.2f}, "
              f"eff_rank={eff_rank}, isometric={isometric_ratio:.2f}, type={transport_type}")

        spectrum_results[f"L{layer_l}_L{layer_next}"] = {
            "singular_values": [round(float(s), 4) for s in S[:10]],  # Top 10
            "s_max": round(s_max, 4),
            "s_min": round(s_min, 4),
            "s_median": round(s_median, 4),
            "condition_number": round(condition_number, 4),
            "effective_rank": eff_rank,
            "isometric_ratio": round(isometric_ratio, 4),
            "amplified_count": amplified,
            "compressed_count": compressed,
            "preserved_count": preserved,
            "transport_type": transport_type,
            "layers": (layer_l, layer_next),
        }

    # Overall summary
    print(f"\n  === Jacobian Spectrum Results ===")
    for key, sr in spectrum_results.items():
        print(f"    {key}: cond={sr['condition_number']:.2f}, "
              f"isometric={sr['isometric_ratio']:.2f}, type={sr['transport_type']}")

    results = {
        "spectrum_by_layer": spectrum_results,
        "n_random": n_random,
        "epsilon": epsilon,
    }

    return results


# ===== EXP 3: METRIC PRESERVATION & HOLONOMY =====

def exp3_metric_preservation(concept_deltas, concepts, sample_layers):
    """
    ★★★ Metric Preservation & Holonomy — 度量保持与完整旋绕 ★★★

    KEY QUESTION: What distance/angle relationships survive coordinate rotation?

    Method:
    - Compute Gram matrix G_l = Δ_l @ Δ_l^T at each layer (C × C)
    - G_l captures ALL pairwise relationships between concepts
    - Compare G_l across layers:
      a. Matrix correlation: corr(vec(G_l), vec(G_{l+1}))
      b. Eigenvalue spectrum: how do eigenvalues of G_l change?
      c. Pairwise distance matrix: how do concept distances change?
    - Compute angle preservation between concept pairs across layers
    - If G_l ≈ G_{l+1} → isometric → no curvature
    - If G_l ≠ G_{l+1} → find WHAT is preserved (gauge invariants)

    INTERPRETATION:
    - High matrix correlation → inner product structure preserved → isometric
    - Low matrix correlation → metric changes → curvature exists
    - Preserved rankings → gauge-invariant structure
    - Angle changes → holonomy (curvature in the fiber bundle)
    """
    print("\n" + "="*60)
    print("EXP 3: Metric Preservation & Holonomy")
    print("="*60)

    n_concepts = len(concepts)
    concept_list = sorted(concepts)

    # Step 1: Compute Gram matrix at each layer
    print(f"  Step 1: Computing Gram matrices at {len(sample_layers)} layers...")

    gram_matrices = {}
    for l in sample_layers:
        # Build delta matrix: (n_concepts, d_model)
        M = np.array([concept_deltas[c][l] for c in concept_list])
        M = np.nan_to_num(M, nan=0.0, posinf=1e4, neginf=-1e4)

        # Gram matrix: (n_concepts, n_concepts)
        G = M @ M.T
        gram_matrices[l] = G

    # Step 2: Matrix correlation between consecutive layers
    print(f"  Step 2: Computing matrix correlations...")

    sorted_layers = sorted(sample_layers)
    matrix_correlations = []

    for i in range(len(sorted_layers) - 1):
        l1 = sorted_layers[i]
        l2 = sorted_layers[i + 1]

        G1 = gram_matrices[l1].flatten()
        G2 = gram_matrices[l2].flatten()

        # Pearson correlation
        corr = np.corrcoef(G1, G2)[0, 1]
        if np.isnan(corr):
            corr = 0.0

        matrix_correlations.append({
            "from_layer": l1,
            "to_layer": l2,
            "correlation": round(float(corr), 4),
        })

        print(f"    L{l1}→L{l2}: corr={corr:.4f}")

    # Step 3: Eigenvalue spectrum evolution
    print(f"  Step 3: Computing eigenvalue spectrum evolution...")

    eigenvalue_evolution = {}
    for l in sorted_layers:
        G = gram_matrices[l]
        try:
            eigenvalues = np.linalg.eigvalsh(G)
            eigenvalues = np.sort(eigenvalues)[::-1]  # Descending
            total = np.sum(np.abs(eigenvalues))
            if total > 0:
                cumvar = np.cumsum(np.abs(eigenvalues)) / total
                rank_90 = int(np.searchsorted(cumvar, 0.90) + 1)
            else:
                rank_90 = 0

            eigenvalue_evolution[l] = {
                "top5": [round(float(e), 4) for e in eigenvalues[:5]],
                "rank_90": rank_90,
                "total_energy": round(float(total), 4),
                "top1_ratio": round(float(eigenvalues[0] / max(total, 1e-10)), 4),
            }
            print(f"    L{l}: rank_90={rank_90}, top1_ratio={eigenvalue_evolution[l]['top1_ratio']:.4f}")
        except np.linalg.LinAlgError:
            eigenvalue_evolution[l] = {"error": "SVD failed"}

    # Step 4: Pairwise distance matrix and its preservation
    print(f"  Step 4: Computing pairwise distance preservation...")

    distance_matrices = {}
    for l in sorted_layers:
        M = np.array([concept_deltas[c][l] for c in concept_list])
        M = np.nan_to_num(M, nan=0.0, posinf=1e4, neginf=-1e4)

        # Pairwise cosine distances
        n = len(concept_list)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                cos = cosine_sim(M[i], M[j])
                D[i, j] = 1 - cos  # Cosine distance
                D[j, i] = D[i, j]
        distance_matrices[l] = D

    # Distance matrix correlation between consecutive layers
    distance_correlations = []
    for i in range(len(sorted_layers) - 1):
        l1 = sorted_layers[i]
        l2 = sorted_layers[i + 1]

        # Use upper triangle only (excluding diagonal)
        D1 = distance_matrices[l1]
        D2 = distance_matrices[l2]
        mask = np.triu(np.ones_like(D1, dtype=bool), k=1)
        d1 = D1[mask]
        d2 = D2[mask]

        corr = np.corrcoef(d1, d2)[0, 1]
        if np.isnan(corr):
            corr = 0.0

        distance_correlations.append({
            "from_layer": l1,
            "to_layer": l2,
            "correlation": round(float(corr), 4),
        })

        print(f"    L{l1}→L{l2}: distance_corr={corr:.4f}")

    # Step 5: Angle preservation between concept pairs
    print(f"  Step 5: Computing angle preservation...")

    # For each concept pair, track the angle across layers
    pair_angle_trajectories = {}
    concrete_families = {"animal", "vehicle", "color"}
    abstract_families = {"abstract", "emotion"}

    for i, c1 in enumerate(concept_list):
        for j, c2 in enumerate(concept_list):
            if i >= j:
                continue
            pair_key = f"{c1}-{c2}"
            family_pair = (CONCEPT_TO_FAMILY[c1], CONCEPT_TO_FAMILY[c2])
            same_family = CONCEPT_TO_FAMILY[c1] == CONCEPT_TO_FAMILY[c2]

            angles = []
            for l in sorted_layers:
                v1 = concept_deltas[c1][l]
                v2 = concept_deltas[c2][l]
                cos = cosine_sim(v1, v2)
                angle_deg = np.degrees(np.arccos(np.clip(cos, -1, 1)))
                angles.append(round(angle_deg, 2))

            # Angle variance across layers (low = preserved)
            angle_var = float(np.var(angles))

            pair_angle_trajectories[pair_key] = {
                "angles": angles,
                "angle_variance": round(angle_var, 2),
                "same_family": same_family,
                "family_pair": family_pair,
            }

    # Step 6: Holonomy test - angle change for same-family vs cross-family pairs
    print(f"  Step 6: Computing holonomy (angle change patterns)...")

    same_family_angle_vars = []
    cross_family_angle_vars = []
    concrete_pair_angle_vars = []
    abstract_pair_angle_vars = []
    mixed_pair_angle_vars = []

    for pair_key, pat in pair_angle_trajectories.items():
        if pat["same_family"]:
            same_family_angle_vars.append(pat["angle_variance"])
        else:
            cross_family_angle_vars.append(pat["angle_variance"])

        f1, f2 = pat["family_pair"]
        c1 = f1 in concrete_families
        c2 = f2 in concrete_families
        a1 = f1 in abstract_families
        a2 = f2 in abstract_families

        if c1 and c2:
            concrete_pair_angle_vars.append(pat["angle_variance"])
        elif a1 and a2:
            abstract_pair_angle_vars.append(pat["angle_variance"])
        else:
            mixed_pair_angle_vars.append(pat["angle_variance"])

    holonomy_results = {
        "same_family_avg_angle_var": round(float(np.mean(same_family_angle_vars)), 4) if same_family_angle_vars else 0,
        "cross_family_avg_angle_var": round(float(np.mean(cross_family_angle_vars)), 4) if cross_family_angle_vars else 0,
        "concrete_pair_avg_angle_var": round(float(np.mean(concrete_pair_angle_vars)), 4) if concrete_pair_angle_vars else 0,
        "abstract_pair_avg_angle_var": round(float(np.mean(abstract_pair_angle_vars)), 4) if abstract_pair_angle_vars else 0,
        "mixed_pair_avg_angle_var": round(float(np.mean(mixed_pair_angle_vars)), 4) if mixed_pair_angle_vars else 0,
        "same_cross_ratio": round(float(np.mean(same_family_angle_vars)) / max(float(np.mean(cross_family_angle_vars)), 1e-6), 2) if same_family_angle_vars and cross_family_angle_vars else 0,
    }

    print(f"    Same-family angle variance: {holonomy_results['same_family_avg_angle_var']:.4f}")
    print(f"    Cross-family angle variance: {holonomy_results['cross_family_avg_angle_var']:.4f}")
    print(f"    Same/Cross ratio: {holonomy_results['same_cross_ratio']:.2f}")
    print(f"    Concrete pair angle variance: {holonomy_results['concrete_pair_avg_angle_var']:.4f}")
    print(f"    Abstract pair angle variance: {holonomy_results['abstract_pair_avg_angle_var']:.4f}")

    # Step 7: Metric change rate (curvature measure)
    print(f"  Step 7: Computing curvature (metric change rate)...")

    curvature_by_layer = []
    for i in range(len(sorted_layers) - 1):
        l1 = sorted_layers[i]
        l2 = sorted_layers[i + 1]

        G1 = gram_matrices[l1]
        G2 = gram_matrices[l2]

        # Relative change in Gram matrix
        diff_norm = np.linalg.norm(G2 - G1)
        base_norm = np.linalg.norm(G1)
        relative_change = diff_norm / max(base_norm, 1e-10)

        curvature_by_layer.append({
            "from_layer": l1,
            "to_layer": l2,
            "relative_change": round(float(relative_change), 4),
            "absolute_change": round(float(diff_norm), 4),
        })

        print(f"    L{l1}→L{l2}: curvature={relative_change:.4f}")

    # Summary
    print(f"\n  === Metric Preservation Results ===")
    avg_gram_corr = float(np.mean([mc["correlation"] for mc in matrix_correlations]))
    avg_dist_corr = float(np.mean([dc["correlation"] for dc in distance_correlations]))
    avg_curvature = float(np.mean([cb["relative_change"] for cb in curvature_by_layer]))

    print(f"  Avg Gram matrix correlation: {avg_gram_corr:.4f}")
    print(f"  Avg distance matrix correlation: {avg_dist_corr:.4f}")
    print(f"  Avg curvature (metric change rate): {avg_curvature:.4f}")

    results = {
        "matrix_correlations": matrix_correlations,
        "distance_correlations": distance_correlations,
        "eigenvalue_evolution": eigenvalue_evolution,
        "holonomy": holonomy_results,
        "curvature_by_layer": curvature_by_layer,
        "pair_angle_trajectories_sample": {k: v for k, v in list(pair_angle_trajectories.items())[:20]},
        "summary": {
            "avg_gram_correlation": round(avg_gram_corr, 4),
            "avg_distance_correlation": round(avg_dist_corr, 4),
            "avg_curvature": round(avg_curvature, 4),
        },
    }

    return results


# ===== EXP 4: SEMANTIC GAUGE INVARIANTS =====

def exp4_gauge_invariants(concept_deltas, concepts, sample_layers):
    """
    ★★★ Semantic Gauge Invariants — 语义规范不变量 ★★★

    KEY QUESTION: What semantic structure survives ALL coordinate rotations?

    Method:
    - At each layer, compute concept rankings by various metrics:
      a. Distance from origin (‖Δ_c‖) → "concept salience"
      b. Pairwise distance ranking → "concept topology"
      c. Within-family vs between-family separation → "cluster structure"
    - Find rankings/relations preserved across ALL layers
    - These "gauge-invariant" structures = the TRUE mathematical structure of language

    INTERPRETATION:
    - Preserved salience ranking → "some concepts are always more prominent"
    - Preserved topology ranking → "some concept pairs are always closer"
    - Preserved cluster structure → "families are always separated"
    - These invariant structures exist REGARDLESS of the coordinate system (layer)
    """
    print("\n" + "="*60)
    print("EXP 4: Semantic Gauge Invariants")
    print("="*60)

    concept_list = sorted(concepts)
    sorted_layers = sorted(sample_layers)
    n_concepts = len(concept_list)

    # Step 1: Concept salience (norm of concept delta) at each layer
    print(f"  Step 1: Computing concept salience...")

    salience_at_layer = {}
    for l in sorted_layers:
        salience = {}
        for c in concept_list:
            delta = concept_deltas[c][l]
            salience[c] = float(np.linalg.norm(delta))
        salience_at_layer[l] = salience

    # Compute salience ranking at each layer
    salience_rankings = {}
    for l in sorted_layers:
        ranked = sorted(salience_at_layer[l].items(), key=lambda x: -x[1])
        salience_rankings[l] = [c for c, _ in ranked]

    # Find gauge-invariant salience relations: pairs that maintain ranking across ALL layers
    print(f"  Step 2: Finding gauge-invariant salience rankings...")

    n_invariant_pairs = 0
    n_total_pairs = 0
    invariant_salience_pairs = []

    for i, c1 in enumerate(concept_list):
        for j, c2 in enumerate(concept_list):
            if i >= j:
                continue
            n_total_pairs += 1

            # Check if c1 is always ranked higher than c2
            always_higher = all(
                salience_at_layer[l][c1] > salience_at_layer[l][c2]
                for l in sorted_layers
            )
            always_lower = all(
                salience_at_layer[l][c1] < salience_at_layer[l][c2]
                for l in sorted_layers
            )

            if always_higher or always_lower:
                n_invariant_pairs += 1
                if len(invariant_salience_pairs) < 30:
                    invariant_salience_pairs.append({
                        "higher": c1 if always_higher else c2,
                        "lower": c2 if always_higher else c1,
                        "family_pair": (CONCEPT_TO_FAMILY[c1], CONCEPT_TO_FAMILY[c2]),
                    })

    salience_invariance_rate = n_invariant_pairs / max(n_total_pairs, 1)
    print(f"    Invariant salience pairs: {n_invariant_pairs}/{n_total_pairs} = {salience_invariance_rate:.4f}")

    # Step 2: Concept topology (pairwise distance ranking) at each layer
    print(f"  Step 3: Computing concept topology rankings...")

    # For each concept, find its nearest neighbor at each layer
    nearest_neighbor_at_layer = {}
    for l in sorted_layers:
        M = np.array([concept_deltas[c][l] for c in concept_list])
        M = np.nan_to_num(M, nan=0.0, posinf=1e4, neginf=-1e4)

        nearest = {}
        for i, c1 in enumerate(concept_list):
            best_c2 = None
            best_sim = -1
            for j, c2 in enumerate(concept_list):
                if i == j:
                    continue
                sim = cosine_sim(M[i], M[j])
                if sim > best_sim:
                    best_sim = sim
                    best_c2 = c2
            nearest[c1] = best_c2
        nearest_neighbor_at_layer[l] = nearest

    # Find gauge-invariant nearest neighbors: pairs that are always nearest across ALL layers
    invariant_neighbors = {}
    for c in concept_list:
        neighbors = [nearest_neighbor_at_layer[l][c] for l in sorted_layers]
        # Check if all neighbors are the same
        if len(set(neighbors)) == 1:
            invariant_neighbors[c] = {
                "neighbor": neighbors[0],
                "family": CONCEPT_TO_FAMILY[c],
                "neighbor_family": CONCEPT_TO_FAMILY[neighbors[0]],
                "same_family": CONCEPT_TO_FAMILY[c] == CONCEPT_TO_FAMILY[neighbors[0]],
            }

    print(f"    Concepts with invariant nearest neighbor: {len(invariant_neighbors)}/{n_concepts}")
    for c, info in list(invariant_neighbors.items())[:10]:
        print(f"      {c} ({info['family']}) → {info['neighbor']} ({info['neighbor_family']}) "
              f"same_family={info['same_family']}")

    # Step 3: Cluster structure invariance
    print(f"  Step 4: Computing cluster structure invariance...")

    # For each layer, compute within-family vs between-family cosine similarity
    concrete_families = {"animal", "vehicle", "color"}
    abstract_families = {"abstract", "emotion"}

    cluster_metrics_at_layer = {}
    for l in sorted_layers:
        M = np.array([concept_deltas[c][l] for c in concept_list])
        M = np.nan_to_num(M, nan=0.0, posinf=1e4, neginf=-1e4)

        within_family_sims = defaultdict(list)
        between_family_sims = defaultdict(list)

        for i, c1 in enumerate(concept_list):
            for j, c2 in enumerate(concept_list):
                if i >= j:
                    continue
                sim = cosine_sim(M[i], M[j])
                f1 = CONCEPT_TO_FAMILY[c1]
                f2 = CONCEPT_TO_FAMILY[c2]

                if f1 == f2:
                    within_family_sims[f1].append(sim)
                else:
                    between_family_sims[(f1, f2)].append(sim)

        # Compute separation: within-family avg - between-family avg
        all_within = [s for sims in within_family_sims.values() for s in sims]
        all_between = [s for sims in between_family_sims.values() for s in sims]

        avg_within = float(np.mean(all_within)) if all_within else 0
        avg_between = float(np.mean(all_between)) if all_between else 0
        separation = avg_within - avg_between

        # Concrete vs abstract within-family similarity
        concrete_within = [s for f, sims in within_family_sims.items() for s in sims if f in concrete_families]
        abstract_within = [s for f, sims in within_family_sims.items() for s in sims if f in abstract_families]

        cluster_metrics_at_layer[l] = {
            "avg_within_family": round(avg_within, 4),
            "avg_between_family": round(avg_between, 4),
            "separation": round(separation, 4),
            "concrete_within": round(float(np.mean(concrete_within)), 4) if concrete_within else 0,
            "abstract_within": round(float(np.mean(abstract_within)), 4) if abstract_within else 0,
        }

    # Check if separation is always positive (families always separated)
    separation_always_positive = all(
        cluster_metrics_at_layer[l]["separation"] > 0 for l in sorted_layers
    )
    separation_variance = float(np.var([cluster_metrics_at_layer[l]["separation"] for l in sorted_layers]))

    print(f"    Cluster separation always positive: {separation_always_positive}")
    print(f"    Separation variance across layers: {separation_variance:.4f}")
    for l in sorted_layers:
        cm = cluster_metrics_at_layer[l]
        print(f"      L{l}: within={cm['avg_within_family']:.4f}, between={cm['avg_between_family']:.4f}, "
              f"sep={cm['separation']:.4f}, concrete_within={cm['concrete_within']:.4f}, "
              f"abstract_within={cm['abstract_within']:.4f}")

    # Step 4: Gauge-invariant summary
    print(f"  Step 5: Computing gauge-invariant summary...")

    # The gauge-invariant structure is what survives ALL coordinate rotations
    gauge_invariant_structure = {
        "salience_invariance_rate": round(salience_invariance_rate, 4),
        "n_invariant_salience_pairs": n_invariant_pairs,
        "n_total_pairs": n_total_pairs,
        "n_invariant_neighbors": len(invariant_neighbors),
        "separation_always_positive": separation_always_positive,
        "separation_variance": round(separation_variance, 4),
    }

    # Which families have the most invariant nearest neighbors?
    family_invariant_counts = defaultdict(int)
    for c, info in invariant_neighbors.items():
        if info["same_family"]:
            family_invariant_counts[CONCEPT_TO_FAMILY[c]] += 1

    print(f"  Family invariant neighbor counts: {dict(family_invariant_counts)}")

    results = {
        "gauge_invariant_structure": gauge_invariant_structure,
        "invariant_salience_pairs_sample": invariant_salience_pairs,
        "invariant_neighbors": invariant_neighbors,
        "cluster_metrics_by_layer": cluster_metrics_at_layer,
        "salience_rankings_sample": {
            l: salience_rankings[l][:10] for l in sorted_layers[:5]
        },
    }

    return results


# ===== MAIN =====

def main():
    import torch

    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    print(f"\n{'='*60}")
    print(f"Phase 170: Dynamical Invariants")
    print(f"Model: {model_name}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")

    # Load model (bfloat16 + device_map="auto")
    model, tokenizer, device, use_8bit = load_model_auto_bf16(model_name)
    info = get_model_info(model, model_name)
    print(f"[0] Model: {info.model_class}, L={info.n_layers}, d={info.d_model}, "
          f"V={info.vocab_size}, 8bit={use_8bit}")

    # Sample layers: every 2 layers for thorough coverage
    n_layers = info.n_layers
    sample_layers = list(range(0, n_layers + 1, 2))
    sample_layers = sorted(set(sample_layers))
    if n_layers not in sample_layers:
        sample_layers.append(n_layers)

    print(f"[0] Sample layers: {sample_layers}")

    # Use ALL 50 concepts and 6 templates for robust results
    concepts = ALL_CONCEPTS  # 50 concepts
    templates = CONTEXT_TEMPLATES[:6]

    all_results = {
        "model": model_name,
        "model_class": info.model_class,
        "n_layers": info.n_layers,
        "d_model": info.d_model,
        "vocab_size": info.vocab_size,
        "use_8bit": use_8bit,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_concepts": len(concepts),
        "n_templates": len(templates),
        "sample_layers": sample_layers,
    }

    # ===== Data Collection =====
    print(f"\n[1] Collecting concept deltas at all sampled layers...")
    t0 = time.time()
    baseline_h, concept_h, concept_deltas = collect_concept_deltas(
        model, tokenizer, device, model_name, use_8bit,
        concepts, templates, sample_layers)
    print(f"  Time: {time.time()-t0:.1f}s")

    # ===== Exp 1: Parallel Transport =====
    print(f"\n[2] Exp 1: Parallel Transport (JVP)")
    t0 = time.time()
    all_results["exp1_transport"] = exp1_parallel_transport(
        model, tokenizer, device, model_name, use_8bit,
        concepts, concept_deltas, sample_layers, epsilon=1.0)
    print(f"  Time: {time.time()-t0:.1f}s")

    # ===== Exp 2: Jacobian Spectrum =====
    print(f"\n[3] Exp 2: Jacobian Spectrum")
    t0 = time.time()
    all_results["exp2_spectrum"] = exp2_jacobian_spectrum(
        model, tokenizer, device, model_name, use_8bit,
        sample_layers, n_random=20, epsilon=1.0)
    print(f"  Time: {time.time()-t0:.1f}s")

    # ===== Exp 3: Metric Preservation =====
    print(f"\n[4] Exp 3: Metric Preservation & Holonomy")
    t0 = time.time()
    all_results["exp3_metric"] = exp3_metric_preservation(
        concept_deltas, concepts, sample_layers)
    print(f"  Time: {time.time()-t0:.1f}s")

    # ===== Exp 4: Gauge Invariants =====
    print(f"\n[5] Exp 4: Semantic Gauge Invariants")
    t0 = time.time()
    all_results["exp4_gauge"] = exp4_gauge_invariants(
        concept_deltas, concepts, sample_layers)
    print(f"  Time: {time.time()-t0:.1f}s")

    # ===== Save Results =====
    os.makedirs("tests/glm5_temp", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = f"tests/glm5_temp/phase170_{model_name}_{timestamp}.json"

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

    print(f"\n[6] Results saved to: {out_path}")

    # Release model
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\nPhase 170 complete for {model_name}!")


if __name__ == "__main__":
    main()
