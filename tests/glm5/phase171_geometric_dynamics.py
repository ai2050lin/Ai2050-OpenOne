"""
Phase 171: Geometric Dynamics — 几何动力学
=============================================

★★★ THE CORE TRANSITION: From "discrete Jacobian" → "continuous generator + curvature"! ★★★

User's key theoretical corrections (ALL CORRECT):

1. "Isometric ≠ Rotation" — σ≈1 only means J^T J ≈ I (length-preserving),
   NOT that J is a pure rotation. A map can preserve length but not direction/parallelism/geodesics.
   → Need to test WHAT KIND of isometry: rotation vs. general diffeomorphism

2. "The real object is log(J), not J" — The Jacobian is a FINITE transformation.
   The generator A_l = log(J_l) is the DYNAMICAL LAW.
   h_{l+Δl} ≈ exp(Δl · A_l) h_l
   → Decompose A_l into A_skew (rotation) and A_sym (expansion/contraction)

3. "Holonomy is the ONLY way to measure curvature" — Not Δ(2ε) - 2Δ(ε),
   but: if P_{γ1} P_{γ2} ≠ P_{γ2} P_{γ1} → curvature exists.
   → Test: do perturbations in different directions COMMUTE?

4. "Concept = Orbit, not vector" — A concept is {v_l} across ALL layers,
   not a single direction at one layer.
   → Language structure lives in COVARIANT relations, not fixed directions

Four critical experiments:

  Exp 1: ★★★ Holonomy — 完整旋绕测试 (MOST IMPORTANT!)
    - At layer l, inject ε*d1+ε*d2 simultaneously → δ12 at l+k
    - Compare with: inject d1 (→ δ1) and d2 (→ δ2) separately, then δ1+δ2
    - Nonlinearity = 1 - cos(δ12, δ1+δ2) = NON-COMMUTATIVITY MEASURE
    - Also: path-dependence test: inject d1 at l→l+j, then inject result at l+j→l+k
      vs inject d1 directly at l→l+k
    - If nonlinearity > 0 → perturbations don't commute → CURVATURE EXISTS
    - This is the TRUE curvature test (not Δ(2ε) - 2Δ(ε))

  Exp 2: ★★★ Generator Field — 生成元场
    - Estimate J_l via randomized JVPs (n=30 for better accuracy)
    - Compute generator spectrum: λ_i = log(σ_i)
    - Compute rotation angles: θ_i = arccos(u_i^T v_i) from SVD
    - Decompose into:
      a. Rotation-dominated directions: |λ_i| < 0.3, θ_i > 30°
      b. Expansion-dominated: λ_i > 0.3
      c. Contraction-dominated: λ_i < -0.3
      d. Isometric (pure transport): |λ_i| < 0.3, θ_i < 30°
    - KEY QUESTION: Is the system rotation-dominated or expansion-dominated?

  Exp 3: ★★ Lie Bracket (State-Dependent Jacobian) — 李括号测试
    - At layer l, compute JVP for direction d2 at BASELINE state: v2_base
    - At layer l, compute JVP for direction d2 at d1-PERTURBED state: v2_pert
    - Lie_bracket = cos(v2_base, v2_pert)
    - If Lie_bracket < 1 → Jacobian is state-dependent → curvature exists
    - This is the DISCRETE version of R(X,Y)Z = ∇_X ∇_Y Z - ∇_Y ∇_X Z
    - REQUIRES: 4 forward passes per concept pair (baseline, d1, d2, d1+d2)

  Exp 4: ★★ Geodesic Structure — 测地线结构
    - At each layer, for concept triples (A,B,C) in the same family
    - Compute triangle ratio: d(A,C) / (d(A,B) + d(B,C))
    - If ratio < 1 → B is "between" A and C (geodesic exists)
    - If ratio varies significantly → manifold has CURVATURE
    - Also: check if linear interpolation stays "on manifold"
    - This reveals the LOCAL GEOMETRY of the concept manifold

Usage: python tests/glm5/phase171_geometric_dynamics.py <model_name>
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


# ===== MODEL LOADING (BF16 + device_map="auto") =====

def load_model_auto_bf16(model_name):
    """Load model with bfloat16 + device_map='auto' (no 8-bit)."""
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
    h = hidden_tensor[0, -1, :].float().cpu().numpy()
    return np.nan_to_num(h, nan=0.0, posinf=1e4, neginf=-1e4)


def cosine_sim(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


# ===== DATA COLLECTION =====

def collect_concept_deltas(model, tokenizer, device, model_name, use_8bit,
                           concepts, templates, sample_layers):
    """Collect concept deltas at all sampled layers."""
    import torch

    input_device = get_device_for_input(model)
    info = get_model_info(model, model_name)

    # Step 1: Baseline
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

    # Step 2: Concept hidden states
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

    # Step 3: Concept deltas
    concept_deltas = {}
    for concept in concepts:
        concept_deltas[concept] = {
            l: concept_h[concept][l] - baseline_h[l] for l in sample_layers
        }

    return baseline_h, concept_h, concept_deltas


def inject_and_read(model, tokenizer, input_device, ref_prompt, layers,
                    transformer_layer_idx, direction, epsilon, target_layer_idx):
    """Inject direction at transformer layer, read at target layer. Returns perturbed hidden."""
    import torch

    direction_tensor = torch.tensor(direction, dtype=torch.float32)

    def make_hook(d, eps, dev):
        def hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0].clone()
                dd = d.to(h.dtype).to(h.device)
                h[0, -1, :] += eps * dd
                return (h,) + output[1:]
            else:
                h = output.clone()
                dd = d.to(h.dtype).to(h.device)
                h[0, -1, :] += eps * dd
                return h
        return hook

    hook = layers[transformer_layer_idx].register_forward_hook(
        make_hook(direction_tensor, epsilon, input_device))

    inputs = tokenizer(ref_prompt, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(input_device)
    attention_mask = inputs["attention_mask"].to(input_device)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                   output_hidden_states=True)

    hook.remove()

    h_perturbed = safe_h(out.hidden_states[target_layer_idx])
    del input_ids, attention_mask, out, direction_tensor

    return h_perturbed


def inject_dual_and_read(model, tokenizer, input_device, ref_prompt, layers,
                         transformer_layer_idx, direction1, direction2, epsilon, target_layer_idx):
    """Inject direction1+direction2 at transformer layer, read at target layer."""
    import torch

    combined = direction1 + direction2
    combined_tensor = torch.tensor(combined, dtype=torch.float32)

    def make_hook(d, eps, dev):
        def hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0].clone()
                dd = d.to(h.dtype).to(h.device)
                h[0, -1, :] += eps * dd
                return (h,) + output[1:]
            else:
                h = output.clone()
                dd = d.to(h.dtype).to(h.device)
                h[0, -1, :] += eps * dd
                return h
        return hook

    hook = layers[transformer_layer_idx].register_forward_hook(
        make_hook(combined_tensor, epsilon, input_device))

    inputs = tokenizer(ref_prompt, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(input_device)
    attention_mask = inputs["attention_mask"].to(input_device)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                   output_hidden_states=True)

    hook.remove()

    h_perturbed = safe_h(out.hidden_states[target_layer_idx])
    del input_ids, attention_mask, out, combined_tensor

    return h_perturbed


# ===== EXP 1: HOLONOMY — Non-Commutative Transport ★★★ =====

def exp1_holonomy(model, tokenizer, device, model_name, use_8bit,
                  concept_deltas, sample_layers, epsilon=1.0):
    """
    ★★★ Holonomy Test — 完整旋绕测试 ★★★

    KEY QUESTION: Do perturbations in different concept directions COMMUTE?

    If P_{γ1} P_{γ2} ≠ P_{γ2} P_{γ1} → curvature exists!

    Method:
    - At layer l, take two concept directions d1, d2
    - Inject d1 → δ1 at l+k (individual)
    - Inject d2 → δ2 at l+k (individual)
    - Inject d1+d2 → δ12 at l+k (combined)
    - Nonlinearity = 1 - cos(δ12, δ1+δ2)
      = How much does the combined perturbation differ from the sum of individual?
      = This is the NON-COMMUTATIVITY of the transport

    Also: PATH-DEPENDENCE test
    - Inject d1 at l → read at l+j (intermediate δ_j)
    - Inject δ_j at l+j → read at l+k (two-step transport)
    - Inject d1 at l → read at l+k (direct transport)
    - Path-dependence = cos(two-step, direct)
      = Does the PATH matter for parallel transport?

    INTERPRETATION:
    - Nonlinearity ≈ 0 → transport is LINEAR → flat connection → no curvature
    - Nonlinearity > 0 → transport is NONLINEAR → curved connection → curvature exists!
    - Path-dependence ≈ 1 → transport is path-INDEPENDENT → flat manifold
    - Path-dependence < 1 → transport is path-DEPENDENT → curved manifold!
    """
    import torch

    print("\n" + "="*60)
    print("EXP 1: Holonomy — Non-Commutative Transport ★★★")
    print("="*60)

    input_device = get_device_for_input(model)
    layers = get_layers(model)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers

    # Select concept pairs from DIFFERENT families (cross-family = most likely to show holonomy)
    concept_pairs = [
        ("cat", "car"),        # animal-vehicle
        ("dog", "freedom"),   # animal-abstract
        ("red", "love"),      # color-emotion
        ("tiger", "justice"), # animal-abstract
        ("blue", "anger"),    # color-emotion
        ("car", "truth"),     # vehicle-abstract
        ("horse", "sadness"), # animal-emotion
        ("green", "power"),   # color-abstract
    ]
    # Also same-family pairs (to compare)
    same_family_pairs = [
        ("cat", "dog"),       # animal-animal
        ("red", "blue"),      # color-color
        ("love", "anger"),    # emotion-emotion
    ]
    all_pairs = concept_pairs + same_family_pairs

    # Select injection layers (3 representative points)
    inject_configs = []
    # Early: L4→L6, Middle: L16→L18, Late: L28→L30 (for 36-layer model)
    for l_start in [4, 12, 20, 28]:
        l_end = l_start + 2
        l_mid = l_start + 1
        if l_start < n_layers - 2 and l_end <= n_layers and l_start >= 1:
            inject_configs.append((l_start, l_mid, l_end))

    print(f"  Concept pairs: {len(all_pairs)} ({len(concept_pairs)} cross, {len(same_family_pairs)} same)")
    print(f"  Injection configs: {inject_configs}")
    print(f"  Epsilon: {epsilon}")

    # Reference prompt for all holonomy tests
    ref_prompt = "The cat is"

    holonomy_results = {}

    for l_start, l_mid, l_end in inject_configs:
        transformer_idx = l_start - 1
        if transformer_idx >= len(layers):
            continue

        # First, compute baseline at l_end
        inputs = tokenizer(ref_prompt, return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        with torch.no_grad():
            out_base = model(input_ids=input_ids, attention_mask=attention_mask,
                            output_hidden_states=True)
        h_baseline_end = safe_h(out_base.hidden_states[l_end])
        h_baseline_mid = safe_h(out_base.hidden_states[l_mid])
        del input_ids, attention_mask, out_base

        pair_results = []

        for c1, c2 in all_pairs:
            # Get concept directions at l_start
            d1 = concept_deltas[c1].get(l_start)
            d2 = concept_deltas[c2].get(l_start)
            if d1 is None or d2 is None:
                continue
            d1_norm = np.linalg.norm(d1)
            d2_norm = np.linalg.norm(d2)
            if d1_norm < 1e-10 or d2_norm < 1e-10:
                continue
            d1_unit = d1 / d1_norm
            d2_unit = d2 / d2_norm

            same_family = CONCEPT_TO_FAMILY[c1] == CONCEPT_TO_FAMILY[c2]

            # === Nonlinearity test (Non-commutativity) ===
            # Inject d1 alone
            h_d1 = inject_and_read(model, tokenizer, input_device, ref_prompt, layers,
                                   transformer_idx, d1_unit, epsilon, l_end)
            delta1 = h_d1 - h_baseline_end

            # Inject d2 alone
            h_d2 = inject_and_read(model, tokenizer, input_device, ref_prompt, layers,
                                   transformer_idx, d2_unit, epsilon, l_end)
            delta2 = h_d2 - h_baseline_end

            # Inject d1+d2 together
            h_d12 = inject_dual_and_read(model, tokenizer, input_device, ref_prompt, layers,
                                         transformer_idx, d1_unit, d2_unit, epsilon, l_end)
            delta12 = h_d12 - h_baseline_end

            # Linear prediction
            delta_linear = delta1 + delta2

            # Nonlinearity measure
            cos_combined = cosine_sim(delta12, delta_linear)
            nonlinearity = 1.0 - cos_combined

            # Relative nonlinearity
            norm_combined = np.linalg.norm(delta12)
            norm_diff = np.linalg.norm(delta12 - delta_linear)
            relative_nonlinearity = norm_diff / max(norm_combined, 1e-10)

            # === Path-dependence test ===
            # Direct: inject d1 at l_start → read at l_end
            delta_direct = delta1.copy()  # Already computed above

            # Two-step: inject d1 at l_start → read at l_mid → inject that at l_mid → read at l_end
            # Step 1: inject d1 at l_start, read at l_mid
            h_d1_mid = inject_and_read(model, tokenizer, input_device, ref_prompt, layers,
                                       transformer_idx, d1_unit, epsilon, l_mid)
            delta_mid = h_d1_mid - h_baseline_mid

            # Step 2: inject delta_mid (normalized) at l_mid, read at l_end
            delta_mid_norm = np.linalg.norm(delta_mid)
            if delta_mid_norm > 1e-10:
                delta_mid_unit = delta_mid / delta_mid_norm
                transformer_idx_mid = l_mid - 1
                if transformer_idx_mid < len(layers):
                    h_twostep = inject_and_read(model, tokenizer, input_device, ref_prompt, layers,
                                               transformer_idx_mid, delta_mid_unit, epsilon, l_end)
                    delta_twostep = h_twostep - h_baseline_end

                    path_dependence = cosine_sim(delta_direct, delta_twostep)
                    path_norm_ratio = np.linalg.norm(delta_twostep) / max(np.linalg.norm(delta_direct), 1e-10)
                else:
                    path_dependence = None
                    path_norm_ratio = None
            else:
                path_dependence = None
                path_norm_ratio = None

            result = {
                "pair": f"{c1}-{c2}",
                "same_family": same_family,
                "nonlinearity": round(nonlinearity, 4),
                "relative_nonlinearity": round(relative_nonlinearity, 4),
                "cos_combined_vs_linear": round(cos_combined, 4),
                "path_dependence": round(path_dependence, 4) if path_dependence is not None else None,
                "path_norm_ratio": round(path_norm_ratio, 4) if path_norm_ratio is not None else None,
                "delta1_norm": round(float(np.linalg.norm(delta1)), 4),
                "delta2_norm": round(float(np.linalg.norm(delta2)), 4),
                "delta12_norm": round(float(np.linalg.norm(delta12)), 4),
            }
            pair_results.append(result)

            del h_d1, h_d2, h_d12, delta1, delta2, delta12, delta_linear

        # Summary for this layer pair
        cross_nonlinearities = [r["nonlinearity"] for r in pair_results if not r["same_family"]]
        same_nonlinearities = [r["nonlinearity"] for r in pair_results if r["same_family"]]
        all_nonlinearities = [r["nonlinearity"] for r in pair_results]
        path_deps = [r["path_dependence"] for r in pair_results if r["path_dependence"] is not None]

        layer_summary = {
            "avg_nonlinearity": round(float(np.mean(all_nonlinearities)), 4) if all_nonlinearities else 0,
            "cross_family_nonlinearity": round(float(np.mean(cross_nonlinearities)), 4) if cross_nonlinearities else 0,
            "same_family_nonlinearity": round(float(np.mean(same_nonlinearities)), 4) if same_nonlinearities else 0,
            "avg_path_dependence": round(float(np.mean(path_deps)), 4) if path_deps else None,
            "n_pairs": len(pair_results),
        }

        print(f"    L{l_start}→L{l_end}: avg_nonlinearity={layer_summary['avg_nonlinearity']:.4f}, "
              f"cross={layer_summary['cross_family_nonlinearity']:.4f}, "
              f"same={layer_summary['same_family_nonlinearity']:.4f}, "
              f"path_dep={layer_summary['avg_path_dependence']:.4f}" if path_deps else "")

        holonomy_results[f"L{l_start}_L{l_end}"] = {
            "pairs": pair_results,
            "summary": layer_summary,
        }

    # Overall summary
    all_nonlin = []
    cross_nonlin = []
    same_nonlin = []
    all_path_dep = []
    for key, hr in holonomy_results.items():
        all_nonlin.append(hr["summary"]["avg_nonlinearity"])
        cross_nonlin.append(hr["summary"]["cross_family_nonlinearity"])
        same_nonlin.append(hr["summary"]["same_family_nonlinearity"])
        if hr["summary"]["avg_path_dependence"] is not None:
            all_path_dep.append(hr["summary"]["avg_path_dependence"])

    print(f"\n  === Holonomy Results ===")
    print(f"  Overall avg nonlinearity: {np.mean(all_nonlin):.4f}")
    print(f"  Cross-family nonlinearity: {np.mean(cross_nonlin):.4f}")
    print(f"  Same-family nonlinearity: {np.mean(same_nonlin):.4f}")
    if all_path_dep:
        print(f"  Overall avg path-dependence: {np.mean(all_path_dep):.4f}")

    results = {
        "holonomy_by_layer": holonomy_results,
        "overall": {
            "avg_nonlinearity": round(float(np.mean(all_nonlin)), 4),
            "cross_family_nonlinearity": round(float(np.mean(cross_nonlin)), 4),
            "same_family_nonlinearity": round(float(np.mean(same_nonlin)), 4),
            "avg_path_dependence": round(float(np.mean(all_path_dep)), 4) if all_path_dep else None,
            "curvature_exists": float(np.mean(all_nonlin)) > 0.05,  # Nonlinearity > 5% → curvature
        },
        "epsilon": epsilon,
        "n_pairs": len(all_pairs),
    }

    return results


# ===== EXP 2: GENERATOR FIELD ★★★ =====

def exp2_generator_field(model, tokenizer, device, model_name, use_8bit,
                        sample_layers, n_random=30, epsilon=1.0):
    """
    ★★★ Generator Field — 生成元场 ★★★

    KEY QUESTION: What is the dynamical law of the system?

    User's insight: "The real object is log(J), not J"
    J_l is a FINITE transformation. A_l = log(J_l) is the DYNAMICAL LAW.

    Method:
    - Estimate J_l via randomized JVPs (n=30 for better accuracy)
    - SVD: J_l ≈ U S V^T
    - Generator spectrum: λ_i = log(σ_i)
    - Rotation angles: θ_i = arccos(u_i^T v_i)
    - Decompose each singular direction into:
      a. ROTATION: |λ_i| < 0.3 AND θ_i > 30° → pure rotation
      b. EXPANSION: λ_i > 0.3 → expanding direction
      c. CONTRACTION: λ_i < -0.3 → contracting direction
      d. ISOMETRIC (pure transport): |λ_i| < 0.3 AND θ_i < 30° → near-identity transport
      e. ROTATION+EXPANSION: |λ_i| > 0.3 AND θ_i > 30° → rotating while expanding

    KEY INSIGHT from user:
    - "Isometric ≠ Rotation" — σ≈1 only means length-preserving
    - A map with σ≈1 could be a rotation (preserving angles) or
      a general diffeomorphism (preserving lengths but not angles)
    - The ROTATION ANGLE θ_i tells us whether the direction is truly rotating

    INTERPRETATION:
    - If most directions are ISOMETRIC (|λ|<0.3, θ<30°) → near-identity transport → flat
    - If most directions are ROTATION (|λ|<0.3, θ>30°) → coordinate rotation → gauge freedom
    - If significant EXPANSION/CONTRACTION → anisotropic → direction-dependent processing
    - The FRACTION of rotation vs isometric reveals the GEOMETRY of the system
    """
    import torch

    print("\n" + "="*60)
    print("EXP 2: Generator Field — log(J_l) Decomposition ★★★")
    print("="*60)

    input_device = get_device_for_input(model)
    layers = get_layers(model)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model

    # Select injection layers
    inject_layers = [l for l in sample_layers if 1 <= l <= n_layers - 2]
    inject_layers = inject_layers[::2]
    if len(inject_layers) > 8:
        inject_layers = inject_layers[::len(inject_layers)//8 + 1]

    print(f"  Injection layers: {inject_layers}")
    print(f"  Random directions: {n_random}")
    print(f"  Epsilon: {epsilon}")

    ref_prompt = "The cat is"
    generator_results = {}

    for lidx, layer_l in enumerate(inject_layers):
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

        # Step 1: Compute baseline at layer_next
        inputs = tokenizer(ref_prompt, return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        with torch.no_grad():
            out_baseline = model(input_ids=input_ids, attention_mask=attention_mask,
                                output_hidden_states=True)
        h_baseline = safe_h(out_baseline.hidden_states[layer_next])
        del input_ids, attention_mask, out_baseline

        # Step 2: Compute J@r_i for random directions
        jvp_matrix = np.zeros((d_model, n_random))
        # Also store the random directions for rotation angle computation
        random_directions = np.zeros((d_model, n_random))

        for r_idx in range(n_random):
            r = np.random.randn(d_model)
            r = r / np.linalg.norm(r)
            random_directions[:, r_idx] = r

            h_perturbed = inject_and_read(model, tokenizer, input_device, ref_prompt, layers,
                                          transformer_layer_idx, r, epsilon, layer_next)
            jvp = (h_perturbed - h_baseline) / epsilon
            jvp_matrix[:, r_idx] = jvp

        # Step 3: SVD of JVP matrix
        try:
            U, S, Vt = np.linalg.svd(jvp_matrix, full_matrices=False)
        except np.linalg.LinAlgError:
            print(f"    SVD failed for layer {layer_l}")
            continue

        # Step 4: Generator spectrum and rotation analysis
        generator_spectrum = np.log(np.maximum(S, 1e-10))  # λ_i = log(σ_i)

        # Compute rotation angles: for each singular direction,
        # measure angle between input direction (Vt row) and output direction (U column)
        rotation_angles = []
        for i in range(min(len(S), 20)):  # Top 20
            if S[i] < 1e-10:
                rotation_angles.append(0.0)
                continue
            u_i = U[:, i]  # Output direction (d_model,)
            # The "input direction" for singular value i is Vt[i, :]
            # But Vt from SVD of Y = J@R, not of J itself
            # So Vt[i,:] is the i-th right singular vector of Y, not of J
            # We need to compare u_i with the "natural" input direction
            # A better measure: angle between u_i and the input that PRODUCED it
            # u_i ≈ J @ v_i where v_i is some input direction
            # The rotation angle for direction i is arccos(cos(u_i, Jv_i/Jv_i_norm))
            # But u_i IS the output direction, so we measure how much the DIRECTION rotates
            # by comparing u_i with the "predicted" direction if J were identity
            # For the top singular vectors, we can compute:
            # θ_i = arccos(|u_i^T v_i'|) where v_i' is the input that maps to u_i
            # Since Y = U S Vt, and Y[:,i] ≈ σ_i u_i, the input that produces u_i is:
            # Vt[i,:] (the i-th row of Vt) — but this is in the random direction basis
            # A simpler approach: just measure ||u_i - v_i|| where v_i is the closest random input
            # The simplest: use the angle between the i-th output and input singular vectors
            v_i = Vt[i, :]  # This is the i-th right singular vector of Y
            # Map v_i back to d_model space: v_i is a vector in R^{n_random}
            # The corresponding input direction in R^{d_model} is:
            # r_input = random_directions @ v_i / ||random_directions @ v_i||
            r_input = random_directions @ v_i
            r_input_norm = np.linalg.norm(r_input)
            if r_input_norm > 1e-10:
                r_input_unit = r_input / r_input_norm
                # Rotation angle: angle between input and output directions
                cos_angle = abs(cosine_sim(u_i, r_input_unit))
                angle_deg = np.degrees(np.arccos(np.clip(cos_angle, 0, 1)))
                rotation_angles.append(round(float(angle_deg), 2))
            else:
                rotation_angles.append(0.0)

        # Step 5: Classify each singular direction
        n_dirs = len(S)
        classification = {"rotation": 0, "expansion": 0, "contraction": 0,
                          "isometric": 0, "rotation_expansion": 0}

        for i in range(min(n_dirs, len(rotation_angles))):
            lam = generator_spectrum[i] if i < len(generator_spectrum) else 0
            theta = rotation_angles[i] if i < len(rotation_angles) else 0

            if abs(lam) < 0.3 and theta > 30:
                classification["rotation"] += 1
            elif lam > 0.3 and theta > 30:
                classification["rotation_expansion"] += 1
            elif lam > 0.3:
                classification["expansion"] += 1
            elif lam < -0.3:
                classification["contraction"] += 1
            else:  # |lam| < 0.3 and theta < 30
                classification["isometric"] += 1

        # Normalize
        total = sum(classification.values())
        for k in classification:
            classification[k] = round(classification[k] / max(total, 1), 4)

        # Summary metrics
        s_max = float(S[0]) if len(S) > 0 else 0
        s_min = float(S[-1]) if len(S) > 0 else 0
        condition_number = s_max / max(s_min, 1e-10)

        gen_max = float(np.max(generator_spectrum)) if len(generator_spectrum) > 0 else 0
        gen_min = float(np.min(generator_spectrum)) if len(generator_spectrum) > 0 else 0
        gen_mean = float(np.mean(generator_spectrum)) if len(generator_spectrum) > 0 else 0

        avg_rotation = float(np.mean(rotation_angles)) if rotation_angles else 0
        max_rotation = float(np.max(rotation_angles)) if rotation_angles else 0

        # Energy decomposition
        rotation_energy = sum(generator_spectrum[i]**2 for i in range(len(generator_spectrum))
                            if i < len(rotation_angles) and abs(generator_spectrum[i]) < 0.3 and rotation_angles[i] > 30)
        expansion_energy = sum(generator_spectrum[i]**2 for i in range(len(generator_spectrum))
                              if generator_spectrum[i] > 0.3)
        contraction_energy = sum(generator_spectrum[i]**2 for i in range(len(generator_spectrum))
                                if generator_spectrum[i] < -0.3)
        isometric_energy = sum(generator_spectrum[i]**2 for i in range(len(generator_spectrum))
                              if i < len(rotation_angles) and abs(generator_spectrum[i]) < 0.3 and rotation_angles[i] < 30)

        total_energy = rotation_energy + expansion_energy + contraction_energy + isometric_energy
        if total_energy > 0:
            energy_fractions = {
                "rotation": round(rotation_energy / total_energy, 4),
                "expansion": round(expansion_energy / total_energy, 4),
                "contraction": round(contraction_energy / total_energy, 4),
                "isometric": round(isometric_energy / total_energy, 4),
            }
        else:
            energy_fractions = {"rotation": 0, "expansion": 0, "contraction": 0, "isometric": 0}

        print(f"    σ_max={s_max:.2f}, σ_min={s_min:.2f}, cond={condition_number:.2f}")
        print(f"    gen_max={gen_max:.2f}, gen_min={gen_min:.2f}, gen_mean={gen_mean:.4f}")
        print(f"    avg_rotation={avg_rotation:.1f}°, max_rotation={max_rotation:.1f}°")
        print(f"    classification={classification}")
        print(f"    energy_fractions={energy_fractions}")

        generator_results[f"L{layer_l}_L{layer_next}"] = {
            "singular_values": [round(float(s), 4) for s in S[:15]],
            "generator_spectrum": [round(float(g), 4) for g in generator_spectrum[:15]],
            "rotation_angles": rotation_angles[:15],
            "classification": classification,
            "energy_fractions": energy_fractions,
            "s_max": round(s_max, 4),
            "s_min": round(s_min, 4),
            "condition_number": round(condition_number, 4),
            "gen_max": round(gen_max, 4),
            "gen_min": round(gen_min, 4),
            "gen_mean": round(gen_mean, 4),
            "avg_rotation_deg": round(avg_rotation, 2),
            "max_rotation_deg": round(max_rotation, 2),
            "layers": (layer_l, layer_next),
        }

    # Overall summary
    print(f"\n  === Generator Field Results ===")
    for key, gr in generator_results.items():
        print(f"    {key}: gen_mean={gr['gen_mean']:.4f}, avg_rot={gr['avg_rotation_deg']:.1f}°, "
              f"class={gr['classification']}, energy={gr['energy_fractions']}")

    results = {
        "generator_by_layer": generator_results,
        "n_random": n_random,
        "epsilon": epsilon,
    }

    return results


# ===== EXP 3: LIE BRACKET (State-Dependent Jacobian) ★★ =====

def exp3_lie_bracket(model, tokenizer, device, model_name, use_8bit,
                     concept_deltas, sample_layers, epsilon=1.0):
    """
    ★★ Lie Bracket — State-Dependent Jacobian ★★

    KEY QUESTION: Is the Jacobian state-dependent? → Does curvature exist?

    In differential geometry: R(X,Y)Z = ∇_X ∇_Y Z - ∇_Y ∇_X Z
    The curvature tensor measures how the Jacobian changes when we move in different directions.

    Method:
    - At layer l, compute JVP for direction d2 at BASELINE state: v2_base
    - At layer l, compute JVP for direction d2 at d1-PERTURBED state: v2_pert
    - Lie_bracket_measure = cos(v2_base, v2_pert)
    - If Lie_bracket < 1 → Jacobian depends on state → CURVATURE EXISTS

    Implementation:
    - JVP_d2 at baseline: inject ε*d2 at l → δ2_base at l+k
    - JVP_d2 at d1-perturbed: inject ε*(d1+d2) at l → δ12 at l+k,
      then subtract δ1 (JVP for d1 alone)
    - More precisely:
      δ2_base = (F(h + ε*d2) - F(h)) / ε
      δ2_pert = (F(h + ε*d1 + ε*d2) - F(h + ε*d1)) / ε
      Lie bracket = 1 - cos(δ2_base, δ2_pert)

    INTERPRETATION:
    - If cos ≈ 1 → Jacobian is state-INDEPENDENT → flat connection → no curvature
    - If cos < 1 → Jacobian is state-DEPENDENT → curved connection → curvature exists!
    - The DEGREE of deviation = the MAGNITUDE of curvature
    """
    import torch

    print("\n" + "="*60)
    print("EXP 3: Lie Bracket — State-Dependent Jacobian ★★")
    print("="*60)

    input_device = get_device_for_input(model)
    layers = get_layers(model)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers

    # Use same concept pairs as Exp 1 but fewer
    test_pairs = [
        ("cat", "car"),       # animal-vehicle
        ("dog", "freedom"),   # animal-abstract
        ("red", "love"),      # color-emotion
        ("tiger", "justice"), # animal-abstract
        ("cat", "dog"),       # animal-animal (same family)
    ]

    # Select injection layers (2 representative points to save time)
    inject_configs = []
    for l_start in [6, 18, 30]:
        l_end = l_start + 2
        if l_start < n_layers - 2 and l_end <= n_layers and l_start >= 1:
            inject_configs.append((l_start, l_end))

    print(f"  Test pairs: {len(test_pairs)}")
    print(f"  Injection configs: {inject_configs}")
    print(f"  Epsilon: {epsilon}")

    ref_prompt = "The cat is"
    lie_results = {}

    for l_start, l_end in inject_configs:
        transformer_idx = l_start - 1
        if transformer_idx >= len(layers):
            continue

        # Baseline at l_end
        inputs = tokenizer(ref_prompt, return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        with torch.no_grad():
            out_base = model(input_ids=input_ids, attention_mask=attention_mask,
                            output_hidden_states=True)
        h_baseline = safe_h(out_base.hidden_states[l_end])
        del input_ids, attention_mask, out_base

        pair_results = []

        for c1, c2 in test_pairs:
            d1 = concept_deltas[c1].get(l_start)
            d2 = concept_deltas[c2].get(l_start)
            if d1 is None or d2 is None:
                continue
            d1_norm = np.linalg.norm(d1)
            d2_norm = np.linalg.norm(d2)
            if d1_norm < 1e-10 or d2_norm < 1e-10:
                continue
            d1_unit = d1 / d1_norm
            d2_unit = d2 / d2_norm

            # 1. JVP for d2 at BASELINE: inject ε*d2 → δ2_base
            h_d2 = inject_and_read(model, tokenizer, input_device, ref_prompt, layers,
                                   transformer_idx, d2_unit, epsilon, l_end)
            delta2_base = h_d2 - h_baseline

            # 2. JVP for d1 alone: inject ε*d1 → δ1
            h_d1 = inject_and_read(model, tokenizer, input_device, ref_prompt, layers,
                                   transformer_idx, d1_unit, epsilon, l_end)
            delta1 = h_d1 - h_baseline

            # 3. JVP for d1+d2 combined: inject ε*(d1+d2) → δ12
            h_d12 = inject_dual_and_read(model, tokenizer, input_device, ref_prompt, layers,
                                         transformer_idx, d1_unit, d2_unit, epsilon, l_end)
            delta12 = h_d12 - h_baseline

            # 4. JVP for d2 at d1-perturbed state:
            #    δ2_pert = δ12 - δ1 (by linearity of the JVP at the perturbed state)
            #    But this is only approximate because the JVP is linearized
            #    More precisely: δ2_pert = (F(h+ε*d1+ε*d2) - F(h+ε*d1)) / ε
            #    ≈ (δ12 - δ1) / ε * ε = δ12 - δ1 (if we scale properly)
            delta2_pert = delta12 - delta1

            # 5. Lie bracket measure: how different is the Jacobian at the perturbed state?
            lie_cos = cosine_sim(delta2_base, delta2_pert)
            lie_deviation = 1.0 - lie_cos

            # Also compute the "reverse" Lie bracket: d1 at d2-perturbed state
            delta1_pert = delta12 - delta2_base
            lie_cos_rev = cosine_sim(delta1, delta1_pert)
            lie_deviation_rev = 1.0 - lie_cos_rev

            # The actual Lie bracket measure: difference between the two orderings
            # [J_d1, J_d2] = (J_d1 at d2-state)(J_d2) - (J_d2 at d1-state)(J_d1)
            # Approximated by: δ2_pert - δ2_base and δ1_pert - δ1
            commutator = np.linalg.norm(delta2_pert - delta2_base)
            base_norm = max(np.linalg.norm(delta2_base), 1e-10)
            commutator_relative = commutator / base_norm

            same_family = CONCEPT_TO_FAMILY[c1] == CONCEPT_TO_FAMILY[c2]

            result = {
                "pair": f"{c1}-{c2}",
                "same_family": same_family,
                "lie_deviation_d2": round(lie_deviation, 4),
                "lie_deviation_d1": round(lie_deviation_rev, 4),
                "commutator_relative": round(commutator_relative, 4),
                "delta2_base_norm": round(float(np.linalg.norm(delta2_base)), 4),
                "delta2_pert_norm": round(float(np.linalg.norm(delta2_pert)), 4),
            }
            pair_results.append(result)

            del h_d1, h_d2, h_d12, delta1, delta2_base, delta12, delta2_pert

        # Summary
        all_lie_dev = [r["lie_deviation_d2"] for r in pair_results]
        all_comm = [r["commutator_relative"] for r in pair_results]

        layer_summary = {
            "avg_lie_deviation": round(float(np.mean(all_lie_dev)), 4) if all_lie_dev else 0,
            "avg_commutator_relative": round(float(np.mean(all_comm)), 4) if all_comm else 0,
            "n_pairs": len(pair_results),
        }

        print(f"    L{l_start}→L{l_end}: avg_lie_dev={layer_summary['avg_lie_deviation']:.4f}, "
              f"avg_commutator={layer_summary['avg_commutator_relative']:.4f}")

        lie_results[f"L{l_start}_L{l_end}"] = {
            "pairs": pair_results,
            "summary": layer_summary,
        }

    # Overall
    all_dev = [lr["summary"]["avg_lie_deviation"] for lr in lie_results.values()]
    all_comm = [lr["summary"]["avg_commutator_relative"] for lr in lie_results.values()]

    print(f"\n  === Lie Bracket Results ===")
    print(f"  Overall avg Lie deviation: {np.mean(all_dev):.4f}")
    print(f"  Overall avg commutator: {np.mean(all_comm):.4f}")

    results = {
        "lie_by_layer": lie_results,
        "overall": {
            "avg_lie_deviation": round(float(np.mean(all_dev)), 4) if all_dev else 0,
            "avg_commutator_relative": round(float(np.mean(all_comm)), 4) if all_comm else 0,
            "state_dependent_jacobian": float(np.mean(all_dev)) > 0.05,
        },
        "epsilon": epsilon,
    }

    return results


# ===== EXP 4: GEODESIC STRUCTURE ★★ =====

def exp4_geodesic_structure(concept_deltas, concepts, sample_layers):
    """
    ★★ Geodesic Structure — 测地线结构 ★★

    KEY QUESTION: Is the concept manifold flat or curved?

    Method:
    1. Triangle inequality test:
       - For concept triples (A, B, C) in the same family
       - Compute cosine distances: d(A,B), d(B,C), d(A,C)
       - Triangle ratio = d(A,C) / (d(A,B) + d(B,C))
       - In flat space: d(A,C) ≤ d(A,B) + d(B,C) (triangle inequality)
       - Ratio ≈ 1 → B is on the geodesic from A to C
       - Ratio < 0.5 → A,B,C are not on a line (curved or dispersed)
       - If ratios vary significantly → manifold has curvature

    2. Linear interpolation test:
       - At layer l, for concepts A and B
       - Linear midpoint: h_mid = (h_A + h_B) / 2
       - Is h_mid close to any actual concept's representation?
       - If yes → the space is "flat" (linear interpolation works)
       - If no → the space is "curved" (linear interpolation leaves the manifold)

    INTERPRETATION:
    - Low triangle ratio variance → flat manifold → Euclidean geometry
    - High triangle ratio variance → curved manifold → Riemannian geometry
    - Linear interpolation works → flat
    - Linear interpolation fails → curved
    """
    print("\n" + "="*60)
    print("EXP 4: Geodesic Structure ★★")
    print("="*60)

    concept_list = sorted(concepts)
    sorted_layers = sorted(sample_layers)
    concrete_families = {"animal", "vehicle", "color"}
    abstract_families = {"abstract", "emotion"}

    # Step 1: Triangle inequality test
    print(f"  Step 1: Triangle inequality test...")

    geodesic_by_layer = {}

    for l in sorted_layers:
        # Build concept vectors
        M = np.array([concept_deltas[c][l] for c in concept_list])
        M = np.nan_to_num(M, nan=0.0, posinf=1e4, neginf=-1e4)

        # Compute pairwise cosine distances
        n = len(concept_list)
        cos_dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                cos = cosine_sim(M[i], M[j])
                cos_dist[i, j] = 1 - cos
                cos_dist[j, i] = cos_dist[i, j]

        # Triangle inequality: for triples within the same family
        triangle_ratios = []
        concrete_ratios = []
        abstract_ratios = []
        cross_ratios = []

        for fam, members in CONCEPT_FAMILIES.items():
            # Get indices of family members
            fam_indices = [concept_list.index(c) for c in members if c in concept_list]

            # All triples
            for i_idx in range(len(fam_indices)):
                for j_idx in range(i_idx+1, len(fam_indices)):
                    for k_idx in range(j_idx+1, len(fam_indices)):
                        ai, bi, ci = fam_indices[i_idx], fam_indices[j_idx], fam_indices[k_idx]

                        d_ab = cos_dist[ai, bi]
                        d_bc = cos_dist[bi, ci]
                        d_ac = cos_dist[ai, ci]

                        if d_ab + d_bc > 1e-10:
                            ratio = d_ac / (d_ab + d_bc)
                            triangle_ratios.append(ratio)
                            if fam in concrete_families:
                                concrete_ratios.append(ratio)
                            else:
                                abstract_ratios.append(ratio)

        # Cross-family triangles
        family_names = list(CONCEPT_FAMILIES.keys())
        for fi in range(len(family_names)):
            for fj in range(fi+1, len(family_names)):
                fam_i_members = [concept_list.index(c) for c in CONCEPT_FAMILIES[family_names[fi]] if c in concept_list][:3]
                fam_j_members = [concept_list.index(c) for c in CONCEPT_FAMILIES[family_names[fj]] if c in concept_list][:3]

                for ai in fam_i_members[:2]:
                    for bi in fam_j_members[:2]:
                        for ci in fam_i_members[:2]:
                            if ai == ci:
                                continue
                            d_ab = cos_dist[ai, bi]
                            d_bc = cos_dist[bi, ci]
                            d_ac = cos_dist[ai, ci]
                            if d_ab + d_bc > 1e-10:
                                ratio = d_ac / (d_ab + d_bc)
                                cross_ratios.append(ratio)

        # Linear interpolation test
        # For each concept pair, check if the midpoint is close to any other concept
        interp_deviations = []
        for i in range(min(n, 20)):
            for j in range(i+1, min(n, 20)):
                h_mid = (M[i] + M[j]) / 2
                # Find closest concept to h_mid
                best_cos = -1
                for k in range(n):
                    if k == i or k == j:
                        continue
                    cos = cosine_sim(h_mid, M[k])
                    best_cos = max(best_cos, cos)
                # The deviation is how far the midpoint is from ANY concept
                # If best_cos is high → midpoint is close to some concept → flat
                # If best_cos is low → midpoint is in "empty space" → curved
                interp_deviations.append(1 - best_cos)

        geodesic_by_layer[l] = {
            "avg_triangle_ratio": round(float(np.mean(triangle_ratios)), 4) if triangle_ratios else 0,
            "triangle_ratio_std": round(float(np.std(triangle_ratios)), 4) if triangle_ratios else 0,
            "concrete_avg_ratio": round(float(np.mean(concrete_ratios)), 4) if concrete_ratios else 0,
            "abstract_avg_ratio": round(float(np.mean(abstract_ratios)), 4) if abstract_ratios else 0,
            "cross_family_avg_ratio": round(float(np.mean(cross_ratios)), 4) if cross_ratios else 0,
            "avg_interp_deviation": round(float(np.mean(interp_deviations)), 4) if interp_deviations else 0,
            "n_triangles": len(triangle_ratios),
            "n_cross_triangles": len(cross_ratios),
        }

        if l in [0, 4, 12, 20, 28, 36]:
            print(f"    L{l}: ratio={geodesic_by_layer[l]['avg_triangle_ratio']:.4f}±"
                  f"{geodesic_by_layer[l]['triangle_ratio_std']:.4f}, "
                  f"concrete={geodesic_by_layer[l]['concrete_avg_ratio']:.4f}, "
                  f"abstract={geodesic_by_layer[l]['abstract_avg_ratio']:.4f}, "
                  f"interp_dev={geodesic_by_layer[l]['avg_interp_deviation']:.4f}")

    # Step 2: Track triangle ratio evolution across layers
    print(f"\n  Step 2: Triangle ratio evolution...")

    ratios_over_layers = [geodesic_by_layer[l]["avg_triangle_ratio"] for l in sorted_layers]
    concrete_over_layers = [geodesic_by_layer[l]["concrete_avg_ratio"] for l in sorted_layers]
    abstract_over_layers = [geodesic_by_layer[l]["abstract_avg_ratio"] for l in sorted_layers]
    interp_over_layers = [geodesic_by_layer[l]["avg_interp_deviation"] for l in sorted_layers]

    # Curvature measure: how much does the triangle ratio change across layers?
    ratio_variance = float(np.var(ratios_over_layers))
    concrete_abstract_gap = float(np.mean(concrete_over_layers)) - float(np.mean(abstract_over_layers))

    print(f"  Triangle ratio variance across layers: {ratio_variance:.6f}")
    print(f"  Concrete-abstract gap: {concrete_abstract_gap:.4f}")
    print(f"  Avg interp deviation: {np.mean(interp_over_layers):.4f}")

    results = {
        "geodesic_by_layer": geodesic_by_layer,
        "overall": {
            "avg_triangle_ratio": round(float(np.mean(ratios_over_layers)), 4),
            "ratio_variance": round(ratio_variance, 6),
            "concrete_avg": round(float(np.mean(concrete_over_layers)), 4),
            "abstract_avg": round(float(np.mean(abstract_over_layers)), 4),
            "concrete_abstract_gap": round(concrete_abstract_gap, 4),
            "avg_interp_deviation": round(float(np.mean(interp_over_layers)), 4),
            "curved_manifold": ratio_variance > 0.001 or abs(concrete_abstract_gap) > 0.05,
        },
    }

    return results


# ===== MAIN =====

def main():
    import torch

    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    print(f"\n{'='*60}")
    print(f"Phase 171: Geometric Dynamics — 几何动力学")
    print(f"Model: {model_name}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")

    # Load model
    model, tokenizer, device, use_8bit = load_model_auto_bf16(model_name)
    info = get_model_info(model, model_name)
    print(f"[0] Model: {info.model_class}, L={info.n_layers}, d={info.d_model}, "
          f"V={info.vocab_size}, 8bit={use_8bit}")

    # Sample layers
    n_layers = info.n_layers
    sample_layers = list(range(0, n_layers + 1, 2))
    sample_layers = sorted(set(sample_layers))
    if n_layers not in sample_layers:
        sample_layers.append(n_layers)

    print(f"[0] Sample layers: {sample_layers}")

    # Use ALL 50 concepts and ALL 8 templates for robust results
    concepts = ALL_CONCEPTS
    templates = CONTEXT_TEMPLATES

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

    # ===== Exp 1: Holonomy =====
    print(f"\n[2] Exp 1: Holonomy — Non-Commutative Transport")
    t0 = time.time()
    all_results["exp1_holonomy"] = exp1_holonomy(
        model, tokenizer, device, model_name, use_8bit,
        concept_deltas, sample_layers, epsilon=1.0)
    print(f"  Time: {time.time()-t0:.1f}s")

    # ===== Exp 2: Generator Field =====
    print(f"\n[3] Exp 2: Generator Field — log(J_l)")
    t0 = time.time()
    all_results["exp2_generator"] = exp2_generator_field(
        model, tokenizer, device, model_name, use_8bit,
        sample_layers, n_random=30, epsilon=1.0)
    print(f"  Time: {time.time()-t0:.1f}s")

    # ===== Exp 3: Lie Bracket =====
    print(f"\n[4] Exp 3: Lie Bracket — State-Dependent Jacobian")
    t0 = time.time()
    all_results["exp3_lie_bracket"] = exp3_lie_bracket(
        model, tokenizer, device, model_name, use_8bit,
        concept_deltas, sample_layers, epsilon=1.0)
    print(f"  Time: {time.time()-t0:.1f}s")

    # ===== Exp 4: Geodesic Structure =====
    print(f"\n[5] Exp 4: Geodesic Structure")
    t0 = time.time()
    all_results["exp4_geodesic"] = exp4_geodesic_structure(
        concept_deltas, concepts, sample_layers)
    print(f"  Time: {time.time()-t0:.1f}s")

    # ===== Save Results =====
    os.makedirs("tests/glm5_temp", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = f"tests/glm5_temp/phase171_{model_name}_{timestamp}.json"

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

    print(f"\nPhase 171 complete for {model_name}!")


if __name__ == "__main__":
    main()
