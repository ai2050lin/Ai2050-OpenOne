"""
Phase 143: Propagation Geometry
================================
From "Semantic Manifold" to "Constrained Propagation System"

Directly addresses user's core critique:
1. "Stable propagation subspace" ≠ "Semantic manifold"
   - Low-rank dynamics also produce semantic >> random
   - Need to test if system is locally linear or piecewise
2. Transformer may be piecewise (LayerNorm, softmax, MLP gating)
3. Core object is "constrained propagation", not "semantic geometry"
4. Need controllability/observability analysis

Three Experiments:

Exp 1: Local Linearity Test (Critical)
- Compare J_{l+1}(h)v at two nearby points h, h'
- If consistent → locally linear → manifold approximately valid
- If inconsistent → piecewise → stratified dynamics
- This is THE test that distinguishes manifold from piecewise system

Exp 2: Observability Landscape
- Inject perturbations at each layer, measure output impact
- Map: layer × direction → observability = ||Δlogits|| / ||δ||
- Identify: which perturbations are "visible" vs "invisible" to decoder?
- This defines the "observable subspace" (control theory view)

Exp 3: Propagation Corridors
- Track perturbation amplitude across layers
- For different injection directions: random, W_U top, semantic
- Identify: stable vs decaying propagation directions
- Key question: is there a "propagation corridor" that persists?

Usage:
  python tests/glm5/phase143_propagation_geometry.py qwen3
  python tests/glm5/phase143_propagation_geometry.py glm4
  python tests/glm5/phase143_propagation_geometry.py deepseek7b
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc
import time
import json
import numpy as np
import torch
from datetime import datetime
from scipy.sparse.linalg import svds
from model_utils import (load_model, get_layers, get_model_info,
                          release_model, get_W_U, MODEL_CONFIGS,
                          get_sample_layers)

# ===== Configuration =====
N_SENTENCE_PAIRS = 12  # Number of base-negated sentence pairs
N_SAMPLE_LAYERS = 6    # Layers to inject at
N_RANDOM_PROBES = 10   # Random probe vectors per injection point
N_WU_PROBES = 5        # W_U top singular vector probes
EPSILON = 0.005        # Perturbation size
N_SEMANTIC_PROBES = 3  # Semantic direction probes (for observability)
N_PROPAGATION_SAMPLES = 5  # Sentences for propagation test

# ===== Sentence Pairs =====
BASE_SENTENCES = [
    "The cat sat on the mat",
    "A dog runs in the park",
    "The student passed the exam",
    "Birds fly across the sky",
    "The scientist discovered a cure",
    "Children play in the garden",
    "The musician played a song",
    "A teacher explained the theory",
    "The artist painted a portrait",
    "The chef prepared a meal",
    "Engineers build tall bridges",
    "The doctor treated the patient",
    "Farmers grow fresh vegetables",
    "The driver crossed the bridge",
    "A programmer wrote the code",
]

# Verb mapping for negation
VERB_MAP = {
    'sat': 'did not sit', 'runs': 'does not run', 'passed': 'did not pass',
    'fly': 'do not fly', 'discovered': 'did not discover', 'play': 'do not play',
    'played': 'did not play', 'explained': 'did not explain', 'painted': 'did not paint',
    'prepared': 'did not prepare', 'build': 'do not build', 'treated': 'did not treat',
    'grow': 'do not grow', 'crossed': 'did not cross', 'wrote': 'did not write',
}

def make_negated(s):
    """Create negated version by replacing main verb"""
    words = s.split()
    for i, w in enumerate(words):
        if w.lower() in VERB_MAP:
            words[i] = VERB_MAP[w.lower()]
            return ' '.join(words)
    return f"It is not true that {s.lower()}"


def get_device_for_input(model):
    """Get device for input tensors"""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===== Core Functions =====

def capture_all_hidden_states(model, tokenizer, device, sentence, n_layers):
    """Run forward pass and capture hidden states at all layers + logits"""
    input_device = get_device_for_input(model)
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(input_device)
    attention_mask = inputs["attention_mask"].to(input_device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                       output_hidden_states=True)

    hs = outputs.hidden_states  # tuple of (n_layers+1,)
    logits = outputs.logits[0, -1].float().cpu().numpy()

    # Extract last token hidden state at each layer
    hidden = {}
    for l in range(len(hs)):
        hidden[l] = hs[l][0, -1, :].float().cpu().numpy()  # [d_model]

    return hidden, logits


def inject_at_layer_and_capture(model, tokenizer, device, sentence,
                                 inject_layer, inject_direction_np,
                                 epsilon, n_layers):
    """
    Inject perturbation at the output of inject_layer and capture all hidden states.

    The injection modifies the residual stream AFTER inject_layer.
    The response at inject_layer+1 tells us about the Jacobian of that layer.

    Returns:
        hidden: dict of {layer_idx: np.ndarray[d_model]} (last token)
        logits: np.ndarray[vocab_size]
    """
    input_device = get_device_for_input(model)
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(input_device)
    attention_mask = inputs["attention_mask"].to(input_device)

    layers = get_layers(model)
    injected = [False]
    direction_tensor = torch.tensor(inject_direction_np, dtype=torch.float32, device=input_device)

    def inject_hook(module, input, output):
        if not injected[0]:
            if isinstance(output, tuple):
                h = output[0].clone()
                h[:, -1, :] += epsilon * direction_tensor.to(h.dtype)
                injected[0] = True
                return (h,) + output[1:]
            else:
                h = output.clone()
                h[:, -1, :] += epsilon * direction_tensor.to(h.dtype)
                injected[0] = True
                return h
        return output

    # Register injection hook
    hook = layers[inject_layer].register_forward_hook(inject_hook)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                       output_hidden_states=True)

    hook.remove()

    hs = outputs.hidden_states
    logits = outputs.logits[0, -1].float().cpu().numpy()

    hidden = {}
    for l in range(len(hs)):
        hidden[l] = hs[l][0, -1, :].float().cpu().numpy()

    return hidden, logits


def compute_wu_top_directions(W_U, n_directions=5):
    """Compute top singular vectors of W_U (most 'readable' directions)"""
    W_U_T = W_U.T.astype(np.float32)
    k = min(n_directions, min(W_U_T.shape) - 2)
    U, s, Vt = svds(W_U_T, k=k)
    # U: [d_model, k] - top right singular vectors of W_U
    # These are the directions that W_U is most sensitive to
    # Sort by singular value (descending)
    idx = np.argsort(s)[::-1]
    U = U[:, idx]
    directions = []
    for i in range(k):
        d = U[:, i].copy()
        d = d / np.linalg.norm(d)
        directions.append(d)
    return directions, s[idx]


# ===== Experiment Functions =====

def exp1_local_linearity(model, tokenizer, device, model_info, W_U,
                          sentence_pairs, sample_layers):
    """
    Exp 1: Local Linearity Test

    For each sentence pair (s, s_not), at each layer l:
    - Compute J_{l+1}(h_{l+1}(s)) v and J_{l+1}(h_{l+1}(s_not)) v
    - Compare their cosine similarity
    - If high (>0.95): locally linear → manifold approximately valid
    - If low (<0.7): piecewise → stratified dynamics

    This directly tests whether the Jacobian is approximately constant
    in a local neighborhood, which is the key distinction between
    smooth manifold and piecewise dynamical system.
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model

    # Generate probe vectors
    np.random.seed(42)
    random_probes = [np.random.randn(d_model) for _ in range(N_RANDOM_PROBES)]
    random_probes = [v / np.linalg.norm(v) for v in random_probes]

    wu_directions, wu_singular_values = compute_wu_top_directions(W_U, N_WU_PROBES)
    all_probes = random_probes + wu_directions
    probe_labels = [f"random_{i}" for i in range(N_RANDOM_PROBES)] + \
                   [f"wu_top_{i}" for i in range(N_WU_PROBES)]

    results = {
        "n_pairs": len(sentence_pairs),
        "sample_layers": sample_layers,
        "n_probes": len(all_probes),
        "probe_labels": probe_labels,
        "epsilon": EPSILON,
        "layer_results": {},
        "pair_details": [],
    }

    print(f"\n{'='*60}")
    print(f"Exp 1: Local Linearity Test")
    print(f"  {len(sentence_pairs)} pairs, {len(sample_layers)} layers, {len(all_probes)} probes")
    print(f"{'='*60}")

    for pair_idx, (s_base, s_neg) in enumerate(sentence_pairs):
        print(f"\n  Pair {pair_idx+1}/{len(sentence_pairs)}: '{s_base[:40]}...' → '{s_neg[:40]}...'")
        t0 = time.time()

        # Capture base hidden states for both sentences
        h_base, logits_base = capture_all_hidden_states(model, tokenizer, device, s_base, n_layers)
        h_neg, logits_neg = capture_all_hidden_states(model, tokenizer, device, s_neg, n_layers)

        pair_result = {
            "base": s_base,
            "negated": s_neg,
            "layer_linearity": {},
        }

        for l_idx, inject_l in enumerate(sample_layers):
            # We inject at the output of layer inject_l
            # Response at layer inject_l+1 gives J_{inject_l+1} v
            # Compare at base point h_{inject_l+1}(s_base) vs h_{inject_l+1}(s_neg)

            # Make sure inject_l+1 exists
            response_l = inject_l + 1
            if response_l >= n_layers + 1:
                continue

            linearities = []
            jv_base_list = []
            jv_neg_list = []

            for p_idx, probe in enumerate(all_probes):
                # Injection at base sentence
                h_inj_base, logits_inj_base = inject_at_layer_and_capture(
                    model, tokenizer, device, s_base,
                    inject_l, probe, EPSILON, n_layers
                )

                # Injection at negated sentence
                h_inj_neg, logits_inj_neg = inject_at_layer_and_capture(
                    model, tokenizer, device, s_neg,
                    inject_l, probe, EPSILON, n_layers
                )

                # Jacobian-vector products
                # J_{l+1} v ≈ (h_{l+2}(inj) - h_{l+2}(base)) / ε
                # But we injected at layer l, so the response is at layer l+1
                # (in hidden_states indexing: inject_l+1 → response at inject_l+2)
                if inject_l + 2 < n_layers + 1:
                    jv_base = (h_inj_base[inject_l + 2] - h_base[inject_l + 2]) / EPSILON
                    jv_neg = (h_inj_neg[inject_l + 2] - h_neg[inject_l + 2]) / EPSILON
                else:
                    # Last layer - compare final hidden states
                    jv_base = (h_inj_base[inject_l + 1] - h_base[inject_l + 1]) / EPSILON
                    jv_neg = (h_inj_neg[inject_l + 1] - h_neg[inject_l + 1]) / EPSILON

                # Cosine similarity between Jacobian-vector products
                norm_base = np.linalg.norm(jv_base)
                norm_neg = np.linalg.norm(jv_neg)

                if norm_base > 1e-8 and norm_neg > 1e-8:
                    cos_sim = float(np.dot(jv_base, jv_neg) / (norm_base * norm_neg))
                else:
                    cos_sim = 0.0

                linearities.append(cos_sim)
                jv_base_list.append(norm_base)
                jv_neg_list.append(norm_neg)

            # Aggregate results for this layer
            lin_arr = np.array(linearities)
            pair_result["layer_linearity"][str(inject_l)] = {
                "mean_cos": float(np.mean(lin_arr)),
                "std_cos": float(np.std(lin_arr)),
                "median_cos": float(np.median(lin_arr)),
                "min_cos": float(np.min(lin_arr)),
                "mean_jv_norm_base": float(np.mean(jv_base_list)),
                "mean_jv_norm_neg": float(np.mean(jv_neg_list)),
                "random_mean_cos": float(np.mean(lin_arr[:N_RANDOM_PROBES])),
                "wu_mean_cos": float(np.mean(lin_arr[N_RANDOM_PROBES:])),
            }

            if l_idx == 0 or (pair_idx + 1) % 4 == 0:
                print(f"    L{inject_l}: mean_cos={np.mean(lin_arr):.4f}, "
                      f"random={np.mean(lin_arr[:N_RANDOM_PROBES]):.4f}, "
                      f"W_U={np.mean(lin_arr[N_RANDOM_PROBES:]):.4f}")

        results["pair_details"].append(pair_result)
        elapsed = time.time() - t0
        print(f"    Time: {elapsed:.1f}s")

    # Aggregate across all pairs
    for inject_l in sample_layers:
        layer_key = str(inject_l)
        all_cos = [p["layer_linearity"][layer_key]["mean_cos"]
                   for p in results["pair_details"] if layer_key in p["layer_linearity"]]
        all_random = [p["layer_linearity"][layer_key]["random_mean_cos"]
                      for p in results["pair_details"] if layer_key in p["layer_linearity"]]
        all_wu = [p["layer_linearity"][layer_key]["wu_mean_cos"]
                  for p in results["pair_details"] if layer_key in p["layer_linearity"]]

        results["layer_results"][layer_key] = {
            "mean_cos": float(np.mean(all_cos)),
            "std_cos": float(np.std(all_cos)),
            "random_mean_cos": float(np.mean(all_random)),
            "wu_mean_cos": float(np.mean(all_wu)),
        }

    return results


def exp2_observability(model, tokenizer, device, model_info, W_U,
                       sentence_pairs, sample_layers):
    """
    Exp 2: Observability Landscape

    For each injection (layer, direction), measure how much the output
    logits change. This defines the "observable subspace" of the system.

    Key insight from user: not all perturbations are observable.
    The LM head (W_U) defines what's "visible" in the output.
    Many hidden perturbations may be "invisible" to the decoder.

    Observability(l, v) = ||Δlogits|| / ||ε * v||

    Compare:
    - Semantic directions (NOT) vs random directions
    - W_U top directions vs random directions
    - Early layers vs late layers
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model

    # Generate probe vectors
    np.random.seed(42)
    random_probes = [np.random.randn(d_model) for _ in range(N_RANDOM_PROBES)]
    random_probes = [v / np.linalg.norm(v) for v in random_probes]
    wu_directions, _ = compute_wu_top_directions(W_U, N_WU_PROBES)

    results = {
        "n_pairs": len(sentence_pairs),
        "sample_layers": sample_layers,
        "epsilon": EPSILON,
        "layer_observability": {},
        "direction_comparison": {},
    }

    print(f"\n{'='*60}")
    print(f"Exp 2: Observability Landscape")
    print(f"  {len(sentence_pairs)} sentences, {len(sample_layers)} layers")
    print(f"{'='*60}")

    all_obs_random = {l: [] for l in sample_layers}
    all_obs_wu = {l: [] for l in sample_layers}
    all_obs_semantic = {l: [] for l in sample_layers}

    for s_idx, (s_base, s_neg) in enumerate(sentence_pairs):
        print(f"  Sentence {s_idx+1}/{len(sentence_pairs)}: '{s_base[:40]}...'")
        t0 = time.time()

        # Base forward pass
        h_base, logits_base = capture_all_hidden_states(model, tokenizer, device, s_base, n_layers)

        # Semantic direction: difference between negated and base at each layer
        h_neg, logits_neg = capture_all_hidden_states(model, tokenizer, device, s_neg, n_layers)

        for inject_l in sample_layers:
            obs_random = []
            obs_wu = []
            obs_semantic = []

            # Test random probes
            for probe in random_probes[:3]:  # Use 3 random for efficiency
                h_inj, logits_inj = inject_at_layer_and_capture(
                    model, tokenizer, device, s_base,
                    inject_l, probe, EPSILON, n_layers
                )
                delta_logits = logits_inj - logits_base
                obs = float(np.linalg.norm(delta_logits)) / EPSILON
                obs_random.append(obs)

            # Test W_U probes
            for probe in wu_directions[:3]:
                h_inj, logits_inj = inject_at_layer_and_capture(
                    model, tokenizer, device, s_base,
                    inject_l, probe, EPSILON, n_layers
                )
                delta_logits = logits_inj - logits_base
                obs = float(np.linalg.norm(delta_logits)) / EPSILON
                obs_wu.append(obs)

            # Test semantic direction (NOT)
            # Use the NOT direction at the injection layer
            not_direction = h_neg[inject_l + 1] - h_base[inject_l + 1]
            not_norm = np.linalg.norm(not_direction)
            if not_norm > 1e-8:
                not_direction_normalized = not_direction / not_norm
                h_inj, logits_inj = inject_at_layer_and_capture(
                    model, tokenizer, device, s_base,
                    inject_l, not_direction_normalized, EPSILON, n_layers
                )
                delta_logits = logits_inj - logits_base
                obs = float(np.linalg.norm(delta_logits)) / EPSILON
                obs_semantic.append(obs)

            all_obs_random[inject_l].extend(obs_random)
            all_obs_wu[inject_l].extend(obs_wu)
            all_obs_semantic[inject_l].extend(obs_semantic)

        elapsed = time.time() - t0
        print(f"    Time: {elapsed:.1f}s")

    # Aggregate
    for inject_l in sample_layers:
        r = all_obs_random[inject_l]
        w = all_obs_wu[inject_l]
        s = all_obs_semantic[inject_l]

        results["layer_observability"][str(inject_l)] = {
            "random_mean": float(np.mean(r)) if r else 0,
            "random_std": float(np.std(r)) if r else 0,
            "wu_mean": float(np.mean(w)) if w else 0,
            "wu_std": float(np.std(w)) if w else 0,
            "semantic_mean": float(np.mean(s)) if s else 0,
            "semantic_std": float(np.std(s)) if s else 0,
            "wu_random_ratio": float(np.mean(w) / max(np.mean(r), 1e-10)) if r and w else 0,
            "semantic_random_ratio": float(np.mean(s) / max(np.mean(r), 1e-10)) if r and s else 0,
        }

    return results


def exp3_propagation_corridors(model, tokenizer, device, model_info, W_U,
                                sentences, sample_layers):
    """
    Exp 3: Propagation Corridors

    Track how perturbation amplitude evolves across layers.
    For injection at layer l, measure:
    - ||Δh_{l+2}||, ||Δh_{l+3}||, ..., ||Δh_{L+1}||
    - This reveals the "propagation corridor" structure

    Key question: which directions survive, which decay?
    Compare: random vs W_U top vs semantic directions.
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model

    # Use fewer sentences for this experiment
    test_sentences = sentences[:N_PROPAGATION_SAMPLES]

    # Injection layers (early, middle, late)
    inject_layers = [
        sample_layers[0],            # Early
        sample_layers[len(sample_layers)//2],  # Middle
        sample_layers[-2] if len(sample_layers) > 2 else sample_layers[-1],  # Late
    ]

    # Direction types
    np.random.seed(42)
    random_dir = np.random.randn(d_model)
    random_dir = random_dir / np.linalg.norm(random_dir)

    wu_directions, wu_sv = compute_wu_top_directions(W_U, 2)
    wu_top_dir = wu_directions[0] if wu_directions else random_dir

    # We'll also test a "W_U bottom" direction (least observable)
    if len(wu_directions) > 1:
        wu_bottom_dir = wu_directions[-1]
    else:
        wu_bottom_dir = random_dir

    directions = {
        "random": random_dir,
        "wu_top1": wu_top_dir,
        "wu_bottom": wu_bottom_dir,
    }

    results = {
        "inject_layers": inject_layers,
        "directions": list(directions.keys()),
        "epsilon": EPSILON,
        "propagation_profiles": {},
    }

    print(f"\n{'='*60}")
    print(f"Exp 3: Propagation Corridors")
    print(f"  {len(test_sentences)} sentences, {len(inject_layers)} injection layers")
    print(f"  Directions: {list(directions.keys())}")
    print(f"{'='*60}")

    for inject_l in inject_layers:
        results["propagation_profiles"][str(inject_l)] = {}

        for dir_name, direction in directions.items():
            # Collect propagation profiles across sentences
            all_profiles = []

            for s_idx, sentence in enumerate(test_sentences):
                # Base forward pass
                h_base, _ = capture_all_hidden_states(model, tokenizer, device, sentence, n_layers)

                # Injection forward pass
                h_inj, _ = inject_at_layer_and_capture(
                    model, tokenizer, device, sentence,
                    inject_l, direction, EPSILON, n_layers
                )

                # Compute perturbation amplitude at each subsequent layer
                # h_base[l] and h_inj[l] are the hidden states at "layer l-1"
                # (since hidden_states[0] = embedding, hidden_states[1] = output of layer 0)
                profile = {}
                for l in range(inject_l + 1, n_layers + 1):
                    delta = h_inj[l] - h_base[l]
                    amp = float(np.linalg.norm(delta))
                    profile[str(l)] = amp

                all_profiles.append(profile)

            # Average across sentences
            avg_profile = {}
            for l_key in all_profiles[0].keys():
                avg_profile[l_key] = float(np.mean([p[l_key] for p in all_profiles]))

            results["propagation_profiles"][str(inject_l)][dir_name] = {
                "avg_profile": avg_profile,
                "n_sentences": len(test_sentences),
            }

            # Print summary
            injected_amps = [avg_profile[k] for k in sorted(avg_profile.keys(), key=int)]
            print(f"  Inject L{inject_l}, {dir_name}: "
                  f"first={injected_amps[0]:.4f}, "
                  f"peak={max(injected_amps):.4f}, "
                  f"last={injected_amps[-1]:.4f}")

    return results


# ===== Main =====

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    print(f"\n{'#'*60}")
    print(f"Phase 143: Propagation Geometry — {model_name}")
    print(f"{'#'*60}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load model
    t0 = time.time()
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    print(f"Model: {model_info.model_class}, L={model_info.n_layers}, d={model_info.d_model}")

    W_U = get_W_U(model, model_name)
    print(f"W_U: shape={W_U.shape}")
    print(f"Load time: {time.time()-t0:.1f}s")

    # Prepare sentence pairs
    sentence_pairs = []
    for s in BASE_SENTENCES[:N_SENTENCE_PAIRS]:
        s_neg = make_negated(s)
        sentence_pairs.append((s, s_neg))

    # Sample layers
    sample_layers = get_sample_layers(model_info.n_layers, N_SAMPLE_LAYERS)
    print(f"Sample layers: {sample_layers}")

    # ===== Exp 1: Local Linearity =====
    print(f"\n{'='*60}")
    print("Starting Exp 1: Local Linearity Test")
    print(f"{'='*60}")
    t1 = time.time()
    exp1_results = exp1_local_linearity(model, tokenizer, device, model_info, W_U,
                                         sentence_pairs, sample_layers)
    print(f"Exp 1 completed in {time.time()-t1:.1f}s")

    # Print summary
    print(f"\n--- Exp 1 Summary: Local Linearity ---")
    for l_key, lr in exp1_results["layer_results"].items():
        verdict = "LINEAR" if lr["mean_cos"] > 0.95 else ("PARTIAL" if lr["mean_cos"] > 0.7 else "PIECEWISE")
        print(f"  L{l_key}: mean_cos={lr['mean_cos']:.4f} "
              f"(random={lr['random_mean_cos']:.4f}, W_U={lr['wu_mean_cos']:.4f}) → {verdict}")

    # ===== Exp 2: Observability =====
    print(f"\n{'='*60}")
    print("Starting Exp 2: Observability Landscape")
    print(f"{'='*60}")
    t2 = time.time()
    exp2_results = exp2_observability(model, tokenizer, device, model_info, W_U,
                                      sentence_pairs, sample_layers)
    print(f"Exp 2 completed in {time.time()-t2:.1f}s")

    # Print summary
    print(f"\n--- Exp 2 Summary: Observability ---")
    for l_key, lr in exp2_results["layer_observability"].items():
        print(f"  L{l_key}: random={lr['random_mean']:.2f}, "
              f"W_U={lr['wu_mean']:.2f}, semantic={lr['semantic_mean']:.2f}, "
              f"W_U/rand={lr['wu_random_ratio']:.2f}x, sem/rand={lr['semantic_random_ratio']:.2f}x")

    # ===== Exp 3: Propagation Corridors =====
    print(f"\n{'='*60}")
    print("Starting Exp 3: Propagation Corridors")
    print(f"{'='*60}")
    t3 = time.time()
    base_sentences = [s for s, _ in sentence_pairs]
    exp3_results = exp3_propagation_corridors(model, tokenizer, device, model_info, W_U,
                                               base_sentences, sample_layers)
    print(f"Exp 3 completed in {time.time()-t3:.1f}s")

    # ===== Save Results =====
    all_results = {
        "model_name": model_name,
        "model_info": {
            "class": model_info.model_class,
            "n_layers": model_info.n_layers,
            "d_model": model_info.d_model,
            "vocab_size": model_info.vocab_size,
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "n_sentence_pairs": N_SENTENCE_PAIRS,
            "n_sample_layers": N_SAMPLE_LAYERS,
            "n_random_probes": N_RANDOM_PROBES,
            "n_wu_probes": N_WU_PROBES,
            "epsilon": EPSILON,
        },
        "exp1_local_linearity": exp1_results,
        "exp2_observability": exp2_results,
        "exp3_propagation": exp3_results,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = f"tests/glm5_temp/phase143_{model_name}_propagation_{timestamp}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to: {output_path}")

    # ===== Final Verdict =====
    print(f"\n{'='*60}")
    print("PHASE 143 VERDICT")
    print(f"{'='*60}")

    # Exp 1 verdict
    all_mean_cos = [lr["mean_cos"] for lr in exp1_results["layer_results"].values()]
    overall_cos = np.mean(all_mean_cos)
    if overall_cos > 0.95:
        linearity_verdict = "LOCALLY LINEAR → manifold framework approximately valid"
    elif overall_cos > 0.8:
        linearity_verdict = "PARTIALLY LINEAR → manifold with some piecewise behavior"
    elif overall_cos > 0.6:
        linearity_verdict = "WEAKLY LINEAR → mixed manifold/piecewise"
    else:
        linearity_verdict = "STRONGLY PIECEWISE → stratified dynamics, NOT smooth manifold"

    print(f"  Exp 1 (Local Linearity): overall cos = {overall_cos:.4f}")
    print(f"    → {linearity_verdict}")

    # Exp 2 verdict
    all_wu_ratios = [lr["wu_random_ratio"] for lr in exp2_results["layer_observability"].values()
                     if lr["wu_random_ratio"] > 0]
    all_sem_ratios = [lr["semantic_random_ratio"] for lr in exp2_results["layer_observability"].values()
                      if lr["semantic_random_ratio"] > 0]
    if all_wu_ratios:
        avg_wu_ratio = np.mean(all_wu_ratios)
        print(f"  Exp 2 (Observability): W_U/random = {avg_wu_ratio:.2f}x")
        if avg_wu_ratio > 3:
            print(f"    → Strong decoder anisotropy: W_U directions are much more observable")
        elif avg_wu_ratio > 1.5:
            print(f"    → Moderate decoder anisotropy")
        else:
            print(f"    → Weak decoder anisotropy: decoder is approximately isotropic")
    if all_sem_ratios:
        avg_sem_ratio = np.mean(all_sem_ratios)
        print(f"    Semantic/random = {avg_sem_ratio:.2f}x")

    # Exp 3 verdict
    print(f"  Exp 3 (Propagation): see detailed profiles above")

    # Release model
    release_model(model)
    print(f"\nTotal time: {time.time()-t0:.1f}s")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
