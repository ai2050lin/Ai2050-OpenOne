"""
Phase 175: Causal Intervention — ★★★ From Observation to Causation! ★★★
=========================================================================

PARADIGM SHIFT: Phase 174 was "observation" (what structure exists).
                Phase 175 is "intervention" (what structure is NECESSARY).

User's core insight: We must distinguish "statistical co-occurrence structure"
from "true computational structure". Only CAUSAL INTERVENTION can do this.

★★★ FOUR KEY EXPERIMENTS ★★★

Exp 1: ★★★ Subspace Ablation (子空间删除) — THE MOST CRITICAL EXPERIMENT
  - Compute "fruit subspace" from PCA of fruit word hidden states
  - Project OUT (ablate) the fruit subspace from apple's hidden state
  - Measure: Does apple's logit for "fruit/food" decrease?
  - Does apple's logit for "vehicle/emotion" stay the same?
  - KEY: If deleting fruit subspace selectively impairs fruit-related outputs
    → Fruit subspace is CAUSALLY NECESSARY, not just correlated!

Exp 2: ★★★ Neuron-Level Category Selectivity (神经元类别选择性)
  - For each MLP intermediate neuron, compute its activation pattern
  - across fruits vs vehicles vs animals vs emotions
  - Find neurons that fire selectively for one category
  - Ablate those neurons → does it selectively impair that category?
  - KEY: Are there "fruit constraint modules"?

Exp 3: ★★★ Context-Dependent Concept Stability (语境依赖概念稳定性)
  - Test same word in different contexts:
    "The apple is" vs "I ate an apple" vs "Apple released" vs "an apple tree"
  - Is the "fruitness" subspace stable across contexts?
  - Does "Apple" (company) share ANY structure with "apple" (fruit)?
  - KEY: Is the concept representation truly compositional, or context-bound?

Exp 4: ★★★ Cross-Lingual Concept Invariant (跨语言概念不变量)
  - apple (EN) vs 苹果 (ZH) vs りんご (JA) vs pomme (FR) vs manzana (ES)
  - In a bilingual context: "The apple / 苹果 is a fruit"
  - Measure: Do these share the same subspace structure?
  - KEY: Is "fruitness" a language-independent computational structure?

CRITICAL METHODOLOGY:
- All ablations use HOOKS to intervene during forward pass
- Compare: normal output vs ablated output
- Measure: change in logit for category-relevant words
- Control: ablate random subspace of same dimension → expect no change

Usage: python tests/glm5/phase175_causal_intervention.py <model_name>
  model_name: qwen3, glm4, deepseek7b
"""

import sys
import os
import time
import json
import gc
import numpy as np
import torch
from datetime import datetime
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'glm5'))

from model_utils import get_layers, get_model_info, release_model, get_W_U, MODEL_CONFIGS


# =====================================================================
# MODEL LOADING (BF16 + device_map="auto")
# =====================================================================

def load_model_bf16(model_name):
    """BF16 + device_map=auto loading for all models"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = MODEL_CONFIGS[model_name]
    print(f"[bf16] Loading {model_name} (bfloat16 + device_map=auto)...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager",
    )
    model.eval()

    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"[bf16] {model_name} loaded: GPU={gpu_mem:.2f}GB", flush=True)

    return model, tokenizer, device


# =====================================================================
# CONCEPT WORDS AND CATEGORY LABELS
# =====================================================================

CATEGORIES = {
    "fruits": [
        "apple", "banana", "orange", "grape", "mango",
        "pear", "peach", "cherry", "lemon", "plum",
        "melon", "kiwi", "coconut", "papaya", "lime",
    ],
    "vehicles": [
        "car", "truck", "bus", "train", "plane",
        "boat", "bike", "van", "taxi", "tram",
        "helicopter", "rocket", "yacht", "canoe", "submarine",
    ],
    "animals": [
        "cat", "dog", "bird", "fish", "horse",
        "cow", "pig", "sheep", "goat", "lion",
        "tiger", "bear", "elephant", "monkey", "rabbit",
    ],
    "emotions": [
        "happy", "sad", "angry", "afraid", "joyful",
        "proud", "ashamed", "guilty", "envious", "calm",
        "anxious", "bored", "lonely", "grateful", "hopeful",
    ],
}

# Category verification words: words that strongly indicate the category
CATEGORY_VERIFICATION = {
    "fruits": ["fruit", "sweet", "juicy", "fresh", "ripe", "delicious", "tropical"],
    "vehicles": ["vehicle", "drive", "fast", "engine", "road", "transport", "wheel"],
    "animals": ["animal", "wild", "pet", "creature", "alive", "species", "mammal"],
    "emotions": ["emotion", "feel", "mood", "mental", "feeling", "express", "sentiment"],
}

# Cross-category control words: should NOT be affected by fruit subspace deletion
CROSS_CATEGORY_CONTROLS = {
    "fruits": ["car", "bus", "cat", "dog", "happy", "sad", "book", "chair"],
    "vehicles": ["apple", "banana", "cat", "dog", "happy", "sad", "book", "chair"],
    "animals": ["apple", "banana", "car", "bus", "happy", "sad", "book", "chair"],
    "emotions": ["apple", "banana", "car", "bus", "cat", "dog", "book", "chair"],
}

# Context variations for Exp 3
CONTEXT_VARIATIONS = {
    "apple": [
        "The apple is",
        "I ate an apple",
        "Apple released a new",
        "An apple tree",
        "The apple fell from",
        "She bought an apple",
    ],
    "car": [
        "The car is",
        "I drove a car",
        "The car crashed",
        "A car factory",
        "His car broke down",
        "She parked the car",
    ],
    "cat": [
        "The cat is",
        "I saw a cat",
        "The cat jumped",
        "A cat owner",
        "The black cat",
        "She fed the cat",
    ],
    "happy": [
        "The happy person",
        "She felt happy",
        "Happy birthday",
        "A happy ending",
        "The happy children",
        "He looked happy",
    ],
}

# Cross-lingual concept words for Exp 4
CROSS_LINGUAL = {
    "apple": {
        "en": "apple",
        "zh": "苹果",
        "ja": "りんご",
        "fr": "pomme",
        "es": "manzana",
    },
    "car": {
        "en": "car",
        "zh": "汽车",
        "ja": "車",
        "fr": "voiture",
        "es": "coche",
    },
    "cat": {
        "en": "cat",
        "zh": "猫",
        "ja": "猫",
        "fr": "chat",
        "es": "gato",
    },
}


# =====================================================================
# HELPER: FIND WORD POSITION
# =====================================================================

def find_word_position(tokenizer, template, word):
    """Find the token position of word in the FULL tokenized template."""
    full_tokens = tokenizer.encode(template, add_special_tokens=True)
    no_special_tokens = tokenizer.encode(template, add_special_tokens=False)
    n_prefix = len(full_tokens) - len(no_special_tokens)
    word_ids = tokenizer.encode(word, add_special_tokens=False)

    for i in range(len(no_special_tokens) - len(word_ids) + 1):
        if no_special_tokens[i:i+len(word_ids)] == word_ids:
            return i + n_prefix

    decoded = [tokenizer.decode([t]) for t in no_special_tokens]
    for i, d in enumerate(decoded):
        if word.lower() in d.lower() and i > 0:
            return i + n_prefix

    return 1 + n_prefix


# =====================================================================
# HELPER: GET HIDDEN STATES + LOGITS
# =====================================================================

def get_hidden_states_and_logits(model, tokenizer, device, template, word_pos_idx,
                                  n_layers_plus1, model_info):
    """Get hidden states at word position + final logits."""
    input_device = next(model.parameters()).device
    inputs = tokenizer(template, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(input_device)
    attention_mask = inputs["attention_mask"].to(input_device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True)

    hs = out.hidden_states
    logits = out.logits[0, -1].float().cpu().numpy()  # [vocab_size]

    result = {}
    for li in range(min(n_layers_plus1, len(hs))):
        h = hs[li][0].float().cpu().numpy()
        if word_pos_idx < h.shape[0]:
            result[li] = h[word_pos_idx]

    return result, logits


# =====================================================================
# HELPER: GET LOGITS FOR SPECIFIC WORDS
# =====================================================================

def get_logit_for_words(logits, tokenizer, words):
    """Get logit values for a list of words."""
    result = {}
    for word in words:
        tok_ids = tokenizer.encode(word, add_special_tokens=False)
        if len(tok_ids) == 1:
            result[word] = float(logits[tok_ids[0]])
        elif len(tok_ids) > 1:
            # Average logit for multi-token words
            result[word] = float(np.mean([logits[tid] for tid in tok_ids]))
    return result


# =====================================================================
# Exp 1: SUBSPACE ABLATION — THE CORE CAUSAL EXPERIMENT
# =====================================================================

def run_subspace_ablation(model, tokenizer, device, model_info):
    """
    ★★★ THE MOST CRITICAL EXPERIMENT OF PHASE 175 ★★★

    Procedure:
    1. Compute "fruit subspace" from PCA of fruit word hidden states at key layers
    2. For each test word (fruit + cross-category controls):
       a. Get normal logits
       b. Ablate fruit subspace from hidden states during forward pass
       c. Get ablated logits
       d. Compare: does ablation selectively impair fruit-related predictions?
    3. Control: ablate random subspace of same dimension

    KEY QUESTION: Is the "fruit subspace" causally necessary for fruit predictions?
    """
    n_layers = model_info.n_layers
    n_layers_plus1 = n_layers + 1
    d_model = model_info.d_model

    print("\n" + "="*70, flush=True)
    print("Exp 1: ★★★ SUBSPACE ABLATION — Causal Necessity of Category Subspace ★★★", flush=True)
    print("="*70, flush=True)

    # Step 1: Compute category subspaces at key layers
    # Use layers where clustering is still strong (from Phase 174: L0 is best)
    key_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]

    print("  Step 1: Collecting category hidden states...", flush=True)

    cat_hiddens = {cat: {li: [] for li in key_layers} for cat in CATEGORIES}

    for cat_name, words in CATEGORIES.items():
        for word in words:
            template = f"The {word} is"
            word_pos = find_word_position(tokenizer, template, word)
            input_device = next(model.parameters()).device

            inputs = tokenizer(template, return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                out = model(input_ids=inputs["input_ids"].to(input_device),
                           attention_mask=inputs["attention_mask"].to(input_device),
                           output_hidden_states=True)

            for li in key_layers:
                if li < len(out.hidden_states):
                    h = out.hidden_states[li][0].float().cpu().numpy()
                    if word_pos < h.shape[0]:
                        cat_hiddens[cat_name][li].append(h[word_pos])

        print(f"    {cat_name}: collected {sum(len(v) for v in cat_hiddens[cat_name].values())} vectors", flush=True)

    # Step 2: Compute PCA subspaces for each category at each key layer
    print("\n  Step 2: Computing category subspaces via PCA...", flush=True)

    cat_subspaces = {}  # {cat: {layer: projection_matrix}}
    cat_centroids = {}  # {cat: {layer: centroid}}

    for cat_name in CATEGORIES:
        cat_subspaces[cat_name] = {}
        cat_centroids[cat_name] = {}
        for li in key_layers:
            vecs = cat_hiddens[cat_name][li]
            if len(vecs) < 3:
                continue
            X = np.array(vecs)  # [n_words, d_model]
            centroid = np.mean(X, axis=0)
            cat_centroids[cat_name][li] = centroid

            # PCA: center the data
            X_centered = X - centroid
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

            # Top-k subspace: use top 10 components (from Phase 174: ~80% variance)
            n_components = min(10, len(Vt))
            subspace_basis = Vt[:n_components]  # [n_components, d_model]
            cat_subspaces[cat_name][li] = subspace_basis

            var_explained = np.sum(S[:n_components]**2) / max(np.sum(S**2), 1e-10)
            print(f"    {cat_name} L{li}: {n_components} components, "
                  f"var_explained={var_explained:.4f}", flush=True)

    # Step 3: Causal intervention — ablate subspace and measure logit change
    print("\n  Step 3: Causal intervention — ablating category subspaces...", flush=True)

    # Test words: fruits + controls
    test_words = {
        "fruits": ["apple", "banana", "orange", "grape", "mango"],
        "controls_for_fruit": ["car", "bus", "cat", "dog", "happy", "sad", "book", "chair"],
    }

    results = {}

    for ablation_cat in ["fruits", "vehicles", "animals", "emotions"]:
        print(f"\n  ★ Ablating {ablation_cat} subspace...", flush=True)

        ablation_results = {}

        for target_cat, target_words in test_words.items():
            for word in target_words:
                template = f"The {word} is"
                word_pos = find_word_position(tokenizer, template, word)

                # Get normal logits
                normal_hiddens, normal_logits = get_hidden_states_and_logits(
                    model, tokenizer, device, template, word_pos, n_layers_plus1, model_info)

                # Get verification word logits
                verify_words = CATEGORY_VERIFICATION.get(ablation_cat, [])
                control_words = CROSS_CATEGORY_CONTROLS.get(ablation_cat, [])

                normal_verify = get_logit_for_words(normal_logits, tokenizer, verify_words)
                normal_control = get_logit_for_words(normal_logits, tokenizer, control_words)

                # Ablate subspace at each key layer using hooks
                for li in key_layers:
                    if ablation_cat not in cat_subspaces or li not in cat_subspaces[ablation_cat]:
                        continue

                    subspace = cat_subspaces[ablation_cat][li]  # [n_comp, d_model]
                    n_comp = subspace.shape[0]

                    # Create projection matrix to remove subspace
                    # P = I - V^T V  where V is the subspace basis
                    # This removes the component in the subspace
                    subspace_t = torch.tensor(subspace, dtype=torch.float32)

                    # Use hook to ablate subspace
                    ablated_logits = _ablate_with_hook(
                        model, tokenizer, template, li, subspace_t)

                    # Measure logit changes
                    ablated_verify = get_logit_for_words(ablated_logits, tokenizer, verify_words)
                    ablated_control = get_logit_for_words(ablated_logits, tokenizer, control_words)

                    # Compute logit changes
                    verify_changes = {w: ablated_verify.get(w, 0) - normal_verify.get(w, 0)
                                     for w in verify_words if w in normal_verify and w in ablated_verify}
                    control_changes = {w: ablated_control.get(w, 0) - normal_control.get(w, 0)
                                      for w in control_words if w in normal_control and w in ablated_control}

                    avg_verify_change = float(np.mean(list(verify_changes.values()))) if verify_changes else 0
                    avg_control_change = float(np.mean(list(control_changes.values()))) if control_changes else 0

                    # Selectivity: verify change / (verify change + control change)
                    # If ablation selectively impairs category → verify change >> control change
                    selectivity = 0.0
                    if abs(avg_verify_change) > 1e-6:
                        selectivity = avg_verify_change / (abs(avg_verify_change) + abs(avg_control_change) + 1e-10)

                    result_key = f"{word}_L{li}"
                    ablation_results[result_key] = {
                        "word": word,
                        "target_cat": target_cat,
                        "ablation_cat": ablation_cat,
                        "layer": li,
                        "normal_verify_avg": round(float(np.mean(list(normal_verify.values()))) if normal_verify else 0, 4),
                        "ablated_verify_avg": round(float(np.mean(list(ablated_verify.values()))) if ablated_verify else 0, 4),
                        "normal_control_avg": round(float(np.mean(list(normal_control.values()))) if normal_control else 0, 4),
                        "ablated_control_avg": round(float(np.mean(list(ablated_control.values()))) if ablated_control else 0, 4),
                        "verify_change": round(avg_verify_change, 4),
                        "control_change": round(avg_control_change, 4),
                        "selectivity": round(float(selectivity), 4),
                    }

        results[ablation_cat] = ablation_results

    # Step 4: Random subspace control — ablate random subspace of same dimension
    print("\n  Step 4: Random subspace control...", flush=True)

    random_ablation_results = {}
    for li in key_layers[:3]:  # Just test a few layers
        # Generate random subspace of same dimension
        n_comp = 10
        random_subspace = torch.randn(n_comp, d_model)
        # Orthogonalize via QR
        random_subspace, _ = torch.linalg.qr(random_subspace.T)
        random_subspace = random_subspace.T[:n_comp]  # [n_comp, d_model]

        for word in ["apple", "car", "cat", "happy"]:
            template = f"The {word} is"
            normal_hiddens, normal_logits = get_hidden_states_and_logits(
                model, tokenizer, device, template,
                find_word_position(tokenizer, template, word),
                n_layers_plus1, model_info)

            ablated_logits = _ablate_with_hook(model, tokenizer, template, li, random_subspace)

            # Measure total logit change
            logit_change = float(np.mean(np.abs(ablated_logits - normal_logits)))
            max_logit_change = float(np.max(np.abs(ablated_logits - normal_logits)))

            random_ablation_results[f"{word}_L{li}"] = {
                "word": word,
                "layer": li,
                "avg_logit_change": round(logit_change, 6),
                "max_logit_change": round(max_logit_change, 6),
            }

    # Summarize
    print("\n  ★★★ SUBSPACE ABLATION SUMMARY ★★★", flush=True)
    for ablation_cat, ablation_data in results.items():
        print(f"\n  Ablating '{ablation_cat}' subspace:", flush=True)
        # Aggregate: average selectivity for same-category vs cross-category words
        same_cat_selectivities = []
        cross_cat_selectivities = []
        for key, data in ablation_data.items():
            if data["target_cat"] == ablation_cat or (data["target_cat"] == "controls_for_fruit" and ablation_cat != "fruits"):
                # This is a cross-category word
                cross_cat_selectivities.append(data["selectivity"])
            else:
                same_cat_selectivities.append(data["selectivity"])

        # More careful: for "fruits" ablation, fruit words are same-cat, controls are cross-cat
        for key, data in ablation_data.items():
            if data["word"] in CATEGORIES.get(ablation_cat, []):
                same_cat_selectivities.append(data["selectivity"])
            else:
                cross_cat_selectivities.append(data["selectivity"])

        avg_same = float(np.mean(same_cat_selectivities)) if same_cat_selectivities else 0
        avg_cross = float(np.mean(cross_cat_selectivities)) if cross_cat_selectivities else 0

        print(f"    Same-category selectivity: {avg_same:.4f}", flush=True)
        print(f"    Cross-category selectivity: {avg_cross:.4f}", flush=True)
        print(f"    Differential: {avg_same - avg_cross:.4f}", flush=True)

    return {
        "category_ablation": results,
        "random_ablation_control": random_ablation_results,
    }


def _ablate_with_hook(model, tokenizer, template, target_layer, subspace_t):
    """
    Ablate a subspace from the hidden state at target_layer during forward pass.

    Uses a forward hook to:
    1. Capture the output at target_layer
    2. Remove the subspace component
    3. Continue forward pass with modified hidden state

    Returns the final logits after ablation.
    """
    input_device = next(model.parameters()).device
    inputs = tokenizer(template, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(input_device)
    attention_mask = inputs["attention_mask"].to(input_device)

    layers = get_layers(model)

    # Create hook that removes subspace projection
    captured = {}
    modified = {}

    def make_ablation_hook(layer_idx, subspace):
        def hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0]  # [batch, seq, d_model]
            else:
                h = output

            # Project out the subspace: h_ablated = h - V^T V h
            # where V is the subspace basis
            subspace_dev = subspace.to(h.device).to(h.dtype)  # [n_comp, d_model]
            # V h^T → [n_comp, seq] (projections onto each component)
            proj = torch.matmul(subspace_dev, h.transpose(-1, -2))  # [n_comp, seq]
            # V^T proj → [d_model, seq] (reconstruction in subspace)
            recon = torch.matmul(subspace_dev.T, proj)  # [d_model, seq]
            # Remove subspace component
            h_ablated = h - recon.transpose(-1, -2)

            captured[f"L{layer_idx}"] = h.detach().float().cpu()

            if isinstance(output, tuple):
                return (h_ablated,) + output[1:]
            return h_ablated
        return hook

    # Register hook
    hooks = [layers[target_layer].register_forward_hook(
        make_ablation_hook(target_layer, subspace_t))]

    with torch.no_grad():
        try:
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            ablated_logits = out.logits[0, -1].float().cpu().numpy()
        except Exception as e:
            print(f"    [WARNING] Ablation forward failed at L{target_layer}: {e}", flush=True)
            ablated_logits = None

    for h in hooks:
        h.remove()

    if ablated_logits is None:
        # Fallback: return zeros
        vocab_size = len(tokenizer)
        return np.zeros(vocab_size)

    return ablated_logits


# =====================================================================
# Exp 2: NEURON-LEVEL CATEGORY SELECTIVITY
# =====================================================================

def run_neuron_selectivity(model, tokenizer, device, model_info):
    """
    Find MLP neurons that are selectively active for specific categories.

    For each key layer:
    1. Run forward pass for words from each category
    2. Record MLP intermediate activations (post-gate, post-activation)
    3. For each neuron, compute its selectivity index:
       selectivity = (activation_for_cat - mean_other_cats) / std
    4. Identify top selective neurons
    5. Ablate top selective neurons → measure category-specific impairment
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    intermediate_size = model_info.intermediate_size

    print("\n" + "="*70, flush=True)
    print("Exp 2: ★★★ NEURON-LEVEL CATEGORY SELECTIVITY ★★★", flush=True)
    print("="*70, flush=True)

    key_layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4]

    # Step 1: Collect MLP activations for each category
    print("  Collecting MLP activations...", flush=True)

    cat_mlp_activations = {cat: {li: [] for li in key_layers} for cat in CATEGORIES}
    layers = get_layers(model)

    for cat_name, words in CATEGORIES.items():
        # Sample 8 words per category to save time
        sample_words = words[:8]
        for word in sample_words:
            template = f"The {word} is"

            # Use hook to capture MLP output
            captured = {}

            def make_mlp_hook(layer_idx):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        captured[f"mlp_L{layer_idx}"] = output[0].detach().float().cpu()
                    else:
                        captured[f"mlp_L{layer_idx}"] = output.detach().float().cpu()
                return hook

            hooks = []
            for li in key_layers:
                if hasattr(layers[li], 'mlp'):
                    hooks.append(layers[li].mlp.register_forward_hook(make_mlp_hook(li)))

            input_device = next(model.parameters()).device
            inputs = tokenizer(template, return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                model(input_ids=inputs["input_ids"].to(input_device),
                      attention_mask=inputs["attention_mask"].to(input_device))

            for h in hooks:
                h.remove()

            # Store last-token MLP output
            for li in key_layers:
                key = f"mlp_L{li}"
                if key in captured:
                    mlp_out = captured[key][0, -1].numpy()  # [d_model]
                    cat_mlp_activations[cat_name][li].append(mlp_out)

        print(f"    {cat_name}: collected", flush=True)

    # Step 2: Compute neuron selectivity
    print("\n  Computing neuron selectivity...", flush=True)

    neuron_selectivity_results = {}

    for li in key_layers:
        # Collect category activation matrices
        cat_activations = {}
        for cat_name in CATEGORIES:
            acts = cat_mlp_activations[cat_name][li]
            if acts:
                cat_activations[cat_name] = np.array(acts)  # [n_words, d_model]

        if len(cat_activations) < 4:
            continue

        # For each dimension (neuron in output space), compute selectivity
        # Selectivity = mean(cat_activation) - mean(other_cat_activations)
        all_selectivities = {}

        for target_cat in CATEGORIES:
            target_acts = cat_activations.get(target_cat)
            if target_acts is None:
                continue

            other_acts = np.concatenate([cat_activations[cat] for cat in cat_activations
                                         if cat != target_cat], axis=0)

            # Mean activation per dimension
            target_mean = np.mean(target_acts, axis=0)  # [d_model]
            other_mean = np.mean(other_acts, axis=0)  # [d_model]

            # Selectivity per dimension
            selectivity = target_mean - other_mean  # [d_model]

            # Top selective neurons (by absolute selectivity)
            top_indices = np.argsort(np.abs(selectivity))[::-1][:20]
            top_neurons = [(int(idx), float(selectivity[idx])) for idx in top_indices]

            # Also compute: for top neurons, what fraction of target words activate them?
            threshold = 0.0
            for idx, sel in top_neurons[:10]:
                target_positive = np.mean(target_acts[:, idx] > threshold)
                other_positive = np.mean(other_acts[:, idx] > threshold)

            all_selectivities[target_cat] = {
                "top_neurons": [(int(i), round(float(s), 4)) for i, s in top_neurons],
                "max_selectivity": round(float(np.max(np.abs(selectivity))), 4),
                "mean_abs_selectivity": round(float(np.mean(np.abs(selectivity))), 4),
            }

        neuron_selectivity_results[f"L{li}"] = all_selectivities

        for cat, sel in all_selectivities.items():
            print(f"    L{li} {cat}: max_sel={sel['max_selectivity']:.4f}, "
                  f"mean_sel={sel['mean_abs_selectivity']:.4f}", flush=True)

    # Step 3: Ablate top selective neurons
    print("\n  Ablating top selective neurons...", flush=True)

    ablation_results = {}

    for li in key_layers:
        for target_cat in CATEGORIES:
            sel_data = neuron_selectivity_results.get(f"L{li}", {}).get(target_cat)
            if not sel_data:
                continue

            top_neurons = sel_data["top_neurons"][:10]
            neuron_indices = [idx for idx, _ in top_neurons]

            # Test: does ablating these neurons selectively impair target category?
            for test_word in CATEGORIES[target_cat][:3]:
                template = f"The {test_word} is"

                # Normal forward pass
                input_device = next(model.parameters()).device
                inputs = tokenizer(template, return_tensors="pt", truncation=True, max_length=64)
                with torch.no_grad():
                    normal_out = model(input_ids=inputs["input_ids"].to(input_device),
                                      attention_mask=inputs["attention_mask"].to(input_device))
                normal_logits = normal_out.logits[0, -1].float().cpu().numpy()

                # Verify words
                verify_words = CATEGORY_VERIFICATION[target_cat]
                normal_verify = get_logit_for_words(normal_logits, tokenizer, verify_words)

                # Ablate top neurons with hook
                def make_neuron_ablation_hook(layer_idx, neuron_indices_list):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            h = output[0].clone()
                        else:
                            h = output.clone()
                        # Zero out the selected neurons at last token position
                        for ni in neuron_indices_list:
                            h[0, -1, ni] = 0.0
                        if isinstance(output, tuple):
                            return (h,) + output[1:]
                        return h
                    return hook

                hooks = [layers[li].register_forward_hook(
                    make_neuron_ablation_hook(li, neuron_indices))]

                with torch.no_grad():
                    ablated_out = model(input_ids=inputs["input_ids"].to(input_device),
                                       attention_mask=inputs["attention_mask"].to(input_device))

                for h in hooks:
                    h.remove()

                ablated_logits = ablated_out.logits[0, -1].float().cpu().numpy()
                ablated_verify = get_logit_for_words(ablated_logits, tokenizer, verify_words)

                verify_change = np.mean([ablated_verify.get(w, 0) - normal_verify.get(w, 0)
                                        for w in verify_words if w in normal_verify and w in ablated_verify])

                # Also test on a cross-category word
                cross_cat = "vehicles" if target_cat != "vehicles" else "fruits"
                cross_word = CATEGORIES[cross_cat][0]
                cross_template = f"The {cross_word} is"
                cross_inputs = tokenizer(cross_template, return_tensors="pt",
                                        truncation=True, max_length=64)
                with torch.no_grad():
                    cross_normal = model(input_ids=cross_inputs["input_ids"].to(input_device),
                                        attention_mask=cross_inputs["attention_mask"].to(input_device))
                cross_normal_logits = cross_normal.logits[0, -1].float().cpu().numpy()
                cross_normal_verify = get_logit_for_words(cross_normal_logits, tokenizer, verify_words)

                hooks = [layers[li].register_forward_hook(
                    make_neuron_ablation_hook(li, neuron_indices))]
                with torch.no_grad():
                    cross_ablated = model(input_ids=cross_inputs["input_ids"].to(input_device),
                                         attention_mask=cross_inputs["attention_mask"].to(input_device))
                for h in hooks:
                    h.remove()

                cross_ablated_logits = cross_ablated.logits[0, -1].float().cpu().numpy()
                cross_ablated_verify = get_logit_for_words(cross_ablated_logits, tokenizer, verify_words)

                cross_change = np.mean([cross_ablated_verify.get(w, 0) - cross_normal_verify.get(w, 0)
                                       for w in verify_words if w in cross_normal_verify and w in cross_ablated_verify])

                result_key = f"{target_cat}_{test_word}_L{li}"
                ablation_results[result_key] = {
                    "category": target_cat,
                    "word": test_word,
                    "layer": li,
                    "same_cat_verify_change": round(float(verify_change), 4),
                    "cross_cat_verify_change": round(float(cross_change), 4),
                    "selectivity_ratio": round(float(verify_change / max(abs(cross_change), 1e-6)), 2),
                }

    # Summarize neuron ablation
    print("\n  ★★★ NEURON ABLATION SUMMARY ★★★", flush=True)
    for cat in CATEGORIES:
        cat_results = {k: v for k, v in ablation_results.items() if v["category"] == cat}
        if cat_results:
            avg_same = np.mean([v["same_cat_verify_change"] for v in cat_results.values()])
            avg_cross = np.mean([v["cross_cat_verify_change"] for v in cat_results.values()])
            print(f"    {cat}: same_cat_change={avg_same:.4f}, cross_cat_change={avg_cross:.4f}, "
                  f"ratio={avg_same/max(abs(avg_cross), 1e-6):.2f}", flush=True)

    return {
        "neuron_selectivity": neuron_selectivity_results,
        "neuron_ablation": ablation_results,
    }


# =====================================================================
# Exp 3: CONTEXT-DEPENDENT CONCEPT STABILITY
# =====================================================================

def run_context_stability(model, tokenizer, device, model_info):
    """
    Test whether concept representations are stable across different contexts.

    Key questions:
    1. Does "apple" (fruit) share structure with "Apple" (company)?
    2. Is the "fruitness" subspace stable across different sentences?
    3. Does context change which dimensions are active?
    """
    n_layers = model_info.n_layers
    n_layers_plus1 = n_layers + 1

    print("\n" + "="*70, flush=True)
    print("Exp 3: ★★★ CONTEXT-DEPENDENT CONCEPT STABILITY ★★★", flush=True)
    print("="*70, flush=True)

    key_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]

    results = {}

    for base_word, contexts in CONTEXT_VARIATIONS.items():
        print(f"\n  Word: '{base_word}' ({len(contexts)} contexts)", flush=True)
        word_hiddens = []  # Hidden states across contexts

        for ctx in contexts:
            # Find the word position in this context
            word_pos = find_word_position(tokenizer, ctx, base_word)

            input_device = next(model.parameters()).device
            inputs = tokenizer(ctx, return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                out = model(input_ids=inputs["input_ids"].to(input_device),
                           attention_mask=inputs["attention_mask"].to(input_device),
                           output_hidden_states=True)

            ctx_hiddens = {}
            for li in key_layers:
                if li < len(out.hidden_states):
                    h = out.hidden_states[li][0].float().cpu().numpy()
                    if word_pos < h.shape[0]:
                        ctx_hiddens[li] = h[word_pos]

            word_hiddens.append({"context": ctx, "hiddens": ctx_hiddens})

        # Compute pairwise similarity across contexts at each layer
        stability_results = {}
        for li in key_layers:
            vecs = [wh["hiddens"][li] for wh in word_hiddens if li in wh["hiddens"]]
            contexts_list = [wh["context"] for wh in word_hiddens if li in wh["hiddens"]]

            if len(vecs) < 2:
                continue

            # Pairwise cosine similarity
            norms = [np.linalg.norm(v) for v in vecs]
            normed = [v / max(n, 1e-10) for v, n in zip(vecs, norms)]

            pairwise_sims = []
            for i in range(len(normed)):
                for j in range(i+1, len(normed)):
                    sim = float(np.dot(normed[i], normed[j]))
                    pairwise_sims.append(sim)

            # Also: compare "fruit context" vs "non-fruit context"
            # Identify fruit vs non-fruit contexts
            fruit_context_idx = []
            non_fruit_context_idx = []
            for i, ctx in enumerate(contexts_list):
                if "ate" in ctx.lower() or "tree" in ctx.lower() or "fell" in ctx.lower() or "bought" in ctx.lower():
                    fruit_context_idx.append(i)
                elif "released" in ctx.lower() or "Apple" in ctx[:10]:
                    non_fruit_context_idx.append(i)
                else:
                    fruit_context_idx.append(i)  # Default to fruit context

            # Average similarity within fruit contexts vs across contexts
            fruit_sims = []
            cross_sims = []
            for i in range(len(normed)):
                for j in range(i+1, len(normed)):
                    sim = float(np.dot(normed[i], normed[j]))
                    if i in fruit_context_idx and j in fruit_context_idx:
                        fruit_sims.append(sim)
                    elif (i in fruit_context_idx and j in non_fruit_context_idx) or \
                         (i in non_fruit_context_idx and j in fruit_context_idx):
                        cross_sims.append(sim)

            stability_results[f"L{li}"] = {
                "avg_pairwise_sim": round(float(np.mean(pairwise_sims)), 4),
                "min_pairwise_sim": round(float(np.min(pairwise_sims)), 4),
                "max_pairwise_sim": round(float(np.max(pairwise_sims)), 4),
                "avg_fruit_context_sim": round(float(np.mean(fruit_sims)), 4) if fruit_sims else 0,
                "avg_cross_context_sim": round(float(np.mean(cross_sims)), 4) if cross_sims else 0,
                "n_fruit_contexts": len(fruit_context_idx),
                "n_nonfruit_contexts": len(non_fruit_context_idx),
            }

        results[base_word] = stability_results

        # Print summary
        for li_key, data in stability_results.items():
            print(f"    {li_key}: avg_sim={data['avg_pairwise_sim']:.4f}, "
                  f"fruit_sim={data['avg_fruit_context_sim']:.4f}, "
                  f"cross_sim={data['avg_cross_context_sim']:.4f}", flush=True)

    return results


# =====================================================================
# Exp 4: CROSS-LINGUAL CONCEPT INVARIANT
# =====================================================================

def run_cross_lingual(model, tokenizer, device, model_info):
    """
    Test whether concept representations share structure across languages.

    Use bilingual contexts:
    - "apple" in English context
    - "苹果" in Chinese context (if tokenizer supports it)
    - Compare: do they share the same subspace structure?
    """
    n_layers = model_info.n_layers
    n_layers_plus1 = n_layers + 1

    print("\n" + "="*70, flush=True)
    print("Exp 4: ★★★ CROSS-LINGUAL CONCEPT INVARIANT ★★★", flush=True)
    print("="*70, flush=True)

    key_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]

    results = {}

    # For each concept, get hidden states in different language contexts
    for concept_key, translations in CROSS_LINGUAL.items():
        print(f"\n  Concept: '{concept_key}' ({len(translations)} languages)", flush=True)

        lang_hiddens = {}

        for lang, word in translations.items():
            # English context template (all languages use English sentence structure
            # with the translated word inserted)
            templates = [
                f"The {word} is",
                f"I like the {word}",
                f"A {word} can be",
            ]

            word_hiddens = []
            for template in templates:
                # Check if the word can be tokenized
                try:
                    word_ids = tokenizer.encode(word, add_special_tokens=False)
                    if len(word_ids) == 0:
                        continue
                except:
                    continue

                word_pos = find_word_position(tokenizer, template, word)
                input_device = next(model.parameters()).device
                inputs = tokenizer(template, return_tensors="pt", truncation=True, max_length=64)

                with torch.no_grad():
                    out = model(input_ids=inputs["input_ids"].to(input_device),
                               attention_mask=inputs["attention_mask"].to(input_device),
                               output_hidden_states=True)

                for li in key_layers:
                    if li < len(out.hidden_states):
                        h = out.hidden_states[li][0].float().cpu().numpy()
                        if word_pos < h.shape[0]:
                            word_hiddens.append({"layer": li, "vector": h[word_pos]})

            if word_hiddens:
                lang_hiddens[lang] = word_hiddens

        # Compare cross-lingual similarity at each layer
        lang_names = list(lang_hiddens.keys())
        if len(lang_names) < 2:
            print(f"    Skipping: only {len(lang_names)} languages tokenized", flush=True)
            continue

        layer_comparisons = {}
        for li in key_layers:
            # Get average vector for each language at this layer
            lang_vecs = {}
            for lang in lang_names:
                vecs = [wh["vector"] for wh in lang_hiddens[lang] if wh["layer"] == li]
                if vecs:
                    lang_vecs[lang] = np.mean(vecs, axis=0)

            if len(lang_vecs) < 2:
                continue

            # Pairwise cosine similarity between languages
            lang_list = list(lang_vecs.keys())
            pairwise = {}
            for i in range(len(lang_list)):
                for j in range(i+1, len(lang_list)):
                    la, lb = lang_list[i], lang_list[j]
                    va, vb = lang_vecs[la], lang_vecs[lb]
                    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
                    if na > 1e-10 and nb > 1e-10:
                        sim = float(np.dot(va, vb) / (na * nb))
                    else:
                        sim = 0.0
                    pairwise[f"{la}-{lb}"] = round(sim, 4)

            # Compare with English as anchor
            en_sim = {}
            if "en" in lang_vecs:
                en_vec = lang_vecs["en"]
                for lang in lang_list:
                    if lang == "en":
                        continue
                    lv = lang_vecs[lang]
                    na, nb = np.linalg.norm(en_vec), np.linalg.norm(lv)
                    if na > 1e-10 and nb > 1e-10:
                        sim = float(np.dot(en_vec, lv) / (na * nb))
                    else:
                        sim = 0.0
                    en_sim[lang] = round(sim, 4)

            layer_comparisons[f"L{li}"] = {
                "pairwise": pairwise,
                "en_anchor_sim": en_sim,
            }

        results[concept_key] = layer_comparisons

        # Print summary
        for li_key, data in layer_comparisons.items():
            en_sims = data.get("en_anchor_sim", {})
            if en_sims:
                print(f"    {li_key}: EN→{en_sims}", flush=True)

    return results


# =====================================================================
# MAIN
# =====================================================================

def run_phase175(model_name):
    print(f"\n{'='*70}", flush=True)
    print(f"Phase 175: Causal Intervention — {model_name}", flush=True)
    print(f"{'='*70}", flush=True)

    t_start = time.time()

    # Load model
    model, tokenizer, device = load_model_bf16(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model

    print(f"  Model: {model_info.model_class}, L={n_layers}, d={d_model}", flush=True)

    # =====================================================================
    # Run all experiments
    # =====================================================================

    # Exp 1: Subspace Ablation (most critical)
    exp1_results = run_subspace_ablation(model, tokenizer, device, model_info)

    # Exp 2: Neuron Selectivity
    exp2_results = run_neuron_selectivity(model, tokenizer, device, model_info)

    # Exp 3: Context Stability
    exp3_results = run_context_stability(model, tokenizer, device, model_info)

    # Exp 4: Cross-lingual Invariant
    exp4_results = run_cross_lingual(model, tokenizer, device, model_info)

    # =====================================================================
    # SAVE RESULTS
    # =====================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output = {
        "model": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "timestamp": timestamp,
        "exp1_subspace_ablation": exp1_results,
        "exp2_neuron_selectivity": exp2_results,
        "exp3_context_stability": exp3_results,
        "exp4_cross_lingual": exp4_results,
    }

    out_path = f"tests/glm5_temp/phase175_{model_name}_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {out_path}", flush=True)

    # Release model
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    print(f"\nPhase 175 ({model_name}) completed in {elapsed:.1f}s", flush=True)

    return output


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python phase175_causal_intervention.py <model_name>")
        print("  model_name: qwen3, glm4, deepseek7b")
        sys.exit(1)

    model_name = sys.argv[1]
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    run_phase175(model_name)
