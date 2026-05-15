"""
Phase 174: Minimum Concept Universe (最小概念宇宙)
====================================================

★★★ PARADIGM: From "constraint dynamics" → "concept structure dynamics"! ★★★

User's core insight: The "hydrogen atom" of language is the concept hierarchy:
  apple → fruit → food → object

This single structure contains:
- Concept hierarchy (层级)
- Abstraction (抽象化)
- Feature reuse (特征复用)
- Generalization (泛化)
- Composition (组合编码)
- Semantic distance (语义距离)

★★★ FIVE KEY EXPERIMENTS ★★★

Exp 1: ★★★ Category Clustering Dynamics (60 words, 4 categories)
  - Fruits, Vehicles, Animals, Emotions — 15 words each
  - Template: "The [word] is"
  - Measure: intra-category similarity vs inter-category similarity at each layer
  - KEY: At which layer do categories separate? Is there a universal "clustering phase transition"?

Exp 2: ★★★ Hierarchical Abstraction (fruit ⊂ food ⊂ object)
  - Concrete fruits: apple, banana, orange
  - Mid-level: bread, meat, rice (food, not fruit)
  - Abstract: fruit, food, object
  - KEY: Does a hierarchical structure emerge? Is "fruit" between "apple" and "food" in representation space?

Exp 3: ★★★ Compositional Decomposition
  - fruit_base = PCA shared subspace of all fruits
  - For each fruit f: f_residual = h(f) - projection onto fruit_base
  - KEY: Is ||f_residual|| << ||h(f)||? → Compositional encoding!
  - Is the "fruit-ness" direction stable across layers?

Exp 4: ★★★ Cross-Category Differential Structure
  - same_cat_diff: apple - banana (both fruits)
  - cross_cat_diff: apple - car (different categories)
  - KEY: Is ||same_cat_diff|| < ||cross_cat_diff||?
  - Is same_cat_diff direction more stable across pairs?

Exp 5: ★★★ Novel Concept Combinations
  - "The blue apple" vs "The apple" — how does adjective modify noun?
  - "The metal fruit" — contradictory combination
  - "The flying banana" — novel combination
  - KEY: Is concept combination additive? Or nonlinear?

Usage: python tests/glm5/phase174_concept_universe.py <model_name>
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

# Force unbuffered output
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
# CONCEPT CATEGORIES — 60 words, 4 categories, 15 each
# =====================================================================

FRUITS = [
    "apple", "banana", "orange", "grape", "mango",
    "pear", "peach", "cherry", "lemon", "plum",
    "melon", "kiwi", "coconut", "papaya", "lime",
]

VEHICLES = [
    "car", "truck", "bus", "train", "plane",
    "boat", "bike", "van", "taxi", "tram",
    "helicopter", "rocket", "yacht", "canoe", "submarine",
]

ANIMALS = [
    "cat", "dog", "bird", "fish", "horse",
    "cow", "pig", "sheep", "goat", "lion",
    "tiger", "bear", "elephant", "monkey", "rabbit",
]

EMOTIONS = [
    "happy", "sad", "angry", "afraid", "joyful",
    "proud", "ashamed", "guilty", "envious", "calm",
    "anxious", "bored", "lonely", "grateful", "hopeful",
]

# Hierarchical concepts for Exp 2
CONCRETE_FRUITS = ["apple", "banana", "orange", "grape", "mango"]
FOOD_NOT_FRUIT = ["bread", "meat", "rice", "cheese", "pasta"]
ABSTRACT_FOOD = ["fruit", "food", "object", "thing", "item"]

# Novel combinations for Exp 5
NOVEL_COMBINATIONS = [
    # (modified_noun_phrase, base_noun, adjective)
    ("The blue apple", "The apple", "blue"),
    ("The red banana", "The banana", "red"),
    ("The metal fruit", "The fruit", "metal"),
    ("The flying banana", "The banana", "flying"),
    ("The tiny elephant", "The elephant", "tiny"),
    ("The cold fire", "The fire", "cold"),
    ("The sweet lemon", "The lemon", "sweet"),
    ("The wooden car", "The car", "wooden"),
]


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def get_word_hidden_states(model, tokenizer, device, template, word_pos_idx,
                           n_layers_plus1, model_info):
    """
    Get hidden states at word position for all layers.

    Args:
        template: e.g. "The apple is" — word should be at word_pos_idx
        word_pos_idx: 0-based index of the target word in the FULL token sequence
                      (including any BOS/special tokens added by tokenizer)

    Returns:
        dict: {layer_idx: hidden_state_vector [d_model]}
    """
    input_device = next(model.parameters()).device
    inputs = tokenizer(template, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(input_device)
    attention_mask = inputs["attention_mask"].to(input_device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True)

    hs = out.hidden_states  # tuple of (1, seq_len, d_model), len = n_layers+1

    result = {}
    for li in range(min(n_layers_plus1, len(hs))):
        h = hs[li][0].float().cpu().numpy()  # [seq_len, d_model]
        if word_pos_idx < h.shape[0]:
            result[li] = h[word_pos_idx]  # [d_model]

    return result


def find_word_position(tokenizer, template, word):
    """
    Find the token position of word in the FULL tokenized template
    (including any BOS/special tokens added by the tokenizer).

    Key insight: tokenizer may add BOS tokens at the beginning, and
    the word may be tokenized differently in context (e.g., " apple" with leading space).
    So we search for the contextual token, not the isolated word token.
    """
    # Full token sequence (with special tokens)
    full_tokens = tokenizer.encode(template, add_special_tokens=True)
    no_special_tokens = tokenizer.encode(template, add_special_tokens=False)

    # The number of special tokens prepended
    n_prefix = len(full_tokens) - len(no_special_tokens)

    # Find the word in the no-special-tokens sequence
    # The word appears after "The " in "The [word] is"
    # In the tokenizer, it will be "▁word" or " word" (with leading space)
    # We search for the word tokens in the no_special sequence
    word_ids = tokenizer.encode(word, add_special_tokens=False)

    # Try direct match in no_special sequence
    for i in range(len(no_special_tokens) - len(word_ids) + 1):
        if no_special_tokens[i:i+len(word_ids)] == word_ids:
            return i + n_prefix

    # If word_ids don't match directly, try with leading space
    # Because "The apple" tokenizes as [The, " apple"] not [The, "apple"]
    # So we need to search for the token that starts with the word
    decoded = [tokenizer.decode([t]) for t in no_special_tokens]
    for i, d in enumerate(decoded):
        if word.lower() in d.lower() and i > 0:  # Skip first token ("The")
            return i + n_prefix

    # Fallback: position after "The" with BOS offset
    return 1 + n_prefix


def compute_category_stats(all_hiddens_by_category, sample_layers):
    """
    Compute intra-category and inter-category similarities.

    Args:
        all_hiddens_by_category: {category: {word: {layer: vector}}}
        sample_layers: list of layer indices to analyze

    Returns:
        dict with category statistics
    """
    results = {}

    for li in sample_layers:
        layer_data = {}
        categories = list(all_hiddens_by_category.keys())

        # Collect vectors for this layer
        cat_vectors = {}
        for cat in categories:
            vecs = []
            for word, layers_dict in all_hiddens_by_category[cat].items():
                if li in layers_dict:
                    vecs.append(layers_dict[li])
            if vecs:
                cat_vectors[cat] = np.array(vecs)  # [n_words, d_model]

        if len(cat_vectors) < 2:
            continue

        # Intra-category similarity: avg cosine similarity within each category
        intra_sims = {}
        for cat, vecs in cat_vectors.items():
            if len(vecs) < 2:
                continue
            # Normalize
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            vecs_norm = vecs / norms

            # Pairwise cosine similarity
            sim_matrix = vecs_norm @ vecs_norm.T
            n = len(vecs)
            # Average off-diagonal
            total = 0
            count = 0
            for i in range(n):
                for j in range(i+1, n):
                    total += sim_matrix[i, j]
                    count += 1
            intra_sims[cat] = total / max(count, 1)

        # Inter-category similarity: avg cosine similarity between categories
        inter_sims = {}
        cat_names = list(cat_vectors.keys())
        for i in range(len(cat_names)):
            for j in range(i+1, len(cat_names)):
                cat_a, cat_b = cat_names[i], cat_names[j]
                vecs_a = cat_vectors[cat_a]
                vecs_b = cat_vectors[cat_b]

                norms_a = np.linalg.norm(vecs_a, axis=1, keepdims=True)
                norms_a = np.maximum(norms_a, 1e-10)
                norms_b = np.linalg.norm(vecs_b, axis=1, keepdims=True)
                norms_b = np.maximum(norms_b, 1e-10)

                vecs_a_norm = vecs_a / norms_a
                vecs_b_norm = vecs_b / norms_b

                cross_sim = vecs_a_norm @ vecs_b_norm.T
                inter_sims[f"{cat_a}-{cat_b}"] = float(np.mean(cross_sim))

        avg_intra = np.mean(list(intra_sims.values())) if intra_sims else 0
        avg_inter = np.mean(list(inter_sims.values())) if inter_sims else 0
        separation_ratio = avg_intra / max(avg_inter, 1e-10)

        # Category centroid distances
        centroids = {}
        for cat, vecs in cat_vectors.items():
            centroids[cat] = np.mean(vecs, axis=0)

        # For each word, compute distance to own centroid vs other centroids
        correct_cluster = 0
        total_words = 0
        for cat, vecs in cat_vectors.items():
            for v in vecs:
                dist_own = np.linalg.norm(v - centroids[cat])
                min_dist_other = min(np.linalg.norm(v - centroids[oc])
                                    for oc in centroids if oc != cat)
                if dist_own < min_dist_other:
                    correct_cluster += 1
                total_words += 1
        cluster_accuracy = correct_cluster / max(total_words, 1)

        layer_data = {
            "avg_intra_sim": round(float(avg_intra), 4),
            "avg_inter_sim": round(float(avg_inter), 4),
            "separation_ratio": round(float(separation_ratio), 4),
            "cluster_accuracy": round(float(cluster_accuracy), 4),
            "intra_by_cat": {k: round(float(v), 4) for k, v in intra_sims.items()},
            "inter_by_pair": {k: round(float(v), 4) for k, v in inter_sims.items()},
        }
        results[f"L{li}"] = layer_data

    return results


def compute_compositional_decomposition(all_hiddens_by_category, category_name,
                                         sample_layers, n_components=10):
    """
    Compute PCA-based decomposition: word = category_base + individual_offset.

    For each layer:
    1. Stack all word vectors in the category
    2. PCA to find shared subspace
    3. Measure: how much variance is captured by top-k components?
    4. Measure: how much of each word is in the shared subspace?

    Returns:
        dict with decomposition statistics per layer
    """
    cat_data = all_hiddens_by_category.get(category_name, {})
    if len(cat_data) < 3:
        return {}

    results = {}
    for li in sample_layers:
        vecs = []
        words = []
        for word, layers_dict in cat_data.items():
            if li in layers_dict:
                vecs.append(layers_dict[li])
                words.append(word)

        if len(vecs) < 3:
            continue

        X = np.array(vecs)  # [n_words, d_model]
        n = X.shape[0]

        # Compute mean (category centroid)
        centroid = np.mean(X, axis=0)

        # Compute PCA via SVD of centered data
        X_centered = X - centroid
        # SVD of centered data
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        # Vt[0:k] are the top-k principal directions in d_model space
        # S[0:k] are the singular values

        # Variance explained by top-k components
        total_var = np.sum(S ** 2)
        var_explained = {}
        for k in [1, 2, 3, 5, min(10, n-1)]:
            if k <= len(S):
                var_explained[f"top{k}"] = round(float(np.sum(S[:k] ** 2) / max(total_var, 1e-20)), 4)

        # For each word, how much is in the shared subspace?
        # Project each word onto the top-k principal components
        proj_ratios = {}
        for k in [1, 2, 3, 5]:
            if k > len(Vt):
                continue
            basis = Vt[:k].T  # [d_model, k]
            projections = []
            for i, word in enumerate(words):
                v = X_centered[i]
                proj = basis.T @ v  # [k]
                proj_energy = np.sum(proj ** 2)
                total_energy = np.sum(v ** 2)
                ratio = proj_energy / max(total_energy, 1e-20)
                projections.append(ratio)
            proj_ratios[f"top{k}_avg_ratio"] = round(float(np.mean(projections)), 4)

        # Distance from each word to centroid
        dists_to_centroid = [float(np.linalg.norm(X[i] - centroid)) for i in range(n)]
        avg_dist = np.mean(dists_to_centroid)
        avg_norm = np.mean([float(np.linalg.norm(X[i])) for i in range(n)])
        dist_norm_ratio = avg_dist / max(avg_norm, 1e-10)

        results[f"L{li}"] = {
            "n_words": n,
            "var_explained": var_explained,
            "proj_ratios": proj_ratios,
            "avg_dist_to_centroid": round(float(avg_dist), 4),
            "avg_norm": round(float(avg_norm), 4),
            "dist_norm_ratio": round(float(dist_norm_ratio), 4),
            "singular_values_top5": [round(float(s), 2) for s in S[:5]],
        }

    return results


def compute_hierarchical_structure(all_hiddens_by_concept, sample_layers):
    """
    Compute hierarchical structure: fruit ⊂ food ⊂ object

    For each layer:
    1. Compute centroids of each concept group
    2. Measure: is the "fruit" centroid between "apple" and "food"?
    3. Measure: is the "food" centroid between "fruit" and "object"?
    4. Compute: triangle inequality violations → evidence of hierarchy
    """
    results = {}

    for li in sample_layers:
        # Collect vectors
        concrete_fruit_vecs = []
        food_not_fruit_vecs = []
        abstract_food_vecs = []

        for word, layers_dict in all_hiddens_by_concept.get("concrete_fruits", {}).items():
            if li in layers_dict:
                concrete_fruit_vecs.append(layers_dict[li])

        for word, layers_dict in all_hiddens_by_concept.get("food_not_fruit", {}).items():
            if li in layers_dict:
                food_not_fruit_vecs.append(layers_dict[li])

        for word, layers_dict in all_hiddens_by_concept.get("abstract_food", {}).items():
            if li in layers_dict:
                abstract_food_vecs.append(layers_dict[li])

        if len(concrete_fruit_vecs) < 2 or len(abstract_food_vecs) < 1:
            continue

        # Centroids
        c_fruit = np.mean(concrete_fruit_vecs, axis=0) if concrete_fruit_vecs else None
        c_food_nf = np.mean(food_not_fruit_vecs, axis=0) if food_not_fruit_vecs else None

        # Abstract concept vectors
        abstract_vecs = {}
        for word, layers_dict in all_hiddens_by_concept.get("abstract_food", {}).items():
            if li in layers_dict:
                abstract_vecs[word] = layers_dict[li]

        layer_data = {}

        # Distance from each concrete fruit to "fruit" abstract concept
        if "fruit" in abstract_vecs and c_fruit is not None:
            fruit_abstract = abstract_vecs["fruit"]
            dists_to_fruit = [float(np.linalg.norm(v - fruit_abstract))
                             for v in concrete_fruit_vecs]
            dists_from_centroid = [float(np.linalg.norm(v - c_fruit))
                                   for v in concrete_fruit_vecs]
            layer_data["avg_dist_to_fruit_abstract"] = round(float(np.mean(dists_to_fruit)), 4)
            layer_data["avg_dist_to_fruit_centroid"] = round(float(np.mean(dists_from_centroid)), 4)

            # Is "fruit" abstract closer to concrete fruits than "object" abstract?
            if "object" in abstract_vecs:
                object_abstract = abstract_vecs["object"]
                dist_fruit_to_concrete = float(np.linalg.norm(fruit_abstract - c_fruit))
                dist_object_to_concrete = float(np.linalg.norm(object_abstract - c_fruit))
                layer_data["dist_fruit_abstract_to_concrete_centroid"] = round(dist_fruit_to_concrete, 4)
                layer_data["dist_object_abstract_to_concrete_centroid"] = round(dist_object_to_concrete, 4)
                layer_data["fruit_closer_than_object"] = dist_fruit_to_concrete < dist_object_to_concrete

            # Is "food" abstract between "fruit" and "object"?
            if "food" in abstract_vecs and "object" in abstract_vecs:
                food_abstract = abstract_vecs["food"]
                object_abstract = abstract_vecs["object"]
                d_fruit_food = float(np.linalg.norm(fruit_abstract - food_abstract))
                d_food_object = float(np.linalg.norm(food_abstract - object_abstract))
                d_fruit_object = float(np.linalg.norm(fruit_abstract - object_abstract))
                # Triangle inequality: d_fruit_object ≤ d_fruit_food + d_food_object
                # If food is "between" fruit and object, this should be nearly equal
                triangle_ratio = d_fruit_object / max(d_fruit_food + d_food_object, 1e-10)
                layer_data["d_fruit_food"] = round(d_fruit_food, 4)
                layer_data["d_food_object"] = round(d_food_object, 4)
                layer_data["d_fruit_object"] = round(d_fruit_object, 4)
                layer_data["triangle_ratio"] = round(float(triangle_ratio), 4)
                # ratio ≈ 1 means food is "on the path" from fruit to object → hierarchical!

        # Inter-group similarity
        if c_fruit is not None and c_food_nf is not None:
            # Cosine similarity between fruit centroid and food-not-fruit centroid
            cos_cc = float(np.dot(c_fruit, c_food_nf) /
                          max(np.linalg.norm(c_fruit) * np.linalg.norm(c_food_nf), 1e-10))
            layer_data["cos_fruit_food_centroids"] = round(cos_cc, 4)

        if layer_data:
            results[f"L{li}"] = layer_data

    return results


def compute_differential_structure(all_hiddens_by_category, sample_layers):
    """
    Compute differential structure:
    - same_cat_diff: ||h(apple) - h(banana)|| (both fruits)
    - cross_cat_diff: ||h(apple) - h(car)|| (different categories)
    - Direction stability of same-cat diffs
    """
    results = {}

    categories = list(all_hiddens_by_category.keys())
    if len(categories) < 2:
        return results

    for li in sample_layers:
        layer_data = {}

        # Collect vectors for this layer
        cat_vecs = {}
        for cat in categories:
            vecs = {}
            for word, layers_dict in all_hiddens_by_category[cat].items():
                if li in layers_dict:
                    vecs[word] = layers_dict[li]
            if vecs:
                cat_vecs[cat] = vecs

        if len(cat_vecs) < 2:
            continue

        # Same-category diffs
        same_cat_diffs = []
        for cat, vecs in cat_vecs.items():
            words = list(vecs.keys())
            for i in range(min(5, len(words))):
                for j in range(i+1, min(5, len(words))):
                    diff = vecs[words[i]] - vecs[words[j]]
                    same_cat_diffs.append(diff)

        # Cross-category diffs (take first 5 words from each category)
        cross_cat_diffs = []
        cat_list = list(cat_vecs.keys())
        for ci in range(min(3, len(cat_list))):
            for cj in range(ci+1, min(3, len(cat_list))):
                cat_a, cat_b = cat_list[ci], cat_list[cj]
                vecs_a = list(cat_vecs[cat_a].values())[:3]
                vecs_b = list(cat_vecs[cat_b].values())[:3]
                for va in vecs_a:
                    for vb in vecs_b:
                        cross_cat_diffs.append(va - vb)

        if same_cat_diffs and cross_cat_diffs:
            same_norms = [float(np.linalg.norm(d)) for d in same_cat_diffs]
            cross_norms = [float(np.linalg.norm(d)) for d in cross_cat_diffs]

            layer_data["avg_same_cat_diff_norm"] = round(float(np.mean(same_norms)), 4)
            layer_data["avg_cross_cat_diff_norm"] = round(float(np.mean(cross_norms)), 4)
            layer_data["ratio_cross_over_same"] = round(
                float(np.mean(cross_norms) / max(np.mean(same_norms), 1e-10)), 4)

            # Direction stability: for same-cat diffs, compute pairwise cosine similarity
            if len(same_cat_diffs) >= 2:
                # Normalize
                diff_norms = [np.linalg.norm(d) for d in same_cat_diffs]
                valid_diffs = [d for d, n in zip(same_cat_diffs, diff_norms) if n > 1e-10]
                if len(valid_diffs) >= 2:
                    normed = [d / np.linalg.norm(d) for d in valid_diffs]
                    cos_sims = []
                    for i in range(min(10, len(normed))):
                        for j in range(i+1, min(10, len(normed))):
                            cos_sims.append(float(np.dot(normed[i], normed[j])))
                    layer_data["same_cat_diff_direction_stability"] = round(
                        float(np.mean(cos_sims)), 4) if cos_sims else 0

        if layer_data:
            results[f"L{li}"] = layer_data

    return results


def compute_novel_combinations(model, tokenizer, device, n_layers_plus1, model_info,
                                sample_layers):
    """
    Exp 5: Novel concept combinations.

    For each combination:
    - Compare hidden state of noun in modified context vs base context
    - Measure: how much does the adjective shift the noun representation?
    - Is the shift additive (along adjective direction) or nonlinear?
    """
    results = {}

    for combo, base, adj in NOVEL_COMBINATIONS:
        print(f"    Combo: '{combo}' vs '{base}' (adj={adj})", flush=True)

        combo_result = {}

        # Find positions in the FULL tokenized sequence
        combo_full_tokens = tokenizer.encode(combo, add_special_tokens=True)
        base_full_tokens = tokenizer.encode(base, add_special_tokens=True)

        # Find noun position: use find_word_position
        # For "The apple", noun is "apple"; for "The blue apple", noun is "apple"
        base_noun = base.split()[-1]  # Last word in base template
        combo_noun = combo.split()[-1]  # Last word in combo template
        combo_adj_word = adj

        base_noun_pos = find_word_position(tokenizer, base, base_noun)
        combo_noun_pos = find_word_position(tokenizer, combo, combo_noun)
        combo_adj_pos = find_word_position(tokenizer, combo, combo_adj_word)

        # Get hidden states
        input_device = next(model.parameters()).device

        # Base context
        base_inputs = tokenizer(base, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            base_out = model(input_ids=base_inputs["input_ids"].to(input_device),
                           attention_mask=base_inputs["attention_mask"].to(input_device),
                           output_hidden_states=True)

        # Combo context
        combo_inputs = tokenizer(combo, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            combo_out = model(input_ids=combo_inputs["input_ids"].to(input_device),
                             attention_mask=combo_inputs["attention_mask"].to(input_device),
                             output_hidden_states=True)

        # Compare at sample layers
        layer_shifts = {}
        for li in sample_layers:
            if li >= len(base_out.hidden_states) or li >= len(combo_out.hidden_states):
                continue

            h_base = base_out.hidden_states[li][0].float().cpu().numpy()  # [seq_len, d_model]
            h_combo = combo_out.hidden_states[li][0].float().cpu().numpy()

            if base_noun_pos >= h_base.shape[1] or combo_noun_pos >= h_combo.shape[1]:
                continue

            # Noun representation in base vs combo context
            noun_base = h_base[0, base_noun_pos] if base_noun_pos < h_base.shape[1] else None
            noun_combo = h_combo[0, combo_noun_pos] if combo_noun_pos < h_combo.shape[1] else None

            if noun_base is None or noun_combo is None:
                continue

            # Shift = noun in combo context - noun in base context
            shift = noun_combo - noun_base
            shift_norm = float(np.linalg.norm(shift))
            noun_base_norm = float(np.linalg.norm(noun_base))
            relative_shift = shift_norm / max(noun_base_norm, 1e-10)

            # Cosine between shift direction and noun direction
            if shift_norm > 1e-10 and noun_base_norm > 1e-10:
                cos_shift_noun = float(np.dot(shift, noun_base) /
                                      (shift_norm * noun_base_norm))
            else:
                cos_shift_noun = 0

            # If there's an adjective representation, compare shift to adjective direction
            adj_info = {}
            if combo_adj_pos < h_combo.shape[1]:
                adj_repr = h_combo[0, combo_adj_pos]
                adj_norm = float(np.linalg.norm(adj_repr))
                if shift_norm > 1e-10 and adj_norm > 1e-10:
                    cos_shift_adj = float(np.dot(shift, adj_repr) /
                                          (shift_norm * adj_norm))
                    adj_info["cos_shift_adj"] = round(float(cos_shift_adj), 4)

            layer_shifts[f"L{li}"] = {
                "shift_norm": round(shift_norm, 4),
                "relative_shift": round(relative_shift, 4),
                "cos_shift_noun": round(float(cos_shift_noun), 4),
                **adj_info,
            }

        combo_result["layer_shifts"] = layer_shifts

        # Summarize: early, mid, late shifts
        early_layers = [k for k in layer_shifts if int(k[1:]) <= model_info.n_layers // 3]
        mid_layers = [k for k in layer_shifts if model_info.n_layers // 3 < int(k[1:]) <= 2 * model_info.n_layers // 3]
        late_layers = [k for k in layer_shifts if int(k[1:]) > 2 * model_info.n_layers // 3]

        for phase_name, phase_layers in [("early", early_layers), ("mid", mid_layers), ("late", late_layers)]:
            if phase_layers:
                shifts = [layer_shifts[k]["relative_shift"] for k in phase_layers]
                cos_nouns = [layer_shifts[k]["cos_shift_noun"] for k in phase_layers]
                combo_result[f"avg_relative_shift_{phase_name}"] = round(float(np.mean(shifts)), 4)
                combo_result[f"avg_cos_shift_noun_{phase_name}"] = round(float(np.mean(cos_nouns)), 4)

        results[combo] = combo_result

    return results


# =====================================================================
# MAIN EXPERIMENT
# =====================================================================

def run_phase174(model_name):
    print(f"\n{'='*70}", flush=True)
    print(f"Phase 174: Minimum Concept Universe — {model_name}", flush=True)
    print(f"{'='*70}", flush=True)

    t_start = time.time()

    # Load model
    model, tokenizer, device = load_model_bf16(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    n_layers_plus1 = n_layers + 1
    d_model = model_info.d_model

    print(f"  Model: {model_info.model_class}, L={n_layers}, d={d_model}", flush=True)

    # Sample layers
    sample_layers = list(range(0, n_layers_plus1, max(1, n_layers // 15))) + [n_layers]
    sample_layers = sorted(set(sample_layers))
    print(f"  Sample layers: {sample_layers}", flush=True)

    # =====================================================================
    # Exp 1: Category Clustering Dynamics
    # =====================================================================
    print(f"\n--- Exp 1: Category Clustering ({len(FRUITS)+len(VEHICLES)+len(ANIMALS)+len(EMOTIONS)} words) ---", flush=True)

    categories = {
        "fruits": FRUITS,
        "vehicles": VEHICLES,
        "animals": ANIMALS,
        "emotions": EMOTIONS,
    }

    all_hiddens_by_category = {cat: {} for cat in categories}

    for cat_name, words in categories.items():
        print(f"  Category: {cat_name} ({len(words)} words)", flush=True)
        for wi, word in enumerate(words):
            template = f"The {word} is"
            # Find word position dynamically (models may add BOS tokens)
            word_pos = find_word_position(tokenizer, template, word)

            hiddens = get_word_hidden_states(model, tokenizer, device, template,
                                            word_pos, n_layers_plus1, model_info)
            all_hiddens_by_category[cat_name][word] = hiddens

            if wi % 5 == 0:
                print(f"    {cat_name}: {wi+1}/{len(words)} done", flush=True)

    # Compute category statistics
    print("  Computing category clustering statistics...", flush=True)
    cat_stats = compute_category_stats(all_hiddens_by_category, sample_layers)

    # Find best clustering layer
    best_sep_layer = None
    best_sep = 0
    for key, data in cat_stats.items():
        if data["separation_ratio"] > best_sep:
            best_sep = data["separation_ratio"]
            best_sep_layer = key

    print(f"  Best separation: {best_sep_layer} (ratio={best_sep:.4f})", flush=True)

    # =====================================================================
    # Exp 2: Hierarchical Abstraction
    # =====================================================================
    print(f"\n--- Exp 2: Hierarchical Abstraction ---", flush=True)

    hierarchical_concepts = {
        "concrete_fruits": CONCRETE_FRUITS,
        "food_not_fruit": FOOD_NOT_FRUIT,
        "abstract_food": ABSTRACT_FOOD,
    }

    all_hiddens_by_concept = {cat: {} for cat in hierarchical_concepts}

    for concept_cat, words in hierarchical_concepts.items():
        print(f"  Concept group: {concept_cat} ({len(words)} words)", flush=True)
        for word in words:
            template = f"The {word} is"
            word_pos = find_word_position(tokenizer, template, word)
            hiddens = get_word_hidden_states(model, tokenizer, device, template,
                                            word_pos, n_layers_plus1, model_info)
            all_hiddens_by_concept[concept_cat][word] = hiddens

    hier_stats = compute_hierarchical_structure(all_hiddens_by_concept, sample_layers)

    # =====================================================================
    # Exp 3: Compositional Decomposition
    # =====================================================================
    print(f"\n--- Exp 3: Compositional Decomposition ---", flush=True)

    decomp_stats = {}
    for cat_name in ["fruits", "vehicles", "animals"]:
        print(f"  Decomposing: {cat_name}", flush=True)
        decomp_stats[cat_name] = compute_compositional_decomposition(
            all_hiddens_by_category, cat_name, sample_layers)

    # =====================================================================
    # Exp 4: Cross-Category Differential Structure
    # =====================================================================
    print(f"\n--- Exp 4: Cross-Category Differential Structure ---", flush=True)

    diff_stats = compute_differential_structure(all_hiddens_by_category, sample_layers)

    # =====================================================================
    # Exp 5: Novel Concept Combinations
    # =====================================================================
    print(f"\n--- Exp 5: Novel Concept Combinations ---", flush=True)

    combo_stats = compute_novel_combinations(model, tokenizer, device, n_layers_plus1,
                                             model_info, sample_layers)

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print(f"\n{'='*70}", flush=True)
    print(f"Phase 174 SUMMARY — {model_name}", flush=True)
    print(f"{'='*70}", flush=True)

    # Exp 1 summary
    print("\n[Exp 1: Category Clustering]", flush=True)
    for key in sorted(cat_stats.keys()):
        d = cat_stats[key]
        print(f"  {key}: intra={d['avg_intra_sim']:.4f}, inter={d['avg_inter_sim']:.4f}, "
              f"sep={d['separation_ratio']:.4f}, cluster_acc={d['cluster_accuracy']:.4f}", flush=True)

    # Exp 3 summary
    print("\n[Exp 3: Compositional Decomposition]", flush=True)
    for cat_name, decomp in decomp_stats.items():
        if decomp:
            # Print key layers: L0, L_mid, L_last
            key_layers = [f"L{sample_layers[0]}", f"L{sample_layers[len(sample_layers)//2]}", f"L{sample_layers[-1]}"]
            for kl in key_layers:
                if kl in decomp:
                    d = decomp[kl]
                    ve = d.get("var_explained", {})
                    top1 = ve.get("top1", "N/A")
                    top3 = ve.get("top3", "N/A")
                    print(f"  {cat_name} {kl}: var_top1={top1}, var_top3={top3}, "
                          f"dist/norm={d.get('dist_norm_ratio', 'N/A')}", flush=True)

    # Exp 4 summary
    print("\n[Exp 4: Differential Structure]", flush=True)
    for key in sorted(diff_stats.keys()):
        d = diff_stats[key]
        print(f"  {key}: same_diff={d.get('avg_same_cat_diff_norm', 'N/A')}, "
              f"cross_diff={d.get('avg_cross_cat_diff_norm', 'N/A')}, "
              f"ratio={d.get('ratio_cross_over_same', 'N/A')}", flush=True)

    # Exp 5 summary
    print("\n[Exp 5: Novel Combinations]", flush=True)
    for combo, data in combo_stats.items():
        early_shift = data.get("avg_relative_shift_early", "N/A")
        late_shift = data.get("avg_relative_shift_late", "N/A")
        print(f"  '{combo}': shift_early={early_shift}, shift_late={late_shift}", flush=True)

    # =====================================================================
    # SAVE RESULTS
    # =====================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output = {
        "model": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "timestamp": timestamp,
        "sample_layers": sample_layers,
        "exp1_category_clustering": cat_stats,
        "exp2_hierarchical_abstraction": hier_stats,
        "exp3_compositional_decomposition": decomp_stats,
        "exp4_differential_structure": diff_stats,
        "exp5_novel_combinations": {k: v for k, v in combo_stats.items()},
    }

    out_path = f"tests/glm5_temp/phase174_{model_name}_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}", flush=True)

    # Release model
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    print(f"\nPhase 174 ({model_name}) completed in {elapsed:.1f}s", flush=True)

    return output


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python phase174_concept_universe.py <model_name>")
        print("  model_name: qwen3, glm4, deepseek7b")
        sys.exit(1)

    model_name = sys.argv[1]
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    run_phase174(model_name)
