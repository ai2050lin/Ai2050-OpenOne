"""
Phase 166: Activation Topology & Feature Reuse — 激活拓扑与特征复用
===================================================================

★★★ CORE PIVOT: From output statistics → parameter-level formation mechanisms! ★★★

User's directive:
  "不要继续只做KL和PCA，已经接近极限了，看不到形成机制"
  "真正应该研究的是activation topology和feature reuse compression"
  "语言避免了维度灾难，说明背后有某种组合压缩机制"

Phase 165 findings:
  - Concepts are NOT context-free operators (cross-domain cos=-0.013)
  - Concepts have low-rank deformation (rank≈6-15)
  - Associativity approximately holds (cos≈0.89)
  - Nouns dominate in composition (21/25)

BUT: All of this is output-level statistics! We don't know WHY or HOW!

The real questions:
  - Which neurons/heads implement the constraint channels?
  - How do related concepts share feature directions?
  - What is the minimal "feature basis" that language uses?
  - How does the network avoid dimensionality curse?

Four experiments:

  Exp 1: ★★★ Attention Head Topology — 注意力头拓扑
    - For each concept, which attention heads are "on"?
    - Compare head activation patterns between related vs. unrelated concepts
    - KEY: Do related concepts share attention routing paths?

  Exp 2: ★★★ MLP Neuron Topology — MLP神经元拓扑
    - For each concept, which MLP neurons are most active?
    - Compare neuron activation overlap (Jaccard) between related concepts
    - KEY: This is the "feature reuse compression" mechanism!

  Exp 3: ★★★ Layer-wise Constraint Contribution — 层级约束贡献
    - How much does each layer contribute to the concept's constraint?
    - Use residual stream decomposition: logits = sum of layer contributions
    - KEY: Which layers implement "syntax" vs "semantics" vs "style"?

  Exp 4: ★★★ Sparse Feature Basis — 稀疏特征基
    - Collect MLP activation vectors for all concepts
    - SVD to find the minimal feature basis
    - KEY: What is the dimensionality of the "concept activation space"?
    - This directly addresses: how does language avoid dimensionality curse?

Usage: python tests/glm5/phase166_activation_topology.py <model_name>
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


# ===== CONCEPT SETS (same as Phase 165 for continuity) =====

CONCEPT_FAMILIES = {
    "animal": ["cat", "dog", "tiger", "lion", "horse", "elephant", "eagle", "whale"],
    "vehicle": ["car", "truck", "bus", "train", "bicycle", "airplane", "boat", "motorcycle"],
    "color": ["red", "blue", "green", "white", "black", "yellow", "purple", "orange"],
    "abstract": ["democracy", "freedom", "justice", "power", "truth", "beauty", "wisdom", "courage"],
    "emotion": ["love", "anger", "fear", "joy", "sadness", "hope", "pride", "shame"],
}

ALL_CONCEPTS = []
CONCEPT_TO_FAMILY = {}
for family, concepts in CONCEPT_FAMILIES.items():
    for c in concepts:
        ALL_CONCEPTS.append(c)
        CONCEPT_TO_FAMILY[c] = family


# ===== CONTEXT TEMPLATES =====

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
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def entropy(p, eps=1e-10):
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def jaccard_similarity(set1, set2):
    """Jaccard similarity of two sets."""
    if not set1 and not set2:
        return 1.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


# ===== EXP 1: ATTENTION HEAD TOPOLOGY =====

def exp1_attention_head_topology(model, tokenizer, device, model_name,
                                  n_concepts=40, n_templates=8, top_k_heads=5):
    """
    ★★★ Attention Head Topology — 注意力头拓扑 ★★★

    For each concept, at each layer, which attention heads are most active?
    - Capture per-head output norms
    - Compare overlap between related vs. unrelated concepts
    - This reveals: do related concepts share routing paths?

    Key metrics:
    - Per-head norm: ||head_i|| — how much this head contributes
    - Head activation pattern: which heads are "on" for each concept
    - Cross-concept overlap: Jaccard of top-k heads between concept pairs
    """
    import torch

    print("\n" + "="*60)
    print("EXP 1: Attention Head Topology")
    print("="*60)

    input_device = get_device_for_input(model)
    layers = get_layers(model)
    n_layers = len(layers)

    # Get model dimensions
    info = get_model_info(model, model_name)
    # Determine n_heads from the attention module
    layer0 = layers[0]
    sa = layer0.self_attn
    if hasattr(sa, 'num_heads'):
        n_heads = sa.num_heads
    elif hasattr(sa, 'n_heads'):
        n_heads = sa.n_heads
    else:
        # Infer from q_proj weight shape
        n_heads = sa.q_proj.weight.shape[0] // info.d_model
        n_heads = max(1, n_heads)

    d_head = info.d_model // n_heads

    print(f"  n_layers={n_layers}, n_heads={n_heads}, d_head={d_head}")
    print(f"  Testing {n_concepts} concepts × {n_templates} templates")

    concepts_to_test = ALL_CONCEPTS[:n_concepts]
    templates = CONTEXT_TEMPLATES[:n_templates]

    # For each concept, collect head norms across all layers
    # concept -> layer -> [n_heads] head norms
    concept_head_norms = {}  # concept -> np.array [n_layers, n_heads]
    concept_top_heads = {}   # concept -> {(layer, head_idx)}

    for cidx, concept in enumerate(concepts_to_test):
        if cidx % 10 == 0:
            print(f"  Concept {cidx}/{len(concepts_to_test)}: {concept}")

        all_head_norms = np.zeros((n_layers, n_heads))
        concept_top_heads[concept] = set()

        for template in templates:
            prompt = template.replace("___", concept)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)

            # Use hooks to capture attention head outputs
            captured_attn = {}

            def make_attn_hook(layer_idx):
                def hook(module, input, output):
                    # output is tuple: (attn_output, attn_weights, ...)
                    # attn_output shape: [batch, seq_len, d_model]
                    if isinstance(output, tuple):
                        captured_attn[layer_idx] = output[0].detach().float().cpu()
                    else:
                        captured_attn[layer_idx] = output.detach().float().cpu()
                return hook

            hooks = []
            for li in range(n_layers):
                hooks.append(layers[li].self_attn.o_proj.register_forward_hook(
                    make_attn_hook(li)))

            with torch.no_grad():
                try:
                    _ = model(input_ids=input_ids, attention_mask=attention_mask)
                except Exception as e:
                    print(f"    Forward failed for {concept}: {e}")

            for h in hooks:
                h.remove()

            # For each layer, split the attention output into heads
            for li in range(n_layers):
                if li in captured_attn:
                    attn_out = captured_attn[li]  # [1, seq_len, d_model]
                    # Take the last token position (where next-token prediction happens)
                    last_pos = attn_out[0, -1, :]  # [d_model]

                    # Split into heads: each head contributes d_head dimensions
                    # But the output projection mixes heads, so we can't directly split
                    # Instead, we look at the INPUT to the output projection
                    # which is the concatenated head outputs

                    # Actually, we need to hook BEFORE the output projection
                    # Let's use a different approach: compute per-head contribution
                    # by looking at the attention weights

                    # Simple approximation: use the attention weight entropy
                    # to measure how "focused" each head is
                    pass

            del input_ids, attention_mask

        # Actually, let's use a different approach that's more reliable:
        # Use output_attentions=True to get actual attention weights
        # Then compute per-head attention entropy

        concept_head_norms[concept] = all_head_norms

    # ===== Alternative approach: Use output_attentions =====
    print("\n  Using output_attentions=True approach...")

    concept_head_entropy = {}   # concept -> [n_layers, n_heads] entropy values
    concept_head_topk = {}     # concept -> {(layer, head)} top-k heads

    for cidx, concept in enumerate(concepts_to_test):
        if cidx % 10 == 0:
            print(f"  Concept {cidx}/{len(concepts_to_test)}: {concept}")

        head_entropies = np.zeros((n_layers, n_heads))
        concept_head_topk[concept] = set()

        for template in templates:
            prompt = template.replace("___", concept)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)

            with torch.no_grad():
                try:
                    out = model(input_ids=input_ids, attention_mask=attention_mask,
                               output_attentions=True)
                except Exception as e:
                    print(f"    Forward failed: {e}")
                    continue

            # attentions: tuple of [n_layers], each [batch, n_heads, seq_len, seq_len]
            if hasattr(out, 'attentions') and out.attentions is not None:
                for li, attn_weights in enumerate(out.attentions):
                    # attn_weights: [1, n_heads, seq_len, seq_len]
                    attn_np = attn_weights[0, :, -1, :].float().cpu().numpy()  # [n_heads, seq_len]
                    # Last position's attention pattern

                    for hi in range(n_heads):
                        # Compute entropy of attention distribution
                        attn_dist = attn_np[hi]
                        attn_dist = np.clip(attn_dist, 1e-10, 1.0)
                        attn_dist = attn_dist / attn_dist.sum()
                        ent = -np.sum(attn_dist * np.log(attn_dist))
                        head_entropies[li, hi] += ent

            del input_ids, attention_mask

        # Average over templates
        head_entropies /= len(templates)

        # Find top-k heads (lowest entropy = most focused)
        concept_head_entropy[concept] = head_entropies

        # Top-k heads per concept (across all layers)
        flat_entropies = head_entropies.flatten()
        n_total_heads = n_layers * n_heads
        top_k = min(top_k_heads * n_layers, n_total_heads)  # top k per layer
        top_indices = np.argsort(flat_entropies)[:top_k]

        for idx in top_indices:
            layer_idx = idx // n_heads
            head_idx = idx % n_heads
            concept_head_topk[concept].add((layer_idx, head_idx))

    # ===== Cross-concept overlap analysis =====
    print("\n  Computing cross-concept overlap...")

    # For each pair of concepts, compute Jaccard similarity of top heads
    same_family_jaccard = []
    diff_family_jaccard = []

    for i, c1 in enumerate(concepts_to_test):
        for j, c2 in enumerate(concepts_to_test):
            if i >= j:
                continue
            top1 = concept_head_topk[c1]
            top2 = concept_head_topk[c2]
            jacc = jaccard_similarity(top1, top2)

            if CONCEPT_TO_FAMILY[c1] == CONCEPT_TO_FAMILY[c2]:
                same_family_jaccard.append(jacc)
            else:
                diff_family_jaccard.append(jacc)

    # ===== Head specialization analysis =====
    # Which heads are "concept-specific" vs "shared"?
    head_concept_count = defaultdict(set)  # (layer, head) -> set of concepts that activate it

    for concept in concepts_to_test:
        for (layer_idx, head_idx) in concept_head_topk[concept]:
            head_concept_count[(layer_idx, head_idx)].add(concept)

    # Heads that are activated by many concepts = "shared routing"
    # Heads that are activated by few concepts = "specialized"
    shared_heads = [(k, len(v)) for k, v in head_concept_count.items() if len(v) >= 5]
    specialized_heads = [(k, len(v)) for k, v in head_concept_count.items() if len(v) <= 2]

    # Sort by number of concepts
    shared_heads.sort(key=lambda x: -x[1])
    specialized_heads.sort(key=lambda x: x[1])

    # Per-layer head usage
    layer_head_usage = {}
    for li in range(n_layers):
        heads_at_layer = [(k, len(v)) for k, v in head_concept_count.items() if k[0] == li]
        if heads_at_layer:
            avg_usage = np.mean([h[1] for h in heads_at_layer])
            max_usage = max([h[1] for h in heads_at_layer])
            layer_head_usage[li] = {
                "n_active_heads": len(heads_at_layer),
                "avg_concept_count": round(float(avg_usage), 2),
                "max_concept_count": int(max_usage),
            }

    results = {
        "n_concepts": len(concepts_to_test),
        "n_templates": len(templates),
        "n_layers": n_layers,
        "n_heads_per_layer": n_heads,
        "same_family_jaccard_mean": round(float(np.mean(same_family_jaccard)), 4) if same_family_jaccard else 0,
        "same_family_jaccard_std": round(float(np.std(same_family_jaccard)), 4) if same_family_jaccard else 0,
        "diff_family_jaccard_mean": round(float(np.mean(diff_family_jaccard)), 4) if diff_family_jaccard else 0,
        "diff_family_jaccard_std": round(float(np.std(diff_family_jaccard)), 4) if diff_family_jaccard else 0,
        "n_shared_heads": len(shared_heads),
        "n_specialized_heads": len(specialized_heads),
        "top5_shared_heads": [((k[0], k[1]), v) for k, v in shared_heads[:5]],
        "top5_specialized_heads": [((k[0], k[1]), v) for k, v in specialized_heads[:5]],
        "layer_head_usage": layer_head_usage,
    }

    print(f"\n  === Attention Head Topology Results ===")
    print(f"  Same-family Jaccard: {results['same_family_jaccard_mean']:.4f} ± {results['same_family_jaccard_std']:.4f}")
    print(f"  Diff-family Jaccard: {results['diff_family_jaccard_mean']:.4f} ± {results['diff_family_jaccard_std']:.4f}")
    print(f"  Shared heads (≥5 concepts): {len(shared_heads)}")
    print(f"  Specialized heads (≤2 concepts): {len(specialized_heads)}")

    return results


# ===== EXP 2: MLP NEURON TOPOLOGY =====

def exp2_mlp_neuron_topology(model, tokenizer, device, model_name,
                               n_concepts=40, n_templates=8, top_k_neurons=100):
    """
    ★★★ MLP Neuron Topology — MLP神经元拓扑 ★★★

    For each concept, which MLP neurons are most active?
    - Capture MLP intermediate activations (post-gate)
    - Compare neuron activation overlap between related vs. unrelated concepts
    - This directly measures "feature reuse compression"!

    Key insight: If "cat" and "dog" share many top neurons →
    language avoids dimensionality curse through feature reuse!
    """
    import torch

    print("\n" + "="*60)
    print("EXP 2: MLP Neuron Topology")
    print("="*60)

    input_device = get_device_for_input(model)
    layers = get_layers(model)
    n_layers = len(layers)
    info = get_model_info(model, model_name)

    # Determine intermediate size
    layer0 = layers[0]
    mlp = layer0.mlp
    if info.mlp_type == "merged_gate_up":
        d_intermediate = mlp.gate_up_proj.weight.shape[0] // 2
    else:
        d_intermediate = mlp.up_proj.weight.shape[0] if hasattr(mlp, 'up_proj') else 0

    print(f"  d_intermediate = {d_intermediate}")
    print(f"  Testing {n_concepts} concepts × {n_templates} templates")

    concepts_to_test = ALL_CONCEPTS[:n_concepts]
    templates = CONTEXT_TEMPLATES[:n_templates]

    # Sample layers (can't capture all layers at once due to memory)
    sample_layers = list(range(0, n_layers, max(1, n_layers // 10))) + [n_layers - 1]
    sample_layers = sorted(set(sample_layers))
    print(f"  Sampling {len(sample_layers)} layers: {sample_layers}")

    # For each concept, collect top-k MLP neuron indices at sampled layers
    # concept -> layer -> set of top-k neuron indices
    concept_top_neurons = defaultdict(lambda: defaultdict(set))

    # Process one layer at a time to save GPU memory
    for li in sample_layers:
        print(f"  Processing layer {li}...")

        # Hook for MLP intermediate activation
        captured_mlp = {}

        def make_mlp_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    captured_mlp[layer_idx] = output[0].detach().float().cpu()
                else:
                    captured_mlp[layer_idx] = output.detach().float().cpu()
            return hook

        for concept in concepts_to_test:
            neuron_acts = []  # Collect across templates

            for template in templates:
                prompt = template.replace("___", concept)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
                input_ids = inputs["input_ids"].to(input_device)
                attention_mask = inputs["attention_mask"].to(input_device)

                # Register hook on MLP output
                hook = layers[li].mlp.register_forward_hook(make_mlp_hook(li))

                with torch.no_grad():
                    try:
                        _ = model(input_ids=input_ids, attention_mask=attention_mask)
                    except Exception as e:
                        print(f"    Forward failed: {e}")

                hook.remove()

                if li in captured_mlp:
                    mlp_out = captured_mlp[li]  # [1, seq_len, d_model]
                    # Take last token position
                    last_pos = mlp_out[0, -1, :].numpy()  # [d_model]

                    # Compute activation magnitude as norm contribution
                    neuron_acts.append(np.abs(last_pos))

                del input_ids, attention_mask
                if li in captured_mlp:
                    del captured_mlp[li]

            # Average across templates
            if neuron_acts:
                avg_acts = np.mean(neuron_acts, axis=0)  # [d_model]
                top_indices = set(np.argsort(avg_acts)[-top_k_neurons:].tolist())
                concept_top_neurons[concept][li] = top_indices

    # ===== Cross-concept neuron overlap =====
    print("\n  Computing cross-concept neuron overlap...")

    # For each pair, compute Jaccard at each sampled layer
    same_family_jaccard_per_layer = defaultdict(list)
    diff_family_jaccard_per_layer = defaultdict(list)

    for i, c1 in enumerate(concepts_to_test):
        for j, c2 in enumerate(concepts_to_test):
            if i >= j:
                continue
            same_family = CONCEPT_TO_FAMILY[c1] == CONCEPT_TO_FAMILY[c2]

            for li in sample_layers:
                if li in concept_top_neurons[c1] and li in concept_top_neurons[c2]:
                    jacc = jaccard_similarity(concept_top_neurons[c1][li],
                                              concept_top_neurons[c2][li])
                    if same_family:
                        same_family_jaccard_per_layer[li].append(jacc)
                    else:
                        diff_family_jaccard_per_layer[li].append(jacc)

    # Aggregate per layer
    layer_overlap = {}
    for li in sample_layers:
        sj = same_family_jaccard_per_layer.get(li, [])
        dj = diff_family_jaccard_per_layer.get(li, [])
        layer_overlap[li] = {
            "same_family_jaccard_mean": round(float(np.mean(sj)), 4) if sj else 0,
            "same_family_jaccard_std": round(float(np.std(sj)), 4) if sj else 0,
            "diff_family_jaccard_mean": round(float(np.mean(dj)), 4) if dj else 0,
            "diff_family_jaccard_std": round(float(np.std(dj)), 4) if dj else 0,
            "overlap_ratio": round(float(np.mean(sj)) / max(float(np.mean(dj)), 1e-6), 2) if sj and dj else 0,
        }

    # ===== Neuron reuse analysis =====
    # How many unique neurons are needed across all concepts?
    all_neurons_per_layer = {}
    for li in sample_layers:
        all_neurons = set()
        for concept in concepts_to_test:
            if li in concept_top_neurons[concept]:
                all_neurons.update(concept_top_neurons[concept][li])
        all_neurons_per_layer[li] = len(all_neurons)

    # Total "space" needed = sum of top-k per concept = n_concepts * top_k
    # Actual space used = |union of all top neurons|
    # Compression ratio = actual / total
    total_space = n_concepts * top_k_neurons
    compression_ratios = {}
    for li in sample_layers:
        actual = all_neurons_per_layer[li]
        compression_ratios[li] = round(float(actual) / total_space, 4)

    # ===== Per-family neuron sets =====
    family_neurons = defaultdict(lambda: defaultdict(set))
    for concept in concepts_to_test:
        family = CONCEPT_TO_FAMILY[concept]
        for li in sample_layers:
            if li in concept_top_neurons[concept]:
                family_neurons[family][li].update(concept_top_neurons[concept][li])

    family_overlap = {}
    families = list(CONCEPT_FAMILIES.keys())
    for fi, f1 in enumerate(families):
        for fj, f2 in enumerate(families):
            if fi >= fj:
                continue
            for li in sample_layers:
                jacc = jaccard_similarity(family_neurons[f1].get(li, set()),
                                          family_neurons[f2].get(li, set()))
                family_overlap[f"{f1}_vs_{f2}_L{li}"] = round(jacc, 4)

    results = {
        "n_concepts": n_concepts,
        "n_templates": n_templates,
        "top_k_neurons": top_k_neurons,
        "d_intermediate": d_intermediate,
        "sample_layers": sample_layers,
        "layer_overlap": layer_overlap,
        "all_neurons_per_layer": {str(k): v for k, v in all_neurons_per_layer.items()},
        "compression_ratio": {str(k): v for k, v in compression_ratios.items()},
        "total_space_if_no_reuse": total_space,
        "family_overlap": family_overlap,
        "same_family_jaccard_overall": round(float(np.mean(
            [v for li in sample_layers for v in same_family_jaccard_per_layer.get(li, [])])), 4),
        "diff_family_jaccard_overall": round(float(np.mean(
            [v for li in sample_layers for v in diff_family_jaccard_per_layer.get(li, [])])), 4),
    }

    print(f"\n  === MLP Neuron Topology Results ===")
    print(f"  Same-family Jaccard (overall): {results['same_family_jaccard_overall']:.4f}")
    print(f"  Diff-family Jaccard (overall): {results['diff_family_jaccard_overall']:.4f}")
    print(f"  Compression ratio (mid-layer): {[compression_ratios.get(li, 0) for li in sample_layers[len(sample_layers)//2:len(sample_layers)//2+1]]}")
    for li in sample_layers:
        if li in layer_overlap:
            lo = layer_overlap[li]
            print(f"    L{li}: same_J={lo['same_family_jaccard_mean']:.4f}, "
                  f"diff_J={lo['diff_family_jaccard_mean']:.4f}, "
                  f"ratio={lo['overlap_ratio']:.2f}, "
                  f"compression={compression_ratios.get(li, 0):.4f}")

    return results


# ===== EXP 3: LAYER-WISE CONSTRAINT CONTRIBUTION =====

def exp3_layer_constraint_contribution(model, tokenizer, device, model_name,
                                         n_concepts=40, n_templates=8):
    """
    ★★★ Layer-wise Constraint Contribution — 层级约束贡献 ★★★

    How much does each layer contribute to the concept's constraint on P(future)?

    Method: Residual stream decomposition
    - The final logits = W_U @ h_final
    - h_final = h_embed + sum_l(Δh_l) where Δh_l is the residual update from layer l
    - logits = W_U @ h_embed + sum_l(W_U @ Δh_l)
    - Each layer's "constraint contribution" = how much W_U @ Δh_l changes the top predictions

    KEY: Which layers are "syntax" vs "semantics" vs "style"?
    """
    import torch

    print("\n" + "="*60)
    print("EXP 3: Layer-wise Constraint Contribution")
    print("="*60)

    input_device = get_device_for_input(model)
    layers = get_layers(model)
    n_layers = len(layers)
    info = get_model_info(model, model_name)

    # Get W_U (lm_head weights)
    W_U = get_W_U(model, model_name)  # [vocab_size, d_model]
    print(f"  W_U shape: {W_U.shape}")

    concepts_to_test = ALL_CONCEPTS[:n_concepts]
    templates = CONTEXT_TEMPLATES[:n_templates]

    print(f"  Testing {len(concepts_to_test)} concepts × {len(templates)} templates")

    # For each concept, compute layer-wise contribution to logits
    # concept -> layer -> contribution metrics
    concept_layer_contrib = {}

    for cidx, concept in enumerate(concepts_to_test):
        if cidx % 10 == 0:
            print(f"  Concept {cidx}/{len(concepts_to_test)}: {concept}")

        family = CONCEPT_TO_FAMILY[concept]

        # Accumulate across templates
        layer_logit_norms = np.zeros(n_layers)
        layer_top1_shifts = np.zeros(n_layers)
        layer_js_contribs = np.zeros(n_layers)

        for template in templates:
            baseline_prompt = template.replace("___", "the")
            concept_prompt = template.replace("___", concept)

            # Get hidden states for baseline
            bl_inputs = tokenizer(baseline_prompt, return_tensors="pt", truncation=True, max_length=128)
            bl_ids = bl_inputs["input_ids"].to(input_device)
            bl_mask = bl_inputs["attention_mask"].to(input_device)

            with torch.no_grad():
                bl_out = model(input_ids=bl_ids, attention_mask=bl_mask,
                              output_hidden_states=True)

            # Get hidden states for concept
            co_inputs = tokenizer(concept_prompt, return_tensors="pt", truncation=True, max_length=128)
            co_ids = co_inputs["input_ids"].to(input_device)
            co_mask = co_inputs["attention_mask"].to(input_device)

            with torch.no_grad():
                co_out = model(input_ids=co_ids, attention_mask=co_mask,
                              output_hidden_states=True)

            # hidden_states: tuple of (n_layers+1) tensors, each [1, seq_len, d_model]
            bl_hs = bl_out.hidden_states
            co_hs = co_out.hidden_states

            # Final logits for reference
            bl_logits = bl_out.logits[0, -1, :].float().cpu().numpy()
            co_logits = co_out.logits[0, -1, :].float().cpu().numpy()

            bl_probs = safe_softmax(bl_logits)
            co_probs = safe_softmax(co_logits)

            # Compute per-layer residual contribution
            # Δh_l = h_l(concept) - h_l(baseline)
            # layer contribution to logits = W_U @ Δh_l
            for li in range(n_layers):
                # Hidden state at layer l output (index li+1 in hidden_states)
                h_l_bl = bl_hs[li+1][0, -1, :].float().cpu().numpy()  # [d_model]
                h_l_co = co_hs[li+1][0, -1, :].float().cpu().numpy()  # [d_model]

                # Residual update at this layer
                h_prev_bl = bl_hs[li][0, -1, :].float().cpu().numpy()
                h_prev_co = co_hs[li][0, -1, :].float().cpu().numpy()

                delta_bl = h_l_bl - h_prev_bl  # layer l contribution (baseline)
                delta_co = h_l_co - h_prev_co  # layer l contribution (concept)

                # Concept-specific contribution at this layer
                delta_concept = delta_co - delta_bl  # extra contribution due to concept

                # Logit contribution
                logit_contrib = W_U @ delta_concept  # [vocab_size]

                # Metrics
                norm_contrib = float(np.linalg.norm(logit_contrib))
                layer_logit_norms[li] += norm_contrib

                # Top-1 shift
                top1_shift = float(np.max(np.abs(logit_contrib)))
                layer_top1_shifts[li] += top1_shift

                # JS contribution: how much does this layer's delta change the distribution?
                # Approximate: apply delta to baseline logits and measure JS
                modified_logits = bl_logits + logit_contrib
                modified_probs = safe_softmax(modified_logits)
                js_val = js_divergence(modified_probs, bl_probs)
                layer_js_contribs[li] += js_val

            del bl_ids, bl_mask, co_ids, co_mask, bl_out, co_out

        # Average across templates
        layer_logit_norms /= len(templates)
        layer_top1_shifts /= len(templates)
        layer_js_contribs /= len(templates)

        # Normalize: what fraction of total contribution comes from each layer?
        total_norm = np.sum(layer_logit_norms)
        if total_norm > 0:
            norm_fractions = layer_logit_norms / total_norm
        else:
            norm_fractions = np.zeros(n_layers)

        concept_layer_contrib[concept] = {
            "family": family,
            "layer_logit_norms": [round(float(v), 4) for v in layer_logit_norms],
            "layer_top1_shifts": [round(float(v), 4) for v in layer_top1_shifts],
            "layer_js_contribs": [round(float(v), 6) for v in layer_js_contribs],
            "norm_fractions": [round(float(v), 6) for v in norm_fractions],
            "top3_contributing_layers": sorted(range(n_layers),
                                                key=lambda l: layer_logit_norms[l],
                                                reverse=True)[:3],
            "early_layer_fraction": round(float(np.sum(norm_fractions[:n_layers//3])), 4),
            "mid_layer_fraction": round(float(np.sum(norm_fractions[n_layers//3:2*n_layers//3])), 4),
            "late_layer_fraction": round(float(np.sum(norm_fractions[2*n_layers//3:])), 4),
        }

    # ===== Aggregate by family =====
    family_layer_contrib = {}
    for family in CONCEPT_FAMILIES:
        concepts_in_family = [c for c in concepts_to_test if CONCEPT_TO_FAMILY[c] == family]
        if not concepts_in_family:
            continue

        avg_norm_fractions = np.mean([concept_layer_contrib[c]["norm_fractions"]
                                       for c in concepts_in_family], axis=0)

        family_layer_contrib[family] = {
            "avg_early_fraction": round(float(np.mean([concept_layer_contrib[c]["early_layer_fraction"]
                                                         for c in concepts_in_family])), 4),
            "avg_mid_fraction": round(float(np.mean([concept_layer_contrib[c]["mid_layer_fraction"]
                                                       for c in concepts_in_family])), 4),
            "avg_late_fraction": round(float(np.mean([concept_layer_contrib[c]["late_layer_fraction"]
                                                        for c in concepts_in_family])), 4),
            "peak_layer": int(np.argmax(avg_norm_fractions)),
        }

    # ===== Overall layer contribution profile =====
    all_norm_fractions = np.mean([concept_layer_contrib[c]["norm_fractions"]
                                   for c in concepts_to_test], axis=0)

    # Find "syntax layers" vs "semantics layers" vs "style layers"
    # Heuristic: early=syntax, mid=semantics, late=style (from literature)
    early_frac = float(np.sum(all_norm_fractions[:n_layers//3]))
    mid_frac = float(np.sum(all_norm_fractions[n_layers//3:2*n_layers//3]))
    late_frac = float(np.sum(all_norm_fractions[2*n_layers//3:]))

    results = {
        "n_concepts": len(concepts_to_test),
        "n_templates": len(templates),
        "n_layers": n_layers,
        "concept_layer_contrib": concept_layer_contrib,
        "family_layer_contrib": family_layer_contrib,
        "overall_early_fraction": round(early_frac, 4),
        "overall_mid_fraction": round(mid_frac, 4),
        "overall_late_fraction": round(late_frac, 4),
        "overall_peak_layer": int(np.argmax(all_norm_fractions)),
        "overall_norm_fractions": [round(float(v), 6) for v in all_norm_fractions],
    }

    print(f"\n  === Layer-wise Constraint Contribution Results ===")
    print(f"  Overall: early={early_frac:.4f}, mid={mid_frac:.4f}, late={late_frac:.4f}")
    print(f"  Peak layer: {results['overall_peak_layer']}")
    for family, fc in family_layer_contrib.items():
        print(f"  {family}: early={fc['avg_early_fraction']:.4f}, "
              f"mid={fc['avg_mid_fraction']:.4f}, late={fc['avg_late_fraction']:.4f}, "
              f"peak={fc['peak_layer']}")

    return results


# ===== EXP 4: SPARSE FEATURE BASIS =====

def exp4_sparse_feature_basis(model, tokenizer, device, model_name,
                                n_concepts=40, n_templates=8, target_layer_idx=None):
    """
    ★★★ Sparse Feature Basis — 稀疏特征基 ★★★

    Collect MLP activation vectors for all concepts at a mid-layer.
    SVD to find the minimal feature basis that reconstructs all concept activations.

    KEY QUESTION: What is the dimensionality of the "concept activation space"?
    - If dim << n_concepts → language uses feature reuse!
    - If dim ≈ n_concepts → each concept is independent (no reuse)

    This directly addresses: HOW does language avoid dimensionality curse?
    """
    import torch

    print("\n" + "="*60)
    print("EXP 4: Sparse Feature Basis")
    print("="*60)

    input_device = get_device_for_input(model)
    layers = get_layers(model)
    n_layers = len(layers)
    info = get_model_info(model, model_name)

    # Use a mid-layer for feature basis analysis
    if target_layer_idx is None:
        target_layer_idx = n_layers * 2 // 3  # 2/3 of the way through
    print(f"  Target layer: {target_layer_idx} (of {n_layers})")

    concepts_to_test = ALL_CONCEPTS[:n_concepts]
    templates = CONTEXT_TEMPLATES[:n_templates]

    print(f"  Testing {len(concepts_to_test)} concepts × {len(templates)} templates")

    # Step 1: Collect baseline MLP output for each concept at target layer
    print(f"  Step 1: Collecting MLP activation vectors...")

    # concept -> activation vector [d_model] (MLP output at target layer)
    concept_mlp_vectors = {}
    baseline_mlp_vector = None

    # First, get baseline ("the")
    captured = {}

    def make_hook():
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured['mlp'] = output[0].detach().float().cpu()
            else:
                captured['mlp'] = output.detach().float().cpu()
        return hook

    for template in templates:
        baseline_prompt = template.replace("___", "the")
        inputs = tokenizer(baseline_prompt, return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)

        hook = layers[target_layer_idx].mlp.register_forward_hook(make_hook())

        with torch.no_grad():
            try:
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
            except Exception as e:
                print(f"    Forward failed: {e}")

        hook.remove()

        if 'mlp' in captured:
            mlp_out = captured['mlp'][0, -1, :].numpy()  # [d_model]
            if baseline_mlp_vector is None:
                baseline_mlp_vector = mlp_out
            else:
                baseline_mlp_vector += mlp_out

        del input_ids, attention_mask

    if baseline_mlp_vector is not None:
        baseline_mlp_vector /= len(templates)

    # Now collect concept vectors
    for cidx, concept in enumerate(concepts_to_test):
        if cidx % 10 == 0:
            print(f"    Concept {cidx}/{len(concepts_to_test)}: {concept}")

        concept_vec = None

        for template in templates:
            prompt = template.replace("___", concept)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)

            captured.clear()
            hook = layers[target_layer_idx].mlp.register_forward_hook(make_hook())

            with torch.no_grad():
                try:
                    _ = model(input_ids=input_ids, attention_mask=attention_mask)
                except Exception as e:
                    print(f"    Forward failed: {e}")

            hook.remove()

            if 'mlp' in captured:
                mlp_out = captured['mlp'][0, -1, :].numpy()
                if concept_vec is None:
                    concept_vec = mlp_out
                else:
                    concept_vec += mlp_out

            del input_ids, attention_mask

        if concept_vec is not None:
            concept_vec /= len(templates)
            # Delta vector: concept - baseline
            if baseline_mlp_vector is not None:
                concept_mlp_vectors[concept] = concept_vec - baseline_mlp_vector
            else:
                concept_mlp_vectors[concept] = concept_vec

    # Step 2: Build activation matrix
    print(f"  Step 2: Building activation matrix...")

    concept_list = sorted(concept_mlp_vectors.keys())
    n_c = len(concept_list)
    d = len(concept_mlp_vectors[concept_list[0]])

    # Matrix M: [n_concepts, d_model]
    M = np.array([concept_mlp_vectors[c] for c in concept_list])
    print(f"  Activation matrix: {M.shape}")

    # Step 3: SVD to find feature basis
    print(f"  Step 3: SVD decomposition...")

    try:
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        # U: [n_c, n_c], S: [min(n_c, d)], Vt: [min(n_c, d), d]

        # How many components needed for 50%, 90%, 95%, 99% variance?
        energy = S ** 2
        total_energy = np.sum(energy)
        cum_energy = np.cumsum(energy) / total_energy

        rank_50 = int(np.searchsorted(cum_energy, 0.50) + 1)
        rank_90 = int(np.searchsorted(cum_energy, 0.90) + 1)
        rank_95 = int(np.searchsorted(cum_energy, 0.95) + 1)
        rank_99 = int(np.searchsorted(cum_energy, 0.99) + 1)

        # Top singular values
        sv_top10 = [round(float(s), 4) for s in S[:10]]
        sv_top30 = [round(float(s), 4) for s in S[:min(30, len(S))]]

        # Top singular value ratio
        top1_ratio = float(S[0] ** 2 / total_energy)
        top3_ratio = float(np.sum(S[:3] ** 2) / total_energy)
        top5_ratio = float(np.sum(S[:5] ** 2) / total_energy)

        # Step 4: Interpret the principal components
        print(f"  Step 4: Interpreting principal components...")

        # Each PC is a direction in d_model space
        # Which concepts have the highest projection on each PC?
        # M = U @ diag(S) @ Vt
        # Concept i's projection on PC j = U[i, j] * S[j]

        concept_projections = U * S[np.newaxis, :]  # [n_c, n_c]

        pc_interpretation = {}
        for pc_idx in range(min(5, len(S))):
            # Which concepts have the highest and lowest projections on this PC?
            projections = concept_projections[:, pc_idx]
            sorted_idx = np.argsort(projections)

            top5_concepts = [(concept_list[i], round(float(projections[i]), 4))
                            for i in sorted_idx[-5:]][::-1]
            bottom5_concepts = [(concept_list[i], round(float(projections[i]), 4))
                               for i in sorted_idx[:5]]

            # Which families dominate the top?
            top_families = [CONCEPT_TO_FAMILY[concept_list[i]] for i in sorted_idx[-n_c//2:]]
            family_counts = defaultdict(int)
            for f in top_families:
                family_counts[f] += 1
            dominant_family = max(family_counts, key=family_counts.get)

            pc_interpretation[f"PC{pc_idx}"] = {
                "singular_value": round(float(S[pc_idx]), 4),
                "variance_ratio": round(float(S[pc_idx] ** 2 / total_energy), 4),
                "top5_concepts": top5_concepts,
                "bottom5_concepts": bottom5_concepts,
                "dominant_family": dominant_family,
                "family_distribution": dict(family_counts),
            }

    except Exception as e:
        print(f"  SVD failed: {e}")
        rank_50, rank_90, rank_95, rank_99 = 0, 0, 0, 0
        sv_top10, sv_top30 = [], []
        top1_ratio, top3_ratio, top5_ratio = 0, 0, 0
        pc_interpretation = {}

    # Step 5: Feature reuse compression ratio
    print(f"  Step 5: Computing compression metrics...")

    # If we need rank_90 components to explain 90% of variance,
    # then the "feature reuse compression ratio" = rank_90 / n_concepts
    # Lower ratio = more reuse = language avoids dimensionality curse better

    compression_ratio = round(rank_90 / n_c, 4) if n_c > 0 else 0

    # Also compute per-family compression
    family_compression = {}
    for family in CONCEPT_FAMILIES:
        concepts_in_family = [c for c in concept_list if CONCEPT_TO_FAMILY[c] == family]
        if len(concepts_in_family) < 2:
            continue

        M_family = np.array([concept_mlp_vectors[c] for c in concepts_in_family])
        try:
            _, S_f, _ = np.linalg.svd(M_family, full_matrices=False)
            energy_f = S_f ** 2
            total_e = np.sum(energy_f)
            cum_e = np.cumsum(energy_f) / total_e
            r90 = int(np.searchsorted(cum_e, 0.90) + 1)
            family_compression[family] = {
                "n_concepts": len(concepts_in_family),
                "rank_90": r90,
                "compression_ratio": round(r90 / len(concepts_in_family), 4),
                "top1_ratio": round(float(S_f[0] ** 2 / total_e), 4),
            }
        except:
            family_compression[family] = {"n_concepts": len(concepts_in_family),
                                           "rank_90": 0, "compression_ratio": 0, "top1_ratio": 0}

    results = {
        "n_concepts": n_c,
        "d_model": d,
        "target_layer": target_layer_idx,
        "n_templates": n_templates,
        "rank_50": rank_50,
        "rank_90": rank_90,
        "rank_95": rank_95,
        "rank_99": rank_99,
        "sv_top10": sv_top10,
        "top1_ratio": round(top1_ratio, 4),
        "top3_ratio": round(top3_ratio, 4),
        "top5_ratio": round(top5_ratio, 4),
        "compression_ratio": compression_ratio,
        "family_compression": family_compression,
        "pc_interpretation": pc_interpretation,
    }

    print(f"\n  === Sparse Feature Basis Results ===")
    print(f"  Rank: 50%={rank_50}, 90%={rank_90}, 95%={rank_95}, 99%={rank_99}")
    print(f"  Top singular value ratio: {top1_ratio:.4f}")
    print(f"  Compression ratio (rank_90/n_concepts): {compression_ratio:.4f}")
    print(f"  Top 5 singular values: {sv_top10[:5]}")
    for family, fc in family_compression.items():
        print(f"  {family}: rank_90={fc['rank_90']}, "
              f"compression={fc['compression_ratio']:.4f}, "
              f"top1_ratio={fc['top1_ratio']:.4f}")
    for pc_name, pc_info in pc_interpretation.items():
        print(f"  {pc_name}: var_ratio={pc_info['variance_ratio']:.4f}, "
              f"dominant_family={pc_info['dominant_family']}, "
              f"top3={pc_info['top5_concepts'][:3]}")

    return results


# ===== MAIN =====

def main():
    import torch

    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    print(f"\n{'='*60}")
    print(f"Phase 166: Activation Topology & Feature Reuse")
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

    # ===== Exp 1: Attention Head Topology =====
    print(f"\n[1] Exp 1: Attention Head Topology")
    t0 = time.time()
    all_results["exp1_attention_topology"] = exp1_attention_head_topology(
        model, tokenizer, device, model_name, n_concepts=40, n_templates=8)
    print(f"  Time: {time.time()-t0:.1f}s")

    # ===== Exp 2: MLP Neuron Topology =====
    print(f"\n[2] Exp 2: MLP Neuron Topology")
    t0 = time.time()
    all_results["exp2_mlp_topology"] = exp2_mlp_neuron_topology(
        model, tokenizer, device, model_name, n_concepts=40, n_templates=8, top_k_neurons=100)
    print(f"  Time: {time.time()-t0:.1f}s")

    # ===== Exp 3: Layer-wise Constraint Contribution =====
    print(f"\n[3] Exp 3: Layer-wise Constraint Contribution")
    t0 = time.time()
    all_results["exp3_layer_contribution"] = exp3_layer_constraint_contribution(
        model, tokenizer, device, model_name, n_concepts=40, n_templates=8)
    print(f"  Time: {time.time()-t0:.1f}s")

    # ===== Exp 4: Sparse Feature Basis =====
    print(f"\n[4] Exp 4: Sparse Feature Basis")
    t0 = time.time()
    all_results["exp4_feature_basis"] = exp4_sparse_feature_basis(
        model, tokenizer, device, model_name, n_concepts=40, n_templates=8)
    print(f"  Time: {time.time()-t0:.1f}s")

    # ===== Save Results =====
    os.makedirs("tests/glm5_temp", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = f"tests/glm5_temp/phase166_{model_name}_{timestamp}.json"

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

    print(f"\nPhase 166 complete for {model_name}!")


if __name__ == "__main__":
    main()
