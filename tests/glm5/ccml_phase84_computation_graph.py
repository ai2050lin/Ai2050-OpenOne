"""
Phase 84: Computation Graph Dynamics — From Jacobian Geometry to Algorithmic Execution
========================================================================================

Phase 83 critique identified 4 CRITICAL flaws:

1. Jacobian = local tangent geometry ≠ actual computation
   - Jacobian: "if I perturb input ε, output changes by Jε"
   - Actual computation: token selection → routing → KV retrieval → state recomposition
   - These are DISCRETE STRUCTURAL CHANGES, not infinitesimal perturbations

2. Attention Jacobian oversimplified: d→d instead of (T×d)→(T×d)
   - Token-token coupling, causal masking, multi-head composition all missing

3. Operator Basis with n=4 tasks is pseudo-low-rank (mathematical tautology)
   - 4 samples → SVD gives ≤4 components. NOT a discovery.

4. No recursive rollout analysis
   - Single-layer local operator ≠ recursive computation
   - Reasoning/CoT/planning all require multi-step rollout

THIS PHASE: Shift from Jacobian geometry to Computation Graph Dynamics
========================================================================

The real computation of a Transformer is NOT:
  - A dense matrix J that maps h_in → h_out
  - A set of basis operators A_i with coefficients c_i

The real computation IS:
  - A SPARSE ROUTING GRAPH: which tokens → which heads → which information paths
  - A MEMORY RETRIEVAL SEQUENCE: what KV pairs are accessed, in what order
  - A RECURSIVE STATE UPDATE: how the graph evolves across layers and across generation steps

Key Insight from Critique:
  "operator geometry" ≠ "algorithmic execution"
  
  The question is not "what is the local derivative?"
  The question is "what is the actual information flow?"

Four Experiments:

A. Attention Routing Graph ★★★★★ (MOST FUNDAMENTAL)
   - For each (token_position, head): what are the top-k source tokens?
   - This reveals the ACTUAL information flow, not the Jacobian
   - Key metrics: routing entropy, routing diversity, task-specific routing patterns
   - Compare across tasks: do different tasks use different routing graphs?

B. Head Specialization & Task-Conditioned Routing ★★★★★
   - Which heads are "active" (low entropy = focused) vs "diffuse" (high entropy)?
   - Are certain heads task-specific? Or is task-specificity distributed?
   - Head clustering: do heads form functional groups?
   - USE 10+ TASKS to avoid n=4 trap

C. Cross-Layer Information Path ★★★★★
   - Track information flow from input tokens through layers to output
   - Path = sequence of (layer, head, source_token) triplets
   - This is the actual "algorithm trace", not the Jacobian
   - How many distinct paths exist? Are some paths task-specific?

D. Recursive Rollout: Graph Evolution During Generation ★★★★★
   - Generate tokens autoregressively
   - Track how the routing graph changes at each step
   - G_t → token_t → G_(t+1)
   - This addresses: CoT, planning, reasoning dynamics
   - Key question: does the graph stabilize? diverge? oscillate?

Why this matters:
  If Transformer computation is truly a dynamic sparse graph,
  then the "language computation algebra" is NOT in individual operators,
  but in the GRAPH EVOLUTION RULES.

  The mathematical structure may be:
  - A routing automaton (finite graph transitions)
  - A memory access protocol (which KV entries, in what order)
  - A recursive graph grammar (how G_t transforms to G_{t+1})

Usage:
  python ccml_phase84_computation_graph.py --exp a
  python ccml_phase84_computation_graph.py --exp b
  python ccml_phase84_computation_graph.py --exp c
  python ccml_phase84_computation_graph.py --exp d
  python ccml_phase84_computation_graph.py --exp all
"""

import torch
import numpy as np
import argparse
import time
import json
from collections import defaultdict
from transformer_lens import HookedTransformer


def get_model():
    model = HookedTransformer.from_pretrained(
        "gpt2-small",
        center_unembed=False,
        center_writing_weights=False,
        fold_ln=False,
        device="cpu",
    )
    model.eval()
    return model


# ============================================================
# Extended Task Set (10+ tasks to avoid n=4 pseudo-low-rank)
# ============================================================
def generate_prompts(task, n):
    prompts = []
    np.random.seed(42)
    
    # Arithmetic tasks
    if task == "addition":
        for i in range(n):
            a, b = np.random.randint(1, 50), np.random.randint(1, 50)
            prompts.append(f"{a} + {b} =")
    elif task == "subtraction":
        for i in range(n):
            a = np.random.randint(10, 99)
            b = np.random.randint(1, a)
            prompts.append(f"{a} - {b} =")
    elif task == "multiplication":
        for i in range(n):
            a, b = np.random.randint(2, 12), np.random.randint(2, 12)
            prompts.append(f"{a} * {b} =")
    
    # Language tasks
    elif task == "antonym":
        words = ["hot", "big", "fast", "happy", "strong", "light", "good", "old",
                 "rich", "tall", "love", "warm", "clean", "safe", "hard", "loud",
                 "bright", "sharp", "sweet", "brave"]
        for i in range(n):
            w = words[i % len(words)]
            prompts.append(f"The opposite of {w} is")
    elif task == "synonym":
        words = [("big", "large"), ("small", "tiny"), ("fast", "quick"), ("smart", "clever"),
                 ("happy", "glad"), ("sad", "upset"), ("strong", "powerful"), ("weak", "feeble"),
                 ("pretty", "beautiful"), ("ugly", "hideous"), ("kind", "gentle"), ("cruel", "mean"),
                 ("rich", "wealthy"), ("poor", "needy"), ("brave", "courageous"), ("scared", "afraid"),
                 ("loud", "noisy"), ("quiet", "silent"), ("bright", "shining"), ("dark", "dim")]
        for i in range(n):
            w, _ = words[i % len(words)]
            prompts.append(f"Another word for {w} is")
    elif task == "past_tense":
        verbs = ["walk", "talk", "jump", "play", "smile", "laugh", "cry", "run",
                 "think", "speak", "write", "read", "sing", "dance", "cook", "clean",
                 "paint", "draw", "sleep", "wake"]
        for i in range(n):
            v = verbs[i % len(verbs)]
            prompts.append(f"The past tense of {v} is")
    elif task == "plural":
        nouns = ["cat", "dog", "house", "tree", "book", "car", "bird", "fish",
                 "chair", "table", "phone", "lamp", "ring", "cup", "plate",
                 "shirt", "shoe", "hat", "sock", "fork"]
        for i in range(n):
            w = nouns[i % len(nouns)]
            prompts.append(f"The plural of {w} is")
    
    # Knowledge tasks
    elif task == "capital":
        countries = ["France", "Germany", "Japan", "Brazil", "Italy",
                     "Spain", "China", "India", "Egypt", "Australia",
                     "Canada", "Mexico", "Korea", "Russia", "Turkey",
                     "Norway", "Sweden", "Poland", "Greece", "Thailand"]
        for i in range(n):
            c = countries[i % len(countries)]
            prompts.append(f"The capital of {c} is")
    elif task == "country":
        cities = ["Paris", "Berlin", "Tokyo", "Brasilia", "Rome",
                  "Madrid", "Beijing", "Delhi", "Cairo", "Canberra",
                  "Ottawa", "Mexico City", "Seoul", "Moscow", "Ankara",
                  "Oslo", "Stockholm", "Warsaw", "Athens", "Bangkok"]
        for i in range(n):
            c = cities[i % len(cities)]
            prompts.append(f"The country of {c} is")
    elif task == "translate_fr":
        words = ["cat", "dog", "house", "water", "book", "tree", "sun", "moon",
                 "fire", "earth", "heart", "hand", "night", "day", "star",
                 "flower", "bird", "fish", "rain", "snow"]
        for i in range(n):
            w = words[i % len(words)]
            prompts.append(f"The French word for {w} is")
    elif task == "translate_es":
        words = ["cat", "dog", "house", "water", "book", "tree", "sun", "moon",
                 "fire", "earth", "heart", "hand", "night", "day", "star",
                 "flower", "bird", "fish", "rain", "snow"]
        for i in range(n):
            w = words[i % len(words)]
            prompts.append(f"The Spanish word for {w} is")
    elif task == "animal_sound":
        animals = [("cat", "meow"), ("dog", "bark"), ("cow", "moo"), ("pig", "oink"),
                   ("duck", "quack"), ("chicken", "cluck"), ("horse", "neigh"),
                   ("sheep", "baa"), ("frog", "ribbit"), ("lion", "roar"),
                   ("wolf", "howl"), ("snake", "hiss"), ("bee", "buzz"),
                   ("owl", "hoot"), ("crow", "caw")]
        for i in range(n):
            a, _ = animals[i % len(animals)]
            prompts.append(f"The sound a {a} makes is")
    
    return prompts


ALL_TASKS = ["addition", "subtraction", "multiplication", 
             "antonym", "synonym", "past_tense", "plural",
             "capital", "country", "translate_fr", "translate_es", "animal_sound"]


# ============================================================
# Core Analysis Functions
# ============================================================
def compute_routing_graph(model, prompt, layer=None):
    """
    Compute the attention routing graph for a prompt.
    
    Returns: dict with keys:
      - 'pattern': [n_layers, n_heads, seq_q, seq_k] attention patterns
      - 'tokens': list of token strings
      - 'entropy': [n_layers, n_heads, seq_q] routing entropy per (layer, head, query_pos)
      - 'top_sources': [n_layers, n_heads, seq_q, top_k] top-k source positions per (layer, head, query_pos)
    """
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
    
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    
    layers_to_analyze = [layer] if layer is not None else range(n_layers)
    
    result = {
        'pattern': {},
        'tokens': model.to_str_tokens(tokens),
        'entropy': {},
        'top_sources': {},
    }
    
    for l in layers_to_analyze:
        pattern = cache[f'blocks.{l}.attn.hook_pattern']  # [n_heads, seq_q, seq_k]
        result['pattern'][l] = pattern.detach()
        
        # Routing entropy: H = -Σ α_s log α_s
        # Low entropy = focused routing (specific token), High entropy = diffuse
        eps = 1e-10
        log_pattern = torch.log(pattern + eps)
        entropy = -torch.sum(pattern * log_pattern, dim=-1)  # [n_heads, seq_q]
        result['entropy'][l] = entropy.detach()
        
        # Top-k source positions for each (head, query_pos)
        top_k = 5
        top_sources = torch.topk(pattern, k=min(top_k, pattern.shape[-1]), dim=-1)
        result['top_sources'][l] = {
            'indices': top_sources.indices.detach(),  # [n_heads, seq_q, top_k]
            'values': top_sources.values.detach(),    # [n_heads, seq_q, top_k]
        }
    
    return result


def routing_graph_fingerprint(routing, layer, pos=-1):
    """
    Create a fingerprint of the routing graph at a specific layer and position.
    
    This captures: for each head, which source tokens carry the most weight.
    This is the ACTUAL computation path, not the Jacobian.
    """
    top_sources = routing['top_sources'][layer]
    n_heads = top_sources['indices'].shape[0]
    
    # For each head: which source positions are attended to?
    head_fingerprints = []
    for h in range(n_heads):
        src_idx = top_sources['indices'][h, pos, :].tolist()  # top-k source positions
        src_val = top_sources['values'][h, pos, :].tolist()   # top-k attention weights
        head_fingerprints.append({
            'head': h,
            'source_positions': src_idx,
            'source_weights': src_val,
            'entropy': routing['entropy'][layer][h, pos].item()
        })
    
    return head_fingerprints


# ============================================================
# Experiment A: Attention Routing Graph
# ============================================================
def exp_a_routing_graph():
    """
    THE CORE EXPERIMENT: What is the actual information flow?
    
    Not: "what is the Jacobian?" (local differential geometry)
    But: "which tokens feed information to which other tokens?" (actual computation)
    
    Key questions:
    1. How focused is routing? (entropy distribution)
    2. Do different tasks produce different routing graphs?
    3. Is routing determined by syntactic position or semantic content?
    4. How does routing vary across layers?
    """
    print("=" * 70)
    print("EXPERIMENT A: Attention Routing Graph — Actual Information Flow")
    print("=" * 70)
    
    model = get_model()
    tasks = ALL_TASKS  # 12 tasks to avoid n=4 trap
    layers = [0, 2, 4, 6, 8, 10]  # More layers for comprehensive view
    n_samples = 50  # Large sample for reliability
    
    # Collect routing graphs
    print("\n--- Collecting routing graphs ---")
    all_routing = {}  # (task, layer) -> routing data
    
    for task in tasks:
        prompts = generate_prompts(task, n_samples)
        for layer in layers:
            entropies = []
            top_source_distributions = []  # For analyzing "which positions are attended to"
            
            for prompt in prompts:
                routing = compute_routing_graph(model, prompt, layer=layer)
                entropies.append(routing['entropy'][layer])  # [n_heads, seq_q]
                
                # For last position, collect top source positions
                top_src = routing['top_sources'][layer]
                # Normalize source positions by sequence length (relative position)
                seq_len = top_src['indices'].shape[1]
                # Collect absolute source positions for last query token
                src_pos = top_src['indices'][:, -1, :].numpy()  # [n_heads, top_k]
                src_wt = top_src['values'][:, -1, :].numpy()    # [n_heads, top_k]
                top_source_distributions.append({
                    'positions': src_pos,
                    'weights': src_wt,
                    'seq_len': seq_len
                })
            
            all_routing[(task, layer)] = {
                'entropies': entropies,
                'source_dists': top_source_distributions
            }
            print(f"  ({task}, L{layer}): done")
    
    # ---- Analysis 1: Routing entropy distribution ----
    print(f"\n{'='*60}")
    print("ANALYSIS 1: Routing Entropy Distribution")
    print("  Low entropy = focused routing (specific token selected)")
    print("  High entropy = diffuse routing (averaging many tokens)")
    print(f"{'='*60}")
    
    for layer in layers:
        print(f"\n  Layer {layer}:")
        print(f"  {'Task':20s} {'Mean_H':>8s} {'Std_H':>8s} {'%Focused':>10s} {'%Diffuse':>10s}")
        print(f"  {'-'*58}")
        
        for task in tasks:
            entropies = all_routing[(task, layer)]['entropies']
            # Extract last position entropy (most relevant for prediction)
            # Each entry is [n_heads, seq_q] with variable seq_q
            h_last_list = [e[:, -1] for e in entropies]  # [n_heads] each
            h_last = torch.stack(h_last_list)  # [n_samples, n_heads]
            mean_h = h_last.mean().item()
            std_h = h_last.std().item()
            
            # Classify heads as "focused" (H < 1.0) or "diffuse" (H > 2.0)
            max_h = np.log(h_last.shape[1])  # Maximum possible entropy
            pct_focused = (h_last < 1.0).float().mean().item() * 100
            pct_diffuse = (h_last > 2.0).float().mean().item() * 100
            
            print(f"  {task:20s} {mean_h:8.3f} {std_h:8.3f} {pct_focused:10.1f} {pct_diffuse:10.1f}")
    
    # ---- Analysis 2: Task-conditioned routing divergence ----
    print(f"\n{'='*60}")
    print("ANALYSIS 2: Task-Conditioned Routing Divergence")
    print("  Do different tasks produce different routing graphs?")
    print(f"{'='*60}")
    
    for layer in [2, 6, 10]:
        print(f"\n  Layer {layer}: Cross-task routing divergence")
        
        # For each task, compute the average routing pattern at last position
        task_avg_patterns = {}
        for task in tasks:
            source_dists = all_routing[(task, layer)]['source_dists']
            # Average the source position weights across samples
            # We'll compare which relative positions get attention
            n_samples_actual = len(source_dists)
            # Normalize positions by seq_len → relative position [0, 1]
            avg_weight_by_relpos = defaultdict(float)
            count_by_relpos = defaultdict(int)
            for sd in source_dists:
                seq_len = sd['seq_len']
                for h in range(sd['positions'].shape[0]):  # n_heads
                    for k in range(sd['positions'].shape[1]):  # top_k
                        rel_pos = sd['positions'][h, k] / seq_len
                        weight = sd['weights'][h, k]
                        # Discretize relative position
                        bucket = int(rel_pos * 10)  # 10 buckets
                        avg_weight_by_relpos[bucket] += weight
                        count_by_relpos[bucket] += 1
            
            for bucket in avg_weight_by_relpos:
                avg_weight_by_relpos[bucket] /= count_by_relpos[bucket]
            
            task_avg_patterns[task] = avg_weight_by_relpos
        
        # Compare task routing patterns
        # Use cosine similarity of the weight-by-position vectors
        all_buckets = sorted(set().union(*[set(p.keys()) for p in task_avg_patterns.values()]))
        
        # Pick representative tasks for display
        display_tasks = ["addition", "antonym", "capital", "translate_fr", "past_tense", "animal_sound"]
        
        print(f"\n  Cross-task routing pattern similarity (cosine):")
        header = f"  {'':20s}" + "".join(f"{t[:10]:>11s}" for t in display_tasks)
        print(header)
        
        for t1 in display_tasks:
            v1 = np.array([task_avg_patterns[t1].get(b, 0) for b in all_buckets])
            row = f"  {t1:20s}"
            for t2 in display_tasks:
                v2 = np.array([task_avg_patterns[t2].get(b, 0) for b in all_buckets])
                if v1.sum() > 0 and v2.sum() > 0:
                    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
                else:
                    cos = 0.0
                row += f"{cos:>11.4f}"
            print(row)
    
    # ---- Analysis 3: Head-level routing specificity ----
    print(f"\n{'='*60}")
    print("ANALYSIS 3: Head-Level Routing Specificity")
    print("  Which heads are task-specific? Which are universal?")
    print(f"{'='*60}")
    
    for layer in [2, 6, 10]:
        print(f"\n  Layer {layer}:")
        
        # For each head, compute the cross-task entropy variance
        # High variance = task-specific head, Low variance = universal head
        head_entropy_by_task = {}
        for task in tasks:
            entropies = all_routing[(task, layer)]['entropies']
            h_last_list = [e[:, -1] for e in entropies]  # [n_heads] each
            h_last = torch.stack(h_last_list)  # [n_samples, n_heads]
            head_entropy_by_task[task] = h_last.mean(dim=0)  # [n_heads]
        
        # Compute per-head variance across tasks
        task_entropy_matrix = torch.stack([head_entropy_by_task[t] for t in tasks])  # [n_tasks, n_heads]
        per_head_variance = task_entropy_matrix.var(dim=0)  # [n_heads]
        per_head_mean = task_entropy_matrix.mean(dim=0)  # [n_heads]
        
        # Classify heads
        var_threshold = per_head_variance.median().item()
        print(f"  Head classification (variance threshold = {var_threshold:.4f}):")
        
        universal_heads = (per_head_variance < var_threshold).nonzero().squeeze().tolist()
        specific_heads = (per_head_variance >= var_threshold).nonzero().squeeze().tolist()
        
        if isinstance(universal_heads, int):
            universal_heads = [universal_heads]
        if isinstance(specific_heads, int):
            specific_heads = [specific_heads]
        
        print(f"    Universal heads (low cross-task variance): {universal_heads}")
        print(f"    Task-specific heads (high cross-task variance): {specific_heads}")
        print(f"    Fraction task-specific: {len(specific_heads)}/{len(specific_heads)+len(universal_heads)}")
        
        # Show per-head entropy for task-specific heads
        if len(specific_heads) > 0:
            print(f"\n    Task-specific head details:")
            for h in specific_heads[:5]:  # Show top 5
                row = f"      Head {h}: mean_H={per_head_mean[h]:.3f}"
                for task in tasks:
                    row += f", {task[:8]}={head_entropy_by_task[task][h]:.2f}"
                print(row)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT A COMPLETE")
    print("=" * 70)


# ============================================================
# Experiment B: Head Specialization & Task-Conditioned Routing
# ============================================================
def exp_b_head_specialization():
    """
    Key questions:
    1. Are there "functional head groups" (e.g., copy heads, induction heads, etc.)?
    2. How many heads are task-specific vs universal?
    3. Does task-specificity increase with layer depth?
    4. With 12 tasks, what is the TRUE dimensionality of the operator space?
    """
    print("=" * 70)
    print("EXPERIMENT B: Head Specialization & Task-Conditioned Routing")
    print("=" * 70)
    
    model = get_model()
    tasks = ALL_TASKS  # 12 tasks
    layers = [0, 2, 4, 6, 8, 10]
    n_samples = 50
    
    # ---- Part 1: Head activation profiles ----
    print(f"\n--- Part 1: Head Activation Profiles ---")
    print("  For each head, compute its 'routing fingerprint' across tasks")
    
    # For each (layer, head, task): compute average routing pattern
    # This is a vector of [top_source_relative_positions, attention_weights]
    head_profiles = {}  # (layer, head) -> profile matrix [n_tasks, feature_dim]
    
    for layer in [2, 6, 10]:
        print(f"\n  Layer {layer}:")
        
        for task in tasks:
            prompts = generate_prompts(task, n_samples)
            for p_idx, prompt in enumerate(prompts[:20]):
                routing = compute_routing_graph(model, prompt, layer=layer)
                
                if p_idx == 0:
                    # Initialize accumulators
                    if (layer, task) not in head_profiles:
                        head_profiles[(layer, task)] = []
                
                # For each head: extract routing features
                top_src = routing['top_sources'][layer]
                n_heads = top_src['indices'].shape[0]
                seq_len = top_src['indices'].shape[1]
                
                # Feature: for each head, the attention weight distribution over
                # relative positions (binned into 10 buckets)
                head_features = np.zeros((n_heads, 10))  # [n_heads, 10 relative position buckets]
                
                for h in range(n_heads):
                    src_pos = top_src['indices'][h, -1, :].numpy()  # top-k source positions for last query
                    src_wt = top_src['values'][h, -1, :].numpy()
                    for k_idx in range(len(src_pos)):
                        rel_pos = src_pos[k_idx] / seq_len
                        bucket = min(int(rel_pos * 10), 9)
                        head_features[h, bucket] += src_wt[k_idx]
                
                if (layer, task) not in head_profiles:
                    head_profiles[(layer, task)] = []
                head_profiles[(layer, task)].append(head_features)
        
        # Average across samples
        for task in tasks:
            if (layer, task) in head_profiles:
                head_profiles[(layer, task)] = np.mean(head_profiles[(layer, task)], axis=0)
    
    # ---- Part 2: Head clustering ----
    print(f"\n--- Part 2: Head Clustering ---")
    print("  Do heads form functional groups?")
    
    for layer in [2, 6, 10]:
        # Create head feature matrix: [n_tasks * n_heads, feature_dim]
        feature_list = []
        head_labels = []
        
        for task in tasks:
            if (layer, task) in head_profiles:
                features = head_profiles[(layer, task)]  # [n_heads, 10]
                for h in range(features.shape[0]):
                    feature_list.append(features[h])
                    head_labels.append(f"{task[:6]}_h{h}")
        
        feature_matrix = np.array(feature_list)  # [n_tasks * n_heads, 10]
        
        # Cluster heads based on their routing profiles
        # Use SVD to find the principal routing patterns
        from numpy.linalg import svd
        U, S, Vt = svd(feature_matrix, full_matrices=False)
        
        print(f"\n  Layer {layer}: Routing pattern SVD")
        print(f"  Singular values: {S[:10].round(4).tolist()}")
        cumvar = np.cumsum(S**2) / np.sum(S**2)
        print(f"  Cumulative variance explained:")
        for k in [1, 2, 3, 5, 8, 12]:
            if k <= len(cumvar):
                print(f"    k={k}: {cumvar[k-1]:.4f}")
        
        # How many distinct routing patterns?
        n_significant = np.sum(S > 0.1 * S[0])
        print(f"  Number of significant routing patterns (SV > 10% of max): {n_significant}")
        
        # Project heads onto first 2 components for visualization
        proj = feature_matrix @ Vt[:2].T  # [N, 2]
        print(f"  2D projection range: x=[{proj[:,0].min():.2f}, {proj[:,0].max():.2f}], "
              f"y=[{proj[:,1].min():.2f}, {proj[:,1].max():.2f}]")
    
    # ---- Part 3: TRUE operator dimensionality with 12 tasks ----
    print(f"\n--- Part 3: TRUE Operator Dimensionality (12 tasks) ---")
    print("  With 12 tasks, the pseudo-low-rank concern from Phase 83 is addressed.")
    print("  If we still find low effective rank, it's real, not an artifact of n=4.")
    
    # Compute the effective rank of the attention routing pattern matrix
    for layer in [2, 6, 10]:
        # Create task-routing matrix: [n_tasks, n_heads * feature_dim]
        task_routing_matrix = []
        for task in tasks:
            if (layer, task) in head_profiles:
                # Flatten all head features into one vector
                task_routing_matrix.append(head_profiles[(layer, task)].flatten())
        
        task_routing_matrix = np.array(task_routing_matrix)  # [n_tasks, n_heads * 10]
        
        U, S, Vt = svd(task_routing_matrix, full_matrices=False)
        
        print(f"\n  Layer {layer}: Task-Routing Matrix SVD")
        print(f"  Shape: {task_routing_matrix.shape}")
        print(f"  Singular values: {S.round(4).tolist()}")
        cumvar = np.cumsum(S**2) / np.sum(S**2)
        print(f"  Cumulative variance:")
        for k in [1, 2, 3, 4, 6, 8, 12]:
            if k <= len(cumvar):
                print(f"    k={k}: {cumvar[k-1]:.4f}")
        
        # This is the critical test:
        # Phase 83 found k=3 explains 91-96% with 4 tasks
        # With 12 tasks, does this still hold?
        n_for_90 = np.searchsorted(cumvar, 0.90) + 1
        n_for_95 = np.searchsorted(cumvar, 0.95) + 1
        n_for_99 = np.searchsorted(cumvar, 0.99) + 1
        print(f"  Components needed: 90%={n_for_90}, 95%={n_for_95}, 99%={n_for_99}")
        
        if n_for_90 <= 4:
            print(f"  *** LOW RANK CONFIRMED with 12 tasks! ***")
        else:
            print(f"  *** Phase 83's low-rank was an artifact of n=4! ***")
    
    # ---- Part 4: Task-specificity of routing increases with depth ----
    print(f"\n--- Part 4: Task-Specificity vs Layer Depth ---")
    
    for layer in layers:
        # Measure task-specificity: how different are routing patterns across tasks?
        task_patterns = []
        for task in tasks:
            if (layer, task) in head_profiles:
                task_patterns.append(head_profiles[(layer, task)].flatten())
        
        if len(task_patterns) < 2:
            continue
        
        task_matrix = np.array(task_patterns)  # [n_tasks, feature_dim]
        mean_pattern = task_matrix.mean(axis=0)
        
        # Task-specificity = average deviation from mean
        deviations = np.linalg.norm(task_matrix - mean_pattern, axis=1)
        specificity = deviations.mean() / (np.linalg.norm(mean_pattern) + 1e-10)
        
        print(f"  Layer {layer}: routing specificity = {specificity:.4f}")
    
    print("\n" + "=" * 70)
    print("EXPERIMENT B COMPLETE")
    print("=" * 70)


# ============================================================
# Experiment C: Cross-Layer Information Path
# ============================================================
def exp_c_information_path():
    """
    Track the actual information flow from input to output.
    
    Not: "what is the Jacobian at each layer?"
    But: "what path does information actually take through the network?"
    
    A path = sequence of (layer, head, source_token) triplets
    that carry information from input tokens to the output prediction.
    """
    print("=" * 70)
    print("EXPERIMENT C: Cross-Layer Information Path — Algorithm Trace")
    print("=" * 70)
    
    model = get_model()
    tasks = ["addition", "antonym", "capital", "translate_fr", "past_tense", "animal_sound"]
    n_samples = 30
    
    # For each task, trace information flow through layers
    print(f"\n--- Information Flow Trace ---")
    
    for task in tasks:
        prompts = generate_prompts(task, n_samples)
        
        # Collect routing at all layers for multiple samples
        sample_paths = []
        
        for prompt in prompts[:15]:
            tokens = model.to_tokens(prompt)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
            
            str_tokens = model.to_str_tokens(tokens)
            seq_len = len(str_tokens)
            
            # For each layer, find the most important source token for the LAST position
            path = []  # List of (layer, head, source_token_idx, source_token_str, weight)
            
            for layer in range(model.cfg.n_layers):
                pattern = cache[f'blocks.{layer}.attn.hook_pattern']  # [n_heads, seq_q, seq_k]
                
                # For the last query position
                pattern_last = pattern[:, -1, :]  # [n_heads, seq_k]
                
                # Find the dominant head (highest max attention weight)
                max_attn_per_head = pattern_last.max(dim=-1).values  # [n_heads]
                dominant_head = max_attn_per_head.argmax().item()
                
                # For the dominant head, find the top source token
                top_src_idx = pattern_last[dominant_head].argmax().item()
                top_src_weight = pattern_last[dominant_head, top_src_idx].item()
                
                # Also find top-3 sources for this head
                top3_vals, top3_idx = pattern_last[dominant_head].topk(3)
                
                path.append({
                    'layer': layer,
                    'dominant_head': dominant_head,
                    'top_source_idx': top_src_idx,
                    'top_source_token': str_tokens[top_src_idx] if top_src_idx < len(str_tokens) else '<unk>',
                    'top_source_weight': top_src_weight,
                    'top3_sources': [(str_tokens[i] if i < len(str_tokens) else '<unk>', v.item()) 
                                    for i, v in zip(top3_idx, top3_vals)],
                    'all_heads_top_src': pattern_last.argmax(dim=-1).tolist(),  # top source for each head
                })
            
            sample_paths.append(path)
        
        # ---- Analyze paths for this task ----
        print(f"\n  Task: {task}")
        print(f"  {'='*50}")
        
        # For each layer, what fraction of the time does the dominant head attend to:
        # 1. The last token (self-attention)
        # 2. The first token (BOS or start of pattern)
        # 3. Some content token
        layer_attention_targets = defaultdict(lambda: defaultdict(int))
        
        for path in sample_paths:
            for step in path:
                layer = step['layer']
                src_idx = step['top_source_idx']
                seq_len_approx = max(step['top_source_idx'] + 1, 10)  # rough estimate
                
                if src_idx == 0:
                    layer_attention_targets[layer]['first_token'] += 1
                elif src_idx >= len(step['top3_sources']) - 1:
                    layer_attention_targets[layer]['last_token'] += 1
                else:
                    layer_attention_targets[layer]['content_token'] += 1
        
        print(f"\n  Dominant source token type by layer:")
        print(f"  {'Layer':>6s} {'First':>8s} {'Last':>8s} {'Content':>8s}")
        for layer in [0, 2, 4, 6, 8, 10]:
            targets = layer_attention_targets[layer]
            total = sum(targets.values()) + 1e-10
            print(f"  {layer:6d} {targets['first_token']/total*100:8.1f}% "
                  f"{targets['last_token']/total*100:8.1f}% "
                  f"{targets['content_token']/total*100:8.1f}%")
        
        # Track information flow: which heads are consistently important?
        head_importance = defaultdict(float)
        for path in sample_paths:
            for step in path:
                head_importance[(step['layer'], step['dominant_head'])] += step['top_source_weight']
        
        # Top-10 most important (layer, head) pairs
        sorted_heads = sorted(head_importance.items(), key=lambda x: -x[1])[:10]
        print(f"\n  Top-10 most important (layer, head) pairs:")
        for (l, h), imp in sorted_heads:
            print(f"    Layer {l}, Head {h}: total_weight={imp:.2f}")
    
    # ---- Cross-task comparison: information path divergence ----
    print(f"\n{'='*60}")
    print("Cross-Task Information Path Divergence")
    print(f"{'='*60}")
    
    # For each pair of tasks, compare which (layer, head) pairs are important
    task_head_importance = {}
    for task in tasks:
        prompts = generate_prompts(task, 30)
        head_imp = defaultdict(float)
        
        for prompt in prompts[:15]:
            tokens = model.to_tokens(prompt)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
            
            for layer in range(model.cfg.n_layers):
                pattern = cache[f'blocks.{layer}.attn.hook_pattern'][:, -1, :]  # [n_heads, seq_k]
                # Max attention weight per head
                max_attn = pattern.max(dim=-1).values  # [n_heads]
                for h in range(max_attn.shape[0]):
                    head_imp[(layer, h)] += max_attn[h].item()
        
        # Normalize
        total = sum(head_imp.values())
        for k in head_imp:
            head_imp[k] /= total
        task_head_importance[task] = head_imp
    
    # Compare pairs
    print(f"\n  Cross-task head usage similarity (cosine):")
    display_tasks = ["addition", "antonym", "capital", "translate_fr", "past_tense", "animal_sound"]
    header = f"  {'':20s}" + "".join(f"{t[:10]:>11s}" for t in display_tasks)
    print(header)
    
    for t1 in display_tasks:
        row = f"  {t1:20s}"
        for t2 in display_tasks:
            # Create vectors from head importance
            all_keys = sorted(set(list(task_head_importance[t1].keys()) + list(task_head_importance[t2].keys())))
            v1 = np.array([task_head_importance[t1].get(k, 0) for k in all_keys])
            v2 = np.array([task_head_importance[t2].get(k, 0) for k in all_keys])
            cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            row += f"{cos:>11.4f}"
        print(row)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT C COMPLETE")
    print("=" * 70)


# ============================================================
# Experiment D: Recursive Rollout — Graph Evolution During Generation
# ============================================================
def exp_d_recursive_rollout():
    """
    THE MOST IMPORTANT MISSING PIECE:
    How does the computation graph evolve during autoregressive generation?
    
    This is NOT about single-step Jacobians.
    This is about RECURSIVE DYNAMICS:
    
    G_t → token_t → G_{t+1}
    
    Key questions:
    1. Does the routing graph stabilize? Diverge? Oscillate?
    2. What is the "attractor structure" of the routing dynamics?
    3. Does reasoning/CoT correspond to a specific graph evolution pattern?
    """
    print("=" * 70)
    print("EXPERIMENT D: Recursive Rollout — Graph Evolution During Generation")
    print("=" * 70)
    
    model = get_model()
    tasks = ["addition", "antonym", "capital", "translate_fr", "past_tense", "animal_sound"]
    n_generate = 5  # Number of tokens to generate
    n_samples = 20
    
    # ---- Part 1: Graph evolution during generation ----
    print(f"\n--- Part 1: Graph Evolution During Generation ---")
    
    for task in tasks:
        prompts = generate_prompts(task, n_samples)
        
        all_graphs = []  # [n_samples, n_generate+1] routing graphs
        
        for prompt in prompts[:10]:
            tokens = model.to_tokens(prompt)
            graphs = []
            
            # Initial routing graph (before generation)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
            
            # Extract routing graph at layer 6 (middle layer)
            target_layer = 6
            pattern = cache[f'blocks.{target_layer}.attn.hook_pattern'][:, -1, :].detach()
            
            # Graph fingerprint: attention weights from last position to all source positions
            # Average across heads
            avg_pattern = pattern.mean(dim=0).numpy()  # [seq_k]
            graphs.append(avg_pattern.copy())
            
            # Generate n_generate tokens
            current_tokens = tokens.clone()
            for gen_step in range(n_generate):
                with torch.no_grad():
                    next_token = model(current_tokens)[0, -1].argmax()
                    current_tokens = torch.cat([current_tokens, next_token.unsqueeze(0).unsqueeze(0)], dim=-1)
                    _, cache = model.run_with_cache(current_tokens, remove_batch_dim=True)
                
                pattern = cache[f'blocks.{target_layer}.attn.hook_pattern'][:, -1, :].detach()
                avg_pattern = pattern.mean(dim=0).numpy()
                graphs.append(avg_pattern.copy())
            
            all_graphs.append(graphs)
        
        # Analyze graph evolution
        print(f"\n  Task: {task}")
        
        # Compute graph stability: cosine between consecutive graphs
        stability_scores = []
        for graphs in all_graphs:
            for i in range(len(graphs) - 1):
                g1 = graphs[i]
                g2 = graphs[i+1]
                # Pad to same length
                max_len = max(len(g1), len(g2))
                g1_padded = np.zeros(max_len)
                g2_padded = np.zeros(max_len)
                g1_padded[:len(g1)] = g1
                g2_padded[:len(g2)] = g2
                
                cos = np.dot(g1_padded, g2_padded) / (np.linalg.norm(g1_padded) * np.linalg.norm(g2_padded) + 1e-10)
                stability_scores.append(cos)
        
        print(f"    Graph stability (cosine between consecutive steps): {np.mean(stability_scores):.4f} ± {np.std(stability_scores):.4f}")
        
        # Compute graph drift: cosine between initial and step-t
        drift_scores = []
        for graphs in all_graphs:
            g0 = graphs[0]
            for i in range(1, len(graphs)):
                gt = graphs[i]
                max_len = max(len(g0), len(gt))
                g0_padded = np.zeros(max_len)
                gt_padded = np.zeros(max_len)
                g0_padded[:len(g0)] = g0
                gt_padded[:len(gt)] = gt
                
                cos = np.dot(g0_padded, gt_padded) / (np.linalg.norm(g0_padded) * np.linalg.norm(gt_padded) + 1e-10)
                drift_scores.append((i, cos))
        
        # Average drift by step
        for step in range(1, n_generate + 1):
            step_scores = [cos for s, cos in drift_scores if s == step]
            if step_scores:
                print(f"    Drift from initial (step {step}): {np.mean(step_scores):.4f}")
    
    # ---- Part 2: Routing entropy dynamics during generation ----
    print(f"\n--- Part 2: Routing Entropy Dynamics During Generation ---")
    
    for task in tasks:
        prompts = generate_prompts(task, n_samples)
        
        entropy_trajectories = []  # [n_samples, n_generate+1]
        
        for prompt in prompts[:10]:
            tokens = model.to_tokens(prompt)
            entropies = []
            
            # Initial
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
            
            pattern = cache[f'blocks.{target_layer}.attn.hook_pattern'][:, -1, :].detach()
            eps = 1e-10
            h = -torch.sum(pattern * torch.log(pattern + eps), dim=-1).mean().item()
            entropies.append(h)
            
            # Generate
            current_tokens = tokens.clone()
            for gen_step in range(n_generate):
                with torch.no_grad():
                    next_token = model(current_tokens)[0, -1].argmax()
                    current_tokens = torch.cat([current_tokens, next_token.unsqueeze(0).unsqueeze(0)], dim=-1)
                    _, cache = model.run_with_cache(current_tokens, remove_batch_dim=True)
                
                pattern = cache[f'blocks.{target_layer}.attn.hook_pattern'][:, -1, :].detach()
                h = -torch.sum(pattern * torch.log(pattern + eps), dim=-1).mean().item()
                entropies.append(h)
            
            entropy_trajectories.append(entropies)
        
        # Average trajectory
        avg_trajectory = np.mean(entropy_trajectories, axis=0)
        std_trajectory = np.std(entropy_trajectories, axis=0)
        
        print(f"\n  Task: {task}")
        print(f"  Entropy trajectory: ", end="")
        for i, (m, s) in enumerate(zip(avg_trajectory, std_trajectory)):
            print(f"step{i}={m:.3f}±{s:.3f}", end="  ")
        print()
    
    # ---- Part 3: Information path persistence during generation ----
    print(f"\n--- Part 3: Information Path Persistence ---")
    print("  Does the same (layer, head) stay dominant across generation steps?")
    
    for task in tasks:
        prompts = generate_prompts(task, n_samples)
        
        head_persistence = defaultdict(int)  # (layer, head) -> count of steps as dominant
        head_changes = []  # number of dominant-head changes per generation sequence
        
        for prompt in prompts[:15]:
            tokens = model.to_tokens(prompt)
            dominant_heads = []
            
            # Initial
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
            
            for layer in [4, 6, 8]:
                pattern = cache[f'blocks.{layer}.attn.hook_pattern'][:, -1, :].detach()
                max_attn = pattern.max(dim=-1).values
                dom_head = max_attn.argmax().item()
                dominant_heads.append((layer, dom_head))
            
            # Generate
            current_tokens = tokens.clone()
            for gen_step in range(n_generate):
                with torch.no_grad():
                    next_token = model(current_tokens)[0, -1].argmax()
                    current_tokens = torch.cat([current_tokens, next_token.unsqueeze(0).unsqueeze(0)], dim=-1)
                    _, cache = model.run_with_cache(current_tokens, remove_batch_dim=True)
                
                current_heads = []
                for layer in [4, 6, 8]:
                    pattern = cache[f'blocks.{layer}.attn.hook_pattern'][:, -1, :].detach()
                    max_attn = pattern.max(dim=-1).values
                    dom_head = max_attn.argmax().item()
                    current_heads.append((layer, dom_head))
                    head_persistence[(layer, dom_head)] += 1
                
                # Count changes
                changes = sum(1 for a, b in zip(dominant_heads, current_heads) if a != b)
                head_changes.append(changes)
                
                dominant_heads = current_heads
        
        avg_changes = np.mean(head_changes) if head_changes else 0
        print(f"  Task: {task:20s} avg dominant-head changes per step: {avg_changes:.2f}")
    
    # ---- Part 4: Recursive graph evolution rules ----
    print(f"\n--- Part 4: Recursive Graph Evolution Rules ---")
    print("  KEY QUESTION: Is there a finite set of graph transition rules?")
    
    # Discretize routing graphs and look for repeated transition patterns
    for task in ["addition", "antonym", "capital"]:
        prompts = generate_prompts(task, 30)
        
        # For each sample, collect the sequence of routing graph "states"
        # State = which head is dominant at layers 4, 6, 8
        state_sequences = []
        
        for prompt in prompts[:15]:
            tokens = model.to_tokens(prompt)
            states = []
            
            # Initial state
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
            
            state = tuple()
            for layer in [4, 6, 8]:
                pattern = cache[f'blocks.{layer}.attn.hook_pattern'][:, -1, :].detach()
                max_attn = pattern.max(dim=-1).values
                dom_head = max_attn.argmax().item()
                state += (dom_head,)
            states.append(state)
            
            # Generate
            current_tokens = tokens.clone()
            for gen_step in range(n_generate):
                with torch.no_grad():
                    next_token = model(current_tokens)[0, -1].argmax()
                    current_tokens = torch.cat([current_tokens, next_token.unsqueeze(0).unsqueeze(0)], dim=-1)
                    _, cache = model.run_with_cache(current_tokens, remove_batch_dim=True)
                
                state = tuple()
                for layer in [4, 6, 8]:
                    pattern = cache[f'blocks.{layer}.attn.hook_pattern'][:, -1, :].detach()
                    max_attn = pattern.max(dim=-1).values
                    dom_head = max_attn.argmax().item()
                    state += (dom_head,)
                states.append(state)
            
            state_sequences.append(states)
        
        # Count unique states and transitions
        unique_states = set()
        transitions = defaultdict(int)
        
        for seq in state_sequences:
            for i in range(len(seq)):
                unique_states.add(seq[i])
                if i < len(seq) - 1:
                    transitions[(seq[i], seq[i+1])] += 1
        
        print(f"\n  Task: {task}")
        print(f"    Unique states: {len(unique_states)}")
        print(f"    Unique transitions: {len(transitions)}")
        print(f"    Total state visits: {sum(len(s) for s in state_sequences)}")
        
        # Most common transitions
        sorted_trans = sorted(transitions.items(), key=lambda x: -x[1])[:5]
        print(f"    Top-5 transitions:")
        for (s1, s2), count in sorted_trans:
            print(f"      {s1} → {s2}: {count}")
    
    print("\n" + "=" * 70)
    print("EXPERIMENT D COMPLETE")
    print("=" * 70)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True,
                       choices=["a", "b", "c", "d", "all"])
    args = parser.parse_args()

    start = time.time()
    if args.exp in ["a", "all"]:
        exp_a_routing_graph()
    if args.exp in ["b", "all"]:
        exp_b_head_specialization()
    if args.exp in ["c", "all"]:
        exp_c_information_path()
    if args.exp in ["d", "all"]:
        exp_d_recursive_rollout()

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")
