"""
Phase 85: Representation Dynamics — From Transport Graph to Algorithm Execution
================================================================================

Phase 84 critique identified 4 CRITICAL flaws:

1. Attention pattern ≠ computation graph
   - A_{ij} is just "transport weight" = traffic map
   - A_{ij}V_j is "what is actually transported" = information content
   - Same routing graph + different V → completely different computation
   - We studied traffic map, not algorithm execution

2. "Fixed-point attractor" is overclaimed
   - High stability is EXPECTED for autoregressive with fixed KV cache
   - Need: basin of attraction, perturbation recovery, phase transition, Lyapunov
   - Current evidence = baseline, not discovery

3. Routing manifold low-dim ≠ computation low-dim
   - Routing naturally constrained by: position, causal mask, BOS
   - Real computation complexity in: value subspace, MLP transforms, recursive rewriting
   - Low routing rank ≠ low computational complexity

4. Missing representation dynamics entirely
   - We studied: how information MOVES (routing)
   - We should study: how information TRANSFORMS (representation)
   - h^(l) → h^(l+1): compress, rewrite, bind, bifurcate, merge
   - THIS is where the algorithm lives

THIS PHASE: Shift from routing statistics to representation dynamics
=====================================================================

The real computation of a Transformer is NOT:
  - Which tokens attend to which other tokens (routing)
  - How focused the attention is (entropy)

The real computation IS:
  - What representation state is at each layer (h^(l))
  - How that state transforms from layer to layer (Δh = h^(l+1) - h^(l))
  - What information is actually carried (A_{ij}V_j, not A_{ij})
  - Whether representations form trajectories, attractors, bifurcations

Key Insight from Critique:
  "attention statistics" ≠ "algorithm execution"
  
  The question is not "where does information flow?"
  The question is "what happens to the representation at each step?"

Four Experiments:

A. Value Transport Analysis ★★★★★ (MOST CRITICAL)
   - Compute A_{ij}V_j for each head, not just A_{ij}
   - What information is actually being transported?
   - Compare: same routing → different value transport across tasks
   - This directly tests: "routing ≠ computation"

B. Representation Trajectory Field ★★★★★
   - Track h^(l) across layers for each token
   - Compute Δh^(l) = h^(l+1) - h^(l) (incremental rewriting)
   - Analyze: compression, bifurcation, merging of representation states
   - Does the representation manifold have geometric structure?

C. Causal Intervention ★★★★★
   - Ablate specific OV circuits (not just attention patterns)
   - Ablate specific residual stream subspaces
   - Measure: does reasoning collapse? Does retrieval fail?
   - This is the ONLY way to distinguish correlation from causation

D. Recursive Representation Rewriting ★★★★★
   - Track how a single token's representation evolves across layers
   - Decompose: how much of Δh comes from attention vs MLP vs residual?
   - Study: iterative refinement, abstraction, compression
   - This is where "reasoning" actually happens

Why this matters:
  If Transformer computation is truly representation dynamics,
  then the "language computation algebra" lives in the
  REPRESENTATION TRANSFORMATION RULES, not the routing rules.

  The mathematical structure may be:
  - A representation rewriting system (like lambda calculus)
  - A compression-expansion cycle across layers
  - A variable binding mechanism in specific subspaces

Usage:
  python ccml_phase85_representation_dynamics.py --exp a
  python ccml_phase85_representation_dynamics.py --exp b
  python ccml_phase85_representation_dynamics.py --exp c
  python ccml_phase85_representation_dynamics.py --exp d
  python ccml_phase85_representation_dynamics.py --exp all
"""

import torch
import numpy as np
import argparse
import time
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


def generate_prompts(task, n):
    prompts = []
    np.random.seed(42)
    
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
# Experiment A: Value Transport Analysis
# ============================================================
def exp_a_value_transport():
    """
    THE MOST CRITICAL EXPERIMENT: What information is actually transported?
    
    Phase 84 studied A_{ij} (routing weights).
    This experiment studies A_{ij}V_j (actual transported information).
    
    Key test: Can two tasks have the same routing but different computation?
    If yes → routing ≠ computation (Phase 84 critique confirmed).
    """
    print("=" * 70)
    print("EXPERIMENT A: Value Transport — What Information Is Actually Carried?")
    print("=" * 70)
    
    model = get_model()
    tasks = ["addition", "antonym", "capital", "translate_fr", "past_tense", "animal_sound"]
    n_samples = 30
    
    # ---- Part 1: A_{ij}V_j analysis ----
    print(f"\n--- Part 1: Attention Weights vs Transported Vectors ---")
    print(f"  Comparing: routing (A) vs content (AV) across tasks")
    
    for layer in [2, 6, 10]:
        print(f"\n  Layer {layer}:")
        
        task_av_vectors = {}  # task -> list of AV vectors [n_heads, d_model] per sample
        task_routing_features = {}  # task -> list of routing features [2*n_heads] per sample
        
        for task in tasks:
            prompts = generate_prompts(task, n_samples)
            av_list = []
            routing_feat_list = []
            
            for prompt in prompts[:15]:
                tokens = model.to_tokens(prompt)
                with torch.no_grad():
                    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
                
                # Attention pattern: [n_heads, seq_q, seq_k]
                pattern = cache[f'blocks.{layer}.attn.hook_pattern'][:, -1, :].detach()
                
                # Value vectors: z = pattern @ V = [n_heads, d_head]
                z = cache[f'blocks.{layer}.attn.hook_z'][-1, :, :].detach()  # [n_heads, d_head]
                
                # Output: attn_out = W_O @ z, but for per-head analysis:
                W_O = model.blocks[layer].attn.W_O.detach()  # [n_heads, d_head, d_model]
                
                # Per-head output: h_out_head = z[h] @ W_O[h]  (z is row vector, W_O is [d_head, d_model])
                # This is the ACTUALLY TRANSPORTED information
                head_outputs = []
                for h in range(z.shape[0]):
                    out_h = z[h] @ W_O[h]  # [d_model]
                    head_outputs.append(out_h)
                
                av_vectors = torch.stack(head_outputs)  # [n_heads, d_model]
                
                # For routing comparison: use entropy instead of raw pattern (avoids seq_len mismatch)
                eps = 1e-10
                routing_entropy = -torch.sum(pattern * torch.log(pattern + eps), dim=-1)  # [n_heads]
                # Also use max attention weight per head
                max_attn = pattern.max(dim=-1).values  # [n_heads]
                routing_features = torch.cat([routing_entropy, max_attn])  # [2*n_heads]
                
                av_list.append(av_vectors)
                routing_feat_list.append(routing_features)
            
            task_av_vectors[task] = torch.stack(av_list)  # [n_samples, n_heads, d_model]
            task_routing_features[task] = torch.stack(routing_feat_list)  # [n_samples, 2*n_heads]
        
        # ---- Compare routing similarity vs content similarity ----
        print(f"\n  === ROUTING (A) similarity vs CONTENT (AV) similarity ===")
        
        # Average AV vectors per task
        task_av_avg = {t: v.mean(0) for t, v in task_av_vectors.items()}  # [n_heads, d_model]
        task_routing_avg = {t: r.mean(0) for t, r in task_routing_features.items()}  # [2*n_heads]
        
        # Compute cross-task cosine for routing features (entropy + max_attn)
        display_tasks = tasks
        print(f"\n  Routing features (entropy + max_attn) cross-task cosine:")
        header = f"  {'':20s}" + "".join(f"{t[:10]:>11s}" for t in display_tasks)
        print(header)
        for t1 in display_tasks:
            row = f"  {t1:20s}"
            for t2 in display_tasks:
                v1 = task_routing_avg[t1]
                v2 = task_routing_avg[t2]
                cos = torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
                row += f"{cos:>11.4f}"
            print(row)
        
        # Compute cross-task cosine for content (AV)
        print(f"\n  Content (AV) cross-task cosine:")
        header = f"  {'':20s}" + "".join(f"{t[:10]:>11s}" for t in display_tasks)
        print(header)
        for t1 in display_tasks:
            row = f"  {t1:20s}"
            for t2 in display_tasks:
                v1 = task_av_avg[t1].flatten()
                v2 = task_av_avg[t2].flatten()
                cos = torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
                row += f"{cos:>11.4f}"
            print(row)
        
        # ---- Key comparison: routing-similar pairs vs content-similar pairs ----
        print(f"\n  === KEY TEST: Do similar routings produce similar content? ===")
        routing_cosines = []
        content_cosines = []
        for i, t1 in enumerate(display_tasks):
            for j, t2 in enumerate(display_tasks):
                if j > i:
                    r_cos = torch.nn.functional.cosine_similarity(
                        task_routing_avg[t1].unsqueeze(0),
                        task_routing_avg[t2].unsqueeze(0)
                    ).item()
                    c_cos = torch.nn.functional.cosine_similarity(
                        task_av_avg[t1].flatten().unsqueeze(0),
                        task_av_avg[t2].flatten().unsqueeze(0)
                    ).item()
                    routing_cosines.append(r_cos)
                    content_cosines.append(c_cos)
        
        print(f"  Routing cosine:  mean={np.mean(routing_cosines):.4f}, std={np.std(routing_cosines):.4f}")
        print(f"  Content cosine:  mean={np.mean(content_cosines):.4f}, std={np.std(content_cosines):.4f}")
        print(f"  Content std / Routing std = {np.std(content_cosines)/(np.std(routing_cosines)+1e-10):.2f}")
        
        if np.std(content_cosines) > np.std(routing_cosines) * 1.5:
            print(f"  *** CONTENT IS MORE TASK-SPECIFIC THAN ROUTING! ***")
            print(f"  *** Phase 84 critique CONFIRMED: routing ≠ computation ***")
        else:
            print(f"  Routing and content have similar task-specificity")
    
    # ---- Part 2: Per-head value analysis ----
    print(f"\n--- Part 2: Per-Head Value Analysis ---")
    print(f"  Which heads carry task-specific information in their VALUES?")
    
    for layer in [2, 6, 10]:
        print(f"\n  Layer {layer}:")
        
        # For each head, compute cross-task cosine of AV vectors
        for h in range(min(4, model.cfg.n_heads)):  # Show first 4 heads
            task_head_avs = {}
            for task in tasks:
                av = task_av_vectors[task][:, h, :]  # [n_samples, d_model]
                task_head_avs[task] = av.mean(0)  # [d_model]
            
            cross_cos = []
            for i, t1 in enumerate(tasks):
                for j, t2 in enumerate(tasks):
                    if j > i:
                        cos = torch.nn.functional.cosine_similarity(
                            task_head_avs[t1].unsqueeze(0),
                            task_head_avs[t2].unsqueeze(0)
                        ).item()
                        cross_cos.append(cos)
            
            print(f"    Head {h}: AV cross-task cosine = {np.mean(cross_cos):.4f} ± {np.std(cross_cos):.4f}")
    
    # ---- Part 3: OV circuit task-specificity ----
    print(f"\n--- Part 3: OV Circuit Task-Specificity ---")
    print(f"  The OV circuit (W_O @ W_V) is the 'semantic transformation'.")
    print(f"  Is it the same across tasks? Or does task-specific V routing make it different?")
    
    for layer in [6]:
        print(f"\n  Layer {layer}:")
        W_O = model.blocks[layer].attn.W_O.detach()  # [n_heads, d_head, d_model]
        W_V = model.blocks[layer].attn.W_V.detach()  # [n_heads, d_model, d_head]
        
        # OV circuit per head (FIXED across tasks)
        OV = torch.einsum('hod,hdm->hom', W_O, W_V)  # [n_heads, d_model, d_model]
        
        # The actual computation: OV @ x, where x depends on the task
        # So even with fixed OV, different inputs → different outputs
        
        # Compare: fixed OV transform vs actual per-task output
        for task in tasks[:3]:
            prompts = generate_prompts(task, 10)
            for prompt in prompts[:1]:
                tokens = model.to_tokens(prompt)
                with torch.no_grad():
                    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
                
                h_pre = cache[f'blocks.{layer}.hook_resid_pre'][-1].detach()
                h_mid = cache[f'blocks.{layer}.hook_resid_mid'][-1].detach()
                
                # Delta from attention = h_mid - h_pre
                delta_attn = h_mid - h_pre  # [d_model]
                
                # What fraction of delta_attn is explained by each head?
                per_head_deltas = []
                for h_idx in range(model.cfg.n_heads):
                    z_h = cache[f'blocks.{layer}.attn.hook_z'][-1, h_idx, :].detach()
                    head_out = z_h @ W_O[h_idx]  # [d_model]
                    per_head_deltas.append(head_out)
                
                per_head_deltas = torch.stack(per_head_deltas)  # [n_heads, d_model]
                
                # Top-3 contributing heads
                head_norms = per_head_deltas.norm(dim=-1)  # [n_heads]
                top_heads = head_norms.argsort(descending=True)[:3]
                
                print(f"    {task:15s}: top-3 heads by ||Δh|| = {top_heads.tolist()}, "
                      f"norms = {[f'{head_norms[h]:.2f}' for h in top_heads]}, "
                      f"total ||Δh_attn|| = {delta_attn.norm():.2f}")
    
    print("\n" + "=" * 70)
    print("EXPERIMENT A COMPLETE")
    print("=" * 70)


# ============================================================
# Experiment B: Representation Trajectory Field
# ============================================================
def exp_b_representation_trajectory():
    """
    Track h^(l) across layers for each token.
    
    The question is not "where does information flow?" (routing)
    The question is "what happens to the representation at each step?"
    
    Key analyses:
    1. Representation trajectory: h^(0) → h^(1) → ... → h^(L)
    2. Incremental rewriting: Δh^(l) = h^(l+1) - h^(l)
    3. Representation compression/expansion
    4. Bifurcation: does the trajectory split by task?
    """
    print("=" * 70)
    print("EXPERIMENT B: Representation Trajectory Field")
    print("=" * 70)
    
    model = get_model()
    tasks = ["addition", "antonym", "capital", "translate_fr", "past_tense", "animal_sound"]
    n_samples = 30
    
    # ---- Part 1: Representation trajectory across layers ----
    print(f"\n--- Part 1: Representation Trajectory Across Layers ---")
    
    for task in tasks:
        prompts = generate_prompts(task, n_samples)
        trajectories = []  # [n_samples, n_layers+1, d_model]
        
        for prompt in prompts[:15]:
            tokens = model.to_tokens(prompt)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
            
            # Collect residual stream at each layer
            h_trajectory = []
            # Initial embedding
            h_0 = cache['hook_embed'][-1].detach()  # [d_model]
            h_trajectory.append(h_0)
            
            # After each layer
            for layer in range(model.cfg.n_layers):
                h_l = cache[f'blocks.{layer}.hook_resid_post'][-1].detach()
                h_trajectory.append(h_l)
            
            trajectories.append(torch.stack(h_trajectory))  # [n_layers+1, d_model]
        
        trajectories = torch.stack(trajectories)  # [n_samples, n_layers+1, d_model]
        
        # ---- Analysis 1: Trajectory length (total distance traveled) ----
        # Total distance from embedding to final output
        h_start = trajectories[:, 0, :]  # [n_samples, d_model]
        h_end = trajectories[:, -1, :]   # [n_samples, d_model]
        total_distance = (h_end - h_start).norm(dim=-1).mean().item()
        
        # Per-step distance
        step_distances = []
        for l in range(trajectories.shape[1] - 1):
            d = (trajectories[:, l+1, :] - trajectories[:, l, :]).norm(dim=-1).mean().item()
            step_distances.append(d)
        
        print(f"\n  Task: {task}")
        print(f"    Total trajectory length: {total_distance:.2f}")
        print(f"    Per-step distances: [{', '.join(f'{d:.2f}' for d in step_distances)}]")
    
    # ---- Part 2: Cross-task representation divergence ----
    print(f"\n--- Part 2: Cross-Task Representation Divergence ---")
    print(f"  Does the representation trajectory bifurcate by task?")
    
    task_trajectories = {}
    for task in tasks:
        prompts = generate_prompts(task, 30)
        layer_representations = defaultdict(list)  # layer -> list of h vectors
        
        for prompt in prompts[:15]:
            tokens = model.to_tokens(prompt)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
            
            for layer in range(model.cfg.n_layers):
                h = cache[f'blocks.{layer}.hook_resid_post'][-1].detach()
                layer_representations[layer].append(h)
        
        # Average representation per layer
        task_trajectories[task] = {
            l: torch.stack(v).mean(0) for l, v in layer_representations.items()
        }
    
    # Cross-task cosine at each layer
    print(f"\n  Cross-task representation cosine by layer:")
    print(f"  {'Layer':>6s}  {'Mean_cos':>10s}  {'Min_cos':>10s}  {'Std_cos':>10s}  {'Divergence':>12s}")
    
    for layer in [0, 2, 4, 6, 8, 10]:
        cross_cos = []
        for i, t1 in enumerate(tasks):
            for j, t2 in enumerate(tasks):
                if j > i:
                    cos = torch.nn.functional.cosine_similarity(
                        task_trajectories[t1][layer].unsqueeze(0),
                        task_trajectories[t2][layer].unsqueeze(0)
                    ).item()
                    cross_cos.append(cos)
        
        # Divergence = 1 - cosine (how different are representations)
        mean_cos = np.mean(cross_cos)
        min_cos = np.min(cross_cos)
        std_cos = np.std(cross_cos)
        divergence = 1 - mean_cos
        
        print(f"  {layer:6d}  {mean_cos:10.4f}  {min_cos:10.4f}  {std_cos:10.4f}  {divergence:12.4f}")
    
    # ---- Part 3: Representation compression analysis ----
    print(f"\n--- Part 3: Representation Compression ---")
    print(f"  Does the representation compress into a lower-dimensional subspace?")
    
    for task in tasks[:3]:
        prompts = generate_prompts(task, 50)
        
        for layer in [0, 6, 11]:
            reps = []
            for prompt in prompts[:30]:
                tokens = model.to_tokens(prompt)
                with torch.no_grad():
                    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
                
                if layer == 0:
                    h = cache['hook_embed'][-1].detach()
                else:
                    h = cache[f'blocks.{layer-1}.hook_resid_post'][-1].detach()
                reps.append(h)
            
            reps = torch.stack(reps)  # [n_samples, d_model]
            
            # SVD to measure effective dimensionality
            U, S, Vt = torch.linalg.svd(reps - reps.mean(0), full_matrices=False)
            
            # Effective rank (number of SVs needed for 90% variance)
            cumvar = torch.cumsum(S**2, dim=0) / (S**2).sum()
            eff_rank_90 = np.searchsorted(cumvar.numpy(), 0.90) + 1
            eff_rank_99 = np.searchsorted(cumvar.numpy(), 0.99) + 1
            
            # Participation ratio (effective dimensionality)
            pr = (S.sum()**2) / (S**2).sum()
            
            if layer in [0, 6, 11]:
                print(f"  {task:15s} L{layer:2d}: eff_rank_90={eff_rank_90}, eff_rank_99={eff_rank_99}, "
                      f"participation_ratio={pr:.1f}, top-5 SVs={[round(s,2) for s in S[:5].tolist()]}")
    
    # ---- Part 4: Incremental rewriting Δh analysis ----
    print(f"\n--- Part 4: Incremental Rewriting Δh = h^(l+1) - h^(l) ---")
    print(f"  How much does each layer rewrite the representation?")
    
    for task in tasks:
        prompts = generate_prompts(task, 20)
        delta_norms = defaultdict(list)
        
        for prompt in prompts[:10]:
            tokens = model.to_tokens(prompt)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
            
            for layer in range(model.cfg.n_layers):
                h_pre = cache[f'blocks.{layer}.hook_resid_pre'][-1].detach()
                h_post = cache[f'blocks.{layer}.hook_resid_post'][-1].detach()
                
                delta = h_post - h_pre
                # Relative rewriting: ||Δh|| / ||h||
                rel_rewrite = delta.norm().item() / (h_pre.norm().item() + 1e-10)
                delta_norms[layer].append(rel_rewrite)
        
        # Average relative rewriting per layer
        avg_rewrites = [np.mean(delta_norms[l]) for l in range(model.cfg.n_layers)]
        print(f"  {task:15s}: " + ", ".join(f"L{l}={r:.3f}" for l, r in enumerate(avg_rewrites) if l % 2 == 0))
    
    print("\n" + "=" * 70)
    print("EXPERIMENT B COMPLETE")
    print("=" * 70)


# ============================================================
# Experiment C: Causal Intervention
# ============================================================
def exp_c_causal_intervention():
    """
    THE ONLY WAY to distinguish correlation from causation.
    
    Phase 84 found "universal routing hubs" — but are they CAUSALLY important?
    
    Interventions:
    1. Zero-ablate specific OV circuits (remove value transport, keep routing)
    2. Zero-ablate specific heads (remove both routing and value transport)
    3. Compare: does removing VALUE transport have a different effect than removing ROUTING?
    
    This directly tests: routing ≠ computation
    """
    print("=" * 70)
    print("EXPERIMENT C: Causal Intervention — Correlation vs Causation")
    print("=" * 70)
    
    model = get_model()
    tasks = ["addition", "antonym", "capital", "translate_fr", "past_tense", "animal_sound"]
    
    # ---- Part 1: OV circuit ablation ----
    print(f"\n--- Part 1: OV Circuit Ablation ---")
    print(f"  Remove VALUE transport (zero z) while keeping attention routing intact.")
    print(f"  This tests: is the VALUE more important than the ROUTING?")
    
    # Baseline: get predictions for each task
    def get_prediction(model, prompt):
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            logits = model(tokens)[0, -1]
            pred_token = logits.argmax().item()
            pred_str = model.to_string(pred_token)
        return pred_str, logits
    
    # Ablation function: zero out specific head's z vector
    def ablate_head_z(model, prompt, layer, head):
        """Zero out head's z vector (value transport) while keeping attention pattern."""
        def hook_fn(z, hook):
            z[:, :, head, :] = 0
            return z
        
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            model.reset_hooks()
            model.add_hook(f'blocks.{layer}.attn.hook_z', hook_fn)
            logits = model(tokens)[0, -1]
            model.reset_hooks()
        
        pred_token = logits.argmax().item()
        pred_str = model.to_string(pred_token)
        return pred_str, logits
    
    # Ablation function: zero out specific head's attention pattern
    def ablate_head_pattern(model, prompt, layer, head):
        """Replace head's attention with uniform distribution (destroy routing)."""
        def hook_fn(pattern, hook):
            n_pos = pattern.shape[-1]
            uniform = torch.ones_like(pattern[:, head, :, :]) / n_pos
            pattern[:, head, :, :] = uniform
            return pattern
        
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            model.reset_hooks()
            model.add_hook(f'blocks.{layer}.attn.hook_pattern', hook_fn)
            logits = model(tokens)[0, -1]
            model.reset_hooks()
        
        pred_token = logits.argmax().item()
        pred_str = model.to_string(pred_token)
        return pred_str, logits
    
    # ---- Test ablation on key routing hubs from Phase 84 ----
    hub_heads = [(4, 11), (5, 1), (7, 2)]  # Universal routing hubs from Phase 84
    
    for task in tasks:
        prompts = generate_prompts(task, 10)
        print(f"\n  Task: {task}")
        
        # Baseline predictions
        baseline_preds = []
        for prompt in prompts[:5]:
            pred, _ = get_prediction(model, prompt)
            baseline_preds.append(pred)
        print(f"    Baseline: {baseline_preds}")
        
        # Ablate each hub head's VALUE (z)
        for layer, head in hub_heads:
            ablated_preds = []
            for prompt in prompts[:5]:
                pred, _ = ablate_head_z(model, prompt, layer, head)
                ablated_preds.append(pred)
            
            # Count how many predictions changed
            n_changed = sum(1 for a, b in zip(baseline_preds, ablated_preds) if a != b)
            print(f"    Ablate VALUE (L{layer}, H{head}): {ablated_preds} ({n_changed}/5 changed)")
        
        # Compare: ablate ROUTING (uniform attention) for one hub
        for layer, head in hub_heads[:1]:
            ablated_preds = []
            for prompt in prompts[:5]:
                pred, _ = ablate_head_pattern(model, prompt, layer, head)
                ablated_preds.append(pred)
            
            n_changed = sum(1 for a, b in zip(baseline_preds, ablated_preds) if a != b)
            print(f"    Ablate ROUTING (L{layer}, H{head}): {ablated_preds} ({n_changed}/5 changed)")
    
    # ---- Part 2: Full head ablation vs value-only ablation ----
    print(f"\n--- Part 2: Full Ablation vs Value-Only Ablation ---")
    print(f"  Systematic comparison across all (layer, head) pairs.")
    
    # For efficiency, focus on one task and a selection of heads
    task = "capital"
    prompts = generate_prompts(task, 10)
    
    # Get baseline logit difference (correct - incorrect)
    def get_logit_diff(model, prompt, correct_token=" Paris"):
        tokens = model.to_tokens(prompt)
        correct_id = model.to_single_token(correct_token)
        with torch.no_grad():
            logits = model(tokens)[0, -1]
        return logits[correct_id].item()
    
    print(f"\n  Task: {task}")
    print(f"  Ablation impact on logit for correct answer:")
    print(f"  {'(L,H)':>8s}  {'Baseline':>10s}  {'Ablate_z':>10s}  {'Ablate_attn':>12s}  {'z_impact':>10s}  {'attn_impact':>12s}")
    
    # Test a selection of heads
    test_heads = [(l, h) for l in [4, 5, 6, 7] for h in range(12)]
    
    baseline_logit = np.mean([get_logit_diff(model, p) for p in prompts[:5]])
    
    for layer, head in test_heads:
        # Value ablation
        z_logits = []
        for prompt in prompts[:5]:
            tokens = model.to_tokens(prompt)
            correct_id = model.to_single_token(" Paris")
            with torch.no_grad():
                model.reset_hooks()
                def hook_z(z, hook, h=head):
                    z[:, :, h, :] = 0
                    return z
                model.add_hook(f'blocks.{layer}.attn.hook_z', hook_z)
                logits = model(tokens)[0, -1]
                z_logits.append(logits[correct_id].item())
                model.reset_hooks()
        z_logit = np.mean(z_logits)
        
        # Attention pattern ablation (uniform)
        attn_logits = []
        for prompt in prompts[:5]:
            tokens = model.to_tokens(prompt)
            correct_id = model.to_single_token(" Paris")
            with torch.no_grad():
                model.reset_hooks()
                def hook_attn(pattern, hook, h=head):
                    n_pos = pattern.shape[-1]
                    uniform = torch.ones_like(pattern[:, h, :, :]) / n_pos
                    pattern[:, h, :, :] = uniform
                    return pattern
                model.add_hook(f'blocks.{layer}.attn.hook_pattern', hook_attn)
                logits = model(tokens)[0, -1]
                attn_logits.append(logits[correct_id].item())
                model.reset_hooks()
        attn_logit = np.mean(attn_logits)
        
        z_impact = baseline_logit - z_logit
        attn_impact = baseline_logit - attn_logit
        
        if abs(z_impact) > 0.5 or abs(attn_impact) > 0.5:  # Only show significant effects
            print(f"  ({layer},{head:2d})  {baseline_logit:10.3f}  {z_logit:10.3f}  {attn_logit:12.3f}  {z_impact:10.3f}  {attn_impact:12.3f}")
    
    # ---- Part 3: Perturbation recovery test ----
    print(f"\n--- Part 3: Perturbation Recovery (Attractor Test) ---")
    print(f"  Phase 84 claimed 'fixed-point attractor' without proper evidence.")
    print(f"  Real attractor: perturb state → system returns to same state.")
    
    # Test: add noise to residual stream at specific layer → does the model recover?
    for noise_level in [0.01, 0.1, 1.0]:
        recovery_scores = []
        
        for task in ["addition", "capital"]:
            prompts = generate_prompts(task, 10)
            
            for prompt in prompts[:5]:
                tokens = model.to_tokens(prompt)
                
                # Baseline prediction
                with torch.no_grad():
                    baseline_logits = model(tokens)[0, -1]
                    baseline_pred = baseline_logits.argmax().item()
                
                # Perturbed prediction: add noise to residual stream at layer 6
                def hook_noise(h, hook):
                    noise = torch.randn_like(h) * noise_level
                    return h + noise
                
                with torch.no_grad():
                    model.reset_hooks()
                    model.add_hook('blocks.6.hook_resid_mid', hook_noise)
                    perturbed_logits = model(tokens)[0, -1]
                    model.reset_hooks()
                
                # Recovery: does the perturbed model predict the same token?
                perturbed_pred = perturbed_logits.argmax().item()
                recovery_scores.append(1.0 if baseline_pred == perturbed_pred else 0.0)
        
        recovery_rate = np.mean(recovery_scores)
        print(f"  Noise level {noise_level}: recovery rate = {recovery_rate:.2f} "
              f"({'strong attractor' if recovery_rate > 0.8 else 'weak attractor' if recovery_rate > 0.5 else 'no attractor'})")
    
    print("\n" + "=" * 70)
    print("EXPERIMENT C COMPLETE")
    print("=" * 70)


# ============================================================
# Experiment D: Recursive Representation Rewriting
# ============================================================
def exp_d_recursive_rewriting():
    """
    Track how a single token's representation evolves across layers.
    
    The REAL question: what does each layer DO to the representation?
    Not: where does information come from? (routing)
    But: what transformation is applied? (computation)
    
    Decompose Δh = h^(l+1) - h^(l) into:
    - Δh_attn = contribution from attention (value transport)
    - Δh_mlp = contribution from MLP (nonlinear transformation)
    - The residual stream carries the "state" that gets iteratively rewritten
    """
    print("=" * 70)
    print("EXPERIMENT D: Recursive Representation Rewriting")
    print("=" * 70)
    
    model = get_model()
    tasks = ["addition", "antonym", "capital", "translate_fr", "past_tense", "animal_sound"]
    n_samples = 30
    
    # ---- Part 1: Attention vs MLP contribution to representation rewriting ----
    print(f"\n--- Part 1: Attention vs MLP Contribution to Δh ---")
    print(f"  Decompose: h^(l+1) - h^(l) = Δh_attn + Δh_mlp")
    
    for task in tasks:
        prompts = generate_prompts(task, n_samples)
        attn_deltas = defaultdict(list)  # layer -> list of ||Δh_attn||
        mlp_deltas = defaultdict(list)    # layer -> list of ||Δh_mlp||
        total_deltas = defaultdict(list)  # layer -> list of ||Δh_total||
        
        for prompt in prompts[:15]:
            tokens = model.to_tokens(prompt)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
            
            for layer in range(model.cfg.n_layers):
                h_pre = cache[f'blocks.{layer}.hook_resid_pre'][-1].detach()
                h_mid = cache[f'blocks.{layer}.hook_resid_mid'][-1].detach()
                h_post = cache[f'blocks.{layer}.hook_resid_post'][-1].detach()
                
                delta_attn = (h_mid - h_pre).norm().item()
                delta_mlp = (h_post - h_mid).norm().item()
                delta_total = (h_post - h_pre).norm().item()
                
                attn_deltas[layer].append(delta_attn)
                mlp_deltas[layer].append(delta_mlp)
                total_deltas[layer].append(delta_total)
        
        # Average per layer
        avg_attn = [np.mean(attn_deltas[l]) for l in range(model.cfg.n_layers)]
        avg_mlp = [np.mean(mlp_deltas[l]) for l in range(model.cfg.n_layers)]
        avg_total = [np.mean(total_deltas[l]) for l in range(model.cfg.n_layers)]
        
        print(f"\n  Task: {task}")
        print(f"  {'Layer':>6s}  {'||Δ_attn||':>12s}  {'||Δ_mlp||':>12s}  {'||Δ_total||':>12s}  {'Attn%':>8s}  {'MLP%':>8s}")
        for layer in [0, 2, 4, 6, 8, 10]:
            a = avg_attn[layer]
            m = avg_mlp[layer]
            t = avg_total[layer]
            print(f"  {layer:6d}  {a:12.3f}  {m:12.3f}  {t:12.3f}  {a/(t+1e-10)*100:8.1f}  {m/(t+1e-10)*100:8.1f}")
    
    # ---- Part 2: Representation rewriting direction analysis ----
    print(f"\n--- Part 2: Representation Rewriting Direction ---")
    print(f"  Not just HOW MUCH is rewritten, but IN WHICH DIRECTION.")
    print(f"  Key question: does Δh point in a consistent direction across samples?")
    
    for task in tasks[:3]:
        prompts = generate_prompts(task, 50)
        
        for layer in [4, 6, 8]:
            attn_deltas = []
            mlp_deltas = []
            
            for prompt in prompts[:30]:
                tokens = model.to_tokens(prompt)
                with torch.no_grad():
                    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
                
                h_pre = cache[f'blocks.{layer}.hook_resid_pre'][-1].detach()
                h_mid = cache[f'blocks.{layer}.hook_resid_mid'][-1].detach()
                h_post = cache[f'blocks.{layer}.hook_resid_post'][-1].detach()
                
                attn_deltas.append(h_mid - h_pre)  # [d_model]
                mlp_deltas.append(h_post - h_mid)    # [d_model]
            
            attn_deltas = torch.stack(attn_deltas)  # [n_samples, d_model]
            mlp_deltas = torch.stack(mlp_deltas)    # [n_samples, d_model]
            
            # Compute consistency: average pairwise cosine of Δh vectors
            # This measures: do all samples rewrite in the same direction?
            attn_mean = attn_deltas.mean(0)
            mlp_mean = mlp_deltas.mean(0)
            
            # Cosine of each sample's delta with the mean delta
            attn_consistency = torch.nn.functional.cosine_similarity(
                attn_deltas, attn_mean.unsqueeze(0).expand_as(attn_deltas), dim=-1
            ).mean().item()
            
            mlp_consistency = torch.nn.functional.cosine_similarity(
                mlp_deltas, mlp_mean.unsqueeze(0).expand_as(mlp_deltas), dim=-1
            ).mean().item()
            
            print(f"  {task:15s} L{layer}: attn_direction_consistency={attn_consistency:.4f}, "
                  f"mlp_direction_consistency={mlp_consistency:.4f}")
    
    # ---- Part 3: Cross-task rewriting direction divergence ----
    print(f"\n--- Part 3: Cross-Task Rewriting Direction Divergence ---")
    print(f"  Do different tasks rewrite the representation in different directions?")
    
    for layer in [2, 6, 10]:
        task_attn_directions = {}
        task_mlp_directions = {}
        
        for task in tasks:
            prompts = generate_prompts(task, 50)
            attn_deltas = []
            mlp_deltas = []
            
            for prompt in prompts[:30]:
                tokens = model.to_tokens(prompt)
                with torch.no_grad():
                    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
                
                h_pre = cache[f'blocks.{layer}.hook_resid_pre'][-1].detach()
                h_mid = cache[f'blocks.{layer}.hook_resid_mid'][-1].detach()
                h_post = cache[f'blocks.{layer}.hook_resid_post'][-1].detach()
                
                attn_deltas.append(h_mid - h_pre)
                mlp_deltas.append(h_post - h_mid)
            
            task_attn_directions[task] = torch.stack(attn_deltas).mean(0)  # [d_model]
            task_mlp_directions[task] = torch.stack(mlp_deltas).mean(0)    # [d_model]
        
        # Cross-task cosine of rewriting directions
        print(f"\n  Layer {layer}: Cross-task Δh_attn direction cosine")
        display = tasks
        header = f"  {'':20s}" + "".join(f"{t[:10]:>11s}" for t in display)
        print(header)
        for t1 in display:
            row = f"  {t1:20s}"
            for t2 in display:
                cos = torch.nn.functional.cosine_similarity(
                    task_attn_directions[t1].unsqueeze(0),
                    task_attn_directions[t2].unsqueeze(0)
                ).item()
                row += f"{cos:>11.4f}"
            print(row)
        
        print(f"\n  Layer {layer}: Cross-task Δh_mlp direction cosine")
        header = f"  {'':20s}" + "".join(f"{t[:10]:>11s}" for t in display)
        print(header)
        for t1 in display:
            row = f"  {t1:20s}"
            for t2 in display:
                cos = torch.nn.functional.cosine_similarity(
                    task_mlp_directions[t1].unsqueeze(0),
                    task_mlp_directions[t2].unsqueeze(0)
                ).item()
                row += f"{cos:>11.4f}"
            print(row)
    
    # ---- Part 4: Subspace analysis of rewriting ----
    print(f"\n--- Part 4: Subspace Analysis of Representation Rewriting ---")
    print(f"  In which subspaces does rewriting occur?")
    
    for layer in [4, 6, 8]:
        # Collect Δh vectors across all tasks
        all_attn_deltas = []
        all_mlp_deltas = []
        task_labels = []
        
        for task in tasks:
            prompts = generate_prompts(task, 20)
            for prompt in prompts[:10]:
                tokens = model.to_tokens(prompt)
                with torch.no_grad():
                    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
                
                h_pre = cache[f'blocks.{layer}.hook_resid_pre'][-1].detach()
                h_mid = cache[f'blocks.{layer}.hook_resid_mid'][-1].detach()
                h_post = cache[f'blocks.{layer}.hook_resid_post'][-1].detach()
                
                all_attn_deltas.append(h_mid - h_pre)
                all_mlp_deltas.append(h_post - h_mid)
                task_labels.append(task)
        
        attn_deltas = torch.stack(all_attn_deltas)  # [N, d_model]
        mlp_deltas = torch.stack(all_mlp_deltas)    # [N, d_model]
        
        # SVD of Δh matrix to find principal rewriting directions
        attn_centered = attn_deltas - attn_deltas.mean(0)
        mlp_centered = mlp_deltas - mlp_deltas.mean(0)
        
        U_a, S_a, _ = torch.linalg.svd(attn_centered, full_matrices=False)
        U_m, S_m, _ = torch.linalg.svd(mlp_centered, full_matrices=False)
        
        print(f"\n  Layer {layer}:")
        print(f"    Attention Δh SVD top-5: {[round(s,2) for s in S_a[:5].tolist()]}")
        print(f"    MLP Δh SVD top-5: {[round(s,2) for s in S_m[:5].tolist()]}")
        
        # Participation ratio (effective dimensionality of rewriting)
        pr_attn = (S_a.sum()**2) / (S_a**2).sum()
        pr_mlp = (S_m.sum()**2) / (S_m**2).sum()
        
        print(f"    Participation ratio: attn={pr_attn:.1f}, mlp={pr_mlp:.1f}")
        
        # Cross-task divergence in principal rewriting directions
        # Project each task's Δh onto the principal components and compare
        cumvar_a = torch.cumsum(S_a**2, dim=0) / (S_a**2).sum()
        cumvar_m = torch.cumsum(S_m**2, dim=0) / (S_m**2).sum()
        
        for k in [1, 3, 5, 10]:
            print(f"    Variance explained by k={k}: attn={cumvar_a[k-1]:.4f}, mlp={cumvar_m[k-1]:.4f}")
    
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
        exp_a_value_transport()
    if args.exp in ["b", "all"]:
        exp_b_representation_trajectory()
    if args.exp in ["c", "all"]:
        exp_c_causal_intervention()
    if args.exp in ["d", "all"]:
        exp_d_recursive_rewriting()

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")
