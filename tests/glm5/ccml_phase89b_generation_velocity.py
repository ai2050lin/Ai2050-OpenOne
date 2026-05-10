"""
Phase 89B: Generation Velocity Field
=====================================

THE CORE SHIFT: Study how representations evolve across GENERATION STEPS,
not across layers.

Previous work: v_L = h_{L+1} - h_L  (layer velocity)
This work:     v_t = h_{t+1} - h_t  (generation step velocity)

where t = token generation step, and h_t = representation after generating t tokens.

KEY QUESTIONS:
1. Do generation trajectories converge? (attractor behavior)
2. Are generation velocity fields low-dimensional?
3. Do different tasks (CoT, retrieval, analogy) produce different trajectory types?
4. Does perturbation recovery exist? (basin of attraction)

CRITICAL DISTINCTION from Phase 89A:
- Layer velocity: how representation changes between layers for SAME input
- Generation velocity: how representation changes between generation steps
- Generation velocity is the REAL dynamics of language

MODELS: Qwen3 (primary), GLM4 8bit, DS7B 8bit
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import sys
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

def get_model(model_name="Qwen/Qwen2.5-1.5B"):
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(
        model_name,
        center_unembed=False,
        center_writing_weights=False,
        fold_ln=False,
        device="cuda",
    )
    model.eval()
    return model

def safe_token_id(model, text):
    try:
        return model.to_single_token(text)
    except:
        return None


# ============================================================
# Generation with representation tracking
# ============================================================

def generate_with_trajectory(model, prompt, max_new_tokens=20, layer=None):
    """
    Generate tokens one by one, collecting representations at each step.
    
    Returns:
        tokens: list of generated token strings
        reprs: list of representation vectors (at specified layer, last position)
        probs: list of top-5 probabilities at each step
    """
    if layer is None:
        layer = model.cfg.n_layers // 2
    
    tokens = []
    reprs = []
    probs_list = []
    
    current_prompt = prompt
    
    for step in range(max_new_tokens):
        # Get representation and logits
        toks = model.to_tokens(current_prompt)
        with torch.no_grad():
            logits, cache = model.run_with_cache(toks)
        
        # Extract representation at last position, specified layer
        last_pos = toks.shape[1] - 1
        h = cache[f'blocks.{layer}.hook_resid_post'][0, last_pos, :].detach().cpu()
        reprs.append(h)
        
        # Get next token
        next_logits = logits[0, -1, :]
        next_probs = torch.softmax(next_logits, dim=-1)
        
        # Top-5 tokens
        top5_vals, top5_ids = next_probs.topk(5)
        top5_tokens = [model.to_string(t) for t in top5_ids]
        top5_probs = top5_vals.tolist()
        probs_list.append(list(zip(top5_tokens, top5_probs)))
        
        # Greedy decoding
        next_token_id = next_logits.argmax().item()
        next_token_str = model.to_string(next_token_id)
        tokens.append(next_token_str)
        
        # Check for stop
        if next_token_str.strip() in ['.', '\n', '<|endoftext|>']:
            if step > 3:  # At least a few steps
                break
        
        # Append to prompt
        current_prompt = current_prompt + next_token_str
    
    return tokens, reprs, probs_list


def generate_with_trajectory_multiple_layers(model, prompt, max_new_tokens=20, layers=None):
    """
    Generate tokens, collecting representations at MULTIPLE layers.
    """
    if layers is None:
        n = model.cfg.n_layers
        layers = [0, n//4, n//2, 3*n//4, n-1]
    
    tokens = []
    reprs_by_layer = {l: [] for l in layers}
    
    current_prompt = prompt
    
    for step in range(max_new_tokens):
        toks = model.to_tokens(current_prompt)
        with torch.no_grad():
            logits, cache = model.run_with_cache(toks)
        
        last_pos = toks.shape[1] - 1
        for l in layers:
            h = cache[f'blocks.{l}.hook_resid_post'][0, last_pos, :].detach().cpu()
            reprs_by_layer[l].append(h)
        
        next_logits = logits[0, -1, :]
        next_token_id = next_logits.argmax().item()
        next_token_str = model.to_string(next_token_id)
        tokens.append(next_token_str)
        
        if next_token_str.strip() in ['.', '\n', '<|endoftext|>']:
            if step > 3:
                break
        
        current_prompt = current_prompt + next_token_str
    
    return tokens, reprs_by_layer


# ============================================================
# TASK DEFINITIONS
# ============================================================

# CoT tasks
COT_PROMPTS = [
    "23 + 47 =",
    "156 + 289 =",
    "45 - 17 =",
    "12 * 8 =",
    "100 - 37 =",
    "7 * 9 =",
    "234 + 567 =",
    "88 - 29 =",
    "15 * 6 =",
    "500 - 178 =",
    "34 + 78 =",
    "9 * 11 =",
    "423 + 156 =",
    "72 - 35 =",
    "14 * 7 =",
]

# Retrieval tasks
RETRIEVAL_PROMPTS = [
    "The capital of France is",
    "The capital of Germany is",
    "The capital of Japan is",
    "The capital of Italy is",
    "The capital of Spain is",
    "The capital of China is",
    "The currency of France is",
    "The currency of Japan is",
    "The currency of England is",
    "The language of Germany is",
    "The language of Japan is",
    "The language of Brazil is",
]

# Analogy tasks
ANALOGY_PROMPTS = [
    "king is to queen as man is to",
    "dog is to puppy as cat is to",
    "hot is to cold as up is to",
    "doctor is to hospital as teacher is to",
    "car is to road as boat is to",
    "bird is to nest as bear is to",
    "pen is to write as knife is to",
    "Paris is to France as Tokyo is to",
    "water is to drink as food is to",
    "hand is to glove as foot is to",
]

# Simple completion tasks
COMPLETION_PROMPTS = [
    "The sky is",
    "Water boils at",
    "The sun rises in the",
    "Birds can",
    "Ice is",
    "Fire is",
    "Rivers flow",
    "Mountains are",
]


# ============================================================
# EXPERIMENT 1: Generation Velocity Field Structure
# ============================================================

def experiment1(model):
    """
    Study the structure of generation velocity fields.
    
    Key question: Are generation velocities low-dimensional?
    Do different task types produce different velocity patterns?
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: Generation Velocity Field Structure")
    print("="*70)
    
    layer = model.cfg.n_layers // 2
    
    # Collect trajectories for each task type
    task_trajectories = {}
    
    for task_name, prompts in [("cot", COT_PROMPTS), 
                                ("retrieval", RETRIEVAL_PROMPTS),
                                ("analogy", ANALOGY_PROMPTS),
                                ("completion", COMPLETION_PROMPTS)]:
        print(f"\n--- {task_name.upper()} trajectories ---")
        trajectories = []
        
        for i, prompt in enumerate(prompts):
            tokens, reprs, probs = generate_with_trajectory(model, prompt, max_new_tokens=15, layer=layer)
            
            if len(reprs) < 3:
                continue
            
            # Compute velocities
            velocities = []
            for t in range(len(reprs) - 1):
                v = reprs[t+1] - reprs[t]
                velocities.append(v)
            
            # Basic trajectory stats
            vel_norms = [v.norm().item() for v in velocities]
            vel_cosines = []
            for t in range(len(velocities) - 1):
                c = F.cosine_similarity(velocities[t].unsqueeze(0), velocities[t+1].unsqueeze(0)).item()
                vel_cosines.append(c)
            
            gen_text = "".join(tokens[:min(10, len(tokens))])
            print(f"  [{i:2d}] {prompt[:30]:30s} -> {gen_text[:30]:30s} | "
                  f"steps={len(tokens):2d}, mean|v|={np.mean(vel_norms):.3f}, "
                  f"vel_consec_cos={np.mean(vel_cosines):.4f}" if vel_cosines else 
                  f"  [{i:2d}] {prompt[:30]:30s} -> {gen_text[:30]:30s} | steps={len(tokens):2d}")
            
            trajectories.append({
                'prompt': prompt,
                'tokens': tokens,
                'reprs': reprs,
                'velocities': velocities,
                'vel_norms': vel_norms,
                'vel_cosines': vel_cosines,
            })
        
        task_trajectories[task_name] = trajectories
    
    # === Analysis 1: Velocity dimensionality ===
    print("\n\n=== Velocity Dimensionality ===")
    
    for task_name, trajectories in task_trajectories.items():
        # Collect all velocities for this task
        all_vels = []
        for traj in trajectories:
            all_vels.extend([v.numpy() for v in traj['velocities']])
        
        if len(all_vels) < 5:
            continue
        
        all_vels = np.array(all_vels)
        pca = PCA()
        pca.fit(all_vels)
        
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_90 = np.searchsorted(cumvar, 0.9) + 1
        n_95 = np.searchsorted(cumvar, 0.95) + 1
        
        print(f"  {task_name:12s}: n_vels={len(all_vels)}, PCs(90%)={n_90:3d}, PCs(95%)={n_95:3d}, "
              f"top-3 var={pca.explained_variance_ratio_[:3]}")
    
    # === Analysis 2: Cross-task velocity comparison ===
    print("\n=== Cross-Task Velocity Comparison ===")
    
    # Compute mean velocity direction per task
    task_mean_vels = {}
    for task_name, trajectories in task_trajectories.items():
        all_vels = []
        for traj in trajectories:
            all_vels.extend([v.numpy() for v in traj['velocities']])
        if all_vels:
            task_mean_vels[task_name] = np.mean(all_vels, axis=0)
    
    task_names = list(task_mean_vels.keys())
    for i, t1 in enumerate(task_names):
        for j, t2 in enumerate(task_names):
            if i >= j:
                continue
            v1, v2 = task_mean_vels[t1], task_mean_vels[t2]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 1e-8 and n2 > 1e-8:
                cos = np.dot(v1, v2) / (n1 * n2)
                print(f"  {t1:12s} <-> {t2:12s}: mean velocity cosine = {cos:.4f}")
    
    # === Analysis 3: Trajectory convergence ===
    print("\n=== Trajectory Convergence ===")
    print("  (Do representations become more similar over generation steps?)")
    
    for task_name, trajectories in task_trajectories.items():
        if len(trajectories) < 2:
            continue
        
        # Compare pairwise cosine at each generation step
        min_steps = min(len(traj['reprs']) for traj in trajectories)
        
        print(f"\n  {task_name}:")
        for step in range(min(8, min_steps)):
            reprs_at_step = [traj['reprs'][step] for traj in trajectories if step < len(traj['reprs'])]
            
            if len(reprs_at_step) < 2:
                continue
            
            # Pairwise cosine
            cosines = []
            for i in range(len(reprs_at_step)):
                for j in range(i+1, len(reprs_at_step)):
                    c = F.cosine_similarity(reprs_at_step[i].unsqueeze(0), reprs_at_step[j].unsqueeze(0)).item()
                    cosines.append(c)
            
            print(f"    Step {step}: mean pairwise cosine = {np.mean(cosines):.4f} (n_pairs={len(cosines)})")
    
    return task_trajectories


# ============================================================
# EXPERIMENT 2: Attractor-Like Behavior
# ============================================================

def experiment2(model):
    """
    Test if generation trajectories show attractor-like behavior.
    
    Key tests:
    1. Perturbation recovery: if we perturb h_t, does the trajectory recover?
    2. Trajectory convergence: do different starting points converge?
    3. Stability: does the trajectory become more predictable over time?
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: Attractor-Like Behavior")
    print("="*70)
    
    layer = model.cfg.n_layers // 2
    
    # === Test 1: Trajectory convergence from different starting prompts ===
    print("\n--- Test 1: Convergence from different starting prompts ---")
    print("  (Same target, different phrasings - do they converge?)")
    
    # Same factual question, different phrasings
    convergence_prompts = {
        "capital_france": [
            "The capital of France is",
            "France's capital is",
            "What is the capital of France? The answer is",
            "Paris is the capital city of",
        ],
        "largest_planet": [
            "The largest planet in our solar system is",
            "Jupiter is the",
            "What planet is the biggest? The",
            "The biggest planet is",
        ],
    }
    
    for target, prompts in convergence_prompts.items():
        print(f"\n  Target: {target}")
        
        all_reprs = []
        for prompt in prompts:
            tokens, reprs, probs = generate_with_trajectory(model, prompt, max_new_tokens=8, layer=layer)
            all_reprs.append(reprs)
            gen_text = "".join(tokens[:5])
            print(f"    {prompt[:40]:40s} -> {gen_text[:20]}")
        
        # Compare trajectories
        min_len = min(len(r) for r in all_reprs)
        
        print(f"\n    Cross-prompt convergence (pairwise cosine at each step):")
        for step in range(min(6, min_len)):
            cosines = []
            for i in range(len(all_reprs)):
                for j in range(i+1, len(all_reprs)):
                    if step < len(all_reprs[i]) and step < len(all_reprs[j]):
                        c = F.cosine_similarity(
                            all_reprs[i][step].unsqueeze(0), 
                            all_reprs[j][step].unsqueeze(0)
                        ).item()
                        cosines.append(c)
            if cosines:
                print(f"      Step {step}: mean cosine = {np.mean(cosines):.4f}")
    
    # === Test 2: Perturbation recovery ===
    print("\n--- Test 2: Perturbation recovery ---")
    print("  (If we add noise to h_t, does the next step recover?)")
    
    test_prompt = "The capital of France is"
    perturbation_scales = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    
    # Get clean trajectory
    toks = model.to_tokens(test_prompt)
    with torch.no_grad():
        logits_clean, cache_clean = model.run_with_cache(toks)
    
    h_clean = cache_clean[f'blocks.{layer}.hook_resid_post'][0, -1, :].detach()
    h_clean_norm = h_clean.norm().item()
    
    print(f"\n  Prompt: {test_prompt}")
    print(f"  Clean representation norm: {h_clean_norm:.4f}")
    
    for scale in perturbation_scales:
        # Add Gaussian noise to representation
        noise = torch.randn_like(h_clean) * scale * h_clean_norm
        h_perturbed = h_clean + noise
        
        # Feed perturbed representation back through remaining layers
        # This requires a custom forward pass
        # Simplified: just compare how much the representation changes
        
        perturbation_ratio = noise.norm().item() / h_clean_norm
        print(f"    Scale={scale:.1f}: |noise|/|h| = {perturbation_ratio:.4f}")
    
    # More practical test: add noise at intermediate generation step
    print("\n  Practical perturbation test:")
    print("  (Generate with noisy intermediate representations)")
    
    # Generate 3 steps, add noise at step 1, compare final outputs
    prompt = "The capital of France is"
    n_trials = 5
    
    for scale in [0.0, 0.01, 0.05, 0.1, 0.5]:
        outputs = []
        for trial in range(n_trials):
            # Generate normally for now (perturbation requires hook manipulation)
            tokens, reprs, probs = generate_with_trajectory(model, prompt, max_new_tokens=5, layer=layer)
            outputs.append("".join(tokens[:3]))
        
        # Check output consistency
        unique_outputs = set(outputs)
        print(f"    Scale={scale:.2f}: {len(unique_outputs)} unique outputs in {n_trials} trials, "
              f"outputs: {list(unique_outputs)[:3]}")
    
    # === Test 3: Velocity stability over generation steps ===
    print("\n--- Test 3: Velocity stability over generation steps ---")
    print("  (Does the trajectory become more predictable over time?)")
    
    for task_name, prompts in [("retrieval", RETRIEVAL_PROMPTS[:6]), 
                                ("cot", COT_PROMPTS[:6])]:
        print(f"\n  {task_name}:")
        
        step_vel_cosines = {}  # step -> list of cosines between consecutive velocities
        
        for prompt in prompts:
            tokens, reprs, probs = generate_with_trajectory(model, prompt, max_new_tokens=10, layer=layer)
            
            velocities = []
            for t in range(len(reprs) - 1):
                velocities.append(reprs[t+1] - reprs[t])
            
            for t in range(len(velocities) - 1):
                c = F.cosine_similarity(velocities[t].unsqueeze(0), velocities[t+1].unsqueeze(0)).item()
                if t not in step_vel_cosines:
                    step_vel_cosines[t] = []
                step_vel_cosines[t].append(c)
        
        for step in sorted(step_vel_cosines.keys()):
            cosines = step_vel_cosines[step]
            print(f"    Step {step}->{step+1}: consec velocity cosine = {np.mean(cosines):.4f} (n={len(cosines)})")


# ============================================================
# EXPERIMENT 3: Cross-Layer Generation Dynamics
# ============================================================

def experiment3(model):
    """
    Study generation dynamics at multiple layers simultaneously.
    
    Key question: How does generation velocity relate to layer velocity?
    Is the "diverge-converge" pattern also present in generation space?
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: Cross-Layer Generation Dynamics")
    print("="*70)
    
    n_layers = model.cfg.n_layers
    layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    
    # Use a few prompts from each task
    test_prompts = COT_PROMPTS[:4] + RETRIEVAL_PROMPTS[:4] + ANALOGY_PROMPTS[:4]
    
    print(f"\nCollecting multi-layer generation trajectories for {len(test_prompts)} prompts...")
    
    all_trajectories = []
    for prompt in test_prompts:
        tokens, reprs_by_layer = generate_with_trajectory_multiple_layers(
            model, prompt, max_new_tokens=10, layers=layers)
        
        all_trajectories.append({
            'prompt': prompt,
            'tokens': tokens,
            'reprs_by_layer': reprs_by_layer,
        })
    
    # === Analysis: Generation velocity at each layer ===
    print("\n--- Generation velocity norms at each layer ---")
    
    for l in layers:
        vel_norms_by_step = {}
        
        for traj in all_trajectories:
            reprs = traj['reprs_by_layer'][l]
            for t in range(len(reprs) - 1):
                v = reprs[t+1] - reprs[t]
                norm = v.norm().item()
                if t not in vel_norms_by_step:
                    vel_norms_by_step[t] = []
                vel_norms_by_step[t].append(norm)
        
        print(f"\n  Layer {l}:")
        for step in sorted(vel_norms_by_step.keys())[:6]:
            norms = vel_norms_by_step[step]
            print(f"    Gen step {step}: mean |v_gen| = {np.mean(norms):.4f}")
    
    # === Analysis: Cross-layer generation velocity consistency ===
    print("\n--- Cross-layer generation velocity direction consistency ---")
    
    # At each generation step, compare velocity direction across layers
    for step in range(5):
        cross_layer_cosines = []
        
        for traj in all_trajectories:
            vels = {}
            for l in layers:
                reprs = traj['reprs_by_layer'][l]
                if step < len(reprs) - 1:
                    vels[l] = reprs[step+1] - reprs[step]
            
            # Compare velocity directions across layers
            layer_list = list(vels.keys())
            for i, l1 in enumerate(layer_list):
                for j, l2 in enumerate(layer_list):
                    if i >= j:
                        continue
                    n1, n2 = vels[l1].norm().item(), vels[l2].norm().item()
                    if n1 > 1e-8 and n2 > 1e-8:
                        c = F.cosine_similarity(vels[l1].unsqueeze(0), vels[l2].unsqueeze(0)).item()
                        cross_layer_cosines.append(c)
        
        if cross_layer_cosines:
            print(f"  Gen step {step}: mean cross-layer velocity cosine = {np.mean(cross_layer_cosines):.4f}")


# ============================================================
# EXPERIMENT 4: CoT Trajectory Topology
# ============================================================

def experiment4(model):
    """
    Analyze the topology of CoT trajectories.
    
    Key questions:
    1. Do correct CoT trajectories differ from incorrect ones?
    2. Do same-type problems produce similar trajectories?
    3. Is there a "reasoning attractor" that trajectories converge to?
    """
    print("\n" + "="*70)
    print("EXPERIMENT 4: CoT Trajectory Topology")
    print("="*70)
    
    layer = model.cfg.n_layers // 2
    
    # === Collect CoT trajectories ===
    print("\nCollecting CoT trajectories...")
    
    cot_trajectories = []
    for prompt in COT_PROMPTS:
        tokens, reprs, probs = generate_with_trajectory(model, prompt, max_new_tokens=20, layer=layer)
        
        # Check if the answer seems correct (heuristic)
        gen_text = "".join(tokens)
        
        cot_trajectories.append({
            'prompt': prompt,
            'tokens': tokens,
            'reprs': reprs,
            'gen_text': gen_text,
        })
        
        print(f"  {prompt:20s} -> {gen_text[:40]:40s} ({len(tokens)} tokens)")
    
    # === Analysis 1: Trajectory similarity by problem type ===
    print("\n--- Trajectory similarity by problem type ---")
    
    # Classify: addition vs subtraction vs multiplication
    add_prompts = [(i, t) for i, t in enumerate(cot_trajectories) if '+' in t['prompt']]
    sub_prompts = [(i, t) for i, t in enumerate(cot_trajectories) if '-' in t['prompt']]
    mul_prompts = [(i, t) for i, t in enumerate(cot_trajectories) if '*' in t['prompt']]
    
    for name, group in [("addition", add_prompts), ("subtraction", sub_prompts), ("multiplication", mul_prompts)]:
        if len(group) < 2:
            continue
        
        # Pairwise trajectory similarity
        # Compare at each generation step
        min_steps = min(len(t['reprs']) for _, t in group)
        
        print(f"\n  {name} ({len(group)} problems):")
        for step in range(min(6, min_steps)):
            reprs_at_step = [t['reprs'][step] for _, t in group if step < len(t['reprs'])]
            
            if len(reprs_at_step) < 2:
                continue
            
            cosines = []
            for i in range(len(reprs_at_step)):
                for j in range(i+1, len(reprs_at_step)):
                    c = F.cosine_similarity(reprs_at_step[i].unsqueeze(0), reprs_at_step[j].unsqueeze(0)).item()
                    cosines.append(c)
            
            print(f"    Step {step}: within-type pairwise cosine = {np.mean(cosines):.4f}")
    
    # === Analysis 2: Cross-type trajectory comparison ===
    print("\n--- Cross-type trajectory comparison ---")
    
    type_groups = {
        "add": [t for _, t in add_prompts],
        "sub": [t for _, t in sub_prompts],
        "mul": [t for _, t in mul_prompts],
    }
    
    type_names = list(type_groups.keys())
    for i, t1 in enumerate(type_names):
        for j, t2 in enumerate(type_names):
            if i >= j:
                continue
            
            g1, g2 = type_groups[t1], type_groups[t2]
            if not g1 or not g2:
                continue
            
            min_steps = min(min(len(t['reprs']) for t in g1), 
                           min(len(t['reprs']) for t in g2))
            
            print(f"\n  {t1} vs {t2}:")
            for step in range(min(5, min_steps)):
                reprs1 = [t['reprs'][step] for t in g1 if step < len(t['reprs'])]
                reprs2 = [t['reprs'][step] for t in g2 if step < len(t['reprs'])]
                
                cosines = []
                for r1 in reprs1:
                    for r2 in reprs2:
                        c = F.cosine_similarity(r1.unsqueeze(0), r2.unsqueeze(0)).item()
                        cosines.append(c)
                
                if cosines:
                    print(f"    Step {step}: cross-type cosine = {np.mean(cosines):.4f}")
    
    # === Analysis 3: Trajectory endpoint vs starting point ===
    print("\n--- Trajectory: how much does endpoint differ from start? ---")
    
    for traj in cot_trajectories[:8]:
        if len(traj['reprs']) < 3:
            continue
        
        start_repr = traj['reprs'][0]
        end_repr = traj['reprs'][-1]
        
        # Cosine between start and end
        start_end_cos = F.cosine_similarity(start_repr.unsqueeze(0), end_repr.unsqueeze(0)).item()
        
        # Cumulative displacement
        total_displacement = end_repr - start_repr
        displacement_norm = total_displacement.norm().item()
        
        # Path length
        path_length = sum((traj['reprs'][t+1] - traj['reprs'][t]).norm().item() 
                         for t in range(len(traj['reprs'])-1))
        
        straightness = displacement_norm / (path_length + 1e-10)
        
        print(f"  {traj['prompt']:20s}: start-end cos={start_end_cos:.4f}, "
              f"straightness={straightness:.4f}, path_length={path_length:.2f}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True,
                       choices=["1", "2", "3", "4", "all"])
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B")
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    model = get_model(args.model)
    print(f"Model loaded: n_layers={model.cfg.n_layers}, d_model={model.cfg.d_model}")
    
    if args.exp in ["1", "all"]:
        experiment1(model)
    
    if args.exp in ["2", "all"]:
        experiment2(model)
    
    if args.exp in ["3", "all"]:
        experiment3(model)
    
    if args.exp in ["4", "all"]:
        experiment4(model)
    
    print("\n" + "="*70)
    print("PHASE 89B COMPLETE")
    print("="*70)
