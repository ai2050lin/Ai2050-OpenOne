import traceback

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity


def run_agi_verification(model):
    results = {}
    print("Starting AGI Verification...")
    try:
        results['rsa'] = analyze_rsa_layers(model)
        results['curvature'] = analyze_curvature(model)
    except Exception as e:
        print(f"AGI Verification Error: {e}")
        results['error'] = str(e)
        results['traceback'] = traceback.format_exc()
    return results

def get_activations(model, prompts, token_idx=-1):
    act_dict = {i: [] for i in range(model.cfg.n_layers)}
    
    with torch.no_grad():
        # Process in chunks to avoid OOM even here
        chunk_size = 4
        for i in range(0, len(prompts), chunk_size):
            batch = prompts[i:i+chunk_size]
            logits, cache = model.run_with_cache(batch)
            
            for layer in range(model.cfg.n_layers):
                acts = cache[f"blocks.{layer}.hook_resid_post"]
                acts = acts[:, token_idx, :].cpu().numpy()
                act_dict[layer].append(acts)
            
            # Clear cache
            del logits, cache
            torch.cuda.empty_cache()

    for i in range(model.cfg.n_layers):
        act_dict[i] = np.concatenate(act_dict[i], axis=0)
        
    return act_dict

def analyze_rsa_layers(model):
    print("Running RSA...")
    content_pairs = [
        ("apple", "red"), ("sky", "blue"), ("grass", "green"), ("snow", "white"), ("lemon", "yellow"),
        ("cat", "cute"), ("rock", "hard"), ("fire", "hot"), ("ice", "cold"), ("night", "dark")
    ]
    
    prompts = []
    # 0: Active (Syn A), 1: Inverted (Syn B)
    syntax_labels = [] 
    content_labels = []
    
    for i, (n, a) in enumerate(content_pairs):
        prompts.append(f"The {n} is {a}.")
        syntax_labels.append(0)
        content_labels.append(i)
        
        prompts.append(f"Here is a {a} {n}.")
        syntax_labels.append(1)
        content_labels.append(i)
        
    acts_by_layer = get_activations(model, prompts)
    
    layer_stats = []
    
    for layer in range(model.cfg.n_layers):
        acts = acts_by_layer[layer]
        sim_matrix = cosine_similarity(acts)
        
        # Determine masks
        n_samples = len(prompts)
        syn_mask = np.zeros((n_samples, n_samples), dtype=bool)
        con_mask = np.zeros((n_samples, n_samples), dtype=bool)
        
        for r in range(n_samples):
            for c in range(n_samples):
                if r == c: continue
                if syntax_labels[r] == syntax_labels[c]: syn_mask[r,c] = True
                if content_labels[r] == content_labels[c]: con_mask[r,c] = True
                
        avg_syn = sim_matrix[syn_mask].mean()
        avg_diff_syn = sim_matrix[~syn_mask].mean()
        
        avg_con = sim_matrix[con_mask].mean()
        avg_diff_con = sim_matrix[~con_mask].mean()
        
        syn_score = avg_syn - avg_diff_syn
        sem_score = avg_con - avg_diff_con
        
        dom_type = "Mixed"
        if sem_score > syn_score * 1.2: dom_type = "Fiber"
        elif syn_score > sem_score * 1.2: dom_type = "Base"
        
        layer_stats.append({
            "layer": layer,
            "syn_score": float(syn_score),
            "sem_score": float(sem_score),
            "type": dom_type
        })
        
    return layer_stats

def analyze_curvature(model):
    print("Running Curvature...")
    # Base: "The coffee is hot." -> Tea -> Cold -> "The tea is cold."
    triples = [
        ("The coffee is hot.", "The tea is hot.", "The coffee is cold.", "The tea is cold."),
        ("The man is happy.", "The woman is happy.", "The man is sad.", "The woman is sad."),
        ("The day is long.", "The night is long.", "The day is short.", "The night is short."),
        ("The cat is fast.", "The dog is fast.", "The cat is slow.", "The dog is slow.")
    ]
    
    all_prompts = []
    for t in triples: all_prompts.extend(t)
        
    acts_by_layer = get_activations(model, all_prompts)
    
    curvature_stats = []
    
    for layer in range(model.cfg.n_layers):
        acts = acts_by_layer[layer]
        curvatures = []
        
        for i in range(len(triples)):
            base, a, b, ab = acts[i*4], acts[i*4+1], acts[i*4+2], acts[i*4+3]
            pred = base + (a - base) + (b - base)
            interaction = ab - pred
            score = np.linalg.norm(interaction) / (np.linalg.norm(base) + 1e-6)
            curvatures.append(score)
            
        curvature_stats.append({
            "layer": layer,
            "curvature": float(np.mean(curvatures))
        })
        
    return curvature_stats
    return curvature_stats

def run_concept_steering(model, prompt, layer_idx=15, strength=1.0, concept_pair="formal_casual"):
    import gc

    # 1. Define Exemplars for Concept Direction
    exemplars = {
        "formal_casual": [
            ("I appreciate your assistance.", "Thanks for helping."),
            ("Could you please inform me?", "Can you tell me?"),
            ("It is imperative that we proceed.", "We gotta move."),
        ],
        "positive_negative": [
            ("This is absolutely wonderful.", "This is terrible."),
            ("I love this weather.", "I hate this weather."),
            ("The result was a success.", "The result was a failure.")
        ],
        "simple_complex": [
            ("The cat sat on the mat.", "The feline entity positioned itself upon the textile surface."),
            ("Eat your vegetables.", "Consume your fibrous plant matter."),
            ("It is raining.", "Precipitation is currently occurring.")
        ]
    }
    
    pairs = exemplars.get(concept_pair, exemplars["formal_casual"])
    
    # 2. Compute Steering Vector
    # Get activations for pairs
    diffs = []
    
    try:
        with torch.no_grad():
            for p_pos, p_neg in pairs:
                # Run forward pass for positive exemplar
                _, cache_pos = model.run_with_cache(p_pos)
                act_pos = cache_pos[f"blocks.{layer_idx}.hook_resid_post"][:, -1, :].clone()
                del cache_pos # Free cache immediately
                
                # Run forward pass for negative exemplar
                _, cache_neg = model.run_with_cache(p_neg)
                act_neg = cache_neg[f"blocks.{layer_idx}.hook_resid_post"][:, -1, :].clone()
                del cache_neg # Free cache immediately
                
                # Calculate difference
                diff = act_pos - act_neg
                diffs.append(diff)
                
                # Clean up intermediate tensors
                del act_pos, act_neg
                torch.cuda.empty_cache()
                
            steering_vec = torch.stack(diffs).mean(dim=0)
            
        # 3. Define Hook
        def steering_hook(resid_pre, hook):
            # resid_pre: [batch, pos, d_model]
            return resid_pre + (strength * steering_vec)

        # 4. Generate with Hook
        model.reset_hooks()
        
        # Run generation
        hook_name = f"blocks.{layer_idx}.hook_resid_post"
        
        # Ensure we don't have lingering memory
        gc.collect()
        torch.cuda.empty_cache()
        
        with model.hooks(fwd_hooks=[(hook_name, steering_hook)]):
            generated = model.generate(prompt, max_new_tokens=20, temperature=0.7)
            
        return {
            "original_prompt": prompt,
            "generated": generated,
            "layer": layer_idx,
            "strength": strength,
            "concept": concept_pair,
            "vector_norm": float(torch.norm(steering_vec).item())
        }
    except Exception as e:
        print(f"Error in steering: {e}")
        traceback.print_exc()
        raise e
    finally:
        # Cleanup
        model.reset_hooks()
        if 'steering_vec' in locals(): del steering_vec
        if 'diffs' in locals(): del diffs
        gc.collect()
        torch.cuda.empty_cache()
