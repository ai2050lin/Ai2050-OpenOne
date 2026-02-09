import os

import numpy as np
import torch
import tqdm
from sklearn.metrics.pairwise import cosine_similarity

import transformer_lens
from transformer_lens import HookedTransformer

# Setup environment variables for model loading
os.environ["HF_HOME"] = r"D:\develop\model"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def get_activations(model, prompts, token_idx=-1):
    """Get the residual stream activations for the last token (or specified) from all layers."""
    # We use run_with_cache to get activations
    # We'll focus on the residual stream at the end of each block: 'blocks.{layer}.hook_resid_post'
    
    # Run one by one or batch? Batch is better for speed, but memory watch.
    # 4B model might fit small batch.
    
    act_dict = {i: [] for i in range(model.cfg.n_layers)}
    
    with torch.no_grad():
        logits, cache = model.run_with_cache(prompts)
        
        for i in range(model.cfg.n_layers):
            # Shape: [batch, seq_len, d_model]
            acts = cache[f"blocks.{i}.hook_resid_post"]
            # specific token index
            acts = acts[:, token_idx, :].cpu().numpy()
            act_dict[i].append(acts)
            
    # Concatenate if we did loops (here we assume fit in one batch)
    for i in range(model.cfg.n_layers):
        act_dict[i] = np.concatenate(act_dict[i], axis=0)
        
    return act_dict

def analyze_rsa_layers(model):
    print("\n[Scheme A] Layer-wise Structure vs Content Mapping (RSA)")
    
    # Construct dataset
    # We want: 
    # Conditions: Syntax A vs B, Content 1..N
    # Syntax A: "The [noun] is [adj]."
    # Syntax B: "Here is a [adj] [noun]."
    # Content pairs: (noun, adj)
    
    content_pairs = [
        ("apple", "red"), ("sky", "blue"), ("grass", "green"), ("snow", "white"), ("lemon", "yellow"),
        ("cat", "cute"), ("rock", "hard"), ("fire", "hot"), ("ice", "cold"), ("night", "dark")
    ]
    
    prompts = []
    # Labels for RSA
    syntax_labels = []
    content_labels = []
    
    for i, (n, a) in enumerate(content_pairs):
        # Syntax A
        prompts.append(f"The {n} is {a}.")
        syntax_labels.append(0)
        content_labels.append(i)
        
        # Syntax B
        prompts.append(f"Here is a {a} {n}.")
        syntax_labels.append(1)
        content_labels.append(i)
        
    print(f"Analyzing {len(prompts)} prompts...")
    
    # Get activations (last token)
    # Note: "The apple is red." -> "." is last.
    # "Here is a red apple." -> "." is last.
    # Last token captures sentence state.
    acts_by_layer = get_activations(model, prompts, token_idx=-1)
    
    print(f"{'Layer':<5} | {'Syn Score':<10} | {'Sem Score':<10} | {'Type'}")
    print("-" * 50)
    
    results = []
    
    for layer in range(model.cfg.n_layers):
        acts = acts_by_layer[layer] # [batch, d_model]
        
        # Compute Similarity Matrix [batch, batch]
        sim_matrix = cosine_similarity(acts)
        
        # RSA Logic
        # Syntax Score: Avg Sim(Same Syntax) - Avg Sim(Diff Syntax)
        # Semantic Score: Avg Sim(Same Content) - Avg Sim(Diff Content)
        
        same_syn_mask = np.array([s1 == s2 for s1 in syntax_labels for s2 in syntax_labels]).reshape(len(prompts), len(prompts))
        same_con_mask = np.array([c1 == c2 for c1 in content_labels for c2 in content_labels]).reshape(len(prompts), len(prompts))
        
        # Exclude diagonal
        np.fill_diagonal(same_syn_mask, False)
        np.fill_diagonal(same_con_mask, False)
        
        # Calculate scores
        # Allow normalizing to range
        
        avg_sim_same_syn = sim_matrix[same_syn_mask].mean()
        avg_sim_diff_syn = sim_matrix[~same_syn_mask].mean() # Technically includes self-diagonal in diff if not careful, but simpler here
        
        avg_sim_same_con = sim_matrix[same_con_mask].mean()
        avg_sim_diff_con = sim_matrix[~same_con_mask].mean()
        
        syn_score = avg_sim_same_syn - avg_sim_diff_syn
        sem_score = avg_sim_same_con - avg_sim_diff_con
        
        # Determine dominant type
        ratio = sem_score / (syn_score + 1e-6)
        if sem_score > syn_score * 1.5:
            dom_type = "Fiber (Sem)"
        elif syn_score > sem_score * 1.5:
            dom_type = "Base (Syn)"
        else:
            dom_type = "Mixed"
            
        print(f"{layer:<5} | {syn_score:<10.4f} | {sem_score:<10.4f} | {dom_type}")
        results.append((layer, syn_score, sem_score))
        
    return results

def analyze_curvature(model):
    print("\n[Scheme B] Curvature / Interaction Analysis")
    
    # Interaction = || (AB - A) - (B - Base) || ? 
    # Or Commutativity: || Act(Base->A->AB) - Act(Base->B->AB) ||
    # Let's use linear additivity constraint.
    # Flat manifold => Act(AB) - Act(Base) = (Act(A) - Act(Base)) + (Act(B) - Act(Base))
    # Curvature => Deviation from this.
    
    # Base: "The coffee is hot."
    # Change A (Subject): Coffee -> Tea. ("The tea is hot.")
    # Change B (Adj): Hot -> Cold. ("The coffee is cold.")
    # Target (AB): "The tea is cold."
    
    triples = [
        # (Base, A, B, AB)
        ("The coffee is hot.", "The tea is hot.", "The coffee is cold.", "The tea is cold."),
        ("The man is happy.", "The woman is happy.", "The man is sad.", "The woman is sad."),
        ("The day is long.", "The night is long.", "The day is short.", "The night is short."),
        ("The cat is fast.", "The dog is fast.", "The cat is slow.", "The dog is slow.")
    ]
    
    # Encode all
    all_prompts = []
    for t in triples:
        all_prompts.extend(t)
        
    acts_by_layer = get_activations(model, all_prompts, token_idx=-1)
    
    print(f"{'Layer':<5} | {'Curvature (Interaction Magnitude)':<35}")
    print("-" * 50)
    
    for layer in range(model.cfg.n_layers):
        acts = acts_by_layer[layer] # [4*N, d]
        
        curvatures = []
        
        for i in range(len(triples)):
            base_idx = i * 4
            a_idx = i * 4 + 1
            b_idx = i * 4 + 2
            ab_idx = i * 4 + 3
            
            vec_base = acts[base_idx]
            vec_a = acts[a_idx]
            vec_b = acts[b_idx]
            vec_ab = acts[ab_idx]
            
            # Linear Prediction
            delta_a = vec_a - vec_base
            delta_b = vec_b - vec_base
            vec_pred = vec_base + delta_a + delta_b
            
            # Interaction vector
            vec_interaction = vec_ab - vec_pred
            
            # Magnitude (normalized by vector norm to be scale invariant-ish)
            norm = np.linalg.norm(vec_interaction)
            base_norm = np.linalg.norm(vec_base)
            
            curvatures.append(norm / base_norm)
            
        avg_curv = np.mean(curvatures)
        print(f"{layer:<5} | {avg_curv:<10.6f}")

def main():
    print("Initializing analysis...")
    
    try:
        model_name = "Qwen/Qwen3-4B" # User specified
        # Fallback to qwen2.5 or other compatible if not exactly loading, but let's try.
        # Note: Qwen models usually need trust_remote_code=True
        
        model = HookedTransformer.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        model.eval()
        
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        print("Tip: Ensure environment variables HF_HOME are set and model exists.")
        return

    # Run Experiments
    analyze_rsa_layers(model)
    analyze_curvature(model)
    
    print("\nAnalysis Complete.")

if __name__ == "__main__":
    main()
