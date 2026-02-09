
import os
import sys

# Set environment variables for model loading BEFORE other imports
os.environ["HF_HOME"] = r"D:\develop\model"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Redirect stdout to file
sys.stdout = open("riemannian_orthogonal_results.txt", "w", encoding="utf-8")

import numpy as np
import torch
import torch.nn.functional as F

from transformer_lens import HookedTransformer

# Configuration
MODEL_NAME = "gpt2"
LAYER_ID = 6 # Middle layer, where entanglement is severe

def load_model():
    print(f"Loading {MODEL_NAME}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        # Load without folding LayerNorm to access pre/post resid streams correctly
        model = HookedTransformer.from_pretrained(MODEL_NAME, device=device, fold_ln=False)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def get_resid_at_last_token(model, text, layer_id):
    cache_name = f"blocks.{layer_id}.hook_resid_post"
    _, cache = model.run_with_cache(text)
    # Shape: [batch, pos, d_model]
    return cache[cache_name][0, -1, :]

def get_logits_at_last_token(model, text_or_act, input_text_len=None):
    # If text_or_act is a tensor (activation), we need to run through the rest of the model
    if isinstance(text_or_act, torch.Tensor):
        # We need to manually run from layer_id to end.
        # This is complex with HookedTransformer. a simpler way is using hooks.
        # But for "steering", we usually add a hook.
        pass
    else:
        logits, _ = model.run_with_cache(text_or_act)
        return logits[0, -1, :]

def steered_logits(model, text, layer_id, steer_vec, alpha=1.0):
    cache_name = f"blocks.{layer_id}.hook_resid_post"
    
    def hook_fn(resid, hook):
        # Add steering vector to the last token position
        resid[:, -1, :] += alpha * steer_vec
        return resid
    
    # Run with hook
    model.reset_hooks()
    logits = model.run_with_hooks(
        text,
        fwd_hooks=[(cache_name, hook_fn)]
    )
    return logits[0, -1, :]

def analyze_stereotypes(model, layer_id):
    print(f"\n--- Analyzing Riemannian Orthogonalization (Layer {layer_id}) ---")
    
    # 1. Define Vectors
    # Knowledge Axis (Gender): "He" - "She"
    print("Extracting vectors...")
    
    # We use simple prompts to get canonical gender vectors
    # "The man is" -> resid
    v_man = get_resid_at_last_token(model, "The man is", layer_id)
    v_woman = get_resid_at_last_token(model, "The woman is", layer_id)
    v_gender_global = v_woman - v_man
    v_gender_global = v_gender_global / v_gender_global.norm()
    
    # Logic Axis (Profession): "The nurse is" - "The doctor is"
    # This represents the "Profession" direction in the latent space
    v_doctor = get_resid_at_last_token(model, "The doctor is", layer_id)
    v_nurse = get_resid_at_last_token(model, "The nurse is", layer_id)
    v_logic_local = v_nurse - v_doctor
    v_logic_local = v_logic_local / v_logic_local.norm()

    # 2. Check Entanglement (Cosine Similarity)
    sim = F.cosine_similarity(v_gender_global.unsqueeze(0), v_logic_local.unsqueeze(0)).item()
    print(f"Entanglement (Gender vs Profession): {sim:.4f}")
    
    # 3. Construct Steering Vectors
    
    # Vector A: Naive Gender Vector
    vec_naive = v_gender_global
    
    # Vector B: Riemannian Orthogonalized Vector
    # v_ortho = v_gender - proj_logic(v_gender)
    proj = torch.dot(v_gender_global, v_logic_local) * v_logic_local
    vec_riemann = v_gender_global - proj
    vec_riemann = vec_riemann / vec_riemann.norm() # Normalize to compare magnitude effects equally
    
    # 4. Steering Experiment
    # Prompt: "The doctor said that"
    # Target: "she" (token)
    # Hallucination: "nurse" (token)
    
    prompt = "The doctor said that"
    
    token_she = model.to_single_token(" she")
    token_he = model.to_single_token(" he")
    token_nurse = model.to_single_token(" nurse")
    
    # Baseline
    logits_base = get_logits_at_last_token(model, prompt)
    prob_base_she = F.softmax(logits_base, dim=0)[token_she].item()
    prob_base_nurse = F.softmax(logits_base, dim=0)[token_nurse].item()
    
    print(f"\nBaseline:")
    print(f"  P('she'): {prob_base_she:.6f}")
    print(f"  P('nurse'): {prob_base_nurse:.6f}")
    
    # Naive Steering
    scale = 5.0
    logits_naive = steered_logits(model, prompt, layer_id, vec_naive, alpha=scale)
    prob_naive_she = F.softmax(logits_naive, dim=0)[token_she].item()
    prob_naive_nurse = F.softmax(logits_naive, dim=0)[token_nurse].item()
    
    print(f"\nNaive Steering (+Gender):")
    print(f"  P('she'): {prob_naive_she:.6f}")
    print(f"  P('nurse'): {prob_naive_nurse:.6f}")
    
    # Riemannian Steering
    logits_riemann = steered_logits(model, prompt, layer_id, vec_riemann, alpha=scale)
    prob_riem_she = F.softmax(logits_riemann, dim=0)[token_she].item()
    prob_riem_nurse = F.softmax(logits_riemann, dim=0)[token_nurse].item()
    
    print(f"\nRiemannian Steering (+Orthogonal Gender):")
    print(f"  P('she'): {prob_riem_she:.6f}")
    print(f"  P('nurse'): {prob_riem_nurse:.6f}")
    
    # 5. Analysis
    print("\n--- Conclusion ---")
    hallucination_naive = prob_naive_nurse - prob_base_nurse
    hallucination_riem = prob_riem_nurse - prob_base_nurse
    
    print(f"Hallucination (Naive): {hallucination_naive:.6f}")
    print(f"Hallucination (Riemann): {hallucination_riem:.6f}")
    
    if hallucination_riem < hallucination_naive:
        print("SUCCESS: Riemannian Orthogonalization reduced hallucination.")
    else:
        print("FAILURE: Riemannian Orthogonalization did not reduce hallucination.")

def main():
    model = load_model()
    if model:
        analyze_stereotypes(model, LAYER_ID)

if __name__ == "__main__":
    main()
