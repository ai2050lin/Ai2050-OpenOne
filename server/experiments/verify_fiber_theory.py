import os

# Set environment variables for model loading (Mirror & Cache)
os.environ["HF_HOME"] = r"D:\develop\model"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import logging

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from transformer_lens import HookedTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(model_name="gpt2-small"):
    logging.info(f"Loading model: {model_name}...")
    try:
        model = HookedTransformer.from_pretrained(model_name)
        model.eval()
        return model
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return None

def get_activation(model, text, layer, token_idx=-1):
    """Get the residual stream activation for a specific token."""
    with torch.no_grad():
        _, cache = model.run_with_cache(text, names_filter=lambda x: f"blocks.{layer}.hook_resid_post" in x)
        # Check tokenization to ensure we are grabbing the right token if needed.
        # For simplicity, we grab the last token (assuming the prompt ends with the target word)
        act = cache[f"blocks.{layer}.hook_resid_post"][0, token_idx, :].cpu().numpy()
    return act

def experiment_parallel_transport(model, layer=6):
    """
    Experiment 1: Parallel Transport (Analogy Test)
    Tests if vector arithmetic holds: King - Man + Woman ~ Queen
    or equivalently: (King - Man) ~ (Queen - Woman)
    """
    logging.info(f"\n--- Experiment 1: Parallel Transport (Layer {layer}) ---")
    
    pairs = [
        ("man", "king", "woman", "queen"),
        ("paris", "france", "rome", "italy"),
        ("walk", "walked", "play", "played")
    ]
    
    passing = 0
    
    for a, b, c, d in pairs:
        # We use simple prompts. Ideally we should use context, but for zero-shot arithmetic, single words/short phrases work ok in residual stream
        # A better way is: "Man is to King as Woman is to Queen" but here we just look at raw word embeddings in context
        # Let's try to extract the concept vector.
        
        # Strategy: Use a simple template "The word is [WORD]." to give it some context, or just the word itself.
        # GPT-2 tokenization might split words. We'll hope for single tokens or take the last one.
        
        v_a = get_activation(model, a, layer)
        v_b = get_activation(model, b, layer)
        v_c = get_activation(model, c, layer)
        v_d = get_activation(model, d, layer)
        
        # Vector difference (The Transport Vector)
        transport_1 = v_b - v_a # Man -> King
        transport_2 = v_d - v_c # Woman -> Queen
        
        # Cosine Similarity
        sim = cosine_similarity([transport_1], [transport_2])[0][0]
        
        logging.info(f"Analogy: {a}:{b} :: {c}:{d}")
        logging.info(f"  Transport Similarity: {sim:.4f}")
        
        if sim > 0.4: # Threshold for "Directionally Consistent"
            passing += 1
            
    if passing >= 2:
        logging.info("result: PASS")
        return True
    else:
        logging.info("result: WEAK/FAIL (Might need better model or layer selection)")
        return False

def experiment_fiber_decoupling(model, layer=6):
    """
    Experiment 2: Fiber Decoupling (Orthogonality Test)
    Tests if changing the Subject is orthogonal to changing the Object.
    
    Sentence Structure: "The [SUBJ] chased the [OBJ]."
    """
    logging.info(f"\n--- Experiment 2: Fiber Decoupling (Layer {layer}) ---")
    
    template = "The {} chased the {}."
    
    # 1. Define Subject Change (Subject Fiber)
    # Fix Object = "mouse", Change Subject: "cat" -> "dog"
    s1 = template.format("cat", "mouse")
    s2 = template.format("dog", "mouse")
    
    v_s1 = get_activation(model, s1, layer)
    v_s2 = get_activation(model, s2, layer)
    
    vec_subj_change = v_s2 - v_s1 # Vector representing "Subject Change"
    
    # 2. Define Object Change (Object Fiber)
    # Fix Subject = "cat", Change Object: "mouse" -> "bird"
    # Using 'bird' to keep length likely similar
    s3 = template.format("cat", "mouse") 
    s4 = template.format("cat", "bird")
    
    v_s3 = get_activation(model, s3, layer) # Should be identical to v_s1 generally
    v_s4 = get_activation(model, s4, layer)
    
    vec_obj_change = v_s4 - v_s3 # Vector representing "Object Change"
    
    # 3. Check Orthogonality
    sim = cosine_similarity([vec_subj_change], [vec_obj_change])[0][0]
    
    logging.info(f"Subject Change Vector (Cat->Dog) vs Object Change Vector (Mouse->Bird)")
    logging.info(f"  Cosine Similarity: {sim:.4f}")
    
    # We expect this to be LOW (near 0).
    # If it is high, it means changing subject and changing object move the state in the SAME direction (Entangled).
    # If near 0, they are Decoupled (Independent Fibers).
    
    if abs(sim) < 0.3:
        logging.info("result: PASS (Orthogonal/Decoupled)")
        return True
    else:
        logging.info("result: FAIL (Entangled)")
        return False

def main():
    model = load_model("gpt2-small")
    if not model:
        return
    
    # Layer selection: Mid-late layers are usually best for semantic composition
    target_layer = model.cfg.n_layers // 2 + 2 # roughly layer 8 for gpt2-small (12 layers)
    
    pass_1 = experiment_parallel_transport(model, layer=target_layer)
    pass_2 = experiment_fiber_decoupling(model, layer=target_layer)
    
    print("\n" + "="*30)
    print("VERIFICATION SUMMARY")
    print("="*30)
    print(f"Exp 1 (Parallel Transport): {'[SUCCESS]' if pass_1 else '[WEAK]'}")
    print(f"Exp 2 (Fiber Decoupling)  : {'[SUCCESS]' if pass_2 else '[FAIL]'}")
    print("="*30)

if __name__ == "__main__":
    main()
