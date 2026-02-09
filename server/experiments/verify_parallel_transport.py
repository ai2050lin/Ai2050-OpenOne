
import os

import numpy as np

# Set environment variables for model loading (Mirror for China)
os.environ["HF_HOME"] = r"D:\develop\model"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn.functional as F

from transformer_lens import HookedTransformer


def cosine_similarity(v1, v2):
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()

def main():
    print("Loading model...")
    model = HookedTransformer.from_pretrained("gpt2-small")
    model.eval()
    
    # Define Analogies (A : B :: C : D) -> Expect (B - A) ~ (D - C)
    # Using simple tokens. Space prefix is important for GPT-2.
    analogies = [
        (" Man", " King", " Woman", " Queen"),
        (" Paris", " France", " Rome", " Italy"),
        (" walk", " walked", " talk", " talked"),
        (" big", " bigger", " small", " smaller"),
        (" he", " his", " she", " her"),
    ]

    print("\n--- Verifying Parallel Transport (Analogies) ---")
    
    layer = 10 # Check deep layer
    hook_name = f"blocks.{layer}.hook_resid_post"
    
    total_sim = 0
    
    for a, b, c, d in analogies:
        # Get tokens
        tokens_ab = model.to_tokens(f"{a} {b}", prepend_bos=False) # Not using BOS to focus on word embeddings/context
        tokens_cd = model.to_tokens(f"{c} {d}", prepend_bos=False)
        
        # In this simple case, we just look at the embeddings or output of a simple forward pass
        # But to be more rigorous, let's run the model on single words to get their "context-free" vectors 
        # OR run on the pairs and look at the shift.
        # Let's try context-free first (simulating 'fiber' movement in abstract space)
        
        # Actually, let's use the embedding matrix directly for "pure" semantic difference,
        # OR run the model with a prompt.
        # Given the theory is about "generating" trajectory, let's look at the residual stream
        # when processing these words.
        
        # Method: Get activation of the specific token.
        # We process single tokens (wrapped in a list)
        
        def get_vector(text):
            # Prepend space if not present, though we included it in strings
            ts = model.to_tokens(text) 
            with torch.no_grad():
                logits, cache = model.run_with_cache(ts)
            # Get the residual stream at the end of the specified layer for the last token
            return cache[hook_name][0, -1, :]

        vec_a = get_vector(a)
        vec_b = get_vector(b)
        vec_c = get_vector(c)
        vec_d = get_vector(d)
        
        diff_1 = vec_b - vec_a
        diff_2 = vec_d - vec_c
        
        sim = cosine_similarity(diff_1, diff_2)
        total_sim += sim
        
        print(f"Angle({b}-{a}, {d}-{c}) Cos Sim: {sim:.4f}")

    avg_sim = total_sim / len(analogies)
    print(f"\nAverage Cosine Similarity: {avg_sim:.4f}")
    
    if avg_sim > 0.5:
        print("\n[SUCCESS] Strong evidence for Parallel Transport Structure.")
    else:
        print("\n[WEAK] Evidence is weak, structure might be more complex or layer choice is wrong.")

if __name__ == "__main__":
    main()
