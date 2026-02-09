
import os

import matplotlib.pyplot as plt
import numpy as np

# Set environment variables for model loading (Mirror for China)
os.environ["HF_HOME"] = r"D:\develop\model"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from sklearn.decomposition import PCA

from transformer_lens import HookedTransformer


def main():
    print("Loading model...")
    model = HookedTransformer.from_pretrained("gpt2-small")
    model.eval()

    # Fiber Experiment:
    # Base Manifold (Structure): "The [SUBJECT] [VERB] the [OBJECT]."
    # We want to see if changing the tokens (Fiber) while keeping structure constant 
    # results in orthogonal movements or specific "fiber" directions.

    templates = [
        "The cat sat on the mat",
        "The dog sat on the mat",
        "The bird sat on the mat", # Variation 1: Subject (Token 1-2)
        "The cat lay on the mat",
        "The cat stood on the mat", # Variation 2: Verb (Token 3)
        "The cat sat on the rug",
        "The cat sat on the bed",   # Variation 3: Object (Token 6)
    ]
    
    print("\n--- Verifying Fiber Structure (Subspace Analysis) ---")
    
    # We'll look at the residual stream at the end of a middle layer
    layer = 6 
    hook_name = f"blocks.{layer}.hook_resid_post"
    
    activations = []
    
    for sentence in templates:
        with torch.no_grad():
            _, cache = model.run_with_cache(sentence)
        # We perform analysis on the last token's state, which should integrate the whole sentence info
        # OR we can look at specific positions. 
        # Let's look at the last token state.
        activations.append(cache[hook_name][0, -1, :].cpu().numpy())

    X = np.array(activations)
    
    # PCA
    pca = PCA(n_components=3)
    pca.fit(X)
    
    print(f"Explained Variance Ratios: {pca.explained_variance_ratio_}")
    
    # Analyze which sentences cluster together in PC1 vs PC2
    X_pca = pca.transform(X)
    
    print("\nPCA Coordinates for each sentence:")
    for i, sent in enumerate(templates):
        print(f"{sent:<25} : {X_pca[i]}")

    # Hypothesis check:
    # If "Subject" variation is one fiber direction, and "Object" is another,
    # sentences varying only by subject should move along one axis (e.g. PC1),
    # and sentences varying only by object along another (e.g. PC2).
    
    dist_subject = np.linalg.norm(X_pca[0] - X_pca[1]) # cat vs dog
    dist_verb = np.linalg.norm(X_pca[0] - X_pca[3])    # sat vs lay
    dist_object = np.linalg.norm(X_pca[0] - X_pca[5])  # mat vs rug
    
    print(f"\nDistances in reduced space:")
    print(f"Subject Change (Cat->Dog): {dist_subject:.4f}")
    print(f"Verb Change (Sat->Lay):   {dist_verb:.4f}")
    print(f"Object Change (Mat->Rug): {dist_object:.4f}")

    # Dot product of difference vectors to check orthogonality
    vec_subj_diff = X_pca[1] - X_pca[0]
    vec_obj_diff = X_pca[5] - X_pca[0]
    
    cos_sim_orth = np.dot(vec_subj_diff, vec_obj_diff) / (np.linalg.norm(vec_subj_diff) * np.linalg.norm(vec_obj_diff))
    print(f"\nOrthogonality of Subject vs Object difference vectors (Cos Sim): {cos_sim_orth:.4f}")
    
    if abs(cos_sim_orth) < 0.3:
        print("[SUCCESS] Subject and Object variations act as nearly orthogonal fibers.")
    else:
        print("[MIXED] Features are entangled.")

if __name__ == "__main__":
    main()
