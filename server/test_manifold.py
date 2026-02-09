import os
import sys

os.environ["HF_HOME"] = r"D:\develop\model"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json

import torch
from structure_analyzer import ManifoldAnalysis

import transformer_lens


def test_manifold_analysis():
    print("Loading model (gpt2-small) for testing...")
    model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")
    
    print("Model loaded.")
    analyzer = ManifoldAnalysis(model)
    
    prompt = "The quick brown fox jumps over the lazy dog."
    tokens = model.to_tokens(prompt)
    
    print(f"Analyzing prompt: '{prompt}'")
    
    # Test Layer 0
    layer_idx = 0
    print(f"\nTesting Layer {layer_idx}...")
    
    activations = analyzer.get_layer_activations(tokens, layer_idx)
    print(f"Activation shape: {activations.shape}")
    
    # Test PCA
    print("Computing PCA...")
    pca_res = analyzer.compute_pca(activations, n_components=3)
    if "error" in pca_res:
        print(f"PCA Error: {pca_res['error']}")
    else:
        print("PCA keys:", pca_res.keys())
        print("Explained Variance Ratio:", pca_res["explained_variance_ratio"])
        
    # Test Intrinsic Dimensionality
    print("Estimating Intrinsic Dimensionality...")
    id_res = analyzer.estimate_intrinsic_dimensionality(activations)
    if "error" in id_res:
        print(f"ID Error: {id_res['error']}")
    else:
        print("ID result:", json.dumps(id_res, indent=2))
        
    print("\nTest Complete!")

if __name__ == "__main__":
    test_manifold_analysis()
