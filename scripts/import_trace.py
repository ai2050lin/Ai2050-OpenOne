
import json
import os
import sys

import numpy as np
import torch
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Verified Path
snapshot_path = "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"

def compute_betti_curves(point_cloud, max_epsilon=10.0, steps=20):
    if len(point_cloud) > 1000:
        indices = np.random.choice(len(point_cloud), 1000, replace=False)
        point_cloud = point_cloud[indices]
    dists = squareform(pdist(point_cloud, metric='euclidean'))
    epsilons = np.linspace(0.1, max_epsilon, steps)
    betti_0_curve = []
    for eps in epsilons:
        adj = (dists < eps).astype(int)
        n_components, _ = connected_components(adj, directed=False)
        betti_0_curve.append(int(n_components))
    return epsilons, betti_0_curve

def run_qwen3_scan():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Qwen3 from: {snapshot_path}")
    
    hf_model = AutoModelForCausalLM.from_pretrained(
        snapshot_path, local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(snapshot_path, local_files_only=True)
    
    model = HookedTransformer.from_pretrained(
        "qwen2.5-7b", hf_model=hf_model, device=device, tokenizer=tokenizer,
        fold_ln=False, center_writing_weights=False, center_unembed=False
    )
    
    print(f"Model loaded. Starting scan ({model.cfg.n_layers} layers)...")
    
    # Simple data
    text = "AGI Research: The Neural Fiber Bundle Theory proposes that intelligence emerges from geometric alignment." * 100
    tokens = model.to_tokens(text)[:, :1024]
    
    layer_names = [f"blocks.{i}.hook_resid_post" for i in range(model.cfg.n_layers)]
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=layer_names)
        
    results = {}
    for i in tqdm(range(model.cfg.n_layers)):
        hook_name = f"blocks.{i}.hook_resid_post"
        acts = cache[hook_name].reshape(-1, model.cfg.d_model).cpu().numpy()
        pca = PCA(n_components=3)
        proj_3d = pca.fit_transform(acts)
        acts_std = (acts - acts.mean(axis=0)) / (acts.std(axis=0) + 1e-6)
        eps, b0 = compute_betti_curves(acts_std)
        results[str(i)] = {
            "pca": proj_3d.tolist(),
            "betti": b0,
            "epsilon": eps.tolist(),
            "pca_mean": pca.mean_.tolist(),
            "pca_components": pca.components_.tolist()
        }
        
    output_path = "tempdata/topology_qwen3.json"
    if not os.path.exists("tempdata"): os.makedirs("tempdata")
    with open(output_path, "w") as f:
        json.dump({"layers": results}, f)
    print(f"Qwen3 Scan Complete. Saved to {output_path}")

if __name__ == "__main__":
    run_qwen3_scan()
