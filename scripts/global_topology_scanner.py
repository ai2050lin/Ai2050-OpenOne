
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

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_data():
    corpus_path = "d:/develop/TransformerLens-main/data/logic_core/logic_corpus_v1.txt"
    if os.path.exists(corpus_path):
        with open(corpus_path, "r", encoding="utf-8") as f:
            return f.read()
    return "AGI Research: The Neural Fiber Bundle Theory proposes that intelligence emerges from geometric alignment." * 100

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

class TopologyScanner:
    def __init__(self, model):
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def run_full_scan(self, layers=None):
        if layers is None: layers = list(range(self.model.cfg.n_layers))
        layer_names = [f"blocks.{i}.hook_resid_post" for i in layers]
        text = load_data()
        tokens = self.model.to_tokens(text)
        tokens = tokens[:, :min(tokens.shape[1], 1024)]
        
        print(f"Scanning layers: {layers}")
        with torch.no_grad():
            _, cache = self.model.run_with_cache(tokens, names_filter=layer_names)
            
        results = {}
        for l_idx, hook_name in tqdm(zip(layers, layer_names), total=len(layers)):
            acts = cache[hook_name].reshape(-1, self.model.cfg.d_model).cpu().numpy()
            pca = PCA(n_components=3)
            proj_3d = pca.fit_transform(acts)
            acts_std = (acts - acts.mean(axis=0)) / (acts.std(axis=0) + 1e-6)
            eps, b0 = compute_betti_curves(acts_std)
            results[str(l_idx)] = {
                "pca": proj_3d.tolist(),
                "betti": b0,
                "epsilon": eps.tolist(),
                "pca_mean": pca.mean_.tolist(),
                "pca_components": pca.components_.tolist()
            }
        return results

def scan_topology(model_type="gpt2"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing {model_type} on {device}...")
    
    try:
        if model_type == "gpt2":
            snapshot_path = r"D:\develop\model\hub\models--gpt2\snapshots\607a30d783dfa663caf39e06633721c8d4cfcd7e"
            from safetensors.torch import load_file as load_safetensors
            from transformer_lens import HookedTransformerConfig
            cfg = HookedTransformerConfig.from_dict({
                "d_model": 768, "d_head": 64, "n_heads": 12, "n_layers": 12,
                "n_ctx": 1024, "d_vocab": 50257, "act_fn": "gelu_new",
                "normalization_type": "LN", "model_name": "gpt2"
            })
            model = HookedTransformer(cfg).to(device)
            tokenizer = AutoTokenizer.from_pretrained(snapshot_path, local_files_only=True)
            tokenizer.pad_token = tokenizer.eos_token
            model.tokenizer = tokenizer
            state_dict = load_safetensors(os.path.join(snapshot_path, "model.safetensors"), device=device)
            model.load_state_dict(state_dict, strict=False)
            output_path = "tempdata/topology.json"
        
        elif model_type == "qwen3":
            # Use raw string backslashes confirmed by final_check.py
            snapshot_path = r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"
            print(f"Loading Qwen3 with manual config: {snapshot_path}")
            
            config = AutoConfig.from_pretrained(snapshot_path, local_files_only=True, trust_remote_code=True)
            hf_model = AutoModelForCausalLM.from_pretrained(
                snapshot_path, config=config, local_files_only=True, trust_remote_code=True,
                torch_dtype=torch.float16, device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(snapshot_path, local_files_only=True)
            model = HookedTransformer.from_pretrained(
                "qwen2.5-7b", hf_model=hf_model, device=device, tokenizer=tokenizer,
                fold_ln=False, center_writing_weights=False, center_unembed=False
            )
            output_path = "tempdata/topology_qwen3.json"
        
        print(f"Loading complete. Starting scan...")
        scanner = TopologyScanner(model)
        results = scanner.run_full_scan()
        
        if not os.path.exists("tempdata"): os.makedirs("tempdata")
        with open(output_path, "w") as f:
            json.dump({"layers": results}, f)
        print(f"Scan for {model_type} COMPLETE. Saved to {output_path}")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2", choices=["gpt2", "qwen3"])
    args = parser.parse_args()
    scan_topology(args.model)
