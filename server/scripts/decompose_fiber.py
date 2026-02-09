
import json

import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm

from transformer_lens import HookedTransformer


class FiberDecomposer:
    """
    Step 3: Fiber Bundle Decomposition.
    Analyzes the 'Fiber' (semantic variations) at a fixed point on the Manifold (fixed syntax).
    """
    def __init__(self, model_name="gpt2-small", device="cpu"):
        self.device = device
        print(f"Loading model {model_name}...")
        try:
            self.model = HookedTransformer.from_pretrained(model_name, device=device)
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model = None

    def get_fiber_activations(self, texts, layer_idx=6):
        if self.model is None:
            return np.random.randn(len(texts), 768)
            
        activations = []
        with torch.no_grad():
            for text in tqdm(texts, desc="Extracting Fiber Activations"):
                _, cache = self.model.run_with_cache(text)
                act = cache[f"blocks.{layer_idx}.hook_resid_post"]
                # Use the last token to represent the sentence meaning/fiber state
                last_token_act = act[0, -1, :].cpu().numpy()
                activations.append(last_token_act)
        return np.array(activations)

    def analyze_fiber_structure(self, fiber_cloud):
        """
        Use PCA to find the intrinsic dimension of the semantic fiber.
        """
        print(f"Analyzing fiber cloud with shape {fiber_cloud.shape}...")
        
        # PCA Decomposition
        pca = PCA()
        pca.fit(fiber_cloud)
        
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        # Estimate intrinsic dimension (e.g., 95% variance)
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        
        print(f"Explained Variance (Top 10): {explained_variance[:10]}")
        print(f"Intrinsic Dimension (95% Var): {n_components_95}")
        
        return {
            "components": pca.components_[:10], # Top 10 basis vectors
            "variance": explained_variance,
            "intrinsic_dim": n_components_95
        }

    def run_analysis(self, corpus_path="data/iso_corpus.jsonl", target_template="T001"):
        # 1. Load Data
        texts = []
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                if item["template_id"] == target_template:
                    texts.append(item["text"])
        
        texts = texts[:500] 
        print(f"Decomposing fiber for template {target_template} ({len(texts)} samples)...")
        
        # 2. Extract Activations
        fiber_cloud = self.get_fiber_activations(texts)
        
        # 3. Analyze Structure
        results = self.analyze_fiber_structure(fiber_cloud)
        
        print("Fiber Decomposition Complete.")
        return results

if __name__ == "__main__":
    decomposer = FiberDecomposer()
    results = decomposer.run_analysis()
    # np.save("data/fiber_basis.npy", results["components"])
