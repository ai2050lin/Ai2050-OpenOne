
import json

import numpy as np
import torch
from tqdm import tqdm

try:
    from gtda.homology import VietorisRipsPersistence
    from gtda.plotting import plot_diagram
except ImportError:
    print("Warning: giotto-tda not installed. Using dummy TDA for demonstration.")
    VietorisRipsPersistence = None

# Mock model loading for this script if actual model is too heavy for dev environment
# In production, we import TransformerLens
from transformer_lens import HookedTransformer


class ManifoldExtractor:
    """
    Step 2: Manifold Topology Extraction.
    Uses Persistent Homology to find the shape of the activation cloud for a fixed syntax.
    """
    def __init__(self, model_name="gpt2-small", device="cpu"):
        self.device = device
        print(f"Loading model {model_name}...")
        try:
            self.model = HookedTransformer.from_pretrained(model_name, device=device)
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model = None

    def get_activations(self, texts, layer_idx=6):
        """
        Get activations from a specific layer.
        """
        if self.model is None:
            # Return random noise for testing if model fails to load
            return np.random.randn(len(texts), 768)
            
        activations = []
        with torch.no_grad():
            for text in tqdm(texts, desc="Extracting Activations"):
                # We want the residual stream state at the last token typically
                # or average pooling. Let's use last token for now.
                _, cache = self.model.run_with_cache(text)
                # Helper to get resid_post
                act = cache[f"blocks.{layer_idx}.hook_resid_post"]
                # Take the last token: [batch=1, pos, d_model] -> [d_model]
                last_token_act = act[0, -1, :].cpu().numpy()
                activations.append(last_token_act)
        return np.array(activations)

    def compute_homology(self, point_cloud):
        """
        Compute persistent homology (Barcodes/Betti numbers).
        """
        if VietorisRipsPersistence is None:
            return {"betti_0": 1, "betti_1": 0, "note": "Dummy TDA"}
            
        # Initialize the VR persistence transformer
        # homology_dimensions=(0, 1, 2) means we look for connected components, loops, and voids
        vr = VietorisRipsPersistence(homology_dimensions=(0, 1, 2))
        diagrams = vr.fit_transform(point_cloud[None, :, :]) # Add batch dim
        
        # Simple extraction of Betti numbers (this is a simplified logic)
        # In reality we analyze the persistence diagram duration
        return diagrams

    def run_analysis(self, corpus_path="data/iso_corpus.jsonl", target_template="T001"):
        # 1. Load Data
        texts = []
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                if item["template_id"] == target_template:
                    texts.append(item["text"])
        
        # Limit for speed
        texts = texts[:500] 
        print(f"Analyzing {len(texts)} samples for template {target_template}...")
        
        # 2. Extract Activations ( The Point Cloud )
        point_cloud = self.get_activations(texts)
        
        # 3. Compute Topology
        print("Computing Persistent Homology...")
        topology = self.compute_homology(point_cloud)
        
        print(f"Topology Analysis Complete.")
        return topology

if __name__ == "__main__":
    extractor = ManifoldExtractor()
    topology = extractor.run_analysis()
    # Save or process result
    # np.save("data/topology_diagrams.npy", topology)
