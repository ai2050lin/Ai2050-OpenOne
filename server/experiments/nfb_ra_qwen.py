
import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from transformer_lens import HookedTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IsoCorpusGenerator:
    """
    Generates an Isomorphic Corpus (Iso-Corpus) to decouple Syntax (M) from Semantics (F).
    """
    def __init__(self):
        self.templates = [
            "The {} {} the {}.", # S-V-O
            "A {} is {} than a {}.", # Comparison
            "Why did the {} {} the {}?", # Question
        ]
        
        self.semantics = {
            "nouns": ["cat", "dog", "man", "woman", "robot", "alien"],
            "verbs": ["chased", "loved", "ate", "saw", "built", "destroyed"],
            "adjectives": ["red", "blue", "big", "small", "fast", "slow"]
        }

    def generate_batch(self, template_idx: int, batch_size: int = 100) -> List[str]:
        """Generates a batch of sentences with FIXED syntax (template) and RANDOM semantics."""
        template = self.templates[template_idx]
        batch = []
        import random
        for _ in range(batch_size):
            # Simple slot filling logic based on template structure
            # This is a simplified filler for demonstration
            noun1 = random.choice(self.semantics["nouns"])
            noun2 = random.choice(self.semantics["nouns"])
            verb = random.choice(self.semantics["verbs"])
            adj = random.choice(self.semantics["adjectives"])
            
            if template_idx == 0:
                sent = template.format(noun1, verb, noun2)
            elif template_idx == 1:
                sent = template.format(noun1, adj, noun2)
            elif template_idx == 2:
                sent = template.format(noun1, verb, noun2)
            else:
                sent = template.format(noun1) 
            
            batch.append(sent)
        return batch

class ManifoldExtractor:
    """
    Phase I: Extracts the topological features of the Base Manifold (M).
    """
    def __init__(self, model: HookedTransformer):
        self.model = model

    def get_centroid(self, texts: List[str], layer: int) -> np.ndarray:
        """
        Computes the centroid of activations for a set of isomorphic sentences.
        Integrates out the 'Fiber' fluctuations.
        """
        activations = []
        with torch.no_grad():
            for text in texts:
                # Run model and get residual stream at specific layer
                # We take the last token's activation for simplicity, or mean pooling
                _, cache = self.model.run_with_cache(text, names_filter=lambda x: f"blocks.{layer}.hook_resid_post" in x)
                act = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :].cpu().numpy()
                activations.append(act)
        
        # Calculate centroid (Integrate out Fiber)
        centroid = np.mean(np.array(activations), axis=0)
        return centroid

    def compute_topology(self, centroids: np.ndarray):
        """
        Computes Persistent Homology (Betti Numbers) of the manifold point cloud.
        """
        logging.info("Computing Persistent Homology for Manifold Structure...")
        try:
            from gtda.homology import VietorisRipsPersistence

            # Simple TDA pipeline
            VR = VietorisRipsPersistence(homology_dimensions=[0, 1, 2])
            diagrams = VR.fit_transform(centroids[None, :, :])
            logging.info(f"TDA Computed. Diagram shape: {diagrams.shape}")
            return diagrams
        except ImportError:
            logging.warning("giotto-tda not installed. Skipping TDA step.")
            return None

class FiberAnalyzer:
    """
    Phase II: Decomposes the Fiber Space (F) into independent semantic basis.
    """
    def __init__(self, model: HookedTransformer):
        self.model = model

    def analyze_fiber(self, template: str, layer: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs PCA on the Fiber Cloud (activations of same syntax, diff semantics).
        """
        # 1. Generate Fiber Cloud
        corpus = IsoCorpusGenerator()
        # We need a custom generation method here to fill the SPECIFIC template
        # For demo, we just reuse the generic batch generator logic slightly modified
        # In a real impl, we'd have precise slot filling.
        sentences = [template.replace("{}", w) for w in ["cat", "dog", "car", "apple"] for _ in range(1)] # Mock for now
        
        activations = []
        with torch.no_grad():
            for text in sentences:
                _, cache = self.model.run_with_cache(text, names_filter=lambda x: f"blocks.{layer}.hook_resid_post" in x)
                act = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :].cpu().numpy()
                activations.append(act)
        
        fiber_cloud = np.array(activations)
        
        # 2. Spectral Decomposition (PCA)
        pca = PCA(n_components=3) # Look for top 3 semantic dimensions
        pca.fit(fiber_cloud)
        
        explained_variance = pca.explained_variance_ratio_
        components = pca.components_
        
        return explained_variance, components

def generate_nfb_data(model=None) -> Dict:
    """
    Runs the NFB-RA analysis and returns the resulting data dictionary.
    """
    logging.info("Starting NFB-RA (Neural Fiber Bundle Reconstruction Algorithm)...")

    if model is None:
        # SYNTHETIC DATA GENERATION
        import numpy as np
        logging.info("Generating SYNTHETIC NFB data (No model provided)...")
        # Manifold: A noisy circle/torus in 3D
        t = np.linspace(0, 2*np.pi, 20)
        centroids_np = np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], axis=1) * 10
        
        # Fiber: Random basis
        variance = np.array([0.5, 0.3, 0.2])
        basis = np.eye(3) # Simple identity basis
    else:
        # REAL DATA GENERATION
        # 1. Phase I: Manifold Extraction
        manifold = ManifoldExtractor(model)
        generator = IsoCorpusGenerator()
        
        centroids = []
        layer_to_probe = 6 # Middle layer
        if hasattr(model, "cfg") and layer_to_probe >= model.cfg.n_layers:
            layer_to_probe = model.cfg.n_layers // 2
        
        logging.info("Step 1: Sampling Manifold Points (Syntax Templates)...")
        for i in range(len(generator.templates)):
            batch = generator.generate_batch(i, batch_size=20) # Reduced for speed in interactive mode
            centroid = manifold.get_centroid(batch, layer=layer_to_probe)
            centroids.append(centroid)
            logging.info(f"  - Extracted centroid for Template {i}")
        
        centroids_np = np.array(centroids)
        
        # Compute Topology (Optional)
        # manifold.compute_topology(centroids_np)
        
        # 2. Phase II: Fiber Analysis
        logging.info("Step 2: Analyzing Fiber Structure...")
        analyzer = FiberAnalyzer(model)
        # Simple template for fiber analysis
        target_template = "The {} is red." 
        variance, basis = analyzer.analyze_fiber(target_template, layer=layer_to_probe)
    
    logging.info(f"Fiber PCA Explained Variance: {variance}")

    output_data = {
        "manifold_centroids": centroids_np.tolist(), # The base points (p)
        "fiber_variance": variance.tolist(),         # Dimensions of the fiber (F_p)
        "fiber_basis": basis.tolist()                # The directions of the fiber
    }
    return output_data

def main():
    # 1. Load Model (Mocking Qwen availability in TransformerLens for now)
    # in reality, user would use: model = HookedTransformer.from_pretrained("Qwen/Qwen-1_8B-Chat", device="cuda")
    # check if we can load a small supported model for demo purposes
    model_name = "gpt2-small" 
    logging.info(f"Loading model: {model_name} (Proxy for Qwen3 in this demo)")
    model = None
    try:
        model = HookedTransformer.from_pretrained(model_name)
    except Exception as e:
        logging.warning(f"Failed to load model ({e}). Switching to SYNTHETIC DATA MODE for visualization dev.")
        pass

    output_data = generate_nfb_data(model)
    
    # 4. Save Data for Visualization
    import json
    import os

    # Save to frontend public folder for direct access during dev
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "public", "nfb_data.json")
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        logging.info(f"Data saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save data: {e}")

if __name__ == "__main__":
    main()
