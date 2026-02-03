
import json
import random

import numpy as np
import torch
from sklearn.linear_model import LinearRegression, Ridge
from tqdm import tqdm

from transformer_lens import HookedTransformer


class TransportLearner:
    """
    Step 4: Connection/Transport Learning.
    Learns the parallel transport operator T that maps activations between semantic states.
    Assumption: v_B = T * v_A for a systematic semantic shift.
    """
    def __init__(self, model_name="gpt2-small", device="cpu"):
        self.device = device
        print(f"Loading model {model_name}...")
        try:
            self.model = HookedTransformer.from_pretrained(model_name, device=device)
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model = None
            
        self.semantics = {
             "Subj": ["scientist", "engineer", "artist", "doctor"],
             # Simple subset for testing transport
        }
        self.template = "The {Subj} analyzed the structure"

    def generate_pairs(self, n_pairs=500):
        """
        Generate pairs (A, B) where we swap the Subject.
        """
        pairs = []
        for _ in range(n_pairs):
            s1 = random.choice(self.semantics["Subj"])
            s2 = random.choice(self.semantics["Subj"])
            while s2 == s1:
                s2 = random.choice(self.semantics["Subj"])
                
            text_a = self.template.replace("{Subj}", s1)
            text_b = self.template.replace("{Subj}", s2)
            
            pairs.append((text_a, text_b))
        return pairs

    def get_activations_batch(self, texts, layer_idx=6):
        if self.model is None:
            return np.random.randn(len(texts), 768)
        
        # Simple batching (size=1 loop for simplicity or proper batching)
        activations = []
        with torch.no_grad():
            for text in texts:
                _, cache = self.model.run_with_cache(text)
                act = cache[f"blocks.{layer_idx}.hook_resid_post"]
                last_token_act = act[0, -1, :].cpu().numpy()
                activations.append(last_token_act)
        return np.array(activations)

    def learn_transport(self, n_samples=200):
        print(f"Generating {n_samples} training pairs...")
        pairs = self.generate_pairs(n_pairs=n_samples)
        
        texts_a = [p[0] for p in pairs]
        texts_b = [p[1] for p in pairs]
        
        print("Extracting activations...")
        acts_a = self.get_activations_batch(texts_a)
        acts_b = self.get_activations_batch(texts_b)
        
        # Learn T: acts_a -> acts_b
        # Linear Regression: B = A * T_transposed + bias
        print("Training Transport Operator (Ridge Regression)...")
        reg = Ridge(alpha=1.0)
        reg.fit(acts_a, acts_b)
        
        score = reg.score(acts_a, acts_b)
        print(f"Training R2 Score: {score:.4f}")
        
        return reg

    def verify_transport(self, transport_model, n_test=50):
        print(f"Verifying on {n_test} test pairs...")
        pairs = self.generate_pairs(n_pairs=n_test)
        texts_a = [p[0] for p in pairs]
        texts_b = [p[1] for p in pairs]
        
        acts_a = self.get_activations_batch(texts_a)
        acts_b_true = self.get_activations_batch(texts_b)
        
        acts_b_pred = transport_model.predict(acts_a)
        
        # Calculate Cosine Similarity between Pred and True
        cos_sims = []
        for i in range(len(acts_b_true)):
            v_true = acts_b_true[i]
            v_pred = acts_b_pred[i]
            sim = np.dot(v_true, v_pred) / (np.linalg.norm(v_true) * np.linalg.norm(v_pred))
            cos_sims.append(sim)
            
        avg_sim = np.mean(cos_sims)
        print(f"Average Cosine Similarity (Pred vs True): {avg_sim:.4f}")
        return avg_sim

if __name__ == "__main__":
    learner = TransportLearner()
    transport_model = learner.learn_transport()
    learner.verify_transport(transport_model)
    # import joblib
    # joblib.dump(transport_model, "data/transport_operator.pkl")
