
from typing import Dict, List, Optional

import numpy as np
import torch

from transformer_lens import HookedTransformer


class ManifoldSurgeon:

    def __init__(self, model: HookedTransformer, cache_dir: str = "nfb_data"):
        self.model = model
        self.hooks = []
        self.interventions = []
        self.cache_dir = cache_dir
        self.hidden_states = None
        self._load_data()

    def _load_data(self):
        import os
        path = os.path.join(self.cache_dir, "hidden_states.npy")
        if os.path.exists(path):
            self.hidden_states = np.load(path)
            print(f"ManifoldSurgeon loaded hidden states: {self.hidden_states.shape}")
        else:
            print("ManifoldSurgeon Warning: No hidden states found. Grafting may fail.")

    def clear_surgery(self):
        """Removes all surgical hooks."""
        self.model.reset_hooks()
        self.hooks = []
        self.interventions = []
        print("All surgical hooks removed.")

    def graft_connection_by_id(self, source_id: int, target_id: int, layer: int, strength: float = 1.0):
        """
        Grafts a connection using Point IDs. 
        """
        if self.hidden_states is None:
            print("Error: Hidden states not loaded.")
            return

        # Calculate steering vector in High-Dim space
        v_source = torch.tensor(self.hidden_states[source_id], device=self.model.cfg.device, dtype=self.model.cfg.dtype)
        v_target = torch.tensor(self.hidden_states[target_id], device=self.model.cfg.device, dtype=self.model.cfg.dtype)
        
        steering_vec = (v_target - v_source)
        steering_vec = steering_vec / torch.norm(steering_vec) * strength

        hook_name = f"blocks.{layer}.hook_resid_post"
        
        def steering_hook(resid_post, hook):
            # resid_post: [batch, pos, d_model]
            resid_post += steering_vec
            return resid_post

        self.model.add_hook(hook_name, steering_hook)
        self.hooks.append(hook_name)
        
        self.interventions.append({
            "type": "graft",
            "source": source_id,
            "target": target_id,
            "layer": layer,
            "strength": strength
        })
        print(f"Grafted connection {source_id} -> {target_id} at Layer {layer}")

    def ablate_concept(self, point_id: int, layer: int, radius: float = 0.5):
        """
        Ablates a concept by suppressing activations that are similar to the target point.
        """
        if self.hidden_states is None:
            return

        v_concept = torch.tensor(self.hidden_states[point_id], device=self.model.cfg.device, dtype=self.model.cfg.dtype)
        
        hook_name = f"blocks.{layer}.hook_resid_post"
        
        def ablation_hook(resid_post, hook):
            # Simple subtraction for now
            resid_post -= v_concept * 1.5 
            return resid_post
            
        self.model.add_hook(hook_name, ablation_hook)
        self.hooks.append(hook_name)
        
        self.interventions.append({
            "type": "ablation",
            "target": point_id,
            "layer": layer
        })
        print(f"Ablated concept {point_id} at Layer {layer}")


