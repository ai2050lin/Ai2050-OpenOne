from typing import Dict, List, Optional

import numpy as np
import torch

import transformer_lens


class GeometricInterceptor:
    """
    Geometric Interceptor: Implements 'Bias Decoupling' using orthogonal rotation.
    
    Physics Principle: 
    If a bias direction exists in the residual stream (e.g., 'Gender' in 'Doctor'),
    the transport matrix R rotates the neutral concept to the biased one.
    To debias, we apply the inverse rotation R^T (which is also R^-1 due to orthogonality).
    """
    def __init__(self, model: transformer_lens.HookedTransformer):
        self.model = model
        self.active_interceptors = {} # Map of (layer, hook_name) -> R_inverse

    def add_interception(self, layer_idx: int, R: np.ndarray, hook_name: str = "hook_resid_post"):
        """
        Adds a geometric interception point.
        R: The transport matrix found during RPT.
        """
        # Since R is orthogonal, R_inv = R.T
        R_inv = R.T
        R_torch = torch.from_numpy(R_inv).to(next(self.model.parameters()).device).to(next(self.model.parameters()).dtype)
        
        full_hook_name = f"blocks.{layer_idx}.{hook_name}"
        self.active_interceptors[full_hook_name] = R_torch
        print(f"[GeometricInterceptor] Interceptor added to {full_hook_name}")

    def get_hook_fn(self, hook_name: str):
        """Returns the hook function for a specific interception point."""
        R_inv = self.active_interceptors.get(hook_name)
        
        def debias_hook(resid, hook):
            if R_inv is None:
                return resid
            # Apply the inverse rotation to 'neutralize' the space
            # resid: [batch, pos, d_model]
            # R_inv: [d_model, d_model]
            return resid @ R_inv
            
        return debias_hook

    def apply_hooks(self):
        """Registers all active interceptors to the model."""
        self.model.reset_hooks()
        for hook_name in self.active_interceptors.keys():
            self.model.add_hook(hook_name, self.get_hook_fn(hook_name))
        print(f"[GeometricInterceptor] {len(self.active_interceptors)} hooks applied.")

    def clear(self):
        self.active_interceptors = {}
        self.model.reset_hooks()

def run_debias_experiment(model, prompt: str, layer_idx: int, R: np.ndarray):
    """
    Runs a comparison between baseline and geometrically debiased state.
    """
    interceptor = GeometricInterceptor(model)
    
    # 1. Baseline
    model.reset_hooks()
    logits_base = model(prompt)
    
    # 2. Debiased
    interceptor.add_interception(layer_idx, R)
    interceptor.apply_hooks()
    logits_debiased = model(prompt)
    
    # Clean up
    model.reset_hooks()
    
    return logits_base, logits_debiased
