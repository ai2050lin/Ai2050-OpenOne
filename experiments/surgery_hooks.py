import torch


class ManifoldSurgeon:
    def __init__(self, model):
        self.model = model
        self.interventions = {} # layer_idx: activation_offset
        self.pca_basis = {} # layer_idx: PCA components (3, D)
        self.pca_mean = {} # layer_idx: PCA mean (D)
        self._is_active = False

    def set_pca_info(self, layer_idx, components, mean):
        """Stores PCA info to allow inverse projection from 3D to D"""
        self.pca_basis[layer_idx] = torch.tensor(components, dtype=torch.float32).to(self.model.cfg.device)
        self.pca_mean[layer_idx] = torch.tensor(mean, dtype=torch.float32).to(self.model.cfg.device)

    def add_intervention(self, layer_idx, coords_3d):
        """Calculates activation offset from 3D coordinate delta"""
        if layer_idx not in self.pca_basis:
            return False
            
        # Inverse project: delta_D = delta_3D * Basis
        # Note: components are (3, D)
        delta_3d = torch.tensor(coords_3d, dtype=torch.float32).to(self.model.cfg.device)
        delta_d = torch.matmul(delta_3d, self.pca_basis[layer_idx])
        
        self.interventions[layer_idx] = delta_d
        self._is_active = True
        return True

    def clear_interventions(self):
        self.interventions = {}
        self._is_active = False

    def hook_fn(self, activations, hook):
        layer_idx = hook.layer()
        if layer_idx in self.interventions:
            # Apply additive intervention
            return activations + self.interventions[layer_idx]
        return activations

    def activate(self):
        self.model.reset_hooks()
        for layer_idx in self.interventions:
            hook_name = f"blocks.{layer_idx}.hook_resid_post"
            self.model.add_hook(hook_name, self.hook_fn)

    def deactivate(self):
        self.model.reset_hooks()
        self._is_active = False
