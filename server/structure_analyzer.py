"""
Neural Network Structure Extraction Module

This module implements three core algorithms for extracting mathematical structures
from neural networks:
1. Circuit Discovery - Find minimal computational subgraphs
2. Sparse Feature Extraction - Discover interpretable features
3. Causal Mediation Analysis - Quantify component influence
"""

import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class CircuitNode:
    """Represents a node in the computational circuit"""
    layer_idx: int
    component_type: str  # "attn_head", "mlp", "embed", "unembed"
    head_idx: Optional[int] = None
    importance: float = 0.0
    
    def __hash__(self):
        return hash((self.layer_idx, self.component_type, self.head_idx))
    
    def __str__(self):
        if self.head_idx is not None:
            return f"L{self.layer_idx}.{self.component_type}.H{self.head_idx}"
        return f"L{self.layer_idx}.{self.component_type}"


class CircuitDiscovery:
    """
    Automatic circuit discovery using activation patching.
    
    This class implements the activation patching technique to identify
    minimal computational circuits that solve specific tasks.
    """
    
    def __init__(self, model, metric_fn: Callable[[Tensor], float]):
        """
        Args:
            model: HookedTransformer model
            metric_fn: Function that takes model output and returns a scalar metric
                      (higher = better performance on task)
        """
        self.model = model
        self.metric_fn = metric_fn
        self.baseline_cache = None
        self.clean_cache = None
        
    def run_with_cache(self, tokens: Tensor) -> Tuple[Tensor, Dict]:
        """Run model and return output with activation cache"""
        with torch.no_grad():
            logits, cache = self.model.run_with_cache(tokens)
        return logits, cache
    
    def patch_activation(
        self, 
        clean_tokens: Tensor,
        corrupted_tokens: Tensor,
        hook_point: str,
        component_idx: Optional[int] = None
    ) -> float:
        """
        Patch a single component's activation and measure metric change.
        
        Args:
            clean_tokens: Normal input tokens
            corrupted_tokens: Corrupted/counterfactual input tokens
            hook_point: Name of the hook point to patch
            component_idx: Index of component (e.g., head_idx for attention)
            
        Returns:
            Metric value after patching
        """
        # Get clean and corrupted caches
        _, clean_cache = self.run_with_cache(clean_tokens)
        _, corrupted_cache = self.run_with_cache(corrupted_tokens)
        
        def patch_hook(activation, hook):
            """Hook function to patch activations"""
            if component_idx is not None:
                # Patch specific component (e.g., attention head)
                activation[:, :, component_idx] = clean_cache[hook.name][:, :, component_idx]
            else:
                # Patch entire activation
                activation[:] = clean_cache[hook.name]
            return activation
        
        # Run corrupted input with patched activation
        with self.model.hooks([(hook_point, patch_hook)]):
            patched_logits = self.model(corrupted_tokens)
        
        return self.metric_fn(patched_logits)
    
    def discover_circuit(
        self,
        clean_tokens: Tensor,
        corrupted_tokens: Tensor,
        threshold: float = 0.1,
        target_layer: Optional[int] = None
    ) -> Tuple[List[CircuitNode], nx.DiGraph]:
        """
        Discover computational circuit for a task.
        
        Args:
            clean_tokens: Normal input that solves the task
            corrupted_tokens: Corrupted input that fails the task
            threshold: Importance threshold for including components
            
        Returns:
            List of important circuit nodes and dependency graph
        """
        # Baseline metrics
        clean_logits = self.model(clean_tokens)
        corrupted_logits = self.model(corrupted_tokens)
        
        clean_metric = self.metric_fn(clean_logits)
        corrupted_metric = self.metric_fn(corrupted_logits)
        metric_range = clean_metric - corrupted_metric
        
        important_nodes = []
        
        # Test each component
        n_layers = self.model.cfg.n_layers
        n_heads = self.model.cfg.n_heads
        
        # Test attention heads
        layers_to_scan = [target_layer] if target_layer is not None else range(n_layers)
        
        for layer in layers_to_scan:
            for head in range(n_heads):
                hook_point = f"blocks.{layer}.attn.hook_result"
                patched_metric = self.patch_activation(
                    clean_tokens, corrupted_tokens, hook_point, head
                )
                
                # Calculate importance as metric recovery
                importance = (patched_metric - corrupted_metric) / (metric_range + 1e-8)
                
                if importance > threshold:
                    node = CircuitNode(
                        layer_idx=layer,
                        component_type="attn_head",
                        head_idx=head,
                        importance=float(importance)
                    )
                    important_nodes.append(node)
        
        # Test MLPs
        for layer in layers_to_scan:
            hook_point = f"blocks.{layer}.mlp.hook_post"
            patched_metric = self.patch_activation(
                clean_tokens, corrupted_tokens, hook_point
            )
            
            importance = (patched_metric - corrupted_metric) / (metric_range + 1e-8)
            
            if importance > threshold:
                node = CircuitNode(
                    layer_idx=layer,
                    component_type="mlp",
                    importance=float(importance)
                )
                important_nodes.append(node)
        
        # Build dependency graph
        circuit_graph = self._build_circuit_graph(important_nodes)
        
        return important_nodes, circuit_graph
    
    def _build_circuit_graph(self, nodes: List[CircuitNode]) -> nx.DiGraph:
        """Build directed graph showing dependencies between components"""
        G = nx.DiGraph()
        
        # Add nodes
        for node in nodes:
            G.add_node(str(node), **node.__dict__)
        
        # Add edges based on layer ordering (simplified)
        # In practice, you'd use attention patterns to determine actual dependencies
        sorted_nodes = sorted(nodes, key=lambda n: (n.layer_idx, n.component_type))
        for i, node1 in enumerate(sorted_nodes):
            for node2 in sorted_nodes[i+1:]:
                if node2.layer_idx > node1.layer_idx:
                    # Later layers depend on earlier layers
                    G.add_edge(str(node1), str(node2), weight=node2.importance)
        
        return G


class SparseAutoEncoder(nn.Module):
    """
    Sparse Autoencoder for discovering interpretable features.
    
    Trains on layer activations to find a sparse overcomplete basis
    that represents the network's internal representations.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        sparsity_coefficient: float = 1e-3
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_coefficient = sparsity_coefficient
        
        # Encoder: input -> sparse features
        self.encoder = nn.Linear(input_dim, hidden_dim)
        
        # Decoder: sparse features -> reconstruction
        self.decoder = nn.Linear(hidden_dim, input_dim)
        
        # Initialize decoder as transpose of encoder (tied weights work well)
        self.decoder.weight.data = self.encoder.weight.data.t()
    
    def encode(self, x: Tensor) -> Tensor:
        """Encode input to sparse feature space"""
        return F.relu(self.encoder(x))
    
    def decode(self, features: Tensor) -> Tensor:
        """Decode features back to input space"""
        return self.decoder(features)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass returning reconstruction and features.
        
        Returns:
            reconstruction, features
        """
        features = self.encode(x)
        reconstruction = self.decode(features)
        return reconstruction, features
    
    def loss(self, x: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute loss with reconstruction and sparsity terms.
        
        Returns:
            total_loss, loss_dict
        """
        reconstruction, features = self.forward(x)
        
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstruction, x)
        
        # Sparsity loss (L1 on features)
        sparsity_loss = torch.mean(torch.abs(features))
        
        total_loss = recon_loss + self.sparsity_coefficient * sparsity_loss
        
        loss_dict = {
            "total": total_loss.item(),
            "reconstruction": recon_loss.item(),
            "sparsity": sparsity_loss.item(),
            "active_features": (features > 0).float().mean().item()
        }
        
        return total_loss, loss_dict
    
    def train_on_activations(
        self,
        activations: Tensor,
        n_epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3
    ) -> List[Dict[str, float]]:
        """
        Train SAE on a dataset of activations.
        
        Args:
            activations: [n_samples, input_dim] activation tensor
            n_epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            
        Returns:
            List of loss dictionaries for each epoch
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        n_samples = activations.shape[0]
        history = []
        
        self.train()
        for epoch in range(n_epochs):
            epoch_losses = []
            
            # Shuffle data
            perm = torch.randperm(n_samples)
            
            for i in range(0, n_samples, batch_size):
                batch_idx = perm[i:i+batch_size]
                batch = activations[batch_idx]
                
                optimizer.zero_grad()
                loss, loss_dict = self.loss(batch)
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss_dict)
            
            # Average losses for epoch
            avg_loss = {
                k: np.mean([d[k] for d in epoch_losses])
                for k in epoch_losses[0].keys()
            }
            history.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: {avg_loss}")
        
        self.eval()
        return history
    
    def get_feature_activations(
        self,
        activations: Tensor
    ) -> Tuple[Tensor, List[Dict[str, Any]]]:
        """
        Get feature activations and statistics.
        
        Returns:
            features: [n_samples, hidden_dim]
            feature_stats: List of dicts with feature statistics
        """
        with torch.no_grad():
            features = self.encode(activations)
        
        feature_stats = []
        for i in range(self.hidden_dim):
            feature_vec = features[:, i]
            active_mask = feature_vec > 0
            
            stats = {
                "feature_idx": i,
                "activation_frequency": active_mask.float().mean().item(),
                "mean_activation": feature_vec[active_mask].mean().item() if active_mask.any() else 0.0,
                "max_activation": feature_vec.max().item(),
                "decoder_norm": self.decoder.weight[:, i].norm().item()
            }
            feature_stats.append(stats)
        
        return features, feature_stats


class CausalMediation:
    """
    Causal mediation analysis for quantifying component influence.
    
    Computes direct and indirect causal effects of components on outputs
    using interventional analysis.
    """
    
    def __init__(self, model):
        self.model = model
    
    def compute_direct_effect(
        self,
        tokens: Tensor,
        hook_point: str,
        intervention_fn: Callable[[Tensor], Tensor],
        metric_fn: Callable[[Tensor], float]
    ) -> float:
        """
        Compute direct causal effect of a component.
        
        Args:
            tokens: Input tokens
            hook_point: Hook point to intervene on
            intervention_fn: Function to modify activations
            metric_fn: Function to measure output
            
        Returns:
            Direct effect (change in metric)
        """
        # Baseline
        baseline_logits = self.model(tokens)
        baseline_metric = metric_fn(baseline_logits)
        
        # Intervene
        def intervention_hook(activation, hook):
            return intervention_fn(activation)
        
        with self.model.hooks([(hook_point, intervention_hook)]):
            intervened_logits = self.model(tokens)
        
        intervened_metric = metric_fn(intervened_logits)
        
        return intervened_metric - baseline_metric
    
    def compute_total_effect(
        self,
        clean_tokens: Tensor,
        ablated_tokens: Tensor,
        metric_fn: Callable[[Tensor], float]
    ) -> float:
        """Compute total effect by comparing clean vs ablated inputs"""
        clean_metric = metric_fn(self.model(clean_tokens))
        ablated_metric = metric_fn(self.model(ablated_tokens))
        return clean_metric - ablated_metric
    
    def analyze_component_importance(
        self,
        tokens: Tensor,
        metric_fn: Callable[[Tensor], float],
        ablation_value: float = 0.0,
        target_layer: Optional[int] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze causal importance of all components.
        
        Args:
            tokens: Input tokens
            metric_fn: Metric function
            ablation_value: Value to set when ablating
            
        Returns:
            Dictionary mapping component names to importance scores
        """
        importance_scores = {}
        
        n_layers = self.model.cfg.n_layers
        n_heads = self.model.cfg.n_heads
        total_components = n_layers * n_heads + n_layers
        analyzed = 0
        
        print(f"    Analyzing {total_components} components ({n_layers} layers × {n_heads} heads + {n_layers} MLPs)...")
        
        # Analyze attention heads
        layers_to_scan = [target_layer] if target_layer is not None else range(n_layers)
        
        for layer in layers_to_scan:
            print(f"    Layer {layer}/{n_layers-1}: Analyzing {n_heads} attention heads...")
            for head in range(n_heads):
                hook_point = f"blocks.{layer}.attn.hook_result"
                
                def ablate_head(activation):
                    activation = activation.clone()
                    activation[:, :, head] = ablation_value
                    return activation
                
                effect = self.compute_direct_effect(
                    tokens, hook_point, ablate_head, metric_fn
                )
                
                component_name = f"L{layer}.attn.H{head}"
                importance_scores[component_name] = {
                    "direct_effect": abs(effect),
                    "layer": layer,
                    "type": "attention",
                    "head": head
                }
                analyzed += 1
            
            # Analyze MLP for this layer
            hook_point = f"blocks.{layer}.mlp.hook_post"
            
            def ablate_mlp(activation):
                return torch.zeros_like(activation)
            
            effect = self.compute_direct_effect(
                tokens, hook_point, ablate_mlp, metric_fn
            )
            
            component_name = f"L{layer}.mlp"
            importance_scores[component_name] = {
                "direct_effect": abs(effect),
                "layer": layer,
                "type": "mlp"
            }
            analyzed += 1
            print(f"    Layer {layer}/{n_layers-1}: Complete ({analyzed}/{total_components} components)")
        
        return importance_scores
    
    def build_causal_graph(
        self,
        importance_scores: Dict[str, Dict[str, float]],
        threshold: float = 0.01
    ) -> nx.DiGraph:
        """
        Build causal graph from importance scores.
        
        Args:
            importance_scores: Output from analyze_component_importance
            threshold: Minimum importance to include
            
        Returns:
            NetworkX directed graph
        """
        G = nx.DiGraph()
        
        # Add nodes for important components
        important_components = {
            name: scores for name, scores in importance_scores.items()
            if scores["direct_effect"] > threshold
        }
        
        for name, scores in important_components.items():
            G.add_node(name, **scores)
        
        # Add edges based on layer ordering
        components = list(important_components.items())
        for i, (name1, scores1) in enumerate(components):
            for name2, scores2 in components[i+1:]:
                if scores2["layer"] > scores1["layer"]:
                    # Add edge weighted by downstream importance
                    G.add_edge(name1, name2, weight=scores2["direct_effect"])
        
        return G


def export_graph_to_json(G: nx.DiGraph) -> Dict[str, Any]:
    """Export NetworkX graph to JSON-serializable format"""

class ManifoldAnalysis:
    """
    Analyzes the geometric structure of neural representations using manifold learning.
    
     techniques:
    1. PCA/SVD for dimensionality reduction
    2. Intrinsic Dimensionality Estimation
    3. Representation Geometry Analysis
    """
    
    def __init__(self, model):
        self.model = model
        
    def get_layer_activations(self, tokens: Tensor, layer_idx: int) -> Tensor:
        """
        Get activations for a specific layer.
        Returns: [batch * seq_len, d_model]
        """
        _, cache = self.model.run_with_cache(tokens)
        
        # Try different hook points
        potential_hooks = [
            f"blocks.{layer_idx}.hook_resid_post",
            f"blocks.{layer_idx}.hook_mlp_out",
            f"blocks.{layer_idx}.attn.hook_z"
        ]
        
        acts = None
        for hook in potential_hooks:
            if hook in cache:
                acts = cache[hook]
                break
                
        if acts is None:
            raise ValueError(f"Could not find valid activations for layer {layer_idx}")
            
        # Flatten batch and sequence dimensions
        if len(acts.shape) == 3: # [batch, seq, d_model]
            acts = acts.reshape(-1, acts.shape[-1])
        elif len(acts.shape) == 4: # [batch, seq, n_heads, d_head] for attn
            acts = acts.reshape(-1, acts.shape[-1] * acts.shape[-2])
            
        return acts.float() # Ensure float32

    def compute_pca(self, data: Tensor, n_components: int = 3) -> Dict[str, Any]:
        """
        Compute PCA using PyTorch SVD.
        """
        # Center the data
        mean = torch.mean(data, dim=0)
        centered_data = data - mean
        
        # SVD
        try:
            U, S, V = torch.pca_lowrank(centered_data, q=n_components, center=False, niter=2)
            projected = torch.matmul(centered_data, V[:, :n_components])
            
            explained_variance = (S[:n_components] ** 2) / (data.shape[0] - 1)
            total_variance = torch.var(data, dim=0, unbiased=True).sum()
            explained_variance_ratio = explained_variance / total_variance
            
            return {
                "projections": projected.cpu().numpy().tolist(),
                "components": V[:, :n_components].cpu().numpy().tolist(),
                "explained_variance": explained_variance.cpu().numpy().tolist(),
                "explained_variance_ratio": explained_variance_ratio.cpu().numpy().tolist()
            }
        except Exception as e:
            print(f"PCA Error: {e}")
            return {"error": str(e)}

    def estimate_intrinsic_dimensionality(self, data: Tensor) -> Dict[str, float]:
        """
        Estimate intrinsic dimensionality using multiple methods.
        1. Participation Ratio (on covariance eigenvalues)
        2. Explained Variance Threshold (95%)
        """
        # Center data
        centered_data = data - data.mean(dim=0)
        
        # Compute Covariance Matrix (or Gram matrix if N < D for speed)
        n, d = centered_data.shape
        if n < d:
             cov = torch.matmul(centered_data, centered_data.t()) / (n - 1)
        else:
             cov = torch.matmul(centered_data.t(), centered_data) / (n - 1)
             
        # Eigenvalues
        try:
            eigs = torch.linalg.eigvalsh(cov)
            eigs = eigs[eigs > 1e-6] # Filter numerical noise
            eigs = torch.sort(eigs, descending=True).values
            
            # 1. Participation Ratio: (sum(e))^2 / sum(e^2)
            sum_e = torch.sum(eigs)
            sum_e2 = torch.sum(eigs ** 2)
            pr = (sum_e ** 2) / sum_e2
            
            # 2. Explained Variance Thresholds
            cumsum = torch.cumsum(eigs, dim=0)
            total_var = torch.sum(eigs)
            
            def get_dim_at_threshold(threshold):
                mask = (cumsum / total_var) >= threshold
                if not mask.any():
                    return len(eigs)
                return torch.argmax(mask.int()).item() + 1
            
            return {
                "participation_ratio": float(pr.item()),
                "components_90": int(get_dim_at_threshold(0.90)),
                "components_95": int(get_dim_at_threshold(0.95)),
                "components_99": int(get_dim_at_threshold(0.99))
            }
        except Exception as e:
            print(f"ID Estimation Error: {e}")
            return {"error": str(e)}


class LanguageValidity:
    """
    Computes 'Language Validity' metrics based on the mathematical model:
    V(M, x) = Info_Validity + Geometric_Validity + Structural_Validity
    """
    
    def __init__(self, model):
        self.model = model
        
    def compute_perplexity(self, text: str, stride: int = 512) -> float:
        """
        Compute perplexity (exp of cross entropy).
        """
        tokens = self.model.to_tokens(text)
        
        # Sliding window
        # Note: HookedTransformer loss() usually handles batching, but we do simple manual window
        # for clarity on long texts if needed.
        # Actually, self.model(tokens, return_type="loss") is easiest for single batch.
        
        with torch.no_grad():
            loss = self.model(tokens, return_type="loss")
            
        return torch.exp(loss).item()

    def compute_entropy_profile(self, text: str) -> Dict[str, float]:
        """
        Compute statistics of the entropy distribution of next-token predictions.
        """
        tokens = self.model.to_tokens(text)
        
        with torch.no_grad():
            logits = self.model(tokens) # [batch, seq_len, vocab_size]
            
        # Logits -> Probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Entropy = - sum(p * log p)
        # Avoid log(0)
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs, dim=-1) # [batch, seq_len]
        
        # Skip the first position if it's BOS? 
        # Usually we care about prediction of the next token, so entropy at pos i 
        # is uncertainty about token i+1.
        
        mean_entropy = entropy.mean().item()
        var_entropy = entropy.var().item() if entropy.numel() > 1 else 0.0
        
        return {
            "mean_entropy": mean_entropy,
            "variance_entropy": var_entropy,
            "min_entropy": entropy.min().item(),
            "max_entropy": entropy.max().item()
        }

    def compute_anisotropy(self, text: str, layer_idx: int) -> float:
        """
        Compute anisotropy of representations in a specific layer.
        Defined as average cosine similarity between all pairs of tokens in the sequence.
        High anisotropy (close to 1.0) means Representation Collapse (bad).
        """
        tokens = self.model.to_tokens(text)
        
        _, cache = self.model.run_with_cache(tokens)
        
        # Get activations for layer (resid_post is standard for 'representation')
        hook_name = f"blocks.{layer_idx}.hook_resid_post"
        if hook_name not in cache:
            return 0.0
            
        acts = cache[hook_name] # [batch, seq_len, d_model]
        
        # Flatten to [seq_len, d_model]
        if acts.shape[0] == 1:
            acts = acts.squeeze(0)
        else:
            # If batch > 1, just take first sequence for simplicity or flatten all
            acts = acts.reshape(-1, acts.shape[-1])
            
        n_tokens = acts.shape[0]
        if n_tokens < 2:
            return 0.0
            
        # Global cosine similarity
        # Normalize vectors
        acts_norm = F.normalize(acts, p=2, dim=1)
        
        # Compute cosine similarity matrix
        # [N, D] @ [D, N] -> [N, N]
        sim_matrix = torch.matmul(acts_norm, acts_norm.t())
        
        # Average of upper triangle (excluding diagonal)
        mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()
        avg_cosine = sim_matrix[mask].mean().item()
        
        return avg_cosine

    def analyze_holistic_validity(
        self, 
        text: str, 
        target_layers: List[int] = None
    ) -> Dict[str, Any]:
        """
        Run all validity checks.
        """
        if target_layers is None:
            # Default to middle and last layer
            n_layers = self.model.cfg.n_layers
            target_layers = [n_layers // 2, n_layers - 1]
            
        results = {}
        
        # 1. Info Validity
        results["perplexity"] = self.compute_perplexity(text)
        results["entropy_stats"] = self.compute_entropy_profile(text)
        
        # 2. Geometric Validity
        geo_stats = {}
        for layer in target_layers:
            anisotropy = self.compute_anisotropy(text, layer)
            geo_stats[f"layer_{layer}_anisotropy"] = anisotropy
        results["geometric_stats"] = geo_stats
        
        return results


class CompositionalAnalysis:
    """
    Analyzes the 'Compositionality' of neural representations using Category Theory principles.
    
    The core hypothesis is that a 'structure-preserving' representation (Functor) should map 
    syntactic composition to semantic composition. 
    
    e.g. F(A • B) ≈ F(A) + F(B) (Linear) or F(A) ⊗ F(B) (Tensor)
    """
    
    def __init__(self, model):
        self.model = model
        
    def get_token_activation(self, text: str, layer_idx: int) -> Tensor:
        """Get the activation of the LAST token for a given text."""
        # Using run_with_cache to get internals
        tokens = self.model.to_tokens(text)
        _, cache = self.model.run_with_cache(tokens)
        
        # We usually look at the residual stream after the layer (resid_post)
        hook_name = f"blocks.{layer_idx}.hook_resid_post"
        if hook_name not in cache:
             # Fallback to hook_resid_pre or final ln
             hook_name = f"blocks.{layer_idx}.hook_resid_pre"
             
        if hook_name not in cache:
            raise ValueError(f"Could not find activation hook for layer {layer_idx}")
            
        act = cache[hook_name] # [batch, seq, d_model]
        
        # Take the last token's activation (representing the whole phrase processing)
        # Assuming batch size 1
        last_token_act = act[0, -1, :] 
        return last_token_act
        
    def analyze_compositionality(
        self, 
        phrases: List[Tuple[str, str, str]], 
        layer_idx: int
    ) -> Dict[str, Any]:
        """
        Test for Additive Compositionality: Repr(Compound) ≈ Repr(Part1) + Repr(Part2)
        
        Args:
            phrases: List of (part1, part2, compound). e.g. [("black", "cat", "black cat")]
            layer_idx: Layer to analyze
            
        Returns:
            Stats including R^2 score of the fit.
        """
        X_list = []
        Y_list = []
        
        for p1, p2, comp in phrases:
            v1 = self.get_token_activation(p1, layer_idx)
            v2 = self.get_token_activation(p2, layer_idx)
            v_comp = self.get_token_activation(comp, layer_idx)
            
            # Input: Concatenation of parts [v1, v2]
            # We want to see if v_comp is a linear function of v1 and v2
            # Simple addition hypothesis: v_comp = 1*v1 + 1*v2
            # General linear hypothesis: v_comp = W * [v1, v2]
            
            # For this test, we construct X as [v1, v2]
            combined = torch.cat([v1, v2], dim=0)
            X_list.append(combined)
            Y_list.append(v_comp)
            
        if not X_list:
            return {"error": "No data"}
            
        X = torch.stack(X_list).float() # [N, 2*d_model]
        Y = torch.stack(Y_list).float() # [N, d_model]
        
        # Solve Least Squares: Y = X @ W.T
        # W.T = (X.T X)^-1 X.T Y
        try:
            # torch.linalg.lstsq returns (solution, residuals, rank, singular_values)
            # solution shape: [2*d_model, d_model]
            result = torch.linalg.lstsq(X, Y)
            W_T = result.solution
            
            Y_pred = X @ W_T
            
            # Compute R^2
            # SS_res = sum( (y_true - y_pred)^2 )
            # SS_tot = sum( (y_true - y_mean)^2 )
            
            ss_res = torch.sum((Y - Y_pred) ** 2)
            y_mean = torch.mean(Y, dim=0, keepdim=True)
            ss_tot = torch.sum((Y - y_mean) ** 2)
            
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            
            # Compute Cosine Similarity between Predicted and Actual (often more stable for high dim)
            cos_sim = F.cosine_similarity(Y, Y_pred, dim=1).mean()
            
            return {
                "layer_idx": layer_idx,
                "n_samples": len(phrases),
                "r2_score": r2.item(),
                "cosine_similarity": cos_sim.item(),
                "residual_loss": ss_res.item()
            }
            
        except Exception as e:
            return {"error": str(e)}
