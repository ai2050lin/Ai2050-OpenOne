
import torch
import torch.nn as nn

from transformer_lens import HookedTransformer, HookedTransformerConfig


class FiberNetLogic(nn.Module):
    """
    A pure Logic Stream model designed to learn abstract geometric structures.
    It uses random initialization and no pre-trained weights.
    """
    def __init__(self, vocab_size: int, d_model: int = 128, n_layers: int = 4, n_heads: int = 4, d_head: int = 32, init_embeddings=None):
        super().__init__()
        
        self.cfg = HookedTransformerConfig(
            n_layers=n_layers,
            d_model=d_model,
            n_ctx=64,  # Logic sequences are short
            d_head=d_head,
            n_heads=n_heads,
            d_vocab=vocab_size,
            act_fn="gelu",
            normalization_type="LN",
            seed=42,
        )
        
        self.model = HookedTransformer(self.cfg)
        
        # Structured Initialization
        if init_embeddings is not None:
            # init_embeddings should be a tensor of shape [vocab_size, d_model]
            # specific: Graph Laplacian Eigenvectors
            print(f"FiberNet: Applying Structured Initialization (Shape: {init_embeddings.shape})")
            
            # We need to access the embedding weights of HookedTransformer
            # Typically: model.embed.W_E
            with torch.no_grad():
                # Ensure shapes match
                if init_embeddings.shape != self.model.embed.W_E.shape:
                    raise ValueError(f"Shape mismatch: Init {init_embeddings.shape} vs Model {self.model.embed.W_E.shape}")
                
                self.model.embed.W_E.data.copy_(init_embeddings)
                
    def forward(self, input_ids):
        return self.model(input_ids)
    
    @classmethod
    def from_config(cls, config_dict):
        return cls(**config_dict)

def create_logic_model(vocab_size=1000, **kwargs):
    """Factory function for FiberNet Logic Core"""
    return FiberNetLogic(vocab_size=vocab_size, **kwargs)

