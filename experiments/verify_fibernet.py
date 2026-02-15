
import os
import sys

import torch

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fibernet_v2 import DecoupledFiberNet, LieGroupEmbedding


def verify_lie_group_embedding():
    print("Verifying LieGroupEmbedding...")
    vocab_size = 10
    dim = 8 # Must be even for circle
    embed = LieGroupEmbedding(vocab_size, dim, group_type='circle')
    
    input_ids = torch.tensor([[0, 1], [2, 3]])
    output = embed(input_ids)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {output.shape}")
    
    expected_shape = (2, 2, 8)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    
    # Check if norm is 1 (for circle group, cos^2 + sin^2 = 1)
    # The output is [cos, sin, cos, sin, ...]
    # We need to reshape to check norms of pairs.
    # Actually, the implementation stacks [cos, sin] then flattens.
    # So every pair (2*i, 2*i+1) should have norm 1.
    
    reshaped = output.view(2, 2, 4, 2) # [Batch, Seq, Dim/2, 2]
    norms = torch.norm(reshaped, dim=-1)
    print(f"Norms (should be close to 1): {norms[0,0]}")
    assert torch.allclose(norms, torch.ones_like(norms)), "Norms are not 1!"
    print("LieGroupEmbedding verified.\n")

def verify_fibernet_forward():
    print("Verifying DecoupledFiberNet Forward Pass...")
    vocab_size = 100
    d_model = 32
    seq_len = 5
    batch_size = 4
    
    model = DecoupledFiberNet(vocab_size, d_model=d_model, n_layers=2, group_type='circle', max_len=10)
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    logits = model(input_ids)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Logits shape: {logits.shape}")
    
    expected_shape = (batch_size, seq_len, vocab_size)
    assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"
    print("DecoupledFiberNet Forward Pass verified.\n")

if __name__ == "__main__":
    verify_lie_group_embedding()
    verify_fibernet_forward()
    print("All verifications passed!")
