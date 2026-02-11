
import os
import sys

import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.fiber_net import FiberBundle, FiberNetV2


def test_fiber_bundle():
    print("--- Testing FiberBundle ---")
    s_vocab = 100
    c_vocab = 500
    d_m = 32
    d_f = 64
    
    model = FiberBundle(s_vocab, c_vocab, d_manifold=d_m, d_fiber=d_f)
    print("Model initialized.")
    
    # Dummy Input
    batch_size = 2
    seq_len = 5
    struct_ids = torch.randint(0, s_vocab, (batch_size, seq_len))
    content_ids = torch.randint(0, c_vocab, (batch_size, seq_len))
    
    # Forward Pass
    logits, fiber_out, manifold_out = model(struct_ids, content_ids)
    
    print(f"Logits Shape: {logits.shape} (Expected: [{batch_size}, {seq_len}, {c_vocab}])")
    print(f"Fiber Out Shape: {fiber_out.shape} (Expected: [{batch_size}, {seq_len}, {d_f}])")
    print(f"Manifold Out Shape: {manifold_out.shape} (Expected: [{batch_size}, {seq_len}, {d_m}])")
    
    # Check Curvature Monitor
    if model.connection.last_transport_matrix is not None:
        print(f"Transport Matrix Shape: {model.connection.last_transport_matrix.shape}")
        
        # Log curvature
        error = model.curvature_monitor.log(model.connection.last_transport_matrix)
        print(f"Curvature Error (Entropy): {error:.6f}")
        print(f"History Length: {len(model.curvature_monitor.history)}")
    else:
        print("ERROR: Transport matrix not captured.")

    assert logits.shape == (batch_size, seq_len, c_vocab)

def test_fibernet_v2():
    print("\n--- Testing FiberNetV2 ---")
    s_vocab = 100
    c_vocab = 500
    model = FiberNetV2(s_vocab, c_vocab)
    print("V2 Model initialized.")
    
    struct_ids = torch.randint(0, s_vocab, (2, 5))
    content_ids = torch.randint(0, c_vocab, (2, 5))
    
    logits, f_out, m_out = model(struct_ids, content_ids)
    print(f"V2 Logits Shape: {logits.shape}")

if __name__ == "__main__":
    test_fiber_bundle()
    test_fibernet_v2()
    print("\nAll tests passed!")
