
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.fibernet_v2 import DecoupledFiberNet
from scripts.generate_logic_data import LogicDatasetGenerator


class VectorDatabase:
    """
    A simple in-memory Vector Database for the 'Infinite Memory Stream'.
    Stores key-value pairs: Vector -> Token (String/ID).
    Supports Nearest Neighbor Search.
    """
    def __init__(self, dim):
        self.dim = dim
        self.vectors = []
        self.mnemonics = [] # The 'Concept' name (e.g. "Paris", "11")
        
    def add(self, vector, mnemonic):
        """
        Add a new memory item.
        vector: tensor [dim]
        mnemonic: str or int
        """
        # Normalize for Cosine Similarity
        v = vector / vector.norm(p=2)
        self.vectors.append(v)
        self.mnemonics.append(mnemonic)
        
    def query(self, query_vec, k=1):
        """
        Find k nearest neighbors.
        """
        if not self.vectors:
            return []
            
        q = query_vec / query_vec.norm(p=2)
        
        # Stack vectors: [N, D]
        # Inefficient loop, but fine for demo
        stack = torch.stack(self.vectors).to(q.device)
        
        # Cosine similarity: q @ stack.T
        scores = torch.matmul(q, stack.T) # [N]
        
        best_scores, indices = torch.topk(scores, k)
        
        results = []
        for score, idx in zip(best_scores, indices):
            results.append((self.mnemonics[idx.item()], score.item()))
            
        return results

def experiment_infinite_memory():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running HLAI Step 3: Infinite Memory on {device}...")
    
    # 1. Setup Logic Engine with STRUCTURED Embeddings
    # We use 'circle' group type so embeddings have semantic math properties (Phases).
    # This allows us to predict the 'correct' embedding for new numbers.
    VOCAB_SIZE = 12
    D_MODEL = 64
    model = DecoupledFiberNet(vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layers=2, group_type='circle').to(device)
    
    # Train heavily on 0..9 (Knowledge Base A)
    # Leave 10, 11 UNSEEN (Knowledge Base B)
    print(">>> Training Logic Engine on Z_10 (0..9)...")
    
    gen = LogicDatasetGenerator(vocab_size=VOCAB_SIZE)
    # Generate data but filter out any 10 or 11
    # Actually simpler: Generate Z_12 data, filter
    x_all, y_all = gen.generate_cyclic_data(3000, modulus=12)
    
    train_mask = (x_all < 10).all(dim=1) & (y_all < 10)
    x_train = x_all[train_mask].to(device)
    y_train = y_all[train_mask].to(device)
    
    print(f"Training Samples (excluding 10, 11): {len(x_train)}")
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    for epoch in range(50):
        optimizer.zero_grad()
        logits = model(x_train)
        loss = criterion(logits[:, -1, :], y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 10 == 0:
            print(f"Ep {epoch}: Loss {loss.item():.4f}")
            
    print(">>> Training Complete.")
    
    # 2. Build Vector Database (The 'Extended Memory')
    print(">>> Building Vector Database...")
    db = VectorDatabase(D_MODEL)
    
    # Populate DB with KNOWN items (0..9)
    # We extract their learned embeddings from the model
    # Note: LieGroupEmbedding creates embeddings on the fly from indices.
    # We can get them by passing IDs.
    known_ids = torch.arange(10, device=device)
    known_vecs = model.content_embed(known_ids) # [10, 64]
    
    for i in range(10):
        db.add(known_vecs[i], str(i))
        
    print("Database populated with 0..9.")
    
    # 3. Verify Retrieval on Known Facts
    # Test: 2 + 3 = 5
    # Input: [2, 3]
    test_input = torch.tensor([[2, 3]], dtype=torch.long).to(device)
    # Model forward returns LOGITS (projection to vocab).
    # BUT we want the VECTOR before the head.
    # FiberNet forward returns `self.head(curr_memory)`.
    # We need to access `curr_memory`.
    # Let's hook or modify... or just assume the model's 'Head' is linear and invertible?
    # No, let's just use the `forward` code manually to get the vector.
    
    # Manual forward to get vector
    with torch.no_grad():
        # Copied logic from forward
        ids = test_input
        batch, seq = ids.shape
        pos = torch.arange(seq, device=device).unsqueeze(0)
        curr_logic = model.pos_embed(pos)
        curr_mem = model.content_embed(ids)
        for layer in model.layers:
            curr_logic, curr_mem = layer(curr_logic, curr_mem)
        output_vector = curr_mem[0, -1, :] # Vector for result
    
    # Query DB
    results = db.query(output_vector, k=3)
    print(f"\nQuery: 2 + 3 = ?")
    print(f"Logic Output Vector Retrieval: {results}")
    
    # 4. KNOWLEDGE INJECTION (The "Infinite" Part)
    # We want to add knowledge "10" and "11".
    # And we want the model to solve "9 + 1 = 10" WITHOUT training.
    
    # Problem: The model has never seen 10. `model.content_embed` has parameters for 10, 11 (initialized random/phases).
    # Since we use `LieGroupEmbedding` (Circle), the initialization is random phases.
    # The model learned to add phases for 0..9.
    # Does it generalize to 10?
    # If the "Logic" is "Add Phase A + Phase B = Phase Result", then YES, it applies to ANY phases.
    # BUT 10's phase is random. 9+1's phase is detemined by 9 and 1.
    # So "10" (the symbol) corresponds to a specific phase that satisfies 9+1=10, 8+2=10 etc.
    # In a real infinite memory, we would *Construct* the embedding of 10 to be consistent.
    # OR we would define 10 as "The result of 9+1".
    
    print("\n>>> Injecting New Knowledge: '10' and '11'...")
    # Theoretical Injection: We define vector(10) = vector(9) + vector(1) (Group operation).
    # But LieGroupEmbedding uses cos/sin. The group operation is rotation.
    # Let's CALCULATE what vector(10) SHOULD be based on the engine's logic.
    
    # Calculate target vector for 9+1
    in_9_1 = torch.tensor([[9, 1]], dtype=torch.long).to(device)
    with torch.no_grad():
        # Manual forward
        pos = torch.arange(2, device=device).unsqueeze(0)
        cl = model.pos_embed(pos)
        cm = model.content_embed(in_9_1)
        for layer in model.layers: cl, cm = layer(cl, cm)
        vec_10_calculated = cm[0, -1, :]
        
    print("Calculated Vector for '10' (derived from 9+1).")
    
    # Inject into DB
    db.add(vec_10_calculated, "10 (Injected)")
    print("Added '10' to Database.")
    
    # NOW TEST RECALL: 5 + 5 = ?
    # The model has never seen 5+5 in training (result 10 was excluded).
    # It has never stored '10'.
    # We just added '10' derived from 9+1.
    # Does 5+5 map to the same vector?
    query_5_5 = torch.tensor([[5, 5]], dtype=torch.long).to(device)
    with torch.no_grad():
        pos = torch.arange(2, device=device).unsqueeze(0)
        cl = model.pos_embed(pos)
        cm = model.content_embed(query_5_5)
        for layer in model.layers: cl, cm = layer(cl, cm)
        vec_5_5 = cm[0, -1, :]
        
    results_new = db.query(vec_5_5, k=1)
    print(f"\nQuery: 5 + 5 = ? (Unseen combination)")
    print(f"Retrieval: {results_new}")
    
    if "10" in results_new[0][0]:
        print("SUCCESS: Zero-shot Generalization + Infinite Memory Injection!")
        print("The model connected 5+5 to the '10' we defined via 9+1.")
    else:
        print("FAILURE: Inconsistent Geometry.")

    # 5. Injection '11'
    # Calculate 10 + 1 (We use the injected 10? No, model doesn't have token 10 input capability yet without updating Embedding layer)
    # Wait, the Memory Stream Input `model.content_embed` takes TOKEN IDs.
    # To input '10', we need `model.content_embed` to handle index 10.
    # We can MANUALLY update `model.content_embed` weights to match our injected vector?
    # YES! That's "Writing to Hippocampus".
    
    # Update Model's Input Embedding for 10 to match the calculated vector?
    # LieGroupEmbedding stores phases. We can't easily invert vec->phase generally without solving.
    # But for demo, let's just stick to "Retrieval Output".
    # We can input pairs that sum to 11 using existing numbers? e.g. 5+6.
    
    # Let's test 5+6 = 11.
    # We inject 11 derived from 9+2.
    in_9_2 = torch.tensor([[9, 2]], dtype=torch.long).to(device)
    with torch.no_grad():
        pos = torch.arange(2, device=device).unsqueeze(0)
        cl = model.pos_embed(pos)
        cm = model.content_embed(in_9_2)
        for layer in model.layers: cl, cm = layer(cl, cm)
        vec_11_calculated = cm[0, -1, :]
        
    db.add(vec_11_calculated, "11 (Injected)")
    
    # Query 5+6
    in_5_6 = torch.tensor([[5, 6]], dtype=torch.long).to(device)
    with torch.no_grad():
        pos = torch.arange(2, device=device).unsqueeze(0)
        cl = model.pos_embed(pos)
        cm = model.content_embed(in_5_6)
        for layer in model.layers: cl, cm = layer(cl, cm)
        vec_out = cm[0, -1, :]
        
    res = db.query(vec_out, k=1)
    print(f"\nQuery: 5 + 6 = ?")
    print(f"Retrieval: {res}")
    
    if "11" in res[0][0]:
        print("SUCCESS: 11 retrieved.")

if __name__ == "__main__":
    experiment_infinite_memory()
