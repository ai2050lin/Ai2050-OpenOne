
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from torch.utils.data import DataLoader, Dataset


# --- 1. Structured Initialization Logic ---
def compute_laplacian_embeddings(n, dim):
    """Computes Laplacian Eigenmaps for a cycle graph of size n."""
    G = nx.cycle_graph(n)
    adj = nx.adjacency_matrix(G).toarray().astype(float)
    laplacian = csgraph.laplacian(adj, normed=True)
    # Get k+1 smallest eigenvectors
    vals, vecs = eigsh(laplacian, k=dim+1, which='SM')
    # Skip the first constant eigenvector
    return torch.tensor(vecs[:, 1:], dtype=torch.float32)

# --- 2. Model Definition (Simplified FiberNet) ---
class StructuredFiberNet(nn.Module):
    def __init__(self, n_vocab, d_model, init_embeddings=None):
        super().__init__()
        self.n_vocab = n_vocab
        self.d_model = d_model
        
        # Fiber (Memory): Holds the concept representations
        self.fiber = nn.Embedding(n_vocab, d_model)
        
        # Logic (Structure): Computes interactions
        # For Z_n addition, we just need a simple interaction mechanism
        # Simulating Attention: Q * K^T
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.unembed = nn.Linear(d_model, n_vocab, bias=False)
        
        # Initialize
        if init_embeddings is not None:
            print(f"Applying Structured Initialization (Shape: {init_embeddings.shape})")
            # Scale it up because eigenvectors are small
            self.fiber.weight.data = init_embeddings * np.sqrt(d_model) # Scaling heuristics
            # Freeze fiber to test if "Structure" alone is enough? 
            # OR let it fine-tune? Let's let it fine-tune but start from good place.
        else:
            nn.init.normal_(self.fiber.weight, std=0.02)
            
    def forward(self, a_idx, b_idx):
        # 1. Fetch Fibers
        a_vec = self.fiber(a_idx) # [Batch, D]
        b_vec = self.fiber(b_idx) # [Batch, D]
        
        # 2. Logic Step (Group Operation)
        # In a circle, rotation is linear. 
        # Ideal: vector_sum = R(a) + R(b) -> c
        # Here we use a simplified attention-like interaction
        # z = a + b (in embedding space)
        # But wait, in circle embedding, a + b (indices) corresponds to rotation R_b applied to a?
        # Or z_c = z_a * z_b (complex multiplication for 2D).
        # Let's trust the network to find the "Add" operator if the space is right.
        
        combo = a_vec + b_vec # Simple superposition
        
        # 3. Decode
        logits = self.unembed(combo)
        return logits

# --- 3. Data & Training ---
class Z113Dataset(Dataset):
    def __init__(self, p=113, mode='train', split_ratio=0.8):
        self.p = p
        self.data = []
        
        # Generate all pairs
        for i in range(p):
            for j in range(p):
                target = (i + j) % p
                self.data.append((i, j, target))
                
        # Shuffle
        np.random.shuffle(self.data)
        
        # Split
        split_idx = int(len(self.data) * split_ratio)
        if mode == 'train':
            self.data = self.data[:split_idx]
        else:
            self.data = self.data[split_idx:]
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def train(use_structured_init=True):
    P = 113
    DIM = 8 # Low dimension to force geometry
    LR = 0.01
    EPOCHS = 100
    
    # Init Weights
    if use_structured_init:
        init_emb = compute_laplacian_embeddings(P, DIM)
    else:
        init_emb = None
        
    model = StructuredFiberNet(P, DIM, init_embeddings=init_emb)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    train_loader = DataLoader(Z113Dataset(P, 'train'), batch_size=32, shuffle=True)
    test_loader = DataLoader(Z113Dataset(P, 'test'), batch_size=128)
    
    history = []
    
    print(f"\nStarting Training (Structured Init: {use_structured_init})...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for a, b, y in train_loader:
            optimizer.zero_grad()
            logits = model(a, b)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for a, b, y in test_loader:
                logits = model(a, b)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        
        acc = correct / total
        history.append(acc)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss {total_loss/len(train_loader):.4f}, Test Acc {acc:.2%}")
            
    return history

if __name__ == "__main__":
    # Compare
    hist_struct = train(use_structured_init=True)
    hist_random = train(use_structured_init=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(hist_struct, label='Structured Init (Laplacian)', linewidth=3)
    plt.plot(hist_random, label='Random Init (Gaussian)', linestyle='--')
    plt.title('Generalization on Z_113 Addition: Structured vs Random')
    plt.xlabel('Epochs')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('tempdata/structured_generalization_test.png')
    print("Saved plot to tempdata/structured_generalization_test.png")
