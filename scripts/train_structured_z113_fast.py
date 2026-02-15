
import os
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from torch.utils.data import DataLoader, TensorDataset


# --- 1. Structured Initialization Logic ---
def compute_laplacian_embeddings(n, dim):
    """Computes Laplacian Eigenmaps for a cycle graph of size n."""
    G = nx.cycle_graph(n)
    adj = nx.adjacency_matrix(G).toarray().astype(float)
    laplacian = csgraph.laplacian(adj, normed=True)
    vals, vecs = eigsh(laplacian, k=dim+1, which='SM')
    return torch.tensor(vecs[:, 1:], dtype=torch.float32)

# --- 2. Model Definition ---
class StructuredFiberNet(nn.Module):
    def __init__(self, n_vocab, d_model, init_embeddings=None):
        super().__init__()
        self.fiber = nn.Embedding(n_vocab, d_model)
        self.unembed = nn.Linear(d_model, n_vocab, bias=False)
        
        if init_embeddings is not None:
            self.fiber.weight.data = init_embeddings * np.sqrt(d_model)
        else:
            nn.init.normal_(self.fiber.weight, std=0.02)
            
    def forward(self, a_idx, b_idx):
        a_vec = self.fiber(a_idx)
        b_vec = self.fiber(b_idx)
        combo = a_vec + b_vec
        logits = self.unembed(combo)
        return logits

# --- 3. Optimized Data & Training ---
def train_fast(use_structured_init=True, device='cuda'):
    P = 113
    DIM = 8
    LR = 0.01
    EPOCHS = 1000 # Increased epochs because it's fast now
    
    print(f"Device: {device}")
    
    # Pre-generate ALL data on GPU
    # X = [0..P-1], Y = [0..P-1]
    x = torch.arange(P, device=device)
    y = torch.arange(P, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    
    # Targets: (x + y) % P
    targets = (grid_x + grid_y) % P
    
    # Flatten
    flat_a = grid_x.reshape(-1)
    flat_b = grid_y.reshape(-1)
    flat_y = targets.reshape(-1)
    
    # Split
    total_samples = P * P
    split = int(total_samples * 0.8)
    perm = torch.randperm(total_samples, device=device)
    
    train_idx = perm[:split]
    test_idx = perm[split:]
    
    # Dataset
    train_ds = TensorDataset(flat_a[train_idx], flat_b[train_idx], flat_y[train_idx])
    test_ds = TensorDataset(flat_a[test_idx], flat_b[test_idx], flat_y[test_idx])
    
    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True) # Large batch size
    test_loader = DataLoader(test_ds, batch_size=1024)
    
    # Model
    if use_structured_init:
        init_emb = compute_laplacian_embeddings(P, DIM).to(device)
    else:
        init_emb = None
        
    model = StructuredFiberNet(P, DIM, init_embeddings=init_emb).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    history = []
    
    start_time = time.time()
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
            
        if epoch % 50 == 0:
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
            print(f"Epoch {epoch}: Loss {total_loss/len(train_loader):.4f}, Test Acc {acc:.2%}")

    end_time = time.time()
    print(f"Training Time: {end_time - start_time:.2f}s")
    return history

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Training Structured...")
    hist_struct = train_fast(True, device)
    
    print("\nTraining Random...")
    hist_random = train_fast(False, device)
    
    plt.figure(figsize=(10, 6))
    plt.plot(hist_struct, label='Structured (Laplacian)', linewidth=3)
    plt.plot(hist_random, label='Random (Gaussian)', linestyle='--')
    plt.title('Grokking Speed: Structured vs Random (Z_113)')
    plt.xlabel('Evals (x50 Epochs)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    os.makedirs('tempdata', exist_ok=True)
    plt.savefig('tempdata/fast_z113_comparison.png')
    print("Saved to tempdata/fast_z113_comparison.png")
