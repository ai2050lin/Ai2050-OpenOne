
import os
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fibernet_logic import create_logic_model

# --- Configuration ---
DATA_PATH = "data/logic_core/logic_corpus_v1.txt"
BATCH_SIZE = 64
D_MODEL = 128
N_LAYERS = 4
N_HEADS = 4
EPOCHS = 20 # Fast training for prototype
LR = 1e-3
CTX_LEN = 64
SEED = 42

# --- 1. Data Loading & Preprocessing ---

class LogicCorpus:
    def __init__(self, path):
        self.path = path
        with open(path, 'r', encoding='utf-8') as f:
            self.lines = [line.strip().split() for line in f if line.strip()]
        
        # Build Vocabulary
        self.token_to_id = {}
        self.id_to_token = {}
        self.vocab_size = 0
        
        # Special tokens
        self.add_token("<PAD>") # 0
        self.add_token("<UNK>") # 1
        self.add_token("<BOS>") # 2
        self.add_token("<EOS>") # 3
        
        for line in self.lines:
            for token in line:
                self.add_token(token)
                
        print(f"Vocab Size: {self.vocab_size}")
        
    def add_token(self, token):
        if token not in self.token_to_id:
            self.token_to_id[token] = self.vocab_size
            self.id_to_token[self.vocab_size] = token
            self.vocab_size += 1
            
    def encode(self, tokens):
        return [self.token_to_id.get(t, self.token_to_id["<UNK>"]) for t in tokens]
    
    def decode(self, ids):
        return [self.id_to_token.get(i, "<UNK>") for i in ids]

class LogicDataset(Dataset):
    def __init__(self, corpus, max_len=CTX_LEN):
        self.corpus = corpus
        self.max_len = max_len
        self.data = []
        
        for line in corpus.lines:
            # Add BOS and EOS
            tokens = ["<BOS>"] + line + ["<EOS>"]
            ids = corpus.encode(tokens)
            
            # Truncate or Pad
            if len(ids) > max_len:
                ids = ids[:max_len]
            else:
                ids = ids + [corpus.token_to_id["<PAD>"]] * (max_len - len(ids))
                
            self.data.append(torch.tensor(ids, dtype=torch.long))
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# --- 2. Graph & Laplacian Embedding ---

def compute_laplacian_embeddings(corpus, d_model):
    """
    Constructs a co-occurrence graph of tokens and calculates Laplacian Eigenmaps.
    """
    print("Construction Co-occurrence Graph...")
    G = nx.Graph()
    G.add_nodes_from(range(corpus.vocab_size))
    
    # Window size for co-occurrence
    window_size = 2
    
    for line in tqdm(corpus.lines, desc="Building Graph"):
        ids = corpus.encode(line)
        for i in range(len(ids)):
            for j in range(max(0, i - window_size), min(len(ids), i + window_size + 1)):
                if i != j:
                    u, v = ids[i], ids[j]
                    if G.has_edge(u, v):
                        G[u][v]['weight'] += 1
                    else:
                        G.add_edge(u, v, weight=1)
                        
    # Ensure graph is connected (handle PAD/UNK/BOS/EOS if not in lines)
    # Actually, let's just compute for the largest component or add weak links
    # For simplicity, we assume the logic corpus connects most symbols.
    # Isolated nodes will get 0-vectors.
    
    adj = nx.adjacency_matrix(G).toarray().astype(float)
    
    # Laplacian
    print("Computing Laplacian...")
    try:
        laplacian = csgraph.laplacian(adj, normed=True)
        # Eigen decomposition
        # We need d_model eigenvectors.
        # Note: vocab_size might be small, ensure k < N
        k = min(d_model + 1, corpus.vocab_size - 1)
        vals, vecs = eigsh(laplacian, k=k, which='SM')
        
        # Skip the first (smallest eigenvalue is 0)
        embedding = vecs[:, 1:]
        
        # If we asked for more dims than available (small vocab), pad with noise? 
        # Or just repeat?
        if embedding.shape[1] < d_model:
            # Pad with zeros or random noise for remaining dimensions
            pad_size = d_model - embedding.shape[1]
            padding = np.random.randn(corpus.vocab_size, pad_size) * 0.01
            embedding = np.hstack([embedding, padding])
        
        # Normalize to typical embedding variance
        embedding = embedding * np.sqrt(d_model)
        
        return torch.tensor(embedding, dtype=torch.float32)
        
    except Exception as e:
        print(f"Laplacian computation failed: {e}")
        print("Fallback to random init.")
        return None

# --- 3. Training Loop ---

def train_model(corpus, dataset, init_embeddings=None, name="Model"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training {name} on {device}...")
    
    model_wrapper = create_logic_model(
        vocab_size=corpus.vocab_size, 
        d_model=D_MODEL, 
        n_layers=N_LAYERS, 
        n_heads=N_HEADS,
        init_embeddings=init_embeddings
    )
    model = model_wrapper.model # Get the HookedTransformer
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=corpus.token_to_id["<PAD>"])
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    history = {'loss': []}
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(loader, desc=f"{name} Epoch {epoch+1}/{EPOCHS}", leave=False)
        for batch in pbar:
            batch = batch.to(device)
            
            # Causal LM: Input = x[:-1], Target = x[1:]
            input_ids = batch[:, :-1]
            target_ids = batch[:, 1:]
            
            outputs = model(input_ids) # [Batch, Seq, Vocab]
            
            # Flatten
            loss = criterion(outputs.reshape(-1, corpus.vocab_size), target_ids.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
    model_path = f"tempdata/{name}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved {name} model to {model_path}")
    
    return history

# --- 4. Main ---

def main():
    # Setup data
    if not os.path.exists("tempdata"):
        os.makedirs("tempdata")
        
    print("Loading Corpus...")
    corpus = LogicCorpus(DATA_PATH)
    dataset = LogicDataset(corpus)
    
    # 1. Random Init
    print("\n--- Baseline: Random Initialization ---")
    hist_random = train_model(corpus, dataset, init_embeddings=None, name="RandomInit")
    
    # 2. Structured Init
    print("\n--- Experiment: Structured Initialization ---")
    laplacian_emb = compute_laplacian_embeddings(corpus, D_MODEL)
    
    if laplacian_emb is not None:
        hist_struct = train_model(corpus, dataset, init_embeddings=laplacian_emb, name="StructInit")
    else:
        print("Skipping Structured Init due to computation failure.")
        hist_struct = {'loss': []}
    
    # 3. Plot Comparison
    plt.figure(figsize=(10, 6))
    plt.plot(hist_random['loss'], label='Random Init', linestyle='--')
    if laplacian_emb is not None:
        plt.plot(hist_struct['loss'], label='Structured Init (Laplacian)', linewidth=2)
    
    plt.title(f'Logic Core Training: Loss Curve\n(V={corpus.vocab_size}, L={N_LAYERS}, H={N_HEADS}, D={D_MODEL})')
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = "tempdata/logic_core_comparison.png"
    plt.savefig(out_path)
    print(f"\nComparison plot saved to {out_path}")

if __name__ == "__main__":
    main()
