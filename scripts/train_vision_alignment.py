
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fibernet_logic import create_logic_model
from models.vision_projector import create_vision_model
from scripts.train_logic_core import D_MODEL, DATA_PATH, N_HEADS, N_LAYERS, LogicCorpus

# --- Configuration ---
LOGIC_MODEL_PATH = "tempdata/StructInit.pth" # Use the Ricci-optimized one ideally, but let's use the trained one first.
# Wait, Ricci flow updated the edges/dist, but did it save back to a .pth for the model weights?
# ricci_flow_optimizer.py didn't save the *model weights*, it just optimized the manifold graph.
# Ideally we should use the post-Ricci embeddings if we could map them back.
# For now, let's use the trained model weights as the "Native Logic Manifold".

BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3

def get_logic_anchors(model, corpus):
    """
    Extracts embeddings for SYM_0 to SYM_9.
    Returns a dict: {digit (int): embedding (tensor)}
    """
    anchors = {}
    print("Extracting Logic Anchors...")
    
    # Ensure model is in eval mode
    model.eval()
    
    with torch.no_grad():
        W_E = model.embed.W_E # [Vocab, D]
        
        for i in range(10):
            token = f"SYM_{i}"
            if token in corpus.token_to_id:
                idx = corpus.token_to_id[token]
                emb = W_E[idx]
                anchors[i] = emb
                print(f"  Found Anchor: {token} -> ID {idx}")
            else:
                print(f"  WARNING: Token {token} not found in logic corpus!")
                
    return anchors

def train_alignment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Aligning Vision to Logic on {device}...")
    
    # 1. Load Logic Model (The Ground Truth)
    corpus = LogicCorpus(DATA_PATH)
    logic_model = create_logic_model(vocab_size=corpus.vocab_size, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS).model
    
    try:
        logic_model.load_state_dict(torch.load(LOGIC_MODEL_PATH))
        print("Logic Model loaded.")
    except FileNotFoundError:
        print("Logic Model validation failed. Please run Phase I training first.")
        return

    logic_model.to(device)
    logic_model.eval() # Freeze
    
    # Get Anchors
    anchors_map = get_logic_anchors(logic_model, corpus)
    if len(anchors_map) < 10:
        print("Error: Could not find all digit anchors in Logic Core.")
        return

    # Convert anchors to a tensor for fast lookup
    # anchors_tensor[i] = embedding of SYM_i
    anchors_tensor = torch.stack([anchors_map[i] for i in range(10)]).to(device) # [10, D]

    # 2. Setup Vision Model
    vision_model = create_vision_model(d_model=D_MODEL).to(device)
    optimizer = optim.Adam(vision_model.parameters(), lr=LR)
    criterion = nn.MSELoss() # Or CosineEmbeddingLoss
    
    # 3. Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # Standard MNIST mean/std
    ])
    
    # We download to tempdata
    if not os.path.exists("tempdata/data"):
        os.makedirs("tempdata/data")
        
    train_dataset = datasets.MNIST('tempdata/data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('tempdata/data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # 4. Training Loop
    history = []
    
    vision_model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device) # [Batch] (0-9)
            
            # Forward Vision
            projected = vision_model(images) # [Batch, D]
            
            # Get Targets
            targets = anchors_tensor[labels] # [Batch, D]
            
            # Loss
            loss = criterion(projected, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_loss = total_loss / len(train_loader)
        history.append(avg_loss)
        print(f"Epoch {epoch+1}: Alignment Loss = {avg_loss:.4f}")
        
    # 5. Visualization & Verification
    print("Generating Alignment Visualization...")
    vision_model.eval()
    
    # Collect test projections
    test_projections = []
    test_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            proj = vision_model(images)
            test_projections.append(proj.detach().cpu().numpy())
            test_labels.append(labels.detach().cpu().numpy())
            
    test_projections = np.concatenate(test_projections, axis=0) # [N_test, D]
    test_labels = np.concatenate(test_labels, axis=0)
    
    # Get Logic Anchors (CPU)
    anchor_vecs = anchors_tensor.detach().cpu().numpy() # [10, D]
    
    # PCA to 2D
    from sklearn.decomposition import PCA

    # Combine data for PCA to share space
    combined = np.vstack([anchor_vecs, test_projections[:2000]]) # Use subset of test to avoid clutter
    pca = PCA(n_components=2)
    combined_2d = pca.fit_transform(combined)
    
    anchors_2d = combined_2d[:10]
    projections_2d = combined_2d[10:]
    labels_subset = test_labels[:2000]
    
    # Plot
    plt.figure(figsize=(10, 10))
    cmap = plt.get_cmap('tab10')
    
    # Plot Projections
    for i in range(10):
        # Digits
        mask = (labels_subset == i)
        plt.scatter(projections_2d[mask, 0], projections_2d[mask, 1], 
                    c=[cmap(i)], label=f'Digit {i}', alpha=0.3, s=10)
        
        # Anchor (Star)
        plt.scatter(anchors_2d[i, 0], anchors_2d[i, 1], 
                    c=[cmap(i)], marker='*', s=300, edgecolors='black', linewidth=2)
        
    plt.title("Multimodal Alignment: Logic Anchors vs Visual Projections")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = "tempdata/vision_alignment.png"
    plt.savefig(out_path)
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    train_alignment()
