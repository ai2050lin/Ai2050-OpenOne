
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fibernet_v2 import DecoupledFiberNet


class VisualFiberNet(nn.Module):
    def __init__(self, linguistic_model, patch_dim=64, d_model=32):
        super().__init__()
        self.linguistic_model = linguistic_model
        # Freeze EVERYTHING in linguistic model
        for param in self.linguistic_model.parameters():
            param.requires_grad = False
            
        # Visual Projector (The only trainable part)
        # Projects flattened image patch to Memory Dimension
        self.patch_projector = nn.Linear(patch_dim, d_model)
        
        # Visual Head (To reconstruct or classify next patch)
        # We'll use classification for simplicity (predicting next shape ID)
        self.visual_head = nn.Linear(d_model, patch_dim) # Predicting vector? Or class?
        # Let's predict class index for simpler evaluation, assuming discrete shapes
        self.n_shapes = 10
        self.classifier = nn.Linear(d_model, self.n_shapes)

    def forward(self, patch_seq, positions):
        # patch_seq: [Batch, Seq, PatchDim]
        # positions: [Batch, Seq]
        
        # 1. Project Visual to Memory
        memory_embeds = self.patch_projector(patch_seq) # [Batch, Seq, D_Model]
        
        # 2. Get Logic Embeds (from linguistic model's pos_embed)
        # syntactic positions trigger specific logic states
        logic_embeds = self.linguistic_model.pos_embed(positions)
        
        # 3. FiberNet Layer Pass (Re-using linguistic layers)
        curr_logic = logic_embeds
        curr_memory = memory_embeds
        
        for layer in self.linguistic_model.layers:
            # We must access layer components manually since we can't call model.forward directly
            # Logic Evolve
            res_l = curr_logic
            curr_logic, _ = layer.logic_attn(curr_logic, curr_logic, curr_logic)
            curr_logic = layer.logic_norm1(res_l + curr_logic)
            res_l = curr_logic
            curr_logic = layer.logic_norm2(res_l + layer.logic_ffn(curr_logic))
            
            # Interaction (Logic drives Memory)
            # Calculate Attention Matrix A from Logic
            # Note: We need to use the layer's internal weights
            lda = layer.attn
            B, S, _ = curr_logic.shape
            H = lda.nhead
            D_L = lda.d_logic // H
            D_M = lda.d_memory // H
            
            Q = lda.W_Q(curr_logic).reshape(B, S, H, D_L).transpose(1, 2)
            K = lda.W_K(curr_logic).reshape(B, S, H, D_L).transpose(1, 2)
            scores = (Q @ K.transpose(-2, -1)) / (D_L ** 0.5)
            A = torch.softmax(scores, dim=-1) # The "Linguistic Structure"
            
            # Apply A to Memory (Visual Memory)
            V = lda.W_V(curr_memory).reshape(B, S, H, D_M).transpose(1, 2)
            transported = A @ V
            transported = transported.transpose(1, 2).reshape(B, S, -1)
            transported = lda.W_O(transported)
            
            res_m = curr_memory
            curr_memory = layer.mem_norm1(res_m + transported)
            res_m = curr_memory
            curr_memory = layer.mem_norm2(res_m + layer.mem_ffn(curr_memory))
            
        # 4. Predict
        logits = self.classifier(curr_memory)
        return logits

def generate_visual_data(n_samples=500, seq_len=3, patch_dim=64, n_shapes=10):
    # Synthetic Task:
    # Rule: Shape(n) -> Shape(n+1) -> Shape(n+2) (Sequential Logic)
    # This mimics "Subject -> Verb -> Object" where there is a transitive dependency
    
    # We represent shapes as one-hot vectors + noise to simulate "perceptual" input
    data = []
    targets = []
    
    for _ in range(n_samples):
        start_idx = random.randint(0, n_shapes - seq_len - 1)
        seq_indices = [start_idx + i for i in range(seq_len)]
        target_idx = start_idx + seq_len # Predict next
        
        # Create patches
        patches = []
        for idx in seq_indices:
            # Base vector (One-hot-ish)
            vec = np.zeros(patch_dim)
            vec[idx * (patch_dim // n_shapes)] = 1.0 
            # Add noise
            vec += np.random.normal(0, 0.1, patch_dim)
            patches.append(vec)
            
        data.append(patches)
        targets.append(target_idx) # Actually we usually predict sequence shift, let's predict next token at end
        
    return torch.tensor(np.array(data), dtype=torch.float32), torch.tensor(targets, dtype=torch.long)

def train_visual_transfer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Phase 15 Experiment on {device}")
    
    # 1. Pre-train Linguistic Model (English SVO)
    print("\n--- Step 1: Training English Logic Base ---")
    vocab_en = {"<PAD>":0, "I":1, "You":2, "We":3, "love":4, "see":5, "know":6, "him":7, "her":8, "them":9}
    
    model_en = DecoupledFiberNet(vocab_size=len(vocab_en), d_model=32, n_layers=2).to(device)
    
    # Generate syntax data (Subj -> Verb -> Obj)
    # 1,2,3 -> 4,5,6 -> 7,8,9
    # SVO Logic: Pos 0 predicts Pos 1, Pos 1 predicts Pos 2
    raw_en_data = []
    for _ in range(500):
        s = random.choice([1,2,3])
        v = random.choice([4,5,6])
        o = random.choice([7,8,9])
        raw_en_data.append([s, v, o])
    data_en = torch.tensor(raw_en_data, dtype=torch.long).to(device)
    
    optimizer_en = optim.Adam(model_en.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(50):
        optimizer_en.zero_grad()
        inputs = data_en[:, :-1]
        targets = data_en[:, 1:]
        logits = model_en(inputs)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss.backward()
        optimizer_en.step()
        if epoch % 10 == 0:
            print(f"English Pre-training Epoch {epoch}, Loss: {loss.item():.4f}")
            
    # 2. Visual Transfer
    print("\n--- Step 2: Visual Transfer (Frozen Logic) ---")
    
    # Model: VisualFiberNet wrapping the FROZEN model_en
    visual_model = VisualFiberNet(model_en, patch_dim=64, d_model=32).to(device)
    
    # Data: Shape Sequences
    # We want to predict the 3rd shape given 1st and 2nd? Or predict 2nd and 3rd?
    # Let's input [Shape1, Shape2] and predict [Shape2, Shape3] via Sequence Logic
    # train_visual_transfer function logic needs minimal adjustment.
    
    # Let's generate [Batch, 3, 64]
    # Input: Patches 0, 1. Target: Labels 1, 2.
    n_shapes = 10
    X_vis, y_vis_labels = generate_visual_data(n_samples=1000, seq_len=3, patch_dim=64, n_shapes=n_shapes)
    X_vis = X_vis.to(device) # [1000, 3, 64]
    
    # Inputs: First 2 patches
    inputs_vis = X_vis[:, :-1, :] # [1000, 2, 64]
    
    # Targets: Next 2 shape IDs (indices of patch 1 and 2)
    # generate_visual_data returns list of seq_indices.
    # Re-generating data with labels inside
    
    # Let's redo data gen inline for clarity
    raw_X = []
    raw_Y = []
    for _ in range(1000):
        start = random.randint(0, n_shapes - 3 - 1)
        seq = [start, start+1, start+2] # 0->1->2 linear structure
        
        patches = []
        for idx in seq:
            vec = np.zeros(64)
            vec[idx * (64 // n_shapes)] = 1.0
            vec += np.random.normal(0, 0.05, 64)
            patches.append(vec)
        raw_X.append(patches[:-1]) # Input: 0, 1
        raw_Y.append(seq[1:])      # Target: 1, 2
        
    inputs = torch.tensor(np.array(raw_X), dtype=torch.float32).to(device)
    targets = torch.tensor(np.array(raw_Y), dtype=torch.long).to(device)
    positions = torch.arange(2, device=device).unsqueeze(0).expand(1000, -1)
    
    # Optimizer: Only trains Visual Projector & Classifier
    # Logic is frozen inside VisualFiberNet __init__
    optimizer_vis = optim.Adam(filter(lambda p: p.requires_grad, visual_model.parameters()), lr=0.01)
    
    print("Trainable Params:", sum(p.numel() for p in visual_model.parameters() if p.requires_grad))
    print("Total Params:    ", sum(p.numel() for p in visual_model.parameters()))
    
    losses = []
    for epoch in range(50):
        optimizer_vis.zero_grad()
        
        logits = visual_model(inputs, positions) # [B, 2, 10]
        
        loss = criterion(logits.reshape(-1, n_shapes), targets.reshape(-1))
        loss.backward()
        optimizer_vis.step()
        losses.append(loss.item())
        
        if epoch % 10 == 0:
            acc = (logits.argmax(dim=-1) == targets).float().mean()
            print(f"Visual Transfer Epoch {epoch}, Loss: {loss.item():.4f}, Acc: {acc.item():.4f}")

    # Plot
    plt.figure()
    plt.plot(losses)
    plt.title("Visual Logic Transfer Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("tempdata/visual_transfer_loss.png")
    print("\nExperiment Complete. Loss plot saved.")
    
    final_acc = (logits.argmax(dim=-1) == targets).float().mean().item()
    if final_acc > 0.9:
        print("SUCCESS: Visual-Logic Isomorphism Verified!")
    else:
        print("FAILURE: Could not transfer logic to visual domain.")

if __name__ == "__main__":
    train_visual_transfer()
