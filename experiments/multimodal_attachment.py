
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


class VisionProjector(nn.Module):
    def __init__(self, input_dim=64*64, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

def generate_visual_z12_data_with_labels(n_samples=1000, modulus=12):
    img_size = 64
    data = []
    targets = [] # c
    sources = [] # [a, b]
    
    for _ in range(n_samples):
        a = np.random.randint(0, modulus)
        b = np.random.randint(0, modulus)
        c = (a + b) % modulus
        
        def make_img(val):
            img = np.zeros((img_size, img_size), dtype=np.float32)
            row = int((val / modulus) * img_size)
            # Make the bar thicker and clearer
            img[max(0, row-3):min(img_size, row+3), :] = 1.0
            img += np.random.normal(0, 0.05, (img_size, img_size)) # less noise
            return img.flatten()
            
        img_a = make_img(a)
        img_b = make_img(b)
        
        data.append([img_a, img_b])
        targets.append(c)
        sources.append([a, b])
            
    return torch.tensor(np.array(data), dtype=torch.float32), torch.tensor(targets, dtype=torch.long), torch.tensor(sources, dtype=torch.long)

def train_multimodal_attachment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running HLAI Step 2: Multimodal Attachment (Revised) on {device}...")
    
    # 1. Logic Engine (Pre-trained & Frozen)
    print(">>> Pre-training Logic Engine...")
    VOCAB_SIZE = 12
    D_MODEL = 64
    logic_model = DecoupledFiberNet(vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layers=2).to(device)
    
    # Freeze Embeddings (Random)
    logic_model.content_embed.requires_grad_(False)
    
    # Train Logic
    gen = LogicDatasetGenerator(vocab_size=VOCAB_SIZE)
    x_sym, y_sym = gen.generate_cyclic_data(2000, modulus=12) # More data
    x_sym, y_sym = x_sym.to(device), y_sym.to(device)
    
    opt_logic = optim.Adam(filter(lambda p: p.requires_grad, logic_model.parameters()), lr=0.01)
    crit_ce = nn.CrossEntropyLoss()
    
    for epoch in range(40):
        opt_logic.zero_grad()
        logits = logic_model(x_sym)
        loss = crit_ce(logits[:, -1, :], y_sym)
        loss.backward()
        opt_logic.step()
        if epoch % 10 == 0:
            print(f"Logic Ep {epoch}: Loss {loss.item():.4f}")
            
    print(">>> Logic Frozen.")
    for p in logic_model.parameters(): p.requires_grad = False
    
    # 2. Vision Projector
    vision_proj = VisionProjector(input_dim=64*64, output_dim=D_MODEL).to(device)
    opt_vis = optim.Adam(vision_proj.parameters(), lr=0.001)
    crit_mse = nn.MSELoss()
    
    # Data
    x_imgs, y_labels, source_labels = generate_visual_z12_data_with_labels(2000, 12)
    x_imgs, y_labels, source_labels = x_imgs.to(device), y_labels.to(device), source_labels.to(device)
    
    losses = []
    
    print(">>> Training Vision with Alignment Loss...")
    for epoch in range(100): # More epochs
        opt_vis.zero_grad()
        
        B, Seq, Dim = x_imgs.shape
        img_flat = x_imgs.reshape(B*Seq, Dim)
        projected = vision_proj(img_flat) # [B*2, 64]
        memory_embeds = projected.reshape(B, Seq, D_MODEL)
        
        # 1. Alignment Loss
        # We want projected[i] to match logic_model.content_embed(source_labels[i])
        source_flat = source_labels.reshape(B*Seq)
        target_embeds = logic_model.content_embed(source_flat) # [B*2, 64]
        
        loss_align = crit_mse(projected, target_embeds)
        
        # 2. Logic Task Loss
        logits = logic_model(memory_embeds=memory_embeds)
        loss_task = crit_ce(logits[:, -1, :], y_labels)
        
        # Combine
        loss = loss_task + 10.0 * loss_align # Strong alignment weight
        
        loss.backward()
        opt_vis.step()
        
        losses.append(loss.item())
        
        if epoch % 10 == 0:
            acc = (logits[:, -1, :].argmax(dim=-1) == y_labels).float().mean()
            print(f"Vis Ep {epoch}: Loss {loss.item():.4f} (Align {loss_align.item():.4f} Task {loss_task.item():.4f}) Acc {acc.item():.4f}")
            
    # Final Check
    if acc > 0.95:
        print("SUCCESS: Vision Grounding Achieved.")
    else:
        print("FAILURE: Grounding failed.")

if __name__ == "__main__":
    train_multimodal_attachment()
