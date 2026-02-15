
import os
import random
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fibernet_v2 import DecoupledFiberNet

# --- Configuration ---
BATCH_SIZE = 64
EPOCHS = 10
LR = 0.001

# --- Synthetic Data Generation ---
# 4x4 Grid = 16 patches
# Top-Left: 0, 1, 4, 5
# Bottom-Right: 10, 11, 14, 15
TL_INDICES = [0, 1, 4, 5]
BR_INDICES = [10, 11, 14, 15]

def generate_data(n_samples=1000):
    data = []
    labels = []
    
    for _ in range(n_samples):
        # Init grid with low noise
        grid = torch.randn(16, 1) * 0.1
        
        label = random.randint(0, 1)
        if label == 0: # Top-Left Active
            for idx in TL_INDICES:
                grid[idx] += 1.0
        else: # Bottom-Right Active
            for idx in BR_INDICES:
                grid[idx] += 1.0
                
        data.append(grid)
        labels.append(label)
        
    return torch.stack(data), torch.tensor(labels)

train_data, train_labels = generate_data(2000)

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]

trainloader = torch.utils.data.DataLoader(SimpleDataset(train_data, train_labels), batch_size=BATCH_SIZE, shuffle=True)

# --- Model Wrapper ---
class FiberVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_len = 16
        self.d_model = 32
        self.d_logic = 16
        self.d_memory = 32
        
        self.pos_embed = nn.Embedding(self.seq_len, self.d_logic)
        self.pixel_proj = nn.Linear(1, self.d_memory) # 1 pixel per patch
        
        # We need to manually construct layers since DecoupledFiberNet is hardcoded for vocab
        # But wait, DecoupledFiberNet just uses FiberLayer.
        # We can just instantiate FiberLayer here.
        # But we need to import FiberLayer. It's not exported.
        # We will use the trick: Instantiate DecoupledFiberNet and validly use its layers.
        self.temp_net = DecoupledFiberNet(vocab_size=1, d_model=32, n_layers=1)
        self.layer = self.temp_net.layers[0] # Just 1 layer for simplicity
        
        self.cls_head = nn.Linear(self.d_memory, 2)
        
    def forward(self, x):
        # x: [B, 16, 1]
        batch, seq, _ = x.shape
        device = x.device
        
        # Logic
        positions = torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)
        curr_logic = self.pos_embed(positions)
        
        # Memory
        curr_memory = self.pixel_proj(x)
        
        # Evolve (1 Layer)
        res_l = curr_logic
        curr_logic, _ = self.layer.logic_attn(curr_logic, curr_logic, curr_logic)
        curr_logic = self.layer.logic_norm1(res_l + curr_logic)
        res_l = curr_logic
        curr_logic = self.layer.logic_norm2(res_l + self.layer.logic_ffn(curr_logic))
        
        # Logic-Driven Attention (Capture Weights)
        lda = self.layer.attn
        head_dim_logic = lda.d_logic // lda.nhead
        Q = lda.W_Q(curr_logic).reshape(batch, seq, lda.nhead, head_dim_logic).transpose(1, 2)
        K = lda.W_K(curr_logic).reshape(batch, seq, lda.nhead, head_dim_logic).transpose(1, 2)
        scores = (Q @ K.transpose(-2, -1)) / (head_dim_logic ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1) # [B, Head, Seq, Seq]

        V = lda.W_V(curr_memory).reshape(batch, seq, lda.nhead, lda.d_memory // lda.nhead).transpose(1, 2)
        transported = attn_weights @ V
        transported = transported.transpose(1, 2).flatten(2)
        transported = lda.W_O(transported)
        
        res_m = curr_memory
        curr_memory = self.layer.mem_norm1(res_m + transported)
        res_m = curr_memory
        curr_memory = self.layer.mem_norm2(res_m + self.layer.mem_ffn(curr_memory))
        
        # Pool
        pooled = curr_memory.mean(dim=1)
        return self.cls_head(pooled), attn_weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FiberVision().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

print("Training FiberVision (Synthetic Corner Task)...")
for epoch in range(EPOCHS):
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch}: Loss {running_loss / len(trainloader):.4f}")

# Visualize
inputs, _ = next(iter(trainloader))
_, attns = model(inputs.to(device))
attn_map = attns[0, 0].detach().cpu().numpy() # [16, 16]

plt.figure(figsize=(6, 5))
plt.imshow(attn_map, cmap='viridis')
plt.title("Logic Stream Attention (4x4 Grid)")
plt.xlabel("Key Position (0-15)")
plt.ylabel("Query Position (0-15)")
plt.colorbar()

os.makedirs('tempdata/fibernet', exist_ok=True)
plt.savefig('tempdata/fibernet/vision_logic.png')
print("Saved plot to tempdata/fibernet/vision_logic.png")
