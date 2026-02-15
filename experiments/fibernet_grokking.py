
import math
import os
import sys
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fibernet_v2 import DecoupledFiberNet

# --- Configuration (Grokking Setup) ---
P = 997 # Larger prime
TOTAL_SAMPLES = P * P
TRAIN_FRACTION = 0.01 # 1% extremely sparse data! (Severe overfitting test)
TRAIN_SAMPLES = int(TOTAL_SAMPLES * TRAIN_FRACTION)

D_MODEL = 128
N_LAYERS = 2
EPOCHS = 2000 # Need long training for grokking
BATCH_SIZE = 512
LR = 1e-3
WD = 1e-2 # Weight decay is crucial for grokking, but let's test if FiberNet needs it less.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")

# --- Standard Transformer Baseline ---
class StandardTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Using standard attention
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=2, dim_feedforward=512, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, x):
        emb = self.embedding(x)
        out = self.transformer(emb)
        # Pooling: Mean of last layer
        pool = out.mean(dim=1)
        return self.head(pool)

# --- Data Generation ---
print(f"Generating data for P={P}...")
X_all = []
Y_all = []
for a in range(P):
    for b in range(P):
        res = (a + b) % P
        X_all.append([a, b])
        Y_all.append(res)
        
X_all = torch.tensor(X_all, dtype=torch.long)
Y_all = torch.tensor(Y_all, dtype=torch.long)

# Shuffle and Split
idx = torch.randperm(len(X_all))
train_idx = idx[:TRAIN_SAMPLES]
test_idx = idx[TRAIN_SAMPLES:] # The other 99%

X_train, Y_train = X_all[train_idx], Y_all[train_idx]
X_test, Y_test = X_all[test_idx], Y_all[test_idx]

print(f"Train Set: {len(X_train)} samples ({TRAIN_FRACTION:.1%})")
print(f"Test Set: {len(X_test)} samples ({1-TRAIN_FRACTION:.1%})")

train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True)
# Test loader (large batch for evaluation)
test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=4096, shuffle=False)

# --- Training Loop ---
def run_experiment(model, name):
    print(f"\n--- Training {name} ---")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    criterion = nn.CrossEntropyLoss()
    
    train_accs = []
    test_accs = []
    loss_history = []
    
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            if len(logits.shape) == 3:
                logits = logits.mean(dim=1)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == by).sum().item()
            total += by.size(0)
            
        train_acc = correct / total
        train_accs.append(train_acc)
        loss_history.append(total_loss / len(train_loader))
        
        # Test every 100 epochs
        if epoch % 100 == 0 or epoch == EPOCHS - 1:
            model.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for bx, by in test_loader:
                    bx, by = bx.to(device), by.to(device)
                    logits = model(bx)
                    if len(logits.shape) == 3:
                        logits = logits.mean(dim=1)
                    preds = logits.argmax(dim=-1)
                    test_correct += (preds == by).sum().item()
                    test_total += by.size(0)
            
            test_acc = test_correct / test_total
            test_accs.append(test_acc)
            
            print(f"[{name}] Epoch {epoch}: Loss {loss_history[-1]:.4f} | Train Acc {train_acc:.2%} | Test Acc {test_acc:.2%}")
            
            if test_acc > 0.99:
                print(f"[{name}] Grokked at Epoch {epoch}!")
                break
                
    duration = time.time() - start_time
    print(f"[{name}] Finished in {duration:.2f}s. Final Test Acc: {test_accs[-1] if test_accs else 0:.2%}")
    return train_accs, test_accs

# --- Run ---
std_model = StandardTransformer(vocab_size=P, d_model=D_MODEL, n_layers=N_LAYERS)
std_train, std_test = run_experiment(std_model, "StandardTransformer")

fiber_model = DecoupledFiberNet(vocab_size=P, d_model=D_MODEL, n_layers=N_LAYERS, group_type='circle', max_len=10)
fiber_train, fiber_test = run_experiment(fiber_model, "FiberNet")

# --- Plot ---
plt.figure(figsize=(10, 5))

# Plot Test Accuracy (Generalization)
x_axis_std = range(0, len(std_test) * 100, 100)
# Adjust if early stopped
if len(x_axis_std) > len(std_test): x_axis_std = x_axis_std[:len(std_test)]

x_axis_fib = range(0, len(fiber_test) * 100, 100)
if len(x_axis_fib) > len(fiber_test): x_axis_fib = x_axis_fib[:len(fiber_test)]

plt.plot(x_axis_std, std_test, label='Standard (Test Acc)', linestyle='--', color='blue')
plt.plot(x_axis_fib, fiber_test, label='FiberNet (Test Acc)', linewidth=2, color='red')

plt.xlabel('Epoch')
plt.ylabel('Test Accuracy (Generalization)')
plt.title(f'Grokking Test: Mod {P} Addition with 1% Data')
plt.legend()
plt.grid(True)

os.makedirs('tempdata/fibernet', exist_ok=True)
plt.savefig('tempdata/fibernet/grokking_test.png')
print("Saved plot to tempdata/fibernet/grokking_test.png")
