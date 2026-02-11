
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from experiments.toy_experiment.group_theory_dataset import GroupTheoryDataset
from experiments.toy_experiment.toy_models import ToyFiberNet, ToyTransformer

LOG_FILE = "d:\\develop\\TransformerLens-main\\experiments\\toy_experiment\\training_log.json"

def log_metrics(epoch, loss, accuracy, model_name):
    # Read existing logs or create new
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r') as f:
                logs = json.load(f)
        except:
            logs = {"Transformer": [], "FiberNet": []}
    else:
        logs = {"Transformer": [], "FiberNet": []}
    
    # Check if key exists (in case of fresh start)
    if model_name not in logs:
        logs[model_name] = []
        
    logs[model_name].append({
        "epoch": epoch,
        "loss": loss,
        "accuracy": accuracy,
        "timestamp": time.time()
    })
    
    with open(LOG_FILE, 'w') as f:
        json.dump(logs, f)

def train_model(model, train_loader, epochs=500, lr=0.001, name="Model"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"\n--- Training {name} ---")
    start_time = time.time()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # inputs: [batch, 2] (a, b)
            # targets: [batch] (result)
            a_idx = inputs[:, 0]
            b_idx = inputs[:, 1]
            
            optimizer.zero_grad()
            outputs = model(a_idx, b_idx) # [batch, vocab]
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        # Log to JSON
        log_metrics(epoch+1, avg_loss, accuracy, name)
        
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
            
        if accuracy > 99.5:
            print(f"converged at epoch {epoch+1}")
            break
            
    end_time = time.time()
    print(f"{name} Training Time: {end_time - start_time:.2f}s")
    return model

def main():
    # Configuration
    GROUP_ORDER = 113 # Prime number for Z_p field
    NUM_SAMPLES = 5000
    BATCH_SIZE = 64
    EPOCHS = 1000
    
    # Clear log file at start
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
        
    # Dataset
    dataset = GroupTheoryDataset(group_type='Z_n', order=GROUP_ORDER, num_samples=NUM_SAMPLES)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Dataset: Z_{GROUP_ORDER} Addition. {len(dataset)} samples.")
    
    # Models
    # Transformer Baseline
    transformer = ToyTransformer(vocab_size=GROUP_ORDER, d_model=64, n_head=4, n_layer=2)
    train_model(transformer, train_loader, epochs=EPOCHS, name="Transformer")
    
    # FiberNet
    # Note: d_manifold=64, d_fiber=256
    fibernet = ToyFiberNet(vocab_size=GROUP_ORDER, d_model=64)
    # FiberNet learns the connection form.
    # The ManifoldStream is trivial (constant operator).
    # The FiberStream learns embeddings.
    # The ConnectionLayer learns the rotation.
    train_model(fibernet, train_loader, epochs=EPOCHS, name="FiberNet")

if __name__ == "__main__":
    main()
