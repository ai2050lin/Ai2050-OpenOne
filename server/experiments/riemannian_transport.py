
import os
import sys

# Set environment variables for model loading BEFORE other imports
os.environ["HF_HOME"] = r"D:\develop\model"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Redirect stdout to file
sys.stdout = open("riemannian_results.txt", "w", encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from transformer_lens import HookedTransformer

# Configuration
MODEL_NAME = "gpt2"
LAYER_ID = 6

def load_model():
    print(f"Loading {MODEL_NAME}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def get_resid_at_last_token(model, text, layer_id):
    cache_name = f"blocks.{layer_id}.hook_resid_post"
    _, cache = model.run_with_cache(text)
    return cache[cache_name][0, -1, :]

class TransportNet(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # Input: Subject Vector (x)
        # Output: Transport Vector (v(x))
        # Goal: x + v(x) approx Object Vector (y)
        
        # We use a simple MLP to model the vector field
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model)
        )
        
    def forward(self, x):
        return self.net(x)

def train_and_evaluate_transport(model, layer_id):
    print(f"\n--- Learning Riemannian Transport (Layer {layer_id}) ---")
    
    # 1. Dataset Construction
    # Relation: Country -> Capital
    pairs = [
        ("China", "Beijing"), ("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), 
        ("Japan", "Tokyo"), ("Russia", "Moscow"), ("Spain", "Madrid"), ("UK", "London"),
        ("Ukraine", "Kiev"), ("Canada", "Ottawa"), ("Egypt", "Cairo"), ("Brazil", "Brasilia"),
        ("India", "Delhi"), ("Australia", "Canberra"), ("Poland", "Warsaw"), ("Sweden", "Stockholm")
    ]
    
    # Split into Train and Test
    # We need enough training data to learn the curvature
    train_pairs = pairs[:12]
    test_pairs = pairs[12:]
    
    context_template = "The capital of {} is"
    
    # 2. Extract Vectors
    X_train = []
    Y_train = []
    
    print("Extracting training vectors...")
    for sub, obj in train_pairs:
        # Input: "The capital of China is" -> Last token resid (Subject context)
        # Target: We actually want to predict the *Vector* that moves to the Object.
        # But wait, standard transport is v = y - x.
        # So we want to learn v(x) s.t. x + v(x) = y.
        # Let's get "Subject" activation and "Object" activation.
        
        # Actually, let's use the standard analogy setup:
        # Prompt: "The capital of China is Beijing" (We need the vector at 'Beijing' vs 'China'?)
        # Simplification:
        # x = Act("The capital of China is") [The state right before prediction]
        # y = Unembed(Object) ? No, we are operating in residual stream.
        # Let's target the residual state of the Object token *if it were there*.
        # Or better: x = Act("Rome"), y = Act("Italy")? No, relations are directional.
        
        # Let's def:
        # Start Point x: Act("The capital of China is")
        # End Point y: Act("The capital of China is Beijing") [State at 'Beijing']
        # The vector v = y - x is the "step" taken by processing 'Beijing'.
        # No, we want to predict 'Beijing' from 'is'.
        
        # Let's go with the Manifold Surgery definition:
        # Transport Vector v maps Sub -> Obj.
        # We define v = Act("Paris") - Act("France") [Isolated]
        # Then we apply it to Act("China") to get Act("Beijing").
        
        x_act = get_resid_at_last_token(model, sub, layer_id)
        y_act = get_resid_at_last_token(model, obj, layer_id)
        
        X_train.append(x_act)
        Y_train.append(y_act)
        
    X_train = torch.stack(X_train).detach()
    Y_train = torch.stack(Y_train).detach() # Target Object Vectors
    
    # 3. Baseline: Constant Transport
    # v_const = Avg(Y - X)
    v_const = (Y_train - X_train).mean(dim=0)
    
    # Evaluate Baseline on Test
    print("Evaluating Baseline (Constant Transport)...")
    base_errors = []
    cos_sims_base = []
    
    for sub, obj in test_pairs:
        x_test = get_resid_at_last_token(model, sub, layer_id)
        y_test = get_resid_at_last_token(model, obj, layer_id)
        
        y_pred = x_test + v_const
        
        error = (y_pred - y_test).norm().item()
        sim = F.cosine_similarity(y_pred.unsqueeze(0), y_test.unsqueeze(0)).item()
        
        base_errors.append(error)
        cos_sims_base.append(sim)
        
    avg_base_error = np.mean(base_errors)
    avg_base_sim = np.mean(cos_sims_base)
    print(f"Baseline - MSE: {avg_base_error:.4f}, CosSim: {avg_base_sim:.4f}")
    
    # 4. Train TransportNet (Riemannian / Vector Field)
    # v(x) = MLP(x)
    # Loss = || (x + v(x)) - y ||^2
    
    print("Training TransportNet...")
    model_dim = model.cfg.d_model
    net = TransportNet(model_dim).to(model.cfg.device)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    
    epochs = 1000
    for epoch in range(epochs):
        optimizer.zero_grad()
        transport_vecs = net(X_train)
        y_preds = X_train + transport_vecs
        loss = F.mse_loss(y_preds, Y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
    # 5. Evaluate TransportNet
    print("Evaluating TransportNet (Vector Field)...")
    net_errors = []
    cos_sims_net = []
    
    for sub, obj in test_pairs:
        x_test = get_resid_at_last_token(model, sub, layer_id)
        y_test = get_resid_at_last_token(model, obj, layer_id)
        
        # Predict transport vector specific to this subject x
        with torch.no_grad():
            v_pred = net(x_test.unsqueeze(0)).squeeze(0)
            
        y_pred = x_test + v_pred
        
        error = (y_pred - y_test).norm().item()
        sim = F.cosine_similarity(y_pred.unsqueeze(0), y_test.unsqueeze(0)).item()
        
        net_errors.append(error)
        cos_sims_net.append(sim)
        
    avg_net_error = np.mean(net_errors)
    avg_net_sim = np.mean(cos_sims_net)
    print(f"TransportNet - MSE: {avg_net_error:.4f}, CosSim: {avg_net_sim:.4f}")
    
    improvement = (avg_base_error - avg_net_error) / avg_base_error * 100
    print(f"Improvement in MSE: {improvement:.2f}%")
    
    # 6. Analyze Curvature (Rotation)
    # If TransportNet works better, it means v(x) changes with x.
    # Let's measure CosSim between predicted vectors for Train set.
    with torch.no_grad():
        vecs = net(X_train) # [N, d_model]
        # Normalize
        vecs_norm = F.normalize(vecs, dim=1)
        # Compute pairwise similarity matrix
        sim_matrix = torch.mm(vecs_norm, vecs_norm.t())
        avg_pairwise_sim = sim_matrix.mean().item()
        
    print(f"Average Pairwise Similarity of Transport Vectors: {avg_pairwise_sim:.4f}")
    if avg_pairwise_sim < 0.9:
        print("Conclusion: Significant rotation detected (Curved Manifold).")
    else:
        print("Conclusion: Vectors are mostly parallel (Flat Manifold).")

def main():
    model = load_model()
    if model:
        train_and_evaluate_transport(model, LAYER_ID)

if __name__ == "__main__":
    main()
