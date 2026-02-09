
import os
import sys

# Set environment variables for model loading BEFORE other imports
os.environ["HF_HOME"] = r"D:\develop\model"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Redirect stdout to file
sys.stdout = open("rotor_results.txt", "w", encoding="utf-8")

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import orthogonal_procrustes
from scipy.spatial import procrustes

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

def train_and_evaluate_rotor(model, layer_id):
    print(f"\n--- Testing Geometric Algebra Rotor Hypothesis (Layer {layer_id}) ---")
    
    # 1. Dataset Construction (Same as before)
    pairs = [
        ("China", "Beijing"), ("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), 
        ("Japan", "Tokyo"), ("Russia", "Moscow"), ("Spain", "Madrid"), ("UK", "London"),
        ("Ukraine", "Kiev"), ("Canada", "Ottawa"), ("Egypt", "Cairo"), ("Brazil", "Brasilia"),
        ("India", "Delhi"), ("Australia", "Canberra"), ("Poland", "Warsaw"), ("Sweden", "Stockholm")
    ]
    
    train_pairs = pairs[:12]
    test_pairs = pairs[12:]
    
    # 2. Extract Vectors
    X_train = []
    Y_train = []
    
    print("Extracting training vectors...")
    for sub, obj in train_pairs:
        x_act = get_resid_at_last_token(model, sub, layer_id)
        y_act = get_resid_at_last_token(model, obj, layer_id)
        X_train.append(x_act.cpu().numpy())
        Y_train.append(y_act.cpu().numpy())
        
    X_train = np.stack(X_train) # [N, D]
    Y_train = np.stack(Y_train) # [N, D]
    
    # 3. Method 1: Vector Translation (Baseline)
    # v = Mean(Y - X)
    v_vec = np.mean(Y_train - X_train, axis=0)
    
    # 4. Method 2: Rotation Matrix (Rotor)
    # Solve min || XR - Y ||_F s.t. R^T R = I
    # Procrustes Analysis
    # Center data? No, we transform absolute vectors.
    # Orthogonal Procrustes: R, scale = orthogonal_procrustes(X, Y)
    # usually computes R s.t. || RX - Y || or similar.
    # Scipy: "Compute R s.t. || A @ R - B ||_F is minimized." -> R maps A to B.
    # Here A=X_train, B=Y_train
    
    R, scale = orthogonal_procrustes(X_train, Y_train)
    print(f"Optimal Rotation Found. Scale (singular values sum): {scale:.4f}")
    
    # 5. Evaluate on Test Set
    print("\nEvaluating on Test Set...")
    
    vec_errors = []
    rot_errors = []
    
    vec_sims = []
    rot_sims = []
    
    # Analyze Norm Preservation
    norm_ratios_orig = []
    norm_ratios_vec = []
    norm_ratios_rot = []
    
    for sub, obj in test_pairs:
        x_test = get_resid_at_last_token(model, sub, layer_id).cpu().numpy()
        y_test = get_resid_at_last_token(model, obj, layer_id).cpu().numpy()
        
        # Ground Truth Ratio
        norm_ratios_orig.append(np.linalg.norm(y_test) / np.linalg.norm(x_test))
        
        # 1. Vector Prediction
        y_pred_vec = x_test + v_vec
        err_vec = np.linalg.norm(y_pred_vec - y_test)
        sim_vec = np.dot(y_pred_vec, y_test) / (np.linalg.norm(y_pred_vec) * np.linalg.norm(y_test))
        vec_errors.append(err_vec)
        vec_sims.append(sim_vec)
        norm_ratios_vec.append(np.linalg.norm(y_pred_vec) / np.linalg.norm(x_test))
        
        # 2. Rotor Prediction
        # y = x @ R
        y_pred_rot = x_test @ R
        err_rot = np.linalg.norm(y_pred_rot - y_test)
        sim_rot = np.dot(y_pred_rot, y_test) / (np.linalg.norm(y_pred_rot) * np.linalg.norm(y_test))
        rot_errors.append(err_rot)
        rot_sims.append(sim_rot)
        norm_ratios_rot.append(np.linalg.norm(y_pred_rot) / np.linalg.norm(x_test))
        
    print(f"Vector Translation - MSE: {np.mean(vec_errors):.4f}, CosSim: {np.mean(vec_sims):.4f}")
    print(f"Rotor (Rotation)   - MSE: {np.mean(rot_errors):.4f}, CosSim: {np.mean(rot_sims):.4f}")
    
    mse_improv = (np.mean(vec_errors) - np.mean(rot_errors)) / np.mean(vec_errors) * 100
    print(f"Improvement in MSE: {mse_improv:.2f}%")
    
    print("\nNorm Analysis (Output/Input Ratio):")
    print(f"True Relation Ratio: {np.mean(norm_ratios_orig):.4f}")
    print(f"Vector Pred Ratio  : {np.mean(norm_ratios_vec):.4f}")
    print(f"Rotor Pred Ratio   : {np.mean(norm_ratios_rot):.4f} (Should be 1.0 for orthogonal matrix)")
    
    if mse_improv > 0:
        print("Conclusion: Rotor hypothesis holds better than Vector hypothesis.")
    else:
        print("Conclusion: Rotor hypothesis does NOT improve over Vector hypothesis.")

def main():
    model = load_model()
    if model:
        train_and_evaluate_rotor(model, LAYER_ID)

if __name__ == "__main__":
    main()
