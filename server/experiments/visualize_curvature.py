
import os
import sys

# Set environment variables for model loading BEFORE other imports
os.environ["HF_HOME"] = r"D:\develop\model"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Redirect stdout to file (optional, but good for logging)
# sys.stdout = open("visualization_log.txt", "w", encoding="utf-8")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

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

def visualize_curvature(model, layer_id):
    print(f"\n--- Visualizing Manifold Curvature (Layer {layer_id}) ---")
    
    # 1. Dataset
    pairs = [
        ("China", "Beijing"), ("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), 
        ("Japan", "Tokyo"), ("Russia", "Moscow"), ("Spain", "Madrid"), ("UK", "London"),
        ("Ukraine", "Kiev"), ("Canada", "Ottawa"), ("Egypt", "Cairo"), ("Brazil", "Brasilia"),
        ("India", "Delhi"), ("Australia", "Canberra"), ("Poland", "Warsaw"), ("Sweden", "Stockholm"),
        ("USA", "Washington"), ("Turkey", "Ankara"), ("Thailand", "Bangkok"), ("Vietnam", "Hanoi")
    ]
    
    X = []
    Y = []
    Labels = []
    
    print("Extracting vectors...")
    for sub, obj in pairs:
        x_act = get_resid_at_last_token(model, sub, layer_id).cpu().numpy()
        y_act = get_resid_at_last_token(model, obj, layer_id).cpu().numpy()
        X.append(x_act)
        Y.append(y_act)
        Labels.append(sub)
        
    X = np.array(X)
    Y = np.array(Y)
    
    # Vectors v = Y - X
    V = Y - X
    
    # 2. PCA Projection
    # We want to visualize the relationship vectors V.
    # Approach A: PCA on V directly to see if they point in same direction.
    # Approach B: PCA on [X, Y] to see the space layout.
    
    # Let's do PCA on V to show the variance in directions.
    pca_v = PCA(n_components=2)
    V_2d = pca_v.fit_transform(V)
    
    # Also project X to 2D to map them
    pca_x = PCA(n_components=2)
    X_2d = pca_x.fit_transform(X)
    
    print(f"Explained Variance (V): {pca_v.explained_variance_ratio_}")
    
    # 3. Plotting
    plt.figure(figsize=(12, 10))
    
    # Plot origin points (Subjects)
    # Note: X_2d space and V_2d space are different PCA spaces.
    # To visualize "Arrow from X", we should ideally project [X, Y] into a common space.
    
    # Better Approach: PCA on ALL data [X, Y]
    all_data = np.concatenate([X, Y], axis=0)
    pca_all = PCA(n_components=2)
    all_2d = pca_all.fit_transform(all_data)
    
    X_2d = all_2d[:len(X)]
    Y_2d = all_2d[len(X):]
    
    # Draw arrows from X to Y
    plt.quiver(
        X_2d[:, 0], X_2d[:, 1], 
        Y_2d[:, 0] - X_2d[:, 0], Y_2d[:, 1] - X_2d[:, 1],
        angles='xy', scale_units='xy', scale=1, alpha=0.5, color='b',
        width=0.003
    )
    
    # Plot points
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c='r', label='Subject (Country)')
    plt.scatter(Y_2d[:, 0], Y_2d[:, 1], c='g', label='Object (Capital)')
    
    # Add labels
    for i, label in enumerate(Labels):
        plt.annotate(label, (X_2d[i, 0], X_2d[i, 1]), fontsize=9, alpha=0.8)
        # plt.annotate(pairs[i][1], (Y_2d[i, 0], Y_2d[i, 1]), fontsize=8, alpha=0.5)

    plt.title(f"Manifold Curvature: Country-Capital Relations (GPT-2 Layer {layer_id})\nPCA Projection of Activations", fontsize=16)
    plt.xlabel(f"PC1 ({pca_all.explained_variance_ratio_[0]:.2%} Var)", fontsize=12)
    plt.ylabel(f"PC2 ({pca_all.explained_variance_ratio_[1]:.2%} Var)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = "manifold_curvature_plot.png"
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")
    
    # 4. Quantitative check for rotation in 2D
    # Calculate angles of 2D arrows
    v_2d = Y_2d - X_2d
    angles = np.arctan2(v_2d[:, 1], v_2d[:, 0])
    
    # Convert to degrees
    angles_deg = np.degrees(angles)
    print(f"Angles of vectors in 2D PCA space: {angles_deg}")
    print(f"Std Dev of Angles: {np.std(angles_deg):.2f} degrees")
    
    if np.std(angles_deg) > 20:
        print("Conclusion: High angular variance in 2D (Visual Curvature Verified).")
    else:
        print("Conclusion: Vectors look parallel in 2D.")

def main():
    model = load_model()
    if model:
        visualize_curvature(model, LAYER_ID)

if __name__ == "__main__":
    main()
