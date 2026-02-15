
import glob
import json
import os
import re

import matplotlib.pyplot as plt


def visualize_latest_embedding():
    # Find latest embedding file
    files = glob.glob("experiments/z113_visuals/embeddings_epoch_*.json")
    if not files:
        print("No embedding files found.")
        return

    # Sort by epoch
    latest_file = max(files, key=lambda f: int(re.search(r"epoch_(\d+)", f).group(1)))
    epoch = re.search(r"epoch_(\d+)", latest_file).group(1)
    
    print(f"Visualizing {latest_file}...")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
        
    embeddings = data["embeddings"] # List of [x, y, z]
    
    # Plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    xs = [e[0] for e in embeddings]
    ys = [e[1] for e in embeddings]
    zs = [e[2] for e in embeddings]
    colors = range(len(embeddings)) # Color by index to see the cycle order
    
    scatter = ax.scatter(xs, ys, zs, c=colors, cmap='hsv', s=100)
    
    # Connect the dots to see the ring
    for i in range(len(embeddings)):
        next_i = (i + 1) % len(embeddings)
        ax.plot([xs[i], xs[next_i]], [ys[i], ys[next_i]], [zs[i], zs[next_i]], 'k-', alpha=0.3)

    ax.set_title(f"Z113 FiberNet Embeddings at Epoch {epoch}\n(Colors represent integer values 0..112)")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_zlabel("PCA 3")
    
    plt.colorbar(scatter, label="Element Value (0-112)")
    
    output_path = f"experiments/z113_visuals/manifold_epoch_{epoch}.png"
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    visualize_latest_embedding()
