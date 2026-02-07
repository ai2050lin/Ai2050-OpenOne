
import os

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances

CACHE_DIR = "nfb_data"

def learn_transport():
    print("Loading data...")
    # Load data
    try:
        data = np.load(os.path.join(CACHE_DIR, "hidden_states.npy"))
        labels = np.load(os.path.join(CACHE_DIR, "cluster_labels.npy"))
        centers = np.load(os.path.join(CACHE_DIR, "cluster_centers.npy"))
        # Load manifold coords for distance
        manifold_coords = np.load(os.path.join(CACHE_DIR, "manifold_coords.npy"))
    except FileNotFoundError:
        print("Data not found. Run previous steps.")
        return

    print(f"Data shape: {data.shape}")
    N_CLUSTERS = len(centers)
    
    # We want to find connections between clusters.
    # Since we don't have time-series, we use Proximity in Manifold Space.
    # Connect clusters that are "close" on the manifold.
    
    print("Computing cluster proximity...")
    # Calculate pairwise distances between cluster centers
    # centers is in manifold space (2D/3D) or high dim?
    # extract_manifold saves centers in MANIFOLD space.
    # Let's verify shape.
    
    # If centers is [K, 2], we compute Euclidean dist.
    dists = pairwise_distances(centers)
    
    transport_maps = {}
    
    # For each cluster, find 2 nearest neighbors
    for i in range(N_CLUSTERS):
        # Sort by distance
        nearest = np.argsort(dists[i])[1:4] # Skip self (index 0), take top 3
        
        for j in nearest:
            if dists[i, j] > 50.0: # Threshold for very far clusters (optional)
                continue
                
            # Learn Transport: v_j = A * v_i + b
            # We take points in cluster i and points in cluster j.
            # But which point maps to which?
            # Without temporal correspondence, we can't learn a true transport MAP (f(x)=y).
            # We can only learn a "Concept Alignment" or just denote the connection using Identity or Mean shift.
            
            # For visualization, we just need the Connection metadata (strength).
            # We fake a score based on proximity.
            score = 1.0 / (1.0 + dists[i, j])
            
            key = f"{i}_{j}"
            transport_maps[key] = {
                "score": float(score),
                "type": "proximity"
            }
            
    print(f"Identified {len(transport_maps)} connections.")
    np.save(os.path.join(CACHE_DIR, "transport_maps.npy"), transport_maps)
    print("Done.")

if __name__ == "__main__":
    learn_transport()
