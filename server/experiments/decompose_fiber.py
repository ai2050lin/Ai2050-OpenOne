
import os
import pickle

import numpy as np
from sklearn.decomposition import PCA

CACHE_DIR = "nfb_data"
FIBER_DIM = 5  # Number of dimensions to retain per fiber (local PCA components)
# In theory, this could be d_model, but we want the "principal" directions of variation at each point.

def decompose_fiber():
    print("Loading data...")
    try:
        data = np.load(os.path.join(CACHE_DIR, "hidden_states.npy")) # [N, D]
        labels = np.load(os.path.join(CACHE_DIR, "cluster_labels.npy")) # [N]
        centers = np.load(os.path.join(CACHE_DIR, "cluster_centers.npy")) # [K, M_dim]
        # We also need the manifold coords to map back if needed, but labels suffice for grouping.
    except FileNotFoundError:
        print("Data/Manifold files missing. Run extract_manifold.py first.")
        return

    n_clusters = len(np.unique(labels))
    d_model = data.shape[1]
    
    print(f"Decomposing fibers for {n_clusters} clusters...")
    
    fiber_bases = np.zeros((n_clusters, FIBER_DIM, d_model))
    fiber_variances = np.zeros((n_clusters, FIBER_DIM))
    
    for i in range(n_clusters):
        # 1. Gather points in this chart
        indices = np.where(labels == i)[0]
        if len(indices) < FIBER_DIM:
            print(f"Cluster {i} has too few points ({len(indices)}), skipping PCA...")
            continue
            
        local_points = data[indices]
        
        # 2. Centering (Relative to cluster mean in high-dim space, effectively "lifting" clean fibers)
        # Note: In fiber bundle theory, the fiber F_x is the space attached to x.
        # Here we approximate F_x by the local variation around the cluster centroid.
        pca = PCA(n_components=FIBER_DIM)
        pca.fit(local_points)
        
        fiber_bases[i] = pca.components_
        fiber_variances[i] = pca.explained_variance_ratio_
        
    print("Saving fiber bases...")
    np.save(os.path.join(CACHE_DIR, "fiber_bases.npy"), fiber_bases)
    np.save(os.path.join(CACHE_DIR, "fiber_variances.npy"), fiber_variances)
    print("Done.")

if __name__ == "__main__":
    decompose_fiber()
