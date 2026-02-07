
import json
import os

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

# from umap import UMAP # Optional, PCA is faster/robust for demo

CACHE_DIR = "nfb_data"
MANIFOLD_DIM = 2 # Intrinsic dimension of syntax
N_CLUSTERS = 20 # Number of charts/patches

def extract_manifold():
    print("Loading hidden states and metadata...")
    data_path = os.path.join(CACHE_DIR, "hidden_states.npy")
    meta_path = os.path.join(CACHE_DIR, "metadata.json")
    
    if not os.path.exists(data_path):
        print("No data found. Run nfb_ra_data.py first.")
        return

    data = np.load(data_path)
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = [{"type": "unknown", "text": ""} for _ in range(len(data))]
    
    print(f"Data shape: {data.shape}")
    
    # 1. Manifold Projection (PCA for stability)
    print("Fitting PCA for Global Manifold...")
    pca = PCA(n_components=MANIFOLD_DIM)
    manifold_2d = pca.fit_transform(data) # [N, 2]
    
    # Lift to 3D for visualization (Optional: adding a non-linear twist or just 0)
    # For "Swiss Roll" visualization, we might want 3 components if doing that.
    # But for "Base Manifold", 2D plane is fine.
    
    # 2. Clustering (Charts)
    print(f"Clustering into {N_CLUSTERS} charts...")
    kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=42)
    cluster_labels = kmeans.fit_predict(manifold_2d)
    cluster_centers = kmeans.cluster_centers_
    
    # 3. Save Structure
    print("Saving manifold structure...")
    np.save(os.path.join(CACHE_DIR, "manifold_coords.npy"), manifold_2d)
    np.save(os.path.join(CACHE_DIR, "cluster_labels.npy"), cluster_labels)
    np.save(os.path.join(CACHE_DIR, "cluster_centers.npy"), cluster_centers)
    
    # 4. Save Point Cloud Data for Visualization (Merged with Metadata)
    # We want to send this to the frontend to draw specific points.
    point_cloud = []
    for i in range(len(data)):
        point_cloud.append({
            "id": i,
            "pos": [float(manifold_2d[i, 0]), 0, float(manifold_2d[i, 1])], # y=0 for base manifold
            "type": metadata[i].get("type", "general"),
            "text": metadata[i].get("text", ""),
            "cluster": int(cluster_labels[i])
        })
        
    with open(os.path.join(CACHE_DIR, "manifold_points.json"), "w") as f:
        json.dump(point_cloud, f, indent=2)
        
    print("Done.")

if __name__ == "__main__":
    extract_manifold()
