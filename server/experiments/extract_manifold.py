
import json
import os

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

# Use Isomap for true manifold distance (geodesic)
from sklearn.manifold import Isomap
from sklearn.metrics import pairwise_distances

CACHE_DIR = "nfb_data"
MANIFOLD_DIM = 2 # Intrinsic dimension of syntax
N_CLUSTERS = 50 # Increased charts for better resolution

def extract_manifold():
    print("Loading hidden states and metadata...")
    # Note: trajectories has shape (N, Time, Dim). We flatten or pick a specific layer?
    # Usually "Manifold" refers to the activation space at a certain layer (or concatenated).
    # nfb_ra_data produces trajectory_data.npy [N, Layers, Dim]
    # Let's use the 'middle layer' (e.g. layer 6 in GPT2-small) or flatten.
    # The AGI Memo says "Manifold Topology Extraction". usually layer-wise or joint.
    # Let's pick Layer 6 (middle depth) as valid repr of "Concept Space".
    
    traj_path = os.path.join(CACHE_DIR, "trajectory_data.npy")
    meta_path = os.path.join(CACHE_DIR, "metadata.json")
    
    if not os.path.exists(traj_path):
        print("No trajectory data found. Run nfb_ra_data.py first.")
        return

    # Load Trajectory Data [N_samples, N_layers, D_model]
    # For GPT2-small: [N, 13, 768]
    full_data = np.load(traj_path)
    
    # Select Layer 6 (Middle Layer) for Manifold Analysis
    # Index 7 corresponds to Block 6 output (0=Embed, 1..12=Blocks)
    layer_idx = 7 
    data = full_data[:, layer_idx, :]
    print(f"Using Layer {layer_idx} activations. Shape: {data.shape}")

    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = [{"type": "unknown", "text": ""} for _ in range(len(data))]
    
    # 1. Manifold Clustering (to find "Charts" or "Patches")
    # Doing Isomap on 20k points is slow (O(N^2)). 
    # Strategy: Cluster first -> Isomap on Centers.
    print(f"Clustering {len(data)} points into {N_CLUSTERS} centers (landmarks)...")
    kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=42, batch_size=1024)
    cluster_labels = kmeans.fit_predict(data)
    cluster_centers = kmeans.cluster_centers_
    
    # 2. Extract Topology of Centers (Simulated TDA via Isomap/MST)
    print("Computing Manifold Topology (Isomap & MST on centers)...")
    
    # Isomap embedding of centers (Geodesic distance preserving)
    embedding = Isomap(n_components=MANIFOLD_DIM, n_neighbors=5)
    center_coords_2d = embedding.fit_transform(cluster_centers)
    
    # Minimum Spanning Tree (MST) on centers to capture "Skeleton"
    # This approximates the 1-skeleton of the Rips complex at connectivity scale
    dist_matrix = pairwise_distances(cluster_centers)
    mst = minimum_spanning_tree(dist_matrix) # Returns sparse matrix
    mst_array = mst.toarray()
    
    # 3. Project all points via PCA (Locally Linear approximation relative to centers?)
    # Or just project them using the same Isomap transform? 
    # Isomap transform is heavy. Let's use PCA for the dense cloud but align it?
    # Actually, simplistic approach: PCA on full data, align to Isomap?
    # Better: Just use PCA for the dense background visualization, but visualize the TOPOLOGY (skeleton) clearly.
    
    print("Projecting dense cloud (PCA)...")
    pca = PCA(n_components=MANIFOLD_DIM)
    cloud_coords_2d = pca.fit_transform(data)
    
    # Optimize visualization alignment? 
    # (Optional: Procrustes to align PCA cloud to Isomap skeleton if needed, but not strictly required)
    
    # 4. Save Structure
    print("Saving manifold structure...")
    np.save(os.path.join(CACHE_DIR, "manifold_coords.npy"), cloud_coords_2d)
    np.save(os.path.join(CACHE_DIR, "cluster_labels.npy"), cluster_labels)
    np.save(os.path.join(CACHE_DIR, "cluster_centers.npy"), cluster_centers)
    
    # Save Topology (MST Edges)
    # List of [start_idx, end_idx, weight]
    rows, cols = np.where(mst_array > 0)
    mst_edges = []
    for r, c in zip(rows, cols):
        mst_edges.append({
            "source": int(r),
            "target": int(c),
            "weight": float(mst_array[r, c])
        })
    
    with open(os.path.join(CACHE_DIR, "manifold_topology.json"), "w") as f:
        json.dump({
            "nodes": [
                {"id": i, "pos": center_coords_2d[i].tolist()} 
                for i in range(len(center_coords_2d))
            ],
            "edges": mst_edges,
            "method": "Isomap + MST"
        }, f, indent=2)

    # 5. Save Point Cloud (Subsampled for Frontend if needed)
    # We save all, frontend handles rendering limit? 
    # Or purely use the skeleton for "Structure" tab and point cloud for "Data".
    
    point_cloud = []
    # Subsample 2000 points for frontend JSON to keep it light
    indices = np.random.choice(len(data), size=min(2000, len(data)), replace=False)
    
    for i in indices:
        point_cloud.append({
            "id": int(i),
            "pos": [float(cloud_coords_2d[i, 0]), 0.0, float(cloud_coords_2d[i, 1])], 
            "type": metadata[i].get("target_category", "unknown"), # Use cleaned metadata key
            "text": metadata[i].get("text", "")[:50] + "...",
            "cluster": int(cluster_labels[i])
        })
        
    with open(os.path.join(CACHE_DIR, "manifold_points.json"), "w") as f:
        json.dump(point_cloud, f, indent=2)
        
    print("Topology extraction complete.")

if __name__ == "__main__":
    extract_manifold()
