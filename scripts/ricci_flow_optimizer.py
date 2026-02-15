
import os
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fibernet_logic import create_logic_model
from scripts.train_logic_core import D_MODEL, DATA_PATH, N_HEADS, N_LAYERS, LogicCorpus

# --- Configuration ---
MODEL_PATH = "tempdata/StructInit.pth"
K_NEIGHBORS = 5
ITERATIONS = 50
ALPHA = 0.1 # Learning rate for edge weights

def form_ricci_curvature(G, edge_weight_key='weight'):
    """
    Computes Forman-Ricci Curvature for all edges.
    Simple version: F(e) = 4 - d(u) - d(v)
    This is a topological proxy. For geometric Ricci flow, usually Ollivier-Ricci is used,
    but it's computationally expensive.
    
    Here we implement a metric-aware version if possible, or use the topological one 
    to drive the edge lengths.
    
    Actually, we want to evolve the metric (distance).
    Ricci Flow: d_new(u,v) = d_old(u,v) - alpha * Ric(u,v) * d_old(u,v)
    
    If Ric > 0 (spherical), distance decreases (shrink).
    If Ric < 0 (hyperbolic), distance increases (expand).
    
    Approximation for Ric(e):
    For unweighted graphs, cliques have high curvature, trees have low.
    
    Let's use a geometric proxy:
    Clusters should become tighter. Bridges should become longer.
    """
    
    # Precompute degrees
    degrees = dict(G.degree(weight=edge_weight_key))
    
    curvature = {}
    for u, v in G.edges():
        # Forman formula for weighted graphs is complex.
        # Let's use simplified combinatorial Ricci:
        # F(e) = 4 - deg(u) - deg(v) (for unweighted)
        # For our case, let's just use degree as a proxy for "crowdedness"
        # High degree nodes -> High Curvature region -> Shrink edges?
        pass
        
    return curvature

def compute_ollivier_ricci_proxy(G, embeddings):
    """
    Simulates Ricci curvature based on local structure.
    If neighbors of u and v are connected, curvature is positive.
    If they are disjoint, curvature is negative.
    
    Jaccard index of neighborhoods?
    """
    ricci = {}
    adj = {n: set(G.neighbors(n)) for n in G.nodes()}
    
    for u, v in G.edges():
        nu = adj[u]
        nv = adj[v]
        
        # Triangles (u, v, w) contribute to positive curvature
        triangles = len(nu.intersection(nv))
        
        # Jaccard like measure
        union = len(nu.union(nv)) - 2 # Exclude u and v themselves
        if union == 0:
             k = 0
        else:
            k = triangles / union # 0 to 1
            
        # Transform to [-1, 1] range loosely
        # k=1 (clique) -> Ric > 0
        # k=0 (tree) -> Ric < 0
        
        # Let's shift: 0.5 is flat?
        ricci[(u, v)] = (k - 0.2) # Bias towards negative curvature for trees
        
    return ricci

def optimize_manifold():
    # 1. Load Model
    print("Loading Model...")
    corpus = LogicCorpus(DATA_PATH)
    vocab_size = corpus.vocab_size
    
    model = create_logic_model(vocab_size=vocab_size, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS).model
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Model not found at {MODEL_PATH}. Run training first.")
        return

    # Extract Embeddings
    embeddings = model.embed.W_E.detach().cpu().numpy() # [Vocab, D]
    
    # 2. Build k-NN Graph
    print(f"Building k-NN Graph (k={K_NEIGHBORS})...")
    dist_matrix = squareform(pdist(embeddings, metric='cosine'))
    
    G = nx.Graph()
    G.add_nodes_from(range(vocab_size))
    
    for i in range(vocab_size):
        # Find k nearest neighbors
        # argsort gives small to large distance
        # [1:k+1] to skip self
        neighbors = np.argsort(dist_matrix[i])[1:K_NEIGHBORS+1]
        for j in neighbors:
            w = dist_matrix[i, j]
            w = max(1e-6, w) # Clamp to avoid negative weights from float error
            G.add_edge(i, j, weight=w, current_dist=w)
            
    print(f"Graph stats: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # 3. Ricci Flow Loop
    print("Starting Ricci Flow Evolution...")
    
    history_mds = []
    
    # Initial MDS for visualization
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, n_jobs=1)
    # We need a full distance matrix for MDS.
    # We can approximate geodesic distance using shortest path on the weighted graph
    
    def get_geo_dist(Graph):
        # Calculate all pairs shortest path
        # return matrix
        lengths = dict(nx.all_pairs_dijkstra_path_length(Graph, weight='current_dist'))
        mat = np.zeros((vocab_size, vocab_size))
        for i in range(vocab_size):
            for j in range(vocab_size):
                mat[i, j] = lengths[i].get(j, 10.0) # Cap at 10
        return mat
    
    initial_geo = get_geo_dist(G)
    coords_init = mds.fit_transform(initial_geo)
    history_mds.append(coords_init)
    
    for it in tqdm(range(ITERATIONS), desc="Evolving"):
        # Calculate Curvature
        ricci = compute_ollivier_ricci_proxy(G, None)
        
        # Update edge weights
        # d_new = d_old - alpha * R * d_old
        max_change = 0
        new_weights = {}
        
        for u, v in G.edges():
            old_w = G[u][v]['current_dist']
            k = ricci.get((u, v), 0)
            
            # Update step
            # Note: In Ricci flow, positive curvature (manifold shrinks) means distance decreases.
            # R > 0 => weight decreases.
            # R < 0 => weight increases.
            
            change = ALPHA * k * old_w
            new_w = max(0.01, old_w - change) # Prevent negative or zero dist
            
            new_weights[(u, v)] = new_w
            max_change = max(max_change, abs(change))
            
        # Apply updates
        for (u, v), w in new_weights.items():
            G[u][v]['current_dist'] = w
            
        # Check convergence
        if max_change < 1e-4:
            print(f"Converged at iteration {it}")
            break
            
    # Final Visualization
    print("Computing final MDS...")
    final_geo = get_geo_dist(G)
    coords_final = mds.fit_transform(final_geo)
    history_mds.append(coords_final)
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    titles = ["Initial Manifold (Before Flow)", "Optimized Manifold (After Ricci Flow)"]
    for i, coords in enumerate(history_mds):
        ax = axes[i]
        ax.scatter(coords[:, 0], coords[:, 1], alpha=0.6, c='blue', s=20)
        ax.set_title(titles[i])
        ax.grid(True, alpha=0.3)
        
        # Draw some edges
        # Just random subset or MST to keep clean
        # Let's draw k-NN edges
        for u, v in list(G.edges())[:500]: # Limit for clutter
             p1 = coords[u]
             p2 = coords[v]
             ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', alpha=0.05)

    plt.suptitle("Ricci Flow Evolution: Smoothing the Logic Manifold")
    out_path = "tempdata/ricci_flow_evolution.png"
    plt.savefig(out_path)
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    optimize_manifold()
