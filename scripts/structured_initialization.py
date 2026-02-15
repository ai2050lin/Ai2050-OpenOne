
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh


def generate_cycle_graph(n):
    """Generates an adjacency matrix for a cycle graph of size n."""
    G = nx.cycle_graph(n)
    adj = nx.adjacency_matrix(G).toarray().astype(float)
    return adj, G

def compute_laplacian_embeddings(adj_matrix, embedding_dim):
    r"""
    Computes Laplacian Eigenmaps embeddings.
    Equation: L v = \lambda v
    Uses the k smallest non-zero eigenvectors.
    """
    # 1. Compute Graph Laplacian (L = D - A)
    # normalized=True gives L = I - D^{-1/2} A D^{-1/2}, often better for spectral clustering
    laplacian = csgraph.laplacian(adj_matrix, normed=True)
    
    # 2. Eigen Decomposition
    # We want the *smallest* eigenvalues (frequency 0 is the constant vector)
    # k+1 because the first eigenvector is usually trivial (constant) for connected graphs
    eigenvalues, eigenvectors = eigsh(laplacian, k=embedding_dim + 1, which='SM')
    
    # 3. Sort and select
    # indices = np.argsort(eigenvalues)
    # The first one is close to 0, skip it.
    embedding = eigenvectors[:, 1:] 
    
    return embedding

def visualize_embeddings(embeddings, title, filename):
    """Plots the 2D embeddings."""
    plt.figure(figsize=(8, 8))
    
    # Color by index to show the cycle order
    n = embeddings.shape[0]
    colors = np.arange(n)
    
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=colors, cmap='hsv', s=100, alpha=0.8)
    
    # Draw edges to show topology
    # Assuming it's a cycle 0->1->...->n-1->0
    for i in range(n):
        j = (i + 1) % n
        p1 = embeddings[i]
        p2 = embeddings[j]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', alpha=0.2)
        
    plt.title(title)
    plt.colorbar(label='Node Index (0-99)')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    output_path = os.path.join('tempdata', filename)
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")
    plt.close()

def main():
    print("Step 3.5: Structured Initialization (The Skeleton)")
    print("------------------------------------------------")
    
    N = 100
    DIM = 2
    
    # 1. Random Initialization (Control Group)
    print(f"1. Generating Random Initialization for Z_{N}...")
    random_emb = np.random.randn(N, DIM)
    visualize_embeddings(random_emb, f"Random Initialization (Z_{N})", "init_random.png")
    
    # 2. Structured Initialization (Experimental Group)
    print(f"2. Generating Laplacian Initialization for Z_{N}...")
    adj, G = generate_cycle_graph(N)
    structured_emb = compute_laplacian_embeddings(adj, DIM)
    visualize_embeddings(structured_emb, f"Laplacian Initialization (Z_{N})", "init_structured.png")
    
    print("\ncomparison completed.")
    print("Check tempdata/init_random.png vs tempdata/init_structured.png")
    print("Random init should look like chaos.")
    print("Structured init should look like a perfect circle.")

if __name__ == "__main__":
    # Ensure tempdata exists
    os.makedirs("tempdata", exist_ok=True)
    main()
