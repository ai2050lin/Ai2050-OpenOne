
import json
import os

import numpy as np
import scipy.spatial.distance as dist
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
DATA_DIR = "nfb_data"
TRAJECTORY_FILE = os.path.join(DATA_DIR, "trajectory_data.npy")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "flow_tubes.json")
ANALYSIS_REPORT = os.path.join(DATA_DIR, "dynamics_report.txt")

def analyze_dynamics():
    if not os.path.exists(TRAJECTORY_FILE) or not os.path.exists(METADATA_FILE):
        print("Data files not found.")
        return

    # 1. Load Data
    print("Loading data...")
    trajectories = np.load(TRAJECTORY_FILE) # (N, L, D)
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)
    
    N, L, D = trajectories.shape
    print(f"Loaded {N} trajectories, {L} layers, {D} dimensions.")

    # 2. Group by Category
    categories = {}
    for i, meta in enumerate(metadata):
        cat = meta.get("target_category", "unknown")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(trajectories[i])
    
    print(f"Categories found: {list(categories.keys())}")

    # 3. Calculate Centroids (The "Mean Flow")
    # Shape: (L, D) for each category
    centroids = {}
    for cat, trajs in categories.items():
        centroids[cat] = np.mean(np.array(trajs), axis=0)

    # 4. Dimensionality Reduction for Visualization (PCA to 3D)
    # Flatten all data to fit PCA: (N*L, D)
    all_points = trajectories.reshape(-1, D)
    pca = PCA(n_components=3)
    print("Fitting PCA...")
    pca.fit(all_points)
    
    # Transform Centroids
    centroids_3d = {}
    for cat, cent in centroids.items():
        centroids_3d[cat] = pca.transform(cent)

    # 5. Velocity Field Analysis (The "Dynamics")
    # v_l = c_{l+1} - c_l
    velocities = {}
    for cat, cent in centroids.items():
        velocities[cat] = np.diff(cent, axis=0) # (L-1, D)

    # 6. Covariant Transport Check (Parallel Transport)
    # Check if "King -> Queen" matches "Man -> Woman" dynamics
    # Ideally, V_female(l) = V_male(l) + Transport(l)
    # Let's check cosine similarity between V_male and V_female at each layer
    
    report_lines = []
    report_lines.append("=== Deep Dynamics Analysis Report ===")
    
    pairs = [("male", "female")] # Assuming these keys exist in categories
    
    for c1, c2 in pairs:
        if c1 in velocities and c2 in velocities:
            sims = []
            for l in range(L-1):
                v1 = velocities[c1][l].reshape(1, -1)
                v2 = velocities[c2][l].reshape(1, -1)
                sim = cosine_similarity(v1, v2)[0][0]
                sims.append(sim)
            
            avg_sim = np.mean(sims)
            report_lines.append(f"Velocity Alignment ({c1} <-> {c2}): {avg_sim:.4f}")
            report_lines.append(f"  Layer-wise: {[round(s, 2) for s in sims]}")
            
            if avg_sim > 0.5:
                report_lines.append("  [CONCLUSION] Strong Covariant Transport detected. Parallel evolution confirmed.")
            else:
                report_lines.append("  [CONCLUSION] Weak alignment. Dynamics might be distinct.")

    # 7. Universal Structure Verification (Kalman Filter Hypothesis)
    # ... (Keep existing Kalman logic/reporting if needed, but we focus on export) ...
    
    # --- GUT METRICS CALCULATION ---
    print("Calculating GUT Metrics (Surprise, Velocity, Curvature)...")
    
    # We need to perform these calculations for *every* trajectory in the clusters, 
    # but for visualization we primarily show the *centroids* or representative tubes.
    # The current code visualizes the CENTROIDS. So we calculate metrics for the centroids.
    
    # Centroid Metrics
    centroid_metrics = {}
    
    for cat, cent in centroids.items():
        # cent shape: (L, D)
        
        # 1. Surprise (Geometric FEP): Distance to global center or specific target?
        # Actually, for the centroid itself, 'surprise' regarding the category cluster is 0 by definition.
        # But maybe we want to visualize the "Variance" of the cluster at that layer as the tube radius.
        # Tube Radius = Cluster Variance at Layer L.
        # This represents "Uncertainty" or "Entropy" -> FEP.
        
        # Calculate cluster variance per layer
        trajs = categories[cat] # List of (L, D)
        trajs_arr = np.array(trajs) # (N_cat, L, D)
        
        # Variance = Mean squared distance from centroid at each layer
        # shape: (L,)
        variances = np.mean(np.linalg.norm(trajs_arr - cent, axis=2), axis=0)
        
        # 2. Velocity (Dynamics)
        # ||v_l||
        # shape: (L-1,) -> pad to L
        vel = np.linalg.norm(np.diff(cent, axis=0), axis=1)
        vel = np.pad(vel, (0, 1), 'edge')
        
        # 3. Curvature (Geometry)
        # 1 - CosSim(v_l, v_{l+1})
        # shape: (L-2,) -> pad to L
        curv = []
        vecs = np.diff(cent, axis=0) # (L-1, D)
        for i in range(len(vecs) - 1):
            v1 = vecs[i].reshape(1, -1)
            v2 = vecs[i+1].reshape(1, -1)
            if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
                curv.append(0)
            else:
                sim = cosine_similarity(v1, v2)[0][0]
                curv.append(1 - sim)
        
        curv = np.array(curv)
        curv = np.pad(curv, (1, 1), 'edge') # Pad start and end
        
        centroid_metrics[cat] = {
            "surprise": variances.tolist(), # Maps to Radius
            "velocity": vel.tolist(),       # Maps to Particle Speed
            "curvature": curv.tolist()      # Maps to Color (Red = High Curvature)
        }

    # 8. Export for Visualization
    # Format for frontend 3D tubes
    tubes_data = {
        "tubes": [],
        "layers": L
    }
    
    colors = {
        "male": "#3498db",   # Blue
        "female": "#e74c3c", # Red
        "royalty": "#f1c40f", # Gold
        "human": "#2ecc71",   # Green
        "profession": "#9b59b6", # Purple
        "positive": "#2ecc71",
        "negative": "#e74c3c"
    }

    for cat, path_3d in centroids_3d.items():
        # path_3d is (L, 3)
        metrics = centroid_metrics.get(cat, {})
        
        tube = {
            "label": cat,
            "path": path_3d.tolist(),
            "color": colors.get(cat, "#bdc3c7"),
            "radius": 0.1, # Base radius
            "metrics": metrics # Inject metrics
        }
        tubes_data["tubes"].append(tube)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(tubes_data, f, indent=2)
    
    with open(ANALYSIS_REPORT, 'w') as f:
        f.write("\n".join(report_lines))

    print(f"Analysis complete. Visualization saved to {OUTPUT_FILE}")
    print("\n".join(report_lines))

if __name__ == "__main__":
    analyze_dynamics()
