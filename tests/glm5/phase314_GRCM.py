"""
Phase 314: GRCM - Global Relative Coding Map (全局相对编码图谱)
================================================================
Core hypothesis: Language encoding is NOT point-encoding (one concept = one vector),
but relational-network encoding (meaning = position in network + relative paths + contextual delta).

This test:
1. Build external semantic relation network G_external with 8 relation types
2. Extract internal representation distances for each concept pair at each layer
3. Compare G_external and G_internal using Mantel correlation and neighborhood overlap
4. Decompose each concept cluster into shared_path + delta_path
5. Build three maps: Reuse Map, Difference Map, Conflict Map

Key question: Does the model preserve human semantic relation structure internally?

Usage:
  python tests/glm5/phase314_GRCM.py qwen3
  python tests/glm5/phase314_GRCM.py glm4
  python tests/glm5/phase314_GRCM.py deepseek7b
"""
import sys, os, gc, time, json, math
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from pathlib import Path
from datetime import datetime
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from model_utils import MODEL_CONFIGS, get_model_info, get_layers, release_model

RESULT_DIR = Path("results/phase314_GRCM")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = Path("tmp"); TMP_DIR.mkdir(parents=True, exist_ok=True)
_log_file = None

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_file:
        try:
            with open(_log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except:
            pass


# =====================================================================
# EXTERNAL SEMANTIC RELATION NETWORK
# =====================================================================
# 8 relation types, each with (node1, node2, relation_type, strength)

RELATIONS = [
    # 1. Same class (同类)
    ("apple", "banana", "same_class", 1.0),
    ("apple", "pear", "same_class", 1.0),
    ("apple", "orange", "same_class", 1.0),
    ("banana", "pear", "same_class", 1.0),
    ("banana", "orange", "same_class", 1.0),
    ("dog", "cat", "same_class", 1.0),
    ("dog", "horse", "same_class", 1.0),
    ("cat", "horse", "same_class", 1.0),
    ("knife", "hammer", "same_class", 1.0),
    ("knife", "key", "same_class", 1.0),
    ("hammer", "key", "same_class", 1.0),
    ("table", "chair", "same_class", 1.0),
    ("car", "bus", "same_class", 1.0),
    ("river", "lake", "same_class", 1.0),
    
    # 2. Hypernym (上下位)
    ("apple", "fruit", "hypernym", 0.8),
    ("banana", "fruit", "hypernym", 0.8),
    ("pear", "fruit", "hypernym", 0.8),
    ("orange", "fruit", "hypernym", 0.8),
    ("dog", "animal", "hypernym", 0.8),
    ("cat", "animal", "hypernym", 0.8),
    ("horse", "animal", "hypernym", 0.8),
    ("knife", "tool", "hypernym", 0.8),
    ("hammer", "tool", "hypernym", 0.8),
    ("key", "tool", "hypernym", 0.8),
    ("car", "vehicle", "hypernym", 0.8),
    ("bus", "vehicle", "hypernym", 0.8),
    ("table", "furniture", "hypernym", 0.8),
    ("chair", "furniture", "hypernym", 0.8),
    ("river", "water", "hypernym", 0.6),
    ("lake", "water", "hypernym", 0.6),
    
    # 3. Attribute (属性)
    ("apple", "red", "attribute", 0.6),
    ("apple", "sweet", "attribute", 0.6),
    ("banana", "yellow", "attribute", 0.6),
    ("banana", "sweet", "attribute", 0.6),
    ("knife", "sharp", "attribute", 0.7),
    ("ice", "cold", "attribute", 0.7),
    ("fire", "hot", "attribute", 0.7),
    ("dog", "loyal", "attribute", 0.5),
    ("cat", "independent", "attribute", 0.5),
    
    # 4. Function (功能)
    ("knife", "cut", "function", 0.7),
    ("key", "open", "function", 0.7),
    ("car", "drive", "function", 0.7),
    ("hammer", "hit", "function", 0.7),
    ("apple", "eat", "function", 0.6),
    
    # 5. Antonym (反义)
    ("happy", "sad", "antonym", 0.9),
    ("bright", "dark", "antonym", 0.9),
    ("warm", "cold", "antonym", 0.9),
    ("strong", "weak", "antonym", 0.9),
    ("fast", "slow", "antonym", 0.9),
    ("big", "small", "antonym", 0.9),
    ("open", "closed", "antonym", 0.8),
    ("good", "bad", "antonym", 0.9),
    ("clean", "dirty", "antonym", 0.9),
    ("rich", "poor", "antonym", 0.9),
    ("light", "heavy", "antonym", 0.9),
    
    # 6. Negation (否定)
    ("happy", "not_happy", "negation", 0.85),
    ("possible", "not_possible", "negation", 0.85),
    ("open", "not_open", "negation", 0.85),
    ("clean", "not_clean", "negation", 0.85),
    ("good", "not_good", "negation", 0.85),
    ("safe", "not_safe", "negation", 0.85),
    
    # 7. Operator (操作)
    ("not_happy", "sad", "operator_similar", 0.6),
    ("not_good", "bad", "operator_similar", 0.6),
    ("not_clean", "dirty", "operator_similar", 0.6),
    ("not_open", "closed", "operator_similar", 0.6),
    ("not_safe", "dangerous", "operator_similar", 0.5),
    
    # 8. Cross-category (跨类别关联)
    ("fruit", "sweet", "cross_category", 0.4),
    ("animal", "alive", "cross_category", 0.4),
    ("tool", "useful", "cross_category", 0.4),
    ("fire", "dangerous", "cross_category", 0.5),
    ("ice", "water", "cross_category", 0.5),
]

# Concept clusters for shared+delta decomposition
CONCEPT_CLUSTERS = {
    "fruit": ["apple", "banana", "pear", "orange"],
    "animal": ["dog", "cat", "horse"],
    "tool": ["knife", "hammer", "key"],
    "emotion_pos": ["happy", "bright", "warm", "strong", "good"],
    "emotion_neg": ["sad", "dark", "cold", "weak", "bad"],
    "antonym_pairs": [("happy", "sad"), ("bright", "dark"), ("warm", "cold"), ("strong", "weak"), ("good", "bad")],
}

# Sentences for each concept (minimal context)
CONCEPT_SENTENCES = {
    # Fruits
    "apple": "the apple was fresh",
    "banana": "the banana was fresh",
    "pear": "the pear was fresh",
    "orange": "the orange was fresh",
    "fruit": "the fruit was fresh",
    
    # Animals
    "dog": "the dog was active",
    "cat": "the cat was active",
    "horse": "the horse was active",
    "animal": "the animal was active",
    
    # Tools
    "knife": "the knife was useful",
    "hammer": "the hammer was useful",
    "key": "the key was useful",
    "tool": "the tool was useful",
    
    # Vehicles
    "car": "the car was fast",
    "bus": "the bus was fast",
    "vehicle": "the vehicle was fast",
    
    # Furniture
    "table": "the table was large",
    "chair": "the chair was large",
    "furniture": "the furniture was large",
    
    # Water features
    "river": "the river was wide",
    "lake": "the lake was wide",
    "water": "the water was clear",
    
    # Attributes
    "red": "the color was red",
    "yellow": "the color was yellow",
    "sweet": "the taste was sweet",
    "sharp": "the edge was sharp",
    "cold": "the temperature was cold",
    "hot": "the temperature was hot",
    "loyal": "the trait was loyal",
    "independent": "the trait was independent",
    
    # Functions
    "cut": "they would cut it",
    "open": "they would open it",
    "drive": "they would drive it",
    "hit": "they would hit it",
    "eat": "they would eat it",
    
    # Positive emotions/attributes
    "happy": "they felt happy",
    "bright": "the light was bright",
    "warm": "the room was warm",
    "strong": "the person was strong",
    "good": "the result was good",
    "fast": "the speed was fast",
    "big": "the size was big",
    "clean": "the room was clean",
    "rich": "the person was rich",
    "light": "the bag was light",
    "safe": "the place was safe",
    "possible": "the task was possible",
    
    # Negative emotions/attributes
    "sad": "they felt sad",
    "dark": "the room was dark",
    "cold_attr": "the wind was cold",
    "weak": "the person was weak",
    "bad": "the result was bad",
    "slow": "the speed was slow",
    "small": "the size was small",
    "dirty": "the room was dirty",
    "poor": "the person was poor",
    "heavy": "the bag was heavy",
    "dangerous": "the place was dangerous",
    "closed": "the door was closed",
    
    # Negation
    "not_happy": "they were not happy",
    "not_good": "the result was not good",
    "not_clean": "the room was not clean",
    "not_open": "the door was not open",
    "not_safe": "the place was not safe",
    "not_possible": "the task was not possible",
    
    # Abstract
    "alive": "the creature was alive",
    "useful": "the item was useful",
}

# Get all unique nodes
ALL_NODES = sorted(set([r[0] for r in RELATIONS] + [r[1] for r in RELATIONS]))


def build_external_distance_matrix():
    """Build external semantic distance matrix from relations."""
    n = len(ALL_NODES)
    node_idx = {node: i for i, node in enumerate(ALL_NODES)}
    
    # Similarity matrix: initialize with 0
    sim_matrix = np.zeros((n, n))
    
    for n1, n2, rel_type, strength in RELATIONS:
        i, j = node_idx[n1], node_idx[n2]
        sim_matrix[i, j] = strength
        sim_matrix[j, i] = strength
    
    # Self-similarity
    np.fill_diagonal(sim_matrix, 1.0)
    
    # Convert to distance: distance = 1 - similarity
    dist_matrix = 1.0 - sim_matrix
    np.fill_diagonal(dist_matrix, 0.0)
    
    return dist_matrix, node_idx


def extract_concept_representations(model, tokenizer, device, layers_to_test, model_info):
    """Extract hidden state representations for each concept at each layer."""
    log("Extracting concept representations...")
    
    representations = {}  # {layer_idx: {concept_name: numpy_vector}}
    
    for li in layers_to_test:
        representations[li] = {}
        layer = get_layers(model)[li]
        
        captures = {}
        def make_hook(name):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    captures[name] = output[0].detach()
                else:
                    captures[name] = output.detach()
            return hook_fn
        
        handle = layer.register_forward_hook(make_hook(f"layer_{li}"))
        
        for concept, sentence in CONCEPT_SENTENCES.items():
            inp = tokenizer(sentence, return_tensors="pt").to(device)
            with torch.no_grad():
                captures.clear()
                model(**inp)
                h = captures.get(f"layer_{li}")
                if h is not None:
                    # Use the concept word position or last token
                    # For simplicity, use last token
                    representations[li][concept] = h[0, -1].cpu().float().numpy()
        
        handle.remove()
        log(f"  Layer {li}: {len(representations[li])} concepts extracted")
    
    return representations


def compute_internal_distance_matrix(representations, layer_idx):
    """Compute cosine distance matrix between all concepts at a layer."""
    concepts = sorted(representations.keys())
    n = len(concepts)
    
    # Build representation matrix
    vecs = np.array([representations[c] for c in concepts])
    
    # Cosine similarity
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    vecs_normed = vecs / norms
    
    sim_matrix = vecs_normed @ vecs_normed.T
    sim_matrix = np.clip(sim_matrix, -1, 1)
    
    # Distance = 1 - similarity
    dist_matrix = 1.0 - sim_matrix
    np.fill_diagonal(dist_matrix, 0.0)
    
    return dist_matrix, concepts


def mantel_test(dist1, dist2, n_permutations=1000):
    """Mantel test: correlation between two distance matrices."""
    # Get upper triangle values
    n = dist1.shape[0]
    idx = np.triu_indices(n, k=1)
    v1 = dist1[idx]
    v2 = dist2[idx]
    
    # Pearson correlation
    if len(v1) < 2 or np.std(v1) < 1e-10 or np.std(v2) < 1e-10:
        return 0.0, 1.0
    
    corr = np.corrcoef(v1, v2)[0, 1]
    
    # Permutation test
    count = 0
    for _ in range(n_permutations):
        perm = np.random.permutation(n)
        v1_perm = dist1[np.ix_(perm, perm)][idx]
        corr_perm = np.corrcoef(v1_perm, v2)[0, 1] if np.std(v1_perm) > 1e-10 else 0
        if abs(corr_perm) >= abs(corr):
            count += 1
    
    p_value = (count + 1) / (n_permutations + 1)
    return float(corr), float(p_value)


def neighborhood_overlap(dist1, dist2, k=5):
    """Compute average neighborhood overlap between two distance matrices."""
    n = dist1.shape[0]
    overlaps = []
    
    for i in range(n):
        # k nearest neighbors in each matrix
        nn1 = set(np.argsort(dist1[i])[1:k+1])
        nn2 = set(np.argsort(dist2[i])[1:k+1])
        
        if nn1 and nn2:
            overlap = len(nn1 & nn2) / len(nn1 | nn2)
            overlaps.append(overlap)
    
    return float(np.mean(overlaps)) if overlaps else 0.0


def relation_type_preservation(ext_dist, int_dist, node_idx, concepts):
    """For each relation type, test if internal distances preserve external similarity."""
    concept_set_idx = {c: i for i, c in enumerate(concepts)}
    
    results = {}
    for rel_type in ["same_class", "hypernym", "attribute", "function", "antonym", "negation", "operator_similar", "cross_category"]:
        # Get pairs with this relation type
        related_pairs = [(n1, n2) for n1, n2, rt, _ in RELATIONS if rt == rel_type]
        
        # Also get random pairs (not in any relation)
        all_related = set((r[0], r[1]) for r in RELATIONS) | set((r[1], r[0]) for r in RELATIONS)
        random_pairs = []
        for i, c1 in enumerate(concepts):
            for j, c2 in enumerate(concepts):
                if i < j and (c1, c2) not in all_related and (c2, c1) not in all_related:
                    random_pairs.append((c1, c2))
        
        # Compute average internal distance for related vs random
        related_dists = []
        for n1, n2 in related_pairs:
            if n1 in concept_set_idx and n2 in concept_set_idx:
                i, j = concept_set_idx[n1], concept_set_idx[n2]
                related_dists.append(int_dist[i, j])
        
        random_dists = []
        import random as rng
        rng.seed(42)
        sample_random = rng.sample(random_pairs, min(len(random_pairs), 100))
        for n1, n2 in sample_random:
            if n1 in concept_set_idx and n2 in concept_set_idx:
                i, j = concept_set_idx[n1], concept_set_idx[n2]
                random_dists.append(int_dist[i, j])
        
        if related_dists and random_dists:
            results[rel_type] = {
                "mean_related_dist": float(np.mean(related_dists)),
                "mean_random_dist": float(np.mean(random_dists)),
                "ratio": float(np.mean(random_dists) / np.mean(related_dists)) if np.mean(related_dists) > 0 else 0,
                "n_pairs": len(related_dists),
            }
    
    return results


def cluster_shared_delta_decomposition(representations, layer_idx, cluster_name, cluster_members):
    """Decompose concept cluster into shared component + individual deltas."""
    member_vecs = {}
    for m in cluster_members:
        if m in representations:
            member_vecs[m] = representations[m]
    
    if len(member_vecs) < 2:
        return None
    
    # Stack vectors
    names = sorted(member_vecs.keys())
    vecs = np.array([member_vecs[n] for n in names])
    
    # PCA to find shared component (first PC)
    mean_vec = vecs.mean(axis=0)
    centered = vecs - mean_vec
    
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    shared_pc1 = Vt[0]
    shared_var = (S[0]**2) / (S**2).sum() if (S**2).sum() > 0 else 0
    
    # Delta for each member: project out shared
    deltas = {}
    for i, name in enumerate(names):
        proj_on_shared = np.dot(centered[i], shared_pc1) * shared_pc1
        delta = centered[i] - proj_on_shared
        deltas[name] = {
            "shared_projection": float(np.dot(centered[i], shared_pc1)),
            "delta_norm": float(np.linalg.norm(delta)),
            "delta_norm_ratio": float(np.linalg.norm(delta) / np.linalg.norm(centered[i])) if np.linalg.norm(centered[i]) > 1e-10 else 0,
        }
    
    return {
        "cluster": cluster_name,
        "n_members": len(names),
        "shared_pc1_var": float(shared_var),
        "members": deltas,
        "mean_norm": float(np.mean([np.linalg.norm(v) for v in vecs])),
    }


def build_three_maps(ext_dist, int_dist, node_idx, concepts):
    """Build Reuse Map, Difference Map, Conflict Map."""
    concept_set_idx = {c: i for i, c in enumerate(concepts)}
    
    reuse_pairs = []
    difference_pairs = []
    conflict_pairs = []
    
    for n1, n2, rel_type, ext_strength in RELATIONS:
        if n1 not in concept_set_idx or n2 not in concept_set_idx:
            continue
        i, j = concept_set_idx[n1], concept_set_idx[n2]
        int_sim = 1.0 - int_dist[i, j]  # internal cosine similarity
        
        # Reuse: high internal similarity means shared representation
        reuse_pairs.append((n1, n2, rel_type, float(int_sim), float(ext_strength)))
        
        # Difference: internal sim != external sim
        diff = abs(int_sim - ext_strength)
        difference_pairs.append((n1, n2, rel_type, float(diff), float(int_sim), float(ext_strength)))
        
        # Conflict: internal and external disagree on direction
        # (e.g., should be similar but are dissimilar, or vice versa)
        if ext_strength > 0.7 and int_sim < 0:
            conflict_pairs.append((n1, n2, rel_type, "should_similar_but_opposite", float(int_sim), float(ext_strength)))
        elif ext_strength < 0.2 and int_sim > 0.5:
            conflict_pairs.append((n1, n2, rel_type, "should_distant_but_similar", float(int_sim), float(ext_strength)))
    
    return {
        "reuse": reuse_pairs,
        "difference": difference_pairs,
        "conflict": conflict_pairs,
    }


def run_model(model_name):
    global _log_file
    _log_file = str(TMP_DIR / f"phase314_{model_name}.log")
    
    log(f"=== Phase 314: GRCM for {model_name} ===")
    
    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    
    log(f"Loading {model_name} (bf16 + device_map=auto)...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    model.eval()
    device = next(model.parameters()).device
    
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log(f"Model loaded: {type(model).__name__}, device={device}, GPU={gpu_mem:.2f}GB")
    
    info = get_model_info(model, model_name)
    log(f"Model info: n_layers={info.n_layers}, d_model={info.d_model}")
    
    # Select layers
    n_layers = info.n_layers
    if n_layers >= 36:
        layers_to_test = [2, 6, 12, 18, 24, 30, n_layers-2]
    elif n_layers >= 24:
        layers_to_test = [2, 4, 8, 12, 16, 20, n_layers-2]
    else:
        layers_to_test = [2, 4, 8, 12, 16, 20, n_layers-2]
    log(f"Test layers: {layers_to_test}")
    
    # Step 1: Build external distance matrix
    log("Building external semantic distance matrix...")
    ext_dist, ext_node_idx = build_external_distance_matrix()
    log(f"External matrix: {ext_dist.shape[0]} nodes, {len(RELATIONS)} relations")
    
    # Step 2: Extract concept representations
    representations = extract_concept_representations(model, tokenizer, device, layers_to_test, info)
    
    # Step 3: For each layer, compute internal distance matrix and compare
    results = {
        "model": model_name,
        "n_layers": info.n_layers,
        "d_model": info.d_model,
        "test_layers": layers_to_test,
        "n_concepts": len(CONCEPT_SENTENCES),
        "n_relations": len(RELATIONS),
        "layers": {},
    }
    
    for li in layers_to_test:
        log(f"Analyzing layer {li}...")
        rep = representations[li]
        
        if len(rep) < 5:
            log(f"  Layer {li}: too few representations ({len(rep)}), skipping")
            continue
        
        # Internal distance matrix
        int_dist, concepts = compute_internal_distance_matrix(rep, li)
        
        # Match with external matrix
        common_concepts = [c for c in concepts if c in ext_node_idx or c in ALL_NODES]
        common_idx_int = [concepts.index(c) for c in common_concepts]
        common_idx_ext = [ext_node_idx.get(c, ALL_NODES.index(c) if c in ALL_NODES else -1) for c in common_concepts]
        
        # Filter valid indices
        valid = [(i, j) for i, j in zip(common_idx_int, common_idx_ext) if j >= 0 and j < ext_dist.shape[0]]
        if len(valid) < 5:
            log(f"  Layer {li}: too few common concepts ({len(valid)}), skipping")
            continue
        
        valid_int_idx = [v[0] for v in valid]
        valid_ext_idx = [v[1] for v in valid]
        
        int_sub = int_dist[np.ix_(valid_int_idx, valid_int_idx)]
        ext_sub = ext_dist[np.ix_(valid_ext_idx, valid_ext_idx)]
        
        # Mantel test
        mantel_corr, mantel_p = mantel_test(ext_sub, int_sub, n_permutations=500)
        
        # Neighborhood overlap
        nn_overlap = neighborhood_overlap(ext_sub, int_sub, k=5)
        
        # Relation type preservation
        rel_pres = relation_type_preservation(ext_dist, int_dist, ext_node_idx, concepts)
        
        # Cluster decomposition
        cluster_results = {}
        for cname, cmembers in CONCEPT_CLUSTERS.items():
            if isinstance(cmembers[0], tuple):
                # Antonym pairs - skip for decomposition
                continue
            valid_members = [m for m in cmembers if m in rep]
            if len(valid_members) >= 2:
                decomp = cluster_shared_delta_decomposition(rep, li, cname, valid_members)
                if decomp:
                    cluster_results[cname] = decomp
        
        # Three maps
        three_maps = build_three_maps(ext_dist, int_dist, ext_node_idx, concepts)
        
        # Key metrics for this layer
        li_data = {
            "n_concepts": len(concepts),
            "mantel_correlation": mantel_corr,
            "mantel_p_value": mantel_p,
            "neighborhood_overlap_k5": nn_overlap,
            "relation_type_preservation": rel_pres,
            "cluster_decomposition": cluster_results,
            "conflict_count": len(three_maps["conflict"]),
            "top_reuse": sorted(three_maps["reuse"], key=lambda x: x[3], reverse=True)[:10],
            "top_conflict": three_maps["conflict"][:10],
        }
        
        results["layers"][str(li)] = li_data
        
        log(f"  Layer {li}: Mantel r={mantel_corr:.3f} (p={mantel_p:.3f}), "
            f"NN overlap={nn_overlap:.3f}, Conflicts={len(three_maps['conflict'])}")
        
        # Print relation type preservation
        for rt, data in sorted(rel_pres.items(), key=lambda x: x[1].get("ratio", 0), reverse=True):
            log(f"    {rt}: related_dist={data['mean_related_dist']:.3f}, random_dist={data['mean_random_dist']:.3f}, "
                f"ratio={data['ratio']:.2f}")
    
    # Save results
    out_path = RESULT_DIR / f"{model_name}_GRCM.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    log(f"Results saved to {out_path}")
    
    # Print summary
    log("\n" + "="*70)
    log(f"GRCM SUMMARY - {model_name}")
    log("="*70)
    
    for li in layers_to_test:
        if str(li) not in results["layers"]:
            continue
        lr = results["layers"][str(li)]
        log(f"\n  Layer {li}:")
        log(f"    Mantel r={lr['mantel_correlation']:.3f} (p={lr['mantel_p_value']:.3f})")
        log(f"    Neighborhood overlap={lr['neighborhood_overlap_k5']:.3f}")
        log(f"    Conflicts={lr['conflict_count']}")
        
        for cname, cdata in lr.get("cluster_decomposition", {}).items():
            log(f"    Cluster '{cname}': shared_pc1_var={cdata['shared_pc1_var']:.3f}, "
                f"members={cdata['n_members']}")
            for mname, mdata in cdata.get("members", {}).items():
                log(f"      {mname}: shared_proj={mdata['shared_projection']:.3f}, "
                    f"delta_norm_ratio={mdata['delta_norm_ratio']:.3f}")
    
    # Release model
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    log(f"Model {model_name} released.")


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    
    if model_name == "all":
        for mn in ["qwen3", "glm4", "deepseek7b"]:
            log(f"\n{'#'*70}")
            log(f"# Starting {mn}")
            log(f"{'#'*70}")
            try:
                run_model(mn)
            except Exception as e:
                log(f"ERROR running {mn}: {e}")
                import traceback
                traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(10)
    else:
        run_model(model_name)
