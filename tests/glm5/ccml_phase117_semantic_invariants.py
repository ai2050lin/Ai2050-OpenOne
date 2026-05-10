"""
Phase 117: Semantic Invariants — What is preserved across layers?
跨层保留的究竟是什么？在spike方向跨层正交、系数不可预测的情况下，
什么几何量在跨层保持不变？

关键认知升级: spike可能是控制信号(control plane), 不是语义内容(data plane)。
本实验不预设spike是"语义"还是"控制", 让数据告诉我们——
移除spike后, 语义结构还在不在?

Four sub-experiments:
Exp 1: Semantic Distance Preservation — 语义距离保持测试
Exp 2: Manifold Local Linearity — 流形局部线性测试  
Exp 3: Control vs Data Separation — 控制vs数据分离测试
Exp 4: Cross-Task Spike Comparison — 跨任务spike对比
"""

import torch
import numpy as np
import json
import argparse
import os
from pathlib import Path
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr, kendalltau
from scipy.spatial.distance import pdist, squareform

# ============================================================
# Configuration
# ============================================================

MODEL_CONFIGS = {
    'qwen3': {
        'name': 'Qwen/Qwen3-4B',
        'n_layers': 36,
        'd_model': 2560,
        'dtype': torch.bfloat16,
    },
    'deepseek7b': {
        'name': 'D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
        'n_layers': 28,
        'd_model': 3584,
        'dtype': torch.float16,
    },
    'glm4': {
        'name': 'D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf',
        'n_layers': 40,
        'd_model': 4096,
        'dtype': torch.float16,
    }
}

OUTPUT_DIR = Path("d:/Ai2050/TransformerLens-Project/tests/glm5_temp")

# ============================================================
# Translation pairs and semantic groups for testing
# ============================================================

# 100 translation pairs (Chinese -> English)
TRANSLATION_PAIRS = [
    ("猫", "cat"), ("狗", "dog"), ("鸟", "bird"), ("鱼", "fish"), ("马", "horse"),
    ("牛", "cow"), ("羊", "sheep"), ("猪", "pig"), ("鸡", "chicken"), ("鸭", "duck"),
    ("苹果", "apple"), ("香蕉", "banana"), ("橙子", "orange"), ("葡萄", "grape"), ("西瓜", "watermelon"),
    ("桃子", "peach"), ("梨", "pear"), ("草莓", "strawberry"), ("柠檬", "lemon"), ("芒果", "mango"),
    ("红色", "red"), ("蓝色", "blue"), ("绿色", "green"), ("黄色", "yellow"), ("白色", "white"),
    ("黑色", "black"), ("紫色", "purple"), ("橙色", "orange"), ("粉色", "pink"), ("灰色", "gray"),
    ("桌子", "table"), ("椅子", "chair"), ("床", "bed"), ("门", "door"), ("窗户", "window"),
    ("书", "book"), ("笔", "pen"), ("纸", "paper"), ("电脑", "computer"), ("电话", "phone"),
    ("太阳", "sun"), ("月亮", "moon"), ("星星", "star"), ("天空", "sky"), ("云", "cloud"),
    ("雨", "rain"), ("雪", "snow"), ("风", "wind"), ("山", "mountain"), ("河", "river"),
    ("大海", "sea"), ("湖", "lake"), ("花", "flower"), ("树", "tree"), ("草", "grass"),
    ("父亲", "father"), ("母亲", "mother"), ("兄弟", "brother"), ("姐妹", "sister"), ("孩子", "child"),
    ("朋友", "friend"), ("老师", "teacher"), ("学生", "student"), ("医生", "doctor"), ("护士", "nurse"),
    ("快乐", "happy"), ("悲伤", "sad"), ("愤怒", "angry"), ("恐惧", "fear"), ("惊讶", "surprise"),
    ("爱", "love"), ("恨", "hate"), ("希望", "hope"), ("梦想", "dream"), ("自由", "freedom"),
    ("跑步", "run"), ("游泳", "swim"), ("飞翔", "fly"), ("跳舞", "dance"), ("唱歌", "sing"),
    ("吃", "eat"), ("喝", "drink"), ("睡", "sleep"), ("走", "walk"), ("看", "see"),
    ("大", "big"), ("小", "small"), ("高", "tall"), ("矮", "short"), ("长", "long"),
    ("快", "fast"), ("慢", "slow"), ("热", "hot"), ("冷", "cold"), ("新", "new"),
    ("旧", "old"), ("好", "good"), ("坏", "bad"), ("美", "beautiful"), ("丑", "ugly"),
    ("聪明", "smart"), ("勇敢", "brave"), ("善良", "kind"), ("诚实", "honest"), ("强大", "strong"),
]

# Semantic groups for manifold linearity test
SEMANTIC_GROUPS = {
    'animals': ["猫", "狗", "鸟", "鱼", "马", "牛", "羊", "猪", "鸡", "鸭"],
    'fruits': ["苹果", "香蕉", "橙子", "葡萄", "西瓜", "桃子", "梨", "草莓", "柠檬", "芒果"],
    'colors': ["红色", "蓝色", "绿色", "黄色", "白色", "黑色", "紫色", "橙色", "粉色", "灰色"],
    'furniture': ["桌子", "椅子", "床", "门", "窗户"],
    'weather': ["太阳", "月亮", "星星", "天空", "云", "雨", "雪", "风"],
    'emotions': ["快乐", "悲伤", "愤怒", "恐惧", "惊讶", "爱", "恨"],
    'actions': ["跑步", "游泳", "飞翔", "跳舞", "唱歌", "吃", "喝", "睡", "走", "看"],
    'size_adjectives': ["大", "小", "高", "矮", "长", "快", "慢", "热", "冷", "新"],
}

# Sentence templates for cross-task comparison
SENTENCE_TEMPLATES = {
    'translate': "将以下中文翻译成英文：{word}",
    'continue': "接下来会发生什么：{word}",
    'define': "请解释以下词语的含义：{word}",
    'antonym': "请说出以下词语的反义词：{word}",
}


# ============================================================
# Core extraction functions
# ============================================================

def load_model(model_key):
    """Load model and return model + tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    config = MODEL_CONFIGS[model_key]
    print(f"Loading {config['name']}...")
    
    tokenizer = AutoTokenizer.from_pretrained(config['name'], trust_remote_code=True)
    
    if model_key in ['deepseek7b', 'glm4']:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
        model = AutoModelForCausalLM.from_pretrained(
            config['name'],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config['name'],
            torch_dtype=config['dtype'],
            device_map="auto",
            trust_remote_code=True,
        )
    model.eval()
    return model, tokenizer


def extract_residuals(model, tokenizer, texts, model_key, last_token_only=True):
    """Extract residual stream representations at each layer for each text.
    
    Returns: dict[layer_idx] -> np.array of shape (n_texts, d_model)
    """
    config = MODEL_CONFIGS[model_key]
    n_layers = config['n_layers']
    
    all_residuals = {l: [] for l in range(n_layers)}
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # tuple of (n_layers+1,) each (1, seq_len, d_model)
            
            for l in range(n_layers):
                h = hidden_states[l + 1]  # +1 because hidden_states[0] is embedding
                if last_token_only:
                    all_residuals[l].append(h[0, -1, :].cpu().float().numpy())
                else:
                    all_residuals[l].append(h[0, :, :].cpu().float().numpy())
    
    # Stack into arrays
    for l in range(n_layers):
        all_residuals[l] = np.stack(all_residuals[l], axis=0)
    
    return all_residuals


def get_spike_subspace(residuals_dict, layer, n_components=None):
    """Get spike subspace (top-k right singular vectors) for a given layer.
    
    Args:
        residuals_dict: dict of layer -> (n_samples, d_model)
        layer: which layer
        n_components: number of components (if None, use known signal dims)
    
    Returns:
        V_k: (d_model, k) orthonormal basis for spike subspace
        s: singular values
    """
    X = residuals_dict[layer]  # (n, d)
    
    # Center
    X_centered = X - X.mean(axis=0, keepdims=True)
    
    # SVD
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    if n_components is None:
        n_components = min(X.shape)
    
    V_k = Vt[:n_components, :].T  # (d, k)
    
    return V_k, s


def project_out_subspace(X, V_k):
    """Remove spike subspace from representations.
    
    X: (n, d) representations
    V_k: (d, k) orthonormal subspace
    
    Returns: (n, d) with spike subspace removed
    """
    proj = X @ V_k @ V_k.T  # (n, d) projection onto spike subspace
    return X - proj


# ============================================================
# Exp 1: Semantic Distance Preservation
# ============================================================

def exp1_semantic_distance_preservation(model, tokenizer, model_key):
    """Test whether semantic distance structure is preserved across layers.
    
    Key idea: If semantic information flows through residual stream,
    the RANK ORDER of pairwise distances should be preserved across layers,
    even if absolute distances change.
    
    Method:
    1. Extract residual representations for 100 Chinese words at each layer
    2. Compute pairwise distance matrices at each layer (cosine + Euclidean)
    3. Compute rank correlation (Spearman) between distance matrices of different layers
    4. Compare: distance preservation WITH vs WITHOUT spike removed
    """
    print("\n" + "="*80)
    print("EXP 1: Semantic Distance Preservation")
    print("="*80)
    
    config = MODEL_CONFIGS[model_key]
    n_layers = config['n_layers']
    
    # Prepare word list
    words = [pair[0] for pair in TRANSLATION_PAIRS[:50]]  # Use 50 words for manageable computation
    
    # Create simple context: just the word in a sentence
    texts = [f"这个词是：{w}" for w in words]
    
    print(f"Extracting residuals for {len(words)} words...")
    residuals = extract_residuals(model, tokenizer, texts, model_key)
    
    results = {
        'model': model_key,
        'n_words': len(words),
        'layers_sampled': list(range(0, n_layers, 3)),  # Sample every 3 layers
        'distance_correlations': {},
        'distance_correlations_no_spike': {},
        'spike_removed_preservation': {},
    }
    
    sampled_layers = list(range(0, n_layers, 3))
    if (n_layers - 1) not in sampled_layers:
        sampled_layers.append(n_layers - 1)
    
    # Compute distance matrices for each sampled layer
    dist_matrices = {}
    dist_matrices_no_spike = {}
    
    for l in sampled_layers:
        X = residuals[l]  # (n, d)
        
        # Full residual distance matrix (cosine)
        cos_sim = cosine_similarity(X)
        cos_dist = 1 - cos_sim
        dist_matrices[l] = cos_dist
        
        # Remove spike subspace
        V_k, s = get_spike_subspace(residuals, l, n_components=min(25, X.shape[0]-1))
        X_no_spike = project_out_subspace(X, V_k)
        cos_sim_no_spike = cosine_similarity(X_no_spike)
        cos_dist_no_spike = 1 - cos_sim_no_spike
        dist_matrices_no_spike[l] = cos_dist_no_spike
    
    # Compute rank correlations between layers (with spike)
    print("\nDistance preservation WITH spike:")
    for i, l1 in enumerate(sampled_layers):
        for j, l2 in enumerate(sampled_layers):
            if i >= j:
                continue
            # Flatten upper triangle of distance matrices
            triu_idx = np.triu_indices(len(words), k=1)
            d1 = dist_matrices[l1][triu_idx]
            d2 = dist_matrices[l2][triu_idx]
            
            rho, p = spearmanr(d1, d2)
            key = f"L{l1}_L{l2}"
            results['distance_correlations'][key] = {
                'spearman_rho': float(rho),
                'p_value': float(p),
            }
            if abs(l1 - l2) <= 4:
                print(f"  L{l1}→L{l2}: ρ={rho:.3f} (p={p:.2e})")
    
    # Compute rank correlations between layers (without spike)
    print("\nDistance preservation WITHOUT spike (spike removed):")
    for i, l1 in enumerate(sampled_layers):
        for j, l2 in enumerate(sampled_layers):
            if i >= j:
                continue
            triu_idx = np.triu_indices(len(words), k=1)
            d1 = dist_matrices_no_spike[l1][triu_idx]
            d2 = dist_matrices_no_spike[l2][triu_idx]
            
            rho, p = spearmanr(d1, d2)
            key = f"L{l1}_L{l2}"
            results['distance_correlations_no_spike'][key] = {
                'spearman_rho': float(rho),
                'p_value': float(p),
            }
    
    # Critical test: spike removal effect on distance preservation
    print("\n*** CRITICAL TEST: Does spike removal preserve semantic structure? ***")
    for l in sampled_layers:
        triu_idx = np.triu_indices(len(words), k=1)
        d_full = dist_matrices[l][triu_idx]
        d_no_spike = dist_matrices_no_spike[l][triu_idx]
        
        rho, p = spearmanr(d_full, d_no_spike)
        
        # Also check semantic group internal distances
        group_preservations = {}
        for gname, gwords in SEMANTIC_GROUPS.items():
            gindices = [words.index(w) for w in gwords if w in words]
            if len(gindices) < 3:
                continue
            g_triu = np.triu_indices(len(gindices), k=1)
            d_full_g = dist_matrices[l][np.ix_(gindices, gindices)][g_triu]
            d_no_g = dist_matrices_no_spike[l][np.ix_(gindices, gindices)][g_triu]
            if len(d_full_g) > 0 and np.std(d_full_g) > 1e-10 and np.std(d_no_g) > 1e-10:
                rho_g, _ = spearmanr(d_full_g, d_no_g)
                group_preservations[gname] = float(rho_g)
        
        results['spike_removed_preservation'][f"L{l}"] = {
            'full_vs_no_spike_rho': float(rho),
            'p_value': float(p),
            'group_preservations': group_preservations,
        }
        print(f"  L{l}: ρ(full vs no-spike) = {rho:.3f}")
        for gname, grho in group_preservations.items():
            print(f"    {gname}: ρ = {grho:.3f}")
    
    return results


# ============================================================
# Exp 2: Manifold Local Linearity
# ============================================================

def exp2_manifold_local_linearity(model, tokenizer, model_key):
    """Test whether semantic groups form locally linear structures.
    
    Key idea: If concepts live on a manifold, semantically related words
    should form locally linear neighborhoods.
    
    Method:
    1. For each semantic group (animals, fruits, etc.),
       check if intra-group distances can be predicted from inter-group structure
    2. Test local linear reconstruction error
    3. Compare across layers
    """
    print("\n" + "="*80)
    print("EXP 2: Manifold Local Linearity")
    print("="*80)
    
    config = MODEL_CONFIGS[model_key]
    n_layers = config['n_layers']
    
    # Collect all words from semantic groups
    all_words = []
    word_to_group = {}
    for gname, gwords in SEMANTIC_GROUPS.items():
        for w in gwords:
            if w not in all_words:
                all_words.append(w)
                word_to_group[w] = gname
    
    texts = [f"这个词是：{w}" for w in all_words]
    
    print(f"Extracting residuals for {len(all_words)} words from {len(SEMANTIC_GROUPS)} groups...")
    residuals = extract_residuals(model, tokenizer, texts, model_key)
    
    results = {
        'model': model_key,
        'n_words': len(all_words),
        'n_groups': len(SEMANTIC_GROUPS),
        'layers_sampled': list(range(0, n_layers, 3)),
        'intra_group_cohesion': {},
        'inter_group_separation': {},
        'local_linearity_error': {},
    }
    
    sampled_layers = list(range(0, n_layers, 3))
    if (n_layers - 1) not in sampled_layers:
        sampled_layers.append(n_layers - 1)
    
    for l in sampled_layers:
        X = residuals[l]  # (n, d)
        
        # Compute pairwise distances
        cos_sim = cosine_similarity(X)
        cos_dist = 1 - cos_sim
        
        # Intra-group cohesion: average cosine distance within each group
        intra_cohesion = {}
        for gname, gwords in SEMANTIC_GROUPS.items():
            gindices = [all_words.index(w) for w in gwords if w in all_words]
            if len(gindices) < 2:
                continue
            intra_dists = []
            for i in range(len(gindices)):
                for j in range(i+1, len(gindices)):
                    intra_dists.append(cos_dist[gindices[i], gindices[j]])
            intra_cohesion[gname] = {
                'mean_dist': float(np.mean(intra_dists)),
                'std_dist': float(np.std(intra_dists)),
                'n_pairs': len(intra_dists),
            }
        
        # Inter-group separation: average cosine distance between groups
        group_names = list(SEMANTIC_GROUPS.keys())
        inter_separation = {}
        for gi in range(len(group_names)):
            for gj in range(gi+1, len(group_names)):
                g1name, g2name = group_names[gi], group_names[gj]
                g1indices = [all_words.index(w) for w in SEMANTIC_GROUPS[g1name] if w in all_words]
                g2indices = [all_words.index(w) for w in SEMANTIC_GROUPS[g2name] if w in all_words]
                
                inter_dists = []
                for i in g1indices:
                    for j in g2indices:
                        inter_dists.append(cos_dist[i, j])
                inter_separation[f"{g1name}_vs_{g2name}"] = {
                    'mean_dist': float(np.mean(inter_dists)),
                }
        
        # Local linear reconstruction error
        # For each word, try to reconstruct it from its k nearest neighbors
        k = min(5, len(all_words) - 1)
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        reconstruction_errors = []
        for i in range(len(all_words)):
            # Neighbors (excluding self)
            neighbor_idx = indices[i, 1:]  # (k,)
            X_neighbors = X[neighbor_idx]  # (k, d)
            X_target = X[i]  # (d,)
            
            # Linear reconstruction: min ||X_target - X_neighbors^T w||^2
            # Solution: w = (X_neighbors X_neighbors^T)^-1 X_neighbors X_target
            try:
                w = np.linalg.lstsq(X_neighbors.T, X_target, rcond=None)[0]
                X_reconstructed = X_neighbors.T @ w
                error = np.linalg.norm(X_target - X_reconstructed) / (np.linalg.norm(X_target) + 1e-10)
                reconstruction_errors.append(float(error))
            except:
                reconstruction_errors.append(float('nan'))
        
        # Compute local linearity for within-group vs between-group neighbors
        within_group_recon = []
        between_group_recon = []
        for i in range(len(all_words)):
            neighbor_idx = indices[i, 1:]
            target_group = word_to_group[all_words[i]]
            for j_idx, j in enumerate(neighbor_idx):
                neighbor_group = word_to_group[all_words[j]]
                # Use distance as proxy for reconstruction quality
                if neighbor_group == target_group:
                    within_group_recon.append(cos_dist[i, j])
                else:
                    between_group_recon.append(cos_dist[i, j])
        
        results['intra_group_cohesion'][f"L{l}"] = intra_cohesion
        results['inter_group_separation'][f"L{l}"] = inter_separation
        results['local_linearity_error'][f"L{l}"] = {
            'mean_recon_error': float(np.nanmean(reconstruction_errors)),
            'within_group_mean_dist': float(np.mean(within_group_recon)) if within_group_recon else None,
            'between_group_mean_dist': float(np.mean(between_group_recon)) if between_group_recon else None,
            'separation_ratio': float(np.mean(between_group_recon) / (np.mean(within_group_recon) + 1e-10)) if within_group_recon and between_group_recon else None,
        }
        
        sep_ratio = results['local_linearity_error'][f"L{l}"]['separation_ratio']
        recon_err = results['local_linearity_error'][f"L{l}"]['mean_recon_error']
        print(f"  L{l}: recon_error={recon_err:.3f}, separation_ratio={sep_ratio:.3f}" if sep_ratio else f"  L{l}: recon_error={recon_err:.3f}")
    
    return results


# ============================================================
# Exp 3: Control vs Data Separation
# ============================================================

def exp3_control_vs_data_separation(model, tokenizer, model_key):
    """Test whether spike subspace carries control or semantic information.
    
    Critical test:
    - If spike is CONTROL: removing spike should preserve semantic distance structure
    - If spike is SEMANTIC: removing spike should destroy semantic distance structure
    
    Method:
    1. Compute semantic distance matrix (within/between groups) with full residual
    2. Remove spike subspace
    3. Compute same distance matrix without spike
    4. Compare rank correlations
    
    Also: decompose residual into spike + complement, test each separately.
    """
    print("\n" + "="*80)
    print("EXP 3: Control vs Data Separation")
    print("="*80)
    
    config = MODEL_CONFIGS[model_key]
    n_layers = config['n_layers']
    
    # Use 50 words for tractability
    words = [pair[0] for pair in TRANSLATION_PAIRS[:50]]
    texts = [f"这个词是：{w}" for w in words]
    
    print(f"Extracting residuals for {len(words)} words...")
    residuals = extract_residuals(model, tokenizer, texts, model_key)
    
    results = {
        'model': model_key,
        'n_words': len(words),
        'layers_tested': [],
        'spike_semantic_content': {},
        'complement_semantic_content': {},
        'spike_group_separation': {},
        'complement_group_separation': {},
    }
    
    # Known spike dimensions from Phase 114
    known_dims = {
        'qwen3': {0: 25, 3: 24, 6: 23, 9: 21, 12: 4, 15: 14, 18: 8, 21: 20, 24: 21, 27: 18, 30: 23, 33: 24, 35: 14},
        'deepseek7b': {0: 29, 6: 22, 9: 22, 15: 16, 21: 21, 27: 11},
        'glm4': {0: 25, 6: 23, 12: 18, 18: 16, 24: 14, 30: 20, 35: 16, 39: 12},
    }
    
    test_layers = list(range(0, n_layers, 3))
    if (n_layers - 1) not in test_layers:
        test_layers.append(n_layers - 1)
    
    for l in test_layers:
        X = residuals[l]  # (n, d)
        
        # Get spike dimension
        spike_dim = known_dims.get(model_key, {}).get(l, min(20, X.shape[0]//2))
        
        # SVD to get spike subspace
        X_centered = X - X.mean(axis=0, keepdims=True)
        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
        V_spike = Vt[:spike_dim, :].T  # (d, spike_dim)
        V_complement = Vt[spike_dim:, :].T  # (d, d-spike_dim)
        
        # Project onto spike and complement
        X_spike = X_centered @ V_spike @ V_spike.T + X.mean(axis=0, keepdims=True)  # spike component + mean
        X_complement = X_centered @ V_complement @ V_complement.T + X.mean(axis=0, keepdims=True)
        X_no_spike = X - (X_centered @ V_spike @ V_spike.T)  # original minus spike projection
        
        # Compute semantic metrics for each representation
        rep_types = {
            'full': X,
            'spike_only': X_spike,
            'complement_only': X_complement,
            'no_spike': X_no_spike,
        }
        
        layer_result = {'spike_dim': spike_dim}
        
        for rep_name, X_rep in rep_types.items():
            # Compute pairwise cosine distances
            cos_sim = cosine_similarity(X_rep)
            cos_dist = 1 - cos_sim
            
            # Semantic group separation
            group_separations = {}
            for gname, gwords in SEMANTIC_GROUPS.items():
                gindices = [words.index(w) for w in gwords if w in words]
                if len(gindices) < 2:
                    continue
                
                # Intra-group distance
                intra_dists = []
                for i in range(len(gindices)):
                    for j in range(i+1, len(gindices)):
                        intra_dists.append(cos_dist[gindices[i], gindices[j]])
                
                # Inter-group distance (to all other groups)
                other_indices = [i for i in range(len(words)) if i not in gindices]
                inter_dists = []
                for i in gindices:
                    for j in other_indices:
                        inter_dists.append(cos_dist[i, j])
                
                group_separations[gname] = {
                    'intra_mean': float(np.mean(intra_dists)),
                    'inter_mean': float(np.mean(inter_dists)),
                    'separation_ratio': float(np.mean(inter_dists) / (np.mean(intra_dists) + 1e-10)),
                }
            
            # kNN accuracy: can we classify words into their semantic groups?
            group_labels = []
            word_indices_by_group = {}
            for gname, gwords in SEMANTIC_GROUPS.items():
                for w in gwords:
                    if w in words:
                        idx = words.index(w)
                        group_labels.append((idx, gname))
                        if gname not in word_indices_by_group:
                            word_indices_by_group[gname] = []
                        word_indices_by_group[gname].append(idx)
            
            # Simple kNN classification (k=3)
            correct = 0
            total = 0
            for idx, true_group in group_labels:
                dists_to_all = cos_dist[idx]
                dists_to_all[idx] = float('inf')  # exclude self
                nearest_idx = np.argsort(dists_to_all)[:3]
                neighbor_groups = [gn for i, gn in group_labels if i in nearest_idx]
                if neighbor_groups:
                    from collections import Counter
                    pred_group = Counter(neighbor_groups).most_common(1)[0][0]
                    if pred_group == true_group:
                        correct += 1
                    total += 1
            
            knn_acc = correct / total if total > 0 else 0
            
            # Rank correlation with full representation
            triu_idx = np.triu_indices(len(words), k=1)
            d_rep = cos_dist[triu_idx]
            
            # Full representation as reference
            cos_sim_full = cosine_similarity(X)
            cos_dist_full = 1 - cos_sim_full
            d_full = cos_dist_full[triu_idx]
            
            rho_with_full, _ = spearmanr(d_full, d_rep) if np.std(d_rep) > 1e-10 else (0, 1)
            
            layer_result[rep_name] = {
                'group_separations': group_separations,
                'knn_accuracy': float(knn_acc),
                'rho_with_full': float(rho_with_full),
            }
        
        results['layers_tested'].append(l)
        results[f"L{l}"] = layer_result
        
        # Print key results
        print(f"\n  L{l} (spike_dim={spike_dim}):")
        for rep_name in ['full', 'spike_only', 'complement_only', 'no_spike']:
            r = layer_result[rep_name]
            avg_sep = np.mean([v['separation_ratio'] for v in r['group_separations'].values()])
            print(f"    {rep_name:20s}: knn_acc={r['knn_accuracy']:.3f}, ρ(full)={r['rho_with_full']:.3f}, avg_sep_ratio={avg_sep:.2f}")
    
    return results


# ============================================================
# Exp 4: Cross-Task Spike Comparison
# ============================================================

def exp4_cross_task_spike(model, tokenizer, model_key):
    """Compare spike subspaces across different tasks.
    
    Key question: Is the spike direction task-specific or task-general?
    
    If spike overlaps across tasks → spike is "task control" signal
    If spike is task-specific → spike encodes task-specific content
    """
    print("\n" + "="*80)
    print("EXP 4: Cross-Task Spike Comparison")
    print("="*80)
    
    config = MODEL_CONFIGS[model_key]
    n_layers = config['n_layers']
    
    # Use a subset of words
    test_words = [pair[0] for pair in TRANSLATION_PAIRS[:30]]
    
    # Generate texts for different tasks
    task_texts = {}
    for task_name, template in SENTENCE_TEMPLATES.items():
        task_texts[task_name] = [template.format(word=w) for w in test_words]
    
    # Extract residuals for each task
    task_residuals = {}
    for task_name, texts in task_texts.items():
        print(f"Extracting residuals for task: {task_name}...")
        task_residuals[task_name] = extract_residuals(model, tokenizer, texts, model_key)
    
    results = {
        'model': model_key,
        'n_words': len(test_words),
        'tasks': list(SENTENCE_TEMPLATES.keys()),
        'layers_sampled': list(range(0, n_layers, 3)),
        'spike_overlap': {},
        'spike_cosine': {},
    }
    
    sampled_layers = list(range(0, n_layers, 3))
    if (n_layers - 1) not in sampled_layers:
        sampled_layers.append(n_layers - 1)
    
    task_names = list(SENTENCE_TEMPLATES.keys())
    
    for l in sampled_layers:
        # Compute spike subspace for each task
        task_spikes = {}
        for task_name in task_names:
            X = task_residuals[task_name][l]
            X_centered = X - X.mean(axis=0, keepdims=True)
            U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
            k = min(10, len(s))  # Use top-10 for comparison
            task_spikes[task_name] = {
                'V': Vt[:k, :].T,  # (d, k)
                's': s[:k],
                'top1': Vt[0, :],  # (d,)
            }
        
        # Pairwise overlap between tasks
        layer_overlaps = {}
        layer_cosines = {}
        for i, t1 in enumerate(task_names):
            for j, t2 in enumerate(task_names):
                if i >= j:
                    continue
                
                # Subspace inclusion ratio (Grassmann)
                V1 = task_spikes[t1]['V']
                V2 = task_spikes[t2]['V']
                
                # V1 subspace inclusion in V2
                proj = V2.T @ V1 @ V1.T @ V2  # (k, k)
                inc_1in2 = np.trace(proj) / V1.shape[1]
                
                # V2 subspace inclusion in V1
                proj2 = V1.T @ V2 @ V2.T @ V1
                inc_2in1 = np.trace(proj2) / V2.shape[1]
                
                # Top-1 vector cosine
                cos_top1 = abs(np.dot(task_spikes[t1]['top1'], task_spikes[t2]['top1']))
                
                key = f"{t1}_vs_{t2}"
                layer_overlaps[key] = {
                    'inc_1in2': float(inc_1in2),
                    'inc_2in1': float(inc_2in1),
                    'avg_inclusion': float((inc_1in2 + inc_2in1) / 2),
                }
                layer_cosines[key] = float(cos_top1)
        
        results['spike_overlap'][f"L{l}"] = layer_overlaps
        results['spike_cosine'][f"L{l}"] = layer_cosines
        
        # Print key overlaps with translate task
        print(f"\n  L{l}:")
        for t2 in task_names:
            if t2 == 'translate':
                continue
            key = f"translate_vs_{t2}"
            if key in layer_overlaps:
                avg_inc = layer_overlaps[key]['avg_inclusion']
                cos_t1 = layer_cosines.get(key, 0)
                print(f"    translate<->{t2}: avg_inclusion={avg_inc:.3f}, |cos(top1)|={cos_t1:.3f}")
    
    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 117: Semantic Invariants")
    parser.add_argument('--model', type=str, required=True, choices=['qwen3', 'deepseek7b', 'glm4'])
    parser.add_argument('--exp', type=str, required=True, choices=['1', '2', '3', '4', 'all'])
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load model
    model, tokenizer = load_model(args.model)
    
    results_all = {}
    
    if args.exp in ['1', 'all']:
        r1 = exp1_semantic_distance_preservation(model, tokenizer, args.model)
        results_all['exp1'] = r1
        out_path = OUTPUT_DIR / f"phase117_exp1_{args.model}_semantic_distance.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(r1, f, ensure_ascii=False, indent=2)
        print(f"\nExp1 results saved to {out_path}")
    
    if args.exp in ['2', 'all']:
        r2 = exp2_manifold_local_linearity(model, tokenizer, args.model)
        results_all['exp2'] = r2
        out_path = OUTPUT_DIR / f"phase117_exp2_{args.model}_manifold_linearity.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(r2, f, ensure_ascii=False, indent=2)
        print(f"\nExp2 results saved to {out_path}")
    
    if args.exp in ['3', 'all']:
        r3 = exp3_control_vs_data_separation(model, tokenizer, args.model)
        results_all['exp3'] = r3
        out_path = OUTPUT_DIR / f"phase117_exp3_{args.model}_control_vs_data.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(r3, f, ensure_ascii=False, indent=2)
        print(f"\nExp3 results saved to {out_path}")
    
    if args.exp in ['4', 'all']:
        r4 = exp4_cross_task_spike(model, tokenizer, args.model)
        results_all['exp4'] = r4
        out_path = OUTPUT_DIR / f"phase117_exp4_{args.model}_cross_task.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(r4, f, ensure_ascii=False, indent=2)
        print(f"\nExp4 results saved to {out_path}")
    
    # Save combined results
    if len(results_all) > 1:
        out_path = OUTPUT_DIR / f"phase117_{args.model}_all_results.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results_all, f, ensure_ascii=False, indent=2)
        print(f"\nAll results saved to {out_path}")
    
    # Free GPU memory
    del model
    torch.cuda.empty_cache()
    print("\nGPU memory freed.")


if __name__ == "__main__":
    main()
