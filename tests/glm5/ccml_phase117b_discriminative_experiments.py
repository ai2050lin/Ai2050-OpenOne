"""
Phase 117b: Discriminative Experiments — Three critical tests
判别性实验：三个关键检验

Exp A: Tokenizer Control — 证明差分信号不是分词器伪影
Exp B: Semantic Feature Disentanglement — 证明spike不只是粗聚类
Exp C: CCA vs PCA Comparison — 找与输出logit真正相关的方向
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
from sklearn.cross_decomposition import CCA
from scipy.stats import spearmanr
from collections import Counter

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
}

OUTPUT_DIR = Path("d:/Ai2050/TransformerLens-Project/tests/glm5_temp")

# ============================================================
# Semantic features for disentanglement test
# ============================================================

# Words with multiple semantic features annotated
# Format: (chinese, english, features_dict)
# Features: animacy (0=non-living, 1=living), concreteness (0=abstract, 1=concrete),
# countability (0=mass, 1=countable), sentiment (-1=negative, 0=neutral, 1=positive),
# size (0=small, 1=medium, 2=large), domain (0=nature, 1=artifact, 2=abstract, 3=person)
WORD_FEATURES = [
    # Animals (animacy=1, concrete=1, countable=1)
    ("猫", "cat", {"animacy": 1, "concrete": 1, "countable": 1, "sentiment": 1, "size": 0, "domain": 0}),
    ("狗", "dog", {"animacy": 1, "concrete": 1, "countable": 1, "sentiment": 1, "size": 0, "domain": 0}),
    ("鸟", "bird", {"animacy": 1, "concrete": 1, "countable": 1, "sentiment": 0, "size": 0, "domain": 0}),
    ("马", "horse", {"animacy": 1, "concrete": 1, "countable": 1, "sentiment": 1, "size": 1, "domain": 0}),
    ("牛", "cow", {"animacy": 1, "concrete": 1, "countable": 1, "sentiment": 0, "size": 2, "domain": 0}),
    ("鱼", "fish", {"animacy": 1, "concrete": 1, "countable": 1, "sentiment": 0, "size": 0, "domain": 0}),
    
    # Fruits (animacy=0, concrete=1, countable=1)
    ("苹果", "apple", {"animacy": 0, "concrete": 1, "countable": 1, "sentiment": 1, "size": 0, "domain": 0}),
    ("香蕉", "banana", {"animacy": 0, "concrete": 1, "countable": 1, "sentiment": 1, "size": 0, "domain": 0}),
    ("西瓜", "watermelon", {"animacy": 0, "concrete": 1, "countable": 1, "sentiment": 1, "size": 1, "domain": 0}),
    ("葡萄", "grape", {"animacy": 0, "concrete": 1, "countable": 1, "sentiment": 1, "size": 0, "domain": 0}),
    
    # Furniture (animacy=0, concrete=1, countable=1)
    ("桌子", "table", {"animacy": 0, "concrete": 1, "countable": 1, "sentiment": 0, "size": 1, "domain": 1}),
    ("椅子", "chair", {"animacy": 0, "concrete": 1, "countable": 1, "sentiment": 0, "size": 0, "domain": 1}),
    ("床", "bed", {"animacy": 0, "concrete": 1, "countable": 1, "sentiment": 1, "size": 2, "domain": 1}),
    ("门", "door", {"animacy": 0, "concrete": 1, "countable": 1, "sentiment": 0, "size": 1, "domain": 1}),
    
    # Colors (animacy=0, concrete=0 [abstract property], countable=0)
    ("红色", "red", {"animacy": 0, "concrete": 0, "countable": 0, "sentiment": 1, "size": -1, "domain": 2}),
    ("蓝色", "blue", {"animacy": 0, "concrete": 0, "countable": 0, "sentiment": 0, "size": -1, "domain": 2}),
    ("绿色", "green", {"animacy": 0, "concrete": 0, "countable": 0, "sentiment": 1, "size": -1, "domain": 2}),
    ("黑色", "black", {"animacy": 0, "concrete": 0, "countable": 0, "sentiment": -1, "size": -1, "domain": 2}),
    
    # Weather (animacy=0, concrete=1 [somewhat], countable=0)
    ("太阳", "sun", {"animacy": 0, "concrete": 1, "countable": 0, "sentiment": 1, "size": 2, "domain": 0}),
    ("雨", "rain", {"animacy": 0, "concrete": 1, "countable": 0, "sentiment": -1, "size": -1, "domain": 0}),
    ("风", "wind", {"animacy": 0, "concrete": 1, "countable": 0, "sentiment": 0, "size": -1, "domain": 0}),
    ("雪", "snow", {"animacy": 0, "concrete": 1, "countable": 0, "sentiment": 1, "size": -1, "domain": 0}),
    
    # Emotions (animacy=0, concrete=0, countable=0)
    ("快乐", "happy", {"animacy": 0, "concrete": 0, "countable": 0, "sentiment": 1, "size": -1, "domain": 2}),
    ("悲伤", "sad", {"animacy": 0, "concrete": 0, "countable": 0, "sentiment": -1, "size": -1, "domain": 2}),
    ("愤怒", "angry", {"animacy": 0, "concrete": 0, "countable": 0, "sentiment": -1, "size": -1, "domain": 2}),
    ("爱", "love", {"animacy": 0, "concrete": 0, "countable": 0, "sentiment": 1, "size": -1, "domain": 2}),
    
    # Actions (animacy=-1, concrete=0, countable=0)
    ("跑步", "run", {"animacy": -1, "concrete": 0, "countable": 0, "sentiment": 1, "size": -1, "domain": 2}),
    ("游泳", "swim", {"animacy": -1, "concrete": 0, "countable": 0, "sentiment": 1, "size": -1, "domain": 2}),
    ("跳舞", "dance", {"animacy": -1, "concrete": 0, "countable": 0, "sentiment": 1, "size": -1, "domain": 2}),
    ("吃", "eat", {"animacy": -1, "concrete": 0, "countable": 0, "sentiment": 1, "size": -1, "domain": 2}),
    
    # People (animacy=1, concrete=1, countable=1)
    ("老师", "teacher", {"animacy": 1, "concrete": 1, "countable": 1, "sentiment": 0, "size": 1, "domain": 3}),
    ("医生", "doctor", {"animacy": 1, "concrete": 1, "countable": 1, "sentiment": 1, "size": 1, "domain": 3}),
    ("朋友", "friend", {"animacy": 1, "concrete": 1, "countable": 1, "sentiment": 1, "size": 1, "domain": 3}),
    
    # Size adjectives (animacy=-1, concrete=0, countable=0)
    ("大", "big", {"animacy": -1, "concrete": 0, "countable": 0, "sentiment": 0, "size": 2, "domain": 2}),
    ("小", "small", {"animacy": -1, "concrete": 0, "countable": 0, "sentiment": 0, "size": 0, "domain": 2}),
    ("长", "long", {"animacy": -1, "concrete": 0, "countable": 0, "sentiment": 0, "size": 1, "domain": 2}),
    ("快", "fast", {"animacy": -1, "concrete": 0, "countable": 0, "sentiment": 1, "size": -1, "domain": 2}),
    
    # Abstract concepts (animacy=0, concrete=0, countable=0)
    ("自由", "freedom", {"animacy": 0, "concrete": 0, "countable": 0, "sentiment": 1, "size": -1, "domain": 2}),
    ("希望", "hope", {"animacy": 0, "concrete": 0, "countable": 0, "sentiment": 1, "size": -1, "domain": 2}),
    ("梦想", "dream", {"animacy": 0, "concrete": 0, "countable": 0, "sentiment": 1, "size": -1, "domain": 2}),
    ("时间", "time", {"animacy": 0, "concrete": 0, "countable": 0, "sentiment": 0, "size": -1, "domain": 2}),
]

FEATURE_NAMES = ["animacy", "concrete", "countable", "sentiment", "size", "domain"]


# ============================================================
# Core functions
# ============================================================

def load_model(model_key):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    config = MODEL_CONFIGS[model_key]
    print(f"Loading {config['name']}...")
    
    tokenizer = AutoTokenizer.from_pretrained(config['name'], trust_remote_code=True)
    
    if model_key in ['deepseek7b']:
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
    config = MODEL_CONFIGS[model_key]
    n_layers = config['n_layers']
    
    all_residuals = {l: [] for l in range(n_layers)}
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
            for l in range(n_layers):
                h = hidden_states[l + 1]
                if last_token_only:
                    all_residuals[l].append(h[0, -1, :].cpu().float().numpy())
                else:
                    all_residuals[l].append(h[0, :, :].cpu().float().numpy())
    
    for l in range(n_layers):
        all_residuals[l] = np.stack(all_residuals[l], axis=0)
    
    return all_residuals


def get_logit_vector(model, tokenizer, target_token, model_key):
    """Get the logit vector for a specific token."""
    token_id = tokenizer.encode(target_token, add_special_tokens=False)[0]
    return token_id


# ============================================================
# Exp A: Tokenizer Control
# ============================================================

def expA_tokenizer_control(model, tokenizer, model_key):
    """Test whether translation difference is a tokenizer artifact.
    
    Key idea: If the difference between translation and continuation 
    is just due to tokenization differences (different token lengths, 
    different scripts), then a control condition that matches these 
    should show similar low-rank structure.
    
    Control conditions:
    1. Same-script pseudo-translation: "Translate X to Chinese" (zh->zh, same script)
    2. Length-matched continuation: "What is X and Y" (matched token length)
    3. Cross-script but non-translation: "Write X in English letters" (script switch, no translation)
    
    If translation spike survives all controls, it's real.
    If it's killed by script control, it's a tokenizer artifact.
    """
    print("\n" + "="*80)
    print("EXP A: Tokenizer Control — Is the spike a tokenizer artifact?")
    print("="*80)
    
    config = MODEL_CONFIGS[model_key]
    n_layers = config['n_layers']
    
    test_words = [pair[0] for pair in WORD_FEATURES[:30]]
    
    # Four conditions
    conditions = {}
    
    # Condition 1: Real translation (zh -> en)
    conditions['translate_zh_en'] = [f"将以下中文翻译成英文：{w}" for w in test_words]
    
    # Condition 2: Chinese continuation (same script, no translation)
    conditions['continue_zh'] = [f"接下来会发生什么：{w}" for w in test_words]
    
    # Condition 3: Pseudo-translation (zh -> zh, same script, matched instruction)
    # "Translate X to Chinese" — should not trigger real translation mechanism
    conditions['pseudo_translate_zh_zh'] = [f"将以下中文翻译成中文：{w}" for w in test_words]
    
    # Condition 4: Script switch without translation (write pinyin)
    # This switches script but doesn't require translation
    conditions['script_switch'] = [f"请用拼音写出以下词语：{w}" for w in test_words]
    
    # Condition 5: English definition (different language, no translation needed)
    conditions['define_en'] = [f"Please define the meaning of this word in English: {w}" for w in test_words]
    
    # Extract residuals for each condition
    condition_residuals = {}
    for cond_name, texts in conditions.items():
        print(f"\nExtracting residuals for condition: {cond_name}...")
        condition_residuals[cond_name] = extract_residuals(model, tokenizer, texts, model_key)
    
    results = {
        'model': model_key,
        'n_words': len(test_words),
        'conditions': list(conditions.keys()),
        'layers_tested': list(range(0, n_layers, 3)),
        'spike_dimensions': {},
        'cross_condition_overlap': {},
        'spike_vs_control_svd': {},
    }
    
    sampled_layers = list(range(0, n_layers, 3))
    if (n_layers - 1) not in sampled_layers:
        sampled_layers.append(n_layers - 1)
    
    # For each layer, compute spike subspace for each condition
    for l in sampled_layers:
        layer_spikes = {}
        layer_singular_values = {}
        
        for cond_name in conditions.keys():
            X = condition_residuals[cond_name][l]
            X_centered = X - X.mean(axis=0, keepdims=True)
            U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
            
            # Compute participation ratio
            s2 = s ** 2
            pr = (s2.sum() ** 2) / (s2 ** 2).sum()
            
            # Top-1 concentration
            concentration = s2[0] / s2.sum() if len(s2) > 0 else 0
            
            # Eig1/eig2 ratio
            eig_ratio = s[0] / s[1] if len(s) > 1 else float('inf')
            
            layer_spikes[cond_name] = Vt[:10, :].T  # (d, 10) for overlap comparison
            layer_singular_values[cond_name] = {
                'pr': float(pr),
                'concentration': float(concentration),
                'eig1_eig2': float(eig_ratio),
                'top5_svs': [float(x) for x in s[:5]],
            }
        
        results['spike_dimensions'][f"L{l}"] = layer_singular_values
        
        # Cross-condition spike overlap
        cond_names = list(conditions.keys())
        for i, c1 in enumerate(cond_names):
            for j, c2 in enumerate(cond_names):
                if i >= j:
                    continue
                
                V1 = layer_spikes[c1]
                V2 = layer_spikes[c2]
                
                # Top-1 cosine
                cos_top1 = abs(np.dot(V1[:, 0], V2[:, 0]))
                
                # Top-5 subspace inclusion
                proj = V2[:, :5].T @ V1[:, :5] @ V1[:, :5].T @ V2[:, :5]
                avg_inc = np.trace(proj) / 5
                
                key = f"{c1}_vs_{c2}"
                if f"L{l}" not in results['cross_condition_overlap']:
                    results['cross_condition_overlap'][f"L{l}"] = {}
                results['cross_condition_overlap'][f"L{l}"][key] = {
                    'cos_top1': float(cos_top1),
                    'avg_inclusion_top5': float(avg_inc),
                }
        
        # Print key comparisons
        print(f"\n  L{l}:")
        for c1 in ['translate_zh_en']:
            for c2 in cond_names:
                if c2 == c1:
                    continue
                key = f"{c1}_vs_{c2}"
                if key in results['cross_condition_overlap'].get(f"L{l}", {}):
                    ov = results['cross_condition_overlap'][f"L{l}"][key]
                    print(f"    {c1} vs {c2}: cos(top1)={ov['cos_top1']:.3f}, avg_inc={ov['avg_inclusion_top5']:.3f}")
    
    # Critical comparison: translate spike dimensions vs control
    print("\n\n*** CRITICAL: PR and concentration comparison ***")
    for l in sampled_layers:
        print(f"  L{l}:", end="")
        for cond in ['translate_zh_en', 'continue_zh', 'pseudo_translate_zh_zh', 'script_switch']:
            if cond in layer_singular_values:
                ld = results['spike_dimensions'][f"L{l}"][cond]
                print(f"  {cond}: PR={ld['pr']:.1f}, conc={ld['concentration']:.3f}", end="")
        print()
    
    return results


# ============================================================
# Exp B: Semantic Feature Disentanglement
# ============================================================

def expB_semantic_disentanglement(model, tokenizer, model_key):
    """Test whether spike encodes fine-grained semantic features, not just clusters.
    
    Key question: Can we decode individual semantic features (animacy, concreteness, 
    etc.) from spike coefficients? Or does spike only preserve coarse cluster membership?
    
    Method:
    1. For each semantic feature, train a linear probe on spike coefficients
    2. Compare with probe on full residual and complement
    3. If spike outperforms complement on fine-grained features, spike truly encodes semantics
    """
    print("\n" + "="*80)
    print("EXP B: Semantic Feature Disentanglement — Is spike just coarse clustering?")
    print("="*80)
    
    config = MODEL_CONFIGS[model_key]
    n_layers = config['n_layers']
    
    words = [wf[0] for wf in WORD_FEATURES]
    features = {fname: np.array([wf[2][fname] for wf in WORD_FEATURES]) for fname in FEATURE_NAMES}
    
    # Remove samples with size=-1 (undefined) for size feature
    valid_size_mask = features['size'] >= 0
    
    texts = [f"这个词是：{w}" for w in words]
    
    print(f"Extracting residuals for {len(words)} words with {len(FEATURE_NAMES)} features...")
    residuals = extract_residuals(model, tokenizer, texts, model_key)
    
    results = {
        'model': model_key,
        'n_words': len(words),
        'features': FEATURE_NAMES,
        'layers_tested': list(range(0, n_layers, 3)),
        'feature_probe_accuracy': {},
    }
    
    sampled_layers = list(range(0, n_layers, 3))
    if (n_layers - 1) not in sampled_layers:
        sampled_layers.append(n_layers - 1)
    
    for l in sampled_layers:
        X = residuals[l]  # (n, d)
        X_centered = X - X.mean(axis=0, keepdims=True)
        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Define spike dimension (use known values or default to 25)
        spike_dim = min(25, X.shape[0] // 2)
        V_spike = Vt[:spike_dim, :].T  # (d, spike_dim)
        V_complement = Vt[spike_dim:, :].T  # (d, d-spike_dim)
        
        # Coefficients
        X_spike = X_centered @ V_spike  # (n, spike_dim)
        X_complement = X_centered @ V_complement  # (n, d-spike_dim)
        X_full = X_centered  # (n, d)
        
        layer_results = {}
        
        for fname in FEATURE_NAMES:
            y = features[fname]
            
            # Skip size if many undefined values
            if fname == 'size':
                mask = valid_size_mask
            else:
                mask = np.ones(len(y), dtype=bool)
            
            if mask.sum() < 5:
                continue
            
            # Get unique values for this feature
            unique_vals = np.unique(y[mask])
            
            if len(unique_vals) < 2:
                continue
            
            # For binary features: simple threshold probe
            # For multi-class: leave-one-out kNN
            
            feature_results = {}
            
            for rep_name, X_rep in [('full', X_full[mask]), 
                                      ('spike', X_spike[mask]),
                                      ('complement', X_complement[mask])]:
                y_masked = y[mask]
                
                if len(unique_vals) == 2:
                    # Binary: use median split probe
                    # Project onto first component and check classification
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.model_selection import LeaveOneOut
                    
                    # Use leave-one-out with logistic regression
                    correct = 0
                    total = 0
                    
                    # Simplified: use kNN (k=3) for speed
                    for i in range(len(y_masked)):
                        # Leave one out
                        X_train = np.delete(X_rep, i, axis=0)
                        y_train = np.delete(y_masked, i)
                        X_test = X_rep[i:i+1]
                        
                        # Find k nearest neighbors
                        dists = np.sum((X_train - X_test) ** 2, axis=1)
                        nearest = np.argsort(dists)[:3]
                        neighbor_labels = y_train[nearest]
                        
                        # Majority vote
                        pred = Counter(neighbor_labels).most_common(1)[0][0]
                        if pred == y_masked[i]:
                            correct += 1
                        total += 1
                    
                    acc = correct / total if total > 0 else 0
                
                else:
                    # Multi-class: kNN (k=3)
                    correct = 0
                    total = 0
                    
                    for i in range(len(y_masked)):
                        X_train = np.delete(X_rep, i, axis=0)
                        y_train = np.delete(y_masked, i)
                        X_test = X_rep[i:i+1]
                        
                        dists = np.sum((X_train - X_test) ** 2, axis=1)
                        nearest = np.argsort(dists)[:3]
                        neighbor_labels = y_train[nearest]
                        
                        pred = Counter(neighbor_labels).most_common(1)[0][0]
                        if pred == y_masked[i]:
                            correct += 1
                        total += 1
                    
                    acc = correct / total if total > 0 else 0
                
                feature_results[rep_name] = float(acc)
            
            layer_results[fname] = feature_results
        
        results['feature_probe_accuracy'][f"L{l}"] = layer_results
        
        # Print results
        print(f"\n  L{l} (spike_dim={spike_dim}):")
        for fname in FEATURE_NAMES:
            if fname in layer_results:
                r = layer_results[fname]
                full_acc = r.get('full', 0)
                spike_acc = r.get('spike', 0)
                comp_acc = r.get('complement', 0)
                print(f"    {fname:15s}: full={full_acc:.3f}, spike={spike_acc:.3f}, complement={comp_acc:.3f}")
    
    return results


# ============================================================
# Exp C: CCA vs PCA Comparison
# ============================================================

def expC_cca_vs_pca(model, tokenizer, model_key):
    """Compare PCA directions with CCA directions (correlated with output logits).
    
    Key question: Are the highest-variance directions (PCA) the same as 
    the most output-relevant directions (CCA)?
    
    If PCA ≈ CCA: variance = causal relevance, current method is fine
    If PCA ≠ CCA: we've been studying the wrong directions
    """
    print("\n" + "="*80)
    print("EXP C: CCA vs PCA Comparison — Are variance directions causal?")
    print("="*80)
    
    config = MODEL_CONFIGS[model_key]
    n_layers = config['n_layers']
    
    # Translation pairs for logit extraction
    test_pairs = WORD_FEATURES[:30]
    zh_words = [wf[0] for wf in test_pairs]
    en_words = [wf[1] for wf in test_pairs]
    
    # Translation task texts
    translate_texts = [f"将以下中文翻译成英文：{w}" for w in zh_words]
    
    print(f"Extracting residuals for {len(zh_words)} translation pairs...")
    residuals = extract_residuals(model, tokenizer, translate_texts, model_key)
    
    # Get target token logit indices
    target_token_ids = []
    for en_word in en_words:
        ids = tokenizer.encode(" " + en_word, add_special_tokens=False)
        target_token_ids.append(ids[0] if ids else 0)
    
    results = {
        'model': model_key,
        'n_pairs': len(test_pairs),
        'layers_tested': list(range(0, n_layers, 3)),
        'pca_vs_cca_alignment': {},
        'logit_variance_explained': {},
    }
    
    sampled_layers = list(range(0, n_layers, 3))
    if (n_layers - 1) not in sampled_layers:
        sampled_layers.append(n_layers - 1)
    
    for l in sampled_layers:
        X = residuals[l]  # (n, d)
        X_centered = X - X.mean(axis=0, keepdims=True)
        
        # Get logit vector for each sample
        # Use the unembedding matrix to get logit direction
        # Since we can't easily extract per-sample logits, use a proxy:
        # the target token's unembedding vector
        
        # Get unembedding matrix
        if hasattr(model, 'lm_head'):
            W_unembed = model.lm_head.weight.detach().cpu().float().numpy()  # (vocab, d)
        elif hasattr(model, 'model') and hasattr(model.model, 'lm_head'):
            W_unembed = model.model.lm_head.weight.detach().cpu().float().numpy()
        else:
            # Try to find it
            print(f"  Warning: Cannot find lm_head for L{l}, skipping CCA")
            continue
        
        # Get target logit directions
        target_vectors = W_unembed[target_token_ids]  # (n, d)
        
        # Center target vectors
        target_centered = target_vectors - target_vectors.mean(axis=0, keepdims=True)
        
        # PCA on residuals
        U_pca, s_pca, Vt_pca = np.linalg.svd(X_centered, full_matrices=False)
        pca_top1 = Vt_pca[0, :]  # (d,)
        pca_top5 = Vt_pca[:5, :].T  # (d, 5)
        
        # CCA between residuals and target logit vectors
        # Need to match dimensions
        n_components = min(X_centered.shape[0] - 1, X_centered.shape[1], target_centered.shape[1], 5)
        
        try:
            cca = CCA(n_components=n_components)
            X_cca = cca.fit_transform(X_centered, target_centered)
            
            # CCA directions in original space
            cca_x_weights = cca.x_weights_  # (d, n_components) - directions in X space
            cca_y_weights = cca.y_weights_  # (d, n_components) - directions in Y space
            
            # CCA correlations
            cca_correlations = []
            for i in range(n_components):
                corr = np.corrcoef(X_cca[0][:, i], X_cca[1][:, i])[0, 1]
                cca_correlations.append(float(abs(corr)))
            
            cca_top1 = cca_x_weights[:, 0]  # (d,) - direction most correlated with output
            
            # Alignment between PCA and CCA top-1
            cos_pca_cca = abs(np.dot(pca_top1, cca_top1))
            
            # Subspace alignment: how much of PCA top-5 is in CCA top-5?
            proj_cca = cca_x_weights[:, :5].T @ pca_top5 @ pca_top5.T @ cca_x_weights[:, :5]
            pca_in_cca = np.trace(proj_cca) / 5
            
            proj_pca = pca_top5.T @ cca_x_weights[:, :5] @ cca_x_weights[:, :5].T @ pca_top5
            cca_in_pca = np.trace(proj_pca) / 5
            
        except Exception as e:
            print(f"  L{l}: CCA failed: {e}")
            cos_pca_cca = 0
            pca_in_cca = 0
            cca_in_pca = 0
            cca_correlations = []
        
        # How much of PCA variance is related to output?
        # Project PCA directions onto target logit space
        logit_variance_by_pca = {}
        for k in [1, 5, 10, 25]:
            V_k = Vt_pca[:k, :].T  # (d, k)
            # Project target vectors onto PCA subspace
            proj_target = target_centered @ V_k @ V_k.T
            # Fraction of target variance captured
            total_var = np.sum(target_centered ** 2)
            captured_var = np.sum(proj_target ** 2)
            logit_variance_by_pca[f'top{k}'] = float(captured_var / (total_var + 1e-10))
        
        results['pca_vs_cca_alignment'][f"L{l}"] = {
            'cos_pca_top1_cca_top1': float(cos_pca_cca),
            'pca_top5_in_cca_top5': float(pca_in_cca),
            'cca_top5_in_pca_top5': float(cca_in_pca),
            'cca_correlations': cca_correlations[:5],
        }
        results['logit_variance_explained'][f"L{l}"] = logit_variance_by_pca
        
        print(f"\n  L{l}:")
        print(f"    cos(PCA_top1, CCA_top1) = {cos_pca_cca:.3f}")
        print(f"    PCA_top5 in CCA_top5    = {pca_in_cca:.3f}")
        print(f"    CCA_top5 in PCA_top5    = {cca_in_pca:.3f}")
        print(f"    CCA correlations (top5) = {[f'{c:.3f}' for c in cca_correlations[:5]]}")
        print(f"    Logit variance by PCA top1 = {logit_variance_by_pca['top1']:.3f}")
        print(f"    Logit variance by PCA top5 = {logit_variance_by_pca['top5']:.3f}")
        print(f"    Logit variance by PCA top25 = {logit_variance_by_pca['top25']:.3f}")
    
    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 117b: Discriminative Experiments")
    parser.add_argument('--model', type=str, required=True, choices=['qwen3', 'deepseek7b'])
    parser.add_argument('--exp', type=str, required=True, choices=['A', 'B', 'C', 'all'])
    args = parser.parse_args()
    
    model, tokenizer = load_model(args.model)
    
    results_all = {}
    
    if args.exp in ['A', 'all']:
        rA = expA_tokenizer_control(model, tokenizer, args.model)
        results_all['expA'] = rA
        out_path = OUTPUT_DIR / f"phase117b_expA_{args.model}_tokenizer_control.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(rA, f, ensure_ascii=False, indent=2)
        print(f"\nExpA results saved to {out_path}")
    
    if args.exp in ['B', 'all']:
        rB = expB_semantic_disentanglement(model, tokenizer, args.model)
        results_all['expB'] = rB
        out_path = OUTPUT_DIR / f"phase117b_expB_{args.model}_semantic_disentangle.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(rB, f, ensure_ascii=False, indent=2)
        print(f"\nExpB results saved to {out_path}")
    
    if args.exp in ['C', 'all']:
        rC = expC_cca_vs_pca(model, tokenizer, args.model)
        results_all['expC'] = rC
        out_path = OUTPUT_DIR / f"phase117b_expC_{args.model}_cca_vs_pca.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(rC, f, ensure_ascii=False, indent=2)
        print(f"\nExpC results saved to {out_path}")
    
    # Save combined
    if len(results_all) > 1:
        out_path = OUTPUT_DIR / f"phase117b_{args.model}_all_results.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results_all, f, ensure_ascii=False, indent=2)
        print(f"\nAll results saved to {out_path}")
    
    del model
    torch.cuda.empty_cache()
    print("\nGPU memory freed.")


if __name__ == "__main__":
    main()
