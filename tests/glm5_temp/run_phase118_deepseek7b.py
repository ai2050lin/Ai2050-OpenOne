"""
Phase 118 DeepSeek7B Test — Using model_utils.py (verified loading approach)
=============================================================================
参考 model_demo.py 的加载方式: bfloat16, device_map="cpu" -> model.to("cuda")

用法:
  python tests/glm5_temp/run_phase118_deepseek7b.py --exp 1
  python tests/glm5_temp/run_phase118_deepseek7b.py --exp all

每个实验独立加载/释放模型，避免GPU内存溢出
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict

from model_utils import load_model, get_model_info, release_model, get_W_U, MODEL_CONFIGS

OUTPUT_DIR = Path("d:/Ai2050/TransformerLens-Project/tests/glm5_temp")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 100-word list with semantic categories (same as Phase 118)
# ============================================================
WORD_LIST = [
    # Animals (15)
    ("猫", "cat", "animal"), ("狗", "dog", "animal"), ("鸟", "bird", "animal"),
    ("马", "horse", "animal"), ("牛", "cow", "animal"), ("鱼", "fish", "animal"),
    ("兔", "rabbit", "animal"), ("蛇", "snake", "animal"), ("虎", "tiger", "animal"),
    ("象", "elephant", "animal"), ("猴", "monkey", "animal"), ("羊", "sheep", "animal"),
    ("鸡", "chicken", "animal"), ("蜂", "bee", "animal"), ("蝶", "butterfly", "animal"),
    # Fruits (10)
    ("苹果", "apple", "fruit"), ("香蕉", "banana", "fruit"), ("橙子", "orange", "fruit"),
    ("葡萄", "grape", "fruit"), ("西瓜", "watermelon", "fruit"), ("桃子", "peach", "fruit"),
    ("梨", "pear", "fruit"), ("草莓", "strawberry", "fruit"), ("柠檬", "lemon", "fruit"),
    ("芒果", "mango", "fruit"),
    # Furniture/Artifacts (12)
    ("桌子", "table", "artifact"), ("椅子", "chair", "artifact"), ("床", "bed", "artifact"),
    ("门", "door", "artifact"), ("窗户", "window", "artifact"), ("书", "book", "artifact"),
    ("笔", "pen", "artifact"), ("电脑", "computer", "artifact"), ("电话", "phone", "artifact"),
    ("刀", "knife", "artifact"), ("车", "car", "artifact"), ("船", "ship", "artifact"),
    # Nature (10)
    ("太阳", "sun", "nature"), ("月亮", "moon", "nature"), ("星星", "star", "nature"),
    ("天空", "sky", "nature"), ("云", "cloud", "nature"), ("雨", "rain", "nature"),
    ("雪", "snow", "nature"), ("风", "wind", "nature"), ("山", "mountain", "nature"),
    ("河", "river", "nature"),
    # Colors (8)
    ("红色", "red", "color"), ("蓝色", "blue", "color"), ("绿色", "green", "color"),
    ("黄色", "yellow", "color"), ("白色", "white", "color"), ("黑色", "black", "color"),
    ("紫色", "purple", "color"), ("橙色", "orange", "color"),
    # Emotions (10)
    ("快乐", "happy", "emotion"), ("悲伤", "sad", "emotion"), ("愤怒", "angry", "emotion"),
    ("恐惧", "fear", "emotion"), ("惊讶", "surprise", "emotion"), ("爱", "love", "emotion"),
    ("恨", "hate", "emotion"), ("希望", "hope", "emotion"), ("骄傲", "pride", "emotion"),
    ("嫉妒", "jealousy", "emotion"),
    # Actions (12)
    ("跑步", "run", "action"), ("游泳", "swim", "action"), ("飞翔", "fly", "action"),
    ("跳舞", "dance", "action"), ("唱歌", "sing", "action"), ("吃", "eat", "action"),
    ("喝", "drink", "action"), ("睡", "sleep", "action"), ("走", "walk", "action"),
    ("看", "see", "action"), ("写", "write", "action"), ("读", "read", "action"),
    # People/Roles (8)
    ("老师", "teacher", "person"), ("医生", "doctor", "person"), ("工人", "worker", "person"),
    ("农民", "farmer", "person"), ("士兵", "soldier", "person"), ("律师", "lawyer", "person"),
    ("科学家", "scientist", "person"), ("艺术家", "artist", "person"),
    # Size adjectives (8)
    ("大", "big", "adjective"), ("小", "small", "adjective"), ("高", "tall", "adjective"),
    ("矮", "short", "adjective"), ("长", "long", "adjective"), ("快", "fast", "adjective"),
    ("慢", "slow", "adjective"), ("热", "hot", "adjective"),
    # Body parts (7)
    ("手", "hand", "body"), ("脚", "foot", "body"), ("头", "head", "body"),
    ("眼", "eye", "body"), ("耳", "ear", "body"), ("鼻", "nose", "body"),
    ("心", "heart", "body"),
]

CATEGORY_NAMES = sorted(set(w[2] for w in WORD_LIST))
CATEGORY_TO_IDX = {c: i for i, c in enumerate(CATEGORY_NAMES)}

TASK_TEMPLATES = {
    'translate': "将以下中文翻译成英文：{word}",
    'continue': "接下来会发生什么：{word}",
    'define': "请定义以下词语：{word}",
}


# ============================================================
# Core extraction functions
# ============================================================

def extract_residuals(model, tokenizer, device, texts, n_layers):
    """Extract residual stream at all layers for all texts."""
    all_residuals = {l: [] for l in range(n_layers)}

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                          output_hidden_states=True)
            hidden_states = outputs.hidden_states

            for l in range(n_layers):
                h = hidden_states[l + 1]
                all_residuals[l].append(h[0, -1, :].cpu().float().numpy())

    for l in range(n_layers):
        all_residuals[l] = np.stack(all_residuals[l], axis=0)

    return all_residuals


def extract_residuals_with_logits(model, tokenizer, device, texts, n_layers):
    """Extract residuals and logits."""
    all_residuals = {l: [] for l in range(n_layers)}
    all_logits = []

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                          output_hidden_states=True)
            hidden_states = outputs.hidden_states
            logits = outputs.logits[0, -1, :].cpu().float().numpy()
            all_logits.append(logits)

            for l in range(n_layers):
                h = hidden_states[l + 1]
                all_residuals[l].append(h[0, -1, :].cpu().float().numpy())

    for l in range(n_layers):
        all_residuals[l] = np.stack(all_residuals[l], axis=0)
    all_logits = np.stack(all_logits, axis=0)

    return all_residuals, all_logits


def compute_spike_subspace(residuals_task, residuals_base, n_components=25):
    """PCA spike subspace from task-base differences."""
    diffs = residuals_task - residuals_base
    mean_diff = diffs.mean(axis=0)
    diffs_centered = diffs - mean_diff

    U, S, Vt = np.linalg.svd(diffs_centered, full_matrices=False)

    s2 = S ** 2
    pr = (s2.sum()) ** 2 / (s2 ** 2).sum() if (s2 ** 2).sum() > 0 else 0
    total_var = s2.sum()
    concentration = s2[:n_components].sum() / total_var if total_var > 0 else 0

    return {
        'components': Vt,
        'singular_values': S,
        'mean_diff': mean_diff,
        'pr': pr,
        'concentration': concentration,
        'diffs': diffs,
        'diffs_centered': diffs_centered,
    }


# ============================================================
# RSA
# ============================================================

def rsa_correlation(X, Y):
    dX = pdist(X, metric='cosine')
    dY = pdist(Y, metric='cosine')
    if np.std(dX) < 1e-10 or np.std(dY) < 1e-10:
        return 0.0
    return spearmanr(dX, dY).correlation


def neighborhood_preservation(X, Y, k=5):
    n = X.shape[0]
    nn_X = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(X)
    _, idx_X = nn_X.kneighbors(X)
    idx_X = idx_X[:, 1:]
    nn_Y = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(Y)
    _, idx_Y = nn_Y.kneighbors(Y)
    idx_Y = idx_Y[:, 1:]
    overlaps = []
    for i in range(n):
        set_X = set(idx_X[i])
        set_Y = set(idx_Y[i])
        intersection = len(set_X & set_Y)
        union = len(set_X | set_Y)
        overlaps.append(intersection / union if union > 0 else 0)
    return np.mean(overlaps)


def exp1_rsa(task_residuals, n_layers):
    print("\n" + "="*80)
    print("EXP 1: RSA -- Representational Similarity Analysis")
    print("="*80)

    tasks = list(task_residuals.keys())
    results = {
        'rsa_consecutive_layers': {},
        'rsa_across_tasks': {},
        'neighborhood_preservation': {},
        'semantic_rsa': {},
    }

    # 1a: RSA between consecutive layers
    print("\n--- 1a: RSA between consecutive layers ---")
    for task in tasks:
        results['rsa_consecutive_layers'][task] = []
        for l in range(n_layers - 1):
            rsa = rsa_correlation(task_residuals[task][l], task_residuals[task][l+1])
            results['rsa_consecutive_layers'][task].append({
                'layer_from': l, 'layer_to': l+1, 'rsa': rsa
            })
            if l % 4 == 0 or l == n_layers - 2:
                print(f"  {task} L{l}->L{l+1}: RSA={rsa:.4f}")

    # 1b: RSA between tasks
    print("\n--- 1b: RSA between tasks (within layer) ---")
    for l in range(n_layers):
        results['rsa_across_tasks'][l] = {}
        for i, t1 in enumerate(tasks):
            for j, t2 in enumerate(tasks):
                if i < j:
                    rsa = rsa_correlation(task_residuals[t1][l], task_residuals[t2][l])
                    results['rsa_across_tasks'][l][f"{t1}_vs_{t2}"] = rsa
        if l % 4 == 0 or l == n_layers - 1:
            pairs_str = " | ".join([f"{k}={v:.3f}" for k, v in results['rsa_across_tasks'][l].items()])
            print(f"  L{l}: {pairs_str}")

    # 1c: Neighborhood preservation
    print("\n--- 1c: Neighborhood preservation (k=5) ---")
    for task in tasks:
        results['neighborhood_preservation'][task] = []
        for l in range(n_layers - 1):
            np_val = neighborhood_preservation(
                task_residuals[task][l], task_residuals[task][l+1], k=5
            )
            results['neighborhood_preservation'][task].append({
                'layer_from': l, 'layer_to': l+1, 'overlap': np_val
            })
            if l % 4 == 0 or l == n_layers - 2:
                print(f"  {task} L{l}->L{l+1}: overlap={np_val:.4f}")

    # 1d: RSA with semantic distance
    print("\n--- 1d: RSA with semantic structure ---")
    n_words = len(WORD_LIST)
    categories = np.array([CATEGORY_TO_IDX[w[2]] for w in WORD_LIST])
    semantic_dist = np.zeros(n_words * (n_words - 1) // 2)
    idx = 0
    for i in range(n_words):
        for j in range(i+1, n_words):
            semantic_dist[idx] = 0 if categories[i] == categories[j] else 1
            idx += 1

    for task in tasks:
        results['semantic_rsa'][task] = []
        for l in range(n_layers):
            repr_dist = pdist(task_residuals[task][l], metric='cosine')
            if np.std(repr_dist) > 1e-10:
                rsa_sem = spearmanr(repr_dist, semantic_dist).correlation
            else:
                rsa_sem = 0.0
            results['semantic_rsa'][task].append({
                'layer': l, 'rsa_with_semantic': rsa_sem
            })
            if l % 4 == 0 or l == n_layers - 1:
                print(f"  {task} L{l}: RSA(semantic)={rsa_sem:.4f}")

    return results


# ============================================================
# CKA
# ============================================================

def linear_cka(X, Y):
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    K = X @ X.T
    L = Y @ Y.T
    hsic_xy = np.sum(K * L)
    hsic_xx = np.sum(K * K)
    hsic_yy = np.sum(L * L)
    if hsic_xx < 1e-10 or hsic_yy < 1e-10:
        return 0.0
    return hsic_xy / np.sqrt(hsic_xx * hsic_yy)


def exp2_cka(task_residuals, n_layers):
    print("\n" + "="*80)
    print("EXP 2: CKA -- Centered Kernel Alignment")
    print("="*80)

    tasks = list(task_residuals.keys())
    results = {
        'cka_consecutive_layers': {},
        'cka_across_tasks': {},
        'cka_long_range': {},
    }

    # 2a: CKA between consecutive layers
    print("\n--- 2a: CKA between consecutive layers ---")
    for task in tasks:
        results['cka_consecutive_layers'][task] = []
        cka_values = []
        for l in range(n_layers - 1):
            cka = linear_cka(task_residuals[task][l], task_residuals[task][l+1])
            cka_values.append(cka)
            results['cka_consecutive_layers'][task].append({
                'layer_from': l, 'layer_to': l+1, 'cka': cka
            })

        drops = [(i, cka_values[i] - cka_values[i+1]) for i in range(len(cka_values)-1)]
        drops.sort(key=lambda x: x[1])

        print(f"\n  {task} - Top 5 CKA drops:")
        for idx, drop in drops[:5]:
            print(f"    L{idx}->L{idx+1}: drop={drop:.4f} ({cka_values[idx]:.4f}->{cka_values[idx+1]:.4f})")
        print(f"  {task} - CKA range: [{min(cka_values):.4f}, {max(cka_values):.4f}]")

    # 2b: CKA between tasks
    print("\n--- 2b: CKA between tasks (within layer) ---")
    for l in range(n_layers):
        results['cka_across_tasks'][l] = {}
        for i, t1 in enumerate(tasks):
            for j, t2 in enumerate(tasks):
                if i < j:
                    cka = linear_cka(task_residuals[t1][l], task_residuals[t2][l])
                    results['cka_across_tasks'][l][f"{t1}_vs_{t2}"] = cka
        if l % 4 == 0 or l == n_layers - 1:
            pairs_str = " | ".join([f"{k}={v:.3f}" for k, v in results['cka_across_tasks'][l].items()])
            print(f"  L{l}: {pairs_str}")

    # 2c: Long-range CKA
    print("\n--- 2c: Long-range CKA (L0 vs Ll) ---")
    for task in tasks:
        results['cka_long_range'][task] = []
        for l in range(n_layers):
            cka = linear_cka(task_residuals[task][0], task_residuals[task][l])
            results['cka_long_range'][task].append({
                'layer': l, 'cka_with_L0': cka
            })
            if l % 4 == 0 or l == n_layers - 1:
                print(f"  {task} CKA(L0, L{l}): {cka:.4f}")

    return results


# ============================================================
# Exp 3: Causal Direction
# ============================================================

def exp3_causal_direction(W_u, task_residuals, n_layers, d_model):
    print("\n" + "="*80)
    print("EXP 3: Causal Direction at Final Layer")
    print("="*80)

    final_layer = n_layers - 1
    print(f"  W_u shape: {W_u.shape}")

    key_layers = [0, 4, 8, 12, 14, 16, 20, 24, final_layer]
    key_layers = [l for l in key_layers if l < n_layers]

    spike_data = {}
    for layer in key_layers:
        spike = compute_spike_subspace(
            task_residuals['translate'][layer],
            task_residuals['continue'][layer],
            n_components=25
        )
        spike_data[layer] = spike

    results = {'spike_info': {}, 'causal_alignment': {}}

    for layer in spike_data:
        spike = spike_data[layer]
        V_spike = spike['components'][:25, :]
        S_spike = spike['singular_values']

        results['spike_info'][layer] = {
            'pr': spike['pr'],
            'concentration': spike['concentration'],
            'top5_singular_values': S_spike[:5].tolist(),
        }

        W_in_spike = W_u @ V_spike.T
        U_causal, S_causal, Vt_causal = np.linalg.svd(W_in_spike, full_matrices=False)

        causal_dir = V_spike.T @ Vt_causal[0]
        pca_dir = V_spike[0]

        cos_pca_causal = np.abs(np.dot(pca_dir, causal_dir) /
                                (np.linalg.norm(pca_dir) * np.linalg.norm(causal_dir) + 1e-10))

        alignments = []
        for k in range(min(5, len(Vt_causal))):
            ck_dir = V_spike.T @ Vt_causal[k]
            for j in range(min(5, len(V_spike))):
                cos = np.abs(np.dot(V_spike[j], ck_dir) /
                            (np.linalg.norm(V_spike[j]) * np.linalg.norm(ck_dir) + 1e-10))
                alignments.append({'causal_rank': k, 'pca_rank': j, 'cosine': cos})

        logit_change_causal = np.linalg.norm(W_u @ causal_dir)
        logit_change_pca = np.linalg.norm(W_u @ pca_dir)

        np.random.seed(42)
        random_dir = V_spike.T @ np.random.randn(25)
        random_dir /= np.linalg.norm(random_dir)
        logit_change_random = np.linalg.norm(W_u @ random_dir)

        W_u_spike = W_u @ V_spike.T @ V_spike
        W_u_complement = W_u - W_u_spike
        spike_power = np.sum(W_u_spike ** 2) / np.sum(W_u ** 2)
        complement_power = np.sum(W_u_complement ** 2) / np.sum(W_u ** 2)

        results['causal_alignment'][layer] = {
            'cos_pca1_causal1': cos_pca_causal,
            'top5_alignments': alignments,
            'logit_change_causal': logit_change_causal,
            'logit_change_pca': logit_change_pca,
            'logit_change_random': logit_change_random,
            'causal_vs_pca_ratio': logit_change_causal / (logit_change_pca + 1e-10),
            'causal_vs_random_ratio': logit_change_causal / (logit_change_random + 1e-10),
            'spike_power_fraction': spike_power,
            'complement_power_fraction': complement_power,
        }

        print(f"\n  Layer {layer}:")
        print(f"    PR={spike['pr']:.1f}, concentration={spike['concentration']:.4f}")
        print(f"    cos(PCA1, Causal1) = {cos_pca_causal:.4f}")
        print(f"    Logit change: causal={logit_change_causal:.2f}, PCA={logit_change_pca:.2f}, random={logit_change_random:.2f}")
        print(f"    Causal/PCA ratio = {logit_change_causal/(logit_change_pca+1e-10):.4f}")
        print(f"    W_u power: spike={spike_power:.4f}, complement={complement_power:.4f}")

    return results


# ============================================================
# Exp 4: Manifold Dynamics
# ============================================================

def exp4_manifold_dynamics(task_residuals, n_layers):
    print("\n" + "="*80)
    print("EXP 4: Local Manifold Dynamics")
    print("="*80)

    tasks = list(task_residuals.keys())
    n_words = len(WORD_LIST)
    categories = np.array([CATEGORY_TO_IDX[w[2]] for w in WORD_LIST])

    results = {'expansion_ratios': {}, 'semantic_routing': {}, 'attractor_analysis': {}}

    task = 'translate'
    print(f"\n--- 4a: Distance ratio analysis (task={task}) ---")

    dist_matrices = {}
    for l in range(n_layers):
        dist_matrices[l] = pdist(task_residuals[task][l], metric='cosine')

    results['expansion_ratios'][task] = []
    for l in range(n_layers - 1):
        d_l = dist_matrices[l]
        d_l1 = dist_matrices[l + 1]
        mask = d_l > 1e-10
        ratios = np.where(mask, d_l1 / d_l, 1.0)
        mean_ratio = np.mean(ratios[mask]) if mask.any() else 1.0
        std_ratio = np.std(ratios[mask]) if mask.any() else 0.0
        n_expand = np.sum(ratios[mask] > 1.05)
        n_contract = np.sum(ratios[mask] < 0.95)
        n_preserve = mask.sum() - n_expand - n_contract
        results['expansion_ratios'][task].append({
            'layer_from': l, 'layer_to': l+1,
            'mean_ratio': mean_ratio, 'std_ratio': std_ratio,
            'n_expand': int(n_expand), 'n_contract': int(n_contract),
            'n_preserve': int(n_preserve),
        })
        if l % 4 == 0 or l == n_layers - 2:
            print(f"  L{l}->L{l+1}: ratio={mean_ratio:.4f}, expand={n_expand}, contract={n_contract}")

    # 4b: Semantic routing
    print(f"\n--- 4b: Semantic routing ---")
    same_cat_mask = np.zeros(len(dist_matrices[0]), dtype=bool)
    diff_cat_mask = np.zeros(len(dist_matrices[0]), dtype=bool)
    idx = 0
    for i in range(n_words):
        for j in range(i+1, n_words):
            if categories[i] == categories[j]:
                same_cat_mask[idx] = True
            else:
                diff_cat_mask[idx] = True
            idx += 1

    results['semantic_routing'][task] = []
    for l in range(n_layers - 1):
        d_l = dist_matrices[l]
        d_l1 = dist_matrices[l + 1]
        mask_safe = d_l > 1e-10
        ratios = np.where(mask_safe, d_l1 / d_l, 1.0)
        same_mask = same_cat_mask & mask_safe
        diff_mask = diff_cat_mask & mask_safe
        same_ratio = np.mean(ratios[same_mask]) if same_mask.any() else 1.0
        diff_ratio = np.mean(ratios[diff_mask]) if diff_mask.any() else 1.0
        routing_effect = same_ratio - diff_ratio
        results['semantic_routing'][task].append({
            'layer_from': l, 'layer_to': l+1,
            'same_category_ratio': same_ratio,
            'diff_category_ratio': diff_ratio,
            'routing_effect': routing_effect,
        })
        if l % 4 == 0 or l == n_layers - 2:
            print(f"  L{l}->L{l+1}: same={same_ratio:.4f}, diff={diff_ratio:.4f}, routing={routing_effect:.4f}")

    # 4c: Attractor analysis
    print(f"\n--- 4c: Attractor candidates ---")
    consistent_contractions = defaultdict(int)
    for l in range(n_layers - 1):
        d_l = dist_matrices[l]
        d_l1 = dist_matrices[l + 1]
        mask_safe = d_l > 1e-10
        ratios = np.where(mask_safe, d_l1 / d_l, 1.0)
        idx = 0
        for i in range(n_words):
            for j in range(i+1, n_words):
                if ratios[idx] < 0.9:
                    consistent_contractions[(i, j)] += 1
                idx += 1

    top_attractors = sorted(consistent_contractions.items(), key=lambda x: x[1], reverse=True)[:20]
    results['attractor_analysis']['top_attractor_pairs'] = []
    print("  Top 10 attractor pairs:")
    for (i, j), count in top_attractors[:10]:
        w1, w2 = WORD_LIST[i], WORD_LIST[j]
        same_cat = w1[2] == w2[2]
        results['attractor_analysis']['top_attractor_pairs'].append({
            'word1': w1[0], 'word1_en': w1[1], 'word1_cat': w1[2],
            'word2': w2[0], 'word2_en': w2[1], 'word2_cat': w2[2],
            'same_category': same_cat, 'contraction_layers': count,
        })
        print(f"    {w1[0]}({w1[2]}) <-> {w2[0]}({w2[2]}): {count}/{n_layers-1} layers, same_cat={same_cat}")

    # 4d: Cross-task comparison
    print(f"\n--- 4d: Cross-task manifold dynamics ---")
    for task2 in tasks:
        if task2 == task:
            continue
        dist_matrices2 = {}
        for l in range(n_layers):
            dist_matrices2[l] = pdist(task_residuals[task2][l], metric='cosine')
        print(f"\n  Comparing {task} vs {task2}:")
        for l in [0, 4, 8, 12, 14, 20, n_layers-2]:
            if l >= n_layers - 1:
                continue
            d1_l = dist_matrices[l]
            d1_l1 = dist_matrices[l + 1]
            d2_l = dist_matrices2[l]
            d2_l1 = dist_matrices2[l + 1]
            mask1 = d1_l > 1e-10
            mask2 = d2_l > 1e-10
            ratios1 = np.where(mask1, d1_l1 / d1_l, 1.0)
            ratios2 = np.where(mask2, d2_l1 / d2_l, 1.0)
            if np.std(ratios1[mask1 & mask2]) > 1e-10 and np.std(ratios2[mask1 & mask2]) > 1e-10:
                corr = spearmanr(ratios1[mask1 & mask2], ratios2[mask1 & mask2]).correlation
            else:
                corr = 0.0
            print(f"    L{l}->L{l+1}: {task}={np.mean(ratios1[mask1]):.4f}, "
                  f"{task2}={np.mean(ratios2[mask2]):.4f}, ratio_corr={corr:.4f}")

    return results


# ============================================================
# Exp 5: Complement Nonlinear Probe
# ============================================================

def exp5_complement_probe(task_residuals, n_layers, d_model):
    print("\n" + "="*80)
    print("EXP 5: Complement Nonlinear Probe")
    print("="*80)

    from sklearn.model_selection import cross_val_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    categories = np.array([CATEGORY_TO_IDX[w[2]] for w in WORD_LIST])
    n_classes = len(CATEGORY_NAMES)

    task = 'translate'
    base_task = 'continue'

    results = {
        'spike_knn': {}, 'complement_knn': {}, 'random_knn': {},
        'spike_mlp': {}, 'complement_mlp': {}, 'random_mlp': {},
    }

    key_layers = [0, 4, 8, 12, 14, 16, 20, 24, 27, n_layers-1]
    key_layers = [l for l in key_layers if l < n_layers]

    for l in key_layers:
        spike = compute_spike_subspace(
            task_residuals[task][l], task_residuals[base_task][l], n_components=25
        )
        V_spike = spike['components'][:25, :]
        spike_dim = min(25, V_spike.shape[0])

        h_task = task_residuals[task][l]
        h_spike = h_task @ V_spike.T @ V_spike
        h_complement = h_task - h_spike

        h_comp_centered = h_complement - h_complement.mean(axis=0)
        U_comp, S_comp, Vt_comp = np.linalg.svd(h_comp_centered, full_matrices=False)
        h_comp_reduced = U_comp[:, :50] * S_comp[:50]

        np.random.seed(42)
        R = np.random.randn(d_model, 50)
        R = R / np.linalg.norm(R, axis=0, keepdims=True)
        h_random = h_task @ R

        h_spike_repr = h_task @ V_spike[:spike_dim, :].T

        # kNN
        for name, h_repr in [('spike', h_spike_repr), ('complement', h_comp_reduced), ('random', h_random)]:
            knn = NearestNeighbors(n_neighbors=4, metric='cosine').fit(h_repr)
            _, idx = knn.kneighbors(h_repr)
            idx = idx[:, 1:]
            pred = np.array([np.bincount(categories[idx[i]], minlength=n_classes).argmax()
                           for i in range(len(categories))])
            acc = np.mean(pred == categories)
            results[f'{name}_knn'][l] = acc

        # MLP
        for name, h_repr in [('spike', h_spike_repr), ('complement', h_comp_reduced), ('random', h_random)]:
            scaler = StandardScaler()
            h_scaled = scaler.fit_transform(h_repr)
            mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',
                              max_iter=500, random_state=42, early_stopping=True)
            try:
                scores = cross_val_score(mlp, h_scaled, categories, cv=5, scoring='accuracy')
                results[f'{name}_mlp'][l] = {'mean': scores.mean(), 'std': scores.std()}
            except Exception as e:
                results[f'{name}_mlp'][l] = {'mean': 0.0, 'std': 0.0, 'error': str(e)}

        print(f"\n  Layer {l} (PR={spike['pr']:.1f}):")
        print(f"    kNN:  spike={results['spike_knn'][l]:.3f}, comp={results['complement_knn'][l]:.3f}, rand={results['random_knn'][l]:.3f}")
        print(f"    MLP:  spike={results['spike_mlp'][l]['mean']:.3f}, comp={results['complement_mlp'][l]['mean']:.3f}, rand={results['random_mlp'][l]['mean']:.3f}")

    return results


# ============================================================
# Serialization helper
# ============================================================

def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save_results(results, name):
    filepath = OUTPUT_DIR / f"{name}.json"
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(convert_to_serializable(results), f, ensure_ascii=False, indent=2, default=str)
    print(f"  Saved: {filepath}")


# ============================================================
# Main: run each experiment with separate model load/release
# ============================================================

def extract_all_residuals(model_name='deepseek7b'):
    """Load model, extract all residuals for all tasks, then release."""
    print(f"\n{'='*60}")
    print(f"Loading {model_name} for residual extraction...")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    print(f"  {info.model_class}, {n_layers} layers, d_model={d_model}")

    # Extract residuals for all tasks
    task_residuals = {}
    for task_name, template in TASK_TEMPLATES.items():
        texts = [template.format(word=w[0]) for w in WORD_LIST]
        print(f"  Extracting {task_name} ({len(texts)} texts)...")
        task_residuals[task_name] = extract_residuals(model, tokenizer, device, texts, n_layers)

    # Also get W_U for Exp 3
    W_u = get_W_U(model)
    print(f"  W_U shape: {W_u.shape}")

    # Release model
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()

    return task_residuals, W_u, n_layers, d_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='deepseek7b', choices=['deepseek7b', 'qwen3'])
    parser.add_argument('--exp', type=str, default='all', choices=['all', '1', '2', '3', '4', '5'])
    args = parser.parse_args()

    model_name = args.model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'#'*80}")
    print(f"Phase 118: Invariant Structure -- {model_name}")
    print(f"Using model_utils.py (bfloat16, CPU->CUDA approach)")
    print(f"Timestamp: {timestamp}")
    print(f"{'#'*80}")

    # Step 1: Extract all residuals (one model load)
    task_residuals, W_u, n_layers, d_model = extract_all_residuals(model_name)

    all_results = {
        'model': model_name,
        'n_words': len(WORD_LIST),
        'categories': CATEGORY_NAMES,
        'n_layers': n_layers,
        'd_model': d_model,
        'timestamp': timestamp,
    }

    # Step 2: Run experiments (no model needed, just residuals)
    if args.exp in ['all', '1']:
        print(f"\n>>> Running Exp 1: RSA <<<")
        results = exp1_rsa(task_residuals, n_layers)
        all_results['exp1_rsa'] = results
        save_results(all_results, f"phase118_exp1_{model_name}_rsa")

    if args.exp in ['all', '2']:
        print(f"\n>>> Running Exp 2: CKA <<<")
        results = exp2_cka(task_residuals, n_layers)
        all_results['exp2_cka'] = results
        save_results(all_results, f"phase118_exp2_{model_name}_cka")

    if args.exp in ['all', '3']:
        print(f"\n>>> Running Exp 3: Causal Direction <<<")
        results = exp3_causal_direction(W_u, task_residuals, n_layers, d_model)
        all_results['exp3_causal'] = results
        save_results(all_results, f"phase118_exp3_{model_name}_causal")

    if args.exp in ['all', '4']:
        print(f"\n>>> Running Exp 4: Manifold Dynamics <<<")
        results = exp4_manifold_dynamics(task_residuals, n_layers)
        all_results['exp4_manifold'] = results
        save_results(all_results, f"phase118_exp4_{model_name}_manifold")

    if args.exp in ['all', '5']:
        print(f"\n>>> Running Exp 5: Complement Probe <<<")
        results = exp5_complement_probe(task_residuals, n_layers, d_model)
        all_results['exp5_complement'] = results
        save_results(all_results, f"phase118_exp5_{model_name}_complement")

    # Save combined
    save_results(all_results, f"phase118_{model_name}_all_results")

    print(f"\n{'#'*80}")
    print(f"PHASE 118 {model_name.upper()} COMPLETE")
    print(f"{'#'*80}")


if __name__ == "__main__":
    main()
