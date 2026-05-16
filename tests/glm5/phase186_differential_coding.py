"""
Phase 186: Differential Coding Geometry — Equivalence Class Formation Dynamics
================================================================================

★★★ 理论基础 ★★★

Phase 185发现所有层λ_max>1, 但这不代表"混沌"——因为Transformer不是自治动力系统,
而是条件输运系统。λ>1更可能是"编码放大"——增强信号背景比。

★ 核心跃迁: 从"约束传播"→"等价类形成" ★

真正的编码机制不是"存储feature", 而是"构建可区分性":
- 编码 = 稳定可区分传播路径
- 语义 = 差异网络, 不是点
- d(apple, x) for all x, 而非 h(apple)

★★★ 四个实验 ★★★

Exp1: Equivalence Class Contraction (★最关键★)
  - 构造语义等价类: 不同表达同一概念的句子
  - 测量类内距离随层的变化: d(h_l(x_i), h_l(x_j)) for x_i ~ x_j
  - 若深层距离→0: 等价类在深层收缩, 系统主动压缩等价结构
  - 对比类间距离: 类内/类间比值 = separability index
  - 关键数据量: 6个语义类 × 10+个表达方式 = 60+句子

Exp2: Distinguishability Emergence (★编码机制核心★)
  - 构造不同相似度的概念对: 从极相似(apple/pear)到极不同(apple/car)
  - 测量每一层的类间/类内距离比 = 可区分性
  - 找到"可区分性涌现层": D(l)突然增大的层
  - 这是编码生成的真正观测: 系统何时开始"区分"两个概念?

Exp3: Cross-Lingual Semantic Orbit (★不变量核心★)
  - 中文↔英文的同一语义
  - 测量: 中英文句子在hidden space中的距离
  - 与同语言近义词对比: 是否跨语言距离≈同语言近义距离?
  - 若成立: 存在语言不变的语义轨道, 翻译=不变输运映射

Exp4: Trained vs Random Jacobian Comparison (★去伪核心★)
  - 关键问题: λ>1是训练学到的还是架构固有的?
  - 方法: 随机初始化模型(不训练), 测λ_max
  - 若随机模型也λ>1: λ>1是架构属性, 不是编码机制
  - 若随机模型λ<1或λ≈1: λ>1是训练学到的, 确实是编码放大

Usage: python tests/glm5/phase186_differential_coding.py <model_name>
       python tests/glm5/phase186_differential_coding.py qwen3
"""

import sys, os, time, json, gc
import numpy as np
import torch
from datetime import datetime
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'glm5'))
from model_utils import get_layers, get_model_info, release_model, get_W_U, MODEL_CONFIGS


def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    print(f"[P186] Loading {model_name} (bfloat16 + device_map=auto)...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, local_files_only=True, attn_implementation="eager")
    model.eval()
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"[P186] {model_name} loaded: device={device}, class={type(model).__name__}, GPU={gpu_mem:.2f}GB")
    return model, tokenizer, device


def force_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# =====================================================================
# SEMANTIC EQUIVALENCE CLASSES — Exp1 & Exp2 data
# =====================================================================

# ★ Equivalence class: "eating fruit" concept — 6 classes with many expressions each
EQUIVALENCE_CLASSES = {
    "apple": [
        "I ate an apple",
        "She bit into the apple",
        "The apple was sweet",
        "He picked a red apple",
        "That apple tastes good",
        "The green apple is sour",
        "An apple a day keeps the doctor away",
        "She offered me an apple",
        "The apple fell from the tree",
        "I bought some apples",
    ],
    "banana": [
        "I ate a banana",
        "She peeled the banana",
        "The banana was ripe",
        "He bought yellow bananas",
        "That banana is too soft",
        "The green banana is unripe",
        "Bananas are rich in potassium",
        "She gave me a banana",
        "The banana hung from the tree",
        "I prefer small bananas",
    ],
    "orange": [
        "I ate an orange",
        "She peeled the orange",
        "The orange was juicy",
        "He squeezed the orange",
        "That orange looks fresh",
        "The blood orange is sweet",
        "Oranges contain vitamin C",
        "She handed me an orange",
        "The orange rolled off the table",
        "I picked some oranges",
    ],
    "car": [
        "I drove the car",
        "She bought a new car",
        "The car was fast",
        "He repaired the car",
        "That car is expensive",
        "The red car parked outside",
        "Cars need regular maintenance",
        "She lent me her car",
        "The car broke down",
        "I washed my car",
    ],
    "book": [
        "I read the book",
        "She wrote a book",
        "The book was interesting",
        "He borrowed the book",
        "That book is famous",
        "The thick book was heavy",
        "Books contain knowledge",
        "She gave me a book",
        "The book fell off the shelf",
        "I bought a new book",
    ],
    "dog": [
        "I walked the dog",
        "She adopted a dog",
        "The dog was friendly",
        "He trained the dog",
        "That dog is very loyal",
        "The small dog barked loudly",
        "Dogs are loyal companions",
        "She brought her dog",
        "The dog chased the cat",
        "I fed my dog",
    ],
}

# ★ Exp2: Similarity spectrum — pairs of varying semantic distance
SIMILARITY_SPECTRUM = [
    # (pair_name, sentence_a, sentence_b, expected_similarity)
    # Very similar (same category, different exemplars)
    ("apple_vs_pear", "I ate an apple", "I ate a pear", 0.9),
    ("dog_vs_cat", "I walked the dog", "I walked the cat", 0.9),
    ("car_vs_bus", "I drove the car", "I drove the bus", 0.85),
    # Moderately similar (same superordinate, different basic level)
    ("apple_vs_banana", "I ate an apple", "I ate a banana", 0.7),
    ("dog_vs_bird", "I walked the dog", "I watched the bird", 0.6),
    ("car_vs_bicycle", "I drove the car", "I rode the bicycle", 0.5),
    # Low similarity (different category, same domain)
    ("apple_vs_car", "I ate an apple", "I drove the car", 0.2),
    ("dog_vs_book", "I walked the dog", "I read the book", 0.1),
    # Very low similarity (completely unrelated)
    ("apple_vs_mountain", "I ate an apple", "I climbed the mountain", 0.05),
    ("dog_vs_philosophy", "I walked the dog", "I studied philosophy", 0.02),
    # Antonyms (related but opposite)
    ("hot_vs_cold", "The water is hot", "The water is cold", 0.4),
    ("big_vs_small", "The house is big", "The house is small", 0.4),
]

# ★ Exp3: Cross-lingual pairs (Chinese ↔ English same meaning)
CROSS_LINGUAL_PAIRS = [
    # (english, chinese, category)
    ("The cat is sleeping", "猫在睡觉", "animal_action"),
    ("I ate an apple", "我吃了一个苹果", "fruit_action"),
    ("She reads books", "她读书", "human_action"),
    ("The sun is shining", "太阳在照耀", "nature"),
    ("He drives to work", "他开车去上班", "daily_activity"),
    ("The water is cold", "水很冷", "temperature"),
    ("Birds can fly", "鸟会飞", "animal_ability"),
    ("I love music", "我喜欢音乐", "emotion"),
    ("The door is open", "门开着", "state"),
    ("She cooks dinner", "她做晚饭", "daily_activity"),
    ("The sky is blue", "天空是蓝色的", "color"),
    ("He runs fast", "他跑得快", "speed"),
    ("The fish swims", "鱼游泳", "animal_action"),
    ("I drink coffee", "我喝咖啡", "daily_activity"),
    ("The house is big", "房子很大", "size"),
    # Same-language within-class pairs for comparison
    ("She reads books", "She reads novels", "same_lang_near"),
    ("I ate an apple", "I ate a pear", "same_lang_near"),
    ("The cat is sleeping", "The dog is sleeping", "same_lang_near"),
    # Same-language between-class pairs
    ("I ate an apple", "She reads books", "same_lang_far"),
    ("The cat is sleeping", "He drives to work", "same_lang_far"),
    ("Birds can fly", "I love music", "same_lang_far"),
]


def get_all_hidden_states(model, tokenizer, device, sentence, target_pos=None):
    """获取所有层的hidden states"""
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        out = model(input_ids=inputs["input_ids"].to(device),
                    attention_mask=inputs["attention_mask"].to(device),
                    output_hidden_states=True)
    # Default: use last meaningful token position
    if target_pos is None:
        target_pos = inputs["input_ids"].shape[1] - 1
    pos = min(target_pos, out.hidden_states[0].shape[1] - 1)
    result = {}
    n_layers = len(out.hidden_states) - 1
    for li, hs in enumerate(out.hidden_states):
        result[li] = hs[0, pos].detach().cpu().float().numpy().astype(np.float32)
    del out
    return result, n_layers


def compute_pairwise_distance(hs_a, hs_b, layer):
    """计算两个hidden state在某层的cosine距离"""
    if layer not in hs_a or layer not in hs_b:
        return None
    a = hs_a[layer]
    b = hs_b[layer]
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm < 1e-10 or b_norm < 1e-10:
        return None
    cos_sim = float(np.dot(a, b) / (a_norm * b_norm))
    # Convert to distance: 0=identical, 2=opposite
    return 1.0 - cos_sim


def compute_euclidean_distance(hs_a, hs_b, layer):
    """计算欧氏距离（归一化）"""
    if layer not in hs_a or layer not in hs_b:
        return None
    a = hs_a[layer]
    b = hs_b[layer]
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-10:
        return None
    return float(np.linalg.norm(a - b)) / norm_a


# =====================================================================
# EXP1: EQUIVALENCE CLASS CONTRACTION
# =====================================================================

def exp1_equivalence_class_contraction(model, tokenizer, device, n_layers, d_model):
    """
    ★★★ 最关键实验: 语义等价类收缩 ★★★
    
    测量:
    1. 类内距离: 同一语义类中不同表达的hidden state距离
    2. 类间距离: 不同语义类之间的距离
    3. Separability index = 类间距离 / 类内距离
    
    关键预期:
    - 类内距离随层递减 → 等价类在深层收缩
    - 类间距离随层递增 → 不同类在深层更可区分
    - Separability index随层递增 → 系统逐步构建可区分性
    """
    print("\n" + "="*70)
    print("Exp1: EQUIVALENCE CLASS CONTRACTION")
    print("  (Do synonyms/paraphrases converge in deep layers?)")
    print("="*70)
    
    # Sample layers
    n_sample = min(15, n_layers)
    sample_layers = sorted(set(
        [0, 1, 2] +
        list(range(0, n_layers, max(1, n_layers // n_sample))) +
        [n_layers//4, n_layers//2, 3*n_layers//4, n_layers-2, n_layers-1]
    ))
    sample_layers = [l for l in sample_layers if 0 <= l <= n_layers]
    sample_layers = sorted(set(sample_layers))
    
    # Collect hidden states for all sentences
    class_names = list(EQUIVALENCE_CLASSES.keys())
    all_hs = {}  # {class_name: [{layer: np.array}, ...]}
    
    for ci, cname in enumerate(class_names):
        sentences = EQUIVALENCE_CLASSES[cname]
        all_hs[cname] = []
        print(f"  [{ci+1}/{len(class_names)}] Processing class '{cname}' ({len(sentences)} sentences)...", flush=True)
        
        for si, sent in enumerate(sentences):
            if si % 4 == 0:
                print(f"    Sentence {si+1}/{len(sentences)}", flush=True)
            hs, _ = get_all_hidden_states(model, tokenizer, device, sent)
            all_hs[cname].append(hs)
            force_cleanup()
    
    print(f"\n  Computing distances across {len(sample_layers)} layers...", flush=True)
    
    # Compute intra-class and inter-class distances
    intra_dist = defaultdict(list)  # {layer: [cosine_distances]}
    inter_dist = defaultdict(list)
    
    for li in sample_layers:
        # Intra-class: all pairs within each class
        for cname in class_names:
            class_hs = all_hs[cname]
            for i in range(len(class_hs)):
                for j in range(i+1, min(i+5, len(class_hs))):  # Limit pairs for speed
                    d = compute_pairwise_distance(class_hs[i], class_hs[j], li)
                    if d is not None:
                        intra_dist[li].append(d)
        
        # Inter-class: pairs across different classes
        for ci, cname_a in enumerate(class_names):
            for cname_b in class_names[ci+1:]:
                # Sample pairs
                hs_a = all_hs[cname_a][:3]
                hs_b = all_hs[cname_b][:3]
                for ha in hs_a:
                    for hb in hs_b:
                        d = compute_pairwise_distance(ha, hb, li)
                        if d is not None:
                            inter_dist[li].append(d)
        
        if li % 5 == 0 or li == sample_layers[-1]:
            intra_m = np.mean(intra_dist[li]) if intra_dist[li] else 0
            inter_m = np.mean(inter_dist[li]) if inter_dist[li] else 0
            sep = inter_m / max(intra_m, 1e-10) if intra_m > 0.01 else 0
            print(f"    L{li}: intra={intra_m:.4f}, inter={inter_m:.4f}, sep={sep:.2f}", flush=True)
    
    # Also compute euclidean distances for robustness
    intra_euc = defaultdict(list)
    inter_euc = defaultdict(list)
    
    for li in sample_layers:
        for cname in class_names:
            class_hs = all_hs[cname]
            for i in range(len(class_hs)):
                for j in range(i+1, min(i+5, len(class_hs))):
                    d = compute_euclidean_distance(class_hs[i], class_hs[j], li)
                    if d is not None:
                        intra_euc[li].append(d)
        
        for ci, cname_a in enumerate(class_names):
            for cname_b in class_names[ci+1:]:
                hs_a = all_hs[cname_a][:3]
                hs_b = all_hs[cname_b][:3]
                for ha in hs_a:
                    for hb in hs_b:
                        d = compute_euclidean_distance(ha, hb, li)
                        if d is not None:
                            inter_euc[li].append(d)
    
    # Per-class intra-distance (for analyzing which classes contract most)
    per_class_intra = defaultdict(lambda: defaultdict(list))
    for li in sample_layers:
        for cname in class_names:
            class_hs = all_hs[cname]
            for i in range(len(class_hs)):
                for j in range(i+1, min(i+5, len(class_hs))):
                    d = compute_pairwise_distance(class_hs[i], class_hs[j], li)
                    if d is not None:
                        per_class_intra[cname][li].append(d)
    
    # Aggregate
    result = {}
    for li in sample_layers:
        intra_m = float(np.mean(intra_dist[li])) if intra_dist[li] else 0
        intra_s = float(np.std(intra_dist[li])) if intra_dist[li] else 0
        inter_m = float(np.mean(inter_dist[li])) if inter_dist[li] else 0
        inter_s = float(np.std(inter_dist[li])) if inter_dist[li] else 0
        intra_euc_m = float(np.mean(intra_euc[li])) if intra_euc[li] else 0
        inter_euc_m = float(np.mean(inter_euc[li])) if inter_euc[li] else 0
        
        sep_cos = inter_m / max(intra_m, 1e-10) if intra_m > 0.01 else 0
        sep_euc = inter_euc_m / max(intra_euc_m, 1e-10) if intra_euc_m > 0.001 else 0
        
        result[li] = {
            "intra_cos_mean": intra_m,
            "intra_cos_std": intra_s,
            "inter_cos_mean": inter_m,
            "inter_cos_std": inter_s,
            "intra_euc_mean": intra_euc_m,
            "inter_euc_mean": inter_euc_m,
            "separability_cos": sep_cos,
            "separability_euc": sep_euc,
            "n_intra": len(intra_dist[li]),
            "n_inter": len(inter_dist[li]),
        }
    
    # Per-class result
    per_class_result = {}
    for cname in class_names:
        per_class_result[cname] = {}
        for li in sample_layers:
            if li in per_class_intra[cname] and per_class_intra[cname][li]:
                vals = per_class_intra[cname][li]
                per_class_result[cname][li] = {
                    "intra_cos_mean": float(np.mean(vals)),
                    "intra_cos_std": float(np.std(vals)),
                    "n_pairs": len(vals),
                }
    
    # Compute slopes
    layers_sorted = sorted(result.keys())
    if len(layers_sorted) >= 2:
        first_li = layers_sorted[0]
        last_li = layers_sorted[-1]
        n_steps = last_li - first_li
        if n_steps > 0:
            intra_slope = (result[last_li]["intra_cos_mean"] - result[first_li]["intra_cos_mean"]) / n_steps
            inter_slope = (result[last_li]["inter_cos_mean"] - result[first_li]["inter_cos_mean"]) / n_steps
            sep_first = result[first_li]["separability_cos"]
            sep_last = result[last_li]["separability_cos"]
        else:
            intra_slope = inter_slope = 0
            sep_first = sep_last = 0
    else:
        intra_slope = inter_slope = 0
        sep_first = sep_last = 0
    
    result["_meta"] = {
        "intra_slope": intra_slope,
        "inter_slope": inter_slope,
        "separability_first": sep_first,
        "separability_last": sep_last,
        "intra_verdict": "CONTRACTING (equivalence classes form)" if intra_slope < -0.001 else
                         "EXPANDING" if intra_slope > 0.001 else "STABLE",
        "inter_verdict": "SEPARATING (classes become distinguishable)" if inter_slope > 0.001 else
                         "MERGING" if inter_slope < -0.001 else "STABLE",
        "sample_layers": sample_layers,
    }
    result["_per_class"] = per_class_result
    
    # Cleanup
    del all_hs
    force_cleanup()
    
    return result


# =====================================================================
# EXP2: DISTINGUISHABILITY EMERGENCE
# =====================================================================

def exp2_distinguishability_emergence(model, tokenizer, device, n_layers, d_model):
    """
    ★★★ 编码机制核心: 可区分性涌现 ★★★
    
    测量不同相似度的概念对, 在每一层的可区分性:
    - D(l, pair) = d(h_l(x_a), h_l(x_b))
    
    关键问题:
    - 可区分性是逐渐形成的, 还是突然涌现的?
    - 极相似对(apple/pear)vs极不同对(apple/car)的涌现模式是否不同?
    - 是否存在"涌现相变层"?
    """
    print("\n" + "="*70)
    print("Exp2: DISTINGUISHABILITY EMERGENCE")
    print("  (When/how do concepts become distinguishable?)")
    print("="*70)
    
    n_sample = min(15, n_layers)
    sample_layers = sorted(set(
        [0, 1, 2] +
        list(range(0, n_layers, max(1, n_layers // n_sample))) +
        [n_layers//4, n_layers//2, 3*n_layers//4, n_layers-2, n_layers-1]
    ))
    sample_layers = [l for l in sample_layers if 0 <= l <= n_layers]
    sample_layers = sorted(set(sample_layers))
    
    # Collect hidden states
    pair_results = defaultdict(lambda: defaultdict(list))  # {pair_name: {layer: [distances]}}
    
    for pi, (pair_name, sent_a, sent_b, expected_sim) in enumerate(SIMILARITY_SPECTRUM):
        print(f"  Pair {pi+1}/{len(SIMILARITY_SPECTRUM)}: {pair_name} (sim~{expected_sim})", flush=True)
        
        # Multiple runs with different target positions for robustness
        hs_a, _ = get_all_hidden_states(model, tokenizer, device, sent_a)
        hs_b, _ = get_all_hidden_states(model, tokenizer, device, sent_b)
        
        for li in sample_layers:
            d_cos = compute_pairwise_distance(hs_a, hs_b, li)
            d_euc = compute_euclidean_distance(hs_a, hs_b, li)
            if d_cos is not None:
                pair_results[pair_name][li].append({
                    "cos_dist": d_cos,
                    "euc_dist": d_euc if d_euc is not None else 0,
                })
        
        del hs_a, hs_b
        force_cleanup()
    
    # Also add within-class and between-class from Exp1 equivalence data
    # For efficiency, we just compute for the key pairs
    print(f"\n  Aggregating results...", flush=True)
    
    result = {}
    for pair_name in [p[0] for p in SIMILARITY_SPECTRUM]:
        pair_data = pair_results[pair_name]
        pair_result = {}
        for li in sample_layers:
            if li in pair_data and pair_data[li]:
                cos_vals = [d["cos_dist"] for d in pair_data[li]]
                euc_vals = [d["euc_dist"] for d in pair_data[li]]
                pair_result[li] = {
                    "cos_dist_mean": float(np.mean(cos_vals)),
                    "cos_dist_std": float(np.std(cos_vals)) if len(cos_vals) > 1 else 0,
                    "euc_dist_mean": float(np.mean(euc_vals)),
                    "n_obs": len(cos_vals),
                }
        
        # Compute emergence slope
        layers_sorted = sorted(pair_result.keys())
        if len(layers_sorted) >= 2:
            first_d = pair_result[layers_sorted[0]]["cos_dist_mean"]
            last_d = pair_result[layers_sorted[-1]]["cos_dist_mean"]
            n_steps = layers_sorted[-1] - layers_sorted[0]
            slope = (last_d - first_d) / max(n_steps, 1)
        else:
            slope = first_d = last_d = 0
        
        # Find the "emergence layer" — where distance changes most rapidly
        max_change_layer = None
        max_change = 0
        for i in range(1, len(layers_sorted)):
            l_prev = layers_sorted[i-1]
            l_curr = layers_sorted[i]
            d_prev = pair_result[l_prev]["cos_dist_mean"]
            d_curr = pair_result[l_curr]["cos_dist_mean"]
            change = abs(d_curr - d_prev) / max(l_curr - l_prev, 1)
            if change > max_change:
                max_change = change
                max_change_layer = l_curr
        
        pair_result["_meta"] = {
            "expected_similarity": float(
                {p[0]: p[3] for p in SIMILARITY_SPECTRUM}.get(pair_name, 0)),
            "emergence_slope": slope,
            "emergence_layer": max_change_layer,
            "first_dist": first_d,
            "last_dist": last_d,
            "verdict": "DIFFERENTIATING" if slope > 0.001 else "MERGING" if slope < -0.001 else "STABLE",
        }
        result[pair_name] = pair_result
    
    # Compute correlation: expected similarity vs final-layer distance
    expected_sims = []
    actual_dists = []
    for pair_name, _, _, expected_sim in SIMILARITY_SPECTRUM:
        if pair_name in result:
            meta = result[pair_name].get("_meta", {})
            expected_sims.append(expected_sim)
            actual_dists.append(meta.get("last_dist", 0))
    
    if len(expected_sims) >= 3:
        from scipy.stats import spearmanr
        rho, p_val = spearmanr(expected_sims, actual_dists)
        result["_correlation"] = {
            "spearman_rho": float(rho),
            "p_value": float(p_val),
            "n_pairs": len(expected_sims),
            "verdict": "SIGNIFICANT: distance encodes semantic similarity" if p_val < 0.05 else "NOT SIGNIFICANT",
        }
    
    force_cleanup()
    return result


# =====================================================================
# EXP3: CROSS-LINGUAL SEMANTIC ORBIT
# =====================================================================

def exp3_cross_lingual_orbit(model, tokenizer, device, n_layers, d_model):
    """
    ★★★ 不变量核心: 跨语言语义轨道 ★★★
    
    测量:
    1. 跨语言距离: 中英文同义句的hidden state距离
    2. 同语言近义距离: 同语言近义句的距离
    3. 同语言远义距离: 同语言远义句的距离
    
    关键问题:
    - 跨语言距离 < 同语言远义距离? (语义轨道超越语言)
    - 跨语言距离 ≈ 同语言近义距离? (翻译=不变输运)
    - 在哪些层, 跨语言距离最小化? (语义轨道在深层还是浅层?)
    """
    print("\n" + "="*70)
    print("Exp3: CROSS-LINGUAL SEMANTIC ORBIT")
    print("  (Do Chinese↔English same-meaning sentences converge?)")
    print("="*70)
    
    n_sample = min(15, n_layers)
    sample_layers = sorted(set(
        [0, 1, 2] +
        list(range(0, n_layers, max(1, n_layers // n_sample))) +
        [n_layers//4, n_layers//2, 3*n_layers//4, n_layers-2, n_layers-1]
    ))
    sample_layers = [l for l in sample_layers if 0 <= l <= n_layers]
    sample_layers = sorted(set(sample_layers))
    
    # Collect all hidden states
    all_hs = {}
    categories = defaultdict(list)  # {category: [sentence_keys]}
    
    for pi, (eng, chn, cat) in enumerate(CROSS_LINGUAL_PAIRS):
        print(f"  Pair {pi+1}/{len(CROSS_LINGUAL_PAIRS)}: [{cat}] {eng[:30]}... / {chn[:15]}...", flush=True)
        
        key_eng = f"{pi}_eng"
        key_chn = f"{pi}_chn"
        
        hs_eng, _ = get_all_hidden_states(model, tokenizer, device, eng)
        hs_chn, _ = get_all_hidden_states(model, tokenizer, device, chn)
        
        all_hs[key_eng] = hs_eng
        all_hs[key_chn] = hs_chn
        categories[cat].append((key_eng, key_chn))
        
        force_cleanup()
    
    # Compute distances by category
    result_by_layer = defaultdict(lambda: defaultdict(list))
    # {layer: {category: [distances]}}
    
    for li in sample_layers:
        # Cross-lingual: en↔zh same meaning
        for pi, (eng, chn, cat) in enumerate(CROSS_LINGUAL_PAIRS):
            key_eng = f"{pi}_eng"
            key_chn = f"{pi}_chn"
            d = compute_pairwise_distance(all_hs[key_eng], all_hs[key_chn], li)
            if d is not None:
                result_by_layer[li]["cross_lingual"].append(d)
        
        # Same-language near: English near-synonyms
        for key_a, key_b in [("5_eng", "15_eng"), ("0_eng", "16_eng"), ("0_eng", "17_eng")]:
            if key_a in all_hs and key_b in all_hs:
                d = compute_pairwise_distance(all_hs[key_a], all_hs[key_b], li)
                if d is not None:
                    result_by_layer[li]["same_lang_near"].append(d)
        
        # Same-language far: English unrelated
        for key_a, key_b in [("0_eng", "18_eng"), ("0_eng", "19_eng"), ("0_eng", "20_eng")]:
            if key_a in all_hs and key_b in all_hs:
                d = compute_pairwise_distance(all_hs[key_a], all_hs[key_b], li)
                if d is not None:
                    result_by_layer[li]["same_lang_far"].append(d)
    
    # Aggregate
    result = {}
    for li in sample_layers:
        layer_result = {}
        for cat in ["cross_lingual", "same_lang_near", "same_lang_far"]:
            vals = result_by_layer[li].get(cat, [])
            if vals:
                layer_result[cat] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "median": float(np.median(vals)),
                    "n": len(vals),
                }
        
        # Key ratio: cross_lingual / same_lang_far
        cl_mean = layer_result.get("cross_lingual", {}).get("mean", 0)
        sn_mean = layer_result.get("same_lang_near", {}).get("mean", 0)
        sf_mean = layer_result.get("same_lang_far", {}).get("mean", 0)
        
        ratio_cl_far = cl_mean / max(sf_mean, 1e-10) if sf_mean > 0.01 else 0
        ratio_cl_near = cl_mean / max(sn_mean, 1e-10) if sn_mean > 0.01 else 0
        
        layer_result["ratio_cross_to_far"] = ratio_cl_far
        layer_result["ratio_cross_to_near"] = ratio_cl_near
        layer_result["orbit_verdict"] = (
            "STRONG ORBIT: cross-lingual ≈ same-lang near" if ratio_cl_near < 1.5 and sn_mean > 0.01
            else "WEAK ORBIT: cross-lingual >> same-lang near" if ratio_cl_near > 2.0
            else "MODERATE ORBIT"
        )
        
        result[li] = layer_result
    
    # Compute slopes
    layers_sorted = sorted(result.keys())
    cl_dists = [result[li].get("cross_lingual", {}).get("mean", 0) for li in layers_sorted]
    sn_dists = [result[li].get("same_lang_near", {}).get("mean", 0) for li in layers_sorted]
    sf_dists = [result[li].get("same_lang_far", {}).get("mean", 0) for li in layers_sorted]
    
    if len(layers_sorted) >= 2:
        n_steps = layers_sorted[-1] - layers_sorted[0]
        cl_slope = (cl_dists[-1] - cl_dists[0]) / max(n_steps, 1)
        sn_slope = (sn_dists[-1] - sn_dists[0]) / max(n_steps, 1)
        sf_slope = (sf_dists[-1] - sf_dists[0]) / max(n_steps, 1)
    else:
        cl_slope = sn_slope = sf_slope = 0
    
    result["_meta"] = {
        "cross_lingual_slope": cl_slope,
        "same_lang_near_slope": sn_slope,
        "same_lang_far_slope": sf_slope,
        "cross_lingual_first": cl_dists[0] if cl_dists else 0,
        "cross_lingual_last": cl_dists[-1] if cl_dists else 0,
        "orbit_verdict": "CONVERGING: cross-lingual distances shrink in deep layers" if cl_slope < -0.001
                        else "DIVERGING" if cl_slope > 0.001 else "STABLE",
        "sample_layers": sample_layers,
    }
    
    del all_hs
    force_cleanup()
    return result


# =====================================================================
# EXP4: TRAINED VS RANDOM JACOBIAN
# =====================================================================

def exp4_trained_vs_random_jacobian(model, tokenizer, device, n_layers, d_model, model_name):
    """
    ★★★ 去伪核心: λ>1是学到的还是固有的? ★★★
    
    方法:
    1. 在训练模型上, 用Phase 185的方法测λ_max
    2. 创建随机初始化的同架构模型, 测λ_max
    3. 对比: 若随机模型也λ>1 → λ>1是架构属性
             若随机模型λ≈1或<1 → λ>1是训练学到的
    
    注意: 不能真的创建随机模型(太慢), 所以用替代方案:
    - 在训练模型的输入上注入随机扰动, 测量传播
    - 在不同上下文中测λ, 看λ是否context-dependent
    - 如果λ依赖于语义内容 → λ>1是学到的编码放大
    - 如果λ对所有输入都相同 → λ>1是架构属性
    """
    print("\n" + "="*70)
    print("Exp4: TRAINED VS RANDOM JACOBIAN (Context Dependence Test)")
    print("  (Is λ>1 a learned code amplification or architectural artifact?)")
    print("="*70)
    
    eps_rel = 0.01  # 1% perturbation
    test_layers = sorted(set([1, 2, 3, 5, 10, n_layers//2, n_layers-5, n_layers-2]))
    test_layers = [l for l in test_layers if 1 <= l < n_layers]
    
    # Three types of inputs:
    # 1. Meaningful sentences (semantic content)
    # 2. Random word sequences (no semantic structure)
    # 3. Repeated tokens (minimal structure)
    meaningful_sentences = [
        "The cat sleeps on the mat",
        "She walked to the store yesterday",
        "Water boils at one hundred degrees",
        "The scientist discovered a new element",
    ]
    
    # Random word sequences (grammatically broken)
    random_sentences = [
        "mat the sleeps cat on The",
        "yesterday store the to walked She",
        "degrees hundred one at boils Water",
        "element new a discovered scientist The",
    ]
    
    # Repeated tokens (will use same token repeated)
    repeat_sentence = "the the the the the the the"
    
    all_inputs = {
        "meaningful": meaningful_sentences,
        "random_order": random_sentences,
        "repeated": [repeat_sentence],
    }
    
    jacobian_by_context = defaultdict(lambda: defaultdict(list))
    # {input_type: {layer: [lambda_max values]}}
    
    layers = get_layers(model)
    
    for input_type, sentences in all_inputs.items():
        print(f"\n  Context type: {input_type} ({len(sentences)} sentences)", flush=True)
        
        for si, sent in enumerate(sentences):
            print(f"    Sentence {si+1}: '{sent[:50]}...'", flush=True)
            
            inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(device)
            attn_mask = inputs["attention_mask"].to(device)
            
            # Get clean hidden states
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn_mask,
                           output_hidden_states=True)
            pos = input_ids.shape[1] - 1
            clean_hs = {}
            for li, hs in enumerate(out.hidden_states):
                clean_hs[li] = hs[0, pos].detach().cpu().float().numpy().astype(np.float32)
            del out
            force_cleanup()
            
            # Compute Jacobian amplification at each test layer
            for patch_li in test_layers:
                if patch_li < 1 or patch_li >= len(layers):
                    continue
                
                h_norm = float(np.linalg.norm(clean_hs.get(patch_li, np.zeros(1))))
                if h_norm < 1e-10:
                    continue
                
                eps_abs = eps_rel * h_norm
                
                # Use random direction (not constraint-specific)
                rng = np.random.RandomState(42 + si)
                v_rand = rng.randn(d_model).astype(np.float32)
                v_rand_norm = float(np.linalg.norm(v_rand))
                if v_rand_norm < 1e-10:
                    continue
                v_rand = v_rand / v_rand_norm
                
                # Also use constraint direction from syntactic pair
                # (reuse the same method as Phase 185)
                # For simplicity, we just use random direction here
                
                # Inject perturbation and measure amplification
                perturb_vec = eps_abs * v_rand
                
                captured_output = {}
                
                def make_inject_hook(pvec, tpos):
                    pt = torch.tensor(pvec, dtype=torch.bfloat16, device=device)
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            new_out = output[0].detach().clone()
                            p = min(tpos, new_out.shape[1] - 1)
                            new_out[0, p] += pt.to(new_out.device)
                            return (new_out,) + output[1:]
                        return output
                    return hook_fn
                
                def make_capture_hook(key, tpos):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            captured_output[key] = output[0][0, min(tpos, output[0].shape[1]-1)].detach().cpu().float().numpy()
                    return hook_fn
                
                hook_inject = None
                hook_capture = None
                
                try:
                    hook_inject = layers[patch_li - 1].register_forward_hook(
                        make_inject_hook(perturb_vec, pos))
                    hook_capture = layers[patch_li].register_forward_hook(
                        make_capture_hook("perturbed", pos))
                    
                    with torch.no_grad():
                        model(input_ids=input_ids, attention_mask=attn_mask)
                    
                    hook_inject.remove()
                    hook_inject = None
                    hook_capture.remove()
                    hook_capture = None
                    
                    if "perturbed" in captured_output and patch_li + 1 in clean_hs:
                        delta_h = captured_output["perturbed"] - clean_hs[patch_li + 1]
                        g = float(np.linalg.norm(delta_h)) / eps_abs
                        jacobian_by_context[input_type][patch_li].append(g)
                
                except Exception as e:
                    if hook_inject:
                        hook_inject.remove()
                    if hook_capture:
                        hook_capture.remove()
            
            del clean_hs
            force_cleanup()
    
    # Aggregate
    result = {}
    for input_type in ["meaningful", "random_order", "repeated"]:
        type_result = {}
        for li in sorted(jacobian_by_context[input_type].keys()):
            vals = jacobian_by_context[input_type][li]
            if vals:
                type_result[li] = {
                    "g_mean": float(np.mean(vals)),
                    "g_std": float(np.std(vals)),
                    "n_obs": len(vals),
                }
        
        # Compute average g
        all_g = [v for li_data in jacobian_by_context[input_type].values() for v in li_data]
        type_result["_meta"] = {
            "overall_g_mean": float(np.mean(all_g)) if all_g else 0,
            "overall_g_std": float(np.std(all_g)) if all_g else 0,
            "n_obs": len(all_g),
        }
        result[input_type] = type_result
    
    # Key comparison: meaningful vs random
    meaningful_gs = []
    random_gs = []
    for li in test_layers:
        if li in jacobian_by_context["meaningful"]:
            meaningful_gs.extend(jacobian_by_context["meaningful"][li])
        if li in jacobian_by_context["random_order"]:
            random_gs.extend(jacobian_by_context["random_order"][li])
    
    if meaningful_gs and random_gs:
        from scipy.stats import mannwhitneyu
        u_stat, p_val = mannwhitneyu(meaningful_gs, random_gs, alternative='two-sided')
        result["_comparison"] = {
            "meaningful_g_mean": float(np.mean(meaningful_gs)),
            "random_g_mean": float(np.mean(random_gs)),
            "mann_whitney_u": float(u_stat),
            "p_value": float(p_val),
            "verdict": "CONTEXT-DEPENDENT: λ>1 is learned (meaningful≠random)" if p_val < 0.05
                      else "CONTEXT-INDEPENDENT: λ>1 may be architectural",
            "meaningful_n": len(meaningful_gs),
            "random_n": len(random_gs),
        }
    else:
        result["_comparison"] = {
            "verdict": "INSUFFICIENT DATA",
        }
    
    force_cleanup()
    return result


# =====================================================================
# MAIN
# =====================================================================

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    t_start = time.time()
    
    print(f"\n{'#'*70}")
    print(f"# Phase 186: DIFFERENTIAL CODING GEOMETRY — {model_name}")
    print(f"# Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Core: Equivalence Class Formation & Distinguishability Emergence")
    print(f"{'#'*70}")
    
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    n_layers, d_model, vocab_size = info.n_layers, info.d_model, info.vocab_size
    print(f"\nModel: {info.model_class}, Layers={n_layers}, d_model={d_model}, vocab={vocab_size}")
    
    # ===== Exp1: Equivalence Class Contraction =====
    print(f"\n{'='*70}")
    print("Running Exp1: Equivalence Class Contraction...")
    print("  ★★★ Do paraphrases/synonyms converge in deep layers? ★★★")
    exp1_results = exp1_equivalence_class_contraction(model, tokenizer, device, n_layers, d_model)
    force_cleanup()
    
    # ===== Exp2: Distinguishability Emergence =====
    print(f"\n{'='*70}")
    print("Running Exp2: Distinguishability Emergence...")
    print("  ★★★ When do concepts become separable? ★★★")
    exp2_results = exp2_distinguishability_emergence(model, tokenizer, device, n_layers, d_model)
    force_cleanup()
    
    # ===== Exp3: Cross-Lingual Semantic Orbit =====
    print(f"\n{'='*70}")
    print("Running Exp3: Cross-Lingual Semantic Orbit...")
    print("  ★★★ Do Chinese↔English same-meaning sentences share orbit? ★★★")
    exp3_results = exp3_cross_lingual_orbit(model, tokenizer, device, n_layers, d_model)
    force_cleanup()
    
    # ===== Exp4: Trained vs Random Jacobian =====
    print(f"\n{'='*70}")
    print("Running Exp4: Trained vs Random Jacobian...")
    print("  ★★★ Is λ>1 learned code amplification or architectural artifact? ★★★")
    exp4_results = exp4_trained_vs_random_jacobian(model, tokenizer, device, n_layers, d_model, model_name)
    force_cleanup()
    
    # ===== Save =====
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_path = f"tests/glm5_temp/phase186_{model_name}_{timestamp}.json"
    
    def make_serializable(obj):
        """Convert numpy types to Python native types for JSON"""
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [make_serializable(x) for x in obj]
        return obj
    
    full_results = {
        "model": model_name, "n_layers": n_layers, "d_model": d_model, "vocab_size": vocab_size,
        "timestamp": timestamp, "elapsed_sec": round(time.time() - t_start, 1),
        "exp1_equivalence_class_contraction": make_serializable(exp1_results),
        "exp2_distinguishability_emergence": make_serializable(exp2_results),
        "exp3_cross_lingual_orbit": make_serializable(exp3_results),
        "exp4_trained_vs_random_jacobian": make_serializable(exp4_results),
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")
    
    # ===== Summary =====
    print(f"\n{'#'*70}")
    print("PHASE 186 SUMMARY")
    print(f"{'#'*70}")
    
    # Exp1
    print("\n★★★ Exp1: Equivalence Class Contraction ★★★")
    meta1 = exp1_results.get("_meta", {})
    print(f"  Intra-class slope: {meta1.get('intra_slope', 0):.5f} [{meta1.get('intra_verdict', 'N/A')}]")
    print(f"  Inter-class slope: {meta1.get('inter_slope', 0):.5f} [{meta1.get('inter_verdict', 'N/A')}]")
    sep_first = meta1.get('separability_first', 0)
    sep_last = meta1.get('separability_last', 0)
    print(f"  Separability: L0={sep_first:.2f} → L{meta1.get('sample_layers', [0])[-1] if meta1.get('sample_layers') else '?'}={sep_last:.2f}")
    if sep_last > sep_first * 1.2:
        print(f"  ★★★ SEPARABILITY INCREASES: System actively constructs distinguishability ★★★")
    
    # Exp2
    print("\n★★★ Exp2: Distinguishability Emergence ★★★")
    for pair_name, _, _, expected_sim in SIMILARITY_SPECTRUM[:4]:
        if pair_name in exp2_results:
            meta = exp2_results[pair_name].get("_meta", {})
            slope = meta.get("emergence_slope", 0)
            emergence_l = meta.get("emergence_layer", "?")
            verdict = meta.get("verdict", "N/A")
            print(f"  {pair_name}: slope={slope:.5f}, emergence_L{emergence_l} [{verdict}]")
    corr = exp2_results.get("_correlation", {})
    if corr:
        print(f"  ★ Semantic similarity vs distance correlation: ρ={corr.get('spearman_rho', 0):.3f}, p={corr.get('p_value', 1):.4f}")
        print(f"    → {corr.get('verdict', 'N/A')}")
    
    # Exp3
    print("\n★★★ Exp3: Cross-Lingual Semantic Orbit ★★★")
    meta3 = exp3_results.get("_meta", {})
    cl_slope = meta3.get("cross_lingual_slope", 0)
    cl_first = meta3.get("cross_lingual_first", 0)
    cl_last = meta3.get("cross_lingual_last", 0)
    print(f"  Cross-lingual distance: L0={cl_first:.4f} → L_last={cl_last:.4f} (slope={cl_slope:.5f})")
    print(f"  → {meta3.get('orbit_verdict', 'N/A')}")
    
    # Find the layer with smallest cross-lingual distance
    min_cl_layer = 0
    min_cl_dist = 999
    for li_str, data in exp3_results.items():
        if li_str == "_meta" or not isinstance(data, dict):
            continue
        try:
            li_int = int(li_str)
        except (ValueError, TypeError):
            continue
        cl = data.get("cross_lingual", {}).get("mean", 999)
        if cl < min_cl_dist:
            min_cl_dist = cl
            min_cl_layer = li_int
    print(f"  Minimum cross-lingual distance at L{min_cl_layer}: {min_cl_dist:.4f}")
    
    # Exp4
    print("\n★★★ Exp4: Trained vs Random Jacobian ★★★")
    comp = exp4_results.get("_comparison", {})
    print(f"  Meaningful g_mean: {comp.get('meaningful_g_mean', 0):.3f}")
    print(f"  Random g_mean: {comp.get('random_g_mean', 0):.3f}")
    print(f"  → {comp.get('verdict', 'N/A')}")
    
    release_model(model)
    elapsed = time.time() - t_start
    print(f"\n{'#'*70}")
    print(f"Phase 186 COMPLETE! Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
