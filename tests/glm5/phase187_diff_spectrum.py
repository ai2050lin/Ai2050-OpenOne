"""
Phase 187: Differential Spectrum Geometry — Difference Renormalization
========================================================================

★★★ 理论基础 ★★★

Phase 186发现"等距压缩"——所有距离等比缩小,separability≈常数。
但这很可能是一阶统计假象: ||h_i - h_j|| 混合了有效/无效/噪声/残差维度。

用户的深刻洞察:
- 真正的编码机制不是"压缩",而是"差异重整化(difference renormalization)"
- 任务相关差异被增强(主谓一致、逻辑冲突)
- 表面形式差异被压缩(措辞、词序、语言形式)
- 语义差异被保持
- 噪声差异被消灭

★★★ 核心假设 ★★★

全局欧氏距离看不出差异重整化,因为增强和压缩互相抵消。
但如果我们分析不同类型差异向量的传播,就能看到各向异性。

★★★ 三个实验 ★★★

Exp1: Difference Amplification Spectrum (DAS) — 差异放大谱
  核心: 不同类型差异向量在传播中是否被不同对待?
  方法: 计算句对差异向量 Δ_l = h_l(a) - h_l(b)
  测量:
    - ||Δ_l||/||Δ_0|| = 累积放大率
    - ||Δ_{l+1}||/||Δ_l|| = 局部放大率
    - cos(Δ_l, Δ_{l+1}) = 方向保持度
    - Δ在主子空间中的能量占比
  差异类型: category, subordinate, syntactic, paraphrase, random_control
  预期: category/subordinate放大率 > random_control (各向异性)

Exp2: Direction-Selective Jacobian (DSJ) — 方向选择性雅可比
  核心: 雅可比矩阵对不同语义方向的放大率是否不同?
  方法: 在关键层注入语义方向vs随机方向, 测量放大率
  预期: 语义方向放大率 > 随机方向放大率

Exp3: Cross-Lingual Direction Alignment (CLDA) — 跨语言方向对齐
  核心: 同一语义对比在不同语言中的差异向量是否对齐?
  方法: 比较中文和英文中相同语义对比的差异向量方向
  预期: 深层cos(Δ_en, Δ_zh) → 1 (差异结构跨语言不变)

Usage: python tests/glm5/phase187_diff_spectrum.py <model_name>
       python tests/glm5/phase187_diff_spectrum.py qwen3
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
    print(f"[P187] Loading {model_name} (bfloat16 + device_map=auto)...")
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
    print(f"[P187] {model_name} loaded: device={device}, class={type(model).__name__}, GPU={gpu_mem:.2f}GB")
    return model, tokenizer, device


def force_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# =====================================================================
# SENTENCE PAIRS — organized by difference type
# =====================================================================

# ★ 10 pairs per type for statistical power
DIFFERENCE_PAIRS = {
    "category": [
        # Different semantic categories — should be AMPLIFIED or PRESERVED
        ("I ate an apple", "I drove a car"),
        ("The dog sleeps", "The book opens"),
        ("She loves music", "He builds houses"),
        ("Water flows down", "Fire burns bright"),
        ("The bird sings", "The car drives"),
        ("Rain falls gently", "Ice melts slowly"),
        ("The teacher speaks", "The river flows"),
        ("Bread tastes good", "Steel feels cold"),
        ("The child laughs", "The wall stands"),
        ("Flowers bloom bright", "Engines roar loud"),
    ],
    "subordinate": [
        # Within-category differences — should be PRESERVED or slightly AMPLIFIED
        ("I ate an apple", "I ate a pear"),
        ("The dog sleeps", "The cat sleeps"),
        ("She loves music", "She loves art"),
        ("Water flows down", "Water flows up"),
        ("The bird sings", "The bird flies"),
        ("The red apple", "The green apple"),
        ("A tall building", "A short building"),
        ("The hot coffee", "The cold coffee"),
        ("Heavy rain falls", "Light rain falls"),
        ("The fast runner", "The slow runner"),
    ],
    "syntactic": [
        # Grammatical differences — mixed prediction
        ("The cat sleeps", "The cats sleep"),
        ("She walks fast", "She walked fast"),
        ("He is running", "He was running"),
        ("I have a book", "I had a book"),
        ("The dog barks", "The dog barked"),
        ("They are coming", "They were coming"),
        ("She will go", "She would go"),
        ("He can swim", "He could swim"),
        ("The boy reads", "The boys read"),
        ("I am happy", "I was happy"),
    ],
    "paraphrase": [
        # Same meaning, different surface — should be COMPRESSED
        ("She reads books", "She is a reader"),
        ("I ate an apple", "I consumed an apple"),
        ("The cat is sleeping", "The cat sleeps"),
        ("He drives to work", "He commutes by car"),
        ("The water is cold", "The water feels chilly"),
        ("She wrote a letter", "She penned a message"),
        ("The house is big", "The house is large"),
        ("He runs quickly", "He runs fast"),
        ("The food was good", "The meal tasted great"),
        ("She arrived early", "She came ahead of time"),
    ],
    "random_control": [
        # Completely unrelated — baseline for comparison
        ("The apple fell", "Philosophy matters"),
        ("She walked home", "Oxygen exists"),
        ("The water boiled", "Abstract thought"),
        ("He sang loudly", "Gravity pulls"),
        ("The door opened", "Mathematics evolved"),
        ("Rivers flow", "Numbers grow"),
        ("The tree swayed", "Logic dictates"),
        ("Ice cream melts", "Truth prevails"),
        ("She smiled back", "Time passes"),
        ("The wind blew", "Science advances"),
    ],
}

# ★ Exp3: Cross-lingual contrast pairs
# Same semantic contrast in both English and Chinese
CROSS_LINGUAL_CONTRASTS = [
    # (en_a, en_b, zh_a, zh_b, contrast_name)
    ("The cat is sleeping", "The dog is sleeping", "猫在睡觉", "狗在睡觉", "cat_vs_dog"),
    ("I ate an apple", "I ate a banana", "我吃了苹果", "我吃了香蕉", "apple_vs_banana"),
    ("The sun is shining", "The moon is shining", "太阳在照耀", "月亮在照耀", "sun_vs_moon"),
    ("He runs fast", "He runs slowly", "他跑得快", "他跑得慢", "fast_vs_slow"),
    ("The house is big", "The house is small", "房子很大", "房子很小", "big_vs_small"),
    ("She loves music", "She hates music", "她喜欢音乐", "她讨厌音乐", "love_vs_hate"),
    ("The water is hot", "The water is cold", "水很热", "水很冷", "hot_vs_cold"),
    ("The bird can fly", "The bird cannot fly", "鸟会飞", "鸟不会飞", "can_vs_cannot"),
    ("The man is tall", "The man is short", "男人很高", "男人很矮", "tall_vs_short"),
    ("She is happy", "She is sad", "她很开心", "她很伤心", "happy_vs_sad"),
]

# ★ Exp2: Jacobian test sentences
JACOBIAN_BASE_SENTENCES = [
    "The scientist discovered a new element",
    "She walked to the store yesterday",
    "The cat sleeps on the warm mat",
]


def get_all_hidden_states(model, tokenizer, device, sentence, target_pos=None):
    """获取所有层的hidden states"""
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        out = model(input_ids=inputs["input_ids"].to(device),
                    attention_mask=inputs["attention_mask"].to(device),
                    output_hidden_states=True)
    if target_pos is None:
        target_pos = inputs["input_ids"].shape[1] - 1
    pos = min(target_pos, out.hidden_states[0].shape[1] - 1)
    result = {}
    n_layers = len(out.hidden_states) - 1
    for li, hs in enumerate(out.hidden_states):
        result[li] = hs[0, pos].detach().cpu().float().numpy().astype(np.float32)
    del out
    return result, n_layers


# =====================================================================
# EXP1: DIFFERENCE AMPLIFICATION SPECTRUM
# =====================================================================

def exp1_difference_amplification_spectrum(model, tokenizer, device, n_layers, d_model):
    """
    ★★★ 核心实验: 差异放大谱 ★★★

    对每种差异类型(category/subordinate/syntactic/paraphrase/random):
    1. 计算句对差异向量 Δ_l = h_l(a) - h_l(b)
    2. 测量: ||Δ_l||, ||Δ_{l+1}||/||Δ_l||, cos(Δ_l, Δ_{l+1})
    3. 计算Δ在主子空间中的能量占比
    4. 比较不同类型的传播模式

    关键: 这揭示"差异重整化"——不同差异被不同对待
    """
    print("\n" + "="*70)
    print("Exp1: DIFFERENCE AMPLIFICATION SPECTRUM (DAS)")
    print("  ★★★ Do different types of differences get treated differently? ★★★")
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
    all_hs = {}  # {diff_type: [{layer: np.array for a}, {layer: np.array for b}]}
    all_sentences_hs = {}  # Cache: {sentence: {layer: np.array}}

    for diff_type, pairs in DIFFERENCE_PAIRS.items():
        print(f"\n  [{diff_type}] Processing {len(pairs)} pairs...", flush=True)
        all_hs[diff_type] = []

        for pi, (sent_a, sent_b) in enumerate(pairs):
            if pi % 3 == 0:
                print(f"    Pair {pi+1}/{len(pairs)}", flush=True)

            # Get hidden states (with caching)
            if sent_a not in all_sentences_hs:
                hs_a, _ = get_all_hidden_states(model, tokenizer, device, sent_a)
                all_sentences_hs[sent_a] = hs_a
            hs_a = all_sentences_hs[sent_a]

            if sent_b not in all_sentences_hs:
                hs_b, _ = get_all_hidden_states(model, tokenizer, device, sent_b)
                all_sentences_hs[sent_b] = hs_b
            hs_b = all_sentences_hs[sent_b]

            all_hs[diff_type].append((hs_a, hs_b))
            force_cleanup()

    # ===== Compute PCA of all hidden states at each layer =====
    print(f"\n  Computing PCA subspaces at {len(sample_layers)} layers...", flush=True)

    # Collect all sentence hidden states for PCA
    all_hs_list = list(all_sentences_hs.values())
    pca_results = {}  # {layer: (mean, components, explained_var_ratio)}

    for li in sample_layers:
        # Stack all hidden states at this layer
        H = np.stack([hs[li] for hs in all_hs_list], axis=0)  # [N, d_model]
        mean = H.mean(axis=0)
        H_centered = H - mean

        # SVD for PCA (more stable than eigendecomposition of covariance)
        # H_centered = U S Vt, where Vt rows are principal directions
        n_components = min(50, H_centered.shape[0] - 1, H_centered.shape[1] - 1)
        n_components = max(n_components, 10)

        U, S, Vt = np.linalg.svd(H_centered, full_matrices=False)
        # S^2 / (N-1) = eigenvalues of covariance matrix
        total_var = np.sum(S**2)
        explained_var_ratio = S**2 / max(total_var, 1e-20)

        pca_results[li] = {
            "mean": mean,
            "Vt": Vt[:n_components],  # Top-k principal directions
            "explained_var_ratio": explained_var_ratio[:n_components],
            "total_var": total_var,
            "cumvar_10": float(np.sum(explained_var_ratio[:10])),
            "cumvar_50": float(np.sum(explained_var_ratio[:50])),
        }

        if li % 5 == 0 or li == sample_layers[-1]:
            cum10 = pca_results[li]["cumvar_10"]
            cum50 = pca_results[li]["cumvar_50"]
            print(f"    L{li}: cumvar_top10={cum10:.3f}, cumvar_top50={cum50:.3f}", flush=True)

    # ===== Compute difference propagation metrics =====
    print(f"\n  Computing difference propagation metrics...", flush=True)

    result_by_type = {}

    for diff_type, pairs_hs in all_hs.items():
        pair_data = []

        for hs_a, hs_b in pairs_hs:
            pair_metrics = {"layers": {}}

            for li in sample_layers:
                delta = hs_a[li] - hs_b[li]  # Δ_l
                delta_norm = float(np.linalg.norm(delta))

                # Projection onto principal subspace
                if li in pca_results:
                    Vt = pca_results[li]["Vt"]  # [k, d_model]
                    # Project Δ onto principal directions
                    proj_coeffs = Vt @ delta  # [k]
                    proj_energy = float(np.sum(proj_coeffs**2))
                    total_energy = max(delta_norm**2, 1e-20)
                    energy_in_pc = proj_energy / total_energy  # fraction in principal subspace

                    # Energy in top-5 vs bottom directions
                    sorted_coeffs2 = np.sort(proj_coeffs**2)[::-1]
                    energy_top5 = float(np.sum(sorted_coeffs2[:5])) / total_energy
                    energy_top10 = float(np.sum(sorted_coeffs2[:10])) / total_energy
                else:
                    energy_in_pc = 0
                    energy_top5 = 0
                    energy_top10 = 0

                pair_metrics["layers"][li] = {
                    "delta_norm": delta_norm,
                    "energy_in_pc": energy_in_pc,
                    "energy_top5": energy_top5,
                    "energy_top10": energy_top10,
                }

            # Compute inter-layer metrics
            for i in range(1, len(sample_layers)):
                l_prev = sample_layers[i-1]
                l_curr = sample_layers[i]

                if l_prev in pair_metrics["layers"] and l_curr in pair_metrics["layers"]:
                    d_prev = pair_metrics["layers"][l_prev]["delta_norm"]
                    d_curr = pair_metrics["layers"][l_curr]["delta_norm"]

                    # Local amplification
                    local_amp = d_curr / max(d_prev, 1e-10) if d_prev > 1e-10 else 0
                    pair_metrics["layers"][l_curr]["local_amp"] = local_amp

                    # Cumulative amplification from L0
                    d0 = pair_metrics["layers"][sample_layers[0]]["delta_norm"]
                    cumul_amp = d_curr / max(d0, 1e-10) if d0 > 1e-10 else 0
                    pair_metrics["layers"][l_curr]["cumul_amp"] = cumul_amp

                    # Direction preservation: cos(Δ_l, Δ_{l+1})
                    delta_prev = hs_a[l_prev] - hs_b[l_prev]
                    delta_curr = hs_a[l_curr] - hs_b[l_curr]
                    n_prev = np.linalg.norm(delta_prev)
                    n_curr = np.linalg.norm(delta_curr)
                    if n_prev > 1e-10 and n_curr > 1e-10:
                        cos_dir = float(np.dot(delta_prev, delta_curr) / (n_prev * n_curr))
                    else:
                        cos_dir = 0
                    pair_metrics["layers"][l_curr]["cos_direction"] = cos_dir

            pair_data.append(pair_metrics)

        # Aggregate across pairs
        agg = {}
        for li in sample_layers:
            norms = [pd["layers"][li]["delta_norm"] for pd in pair_data if li in pd["layers"]]
            amps = [pd["layers"][li].get("local_amp", 0) for pd in pair_data if li in pd["layers"] and "local_amp" in pd["layers"][li]]
            cumul_amps = [pd["layers"][li].get("cumul_amp", 0) for pd in pair_data if li in pd["layers"] and "cumul_amp" in pd["layers"][li]]
            cos_dirs = [pd["layers"][li].get("cos_direction", 0) for pd in pair_data if li in pd["layers"] and "cos_direction" in pd["layers"][li]]
            energy_pc = [pd["layers"][li]["energy_in_pc"] for pd in pair_data if li in pd["layers"]]
            energy_top5 = [pd["layers"][li]["energy_top5"] for pd in pair_data if li in pd["layers"]]

            agg[li] = {
                "norm_mean": float(np.mean(norms)) if norms else 0,
                "norm_std": float(np.std(norms)) if norms else 0,
                "local_amp_mean": float(np.mean(amps)) if amps else 0,
                "local_amp_std": float(np.std(amps)) if amps else 0,
                "cumul_amp_mean": float(np.mean(cumul_amps)) if cumul_amps else 0,
                "cos_dir_mean": float(np.mean(cos_dirs)) if cos_dirs else 0,
                "cos_dir_std": float(np.std(cos_dirs)) if cos_dirs else 0,
                "energy_in_pc_mean": float(np.mean(energy_pc)) if energy_pc else 0,
                "energy_top5_mean": float(np.mean(energy_top5)) if energy_top5 else 0,
                "n_pairs": len(norms),
            }

        result_by_type[diff_type] = agg

        # Print summary for this type
        first_li = sample_layers[0]
        last_li = sample_layers[-1]
        print(f"\n  [{diff_type}] ({len(pairs_hs)} pairs)", flush=True)
        print(f"    Norm: L{first_li}={agg[first_li]['norm_mean']:.4f} → L{last_li}={agg[last_li]['norm_mean']:.4f}", flush=True)
        print(f"    Cumul_amp: L{last_li}={agg[last_li].get('cumul_amp_mean', 0):.4f}", flush=True)
        print(f"    cos_dir: L{last_li}={agg[last_li].get('cos_dir_mean', 0):.4f}", flush=True)
        print(f"    Energy in PC: L{first_li}={agg[first_li]['energy_in_pc_mean']:.3f} → L{last_li}={agg[last_li]['energy_in_pc_mean']:.3f}", flush=True)
        print(f"    Energy top5: L{first_li}={agg[first_li]['energy_top5_mean']:.3f} → L{last_li}={agg[last_li]['energy_top5_mean']:.3f}", flush=True)

    # ===== Key comparison: category vs random at last layer =====
    print(f"\n  ★★★ KEY COMPARISON: Difference Renormalization ★★★", flush=True)
    last_li = sample_layers[-1]
    for dt in ["category", "subordinate", "syntactic", "paraphrase", "random_control"]:
        if dt in result_by_type and last_li in result_by_type[dt]:
            ca = result_by_type[dt][last_li].get("cumul_amp_mean", 0)
            cd = result_by_type[dt][last_li].get("cos_dir_mean", 0)
            epc = result_by_type[dt][last_li].get("energy_in_pc_mean", 0)
            e5 = result_by_type[dt][last_li].get("energy_top5_mean", 0)
            print(f"    {dt:20s}: cumul_amp={ca:.4f}, cos_dir={cd:.4f}, E_pc={epc:.3f}, E_top5={e5:.3f}", flush=True)

    # ===== Statistical test: category vs random cumulative amplification =====
    # Collect per-pair cumulative amplification at last layer
    cat_cumul = []
    rnd_cumul = []
    for pd_item in all_hs.get("category", []):
        hs_a, hs_b = pd_item
        d0 = float(np.linalg.norm(hs_a[sample_layers[0]] - hs_b[sample_layers[0]]))
        d_last = float(np.linalg.norm(hs_a[last_li] - hs_b[last_li]))
        if d0 > 1e-10:
            cat_cumul.append(d_last / d0)
    for pd_item in all_hs.get("random_control", []):
        hs_a, hs_b = pd_item
        d0 = float(np.linalg.norm(hs_a[sample_layers[0]] - hs_b[sample_layers[0]]))
        d_last = float(np.linalg.norm(hs_a[last_li] - hs_b[last_li]))
        if d0 > 1e-10:
            rnd_cumul.append(d_last / d0)

    stat_result = {}
    if cat_cumul and rnd_cumul:
        from scipy.stats import mannwhitneyu
        u_stat, p_val = mannwhitneyu(cat_cumul, rnd_cumul, alternative='two-sided')
        stat_result = {
            "category_cumul_mean": float(np.mean(cat_cumul)),
            "random_cumul_mean": float(np.mean(rnd_cumul)),
            "mann_whitney_u": float(u_stat),
            "p_value": float(p_val),
            "n_category": len(cat_cumul),
            "n_random": len(rnd_cumul),
            "verdict": "ANISOTROPY DETECTED: category≠random" if p_val < 0.05 else "ISOTROPIC: no significant difference",
        }
        print(f"\n  ★ STAT TEST: category cumul={np.mean(cat_cumul):.4f} vs random cumul={np.mean(rnd_cumul):.4f}, p={p_val:.4f}")
        print(f"    → {stat_result['verdict']}")

    # ===== Collect PCA results =====
    pca_summary = {}
    for li in sample_layers:
        if li in pca_results:
            pca_summary[li] = {
                "cumvar_top10": pca_results[li]["cumvar_10"],
                "cumvar_top50": pca_results[li]["cumvar_50"],
            }

    # Cleanup
    del all_hs, all_sentences_hs
    force_cleanup()

    return {
        "by_type": result_by_type,
        "pca": pca_summary,
        "stat_test": stat_result,
        "sample_layers": sample_layers,
    }


# =====================================================================
# EXP2: DIRECTION-SELECTIVE JACOBIAN
# =====================================================================

def exp2_direction_selective_jacobian(model, tokenizer, device, n_layers, d_model):
    """
    ★★★ 方向选择性雅可比 ★★★

    在关键层注入不同类型的语义方向和随机方向, 测量放大率。
    如果语义方向放大率 > 随机方向, 则Jacobian是各向异性的。

    这直接检验"差异重整化"假设。
    """
    print("\n" + "="*70)
    print("Exp2: DIRECTION-SELECTIVE JACOBIAN (DSJ)")
    print("  ★★★ Is the Jacobian anisotropic for semantic directions? ★★★")
    print("="*70)

    eps_rel = 0.01  # 1% perturbation
    # Test at 4 key layers
    test_layers = sorted(set([
        1, 2, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-2
    ]))
    test_layers = [l for l in test_layers if 1 <= l < n_layers]

    layers = get_layers(model)

    # Pre-compute hidden states for sentence pairs to get semantic directions
    print("  Pre-computing semantic directions...", flush=True)

    # Define direction sources
    direction_sources = {
        "category": [
            ("I ate an apple", "I drove a car"),
            ("The dog sleeps", "The book opens"),
            ("She loves music", "He builds houses"),
        ],
        "subordinate": [
            ("I ate an apple", "I ate a pear"),
            ("The dog sleeps", "The cat sleeps"),
            ("She loves music", "She loves art"),
        ],
        "paraphrase": [
            ("She reads books", "She is a reader"),
            ("I ate an apple", "I consumed an apple"),
            ("The cat is sleeping", "The cat sleeps"),
        ],
    }

    # Get semantic directions at each test layer
    semantic_dirs = {}  # {dir_type: {layer: [direction_vectors]}}
    base_hs_cache = {}

    for dir_type, pairs in direction_sources.items():
        semantic_dirs[dir_type] = defaultdict(list)

        for sent_a, sent_b in pairs:
            # Get hidden states
            if sent_a not in base_hs_cache:
                hs_a, _ = get_all_hidden_states(model, tokenizer, device, sent_a)
                base_hs_cache[sent_a] = hs_a
            if sent_b not in base_hs_cache:
                hs_b, _ = get_all_hidden_states(model, tokenizer, device, sent_b)
                base_hs_cache[sent_b] = hs_b

            hs_a = base_hs_cache[sent_a]
            hs_b = base_hs_cache[sent_b]

            for li in test_layers:
                delta = hs_a[li] - hs_b[li]
                d_norm = float(np.linalg.norm(delta))
                if d_norm > 1e-10:
                    semantic_dirs[dir_type][li].append(delta / d_norm)

            force_cleanup()

    # Now do the Jacobian test
    print(f"\n  Running Jacobian amplification test at {len(test_layers)} layers...", flush=True)

    jacobian_results = defaultdict(lambda: defaultdict(list))
    # {dir_type: {layer: [amplification_values]}}

    for base_sent in JACOBIAN_BASE_SENTENCES:
        print(f"\n  Base: '{base_sent[:50]}...'", flush=True)

        inputs = tokenizer(base_sent, return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(device)
        attn_mask = inputs["attention_mask"].to(device)
        pos = input_ids.shape[1] - 1

        # Get clean hidden states
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
        clean_hs = {}
        for li, hs in enumerate(out.hidden_states):
            clean_hs[li] = hs[0, pos].detach().cpu().float().numpy().astype(np.float32)
        del out
        force_cleanup()

        for li in test_layers:
            h_norm = float(np.linalg.norm(clean_hs.get(li, np.zeros(1))))
            if h_norm < 1e-10:
                continue

            eps_abs = eps_rel * h_norm

            # Test directions
            test_dirs = {}

            # Semantic directions
            for dir_type in ["category", "subordinate", "paraphrase"]:
                if li in semantic_dirs[dir_type] and semantic_dirs[dir_type][li]:
                    # Use first available direction
                    test_dirs[dir_type] = semantic_dirs[dir_type][li][0]

            # Random direction
            rng = np.random.RandomState(42)
            v_rand = rng.randn(d_model).astype(np.float32)
            v_rand = v_rand / np.linalg.norm(v_rand)
            test_dirs["random"] = v_rand

            # Residual stream direction (along h itself)
            h_dir = clean_hs[li] / max(np.linalg.norm(clean_hs[li]), 1e-10)
            test_dirs["along_h"] = h_dir

            # Inject and measure
            for dir_name, v_dir in test_dirs.items():
                perturb_vec = eps_abs * v_dir

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
                    hook_inject = layers[li].register_forward_hook(
                        make_inject_hook(perturb_vec, pos))
                    hook_capture = layers[min(li+1, n_layers-1)].register_forward_hook(
                        make_capture_hook("perturbed", pos))

                    with torch.no_grad():
                        model(input_ids=input_ids, attention_mask=attn_mask)

                    hook_inject.remove()
                    hook_inject = None
                    hook_capture.remove()
                    hook_capture = None

                    next_li = min(li+1, n_layers-1)
                    if "perturbed" in captured_output and next_li in clean_hs:
                        delta_h = captured_output["perturbed"] - clean_hs[next_li]
                        g = float(np.linalg.norm(delta_h)) / eps_abs
                        jacobian_results[dir_name][li].append(g)

                except Exception as e:
                    if hook_inject:
                        hook_inject.remove()
                    if hook_capture:
                        hook_capture.remove()
                    print(f"    Error at L{li} dir={dir_name}: {e}")

        del clean_hs
        force_cleanup()

    # Aggregate
    result = {}
    for dir_name in ["category", "subordinate", "paraphrase", "random", "along_h"]:
        dir_result = {}
        all_g = []
        for li in test_layers:
            vals = jacobian_results[dir_name][li]
            if vals:
                dir_result[li] = {
                    "g_mean": float(np.mean(vals)),
                    "g_std": float(np.std(vals)),
                    "n_obs": len(vals),
                }
                all_g.extend(vals)

        dir_result["_meta"] = {
            "overall_g_mean": float(np.mean(all_g)) if all_g else 0,
            "overall_g_std": float(np.std(all_g)) if all_g else 0,
            "n_obs": len(all_g),
        }
        result[dir_name] = dir_result

    # Key comparison: semantic vs random
    semantic_gs = []
    random_gs = []
    for dir_name in ["category", "subordinate", "paraphrase"]:
        for li in test_layers:
            semantic_gs.extend(jacobian_results[dir_name][li])
    for li in test_layers:
        random_gs.extend(jacobian_results["random"][li])

    if semantic_gs and random_gs:
        from scipy.stats import mannwhitneyu
        u_stat, p_val = mannwhitneyu(semantic_gs, random_gs, alternative='two-sided')
        result["_comparison"] = {
            "semantic_g_mean": float(np.mean(semantic_gs)),
            "semantic_g_std": float(np.std(semantic_gs)),
            "random_g_mean": float(np.mean(random_gs)),
            "random_g_std": float(np.std(random_gs)),
            "mann_whitney_u": float(u_stat),
            "p_value": float(p_val),
            "n_semantic": len(semantic_gs),
            "n_random": len(random_gs),
            "verdict": "ANISOTROPIC: semantic≠random (p<0.05)" if p_val < 0.05
                      else "ISOTROPIC: no significant difference",
        }
        print(f"\n  ★ STAT TEST: semantic g={np.mean(semantic_gs):.3f} vs random g={np.mean(random_gs):.3f}, p={p_val:.4f}")
        print(f"    → {result['_comparison']['verdict']}")
    else:
        result["_comparison"] = {"verdict": "INSUFFICIENT DATA"}

    force_cleanup()
    return result


# =====================================================================
# EXP3: CROSS-LINGUAL DIRECTION ALIGNMENT
# =====================================================================

def exp3_cross_lingual_direction_alignment(model, tokenizer, device, n_layers, d_model):
    """
    ★★★ 跨语言方向对齐 ★★★

    核心测试: 同一语义对比在不同语言中的差异向量是否对齐?
    例如: "猫vs狗"的差异方向在中文和英文中是否一致?

    如果深层 cos(Δ_en, Δ_zh) → 1:
    → 差异结构是语言不变的, 差异重整化机制跨语言共享
    """
    print("\n" + "="*70)
    print("Exp3: CROSS-LINGUAL DIRECTION ALIGNMENT (CLDA)")
    print("  ★★★ Is the same semantic contrast encoded identically across languages? ★★★")
    print("="*70)

    n_sample = min(15, n_layers)
    sample_layers = sorted(set(
        [0, 1, 2] +
        list(range(0, n_layers, max(1, n_layers // n_sample))) +
        [n_layers//4, n_layers//2, 3*n_layers//4, n_layers-2, n_layers-1]
    ))
    sample_layers = [l for l in sample_layers if 0 <= l <= n_layers]
    sample_layers = sorted(set(sample_layers))

    # Get hidden states for all cross-lingual contrast sentences
    all_hs = {}  # {sentence: {layer: np.array}}

    for ci, (en_a, en_b, zh_a, zh_b, contrast_name) in enumerate(CROSS_LINGUAL_CONTRASTS):
        print(f"  Contrast {ci+1}/{len(CROSS_LINGUAL_CONTRASTS)}: {contrast_name}", flush=True)

        for sent in [en_a, en_b, zh_a, zh_b]:
            if sent not in all_hs:
                hs, _ = get_all_hidden_states(model, tokenizer, device, sent)
                all_hs[sent] = hs
                force_cleanup()

    # Compute direction alignment
    result = {}

    for li in sample_layers:
        cos_en_zh_list = []
        cos_en_en_list = []  # Control: direction consistency across English pairs
        norm_en_list = []
        norm_zh_list = []

        for ci, (en_a, en_b, zh_a, zh_b, contrast_name) in enumerate(CROSS_LINGUAL_CONTRASTS):
            # English difference vector
            delta_en = all_hs[en_a][li] - all_hs[en_b][li]
            norm_en = float(np.linalg.norm(delta_en))

            # Chinese difference vector
            delta_zh = all_hs[zh_a][li] - all_hs[zh_b][li]
            norm_zh = float(np.linalg.norm(delta_zh))

            norm_en_list.append(norm_en)
            norm_zh_list.append(norm_zh)

            # cos(Δ_en, Δ_zh) — direction alignment
            if norm_en > 1e-10 and norm_zh > 1e-10:
                cos_alignment = float(np.dot(delta_en, delta_zh) / (norm_en * norm_zh))
                cos_en_zh_list.append(cos_alignment)
            else:
                cos_en_zh_list.append(0)

        result[li] = {
            "cos_en_zh_mean": float(np.mean(cos_en_zh_list)) if cos_en_zh_list else 0,
            "cos_en_zh_std": float(np.std(cos_en_zh_list)) if cos_en_zh_list else 0,
            "cos_en_zh_median": float(np.median(cos_en_zh_list)) if cos_en_zh_list else 0,
            "norm_en_mean": float(np.mean(norm_en_list)),
            "norm_zh_mean": float(np.mean(norm_zh_list)),
            "n_contrasts": len(cos_en_zh_list),
        }

        if li % 5 == 0 or li == sample_layers[-1]:
            ca = result[li]["cos_en_zh_mean"]
            print(f"    L{li}: cos(Δ_en, Δ_zh)={ca:.4f}", flush=True)

    # Compute slope
    layers_sorted = sorted(result.keys())
    cos_vals = [result[li]["cos_en_zh_mean"] for li in layers_sorted]

    if len(layers_sorted) >= 2:
        n_steps = layers_sorted[-1] - layers_sorted[0]
        slope = (cos_vals[-1] - cos_vals[0]) / max(n_steps, 1)
    else:
        slope = 0

    result["_meta"] = {
        "cos_en_zh_first": cos_vals[0] if cos_vals else 0,
        "cos_en_zh_last": cos_vals[-1] if cos_vals else 0,
        "cos_slope": slope,
        "verdict": "CONVERGING: cross-lingual direction alignment increases" if slope > 0.001
                  else "DIVERGING" if slope < -0.001 else "STABLE",
        "sample_layers": sample_layers,
    }

    # Per-contrast alignment at last layer
    per_contrast = {}
    last_li = sample_layers[-1]
    for ci, (en_a, en_b, zh_a, zh_b, contrast_name) in enumerate(CROSS_LINGUAL_CONTRASTS):
        delta_en = all_hs[en_a][last_li] - all_hs[en_b][last_li]
        delta_zh = all_hs[zh_a][last_li] - all_hs[zh_b][last_li]
        ne = float(np.linalg.norm(delta_en))
        nz = float(np.linalg.norm(delta_zh))
        if ne > 1e-10 and nz > 1e-10:
            ca = float(np.dot(delta_en, delta_zh) / (ne * nz))
        else:
            ca = 0
        per_contrast[contrast_name] = {"cos": ca, "norm_en": ne, "norm_zh": nz}

    result["_per_contrast"] = per_contrast

    print(f"\n  ★ Cross-lingual direction alignment: L0={cos_vals[0]:.4f} → L{last_li}={cos_vals[-1]:.4f}")
    print(f"    → {result['_meta']['verdict']}")
    print(f"  Per-contrast at L{last_li}:")
    for cname, cdata in sorted(per_contrast.items()):
        print(f"    {cname}: cos={cdata['cos']:.4f}", flush=True)

    del all_hs
    force_cleanup()

    return result


# =====================================================================
# MAIN
# =====================================================================

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    t_start = time.time()

    print(f"\n{'#'*70}")
    print(f"# Phase 187: DIFFERENTIAL SPECTRUM GEOMETRY — {model_name}")
    print(f"# Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Core: Difference Renormalization — Anisotropy of Encoding")
    print(f"{'#'*70}")

    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    n_layers, d_model, vocab_size = info.n_layers, info.d_model, info.vocab_size
    print(f"\nModel: {info.model_class}, Layers={n_layers}, d_model={d_model}, vocab={vocab_size}")

    # ===== Exp1: Difference Amplification Spectrum =====
    print(f"\n{'='*70}")
    print("Running Exp1: Difference Amplification Spectrum...")
    print("  ★★★ Do different difference types get different treatment? ★★★")
    exp1_results = exp1_difference_amplification_spectrum(model, tokenizer, device, n_layers, d_model)
    force_cleanup()

    # ===== Exp2: Direction-Selective Jacobian =====
    print(f"\n{'='*70}")
    print("Running Exp2: Direction-Selective Jacobian...")
    print("  ★★★ Is the Jacobian anisotropic for semantic directions? ★★★")
    exp2_results = exp2_direction_selective_jacobian(model, tokenizer, device, n_layers, d_model)
    force_cleanup()

    # ===== Exp3: Cross-Lingual Direction Alignment =====
    print(f"\n{'='*70}")
    print("Running Exp3: Cross-Lingual Direction Alignment...")
    print("  ★★★ Is the same semantic contrast encoded identically across languages? ★★★")
    exp3_results = exp3_cross_lingual_direction_alignment(model, tokenizer, device, n_layers, d_model)
    force_cleanup()

    # ===== Save =====
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')

    def make_serializable(obj):
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
        "exp1_diff_amplification_spectrum": make_serializable(exp1_results),
        "exp2_direction_selective_jacobian": make_serializable(exp2_results),
        "exp3_cross_lingual_direction_alignment": make_serializable(exp3_results),
    }

    output_path = f"tests/glm5_temp/phase187_{model_name}_{timestamp}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")

    # ===== Summary =====
    print(f"\n{'#'*70}")
    print("PHASE 187 SUMMARY")
    print(f"{'#'*70}")

    # Exp1 summary
    print("\n★★★ Exp1: Difference Amplification Spectrum ★★★")
    last_li = exp1_results.get("sample_layers", [0])[-1]
    for dt in ["category", "subordinate", "syntactic", "paraphrase", "random_control"]:
        if dt in exp1_results.get("by_type", {}):
            agg = exp1_results["by_type"][dt]
            if last_li in agg:
                ca = agg[last_li].get("cumul_amp_mean", 0)
                cd = agg[last_li].get("cos_dir_mean", 0)
                epc = agg[last_li].get("energy_in_pc_mean", 0)
                print(f"  {dt:20s}: cumul_amp={ca:.4f}, cos_dir={cd:.4f}, E_pc={epc:.3f}")

    stat = exp1_results.get("stat_test", {})
    if stat:
        print(f"  ★ Category vs Random: p={stat.get('p_value', 1):.4f} → {stat.get('verdict', 'N/A')}")

    # Exp2 summary
    print("\n★★★ Exp2: Direction-Selective Jacobian ★★★")
    comp = exp2_results.get("_comparison", {})
    if comp:
        print(f"  Semantic g: {comp.get('semantic_g_mean', 0):.3f} ± {comp.get('semantic_g_std', 0):.3f}")
        print(f"  Random g:   {comp.get('random_g_mean', 0):.3f} ± {comp.get('random_g_std', 0):.3f}")
        print(f"  p={comp.get('p_value', 1):.4f} → {comp.get('verdict', 'N/A')}")

    for dir_name in ["category", "subordinate", "paraphrase", "random", "along_h"]:
        if dir_name in exp2_results:
            meta = exp2_results[dir_name].get("_meta", {})
            print(f"  {dir_name:20s}: g_mean={meta.get('overall_g_mean', 0):.3f}")

    # Exp3 summary
    print("\n★★★ Exp3: Cross-Lingual Direction Alignment ★★★")
    meta3 = exp3_results.get("_meta", {})
    print(f"  cos(Δ_en, Δ_zh): L0={meta3.get('cos_en_zh_first', 0):.4f} → L_last={meta3.get('cos_en_zh_last', 0):.4f}")
    print(f"  → {meta3.get('verdict', 'N/A')}")

    release_model(model)
    elapsed = time.time() - t_start
    print(f"\n{'#'*70}")
    print(f"Phase 187 COMPLETE! Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
