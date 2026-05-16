"""
Phase 189: Universal Polarity Direction — Robustness Validation
================================================================

★★★ 核心问题 ★★★

Phase 188发现: 20个极性对比在低PC空间中几乎完全平行(cross_dim_cos=0.91-0.97)
但这可能是因为使用了相同模板句"Things that are [adj]",结构完全相同。

如果换用不同句式模板,极性方向是否仍然平行?
如果用随机词对(非极性),差异向量是否也平行?

★★★ 实验设计 ★★★

Exp1: 句式模板控制实验 (Template Control)
  核心: 极性方向是否独立于句式模板?
  方法: 用5种不同模板表达同一极性对比,检查差异向量是否仍然平行
  模板:
    a. "Things that are [adj]" (原模板)
    b. "The weather is [adj]" (主语固定)
    c. "[Adj] things exist" (形容词前置)
    d. "I feel [adj] today" (第一人称)
    e. "Something is [adj]" (中性主语)
  预期: 如果极性方向是真正的语义结构, 不同模板的差异向量应平行

Exp2: 极性vs随机控制 (Polarity vs Random Control)
  核心: 极性词对的平行性是否特殊?
  方法: 构造随机词对(非极性), 检查差异向量在低PC空间中的平行度
  随机词对:
    a. "Things that are [word1]" vs "Things that are [word2]"
       其中word1和word2是随机的(非极性)词
    b. 比较: 极性词对的cos vs 随机词对的cos
  预期: 极性词对的cos >> 随机词对的cos

Exp3: 极性方向的正交性检验 (Orthogonality Test)
  核心: 极性方向是否与内容方向正交?
  方法:
    a. 提取通用极性方向(所有极性差异向量的平均)
    b. 计算极性方向与内容差异向量(不同主题类别)的cos
    c. 计算极性方向与PCA主方向的cos
  预期: 极性方向与内容方向低cos → 真正正交

Exp4: 语义分化三维度检验 (Three-Factor Test)
  核心: Osgood三维度(Evaluation, Potency, Activity)是否存在?
  方法:
    a. 构造三类词对:
       - Evaluation: good/bad, love/hate, beautiful/ugly
       - Potency: strong/weak, powerful/powerless, hard/soft
       - Activity: fast/slow, active/passive, hot/cold
    b. 提取每类词对的差异向量
    c. 在低PC空间中做PCA, 看是否有3个独立方向
  预期: 低PC空间中有2-3个独立方向(不只是1个)

★★★ 数据量 ★★★
- 每个极性对比用5种模板 = 5×20 = 100对
- 20个随机词对
- 15个类别差异向量
- 足够做可靠统计

Usage: python tests/glm5/phase189_polarity_validation.py <model_name>
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
    print(f"[P189] Loading {model_name} (bfloat16 + device_map=auto)...")
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
    print(f"[P189] {model_name} loaded: device={device}, class={type(model).__name__}, GPU={gpu_mem:.2f}GB")
    return model, tokenizer, device


def force_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# =====================================================================
# POLARITY DATA
# =====================================================================

POLAR_PAIRS = [
    # (pos, neg, name, osgood_factor)
    ("hot", "cold", "hot_cold", "activity"),
    ("big", "small", "big_small", "potency"),
    ("fast", "slow", "fast_slow", "activity"),
    ("happy", "sad", "happy_sad", "evaluation"),
    ("love", "hate", "love_hate", "evaluation"),
    ("light", "dark", "light_dark", "activity"),
    ("strong", "weak", "strong_weak", "potency"),
    ("rich", "poor", "rich_poor", "evaluation"),
    ("good", "bad", "good_bad", "evaluation"),
    ("high", "low", "high_low", "potency"),
    ("new", "old", "new_old", "evaluation"),
    ("open", "closed", "open_closed", "activity"),
    ("soft", "hard", "soft_hard", "potency"),
    ("wet", "dry", "wet_dry", "activity"),
    ("alive", "dead", "alive_dead", "potency"),
    ("beautiful", "ugly", "beautiful_ugly", "evaluation"),
    ("peace", "war", "peace_war", "evaluation"),
    ("create", "destroy", "create_destroy", "potency"),
    ("remember", "forget", "remember_forget", "activity"),
    ("accept", "reject", "accept_reject", "evaluation"),
]

# 5 different templates for each polarity pair
TEMPLATES = [
    lambda w: f"Things that are {w}",
    lambda w: f"The weather is {w}",
    lambda w: f"I feel {w} today",
    lambda w: f"Something is {w}",
    lambda w: f"That seems {w}",
]

# Random word pairs (NOT polar opposites) for control
RANDOM_PAIRS = [
    ("apple", "table", "apple_table"),
    ("river", "chair", "river_chair"),
    ("mountain", "book", "mountain_book"),
    ("ocean", "lamp", "ocean_lamp"),
    ("forest", "phone", "forest_phone"),
    ("garden", "shoe", "garden_shoe"),
    ("bridge", "cup", "bridge_cup"),
    ("castle", "pen", "castle_pen"),
    ("island", "door", "island_door"),
    ("desert", "plate", "desert_plate"),
    ("valley", "knife", "valley_knife"),
    ("cloud", "ring", "cloud_ring"),
    ("thunder", "glass", "thunder_glass"),
    ("winter", "bread", "winter_bread"),
    ("summer", "stone", "summer_stone"),
    ("autumn", "silk", "autumn_silk"),
    ("spring", "gold", "spring_gold"),
    ("night", "wood", "night_wood"),
    ("morning", "iron", "morning_iron"),
    ("evening", "rice", "evening_rice"),
]

# Category sentences for content direction + PCA reference
# ★ Need enough for reliable PCA (at least 50+ sentences)
CATEGORY_SENTENCES = {
    "animal": [
        "The cat sleeps on the mat", "A dog barks at the door",
        "The bird sings a song", "Fish swim in the pond",
        "The horse runs fast", "A bee flies to the flower",
        "The lion roars loudly", "Eagles soar above clouds",
        "The rabbit hops away", "Whales dive deep below",
    ],
    "object": [
        "The chair stands in the corner", "A book lies on the table",
        "The door opens slowly", "The clock ticks on the wall",
        "A pen writes on paper", "The lamp shines brightly",
        "The cup holds hot tea", "A key opens the lock",
        "The mirror reflects light", "The wheel turns smoothly",
    ],
    "nature": [
        "The river flows downhill", "Rain falls from the sky",
        "The sun shines brightly", "Wind blows through trees",
        "Snow covers the ground", "Mountains rise above clouds",
        "The ocean crashes on shore", "Leaves fall in autumn",
        "Flowers bloom in spring", "Stars twinkle at night",
    ],
    "people": [
        "The teacher writes on board", "A doctor helps patients",
        "The student studies hard", "Children play in the park",
        "The artist paints pictures", "Farmers grow crops",
        "The driver steers the car", "A singer performs songs",
        "The builder constructs houses", "Scientists discover truths",
    ],
    "food": [
        "The bread tastes fresh", "Rice feeds many people",
        "Apples grow on trees", "The soup is warm today",
        "Cheese comes from milk", "Corn grows in fields",
        "The cake smells sweet", "Fish provides protein",
        "Oranges contain vitamins", "Water sustains life",
    ],
}


def get_hidden_states(model, tokenizer, device, sentence, n_layers):
    """Get hidden states at last layer for last token position"""
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True, use_cache=False)

    hidden_states = outputs.hidden_states
    last_pos = attention_mask.sum().item() - 1
    # Return only last layer
    h = hidden_states[-1][0, last_pos, :].float().cpu().numpy()
    return h


def get_all_hidden_states(model, tokenizer, device, sentence, n_layers):
    """Get hidden states at all layers for last token position"""
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True, use_cache=False)

    hidden_states = outputs.hidden_states
    last_pos = attention_mask.sum().item() - 1
    result = {}
    for l in range(n_layers):
        h = hidden_states[l][0, last_pos, :].float().cpu().numpy()
        result[l] = h
    return result


def run_exp1_template_control(model, tokenizer, device, n_layers, d_model):
    """Exp1: Does the polarity direction persist across different templates?"""
    print("\n" + "=" * 70)
    print("Exp1: TEMPLATE CONTROL")
    print("  ★★★ Does polarity direction persist across templates? ★★★")
    print("=" * 70)

    last_li = n_layers - 1

    # Collect all sentences for PCA
    ref_sents = []
    for key in CATEGORY_SENTENCES:
        ref_sents.extend(CATEGORY_SENTENCES[key])
    ref_sents = list(set(ref_sents))
    print(f"  Reference sentences for PCA: {len(ref_sents)}")

    # Get hidden states for reference sentences
    print("  Computing reference hidden states...")
    ref_hs = {}
    for s in ref_sents:
        ref_hs[s] = get_hidden_states(model, tokenizer, device, s, last_li)

    # Compute PCA
    H_ref = np.array([ref_hs[s] for s in ref_sents])
    H_ref_c = H_ref - H_ref.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(H_ref_c, full_matrices=False)
    n_high = min(50, Vt.shape[0])
    V_high = Vt[:n_high, :]
    V_low = Vt[n_high:, :]
    mean_h = H_ref.mean(axis=0)

    print(f"  PCA: top50 var={np.sum(S[:n_high]**2)/np.sum(S**2):.3f}")

    # For each polarity pair, get delta vectors under each template
    results = {}
    for pos_w, neg_w, name, factor in POLAR_PAIRS[:10]:  # Use 10 for speed
        deltas_low = []
        deltas_high = []

        for ti, template in enumerate(TEMPLATES):
            pos_s = template(pos_w)
            neg_s = template(neg_w)

            h_pos = get_hidden_states(model, tokenizer, device, pos_s, last_li)
            h_neg = get_hidden_states(model, tokenizer, device, neg_s, last_li)

            delta = h_pos - h_neg
            delta_c = delta  # Don't subtract mean again, already relative

            # Project onto low-PC space
            proj_low = V_low @ delta_c
            proj_high = V_high @ delta_c

            E_low = np.sum(proj_low ** 2) / max(np.sum(delta_c ** 2), 1e-10)

            deltas_low.append(proj_low)
            deltas_high.append(proj_high)

            if ti == 0:
                results[name] = {"E_low_t0": float(E_low)}
            results[name][f"E_low_t{ti}"] = float(E_low)

        # Check if deltas from different templates are parallel
        min_dim = min(len(d) for d in deltas_low)
        D = np.array([d[:min_dim] for d in deltas_low])
        norms = np.linalg.norm(D, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        D_norm = D / norms
        cos_matrix = D_norm @ D_norm.T

        # Mean off-diagonal cosine similarity
        n = cos_matrix.shape[0]
        off_diag = []
        for i in range(n):
            for j in range(i + 1, n):
                off_diag.append(cos_matrix[i, j])
        mean_cos = np.mean(off_diag) if off_diag else 0

        results[name]["template_cross_cos"] = float(mean_cos)
        print(f"  {name:20s}: cross-template cos(low-PC) = {mean_cos:.4f}")

    return results


def run_exp2_polarity_vs_random(model, tokenizer, device, n_layers, d_model):
    """Exp2: Is polarity parallelism special compared to random word pairs?"""
    print("\n" + "=" * 70)
    print("Exp2: POLARITY vs RANDOM CONTROL")
    print("  ★★★ Is polarity parallelism special? ★★★")
    print("=" * 70)

    last_li = n_layers - 1

    # Reference for PCA
    ref_sents = []
    for key in CATEGORY_SENTENCES:
        ref_sents.extend(CATEGORY_SENTENCES[key])
    ref_sents = list(set(ref_sents))

    ref_hs = {}
    for s in ref_sents:
        ref_hs[s] = get_hidden_states(model, tokenizer, device, s, last_li)

    H_ref = np.array([ref_hs[s] for s in ref_sents])
    H_ref_c = H_ref - H_ref.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(H_ref_c, full_matrices=False)
    n_high = min(50, Vt.shape[0])
    V_high = Vt[:n_high, :]
    V_low = Vt[n_high:, :]
    mean_h = H_ref.mean(axis=0)

    # Template for all comparisons
    template = lambda w: f"Things that are {w}"

    # Polarity delta vectors
    print("  Computing polarity delta vectors...")
    polar_deltas_low = []
    polar_deltas_high = []
    polar_names = []

    for pos_w, neg_w, name, factor in POLAR_PAIRS:
        pos_s = template(pos_w)
        neg_s = template(neg_w)
        h_pos = get_hidden_states(model, tokenizer, device, pos_s, last_li)
        h_neg = get_hidden_states(model, tokenizer, device, neg_s, last_li)
        delta = h_pos - h_neg
        proj_low = V_low @ delta
        proj_high = V_high @ delta
        polar_deltas_low.append(proj_low)
        polar_deltas_high.append(proj_high)
        polar_names.append(name)

    # Random delta vectors
    print("  Computing random delta vectors...")
    random_deltas_low = []
    random_deltas_high = []
    random_names = []

    for w1, w2, name in RANDOM_PAIRS:
        s1 = template(w1)
        s2 = template(w2)
        h1 = get_hidden_states(model, tokenizer, device, s1, last_li)
        h2 = get_hidden_states(model, tokenizer, device, s2, last_li)
        delta = h1 - h2
        proj_low = V_low @ delta
        proj_high = V_high @ delta
        random_deltas_low.append(proj_low)
        random_deltas_high.append(proj_high)
        random_names.append(name)

    # Compute pairwise cosine similarities
    min_dim = min(len(d) for d in polar_deltas_low)
    D_polar = np.array([d[:min_dim] for d in polar_deltas_low])
    norms = np.linalg.norm(D_polar, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    D_polar_norm = D_polar / norms
    cos_polar = D_polar_norm @ D_polar_norm.T

    min_dim_r = min(len(d) for d in random_deltas_low)
    D_random = np.array([d[:min_dim_r] for d in random_deltas_low])
    norms_r = np.linalg.norm(D_random, axis=1, keepdims=True)
    norms_r = np.maximum(norms_r, 1e-10)
    D_random_norm = D_random / norms_r
    cos_random = D_random_norm @ D_random_norm.T

    # Also do high-PC
    D_polar_h = np.array(polar_deltas_high)
    norms_ph = np.linalg.norm(D_polar_h, axis=1, keepdims=True)
    norms_ph = np.maximum(norms_ph, 1e-10)
    D_polar_h_norm = D_polar_h / norms_ph
    cos_polar_h = D_polar_h_norm @ D_polar_h_norm.T

    D_random_h = np.array(random_deltas_high)
    norms_rh = np.linalg.norm(D_random_h, axis=1, keepdims=True)
    norms_rh = np.maximum(norms_rh, 1e-10)
    D_random_h_norm = D_random_h / norms_rh
    cos_random_h = D_random_h_norm @ D_random_h_norm.T

    # Extract off-diagonal values
    def get_off_diag(M):
        vals = []
        n = M.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                vals.append(M[i, j])
        return vals

    off_polar_low = get_off_diag(cos_polar)
    off_random_low = get_off_diag(cos_random)
    off_polar_high = get_off_diag(cos_polar_h)
    off_random_high = get_off_diag(cos_random_h)

    print(f"\n  ★ LOW-PC SPACE ★")
    print(f"    Polar pairs: mean cos = {np.mean(off_polar_low):.4f} ± {np.std(off_polar_low):.4f}")
    print(f"    Random pairs: mean cos = {np.mean(off_random_low):.4f} ± {np.std(off_random_low):.4f}")

    print(f"\n  ★ HIGH-PC SPACE ★")
    print(f"    Polar pairs: mean cos = {np.mean(off_polar_high):.4f} ± {np.std(off_polar_high):.4f}")
    print(f"    Random pairs: mean cos = {np.mean(off_random_high):.4f} ± {np.std(off_random_high):.4f}")

    # Statistical test
    from scipy.stats import mannwhitneyu
    try:
        stat, p_val = mannwhitneyu(off_polar_low, off_random_low, alternative='greater')
        print(f"\n  ★ Mann-Whitney U test (low-PC): U={stat:.0f}, p={p_val:.6f}")
        if p_val < 0.05:
            print(f"    → POLAR pairs are SIGNIFICANTLY more parallel than random")
        else:
            print(f"    → NO significant difference between polar and random pairs")
    except Exception as e:
        print(f"    Statistical test failed: {e}")
        p_val = 1.0

    results = {
        "polar_mean_cos_low": float(np.mean(off_polar_low)),
        "polar_std_cos_low": float(np.std(off_polar_low)),
        "random_mean_cos_low": float(np.mean(off_random_low)),
        "random_std_cos_low": float(np.std(off_random_low)),
        "polar_mean_cos_high": float(np.mean(off_polar_high)),
        "random_mean_cos_high": float(np.mean(off_random_high)),
        "mann_whitney_p": float(p_val),
    }

    return results


def run_exp3_orthogonality(model, tokenizer, device, n_layers, d_model):
    """Exp3: Is polarity direction orthogonal to content direction?"""
    print("\n" + "=" * 70)
    print("Exp3: ORTHOGONALITY TEST")
    print("  ★★★ Is polarity direction orthogonal to content? ★★★")
    print("=" * 70)

    last_li = n_layers - 1

    # Get polarity direction (mean of all polar deltas)
    template = lambda w: f"Things that are {w}"
    polar_deltas = []
    for pos_w, neg_w, name, factor in POLAR_PAIRS:
        h_pos = get_hidden_states(model, tokenizer, device, template(pos_w), last_li)
        h_neg = get_hidden_states(model, tokenizer, device, template(neg_w), last_li)
        polar_deltas.append(h_pos - h_neg)

    polar_dir = np.mean(polar_deltas, axis=0)
    polar_dir = polar_dir / np.linalg.norm(polar_dir)

    # Get content directions (between categories)
    cat_hs = {}
    for cat_name, sents in CATEGORY_SENTENCES.items():
        hs_list = []
        for s in sents:
            hs_list.append(get_hidden_states(model, tokenizer, device, s, last_li))
        cat_hs[cat_name] = np.mean(hs_list, axis=0)

    # Compute content direction (animal vs object)
    content_dir_1 = cat_hs["animal"] - cat_hs["object"]
    content_dir_1 = content_dir_1 / np.linalg.norm(content_dir_1)

    content_dir_2 = cat_hs["animal"] - cat_hs["nature"]
    content_dir_2 = content_dir_2 / np.linalg.norm(content_dir_2)

    content_dir_3 = cat_hs["object"] - cat_hs["nature"]
    content_dir_3 = content_dir_3 / np.linalg.norm(content_dir_3)

    # Compute cos between polarity and content
    cos_pc1 = np.dot(polar_dir, content_dir_1)
    cos_pc2 = np.dot(polar_dir, content_dir_2)
    cos_pc3 = np.dot(polar_dir, content_dir_3)

    print(f"  cos(polarity, animal-object) = {cos_pc1:.4f}")
    print(f"  cos(polarity, animal-nature) = {cos_pc2:.4f}")
    print(f"  cos(polarity, object-nature) = {cos_pc3:.4f}")

    # Also check individual polar deltas
    print(f"\n  Individual polar deltas vs content direction (animal-object):")
    for i, (pos_w, neg_w, name, factor) in enumerate(POLAR_PAIRS):
        delta = polar_deltas[i]
        delta_norm = delta / np.linalg.norm(delta)
        cos_content = np.dot(delta_norm, content_dir_1)
        cos_polar = np.dot(delta_norm, polar_dir)
        print(f"    {name:20s}: cos(content)={cos_content:.4f}, cos(polar_dir)={cos_polar:.4f}")

    results = {
        "cos_polar_content1": float(cos_pc1),
        "cos_polar_content2": float(cos_pc2),
        "cos_polar_content3": float(cos_pc3),
        "mean_cos_polar_content": float(np.mean([abs(cos_pc1), abs(cos_pc2), abs(cos_pc3)])),
    }

    return results


def run_exp4_three_factors(model, tokenizer, device, n_layers, d_model):
    """Exp4: Do Osgood's three factors (E, P, A) exist as separate directions?"""
    print("\n" + "=" * 70)
    print("Exp4: THREE-FACTOR TEST (Osgood's EPA)")
    print("  ★★★ Do Evaluation, Potency, Activity exist as separate directions? ★★★")
    print("=" * 70)

    last_li = n_layers - 1

    # Reference for PCA
    ref_sents = []
    for key in CATEGORY_SENTENCES:
        ref_sents.extend(CATEGORY_SENTENCES[key])
    ref_sents = list(set(ref_sents))

    ref_hs = {}
    for s in ref_sents:
        ref_hs[s] = get_hidden_states(model, tokenizer, device, s, last_li)

    H_ref = np.array([ref_hs[s] for s in ref_sents])
    H_ref_c = H_ref - H_ref.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(H_ref_c, full_matrices=False)
    # Use fewer high PCs to leave enough for low-PC space
    n_high = min(10, Vt.shape[0] // 2)
    V_high = Vt[:n_high, :]
    V_low = Vt[n_high:, :]

    if V_low.shape[0] < 5:
        print(f"  WARNING: Not enough PCs for low subspace (V_low has {V_low.shape[0]} rows)")
        print(f"  Reducing n_high to {Vt.shape[0] // 3}")
        n_high = Vt.shape[0] // 3
        V_high = Vt[:n_high, :]
        V_low = Vt[n_high:, :]

    # Group by Osgood factor
    template = lambda w: f"Things that are {w}"
    factor_deltas = defaultdict(list)

    for pos_w, neg_w, name, factor in POLAR_PAIRS:
        h_pos = get_hidden_states(model, tokenizer, device, template(pos_w), last_li)
        h_neg = get_hidden_states(model, tokenizer, device, template(neg_w), last_li)
        delta = h_pos - h_neg
        proj_low = V_low @ delta
        factor_deltas[factor].append(proj_low)

    # Compute mean direction for each factor
    factor_dirs = {}
    for factor, deltas in factor_deltas.items():
        mean_delta = np.mean(deltas, axis=0)
        norm = np.linalg.norm(mean_delta)
        if norm > 1e-10:
            factor_dirs[factor] = mean_delta / norm

    # Compute pairwise cosine between factors
    print("\n  Cosine similarity between factor directions (low-PC space):")
    factor_names = sorted(factor_dirs.keys())
    for i, f1 in enumerate(factor_names):
        for j, f2 in enumerate(factor_names):
            if j > i:
                cos = np.dot(factor_dirs[f1], factor_dirs[f2])
                print(f"    {f1:12s} vs {f2:12s}: cos = {cos:.4f}")

    # PCA of all polar delta vectors in low-PC space
    all_deltas_low = []
    delta_names = []
    for pos_w, neg_w, name, factor in POLAR_PAIRS:
        h_pos = get_hidden_states(model, tokenizer, device, template(pos_w), last_li)
        h_neg = get_hidden_states(model, tokenizer, device, template(neg_w), last_li)
        delta = h_pos - h_neg
        proj_low = V_low @ delta
        all_deltas_low.append(proj_low)
        delta_names.append(name)

    min_dim = min(len(d) for d in all_deltas_low)
    D = np.array([d[:min_dim] for d in all_deltas_low])
    D_centered = D - D.mean(axis=0, keepdims=True)

    # SVD of delta vectors
    U_d, S_d, Vt_d = np.linalg.svd(D_centered, full_matrices=False)

    print(f"\n  SVD of polar delta vectors in low-PC space:")
    print(f"    Singular values: {S_d[:5]}")
    print(f"    Variance explained: {np.cumsum(S_d[:5]**2) / np.sum(S_d**2)}")

    # How many significant dimensions?
    total_var = np.sum(S_d ** 2)
    for k in [1, 2, 3, 5]:
        var_explained = np.sum(S_d[:k] ** 2) / total_var
        print(f"    Top-{k} components explain {var_explained:.3f} of variance")

    # Project each factor onto the first 3 SVD components
    print(f"\n  Factor projections onto top-3 SVD components:")
    for factor, deltas in factor_deltas.items():
        mean_d = np.mean(deltas, axis=0)[:min_dim]
        proj = Vt_d[:3, :] @ mean_d
        print(f"    {factor:12s}: V1={proj[0]:.4f}, V2={proj[1]:.4f}, V3={proj[2]:.4f}")

    results = {
        "n_significant_dims": int(np.sum(S_d > 0.1 * S_d[0])),
        "var_explained_top1": float(np.sum(S_d[:1] ** 2) / total_var),
        "var_explained_top2": float(np.sum(S_d[:2] ** 2) / total_var),
        "var_explained_top3": float(np.sum(S_d[:3] ** 2) / total_var),
    }

    # Factor cosine similarities
    for i, f1 in enumerate(factor_names):
        for j, f2 in enumerate(factor_names):
            if j > i:
                cos = np.dot(factor_dirs[f1], factor_dirs[f2])
                results[f"cos_{f1}_{f2}"] = float(cos)

    return results


# =====================================================================
# MAIN
# =====================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python phase189_polarity_validation.py <model_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    t0 = time.time()

    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    n_layers, d_model, vocab_size = info.n_layers, info.d_model, info.vocab_size
    print(f"\nModel: {info.model_class}, Layers={n_layers}, d_model={d_model}, vocab={vocab_size}")

    # Run all experiments
    print("\n" + "=" * 70)
    print("Running Exp1: Template Control...")
    exp1_results = run_exp1_template_control(model, tokenizer, device, n_layers, d_model)

    print("\n" + "=" * 70)
    print("Running Exp2: Polarity vs Random Control...")
    exp2_results = run_exp2_polarity_vs_random(model, tokenizer, device, n_layers, d_model)

    print("\n" + "=" * 70)
    print("Running Exp3: Orthogonality Test...")
    exp3_results = run_exp3_orthogonality(model, tokenizer, device, n_layers, d_model)

    print("\n" + "=" * 70)
    print("Running Exp4: Three-Factor Test (EPA)...")
    exp4_results = run_exp4_three_factors(model, tokenizer, device, n_layers, d_model)

    # Save results
    all_results = {
        "model": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "vocab_size": vocab_size,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M"),
        "exp1_template_control": exp1_results,
        "exp2_polarity_vs_random": exp2_results,
        "exp3_orthogonality": exp3_results,
        "exp4_three_factors": exp4_results,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = f"tests/glm5_temp/phase189_{model_name}_{ts}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path}")

    # ===== SUMMARY =====
    print("\n" + "#" * 70)
    print(f"PHASE 189 SUMMARY")
    print("#" * 70)

    print("\n★★★ Exp1: Template Control ★★★")
    for name in sorted(exp1_results.keys()):
        if isinstance(exp1_results[name], dict) and "template_cross_cos" in exp1_results[name]:
            print(f"  {name:20s}: cross-template cos = {exp1_results[name]['template_cross_cos']:.4f}")

    print("\n★★★ Exp2: Polarity vs Random ★★★")
    print(f"  Polar pairs cos (low-PC): {exp2_results.get('polar_mean_cos_low', 0):.4f}")
    print(f"  Random pairs cos (low-PC): {exp2_results.get('random_mean_cos_low', 0):.4f}")
    print(f"  Polar pairs cos (high-PC): {exp2_results.get('polar_mean_cos_high', 0):.4f}")
    print(f"  Random pairs cos (high-PC): {exp2_results.get('random_mean_cos_high', 0):.4f}")
    p = exp2_results.get('mann_whitney_p', 1)
    print(f"  Mann-Whitney p = {p:.6f} → {'SIGNIFICANT' if p < 0.05 else 'NOT significant'}")

    print("\n★★★ Exp3: Orthogonality ★★★")
    print(f"  Mean |cos(polarity, content)| = {exp3_results.get('mean_cos_polar_content', 0):.4f}")

    print("\n★★★ Exp4: Three-Factor (EPA) ★★★")
    print(f"  Variance explained: top1={exp4_results.get('var_explained_top1', 0):.3f}, "
          f"top2={exp4_results.get('var_explained_top2', 0):.3f}, "
          f"top3={exp4_results.get('var_explained_top3', 0):.3f}")

    # Release model
    release_model(model)
    force_cleanup()

    elapsed = time.time() - t0
    print(f"\n{'#' * 70}")
    print(f"Phase 189 COMPLETE! Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'#' * 70}")


if __name__ == "__main__":
    main()
