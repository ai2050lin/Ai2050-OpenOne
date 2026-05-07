"""
Phase 74: Concept Geometry — 知识编码的基本结构
================================================

范式转移: 从局部微分几何 → 概念状态空间几何

Phase 68-73的教训:
- 标量输出梯度天然低秩 — 不能代表网络动力学
- Jacobian几乎满秩(~650) — 局部线性化无法揭示结构
- CKA/cosine在高维空间不可靠 — 需要随机基线

Phase 74的新方法:
- 不研究梯度/Jacobian，直接研究隐藏状态h_l的几何结构
- 核心问题: 概念如何在状态空间中组织?
- 三个子问题:
  A. 概念聚类 — 语义相似的词是否在h空间中聚类?
  B. 属性代数 — "红+苹果"如何组合?
  C. 层级结构 — "水果"和"苹果"的关系

关键方法论改进:
1. 所有结论必须与随机基线比较
2. 使用内部交叉验证(split-half)
3. 量化效应量(effect size)而非仅报告p-value
4. 区分"可解释的结构"和"高维默认行为"

测试模型: GPT-2 Small (float32) — 与Phase 70-73一致
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import torch, numpy as np, gc, argparse, time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

GPT2_PATH = "D:/develop/model/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"


# ============================================================
# 概念分类体系 — 用于测试聚类结构
# ============================================================

# 语义类别: 每类10个词
SEMANTIC_CATEGORIES = {
    "animals":  ["cat", "dog", "bird", "fish", "horse", "bear", "snake", "frog", "fox", "wolf"],
    "fruits":   ["apple", "banana", "orange", "grape", "peach", "pear", "cherry", "mango", "lemon", "plum"],
    "colors":   ["red", "blue", "green", "yellow", "black", "white", "pink", "brown", "gray", "purple"],
    "tools":    ["hammer", "wrench", "drill", "saw", "axe", "chisel", "pliers", "screwdriver", "ruler", "knife"],
    "body":     ["head", "hand", "foot", "arm", "leg", "eye", "ear", "nose", "mouth", "heart"],
    "weather":  ["rain", "snow", "wind", "storm", "cloud", "fog", "hail", "frost", "sun", "heat"],
    "vehicles": ["car", "bus", "train", "plane", "boat", "truck", "bike", "ship", "taxi", "van"],
    "clothing": ["shirt", "pants", "dress", "coat", "hat", "shoe", "sock", "glove", "belt", "scarf"],
}

# 属性词 — 用于测试属性-概念组合
COLOR_ATTRS = ["red", "blue", "green", "yellow", "black", "white"]
SIZE_ATTRS  = ["big", "small", "tiny", "huge", "long", "short"]
TEXTURE_ATTRS = ["soft", "hard", "smooth", "rough", "sharp", "flat"]

# 可着色/可尺寸化的概念
COLORABLE_CONCEPTS = ["apple", "car", "house", "flower", "shirt", "bird", "sky", "stone"]
SIZEABLE_CONCEPTS  = ["dog", "house", "tree", "car", "stone", "river", "mountain", "book"]

# 层级关系 — 用于测试层级结构
HIERARCHY = {
    "apple→fruit":     ("apple", "fruit"),
    "banana→fruit":    ("banana", "fruit"),
    "orange→fruit":    ("orange", "fruit"),
    "cat→animal":      ("cat", "animal"),
    "dog→animal":      ("dog", "animal"),
    "horse→animal":    ("horse", "animal"),
    "car→vehicle":     ("car", "vehicle"),
    "bus→vehicle":     ("bus", "vehicle"),
    "train→vehicle":   ("train", "vehicle"),
    "hammer→tool":     ("hammer", "tool"),
    "drill→tool":      ("drill", "tool"),
    "fruit→food":      ("fruit", "food"),
    "animal→life":     ("animal", "life"),
    "vehicle→machine": ("vehicle", "machine"),
}


def load_gpt2_float32():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    print("[Phase74] Loading GPT-2 Small float32...")
    tokenizer = GPT2Tokenizer.from_pretrained(GPT2_PATH, local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(
        GPT2_PATH, torch_dtype=torch.float32, local_files_only=True,
    )
    if torch.cuda.is_available():
        model = model.to('cuda')
    model.eval()
    device = next(model.parameters()).device
    n_layers = model.config.n_layer
    d_model = model.config.n_embd
    print(f"[Phase74] GPT-2: {n_layers} layers, d={d_model}, device={device}")
    return model, tokenizer, device


def get_hidden_states(model, tokenizer, device, text, layers=None):
    """获取所有层的隐藏状态"""
    toks = tokenizer(text, return_tensors="pt").to(device)
    input_ids = toks.input_ids

    outputs = model(input_ids=input_ids, output_hidden_states=True)

    if layers is None:
        layers = range(len(outputs.hidden_states))

    # 返回最后token的隐藏状态
    result = {}
    for li in layers:
        h = outputs.hidden_states[li][0, -1, :].detach().cpu().numpy()
        result[li] = h

    return result


def get_pooled_hidden_states(model, tokenizer, device, words, template="The {}"):
    """获取一组词的隐藏状态"""
    states = {}
    for li in range(13):  # 0-12
        states[li] = []

    for word in words:
        text = template.format(word)
        hs = get_hidden_states(model, tokenizer, device, text)
        for li in hs:
            states[li].append(hs[li])

    for li in states:
        states[li] = np.array(states[li])

    return states


# ============================================================
# Experiment A: Concept Clustering — 语义类别在h空间中是否聚类?
# ============================================================
def experiment_a(model, tokenizer, device):
    """
    ★★★ 核心测试: 概念聚类结构

    如果语义类别在隐藏空间中形成聚类:
    → 类内距离 < 类间距离
    → 这是最基本的知识编码结构

    关键: 与随机基线比较!
    - 随机分配: 将80个词随机分成8组
    - 如果语义聚类的效果量 ≈ 随机分配 → 不存在真正的聚类
    - 如果语义聚类显著优于随机 → 存在语义编码结构
    """
    print("\n" + "="*70)
    print("Experiment A: Concept Clustering — 语义类别聚类测试")
    print("="*70)

    # 收集所有词和标签
    all_words = []
    all_labels = []
    for cat, words in SEMANTIC_CATEGORIES.items():
        for w in words:
            all_words.append(w)
            all_labels.append(cat)

    n_words = len(all_words)
    n_cats = len(SEMANTIC_CATEGORIES)
    print(f"  {n_words} words in {n_cats} categories")

    # 获取隐藏状态
    print("  Collecting hidden states...")
    template = "The {}"
    states = get_pooled_hidden_states(model, tokenizer, device, all_words, template)

    # 对每层分析
    test_layers = [0, 3, 6, 9, 11, 12]

    for li in test_layers:
        if li not in states:
            continue

        H = states[li]  # [n_words, d_model]
        print(f"\n--- Layer {li} ---")

        # ★★★ 1. 类内距离 vs 类间距离
        intra_dists = []
        inter_dists = []

        for cat in SEMANTIC_CATEGORIES:
            cat_idx = [i for i, l in enumerate(all_labels) if l == cat]
            other_idx = [i for i, l in enumerate(all_labels) if l != cat]

            cat_states = H[cat_idx]

            # 类内距离
            for i in range(len(cat_idx)):
                for j in range(i+1, len(cat_idx)):
                    intra_dists.append(np.linalg.norm(cat_states[i] - cat_states[j]))

            # 类间距离 (采样, 避免O(n²))
            for idx in cat_idx:
                sample_other = np.random.choice(other_idx, min(5, len(other_idx)), replace=False)
                for oidx in sample_other:
                    inter_dists.append(np.linalg.norm(H[idx] - H[oidx]))

        mean_intra = np.mean(intra_dists)
        mean_inter = np.mean(inter_dists)
        ratio = mean_inter / max(mean_intra, 1e-10)

        print(f"  Intra-class dist: {mean_intra:.4f}")
        print(f"  Inter-class dist: {mean_inter:.4f}")
        print(f"  Ratio (inter/intra): {ratio:.3f}")

        # ★★★ 2. 随机基线: 随机分配标签
        n_random = 100
        random_ratios = []
        for _ in range(n_random):
            rand_labels = np.random.permutation(all_labels).tolist()
            r_intra = []
            r_inter = []
            for cat in SEMANTIC_CATEGORIES:
                cat_idx = [i for i, l in enumerate(rand_labels) if l == cat]
                other_idx = [i for i, l in enumerate(rand_labels) if l != cat]
                cat_states = H[cat_idx]
                for i in range(len(cat_idx)):
                    for j in range(i+1, len(cat_idx)):
                        r_intra.append(np.linalg.norm(cat_states[i] - cat_states[j]))
                for idx in cat_idx:
                    sample_other = np.random.choice(other_idx, min(3, len(other_idx)), replace=False)
                    for oidx in sample_other:
                        r_inter.append(np.linalg.norm(H[idx] - H[oidx]))

            r_mean_intra = np.mean(r_intra)
            r_mean_inter = np.mean(r_inter)
            random_ratios.append(r_mean_inter / max(r_mean_intra, 1e-10))

        rand_mean = np.mean(random_ratios)
        rand_std = np.std(random_ratios)
        z_score = (ratio - rand_mean) / max(rand_std, 1e-10)

        print(f"  Random baseline: ratio={rand_mean:.3f}±{rand_std:.3f}")
        print(f"  Z-score: {z_score:.1f}σ")
        if z_score > 3:
            print(f"  ★★★ 语义聚类显著超过随机基线!")
        elif z_score > 1.5:
            print(f"  语义聚类弱于但高于随机基线")
        else:
            print(f"  语义聚类不显著 — 可能是高维默认行为")

        # ★★★ 3. Silhouette Score — 标准聚类质量指标
        from sklearn.metrics import silhouette_score
        sil = silhouette_score(H, all_labels)

        # 随机基线
        rand_sils = []
        for _ in range(50):
            rand_labels = np.random.permutation(all_labels)
            rand_sils.append(silhouette_score(H, rand_labels))

        rand_sil_mean = np.mean(rand_sils)
        rand_sil_std = np.std(rand_sils)
        sil_z = (sil - rand_sil_mean) / max(rand_sil_std, 1e-10)

        print(f"  Silhouette score: {sil:.4f}")
        print(f"  Random silhouette: {rand_sil_mean:.4f}±{rand_sil_std:.4f}")
        print(f"  Silhouette Z-score: {sil_z:.1f}σ")

        # ★★★ 4. 类中心之间的距离矩阵
        cat_centers = {}
        for cat in SEMANTIC_CATEGORIES:
            cat_idx = [i for i, l in enumerate(all_labels) if l == cat]
            cat_centers[cat] = np.mean(H[cat_idx], axis=0)

        # 语义相关的类别应该更近
        # 例如: animals-body, fruits-colors 应该比 animals-vehicles 更近
        sem_related = [("animals", "body"), ("fruits", "colors"),
                       ("vehicles", "tools"), ("weather", "clothing")]
        sem_unrelated = [("animals", "vehicles"), ("fruits", "tools"),
                         ("colors", "body"), ("weather", "tools")]

        related_dists = []
        for c1, c2 in sem_related:
            d = np.linalg.norm(cat_centers[c1] - cat_centers[c2])
            related_dists.append(d)

        unrelated_dists = []
        for c1, c2 in sem_unrelated:
            d = np.linalg.norm(cat_centers[c1] - cat_centers[c2])
            unrelated_dists.append(d)

        print(f"  Semantically related cat dist: {np.mean(related_dists):.4f}")
        print(f"  Semantically unrelated cat dist: {np.mean(unrelated_dists):.4f}")
        print(f"  Related/Unrelated ratio: {np.mean(related_dists)/max(np.mean(unrelated_dists),1e-10):.3f}")

    return states


# ============================================================
# Experiment B: Attribute Algebra — 属性如何与概念组合?
# ============================================================
def experiment_b(model, tokenizer, device):
    """
    ★★★ 核心测试: 属性-概念组合规则

    "红苹果" = "红" + "苹果"? 还是 "红" × "苹果"? 还是其他?

    测试三种假设:
    1. 加法: h("红苹果") ≈ h("苹果") + α·h("红")
    2. 乘法: h("红苹果") ≈ h("苹果") ⊙ (1 + β·h("红"))
    3. 偏移: h("红苹果") ≈ h("苹果") + δ(类别="水果")

    关键: 与随机方向基线比较!
    """
    print("\n" + "="*70)
    print("Experiment B: Attribute Algebra — 属性组合规则")
    print("="*70)

    d_model = model.config.n_embd

    # ---- Part 1: 颜色属性 ----
    print("\n=== Color Attributes ===")

    # 收集所有需要的隐藏状态
    color_words = COLOR_ATTRS
    concept_words = COLORABLE_CONCEPTS

    # "The red apple", "The apple", "The red" 等
    print("  Collecting attribute states...")
    color_states = get_pooled_hidden_states(model, tokenizer, device,
                                            color_words, "The {}")
    concept_states = get_pooled_hidden_states(model, tokenizer, device,
                                              concept_words, "The {}")

    # 组合词: "The red apple"
    combo_texts = [f"The {c} {n}" for c in COLOR_ATTRS for n in COLORABLE_CONCEPTS]
    combo_words = [f"{c}_{n}" for c in COLOR_ATTRS for n in COLORABLE_CONCEPTS]
    combo_states = get_pooled_hidden_states(model, tokenizer, device,
                                            combo_words, "{}")

    # 手动构造组合文本
    combo_states_direct = {}
    for li in range(13):
        combo_states_direct[li] = []

    for c in COLOR_ATTRS:
        for n in COLORABLE_CONCEPTS:
            text = f"The {c} {n}"
            hs = get_hidden_states(model, tokenizer, device, text)
            for li in hs:
                combo_states_direct[li].append(hs[li])

    for li in combo_states_direct:
        combo_states_direct[li] = np.array(combo_states_direct[li])

    test_layers = [0, 3, 6, 9, 11, 12]

    for li in test_layers:
        if li not in color_states or li not in concept_states:
            continue
        if li not in combo_states_direct:
            continue

        print(f"\n--- Layer {li} ---")

        C = color_states[li]      # [n_colors, d]
        N = concept_states[li]    # [n_concepts, d]
        CN = combo_states_direct[li]  # [n_colors*n_concepts, d]

        # 测试加法假设: h(c,n) ≈ h(n) + α·(h(c) - h_ref)
        # 其中h_ref是"基准"隐藏状态 (无颜色修饰的概念)

        # 对每个(颜色, 概念)对
        add_errors = []
        rand_errors = []

        idx = 0
        for ci, color in enumerate(COLOR_ATTRS):
            for ni, concept in enumerate(COLORABLE_CONCEPTS):
                h_combo = CN[idx]       # h("red apple")
                h_concept = N[ni]       # h("apple")
                h_color = C[ci]         # h("red")

                # 加法假设: h("red apple") ≈ h("apple") + α·h("red")
                # 最优α = (h_combo - h_concept) · h_color / ||h_color||²
                diff = h_combo - h_concept
                alpha = np.dot(diff, h_color) / max(np.dot(h_color, h_color), 1e-10)
                h_pred_add = h_concept + alpha * h_color
                err_add = np.linalg.norm(h_combo - h_pred_add) / max(np.linalg.norm(h_combo), 1e-10)

                # 随机方向基线: 用随机方向代替h_color
                rand_dir = np.random.randn(d_model)
                alpha_r = np.dot(diff, rand_dir) / max(np.dot(rand_dir, rand_dir), 1e-10)
                h_pred_rand = h_concept + alpha_r * rand_dir
                err_rand = np.linalg.norm(h_combo - h_pred_rand) / max(np.linalg.norm(h_combo), 1e-10)

                add_errors.append(err_add)
                rand_errors.append(err_rand)
                idx += 1

        mean_add_err = np.mean(add_errors)
        mean_rand_err = np.mean(rand_errors)
        improvement = (mean_rand_err - mean_add_err) / mean_rand_err * 100

        print(f"  Additive model error: {mean_add_err:.4f}")
        print(f"  Random direction error: {mean_rand_err:.4f}")
        print(f"  Improvement over random: {improvement:.1f}%")

        # ★★★ 偏移一致性测试
        # 如果h("red apple") - h("apple") 对所有概念都相似
        # → "红色"有一个通用的偏移方向
        print(f"\n  Offset Consistency Test:")

        # 对每个颜色, 收集所有偏移
        color_offsets = defaultdict(list)
        idx = 0
        for ci, color in enumerate(COLOR_ATTRS):
            for ni, concept in enumerate(COLORABLE_CONCEPTS):
                offset = CN[idx] - N[ni]  # h("red apple") - h("apple")
                color_offsets[color].append(offset)
                idx += 1

        # 同一颜色的偏移是否一致?
        for color in COLOR_ATTRS[:4]:  # 只显示前4个
            offsets = np.array(color_offsets[color])
            mean_offset = np.mean(offsets, axis=0)
            # 归一化余弦
            cosines = [np.dot(o, mean_offset) / max(np.linalg.norm(o) * np.linalg.norm(mean_offset), 1e-10)
                       for o in offsets]
            mean_cos = np.mean(cosines)
            print(f"    {color:8s}: offset consistency cos={mean_cos:.4f}")

        # 随机基线: 随机对之间的偏移一致性
        all_offsets = []
        idx = 0
        for ci, color in enumerate(COLOR_ATTRS):
            for ni, concept in enumerate(COLORABLE_CONCEPTS):
                all_offsets.append(CN[idx] - N[ni])
                idx += 1

        all_offsets = np.array(all_offsets)
        rand_consistencies = []
        for _ in range(100):
            idx1, idx2 = np.random.choice(len(all_offsets), 2, replace=False)
            cos_r = np.dot(all_offsets[idx1], all_offsets[idx2]) / max(
                np.linalg.norm(all_offsets[idx1]) * np.linalg.norm(all_offsets[idx2]), 1e-10)
            rand_consistencies.append(cos_r)

        print(f"    Random offset consistency: {np.mean(rand_consistencies):.4f}±{np.std(rand_consistencies):.4f}")

        # ★★★ 不同颜色的偏移方向是否不同?
        color_mean_offsets = {}
        for color in COLOR_ATTRS:
            offsets = np.array(color_offsets[color])
            color_mean_offsets[color] = np.mean(offsets, axis=0)

        print(f"\n  Cross-Color Offset Angles:")
        for i, c1 in enumerate(COLOR_ATTRS[:4]):
            for c2 in COLOR_ATTRS[i+1:5]:
                cos_cc = np.dot(color_mean_offsets[c1], color_mean_offsets[c2]) / max(
                    np.linalg.norm(color_mean_offsets[c1]) * np.linalg.norm(color_mean_offsets[c2]), 1e-10)
                print(f"    {c1:8s} vs {c2:8s}: cos={cos_cc:.4f}")


# ============================================================
# Experiment C: Hierarchical Structure — 层级关系
# ============================================================
def experiment_c(model, tokenizer, device):
    """
    ★★★ 核心测试: 概念的层级结构

    "苹果"和"水果"的关系是什么?
    - 子集方向: h("水果") ≈ h("苹果")的平均 (上位概念=下位概念的质心)
    - 包含偏移: h("水果") ≈ h("苹果") + δ(hierarchy) (层级方向)
    - 正交结构: 层级维度与实例维度正交

    关键: "水果"是否编码在"苹果"的某个特定方向上?
    """
    print("\n" + "="*70)
    print("Experiment C: Hierarchical Structure — 层级关系")
    print("="*70)

    d_model = model.config.n_embd

    # 收集所有需要的词
    all_words = set()
    for key, (w1, w2) in HIERARCHY.items():
        all_words.add(w1)
        all_words.add(w2)

    all_words = list(all_words)
    print(f"  {len(all_words)} unique words in hierarchy test")

    # 获取隐藏状态
    word_states = get_pooled_hidden_states(model, tokenizer, device,
                                            all_words, "The {}")

    test_layers = [0, 3, 6, 9, 11, 12]

    for li in test_layers:
        if li not in word_states:
            continue

        H = word_states[li]
        word_to_idx = {w: i for i, w in enumerate(all_words)}

        print(f"\n--- Layer {li} ---")

        # ★★★ 1. 子集方向测试: 上位概念是否≈下位概念的均值?
        # "水果" ≈ mean("苹果", "香蕉", "橘子") ?
        categories = {
            "fruit":  ["apple", "banana", "orange"],
            "animal": ["cat", "dog", "horse"],
            "vehicle":["car", "bus", "train"],
            "tool":   ["hammer", "drill"],
        }

        for cat_name, members in categories.items():
            if cat_name not in word_to_idx:
                continue
            cat_idx = word_to_idx[cat_name]
            h_cat = H[cat_idx]

            member_idx = [word_to_idx[m] for m in members if m in word_to_idx]
            if len(member_idx) < 2:
                continue

            h_members = H[member_idx]
            h_mean = np.mean(h_members, axis=0)

            cos_cat_mean = np.dot(h_cat, h_mean) / max(
                np.linalg.norm(h_cat) * np.linalg.norm(h_mean), 1e-10)

            # 随机基线: 随机3个词的均值
            rand_cos = []
            for _ in range(100):
                rand_idx = np.random.choice(len(all_words), len(member_idx), replace=False)
                h_rand_mean = np.mean(H[rand_idx], axis=0)
                cos_r = np.dot(h_cat, h_rand_mean) / max(
                    np.linalg.norm(h_cat) * np.linalg.norm(h_rand_mean), 1e-10)
                rand_cos.append(cos_r)

            z_score = (cos_cat_mean - np.mean(rand_cos)) / max(np.std(rand_cos), 1e-10)

            print(f"  {cat_name:8s}: cos(category, member_mean)={cos_cat_mean:.4f}, "
                  f"random={np.mean(rand_cos):.4f}±{np.std(rand_cos):.4f}, z={z_score:.1f}σ")

        # ★★★ 2. 层级偏移方向测试
        # h("fruit") - h("apple") 是否与 h("animal") - h("cat") 方向相似?
        # 如果相似 → 存在通用"抽象化方向"
        print(f"\n  Hierarchy Offset Direction Test:")

        hierarchy_offsets = {}
        for key, (sub, sup) in HIERARCHY.items():
            if sub in word_to_idx and sup in word_to_idx:
                offset = H[word_to_idx[sup]] - H[word_to_idx[sub]]
                hierarchy_offsets[key] = offset

        if len(hierarchy_offsets) >= 2:
            offset_keys = list(hierarchy_offsets.keys())
            offset_vals = np.array([hierarchy_offsets[k] for k in offset_keys])

            # 偏移方向的成对余弦
            print(f"  Pairwise cosines of hierarchy offsets:")
            for i in range(min(len(offset_keys), 6)):
                for j in range(i+1, min(len(offset_keys), 6)):
                    cos_ij = np.dot(offset_vals[i], offset_vals[j]) / max(
                        np.linalg.norm(offset_vals[i]) * np.linalg.norm(offset_vals[j]), 1e-10)
                    print(f"    {offset_keys[i]:20s} vs {offset_keys[j]:20s}: cos={cos_ij:.4f}")

            # 平均偏移一致性
            mean_offset = np.mean(offset_vals, axis=0)
            cos_with_mean = [np.dot(o, mean_offset) / max(
                np.linalg.norm(o) * np.linalg.norm(mean_offset), 1e-10)
                for o in offset_vals]
            print(f"  Mean offset consistency: {np.mean(cos_with_mean):.4f}")

            # 随机基线
            rand_pair_cos = []
            for _ in range(100):
                i1, i2 = np.random.choice(len(all_words), 2, replace=False)
                j1, j2 = np.random.choice(len(all_words), 2, replace=False)
                r_off1 = H[i1] - H[i2]
                r_off2 = H[j1] - H[j2]
                cos_r = np.dot(r_off1, r_off2) / max(
                    np.linalg.norm(r_off1) * np.linalg.norm(r_off2), 1e-10)
                rand_pair_cos.append(cos_r)
            print(f"  Random offset pair cosine: {np.mean(rand_pair_cos):.4f}±{np.std(rand_pair_cos):.4f}")

        # ★★★ 3. 上位/下位概念与属性词的余弦
        # 如果"苹果"接近"红色"而"水果"不接近 → 下位概念携带属性信息
        print(f"\n  Attribute Proximity Test:")
        attr_words = {"color": "red", "size": "big", "taste": "sweet"}
        test_pairs = [
            ("apple", "fruit"),
            ("cat", "animal"),
            ("car", "vehicle"),
        ]

        for sub, sup in test_pairs:
            if sub not in word_to_idx or sup not in word_to_idx:
                continue
            h_sub = H[word_to_idx[sub]]
            h_sup = H[word_to_idx[sup]]

            for attr_type, attr_word in attr_words.items():
                if attr_word not in word_to_idx:
                    continue
                h_attr = H[word_to_idx[attr_word]]

                cos_sub = np.dot(h_sub, h_attr) / max(
                    np.linalg.norm(h_sub) * np.linalg.norm(h_attr), 1e-10)
                cos_sup = np.dot(h_sup, h_attr) / max(
                    np.linalg.norm(h_sup) * np.linalg.norm(h_attr), 1e-10)

                print(f"    {sub:8s}→{attr_word}: cos={cos_sub:.4f}, "
                      f"{sup:8s}→{attr_word}: cos={cos_sup:.4f}, "
                      f"diff={cos_sub-cos_sup:+.4f}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="a", choices=["a", "b", "c", "all"])
    args = parser.parse_args()

    model, tokenizer, device = load_gpt2_float32()

    try:
        if args.exp in ["a", "all"]:
            experiment_a(model, tokenizer, device)

        if args.exp in ["b", "all"]:
            experiment_b(model, tokenizer, device)

        if args.exp in ["c", "all"]:
            experiment_c(model, tokenizer, device)
    finally:
        del model
        torch.cuda.empty_cache()
        gc.collect()
        print("\n[Phase74] Done. GPU memory released.")
