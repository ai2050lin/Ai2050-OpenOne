"""
Phase 63: 子空间稳定性验证 + 编码结构深度分析
=================================================

四个方案:
  Part 1 (63a): Grassmann距离验证2D子空间稳定性 — 决定性实验
  Part 2 (63b): 跨轴子空间正交性 + 2D平面内精细结构
  Part 3 (63c): 多维度分类准确率曲线 — 信息分散程度量化
  Part 4 (63d): 类别共享+个体差异分解 — 验证相对编码假设

用法:
  python tests/glm5/phase63_subspace_stability.py --model qwen3 --part 1
  python tests/glm5/phase63_subspace_stability.py --model qwen3 --part all
"""

import sys, os, json, argparse, gc, time, warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization"""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)

warnings.filterwarnings("ignore")

PROJECT = Path("d:/Ai2050/TransformerLens-Project")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "tests" / "glm5"))

RESULT_DIR = PROJECT / "results" / "subspace_topology"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

model_name_global = ""


def log_time(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    safe_msg = msg.encode('ascii', errors='replace').decode('ascii')
    print(f"[{ts}] {safe_msg}", flush=True)


# =====================================================================
# 模型加载 (BF16 + device_map="auto" + Flash Attention)
# =====================================================================

def load_model_bf16(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from model_utils import MODEL_CONFIGS

    cfg = MODEL_CONFIGS[model_name]
    log_time(f"Loading {model_name} (bf16 + device_map=auto + flash)...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="flash_attention_2",
        )
        log_time(f"{model_name} loaded with flash_attention_2")
    except Exception as e:
        log_time(f"Flash attention failed ({e}), falling back to eager")
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="eager",
        )

    model.eval()
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log_time(f"{model_name}: device={device}, GPU={gpu_mem:.2f}GB")
    return model, tokenizer, device


import torch


# =====================================================================
# 通用: 收集hidden states
# =====================================================================

def collect_hidden_states(model, tokenizer, device, sentences, target_layers, batch_size=4):
    from model_utils import get_model_info
    info = get_model_info(model, model_name_global)
    all_hidden = {li: [] for li in target_layers}

    for batch_start in range(0, len(sentences), batch_size):
        batch_sents = sentences[batch_start:batch_start + batch_size]
        inputs = tokenizer(batch_sents, return_tensors="pt", padding=True,
                           truncation=True, max_length=64)
        input_device = next(model.parameters()).device
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            output_hidden_states=True)

        for li in target_layers:
            hs = outputs.hidden_states[li].float().cpu().numpy()
            for i in range(len(batch_sents)):
                mask = inputs["attention_mask"][i].numpy()
                last_pos = np.where(mask > 0)[0][-1]
                all_hidden[li].append(hs[i, last_pos])

        if batch_start % (batch_size * 10) == 0:
            log_time(f"  Collected {batch_start + len(batch_sents)}/{len(sentences)}")

    for li in target_layers:
        all_hidden[li] = np.array(all_hidden[li])
    return all_hidden


# =====================================================================
# 通用: 子空间提取与度量
# =====================================================================

def extract_subspace(activations, n_dims=2):
    """提取n_dims维子空间的正交基"""
    mean = activations.mean(axis=0, keepdims=True)
    centered = activations - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    return Vt[:n_dims].T  # [d_model, n_dims] orthonormal basis


def extract_subspace_with_info(activations, n_dims=10):
    """提取子空间基 + 奇异值信息"""
    mean = activations.mean(axis=0, keepdims=True)
    centered = activations - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    basis = Vt[:n_dims].T  # [d_model, n_dims]
    return basis, S[:n_dims]


def grassmann_distance(S1, S2):
    """
    计算两个子空间之间的Grassmann距离
    
    S1, S2: [d_model, k] orthonormal basis matrices
    返回: Grassmann距离 (弧度)
    """
    from scipy.linalg import subspace_angles
    angles = subspace_angles(S1, S2)  # 长度=min(k1, k2)的主角向量
    return float(np.sqrt(np.sum(angles ** 2)))


def principal_angles(S1, S2):
    """计算两个子空间之间的主角(度数)"""
    from scipy.linalg import subspace_angles
    angles_rad = subspace_angles(S1, S2)
    return [float(a) for a in np.degrees(angles_rad)]


def subspace_overlap_principal_angle(S1, S2):
    """用principal angle方法计算子空间overlap"""
    M = S1.T @ S2
    svals = np.linalg.svd(M, compute_uv=False)
    svals = np.clip(svals, 0, 1)
    return float(np.mean(svals ** 2))


# =====================================================================
# 词集定义
# =====================================================================

# 温度轴: 两套词集
TEMPERATURE_WORDS_V1 = ["freezing", "frigid", "cold", "cool", "lukewarm",
                        "warm", "hot", "scorching", "blazing", "searing"]
TEMPERATURE_WORDS_V2 = ["icy", "chilly", "frosty", "tepid", "mild",
                        "toasty", "boiling", "sweltering", "scalding", "infernal"]
# 扩展温度词 (用于精细结构)
TEMPERATURE_WORDS_EXTENDED = [
    "freezing", "icy", "frigid", "cold", "chilly", "frosty",
    "cool", "tepid", "lukewarm", "mild",
    "warm", "toasty", "hot", "boiling", "scorching",
    "sweltering", "blazing", "searing", "scalding", "infernal"
]

# 大小轴: 两套词集
SIZE_WORDS_V1 = ["microscopic", "tiny", "small", "moderate", "medium",
                 "large", "big", "huge", "enormous", "gigantic"]
SIZE_WORDS_V2 = ["minuscule", "puny", "petite", "average", "standard",
                 "substantial", "bulky", "massive", "colossal", "immense"]
SIZE_WORDS_EXTENDED = [
    "microscopic", "minuscule", "tiny", "puny", "petite",
    "small", "moderate", "average", "medium", "standard",
    "large", "substantial", "big", "bulky", "huge",
    "enormous", "massive", "gigantic", "colossal", "immense"
]

# 速度轴: 两套词集
SPEED_WORDS_V1 = ["glacial", "sluggish", "slow", "steady", "moderate",
                  "fast", "quick", "rapid", "swift", "lightning"]
SPEED_WORDS_V2 = ["crawling", "leisurely", "plodding", "cruising", "mid-tempo",
                  "brisk", "hasty", "fleet", "breakneck", "instantaneous"]
SPEED_WORDS_EXTENDED = [
    "glacial", "crawling", "sluggish", "leisurely", "pluggish",
    "slow", "steady", "cruising", "moderate", "mid-tempo",
    "fast", "brisk", "quick", "hasty", "rapid",
    "swift", "fleet", "lightning", "breakneck", "instantaneous"
]

# 情感轴: 两套词集
EMOTION_WORDS_V1 = ["despair", "sad", "dislike", "annoyed", "neutral",
                    "calm", "content", "like", "joy", "love"]
EMOTION_WORDS_V2 = ["anguish", "sorrow", "loathe", "irritated", "indifferent",
                    "peaceful", "pleased", "affection", "elation", "adoration"]
EMOTION_WORDS_EXTENDED = [
    "despair", "anguish", "sad", "sorrow", "dislike", "loathe",
    "annoyed", "irritated", "neutral", "indifferent",
    "calm", "peaceful", "content", "pleased", "like", "affection",
    "joy", "elation", "love", "adoration"
]

# 相对编码验证: 概念类别
CONCEPT_CATEGORIES = {
    "fruit": ["apple", "banana", "orange", "grape", "mango",
              "peach", "cherry", "pear", "strawberry", "watermelon"],
    "animal": ["cat", "dog", "bird", "horse", "fish",
               "bear", "lion", "eagle", "snake", "rabbit"],
    "vehicle": ["car", "bus", "train", "bicycle", "motorcycle",
                "truck", "airplane", "boat", "scooter", "helicopter"],
    "clothing": ["shirt", "pants", "dress", "jacket", "hat",
                 "socks", "shoes", "scarf", "gloves", "coat"],
    "furniture": ["chair", "table", "sofa", "bed", "desk",
                  "cabinet", "shelf", "lamp", "rug", "stool"],
}

# 概念区分实验 (用于Part 3维度-准确率曲线)
CONCEPT_GROUPS_CLASSIFY = {
    "temperature": TEMPERATURE_WORDS_V1[:8],
    "size": SIZE_WORDS_V1[:8],
    "emotion": ["love", "hate", "joy", "sad", "anger", "calm", "like", "dislike"],
    "animal": ["cat", "dog", "bird", "horse", "fish", "bear", "lion", "eagle"],
    "fruit": ["apple", "banana", "orange", "grape", "mango", "peach", "cherry", "pear"],
}

TEMPLATES_PER_WORD = 30


def generate_templates(word_list, category="generic"):
    """为每个词生成模板句子"""
    templates = {}
    for word in word_list:
        sents = [
            f"The {word} was very noticeable today",
            f"She described it as {word}",
            f"He found the {word} quite interesting",
            f"This {word} thing caught my attention",
            f"The weather felt {word} this morning",
            f"Everyone noticed the {word} change",
            f"The {word} conditions were remarkable",
            f"I have never seen anything so {word}",
            f"The {word} experience was memorable",
            f"She thought it was rather {word}",
            f"That {word} feeling was overwhelming",
            f"The situation became {word} quickly",
            f"He considered the {word} aspect important",
            f"The {word} quality stood out immediately",
            f"They described the event as {word}",
            f"A {word} atmosphere filled the room",
            f"The {word} sensation was unmistakable",
            f"Nothing could be more {word} than this",
            f"She appreciated the {word} nature of it",
            f"The {word} phenomenon was well documented",
            f"He encountered a {word} situation",
            f"The {word} characteristic was defining",
            f"People often describe it as {word}",
            f"The {word} element was crucial",
            f"We observed the {word} pattern clearly",
            f"The {word} property was significant",
            f"She recognized the {word} feature",
            f"The {word} condition required attention",
            f"He labeled the experience as {word}",
            f"The {word} state was evident to all",
        ]
        templates[word] = sents[:TEMPLATES_PER_WORD]
    return templates


# =====================================================================
# Part 1 (63a): Grassmann距离验证2D子空间稳定性
# =====================================================================

def run_part1(model, tokenizer, device):
    """
    63a: 用Grassmann距离直接验证"2D子空间稳定,1D方向不稳定"
    
    关键判断:
    - Grassmann距离 < 0.3 (≈17度) → 子空间高度稳定 → "2D平面"假设成立
    - Grassmann距离 > 0.8 (≈46度) → 子空间不稳定 → 需要重新思考
    """
    from model_utils import get_model_info

    info = get_model_info(model, model_name_global)
    n_layers = info.n_layers
    d_model = info.d_model

    # 采样层: 早/中/晚+最上层
    target_layers = [n_layers // 4, n_layers // 2, n_layers * 3 // 4, n_layers - 1]
    log_time(f"Part 1: Grassmann distance verification (layers={target_layers})")

    # 轴定义: v1和v2词集
    axes = {
        "temperature": (TEMPERATURE_WORDS_V1, TEMPERATURE_WORDS_V2),
        "size": (SIZE_WORDS_V1, SIZE_WORDS_V2),
        "speed": (SPEED_WORDS_V1, SPEED_WORDS_V2),
        "emotion": (EMOTION_WORDS_V1, EMOTION_WORDS_V2),
    }

    results = {}

    for axis_name, (words_v1, words_v2) in axes.items():
        log_time(f"  Axis: {axis_name}")

        # 收集两套词集的hidden states
        all_words = list(set(words_v1 + words_v2))
        templates = generate_templates(all_words)
        all_sentences = []
        word_to_idx = {}
        for w in all_words:
            word_to_idx[w] = len(all_sentences)
            all_sentences.extend(templates[w])

        hidden_states = collect_hidden_states(model, tokenizer, device,
                                              all_sentences, target_layers, batch_size=4)

        # 为每个词提取平均hidden state
        word_hidden = {}
        for w in all_words:
            start_idx = word_to_idx[w]
            end_idx = start_idx + len(templates[w])
            word_hidden[w] = {}
            for li in target_layers:
                word_hidden[w][li] = hidden_states[li][start_idx:end_idx].mean(axis=0)

        # 对每个层计算
        axis_results = {}
        for li in target_layers:
            layer_result = {}

            # V1词集的子空间
            v1_acts = np.array([word_hidden[w][li] for w in words_v1])
            v2_acts = np.array([word_hidden[w][li] for w in words_v2])

            # 2D子空间
            S2d_v1 = extract_subspace(v1_acts, n_dims=2)
            S2d_v2 = extract_subspace(v2_acts, n_dims=2)

            # Grassmann距离 (2D)
            gdist_2d = grassmann_distance(S2d_v1, S2d_v2)
            pangles_2d = principal_angles(S2d_v1, S2d_v2)

            # 3D子空间
            S3d_v1 = extract_subspace(v1_acts, n_dims=3)
            S3d_v2 = extract_subspace(v2_acts, n_dims=3)
            gdist_3d = grassmann_distance(S3d_v1, S3d_v2)
            pangles_3d = principal_angles(S3d_v1, S3d_v2)

            # 5D子空间
            S5d_v1 = extract_subspace(v1_acts, n_dims=5)
            S5d_v2 = extract_subspace(v2_acts, n_dims=5)
            gdist_5d = grassmann_distance(S5d_v1, S5d_v2)

            # 10D子空间
            S10d_v1 = extract_subspace(v1_acts, n_dims=10)
            S10d_v2 = extract_subspace(v2_acts, n_dims=10)
            gdist_10d = grassmann_distance(S10d_v1, S10d_v2)

            # 1D方向稳定性 (对比)
            pc1_v1 = extract_subspace(v1_acts, n_dims=1)
            pc1_v2 = extract_subspace(v2_acts, n_dims=1)
            pc1_cos = float(np.abs((pc1_v1 * pc1_v2).sum()))

            pc2_v1 = extract_subspace(v1_acts, n_dims=2)[:, 1:2]
            pc2_v2 = extract_subspace(v2_acts, n_dims=2)[:, 1:2]
            pc2_cos = float(np.abs((pc2_v1 * pc2_v2).sum()))

            # 随机baseline: 随机2D子空间的Grassmann距离
            n_random = 100
            random_gdist = []
            for _ in range(n_random):
                R1 = np.random.randn(d_model, 2)
                R1, _ = np.linalg.qr(R1)
                R2 = np.random.randn(d_model, 2)
                R2, _ = np.linalg.qr(R2)
                random_gdist.append(grassmann_distance(R1, R2))
            random_gdist_mean = float(np.mean(random_gdist))

            # principal angle overlap
            pa_overlap = subspace_overlap_principal_angle(S2d_v1, S2d_v2)

            layer_result = {
                "grassmann_2d": round(gdist_2d, 4),
                "grassmann_3d": round(gdist_3d, 4),
                "grassmann_5d": round(gdist_5d, 4),
                "grassmann_10d": round(gdist_10d, 4),
                "principal_angles_2d_deg": [round(a, 2) for a in pangles_2d],
                "principal_angles_3d_deg": [round(a, 2) for a in pangles_3d],
                "pc1_cosine": round(pc1_cos, 4),
                "pc2_cosine": round(pc2_cos, 4),
                "random_baseline_2d": round(random_gdist_mean, 4),
                "pa_overlap_2d": round(pa_overlap, 4),
            }

            axis_results[f"L{li}"] = layer_result

            log_time(f"    L{li}: Gdist(2D)={gdist_2d:.4f}, PA=[{', '.join(f'{a:.1f}' for a in pangles_2d)}]°, "
                     f"PC1|cos|={pc1_cos:.3f}, PC2|cos|={pc2_cos:.3f}, "
                     f"random_baseline={random_gdist_mean:.4f}")

        results[axis_name] = axis_results

    # 保存结果
    out = {
        "model": model_name_global,
        "part": "63a_grassmann_distance",
        "timestamp": datetime.now().isoformat(),
        "target_layers": target_layers,
        "results": results,
    }

    path = RESULT_DIR / f"phase63_part1_{model_name_global}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    log_time(f"Part 1 results saved to {path}")

    # 打印汇总
    log_time("=" * 60)
    log_time("Part 1 SUMMARY: Grassmann Distance Verification")
    log_time("=" * 60)
    for axis_name, axis_results in results.items():
        for layer_key, lr in axis_results.items():
            gdist = lr["grassmann_2d"]
            pa = lr["principal_angles_2d_deg"]
            pc1c = lr["pc1_cosine"]
            rb = lr["random_baseline_2d"]
            stability = "STABLE" if gdist < 0.3 else ("MARGINAL" if gdist < 0.8 else "UNSTABLE")
            reduction = f"{rb/gdist:.1f}x" if gdist > 0.01 else "N/A"
            log_time(f"  {axis_name} {layer_key}: Gdist={gdist:.4f} PA=[{pa}]° "
                     f"PC1|cos|={pc1c:.3f} vs_random={reduction} -> {stability}")

    return out


# =====================================================================
# Part 2 (63b): 跨轴子空间正交性 + 2D平面内精细结构
# =====================================================================

def run_part2(model, tokenizer, device):
    """
    63b: 跨轴子空间正交性 + 平面内精细结构
    
    A. 温度/大小/速度/情感的2D子空间是否相互正交?
    B. 在2D平面内, 词是否排列成1D线(流形)?
    """
    from model_utils import get_model_info

    info = get_model_info(model, model_name_global)
    n_layers = info.n_layers
    d_model = info.d_model

    target_layer = n_layers - 1  # 最高层
    log_time(f"Part 2: Cross-axis orthogonality + fine structure (layer={target_layer})")

    # 收集扩展词集的hidden states
    all_extended_words = (TEMPERATURE_WORDS_EXTENDED + SIZE_WORDS_EXTENDED +
                          SPEED_WORDS_EXTENDED + EMOTION_WORDS_EXTENDED)
    all_extended_words = list(set(all_extended_words))

    templates = generate_templates(all_extended_words)
    all_sentences = []
    word_to_idx = {}
    for w in all_extended_words:
        word_to_idx[w] = len(all_sentences)
        all_sentences.extend(templates[w])

    hidden_states = collect_hidden_states(model, tokenizer, device,
                                          all_sentences, [target_layer], batch_size=4)

    # 为每个词提取平均hidden state
    word_hidden = {}
    for w in all_extended_words:
        start_idx = word_to_idx[w]
        end_idx = start_idx + len(templates[w])
        word_hidden[w] = hidden_states[target_layer][start_idx:end_idx].mean(axis=0)

    results = {}

    # ---- A. 跨轴子空间正交性 ----
    log_time("  A. Cross-axis subspace orthogonality...")

    axis_words = {
        "temperature": TEMPERATURE_WORDS_EXTENDED,
        "size": SIZE_WORDS_EXTENDED,
        "speed": SPEED_WORDS_EXTENDED,
        "emotion": EMOTION_WORDS_EXTENDED,
    }

    # 也用V1词集 (与Phase 62对齐)
    axis_words_v1 = {
        "temperature": TEMPERATURE_WORDS_V1,
        "size": SIZE_WORDS_V1,
        "speed": SPEED_WORDS_V1,
        "emotion": EMOTION_WORDS_V1,
    }

    # 提取各轴的2D子空间 (使用V1词集, 与Phase 62对齐)
    axis_subspaces_2d = {}
    axis_subspaces_3d = {}
    for axis_name, words in axis_words_v1.items():
        acts = np.array([word_hidden[w] for w in words if w in word_hidden])
        if len(acts) >= 3:
            axis_subspaces_2d[axis_name] = extract_subspace(acts, n_dims=2)
            axis_subspaces_3d[axis_name] = extract_subspace(acts, n_dims=3)

    # 计算跨轴principal angles
    orthogonality_results = {}
    axis_names = list(axis_subspaces_2d.keys())
    for i in range(len(axis_names)):
        for j in range(i + 1, len(axis_names)):
            a1, a2 = axis_names[i], axis_names[j]
            S1_2d = axis_subspaces_2d[a1]
            S2_2d = axis_subspaces_2d[a2]

            pa_2d = principal_angles(S1_2d, S2_2d)
            gdist_2d = grassmann_distance(S1_2d, S2_2d)
            pa_overlap = subspace_overlap_principal_angle(S1_2d, S2_2d)

            # 3D子空间比较
            S1_3d = axis_subspaces_3d[a1]
            S2_3d = axis_subspaces_3d[a2]
            pa_3d = principal_angles(S1_3d, S2_3d)
            gdist_3d = grassmann_distance(S1_3d, S2_3d)

            key = f"{a1}_vs_{a2}"
            orthogonality_results[key] = {
                "principal_angles_2d_deg": [round(a, 2) for a in pa_2d],
                "grassmann_2d": round(gdist_2d, 4),
                "pa_overlap_2d": round(pa_overlap, 4),
                "principal_angles_3d_deg": [round(a, 2) for a in pa_3d],
                "grassmann_3d": round(gdist_3d, 4),
            }

            orth_judgment = "NEAR-ORTHOGONAL" if min(pa_2d) > 60 else (
                "PARTIAL-OVERLAP" if min(pa_2d) < 30 else "PARTIAL-ORTHOGONAL")
            log_time(f"    {key}: PA_2D=[{', '.join(f'{a:.1f}' for a in pa_2d)}]° "
                     f"Gdist={gdist_2d:.4f} -> {orth_judgment}")

    results["cross_axis_orthogonality"] = orthogonality_results

    # ---- B. 2D平面内精细结构 ----
    log_time("  B. Fine structure within 2D planes...")

    fine_structure = {}
    for axis_name, words in axis_words.items():
        valid_words = [w for w in words if w in word_hidden]
        if len(valid_words) < 5:
            continue

        acts = np.array([word_hidden[w] for w in valid_words])

        # 提取2D子空间
        S_2d = extract_subspace(acts, n_dims=2)

        # 将每个词投影到2D平面
        coords_2d = acts @ S_2d  # [n_words, 2]

        # 计算词在2D平面内的线性度
        # 如果词近似排列在一条线上, 用PCA的第一分量解释方差比应该>90%
        mean_coord = coords_2d.mean(axis=0)
        centered_coords = coords_2d - mean_coord
        U_coord, S_coord, Vt_coord = np.linalg.svd(centered_coords, full_matrices=False)
        linearity = float(S_coord[0] ** 2 / (S_coord ** 2).sum()) if len(S_coord) > 1 else 1.0

        # 计算词在2D平面内的1D排序 (沿主轴)
        pc1_coords = centered_coords @ Vt_coord[0]
        word_order = [(valid_words[i], float(pc1_coords[i])) for i in range(len(valid_words))]
        word_order.sort(key=lambda x: x[1])

        # 计算端点距离比 (端点之间的距离 vs 中心到端点的距离)
        if len(valid_words) >= 2:
            extremes = [word_order[0][0], word_order[-1][0]]
            extremes_idx = [valid_words.index(e) for e in extremes]
            dist_extremes = np.linalg.norm(coords_2d[extremes_idx[0]] - coords_2d[extremes_idx[1]])
            center_to_extremes = [np.linalg.norm(coords_2d[ei] - mean_coord) for ei in extremes_idx]
            aspect_ratio = dist_extremes / (max(center_to_extremes) + 1e-10)
        else:
            aspect_ratio = 0

        fine_structure[axis_name] = {
            "linearity": round(linearity, 4),
            "aspect_ratio": round(aspect_ratio, 4),
            "word_order_along_pc1": word_order,
            "svd_variances": [round(float(s ** 2 / (S_coord ** 2).sum()), 4) for s in S_coord],
        }

        log_time(f"    {axis_name}: linearity={linearity:.3f} "
                 f"aspect_ratio={aspect_ratio:.2f} "
                 f"SVD_var=[{', '.join(f'{s:.3f}' for s in [float(x) for x in S_coord**2/(S_coord**2).sum()])}]")
        log_time(f"      PC1 order: {' < '.join([w for w, _ in word_order])}")

    results["fine_structure"] = fine_structure

    # ---- C. 扩展词集的W_U解码 (验证清晰度与词集大小的关系) ----
    log_time("  C. W_U decoding with extended word sets...")

    W_U = None
    try:
        from model_utils import get_W_U
        W_U = get_W_U(model, model_name_global)
    except Exception as e:
        log_time(f"  W_U loading failed: {e}, skipping decoding")

    if W_U is not None:
        decoding_results = {}
        for axis_name, words in axis_words.items():
            valid_words = [w for w in words if w in word_hidden]
            if len(valid_words) < 5:
                continue

            acts = np.array([word_hidden[w] for w in valid_words])
            S_2d = extract_subspace(acts, n_dims=2)

            # PC1方向解码
            pc1_dir = S_2d[:, 0]
            logit_dir = W_U @ pc1_dir
            top_ids = np.argsort(logit_dir)[-15:][::-1]
            bot_ids = np.argsort(logit_dir)[:15]
            top_tokens = [tokenizer.decode([int(i)]).strip() for i in top_ids]
            bot_tokens = [tokenizer.decode([int(i)]).strip() for i in bot_ids]

            # PC2方向解码
            pc2_dir = S_2d[:, 1]
            logit_dir2 = W_U @ pc2_dir
            top_ids2 = np.argsort(logit_dir2)[-15:][::-1]
            bot_ids2 = np.argsort(logit_dir2)[:15]
            top_tokens2 = [tokenizer.decode([int(i)]).strip() for i in top_ids2]
            bot_tokens2 = [tokenizer.decode([int(i)]).strip() for i in bot_ids2]

            decoding_results[axis_name] = {
                "pc1_top": top_tokens[:10],
                "pc1_bot": bot_tokens[:10],
                "pc2_top": top_tokens2[:10],
                "pc2_bot": bot_tokens2[:10],
            }

            log_time(f"    {axis_name} PC1 top: {top_tokens[:5]}")
            log_time(f"    {axis_name} PC2 top: {top_tokens2[:5]}")

        results["wu_decoding_extended"] = decoding_results

    # 保存结果
    out = {
        "model": model_name_global,
        "part": "63b_orthogonality_fine_structure",
        "timestamp": datetime.now().isoformat(),
        "target_layer": target_layer,
        "results": results,
    }

    path = RESULT_DIR / f"phase63_part2_{model_name_global}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    log_time(f"Part 2 results saved to {path}")

    return out


# =====================================================================
# Part 3 (63c): 多维度分类准确率曲线
# =====================================================================

def run_part3(model, tokenizer, device):
    """
    63c: 不同维度数下的分类准确率曲线
    
    核心问题: 语言编码是稀疏低维的还是高维分散的?
    - 如果存在拐点(knee point) → 信息集中在前k个维度
    - 如果线性增长到all → 信息完全分散
    """
    from model_utils import get_model_info
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    info = get_model_info(model, model_name_global)
    n_layers = info.n_layers

    target_layer = n_layers - 1
    log_time(f"Part 3: Dimension-accuracy curve (layer={target_layer})")

    dims_range = [3, 5, 10, 20, 50, 100, 200, 500, "all"]

    # 收集各概念组的hidden states
    concept_hidden = {}
    concept_labels = {}

    for group_name, words in CONCEPT_GROUPS_CLASSIFY.items():
        log_time(f"  Collecting: {group_name} ({len(words)} words)")

        templates = generate_templates(words)
        all_sentences = []
        labels = []
        for w in words:
            all_sentences.extend(templates[w])
            labels.extend([w] * len(templates[w]))

        hidden_states = collect_hidden_states(model, tokenizer, device,
                                              all_sentences, [target_layer], batch_size=4)
        concept_hidden[group_name] = hidden_states[target_layer]
        concept_labels[group_name] = labels

    # 合并所有概念组用于多分类
    all_hidden = np.concatenate(list(concept_hidden.values()), axis=0)
    all_group_labels = []
    for gname, labels in concept_labels.items():
        all_group_labels.extend([gname] * len(labels))
    all_group_labels = np.array(all_group_labels)

    # 同时做组内词分类
    results = {
        "group_classification": {},  # 区分概念组 (5-class)
        "word_classification": {},   # 区分组内词
    }

    # ---- A. 概念组分类 (5分类) ----
    log_time("  A. Group classification (5-class)...")

    # 用discriminative维度选择
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_group = le.fit_transform(all_group_labels)

    # 训练一个全维度分类器来获取discriminative维度排序
    clf_full = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
    clf_full.fit(all_hidden, y_group)
    dim_importance = np.abs(clf_full.coef_).sum(axis=0)  # 多分类: sum over classes
    discrim_dims_order = np.argsort(dim_importance)[::-1]

    for k in dims_range:
        if k == "all":
            X_k = all_hidden
            k_actual = all_hidden.shape[1]
        else:
            k_actual = min(k, all_hidden.shape[1])
            top_dims = discrim_dims_order[:k_actual]
            X_k = all_hidden[:, top_dims]

        # 5-fold cross validation
        clf_k = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
        scores = cross_val_score(clf_k, X_k, y_group, cv=5, scoring='accuracy')

        results["group_classification"][str(k)] = {
            "accuracy": round(float(scores.mean()), 4),
            "std": round(float(scores.std()), 4),
            "n_dims": k_actual,
        }
        log_time(f"    k={k}: acc={scores.mean():.3f} ± {scores.std():.3f}")

    # ---- B. 组内词分类 (每个组内部分词) ----
    log_time("  B. Within-group word classification...")

    for group_name, words in CONCEPT_GROUPS_CLASSIFY.items():
        hidden = concept_hidden[group_name]
        labels = np.array(concept_labels[group_name])

        # 只保留至少有2个样本的词
        unique_labels, counts = np.unique(labels, return_counts=True)
        valid_labels = unique_labels[counts >= 5]
        if len(valid_labels) < 2:
            continue

        mask = np.isin(labels, valid_labels)
        X = hidden[mask]
        y = labels[mask]

        le_w = LabelEncoder()
        y_enc = le_w.fit_transform(y)

        # Discriminative维度选择
        try:
            clf_w_full = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
            clf_w_full.fit(X, y_enc)
            dim_imp_w = np.abs(clf_w_full.coef_).sum(axis=0)
            discrim_order_w = np.argsort(dim_imp_w)[::-1]
        except Exception as e:
            log_time(f"    {group_name}: full clf failed ({e}), using variance order")
            var_order = np.argsort(X.var(axis=0))[::-1]
            discrim_order_w = var_order

        group_curve = {}
        for k in [5, 10, 20, 50, 100, "all"]:
            if k == "all":
                X_k = X
                k_actual = X.shape[1]
            else:
                k_actual = min(k, X.shape[1])
                top_dims = discrim_order_w[:k_actual]
                X_k = X[:, top_dims]

            try:
                clf_k = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
                scores = cross_val_score(clf_k, X_k, y_enc, cv=5, scoring='accuracy')
                group_curve[str(k)] = {
                    "accuracy": round(float(scores.mean()), 4),
                    "std": round(float(scores.std()), 4),
                    "n_dims": k_actual,
                }
            except Exception as e:
                group_curve[str(k)] = {"accuracy": 0, "std": 0, "error": str(e)}

        results["word_classification"][group_name] = group_curve
        acc_all = group_curve.get("all", {}).get("accuracy", 0)
        acc_10 = group_curve.get("10", {}).get("accuracy", 0)
        log_time(f"    {group_name}: acc(k=10)={acc_10:.3f}, acc(all)={acc_all:.3f}")

    # 保存结果
    out = {
        "model": model_name_global,
        "part": "63c_dimension_accuracy_curve",
        "timestamp": datetime.now().isoformat(),
        "target_layer": target_layer,
        "dims_range": [str(d) for d in dims_range],
        "results": results,
    }

    path = RESULT_DIR / f"phase63_part3_{model_name_global}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    log_time(f"Part 3 results saved to {path}")

    return out


# =====================================================================
# Part 4 (63d): 类别共享 + 个体差异分解 — 验证相对编码假设
# =====================================================================

def run_part4(model, tokenizer, device):
    """
    63d: 验证"类别共享激活 + 个体差异激活"的分解结构
    
    核心假设:
    激活(苹果) = 类别共享成分(水果) + 个体差异成分(苹果特有)
    
    如果差异成分在W_U解码后对应苹果独特特征 → 相对编码假设成立
    如果差异成分跨上下文稳定 → 编码是结构化的
    """
    from model_utils import get_model_info

    info = get_model_info(model, model_name_global)
    n_layers = info.n_layers

    target_layer = n_layers - 1
    log_time(f"Part 4: Class-shared + Individual-difference decomposition (layer={target_layer})")

    # 选择3个概念类别进行测试
    test_categories = {
        "fruit": ["apple", "banana", "orange", "grape", "mango",
                  "peach", "cherry", "pear", "strawberry", "watermelon"],
        "animal": ["cat", "dog", "bird", "horse", "fish",
                   "bear", "lion", "eagle", "snake", "rabbit"],
        "vehicle": ["car", "bus", "train", "bicycle", "motorcycle",
                    "truck", "airplane", "boat", "scooter", "helicopter"],
    }

    # 生成模板句子 (多样化的上下文)
    def generate_diverse_templates(word, n=50):
        templates = [
            f"The {word} was remarkable",
            f"She noticed the {word}",
            f"He described the {word}",
            f"I really like {word}",
            f"The {word} caught my eye",
            f"Everyone saw the {word}",
            f"The {word} appeared suddenly",
            f"We found the {word} interesting",
            f"That {word} was unusual",
            f"The {word} stood out",
            f"She preferred the {word}",
            f"The {word} was impressive",
            f"He chose the {word}",
            f"The {word} was distinctive",
            f"They admired the {word}",
            f"A {word} was visible",
            f"The {word} seemed special",
            f"She recognized the {word}",
            f"The {word} looked different",
            f"He appreciated the {word}",
            f"The {word} was noticeable",
            f"We examined the {word}",
            f"That {word} was unique",
            f"The {word} was familiar",
            f"She mentioned the {word}",
            f"The {word} was important",
            f"He studied the {word}",
            f"The {word} had character",
            f"The {word} was exceptional",
            f"We liked the {word}",
            f"The {word} was beautiful",
            f"She found the {word} appealing",
            f"The {word} was typical",
            f"He enjoyed the {word}",
            f"The {word} was unusual here",
            f"They spotted the {word}",
            f"A {word} appeared nearby",
            f"The {word} was striking",
            f"She identified the {word}",
            f"The {word} was prominent",
            f"He remembered the {word}",
            f"The {word} was ordinary",
            f"We observed the {word}",
            f"That {word} was common",
            f"The {word} was rare",
            f"She encountered the {word}",
            f"The {word} was significant",
            f"He selected the {word}",
            f"The {word} was valuable",
        ]
        return templates[:n]

    # 收集hidden states
    results = {}

    for cat_name, words in test_categories.items():
        log_time(f"  Category: {cat_name}")

        all_sentences = []
        word_indices = {}  # word -> (start, end)

        for w in words:
            start = len(all_sentences)
            sents = generate_diverse_templates(w, n=50)
            all_sentences.extend(sents)
            word_indices[w] = (start, start + len(sents))

        hidden_states = collect_hidden_states(model, tokenizer, device,
                                              all_sentences, [target_layer], batch_size=4)

        # 每个词的平均hidden state
        word_means = {}
        for w in words:
            s, e = word_indices[w]
            word_means[w] = hidden_states[target_layer][s:e].mean(axis=0)

        # 类别质心
        cat_centroid = np.mean([word_means[w] for w in words], axis=0)

        # 个体差异成分
        deltas = {}
        for w in words:
            deltas[w] = word_means[w] - cat_centroid

        # ---- 验证1: 差异成分的W_U解码 ----
        W_U = None
        try:
            from model_utils import get_W_U
            W_U = get_W_U(model, model_name_global)
        except Exception as e:
            log_time(f"  W_U loading failed: {e}")

        delta_decoding = {}
        if W_U is not None:
            for w in words:
                logit_dir = W_U @ deltas[w]
                top_ids = np.argsort(logit_dir)[-15:][::-1]
                bot_ids = np.argsort(logit_dir)[:15]
                top_tokens = [tokenizer.decode([int(i)]).strip() for i in top_ids]
                bot_tokens = [tokenizer.decode([int(i)]).strip() for i in bot_ids]

                # 检查是否包含该词本身
                w_in_top = any(w.lower() in t.lower() for t in top_tokens[:10])

                delta_decoding[w] = {
                    "top": top_tokens[:10],
                    "bot": bot_tokens[:10],
                    "word_in_top10": w_in_top,
                }

            log_time(f"    {cat_name} delta decoding:")
            for w in words:
                d = delta_decoding[w]
                log_time(f"      {w}: top={d['top'][:5]}, self_in_top10={d['word_in_top10']}")

        # ---- 验证2: 差异成分跨上下文的稳定性 ----
        # 对每个词, 计算各上下文的差异成分, 然后SVD
        delta_stability = {}
        for w in words:
            s, e = word_indices[w]
            word_hidden_all = hidden_states[target_layer][s:e]  # [n_templates, d_model]

            # 每个上下文的差异成分
            per_context_deltas = word_hidden_all - cat_centroid  # [n_templates, d_model]

            # SVD分析
            mean_d = per_context_deltas.mean(axis=0)
            centered_d = per_context_deltas - mean_d
            U_d, S_d, Vt_d = np.linalg.svd(centered_d, full_matrices=False)

            # 90%方差需要多少维
            var_explained = np.cumsum(S_d ** 2) / (S_d ** 2).sum()
            k90 = int(np.searchsorted(var_explained, 0.9)) + 1
            k95 = int(np.searchsorted(var_explained, 0.95)) + 1

            # 第一主成分的解释力
            pc1_var = float(S_d[0] ** 2 / (S_d ** 2).sum())

            # 差异成分的范数 (绝对强度)
            delta_norm = float(np.linalg.norm(deltas[w]))
            cat_norm = float(np.linalg.norm(cat_centroid))
            delta_ratio = delta_norm / (cat_norm + 1e-10)

            delta_stability[w] = {
                "k90": k90,
                "k95": k95,
                "pc1_variance": round(pc1_var, 4),
                "delta_norm": round(delta_norm, 4),
                "cat_centroid_norm": round(cat_norm, 4),
                "delta_ratio": round(delta_ratio, 4),
            }

        log_time(f"    {cat_name} delta stability:")
        for w in words:
            ds = delta_stability[w]
            log_time(f"      {w}: k90={ds['k90']}, pc1_var={ds['pc1_variance']:.3f}, "
                     f"delta_ratio={ds['delta_ratio']:.4f}")

        # ---- 验证3: 语义距离 vs 差异成分距离 ----
        # W_U空间中的语义距离
        if W_U is not None:
            semantic_distances = {}
            delta_distances = {}
            abs_distances = {}

            for i in range(len(words)):
                for j in range(i + 1, len(words)):
                    w1, w2 = words[i], words[j]
                    # 差异成分的距离
                    delta_dist = float(np.linalg.norm(deltas[w1] - deltas[w2]))
                    # 绝对hidden state距离
                    abs_dist = float(np.linalg.norm(word_means[w1] - word_means[w2]))
                    # W_U空间的语义距离 (logit similarity)
                    logit1 = W_U @ word_means[w1]
                    logit2 = W_U @ word_means[w2]
                    sem_dist = float(np.linalg.norm(logit1 - logit2))

                    semantic_distances[f"{w1}-{w2}"] = round(sem_dist, 4)
                    delta_distances[f"{w1}-{w2}"] = round(delta_dist, 4)
                    abs_distances[f"{w1}-{w2}"] = round(abs_dist, 4)

            # 计算相关性
            from scipy.stats import spearmanr, pearsonr
            sem_vals = list(semantic_distances.values())
            delta_vals = list(delta_distances.values())
            abs_vals = list(abs_distances.values())

            if len(sem_vals) > 2:
                rho_delta, p_delta = spearmanr(delta_vals, sem_vals)
                rho_abs, p_abs = spearmanr(abs_vals, sem_vals)
            else:
                rho_delta, p_delta, rho_abs, p_abs = 0, 1, 0, 1

            distance_corr = {
                "delta_semantic_spearman": round(float(rho_delta), 4),
                "delta_semantic_pvalue": round(float(p_delta), 4),
                "abs_semantic_spearman": round(float(rho_abs), 4),
                "abs_semantic_pvalue": round(float(p_abs), 4),
            }

            log_time(f"    {cat_name} distance correlation: "
                     f"delta-semantic rho={rho_delta:.3f}, abs-semantic rho={rho_abs:.3f}")
        else:
            distance_corr = {}

        results[cat_name] = {
            "delta_decoding": delta_decoding,
            "delta_stability": delta_stability,
            "distance_correlation": distance_corr,
        }

    # 保存结果
    out = {
        "model": model_name_global,
        "part": "63d_relative_encoding_test",
        "timestamp": datetime.now().isoformat(),
        "target_layer": target_layer,
        "results": results,
    }

    path = RESULT_DIR / f"phase63_part4_{model_name_global}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    log_time(f"Part 4 results saved to {path}")

    return out


# =====================================================================
# 主函数
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 63: Subspace Stability + Encoding Analysis")
    parser.add_argument("--model", type=str, required=True,
                        choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--part", type=str, required=True,
                        choices=["1", "2", "3", "4", "all"])
    args = parser.parse_args()

    global model_name_global
    model_name_global = args.model

    log_time(f"Phase 63: model={args.model}, part={args.part}")

    model, tokenizer, device = load_model_bf16(args.model)

    parts = ["1", "2", "3", "4"] if args.part == "all" else [args.part]

    for part in parts:
        log_time(f"\n{'='*60}")
        log_time(f"Starting Part {part}")
        log_time(f"{'='*60}")

        try:
            if part == "1":
                run_part1(model, tokenizer, device)
            elif part == "2":
                run_part2(model, tokenizer, device)
            elif part == "3":
                run_part3(model, tokenizer, device)
            elif part == "4":
                run_part4(model, tokenizer, device)
        except Exception as e:
            log_time(f"Part {part} FAILED: {e}")
            import traceback
            traceback.print_exc()

        # 清理GPU缓存
        gc.collect()
        torch.cuda.empty_cache()
        log_time(f"GPU after Part {part}: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # 释放模型
    from model_utils import release_model
    release_model(model)
    log_time("Phase 63 COMPLETE!")


if __name__ == "__main__":
    main()
