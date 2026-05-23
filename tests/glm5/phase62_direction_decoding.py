"""
Phase 62: 方向解码 + 跨轴验证 + 方法修正
===========================================

四个方案 + 稳定性验证:
  Part 1 (62a): 用Discrim方法+principal angle重算overlap — 验证Phase 59核心结论
  Part 2 (62b): PC2方向精确解码 — 投影W_U解码top tokens
  Part 3 (62c): 句法对照实验 — 排除PC2消融的句法替代解释
  Part 4 (62d): 跨轴PC1一致性 + 词集稳定性检验

用法:
  python tests/glm5/phase62_direction_decoding.py --model qwen3 --part 1
  python tests/glm5/phase62_direction_decoding.py --model qwen3 --part all
"""

import sys, os, json, argparse, gc, time, copy, warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

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


def compute_principal_angle_overlap(S1, S2):
    """
    用principal angle方法计算子空间overlap (更精确的度量)
    
    S1, S2: [d_model, k] orthonormal basis
    返回: mean squared cosine of principal angles
    """
    M = S1.T @ S2
    svals = np.linalg.svd(M, compute_uv=False)
    svals = np.clip(svals, 0, 1)
    return float(np.mean(svals ** 2))


def compute_topk_overlap(S1, S2):
    """原Phase 59-60的top-k方差overlap"""
    M = S1.T @ S2
    svals = np.linalg.svd(M, compute_uv=False)
    return float(np.mean(svals ** 2))


def extract_subspace(activations, n_dims=10):
    mean = activations.mean(axis=0, keepdims=True)
    centered = activations - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    return Vt[:n_dims].T  # [d_model, n_dims] orthonormal basis


# =====================================================================
# 扩展词集
# =====================================================================

# 温度轴: 10个词
TEMPERATURE_WORDS = ["freezing", "frigid", "cold", "cool", "lukewarm", "warm", "hot", "scorching", "blazing", "searing"]

# 温度轴替代词集 (用于稳定性检验)
TEMPERATURE_WORDS_V2 = ["icy", "chilly", "frosty", "tepid", "mild", "toasty", "boiling", "sweltering", "scalding", "infernal"]

# 大小轴: 10个词
SIZE_WORDS = ["microscopic", "tiny", "small", "moderate", "medium", "large", "big", "huge", "enormous", "gigantic"]

# 大小轴替代词集
SIZE_WORDS_V2 = ["minuscule", "puny", "petite", "average", "standard", "substantial", "bulky", "massive", "colossal", "immense"]

# 速度轴: 10个词
SPEED_WORDS = ["glacial", "sluggish", "slow", "steady", "moderate", "fast", "quick", "rapid", "swift", "lightning"]

# 速度轴替代词集
SPEED_WORDS_V2 = ["crawling", "leisurely", "plodding", "cruising", "mid-tempo", "brisk", "hasty", "fleet", "breakneck", "instantaneous"]

# 情感轴: 12个词
EMOTION_WORDS = ["love", "joy", "like", "content", "calm", "neutral",
                 "annoyed", "dislike", "sad", "anger", "hate", "despair"]

# 颜色词 (用于句法对照实验)
COLOR_WORDS = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "black", "white", "gray"]

# 概念区分实验 (Phase 59风格)
CONCEPT_GROUPS = {
    "temperature": ["freezing", "cold", "cool", "warm", "hot", "scorching", "lukewarm", "blazing"],
    "size": ["tiny", "small", "medium", "large", "big", "huge", "enormous", "gigantic"],
    "emotion": ["love", "hate", "like", "dislike", "joy", "sad", "anger", "calm"],
    "animal": ["cat", "dog", "bird", "horse", "fish", "bear", "lion", "eagle"],
    "fruit": ["apple", "banana", "orange", "grape", "mango", "peach", "cherry", "pear"],
}

TEMPLATES_PER_WORD = 30


def generate_templates(word_list, category="generic"):
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
# Part 1: 用Discrim方法+principal angle重算overlap
# 验证Phase 59核心结论是否仍然成立
# =====================================================================

def run_part1(model, tokenizer, device):
    """
    62a: 用区分度维度+principal angle方法重算overlap
    核心验证: 反义词overlap > 近义词overlap是否仍然成立
    """
    from model_utils import get_model_info
    from sklearn.linear_model import LogisticRegression
    from scipy.stats import spearmanr

    info = get_model_info(model, model_name_global)
    n_layers = info.n_layers
    d_model = info.d_model

    target_layers = [n_layers // 3, n_layers * 2 // 3, n_layers - 1]
    log_time(f"Part 1: Discrim overlap re-verification (layers={target_layers})")

    # ---- A. 构建语义关系对 ----
    # 反义词对 (antonym pairs)
    antonym_pairs = [
        ("hot", "cold"), ("warm", "cool"), ("freezing", "scorching"),
        ("big", "small"), ("huge", "tiny"), ("enormous", "microscopic"),
        ("love", "hate"), ("joy", "sad"), ("calm", "anger"),
        ("fast", "slow"), ("rapid", "sluggish"), ("swift", "glacial"),
    ]
    # 近义词对 (synonym pairs)
    synonym_pairs = [
        ("hot", "scorching"), ("cold", "frigid"), ("warm", "lukewarm"),
        ("big", "large"), ("small", "tiny"), ("huge", "enormous"),
        ("love", "like"), ("joy", "content"), ("calm", "neutral"),
        ("fast", "quick"), ("slow", "sluggish"), ("rapid", "swift"),
    ]
    # 无关词对 (unrelated pairs)
    unrelated_pairs = [
        ("hot", "big"), ("cold", "tiny"), ("warm", "love"),
        ("big", "fast"), ("small", "joy"), ("huge", "calm"),
        ("love", "fast"), ("sad", "big"), ("anger", "small"),
        ("fast", "hot"), ("slow", "huge"), ("rapid", "cold"),
    ]
    # 上下位词对 (hyponym pairs)
    hyponym_pairs = [
        ("apple", "fruit"), ("cat", "animal"), ("car", "vehicle"),
        ("rose", "flower"), ("oak", "tree"), ("gold", "metal"),
        ("red", "color"), ("cold", "temperature"), ("fast", "speed"),
    ]

    # 合并所有词
    all_words = set()
    for pairs in [antonym_pairs, synonym_pairs, unrelated_pairs, hyponym_pairs]:
        for w1, w2 in pairs:
            all_words.add(w1)
            all_words.add(w2)
    all_words = list(all_words)

    # 加上区分度分类需要的概念组词
    for gname, words in CONCEPT_GROUPS.items():
        all_words.extend([w for w in words if w not in all_words])
    all_words = list(set(all_words))

    log_time(f"Total unique words: {len(all_words)}")

    # 生成模板并收集hidden states
    all_templates = generate_templates(all_words)
    all_sentences = []
    sent_word_map = []
    for w in all_words:
        for s in all_templates[w]:
            all_sentences.append(s)
            sent_word_map.append(w)

    log_time(f"Total sentences: {len(all_sentences)}")
    hidden = collect_hidden_states(model, tokenizer, device, all_sentences, target_layers)

    results = {}

    for li in target_layers:
        log_time(f"  Processing layer {li}...")
        hs = hidden[li]  # [n_sents, d_model]

        # 构建每个词的平均激活
        word_means = {}
        word_acts = {}
        for w in all_words:
            idxs = [i for i, sw in enumerate(sent_word_map) if sw == w]
            word_means[w] = hs[idxs].mean(axis=0)
            word_acts[w] = hs[idxs]

        # === 方法1: Top-k方差 overlap (原方法) ===
        def compute_pair_overlap_topk(w1, w2, n_dims=10):
            if w1 not in word_acts or w2 not in word_acts:
                return None
            s1 = extract_subspace(word_acts[w1], n_dims=n_dims)
            s2 = extract_subspace(word_acts[w2], n_dims=n_dims)
            return compute_topk_overlap(s1, s2)

        # === 方法2: Discrim overlap (logistic regression + principal angle) ===
        def compute_pair_overlap_discrim(w1, w2, n_dims=10):
            """
            用logistic regression选择区分A与"其他"的维度,
            再计算两个词的区分度子空间的principal angle overlap
            """
            if w1 not in word_acts or w2 not in word_acts:
                return None
            # 选择一个"其他"参照组
            other_words = [w for w in all_words if w != w1 and w != w2]
            # 随机选5个其他词作为参照
            np.random.seed(42)
            ref_words = list(np.random.choice(other_words, min(5, len(other_words)), replace=False))

            def get_discrim_dims(target_w, ref_ws, n_d=n_dims):
                X_target = word_acts[target_w]
                X_ref = np.concatenate([word_acts[ref_w] for ref_w in ref_ws if ref_w in word_acts], axis=0)
                X = np.concatenate([X_target, X_ref], axis=0)
                y = np.array([0] * len(X_target) + [1] * len(X_ref))
                clf = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
                clf.fit(X, y)
                top_dims = np.argsort(np.abs(clf.coef_[0]))[-n_d:]
                return top_dims

            dims1 = get_discrim_dims(w1, ref_words)
            dims2 = get_discrim_dims(w2, ref_words)

            # 构建子空间基 (用SVD确保正交)
            def dims_to_basis(dims, acts):
                selected = acts[:, dims]
                U, S, Vt = np.linalg.svd(selected, full_matrices=False)
                return U  # [n_sents, n_dims] 但我们需要的子空间在d_model空间

            # 更正确: 在全空间中构造区分度子空间
            def dims_to_subspace(dims, d):
                """构建选择维度的正交基"""
                basis = np.zeros((d, len(dims)))
                for i, dim in enumerate(dims):
                    basis[dim, i] = 1.0
                # 正交化
                Q, _ = np.linalg.qr(basis)
                return Q[:, :len(dims)]

            s1 = dims_to_subspace(dims1, d_model)
            s2 = dims_to_subspace(dims2, d_model)
            return compute_principal_angle_overlap(s1, s2)

        # === 方法3: Jaccard overlap (集合交集) ===
        def compute_pair_jaccard(w1, w2, n_dims=10):
            if w1 not in word_acts or w2 not in word_acts:
                return None
            other_words = [w for w in all_words if w != w1 and w != w2]
            np.random.seed(42)
            ref_words = list(np.random.choice(other_words, min(5, len(other_words)), replace=False))

            def get_discrim_dims(target_w, ref_ws, n_d=n_dims):
                X_target = word_acts[target_w]
                X_ref = np.concatenate([word_acts[ref_w] for ref_w in ref_ws if ref_w in word_acts], axis=0)
                X = np.concatenate([X_target, X_ref], axis=0)
                y = np.array([0] * len(X_target) + [1] * len(X_ref))
                clf = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
                clf.fit(X, y)
                top_dims = set(np.argsort(np.abs(clf.coef_[0]))[-n_d:])
                return top_dims

            dims1 = get_discrim_dims(w1, ref_words)
            dims2 = get_discrim_dims(w2, ref_words)
            intersection = dims1 & dims2
            union = dims1 | dims2
            return len(intersection) / max(len(union), 1)

        # 计算所有对的overlap
        pair_types = {
            "antonym": antonym_pairs,
            "synonym": synonym_pairs,
            "unrelated": unrelated_pairs,
            "hyponym": hyponym_pairs,
        }

        layer_results = {}
        for ptype, pairs in pair_types.items():
            topk_vals = []
            discrim_vals = []
            jaccard_vals = []

            for w1, w2 in pairs:
                v1 = compute_pair_overlap_topk(w1, w2)
                v2 = compute_pair_overlap_discrim(w1, w2)
                v3 = compute_pair_jaccard(w1, w2)
                if v1 is not None:
                    topk_vals.append(v1)
                if v2 is not None:
                    discrim_vals.append(v2)
                if v3 is not None:
                    jaccard_vals.append(v3)

            layer_results[ptype] = {
                "topk_mean": float(np.mean(topk_vals)) if topk_vals else None,
                "topk_std": float(np.std(topk_vals)) if topk_vals else None,
                "discrim_mean": float(np.mean(discrim_vals)) if discrim_vals else None,
                "discrim_std": float(np.std(discrim_vals)) if discrim_vals else None,
                "jaccard_mean": float(np.mean(jaccard_vals)) if jaccard_vals else None,
                "jaccard_std": float(np.std(jaccard_vals)) if jaccard_vals else None,
                "n_pairs": len(pairs),
            }
            log_time(f"    {ptype}: topk={np.mean(topk_vals):.4f}, "
                     f"discrim={np.mean(discrim_vals):.4f}, "
                     f"jaccard={np.mean(jaccard_vals):.4f}" if topk_vals else "")

        # === 核心检验: 反义词overlap > 近义词overlap 是否仍成立 ===
        ant_topk = layer_results["antonym"]["topk_mean"]
        syn_topk = layer_results["synonym"]["topk_mean"]
        ant_discrim = layer_results["antonym"]["discrim_mean"]
        syn_discrim = layer_results["synonym"]["discrim_mean"]

        layer_results["key_findings"] = {
            "antonym_gt_synonym_topk": ant_topk > syn_topk if ant_topk and syn_topk else None,
            "antonym_gt_synonym_discrim": ant_discrim > syn_discrim if ant_discrim and syn_discrim else None,
            "ant_topk": ant_topk,
            "syn_topk": syn_topk,
            "ant_discrim": ant_discrim,
            "syn_discrim": syn_discrim,
        }

        log_time(f"  KEY: antonym>synonym topk={ant_topk > syn_topk if ant_topk and syn_topk else 'N/A'}, "
                 f"discrim={ant_discrim > syn_discrim if ant_discrim and syn_discrim else 'N/A'}")

        results[f"layer_{li}"] = layer_results

    # 额外: 区分度维度分类准确率 (与Phase 61 Part1一致，作为校验)
    log_time("Computing discriminative classification accuracy...")
    for li in target_layers:
        hs = hidden[li]
        for gname, words in CONCEPT_GROUPS.items():
            X_all = []
            y_all = []
            for gi, w in enumerate(words):
                idxs = [i for i, sw in enumerate(sent_word_map) if sw == w]
                X_all.extend(hs[idxs])
                y_all.extend([gi] * len(idxs))
            X_all = np.array(X_all)
            y_all = np.array(y_all)

            # All dims
            clf_all = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
            from sklearn.model_selection import cross_val_score
            try:
                acc_all = float(np.mean(cross_val_score(clf_all, X_all, y_all, cv=3)))
            except:
                acc_all = None

            # Top-10 方差维度
            var_per_dim = np.var(X_all, axis=0)
            topk_dims = np.argsort(var_per_dim)[-10:]
            clf_topk = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
            try:
                acc_topk = float(np.mean(cross_val_score(clf_topk, X_all[:, topk_dims], y_all, cv=3)))
            except:
                acc_topk = None

            # Top-10 区分度维度 (1-vs-rest)
            clf_d = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
            try:
                clf_d.fit(X_all, y_all)
                discrim_dims = set()
                for ci in range(len(clf_d.coef_)):
                    top10 = np.argsort(np.abs(clf_d.coef_[ci]))[-10:]
                    discrim_dims.update(top10.tolist())
                discrim_dims = sorted(discrim_dims)[:10]
                clf_discrim = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
                acc_discrim = float(np.mean(cross_val_score(clf_discrim, X_all[:, discrim_dims], y_all, cv=3)))
            except:
                acc_discrim = None

            if f"layer_{li}" not in results:
                results[f"layer_{li}"] = {}
            results[f"layer_{li}"].setdefault("classification", {})[gname] = {
                "all_dims": acc_all,
                "top10_var": acc_topk,
                "top10_discrim": acc_discrim,
            }

    out_path = RESULT_DIR / f"phase62_part1_{model_name_global}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log_time(f"Part 1 results saved to {out_path}")
    return results


# =====================================================================
# Part 2: PC2方向精确解码
# 投影W_U解码top tokens，验证PC2是否对齐温度词汇
# =====================================================================

def run_part2(model, tokenizer, device):
    """
    62b: PC2方向精确语义解码
    - 投影W_U解码PC2的top boosted/suppressed tokens
    - 验证是否对齐温度词汇
    - 同时解码PC1, 验证是否对齐极端/温和词汇
    """
    from model_utils import get_model_info, get_W_U

    info = get_model_info(model, model_name_global)
    n_layers = info.n_layers
    d_model = info.d_model

    # 获取W_U
    log_time("Loading W_U...")
    W_U = get_W_U(model, model_name_global)  # [vocab_size, d_model]
    log_time(f"W_U shape: {W_U.shape}")

    target_layer = n_layers - 1  # 最高层

    # 词集
    axes = {
        "temperature": TEMPERATURE_WORDS,
        "size": SIZE_WORDS,
        "speed": SPEED_WORDS,
        "emotion": EMOTION_WORDS,
    }

    results = {}

    for axis_name, words in axes.items():
        log_time(f"  Decoding {axis_name} axis PC1/PC2...")

        # 收集hidden states
        templates = generate_templates(words)
        all_sents = []
        sent_word_map = []
        for w in words:
            for s in templates[w]:
                all_sents.append(s)
                sent_word_map.append(w)

        hidden = collect_hidden_states(model, tokenizer, device, all_sents, [target_layer])
        hs = hidden[target_layer]  # [n_sents, d_model]

        # 计算每个词的平均激活
        word_means = {}
        for w in words:
            idxs = [i for i, sw in enumerate(sent_word_map) if sw == w]
            word_means[w] = hs[idxs].mean(axis=0)

        # SVD提取PC1/PC2
        mean_act = np.mean(list(word_means.values()), axis=0)
        stacked = np.array([word_means[w] - mean_act for w in words])
        U, S, Vt = np.linalg.svd(stacked, full_matrices=False)

        pc1_direction = Vt[0]  # [d_model]
        pc2_direction = Vt[1]  # [d_model]

        # 投影W_U解码
        logit_pc1 = W_U @ pc1_direction  # [vocab_size]
        logit_pc2 = W_U @ pc2_direction  # [vocab_size]

        # Top boosted/suppressed tokens
        top_boosted_pc1 = np.argsort(logit_pc1)[-30:][::-1]
        top_suppressed_pc1 = np.argsort(logit_pc1)[:30]
        top_boosted_pc2 = np.argsort(logit_pc2)[-30:][::-1]
        top_suppressed_pc2 = np.argsort(logit_pc2)[:30]

        # 解码tokens
        def decode_tokens(ids):
            return [tokenizer.decode([int(i)]).strip() for i in ids]

        # 词在PC1/PC2上的投影分数
        word_pc1_scores = {w: float(np.dot(word_means[w] - mean_act, pc1_direction)) for w in words}
        word_pc2_scores = {w: float(np.dot(word_means[w] - mean_act, pc2_direction)) for w in words}

        # 检查温度词是否出现在PC2 top tokens中
        temp_words_set = set(TEMPERATURE_WORDS) if axis_name == "temperature" else set()
        temp_in_pc2_boosted = sum(1 for t in decode_tokens(top_boosted_pc2[:20]) if t.lower() in temp_words_set)
        temp_in_pc2_suppressed = sum(1 for t in decode_tokens(top_suppressed_pc2[:20]) if t.lower() in temp_words_set)

        axis_result = {
            "singular_values": S[:5].tolist(),
            "word_pc1_scores": word_pc1_scores,
            "word_pc2_scores": word_pc2_scores,
            "pc1_top_boosted": decode_tokens(top_boosted_pc1[:20]),
            "pc1_top_suppressed": decode_tokens(top_suppressed_pc1[:20]),
            "pc2_top_boosted": decode_tokens(top_boosted_pc2[:20]),
            "pc2_top_suppressed": decode_tokens(top_suppressed_pc2[:20]),
            "temperature_words_in_pc2_top20": temp_in_pc2_boosted + temp_in_pc2_suppressed,
            "pc1_variance_explained": float(S[0] ** 2 / np.sum(S ** 2)),
            "pc2_variance_explained": float(S[1] ** 2 / np.sum(S ** 2)),
        }

        log_time(f"  {axis_name} PC1 var={axis_result['pc1_variance_explained']:.3f}, "
                 f"PC2 var={axis_result['pc2_variance_explained']:.3f}")
        log_time(f"  PC1 top boosted: {decode_tokens(top_boosted_pc1[:10])}")
        log_time(f"  PC2 top boosted: {decode_tokens(top_boosted_pc2[:10])}")

        results[axis_name] = axis_result

        # 释放内存
        del hidden, hs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_path = RESULT_DIR / f"phase62_part2_{model_name_global}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log_time(f"Part 2 results saved to {out_path}")
    return results


# =====================================================================
# Part 3: 句法对照实验
# 排除"PC2编码句法模式而非温度语义"的替代解释
# =====================================================================

def run_part3(model, tokenizer, device):
    """
    62c: 句法对照实验
    - 构造温度prompt和颜色prompt (相同句式，不同语义)
    - 对两组prompt分别做PC2消融
    - 如果PC2编码的是温度语义: 温度prompt的KL >> 颜色prompt的KL
    - 如果PC2编码的是句法模式: 两者KL应相似
    """
    from model_utils import get_model_info

    info = get_model_info(model, model_name_global)
    n_layers = info.n_layers
    d_model = info.d_model

    target_layer = n_layers - 1  # 最高层，效果最明显

    # 温度相关prompt
    temp_prompts = [
        "The water felt",
        "He stepped outside and felt",
        "The room was",
        "The soup was",
        "The breeze was",
        "She touched the metal and it felt",
    ]

    # 颜色相关prompt (相同句式，不同语义)
    color_prompts = [
        "The wall was painted",
        "She wore a dress that was",
        "The car was",
        "The sky appeared",
        "The flowers were",
        "The painting was",
    ]

    # 大小相关prompt (第三个语义域)
    size_prompts = [
        "The building was",
        "The animal was",
        "The mountain was",
        "The box was",
        "The lake was",
        "The insect was",
    ]

    # 步骤1: 从温度词收集hidden states, 提取PC1/PC2
    log_time("Step 1: Extracting PC1/PC2 from temperature words...")
    templates = generate_templates(TEMPERATURE_WORDS)
    all_sents = []
    for w in TEMPERATURE_WORDS:
        all_sents.extend(templates[w])

    hidden = collect_hidden_states(model, tokenizer, device, all_sents, [target_layer])
    hs = hidden[target_layer]

    # 分词计算均值
    word_hidden = {}
    idx = 0
    for w in TEMPERATURE_WORDS:
        n = len(templates[w])
        word_hidden[w] = hs[idx:idx+n].mean(axis=0)
        idx += n

    mean_act = np.mean(list(word_hidden.values()), axis=0)
    stacked = np.array([word_hidden[w] - mean_act for w in TEMPERATURE_WORDS])
    U, S, Vt = np.linalg.svd(stacked, full_matrices=False)
    pc1_dir = Vt[0]
    pc2_dir = Vt[1]

    log_time(f"  PC1 var={S[0]**2/np.sum(S**2):.3f}, PC2 var={S[1]**2/np.sum(S**2):.3f}")

    del hidden, hs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 步骤2: 对每组prompt做消融实验
    log_time("Step 2: Running ablation experiments...")

    prompt_groups = {
        "temperature": temp_prompts,
        "color": color_prompts,
        "size": size_prompts,
    }

    results = {"pc_variance": [float(s) for s in S[:5]]}

    for ablation_type in ["remove_pc1", "remove_pc2", "amplify_pc1_pos", "amplify_pc2_pos"]:
        log_time(f"  Ablation: {ablation_type}")
        ablation_results = {}

        for group_name, prompts in prompt_groups.items():
            kl_values = []
            logit_changes = []

            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
                input_device = next(model.parameters()).device
                input_ids = inputs["input_ids"].to(input_device)
                attention_mask = inputs["attention_mask"].to(input_device)

                # Baseline: 正常前向传播
                with torch.no_grad():
                    base_out = model(input_ids=input_ids, attention_mask=attention_mask,
                                     output_hidden_states=True)

                # 收集target_layer的输出
                captured = {}
                layers = model.model.layers if hasattr(model.model, 'layers') else []

                def make_hook(key):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            captured[key] = output[0].detach().clone()
                        else:
                            captured[key] = output.detach().clone()
                    return hook

                hook = layers[target_layer].register_forward_hook(make_hook("target"))

                # Intervened: 在target_layer添加perturbation
                def add_perturbation_hook(direction, ablation_t, beta=20.0):
                    perturb_hooks = []

                    def perturb_hook(module, input, output):
                        if isinstance(output, tuple):
                            h = output[0]
                        else:
                            h = output

                        dir_tensor = torch.tensor(direction, dtype=h.dtype, device=h.device)

                        if ablation_t == "remove_pc1" or ablation_t == "remove_pc2":
                            # 移除PC方向的投影: h_new = h - (h·d / ||d||^2) * d
                            dir_norm_sq = dir_tensor.norm() ** 2 + 1e-10
                            proj_scalar = torch.einsum('bsd,d->bs', h, dir_tensor) / dir_norm_sq
                            proj = proj_scalar.unsqueeze(-1) * dir_tensor.unsqueeze(0).unsqueeze(0)
                            h_new = h - proj
                        elif ablation_t == "amplify_pc1_pos" or ablation_t == "amplify_pc2_pos":
                            h_new = h + beta * dir_tensor.unsqueeze(0).unsqueeze(0)
                        else:
                            h_new = h

                        if isinstance(output, tuple):
                            return (h_new,) + output[1:]
                        return h_new

                    perturb_hooks.append(layers[target_layer].register_forward_hook(perturb_hook))
                    return perturb_hooks

                direction = pc1_dir if "pc1" in ablation_type else pc2_dir
                perturb_hooks = add_perturbation_hook(direction, ablation_type)

                with torch.no_grad():
                    interv_out = model(input_ids=input_ids, attention_mask=attention_mask,
                                       output_hidden_states=True)

                # 移除hooks
                hook.remove()
                for h in perturb_hooks:
                    h.remove()

                # 计算KL散度
                base_logits = base_out.logits[0, -1].float()
                interv_logits = interv_out.logits[0, -1].float()

                # 使用稳定的KL计算
                p = torch.softmax(base_logits, dim=-1)
                q = torch.softmax(interv_logits, dim=-1)
                kl = float((p * (torch.log(p + 1e-10) - torch.log(q + 1e-10))).sum())
                kl_values.append(kl)

                # Top token变化
                base_top = tokenizer.decode([int(base_logits.argmax())]).strip()
                interv_top = tokenizer.decode([int(interv_logits.argmax())]).strip()
                logit_changes.append({"prompt": prompt, "base_top": base_top,
                                      "interv_top": interv_top, "changed": base_top != interv_top})

                del base_out, interv_out
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            ablation_results[group_name] = {
                "kl_mean": float(np.mean(kl_values)),
                "kl_std": float(np.std(kl_values)),
                "kl_values": kl_values,
                "logit_changes": logit_changes,
                "n_changed": sum(1 for lc in logit_changes if lc["changed"]),
            }
            log_time(f"    {group_name}: KL={np.mean(kl_values):.4f}, "
                     f"changed={sum(1 for lc in logit_changes if lc['changed'])}/{len(prompts)}")

        results[ablation_type] = ablation_results

    # === 核心判断 ===
    # 对每种ablation, 检查temperature的KL是否显著大于color和size
    key_findings = {}
    for ablation_type in ["remove_pc1", "remove_pc2", "amplify_pc1_pos", "amplify_pc2_pos"]:
        temp_kl = results[ablation_type]["temperature"]["kl_mean"]
        color_kl = results[ablation_type]["color"]["kl_mean"]
        size_kl = results[ablation_type]["size"]["kl_mean"]

        ratio_tc = temp_kl / max(color_kl, 1e-10)
        ratio_ts = temp_kl / max(size_kl, 1e-10)

        key_findings[ablation_type] = {
            "temp_kl": temp_kl,
            "color_kl": color_kl,
            "size_kl": size_kl,
            "temp_color_ratio": ratio_tc,
            "temp_size_ratio": ratio_ts,
            "semantic_specific": ratio_tc > 1.5 and ratio_ts > 1.5,
        }
        log_time(f"  KEY {ablation_type}: temp/color ratio={ratio_tc:.2f}, "
                 f"temp/size ratio={ratio_ts:.2f}, semantic_specific={ratio_tc > 1.5 and ratio_ts > 1.5}")

    results["key_findings"] = key_findings

    out_path = RESULT_DIR / f"phase62_part3_{model_name_global}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log_time(f"Part 3 results saved to {out_path}")
    return results


# =====================================================================
# Part 4: 跨轴PC1一致性 + 词集稳定性检验
# =====================================================================

def run_part4(model, tokenizer, device):
    """
    62d: 
    A. 跨轴PC1一致性检验 — 是否存在通用"强度/极性"维度
    B. 词集稳定性检验 — 替换词集后PC1/PC2方向是否一致
    """
    from model_utils import get_model_info

    info = get_model_info(model, model_name_global)
    n_layers = info.n_layers
    d_model = info.d_model

    target_layer = n_layers - 1  # 最高层

    # ============ A. 跨轴PC1一致性 ============
    log_time("Part 4A: Cross-axis PC1 consistency check...")

    axes_wordsets = {
        "temperature": TEMPERATURE_WORDS,
        "temperature_v2": TEMPERATURE_WORDS_V2,
        "size": SIZE_WORDS,
        "size_v2": SIZE_WORDS_V2,
        "speed": SPEED_WORDS,
        "speed_v2": SPEED_WORDS_V2,
        "emotion": EMOTION_WORDS,
    }

    # 收集所有轴的hidden states
    pc_directions = {}

    for axis_name, words in axes_wordsets.items():
        log_time(f"  Extracting PC1/PC2 for {axis_name} ({len(words)} words)...")
        templates = generate_templates(words)
        all_sents = []
        for w in words:
            all_sents.extend(templates[w])

        hidden = collect_hidden_states(model, tokenizer, device, all_sents, [target_layer])
        hs = hidden[target_layer]

        word_hidden = {}
        idx = 0
        for w in words:
            n = len(templates[w])
            word_hidden[w] = hs[idx:idx+n].mean(axis=0)
            idx += n

        mean_act = np.mean(list(word_hidden.values()), axis=0)
        stacked = np.array([word_hidden[w] - mean_act for w in words])
        U, S, Vt = np.linalg.svd(stacked, full_matrices=False)

        pc_directions[axis_name] = {
            "pc1": Vt[0],
            "pc2": Vt[1],
            "pc3": Vt[2] if len(Vt) > 2 else None,
            "singular_values": S[:5].tolist(),
            "pc1_var": float(S[0] ** 2 / np.sum(S ** 2)),
            "pc2_var": float(S[1] ** 2 / np.sum(S ** 2)),
            "word_pc1_scores": {w: float(np.dot(word_hidden[w] - mean_act, Vt[0])) for w in words},
            "word_pc2_scores": {w: float(np.dot(word_hidden[w] - mean_act, Vt[1])) for w in words},
        }

        log_time(f"    {axis_name}: PC1 var={pc_directions[axis_name]['pc1_var']:.3f}, "
                 f"PC2 var={pc_directions[axis_name]['pc2_var']:.3f}")

        del hidden, hs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 计算跨轴PC1/PC2 cosine相似度
    log_time("Computing cross-axis cosine similarities...")
    main_axes = ["temperature", "size", "speed", "emotion"]
    cross_axis_results = {}

    # PC1 cross-axis
    pc1_cosines = {}
    for i, a1 in enumerate(main_axes):
        for a2 in main_axes[i+1:]:
            cos_val = float(np.dot(pc_directions[a1]["pc1"], pc_directions[a2]["pc1"]) /
                          (np.linalg.norm(pc_directions[a1]["pc1"]) * np.linalg.norm(pc_directions[a2]["pc1"]) + 1e-10))
            pc1_cosines[f"{a1}_vs_{a2}"] = cos_val
            log_time(f"  PC1 cosine({a1}, {a2}) = {cos_val:.4f}")

    # PC2 cross-axis
    pc2_cosines = {}
    for i, a1 in enumerate(main_axes):
        for a2 in main_axes[i+1:]:
            cos_val = float(np.dot(pc_directions[a1]["pc2"], pc_directions[a2]["pc2"]) /
                          (np.linalg.norm(pc_directions[a1]["pc2"]) * np.linalg.norm(pc_directions[a2]["pc2"]) + 1e-10))
            pc2_cosines[f"{a1}_vs_{a2}"] = cos_val
            log_time(f"  PC2 cosine({a1}, {a2}) = {cos_val:.4f}")

    # ============ B. 词集稳定性检验 ============
    log_time("Part 4B: Word-set stability check...")

    stability_results = {}
    for axis_base in ["temperature", "size", "speed"]:
        axis_v2 = f"{axis_base}_v2"
        if axis_base in pc_directions and axis_v2 in pc_directions:
            # PC1稳定性
            cos_pc1 = float(np.dot(pc_directions[axis_base]["pc1"], pc_directions[axis_v2]["pc1"]) /
                          (np.linalg.norm(pc_directions[axis_base]["pc1"]) * np.linalg.norm(pc_directions[axis_v2]["pc1"]) + 1e-10))
            # PC2稳定性
            cos_pc2 = float(np.dot(pc_directions[axis_base]["pc2"], pc_directions[axis_v2]["pc2"]) /
                          (np.linalg.norm(pc_directions[axis_base]["pc2"]) * np.linalg.norm(pc_directions[axis_v2]["pc2"]) + 1e-10))
            # 注意: PC方向可能反号，取绝对值
            cos_pc1_abs = abs(cos_pc1)
            cos_pc2_abs = abs(cos_pc2)

            stability_results[axis_base] = {
                "pc1_cosine": cos_pc1,
                "pc1_cosine_abs": cos_pc1_abs,
                "pc2_cosine": cos_pc2,
                "pc2_cosine_abs": cos_pc2_abs,
                "pc1_stable": cos_pc1_abs > 0.7,
                "pc2_stable": cos_pc2_abs > 0.7,
            }
            log_time(f"  {axis_base} stability: |cos(PC1)|={cos_pc1_abs:.4f}, "
                     f"|cos(PC2)|={cos_pc2_abs:.4f}, "
                     f"PC1_stable={cos_pc1_abs > 0.7}, PC2_stable={cos_pc2_abs > 0.7}")

    # ============ C. 层间PC1/PC2稳定性 (补充) ============
    log_time("Part 4C: Layer-wise PC1/PC2 stability for temperature axis...")
    sample_layers = [n_layers // 4, n_layers // 2, n_layers * 3 // 4, n_layers - 1]

    layer_pc_directions = {}
    templates = generate_templates(TEMPERATURE_WORDS)
    all_sents = []
    for w in TEMPERATURE_WORDS:
        all_sents.extend(templates[w])

    hidden = collect_hidden_states(model, tokenizer, device, all_sents, sample_layers)

    for li in sample_layers:
        hs = hidden[li]
        word_hidden = {}
        idx = 0
        for w in TEMPERATURE_WORDS:
            n = len(templates[w])
            word_hidden[w] = hs[idx:idx+n].mean(axis=0)
            idx += n

        mean_act = np.mean(list(word_hidden.values()), axis=0)
        stacked = np.array([word_hidden[w] - mean_act for w in TEMPERATURE_WORDS])
        U, S, Vt = np.linalg.svd(stacked, full_matrices=False)

        layer_pc_directions[f"L{li}"] = {
            "pc1": Vt[0],
            "pc2": Vt[1],
            "pc1_var": float(S[0] ** 2 / np.sum(S ** 2)),
            "pc2_var": float(S[1] ** 2 / np.sum(S ** 2)),
        }
        log_time(f"    L{li}: PC1 var={layer_pc_directions[f'L{li}']['pc1_var']:.3f}")

    # 层间PC1一致性
    layer_names = sorted(layer_pc_directions.keys())
    layer_stability = {}
    for i, l1 in enumerate(layer_names):
        for l2 in layer_names[i+1:]:
            cos_pc1 = abs(float(np.dot(layer_pc_directions[l1]["pc1"], layer_pc_directions[l2]["pc1"]) /
                               (np.linalg.norm(layer_pc_directions[l1]["pc1"]) * np.linalg.norm(layer_pc_directions[l2]["pc1"]) + 1e-10)))
            cos_pc2 = abs(float(np.dot(layer_pc_directions[l1]["pc2"], layer_pc_directions[l2]["pc2"]) /
                               (np.linalg.norm(layer_pc_directions[l1]["pc2"]) * np.linalg.norm(layer_pc_directions[l2]["pc2"]) + 1e-10)))
            layer_stability[f"{l1}_vs_{l2}"] = {
                "pc1_cos_abs": cos_pc1,
                "pc2_cos_abs": cos_pc2,
            }
            log_time(f"    {l1} vs {l2}: |cos(PC1)|={cos_pc1:.4f}, |cos(PC2)|={cos_pc2:.4f}")

    # 汇总结果
    results = {
        "cross_axis_pc1_cosines": pc1_cosines,
        "cross_axis_pc2_cosines": pc2_cosines,
        "word_set_stability": stability_results,
        "layer_stability": layer_stability,
        "axis_details": {
            name: {
                "pc1_var": data["pc1_var"],
                "pc2_var": data["pc2_var"],
                "singular_values": data["singular_values"],
                "word_pc1_scores": data["word_pc1_scores"],
                "word_pc2_scores": data["word_pc2_scores"],
            }
            for name, data in pc_directions.items()
        },
        "key_findings": {
            "universal_intensity_axis": all(v > 0.7 for v in pc1_cosines.values()) if pc1_cosines else False,
            "no_universal_axis": all(v < 0.3 for v in pc1_cosines.values()) if pc1_cosines else False,
            "word_set_stable_pc1": all(v["pc1_stable"] for v in stability_results.values()) if stability_results else False,
            "word_set_stable_pc2": all(v["pc2_stable"] for v in stability_results.values()) if stability_results else False,
            "avg_pc1_cross_axis_cos": float(np.mean(list(pc1_cosines.values()))) if pc1_cosines else None,
            "avg_pc2_cross_axis_cos": float(np.mean(list(pc2_cosines.values()))) if pc2_cosines else None,
        }
    }

    log_time(f"  KEY: universal_intensity_axis={results['key_findings']['universal_intensity_axis']}")
    log_time(f"  KEY: no_universal_axis={results['key_findings']['no_universal_axis']}")
    log_time(f"  KEY: avg_pc1_cross_axis_cos={results['key_findings']['avg_pc1_cross_axis_cos']}")
    log_time(f"  KEY: word_set_stable_pc1={results['key_findings']['word_set_stable_pc1']}")
    log_time(f"  KEY: word_set_stable_pc2={results['key_findings']['word_set_stable_pc2']}")

    out_path = RESULT_DIR / f"phase62_part4_{model_name_global}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log_time(f"Part 4 results saved to {out_path}")
    return results


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--part", type=str, required=True, choices=["1", "2", "3", "4", "all"])
    args = parser.parse_args()

    global model_name_global
    model_name_global = args.model

    log_time(f"=== Phase 62: {args.model} Part {args.part} ===")

    model, tokenizer, device = load_model_bf16(args.model)

    if args.part in ("1", "all"):
        log_time("--- Running Part 1: Discrim overlap re-verification ---")
        run_part1(model, tokenizer, device)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.part in ("2", "all"):
        log_time("--- Running Part 2: PC2 direction decoding ---")
        run_part2(model, tokenizer, device)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.part in ("3", "all"):
        log_time("--- Running Part 3: Syntactic control experiment ---")
        run_part3(model, tokenizer, device)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.part in ("4", "all"):
        log_time("--- Running Part 4: Cross-axis PC1 + word-set stability ---")
        run_part4(model, tokenizer, device)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 释放模型
    from model_utils import release_model
    release_model(model)
    log_time(f"=== Phase 62 {args.model} Part {args.part} COMPLETE ===")


if __name__ == "__main__":
    main()
