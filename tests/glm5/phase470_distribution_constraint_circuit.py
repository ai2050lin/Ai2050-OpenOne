"""
Phase 470: 分布约束指纹与关系槽位分离 — 基于分布控制接口理论
==============================================================
理论核心: Meaning = stable control over future probability distribution
          意义 = 对未来概率分布的稳定控制

核心实验:
  Exp1: Distributional Constraint Fingerprint (DCF) 构造
        — 定义每个对象对未来分布的稳定约束向量
        — 验证DCF比原始残差cos更好地聚类类别
  Exp2: Relation Slot 分离
        — 证明同一对象在不同关系槽位下产生不同分布约束
        — 对象码 ≠ 固定概念方向, 而是条件化约束
  Exp3: DCF跨模型对齐检查
        — 比较不同模型的DCF结构是否相似
        — 验证"约束不变性"假设

模型加载: bfloat16 + device_map="auto" + flash_attention_2

用法:
  python tests/glm5/phase470_distribution_constraint_circuit.py qwen3 1
  python tests/glm5/phase470_distribution_constraint_circuit.py glm4 1
  python tests/glm5/phase470_distribution_constraint_circuit.py deepseek7b 1
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')
import os, gc, time, json, math
import numpy as np
import torch
from model_utils import (get_layers, get_model_info, release_model, get_W_U, MODEL_CONFIGS)

def plog(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ==================== 数据定义 ====================
CATEGORIES = {
    "fruit":    ["apple", "banana", "orange", "grape", "pear", "peach", "lemon", "mango"],
    "animal":   ["dog", "cat", "horse", "lion", "bear", "rabbit", "cow", "tiger"],
    "tool":     ["hammer", "knife", "wrench", "saw", "drill", "axe", "shovel", "scissors"],
    "vehicle":  ["car", "bus", "bicycle", "truck", "train", "boat", "plane", "scooter"],
    "clothing": ["shirt", "dress", "hat", "coat", "sock", "glove", "scarf", "boot"],
    "furniture":["chair", "table", "desk", "sofa", "bed", "shelf", "lamp", "cabinet"],
}

# 分布约束指纹的候选族词 — 8个类别维度
FAMILY_WORDS = {
    "fruit":    ["fruit", "produce", "crop", "berry"],
    "animal":   ["animal", "creature", "beast", "pet"],
    "tool":     ["tool", "implement", "device", "instrument"],
    "vehicle":  ["vehicle", "transport", "automobile", "car"],
    "clothing": ["clothing", "attire", "wear", "garment"],
    "furniture":["furniture", "furnishing", "fixture", "seat"],
    "food":     ["food", "meal", "dish", "snack"],
    "plant":    ["plant", "tree", "vegetation", "flora"],
}

# 关系槽位模板 — 5个不同关系
RELATION_TEMPLATES = {
    "kind_of":    "The {obj} is a kind of",
    "used_for":   "The {obj} is commonly used for",
    "found_in":   "The {obj} is typically found in",
    "made_of":    "The {obj} is typically made of",
    "related_to": "The {obj} is closely related to",
}

# 翻译模板(跨语言)
TRANSLATE_TEMPLATES = {
    "en_to_zh": "The Chinese translation of '{obj}' is",
    "zh_to_en": "The English translation of '苹果' is",  # 仅apple
}

ROUNDS = {
    1: {k: v[:6] for k, v in CATEGORIES.items()},   # R1: 6对象/类
    2: {k: v[:8] for k, v in CATEGORIES.items()},   # R2: 8对象/类
}


# ==================== 模型加载 ====================
def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    plog(f"Loading {model_name} (bfloat16 + device_map=auto + flash)...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, local_files_only=True,
            attn_implementation="flash_attention_2",
        )
        plog(f"  flash_attention_2 loaded OK")
    except Exception as e:
        plog(f"  flash_attention_2 failed ({e}), falling back to eager")
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, local_files_only=True,
            attn_implementation="eager",
        )

    model.eval()
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    plog(f"  {model_name}: device={device}, GPU={gpu_mem:.2f}GB")

    if hasattr(model, 'hf_device_map'):
        dmap = model.hf_device_map
        gpu_count = sum(1 for v in dmap.values() if 'cuda' in str(v))
        cpu_count = sum(1 for v in dmap.values() if 'cpu' in str(v))
        plog(f"  Layer dist: GPU={gpu_count}, CPU={cpu_count}")

    return model, tokenizer, device


# ==================== 基础工具 ====================
def get_final_logits(model, tokenizer, prompt, device):
    """获取最后一层logits"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    return out.logits[0, -1].float().cpu().numpy()


def get_residual_at_layer(model, tokenizer, prompt, layer_idx, device, pos=-1):
    """提取指定层残差流"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    seq_len = attention_mask.sum().item()

    captured = {}
    layers = get_layers(model)
    def hook_fn(module, inp, output):
        if isinstance(inp, tuple) and len(inp) > 0:
            captured['resid'] = inp[0].detach().float().cpu()

    h = layers[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)
    h.remove()

    if 'resid' in captured:
        p = seq_len - 1 if pos == -1 else pos
        return captured['resid'][0, p].numpy(), seq_len
    return None, 0


def logit_entropy(logits_vec):
    """计算logit分布的熵"""
    log_probs = logits_vec - np.max(logits_vec)
    log_probs = log_probs - np.log(np.sum(np.exp(log_probs)) + 1e-30)
    return -float(np.sum(np.exp(log_probs) * log_probs))


def top_k_probability(logits_vec, k=5):
    """Top-k候选的平均概率"""
    log_probs = logits_vec - np.max(logits_vec)
    probs = np.exp(log_probs) / np.sum(np.exp(log_probs))
    top_k = np.sort(probs)[-k:]
    return float(np.mean(top_k))


def find_token_id(tokenizer, word):
    """寻找词的token id(处理前缀空格)"""
    vocab = tokenizer.get_vocab()
    for candidate in [word, f" {word}", word.lower(), f" {word.lower()}"]:
        if candidate in vocab:
            return vocab[candidate]
    return None


def compute_dcf(logits, tokenizer, family_words_dict=None):
    """
    计算分布约束指纹 (Distributional Constraint Fingerprint)
    
    DCF(x) = [ΔlogP(family_k | x)] for k in families
    其中 ΔlogP = logP(family_k | x) - mean_logP(family_k)
    
    即: 每个类别族词的logit减去该词在所有样本中的平均logit,
    得到该对象对各类别分布的约束效应。
    
    但更直接的做法: 直接用logit值作为约束强度,
    因为我们要的是"对象x让哪些类别族词的logit升高"。
    
    简化版: DCF(x) = [mean_logit(family_k_words)] for k in families
    """
    if family_words_dict is None:
        family_words_dict = FAMILY_WORDS

    vocab = tokenizer.get_vocab()
    dcf_vector = []
    dcf_details = {}

    for family_name, words in family_words_dict.items():
        logit_values = []
        valid_words = []
        for w in words:
            tid = find_token_id(tokenizer, w)
            if tid is not None:
                logit_values.append(float(logits[tid]))
                valid_words.append(w)

        if logit_values:
            mean_logit = float(np.mean(logit_values))
            dcf_vector.append(mean_logit)
            dcf_details[family_name] = {
                "mean_logit": round(mean_logit, 4),
                "words": valid_words,
                "n_valid": len(valid_words),
            }
        else:
            dcf_vector.append(0.0)
            dcf_details[family_name] = {"mean_logit": 0, "words": [], "n_valid": 0}

    return np.array(dcf_vector), dcf_details


def compute_dcf_centered(dcf_vectors, labels):
    """
    计算中心化的DCF(减去全局均值)
    这是真正的约束指纹: 每个对象相对于平均水平的偏差
    """
    dcf_matrix = np.array(dcf_vectors)
    global_mean = np.mean(dcf_matrix, axis=0)
    centered = dcf_matrix - global_mean
    return centered, global_mean


def cluster_quality(vectors, labels, method='silhouette'):
    """
    评估聚类质量(同一类别的对象是否在向量空间中聚在一起)
    返回轮廓系数(越高越好, 范围[-1,1])
    """
    from scipy.spatial.distance import pdist, squareform

    if len(vectors) < 3 or len(set(labels)) < 2:
        return 0.0

    vectors = np.array(vectors)
    labels = np.array(labels)

    # 归一化
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    vectors_norm = vectors / norms

    # 计算距离矩阵
    dist_matrix = squareform(pdist(vectors_norm, metric='cosine'))

    # 计算轮廓系数(简化版)
    silhouette_values = []
    unique_labels = list(set(labels))

    for i in range(len(vectors)):
        own_label = labels[i]
        own_cluster = [j for j in range(len(vectors)) if labels[j] == own_label and j != i]
        other_clusters = {l: [j for j in range(len(vectors)) if labels[j] == l] for l in unique_labels if l != own_label}

        if not own_cluster:
            continue

        # a: 同类内平均距离
        a = np.mean([dist_matrix[i, j] for j in own_cluster]) if own_cluster else 0

        # b: 最近异类平均距离
        b = float('inf')
        for l, indices in other_clusters.items():
            if indices:
                mean_dist = np.mean([dist_matrix[i, j] for j in indices])
                b = min(b, mean_dist)

        if b == float('inf'):
            b = 0

        # 轮廓系数
        s = (b - a) / max(a, b, 1e-10)
        silhouette_values.append(s)

    return float(np.mean(silhouette_values)) if silhouette_values else 0.0


# ==================== Exp1: DCF构造与聚类验证 ====================
def exp1_dcf_construction(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    核心实验: 构造每个对象的分布约束指纹, 验证DCF比残差cos更好地聚类类别
    
    方法:
    1. 对每个对象在kind_of模板下, 提取最终logits
    2. 计算DCF: 每个类别族词的mean logit向量
    3. 比较DCF聚类质量 vs 残差cos聚类质量
    4. 在多层检查DCF的层间演变
    """
    plog("=== Exp1: DCF构造与聚类验证 ===")
    info = get_model_info(model, model_name)
    n_layers = info.n_layers

    n_obj = 6 if round_num == 1 else 8
    sample_layers = sorted(set([
        0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1,
    ]))

    results = {}
    cat_list = ["fruit", "animal", "vehicle", "tool", "furniture", "clothing"]

    for layer_idx in sample_layers:
        plog(f"  Layer L{layer_idx}...")
        layer_result = {}

        # ---- 1a. 在kind_of模板下收集所有对象的DCF和残差 ----
        all_dcf_vectors = []
        all_resid_vectors = []
        all_labels = []
        all_objects = []

        for cat in cat_list:
            objs = obj_dict.get(cat, [])[:n_obj]
            for obj in objs:
                prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)

                # 最终logits → DCF
                logits = get_final_logits(model, tokenizer, prompt, device)
                dcf_vec, dcf_detail = compute_dcf(logits, tokenizer)
                all_dcf_vectors.append(dcf_vec)
                all_labels.append(cat)
                all_objects.append(obj)

                # 残差向量
                resid, _ = get_residual_at_layer(model, tokenizer, prompt, layer_idx, device)
                if resid is not None:
                    all_resid_vectors.append(resid)

        if len(all_dcf_vectors) < 10:
            plog(f"    L{layer_idx}: too few samples, skip")
            continue

        # ---- 1b. 中心化DCF ----
        dcf_centered, global_mean = compute_dcf_centered(all_dcf_vectors, all_labels)

        # ---- 1c. 聚类质量比较: DCF vs 残差cos ----
        dcf_silhouette = cluster_quality(all_dcf_vectors, all_labels)
        dcf_centered_silhouette = cluster_quality(dcf_centered, all_labels)

        resid_silhouette = 0.0
        if len(all_resid_vectors) == len(all_dcf_vectors):
            resid_silhouette = cluster_quality(all_resid_vectors, all_labels)

        layer_result["clustering"] = {
            "dcf_silhouette": round(dcf_silhouette, 4),
            "dcf_centered_silhouette": round(dcf_centered_silhouette, 4),
            "resid_silhouette": round(resid_silhouette, 4),
            "dcf_advantage": round(dcf_silhouette - resid_silhouette, 4),
            "n_objects": len(all_dcf_vectors),
            "n_categories": len(set(all_labels)),
        }

        plog(f"    Silhouette: DCF={dcf_silhouette:.4f}, DCF_centered={dcf_centered_silhouette:.4f}, "
             f"Resid={resid_silhouette:.4f}, DCF_advantage={dcf_silhouette-resid_silhouette:.4f}")

        # ---- 1d. DCF类别分隔分析 ----
        # 计算每个类别的DCF中心, 看哪些类别在DCF空间中分离得好
        dcf_matrix = np.array(all_dcf_vectors)
        cat_centers = {}
        for cat in cat_list:
            mask = [i for i, l in enumerate(all_labels) if l == cat]
            if mask:
                cat_centers[cat] = np.mean(dcf_matrix[mask], axis=0).tolist()

        # 类别间DCF距离
        cat_pair_dists = {}
        for i, c1 in enumerate(cat_list):
            for j, c2 in enumerate(cat_list):
                if i < j and c1 in cat_centers and c2 in cat_centers:
                    dist = float(np.linalg.norm(
                        np.array(cat_centers[c1]) - np.array(cat_centers[c2])))
                    cat_pair_dists[f"{c1}_vs_{c2}"] = round(dist, 4)

        # 类内DCF散度
        cat_spreads = {}
        for cat in cat_list:
            mask = [i for i, l in enumerate(all_labels) if l == cat]
            if len(mask) >= 2:
                cat_vecs = dcf_matrix[mask]
                center = np.mean(cat_vecs, axis=0)
                spread = float(np.mean([np.linalg.norm(v - center) for v in cat_vecs]))
                cat_spreads[cat] = round(spread, 4)

        layer_result["category_structure"] = {
            "cat_centers_dim_order": list(FAMILY_WORDS.keys()),
            "cat_centers": {k: [round(x, 4) for x in v] for k, v in cat_centers.items()},
            "cat_spreads": cat_spreads,
            "min_inter_cat_dist": round(min(cat_pair_dists.values()), 4) if cat_pair_dists else 0,
            "max_intra_cat_spread": round(max(cat_spreads.values()), 4) if cat_spreads else 0,
            "inter_intra_ratio": round(min(cat_pair_dists.values()) / max(max(cat_spreads.values()), 1e-6), 4) if cat_pair_dists and cat_spreads else 0,
        }

        # ---- 1e. 每个对象的DCF详细记录 ----
        object_dcf_records = []
        for idx in range(len(all_objects)):
            record = {
                "object": all_objects[idx],
                "category": all_labels[idx],
                "dcf_raw": [round(x, 4) for x in all_dcf_vectors[idx].tolist()],
                "dcf_centered": [round(x, 4) for x in dcf_centered[idx].tolist()],
                # 哪个类别维度的DCF最高(排除自身)?
                "top_constraint_dim": list(FAMILY_WORDS.keys())[int(np.argmax(dcf_centered[idx]))],
                "correct_constraint_rank": int(list(np.argsort(dcf_centered[idx])[::-1]).index(list(FAMILY_WORDS.keys()).index(all_labels[idx]))) if all_labels[idx] in FAMILY_WORDS else -1,
            }
            object_dcf_records.append(record)

        layer_result["object_dcf"] = object_dcf_records

        # ---- 1f. DCF vs 残差cos 逐对象对比 ----
        if len(all_resid_vectors) == len(all_dcf_vectors):
            # 对每对同类对象, 计算DCF cos和resid cos
            same_cat_dcf_cos = []
            same_cat_resid_cos = []
            diff_cat_dcf_cos = []
            diff_cat_resid_cos = []

            resid_matrix = np.array(all_resid_vectors)
            for i in range(min(30, len(all_objects))):
                for j in range(i+1, min(30, len(all_objects))):
                    # DCF cos
                    d_i, d_j = dcf_centered[i], dcf_centered[j]
                    n_i, n_j = np.linalg.norm(d_i), np.linalg.norm(d_j)
                    if n_i > 1e-10 and n_j > 1e-10:
                        dcf_cos = float(np.dot(d_i, d_j) / (n_i * n_j))
                    else:
                        dcf_cos = 0.0

                    # Resid cos
                    r_i, r_j = resid_matrix[i], resid_matrix[j]
                    n_ri, n_rj = np.linalg.norm(r_i), np.linalg.norm(r_j)
                    if n_ri > 1e-10 and n_rj > 1e-10:
                        resid_cos = float(np.dot(r_i, r_j) / (n_ri * n_rj))
                    else:
                        resid_cos = 0.0

                    if all_labels[i] == all_labels[j]:
                        same_cat_dcf_cos.append(dcf_cos)
                        same_cat_resid_cos.append(resid_cos)
                    else:
                        diff_cat_dcf_cos.append(dcf_cos)
                        diff_cat_resid_cos.append(resid_cos)

            layer_result["pairwise_comparison"] = {
                "same_cat_dcf_cos_mean": round(float(np.mean(same_cat_dcf_cos)), 4) if same_cat_dcf_cos else 0,
                "same_cat_resid_cos_mean": round(float(np.mean(same_cat_resid_cos)), 4) if same_cat_resid_cos else 0,
                "diff_cat_dcf_cos_mean": round(float(np.mean(diff_cat_dcf_cos)), 4) if diff_cat_dcf_cos else 0,
                "diff_cat_resid_cos_mean": round(float(np.mean(diff_cat_resid_cos)), 4) if diff_cat_resid_cos else 0,
                "dcf_discriminability": round(
                    (float(np.mean(same_cat_dcf_cos)) - float(np.mean(diff_cat_dcf_cos))) if same_cat_dcf_cos and diff_cat_dcf_cos else 0, 4),
                "resid_discriminability": round(
                    (float(np.mean(same_cat_resid_cos)) - float(np.mean(diff_cat_resid_cos))) if same_cat_resid_cos and diff_cat_resid_cos else 0, 4),
            }
            plog(f"    Pairwise: DCF_disc={layer_result['pairwise_comparison']['dcf_discriminability']:.4f}, "
                 f"Resid_disc={layer_result['pairwise_comparison']['resid_discriminability']:.4f}")

        results[f"L{layer_idx}"] = layer_result

    # ---- 汇总 ----
    summary = {
        "dcf_wins_over_resid": sum(1 for k, v in results.items()
                                   if v.get("clustering", {}).get("dcf_advantage", 0) > 0),
        "total_layers_tested": len(results),
        "best_dcf_silhouette_layer": max(results.keys(),
                                          key=lambda k: results[k].get("clustering", {}).get("dcf_silhouette", 0)) if results else "N/A",
        "theory_prediction": "DCF should cluster categories better than residual cosine",
    }
    results["summary"] = summary
    plog(f"  Exp1 Summary: DCF wins in {summary['dcf_wins_over_resid']}/{summary['total_layers_tested']} layers")

    return results


# ==================== Exp2: 关系槽位分离 ====================
def exp2_relation_slot_separation(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    核心实验: 证明对象码不是固定概念方向, 而是条件化约束
    
    方法:
    1. 对同一对象, 在5个不同关系槽位下提取DCF
    2. 比较不同关系下DCF的差异
    3. 验证: kind_of→类别约束, used_for→功能约束, found_in→场景约束
    
    预期: 同一对象的DCF在不同关系下显著不同
    """
    plog("=== Exp2: 关系槽位分离 ===")
    info = get_model_info(model, model_name)

    n_obj = 4 if round_num == 1 else 6
    results = {}

    # 测试对象: 每个类别2个
    test_objects = []
    for cat in ["fruit", "animal", "vehicle", "tool"]:
        objs = obj_dict.get(cat, [])[:2]
        for obj in objs:
            test_objects.append((obj, cat))

    relation_keys = list(RELATION_TEMPLATES.keys())

    # ---- 2a. 对每个对象计算5个关系下的DCF ----
    plog(f"  Testing {len(test_objects)} objects × {len(relation_keys)} relations...")

    all_object_results = {}

    for obj_idx, (obj_name, obj_cat) in enumerate(test_objects):
        plog(f"    [{obj_idx+1}/{len(test_objects)}] {obj_name} ({obj_cat})")
        obj_result = {}

        for rel_key in relation_keys:
            prompt = RELATION_TEMPLATES[rel_key].format(obj=obj_name)
            logits = get_final_logits(model, tokenizer, prompt, device)
            dcf_vec, dcf_detail = compute_dcf(logits, tokenizer)
            ent = logit_entropy(logits)

            obj_result[rel_key] = {
                "dcf_raw": [round(x, 4) for x in dcf_vec.tolist()],
                "dcf_detail": dcf_detail,
                "entropy": round(ent, 4),
                "prompt": prompt,
            }

        all_object_results[obj_name] = obj_result

    # ---- 2b. 计算关系间DCF差异 ----
    # 对每个对象: 不同关系下DCF的中心化版本, 然后计算关系间cos
    relation_separation = {}

    for obj_name, obj_data in all_object_results.items():
        # 收集5个关系下的DCF
        dcf_by_rel = {k: np.array(v["dcf_raw"]) for k, v in obj_data.items()}
        dcf_matrix = np.array([dcf_by_rel[k] for k in relation_keys])

        # 中心化
        dcf_mean = np.mean(dcf_matrix, axis=0)
        dcf_centered = dcf_matrix - dcf_mean

        # 关系间cos矩阵
        cos_matrix = np.zeros((len(relation_keys), len(relation_keys)))
        for i in range(len(relation_keys)):
            for j in range(len(relation_keys)):
                ni, nj = np.linalg.norm(dcf_centered[i]), np.linalg.norm(dcf_centered[j])
                if ni > 1e-10 and nj > 1e-10:
                    cos_matrix[i, j] = float(np.dot(dcf_centered[i], dcf_centered[j]) / (ni * nj))
                else:
                    cos_matrix[i, j] = 0.0

        # 平均非对角cos(越低 = 关系分离越好)
        off_diag_cos = []
        for i in range(len(relation_keys)):
            for j in range(i+1, len(relation_keys)):
                off_diag_cos.append(cos_matrix[i, j])

        mean_off_diag = float(np.mean(off_diag_cos)) if off_diag_cos else 0

        # 每个关系的top约束维度
        top_constraints = {}
        for ri, rel in enumerate(relation_keys):
            top_dim_idx = int(np.argmax(dcf_centered[ri]))
            top_dim_name = list(FAMILY_WORDS.keys())[top_dim_idx]
            top_constraints[rel] = {
                "top_dim": top_dim_name,
                "top_value": round(float(dcf_centered[ri][top_dim_idx]), 4),
            }

        relation_separation[obj_name] = {
            "mean_inter_relation_cos": round(mean_off_diag, 4),
            "relation_cos_matrix": [[round(x, 4) for x in row] for row in cos_matrix.tolist()],
            "top_constraint_by_relation": top_constraints,
            "dcf_variety": round(float(np.std([np.linalg.norm(v) for v in dcf_centered])), 4),
        }

    # ---- 2c. 汇总 ----
    all_inter_cos = [v["mean_inter_relation_cos"] for v in relation_separation.values()]

    # 期望: kind_of → 类别维最高, used_for → 功能维最高
    expected_patterns = {
        "kind_of": ["fruit", "animal", "vehicle", "tool"],  # 类别
        "used_for": ["food", "tool"],  # 功能/用途
        "found_in": ["plant", "furniture"],  # 场景
        "made_of": ["plant", "tool"],  # 材料
        "related_to": ["food", "plant"],  # 关联
    }

    # 检查kind_of是否正确指向对象的类别维度
    kind_of_correct = 0
    kind_of_total = 0
    for obj_name, obj_data in all_object_results.items():
        if "kind_of" in obj_data:
            dcf = np.array(obj_data["kind_of"]["dcf_raw"])
            obj_cat = None
            for cat, objs in CATEGORIES.items():
                if obj_name in objs:
                    obj_cat = cat
                    break
            if obj_cat and obj_cat in FAMILY_WORDS:
                kind_of_total += 1
                cat_idx = list(FAMILY_WORDS.keys()).index(obj_cat)
                if np.argmax(dcf) == cat_idx:
                    kind_of_correct += 1

    results["per_object"] = relation_separation
    results["summary"] = {
        "mean_inter_relation_cos": round(float(np.mean(all_inter_cos)), 4) if all_inter_cos else 0,
        "n_objects": len(relation_separation),
        "kind_of_correct_category": f"{kind_of_correct}/{kind_of_total}",
        "interpretation": "Low inter-relation cos means DCF is relation-dependent (supports constraint theory)" if float(np.mean(all_inter_cos)) < 0.5 else "High inter-relation cos suggests DCF is object-fixed",
        "theory_prediction_confirmed": float(np.mean(all_inter_cos)) < 0.5,
    }

    plog(f"  Exp2 Summary: mean_inter_relation_cos={results['summary']['mean_inter_relation_cos']:.4f}, "
         f"kind_of_correct={results['summary']['kind_of_correct_category']}, "
         f"constraint_theory={'confirmed' if results['summary']['theory_prediction_confirmed'] else 'not confirmed'}")

    return results


# ==================== Exp3: DCF跨模型对齐 ====================
def exp3_dcf_cross_model_alignment(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    验证分布约束不变性: 不同模型的DCF结构是否相似?
    
    核心问题: 语义约束是否跨模型稳定?
    预测: 如果语义 = 分布约束, 则不同模型对同一对象应产生相似的DCF
    
    方法:
    1. 在kind_of模板下计算所有对象的DCF
    2. 保存DCF矩阵(供后续跨模型比较)
    3. 分析DCF维度的重要性排序是否跨模型一致
    """
    plog("=== Exp3: DCF跨模型对齐检查(单模型部分) ===")
    info = get_model_info(model, model_name)

    n_obj = 6 if round_num == 1 else 8
    cat_list = ["fruit", "animal", "vehicle", "tool", "furniture", "clothing"]

    # ---- 3a. 收集所有对象在kind_of下的DCF ----
    all_dcf = {}
    all_entropy = {}

    for cat in cat_list:
        objs = obj_dict.get(cat, [])[:n_obj]
        cat_dcfs = []

        for obj in objs:
            prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
            logits = get_final_logits(model, tokenizer, prompt, device)
            dcf_vec, _ = compute_dcf(logits, tokenizer)
            cat_dcfs.append(dcf_vec)
            all_entropy[obj] = round(logit_entropy(logits), 4)

        all_dcf[cat] = {
            "mean_dcf": [round(x, 4) for x in np.mean(cat_dcfs, axis=0).tolist()],
            "std_dcf": [round(x, 4) for x in np.std(cat_dcfs, axis=0).tolist()],
            "n_objects": len(cat_dcfs),
        }

    # ---- 3b. DCF维度重要性排序 ----
    # 哪些DCF维度(=类别族)区分对象最好?
    all_dcf_matrix = []
    all_labels = []
    for cat in cat_list:
        objs = obj_dict.get(cat, [])[:n_obj]
        for obj in objs:
            prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
            logits = get_final_logits(model, tokenizer, prompt, device)
            dcf_vec, _ = compute_dcf(logits, tokenizer)
            all_dcf_matrix.append(dcf_vec)
            all_labels.append(cat)

    dcf_matrix = np.array(all_dcf_matrix)

    # 每个维度的方差(= 区分力)
    dim_variances = np.var(dcf_matrix, axis=0)
    dim_order = np.argsort(dim_variances)[::-1]
    dim_names = list(FAMILY_WORDS.keys())

    dim_importance = []
    for idx in dim_order:
        dim_importance.append({
            "dimension": dim_names[idx],
            "variance": round(float(dim_variances[idx]), 4),
        })

    # ---- 3c. 类别DCF中心间的相关结构 ----
    cat_center_matrix = np.array([all_dcf[cat]["mean_dcf"] for cat in cat_list])
    # 类别间DCF中心的相关矩阵
    from scipy.stats import pearsonr
    cat_corr = np.zeros((len(cat_list), len(cat_list)))
    for i in range(len(cat_list)):
        for j in range(len(cat_list)):
            r, _ = pearsonr(cat_center_matrix[i], cat_center_matrix[j])
            cat_corr[i, j] = round(float(r), 4)

    results = {
        "category_dcf_profiles": all_dcf,
        "dim_importance_ranking": dim_importance,
        "category_correlation_matrix": {
            "labels": cat_list,
            "matrix": [[round(x, 4) for x in row] for row in cat_corr.tolist()],
        },
        "entropy_stats": {
            "mean": round(float(np.mean(list(all_entropy.values()))), 4),
            "std": round(float(np.std(list(all_entropy.values()))), 4),
        },
        "summary": {
            "top_discriminating_dims": [d["dimension"] for d in dim_importance[:3]],
            "n_categories": len(cat_list),
            "n_objects_per_cat": n_obj,
        },
    }

    plog(f"  Exp3 Summary: Top discriminating dims = {results['summary']['top_discriminating_dims']}")
    plog(f"  Entropy: mean={results['entropy_stats']['mean']:.4f}, std={results['entropy_stats']['std']:.4f}")

    return results


# ==================== 主函数 ====================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}")
        return

    obj_dict = ROUNDS[round_num]

    plog(f"Phase 470: Distribution Constraint Circuit — {model_name}, Round {round_num}")
    plog(f"Objects per category: {len(list(obj_dict.values())[0])}")
    plog(f"Theory: Meaning = stable control over future probability distribution")

    # ---- 1. 加载模型 ----
    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    t_load = time.time() - t0
    plog(f"Model loaded in {t_load:.0f}s")

    info = get_model_info(model, model_name)
    plog(f"Model: {info.model_class}, n_layers={info.n_layers}, d_model={info.d_model}")

    # ---- 2. 运行实验 ----
    all_results = {
        "phase": 470,
        "model": model_name,
        "round": round_num,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "theory": "Distribution-Control Interface Theory",
        "core_formula": "Meaning(x) = ΔP(future | x)",
        "model_info": {
            "class": info.model_class,
            "n_layers": info.n_layers,
            "d_model": info.d_model,
        },
    }

    # Exp1: DCF构造与聚类验证
    t1 = time.time()
    all_results["exp1_dcf_construction"] = exp1_dcf_construction(
        model, tokenizer, model_name, device, obj_dict, round_num)
    plog(f"Exp1 done in {time.time()-t1:.0f}s")

    # Exp2: 关系槽位分离
    t2 = time.time()
    all_results["exp2_relation_slot"] = exp2_relation_slot_separation(
        model, tokenizer, model_name, device, obj_dict, round_num)
    plog(f"Exp2 done in {time.time()-t2:.0f}s")

    # Exp3: DCF跨模型对齐(单模型部分)
    t3 = time.time()
    all_results["exp3_dcf_alignment"] = exp3_dcf_cross_model_alignment(
        model, tokenizer, model_name, device, obj_dict, round_num)
    plog(f"Exp3 done in {time.time()-t3:.0f}s")

    # ---- 3. 保存结果 ----
    os.makedirs("results/glm5", exist_ok=True)
    out_path = f"results/glm5/phase470_{model_name}_r{round_num}.json"

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(x) for x in obj]
        if isinstance(obj, bool):
            return obj
        if isinstance(obj, (int, float, str)):
            return obj
        return str(obj)

    all_results = convert(all_results)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    plog(f"Results saved to {out_path}")

    # ---- 4. 释放模型 ----
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()

    total_time = time.time() - t0
    plog(f"Phase 470 {model_name} Round {round_num} complete in {total_time:.0f}s")


if __name__ == "__main__":
    main()
