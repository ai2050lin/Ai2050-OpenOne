"""
Phase 230: Feature Mechanics (特征力学)
=========================================

战略转向: 从"关系动力学"转向"特征发现"
核心问题: 语言是否存在稳定的基本特征单元?

4个实验:
  Exp1: 形容词组合性 ★★★★★ — 决定性实验
         h("red apple") - h("apple") ≈ h("red banana") - h("banana")?
         如果成立 → "红色"是稳定特征方向 → 语言有基本粒子

  Exp2: 操作编码 — 认知算子是否有稳定方向
         h("Translate: X") - h("X") 在不同X间是否稳定?

  Exp3: 特征可辨识性 — 不同特征方向之间是否正交/可分
         Δ_red vs Δ_blue 的cosine similarity (应低)
         Δ_red vs Δ_big 的cosine similarity (应低)

  Exp4: 特征注入因果验证 — 注入Δ_red是否会改变模型行为?
         如果注入Δ_red使模型更倾向生成红色相关词 → 特征是因果有效的

用法: python tests/glm5/phase230_feature_mechanics.py [qwen3|glm4|deepseek7b]
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc
import time
import json
import numpy as np
import torch
from collections import defaultdict
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr
from model_utils import (get_layers, get_model_info, release_model, MODEL_CONFIGS)


# ===== 数据定义 =====

# 形容词: 覆盖颜色、大小、温度、速度、情感、评价等维度
ADJECTIVES = [
    # 颜色 (6)
    "red", "blue", "green", "white", "black", "yellow",
    # 大小/形状 (6)
    "big", "small", "tall", "short", "wide", "narrow",
    # 温度/物理 (5)
    "hot", "cold", "warm", "heavy", "light",
    # 速度/时间 (4)
    "fast", "slow", "old", "new",
    # 情感/评价 (5)
    "happy", "sad", "angry", "beautiful", "ugly",
    # 状态 (4)
    "clean", "dirty", "safe", "dangerous",
]

# 名词: 覆盖动物、植物、物体、抽象概念等
NOUNS = [
    # 动物 (8)
    "cat", "dog", "bird", "fish", "horse", "bear", "lion", "snake",
    # 水果/食物 (6)
    "apple", "banana", "orange", "cake", "bread", "rice",
    # 自然 (5)
    "mountain", "river", "tree", "flower", "ocean",
    # 建筑/交通 (5)
    "house", "car", "bridge", "road", "building",
    # 人物/职业 (6)
    "doctor", "teacher", "child", "woman", "man", "friend",
]

# 认知操作
OPERATIONS = [
    ("translate", "Translate to French:"),
    ("summarize", "Summarize this:"),
    ("explain", "Explain why:"),
    ("compare", "Compare and contrast:"),
    ("list", "List the reasons:"),
    ("correct", "Correct the errors:"),
    ("rewrite", "Rewrite this:"),
    ("question", "Answer the question:"),
    ("negate", "State the opposite of:"),
    ("justify", "Justify the claim:"),
]

# 操作用基础句子
OP_SENTENCES = [
    "The cat sat on the mat and looked out the window.",
    "Scientists discovered a new element in the laboratory.",
    "The river flows through the valley to the sea.",
    "She finished reading the book before dinner.",
    "The children played happily in the garden.",
    "A strong wind blew across the open field.",
    "The teacher explained the lesson to the students.",
    "He walked slowly along the dark corridor.",
    "The company launched a new product this year.",
    "Rain fell steadily throughout the long night.",
    "The artist painted a beautiful landscape scene.",
    "They built a small cabin near the lake.",
    "The musician played a soft melody on piano.",
    "We watched the sunset from the hilltop.",
    "The old man told stories about his youth.",
    "Birds sang in the trees every morning.",
    "The chef prepared a delicious three course meal.",
    "She wrote a letter to her old friend.",
    "The train arrived late at the station.",
    "Snow covered the mountains during winter.",
]


# ===== 模型加载 =====

def load_model_bf16(model_name):
    """BF16 + device_map="auto" 加载模型"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    cfg = MODEL_CONFIGS[model_name]
    print(f"[load] Loading {model_name} (bfloat16 + device_map=auto)...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
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
    print(f"[load] {model_name}: device={device}, GPU={gpu_mem:.2f}GB")
    return model, tokenizer, device


# ===== 通用: 提取各层hidden states =====

def extract_hidden_states_batch(model, tokenizer, device, texts, n_layers,
                                 position="last", max_length=64):
    """
    批量提取每层hidden states
    
    Args:
        texts: 字符串列表
        position: "last"=取最后一个token, "mean"=取平均
    
    Returns:
        dict: {layer_idx: np.ndarray [n_texts, d_model]}
    """
    all_hidden = {l: [] for l in range(n_layers)}
    
    for i, text in enumerate(texts):
        toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = toks["input_ids"].to(device)
        attn_mask = toks["attention_mask"].to(device)
        
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask,
                       output_hidden_states=True)
        
        hs = out.hidden_states  # tuple of (1, seq_len, d_model)
        seq_len = hs[0].shape[1]
        
        # 找到最后一个非padding位置
        if position == "last":
            # 找attention_mask中最后一个1的位置
            mask_np = attn_mask[0].cpu().numpy()
            last_pos = np.where(mask_np == 1)[0][-1]
            for l in range(n_layers):
                h = hs[l][0, last_pos].float().cpu().numpy()
                all_hidden[l].append(h)
        elif position == "mean":
            for l in range(n_layers):
                # 只对非padding位置取平均
                mask = attn_mask[0].unsqueeze(-1).float()  # [seq_len, 1]
                h = (hs[l][0] * mask.to(hs[l].device)).sum(dim=0) / mask.sum()
                all_hidden[l].append(h.float().cpu().numpy())
        
        if (i + 1) % 50 == 0:
            print(f"    extracted {i+1}/{len(texts)} texts")
            if torch.cuda.is_available():
                print(f"    GPU mem: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    
    # 转为numpy
    for l in range(n_layers):
        all_hidden[l] = np.array(all_hidden[l])  # [n_texts, d_model]
    
    return all_hidden


# ===== Exp1: 形容词组合性 =====

def exp1_adjective_compositionality(model, tokenizer, device, n_layers, d_model):
    """
    ★★★★★ 决定性实验: 形容词方向是否跨名词稳定
    
    方法:
    1. 对每个形容词A和名词N, 提取 h("The A N.") 和 h("The N.")
    2. 计算 Δ_A,N = h("The A N.")[-1] - h("The N.")[-1]
    3. 对每个形容词A, 计算跨名词的 Δ_A 稳定性:
       - 同形容词不同名词的 Δ 之间的平均 cosine similarity
    4. 与基线比较: 不同形容词的 Δ 之间的 cosine similarity
    
    如果 same_adj_sim >> cross_adj_sim → 形容词是稳定特征方向
    """
    print("\n" + "="*60)
    print("Exp1: Adjective Compositionality (形容词组合性) ★★★★★")
    print("="*60)
    print(f"  Adjectives: {len(ADJECTIVES)}, Nouns: {len(NOUNS)}")
    print(f"  Total pairs: {len(ADJECTIVES) * len(NOUNS)}")
    
    t0 = time.time()
    
    # 构造句子
    adj_noun_texts = []
    noun_only_texts = []
    adj_indices = []
    noun_indices = []
    
    for ai, adj in enumerate(ADJECTIVES):
        for ni, noun in enumerate(NOUNS):
            adj_noun_texts.append(f"The {adj} {noun}.")
            noun_only_texts.append(f"The {noun}.")
            adj_indices.append(ai)
            noun_indices.append(ni)
    
    n_pairs = len(adj_noun_texts)
    print(f"  Extracting hidden states for {n_pairs} adj+noun + {n_pairs} noun-only texts...")
    
    # 提取hidden states
    hs_adj_noun = extract_hidden_states_batch(model, tokenizer, device, adj_noun_texts, n_layers)
    hs_noun_only = extract_hidden_states_batch(model, tokenizer, device, noun_only_texts, n_layers)
    
    # 采样层: 浅/中/深各取几层
    sample_layers = get_sample_layers(n_layers, n_samples=12)
    print(f"  Sample layers: {sample_layers}")
    
    results = {}
    
    for l in sample_layers:
        h_an = hs_adj_noun[l]  # [n_pairs, d_model]
        h_n = hs_noun_only[l]  # [n_pairs, d_model]
        
        # 计算 Δ vectors
        deltas = h_an - h_n  # [n_pairs, d_model]
        
        # 按形容词分组
        adj_deltas = defaultdict(list)
        for i, ai in enumerate(adj_indices):
            adj_deltas[ai].append(deltas[i])
        
        # 对每个形容词, 计算跨名词的稳定性
        same_adj_sims = []
        adj_mean_dirs = {}
        
        for ai in range(len(ADJECTIVES)):
            vecs = np.array(adj_deltas[ai])  # [n_nouns, d_model]
            
            # 计算均值方向
            mean_dir = vecs.mean(axis=0)
            mean_norm = np.linalg.norm(mean_dir)
            if mean_norm > 1e-10:
                mean_dir = mean_dir / mean_norm
            adj_mean_dirs[ai] = mean_dir
            
            # 计算每个 Δ 与均值的 cosine similarity
            for vi in range(len(vecs)):
                v = vecs[vi]
                v_norm = np.linalg.norm(v)
                if v_norm > 1e-10:
                    cos_sim = np.dot(v, mean_dir) / v_norm
                    same_adj_sims.append(float(cos_sim))
        
        # 跨形容词相似性 (基线)
        cross_adj_sims = []
        mean_dirs_list = [adj_mean_dirs[ai] for ai in range(len(ADJECTIVES))]
        
        for i in range(len(mean_dirs_list)):
            for j in range(i+1, len(mean_dirs_list)):
                cos = np.dot(mean_dirs_list[i], mean_dirs_list[j])
                cross_adj_sims.append(float(cos))
        
        # 按语义类别分组分析
        category_groups = {
            "color": [0, 1, 2, 3, 4, 5],         # red, blue, green, white, black, yellow
            "size": [6, 7, 8, 9, 10, 11],          # big, small, tall, short, wide, narrow
            "physical": [12, 13, 14, 15, 16],       # hot, cold, warm, heavy, light
            "temporal": [17, 18, 19, 20],            # fast, slow, old, new
            "emotional": [21, 22, 23, 24, 25],       # happy, sad, angry, beautiful, ugly
            "state": [26, 27, 28, 29],               # clean, dirty, safe, dangerous
        }
        
        # 类内相似性 vs 类间相似性
        within_cat_sims = {}
        for cat_name, indices in category_groups.items():
            cat_sims = []
            for i in indices:
                for j in indices:
                    if i < j:
                        cos = np.dot(adj_mean_dirs[i], adj_mean_dirs[j])
                        cat_sims.append(float(cos))
            within_cat_sims[cat_name] = cat_sims
        
        between_cat_sims = []
        cat_names = list(category_groups.keys())
        for ci in range(len(cat_names)):
            for cj in range(ci+1, len(cat_names)):
                for i in category_groups[cat_names[ci]]:
                    for j in category_groups[cat_names[cj]]:
                        cos = np.dot(adj_mean_dirs[i], adj_mean_dirs[j])
                        between_cat_sims.append(float(cos))
        
        # 逐形容词稳定性
        per_adj_stability = {}
        for ai, adj in enumerate(ADJECTIVES):
            vecs = np.array(adj_deltas[ai])
            mean_d = vecs.mean(axis=0)
            cos_sims = []
            for vi in range(len(vecs)):
                v = vecs[vi]
                v_norm = np.linalg.norm(v)
                m_norm = np.linalg.norm(mean_d)
                if v_norm > 1e-10 and m_norm > 1e-10:
                    cos_sims.append(float(np.dot(v, mean_d) / (v_norm * m_norm)))
            per_adj_stability[adj] = {
                "mean_cos": float(np.mean(cos_sims)) if cos_sims else 0.0,
                "std_cos": float(np.std(cos_sims)) if cos_sims else 0.0,
                "min_cos": float(np.min(cos_sims)) if cos_sims else 0.0,
                "delta_norm": float(np.mean(np.linalg.norm(vecs, axis=1))),
            }
        
        # 排名top5最稳定和最不稳定的形容词
        sorted_adj = sorted(per_adj_stability.items(), key=lambda x: x[1]["mean_cos"], reverse=True)
        
        results[f"L{l}"] = {
            "same_adj_mean_cos": float(np.mean(same_adj_sims)),
            "same_adj_std_cos": float(np.std(same_adj_sims)),
            "same_adj_median_cos": float(np.median(same_adj_sims)),
            "cross_adj_mean_cos": float(np.mean(cross_adj_sims)),
            "cross_adj_std_cos": float(np.std(cross_adj_sims)),
            "separation_ratio": float(np.mean(same_adj_sims) / max(abs(np.mean(cross_adj_sims)), 1e-6)),
            "within_cat_sims": {k: float(np.mean(v)) for k, v in within_cat_sims.items()},
            "between_cat_mean_cos": float(np.mean(between_cat_sims)),
            "top5_stable": [(a, d["mean_cos"]) for a, d in sorted_adj[:5]],
            "bottom5_stable": [(a, d["mean_cos"]) for a, d in sorted_adj[-5:]],
            "per_adj_stability": per_adj_stability,
        }
        
        print(f"  L{l:2d}: same_adj_cos={np.mean(same_adj_sims):.4f}±{np.std(same_adj_sims):.4f}, "
              f"cross_adj_cos={np.mean(cross_adj_sims):.4f}±{np.std(cross_adj_sims):.4f}, "
              f"separation={np.mean(same_adj_sims)/max(abs(np.mean(cross_adj_sims)),1e-6):.2f}x")
    
    elapsed = time.time() - t0
    print(f"\n  Exp1 完成 ({elapsed:.1f}s)")
    
    # 释放中间数据
    del hs_adj_noun, hs_noun_only
    gc.collect()
    torch.cuda.empty_cache()
    
    return results


# ===== Exp2: 操作编码 =====

def exp2_operation_encoding(model, tokenizer, device, n_layers, d_model):
    """
    认知操作是否有稳定的编码方向?
    
    方法:
    1. 对每个操作O和句子S, 提取 h("O: S") 和 h("S")
    2. 计算 Δ_O,S = h("O: S")[-1] - h("S")[-1]
    3. 对每个操作O, 计算跨句子的稳定性
    """
    print("\n" + "="*60)
    print("Exp2: Operation Encoding (操作编码)")
    print("="*60)
    print(f"  Operations: {len(OPERATIONS)}, Sentences: {len(OP_SENTENCES)}")
    
    t0 = time.time()
    
    # 构造句子
    op_texts = []
    base_texts = []
    op_indices = []
    
    for oi, (op_name, op_prefix) in enumerate(OPERATIONS):
        for si, sent in enumerate(OP_SENTENCES):
            op_texts.append(f"{op_prefix} {sent}")
            base_texts.append(sent)
            op_indices.append(oi)
    
    n_pairs = len(op_texts)
    print(f"  Extracting hidden states for {n_pairs} op+sent + {n_pairs} base texts...")
    
    hs_op = extract_hidden_states_batch(model, tokenizer, device, op_texts, n_layers)
    hs_base = extract_hidden_states_batch(model, tokenizer, device, base_texts, n_layers)
    
    sample_layers = get_sample_layers(n_layers, n_samples=12)
    
    results = {}
    
    for l in sample_layers:
        h_op = hs_op[l]
        h_base = hs_base[l]
        deltas = h_op - h_base
        
        # 按操作分组
        op_deltas = defaultdict(list)
        for i, oi in enumerate(op_indices):
            op_deltas[oi].append(deltas[i])
        
        # 同操作稳定性
        same_op_sims = []
        op_mean_dirs = {}
        
        for oi in range(len(OPERATIONS)):
            vecs = np.array(op_deltas[oi])
            mean_dir = vecs.mean(axis=0)
            mean_norm = np.linalg.norm(mean_dir)
            if mean_norm > 1e-10:
                mean_dir = mean_dir / mean_norm
            op_mean_dirs[oi] = mean_dir
            
            for vi in range(len(vecs)):
                v = vecs[vi]
                v_norm = np.linalg.norm(v)
                if v_norm > 1e-10:
                    cos_sim = np.dot(v, mean_dir) / v_norm
                    same_op_sims.append(float(cos_sim))
        
        # 跨操作相似性
        cross_op_sims = []
        for i in range(len(OPERATIONS)):
            for j in range(i+1, len(OPERATIONS)):
                cos = np.dot(op_mean_dirs[i], op_mean_dirs[j])
                cross_op_sims.append(float(cos))
        
        # 逐操作分析
        per_op = {}
        for oi, (op_name, _) in enumerate(OPERATIONS):
            vecs = np.array(op_deltas[oi])
            mean_d = vecs.mean(axis=0)
            cos_sims = []
            for vi in range(len(vecs)):
                v = vecs[vi]
                v_norm = np.linalg.norm(v)
                m_norm = np.linalg.norm(mean_d)
                if v_norm > 1e-10 and m_norm > 1e-10:
                    cos_sims.append(float(np.dot(v, mean_d) / (v_norm * m_norm)))
            per_op[op_name] = {
                "mean_cos": float(np.mean(cos_sims)) if cos_sims else 0.0,
                "std_cos": float(np.std(cos_sims)) if cos_sims else 0.0,
                "delta_norm": float(np.mean(np.linalg.norm(vecs, axis=1))),
            }
        
        results[f"L{l}"] = {
            "same_op_mean_cos": float(np.mean(same_op_sims)),
            "same_op_std_cos": float(np.std(same_op_sims)),
            "cross_op_mean_cos": float(np.mean(cross_op_sims)),
            "separation_ratio": float(np.mean(same_op_sims) / max(abs(np.mean(cross_op_sims)), 1e-6)),
            "per_op": per_op,
        }
        
        print(f"  L{l:2d}: same_op_cos={np.mean(same_op_sims):.4f}±{np.std(same_op_sims):.4f}, "
              f"cross_op_cos={np.mean(cross_op_sims):.4f}, "
              f"separation={np.mean(same_op_sims)/max(abs(np.mean(cross_op_sims)),1e-6):.2f}x")
    
    elapsed = time.time() - t0
    print(f"\n  Exp2 完成 ({elapsed:.1f}s)")
    
    del hs_op, hs_base
    gc.collect()
    torch.cuda.empty_cache()
    
    return results


# ===== Exp3: 特征可辨识性 =====

def exp3_feature_discriminability(model, tokenizer, device, n_layers, d_model,
                                   exp1_results):
    """
    不同特征方向之间是否可分?
    
    使用Exp1的形容词方向, 分析:
    1. 同类别形容词(如颜色)之间的cosine similarity
    2. 不同类别形容词之间的cosine similarity
    3. PCA可视化特征方向的空间分布
    """
    print("\n" + "="*60)
    print("Exp3: Feature Discriminability (特征可辨识性)")
    print("="*60)
    
    # 从Exp1结果中提取特征方向矩阵
    sample_layers = get_sample_layers(n_layers, n_samples=12)
    
    results = {}
    
    for l in sample_layers:
        layer_key = f"L{l}"
        if layer_key not in exp1_results:
            continue
        
        per_adj = exp1_results[layer_key]["per_adj_stability"]
        
        # 收集特征稳定性数据
        stability_by_cat = defaultdict(list)
        category_groups = {
            "color": ["red", "blue", "green", "white", "black", "yellow"],
            "size": ["big", "small", "tall", "short", "wide", "narrow"],
            "physical": ["hot", "cold", "warm", "heavy", "light"],
            "temporal": ["fast", "slow", "old", "new"],
            "emotional": ["happy", "sad", "angry", "beautiful", "ugly"],
            "state": ["clean", "dirty", "safe", "dangerous"],
        }
        
        for cat, adjs in category_groups.items():
            for adj in adjs:
                if adj in per_adj:
                    stability_by_cat[cat].append(per_adj[adj]["mean_cos"])
        
        cat_stability = {}
        for cat, vals in stability_by_cat.items():
            if vals:
                cat_stability[cat] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                }
        
        # 类别间区分度 (from within_cat_sims and between_cat_mean_cos)
        within = exp1_results[layer_key]["within_cat_sims"]
        between = exp1_results[layer_key]["between_cat_mean_cos"]
        
        # 每个类别的平均类内相似度
        avg_within = float(np.mean(list(within.values())))
        
        # 类内vs类间比值
        discriminability = avg_within / max(abs(between), 1e-6) if between != 0 else 0
        
        results[layer_key] = {
            "cat_stability": cat_stability,
            "avg_within_cat_cos": avg_within,
            "between_cat_cos": between,
            "discriminability_ratio": discriminability,
            "within_cat_details": within,
        }
        
        print(f"  L{l:2d}: avg_within={avg_within:.4f}, between={between:.4f}, "
              f"discriminability={discriminability:.2f}x")
    
    return results


# ===== Exp4: 特征注入因果验证 =====

def exp4_causal_intervention(model, tokenizer, device, n_layers, d_model,
                              exp1_results):
    """
    注入形容词方向是否会因果地改变模型输出?
    
    方法:
    1. 取中层(约束传播核心层)的形容词方向
    2. 在neutral名词(如"thing")的hidden state上注入方向
    3. 检查模型输出是否偏向该形容词的语义
    
    如果注入Δ_red使模型更倾向生成"red/apple/colored"等 → 特征因果有效
    """
    print("\n" + "="*60)
    print("Exp4: Causal Intervention (特征注入因果验证)")
    print("="*60)
    
    # 选择中层的特征方向
    mid_layer = n_layers // 2
    # 也在1/3和2/3层测试
    test_layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4]
    test_layers = [l for l in test_layers if l < n_layers]
    
    # 先提取特征方向 (用Exp1的方法但只取少量名词)
    adj_list = ["red", "big", "happy", "hot", "fast", "clean"]
    noun_list = ["cat", "dog", "bird", "house", "car", "tree", "book", "river"]
    
    # 提取特征方向
    print("  Extracting feature directions...")
    adj_texts = []
    noun_texts = []
    for adj in adj_list:
        for noun in noun_list:
            adj_texts.append(f"The {adj} {noun}.")
            noun_texts.append(f"The {noun}.")
    
    hs_adj = extract_hidden_states_batch(model, tokenizer, device, adj_texts, n_layers)
    hs_noun = extract_hidden_states_batch(model, tokenizer, device, noun_texts, n_layers)
    
    results = {}
    
    for target_layer in test_layers:
        print(f"\n  --- Target layer L{target_layer} ---")
        
        # 计算每个形容词的特征方向
        feature_dirs = {}
        for ai, adj in enumerate(adj_list):
            deltas = []
            for ni in range(len(noun_list)):
                idx = ai * len(noun_list) + ni
                delta = hs_adj[target_layer][idx] - hs_noun[target_layer][idx]
                deltas.append(delta)
            mean_delta = np.mean(deltas, axis=0)
            norm = np.linalg.norm(mean_delta)
            if norm > 1e-10:
                mean_delta = mean_delta / norm
            feature_dirs[adj] = mean_delta
        
        # 注入测试: 用hook在target_layer注入特征方向
        test_prompts = [
            "The thing on the table is",
            "I saw a thing that was",
            "She described the thing as",
            "The object in the room was",
            "He found a thing that looked",
        ]
        
        # 期望的关联词 (简化版: 只检查形容词本身是否概率上升)
        adj_associated = {
            "red": ["red", "color", "crimson", "scarlet", "pink"],
            "big": ["big", "large", "huge", "enormous", "giant"],
            "happy": ["happy", "joyful", "cheerful", "glad", "pleased"],
            "hot": ["hot", "warm", "burning", "fiery", "heated"],
            "fast": ["fast", "quick", "rapid", "swift", "speedy"],
            "clean": ["clean", "pure", "fresh", "tidy", "neat"],
        }
        
        intervention_results = {}
        
        for adj_name, direction in feature_dirs.items():
            adj_score_changes = []
            
            for prompt in test_prompts:
                # Baseline: 不注入
                toks = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
                input_ids = toks["input_ids"].to(device)
                attn_mask = toks["attention_mask"].to(device)
                
                with torch.no_grad():
                    out_base = model(input_ids=input_ids, attention_mask=attn_mask,
                                    output_hidden_states=True)
                
                # 获取target layer的hidden state, 注入方向
                base_logits = out_base.logits[0, -1].float().cpu().numpy()
                
                # 用hook注入
                layers = get_layers(model)
                captured = {}
                
                def make_inject_hook(layer_idx, feat_dir, beta=5.0):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            h = output[0].clone()
                            # 在最后一个token位置注入
                            dir_tensor = torch.tensor(feat_dir, dtype=h.dtype, device=h.device)
                            h[0, -1, :] += beta * dir_tensor
                            captured[f"L{layer_idx}"] = h
                            return (h,) + output[1:]
                        return output
                    return hook
                
                hooks = [layers[target_layer].register_forward_hook(
                    make_inject_hook(target_layer, direction, beta=5.0))]
                
                with torch.no_grad():
                    out_interv = model(input_ids=input_ids, attention_mask=attn_mask,
                                      output_hidden_states=True)
                
                for h in hooks:
                    h.remove()
                
                interv_logits = out_interv.logits[0, -1].float().cpu().numpy()
                
                # 计算关联词概率变化
                score_change = 0
                n_associated = 0
                for word in adj_associated[adj_name]:
                    word_ids = tokenizer.encode(word, add_special_tokens=False)
                    for wid in word_ids:
                        # softmax概率
                        base_prob = np.exp(base_logits[wid]) / np.sum(np.exp(base_logits))
                        interv_prob = np.exp(interv_logits[wid]) / np.sum(np.exp(interv_logits))
                        score_change += (interv_prob - base_prob)
                        n_associated += 1
                
                if n_associated > 0:
                    adj_score_changes.append(score_change / n_associated)
            
            intervention_results[adj_name] = {
                "mean_prob_change": float(np.mean(adj_score_changes)),
                "std_prob_change": float(np.std(adj_score_changes)),
                "direction_norm": float(np.linalg.norm(feature_dirs[adj_name])),
            }
            
            print(f"    {adj_name}: prob_change={np.mean(adj_score_changes):.6f}±{np.std(adj_score_changes):.6f}")
        
        results[f"L{target_layer}"] = intervention_results
    
    del hs_adj, hs_noun
    gc.collect()
    torch.cuda.empty_cache()
    
    return results


# ===== 辅助函数 =====

def get_sample_layers(n_layers, n_samples=10):
    if n_layers <= n_samples:
        return list(range(n_layers))
    step = n_layers // n_samples
    layers = list(range(0, n_layers, step)) + [n_layers - 1]
    return sorted(set(layers))


# ===== 主函数 =====

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    assert model_name in MODEL_CONFIGS, f"Unknown model: {model_name}"
    
    print("=" * 70)
    print(f"Phase 230: Feature Mechanics — {model_name}")
    print("=" * 70)
    
    t_start = time.time()
    
    # 加载模型
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    print(f"  Model: {info.model_class}, n_layers={n_layers}, d_model={d_model}")
    
    all_results = {
        "model": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "n_adjectives": len(ADJECTIVES),
        "n_nouns": len(NOUNS),
        "n_operations": len(OPERATIONS),
        "n_op_sentences": len(OP_SENTENCES),
    }
    
    # Exp1: 形容词组合性 (最重要的实验)
    exp1_results = exp1_adjective_compositionality(model, tokenizer, device, n_layers, d_model)
    all_results["exp1"] = exp1_results
    
    # Exp2: 操作编码
    exp2_results = exp2_operation_encoding(model, tokenizer, device, n_layers, d_model)
    all_results["exp2"] = exp2_results
    
    # Exp3: 特征可辨识性
    exp3_results = exp3_feature_discriminability(model, tokenizer, device, n_layers, d_model,
                                                  exp1_results)
    all_results["exp3"] = exp3_results
    
    # Exp4: 特征注入因果验证
    exp4_results = exp4_causal_intervention(model, tokenizer, device, n_layers, d_model,
                                             exp1_results)
    all_results["exp4"] = exp4_results
    
    # 释放模型
    release_model(model)
    
    # 保存结果
    out_path = f"tests/glm5_temp/phase230_{model_name}_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存到: {out_path}")
    
    # 打印汇总
    print("\n" + "=" * 70)
    print("PHASE 230 汇总")
    print("=" * 70)
    
    # Exp1 汇总
    print("\n--- Exp1: 形容词组合性 ---")
    for layer_key in sorted(exp1_results.keys(), key=lambda x: int(x[1:])):
        r = exp1_results[layer_key]
        print(f"  {layer_key}: same_cos={r['same_adj_mean_cos']:.4f}, "
              f"cross_cos={r['cross_adj_mean_cos']:.4f}, "
              f"separation={r['separation_ratio']:.2f}x, "
              f"top_stable={r['top5_stable'][0]}")
    
    # Exp2 汇总
    print("\n--- Exp2: 操作编码 ---")
    for layer_key in sorted(exp2_results.keys(), key=lambda x: int(x[1:])):
        r = exp2_results[layer_key]
        print(f"  {layer_key}: same_cos={r['same_op_mean_cos']:.4f}, "
              f"cross_cos={r['cross_op_mean_cos']:.4f}, "
              f"separation={r['separation_ratio']:.2f}x")
    
    # Exp3 汇总
    print("\n--- Exp3: 特征可辨识性 ---")
    for layer_key in sorted(exp3_results.keys(), key=lambda x: int(x[1:])):
        r = exp3_results[layer_key]
        print(f"  {layer_key}: within={r['avg_within_cat_cos']:.4f}, "
              f"between={r['between_cat_cos']:.4f}, "
              f"discriminability={r['discriminability_ratio']:.2f}x")
    
    # Exp4 汇总
    print("\n--- Exp4: 因果验证 ---")
    for layer_key in sorted(exp4_results.keys(), key=lambda x: int(x[1:])):
        r = exp4_results[layer_key]
        changes = [(adj, d["mean_prob_change"]) for adj, d in r.items()]
        pos_count = sum(1 for _, c in changes if c > 0)
        print(f"  {layer_key}: {pos_count}/{len(changes)} features cause positive prob change")
        for adj, change in sorted(changes, key=lambda x: x[1], reverse=True):
            print(f"    {adj}: {change:+.6f}")
    
    elapsed = time.time() - t_start
    print(f"\n总耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print("Phase 230 完成!")


if __name__ == "__main__":
    main()
