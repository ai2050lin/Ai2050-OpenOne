"""
Phase 126: 参数回路拓扑分析 — 从"方向空间"转向"回路空间"
===========================================================

Phase 125结论: 方向主义崩塌, 语义不在任何单个方向中
本阶段: 转向回路分析 — attention heads + MLP neurons的协同结构

5个实验:
- Exp 1: Attention Head功能分化 (不同语义类别激活哪些head组合)
- Exp 2: Head协同激活矩阵 (哪些head形成稳定协同回路)
- Exp 3: 条件轨迹分叉 (上下文如何改变head路由)
- Exp 4: 属性绑定回路 (共享属性是否共享回路)
- Exp 5: 回路消融 vs 方向消融 (证明回路比方向更本质)

注意: head_mask对Qwen3无效, 改用o_proj pre_hook消融
"""

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import time
import gc
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from tests.glm5.model_utils import (
    load_model, get_layers, get_model_info, get_W_U, release_model, MODEL_CONFIGS
)


# ============================================================
# 工具函数
# ============================================================

def get_device_for_input(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_n_heads(model) -> int:
    if hasattr(model, 'config') and hasattr(model.config, 'num_attention_heads'):
        return model.config.num_attention_heads
    return 32


def get_head_dim(model) -> int:
    """获取attention head维度 — 注意: 不是d_model/n_heads! GQA改变了这个关系"""
    if hasattr(model, 'config') and hasattr(model.config, 'head_dim'):
        return model.config.head_dim
    layers = get_layers(model)
    # 从o_proj推断: o_proj.weight shape = [d_model, n_heads * head_dim]
    d_model = model.config.hidden_size
    o_proj_dim = layers[0].self_attn.o_proj.weight.shape[1]
    n_heads = get_n_heads(model)
    return o_proj_dim // n_heads


def make_oproj_pre_hook(head_indices_to_zero, n_heads, head_dim):
    """Hook到o_proj的输入, 将指定head的输出置零"""
    def hook(module, input):
        attn_output = input[0]  # [batch, seq, n_heads * head_dim]
        batch, seq, d = attn_output.shape
        attn_reshaped = attn_output.view(batch, seq, n_heads, head_dim)
        for hi in head_indices_to_zero:
            attn_reshaped[:, :, hi, :] = 0.0
        return input
    return hook


def make_mlp_zero_hook():
    """Hook将MLP输出置零"""
    def hook(module, input, output):
        if isinstance(output, tuple):
            return (torch.zeros_like(output[0]),) + output[1:]
        return torch.zeros_like(output)
    return hook


def compute_kl(p_base_logits, p_abl_logits):
    """计算KL散度, 处理nan/inf"""
    # 检查nan/inf
    if np.isnan(p_base_logits).any() or np.isnan(p_abl_logits).any():
        return -1.0  # 标记为无效
    if np.isinf(p_base_logits).any() or np.isinf(p_abl_logits).any():
        return -1.0
    
    p1 = np.exp(p_base_logits - np.max(p_base_logits)); p1 /= p1.sum()
    p2 = np.exp(p_abl_logits - np.max(p_abl_logits)); p2 /= p2.sum()
    kl = float(np.sum(p1 * (np.log(p1 + 1e-10) - np.log(p2 + 1e-10))))
    
    if np.isnan(kl) or np.isinf(kl):
        return -1.0
    return kl


def safe_cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ============================================================
# 数据
# ============================================================

SEMANTIC_CATEGORIES = {
    "fruit": [
        "apple", "banana", "orange", "grape", "pear", "peach", "mango",
        "cherry", "lemon", "lime", "kiwi", "plum", "melon", "fig",
        "papaya", "guava", "apricot", "blueberry", "strawberry", "raspberry",
        "coconut", "pineapple", "pomegranate", "watermelon", "tangerine",
        "blackberry", "cranberry", "date", "dragonfruit", "persimmon"
    ],
    "animal": [
        "cat", "dog", "horse", "cow", "pig", "sheep", "goat", "chicken",
        "duck", "rabbit", "mouse", "rat", "fox", "wolf", "bear", "lion",
        "tiger", "elephant", "giraffe", "zebra", "monkey", "ape", "deer",
        "elk", "moose", "whale", "dolphin", "shark", "eagle", "hawk"
    ],
    "color": [
        "red", "blue", "green", "yellow", "orange", "purple", "pink",
        "brown", "black", "white", "gray", "silver", "gold", "violet",
        "indigo", "crimson", "scarlet", "teal", "cyan", "magenta",
        "turquoise", "maroon", "navy", "olive", "coral", "beige",
        "ivory", "lavender", "salmon", "khaki"
    ],
    "verb_motion": [
        "run", "walk", "jump", "swim", "fly", "climb", "crawl", "dive",
        "sprint", "stroll", "leap", "bounce", "slide", "roll", "spin",
        "dance", "skip", "dash", "hike", "march", "gallop", "wander",
        "drift", "glide", "soar", "plunge", "tumble", "stumble", "rush", "flee"
    ],
    "place": [
        "Paris", "London", "Tokyo", "Beijing", "Berlin", "Rome", "Madrid",
        "Moscow", "Sydney", "Cairo", "Delhi", "Bangkok", "Seoul", "Vienna",
        "Lisbon", "Athens", "Dublin", "Oslo", "Helsinki", "Warsaw",
        "Prague", "Budapest", "Zurich", "Amsterdam", "Brussels",
        "Stockholm", "Copenhagen", "Vancouver", "Melbourne", "Mumbai"
    ],
    "abstract": [
        "freedom", "justice", "truth", "beauty", "wisdom", "courage",
        "honesty", "loyalty", "patience", "kindness", "equality", "peace",
        "hope", "faith", "love", "trust", "respect", "honor", "dignity",
        "integrity", "compassion", "mercy", "humility", "gratitude",
        "tolerance", "creativity", "curiosity", "ambition", "resilience",
        "perseverance"
    ],
}

CONTEXT_PAIRS = [
    ("I eat the", "I work at", "apple"),
    ("The ripe", "The tech company", "apple"),
    ("The barking", "The hot", "dog"),
    ("I pet the", "I ate the", "dog"),
    ("The wooden", "The baseball", "bat"),
    ("The flying", "The river", "bank"),
    ("The sharp", "The musical", "note"),
    ("The boiling", "The cold spring", "water"),
    ("The chess", "The royal", "queen"),
    ("The dining", "The mathematical", "table"),
    ("I drink from the", "I sit at the", "glass"),
    ("The garden", "The wedding", "rose"),
    ("The ocean", "The computer", "mouse"),
    ("The animal", "The shooting", "star"),
    ("The kitchen", "The police", "chief"),
]

ATTRIBUTE_BINDING = {
    "color_fruit": [
        ("red apple", "green apple", "red car", "green car"),
        ("yellow banana", "green banana", "yellow sun", "green leaf"),
        ("red cherry", "dark cherry", "red fire", "dark night"),
        ("purple grape", "green grape", "purple sky", "green grass"),
        ("orange orange", "green orange", "orange sunset", "green forest"),
    ],
    "size_animal": [
        ("big elephant", "small elephant", "big house", "small house"),
        ("tiny mouse", "large mouse", "tiny ant", "large dog"),
        ("huge whale", "small whale", "huge mountain", "small hill"),
        ("big cat", "small cat", "big problem", "small issue"),
        ("tall giraffe", "short giraffe", "tall building", "short story"),
    ],
    "emotion_face": [
        ("happy face", "sad face", "happy news", "sad news"),
        ("angry voice", "calm voice", "angry storm", "calm sea"),
        ("warm smile", "cold smile", "warm fire", "cold ice"),
        ("bright eyes", "dark eyes", "bright sun", "dark room"),
        ("scared child", "brave child", "scared cat", "brave hero"),
    ],
}


# ============================================================
# Exp 1: Attention Head 功能分化
# ============================================================

def exp1_head_functional_differentiation(model, tokenizer, model_name, model_info):
    """不同语义类别激活哪些attention head组合?"""
    print("\n" + "="*60)
    print("Exp 1: Attention Head 功能分化")
    print("="*60)
    
    device = get_device_for_input(model)
    layers = get_layers(model)
    n_layers = len(layers)
    n_heads = get_n_heads(model)
    
    target_layers = sorted(set([0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]))
    print(f"  n_layers={n_layers}, n_heads={n_heads}, target_layers={target_layers}")
    
    # 收集head attention patterns
    # head_attn_by_cat[category][layer_idx][head_idx] = list of attention entropy values
    head_attn_by_cat = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for category, words in SEMANTIC_CATEGORIES.items():
        print(f"  Category: {category} ({len(words)} words)")
        for word in words:
            text = f"The {word}"
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=32)
            input_ids = inputs["input_ids"].to(device)
            attn_mask = inputs["attention_mask"].to(device)
            
            with torch.no_grad():
                try:
                    out = model(input_ids=input_ids, attention_mask=attn_mask, output_attentions=True)
                except Exception as e:
                    continue
            
            if out.attentions is None:
                continue
            
            for li in target_layers:
                if li < len(out.attentions):
                    attn = out.attentions[li]  # [1, n_heads, seq, seq]
                    seq_len = int(attn_mask.sum())
                    for hi in range(min(n_heads, attn.shape[1])):
                        # head hi对last real token的attention分布
                        head_attn = attn[0, hi, seq_len-1, :seq_len].cpu().float().numpy()
                        # entropy (低=选择性高)
                        entropy = -float(np.sum(head_attn * np.log(head_attn + 1e-10)))
                        # concentration (max weight)
                        concentration = float(np.max(head_attn))
                        head_attn_by_cat[category][li][hi].append({
                            "entropy": entropy,
                            "concentration": concentration,
                        })
    
    # 分析: 每个head的类别profile
    all_categories = list(SEMANTIC_CATEGORIES.keys())
    results = {
        "n_heads": n_heads,
        "target_layers": target_layers,
        "categories": all_categories,
    }
    
    for li in target_layers:
        # 构建profile矩阵 [n_heads, n_categories] (entropy维度)
        ent_profiles = np.zeros((n_heads, len(all_categories)))
        conc_profiles = np.zeros((n_heads, len(all_categories)))
        
        for ci, cat in enumerate(all_categories):
            for hi in range(n_heads):
                vals = head_attn_by_cat[cat][li][hi]
                if vals:
                    ent_profiles[hi, ci] = np.mean([v["entropy"] for v in vals])
                    conc_profiles[hi, ci] = np.mean([v["concentration"] for v in vals])
        
        # Head之间的相关矩阵 (基于entropy profile)
        if ent_profiles.std() > 1e-10:
            corr = np.corrcoef(ent_profiles)  # [n_heads, n_heads]
            
            # 高相关对 (r>0.8)
            high_corr_pairs = []
            for i in range(n_heads):
                for j in range(i+1, n_heads):
                    if abs(corr[i, j]) > 0.8:
                        high_corr_pairs.append((i, j, float(corr[i, j])))
            
            # 功能聚类 (相关>0.7的连通分量)
            abs_corr = np.abs(corr)
            np.fill_diagonal(abs_corr, 1.0)
            adj = (abs_corr > 0.7).astype(float)
            n_components, labels = connected_components(csr_matrix(adj))
            cluster_sizes = sorted([int(np.sum(labels == i)) for i in range(n_components)], reverse=True)
            
            # 每个head的"选择性指数": max类内熵 / 跨类均值
            selectivities = []
            for hi in range(n_heads):
                cat_avgs = ent_profiles[hi]
                cross_avg = cat_avgs.mean() + 1e-10
                max_sel = float(np.max(np.abs(cat_avgs - cross_avg)) / cross_avg)
                best_cat = all_categories[int(np.argmax(np.abs(cat_avgs - cross_avg)))]
                selectivities.append({
                    "head": hi,
                    "max_selectivity": max_sel,
                    "best_category": best_cat,
                })
            selectivities.sort(key=lambda x: x["max_selectivity"], reverse=True)
            
            results[f"L{li}"] = {
                "n_high_corr_pairs": len(high_corr_pairs),
                "n_clusters": n_components,
                "cluster_sizes_top5": cluster_sizes[:5],
                "mean_abs_corr": float(np.mean(np.abs(corr[np.triu_indices(n_heads, k=1)]))),
                "top5_selective_heads": selectivities[:5],
            }
            
            print(f"  L{li}: {len(high_corr_pairs)} high-corr pairs, "
                  f"{n_components} clusters, top sizes: {cluster_sizes[:5]}")
    
    return results


# ============================================================
# Exp 2: Head协同激活矩阵
# ============================================================

def exp2_head_coactivation(model, tokenizer, model_name, model_info):
    """对大量输入, 收集各head的attention pattern, 计算head间相关矩阵"""
    print("\n" + "="*60)
    print("Exp 2: Head协同激活矩阵")
    print("="*60)
    
    device = get_device_for_input(model)
    layers = get_layers(model)
    n_layers = len(layers)
    n_heads = get_n_heads(model)
    
    target_layers = sorted(set([0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]))
    
    # 合并所有输入
    all_words = []
    for cat, words in SEMANTIC_CATEGORIES.items():
        all_words.extend(words)
    
    diverse_prompts = [
        "The weather today is very", "She walked to the store and",
        "In mathematics, the function", "The city was built near",
        "He played the piano with great", "The experiment showed that",
        "After the rain, the garden", "The old man sat quietly and",
        "She opened the book and began to", "The river flows through the",
        "In the kitchen, the chef", "The children played happily in the",
        "The scientist studied the effects of", "The mountain was covered with",
        "The music filled the room with", "The teacher explained that",
        "They traveled across the vast", "The painting depicted a scene of",
        "The engine roared as the car", "The flower bloomed in the spring",
        "He solved the puzzle by thinking", "The ocean waves crashed against the",
        "The artist created a beautiful", "The recipe required fresh",
        "The documentary explored the history of", "The students discussed the topic of",
        "The forest was home to many", "The bridge connected the two",
        "The star shone brightly in the", "The computer processed the data and",
        "The doctor recommended regular", "The garden produced many delicious",
        "The athlete trained hard for the", "The novel told the story of",
        "The telescope revealed a distant", "The vaccine protected against the",
        "The museum displayed ancient", "The concert attracted a large",
        "The satellite orbited around the", "The poet wrote about the beauty of",
        "The earthquake caused significant", "The factory produced various",
        "The ceremony celebrated the", "The microscope revealed tiny",
        "The library contained thousands of", "The volcano erupted with tremendous",
        "The election determined the next", "The spacecraft traveled beyond the",
        "The glacier slowly retreated from the", "The festival featured traditional",
        "The coral reef supported diverse", "The telescope observed a new",
        "The algorithm sorted the data by", "The philosopher questioned the nature of",
        "The architect designed a modern", "The symphony ended with a powerful",
        "The detective investigated the mysterious", "The rocket launched into the dark",
        "The whale migrated across the deep", "The robot performed the complex",
        "The diamond sparkled in the bright", "The treaty established peace between the",
        "The microscope showed the cell", "The opera singer performed a beautiful",
        "The submarine dived deep into the", "The orchestra played a moving",
        "The inventor created a revolutionary", "The dancer moved with incredible",
        "The meteor crashed into the barren", "The programmer wrote efficient",
        "The expedition discovered an ancient", "The thunder echoed across the stormy",
        "The biologist examined the rare", "The judge ruled that the",
        "The composer wrote a haunting", "The pilot navigated through the thick",
        "The chemist synthesized a new", "The sculptor carved the marble into a",
        "The glacier carved the valley over", "The programmer debugged the complex",
        "The historian analyzed the primary", "The fisherman caught a large",
        "The nurse cared for the sick", "The engineer calculated the structural",
        "The astronaut floated in the zero", "The baker prepared a delicious",
        "The philosopher debated the ethical", "The musician tuned the instrument before",
        "The photographer captured the stunning", "The linguist studied the ancient",
        "The mathematician proved the elegant", "The ecologist monitored the endangered",
        "The mechanic repaired the broken", "The journalist reported on the breaking",
        "The surgeon performed the delicate", "The astronomer detected the faint",
        "The botanist identified the rare", "The economist predicted the market",
        "The psychologist analyzed the cognitive", "The geologist examined the rock",
    ]
    
    all_inputs = all_words + diverse_prompts
    n_inputs = len(all_inputs)
    print(f"  Total inputs: {n_inputs}")
    
    # 收集head attention features
    # head_features[layer][head] = [n_inputs, 4] (entropy, concentration, first_attn, self_attn)
    head_features = defaultdict(lambda: defaultdict(lambda: np.zeros((n_inputs, 4))))
    input_idx = 0
    
    for batch_start in range(0, n_inputs, 8):
        batch_end = min(batch_start + 8, n_inputs)
        batch_texts = all_inputs[batch_start:batch_end]
        
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, 
                          truncation=True, max_length=32)
        input_ids = inputs["input_ids"].to(device)
        attn_mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            try:
                out = model(input_ids=input_ids, attention_mask=attn_mask,
                           output_attentions=True)
            except Exception as e:
                print(f"    Batch {batch_start}-{batch_end} failed: {e}")
                input_idx += len(batch_texts)
                continue
        
        if out.attentions is None:
            input_idx += len(batch_texts)
            continue
        
        for li_idx, li in enumerate(target_layers):
            if li >= len(out.attentions):
                continue
            attn = out.attentions[li]  # [batch, n_heads, seq, seq]
            
            for b in range(attn.shape[0]):
                if input_idx + b >= n_inputs:
                    break
                seq_len = int(attn_mask[b].sum()) if b < attn_mask.shape[0] else attn.shape[2]
                for hi in range(min(n_heads, attn.shape[1])):
                    head_attn = attn[b, hi, min(seq_len-1, attn.shape[2]-1), :seq_len].cpu().float().numpy()
                    entropy = -float(np.sum(head_attn * np.log(head_attn + 1e-10)))
                    concentration = float(np.max(head_attn))
                    first_attn = float(head_attn[0]) if len(head_attn) > 0 else 0
                    self_attn = float(head_attn[-1]) if len(head_attn) > 0 else 0
                    head_features[li][hi][input_idx + b] = [entropy, concentration, first_attn, self_attn]
        
        input_idx += len(batch_texts)
    
    # 分析: head相关矩阵和聚类
    results = {"n_inputs": n_inputs, "n_heads": n_heads, "target_layers": target_layers}
    
    for li in target_layers:
        # 构建feature矩阵 [n_heads, n_inputs * 4]
        feat_list = []
        for hi in range(n_heads):
            feat_list.append(head_features[li][hi].flatten())
        feat_matrix = np.array(feat_list)
        
        # 相关矩阵
        if feat_matrix.std() > 1e-10:
            # 只用entropy维度 (column 0 per input)
            ent_matrix = np.zeros((n_heads, n_inputs))
            for hi in range(n_heads):
                for b in range(n_inputs):
                    ent_matrix[hi, b] = head_features[li][hi][b, 0]
            
            if ent_matrix.std() > 1e-10:
                corr = np.corrcoef(ent_matrix)
            else:
                corr = np.eye(n_heads)
        else:
            corr = np.eye(n_heads)
        
        # 统计
        triu_corr = corr[np.triu_indices(n_heads, k=1)]
        abs_corr = np.abs(corr)
        np.fill_diagonal(abs_corr, 1.0)
        
        # 聚类
        adj = (abs_corr > 0.7).astype(float)
        n_components, labels = connected_components(csr_matrix(adj))
        cluster_sizes = sorted([int(np.sum(labels == i)) for i in range(n_components)], reverse=True)
        
        results[f"L{li}"] = {
            "mean_abs_corr": float(np.mean(np.abs(triu_corr))),
            "max_corr": float(np.max(np.abs(triu_corr))),
            "frac_gt05": float(np.mean(np.abs(triu_corr) > 0.5)),
            "frac_gt08": float(np.mean(np.abs(triu_corr) > 0.8)),
            "n_clusters": n_components,
            "cluster_sizes_top5": cluster_sizes[:5],
        }
        
        print(f"  L{li}: mean|r|={results[f'L{li}']['mean_abs_corr']:.3f}, "
              f"frac>0.5={results[f'L{li}']['frac_gt05']:.3f}, "
              f"clusters={n_components}, top sizes={cluster_sizes[:3]}")
    
    return results


# ============================================================
# Exp 3: 条件轨迹分叉
# ============================================================

def exp3_conditional_branching(model, tokenizer, model_name, model_info):
    """上下文如何改变head的路由模式?"""
    print("\n" + "="*60)
    print("Exp 3: 条件轨迹分叉")
    print("="*60)
    
    device = get_device_for_input(model)
    layers = get_layers(model)
    n_layers = len(layers)
    n_heads = get_n_heads(model)
    
    target_layers = sorted(set([
        0, n_layers//6, n_layers//3, n_layers//2, 2*n_layers//3, 5*n_layers//6, n_layers-1
    ]))
    
    results = {"context_pairs": [], "target_layers": target_layers, "n_heads": n_heads}
    
    for ctx1, ctx2, target in CONTEXT_PAIRS:
        text1 = f"{ctx1} {target}"
        text2 = f"{ctx2} {target}"
        
        inputs1 = tokenizer(text1, return_tensors="pt", truncation=True, max_length=32)
        inputs2 = tokenizer(text2, return_tensors="pt", truncation=True, max_length=32)
        input_ids1 = inputs1["input_ids"].to(device)
        input_ids2 = inputs2["input_ids"].to(device)
        attn_mask1 = inputs1["attention_mask"].to(device)
        attn_mask2 = inputs2["attention_mask"].to(device)
        
        pair_result = {
            "context1": ctx1, "context2": ctx2, "target": target,
            "layer_divergence": {},
            "head_attn_divergence": {},
        }
        
        with torch.no_grad():
            out1 = model(input_ids=input_ids1, attention_mask=attn_mask1,
                        output_hidden_states=True, output_attentions=True)
            out2 = model(input_ids=input_ids2, attention_mask=attn_mask2,
                        output_hidden_states=True, output_attentions=True)
        
        # Hidden state轨迹分叉
        for li in target_layers:
            h1 = out1.hidden_states[li+1][0, -1].float().cpu().numpy()
            h2 = out2.hidden_states[li+1][0, -1].float().cpu().numpy()
            cos = safe_cos(h1, h2)
            pair_result["layer_divergence"][str(li)] = cos
        
        # Head attention pattern分叉
        if out1.attentions is not None and out2.attentions is not None:
            for li in target_layers:
                if li < len(out1.attentions) and li < len(out2.attentions):
                    attn1 = out1.attentions[li]
                    attn2 = out2.attentions[li]
                    
                    seq1 = int(attn_mask1.sum())
                    seq2 = int(attn_mask2.sum())
                    
                    head_divs = {}
                    for hi in range(min(n_heads, attn1.shape[1])):
                        p1 = attn1[0, hi, -1, :seq1].cpu().float().numpy()
                        p2 = attn2[0, hi, -1, :seq2].cpu().float().numpy()
                        
                        min_len = min(len(p1), len(p2))
                        p1t = p1[:min_len] + 1e-10; p1t /= p1t.sum()
                        p2t = p2[:min_len] + 1e-10; p2t /= p2t.sum()
                        
                        m = 0.5 * (p1t + p2t)
                        js = 0.5 * float(np.sum(p1t * np.log(p1t / m))) + \
                             0.5 * float(np.sum(p2t * np.log(p2t / m)))
                        head_divs[str(hi)] = float(js)
                    
                    pair_result["head_attn_divergence"][str(li)] = head_divs
        
        results["context_pairs"].append(pair_result)
        l0_cos = pair_result["layer_divergence"].get("0", "N/A")
        lmid_cos = pair_result["layer_divergence"].get(str(n_layers//2), "N/A")
        llast_cos = pair_result["layer_divergence"].get(str(n_layers-1), "N/A")
        if isinstance(l0_cos, float):
            print(f"  '{text1}' vs '{text2}': "
                  f"L0 cos={l0_cos:.3f}, Lmid={lmid_cos:.3f}, Llast={llast_cos:.3f}")
    
    # 聚合: 最上下文敏感的层和head
    layer_sensitivity = defaultdict(list)
    head_sensitivity = defaultdict(list)
    
    for pr in results["context_pairs"]:
        for li_str, cos_val in pr["layer_divergence"].items():
            layer_sensitivity[int(li_str)].append(1 - cos_val)
        for li_str, head_divs in pr["head_attn_divergence"].items():
            for hi_str, js_val in head_divs.items():
                head_sensitivity[(int(li_str), int(hi_str))].append(js_val)
    
    results["layer_context_sensitivity"] = {
        str(li): {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        for li, vals in layer_sensitivity.items()
    }
    
    head_sens_list = [
        (li_hi, float(np.mean(vals)), float(np.std(vals)))
        for li_hi, vals in head_sensitivity.items()
    ]
    head_sens_list.sort(key=lambda x: x[1], reverse=True)
    results["top_context_sensitive_heads"] = [
        {"layer": int(li_hi[0]), "head": int(li_hi[1]), "mean_js": mj, "std_js": sj}
        for li_hi, mj, sj in head_sens_list[:20]
    ]
    
    return results


# ============================================================
# Exp 4: 属性绑定回路
# ============================================================

def exp4_attribute_binding(model, tokenizer, model_name, model_info):
    """共享属性是否共享回路?"""
    print("\n" + "="*60)
    print("Exp 4: 属性绑定回路")
    print("="*60)
    
    device = get_device_for_input(model)
    layers = get_layers(model)
    n_layers = len(layers)
    n_heads = get_n_heads(model)
    
    target_layers = sorted(set([0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]))
    results = {"binding_groups": [], "target_layers": target_layers}
    
    for group_name, tuples in ATTRIBUTE_BINDING.items():
        group_result = {"group": group_name, "comparisons": []}
        
        for tuple_vals in tuples:
            phrases = list(tuple_vals)
            
            phrase_data = []
            for phrase in phrases:
                inputs = tokenizer(phrase, return_tensors="pt", truncation=True, max_length=32)
                input_ids = inputs["input_ids"].to(device)
                attn_mask = inputs["attention_mask"].to(device)
                
                with torch.no_grad():
                    out = model(input_ids=input_ids, attention_mask=attn_mask,
                               output_hidden_states=True, output_attentions=True)
                
                data = {"phrase": phrase, "hidden": {}, "attn": {}}
                for li in target_layers:
                    h = out.hidden_states[li+1][0, -1].float().cpu().numpy()
                    data["hidden"][str(li)] = h
                
                if out.attentions is not None:
                    for li in target_layers:
                        if li < len(out.attentions):
                            attn = out.attentions[li][0, :, -1, :].float().cpu().numpy()
                            data["attn"][str(li)] = attn
                
                phrase_data.append(data)
            
            comp = {"phrases": phrases}
            
            for li in target_layers:
                h0 = phrase_data[0]["hidden"][str(li)]
                h1 = phrase_data[1]["hidden"][str(li)]
                h2 = phrase_data[2]["hidden"][str(li)]
                h3 = phrase_data[3]["hidden"][str(li)]
                
                comp[f"cos_L{li}"] = {
                    "same_obj_diff_attr": safe_cos(h0, h1),
                    "same_attr_diff_obj": safe_cos(h0, h2),
                    "diff_obj_diff_attr": safe_cos(h0, h3),
                }
            
            for li in target_layers:
                li_str = str(li)
                if li_str in phrase_data[0]["attn"] and li_str in phrase_data[1]["attn"]:
                    a0 = phrase_data[0]["attn"][li_str]
                    a1 = phrase_data[1]["attn"][li_str]
                    a2 = phrase_data[2]["attn"].get(li_str)
                    a3 = phrase_data[3]["attn"].get(li_str)
                    
                    head_js_01, head_js_02, head_js_03 = [], [], []
                    
                    for hi in range(min(n_heads, a0.shape[0])):
                        p0 = a0[hi] + 1e-10; p0 /= p0.sum()
                        p1 = a1[hi] + 1e-10; p1 /= p1.sum()
                        min_len = min(len(p0), len(p1))
                        p0t, p1t = p0[:min_len], p1[:min_len]
                        m = 0.5 * (p0t + p1t)
                        js01 = 0.5 * float(np.sum(p0t * np.log(p0t / m))) + \
                               0.5 * float(np.sum(p1t * np.log(p1t / m)))
                        head_js_01.append(js01)
                        
                        if a2 is not None:
                            p2 = a2[hi] + 1e-10; p2 /= p2.sum()
                            min_len2 = min(len(p0), len(p2))
                            p0t2, p2t = p0[:min_len2], p2[:min_len2]
                            m2 = 0.5 * (p0t2 + p2t)
                            js02 = 0.5 * float(np.sum(p0t2 * np.log(p0t2 / m2))) + \
                                   0.5 * float(np.sum(p2t * np.log(p2t / m2)))
                            head_js_02.append(js02)
                        
                        if a3 is not None:
                            p3 = a3[hi] + 1e-10; p3 /= p3.sum()
                            min_len3 = min(len(p0), len(p3))
                            p0t3, p3t = p0[:min_len3], p3[:min_len3]
                            m3 = 0.5 * (p0t3 + p3t)
                            js03 = 0.5 * float(np.sum(p0t3 * np.log(p0t3 / m3))) + \
                                   0.5 * float(np.sum(p3t * np.log(p3t / m3)))
                            head_js_03.append(js03)
                    
                    comp[f"attn_js_L{li}"] = {
                        "same_obj_diff_attr_mean": float(np.mean(head_js_01)),
                        "same_attr_diff_obj_mean": float(np.mean(head_js_02)) if head_js_02 else None,
                        "diff_obj_diff_attr_mean": float(np.mean(head_js_03)) if head_js_03 else None,
                    }
            
            group_result["comparisons"].append(comp)
        
        results["binding_groups"].append(group_result)
        
        # 聚合该group
        obj_sims = defaultdict(list)
        attr_sims = defaultdict(list)
        diff_sims = defaultdict(list)
        
        for comp in group_result["comparisons"]:
            for key, val in comp.items():
                if key.startswith("cos_L"):
                    li = key[2:]
                    if isinstance(val, dict):
                        obj_sims[li].append(val["same_obj_diff_attr"])
                        attr_sims[li].append(val["same_attr_diff_obj"])
                        diff_sims[li].append(val["diff_obj_diff_attr"])
        
        results[f"aggregated_{group_name}"] = {
            "obj_sim_by_layer": {li: float(np.mean(vals)) for li, vals in obj_sims.items()},
            "attr_sim_by_layer": {li: float(np.mean(vals)) for li, vals in attr_sims.items()},
            "diff_sim_by_layer": {li: float(np.mean(vals)) for li, vals in diff_sims.items()},
        }
        print(f"  Group '{group_name}': done")
    
    return results


# ============================================================
# Exp 5: 回路消融 vs 方向消融
# ============================================================

def exp5_circuit_vs_direction_ablation(model, tokenizer, model_name, model_info):
    """删除特定head组合(回路消融) vs 投影掉特定方向(方向消融)"""
    print("\n" + "="*60)
    print("Exp 5: 回路消融 vs 方向消融")
    print("="*60)
    
    device = get_device_for_input(model)
    layers = get_layers(model)
    n_layers = len(layers)
    n_heads = get_n_heads(model)
    head_dim = get_head_dim(model)
    d_model = model_info.d_model
    
    test_sentences = [
        "The apple is a type of fruit that grows on",
        "The dog is a loyal animal that can",
        "The color red is often associated with",
        "Paris is the capital city of",
        "Running is a form of exercise that improves",
        "Freedom is a fundamental right that every",
        "The cat sat quietly on the",
        "The ocean is a vast body of",
    ]
    
    target_layer = n_layers // 2
    
    # 基线预测
    print("  Getting baseline predictions...")
    baseline_logits = {}
    for sent in test_sentences:
        inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=32)
        input_ids = inputs["input_ids"].to(device)
        attn_mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask)
        baseline_logits[sent] = out.logits[0, -1].float().cpu().numpy()
    
    # === 方向消融 (PCA) ===
    print("  Direction ablation (PCA)...")
    
    all_hidden = []
    for sent in test_sentences:
        inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=32)
        input_ids = inputs["input_ids"].to(device)
        attn_mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
        h = out.hidden_states[target_layer+1][0, -1].float().cpu().numpy()
        all_hidden.append(h)
    
    all_hidden = np.array(all_hidden)
    centered = all_hidden - all_hidden.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    pca_directions = Vt[:50]
    
    W_U = get_W_U(model, model_name)
    
    direction_ablation_results = {}
    for k in [5, 10, 25, 50]:
        dirs = pca_directions[:k]
        proj = dirs.T @ dirs
        
        kl_divs = []
        for sent in test_sentences:
            inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=32)
            input_ids = inputs["input_ids"].to(device)
            attn_mask = inputs["attention_mask"].to(device)
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
            h = out.hidden_states[target_layer+1][0, -1].float().cpu().numpy()
            
            h_ablated = h - proj @ h
            logits_ablated = W_U @ h_ablated
            logits_orig = W_U @ h
            
            kl = compute_kl(logits_orig, logits_ablated)
            kl_divs.append(kl)
        
        valid_kls = [x for x in kl_divs if x >= 0]
        direction_ablation_results[str(k)] = {
            "mean_kl": float(np.mean(valid_kls)) if valid_kls else -1.0,
            "std_kl": float(np.std(valid_kls)) if valid_kls else 0.0,
        }
        print(f"    Direction top-{k} PCA dirs: mean_KL={np.mean(valid_kls) if valid_kls else 'N/A':.4f}")
    
    # === 回路消融 (Hook-based head zeroing) ===
    print("  Circuit ablation (hook-based head zeroing)...")
    
    circuit_ablation_results = {}
    
    for k in [1, 3, 5, 10]:
        kl_random = []
        kl_same_layer = []
        kl_cross_layer = []
        
        for trial in range(5):
            np.random.seed(trial * 42 + 7)
            
            # Random k heads across all layers
            all_head_ids = [(li, hi) for li in range(n_layers) for hi in range(n_heads)]
            random_heads = [all_head_ids[i] for i in np.random.choice(len(all_head_ids), min(k, len(all_head_ids)), replace=False)]
            
            # Same layer k heads
            random_layer = np.random.randint(0, n_layers)
            same_layer_heads = [(random_layer, hi) for hi in range(min(k, n_heads))]
            
            # Cross layer: 1 head per layer
            cross_layer_heads = [(li, np.random.randint(0, n_heads)) for li in range(min(k, n_layers))]
            
            for head_set_name, head_set in [
                ("random", random_heads),
                ("same_layer", same_layer_heads),
                ("cross_layer", cross_layer_heads),
            ]:
                # 按层分组
                heads_by_layer = defaultdict(list)
                for li, hi in head_set:
                    heads_by_layer[li].append(hi)
                
                # 注册hooks
                hooks = []
                for li, head_indices in heads_by_layer.items():
                    hooks.append(layers[li].self_attn.o_proj.register_forward_pre_hook(
                        make_oproj_pre_hook(head_indices, n_heads, head_dim)
                    ))
                
                kl_divs = []
                for sent in test_sentences:
                    inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=32)
                    input_ids = inputs["input_ids"].to(device)
                    attn_mask = inputs["attention_mask"].to(device)
                    with torch.no_grad():
                        out_abl = model(input_ids=input_ids, attention_mask=attn_mask)
                    abl_logits = out_abl.logits[0, -1].float().cpu().numpy()
                    kl = compute_kl(baseline_logits[sent], abl_logits)
                    kl_divs.append(kl)
                
                for h in hooks: h.remove()
                
                # 过滤无效值 (compute_kl返回-1表示nan/inf)
                valid_kls = [x for x in kl_divs if x >= 0]
                mean_kl = float(np.mean(valid_kls)) if valid_kls else -1.0
                if head_set_name == "random":
                    kl_random.append(mean_kl)
                elif head_set_name == "same_layer":
                    kl_same_layer.append(mean_kl)
                elif head_set_name == "cross_layer":
                    kl_cross_layer.append(mean_kl)
        
        circuit_ablation_results[str(k)] = {
            "random": {"mean_kl": float(np.mean(kl_random)) if kl_random else -1},
            "same_layer": {"mean_kl": float(np.mean(kl_same_layer)) if kl_same_layer else -1},
            "cross_layer": {"mean_kl": float(np.mean(kl_cross_layer)) if kl_cross_layer else -1},
        }
        print(f"    k={k}: random={np.mean(kl_random):.4f}, same_layer={np.mean(kl_same_layer):.4f}, cross_layer={np.mean(kl_cross_layer):.4f}")
    
    # === MLP消融 (对比) ===
    print("  MLP ablation (for comparison)...")
    mlp_ablation_results = {}
    
    for k in [1, 3, 5, 10]:
        kl_mlp = []
        for trial in range(5):
            np.random.seed(trial * 42 + 13)
            random_layers = np.random.choice(n_layers, min(k, n_layers), replace=False)
            
            hooks = []
            for li in random_layers:
                hooks.append(layers[int(li)].mlp.register_forward_hook(make_mlp_zero_hook()))
            
            kl_divs = []
            for sent in test_sentences:
                inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=32)
                input_ids = inputs["input_ids"].to(device)
                attn_mask = inputs["attention_mask"].to(device)
                with torch.no_grad():
                    out_abl = model(input_ids=input_ids, attention_mask=attn_mask)
                abl_logits = out_abl.logits[0, -1].float().cpu().numpy()
                kl = compute_kl(baseline_logits[sent], abl_logits)
                kl_divs.append(kl)
            
            for h in hooks: h.remove()
            valid_kls = [x for x in kl_divs if x >= 0]
            kl_mlp.append(float(np.mean(valid_kls)) if valid_kls else -1.0)
        
        mlp_ablation_results[str(k)] = {"mean_kl": float(np.mean(kl_mlp))}
        print(f"    MLP k={k} layers: mean_KL={np.mean(kl_mlp):.4f}")
    
    results = {
        "direction_ablation": direction_ablation_results,
        "circuit_ablation": circuit_ablation_results,
        "mlp_ablation": mlp_ablation_results,
        "test_sentences": test_sentences,
        "target_layer": target_layer,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "head_dim": head_dim,
    }
    
    return results


# ============================================================
# 主函数
# ============================================================

def run_all_experiments(model_name: str):
    print(f"\n{'#'*60}")
    print(f"# Phase 126: 参数回路拓扑分析 — {model_name}")
    print(f"{'#'*60}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_heads = get_n_heads(model)
    head_dim = get_head_dim(model)
    
    print(f"Model: {model_info.model_class}, L={model_info.n_layers}, "
          f"d={model_info.d_model}, n_heads={n_heads}, head_dim={head_dim}")
    
    all_results = {"model_name": model_name, "model_info": {
        "class": model_info.model_class,
        "n_layers": model_info.n_layers,
        "d_model": model_info.d_model,
        "n_heads": n_heads,
        "head_dim": head_dim,
    }}
    
    for exp_name, exp_fn in [
        ("exp1", exp1_head_functional_differentiation),
        ("exp2", exp2_head_coactivation),
        ("exp3", exp3_conditional_branching),
        ("exp4", exp4_attribute_binding),
        ("exp5", exp5_circuit_vs_direction_ablation),
    ]:
        try:
            t0 = time.time()
            result = exp_fn(model, tokenizer, model_name, model_info)
            all_results[exp_name] = result
            print(f"  {exp_name} done in {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"  {exp_name} FAILED: {e}")
            import traceback; traceback.print_exc()
            all_results[f"{exp_name}_error"] = str(e)
    
    release_model(model)
    return all_results


def convert(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert(x) for x in obj]
    return obj


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    
    results = run_all_experiments(model_name)
    
    out_path = f"tests/glm5_temp/phase126_{model_name}_circuit_topology.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(convert(results), f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")
