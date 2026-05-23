"""
Phase 64: 语言编码架构图 — 从"方向"到"分布"的范式转换
======================================================

基于Phase 63的关键发现:
  - 跨轴子空间完全近正交 (最稳健发现, 需扩展)
  - 区分信息集中在前50维 (有拐点)
  - 2D子空间不稳定 (推翻低维子空间假设)
  - 差异成分解码出概念特征 (支持相对编码)

四方案:
  Part 1 (64a): 跨轴正交性扩展到12轴 — 建立语言编码整体架构图
  Part 2 (64b): 协方差结构稳定性 — 解决"子空间不稳定+W_U稳定"的核心矛盾
  Part 3 (64c): 共享成分(85%)层次分解 — 破解hidden state主体编码
  Part 4 (64d): 独立子空间维度估算 — 语言编码的容量分析

用法:
  python tests/glm5/phase64_encoding_architecture.py --model qwen3 --part 1
  python tests/glm5/phase64_encoding_architecture.py --model qwen3 --part all
"""

import sys, os, json, argparse, gc, time, warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from itertools import combinations

class NumpyEncoder(json.JSONEncoder):
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

# ========== 模型加载 ==========
def load_model_bf16(model_name: str):
    """BF16 + device_map=auto 加载模型"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from model_utils import MODEL_CONFIGS

    cfg = MODEL_CONFIGS[model_name]
    log_time(f"Loading {model_name} (bfloat16 + device_map=auto)...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 尝试flash_attention_2, 不支持则回退eager
    for attn_impl in ["flash_attention_2", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"],
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
                attn_implementation=attn_impl,
            )
            log_time(f"Loaded with attn_implementation={attn_impl}")
            break
        except Exception as e:
            log_time(f"flash_attention_2 failed ({e}), falling back to eager")
            if attn_impl == "eager":
                raise
    model.eval()

    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    if hasattr(model, 'hf_device_map'):
        dmap = model.hf_device_map
        gpu_count = sum(1 for v in dmap.values() if 'cuda' in str(v))
        cpu_count = sum(1 for v in dmap.values() if 'cpu' in str(v))
        log_time(f"{model_name} loaded: GPU={gpu_count} comps, CPU={cpu_count} comps, GPU mem={gpu_mem:.2f}GB")
    else:
        log_time(f"{model_name} loaded: device={device}, GPU mem={gpu_mem:.2f}GB")

    return model, tokenizer, device


import torch

# ========== 语义轴词集定义 ==========
SEMANTIC_AXES = {
    "temperature": {
        "v1": ["freezing", "cold", "cool", "warm", "hot", "scorching"],
        "v2": ["icy", "chilly", "lukewarm", "tepid", "boiling", "scalding"],
        "full": ["freezing", "icy", "cold", "chilly", "cool", "lukewarm",
                 "tepid", "warm", "hot", "scorching", "boiling", "scalding"],
    },
    "size": {
        "v1": ["tiny", "small", "medium", "large", "huge", "enormous"],
        "v2": ["microscopic", "little", "moderate", "big", "massive", "gigantic"],
        "full": ["tiny", "microscopic", "small", "little", "medium", "moderate",
                 "large", "big", "huge", "massive", "enormous", "gigantic"],
    },
    "speed": {
        "v1": ["crawling", "slow", "moderate", "fast", "rapid", "instant"],
        "v2": ["sluggish", "leisurely", "steady", "swift", "quick", "immediate"],
        "full": ["crawling", "sluggish", "slow", "leisurely", "moderate", "steady",
                 "fast", "swift", "rapid", "quick", "instant", "immediate"],
    },
    "emotion": {
        "v1": ["serene", "calm", "neutral", "excited", "passionate", "ecstatic"],
        "v2": ["tranquil", "relaxed", "indifferent", "thrilled", "fervent", "euphoric"],
        "full": ["serene", "tranquil", "calm", "relaxed", "neutral", "indifferent",
                 "excited", "thrilled", "passionate", "fervent", "ecstatic", "euphoric"],
    },
    # === 新增轴 ===
    "brightness": {
        "v1": ["dark", "dim", "soft", "bright", "glaring", "blinding"],
        "v2": ["pitch", "murky", "muted", "luminous", "dazzling", "radiant"],
        "full": ["dark", "pitch", "dim", "murky", "soft", "muted",
                 "bright", "luminous", "glaring", "dazzling", "blinding", "radiant"],
    },
    "weight": {
        "v1": ["featherlight", "light", "medium", "heavy", "massive", "crushing"],
        "v2": ["weightless", "slight", "moderate", "hefty", "ponderous", "overwhelming"],
        "full": ["featherlight", "weightless", "light", "slight", "medium", "moderate",
                 "heavy", "hefty", "massive", "ponderous", "crushing", "overwhelming"],
    },
    "age": {
        "v1": ["newborn", "young", "adolescent", "adult", "old", "ancient"],
        "v2": ["infant", "juvenile", "teenage", "mature", "elderly", "primeval"],
        "full": ["newborn", "infant", "young", "juvenile", "adolescent", "teenage",
                 "adult", "mature", "old", "elderly", "ancient", "primeval"],
    },
    "distance": {
        "v1": ["adjacent", "near", "moderate", "far", "distant", "remote"],
        "v2": ["touching", "close", "midway", "removed", "faraway", "inaccessible"],
        "full": ["adjacent", "touching", "near", "close", "moderate", "midway",
                 "far", "removed", "distant", "faraway", "remote", "inaccessible"],
    },
    "complexity": {
        "v1": ["trivial", "simple", "moderate", "complex", "intricate", "convoluted"],
        "v2": ["basic", "elementary", "intermediate", "sophisticated", "elaborate", "byzantine"],
        "full": ["trivial", "basic", "simple", "elementary", "moderate", "intermediate",
                 "complex", "sophisticated", "intricate", "elaborate", "convoluted", "byzantine"],
    },
    "certainty": {
        "v1": ["impossible", "unlikely", "possible", "probable", "certain", "definite"],
        "v2": ["inconceivable", "doubtful", "plausible", "likely", "sure", "absolute"],
        "full": ["impossible", "inconceivable", "unlikely", "doubtful", "possible", "plausible",
                 "probable", "likely", "certain", "sure", "definite", "absolute"],
    },
    "danger": {
        "v1": ["harmless", "safe", "moderate", "risky", "dangerous", "lethal"],
        "v2": ["benign", "secure", "cautious", "perilous", "hazardous", "deadly"],
        "full": ["harmless", "benign", "safe", "secure", "moderate", "cautious",
                 "risky", "perilous", "dangerous", "hazardous", "lethal", "deadly"],
    },
    "beauty": {
        "v1": ["hideous", "ugly", "plain", "attractive", "beautiful", "gorgeous"],
        "v2": ["repulsive", "unsightly", "ordinary", "lovely", "stunning", "exquisite"],
        "full": ["hideous", "repulsive", "ugly", "unsightly", "plain", "ordinary",
                 "attractive", "lovely", "beautiful", "stunning", "gorgeous", "exquisite"],
    },
}

# 概念类别(用于Part 3/4)
CONCEPT_CATEGORIES = {
    "fruit": ["apple", "banana", "orange", "strawberry", "grape", "peach", "cherry", "mango", "pear", "lemon"],
    "animal": ["dog", "cat", "horse", "eagle", "dolphin", "snake", "bear", "rabbit", "lion", "whale"],
    "vehicle": ["car", "bicycle", "airplane", "train", "boat", "motorcycle", "bus", "helicopter", "truck", "scooter"],
    "tool": ["hammer", "wrench", "screwdriver", "pliers", "saw", "drill", "chisel", "level", "clamp", "ruler"],
    "clothing": ["shirt", "pants", "dress", "jacket", "shoes", "hat", "gloves", "socks", "coat", "scarf"],
}

# 更高层概念类别
HIGH_LEVEL_CATEGORIES = {
    "living": ["fruit", "animal"],
    "artifact": ["vehicle", "tool", "clothing"],
}

# 上下文模板(用于Part 2/3)
CONTEXT_TEMPLATES = [
    "The {word} is very interesting.",
    "I like this {word}.",
    "She mentioned the {word}.",
    "We found a {word} nearby.",
    "The {word} was remarkable.",
    "He described the {word} carefully.",
    "That {word} caught my attention.",
    "Everyone noticed the {word}.",
    "The {word} appeared suddenly.",
    "I remember the {word} clearly.",
    "What about the {word}?",
    "The {word} changed everything.",
    "She picked up the {word}.",
    "A {word} can be useful.",
    "The {word} made a sound.",
    "We observed the {word} closely.",
    "The {word} was unusual.",
    "He brought the {word} inside.",
    "The {word} is important.",
    "Look at the {word}.",
]


def get_hidden_states_for_words(model, tokenizer, device, words, layer_idx, 
                                  template="The word is {word}."):
    """获取一批词在指定层的hidden states"""
    from model_utils import get_layers
    layers = get_layers(model)
    layer = layers[layer_idx]
    
    captured = {}
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured['h'] = output[0].detach().float().cpu()
        else:
            captured['h'] = output.detach().float().cpu()
    
    handle = layer.register_forward_hook(hook_fn)
    
    hidden_states = {}
    for word in words:
        text = template.format(word=word)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        captured.clear()
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        
        if 'h' in captured:
            # 取最后一个token的hidden state
            h = captured['h'][0, -1].numpy()
            hidden_states[word] = h
    
    handle.remove()
    return hidden_states


def get_hidden_states_multi_context(model, tokenizer, device, word, layer_idx,
                                      templates, n_contexts=None):
    """获取同一个词在多个上下文中的hidden states"""
    from model_utils import get_layers
    layers = get_layers(model)
    layer = layers[layer_idx]
    
    captured = {}
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured['h'] = output[0].detach().float().cpu()
        else:
            captured['h'] = output.detach().float().cpu()
    
    handle = layer.register_forward_hook(hook_fn)
    
    if n_contexts is not None:
        templates = templates[:n_contexts]
    
    states = []
    for tmpl in templates:
        text = tmpl.format(word=word)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        captured.clear()
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        
        if 'h' in captured:
            h = captured['h'][0, -1].numpy()
            states.append(h)
    
    handle.remove()
    return np.array(states)


def extract_subspace(activations, n_dims=2):
    """从激活矩阵中提取n_dims维子空间基"""
    # activations: [n_samples, d_model]
    centered = activations - activations.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    return Vt[:n_dims]  # [n_dims, d_model]


def principal_angles(S1, S2):
    """计算两个子空间之间的主角(度数)"""
    from scipy.linalg import subspace_angles
    angles_rad = subspace_angles(S1.T, S2.T)
    return [float(a) for a in np.degrees(angles_rad)]


def subspace_k90(activations, threshold=0.90):
    """计算达到90%方差需要的维度数"""
    centered = activations - activations.mean(axis=0, keepdims=True)
    _, S, _ = np.linalg.svd(centered, full_matrices=False)
    var_explained = np.cumsum(S**2) / np.sum(S**2)
    k90 = int(np.searchsorted(var_explained, threshold)) + 1
    return k90, var_explained.tolist()


def decode_wu(direction, W_U, tokenizer, top_k=15):
    """W_U解码: 给定方向, 找最对齐的token"""
    scores = W_U @ direction
    top_ids = np.argsort(scores)[-top_k:][::-1]
    results = []
    for tid in top_ids:
        tok = tokenizer.decode([tid]).strip()
        results.append({"token": tok, "score": float(scores[tid])})
    return results


# ================================================================
# Part 1: 跨轴正交性扩展到12轴 — 语言编码整体架构图
# ================================================================
def run_part1(model_name):
    """扩展跨轴正交性到12个语义轴, 建立12×12正交性热力图"""
    global model_name_global
    model_name_global = model_name
    
    log_time(f"=== Part 1: 12轴跨轴正交性 — {model_name} ===")
    
    model, tokenizer, device = load_model_bf16(model_name)
    from model_utils import get_model_info, get_layers, get_W_U, release_model
    info = get_model_info(model, model_name)
    W_U = get_W_U(model, model_name)
    
    # 采样层
    n_layers = info.n_layers
    sample_layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    log_time(f"Model: {info.model_class}, L={n_layers}, d={info.d_model}, layers={sample_layers}")
    
    axis_names = list(SEMANTIC_AXES.keys())
    n_axes = len(axis_names)
    
    out = {
        "phase": "64a",
        "model": model_name,
        "model_class": info.model_class,
        "n_layers": n_layers,
        "d_model": info.d_model,
        "n_axes": n_axes,
        "axis_names": axis_names,
        "sample_layers": sample_layers,
        "results_by_layer": {},
    }
    
    for layer_idx in sample_layers:
        log_time(f"  Layer {layer_idx}...")
        layer_result = {"orthogonality_matrix": {}, "axis_k90": {}, "axis_pc_decode": {}}
        
        # 为每个轴提取子空间
        subspaces = {}
        k90s = {}
        pc_decodes = {}
        
        for axis_name in axis_names:
            words = SEMANTIC_AXES[axis_name]["full"]
            hs = get_hidden_states_for_words(model, tokenizer, device, words, layer_idx)
            
            if len(hs) < 2:
                log_time(f"    {axis_name}: insufficient hidden states ({len(hs)})")
                continue
            
            act_matrix = np.array(list(hs.values()))
            
            # k90 (有效维度)
            k90, var_explained = subspace_k90(act_matrix)
            k90s[axis_name] = {"k90": k90, "var_explained": var_explained[:20]}
            
            # 5D子空间 (用于正交性检验)
            n_sub = min(5, act_matrix.shape[0] - 1)
            subspace = extract_subspace(act_matrix, n_dims=n_sub)
            subspaces[axis_name] = subspace
            
            # PC1/PC2 W_U解码
            pc1 = extract_subspace(act_matrix, n_dims=1)[0]
            pc2_dir = extract_subspace(act_matrix, n_dims=2)[1]
            pc_decodes[axis_name] = {
                "pc1_top5": decode_wu(pc1, W_U, tokenizer, top_k=5),
                "pc2_top5": decode_wu(pc2_dir, W_U, tokenizer, top_k=5),
            }
        
        layer_result["axis_k90"] = k90s
        layer_result["axis_pc_decode"] = pc_decodes
        
        # 计算所有轴对的正交性
        for i, j in combinations(range(n_axes), 2):
            ai, aj = axis_names[i], axis_names[j]
            if ai not in subspaces or aj not in subspaces:
                continue
            
            angles = principal_angles(subspaces[ai], subspaces[aj])
            key = f"{ai}_vs_{aj}"
            layer_result["orthogonality_matrix"][key] = {
                "angles": angles,
                "min_angle": min(angles),
                "mean_angle": float(np.mean(angles)),
            }
        
        out["results_by_layer"][str(layer_idx)] = layer_result
        log_time(f"    {n_axes} axes, {len(layer_result['orthogonality_matrix'])} pairs computed")
    
    # 保存
    path = RESULT_DIR / f"phase64_part1_{model_name_global}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    log_time(f"Part 1 saved: {path}")
    
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    return out


# ================================================================
# Part 2: 协方差结构稳定性 — 解决核心矛盾
# ================================================================
def run_part2(model_name):
    """验证语义信号是否以"分布形状"而非"点位置"存在"""
    global model_name_global
    model_name_global = model_name
    
    log_time(f"=== Part 2: 协方差结构稳定性 — {model_name} ===")
    
    model, tokenizer, device = load_model_bf16(model_name)
    from model_utils import get_model_info, get_layers, get_W_U, release_model
    info = get_model_info(model, model_name)
    
    n_layers = info.n_layers
    target_layer = n_layers - 1  # 最后一层
    log_time(f"Model: {info.model_class}, target_layer={target_layer}, d={info.d_model}")
    
    # 选择测试词: 同轴词(温度) vs 不同轴词
    test_words = {
        "temperature_hot": ["hot", "scorching", "boiling", "burning", "fiery"],
        "temperature_cold": ["cold", "freezing", "icy", "frozen", "chilly"],
        "size_big": ["huge", "enormous", "massive", "gigantic", "vast"],
        "size_small": ["tiny", "small", "little", "minute", "miniature"],
        "animal": ["dog", "cat", "horse", "eagle", "dolphin"],
        "fruit": ["apple", "banana", "orange", "grape", "cherry"],
    }
    
    n_contexts = 20  # 每个词用20个不同上下文
    templates = CONTEXT_TEMPLATES[:n_contexts]
    
    out = {
        "phase": "64b",
        "model": model_name,
        "model_class": info.model_class,
        "target_layer": target_layer,
        "n_contexts": n_contexts,
        "results": {},
    }
    
    # 对每个词组, 收集多上下文hidden states
    all_stats = {}
    for group_name, words in test_words.items():
        log_time(f"  Processing {group_name}...")
        group_stats = {}
        
        for word in words:
            states = get_hidden_states_multi_context(
                model, tokenizer, device, word, target_layer, templates, n_contexts
            )
            
            if len(states) < 5:
                log_time(f"    {word}: only {len(states)} states, skipping")
                continue
            
            # 计算统计量
            mean_vec = np.mean(states, axis=0)
            cov_mat = np.cov(states, rowvar=False)  # [d, d]
            
            # 协方差矩阵的特征值分解(只取top-20)
            eigvals = np.linalg.eigvalsh(cov_mat)
            eigvals = np.sort(eigvals)[::-1][:20]
            
            group_stats[word] = {
                "mean_norm": float(np.linalg.norm(mean_vec)),
                "mean_std": float(np.std(states, axis=0).mean()),
                "cov_trace": float(np.trace(cov_mat)),
                "cov_frobenius": float(np.linalg.norm(cov_mat, 'fro')),
                "cov_top20_eigvals": eigvals.tolist(),
                "n_states": len(states),
            }
        
        all_stats[group_name] = group_stats
    
    # 关键检验1: 协方差矩阵的稳定性
    # 把contexts分成两半, 计算两个半的协方差相似度
    log_time("  Computing covariance stability (split-half)...")
    cov_stability = {}
    
    for group_name, words in test_words.items():
        group_results = []
        for word in words:
            states = get_hidden_states_multi_context(
                model, tokenizer, device, word, target_layer, templates, n_contexts
            )
            if len(states) < 10:
                continue
            
            # 分成两半
            mid = len(states) // 2
            s1, s2 = states[:mid], states[mid:]
            cov1 = np.cov(s1, rowvar=False)
            cov2 = np.cov(s2, rowvar=False)
            
            # Frobenius内积相似度
            cov1_flat = cov1.flatten()
            cov2_flat = cov2.flatten()
            cos_sim = float(np.dot(cov1_flat, cov2_flat) / 
                           (np.linalg.norm(cov1_flat) * np.linalg.norm(cov2_flat) + 1e-10))
            
            # 特征向量对齐度(top-5)
            eigvals1, eigvecs1 = np.linalg.eigh(cov1)
            eigvals2, eigvecs2 = np.linalg.eigh(cov2)
            # 取最大的5个特征向量
            top5_v1 = eigvecs1[:, -5:][:, ::-1]  # [d, 5]
            top5_v2 = eigvecs2[:, -5:][:, ::-1]  # [d, 5]
            # 子空间主角
            try:
                angles = principal_angles(top5_v1.T, top5_v2.T)
                group_results.append({
                    "word": word,
                    "cov_cosine": cos_sim,
                    "subspace_angles": angles,
                    "subspace_mean_angle": float(np.mean(angles)),
                })
            except Exception as e:
                group_results.append({
                    "word": word,
                    "cov_cosine": cos_sim,
                    "subspace_angles_error": str(e),
                })
        
        cov_stability[group_name] = group_results
    
    out["cov_stability"] = cov_stability
    
    # 关键检验2: 同轴词的协方差结构是否比跨轴词更相似
    log_time("  Computing cross-group covariance similarity...")
    cross_group_sim = {}
    
    # 计算每组的平均协方差矩阵(用低维近似,节省内存)
    group_covs = {}
    for group_name, words in test_words.items():
        all_states = []
        for word in words:
            states = get_hidden_states_multi_context(
                model, tokenizer, device, word, target_layer, templates, n_contexts
            )
            all_states.append(states)
        
        if all_states:
            combined = np.vstack(all_states)
            # 低秩近似: 只存top-50特征向量
            centered = combined - combined.mean(axis=0, keepdims=True)
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            group_covs[group_name] = {
                "top50_vecs": Vt[:50],  # [50, d]
                "top50_vals": (S[:50]**2 / len(combined)).tolist(),
                "total_var": float(np.sum(S**2) / len(combined)),
                "n_samples": len(combined),
            }
    
    # 计算组间协方差相似度
    group_names = list(group_covs.keys())
    for i, j in combinations(range(len(group_names)), 2):
        gi, gj = group_names[i], group_names[j]
        if gi not in group_covs or gj not in group_covs:
            continue
        
        # 子空间主角(50D vs 50D)
        v1 = group_covs[gi]["top50_vecs"]
        v2 = group_covs[gj]["top50_vecs"]
        
        # 取前10个主角(计算量考虑)
        angles = principal_angles(v1[:10], v2[:10])
        
        key = f"{gi}_vs_{gj}"
        cross_group_sim[key] = {
            "subspace_angles_10d": angles,
            "min_angle": min(angles) if angles else None,
            "mean_angle": float(np.mean(angles)) if angles else None,
            "var_ratio": group_covs[gi]["total_var"] / max(group_covs[gj]["total_var"], 1e-10),
            "is_same_axis": (gi.split("_")[0] == gj.split("_")[0] and 
                            gi.split("_")[0] in ["temperature", "size"]),
        }
    
    out["cross_group_cov_similarity"] = cross_group_sim
    out["group_covs_summary"] = {
        k: {"total_var": v["total_var"], "n_samples": v["n_samples"],
            "top5_vals_ratio": sum(v["top50_vals"][:5]) / max(sum(v["top50_vals"]), 1e-10)}
        for k, v in group_covs.items()
    }
    
    # 保存
    path = RESULT_DIR / f"phase64_part2_{model_name_global}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    log_time(f"Part 2 saved: {path}")
    
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    return out


# ================================================================
# Part 3: 共享成分(85%)层次分解
# ================================================================
def run_part3(model_name):
    """分析hidden state中85%共享成分的层次结构"""
    global model_name_global
    model_name_global = model_name
    
    log_time(f"=== Part 3: 共享成分层次分解 — {model_name} ===")
    
    model, tokenizer, device = load_model_bf16(model_name)
    from model_utils import get_model_info, get_layers, get_W_U, release_model
    info = get_model_info(model, model_name)
    W_U = get_W_U(model, model_name)
    
    n_layers = info.n_layers
    target_layer = n_layers - 1
    log_time(f"Model: {info.model_class}, target_layer={target_layer}, d={info.d_model}")
    
    template = "The word is {word}."
    
    out = {
        "phase": "64c",
        "model": model_name,
        "model_class": info.model_class,
        "target_layer": target_layer,
        "results": {},
    }
    
    # Step 1: 收集所有概念词的hidden states
    log_time("  Step 1: Collecting hidden states for all concept words...")
    all_hiddens = {}  # {category: {word: hidden_state}}
    for cat_name, words in CONCEPT_CATEGORIES.items():
        cat_hiddens = {}
        for word in words:
            hs = get_hidden_states_for_words(model, tokenizer, device, [word], target_layer, template)
            if word in hs:
                cat_hiddens[word] = hs[word]
        all_hiddens[cat_name] = cat_hiddens
        log_time(f"    {cat_name}: {len(cat_hiddens)} words")
    
    # Step 2: 计算每类的质心(共享成分)
    log_time("  Step 2: Computing category centroids...")
    centroids = {}
    for cat_name, word_hiddens in all_hiddens.items():
        if len(word_hiddens) >= 3:
            mat = np.array(list(word_hiddens.values()))
            centroids[cat_name] = np.mean(mat, axis=0)
    
    # Step 3: 计算高层质心
    log_time("  Step 3: Computing higher-level centroids...")
    high_centroids = {}
    for high_cat, sub_cats in HIGH_LEVEL_CATEGORIES.items():
        sub_centroids = [centroids[sc] for sc in sub_cats if sc in centroids]
        if sub_centroids:
            high_centroids[high_cat] = np.mean(sub_centroids, axis=0)
    
    # 全局质心
    all_centroids_list = [v for v in centroids.values()]
    global_centroid = np.mean(all_centroids_list, axis=0)
    
    # Step 4: 层次分解 + W_U解码
    log_time("  Step 4: Hierarchical decomposition + W_U decoding...")
    decomposition = {}
    
    for cat_name, word_hiddens in all_hiddens.items():
        cat_result = {}
        
        # 类别共享成分 (vs 全局质心)
        cat_shared = centroids[cat_name] - global_centroid
        cat_shared_decode = decode_wu(cat_shared, W_U, tokenizer, top_k=15)
        
        # 类别特有成分 (vs 高层质心)
        high_cat = None
        for hc, subs in HIGH_LEVEL_CATEGORIES.items():
            if cat_name in subs:
                high_cat = hc
                break
        
        if high_cat and high_cat in high_centroids:
            cat_specific = centroids[cat_name] - high_centroids[high_cat]
            cat_specific_decode = decode_wu(cat_specific, W_U, tokenizer, top_k=15)
        else:
            cat_specific = None
            cat_specific_decode = []
        
        cat_result["shared_vs_global"] = cat_shared_decode[:10]
        cat_result["specific_vs_highlevel"] = cat_specific_decode[:10] if cat_specific is not None else []
        cat_result["shared_norm"] = float(np.linalg.norm(cat_shared))
        cat_result["specific_norm"] = float(np.linalg.norm(cat_specific)) if cat_specific is not None else 0
        
        # 个体差异成分
        delta_results = {}
        for word, h in word_hiddens.items():
            delta = h - centroids[cat_name]
            delta_decode = decode_wu(delta, W_U, tokenizer, top_k=10)
            delta_results[word] = {
                "decode": delta_decode[:5],
                "delta_norm": float(np.linalg.norm(delta)),
                "delta_ratio": float(np.linalg.norm(delta) / (np.linalg.norm(h) + 1e-10)),
            }
        
        cat_result["individual_deltas"] = delta_results
        decomposition[cat_name] = cat_result
    
    out["results"] = decomposition
    
    # Step 5: 共享成分的跨子集稳定性
    log_time("  Step 5: Testing shared component stability across subsets...")
    stability_results = {}
    
    for cat_name, words in CONCEPT_CATEGORIES.items():
        if len(words) < 5:
            continue
        
        # 用不同子集计算质心
        n = len(words)
        subsets = [
            words[:n//2],  # 前半
            words[n//2:],  # 后半
            words[::2],    # 奇数位
            words[1::2],   # 偶数位
        ]
        
        subset_centroids = []
        for i, subset in enumerate(subsets):
            hs = get_hidden_states_for_words(model, tokenizer, device, subset, target_layer, template)
            if len(hs) >= 2:
                mat = np.array(list(hs.values()))
                subset_centroids.append(np.mean(mat, axis=0))
        
        # 计算子集质心之间的cosine similarity
        cos_sims = []
        for i, j in combinations(range(len(subset_centroids)), 2):
            c1, c2 = subset_centroids[i], subset_centroids[j]
            cos = float(np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-10))
            cos_sims.append(cos)
        
        stability_results[cat_name] = {
            "n_subsets": len(subset_centroids),
            "pairwise_cos_sim": cos_sims,
            "mean_cos_sim": float(np.mean(cos_sims)) if cos_sims else 0,
            "min_cos_sim": float(np.min(cos_sims)) if cos_sims else 0,
        }
    
    out["shared_stability"] = stability_results
    
    # Step 6: 共享成分的维度分析
    log_time("  Step 6: Shared component dimensionality...")
    dim_analysis = {}
    for cat_name, word_hiddens in all_hiddens.items():
        if len(word_hiddens) < 3:
            continue
        mat = np.array(list(word_hiddens.values()))
        k90, var = subspace_k90(mat)
        dim_analysis[cat_name] = {
            "k90": k90,
            "var_explained_top10": var[:10],
            "n_words": len(word_hiddens),
        }
    
    out["dimensionality"] = dim_analysis
    
    # 保存
    path = RESULT_DIR / f"phase64_part3_{model_name_global}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    log_time(f"Part 3 saved: {path}")
    
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    return out


# ================================================================
# Part 4: 独立子空间维度估算 — 语言编码的容量分析
# ================================================================
def run_part4(model_name):
    """估算独立语义轴的总维度占用, 分析hidden state的编码容量"""
    global model_name_global
    model_name_global = model_name
    
    log_time(f"=== Part 4: 独立子空间维度估算 — {model_name} ===")
    
    model, tokenizer, device = load_model_bf16(model_name)
    from model_utils import get_model_info, get_layers, get_W_U, release_model
    info = get_model_info(model, model_name)
    
    n_layers = info.n_layers
    d_model = info.d_model
    sample_layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    log_time(f"Model: {info.model_class}, L={n_layers}, d={d_model}, layers={sample_layers}")
    
    template = "The word is {word}."
    
    out = {
        "phase": "64d",
        "model": model_name,
        "model_class": info.model_class,
        "n_layers": n_layers,
        "d_model": d_model,
        "results_by_layer": {},
    }
    
    axis_names = list(SEMANTIC_AXES.keys())
    
    for layer_idx in sample_layers:
        log_time(f"  Layer {layer_idx}...")
        layer_result = {
            "axis_k90": {},
            "axis_k95": {},
            "axis_k99": {},
            "total_dim_used": {},
        }
        
        # 对每个轴, 计算不同阈值下的有效维度
        axis_subspaces = {}
        for axis_name in axis_names:
            words = SEMANTIC_AXES[axis_name]["full"]
            hs = get_hidden_states_for_words(model, tokenizer, device, words, layer_idx, template)
            
            if len(hs) < 3:
                continue
            
            act_matrix = np.array(list(hs.values()))
            centered = act_matrix - act_matrix.mean(axis=0, keepdims=True)
            _, S, Vt = np.linalg.svd(centered, full_matrices=False)
            var_explained = np.cumsum(S**2) / np.sum(S**2)
            
            k90 = int(np.searchsorted(var_explained, 0.90)) + 1
            k95 = int(np.searchsorted(var_explained, 0.95)) + 1
            k99 = int(np.searchsorted(var_explained, 0.99)) + 1
            
            layer_result["axis_k90"][axis_name] = k90
            layer_result["axis_k95"][axis_name] = k95
            layer_result["axis_k99"][axis_name] = k99
            
            # 存储子空间用于正交性检验
            axis_subspaces[axis_name] = Vt[:k90]  # k90维子空间
        
        # 总维度占用(简单求和)
        total_k90 = sum(layer_result["axis_k90"].values())
        total_k95 = sum(layer_result["axis_k95"].values())
        total_k99 = sum(layer_result["axis_k99"].values())
        
        layer_result["total_dim_used"] = {
            "sum_k90": total_k90,
            "sum_k95": total_k95,
            "sum_k99": total_k99,
            "utilization_k90": float(total_k90 / d_model),
            "utilization_k95": float(total_k95 / d_model),
            "utilization_k99": float(total_k99 / d_model),
        }
        
        # 修正估算: 考虑轴间正交性
        # 如果轴A和轴B不正交, 它们共享维度, 实际占用比简单求和小
        log_time(f"    Computing corrected dimension (with overlap)...")
        
        # 合并所有轴的hidden states, 做全局SVD
        all_hiddens = []
        for axis_name in axis_names:
            words = SEMANTIC_AXES[axis_name]["full"]
            hs = get_hidden_states_for_words(model, tokenizer, device, words, layer_idx, template)
            all_hiddens.extend(list(hs.values()))
        
        if all_hiddens:
            all_mat = np.array(all_hiddens)
            centered = all_mat - all_mat.mean(axis=0, keepdims=True)
            _, S_all, _ = np.linalg.svd(centered, full_matrices=False)
            var_all = np.cumsum(S_all**2) / np.sum(S_all**2)
            
            global_k90 = int(np.searchsorted(var_all, 0.90)) + 1
            global_k95 = int(np.searchsorted(var_all, 0.95)) + 1
            global_k99 = int(np.searchsorted(var_all, 0.99)) + 1
            
            layer_result["global_dimensionality"] = {
                "global_k90": global_k90,
                "global_k95": global_k95,
                "global_k99": global_k99,
                "overlap_ratio_k90": float(total_k90 / max(global_k90, 1)),
                # overlap_ratio > 1 说明轴之间有维度重叠(不正交)
                # overlap_ratio ≈ 1 说明轴之间完全正交, 无重叠
            }
        
        # 检验: 哪些轴对有维度共享?
        log_time(f"    Computing pairwise axis overlap...")
        pairwise_overlap = {}
        for i, j in combinations(range(len(axis_names)), 2):
            ai, aj = axis_names[i], axis_names[j]
            if ai not in axis_subspaces or aj not in axis_subspaces:
                continue
            
            Si, Sj = axis_subspaces[ai], axis_subspaces[aj]
            # 投影矩阵: Si @ Si^T 投影到ai的子空间
            # 重叠度 = ||Si @ Si^T @ Sj^T||_F / ||Sj||_F
            proj = Si.T @ Si @ Sj.T  # [d, d] @ [d, k] = [d, k]
            overlap = float(np.linalg.norm(proj, 'fro') / (np.linalg.norm(Sj.T, 'fro') + 1e-10))
            pairwise_overlap[f"{ai}_vs_{aj}"] = min(overlap, 1.0)
        
        layer_result["pairwise_overlap"] = pairwise_overlap
        
        out["results_by_layer"][str(layer_idx)] = layer_result
        log_time(f"    k90: sum={total_k90}, global={global_k90 if all_hiddens else 'N/A'}, "
                 f"utilization={total_k90/d_model:.1%}")
    
    # 保存
    path = RESULT_DIR / f"phase64_part4_{model_name_global}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    log_time(f"Part 4 saved: {path}")
    
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    return out


# ================================================================
# 主入口
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="Phase 64: Language Encoding Architecture")
    parser.add_argument("--model", type=str, required=True,
                        choices=["qwen3", "deepseek7b", "glm4"])
    parser.add_argument("--part", type=str, default="all",
                        choices=["1", "2", "3", "4", "all"])
    args = parser.parse_args()
    
    log_time(f"Phase 64 start: model={args.model}, part={args.part}")
    
    if args.part == "all":
        for p in ["1", "2", "3", "4"]:
            log_time(f"--- Running Part {p} ---")
            try:
                if p == "1": run_part1(args.model)
                elif p == "2": run_part2(args.model)
                elif p == "3": run_part3(args.model)
                elif p == "4": run_part4(args.model)
            except Exception as e:
                log_time(f"Part {p} FAILED: {e}")
                import traceback; traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(3)
    else:
        if args.part == "1": run_part1(args.model)
        elif args.part == "2": run_part2(args.model)
        elif args.part == "3": run_part3(args.model)
        elif args.part == "4": run_part4(args.model)
    
    log_time(f"Phase 64 complete: model={args.model}, part={args.part}")


if __name__ == "__main__":
    main()
