"""
Phase 127: 语义动力学轨迹分析 — 从"回路空间"到"轨迹空间"
===========================================================

Phase 126发现: MLP消融影响远大于attention消融, 但存在容量不匹配问题
用户关键修正: 
  1. MLP/Attn比较不公平 — MLP参数量远大于单个head
  2. 语义不是"空间对象", 而是"动力学过程"
  3. 需要分析完整轨迹 (h_0, h_1, ..., h_L), 而非单层hidden state

本阶段5个实验:
- Exp 1: 容量匹配消融 — 公平比较MLP vs Attention的信息贡献
- Exp 2: 全层轨迹聚类 — 不同语义是否对应不同轨迹族
- Exp 3: MLP神经元选择性 — 哪些neurons对哪些语义类别响应
- Exp 4: 轨迹分叉深度分析 — 歧义词在哪些层、哪些维度分叉
- Exp 5: 动力学邻接性 — "概念"在动力学空间中的邻近关系
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
    if hasattr(model, 'config') and hasattr(model.config, 'head_dim'):
        return model.config.head_dim
    layers = get_layers(model)
    sa = layers[0].self_attn
    if hasattr(sa, 'o_proj'):
        o_shape = sa.o_proj.weight.shape
        n_h = get_n_heads(model)
        return o_shape[1] // n_h
    return 128


def compute_kl(p_base_logits, p_abl_logits):
    """计算KL散度, 处理nan/inf"""
    if np.isnan(p_base_logits).any() or np.isnan(p_abl_logits).any():
        return -1.0
    if np.isinf(p_base_logits).any() or np.isinf(p_abl_logits).any():
        return -1.0
    p1 = np.exp(p_base_logits - np.max(p_base_logits)); p1 /= p1.sum()
    p2 = np.exp(p_abl_logits - np.max(p_abl_logits)); p2 /= p2.sum()
    kl = float(np.sum(p1 * (np.log(p1 + 1e-10) - np.log(p2 + 1e-10))))
    if np.isnan(kl) or np.isinf(kl):
        return -1.0
    return kl


def get_base_logits(model, tokenizer, device, text):
    """获取基础logits"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attn_mask = inputs["attention_mask"].to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn_mask)
    logits = out.logits[0, -1].float().cpu().numpy()
    return logits


def get_all_hidden_states(model, tokenizer, device, text):
    """获取所有层的hidden states"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attn_mask = inputs["attention_mask"].to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
    # hidden_states: (n_layers+1) x [1, seq_len, d_model]
    hs = out.hidden_states
    # 取每层最后一个token的hidden state
    result = []
    for h in hs:
        result.append(h[0, -1, :].float().cpu().numpy())
    return np.array(result)  # [n_layers+1, d_model]


def register_o_proj_hook(layers, layer_idx, head_idx, n_heads, head_dim):
    """注册o_proj pre_hook来零化特定head的输出"""
    hooks = []
    def make_pre_hook(hi, hd):
        def pre_hook(module, args):
            attn_output = args[0]  # [batch, seq, n_heads * head_dim]
            if attn_output.dim() == 3:
                attn_output[:, :, hi*hd:(hi+1)*hd] = 0.0
            return (attn_output,) + args[1:]
        return pre_hook
    
    h = layers[layer_idx].self_attn.o_proj.register_forward_pre_hook(
        make_pre_hook(head_idx, head_dim)
    )
    hooks.append(h)
    return hooks


def register_mlp_neuron_hook(layers, layer_idx, neuron_indices):
    """注册down_proj pre_hook来零化特定MLP neurons"""
    hooks = []
    def make_pre_hook(n_indices):
        def pre_hook(module, args):
            # args[0] = down_proj的输入 = [batch, seq, intermediate_size]
            mlp_output = args[0]
            if mlp_output.dim() == 3:
                for ni in n_indices:
                    if ni < mlp_output.shape[-1]:
                        mlp_output[:, :, ni] = 0.0
            return (mlp_output,) + args[1:]
        return pre_hook
    
    h = layers[layer_idx].mlp.down_proj.register_forward_pre_hook(
        make_pre_hook(neuron_indices)
    )
    hooks.append(h)
    return hooks


def register_mlp_layer_hook(layers, layer_idx):
    """注册hook来零化整个MLP层的输出"""
    hooks = []
    def make_hook():
        def hook(module, input, output):
            if isinstance(output, tuple):
                return (torch.zeros_like(output[0]),) + output[1:]
            return torch.zeros_like(output)
        return hook
    
    h = layers[layer_idx].mlp.register_forward_hook(make_hook())
    hooks.append(h)
    return hooks


# ============================================================
# 语义测试数据 — 大样本量
# ============================================================

# 6个语义类别, 每类15个名词
SEMANTIC_CATEGORIES = {
    "fruit": [
        "apple", "banana", "orange", "grape", "mango",
        "peach", "cherry", "lemon", "pear", "plum",
        "strawberry", "blueberry", "watermelon", "pineapple", "coconut"
    ],
    "animal": [
        "cat", "dog", "horse", "elephant", "lion",
        "tiger", "bear", "rabbit", "deer", "wolf",
        "eagle", "dolphin", "whale", "snake", "monkey"
    ],
    "place": [
        "Paris", "London", "Tokyo", "Beijing", "Sydney",
        "Cairo", "Rome", "Berlin", "Moscow", "Toronto",
        "Mumbai", "Seoul", "Bangkok", "Vienna", "Dubai"
    ],
    "abstract": [
        "freedom", "justice", "truth", "beauty", "love",
        "courage", "wisdom", "peace", "hope", "trust",
        "anger", "fear", "joy", "sorrow", "pride"
    ],
    "tool": [
        "hammer", "screwdriver", "wrench", "drill", "saw",
        "pliers", "chisel", "level", "clamp", "ruler",
        "knife", "scissors", "axe", "shovel", "ladder"
    ],
    "vehicle": [
        "car", "bus", "train", "airplane", "bicycle",
        "motorcycle", "truck", "boat", "helicopter", "subway",
        "taxi", "van", "scooter", "yacht", "tractor"
    ],
}

# 歧义词和上下文
AMBIGUOUS_WORDS = {
    "apple": ["I ate a fresh apple", "Apple released the new iPhone"],
    "bank": ["I sat by the river bank", "I deposited money at the bank"],
    "orange": ["The orange was sweet and juicy", "She wore an orange dress"],
    "plant": ["The plant needs more water", "They built a new power plant"],
    "bass": ["He caught a large bass", "The bass guitar was very loud"],
    "match": ["They won the tennis match", "Use a match to light the fire"],
    "rock": ["The rock was very heavy", "She listened to rock music"],
    "bat": ["The bat flew through the cave", "He swung the baseball bat"],
    "nail": ["She painted her fingernail", "I hit the nail with a hammer"],
    "draft": ["The cold draft came through the window", "He was drafted into the army"],
    "fair": ["The county fair was exciting", "The judge was fair and impartial"],
    "ring": ["She wore a gold ring", "The telephone began to ring"],
    "letter": ["He sent a letter by mail", "The letter A is the first letter"],
    "date": ["They went on a romantic date", "What is the date today"],
    "right": ["Turn right at the corner", "She had the right answer"],
}

# 属性绑定测试: 共享对象不同属性 vs 共享属性不同对象
BINDING_TEST = {
    "same_object_diff_attr": [
        ("red apple", "green apple"),
        ("big house", "small house"),
        ("hot water", "cold water"),
        ("fast car", "slow car"),
        ("old man", "young man"),
        ("sweet cake", "bitter coffee"),
        ("loud music", "quiet music"),
        ("bright light", "dim light"),
        ("sharp knife", "dull knife"),
        ("heavy stone", "light feather"),
    ],
    "same_attr_diff_object": [
        ("red apple", "red car"),
        ("big house", "big mountain"),
        ("hot water", "hot fire"),
        ("fast car", "fast runner"),
        ("old man", "old tree"),
        ("sweet cake", "sweet honey"),
        ("loud music", "loud thunder"),
        ("bright light", "bright star"),
        ("sharp knife", "sharp mind"),
        ("heavy stone", "heavy rain"),
    ],
    "all_diff": [
        ("red apple", "big house"),
        ("hot water", "fast car"),
        ("old man", "sweet cake"),
        ("loud music", "sharp knife"),
        ("bright light", "heavy stone"),
        ("green apple", "small house"),
        ("cold water", "slow car"),
        ("young man", "bitter coffee"),
        ("quiet music", "dull knife"),
        ("dim light", "light feather"),
    ],
}


# ============================================================
# Exp 1: 容量匹配消融 — 公平比较MLP vs Attention
# ============================================================

def exp1_capacity_matched_ablation(model, tokenizer, device, model_info):
    """
    核心问题: Phase 126的"MLP更重要"结论是否只是参数量效应?
    
    公平比较策略:
    - 参数量匹配: 消融相同参数量的MLP neurons vs Attention heads
    - MLP参数: 每个neuron = d_model (down_proj的一列) 
      → intermediate_size * d_model 总参数
    - Attn参数: 每个head = 4 * head_dim * d_model (Q/K/V/O)
      → n_heads * 4 * head_dim * d_model 总参数
    
    计算每个"单位参数"的KL贡献:
    - MLP: 关闭k个neurons → KL / (k * d_model)
    - Attn: 关闭k个heads → KL / (k * 4 * head_dim * d_model)
    """
    print("\n" + "="*60)
    print("Exp 1: 容量匹配消融 — 公平比较MLP vs Attention")
    print("="*60)
    
    layers = get_layers(model)
    n_layers = model_info.n_layers
    n_heads = get_n_heads(model)
    head_dim = get_head_dim(model)
    d_model = model_info.d_model
    intermediate_size = model_info.intermediate_size
    
    # 参数量计算
    mlp_params_per_neuron = d_model  # down_proj一列
    attn_params_per_head = 4 * head_dim * d_model  # Q+K+V+O for one head
    
    # MLP层总参数 vs Attention层总参数
    mlp_params_per_layer = intermediate_size * d_model * 3  # gate+up+down (近似)
    attn_params_per_layer = n_heads * attn_params_per_head
    
    print(f"  MLP params/layer: {mlp_params_per_layer:,} (intermediate={intermediate_size})")
    print(f"  Attn params/layer: {attn_params_per_layer:,} (n_heads={n_heads}, head_dim={head_dim})")
    print(f"  MLP/Attn参数比: {mlp_params_per_layer/attn_params_per_layer:.2f}x")
    
    # 测试句子 — 大样本
    test_sentences = [
        "The cat sat on the mat",
        "She walked to the store",
        "The sun was very bright",
        "He read an interesting book",
        "The water was ice cold",
        "Birds fly in the sky",
        "The food tasted great",
        "Music filled the room",
        "Time passes quickly",
        "The garden looked beautiful",
        "Rain fell from the clouds",
        "Children played in the park",
        "The mountain was very tall",
        "She wrote a long letter",
        "The river flowed gently",
        "Stars shine at night",
        "The cake smelled wonderful",
        "Wind blew through the trees",
        "The dog barked loudly",
        "Fire burned in the fireplace",
    ]
    
    # 目标层: 中间5层 (语义计算最密集的区域)
    target_layers = sorted(set([
        n_layers//4, n_layers//3, n_layers//2, 2*n_layers//3, 3*n_layers//4
    ]))
    print(f"  目标层: {target_layers}")
    
    results = {
        "mlp_params_per_layer": mlp_params_per_layer,
        "attn_params_per_layer": attn_params_per_layer,
        "param_ratio": mlp_params_per_layer / attn_params_per_layer,
        "target_layers": target_layers,
    }
    
    # --- 1a. 逐层MLP全消融 ---
    print("\n  [1a] 逐层MLP全消融...")
    mlp_layer_kls = {}
    for li in target_layers:
        kl_vals = []
        for sent in test_sentences:
            base_logits = get_base_logits(model, tokenizer, device, sent)
            hooks = register_mlp_layer_hook(layers, li)
            try:
                abl_logits = get_base_logits(model, tokenizer, device, sent)
                kl = compute_kl(base_logits, abl_logits)
                if kl >= 0:
                    kl_vals.append(kl)
            finally:
                for h in hooks: h.remove()
        mean_kl = float(np.mean(kl_vals)) if kl_vals else -1.0
        mlp_layer_kls[str(li)] = mean_kl
        print(f"    MLP L{li}: KL={mean_kl:.4f} (n={len(kl_vals)})")
    results["mlp_full_layer_ablation"] = mlp_layer_kls
    
    # --- 1b. 逐层Attention全消融 ---
    print("\n  [1b] 逐层Attention全消融...")
    attn_layer_kls = {}
    for li in target_layers:
        kl_vals = []
        for sent in test_sentences:
            base_logits = get_base_logits(model, tokenizer, device, sent)
            hooks = []
            for hi in range(n_heads):
                hooks.extend(register_o_proj_hook(layers, li, hi, n_heads, head_dim))
            try:
                abl_logits = get_base_logits(model, tokenizer, device, sent)
                kl = compute_kl(base_logits, abl_logits)
                if kl >= 0:
                    kl_vals.append(kl)
            finally:
                for h in hooks: h.remove()
        mean_kl = float(np.mean(kl_vals)) if kl_vals else -1.0
        attn_layer_kls[str(li)] = mean_kl
        print(f"    Attn L{li}: KL={mean_kl:.4f} (n={len(kl_vals)})")
    results["attn_full_layer_ablation"] = attn_layer_kls
    
    # --- 1c. 参数量匹配的MLP neuron消融 ---
    # 等价于1个head的参数量 = attn_params_per_head
    # 需要消融的neuron数 = attn_params_per_head / mlp_params_per_neuron
    neurons_per_head_eq = attn_params_per_head // mlp_params_per_neuron
    print(f"\n  [1c] 参数量匹配: 1个head ≈ {neurons_per_head_eq} MLP neurons")
    
    mlp_neuron_kls = {}
    for li in target_layers[:3]:  # 只测试3层以节省时间
        kl_vals = []
        for sent in test_sentences[:10]:  # 减少样本量
            base_logits = get_base_logits(model, tokenizer, device, sent)
            # 随机选择neurons
            rng = np.random.RandomState(42 + li)
            for trial in range(3):
                neuron_indices = rng.choice(intermediate_size, neurons_per_head_eq, replace=False).tolist()
                hooks = register_mlp_neuron_hook(layers, li, neuron_indices)
                try:
                    abl_logits = get_base_logits(model, tokenizer, device, sent)
                    kl = compute_kl(base_logits, abl_logits)
                    if kl >= 0:
                        kl_vals.append(kl)
                finally:
                    for h in hooks: h.remove()
        mean_kl = float(np.mean(kl_vals)) if kl_vals else -1.0
        mlp_neuron_kls[str(li)] = {
            "n_neurons": neurons_per_head_eq,
            "mean_kl": mean_kl,
            "kl_per_neuron": mean_kl / neurons_per_head_eq if mean_kl > 0 else 0,
        }
        print(f"    MLP {neurons_per_head_eq} neurons L{li}: KL={mean_kl:.6f} "
              f"(per-neuron={mean_kl/neurons_per_head_eq:.8f})")
    results["mlp_matched_neuron_ablation"] = mlp_neuron_kls
    
    # --- 1d. 单head消融 ---
    print("\n  [1d] 单head消融 (per-head KL)...")
    head_kls = {}
    for li in target_layers[:3]:
        head_kls_layer = []
        for hi in range(n_heads):
            kl_vals = []
            for sent in test_sentences[:5]:
                base_logits = get_base_logits(model, tokenizer, device, sent)
                hooks = register_o_proj_hook(layers, li, hi, n_heads, head_dim)
                try:
                    abl_logits = get_base_logits(model, tokenizer, device, sent)
                    kl = compute_kl(base_logits, abl_logits)
                    if kl >= 0:
                        kl_vals.append(kl)
                finally:
                    for h in hooks: h.remove()
            mean_kl = float(np.mean(kl_vals)) if kl_vals else -1.0
            head_kls_layer.append(mean_kl)
        head_kls[str(li)] = {
            "mean": float(np.mean([x for x in head_kls_layer if x >= 0])) if any(x >= 0 for x in head_kls_layer) else -1,
            "max": float(np.max([x for x in head_kls_layer if x >= 0])) if any(x >= 0 for x in head_kls_layer) else -1,
            "per_head_values": head_kls_layer,
        }
        valid = [x for x in head_kls_layer if x >= 0]
        print(f"    L{li}: mean_head_KL={np.mean(valid):.6f}, max={np.max(valid):.6f}")
    results["single_head_ablation"] = head_kls
    
    # --- 1e. 归一化比较 ---
    print("\n  [1e] 归一化比较 (KL per parameter)...")
    for li in target_layers:
        li_str = str(li)
        mlp_kl = mlp_layer_kls.get(li_str, -1)
        attn_kl = attn_layer_kls.get(li_str, -1)
        
        if mlp_kl > 0 and attn_kl > 0:
            mlp_kl_per_param = mlp_kl / mlp_params_per_layer
            attn_kl_per_param = attn_kl / attn_params_per_layer
            print(f"    L{li}: MLP KL/param={mlp_kl_per_param:.10f}, "
                  f"Attn KL/param={attn_kl_per_param:.10f}, "
                  f"ratio={attn_kl_per_param/mlp_kl_per_param:.2f}x")
    
    return results


# ============================================================
# Exp 2: 全层轨迹聚类 — 轨迹族分析
# ============================================================

def exp2_trajectory_clustering(model, tokenizer, device, model_info):
    """
    核心问题: 不同语义是否对应不同轨迹族?
    
    方法: 
    1. 对6个类别的90个词, 计算(h_0, h_1, ..., h_L)完整轨迹
    2. 将轨迹展平为向量, 做PCA降维
    3. 计算类内/类间距离
    4. 聚类分析: 轨迹能否按语义类别分开?
    """
    print("\n" + "="*60)
    print("Exp 2: 全层轨迹聚类 — 轨迹族分析")
    print("="*60)
    
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    # 收集所有词的完整轨迹
    all_trajectories = {}  # {word: [n_layers+1, d_model]}
    all_labels = {}  # {word: category}
    
    for cat, words in SEMANTIC_CATEGORIES.items():
        print(f"  Processing {cat} ({len(words)} words)...")
        for word in words:
            prompt = f"The {word}"
            try:
                traj = get_all_hidden_states(model, tokenizer, device, prompt)
                # 检查nan/inf
                if np.isnan(traj).any() or np.isinf(traj).any():
                    print(f"    Skipping {word}: nan/inf in trajectory")
                    continue
                all_trajectories[word] = traj
                all_labels[word] = cat
            except Exception as e:
                print(f"    Failed: {word} - {e}")
    
    n_words = len(all_trajectories)
    print(f"  成功收集 {n_words} 个词的轨迹")
    
    if n_words < 10:
        return {"error": "too few words"}
    
    # 构建轨迹矩阵: [n_words, (n_layers+1) * d_model]
    words = sorted(all_trajectories.keys())
    traj_flat = np.array([all_trajectories[w].flatten() for w in words])
    
    # 归一化: L2 normalize每个轨迹
    norms = np.linalg.norm(traj_flat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    traj_flat_norm = traj_flat / norms
    
    # PCA降维
    from sklearn.decomposition import PCA
    n_components = min(50, traj_flat_norm.shape[0] - 1, traj_flat_norm.shape[1])
    pca = PCA(n_components=n_components)
    traj_pca = pca.fit_transform(traj_flat_norm)
    
    print(f"  PCA: {n_components} components, explained variance ratio sum={sum(pca.explained_variance_ratio_):.4f}")
    print(f"  Top-10 variance ratio: {pca.explained_variance_ratio_[:10]}")
    
    # 类内/类间距离
    categories = list(SEMANTIC_CATEGORIES.keys())
    intra_dists = defaultdict(list)
    inter_dists = defaultdict(list)
    
    labels = np.array([all_labels[w] for w in words])
    
    for cat in categories:
        cat_mask = labels == cat
        cat_indices = np.where(cat_mask)[0]
        other_mask = ~cat_mask
        
        # 类内距离
        if len(cat_indices) > 1:
            for i in range(len(cat_indices)):
                for j in range(i+1, len(cat_indices)):
                    d = np.linalg.norm(traj_pca[cat_indices[i]] - traj_pca[cat_indices[j]])
                    intra_dists[cat].append(d)
        
        # 类间距离 (采样)
        other_indices = np.where(other_mask)[0]
        for ci in cat_indices[:5]:
            for oi in other_indices[:10]:
                d = np.linalg.norm(traj_pca[ci] - traj_pca[oi])
                inter_dists[cat].append(d)
    
    # 聚类质量
    from sklearn.metrics import silhouette_score
    sil_score = silhouette_score(traj_pca[:, :10], labels)
    
    print(f"  Silhouette score (轨迹聚类): {sil_score:.4f}")
    for cat in categories:
        intra = np.mean(intra_dists[cat]) if intra_dists[cat] else 0
        inter = np.mean(inter_dists[cat]) if inter_dists[cat] else 0
        ratio = inter / max(intra, 1e-10)
        print(f"    {cat}: intra_dist={intra:.4f}, inter_dist={inter:.4f}, "
              f"inter/intra={ratio:.2f}")
    
    # 层级分解: 单层的聚类质量
    layer_sil = {}
    for li in range(n_layers + 1):
        layer_vecs = np.array([all_trajectories[w][li] for w in words])
        layer_norms = np.linalg.norm(layer_vecs, axis=1, keepdims=True)
        layer_norms = np.maximum(layer_norms, 1e-10)
        layer_vecs_norm = layer_vecs / layer_norms
        
        try:
            sil = silhouette_score(layer_vecs_norm, labels)
            layer_sil[str(li)] = float(sil)
        except:
            layer_sil[str(li)] = -1.0
    
    # 找最佳聚类层
    best_layer = max(layer_sil, key=lambda k: layer_sil[k])
    print(f"  最佳单层聚类: L{best_layer}, sil={layer_sil[best_layer]:.4f}")
    
    # 轨迹vs单层的比较
    print(f"  轨迹聚类 sil={sil_score:.4f} vs 最佳单层 sil={layer_sil[best_layer]:.4f}")
    
    # 层间轨迹差异: 逐层cosine similarity
    cat_trajectories = defaultdict(list)
    for w in words:
        cat_trajectories[all_labels[w]].append(all_trajectories[w])
    
    # 每个类别的平均轨迹
    cat_mean_trajs = {}
    for cat, trajs in cat_trajectories.items():
        cat_mean_trajs[cat] = np.mean(trajs, axis=0)  # [n_layers+1, d_model]
    
    # 类间轨迹的层间余弦相似度
    inter_cat_cos = {}
    cat_list = list(cat_mean_trajs.keys())
    for i in range(len(cat_list)):
        for j in range(i+1, len(cat_list)):
            c1, c2 = cat_list[i], cat_list[j]
            layer_cos = []
            for li in range(n_layers + 1):
                v1 = cat_mean_trajs[c1][li]
                v2 = cat_mean_trajs[c2][li]
                cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
                layer_cos.append(float(cos))
            inter_cat_cos[f"{c1}_vs_{c2}"] = layer_cos
    
    results = {
        "n_words": n_words,
        "silhouette_score_full_trajectory": float(sil_score),
        "layer_silhouette_scores": layer_sil,
        "best_single_layer": int(best_layer),
        "best_single_layer_sil": layer_sil[best_layer],
        "pca_explained_variance_top10": pca.explained_variance_ratio_[:10].tolist(),
        "category_distances": {
            cat: {
                "intra_mean": float(np.mean(intra_dists[cat])) if intra_dists[cat] else 0,
                "inter_mean": float(np.mean(inter_dists[cat])) if inter_dists[cat] else 0,
            }
            for cat in categories
        },
        "inter_cat_cos_layer_profile": inter_cat_cos,
    }
    
    return results


# ============================================================
# Exp 3: MLP神经元选择性
# ============================================================

def exp3_mlp_neuron_selectivity(model, tokenizer, device, model_info):
    """
    核心问题: 哪些MLP neurons对哪些语义类别有选择性响应?
    
    方法:
    1. 对6类90个词, 计算MLP gate activations
    2. 找对特定类别选择性响应的neurons
    3. 分析选择性neurons的空间分布
    """
    print("\n" + "="*60)
    print("Exp 3: MLP神经元选择性")
    print("="*60)
    
    layers = get_layers(model)
    n_layers = model_info.n_layers
    intermediate_size = model_info.intermediate_size
    d_model = model_info.d_model
    
    target_layers = sorted(set([
        n_layers//6, n_layers//3, n_layers//2, 2*n_layers//3, 5*n_layers//6
    ]))
    print(f"  目标层: {target_layers}, intermediate_size={intermediate_size}")
    
    # 收集MLP gate activations
    # MLP(x) = W_down · (σ(W_gate · x) ⊙ (W_up · x))
    # gate activation = σ(W_gate · x) — 这是neuron是否被激活的指示器
    
    all_gate_acts = {}  # {(layer_idx, word): [intermediate_size]}
    all_labels = {}
    
    for cat, words in SEMANTIC_CATEGORIES.items():
        for word in words[:10]:  # 每类10个词
            prompt = f"The {word}"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(device)
            attn_mask = inputs["attention_mask"].to(device)
            all_labels[word] = cat
            
            for li in target_layers:
                captured = {}
                def make_gate_hook(key):
                    def hook(module, input, output):
                        # gate_proj的输出 = [batch, seq, intermediate_size]
                        # 对于merged gate_up_proj, 输出是 [batch, seq, 2*intermediate_size]
                        # 我们只取前半部分(gate部分)
                        out = output[0, -1, :].float().cpu().numpy()
                        captured[key] = out
                    return hook
                
                # 兼容不同MLP架构
                mlp = layers[li].mlp
                if hasattr(mlp, 'gate_proj'):
                    h = mlp.gate_proj.register_forward_hook(
                        make_gate_hook(f"L{li}")
                    )
                elif hasattr(mlp, 'gate_up_proj'):
                    # GLM4: gate_up_proj输出是2*intermediate, 取前半为gate
                    def make_merged_gate_hook(key, intermediate_size):
                        def hook(module, input, output):
                            out = output[0, -1, :].float().cpu().numpy()
                            gate_out = out[:intermediate_size]  # 只取gate部分
                            captured[key] = gate_out
                        return hook
                    h = mlp.gate_up_proj.register_forward_hook(
                        make_merged_gate_hook(f"L{li}", model_info.intermediate_size)
                    )
                else:
                    continue
                try:
                    with torch.no_grad():
                        model(input_ids=input_ids, attention_mask=attn_mask)
                finally:
                    h.remove()
                
                if f"L{li}" in captured:
                    val = captured[f"L{li}"]
                    if np.isnan(val).any() or np.isinf(val).any():
                        continue
                    all_gate_acts[(li, word)] = val
    
    n_collected = len(all_gate_acts)
    print(f"  收集了 {n_collected} 个gate activation vectors")
    
    if n_collected < 10:
        return {"error": "too few gate activations"}
    
    # 对每层分析neuron选择性
    results = {}
    categories = list(SEMANTIC_CATEGORIES.keys())
    
    for li in target_layers:
        # 构建gate activation矩阵: [n_words, intermediate_size]
        layer_words = [w for (l, w) in all_gate_acts if l == li]
        if not layer_words:
            continue
        
        gate_mat = np.array([all_gate_acts[(li, w)] for w in layer_words])
        layer_labels = np.array([all_labels[w] for w in layer_words])
        
        # 统计每类每neuron的mean activation
        cat_means = {}
        for cat in categories:
            mask = layer_labels == cat
            if mask.sum() > 0:
                cat_means[cat] = gate_mat[mask].mean(axis=0)  # [intermediate_size]
        
        if len(cat_means) < 2:
            continue
        
        # 计算选择性: 对每个neuron, 计算它在某类上的activation vs 其他类
        # 选择性 = (cat_mean - other_mean) / (cat_mean + other_mean + eps)
        selectivity = {}
        for cat in categories:
            if cat not in cat_means:
                continue
            cat_mean = cat_means[cat]
            other_mean = np.mean([cat_means[c] for c in cat_means if c != cat], axis=0)
            sel = (cat_mean - other_mean) / (np.abs(cat_mean) + np.abs(other_mean) + 1e-6)
            selectivity[cat] = sel  # [intermediate_size]
        
        # 找top选择性neurons
        top_selective = {}
        for cat in categories:
            if cat not in selectivity:
                continue
            sel = selectivity[cat]
            # 只看正选择性 (对该类激活更强)
            top_indices = np.argsort(sel)[-20:][::-1]
            top_selective[cat] = {
                "top_neuron_indices": top_indices.tolist(),
                "top_selectivity_values": sel[top_indices].tolist(),
            }
        
        # 类间选择性重叠: 不同类别的top neurons是否重叠?
        overlap_matrix = {}
        cat_list = [c for c in categories if c in top_selective]
        for i, c1 in enumerate(cat_list):
            for j, c2 in enumerate(cat_list):
                if i >= j:
                    continue
                top1 = set(top_selective[c1]["top_neuron_indices"][:10])
                top2 = set(top_selective[c2]["top_neuron_indices"][:10])
                overlap = len(top1 & top2)
                overlap_matrix[f"{c1}_vs_{c2}"] = overlap
        
        # Neuron激活率的类别差异
        # 多少neurons被某类显著激活?
        activation_rate = {}
        for cat in categories:
            if cat not in cat_means:
                continue
            # 被激活 = gate activation > 0 (SiLU激活前)
            active_rate = float((cat_means[cat] > 0).mean())
            strong_active_rate = float((cat_means[cat] > 1.0).mean())
            activation_rate[cat] = {
                "active_rate": active_rate,
                "strong_active_rate": strong_active_rate,
            }
        
        results[str(li)] = {
            "top_selective_neurons": top_selective,
            "overlap_matrix": overlap_matrix,
            "activation_rates": activation_rate,
            "gate_sparsity": float((gate_mat == 0).mean()),
            "gate_mean": float(gate_mat.mean()),
            "gate_std": float(gate_mat.std()),
        }
        
        print(f"  L{li}: gate_mean={gate_mat.mean():.4f}, gate_std={gate_mat.std():.4f}, "
              f"sparsity={float((gate_mat == 0).mean()):.4f}")
        for cat in cat_list:
            sel_vals = selectivity[cat]
            n_high_sel = int((sel_vals > 0.3).sum())
            print(f"    {cat}: {n_high_sel} neurons with selectivity>0.3")
    
    return results


# ============================================================
# Exp 4: 轨迹分叉深度分析
# ============================================================

def exp4_trajectory_divergence(model, tokenizer, device, model_info):
    """
    核心问题: 歧义词在哪些层、哪些维度分叉?
    
    方法:
    1. 对15个歧义词, 比较不同上下文中的完整轨迹
    2. 逐层分析分叉程度 (cosine similarity, 欧氏距离)
    3. 分叉维度的PCA分析: 分叉主要在哪些方向?
    4. 分叉层的Jacobian分析: 分叉是否在Jacobian高敏感方向?
    """
    print("\n" + "="*60)
    print("Exp 4: 轨迹分叉深度分析")
    print("="*60)
    
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    results = {}
    
    for word, contexts in AMBIGUOUS_WORDS.items():
        ctx1, ctx2 = contexts
        
        # 获取两条轨迹
        traj1 = get_all_hidden_states(model, tokenizer, device, ctx1)
        traj2 = get_all_hidden_states(model, tokenizer, device, ctx2)
        
        # 检查nan
        if np.isnan(traj1).any() or np.isnan(traj2).any():
            results[word] = {"error": "nan in trajectory"}
            continue
        
        # 逐层计算分叉程度
        layer_cos = []
        layer_dist = []
        for li in range(n_layers + 1):
            v1 = traj1[li]
            v2 = traj2[li]
            cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            dist = np.linalg.norm(v1 - v2)
            layer_cos.append(float(cos))
            layer_dist.append(float(dist))
        
        # 找分叉最大的层
        min_cos_layer = int(np.argmin(layer_cos))
        max_dist_layer = int(np.argmax(layer_dist))
        
        # 分叉方向: 差向量的PCA
        # 在分叉最大层, 差向量 = traj1[min_cos_layer] - traj2[min_cos_layer]
        delta = traj1[min_cos_layer] - traj2[min_cos_layer]
        delta_norm = np.linalg.norm(delta)
        delta_direction = delta / max(delta_norm, 1e-10)
        
        # 差向量在各层的投影
        delta_projections = []
        for li in range(n_layers + 1):
            proj = np.dot(traj1[li] - traj2[li], delta_direction)
            delta_projections.append(float(proj))
        
        # 逐层的分叉角度变化率
        cos_changes = [abs(layer_cos[i+1] - layer_cos[i]) for i in range(len(layer_cos)-1)]
        max_change_layer = int(np.argmax(cos_changes))
        
        results[word] = {
            "contexts": contexts,
            "layer_cosine": layer_cos,
            "layer_distance": layer_dist,
            "min_cos_layer": min_cos_layer,
            "min_cos_value": layer_cos[min_cos_layer],
            "max_dist_layer": max_dist_layer,
            "max_dist_value": layer_dist[max_dist_layer],
            "max_cos_change_layer": max_change_layer,
            "max_cos_change": cos_changes[max_change_layer],
            "delta_projection_per_layer": delta_projections,
        }
        
        print(f"  '{word}': min_cos L{min_cos_layer} ({layer_cos[min_cos_layer]:.4f}), "
              f"max_dist L{max_dist_layer} ({layer_dist[max_dist_layer]:.4f}), "
              f"max_change L{max_change_layer}")
    
    # 统计: 分叉最常发生在哪些层
    valid_results = {w: r for w, r in results.items() if "error" not in r}
    min_cos_layers = [r["min_cos_layer"] for r in valid_results.values()]
    max_dist_layers = [r["max_dist_layer"] for r in valid_results.values()]
    max_change_layers = [r["max_cos_change_layer"] for r in valid_results.values()]
    
    if not min_cos_layers:
        results["_summary"] = {"error": "no valid results"}
        return results
    
    print(f"\n  分叉统计:")
    print(f"    min_cos 层分布: mean={np.mean(min_cos_layers):.1f}, "
          f"range=[{np.min(min_cos_layers)}, {np.max(min_cos_layers)}]")
    print(f"    max_dist 层分布: mean={np.mean(max_dist_layers):.1f}, "
          f"range=[{np.min(max_dist_layers)}, {np.max(max_dist_layers)}]")
    print(f"    max_change 层分布: mean={np.mean(max_change_layers):.1f}, "
          f"range=[{np.min(max_change_layers)}, {np.max(max_change_layers)}]")
    
    results["_summary"] = {
        "min_cos_layer_stats": {
            "mean": float(np.mean(min_cos_layers)),
            "std": float(np.std(min_cos_layers)),
        },
        "max_dist_layer_stats": {
            "mean": float(np.mean(max_dist_layers)),
            "std": float(np.std(max_dist_layers)),
        },
        "max_change_layer_stats": {
            "mean": float(np.mean(max_change_layers)),
            "std": float(np.std(max_change_layers)),
        },
    }
    
    return results


# ============================================================
# Exp 5: 动力学邻接性 — 概念的动力学空间结构
# ============================================================

def exp5_dynamical_adjacency(model, tokenizer, device, model_info):
    """
    核心问题: "概念"在动力学空间中的邻近关系是什么?
    
    方法:
    1. 对大量词对, 计算完整轨迹间的距离
    2. 构建动力学邻接图
    3. 分析: 语义相关词是否在动力学空间中邻近?
    4. 与embedding空间的邻接性比较
    
    关键比较:
    - Embedding空间 (L0) 的邻接性
    - 中间层空间 (L_mid) 的邻接性
    - 输出层空间 (L_last) 的邻接性
    - 完整轨迹空间的邻接性
    """
    print("\n" + "="*60)
    print("Exp 5: 动力学邻接性")
    print("="*60)
    
    n_layers = model_info.n_layers
    
    # 定义语义相关的词对 vs 无关词对
    related_pairs = [
        ("apple", "banana"), ("cat", "dog"), ("car", "bus"),
        ("Paris", "London"), ("freedom", "justice"), ("hammer", "screwdriver"),
        ("apple", "orange"), ("horse", "elephant"), ("train", "airplane"),
        ("Tokyo", "Seoul"), ("truth", "beauty"), ("knife", "scissors"),
        ("grape", "cherry"), ("tiger", "lion"), ("bicycle", "motorcycle"),
        ("Berlin", "Vienna"), ("wisdom", "courage"), ("drill", "saw"),
        ("mango", "peach"), ("eagle", "dolphin"), ("boat", "ship"),
        ("Cairo", "Dubai"), ("peace", "hope"), ("chisel", "clamp"),
        ("watermelon", "strawberry"), ("bear", "wolf"), ("truck", "van"),
        ("Mumbai", "Bangkok"), ("love", "trust"), ("ruler", "level"),
        # 跨类别但语义相关
        ("apple", "red"), ("sky", "blue"), ("fire", "hot"),
        ("ice", "cold"), ("cat", "fur"), ("bird", "fly"),
        ("fish", "water"), ("sun", "bright"), ("moon", "night"),
        ("king", "crown"), ("doctor", "hospital"), ("teacher", "school"),
    ]
    
    unrelated_pairs = [
        ("apple", "freedom"), ("cat", "hammer"), ("car", "justice"),
        ("Paris", "screwdriver"), ("banana", "truth"), ("dog", "ruler"),
        ("orange", "courage"), ("horse", "chisel"), ("train", "beauty"),
        ("Tokyo", "scissors"), ("grape", "wisdom"), ("elephant", "clamp"),
        ("cherry", "peace"), ("tiger", "drill"), ("bicycle", "Vienna"),
        ("Berlin", "mango"), ("eagle", "level"), ("boat", "hope"),
        ("Cairo", "watermelon"), ("bear", "Bangkok"), ("truck", "love"),
        ("Mumbai", "strawberry"), ("saw", "wolf"), ("van", "Dubai"),
        ("king", "screwdriver"), ("doctor", "banana"), ("teacher", "hammer"),
        ("sun", "freedom"), ("moon", "justice"), ("fire", "truth"),
    ]
    
    def compute_pair_dynamics(word1, word2):
        """计算两个词在动力学空间的距离"""
        prompt1 = f"The {word1}"
        prompt2 = f"The {word2}"
        
        traj1 = get_all_hidden_states(model, tokenizer, device, prompt1)
        traj2 = get_all_hidden_states(model, tokenizer, device, prompt2)
        
        # 检查nan
        if np.isnan(traj1).any() or np.isnan(traj2).any():
            return None
        
        # 各层cosine
        layer_cos = []
        for li in range(n_layers + 1):
            v1 = traj1[li]
            v2 = traj2[li]
            cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            layer_cos.append(float(cos))
        
        # 轨迹整体距离
        traj_dist = np.linalg.norm(traj1.flatten() - traj2.flatten())
        
        return {
            "layer_cos": layer_cos,
            "traj_dist": float(traj_dist),
            "L0_cos": layer_cos[0],
            "Lmid_cos": layer_cos[n_layers//2],
            "Llast_cos": layer_cos[-1],
        }
    
    related_results = []
    for w1, w2 in related_pairs:
        try:
            r = compute_pair_dynamics(w1, w2)
            if r is not None:
                r["pair"] = (w1, w2)
                related_results.append(r)
        except Exception as e:
            print(f"    Failed: {w1}-{w2}: {e}")
    
    unrelated_results = []
    for w1, w2 in unrelated_pairs:
        try:
            r = compute_pair_dynamics(w1, w2)
            if r is not None:
                r["pair"] = (w1, w2)
                unrelated_results.append(r)
        except Exception as e:
            print(f"    Failed: {w1}-{w2}: {e}")
    
    # 统计比较
    def summarize(results_list, name):
        if not results_list:
            return {}
        l0_cos = [r["L0_cos"] for r in results_list]
        lmid_cos = [r["Lmid_cos"] for r in results_list]
        llast_cos = [r["Llast_cos"] for r in results_list]
        traj_dists = [r["traj_dist"] for r in results_list]
        return {
            "n_pairs": len(results_list),
            "L0_cos_mean": float(np.mean(l0_cos)),
            "Lmid_cos_mean": float(np.mean(lmid_cos)),
            "Llast_cos_mean": float(np.mean(llast_cos)),
            "traj_dist_mean": float(np.mean(traj_dists)),
        }
    
    related_summary = summarize(related_results, "related")
    unrelated_summary = summarize(unrelated_results, "unrelated")
    
    print(f"  相关词对 ({related_summary.get('n_pairs', 0)}):")
    print(f"    L0 cos={related_summary.get('L0_cos_mean', 0):.4f}, "
          f"Lmid cos={related_summary.get('Lmid_cos_mean', 0):.4f}, "
          f"Llast cos={related_summary.get('Llast_cos_mean', 0):.4f}")
    print(f"  无关词对 ({unrelated_summary.get('n_pairs', 0)}):")
    print(f"    L0 cos={unrelated_summary.get('L0_cos_mean', 0):.4f}, "
          f"Lmid cos={unrelated_summary.get('Lmid_cos_mean', 0):.4f}, "
          f"Llast cos={unrelated_summary.get('Llast_cos_mean', 0):.4f}")
    
    # 关键比较: 哪层的cos差异最大?
    if related_summary and unrelated_summary:
        for layer_name in ["L0", "Lmid", "Llast"]:
            r_val = related_summary.get(f"{layer_name}_cos_mean", 0)
            u_val = unrelated_summary.get(f"{layer_name}_cos_mean", 0)
            diff = r_val - u_val
            print(f"    {layer_name}: related-unrelated cos差异 = {diff:.4f}")
    
    # 层级cos profile
    if related_results and unrelated_results:
        mean_related_cos = np.mean([r["layer_cos"] for r in related_results], axis=0)
        mean_unrelated_cos = np.mean([r["layer_cos"] for r in unrelated_results], axis=0)
        cos_diff = mean_related_cos - mean_unrelated_cos
        
        max_diff_layer = int(np.argmax(cos_diff))
        print(f"  语义区分度最大层: L{max_diff_layer} (cos差异={cos_diff[max_diff_layer]:.4f})")
    
    results = {
        "related_pairs_summary": related_summary,
        "unrelated_pairs_summary": unrelated_summary,
        "n_related": len(related_results),
        "n_unrelated": len(unrelated_results),
    }
    
    if related_results and unrelated_results:
        results["mean_related_cos_profile"] = mean_related_cos.tolist()
        results["mean_unrelated_cos_profile"] = mean_unrelated_cos.tolist()
        results["cos_diff_profile"] = cos_diff.tolist()
        results["max_discrimination_layer"] = max_diff_layer
    
    return results


# ============================================================
# 主流程
# ============================================================

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    exp_num = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # 0=全部
    
    print(f"="*60)
    print(f"Phase 127: 语义动力学轨迹分析")
    print(f"模型: {model_name}")
    print(f"="*60)
    
    # 加载模型
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    print(f"  n_layers={model_info.n_layers}, d_model={model_info.d_model}, "
          f"intermediate={model_info.intermediate_size}")
    
    all_results = {
        "model": model_name,
        "model_info": {
            "class": model_info.model_class,
            "n_layers": model_info.n_layers,
            "d_model": model_info.d_model,
            "intermediate_size": model_info.intermediate_size,
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    try:
        if exp_num == 0 or exp_num == 1:
            t0 = time.time()
            all_results["exp1_capacity_matched_ablation"] = exp1_capacity_matched_ablation(
                model, tokenizer, device, model_info
            )
            print(f"  Exp 1 完成: {time.time()-t0:.1f}s")
        
        if exp_num == 0 or exp_num == 2:
            t0 = time.time()
            all_results["exp2_trajectory_clustering"] = exp2_trajectory_clustering(
                model, tokenizer, device, model_info
            )
            print(f"  Exp 2 完成: {time.time()-t0:.1f}s")
        
        if exp_num == 0 or exp_num == 3:
            t0 = time.time()
            all_results["exp3_mlp_neuron_selectivity"] = exp3_mlp_neuron_selectivity(
                model, tokenizer, device, model_info
            )
            print(f"  Exp 3 完成: {time.time()-t0:.1f}s")
        
        if exp_num == 0 or exp_num == 4:
            t0 = time.time()
            all_results["exp4_trajectory_divergence"] = exp4_trajectory_divergence(
                model, tokenizer, device, model_info
            )
            print(f"  Exp 4 完成: {time.time()-t0:.1f}s")
        
        if exp_num == 0 or exp_num == 5:
            t0 = time.time()
            all_results["exp5_dynamical_adjacency"] = exp5_dynamical_adjacency(
                model, tokenizer, device, model_info
            )
            print(f"  Exp 5 完成: {time.time()-t0:.1f}s")
    
    finally:
        release_model(model)
    
    # 保存结果
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'glm5_temp')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"phase127_{model_name}_semantic_dynamics.json")
    
    # 转换numpy类型
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(x) for x in obj]
        return obj
    
    all_results = convert(all_results)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存: {out_path}")
    print(f"Phase 127 完成!")


if __name__ == "__main__":
    main()
