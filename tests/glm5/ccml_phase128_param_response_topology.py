"""
Phase 128: 参数响应拓扑分析 — 从"欧氏空间"到"激活图空间"
===========================================================

用户对Phase 127的关键修正:
  1. 轨迹聚类3倍优势可能是维度作弊 (94720 vs 2560)
  2. 语义可能不是欧氏几何对象, 而是拓扑对象
  3. 真正该研究的是"参数激活拓扑" — 输入激活哪些参数形成激活图
  4. "概念"本质上是激活图的重叠模式, 不是向量距离

本阶段5个实验:
- Exp 1: 维度控制轨迹比较 — 修复维度作弊问题
- Exp 2: 参数激活图 G(x) — 构建输入的条件参数响应图
- Exp 3: 概念拓扑邻接 — 激活图重叠 vs 欧氏距离, 谁更好预测语义?
- Exp 4: 组合语义(属性绑定) — "red apple" vs "green apple", 属性如何改变激活图
- Exp 5: 语法绑定 — "dog bites man" vs "man bites dog", 同token不同回路
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
# 语义类别和测试数据
# ============================================================

SEMANTIC_CATEGORIES = {
    "animal": ["cat", "dog", "horse", "lion", "tiger", "bear", "wolf", "fox", "deer", "rabbit",
               "elephant", "giraffe", "zebra", "monkey", "whale", "dolphin", "eagle", "snake",
               "shark", "penguin", "owl", "parrot", "turtle", "frog", "beaver"],
    "fruit": ["apple", "banana", "orange", "grape", "mango", "peach", "cherry", "lemon",
              "melon", "pear", "plum", "kiwi", "fig", "lime", "coconut", "papaya",
              "apricot", "avocado", "blueberry", "strawberry", "raspberry", "pineapple"],
    "place": ["city", "village", "mountain", "river", "ocean", "forest", "desert", "island",
              "valley", "lake", "cave", "bridge", "tower", "castle", "temple", "harbor",
              "canyon", "cliff", "meadow", "swamp", "glacier", "volcano"],
    "tool": ["hammer", "knife", "saw", "drill", "wrench", "screwdriver", "pliers", "axe",
             "shovel", "rake", "chisel", "level", "welder", "clamp", "plier", "mallet",
             "anvil", "crowbar", "pliers", "caliper", "compass", "ruler"],
    "abstract": ["freedom", "justice", "beauty", "truth", "courage", "wisdom", "love",
                 "hope", "peace", "anger", "fear", "joy", "sorrow", "guilt", "pride",
                 "shame", "trust", "loyalty", "honor", "mercy", "faith", "doubt"],
}

# Exp 4: 组合语义 — 属性+名词
COMPOSITIONAL_PAIRS = [
    # 核心名词: apple, 同属性对比
    ("red apple", "fruit"), ("green apple", "fruit"), ("big apple", "fruit"),
    ("rotten apple", "fruit"), ("sweet apple", "fruit"), ("sour apple", "fruit"),
    # 核心名词: dog, 同属性对比
    ("big dog", "animal"), ("small dog", "animal"), ("fierce dog", "animal"),
    ("friendly dog", "animal"), ("old dog", "animal"), ("young dog", "animal"),
    # 核心名词: city, 同属性对比
    ("big city", "place"), ("small city", "place"), ("ancient city", "place"),
    ("modern city", "place"), ("busy city", "place"), ("quiet city", "place"),
    # 不同名词同属性: 测试属性是否形成独立激活模式
    ("red apple", "fruit"), ("red car", "tool"), ("red house", "place"),
    ("big dog", "animal"), ("big city", "place"), ("big mountain", "place"),
]

# Exp 5: 语法绑定 — 同token不同语序
SYNTACTIC_PAIRS = [
    # 主宾互换
    ("dog bites man", "man bites dog"),
    ("cat chases mouse", "mouse chases cat"),
    ("teacher praises student", "student praises teacher"),
    ("doctor helps patient", "patient helps doctor"),
    ("police arrests criminal", "criminal arrests police"),
    # 被动 vs 主动
    ("dog bites man", "man is bitten by dog"),
    ("cat chases mouse", "mouse is chased by cat"),
    # 否定 vs 肯定
    ("dog bites man", "dog does not bite man"),
    ("cat likes fish", "cat does not like fish"),
    # 条件 vs 陈述
    ("if it rains, I stay home", "it rains and I stay home"),
    # 时态变化
    ("dog bit man", "dog will bite man"),
    ("cat caught mouse", "cat will catch mouse"),
]


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
    return 20  # fallback


def get_all_hidden_states(model, tokenizer, device, prompt, max_length=64):
    """获取所有层的hidden states, 返回 [n_layers+1, d_model]"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True)
    
    hs = out.hidden_states  # tuple of [1, seq_len, d_model]
    # 取最后一个token的hidden state
    trajectory = torch.stack([h[0, -1, :].float().cpu() for h in hs])  # [n_layers+1, d_model]
    return trajectory.numpy()


def get_intermediate_size(model, model_info):
    """获取intermediate_size"""
    if model_info.intermediate_size > 0:
        return model_info.intermediate_size
    layers = get_layers(model)
    mlp = layers[0].mlp
    if hasattr(mlp, 'gate_up_proj'):
        return mlp.gate_up_proj.weight.shape[0] // 2
    elif hasattr(mlp, 'up_proj'):
        return mlp.up_proj.weight.shape[0]
    return 0


# ============================================================
# Exp 1: 维度控制轨迹比较
# ============================================================

def exp1_dimension_controlled_comparison(model, tokenizer, device, model_info):
    """
    修复维度作弊: 在相同维度下比较轨迹 vs 单层
    
    方法:
    1. 单层PCA → 100维, 计算silhouette
    2. 轨迹PCA → 100维, 计算silhouette
    3. 轨迹Fourier谱 → 比较频域信息
    4. Dynamic Mode Decomposition → 提取动力学模式
    """
    print("\n" + "="*60)
    print("Exp 1: 维度控制轨迹比较")
    print("="*60)
    
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    # 收集所有词的轨迹
    all_trajectories = {}
    all_labels = {}
    
    for cat, words in SEMANTIC_CATEGORIES.items():
        print(f"  Processing {cat} ({len(words)} words)...")
        for word in words:
            prompt = f"The {word}"
            try:
                traj = get_all_hidden_states(model, tokenizer, device, prompt)
                if np.isnan(traj).any() or np.isinf(traj).any():
                    continue
                all_trajectories[word] = traj  # [n_layers+1, d_model]
                all_labels[word] = cat
            except Exception as e:
                pass
    
    n_words = len(all_trajectories)
    print(f"  Collected {n_words} valid trajectories")
    
    if n_words < 10:
        return {"error": "too few valid trajectories"}
    
    # 构建数据矩阵
    words = list(all_trajectories.keys())
    labels = np.array([all_labels[w] for w in words])
    
    # --- 方法1: 单层PCA vs 轨迹PCA, 同维度比较 ---
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    
    # 找最佳单层
    best_single_sil = -1
    best_single_layer = -1
    single_layer_sils = {}
    
    for li in range(n_layers + 1):
        # 单层数据: [n_words, d_model]
        single_data = np.array([all_trajectories[w][li] for w in words])
        if single_data.shape[0] < 10:
            continue
        
        # PCA到100维
        n_comp = min(100, min(single_data.shape) - 1)
        try:
            pca = PCA(n_components=n_comp)
            reduced = pca.fit_transform(single_data)
            sil = silhouette_score(reduced, labels)
            single_layer_sils[li] = sil
            if sil > best_single_sil:
                best_single_sil = sil
                best_single_layer = li
        except Exception:
            pass
    
    print(f"  Best single layer: L{best_single_layer}, sil={best_single_sil:.4f}")
    
    # 轨迹数据: [n_words, (n_layers+1)*d_model]
    traj_flat = np.array([all_trajectories[w].flatten() for w in words])
    
    # PCA到相同维度(100维)
    n_comp = min(100, min(traj_flat.shape) - 1)
    try:
        pca_traj = PCA(n_components=n_comp)
        traj_reduced = pca_traj.fit_transform(traj_flat)
        traj_sil_100 = silhouette_score(traj_reduced, labels)
    except Exception:
        traj_sil_100 = -1
    
    print(f"  Trajectory PCA-{n_comp}: sil={traj_sil_100:.4f}")
    
    # --- 逐维度比较: 10, 20, 50, 100, 200 ---
    dim_comparison = {}
    for dim in [10, 20, 50, 100, 200]:
        # 单层最佳层 PCA
        single_data = np.array([all_trajectories[w][best_single_layer] for w in words])
        n_comp_s = min(dim, min(single_data.shape) - 1)
        try:
            pca_s = PCA(n_components=n_comp_s)
            s_reduced = pca_s.fit_transform(single_data)
            s_sil = silhouette_score(s_reduced, labels)
        except Exception:
            s_sil = -1
        
        # 轨迹 PCA
        n_comp_t = min(dim, min(traj_flat.shape) - 1)
        try:
            pca_t = PCA(n_components=n_comp_t)
            t_reduced = pca_t.fit_transform(traj_flat)
            t_sil = silhouette_score(t_reduced, labels)
        except Exception:
            t_sil = -1
        
        dim_comparison[str(dim)] = {
            "single_layer_sil": round(s_sil, 4),
            "trajectory_sil": round(t_sil, 4),
            "trajectory_advantage": round(t_sil / max(s_sil, 1e-6), 2),
        }
        print(f"  dim={dim}: single={s_sil:.4f}, traj={t_sil:.4f}, adv={t_sil/max(s_sil,1e-6):.2f}x")
    
    # --- 方法2: 轨迹Fourier谱分析 ---
    # 对轨迹做FFT, 提取频域特征
    # 每个词的轨迹: [n_layers+1, d_model]
    # 对d_model维度做PCA到50维, 然后对layer维度做FFT
    try:
        # 先对d_model降维
        all_traj_stacked = np.concatenate([all_trajectories[w] for w in words], axis=0)  # [n_words*(n_layers+1), d_model]
        pca_d = PCA(n_components=50)
        all_traj_pca = pca_d.fit_transform(all_traj_stacked)  # [n_words*(n_layers+1), 50]
        
        # 重塑为 [n_words, n_layers+1, 50]
        traj_pca = all_traj_pca.reshape(n_words, n_layers + 1, 50)
        
        # 对layer维度做FFT
        traj_fft = np.abs(np.fft.rfft(traj_pca, axis=1))  # [n_words, freq_bins, 50]
        traj_fft_flat = traj_fft.reshape(n_words, -1)  # [n_words, freq_bins*50]
        
        # PCA降维后计算silhouette
        n_comp_f = min(100, min(traj_fft_flat.shape) - 1)
        pca_f = PCA(n_components=n_comp_f)
        fft_reduced = pca_f.fit_transform(traj_fft_flat)
        fft_sil = silhouette_score(fft_reduced, labels)
        
        print(f"  Fourier spectrum PCA-{n_comp_f}: sil={fft_sil:.4f}")
    except Exception as e:
        fft_sil = -1
        print(f"  Fourier analysis failed: {e}")
    
    # --- 方法3: 层间增量(Δh)分析 ---
    # Δh_l = h_{l+1} - h_l, 只保留增量信息
    try:
        all_deltas = {}
        for w in words:
            deltas = np.diff(all_trajectories[w], axis=0)  # [n_layers, d_model]
            all_deltas[w] = deltas
        
        # 增量展开
        delta_flat = np.array([all_deltas[w].flatten() for w in words])  # [n_words, n_layers*d_model]
        
        n_comp_d = min(100, min(delta_flat.shape) - 1)
        pca_d = PCA(n_components=n_comp_d)
        delta_reduced = pca_d.fit_transform(delta_flat)
        delta_sil = silhouette_score(delta_reduced, labels)
        
        print(f"  Layer-delta PCA-{n_comp_d}: sil={delta_sil:.4f}")
    except Exception as e:
        delta_sil = -1
        print(f"  Delta analysis failed: {e}")
    
    # --- 方法4: 每层PCA10维拼接 ---
    try:
        per_layer_pca10 = []
        for li in range(n_layers + 1):
            single_data = np.array([all_trajectories[w][li] for w in words])
            pca_li = PCA(n_components=min(10, min(single_data.shape) - 1))
            reduced_li = pca_li.fit_transform(single_data)
            per_layer_pca10.append(reduced_li)
        
        concat_pca = np.concatenate(per_layer_pca10, axis=1)  # [n_words, (n_layers+1)*10]
        
        # 再PCA到100维
        n_comp_c = min(100, min(concat_pca.shape) - 1)
        pca_c = PCA(n_components=n_comp_c)
        concat_reduced = pca_c.fit_transform(concat_pca)
        concat_sil = silhouette_score(concat_reduced, labels)
        
        print(f"  Per-layer-PCA10 concat → PCA-{n_comp_c}: sil={concat_sil:.4f}")
    except Exception as e:
        concat_sil = -1
        print(f"  Concat analysis failed: {e}")
    
    return {
        "n_words": n_words,
        "best_single_layer": best_single_layer,
        "best_single_sil": round(best_single_sil, 4),
        "traj_sil_same_dim": round(traj_sil_100, 4),
        "fourier_sil": round(fft_sil, 4),
        "delta_sil": round(delta_sil, 4),
        "concat_pca10_sil": round(concat_sil, 4),
        "dim_comparison": dim_comparison,
        "single_layer_sils_top5": dict(sorted(single_layer_sils.items(), key=lambda x: -x[1])[:5]),
    }


# ============================================================
# Exp 2: 参数激活图 G(x)
# ============================================================

def exp2_parameter_activation_graph(model, tokenizer, device, model_info):
    """
    构建输入x的参数激活图 G(x)
    
    G(x) = {
        attn_heads: { (layer, head): activation_weight },
        mlp_neurons: { (layer, neuron): activation_value },
        residual_channels: { (layer, channel): magnitude }
    }
    
    核心思想: 语义不是向量距离, 而是参数激活拓扑
    """
    print("\n" + "="*60)
    print("Exp 2: 参数激活图 G(x)")
    print("="*60)
    
    n_layers = model_info.n_layers
    n_heads = get_n_heads(model)
    d_model = model_info.d_model
    intermediate_size = get_intermediate_size(model, model_info)
    layers = get_layers(model)
    
    # 采样层: 前1/4, 中1/2, 后3/4
    target_layers = sorted(set([
        0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1
    ] + list(range(0, n_layers, max(1, n_layers // 8)))))
    
    print(f"  Target layers: {target_layers}")
    print(f"  n_heads={n_heads}, d_model={d_model}, intermediate={intermediate_size}")
    
    # 收集激活图
    activation_graphs = {}  # word -> activation_data
    
    all_words = []
    for cat, words in SEMANTIC_CATEGORIES.items():
        for w in words:
            all_words.append((w, cat))
    
    for idx, (word, cat) in enumerate(all_words):
        if idx % 20 == 0:
            print(f"  Processing {idx}/{len(all_words)}: {word}")
        
        prompt = f"The {word}"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        # Hook收集
        captured_attn = {}   # (layer, head) -> attention_weight
        captured_mlp = {}    # (layer, neuron_idx) -> activation
        captured_residual = {}  # (layer, channel) -> magnitude
        captured_attn_pattern = {}  # layer -> attention_pattern
        
        hooks = []
        
        for li in target_layers:
            layer = layers[li]
            
            # 1. Attention hooks: 收集注意力权重
            def make_attn_hook(layer_idx):
                def hook(module, input, output):
                    # output_attentions=True时, output[1]是attention weights
                    # [batch, n_heads, seq_len, seq_len]
                    if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                        attn_w = output[1][0, :, -1, :].float().cpu().numpy()  # [n_heads, seq_len]
                        # 每个head对最后一个token的关注分布
                        for h in range(attn_w.shape[0]):
                            # 用entropy衡量head的"选择性"
                            attn_ent = -np.sum(attn_w[h] * np.log(attn_w[h] + 1e-10))
                            max_attn = float(np.max(attn_w[h]))
                            captured_attn[(layer_idx, h)] = {
                                "entropy": float(attn_ent),
                                "max_weight": max_attn,
                                "focus_position": int(np.argmax(attn_w[h])),
                            }
                return hook
            
            h = layer.self_attn.register_forward_hook(make_attn_hook(li))
            hooks.append(h)
            
            # 2. MLP hooks: 收集MLP激活
            # 我们用down_proj的输入(=SiLU(gate) * up)来获取neuron激活
            def make_mlp_hook(layer_idx, inter_size):
                captured_pre_down = {}
                def pre_down_hook(module, input, output):
                    # input[0] = [batch, seq_len, intermediate_size]
                    if isinstance(input, tuple):
                        pre_down = input[0][0, -1, :].float().cpu().numpy()  # [intermediate]
                        if not (np.isnan(pre_down).any() or np.isinf(pre_down).any()):
                            captured_pre_down["val"] = pre_down
                    return output
                return pre_down_hook, captured_pre_down
            
            mlp = layer.mlp
            if hasattr(mlp, 'down_proj'):
                pre_down_hook, captured_pre_down = make_mlp_hook(li, intermediate_size)
                h2 = mlp.down_proj.register_forward_hook(pre_down_hook)
                hooks.append(h2)
                # 存储引用以便后续读取
                if not hasattr(exp2_parameter_activation_graph, '_mlp_captures'):
                    exp2_parameter_activation_graph._mlp_captures = {}
                exp2_parameter_activation_graph._mlp_captures[li] = captured_pre_down
        
        # Forward pass with output_attentions
        with torch.no_grad():
            try:
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True, output_attentions=True)
            except Exception as e:
                for h in hooks:
                    h.remove()
                continue
        
        # 收集hidden states用于residual channel分析
        hs = out.hidden_states
        for li_idx, li in enumerate(target_layers):
            # Residual channel magnitudes
            h_state = hs[li][0, -1, :].float().cpu().numpy()  # [d_model]
            if not (np.isnan(h_state).any() or np.isinf(h_state).any()):
                # 记录top-k channel
                top_k = min(50, d_model)
                top_indices = np.argsort(np.abs(h_state))[-top_k:]
                for ch in top_indices:
                    captured_residual[(li, int(ch))] = float(h_state[ch])
        
        # 从forward输出直接获取attention patterns (更可靠)
        if hasattr(out, 'attentions') and out.attentions is not None:
            for li_idx, li in enumerate(target_layers):
                if li_idx < len(out.attentions) and out.attentions[li_idx] is not None:
                    attn_w = out.attentions[li_idx][0, :, -1, :].float().cpu().numpy()  # [n_heads, seq_len]
                    for h in range(min(n_heads, attn_w.shape[0])):
                        attn_ent = -np.sum(attn_w[h] * np.log(attn_w[h] + 1e-10))
                        max_attn = float(np.max(attn_w[h]))
                        captured_attn[(li, h)] = {
                            "entropy": float(attn_ent),
                            "max_weight": max_attn,
                            "focus_position": int(np.argmax(attn_w[h])),
                        }
        
        # 读取MLP pre-down activations
        for li in target_layers:
            if hasattr(exp2_parameter_activation_graph, '_mlp_captures') and li in exp2_parameter_activation_graph._mlp_captures:
                pre_down = exp2_parameter_activation_graph._mlp_captures[li].get("val")
                if pre_down is not None:
                    # 取top-k activated neurons
                    top_k = min(200, len(pre_down))
                    top_indices = np.argsort(np.abs(pre_down))[-top_k:]
                    for ni in top_indices:
                        captured_mlp[(li, int(ni))] = float(pre_down[ni])
        
        for h in hooks:
            h.remove()
        
        # 存储激活图
        activation_graphs[word] = {
            "category": cat,
            "attn_heads": {f"L{k[0]}_H{k[1]}": v for k, v in captured_attn.items()},
            "mlp_neurons": {f"L{k[0]}_N{k[1]}": round(v, 4) for k, v in captured_mlp.items()},
            "residual_channels": {f"L{k[0]}_C{k[1]}": round(v, 4) for k, v in captured_residual.items()},
            "n_attn_entries": len(captured_attn),
            "n_mlp_entries": len(captured_mlp),
            "n_residual_entries": len(captured_residual),
        }
    
    # 汇总统计
    n_graphs = len(activation_graphs)
    avg_attn = np.mean([g["n_attn_entries"] for g in activation_graphs.values()])
    avg_mlp = np.mean([g["n_mlp_entries"] for g in activation_graphs.values()])
    avg_res = np.mean([g["n_residual_entries"] for g in activation_graphs.values()])
    
    print(f"  Collected {n_graphs} activation graphs")
    print(f"  Avg entries: attn={avg_attn:.0f}, mlp={avg_mlp:.0f}, residual={avg_res:.0f}")
    
    return {
        "n_graphs": n_graphs,
        "target_layers": target_layers,
        "avg_attn_entries": round(avg_attn, 1),
        "avg_mlp_entries": round(avg_mlp, 1),
        "avg_residual_entries": round(avg_res, 1),
        "activation_graphs": activation_graphs,
    }


# ============================================================
# Exp 3: 概念拓扑邻接 — 激活图重叠 vs 欧氏距离
# ============================================================

def exp3_concept_topology_adjacency(model, tokenizer, device, model_info, activation_graphs):
    """
    核心问题: 激活图重叠 vs 欧氏距离, 谁更好预测语义相似性?
    
    方法:
    1. 计算所有词对的激活图Jaccard重叠
    2. 计算所有词对的cosine距离
    3. 比较哪种距离更与语义类别一致
    """
    print("\n" + "="*60)
    print("Exp 3: 概念拓扑邻接 — 激活图重叠 vs 欧氏距离")
    print("="*60)
    
    if not activation_graphs or len(activation_graphs) < 10:
        return {"error": "insufficient activation graphs"}
    
    words = list(activation_graphs.keys())
    labels = {w: activation_graphs[w]["category"] for w in words}
    n = len(words)
    
    # --- 1. 激活图Jaccard重叠 ---
    def jaccard_overlap(set1, set2):
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / max(union, 1)
    
    # 构建每词的激活集合
    attn_sets = {}
    mlp_sets = {}
    combined_sets = {}
    
    for w in words:
        g = activation_graphs[w]
        attn_sets[w] = set(g["attn_heads"].keys())
        mlp_sets[w] = set(g["mlp_neurons"].keys())
        combined_sets[w] = attn_sets[w] | mlp_sets[w]
    
    # 计算词对的重叠和距离
    same_cat_jaccard_attn = []
    diff_cat_jaccard_attn = []
    same_cat_jaccard_mlp = []
    diff_cat_jaccard_mlp = []
    same_cat_jaccard_combined = []
    diff_cat_jaccard_combined = []
    same_cat_cosine = []
    diff_cat_cosine = []
    
    # 收集hidden states用于cosine距离
    hidden_states = {}
    for w in words:
        prompt = f"The {w}"
        try:
            traj = get_all_hidden_states(model, tokenizer, device, prompt)
            if not (np.isnan(traj).any() or np.isinf(traj).any()):
                hidden_states[w] = traj
        except Exception:
            pass
    
    # 计算所有词对 (采样避免O(n^2)过大)
    import random
    random.seed(42)
    
    max_pairs = min(2000, n * (n - 1) // 2)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((words[i], words[j]))
    
    if len(pairs) > max_pairs:
        pairs = random.sample(pairs, max_pairs)
    
    print(f"  Computing {len(pairs)} word pairs...")
    
    for w1, w2 in pairs:
        same_cat = labels[w1] == labels[w2]
        
        # Jaccard重叠
        j_attn = jaccard_overlap(attn_sets[w1], attn_sets[w2])
        j_mlp = jaccard_overlap(mlp_sets[w1], mlp_sets[w2])
        j_combined = jaccard_overlap(combined_sets[w1], combined_sets[w2])
        
        if same_cat:
            same_cat_jaccard_attn.append(j_attn)
            same_cat_jaccard_mlp.append(j_mlp)
            same_cat_jaccard_combined.append(j_combined)
        else:
            diff_cat_jaccard_attn.append(j_attn)
            diff_cat_jaccard_mlp.append(j_mlp)
            diff_cat_jaccard_combined.append(j_combined)
        
        # Cosine距离 (用中间层)
        if w1 in hidden_states and w2 in hidden_states:
            mid_layer = model_info.n_layers // 2
            h1 = hidden_states[w1][mid_layer]
            h2 = hidden_states[w2][mid_layer]
            cos = float(np.dot(h1, h2) / (np.linalg.norm(h1) * np.linalg.norm(h2) + 1e-10))
            if same_cat:
                same_cat_cosine.append(cos)
            else:
                diff_cat_cosine.append(cos)
    
    # 统计
    def safe_mean(lst):
        return round(np.mean(lst), 4) if lst else 0
    
    def safe_std(lst):
        return round(np.std(lst), 4) if lst else 0
    
    results = {
        "n_same_cat_pairs": len(same_cat_jaccard_attn),
        "n_diff_cat_pairs": len(diff_cat_jaccard_attn),
        "jaccard_attn": {
            "same_cat_mean": safe_mean(same_cat_jaccard_attn),
            "diff_cat_mean": safe_mean(diff_cat_jaccard_attn),
            "discrimination": round(safe_mean(same_cat_jaccard_attn) - safe_mean(diff_cat_jaccard_attn), 4),
        },
        "jaccard_mlp": {
            "same_cat_mean": safe_mean(same_cat_jaccard_mlp),
            "diff_cat_mean": safe_mean(diff_cat_jaccard_mlp),
            "discrimination": round(safe_mean(same_cat_jaccard_mlp) - safe_mean(diff_cat_jaccard_mlp), 4),
        },
        "jaccard_combined": {
            "same_cat_mean": safe_mean(same_cat_jaccard_combined),
            "diff_cat_mean": safe_mean(diff_cat_jaccard_combined),
            "discrimination": round(safe_mean(same_cat_jaccard_combined) - safe_mean(diff_cat_jaccard_combined), 4),
        },
        "cosine_mid_layer": {
            "same_cat_mean": safe_mean(same_cat_cosine),
            "diff_cat_mean": safe_mean(diff_cat_cosine),
            "discrimination": round(safe_mean(same_cat_cosine) - safe_mean(diff_cat_cosine), 4),
        },
    }
    
    # 哪种度量更好区分语义类别?
    print(f"  Jaccard-attn discrimination: {results['jaccard_attn']['discrimination']:.4f}")
    print(f"  Jaccard-mlp discrimination: {results['jaccard_mlp']['discrimination']:.4f}")
    print(f"  Jaccard-combined discrimination: {results['jaccard_combined']['discrimination']:.4f}")
    print(f"  Cosine discrimination: {results['cosine_mid_layer']['discrimination']:.4f}")
    
    # --- 额外: 每个类别的激活图特征 ---
    cat_attn_patterns = defaultdict(list)
    cat_mlp_patterns = defaultdict(list)
    
    for w in words:
        cat = labels[w]
        g = activation_graphs[w]
        cat_attn_patterns[cat].append(set(g["attn_heads"].keys()))
        cat_mlp_patterns[cat].append(set(g["mlp_neurons"].keys()))
    
    # 每类别内部的平均Jaccard重叠
    cat_intra_overlap = {}
    for cat in cat_attn_patterns:
        attn_sets_cat = cat_attn_patterns[cat]
        mlp_sets_cat = cat_mlp_patterns[cat]
        
        # 类内Jaccard
        intra_j = []
        for i in range(min(len(attn_sets_cat), 20)):
            for j in range(i + 1, min(len(attn_sets_cat), 20)):
                intra_j.append(jaccard_overlap(attn_sets_cat[i], attn_sets_cat[j]))
        
        cat_intra_overlap[cat] = round(np.mean(intra_j), 4) if intra_j else 0
    
    results["category_intra_overlap"] = cat_intra_overlap
    
    return results


# ============================================================
# Exp 4: 组合语义(属性绑定)
# ============================================================

def exp4_compositional_semantics(model, tokenizer, device, model_info):
    """
    "red apple" vs "green apple" vs "big apple"
    属性如何改变激活图? 是否存在"核心名词"子图?
    """
    print("\n" + "="*60)
    print("Exp 4: 组合语义 — 属性绑定")
    print("="*60)
    
    n_layers = model_info.n_layers
    n_heads = get_n_heads(model)
    d_model = model_info.d_model
    intermediate_size = get_intermediate_size(model, model_info)
    layers = get_layers(model)
    
    target_layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4]
    
    def get_activation_signature(prompt):
        """获取一个prompt的激活签名: MLP top neurons + attention patterns"""
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        mlp_acts = {}
        attn_patterns = {}
        
        hooks = []
        mlp_captures = {}
        
        for li in target_layers:
            # MLP hook
            def make_mlp_h(layer_idx):
                cap = {}
                def hook(module, input, output):
                    if isinstance(input, tuple):
                        pre_down = input[0][0, -1, :].float().cpu().numpy()
                        if not (np.isnan(pre_down).any() or np.isinf(pre_down).any()):
                            cap["val"] = pre_down
                    return output
                return hook, cap
            
            h, cap = make_mlp_h(li)
            if hasattr(layers[li].mlp, 'down_proj'):
                hooks.append(layers[li].mlp.down_proj.register_forward_hook(h))
                mlp_captures[li] = cap
        
        with torch.no_grad():
            try:
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True, output_attentions=True)
            except Exception:
                for h in hooks: h.remove()
                return None
        
        for h in hooks:
            h.remove()
        
        # 收集MLP top neurons
        for li in target_layers:
            if li in mlp_captures and "val" in mlp_captures[li]:
                pre_down = mlp_captures[li]["val"]
                top_k = min(100, len(pre_down))
                top_idx = set(np.argsort(np.abs(pre_down))[-top_k:])
                mlp_acts[li] = top_idx
        
        # 收集attention patterns
        if hasattr(out, 'attentions') and out.attentions is not None:
            for li_idx, li in enumerate(target_layers):
                if li_idx < len(out.attentions) and out.attentions[li_idx] is not None:
                    attn_w = out.attentions[li_idx][0, :, -1, :].float().cpu().numpy()
                    # 每个head的focus position
                    head_focus = {}
                    for h in range(min(n_heads, attn_w.shape[0])):
                        head_focus[h] = int(np.argmax(attn_w[h]))
                    attn_patterns[li] = head_focus
        
        # 收集hidden states
        hs = out.hidden_states
        mid_hs = hs[n_layers // 2][0, -1, :].float().cpu().numpy()
        
        return {
            "mlp_top_neurons": mlp_acts,
            "attn_patterns": attn_patterns,
            "mid_hidden": mid_hs,
        }
    
    # 分析组合语义
    results = {}
    
    # 按核心名词分组
    noun_groups = defaultdict(list)
    for phrase, cat in COMPOSITIONAL_PAIRS:
        # 提取名词
        parts = phrase.split()
        if len(parts) >= 2:
            noun = parts[-1]  # 最后一个词作为名词
            noun_groups[noun].append((phrase, cat))
    
    print(f"  Noun groups: {list(noun_groups.keys())}")
    
    for noun, phrases in noun_groups.items():
        print(f"\n  Analyzing noun: {noun} ({len(phrases)} phrases)")
        
        # 收集所有phrase的激活签名
        sigs = {}
        for phrase, cat in phrases:
            sig = get_activation_signature(phrase)
            if sig is not None:
                sigs[phrase] = sig
        
        if len(sigs) < 2:
            continue
        
        # 计算名词内部的重叠
        phrase_list = list(sigs.keys())
        mlp_overlaps = []
        attn_overlaps = []
        cosine_sims = []
        
        for i in range(len(phrase_list)):
            for j in range(i + 1, len(phrase_list)):
                p1, p2 = phrase_list[i], phrase_list[j]
                s1, s2 = sigs[p1], sigs[p2]
                
                # MLP neuron overlap
                for li in target_layers:
                    if li in s1["mlp_top_neurons"] and li in s2["mlp_top_neurons"]:
                        set1 = s1["mlp_top_neurons"][li]
                        set2 = s2["mlp_top_neurons"][li]
                        if set1 and set2:
                            overlap = len(set1 & set2) / len(set1 | set2)
                            mlp_overlaps.append(overlap)
                
                # Attention pattern overlap
                for li in target_layers:
                    if li in s1["attn_patterns"] and li in s2["attn_patterns"]:
                        ap1 = s1["attn_patterns"][li]
                        ap2 = s2["attn_patterns"][li]
                        same_focus = sum(1 for h in ap1 if h in ap2 and ap1[h] == ap2[h])
                        total = len(set(ap1.keys()) | set(ap2.keys()))
                        if total > 0:
                            attn_overlaps.append(same_focus / total)
                
                # Cosine similarity
                h1 = s1["mid_hidden"]
                h2 = s2["mid_hidden"]
                cos = float(np.dot(h1, h2) / (np.linalg.norm(h1) * np.linalg.norm(h2) + 1e-10))
                cosine_sims.append(cos)
        
        results[noun] = {
            "n_phrases": len(sigs),
            "mlp_overlap_mean": round(np.mean(mlp_overlaps), 4) if mlp_overlaps else 0,
            "attn_overlap_mean": round(np.mean(attn_overlaps), 4) if attn_overlaps else 0,
            "cosine_sim_mean": round(np.mean(cosine_sims), 4) if cosine_sims else 0,
        }
        print(f"    MLP overlap: {results[noun]['mlp_overlap_mean']:.4f}")
        print(f"    Attn overlap: {results[noun]['attn_overlap_mean']:.4f}")
        print(f"    Cosine sim: {results[noun]['cosine_sim_mean']:.4f}")
    
    # 跨名词对比: 同属性不同名词
    print("\n  Cross-noun comparison (same attribute, different noun)...")
    cross_noun_results = {}
    
    # 对比: "red apple" vs "red car" vs "red house"
    attr_groups = defaultdict(list)
    for phrase, cat in COMPOSITIONAL_PAIRS:
        parts = phrase.split()
        if len(parts) >= 2:
            attr = parts[0]
            attr_groups[attr].append(phrase)
    
    for attr, phrases in attr_groups.items():
        if len(phrases) < 2:
            continue
        
        sigs = {}
        for phrase in phrases:
            sig = get_activation_signature(phrase)
            if sig is not None:
                sigs[phrase] = sig
        
        if len(sigs) < 2:
            continue
        
        phrase_list = list(sigs.keys())
        mlp_overlaps = []
        cosine_sims = []
        
        for i in range(len(phrase_list)):
            for j in range(i + 1, len(phrase_list)):
                s1, s2 = sigs[phrase_list[i]], sigs[phrase_list[j]]
                for li in target_layers:
                    if li in s1["mlp_top_neurons"] and li in s2["mlp_top_neurons"]:
                        set1, set2 = s1["mlp_top_neurons"][li], s2["mlp_top_neurons"][li]
                        if set1 and set2:
                            mlp_overlaps.append(len(set1 & set2) / len(set1 | set2))
                h1, h2 = s1["mid_hidden"], s2["mid_hidden"]
                cos = float(np.dot(h1, h2) / (np.linalg.norm(h1) * np.linalg.norm(h2) + 1e-10))
                cosine_sims.append(cos)
        
        cross_noun_results[attr] = {
            "n_phrases": len(sigs),
            "mlp_overlap_mean": round(np.mean(mlp_overlaps), 4) if mlp_overlaps else 0,
            "cosine_sim_mean": round(np.mean(cosine_sims), 4) if cosine_sims else 0,
        }
        print(f"    {attr}: MLP overlap={cross_noun_results[attr]['mlp_overlap_mean']:.4f}, "
              f"COS={cross_noun_results[attr]['cosine_sim_mean']:.4f}")
    
    results["cross_noun"] = cross_noun_results
    
    return results


# ============================================================
# Exp 5: 语法绑定 — 同token不同回路
# ============================================================

def exp5_syntactic_binding(model, tokenizer, device, model_info):
    """
    "dog bites man" vs "man bites dog"
    同token, 不同语序 → 是否激活不同参数回路?
    
    这是最关键的实验: 直接测试"语义=条件计算"
    """
    print("\n" + "="*60)
    print("Exp 5: 语法绑定 — 同token不同回路")
    print("="*60)
    
    n_layers = model_info.n_layers
    n_heads = get_n_heads(model)
    d_model = model_info.d_model
    intermediate_size = get_intermediate_size(model, model_info)
    layers = get_layers(model)
    
    target_layers = sorted(set([0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]))
    
    def get_full_activation_signature(prompt):
        """获取完整的激活签名"""
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        mlp_acts = {}   # layer -> set of top neuron indices
        attn_patterns = {}  # layer -> {head: focus_position}
        hidden_states_all = {}  # layer -> [d_model]
        
        hooks = []
        mlp_captures = {}
        
        for li in target_layers:
            # MLP hook
            def make_mlp_h(layer_idx):
                cap = {}
                def hook(module, input, output):
                    if isinstance(input, tuple):
                        pre_down = input[0][0, -1, :].float().cpu().numpy()
                        if not (np.isnan(pre_down).any() or np.isinf(pre_down).any()):
                            cap["val"] = pre_down
                    return output
                return hook, cap
            
            h, cap = make_mlp_h(li)
            if hasattr(layers[li].mlp, 'down_proj'):
                hooks.append(layers[li].mlp.down_proj.register_forward_hook(h))
                mlp_captures[li] = cap
        
        with torch.no_grad():
            try:
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True, output_attentions=True)
            except Exception:
                for h in hooks: h.remove()
                return None
        
        for h in hooks:
            h.remove()
        
        # MLP top neurons
        for li in target_layers:
            if li in mlp_captures and "val" in mlp_captures[li]:
                pre_down = mlp_captures[li]["val"]
                top_k = min(200, len(pre_down))
                top_idx = set(np.argsort(np.abs(pre_down))[-top_k:].tolist())
                mlp_acts[li] = top_idx
        
        # Attention patterns
        if hasattr(out, 'attentions') and out.attentions is not None:
            for li_idx, li in enumerate(target_layers):
                if li_idx < len(out.attentions) and out.attentions[li_idx] is not None:
                    attn_w = out.attentions[li_idx][0, :, -1, :].float().cpu().numpy()
                    head_focus = {}
                    for h in range(min(n_heads, attn_w.shape[0])):
                        head_focus[h] = {
                            "focus_pos": int(np.argmax(attn_w[h])),
                            "max_weight": float(np.max(attn_w[h])),
                            "entropy": float(-np.sum(attn_w[h] * np.log(attn_w[h] + 1e-10))),
                        }
                    attn_patterns[li] = head_focus
        
        # Hidden states at all target layers
        hs = out.hidden_states
        for li in target_layers:
            h_state = hs[li][0, -1, :].float().cpu().numpy()
            if not (np.isnan(h_state).any() or np.isinf(h_state).any()):
                hidden_states_all[li] = h_state
        
        return {
            "mlp_acts": mlp_acts,
            "attn_patterns": attn_patterns,
            "hidden_states": hidden_states_all,
        }
    
    # 分析每对语法变体
    pair_results = []
    
    for s1, s2 in SYNTACTIC_PAIRS:
        print(f"\n  Comparing: '{s1}' vs '{s2}'")
        
        sig1 = get_full_activation_signature(s1)
        sig2 = get_full_activation_signature(s2)
        
        if sig1 is None or sig2 is None:
            print(f"    Failed to get signatures")
            continue
        
        # --- 1. MLP neuron overlap ---
        mlp_overlaps = {}
        for li in target_layers:
            if li in sig1["mlp_acts"] and li in sig2["mlp_acts"]:
                set1, set2 = sig1["mlp_acts"][li], sig2["mlp_acts"][li]
                if set1 and set2:
                    jaccard = len(set1 & set2) / len(set1 | set2)
                    mlp_overlaps[li] = round(jaccard, 4)
        
        # --- 2. Attention pattern difference ---
        attn_diffs = {}
        for li in target_layers:
            if li in sig1["attn_patterns"] and li in sig2["attn_patterns"]:
                ap1, ap2 = sig1["attn_patterns"][li], sig2["attn_patterns"][li]
                n_changed = sum(1 for h in ap1 if h in ap2 and ap1[h]["focus_pos"] != ap2[h]["focus_pos"])
                n_total = len(set(ap1.keys()) & set(ap2.keys()))
                if n_total > 0:
                    attn_diffs[li] = round(n_changed / n_total, 4)
        
        # --- 3. Hidden state cosine at each layer ---
        layer_cosines = {}
        for li in target_layers:
            if li in sig1["hidden_states"] and li in sig2["hidden_states"]:
                h1, h2 = sig1["hidden_states"][li], sig2["hidden_states"][li]
                cos = float(np.dot(h1, h2) / (np.linalg.norm(h1) * np.linalg.norm(h2) + 1e-10))
                layer_cosines[li] = round(cos, 4)
        
        # --- 4. 层间增量差异 ---
        delta_diffs = {}
        hs1 = sig1["hidden_states"]
        hs2 = sig2["hidden_states"]
        sorted_layers = sorted([l for l in hs1.keys() if l in hs2])
        for i in range(len(sorted_layers) - 1):
            l1, l2 = sorted_layers[i], sorted_layers[i + 1]
            delta1 = hs1[l2] - hs1[l1]
            delta2 = hs2[l2] - hs2[l1]
            cos_delta = float(np.dot(delta1, delta2) / (np.linalg.norm(delta1) * np.linalg.norm(delta2) + 1e-10))
            delta_diffs[f"L{l1}_L{l2}"] = round(cos_delta, 4)
        
        pair_result = {
            "pair": (s1, s2),
            "mlp_overlap": mlp_overlaps,
            "attn_change_ratio": attn_diffs,
            "layer_cosine": layer_cosines,
            "delta_cosine": delta_diffs,
        }
        pair_results.append(pair_result)
        
        # 打印关键指标
        print(f"    MLP overlap: {mlp_overlaps}")
        print(f"    Attn change: {attn_diffs}")
        print(f"    Mid-layer cos: {layer_cosines.get(n_layers // 2, 'N/A')}")
    
    # 汇总
    if not pair_results:
        return {"error": "no valid pairs"}
    
    # 主宾互换 vs 被动 vs 否定的差异
    swap_pairs = [r for r in pair_results if any(k in r["pair"][0] for k in ["bites", "chases", "praises", "helps", "arrests"])]
    
    avg_mlp_overlap = np.mean([np.mean(list(r["mlp_overlap"].values())) for r in pair_results if r["mlp_overlap"]])
    avg_attn_change = np.mean([np.mean(list(r["attn_change_ratio"].values())) for r in pair_results if r["attn_change_ratio"]])
    avg_mid_cos = np.mean([list(r["layer_cosine"].values())[len(r["layer_cosine"])//2] for r in pair_results if r["layer_cosine"]])
    
    print(f"\n  === Summary ===")
    print(f"  Avg MLP overlap: {avg_mlp_overlap:.4f}")
    print(f"  Avg Attn change: {avg_attn_change:.4f}")
    print(f"  Avg mid-layer cosine: {avg_mid_cos:.4f}")
    
    return {
        "n_pairs": len(pair_results),
        "avg_mlp_overlap": round(avg_mlp_overlap, 4),
        "avg_attn_change_ratio": round(avg_attn_change, 4),
        "avg_mid_layer_cosine": round(avg_mid_cos, 4),
        "pair_details": pair_results,
    }


# ============================================================
# 主函数
# ============================================================

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    
    print(f"\n{'#'*60}")
    print(f"Phase 128: 参数响应拓扑分析 — {model_name}")
    print(f"{'#'*60}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    
    print(f"Model: {model_info.model_class}, layers={model_info.n_layers}, d_model={model_info.d_model}")
    
    results = {
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
        # Exp 1: 维度控制轨迹比较
        results["exp1_dim_controlled"] = exp1_dimension_controlled_comparison(
            model, tokenizer, device, model_info
        )
        gc.collect()
        torch.cuda.empty_cache()
        
        # Exp 2: 参数激活图
        exp2_results = exp2_parameter_activation_graph(
            model, tokenizer, device, model_info
        )
        results["exp2_activation_graph"] = {
            k: v for k, v in exp2_results.items() if k != "activation_graphs"
        }
        gc.collect()
        torch.cuda.empty_cache()
        
        # Exp 3: 概念拓扑邻接
        activation_graphs = exp2_results.get("activation_graphs", {})
        results["exp3_topology_adjacency"] = exp3_concept_topology_adjacency(
            model, tokenizer, device, model_info, activation_graphs
        )
        gc.collect()
        torch.cuda.empty_cache()
        
        # Exp 4: 组合语义
        results["exp4_compositional"] = exp4_compositional_semantics(
            model, tokenizer, device, model_info
        )
        gc.collect()
        torch.cuda.empty_cache()
        
        # Exp 5: 语法绑定
        results["exp5_syntactic"] = exp5_syntactic_binding(
            model, tokenizer, device, model_info
        )
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        results["error"] = str(e)
    
    finally:
        release_model(model)
    
    # 保存结果
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'glm5_temp')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"phase128_{model_name}_param_topology.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\nResults saved to: {output_path}")
    print(f"Done!")


if __name__ == "__main__":
    main()
