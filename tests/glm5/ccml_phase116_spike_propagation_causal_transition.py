"""
Phase 116: Spike传播动力学与因果过渡点
==========================================
核心目标 (基于Phase 115的关键发现):

Phase 115发现: L12的4维spike翻译特异性极强(z=29.9)但无因果效应,
而L35的14维spike因果效应极强(+2.089)但不翻译配对特异(AMBIGUOUS)。

核心问题: "状态协调码"(L12)如何变成"输出决策码"(L35)?

实验设计:

Exp 1: Layer-by-Layer Causal Sweep (逐层因果扫描)
   - 对所有36层做spike子空间干预(remove), 找到因果过渡点
   - Phase 115只测了4层(L12/L18/L27/L35), 遗漏了关键过渡区
   - 预期: 因果力从某层开始急剧上升, 形成相变

Exp 2: Spike Subspace Continuity (spike子空间连续性)
   - L12的4维spike子空间 vs L15/L18/L21/.../L35的spike子空间
   - 计算子空间重叠(subspace overlap / principal angles)
   - 如果L12 spike ⊂ L35 spike → 信息在子空间内传播
   - 如果L12 spike ⊄ L35 spike → 信息被变换/旋转

Exp 3: Spike Coefficient Decodability (spike系数可解码性)
   - 在每层的spike子空间中, 投影系数能否预测目标英文词?
   - L12: 如果系数不能预测目标词 → 确认是"状态码"(不含词义信息)
   - L35: 如果系数能预测目标词 → 确认是"决策码"(含词义信息)
   - 方法: 简单linear probe (kNN / logistic regression)

Exp 4: Inter-Layer Spike Propagation (层间spike传播)
   - 从L12 spike到L15/L18/.../L35 spike的传播映射
   - L12 spike投影 → 能否预测L35 spike投影?
   - 传播是线性的还是非线性的?

理论纪律:
- 不使用"信息流"等泛化术语, 只报告可测量的统计量
- 不假设spike是"语义", 只报告可解码性
- "因果力"严格定义为: remove spike后目标词log_prob的变化量
- 区分"方向特异性"和"系数特异性"
"""

import os, sys, json, gc, time, argparse
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_utils import load_model, get_layers, get_model_info, release_model

# ============================================================
# 设置
# ============================================================
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'glm5_temp')
OUT_DIR = os.path.abspath(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# 翻译词对 — 扩大到120个以增加统计力
WORD_PAIRS = [
    ("猫", "cat"), ("狗", "dog"), ("鸟", "bird"), ("鱼", "fish"), ("马", "horse"),
    ("牛", "cow"), ("羊", "sheep"), ("猪", "pig"), ("鸡", "chicken"), ("鼠", "mouse"),
    ("水", "water"), ("火", "fire"), ("土", "earth"), ("风", "wind"), ("金", "gold"),
    ("木", "wood"), ("铁", "iron"), ("石", "stone"), ("沙", "sand"), ("冰", "ice"),
    ("月", "moon"), ("星", "star"), ("云", "cloud"), ("雨", "rain"), ("雪", "snow"),
    ("日", "sun"), ("天", "sky"), ("海", "sea"), ("河", "river"), ("山", "mountain"),
    ("红", "red"), ("蓝", "blue"), ("绿", "green"), ("白", "white"), ("黑", "black"),
    ("黄", "yellow"), ("紫", "purple"), ("灰", "gray"), ("棕", "brown"), ("粉", "pink"),
    ("手", "hand"), ("足", "foot"), ("目", "eye"), ("耳", "ear"), ("口", "mouth"),
    ("心", "heart"), ("头", "head"), ("骨", "bone"), ("血", "blood"), ("发", "hair"),
    ("父", "father"), ("母", "mother"), ("子", "son"), ("女", "daughter"), ("友", "friend"),
    ("王", "king"), ("师", "teacher"), ("医", "doctor"), ("兵", "soldier"), ("农", "farmer"),
    ("爱", "love"), ("恨", "hate"), ("善", "good"), ("恶", "evil"), ("真", "truth"),
    ("美", "beauty"), ("智", "wisdom"), ("力", "power"), ("光", "light"), ("影", "shadow"),
    ("走", "walk"), ("跑", "run"), ("飞", "fly"), ("吃", "eat"), ("喝", "drink"),
    ("看", "see"), ("听", "hear"), ("说", "say"), ("写", "write"), ("读", "read"),
    ("年", "year"), ("月", "month"), ("日", "day"), ("时", "hour"), ("分", "minute"),
    ("米", "rice"), ("茶", "tea"), ("酒", "wine"), ("肉", "meat"), ("盐", "salt"),
    ("车", "car"), ("船", "ship"), ("路", "road"), ("桥", "bridge"), ("门", "door"),
    ("花", "flower"), ("草", "grass"), ("树", "tree"), ("叶", "leaf"), ("根", "root"),
    ("喜", "joy"), ("怒", "anger"), ("哀", "sorrow"), ("惧", "fear"), ("思", "thought"),
    ("法", "law"), ("国", "country"), ("城", "city"), ("家", "home"), ("书", "book"),
    ("电", "electricity"), ("网", "network"), ("数", "number"), ("算", "compute"), ("器", "device"),
    ("湖", "lake"), ("岛", "island"), ("春", "spring"), ("夏", "summer"), ("秋", "autumn"),
    ("冬", "winter"), ("晨", "morning"), ("暮", "dusk"), ("雷", "thunder"), ("雾", "fog"),
    ("龙", "dragon"), ("蛇", "snake"), ("虎", "tiger"), ("鹿", "deer"), ("兔", "rabbit"),
    ("道", "way"), ("德", "virtue"), ("礼", "ritual"), ("义", "justice"), ("信", "trust"),
    ("剑", "sword"), ("笔", "pen"), ("琴", "lute"), ("画", "painting"), ("棋", "chess"),
]

# 去重
seen_zh = set()
UNIQUE_PAIRS = []
for zh, en in WORD_PAIRS:
    if zh not in seen_zh:
        seen_zh.add(zh)
        UNIQUE_PAIRS.append((zh, en))
WORD_PAIRS = UNIQUE_PAIRS[:120]

# 采样层 — 加密后期层采样
SAMPLE_LAYERS = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35]

# Phase 114已知各层信号维度 (Qwen3)
KNOWN_SIGNAL_DIMS = {0: 25, 3: 24, 6: 23, 9: 21, 12: 4, 15: 14, 18: 8, 21: 20, 24: 21, 27: 18, 30: 23, 33: 24, 35: 14}


# ============================================================
# 数据收集: MLP输出
# ============================================================
def collect_mlp_outputs(model, tokenizer, device, n_layers, word_pairs, batch_size=5):
    """收集MLP输出"""
    layers = get_layers(model)
    
    all_mlp = {'zh': defaultdict(list), 'trans': defaultdict(list)}
    
    for batch_start in range(0, len(word_pairs), batch_size):
        batch = word_pairs[batch_start:batch_start+batch_size]
        
        for zh, en in batch:
            zh_prompt = f"翻译以下中文词：{zh}"
            trans_prompt = f"Translate the following Chinese word: {zh}"
            
            for task, prompt in [('zh', zh_prompt), ('trans', trans_prompt)]:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                mlp_outs = {}
                hooks = []
                
                def make_mlp_hook(l):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            mlp_outs[l] = output[0][0, -1, :].detach().float().cpu().numpy()
                        else:
                            mlp_outs[l] = output[0, -1, :].detach().float().cpu().numpy()
                    return hook_fn
                
                for l, layer in enumerate(layers):
                    if hasattr(layer, 'mlp'):
                        hooks.append(layer.mlp.register_forward_hook(make_mlp_hook(l)))
                
                with torch.no_grad():
                    _ = model(inputs["input_ids"])
                
                for h in hooks:
                    h.remove()
                del inputs
                gc.collect()
                torch.cuda.empty_cache()
                
                for l in mlp_outs:
                    all_mlp[task][l].append(mlp_outs[l])
        
        print(f"  [collect_mlp] {min(batch_start+batch_size, len(word_pairs))}/{len(word_pairs)}")
    
    result = {}
    for task in ['zh', 'trans']:
        result[task] = {}
        for l in all_mlp[task]:
            result[task][l] = np.array(all_mlp[task][l])
    
    return result


# ============================================================
# MP分析 — 获取spike子空间
# ============================================================
def get_spike_subspace(diffs, k, N, P):
    """获取差分的top-k右奇异向量(spike子空间)"""
    diffs_centered = diffs - diffs.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(diffs_centered, full_matrices=False)
    k_actual = min(k, Vt.shape[0])
    return {
        'V': Vt[:k_actual, :].T,  # [P, k] 投影矩阵
        'S': S[:k_actual],
        'Vt': Vt[:k_actual, :],   # [k, P]
        'k': k_actual,
        'coefficients': U[:, :k_actual] * S[:k_actual],  # [N, k] 投影系数
        'mean_diff': diffs.mean(axis=0),
    }


# ============================================================
# Exp 1: Layer-by-Layer Causal Sweep — 逐层因果扫描
# ============================================================
def exp1_causal_sweep(model, tokenizer, device, n_layers, mlp_outs, d_model,
                       test_pairs=None, n_test=50):
    """
    逐层因果扫描: 对所有采样层做spike子空间remove干预
    
    Phase 115只测了4层, 这里对13层全部扫描, 找到因果过渡点
    
    每层:
    1. Baseline: 正常翻译, 记录目标词log_prob
    2. Remove spike: 减去spike子空间投影(完全移除)
    3. Remove random: 减去同维度随机子空间投影(控制)
    """
    print("\n" + "="*70)
    print("Exp 1: Layer-by-Layer Causal Sweep — 逐层因果扫描")
    print("="*70)
    
    if test_pairs is None:
        test_pairs = WORD_PAIRS[:n_test]
    
    layers_list = get_layers(model)
    
    results = {}
    
    for l in SAMPLE_LAYERS:
        if l >= len(layers_list):
            continue
        if l not in mlp_outs['zh'] or l not in mlp_outs['trans']:
            continue
        
        zh_data = mlp_outs['zh'][l]
        trans_data = mlp_outs['trans'][l]
        N = zh_data.shape[0]
        P = zh_data.shape[1]
        
        # 获取spike子空间
        diffs = trans_data - zh_data
        k = KNOWN_SIGNAL_DIMS.get(l, min(10, N-1))
        spike = get_spike_subspace(diffs, k, N, P)
        V = spike['V']  # [P, k]
        
        print(f"\n  L{l} (spike_dim={spike['k']}):")
        
        # 定义评估函数
        def evaluate_batch(pairs, hook_fn_factory=None):
            """批量评估翻译"""
            log_probs = []
            target_in_top10 = []
            
            for zh, en in pairs:
                trans_prompt = f"Translate the following Chinese word: {zh}"
                inputs = tokenizer(trans_prompt, return_tensors="pt").to(device)
                
                hooks = []
                if hook_fn_factory is not None:
                    hooks.append(layers_list[l].register_forward_hook(hook_fn_factory))
                
                with torch.no_grad():
                    outputs = model(inputs["input_ids"])
                    logits = outputs.logits[0, -1, :]
                    probs = torch.softmax(logits, dim=-1)
                
                target_ids = tokenizer.encode(en, add_special_tokens=False)
                top10_ids = torch.argsort(logits, descending=True)[:10].cpu().numpy()
                
                target_log_prob = -100
                for tid in target_ids:
                    if tid < len(probs):
                        target_log_prob = max(target_log_prob, float(torch.log(probs[tid] + 1e-10)))
                
                log_probs.append(target_log_prob)
                target_in_top10.append(any(tid in top10_ids for tid in target_ids))
                
                for h in hooks:
                    h.remove()
                del inputs, outputs, logits, probs
                gc.collect()
                torch.cuda.empty_cache()
            
            return {
                "mean_log_prob": float(np.mean(log_probs)),
                "std_log_prob": float(np.std(log_probs)),
                "top10_rate": float(np.mean(target_in_top10)),
            }
        
        # 1. Baseline
        base = evaluate_batch(test_pairs)
        print(f"    Baseline: log_prob={base['mean_log_prob']:.3f}")
        
        # 2. Remove spike (完全移除, 不像Phase115的50%)
        def make_remove_hook(proj_matrix_np):
            proj_matrix = torch.tensor(proj_matrix_np, dtype=torch.float32, device=device)
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0].detach().clone().float()
                    last_h = h[0, -1, :].unsqueeze(0)  # [1, P]
                    proj = proj_matrix @ (proj_matrix.T @ last_h.T)  # [P, 1]
                    h[0, -1, :] -= proj.squeeze()  # 完全移除
                    return (h.to(output[0].dtype),) + output[1:]
                else:
                    h = output.detach().clone().float()
                    last_h = h[0, -1, :].unsqueeze(0)
                    proj = proj_matrix @ (proj_matrix.T @ last_h.T)
                    h[0, -1, :] -= proj.squeeze()
                    return h.to(output.dtype)
            return hook_fn
        
        remove_result = evaluate_batch(test_pairs, make_remove_hook(V))
        print(f"    Remove spike: log_prob={remove_result['mean_log_prob']:.3f} "
              f"(Δ={remove_result['mean_log_prob'] - base['mean_log_prob']:.3f})")
        
        # 3. Amplify spike (x2)
        def make_amplify_hook(proj_matrix_np, factor=1.0):
            proj_matrix = torch.tensor(proj_matrix_np, dtype=torch.float32, device=device)
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0].detach().clone().float()
                    last_h = h[0, -1, :].unsqueeze(0)
                    proj = proj_matrix @ (proj_matrix.T @ last_h.T)
                    h[0, -1, :] += proj.squeeze() * factor
                    return (h.to(output[0].dtype),) + output[1:]
                else:
                    h = output.detach().clone().float()
                    last_h = h[0, -1, :].unsqueeze(0)
                    proj = proj_matrix @ (proj_matrix.T @ last_h.T)
                    h[0, -1, :] += proj.squeeze() * factor
                    return h.to(output.dtype)
            return hook_fn
        
        amplify_result = evaluate_batch(test_pairs, make_amplify_hook(V, factor=1.0))
        print(f"    Amplify spike: log_prob={amplify_result['mean_log_prob']:.3f} "
              f"(Δ={amplify_result['mean_log_prob'] - base['mean_log_prob']:.3f})")
        
        # 4. Remove random subspace (控制)
        V_random = np.random.randn(P, spike['k'])
        V_random, _ = np.linalg.qr(V_random)
        random_result = evaluate_batch(test_pairs, make_remove_hook(V_random))
        print(f"    Remove random: log_prob={random_result['mean_log_prob']:.3f} "
              f"(Δ={random_result['mean_log_prob'] - base['mean_log_prob']:.3f})")
        
        # 计算因果效应
        delta_remove = remove_result['mean_log_prob'] - base['mean_log_prob']
        delta_random = random_result['mean_log_prob'] - base['mean_log_prob']
        delta_amplify = amplify_result['mean_log_prob'] - base['mean_log_prob']
        
        causal_effect = delta_remove - delta_random  # >0 = spike移除比random移除更损害翻译
        
        results[f"L{l}"] = {
            "spike_dim": spike['k'],
            "baseline_log_prob": base['mean_log_prob'],
            "delta_remove_spike": float(delta_remove),
            "delta_amplify_spike": float(delta_amplify),
            "delta_remove_random": float(delta_random),
            "causal_effect": float(causal_effect),
        }
    
    return results


# ============================================================
# Exp 2: Spike Subspace Continuity — spike子空间连续性
# ============================================================
def exp2_spike_continuity(mlp_outs, d_model):
    """
    spike子空间连续性: L12的spike子空间与其他层的spike子空间的重叠
    
    核心问题: L12的4维spike是否嵌入在L35的14维spike中?
    """
    print("\n" + "="*70)
    print("Exp 2: Spike Subspace Continuity — spike子空间连续性")
    print("="*70)
    
    # 收集所有层的spike子空间
    spike_subspaces = {}
    for l in SAMPLE_LAYERS:
        if l not in mlp_outs['zh'] or l not in mlp_outs['trans']:
            continue
        
        zh_data = mlp_outs['zh'][l]
        trans_data = mlp_outs['trans'][l]
        N = zh_data.shape[0]
        P = zh_data.shape[1]
        
        diffs = trans_data - zh_data
        k = KNOWN_SIGNAL_DIMS.get(l, min(10, N-1))
        spike_subspaces[l] = get_spike_subspace(diffs, k, N, P)
        
        print(f"  L{l}: spike_dim={spike_subspaces[l]['k']}, "
              f"top-3 S={[f'{x:.2f}' for x in spike_subspaces[l]['S'][:3]]}")
    
    if len(spike_subspaces) < 2:
        print("  层数不足, 跳过")
        return {}
    
    # 计算层间子空间重叠
    # 方法: 子空间A到子空间B的投影比 = ||P_B @ v||^2 / ||v||^2, 对A的所有基向量平均
    layer_list = sorted(spike_subspaces.keys())
    results = {}
    
    # 参考层: L12 (状态协调码)
    ref_layer = 12
    if ref_layer not in spike_subspaces:
        ref_layer = min(spike_subspaces.keys(), key=lambda x: abs(x - 12))
    
    V_ref = spike_subspaces[ref_layer]['V']  # [P, k_ref]
    k_ref = spike_subspaces[ref_layer]['k']
    
    print(f"\n  参考层: L{ref_layer} (k={k_ref})")
    
    for l in layer_list:
        V_l = spike_subspaces[l]['V']  # [P, k_l]
        k_l = spike_subspaces[l]['k']
        
        # 1. L12→L的子空间包含度: L12的spike有多少在L的spike中?
        # P_L = V_l @ V_l^T, 投影比 = mean(||P_L @ v_i||^2 / ||v_i||^2)
        proj_matrix_l = V_l @ V_l.T  # [P, P]
        inclusion_ratios = []
        for i in range(k_ref):
            v = V_ref[:, i]  # [P]
            proj_v = proj_matrix_l @ v
            ratio = np.dot(proj_v, proj_v) / (np.dot(v, v) + 1e-10)
            inclusion_ratios.append(ratio)
        
        mean_inclusion = np.mean(inclusion_ratios)
        
        # 2. L→L12的子空间包含度: L的spike有多少在L12的spike中?
        proj_matrix_ref = V_ref @ V_ref.T
        reverse_inclusion_ratios = []
        for i in range(k_l):
            v = V_l[:, i]
            proj_v = proj_matrix_ref @ v
            ratio = np.dot(proj_v, proj_v) / (np.dot(v, v) + 1e-10)
            reverse_inclusion_ratios.append(ratio)
        
        mean_reverse_inclusion = np.mean(reverse_inclusion_ratios)
        
        # 3. 逐向量cosine相似度 (L12的每个spike方向与L的最近spike方向)
        cosine_to_nearest = []
        for i in range(k_ref):
            v = V_ref[:, i]
            cosines = []
            for j in range(k_l):
                u = V_l[:, j]
                cos = abs(np.dot(v, u) / (np.linalg.norm(v) * np.linalg.norm(u) + 1e-10))
                cosines.append(cos)
            cosine_to_nearest.append(max(cosines))
        
        mean_cosine_nearest = np.mean(cosine_to_nearest)
        
        # 4. Grassmann距离 (子空间距离的标准度量)
        # d_G = ||P_A - P_B||_F / sqrt(k_A + k_B)
        diff = proj_matrix_ref[:min(P, d_model), :min(P, d_model)] - proj_matrix_l[:min(P, d_model), :min(P, d_model)]
        grassmann_dist = np.linalg.norm(diff, 'fro') / np.sqrt(k_ref + k_l)
        
        results[f"L{l}"] = {
            "spike_dim": k_l,
            "inclusion_ref_to_l": float(mean_inclusion),  # L12→L 包含度
            "inclusion_l_to_ref": float(mean_reverse_inclusion),  # L→L12 包含度
            "cosine_to_nearest": float(mean_cosine_nearest),
            "grassmann_dist": float(grassmann_dist),
            "per_vector_inclusion": [float(x) for x in inclusion_ratios],
            "per_vector_cosine_nearest": [float(x) for x in cosine_to_nearest],
        }
        
        print(f"  L{l}: inclusion(L12→L)={mean_inclusion:.4f}, "
              f"inclusion(L→L12)={mean_reverse_inclusion:.4f}, "
              f"cos_nearest={mean_cosine_nearest:.4f}, "
              f"grassmann={grassmann_dist:.4f}")
    
    # 也计算相邻层之间的子空间重叠
    print("\n  --- 相邻层子空间重叠 ---")
    adjacency_results = {}
    for i in range(len(layer_list) - 1):
        l1, l2 = layer_list[i], layer_list[i+1]
        V1, V2 = spike_subspaces[l1]['V'], spike_subspaces[l2]['V']
        k1, k2 = spike_subspaces[l1]['k'], spike_subspaces[l2]['k']
        
        # 双向包含度
        proj2 = V2 @ V2.T
        inc1to2 = np.mean([np.dot(proj2 @ V1[:, j], proj2 @ V1[:, j]) / (np.dot(V1[:, j], V1[:, j]) + 1e-10) for j in range(k1)])
        
        proj1 = V1 @ V1.T
        inc2to1 = np.mean([np.dot(proj1 @ V2[:, j], proj1 @ V2[:, j]) / (np.dot(V2[:, j], V2[:, j]) + 1e-10) for j in range(k2)])
        
        adjacency_results[f"L{l1}_L{l2}"] = {
            "inclusion_l1_to_l2": float(inc1to2),
            "inclusion_l2_to_l1": float(inc2to1),
        }
        print(f"    L{l1}→L{l2}: inc1to2={inc1to2:.4f}, inc2to1={inc2to1:.4f}")
    
    results["_adjacency"] = adjacency_results
    results["_ref_layer"] = ref_layer
    
    return results


# ============================================================
# Exp 3: Spike Coefficient Decodability — spike系数可解码性
# ============================================================
def exp3_spike_decodability(mlp_outs, d_model, word_pairs):
    """
    spike系数可解码性: 在spike子空间中的投影系数能否预测目标词?
    
    方法:
    1. 将每个样本投影到该层的spike子空间, 得到k维系数
    2. 用系数做kNN分类: 给定一个样本的系数, 能否找到最近邻的正确翻译?
    3. 对比: 用同等维度的随机子空间做同样操作(基线)
    
    如果L12的系数不能解码 → 确认是"状态码"
    如果L35的系数能解码 → 确认是"决策码"
    """
    print("\n" + "="*70)
    print("Exp 3: Spike Coefficient Decodability — spike系数可解码性")
    print("="*70)
    
    results = {}
    
    for l in SAMPLE_LAYERS:
        if l not in mlp_outs['zh'] or l not in mlp_outs['trans']:
            continue
        
        zh_data = mlp_outs['zh'][l]
        trans_data = mlp_outs['trans'][l]
        N = zh_data.shape[0]
        P = zh_data.shape[1]
        
        if N < 20:
            continue
        
        diffs = trans_data - zh_data
        k = KNOWN_SIGNAL_DIMS.get(l, min(10, N-1))
        spike = get_spike_subspace(diffs, k, N, P)
        k_actual = spike['k']
        
        # 投影系数: [N, k]
        spike_coeffs = spike['coefficients']  # [N, k]
        
        # 随机子空间系数 (控制)
        V_random = np.random.randn(P, k_actual)
        V_random, _ = np.linalg.qr(V_random)
        diffs_centered = diffs - diffs.mean(axis=0, keepdims=True)
        random_coeffs = diffs_centered @ V_random  # [N, k]
        
        # 英文词标签
        en_labels = [en for _, en in word_pairs[:N]]
        unique_labels = list(set(en_labels))
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        
        # kNN解码 (k=5, leave-one-out)
        def knn_accuracy(coeffs, labels, k_nn=5):
            """Leave-one-out kNN准确率"""
            N = len(labels)
            correct = 0
            for i in range(N):
                # 计算到所有其他样本的距离
                dists = np.sum((coeffs - coeffs[i]) ** 2, axis=1)
                dists[i] = np.inf  # 排除自身
                nearest_k = np.argsort(dists)[:k_nn]
                # 投票
                vote_labels = [labels[j] for j in nearest_k]
                from collections import Counter
                most_common = Counter(vote_labels).most_common(1)[0][0]
                if most_common == labels[i]:
                    correct += 1
            return correct / N
        
        # Spike子空间kNN
        spike_acc = knn_accuracy(spike_coeffs, en_labels, k_nn=5)
        
        # 随机子空间kNN
        random_acc = knn_accuracy(random_coeffs, en_labels, k_nn=5)
        
        # 全空间kNN (在完整d_model维空间中)
        full_acc = knn_accuracy(diffs_centered, en_labels, k_nn=5)
        
        # 信息比: spike系数保留了多少翻译区分信息?
        # 如果spike_acc ≈ full_acc → spike子空间几乎包含所有翻译区分信息
        # 如果spike_acc ≈ random_acc → spike子空间不包含翻译区分信息
        info_ratio = spike_acc / full_acc if full_acc > 0 else 0
        
        # 随机基线的期望准确率
        random_baseline = 1.0 / len(unique_labels)
        
        results[f"L{l}"] = {
            "spike_dim": k_actual,
            "spike_knn_acc": float(spike_acc),
            "random_knn_acc": float(random_acc),
            "full_knn_acc": float(full_acc),
            "info_ratio": float(info_ratio),
            "random_baseline": float(random_baseline),
            "n_unique_labels": len(unique_labels),
        }
        
        print(f"  L{l}: spike_acc={spike_acc:.3f}, random_acc={random_acc:.3f}, "
              f"full_acc={full_acc:.3f}, info_ratio={info_ratio:.3f}, "
              f"baseline={random_baseline:.4f}")
    
    return results


# ============================================================
# Exp 4: Inter-Layer Spike Propagation — 层间spike传播
# ============================================================
def exp4_spike_propagation(mlp_outs, d_model):
    """
    层间spike传播: L12的spike系数能否预测L35的spike系数?
    
    方法:
    1. 获取L12和L35的spike系数
    2. 线性回归: L12_coeffs → L35_coeffs
    3. 非线性测试: 添加多项式特征后的预测力
    4. 逐步传播: L12→L15→L18→L21→L24→L27→L30→L33→L35
    """
    print("\n" + "="*70)
    print("Exp 4: Inter-Layer Spike Propagation — 层间spike传播")
    print("="*70)
    
    # 收集所有层的spike系数
    spike_data = {}
    for l in SAMPLE_LAYERS:
        if l not in mlp_outs['zh'] or l not in mlp_outs['trans']:
            continue
        
        zh_data = mlp_outs['zh'][l]
        trans_data = mlp_outs['trans'][l]
        N = zh_data.shape[0]
        P = zh_data.shape[1]
        
        diffs = trans_data - zh_data
        k = KNOWN_SIGNAL_DIMS.get(l, min(10, N-1))
        spike = get_spike_subspace(diffs, k, N, P)
        
        spike_data[l] = {
            'coeffs': spike['coefficients'],  # [N, k]
            'V': spike['V'],
            'k': spike['k'],
            'S': spike['S'],
        }
    
    if len(spike_data) < 2:
        print("  层数不足, 跳过")
        return {}
    
    layer_list = sorted(spike_data.keys())
    results = {}
    
    # 1. 逐步传播: 每对相邻层之间的系数预测力
    print("\n  --- 逐步传播 (相邻层) ---")
    step_results = {}
    for i in range(len(layer_list) - 1):
        l_src, l_tgt = layer_list[i], layer_list[i+1]
        coeffs_src = spike_data[l_src]['coeffs']  # [N, k_src]
        coeffs_tgt = spike_data[l_tgt]['coeffs']  # [N, k_tgt]
        N = coeffs_src.shape[0]
        
        # Leave-one-out线性预测
        # 简化: 用全样本做线性回归, 报告R²
        # coeffs_tgt ≈ coeffs_src @ W + b
        
        # 中心化
        src_mean = coeffs_src.mean(axis=0)
        tgt_mean = coeffs_tgt.mean(axis=0)
        src_c = coeffs_src - src_mean
        tgt_c = coeffs_tgt - tgt_mean
        
        # 线性回归 (正规方程)
        # W = (X^T X)^{-1} X^T Y
        try:
            W = np.linalg.lstsq(src_c, tgt_c, rcond=None)[0]  # [k_src, k_tgt]
            pred = src_c @ W
            # R² per target dimension
            ss_res = np.sum((tgt_c - pred) ** 2, axis=0)
            ss_tot = np.sum(tgt_c ** 2, axis=0)
            r2_per_dim = 1 - ss_res / (ss_tot + 1e-10)
            r2_mean = np.mean(r2_per_dim)
            r2_best = np.max(r2_per_dim)
        except:
            r2_mean = 0
            r2_best = 0
            r2_per_dim = [0] * coeffs_tgt.shape[1]
        
        step_results[f"L{l_src}_L{l_tgt}"] = {
            "r2_mean": float(r2_mean),
            "r2_best": float(r2_best),
            "r2_per_dim": [float(x) for x in r2_per_dim[:10]],  # 最多10维
            "src_dim": spike_data[l_src]['k'],
            "tgt_dim": spike_data[l_tgt]['k'],
        }
        print(f"    L{l_src}→L{l_tgt}: R²_mean={r2_mean:.4f}, R²_best={r2_best:.4f}")
    
    # 2. 从L12直接到所有层的传播
    print("\n  --- L12直接传播 ---")
    ref_layer = 12
    if ref_layer not in spike_data:
        ref_layer = min(spike_data.keys(), key=lambda x: abs(x - 12))
    
    ref_coeffs = spike_data[ref_layer]['coeffs']
    ref_mean = ref_coeffs.mean(axis=0)
    ref_c = ref_coeffs - ref_mean
    
    direct_results = {}
    for l in layer_list:
        if l == ref_layer:
            continue
        
        tgt_coeffs = spike_data[l]['coeffs']
        tgt_mean = tgt_coeffs.mean(axis=0)
        tgt_c = tgt_coeffs - tgt_mean
        
        try:
            W = np.linalg.lstsq(ref_c, tgt_c, rcond=None)[0]
            pred = ref_c @ W
            ss_res = np.sum((tgt_c - pred) ** 2, axis=0)
            ss_tot = np.sum(tgt_c ** 2, axis=0)
            r2_per_dim = 1 - ss_res / (ss_tot + 1e-10)
            r2_mean = np.mean(r2_per_dim)
            r2_best = np.max(r2_per_dim)
        except:
            r2_mean = 0
            r2_best = 0
        
        direct_results[f"L{ref_layer}_L{l}"] = {
            "r2_mean": float(r2_mean),
            "r2_best": float(r2_best),
        }
        print(f"    L{ref_layer}→L{l}: R²_mean={r2_mean:.4f}, R²_best={r2_best:.4f}")
    
    # 3. 非线性传播测试: L12→L35
    print("\n  --- 非线性传播测试 (L12→L35) ---")
    tgt_layer = 35
    if ref_layer in spike_data and tgt_layer in spike_data:
        src_c = ref_c
        tgt_c = spike_data[tgt_layer]['coeffs'] - spike_data[tgt_layer]['coeffs'].mean(axis=0)
        
        # 线性R²
        try:
            W_lin = np.linalg.lstsq(src_c, tgt_c, rcond=None)[0]
            pred_lin = src_c @ W_lin
            r2_lin = np.mean(1 - np.sum((tgt_c - pred_lin)**2, axis=0) / (np.sum(tgt_c**2, axis=0) + 1e-10))
        except:
            r2_lin = 0
        
        # 二次特征
        from itertools import combinations
        k_src = src_c.shape[1]
        quad_features = []
        for i in range(k_src):
            quad_features.append(src_c[:, i:i+1] ** 2)
        for i, j in combinations(range(k_src), 2):
            quad_features.append((src_c[:, i] * src_c[:, j]).reshape(-1, 1))
        
        if quad_features:
            src_quad = np.hstack([src_c] + quad_features)
            try:
                W_quad = np.linalg.lstsq(src_quad, tgt_c, rcond=None)[0]
                pred_quad = src_quad @ W_quad
                r2_quad = np.mean(1 - np.sum((tgt_c - pred_quad)**2, axis=0) / (np.sum(tgt_c**2, axis=0) + 1e-10))
            except:
                r2_quad = 0
        else:
            r2_quad = 0
        
        nonlinearity_gain = r2_quad - r2_lin
        print(f"    线性R²={r2_lin:.4f}, 二次R²={r2_quad:.4f}, 增益={nonlinearity_gain:.4f}")
        
        results["_nonlinear_L12_L35"] = {
            "r2_linear": float(r2_lin),
            "r2_quadratic": float(r2_quad),
            "nonlinearity_gain": float(nonlinearity_gain),
        }
    
    results["_step_propagation"] = step_results
    results["_direct_from_L12"] = direct_results
    results["_ref_layer"] = ref_layer
    
    return results


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=str, default="all", choices=["all", "1", "2", "3", "4"])
    args = parser.parse_args()
    
    model_name = args.model
    print(f"\n{'='*70}")
    print(f"Phase 116: Spike传播动力学与因果过渡点")
    print(f"模型: {model_name}")
    print(f"{'='*70}")
    
    # 加载模型
    model, tokenizer, device = load_model(model_name)
    layers = get_layers(model)
    n_layers = len(layers)
    model_info = get_model_info(model, model_name)
    d_model = model_info.d_model
    print(f"模型: {model_info.model_class}, {n_layers}层, d_model={d_model}, device={device}")
    
    # 更新采样层
    global SAMPLE_LAYERS, KNOWN_SIGNAL_DIMS
    if n_layers <= 36:
        SAMPLE_LAYERS = [l for l in [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35] if l < n_layers]
    
    all_results = {"model": model_name, "n_layers": n_layers, "d_model": d_model}
    
    # 收集MLP输出
    print("\n--- 收集MLP输出 ---")
    mlp_outs = collect_mlp_outputs(model, tokenizer, device, n_layers, WORD_PAIRS)
    
    # Exp 1: Layer-by-Layer Causal Sweep
    if args.exp in ["all", "1"]:
        print("\n--- Exp 1: Layer-by-Layer Causal Sweep ---")
        exp1_result = exp1_causal_sweep(model, tokenizer, device, n_layers, mlp_outs, d_model)
        all_results["exp1_causal_sweep"] = exp1_result
        
        out_path = os.path.join(OUT_DIR, f"phase116_exp1_{model_name}_causal_sweep.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(exp1_result, f, indent=2, ensure_ascii=False)
        print(f"  保存到 {out_path}")
    
    # Exp 2: Spike Subspace Continuity
    if args.exp in ["all", "2"]:
        print("\n--- Exp 2: Spike Subspace Continuity ---")
        exp2_result = exp2_spike_continuity(mlp_outs, d_model)
        all_results["exp2_continuity"] = exp2_result
        
        out_path = os.path.join(OUT_DIR, f"phase116_exp2_{model_name}_continuity.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(exp2_result, f, indent=2, ensure_ascii=False, default=str)
        print(f"  保存到 {out_path}")
    
    # Exp 3: Spike Coefficient Decodability
    if args.exp in ["all", "3"]:
        print("\n--- Exp 3: Spike Coefficient Decodability ---")
        exp3_result = exp3_spike_decodability(mlp_outs, d_model, WORD_PAIRS)
        all_results["exp3_decodability"] = exp3_result
        
        out_path = os.path.join(OUT_DIR, f"phase116_exp3_{model_name}_decodability.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(exp3_result, f, indent=2, ensure_ascii=False)
        print(f"  保存到 {out_path}")
    
    # Exp 4: Inter-Layer Spike Propagation
    if args.exp in ["all", "4"]:
        print("\n--- Exp 4: Inter-Layer Spike Propagation ---")
        exp4_result = exp4_spike_propagation(mlp_outs, d_model)
        all_results["exp4_propagation"] = exp4_result
        
        out_path = os.path.join(OUT_DIR, f"phase116_exp4_{model_name}_propagation.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(exp4_result, f, indent=2, ensure_ascii=False, default=str)
        print(f"  保存到 {out_path}")
    
    # 保存全部结果
    all_out_path = os.path.join(OUT_DIR, f"phase116_{model_name}_all_results.json")
    with open(all_out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n全部结果保存到 {all_out_path}")
    
    # 释放模型
    release_model(model)
    print("\nPhase 116 完成!")


if __name__ == "__main__":
    main()
