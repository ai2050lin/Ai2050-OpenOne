"""
Phase 113: Null Hypothesis + Subspace Dynamics + Operator Decomposition
============================================================
核心目标:
1. 生死实验: 90°旋转是否只是高维几何零假设?
2. 子空间动力学: principal angles代替单向量夹角
3. 算子分解: attention vs MLP如何分别改变translation subspace
4. 路径拓扑稳定性: 计算路径跨层持续度

关键批判回应:
- "90°旋转"可能是null expectation → 必须比较随机基线
- 单方向夹角不够 → 需要principal angles between subspaces
- 不知道驱动器 → 需要operator decomposition
- 方向不稳定 → 应该研究route topology
"""

import os, sys, json, gc, time, argparse
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import torch
from collections import defaultdict

# 添加路径以使用model_utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_utils import load_model, get_layers, get_model_info, release_model

# ============================================================
# 设置
# ============================================================
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'glm5_temp')
OUT_DIR = os.path.abspath(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 翻译词对 - 扩大到100个
WORD_PAIRS = [
    # 自然
    ("猫", "cat"), ("狗", "dog"), ("鸟", "bird"), ("鱼", "fish"), ("马", "horse"),
    ("牛", "cow"), ("羊", "sheep"), ("猪", "pig"), ("鸡", "chicken"), ("鼠", "mouse"),
    # 物质
    ("水", "water"), ("火", "fire"), ("土", "earth"), ("风", "wind"), ("金", "gold"),
    ("木", "wood"), ("铁", "iron"), ("石", "stone"), ("沙", "sand"), ("冰", "ice"),
    # 天体
    ("月", "moon"), ("星", "star"), ("云", "cloud"), ("雨", "rain"), ("雪", "snow"),
    ("日", "sun"), ("天", "sky"), ("海", "sea"), ("河", "river"), ("山", "mountain"),
    # 颜色
    ("红", "red"), ("蓝", "blue"), ("绿", "green"), ("白", "white"), ("黑", "black"),
    ("黄", "yellow"), ("紫", "purple"), ("灰", "gray"), ("棕", "brown"), ("粉", "pink"),
    # 身体
    ("手", "hand"), ("足", "foot"), ("目", "eye"), ("耳", "ear"), ("口", "mouth"),
    ("心", "heart"), ("头", "head"), ("骨", "bone"), ("血", "blood"), ("发", "hair"),
    # 人类
    ("父", "father"), ("母", "mother"), ("子", "son"), ("女", "daughter"), ("友", "friend"),
    ("王", "king"), ("师", "teacher"), ("医", "doctor"), ("兵", "soldier"), ("农", "farmer"),
    # 抽象
    ("爱", "love"), ("恨", "hate"), ("善", "good"), ("恶", "evil"), ("真", "truth"),
    ("美", "beauty"), ("智", "wisdom"), ("力", "power"), ("光", "light"), ("影", "shadow"),
    # 动作
    ("走", "walk"), ("跑", "run"), ("飞", "fly"), ("吃", "eat"), ("喝", "drink"),
    ("看", "see"), ("听", "hear"), ("说", "say"), ("写", "write"), ("读", "read"),
    # 时间
    ("年", "year"), ("月", "month"), ("日", "day"), ("时", "hour"), ("分", "minute"),
    # 食物
    ("米", "rice"), ("茶", "tea"), ("酒", "wine"), ("肉", "meat"), ("盐", "salt"),
    # 交通
    ("车", "car"), ("船", "ship"), ("路", "road"), ("桥", "bridge"), ("门", "door"),
    # 自然2
    ("花", "flower"), ("草", "grass"), ("树", "tree"), ("叶", "leaf"), ("根", "root"),
    # 情感
    ("喜", "joy"), ("怒", "anger"), ("哀", "sorrow"), ("惧", "fear"), ("思", "thought"),
    # 社会制度
    ("法", "law"), ("国", "country"), ("城", "city"), ("家", "home"), ("书", "book"),
    # 科技
    ("电", "electricity"), ("网", "network"), ("数", "number"), ("算", "compute"), ("器", "device"),
]

# 去重
seen_zh = set()
UNIQUE_PAIRS = []
for zh, en in WORD_PAIRS:
    if zh not in seen_zh:
        seen_zh.add(zh)
        UNIQUE_PAIRS.append((zh, en))
WORD_PAIRS = UNIQUE_PAIRS[:100]

# 采样层
SAMPLE_LAYERS = [0, 6, 12, 18, 24, 27, 30, 33, 35]

# ============================================================
# 模型加载 - 使用model_utils
# ============================================================

# ============================================================
# 数据收集: MLP gate activations (翻译 vs 中文)
# ============================================================
def collect_gate_activations(model, tokenizer, device, n_layers, word_pairs, batch_size=10):
    """收集所有词对的MLP gate activations"""
    layers = get_layers(model)
    n_neurons = None
    
    # 存储: {层: {zh: [样本×neuron], trans: [样本×neuron]}}
    all_acts = {'zh': defaultdict(list), 'trans': defaultdict(list)}
    
    for batch_start in range(0, len(word_pairs), batch_size):
        batch = word_pairs[batch_start:batch_start+batch_size]
        
        for zh, en in batch:
            # 中文prompt
            zh_prompt = f"翻译以下中文词：{zh}"
            # 翻译prompt
            trans_prompt = f"Translate the following Chinese word: {zh}"
            
            for task, prompt in [('zh', zh_prompt), ('trans', trans_prompt)]:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                gate_acts = {}
                hooks = []
                
                def make_hook(l):
                    def hook_fn(module, input, output):
                        gate_act = torch.nn.functional.silu(output)
                        gate_acts[l] = gate_act[0, -1, :].detach().float().cpu().numpy()
                    return hook_fn
                
                for l, layer in enumerate(layers):
                    if hasattr(layer.mlp, 'gate_proj'):
                        h = layer.mlp.gate_proj.register_forward_hook(make_hook(l))
                        hooks.append(h)
                
                with torch.no_grad():
                    outputs = model(inputs["input_ids"])
                
                for h in hooks:
                    h.remove()
                
                del outputs, inputs
                gc.collect()
                torch.cuda.empty_cache()
                
                for l in gate_acts:
                    all_acts[task][l].append(gate_acts[l])
                    if n_neurons is None:
                        n_neurons = gate_acts[l].shape[0]
        
        print(f"  已收集 {min(batch_start+batch_size, len(word_pairs))}/{len(word_pairs)} 词对")
    
    # 转numpy
    result = {}
    for task in ['zh', 'trans']:
        result[task] = {}
        for l in all_acts[task]:
            result[task][l] = np.array(all_acts[task][l])  # shape: (n_samples, n_neurons)
    
    return result, n_neurons

# ============================================================
# 数据收集: Attention vs MLP 分离输出
# ============================================================
def collect_attn_mlp_outputs(model, tokenizer, device, n_layers, word_pairs, batch_size=10):
    """收集attention输出和MLP输出(分离)"""
    layers = get_layers(model)
    
    # 存储: {层: {zh_attn: [], zh_mlp: [], trans_attn: [], trans_mlp: []}}
    all_outputs = defaultdict(lambda: defaultdict(list))
    
    for batch_start in range(0, len(word_pairs), batch_size):
        batch = word_pairs[batch_start:batch_start+batch_size]
        
        for zh, en in batch:
            zh_prompt = f"翻译以下中文词：{zh}"
            trans_prompt = f"Translate the following Chinese word: {zh}"
            
            for task, prompt in [('zh', zh_prompt), ('trans', trans_prompt)]:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                
                # 需要hook: residual stream before/after attention, after MLP
                # strategy: hook residual_stream at three points
                #   - after attention (before MLP): captures attention contribution
                #   - after MLP: captures MLP contribution by subtraction
                
                residuals = {}
                hooks = []
                
                for l, layer in enumerate(layers):
                    # Hook the residual stream at layer output
                    # In most models, layer forward = x + attn(x) + mlp(x + attn(x))
                    # We need: attn_output and mlp_output separately
                    
                    # Method: hook attention output and MLP output directly
                    def make_attn_hook(l):
                        def hook_fn(module, input, output):
                            # output is typically (attn_output, attn_weights, ...)
                            if isinstance(output, tuple):
                                residuals[(l, 'attn')] = output[0][0, -1, :].detach().float().cpu().numpy()
                            else:
                                residuals[(l, 'attn')] = output[0, -1, :].detach().float().cpu().numpy()
                        return hook_fn
                    
                    def make_mlp_hook(l):
                        def hook_fn(module, input, output):
                            residuals[(l, 'mlp')] = output[0, -1, :].detach().float().cpu().numpy()
                        return hook_fn
                    
                    # Hook self_attn output
                    if hasattr(layer, 'self_attn'):
                        h1 = layer.self_attn.register_forward_hook(make_attn_hook(l))
                        hooks.append(h1)
                    
                    # Hook mlp output
                    if hasattr(layer, 'mlp'):
                        h2 = layer.mlp.register_forward_hook(make_mlp_hook(l))
                        hooks.append(h2)
                
                with torch.no_grad():
                    outputs = model(inputs["input_ids"])
                
                for h in hooks:
                    h.remove()
                del outputs, inputs
                gc.collect()
                torch.cuda.empty_cache()
                
                for l in SAMPLE_LAYERS:
                    key_a = (l, 'attn')
                    key_m = (l, 'mlp')
                    if key_a in residuals:
                        all_outputs[l][f'{task}_attn'].append(residuals[key_a])
                    if key_m in residuals:
                        all_outputs[l][f'{task}_mlp'].append(residuals[key_m])
        
        print(f"  已收集 {min(batch_start+batch_size, len(word_pairs))}/{len(word_pairs)} 词对")
    
    # 转numpy
    result = {}
    for l in SAMPLE_LAYERS:
        result[l] = {}
        for key in all_outputs[l]:
            result[l][key] = np.array(all_outputs[l][key])
    
    return result

# ============================================================
# Exp 1: 零假设检验 — 90°旋转是否只是高维几何?
# ============================================================
def exp1_null_rotation(gate_acts, n_neurons, n_samples, n_bootstrap=500):
    """
    核心实验: 比较真实旋转角与随机基线
    
    零假设: 在n_neurons维空间中, 两个随机向量天然正交(θ≈90°)
    备择假设: 翻译-中文差分方向的层间旋转角偏离随机基线
    
    方法:
    1. 真实: 翻译-中文差分向量在相邻层间的夹角
    2. Null A: 从N(0,1)采样的随机向量的夹角
    3. Null B: 随机prompt的激活差分的夹角(控制模型结构效应)
    4. Null C: 同任务内的样本间差分向量的夹角
    """
    print("\n" + "="*60)
    print("Exp 1: 零假设检验 — 90°旋转是否只是高维几何?")
    print("="*60)
    
    results = {}
    
    # --- 真实旋转角 ---
    print("\n计算真实旋转角...")
    real_angles = {}
    for l in sorted(gate_acts['zh'].keys()):
        zh = gate_acts['zh'][l]  # (n_samples, n_neurons)
        trans = gate_acts['trans'][l]
        
        # 差分向量: 逐样本
        diffs = trans - zh  # (n_samples, n_neurons)
        
        # 均值差分方向
        mean_diff = diffs.mean(axis=0)
        mean_diff_norm = mean_diff / (np.linalg.norm(mean_diff) + 1e-10)
        
        real_angles[l] = mean_diff_norm
    
    # 计算相邻层间夹角
    sorted_layers = sorted(real_angles.keys())
    real_layer_angles = {}
    for i in range(len(sorted_layers) - 1):
        l1, l2 = sorted_layers[i], sorted_layers[i+1]
        cos_theta = np.abs(np.dot(real_angles[l1], real_angles[l2]))
        theta = np.degrees(np.arccos(np.clip(cos_theta, 0, 1)))
        real_layer_angles[(l1, l2)] = theta
    
    print("真实旋转角:")
    for (l1, l2), theta in real_layer_angles.items():
        print(f"  L{l1}→L{l2}: {theta:.1f}°")
    
    # --- Null A: 纯随机向量 ---
    print("\nNull A: 纯随机向量的夹角分布...")
    null_a_angles = []
    for _ in range(n_bootstrap):
        v1 = np.random.randn(n_neurons)
        v2 = np.random.randn(n_neurons)
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)
        cos_theta = np.abs(np.dot(v1, v2))
        theta = np.degrees(np.arccos(np.clip(cos_theta, 0, 1)))
        null_a_angles.append(theta)
    
    null_a_mean = np.mean(null_a_angles)
    null_a_std = np.std(null_a_angles)
    print(f"  Null A (随机向量): {null_a_mean:.1f}° ± {null_a_std:.1f}°")
    
    # --- Null B: 同分布内的随机差分 ---
    # 对每层, 随机采样两个子集, 计算子集均值差
    print("\nNull B: 同分布内随机差分的层间夹角...")
    null_b_angles = defaultdict(list)
    for _ in range(n_bootstrap):
        for l in sorted(gate_acts['zh'].keys()):
            zh = gate_acts['zh'][l]
            trans = gate_acts['trans'][l]
            n = len(zh)
            
            # 随机分两半
            idx1 = np.random.choice(n, n//2, replace=False)
            idx2 = np.array([i for i in range(n) if i not in idx1])
            
            # 随机差分方向
            diff1 = trans[idx1].mean(0) - zh[idx1].mean(0)
            diff1 /= np.linalg.norm(diff1) + 1e-10
            diff2 = trans[idx2].mean(0) - zh[idx2].mean(0)
            diff2 /= np.linalg.norm(diff2) + 1e-10
            
            # 存储每层的随机差分方向
            if l not in results:
                results[l] = {}
            if 'null_b_diffs' not in results[l]:
                results[l]['null_b_diffs'] = []
            results[l]['null_b_diffs'].append(diff1)
    
    # 计算null_b的层间夹角
    sorted_layers_list = sorted(gate_acts['zh'].keys())
    for i in range(len(sorted_layers_list) - 1):
        l1, l2 = sorted_layers_list[i], sorted_layers_list[i+1]
        for b in range(n_bootstrap):
            if l1 in results and l2 in results:
                v1 = results[l1]['null_b_diffs'][b]
                v2 = results[l2]['null_b_diffs'][b]
                cos_theta = np.abs(np.dot(v1, v2))
                theta = np.degrees(np.arccos(np.clip(cos_theta, 0, 1)))
                null_b_angles[(l1, l2)].append(theta)
    
    print("Null B (同分布随机差分) 的层间夹角:")
    null_b_summary = {}
    for (l1, l2) in sorted(null_b_angles.keys()):
        angles = null_b_angles[(l1, l2)]
        null_b_summary[(l1, l2)] = {'mean': np.mean(angles), 'std': np.std(angles)}
        print(f"  L{l1}→L{l2}: {np.mean(angles):.1f}° ± {np.std(angles):.1f}°")
    
    # --- Null C: 同任务内的差分 (纯中文内随机差分) ---
    print("\nNull C: 同任务内的随机差分层间夹角...")
    null_c_angles = defaultdict(list)
    for _ in range(n_bootstrap):
        prev_diff = None
        for l in sorted(gate_acts['zh'].keys()):
            zh = gate_acts['zh'][l]
            n = len(zh)
            idx1 = np.random.choice(n, n//2, replace=False)
            idx2 = np.array([i for i in range(n) if i not in idx1])
            
            diff = zh[idx1].mean(0) - zh[idx2].mean(0)
            diff /= np.linalg.norm(diff) + 1e-10
            
            if prev_diff is not None:
                cos_theta = np.abs(np.dot(prev_diff, diff))
                theta = np.degrees(np.arccos(np.clip(cos_theta, 0, 1)))
                null_c_angles[prev_l].append(theta)
            
            prev_diff = diff
            prev_l = l
    
    print("Null C (同任务随机差分) 的层间夹角:")
    null_c_summary = {}
    for l_pair in sorted(null_c_angles.keys()):
        angles = null_c_angles[l_pair]
        null_c_summary[l_pair] = {'mean': np.mean(angles), 'std': np.std(angles)}
    
    # --- 综合比较 ---
    print("\n" + "="*60)
    print("零假设检验总结:")
    print("="*60)
    
    summary = {}
    for (l1, l2) in sorted(real_layer_angles.keys()):
        real_theta = real_layer_angles[(l1, l2)]
        null_a_z = (real_theta - null_a_mean) / (null_a_std + 1e-10)
        
        null_b_mean = null_b_summary.get((l1, l2), {}).get('mean', null_a_mean)
        null_b_std = null_b_summary.get((l1, l2), {}).get('std', null_a_std)
        null_b_z = (real_theta - null_b_mean) / (null_b_std + 1e-10)
        
        # 判定: 是否显著偏离null B?
        is_significant = abs(null_b_z) > 2.0
        
        summary[f"L{l1}_L{l2}"] = {
            'real_angle': float(real_theta),
            'null_a_mean': float(null_a_mean),
            'null_a_std': float(null_a_std),
            'null_a_z': float(null_a_z),
            'null_b_mean': float(null_b_mean),
            'null_b_std': float(null_b_std),
            'null_b_z': float(null_b_z),
            'is_significant': bool(is_significant),
            'verdict': 'REJECT_NULL' if is_significant else 'FAIL_REJECT_NULL'
        }
        
        status = "★ 显著偏离null" if is_significant else "✗ 不显著 (=null expectation)"
        print(f"  L{l1}→L{l2}: real={real_theta:.1f}°, null_B={null_b_mean:.1f}°±{null_b_std:.1f}°, z={null_b_z:.2f} → {status}")
    
    # --- 个体样本旋转角分布 ---
    print("\n个体样本旋转角分布 (非均值)...")
    individual_angles = defaultdict(list)
    for l in sorted(gate_acts['zh'].keys()):
        zh = gate_acts['zh'][l]
        trans = gate_acts['trans'][l]
        n = len(zh)
        
        for i in range(n):
            diff = trans[i] - zh[i]
            norm = np.linalg.norm(diff)
            if norm > 1e-6:
                individual_angles[l].append(diff / norm)
    
    # 相邻层个体旋转角
    ind_layer_angles = {}
    for i in range(len(sorted_layers) - 1):
        l1, l2 = sorted_layers[i], sorted_layers[i+1]
        if l1 in individual_angles and l2 in individual_angles:
            angles_list = []
            n_ind = min(len(individual_angles[l1]), len(individual_angles[l2]))
            for j in range(n_ind):
                cos_theta = np.abs(np.dot(individual_angles[l1][j], individual_angles[l2][j]))
                theta = np.degrees(np.arccos(np.clip(cos_theta, 0, 1)))
                angles_list.append(theta)
            ind_layer_angles[(l1, l2)] = angles_list
    
    print("个体样本旋转角分布:")
    for (l1, l2) in sorted(ind_layer_angles.keys()):
        angles = ind_layer_angles[(l1, l2)]
        print(f"  L{l1}→L{l2}: mean={np.mean(angles):.1f}°, std={np.std(angles):.1f}°, "
              f"median={np.median(angles):.1f}°, [10th,90th]=[{np.percentile(angles,10):.1f}°,{np.percentile(angles,90):.1f}°]")
    
    return {
        'real_layer_angles': {f"{k[0]}_{k[1]}": v for k, v in real_layer_angles.items()},
        'null_a_mean': float(null_a_mean),
        'null_a_std': float(null_a_std),
        'null_b_summary': {f"{k[0]}_{k[1]}": v for k, v in null_b_summary.items()},
        'summary': summary,
        'individual_angles': {f"{k[0]}_{k[1]}": {'mean': float(np.mean(v)), 'std': float(np.std(v)),
                             'median': float(np.median(v)), 'p10': float(np.percentile(v,10)), 
                             'p90': float(np.percentile(v,90))} 
                             for k, v in ind_layer_angles.items()}
    }

# ============================================================
# Exp 2: 子空间动力学 — Principal Angles
# ============================================================
def exp2_subspace_dynamics(gate_acts, n_neurons, n_samples, n_components=10):
    """
    从单向量夹角升级到子空间principal angles
    
    核心思想: 不再只看均值差分方向, 而是看翻译-中文差分的整个子空间
    子空间间的principal angles比单向量夹角更稳定、更信息丰富
    
    数学: 
    1. 对每层, 计算N个样本的差分矩阵 D (n_samples × n_neurons)
    2. SVD分解: D = U Σ V^T
    3. 取前k个主成分构成子空间
    4. 计算相邻层子空间间的principal angles
    """
    print("\n" + "="*60)
    print("Exp 2: 子空间动力学 — Principal Angles (Grassmannian)")
    print("="*60)
    
    results = {}
    
    # --- 构建每层的翻译差分子空间 ---
    print("构建翻译差分子空间...")
    subspaces = {}
    subspace_info = {}
    
    for l in sorted(gate_acts['zh'].keys()):
        zh = gate_acts['zh'][l]
        trans = gate_acts['trans'][l]
        
        # 差分矩阵
        diffs = trans - zh  # (n_samples, n_neurons)
        
        # 中心化
        diffs_centered = diffs - diffs.mean(axis=0, keepdims=True)
        
        # SVD
        U, S, Vt = np.linalg.svd(diffs_centered, full_matrices=False)
        
        # 有效维度 (participation ratio)
        S2 = S ** 2
        pr = np.sum(S2) ** 2 / (np.sum(S2 ** 2) + 1e-10)
        
        # 累积方差解释
        cumvar = np.cumsum(S2) / (np.sum(S2) + 1e-10)
        
        # 子空间: 前k个主方向
        k = min(n_components, len(S))
        subspace = Vt[:k, :]  # (k, n_neurons) — 行是主方向
        
        subspaces[l] = subspace
        subspace_info[l] = {
            'singular_values': S[:20].tolist(),
            'participation_ratio': float(pr),
            'cumvar_50': int(np.searchsorted(cumvar, 0.5) + 1),
            'cumvar_90': int(np.searchsorted(cumvar, 0.9) + 1),
            'cumvar_95': int(np.searchsorted(cumvar, 0.95) + 1),
        }
        
        print(f"  L{l}: PR={pr:.1f}, dim_50={subspace_info[l]['cumvar_50']}, "
              f"dim_90={subspace_info[l]['cumvar_90']}, dim_95={subspace_info[l]['cumvar_95']}")
    
    # --- Principal Angles between adjacent layer subspaces ---
    print("\n计算相邻层子空间的Principal Angles...")
    
    sorted_layers_list = sorted(subspaces.keys())
    principal_angle_results = {}
    
    for i in range(len(sorted_layers_list) - 1):
        l1, l2 = sorted_layers_list[i], sorted_layers_list[i+1]
        
        U1 = subspaces[l1].T  # (n_neurons, k)
        U2 = subspaces[l2].T  # (n_neurons, k)
        
        # Principal angles via SVD of U1^T U2
        M = U1.T @ U2  # (k, k)
        cos_angles = np.linalg.svd(M, compute_uv=False)
        cos_angles = np.clip(cos_angles, 0, 1)
        angles_deg = np.degrees(np.arccos(cos_angles))
        
        # Grassmannian distance: d = ||angles||
        grassmann_dist = np.sqrt(np.sum(angles_deg ** 2))
        
        # 最大principal angle
        max_angle = angles_deg[0]
        
        # 最小principal angle (最重要: 子空间最接近的方向)
        min_angle = angles_deg[-1]
        
        # 平均principal angle
        mean_angle = np.mean(angles_deg)
        
        principal_angle_results[f"L{l1}_L{l2}"] = {
            'principal_angles': angles_deg.tolist(),
            'max_angle': float(max_angle),
            'min_angle': float(min_angle),
            'mean_angle': float(mean_angle),
            'grassmann_distance': float(grassmann_dist),
        }
        
        print(f"  L{l1}→L{l2}: min={min_angle:.1f}°, max={max_angle:.1f}°, "
              f"mean={mean_angle:.1f}°, Grassmann_dist={grassmann_dist:.1f}")
    
    # --- 非相邻层比较: 子空间对齐随距离衰减 ---
    print("\n子空间对齐随层距离衰减...")
    distance_decay = {}
    for i in range(len(sorted_layers_list)):
        for j in range(i+1, len(sorted_layers_list)):
            l1, l2 = sorted_layers_list[i], sorted_layers_list[j]
            dist = abs(l2 - l1)
            
            U1 = subspaces[l1].T
            U2 = subspaces[l2].T
            M = U1.T @ U2
            cos_angles = np.linalg.svd(M, compute_uv=False)
            cos_angles = np.clip(cos_angles, 0, 1)
            min_angle = np.degrees(np.arccos(cos_angles[-1]))
            
            if dist not in distance_decay:
                distance_decay[dist] = []
            distance_decay[dist].append(min_angle)
    
    print("最小principal angle vs 层距离:")
    decay_summary = {}
    for dist in sorted(distance_decay.keys()):
        angles = distance_decay[dist]
        decay_summary[str(dist)] = {'mean': float(np.mean(angles)), 'std': float(np.std(angles))}
        print(f"  distance={dist}: min_angle={np.mean(angles):.1f}° ± {np.std(angles):.1f}°")
    
    # --- Null: 随机子空间的principal angles ---
    print("\nNull: 随机子空间的principal angles...")
    k = min(n_components, n_samples)
    null_principal_angles = []
    for _ in range(200):
        U1 = np.random.randn(n_neurons, k)
        U2 = np.random.randn(n_neurons, k)
        # 正交化
        U1, _ = np.linalg.qr(U1)
        U2, _ = np.linalg.qr(U2)
        M = U1.T @ U2
        cos_angles = np.linalg.svd(M, compute_uv=False)
        cos_angles = np.clip(cos_angles, 0, 1)
        null_angles = np.degrees(np.arccos(cos_angles))
        null_principal_angles.append(null_angles)
    
    null_pa = np.array(null_principal_angles)
    print(f"  Null principal angles (随机子空间): min_angle mean={np.mean(null_pa[:,-1]):.1f}°, "
          f"mean_angle mean={np.mean(null_pa):.1f}°")
    
    return {
        'subspace_info': subspace_info,
        'principal_angles': principal_angle_results,
        'distance_decay': decay_summary,
        'null_min_angle_mean': float(np.mean(null_pa[:,-1])),
        'null_min_angle_std': float(np.std(null_pa[:,-1])),
        'null_mean_angle_mean': float(np.mean(null_pa)),
        'null_mean_angle_std': float(np.std(null_pa)),
    }

# ============================================================
# Exp 3: 算子分解 — Attention vs MLP
# ============================================================
def exp3_operator_decomposition(attn_mlp_outputs, n_samples):
    """
    分离attention和MLP对翻译子空间的贡献
    
    核心思想:
    - translation subspace的变化由attention和MLP共同贡献
    - 分别分析: attention输出和MLP输出的差分子空间
    - 比较哪个算子对翻译-中文分离贡献更大
    """
    print("\n" + "="*60)
    print("Exp 3: 算子分解 — Attention vs MLP贡献")
    print("="*60)
    
    results = {}
    
    for l in sorted(attn_mlp_outputs.keys()):
        data = attn_mlp_outputs[l]
        
        zh_attn = data.get('zh_attn')
        trans_attn = data.get('trans_attn')
        zh_mlp = data.get('zh_mlp')
        trans_mlp = data.get('trans_mlp')
        
        if any(x is None for x in [zh_attn, trans_attn, zh_mlp, trans_mlp]):
            print(f"  L{l}: 缺少数据, 跳过")
            continue
        
        layer_result = {}
        
        # --- Attention差分 ---
        attn_diff = trans_attn - zh_attn  # (n_samples, dim)
        attn_diff_mean = attn_diff.mean(axis=0)
        attn_diff_norm = np.linalg.norm(attn_diff_mean)
        
        # 差分的有效维度
        attn_centered = attn_diff - attn_diff_mean
        if attn_centered.shape[0] > 1:
            S_attn = np.linalg.svd(attn_centered, compute_uv=False)
            S2_attn = S_attn ** 2
            attn_pr = np.sum(S2_attn) ** 2 / (np.sum(S2_attn ** 2) + 1e-10)
            attn_cumvar = np.cumsum(S2_attn) / (np.sum(S2_attn) + 1e-10)
        else:
            attn_pr = 1.0
            attn_cumvar = np.array([1.0])
        
        # --- MLP差分 ---
        mlp_diff = trans_mlp - zh_mlp
        mlp_diff_mean = mlp_diff.mean(axis=0)
        mlp_diff_norm = np.linalg.norm(mlp_diff_mean)
        
        mlp_centered = mlp_diff - mlp_diff_mean
        if mlp_centered.shape[0] > 1:
            S_mlp = np.linalg.svd(mlp_centered, compute_uv=False)
            S2_mlp = S_mlp ** 2
            mlp_pr = np.sum(S2_mlp) ** 2 / (np.sum(S2_mlp ** 2) + 1e-10)
            mlp_cumvar = np.cumsum(S2_mlp) / (np.sum(S2_mlp) + 1e-10)
        else:
            mlp_pr = 1.0
            mlp_cumvar = np.array([1.0])
        
        # --- 两个差分方向的对齐 ---
        if attn_diff_norm > 1e-6 and mlp_diff_norm > 1e-6:
            cos_align = np.dot(attn_diff_mean, mlp_diff_mean) / (attn_diff_norm * mlp_diff_norm)
        else:
            cos_align = 0.0
        
        # --- 差分能量的比较 ---
        total_diff_norm = attn_diff_norm + mlp_diff_norm
        if total_diff_norm > 1e-6:
            attn_frac = attn_diff_norm / total_diff_norm
            mlp_frac = mlp_diff_norm / total_diff_norm
        else:
            attn_frac = mlp_frac = 0.5
        
        layer_result = {
            'attn_diff_norm': float(attn_diff_norm),
            'mlp_diff_norm': float(mlp_diff_norm),
            'attn_diff_pr': float(attn_pr),
            'mlp_diff_pr': float(mlp_pr),
            'cos_alignment': float(cos_align),
            'attn_energy_fraction': float(attn_frac),
            'mlp_energy_fraction': float(mlp_frac),
            'attn_dim_90': int(np.searchsorted(attn_cumvar, 0.9) + 1) if len(attn_cumvar) > 0 else 0,
            'mlp_dim_90': int(np.searchsorted(mlp_cumvar, 0.9) + 1) if len(mlp_cumvar) > 0 else 0,
        }
        
        results[f"L{l}"] = layer_result
        
        print(f"  L{l}: attn_norm={attn_diff_norm:.3f}, mlp_norm={mlp_diff_norm:.3f}, "
              f"attn_frac={attn_frac:.2%}, mlp_frac={mlp_frac:.2%}, "
              f"cos_align={cos_align:.3f}, attn_PR={attn_pr:.1f}, mlp_PR={mlp_pr:.1f}")
    
    # --- 综合分析: 哪个算子主导翻译分离? ---
    print("\n综合分析:")
    if results:
        attn_fracs = [v['attn_energy_fraction'] for v in results.values()]
        mlp_fracs = [v['mlp_energy_fraction'] for v in results.values()]
        cos_aligns = [v['cos_alignment'] for v in results.values()]
        
        print(f"  平均attn能量分数: {np.mean(attn_fracs):.2%}")
        print(f"  平均mlp能量分数: {np.mean(mlp_fracs):.2%}")
        print(f"  平均cos对齐: {np.mean(cos_aligns):.3f}")
        
        # 找MLP主导的层
        mlp_dominant = [k for k, v in results.items() if v['mlp_energy_fraction'] > 0.6]
        attn_dominant = [k for k, v in results.items() if v['attn_energy_fraction'] > 0.6]
        print(f"  MLP主导层: {mlp_dominant}")
        print(f"  Attn主导层: {attn_dominant}")
    
    return results

# ============================================================
# Exp 4: 路径拓扑稳定性
# ============================================================
def exp4_route_topology(gate_acts, n_neurons, n_samples):
    """
    从方向稳定性升级到路径拓扑稳定性
    
    核心思想:
    - 不再问"差分方向是否跨层稳定"(答案是NO, 因为高维正交)
    - 而是问"哪些计算路径被跨层持续激活"
    
    方法:
    1. 对每层, 找出top-k active neurons (翻译差分最大的)
    2. 研究这些neuron的集合随层的演化
    3. 不是看neuron ID是否重合, 而是看:
       - 稀疏模式(sparse pattern)的拓扑相似度
       - 激活分布的排序稳定性
       - 功能路径的持续度
    """
    print("\n" + "="*60)
    print("Exp 4: 路径拓扑稳定性 — 从方向到路径")
    print("="*60)
    
    results = {}
    k_values = [10, 50, 97, 200, 500]
    
    # --- 1. 每层的翻译差分neuron排名 ---
    print("计算每层翻译差分neuron排名...")
    rankings = {}
    diff_magnitudes = {}
    
    for l in sorted(gate_acts['zh'].keys()):
        zh = gate_acts['zh'][l]
        trans = gate_acts['trans'][l]
        
        # 逐样本差分
        diffs = trans - zh  # (n_samples, n_neurons)
        
        # 平均差分幅度
        mean_diff = np.mean(np.abs(diffs), axis=0)  # (n_neurons,)
        
        # 排名 (降序)
        rank = np.argsort(-mean_diff)
        
        rankings[l] = rank
        diff_magnitudes[l] = mean_diff
    
    # --- 2. 排名稳定性 (Spearman correlation) ---
    print("\n排名稳定性 (Spearman correlation between adjacent layers)...")
    sorted_layers_list = sorted(rankings.keys())
    rank_stability = {}
    
    for i in range(len(sorted_layers_list) - 1):
        l1, l2 = sorted_layers_list[i], sorted_layers_list[i+1]
        
        # 使用差分幅度值的Spearman相关
        from scipy.stats import spearmanr
        rho, p = spearmanr(diff_magnitudes[l1], diff_magnitudes[l2])
        
        rank_stability[f"L{l1}_L{l2}"] = {'spearman_rho': float(rho), 'p_value': float(p)}
        print(f"  L{l1}→L{l2}: ρ={rho:.4f}, p={p:.2e}")
    
    # --- 3. Sparse Pattern Overlap (top-k的Jaccard overlap) ---
    print("\nSparse Pattern Overlap (Jaccard)...")
    jaccard_results = {}
    
    for k in k_values:
        for i in range(len(sorted_layers_list) - 1):
            l1, l2 = sorted_layers_list[i], sorted_layers_list[i+1]
            set1 = set(rankings[l1][:k])
            set2 = set(rankings[l2][:k])
            jaccard = len(set1 & set2) / len(set1 | set2)
            
            key = f"L{l1}_L{l2}_k{k}"
            jaccard_results[key] = float(jaccard)
        
        print(f"  k={k}: ", end="")
        for i in range(len(sorted_layers_list) - 1):
            l1, l2 = sorted_layers_list[i], sorted_layers_list[i+1]
            key = f"L{l1}_L{l2}_k{k}"
            print(f"L{l1}→L{l2}={jaccard_results[key]:.3f}  ", end="")
        print()
    
    # --- 4. Null: 随机排名的Jaccard ---
    print("\nNull: 随机排名的期望Jaccard...")
    null_jaccard = {}
    for k in k_values:
        # 随机两个k子集的Jaccard期望
        # E[Jaccard] = k / (2*n - k) where n = n_neurons
        expected = k / (2 * n_neurons - k)
        null_jaccard[str(k)] = float(expected)
        print(f"  k={k}: E[Jaccard_random]={expected:.4f}")
    
    # --- 5. 激活分布的Gini系数 (稀疏度度量) ---
    print("\n激活分布的Gini系数...")
    gini_results = {}
    for l in sorted(gate_acts['zh'].keys()):
        for task in ['zh', 'trans']:
            acts = gate_acts[task][l]
            mean_acts = np.mean(np.abs(acts), axis=0)
            # Gini coefficient
            sorted_acts = np.sort(mean_acts)
            n = len(sorted_acts)
            index = np.arange(1, n + 1)
            gini = (2 * np.sum(index * sorted_acts) / (n * np.sum(sorted_acts) + 1e-10)) - (n + 1) / n
            
            gini_results[f"L{l}_{task}"] = float(gini)
        
        zh_gini = gini_results[f"L{l}_zh"]
        trans_gini = gini_results[f"L{l}_trans"]
        print(f"  L{l}: zh_gini={zh_gini:.3f}, trans_gini={trans_gini:.3f}, diff={trans_gini-zh_gini:+.3f}")
    
    # --- 6. 功能路径持续度 ---
    print("\n功能路径持续度 (path persistence)...")
    # 对每个词对, 追踪其在各层的top-10差分neuron, 看路径是否持续
    path_persistence = defaultdict(list)
    
    for sample_idx in range(min(n_samples, len(gate_acts['zh'][0]))):
        # 追踪该样本在各层的top-10差分neuron
        layer_top10 = {}
        for l in sorted(gate_acts['zh'].keys()):
            zh_act = gate_acts['zh'][l][sample_idx]
            trans_act = gate_acts['trans'][l][sample_idx]
            diff = np.abs(trans_act - zh_act)
            top10 = set(np.argsort(-diff)[:10])
            layer_top10[l] = top10
        
        # 计算相邻层的路径持续度
        sorted_l = sorted(layer_top10.keys())
        for i in range(len(sorted_l) - 1):
            l1, l2 = sorted_l[i], sorted_l[i+1]
            overlap = len(layer_top10[l1] & layer_top10[l2]) / 10.0
            path_persistence[(l1, l2)].append(overlap)
    
    print("路径持续度 (个体样本top-10差分neuron的跨层overlap):")
    persistence_summary = {}
    for (l1, l2) in sorted(path_persistence.keys()):
        overlaps = path_persistence[(l1, l2)]
        persistence_summary[f"L{l1}_L{l2}"] = {
            'mean': float(np.mean(overlaps)),
            'std': float(np.std(overlaps)),
            'p10': float(np.percentile(overlaps, 10)),
            'p90': float(np.percentile(overlaps, 90)),
        }
        print(f"  L{l1}→L{l2}: mean={np.mean(overlaps):.2%}, std={np.std(overlaps):.2%}")
    
    return {
        'rank_stability': rank_stability,
        'jaccard_overlap': jaccard_results,
        'null_jaccard': null_jaccard,
        'gini': gini_results,
        'path_persistence': persistence_summary,
    }

# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen3', choices=['qwen3', 'glm4', 'ds7b'])
    parser.add_argument('--exp', type=int, default=0, help='0=all, 1-4=specific')
    parser.add_argument('--n_pairs', type=int, default=100, help='Number of word pairs')
    args = parser.parse_args()
    
    model_name = args.model
    print(f"Phase 113: 零假设检验 + 子空间动力学 + 算子分解")
    print(f"模型: {model_name}")
    
    # 加载模型
    print("\n加载模型...")
    model, tokenizer, device = load_model(model_name)
    layers = get_layers(model)
    n_layers = len(layers)
    model_info = get_model_info(model, model_name)
    n_neurons = model_info.intermediate_size if model_info.intermediate_size > 0 else 9728
    print(f"模型加载完成, {n_layers}层, intermediate={n_neurons}, device={device}")
    
    # 使用子集
    word_pairs = WORD_PAIRS[:args.n_pairs]
    print(f"使用{len(word_pairs)}个词对")
    
    # 数据收集 (一次性)
    print("\n" + "="*60)
    print("Step 1: 收集MLP gate activations")
    print("="*60)
    gate_acts, n_neurons = collect_gate_activations(model, tokenizer, device, n_layers, word_pairs)
    n_samples = len(word_pairs)
    print(f"收集完成: {n_samples}样本, {n_neurons}维")
    
    print("\n" + "="*60)
    print("Step 2: 收集Attention/MLP分离输出")
    print("="*60)
    attn_mlp_out = collect_attn_mlp_outputs(model, tokenizer, device, n_layers, word_pairs)
    
    # 释放模型
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("模型已释放")
    
    # 运行实验
    all_results = {}
    
    if args.exp in [0, 1]:
        r1 = exp1_null_rotation(gate_acts, n_neurons, n_samples)
        all_results['exp1_null_rotation'] = r1
        
        # 保存
        with open(os.path.join(OUT_DIR, f'phase113_exp1_{model_name}_null_rotation.json'), 'w') as f:
            json.dump(r1, f, indent=2, ensure_ascii=False)
        print("Exp 1 结果已保存")
    
    if args.exp in [0, 2]:
        r2 = exp2_subspace_dynamics(gate_acts, n_neurons, n_samples)
        all_results['exp2_subspace_dynamics'] = r2
        
        with open(os.path.join(OUT_DIR, f'phase113_exp2_{model_name}_subspace_dynamics.json'), 'w') as f:
            json.dump(r2, f, indent=2, ensure_ascii=False)
        print("Exp 2 结果已保存")
    
    if args.exp in [0, 3]:
        r3 = exp3_operator_decomposition(attn_mlp_out, n_samples)
        all_results['exp3_operator_decomposition'] = r3
        
        with open(os.path.join(OUT_DIR, f'phase113_exp3_{model_name}_operator_decomposition.json'), 'w') as f:
            json.dump(r3, f, indent=2, ensure_ascii=False)
        print("Exp 3 结果已保存")
    
    if args.exp in [0, 4]:
        r4 = exp4_route_topology(gate_acts, n_neurons, n_samples)
        all_results['exp4_route_topology'] = r4
        
        with open(os.path.join(OUT_DIR, f'phase113_exp4_{model_name}_route_topology.json'), 'w') as f:
            json.dump(r4, f, indent=2, ensure_ascii=False)
        print("Exp 4 结果已保存")
    
    # 保存完整结果
    with open(os.path.join(OUT_DIR, f'phase113_{model_name}_all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("Phase 113 完成!")
    print("="*60)

if __name__ == '__main__':
    main()
