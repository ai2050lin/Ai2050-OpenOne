"""
Phase 123: 子空间输运 (Subspace Transport)
==========================================

Phase 122 证明了:
1. 局部Jacobian温和 (σ∈[0.8, 4.5]), 不是极端病态
2. 不存在吸引子 (恢复率=0%, 扰动被放大)
3. 敏感性场近似各向同性 (spike/comp/random KL无显著差异)

但"温和放大器"理论过度简化了:
- 局部σ≈3不代表全局动力学简单 (3^36是天文级)
- 没有吸引子恢复≠没有动力学流形 (可能是中性/不稳定流形)
- 敏感性各向同性的结论可能还在数值噪声底 (差异10^-5级)

Phase 123 的核心转向: 从"能量/敏感性"转向"子空间几何"
=============================================================

核心问题: Transformer 各层之间的子空间输运几何是什么?

实验1: 子空间输运角度
  - 测量 span(V_l) → span(J_l V_l) 的 principal angles
  - V_l: 层l的spike子空间 (PCA top-k方向)
  - J_l V_l: 这些方向经过一层变换后的输出方向
  - 如果 principal angles 小 → 子空间被稳定输运
  - 如果 principal angles 大 → 子空间被旋转/扭曲

实验2: 累积Jacobian有效秩
  - 测量 J_{L:l} = J_L ··· J_l 的 effective rank
  - 这决定长期方向选择性
  - 如果 eff_rank(J_{L:0}) << d_model → 存在低维输运通道
  - 如果 eff_rank(J_{L:0}) ≈ d_model → 全空间均匀输运

实验3: Token类型输运稳定性
  - 不同token类型 (名词/动词/语法) 的子空间输运是否有差异？
  - 语言能力可能来自: 不同信息类型在输运中的稳定性差异
  - 如果名词子空间比动词更稳定 → 语义输运有结构

数据量: 500词 (加大数据量, 避免Phase 122数据量不足的问题)
模型: Qwen3 (主), DS7B (验证), GLM4 (验证) - 逐个测试避免OOM

理论框架: 表征输运理论 (Representation Transport Theory)
  Transformer 不存在显式"语义编码器"或"吸引子流形";
  语言输入诱导出低维高能统计结构,
  网络层间动力学对这些结构进行相对稳定但非恢复性的输运与重分配,
  最终通过残差累积形成输出行为。
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.linalg import subspace_angles, orthogonal_procrustes
from scipy.sparse.linalg import svds

import torch
from model_utils import (load_model, get_layers, get_model_info,
                          release_model, get_W_U, MODEL_CONFIGS)

TEMP_DIR = Path('tests/glm5_temp')
TEMP_DIR.mkdir(exist_ok=True)

# ===== 500词数据集 =====
SEMANTIC_CATEGORIES = {
    'animals': ['cat', 'dog', 'bird', 'fish', 'lion', 'tiger', 'bear', 'wolf', 'deer', 'fox',
                'eagle', 'snake', 'whale', 'shark', 'rabbit', 'horse', 'cow', 'pig', 'sheep', 'goat',
                'duck', 'goose', 'swan', 'owl', 'crow', 'ant', 'bee', 'fly', 'moth', 'crab',
                'leopard', 'cheetah', 'giraffe', 'elephant', 'monkey', 'gorilla', 'dolphin', 'penguin', 'parrot', 'falcon'],
    'colors': ['red', 'blue', 'green', 'yellow', 'black', 'white', 'pink', 'purple', 'orange', 'brown',
               'gray', 'gold', 'silver', 'violet', 'crimson', 'scarlet', 'azure', 'teal', 'cyan', 'magenta',
               'ivory', 'amber', 'coral', 'salmon', 'tan', 'beige', 'lime', 'olive', 'navy', 'maroon',
               'turquoise', 'indigo', 'lavender', 'peach', 'mint', 'cherry', 'ruby', 'sapphire', 'emerald', 'bronze'],
    'emotions': ['happy', 'sad', 'angry', 'fear', 'love', 'joy', 'hate', 'hope', 'pride', 'shame',
                 'guilt', 'envy', 'calm', 'grief', 'bliss', 'rage', 'dread', 'trust', 'awe', 'disgust',
                 'pity', 'sorrow', 'delight', 'terror', 'serene', 'fury', 'panic', 'despair', 'ecstasy', 'shy',
                 'jealousy', 'nostalgia', 'melancholy', 'euphoria', 'anxiety', 'relief', 'contempt', 'adoration', 'resentment', 'gratitude'],
    'food': ['bread', 'rice', 'meat', 'fruit', 'cake', 'soup', 'cheese', 'milk', 'wine', 'beer',
             'honey', 'sugar', 'salt', 'pepper', 'flour', 'butter', 'cream', 'olive', 'vinegar', 'ginger',
             'garlic', 'onion', 'potato', 'tomato', 'carrot', 'apple', 'grape', 'lemon', 'mango', 'peach',
             'pasta', 'sushi', 'taco', 'pizza', 'salad', 'steak', 'bacon', 'sausage', 'mushroom', 'chocolate'],
    'body': ['head', 'hand', 'foot', 'eye', 'ear', 'nose', 'mouth', 'heart', 'brain', 'arm',
             'leg', 'back', 'neck', 'chest', 'finger', 'thumb', 'wrist', 'elbow', 'knee', 'ankle',
             'shoulder', 'hip', 'chin', 'cheek', 'brow', 'lip', 'tongue', 'throat', 'lung', 'liver',
             'spine', 'rib', 'palm', 'toe', 'waist', 'forehead', 'eyelash', 'nostril', 'temple', 'navel'],
    'weather': ['rain', 'snow', 'wind', 'storm', 'cloud', 'sun', 'fog', 'ice', 'heat', 'cold',
                'frost', 'thunder', 'mist', 'hail', 'dew', 'breeze', 'gale', 'tornado', 'blizzard', 'drought',
                'flood', 'humid', 'arid', 'sleet', 'drizzle', 'overcast', 'clear', 'warm', 'chill', 'typhoon',
                'monsoon', 'cyclone', 'hurricane', 'downpour', 'rainbow', 'aurora', 'smog', 'haze', 'vapour', 'whirlwind'],
    'tools': ['hammer', 'knife', 'saw', 'drill', 'wrench', 'chisel', 'pliers', 'ruler', 'compass', 'level',
              'axe', 'shovel', 'mallet', 'clamp', 'vise', 'plane', 'file', 'wedge', 'lever', 'pulley',
              'screwdriver', 'tape', 'glue', 'nail', 'screw', 'bolt', 'nut', 'hinge', 'spring', 'gear',
              'anvil', 'tongs', 'lathe', 'caliper', 'socket', 'welder', 'plumb', 'spade', 'pickaxe', 'scalpel'],
    'clothing': ['shirt', 'pants', 'dress', 'coat', 'hat', 'shoe', 'sock', 'glove', 'scarf', 'belt',
                 'jacket', 'skirt', 'boot', 'vest', 'cap', 'tie', 'hood', 'cape', 'robe', 'gown',
                 'blouse', 'sweater', 'cardigan', 'parka', 'raincoat', 'sandal', 'sneaker', 'mitten', 'apron', 'helm',
                 'corset', 'tunic', 'poncho', 'culottes', 'leggings', 'overalls', 'blazer', 'cummerbund', 'tuxedo', 'kimono'],
    'vehicles': ['car', 'bus', 'train', 'boat', 'plane', 'bike', 'truck', 'ship', 'subway', 'taxi',
                 'van', 'scooter', 'yacht', 'helicopter', 'tram', 'wagon', 'canoe', 'kayak', 'raft', 'ferry',
                 'jet', 'rocket', 'tank', 'tractor', 'bulldozer', 'crane', 'ambulance', 'firetruck', 'sled', 'cart',
                 'glider', 'hovercraft', 'catamaran', 'gondola', 'rickshaw', 'segway', 'skateboard', 'surfboard', 'chariot', 'cablecar'],
    'buildings': ['house', 'church', 'school', 'tower', 'bridge', 'castle', 'hotel', 'museum', 'palace', 'temple',
                  'factory', 'library', 'prison', 'market', 'barn', 'cabin', 'cottage', 'villa', 'manor', 'fort',
                  'mosque', 'shrine', 'stadium', 'theater', 'hospital', 'warehouse', 'lighthouse', 'dock', 'pier', 'vault',
                  'archway', 'basilica', 'chapel', 'monastery', 'skyscraper', 'barracks', 'lodge', 'greenhouse', 'observatory', 'aqueduct'],
    # 新增: 抽象概念类 (测试语法vs语义)
    'abstract': ['freedom', 'justice', 'truth', 'beauty', 'wisdom', 'courage', 'peace', 'chaos', 'order', 'time',
                 'space', 'mind', 'soul', 'spirit', 'faith', 'reason', 'logic', 'science', 'art', 'law',
                 'power', 'nature', 'culture', 'history', 'future', 'destiny', 'virtue', 'honor', 'duty', 'love',
                 'knowledge', 'progress', 'equality', 'liberty', 'democracy', 'harmony', 'infinity', 'eternity', 'reality', 'existence'],
    # 新增: 动作类 (测试动词输运)
    'actions': ['run', 'walk', 'jump', 'swim', 'fly', 'eat', 'drink', 'sleep', 'think', 'speak',
                'write', 'read', 'build', 'destroy', 'create', 'change', 'grow', 'move', 'stop', 'start',
                'push', 'pull', 'lift', 'carry', 'throw', 'catch', 'hold', 'release', 'open', 'close',
                'break', 'fix', 'teach', 'learn', 'give', 'take', 'find', 'lose', 'win', 'fail'],
}

# Token类型分类 (用于Exp 3)
TOKEN_CATEGORIES = {
    'nouns': [],      # 从SEMANTIC_CATEGORIES提取
    'verbs': [],      # 从actions提取
    'adjectives': [], # 从colors/emotions提取
}

# 模板
TEMPLATES = [
    'Translate the word "{}" into Chinese.',
    'Define the word "{}" in one sentence.',
    'The {} is',
    'What is a {}?',
    'Tell me about {}.',
]


def get_all_words(n=500):
    """获取前n个词"""
    all_words = []
    for cat, words in SEMANTIC_CATEGORIES.items():
        all_words.extend(words[:40])
    return all_words[:n]


def classify_token_type(word):
    """粗略分类token类型"""
    if word in SEMANTIC_CATEGORIES.get('actions', []):
        return 'verb'
    elif word in SEMANTIC_CATEGORIES.get('colors', []) + SEMANTIC_CATEGORIES.get('emotions', []):
        return 'adjective'
    else:
        return 'noun'


def collect_residuals_with_hooks(model, tokenizer, prompts, device, n_layers, max_batch=10):
    """对一组prompts收集每层residual stream (last token position)"""
    all_residuals = {}
    
    for l in range(n_layers):
        layer_acts = []
        
        for i in range(0, len(prompts), max_batch):
            batch = prompts[i:i+max_batch]
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=64)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            
            hs = out.hidden_states[l]  # (batch, seq, d_model)
            
            for j in range(len(batch)):
                mask = attention_mask[j]
                non_pad = mask.nonzero()
                last_pos = non_pad[-1].item() if len(non_pad) > 0 else -1
                layer_acts.append(hs[j, last_pos, :].float().cpu().numpy())
        
        all_residuals[l] = np.array(layer_acts)
    
    return all_residuals


def compute_pca_per_layer(residuals, n_components=50):
    """对每层的residuals做PCA，返回spike和complement方向"""
    pca_results = {}
    
    for l, H in residuals.items():
        H_centered = H - H.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(H_centered, full_matrices=False)
        
        pca_results[l] = {
            'U': U,
            'S': S,
            'Vt': Vt,
            'spike_5': Vt[:5],
            'spike_10': Vt[:10],
            'spike_25': Vt[:25],
            'comp_5': Vt[5:10],
            'comp_10': Vt[10:20],
            'comp_25': Vt[25:50],
            'pr': (np.sum(S**2))**2 / (np.sum(S**4) + 1e-10),
            'spike_frac_5': np.sum(S[:5]**2) / (np.sum(S**2) + 1e-10),
            'spike_frac_25': np.sum(S[:25]**2) / (np.sum(S**2) + 1e-10),
        }
    
    return pca_results


def compute_principal_angles(V1, V2):
    """
    计算两个子空间之间的principal angles
    
    Args:
        V1: (d, k1) - 子空间1的基向量
        V2: (d, k2) - 子空间2的基向量
    
    Returns:
        angles: 弧度数组, 长度=min(k1, k2)
    """
    # 确保 V1, V2 是正交的
    Q1, _ = np.linalg.qr(V1)
    Q2, _ = np.linalg.qr(V2)
    
    # SVD of Q1^T Q2
    M = Q1.T @ Q2
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1, 1)
    angles = np.arccos(s)
    
    return angles


def compute_effective_rank(M, threshold=0.99):
    """
    计算矩阵的有效秩 (energy-based)
    
    eff_rank = min k s.t. Σ_{i=1}^{k} σ_i^2 / Σ_{i=1}^{r} σ_i^2 ≥ threshold
    """
    s = np.linalg.svd(M, compute_uv=False)
    total_energy = np.sum(s**2)
    if total_energy < 1e-10:
        return 0
    cum_energy = np.cumsum(s**2)
    k = np.searchsorted(cum_energy / total_energy, threshold) + 1
    return int(k)


# ============================================================
# Exp 1: 子空间输运角度
# ============================================================
def exp1_subspace_transport_angles(model_name, model, tokenizer, device, pca_results, n_prompts=50, k_subspace=10):
    """
    核心问题: spike子空间经过一层变换后, 是否被稳定输运?
    
    方法:
    1. 对每个prompt, 用hook收集h_l和h_{l+1}
    2. 对h_l的spike子空间V_l (PCA top-k方向), 计算J_l V_l
       - J_l V_l通过有限差分: J_l v ≈ (h_{l+1}(h_l+εv) - h_{l+1}(h_l)) / ε
    3. 计算 span(V_l) 和 span(J_l V_l) 之间的 principal angles
    4. 如果 principal angles 小 → 子空间被稳定输运 (旋转少)
    5. 如果 principal angles 大 → 子空间被旋转/扭曲
    
    同时也计算:
    - spike子空间 vs complement子空间 的输运角度对比
    - 不同层的输运角度变化
    """
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    layers = get_layers(model)
    
    print(f"\n{'='*60}")
    print(f"Exp 1: Subspace Transport Angles ({model_name})")
    print(f"d_model={d_model}, n_layers={n_layers}")
    print(f"n_prompts={n_prompts}, k_subspace={k_subspace}")
    print(f"{'='*60}")
    
    # 生成多样化prompts
    all_words = get_all_words(200)
    np.random.seed(42)
    test_prompts = []
    for i in range(n_prompts):
        w = all_words[i * 4]
        t = TEMPLATES[i % len(TEMPLATES)]
        test_prompts.append(t.format(w))
    
    # 选取关键层
    key_layers = list(range(0, n_layers, 4)) + [n_layers - 1]
    key_layers = sorted(set([l for l in key_layers if l < n_layers]))
    
    results = {}
    
    for prompt_idx, prompt in enumerate(test_prompts):
        print(f"\n  Prompt {prompt_idx+1}/{n_prompts}: '{prompt[:50]}...'", flush=True)
        
        # Baseline forward: 保存所有层的hidden states
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        mask = attention_mask[0]
        last_pos = mask.nonzero()[-1].item() if len(mask.nonzero()) > 0 else -1
        
        # 收集baseline
        baseline_hs = {}
        
        def make_baseline_hook(l):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    baseline_hs[l] = output[0].detach().clone()
                else:
                    baseline_hs[l] = output.detach().clone()
            return hook
        
        hooks_b = [layers[l].register_forward_hook(make_baseline_hook(l)) for l in range(n_layers)]
        
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        
        for h in hooks_b:
            h.remove()
        
        # 对每个关键层计算子空间输运
        for l_idx, l in enumerate(key_layers):
            if l + 1 >= n_layers:
                continue
            
            print(f"    Layer {l}...", end=' ', flush=True)
            t0 = time.time()
            
            h_l_base = baseline_hs[l][0, last_pos, :].float()
            h_l_norm = h_l_base.norm().item()
            eps = 0.01 * h_l_norm
            
            # 获取spike和complement子空间方向
            if l not in pca_results:
                print("no PCA, skip")
                continue
            
            V_spike = pca_results[l]['spike_10'].T  # (d_model, 10)
            V_comp = pca_results[l]['comp_10'].T    # (d_model, 10)
            
            # 计算 J_l V_spike 和 J_l V_comp (通过有限差分)
            def compute_JV(V_basis, eps_val):
                """计算 J @ V_basis, 返回 (d_model, k) 矩阵"""
                k = V_basis.shape[1]
                JV = np.zeros((d_model, k))
                
                for j in range(k):
                    v = V_basis[:, j]
                    v_tensor = torch.tensor(v, dtype=torch.float32, device=device)
                    
                    captured = {}
                    
                    def perturb_hook(module, input, output, v_in=v_tensor, eps_in=eps_val):
                        hs = output[0].clone()
                        hs[0, last_pos, :] = hs[0, last_pos, :].float() + eps_in * v_in.to(hs.dtype)
                        return (hs,) + output[1:]
                    
                    def capture_hook(module, input, output, l_next=l+1):
                        if isinstance(output, tuple):
                            captured[l_next] = output[0].detach().clone()
                        else:
                            captured[l_next] = output.detach().clone()
                    
                    h1 = layers[l].register_forward_hook(perturb_hook)
                    h2 = layers[l+1].register_forward_hook(capture_hook)
                    
                    with torch.no_grad():
                        model(input_ids=input_ids, attention_mask=attention_mask)
                    
                    h1.remove()
                    h2.remove()
                    
                    h_l1_pert = captured[l+1][0, last_pos, :].float().cpu().numpy()
                    h_l1_base_np = baseline_hs[l+1][0, last_pos, :].float().cpu().numpy()
                    
                    jvp = (h_l1_pert - h_l1_base_np) / eps_val
                    JV[:, j] = jvp
                
                return JV
            
            JV_spike = compute_JV(V_spike, eps)
            JV_comp = compute_JV(V_comp, eps)
            
            # 计算principal angles
            # spike子空间输运: span(V_spike) vs span(JV_spike)
            angles_spike = compute_principal_angles(V_spike, JV_spike)
            # complement子空间输运: span(V_comp) vs span(JV_comp)
            angles_comp = compute_principal_angles(V_comp, JV_comp)
            # 交叉输运: spike输入 → complement输出?
            angles_spike_to_comp = compute_principal_angles(V_spike, JV_comp)
            
            # 计算Procrustes距离 (最优旋转匹配后的残差)
            R_spike, scale_spike = orthogonal_procrustes(JV_spike, V_spike)
            residual_spike = np.linalg.norm(JV_spike @ R_spike.T - V_spike) / np.linalg.norm(V_spike)
            
            R_comp, scale_comp = orthogonal_procrustes(JV_comp, V_comp)
            residual_comp = np.linalg.norm(JV_comp @ R_comp.T - V_comp) / np.linalg.norm(V_comp)
            
            # 保存结果
            if l not in results:
                results[l] = {
                    'angles_spike': [],      # spike子空间输运角度
                    'angles_comp': [],        # complement子空间输运角度
                    'angles_cross': [],       # 交叉输运角度
                    'procrustes_spike': [],   # spike Procrustes残差
                    'procrustes_comp': [],    # complement Procrustes残差
                    'mean_angle_spike': [],
                    'mean_angle_comp': [],
                    'mean_angle_cross': [],
                }
            
            results[l]['angles_spike'].append(angles_spike.tolist())
            results[l]['angles_comp'].append(angles_comp.tolist())
            results[l]['angles_cross'].append(angles_spike_to_comp.tolist())
            results[l]['procrustes_spike'].append(float(residual_spike))
            results[l]['procrustes_comp'].append(float(residual_comp))
            results[l]['mean_angle_spike'].append(float(np.mean(angles_spike)))
            results[l]['mean_angle_comp'].append(float(np.mean(angles_comp)))
            results[l]['mean_angle_cross'].append(float(np.mean(angles_spike_to_comp)))
            
            t1 = time.time()
            print(f"done in {t1-t0:.1f}s, "
                  f"spike_angle={np.degrees(np.mean(angles_spike)):.1f}°, "
                  f"comp_angle={np.degrees(np.mean(angles_comp)):.1f}°, "
                  f"cross_angle={np.degrees(np.mean(angles_spike_to_comp)):.1f}°")
    
    # 汇总统计
    summary = {}
    for l, data in results.items():
        summary[l] = {
            'mean_angle_spike_deg': float(np.degrees(np.mean([np.mean(a) for a in data['angles_spike']]))),
            'mean_angle_comp_deg': float(np.degrees(np.mean([np.mean(a) for a in data['angles_comp']]))),
            'mean_angle_cross_deg': float(np.degrees(np.mean([np.mean(a) for a in data['angles_cross']]))),
            'std_angle_spike_deg': float(np.degrees(np.std([np.mean(a) for a in data['angles_spike']]))),
            'std_angle_comp_deg': float(np.degrees(np.std([np.mean(a) for a in data['angles_comp']]))),
            'mean_procrustes_spike': float(np.mean(data['procrustes_spike'])),
            'mean_procrustes_comp': float(np.mean(data['procrustes_comp'])),
            'n_prompts': n_prompts,
            'k_subspace': k_subspace,
        }
        
        print(f"\n  L{l} Summary: "
              f"spike={summary[l]['mean_angle_spike_deg']:.1f}°±{summary[l]['std_angle_spike_deg']:.1f}°, "
              f"comp={summary[l]['mean_angle_comp_deg']:.1f}°±{summary[l]['std_angle_comp_deg']:.1f}°, "
              f"cross={summary[l]['mean_angle_cross_deg']:.1f}°, "
              f"proc_spike={summary[l]['mean_procrustes_spike']:.3f}, "
              f"proc_comp={summary[l]['mean_procrustes_comp']:.3f}")
    
    return {'per_layer': results, 'summary': summary}


# ============================================================
# Exp 2: 累积Jacobian有效秩
# ============================================================
def exp2_cumulative_jacobian_effrank(model_name, model, tokenizer, device, n_prompts=20, k_probes=50):
    """
    核心问题: 累积Jacobian J_{L:l} = J_L ··· J_l 的有效秩是什么?
    
    这决定长期方向选择性:
    - 如果 eff_rank(J_{L:0}) << d_model → 存在低维输运通道
    - 如果 eff_rank(J_{L:0}) ≈ d_model → 全空间均匀输运
    
    方法:
    1. 对每个prompt, 用hook逐层注入扰动并追踪传播
    2. 对每个注入层l, 计算J_{L:l}的k个JvP列
    3. 对JvP矩阵做SVD, 计算有效秩
    
    注意: 完整累积Jacobian计算量巨大(O(L * k * d))
    这里用近似: 对每对(l_inject, l_observe), 用有限差分计算J_{l_observe:l_inject}
    """
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    layers = get_layers(model)
    
    print(f"\n{'='*60}")
    print(f"Exp 2: Cumulative Jacobian Effective Rank ({model_name})")
    print(f"d_model={d_model}, n_layers={n_layers}")
    print(f"n_prompts={n_prompts}, k_probes={k_probes}")
    print(f"{'='*60}")
    
    all_words = get_all_words(100)
    np.random.seed(42)
    test_prompts = []
    for i in range(n_prompts):
        w = all_words[i * 5]
        t = TEMPLATES[i % len(TEMPLATES)]
        test_prompts.append(t.format(w))
    
    # 注入层和观察层
    inject_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4]
    inject_layers = [l for l in inject_layers if l < n_layers]
    observe_layers = list(range(0, n_layers, 4)) + [n_layers - 1]
    observe_layers = sorted(set([l for l in observe_layers if l < n_layers]))
    
    results = {}
    
    for prompt_idx, prompt in enumerate(test_prompts):
        print(f"\n  Prompt {prompt_idx+1}/{n_prompts}: '{prompt[:50]}...'", flush=True)
        
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        mask = attention_mask[0]
        last_pos = mask.nonzero()[-1].item() if len(mask.nonzero()) > 0 else -1
        
        # Baseline: 保存所有层的hidden states
        baseline_hs = {}
        
        def make_baseline_hook(l):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    baseline_hs[l] = output[0].detach().clone()
                else:
                    baseline_hs[l] = output.detach().clone()
            return hook
        
        hooks_b = [layers[l].register_forward_hook(make_baseline_hook(l)) for l in range(n_layers)]
        
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        
        for h in hooks_b:
            h.remove()
        
        # 对每个注入层计算累积Jacobian
        for inject_l in inject_layers:
            print(f"    Inject L{inject_l}...", end=' ', flush=True)
            t0 = time.time()
            
            h_inject_norm = baseline_hs[inject_l][0, last_pos, :].float().norm().item()
            eps = 0.01 * h_inject_norm
            
            # 生成随机探测向量
            np.random.seed(42 + prompt_idx * 10 + inject_l)
            probe_vectors = np.random.randn(k_probes, d_model)
            Q, _ = np.linalg.qr(probe_vectors.T)
            probe_vectors = Q.T[:k_probes]
            
            # 对每个观察层, 计算J_{observe:inject}的JvP
            cumulative_JvP = {}  # {observe_l: (d_model, k_probes)}
            
            for j in range(k_probes):
                v = probe_vectors[j]
                v_tensor = torch.tensor(v, dtype=torch.float32, device=device)
                
                # 在inject_l注入扰动, 收集所有后续层的响应
                perturbed_hs = {}
                
                def perturb_hook(module, input, output, v_in=v_tensor, eps_in=eps):
                    hs = output[0].clone()
                    hs[0, last_pos, :] = hs[0, last_pos, :].float() + eps_in * v_in.to(hs.dtype)
                    return (hs,) + output[1:]
                
                def make_capture_hook(l_cap):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            perturbed_hs[l_cap] = output[0][0, last_pos, :].detach().float().cpu().numpy()
                        else:
                            perturbed_hs[l_cap] = output[0, last_pos, :].detach().float().cpu().numpy()
                    return hook
                
                h1 = layers[inject_l].register_forward_hook(perturb_hook)
                capture_hooks = [layers[l].register_forward_hook(make_capture_hook(l))
                                 for l in observe_layers if l > inject_l]
                
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                
                h1.remove()
                for h in capture_hooks:
                    h.remove()
                
                # 计算JvP for each observe layer
                for obs_l in observe_layers:
                    if obs_l <= inject_l:
                        continue
                    if obs_l in perturbed_hs and obs_l in baseline_hs:
                        h_base = baseline_hs[obs_l][0, last_pos, :].float().cpu().numpy()
                        h_pert = perturbed_hs[obs_l]
                        jvp = (h_pert - h_base) / eps
                        
                        if obs_l not in cumulative_JvP:
                            cumulative_JvP[obs_l] = np.zeros((d_model, k_probes))
                        cumulative_JvP[obs_l][:, j] = jvp
                
                if j % 10 == 9:
                    torch.cuda.empty_cache()
            
            # 计算每个观察层的有效秩
            for obs_l, JV_matrix in cumulative_JvP.items():
                eff_rank = compute_effective_rank(JV_matrix)
                s = np.linalg.svd(JV_matrix, compute_uv=False)
                
                key = f"L{inject_l}_to_L{obs_l}"
                if key not in results:
                    results[key] = {
                        'inject_layer': inject_l,
                        'observe_layer': obs_l,
                        'n_hops': obs_l - inject_l,
                        'eff_ranks': [],
                        'top_sv_ratios': [],  # σ_1 / σ_k
                    }
                
                results[key]['eff_ranks'].append(eff_rank)
                if len(s) > 1:
                    results[key]['top_sv_ratios'].append(float(s[0] / (s[min(k_probes-1, len(s)-1)] + 1e-10)))
            
            t1 = time.time()
            print(f"done in {t1-t0:.1f}s")
    
    # 汇总
    summary = {}
    for key, data in results.items():
        summary[key] = {
            'inject_layer': data['inject_layer'],
            'observe_layer': data['observe_layer'],
            'n_hops': data['n_hops'],
            'mean_eff_rank': float(np.mean(data['eff_ranks'])),
            'std_eff_rank': float(np.std(data['eff_ranks'])),
            'mean_sv_ratio': float(np.mean(data['top_sv_ratios'])) if data['top_sv_ratios'] else 0,
            'd_model': d_model,
            'rank_ratio': float(np.mean(data['eff_ranks'])) / d_model,
        }
        
        print(f"  {key}: eff_rank={summary[key]['mean_eff_rank']:.1f}±{summary[key]['std_eff_rank']:.1f} "
              f"({summary[key]['rank_ratio']:.1%} of d_model={d_model}), "
              f"sv_ratio={summary[key]['mean_sv_ratio']:.1f}")
    
    return {'per_pair': results, 'summary': summary}


# ============================================================
# Exp 3: Token类型输运稳定性
# ============================================================
def exp3_token_type_transport(model_name, model, tokenizer, device, n_words_per_type=100, k_probes=30):
    """
    核心问题: 不同token类型(名词/动词/形容词)的子空间输运是否有差异?
    
    假设: 语言能力可能来自不同信息类型在输运中的稳定性差异
    
    方法:
    1. 分别对名词/动词/形容词prompts收集residuals
    2. 对每类, 计算PCA得到子空间
    3. 计算子空间输运角度 (同Exp1, 但按token类型分组)
    4. 对比不同类型的输运稳定性
    
    同时测量:
    - 子空间重叠度 (名词子空间 vs 动量子空间)
    - 跨层子空间一致性
    - 扰动传播速率 (名词扰动 vs 动词扰动)
    """
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    layers = get_layers(model)
    
    print(f"\n{'='*60}")
    print(f"Exp 3: Token Type Transport Stability ({model_name})")
    print(f"n_words_per_type={n_words_per_type}, k_probes={k_probes}")
    print(f"{'='*60}")
    
    # 按类型分组
    nouns = SEMANTIC_CATEGORIES['animals'][:20] + SEMANTIC_CATEGORIES['food'][:20] + \
            SEMANTIC_CATEGORIES['body'][:20] + SEMANTIC_CATEGORIES['buildings'][:20] + \
            SEMANTIC_CATEGORIES['tools'][:20]
    verbs = SEMANTIC_CATEGORIES['actions'][:40]
    adjectives = SEMANTIC_CATEGORIES['colors'][:20] + SEMANTIC_CATEGORIES['emotions'][:20]
    
    word_groups = {
        'noun': nouns[:n_words_per_type],
        'verb': verbs[:n_words_per_type],
        'adjective': adjectives[:n_words_per_type],
    }
    
    results = {}
    
    for type_name, words in word_groups.items():
        print(f"\n  --- {type_name} ({len(words)} words) ---", flush=True)
        
        # 收集该类型的residuals
        template = 'Translate the word "{}" into Chinese.'
        prompts = [template.format(w) for w in words]
        
        residuals = collect_residuals_with_hooks(model, tokenizer, prompts, device, n_layers, max_batch=10)
        
        # PCA
        pca_type = compute_pca_per_layer(residuals, n_components=min(50, len(words)-1))
        
        # 子空间输运角度 (采样关键层)
        key_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
        key_layers = [l for l in key_layers if l < n_layers and l+1 < n_layers]
        
        type_results = {
            'pca_spike_frac': {},  # 每层spike fraction
            'transport_angles': {}, # 每层输运角度
            'subspace_overlap': {}, # 与其他类型的子空间重叠
        }
        
        for l in key_layers:
            if l not in pca_type:
                continue
            
            # Spike fraction
            S = pca_type[l]['S']
            spike_frac = np.sum(S[:5]**2) / (np.sum(S**2) + 1e-10)
            type_results['pca_spike_frac'][l] = float(spike_frac)
            
            # 单层输运角度 (用spike_5方向)
            V_spike = pca_type[l]['spike_5'].T  # (d_model, 5)
            
            # 采样5个prompts计算输运角度
            angle_list = []
            for p_idx in range(min(5, len(prompts))):
                prompt = prompts[p_idx]
                inputs_pt = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64)
                iid = inputs_pt['input_ids'].to(device)
                amask = inputs_pt['attention_mask'].to(device)
                lpos = amask[0].nonzero()[-1].item()
                
                # baseline
                bl = {}
                def mk_bl(lb):
                    def hook(m, i, o):
                        bl[lb] = o[0].detach().clone() if isinstance(o, tuple) else o.detach().clone()
                    return hook
                
                hks = [layers[lb].register_forward_hook(mk_bl(lb)) for lb in range(n_layers)]
                with torch.no_grad():
                    model(input_ids=iid, attention_mask=amask)
                for h in hks: h.remove()
                
                if l not in bl or l+1 not in bl:
                    continue
                
                h_l_norm = bl[l][0, lpos, :].float().norm().item()
                eps = 0.01 * h_l_norm
                
                JV = np.zeros((d_model, 5))
                for j in range(5):
                    v = V_spike[:, j]
                    vt = torch.tensor(v, dtype=torch.float32, device=device)
                    
                    cap = {}
                    def ph(m, i, o, v_in=vt, e=eps):
                        hs = o[0].clone()
                        hs[0, lpos, :] = hs[0, lpos, :].float() + e * v_in.to(hs.dtype)
                        return (hs,) + o[1:]
                    
                    def ch(m, i, o, ln=l+1):
                        cap[ln] = o[0].detach().clone() if isinstance(o, tuple) else o.detach().clone()
                    
                    h1 = layers[l].register_forward_hook(ph)
                    h2 = layers[l+1].register_forward_hook(ch)
                    with torch.no_grad():
                        model(input_ids=iid, attention_mask=amask)
                    h1.remove(); h2.remove()
                    
                    h_pert = cap[l+1][0, lpos, :].float().cpu().numpy()
                    h_base = bl[l+1][0, lpos, :].float().cpu().numpy()
                    JV[:, j] = (h_pert - h_base) / eps
                
                angles = compute_principal_angles(V_spike, JV)
                angle_list.append(np.degrees(np.mean(angles)))
                
                torch.cuda.empty_cache()
            
            type_results['transport_angles'][l] = {
                'mean': float(np.mean(angle_list)) if angle_list else -1,
                'std': float(np.std(angle_list)) if len(angle_list) > 1 else 0,
                'n_samples': len(angle_list),
            }
            
            print(f"    L{l}: spike_frac={spike_frac:.4f}, "
                  f"transport_angle={type_results['transport_angles'][l]['mean']:.1f}°"
                  f"±{type_results['transport_angles'][l]['std']:.1f}°")
        
        results[type_name] = type_results
    
    # 子空间重叠分析
    print(f"\n  --- Subspace Overlap Analysis ---", flush=True)
    overlap_results = {}
    
    # 收集每类的PCA (已在上面计算, 但需要保存)
    # 这里重新计算一次简化版本
    type_pcas = {}
    for type_name, words in word_groups.items():
        template = 'Translate the word "{}" into Chinese.'
        prompts = [template.format(w) for w in words[:50]]
        residuals = collect_residuals_with_hooks(model, tokenizer, prompts, device, n_layers, max_batch=10)
        type_pcas[type_name] = compute_pca_per_layer(residuals, n_components=min(30, len(words)-1))
    
    for l in [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]:
        if l >= n_layers:
            continue
        overlap_results[l] = {}
        types = list(type_pcas.keys())
        for i, t1 in enumerate(types):
            for j, t2 in enumerate(types):
                if i >= j:
                    continue
                if l not in type_pcas[t1] or l not in type_pcas[t2]:
                    continue
                V1 = type_pcas[t1][l]['spike_10']  # (10, d_model)
                V2 = type_pcas[t2][l]['spike_10']  # (10, d_model)
                angles = compute_principal_angles(V1.T, V2.T)
                mean_angle = float(np.degrees(np.mean(angles)))
                overlap_results[l][f'{t1}_vs_{t2}'] = mean_angle
                print(f"    L{l} {t1} vs {t2}: principal angle = {mean_angle:.1f}°")
    
    return {
        'per_type': results,
        'overlap': overlap_results,
    }


# ============================================================
# 主函数
# ============================================================
def run_all_experiments(model_name):
    """对单个模型运行所有实验"""
    print(f"\n{'#'*60}")
    print(f"# Phase 123: Subspace Transport ({model_name})")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'#'*60}")
    
    # 加载模型
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    use_8bit = model_name in ("deepseek7b", "glm4") and gpu_mem_gb < 16
    
    model, tokenizer, device = load_model(model_name, use_8bit=use_8bit)
    info = get_model_info(model, model_name)
    print(f"Model: {info.model_class}, n_layers={info.n_layers}, d_model={info.d_model}")
    
    # 收集residuals并做PCA
    all_words = get_all_words(200)
    template = 'Translate the word "{}" into Chinese.'
    prompts = [template.format(w) for w in all_words[:200]]
    
    print(f"\nCollecting residuals for PCA (200 prompts)...")
    residuals = collect_residuals_with_hooks(model, tokenizer, prompts, device, info.n_layers, max_batch=10)
    pca_results = compute_pca_per_layer(residuals, n_components=50)
    
    # Exp 1: 子空间输运角度
    print(f"\n{'='*60}")
    print("Running Exp 1: Subspace Transport Angles")
    print(f"{'='*60}")
    exp1_results = exp1_subspace_transport_angles(
        model_name, model, tokenizer, device, pca_results,
        n_prompts=30,  # 加大到30个prompts
        k_subspace=10,
    )
    
    # 保存Exp 1结果
    exp1_path = TEMP_DIR / f'phase123_exp1_{model_name}_subspace_angles.json'
    with open(exp1_path, 'w', encoding='utf-8') as f:
        json.dump(exp1_results['summary'], f, indent=2, ensure_ascii=False)
    print(f"Exp 1 saved to {exp1_path}")
    
    # Exp 2: 累积Jacobian有效秩
    print(f"\n{'='*60}")
    print("Running Exp 2: Cumulative Jacobian Effective Rank")
    print(f"{'='*60}")
    exp2_results = exp2_cumulative_jacobian_effrank(
        model_name, model, tokenizer, device,
        n_prompts=15,  # 15个prompts (计算量大)
        k_probes=50,
    )
    
    # 保存Exp 2结果
    exp2_path = TEMP_DIR / f'phase123_exp2_{model_name}_cumulative_jacobian.json'
    with open(exp2_path, 'w', encoding='utf-8') as f:
        json.dump(exp2_results['summary'], f, indent=2, ensure_ascii=False)
    print(f"Exp 2 saved to {exp2_path}")
    
    # Exp 3: Token类型输运稳定性
    print(f"\n{'='*60}")
    print("Running Exp 3: Token Type Transport Stability")
    print(f"{'='*60}")
    exp3_results = exp3_token_type_transport(
        model_name, model, tokenizer, device,
        n_words_per_type=50,  # 每类型50个词
        k_probes=30,
    )
    
    # 保存Exp 3结果
    exp3_path = TEMP_DIR / f'phase123_exp3_{model_name}_token_transport.json'
    with open(exp3_path, 'w', encoding='utf-8') as f:
        # 只保存可序列化的部分
        save_data = {}
        for type_name, data in exp3_results['per_type'].items():
            save_data[type_name] = {
                'pca_spike_frac': data.get('pca_spike_frac', {}),
                'transport_angles': data.get('transport_angles', {}),
            }
        save_data['overlap'] = exp3_results.get('overlap', {})
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"Exp 3 saved to {exp3_path}")
    
    # 释放模型
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()
    
    return {
        'exp1': exp1_results['summary'],
        'exp2': exp2_results['summary'],
        'exp3': exp3_results.get('per_type', {}),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Phase 123: Subspace Transport')
    parser.add_argument('model', type=str, default='qwen3',
                       choices=['qwen3', 'deepseek7b', 'glm4', 'all'],
                       nargs='?', help='Model to test')
    args = parser.parse_args()
    
    if args.model == 'all':
        all_results = {}
        for model_name in ['qwen3', 'deepseek7b', 'glm4']:
            try:
                result = run_all_experiments(model_name)
                all_results[model_name] = result
            except Exception as e:
                print(f"!!! {model_name} failed: {e}")
                import traceback; traceback.print_exc()
                all_results[model_name] = {'error': str(e)}
            
            # 确保GPU完全释放
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(3)
        
        # 保存汇总
        summary_path = TEMP_DIR / 'phase123_all_models_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nAll models summary saved to {summary_path}")
    else:
        run_all_experiments(args.model)
    
    print(f"\n{'#'*60}")
    print(f"# Phase 123 Complete! {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
