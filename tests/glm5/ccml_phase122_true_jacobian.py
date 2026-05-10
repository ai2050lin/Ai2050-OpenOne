"""
Phase 122: 真正的局部Jacobian + 轨迹恢复 + 敏感性场
====================================================

基于 Phase 121 批判性分析，当前最关键的三个"生死实验"：

实验1: 真正局部Jacobian ∂h_{l+1}/∂h_l (逐点微分)
  - 使用有限差分 + hook注入，计算单层前向的Jacobian
  - 奇异值谱: 真正的扩张/收缩方向
  - 与Phase 121的协方差输运谱(广义特征值)对比
  - 关键判别: bimodal谱是局部动力学还是协方差输运效应？

实验2: 轨迹恢复测试（吸引子证据）
  - 在关键层注入扰动，测量后续层的扰动演化
  - 扰动方向: spike top-5, complement top-5, random
  - 关键判别: 扰动是否被恢复？(吸引子) 还是持续/扩大？(非吸引子)

实验3: 敏感性场 ∂logit/∂(v^T h_l)
  - 测量输出对不同隐藏状态的敏感性
  - 分解: Causal Effect ≈ Energy × Sensitivity
  - 关键判别: spike方向的敏感性是否更高？

数据量: 100词(Jacobian), 300词(轨迹恢复+敏感性)
模型: Qwen3-4B (主), GLM4-9B (验证), DeepSeek7B (验证)
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc
import json
import argparse
import time
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
from model_utils import load_model, get_layers, get_model_info, release_model, MODEL_CONFIGS

TEMP_DIR = Path('tests/glm5_temp')
TEMP_DIR.mkdir(exist_ok=True)

# ===== 300词数据集 (同Phase 121) =====
SEMANTIC_CATEGORIES = {
    'animals': ['cat', 'dog', 'bird', 'fish', 'lion', 'tiger', 'bear', 'wolf', 'deer', 'fox',
                'eagle', 'snake', 'whale', 'shark', 'rabbit', 'horse', 'cow', 'pig', 'sheep', 'goat',
                'duck', 'goose', 'swan', 'owl', 'crow', 'ant', 'bee', 'fly', 'moth', 'crab'],
    'colors': ['red', 'blue', 'green', 'yellow', 'black', 'white', 'pink', 'purple', 'orange', 'brown',
               'gray', 'gold', 'silver', 'violet', 'crimson', 'scarlet', 'azure', 'teal', 'cyan', 'magenta',
               'ivory', 'amber', 'coral', 'salmon', 'tan', 'beige', 'lime', 'olive', 'navy', 'maroon'],
    'emotions': ['happy', 'sad', 'angry', 'fear', 'love', 'joy', 'hate', 'hope', 'pride', 'shame',
                 'guilt', 'envy', 'calm', 'grief', 'bliss', 'rage', 'dread', 'trust', 'awe', 'disgust',
                 'pity', 'sorrow', 'delight', 'terror', 'serene', 'fury', 'panic', 'despair', 'ecstasy', 'shy'],
    'food': ['bread', 'rice', 'meat', 'fruit', 'cake', 'soup', 'cheese', 'milk', 'wine', 'beer',
             'honey', 'sugar', 'salt', 'pepper', 'flour', 'butter', 'cream', 'olive', 'vinegar', 'ginger',
             'garlic', 'onion', 'potato', 'tomato', 'carrot', 'apple', 'grape', 'lemon', 'mango', 'peach'],
    'body': ['head', 'hand', 'foot', 'eye', 'ear', 'nose', 'mouth', 'heart', 'brain', 'arm',
             'leg', 'back', 'neck', 'chest', 'finger', 'thumb', 'wrist', 'elbow', 'knee', 'ankle',
             'shoulder', 'hip', 'chin', 'cheek', 'brow', 'lip', 'tongue', 'throat', 'lung', 'liver'],
    'weather': ['rain', 'snow', 'wind', 'storm', 'cloud', 'sun', 'fog', 'ice', 'heat', 'cold',
                'frost', 'thunder', 'mist', 'hail', 'dew', 'breeze', 'gale', 'tornado', 'blizzard', 'drought',
                'flood', 'humid', 'arid', 'sleet', 'drizzle', 'overcast', 'clear', 'warm', 'chill', 'frost'],
    'tools': ['hammer', 'knife', 'saw', 'drill', 'wrench', 'chisel', 'pliers', 'ruler', 'compass', 'level',
              'axe', 'shovel', 'mallet', 'clamp', 'vise', 'plane', 'file', 'wedge', 'lever', 'pulley',
              'screwdriver', 'tape', 'glue', 'nail', 'screw', 'bolt', 'nut', 'hinge', 'spring', 'gear'],
    'clothing': ['shirt', 'pants', 'dress', 'coat', 'hat', 'shoe', 'sock', 'glove', 'scarf', 'belt',
                 'jacket', 'skirt', 'boot', 'vest', 'cap', 'tie', 'hood', 'cape', 'robe', 'gown',
                 'blouse', 'sweater', 'cardigan', 'parka', 'raincoat', 'sandal', 'sneaker', 'mitten', 'apron', 'helm'],
    'vehicles': ['car', 'bus', 'train', 'boat', 'plane', 'bike', 'truck', 'ship', 'subway', 'taxi',
                 'van', 'scooter', 'yacht', 'helicopter', 'tram', 'wagon', 'canoe', 'kayak', 'raft', 'ferry',
                 'jet', 'rocket', 'tank', 'tractor', 'bulldozer', 'crane', 'ambulance', 'firetruck', 'sled', 'cart'],
    'buildings': ['house', 'church', 'school', 'tower', 'bridge', 'castle', 'hotel', 'museum', 'palace', 'temple',
                  'factory', 'library', 'prison', 'market', 'barn', 'cabin', 'cottage', 'villa', 'manor', 'fort',
                  'mosque', 'shrine', 'stadium', 'theater', 'hospital', 'warehouse', 'lighthouse', 'dock', 'pier', 'vault'],
}

# 多样化模板 (用于Jacobian计算的输入多样性)
TEMPLATES = [
    'Translate the word "{}" into Chinese.',
    'Define the word "{}" in one sentence.',
    'The {} is',
    'What is a {}?',
]


def get_all_words(n=300):
    """获取前n个词"""
    all_words = []
    for cat, words in SEMANTIC_CATEGORIES.items():
        all_words.extend(words[:30])
    return all_words[:n]


def collect_residuals(model, tokenizer, prompts, device, n_layers, max_batch=10):
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
            'spike_25': Vt[:25],
            'comp_5': Vt[5:10],  # complement top 5 (after spike)
            'comp_25': Vt[25:50],
            'pr': (np.sum(S**2))**2 / (np.sum(S**4) + 1e-10),
            'spike_frac_5': np.sum(S[:5]**2) / (np.sum(S**2) + 1e-10),
            'spike_frac_25': np.sum(S[:25]**2) / (np.sum(S**2) + 1e-10),
        }

    return pca_results


# ============================================================
# Exp 1: 真正局部Jacobian ∂h_{l+1}/∂h_l
# ============================================================
def exp1_true_jacobian(model_name, model, tokenizer, device, pca_results, n_prompts=10, k_probes=100):
    """
    核心问题: 真正的局部Jacobian谱是什么？
    Phase 121的"Jacobian"是协方差输运 Σ_{l+1}/Σ_l，不是真正的∂h_{l+1}/∂h_l

    方法:
    1. 对每个prompt，运行baseline前向，保存h_l和h_{l+1}
    2. 对每个探测向量v，注入扰动εv到h_l，测量h_{l+1}的变化
    3. JvP = (h_{l+1}(h_l + εv) - h_{l+1}(h_l)) / ε
    4. 用k个随机探测向量的JvP做随机SVD，得到奇异值谱

    关键判别:
    - 真Jacobian的奇异值谱是否也是bimodal？
    - 真Jacobian的收缩方向是否对应PCA spike？
    - 与协方差输运谱的差异有多大？
    """
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    layers = get_layers(model)

    print(f"\n{'='*60}")
    print(f"Exp 1: True Local Jacobian ({model_name})")
    print(f"d_model={d_model}, n_layers={n_layers}")
    print(f"n_prompts={n_prompts}, k_probes={k_probes}")
    print(f"{'='*60}")

    # 生成多样化prompts
    all_words = get_all_words(100)
    np.random.seed(42)
    test_prompts = []
    for i in range(n_prompts):
        w = all_words[i * 10]
        t = TEMPLATES[i % len(TEMPLATES)]
        test_prompts.append(t.format(w))

    # 选取关键层
    key_layers = list(range(0, n_layers, 6)) + [n_layers - 1]
    key_layers = sorted(set([l for l in key_layers if l < n_layers]))

    results = {}

    for prompt_idx, prompt in enumerate(test_prompts):
        print(f"\n  Prompt {prompt_idx+1}/{n_prompts}: '{prompt[:50]}...'")

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

        # 对每个关键层计算Jacobian
        for l_idx, l in enumerate(key_layers):
            if l + 1 >= n_layers:
                continue

            print(f"    Layer {l}...", end=' ', flush=True)
            t0 = time.time()

            # h_l的范数 (用于确定扰动步长)
            h_l_base = baseline_hs[l][0, last_pos, :].float()
            h_l_norm = h_l_base.norm().item()

            # 扰动步长: 0.01 * ||h_l|| (小扰动保证线性)
            eps = 0.01 * h_l_norm

            # 生成随机探测向量
            np.random.seed(42 + prompt_idx * 100 + l_idx)
            probe_vectors = np.random.randn(k_probes, d_model)
            # 正交化 (QR分解) 以获得更好的覆盖
            Q, _ = np.linalg.qr(probe_vectors.T)
            probe_vectors = Q.T[:k_probes]  # (k_probes, d_model)

            # 计算JvP: 对每个探测向量v，注入扰动并测量响应
            JvP_matrix = np.zeros((d_model, k_probes))

            for j in range(k_probes):
                v = probe_vectors[j]  # (d_model,)
                v_tensor = torch.tensor(v, dtype=torch.float32, device=device)

                # 在层l注入扰动
                captured = {}

                def perturb_hook(module, input, output, v_in=v_tensor, eps_in=eps):
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

                # JvP = (h_{l+1}^{pert} - h_{l+1}^{base}) / eps
                h_l1_pert = captured[l+1][0, last_pos, :].float().cpu().numpy()
                h_l1_base_np = baseline_hs[l+1][0, last_pos, :].float().cpu().numpy()

                jvp = (h_l1_pert - h_l1_base_np) / eps
                JvP_matrix[:, j] = jvp

                # 清理GPU缓存 (每20个探测向量清理一次)
                if j % 20 == 19:
                    torch.cuda.empty_cache()

            # 随机SVD: JvP_matrix ≈ J @ V，对JvP_matrix做SVD
            # 奇异值 = J的奇异值近似
            U_approx, S_approx, Vt_approx = np.linalg.svd(JvP_matrix, full_matrices=False)

            # 分析奇异值谱
            n_expanding = np.sum(S_approx > 1.0)
            n_contracting = np.sum(S_approx < 1.0)
            n_neutral = np.sum(np.abs(S_approx - 1.0) < 0.01)

            # 与PCA spike方向的对齐
            if l in pca_results:
                V_spike5 = pca_results[l]['spike_5']  # (5, d_model)
                # 检查前5个左奇异向量与spike方向的对齐
                for i in range(min(5, U_approx.shape[1])):
                    u_i = U_approx[:, i]
                    for j in range(5):
                        alignment = abs(np.dot(u_i, V_spike5[j]))

            t1 = time.time()
            print(f"done in {t1-t0:.1f}s, σ_max={S_approx[0]:.4f}, σ_min={S_approx[-1]:.6f}, "
                  f"n_expand={n_expanding}, n_contract={n_contracting}")

            # 保存结果
            if l not in results:
                results[l] = {
                    'singular_values_per_prompt': [],
                    'n_expanding_per_prompt': [],
                    'n_contracting_per_prompt': [],
                }

            results[l]['singular_values_per_prompt'].append(S_approx[:50].tolist())
            results[l]['n_expanding_per_prompt'].append(int(n_expanding))
            results[l]['n_contracting_per_prompt'].append(int(n_contracting))

    # 汇总统计
    summary = {}
    for l, data in results.items():
        sv_arrays = np.array(data['singular_values_per_prompt'])  # (n_prompts, 50)
        mean_sv = np.mean(sv_arrays, axis=0)
        std_sv = np.std(sv_arrays, axis=0)

        summary[l] = {
            'mean_singular_values_top50': mean_sv.tolist(),
            'std_singular_values_top50': std_sv.tolist(),
            'mean_n_expanding': float(np.mean(data['n_expanding_per_prompt'])),
            'mean_n_contracting': float(np.mean(data['n_contracting_per_prompt'])),
            'condition_number': float(mean_sv[0] / (mean_sv[-1] + 1e-10)),
            'singular_value_range': f"[{mean_sv[0]:.4f}, {mean_sv[-1]:.6f}]",
            'n_prompts': n_prompts,
            'k_probes': k_probes,
            'eps_relative': 0.01,
        }

        print(f"\n  L{l} Summary: σ=[{mean_sv[0]:.4f} ... {mean_sv[-1]:.6f}], "
              f"κ={mean_sv[0]/(mean_sv[-1]+1e-10):.1f}, "
              f"n_expand={summary[l]['mean_n_expanding']:.0f}, "
              f"n_contract={summary[l]['mean_n_contracting']:.0f}")

    # 与Phase 121协方差输运谱对比
    print(f"\n  === 对比: True Jacobian vs 协方差输运 ===")
    print(f"  True Jacobian: 局部微分 ∂h_{{l+1}}/∂h_l, 逐点性质")
    print(f"  协方差输运: 全局统计 Σ_{{l+1}}/Σ_l, 分布性质")
    print(f"  如果两者谱形不同 → 协方差输运不是局部动力学的可靠代理")

    return {'per_layer': results, 'summary': summary}


# ============================================================
# Exp 2: 轨迹恢复测试 (吸引子证据)
# ============================================================
def exp2_trajectory_recovery(model_name, model, tokenizer, device, pca_results, n_words=100):
    """
    核心问题: 扰动是否被恢复？(吸引子行为)
    
    这是"吸引子"理论的生死判别实验:
    - 如果扰动在后续层被恢复(距离减小) → 支持吸引子假说
    - 如果扰动持续或扩大 → 不支持吸引子，只是输运
    
    方法:
    1. 在层l注入扰动δ (在spike/complement/random方向)
    2. 测量后续层l+1, l+2, ..., L的扰动大小
    3. distance_ratio(k) = ||δ_{l+k}|| / ||δ_l||
       - ratio < 1: 恢复 (吸引子)
       - ratio > 1: 扩大
       - ratio ≈ 1: 中性输运
    
    关键判别:
    - 不同方向是否恢复速度不同？
    - spike方向是否被特别保护(不恢复)？
    - 中间层(L12-L18)是否有更强的恢复？
    """
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    layers = get_layers(model)

    print(f"\n{'='*60}")
    print(f"Exp 2: Trajectory Recovery ({model_name})")
    print(f"n_words={n_words}")
    print(f"{'='*60}")

    all_words = get_all_words(n_words)
    template = 'Translate the word "{}" into Chinese.'

    # 选取关键层作为扰动注入点
    inject_layers = list(range(0, n_layers, 6)) + [n_layers - 1]
    inject_layers = sorted(set([l for l in inject_layers if l < n_layers]))

    # 扰动方向类型
    direction_types = ['spike_5', 'comp_5', 'random_5']

    # 扰动幅度 (相对于h_l的范数)
    eps_relative = 0.01  # 1%的hidden state范数

    results = {}

    for word_idx, word in enumerate(all_words[:50]):  # 用50个词做轨迹恢复
        prompt = template.format(word)
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
                    baseline_hs[l] = output[0][0, last_pos, :].detach().float().cpu().numpy()
                else:
                    baseline_hs[l] = output[0, last_pos, :].detach().float().cpu().numpy()
            return hook

        hooks_b = [layers[l].register_forward_hook(make_baseline_hook(l)) for l in range(n_layers)]

        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)

        for h in hooks_b:
            h.remove()

        # 对每个注入层和方向类型
        for inject_l in inject_layers:
            if inject_l + 1 >= n_layers:
                continue

            h_l_norm = np.linalg.norm(baseline_hs[inject_l])
            eps = eps_relative * h_l_norm

            for dir_type in direction_types:
                # 获取扰动方向
                if dir_type == 'spike_5' and inject_l in pca_results:
                    # Spike top-5的第一个方向
                    direction = pca_results[inject_l]['spike_5'][0]
                elif dir_type == 'comp_5' and inject_l in pca_results:
                    # Complement的第一个方向 (第6个PCA方向)
                    direction = pca_results[inject_l]['comp_5'][0]
                elif dir_type == 'random_5':
                    np.random.seed(42 + word_idx)
                    direction = np.random.randn(d_model)
                    direction = direction / np.linalg.norm(direction)
                else:
                    continue

                # 注入扰动
                perturbed_hs = {}
                direction_tensor = torch.tensor(direction, dtype=torch.float32, device=device)

                def perturb_hook(module, input, output, d_in=direction_tensor, eps_in=eps):
                    hs = output[0].clone()
                    hs[0, last_pos, :] = hs[0, last_pos, :].float() + eps_in * d_in.to(hs.dtype)
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
                                 for l in range(inject_l + 1, n_layers)]

                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)

                h1.remove()
                for h in capture_hooks:
                    h.remove()

                # 计算距离比率
                delta_initial = eps * direction
                delta_initial_norm = np.linalg.norm(delta_initial)

                distance_ratios = {}
                for l in range(inject_l + 1, n_layers):
                    if l in perturbed_hs and l in baseline_hs:
                        delta_h = perturbed_hs[l] - baseline_hs[l]
                        ratio = np.linalg.norm(delta_h) / (delta_initial_norm + 1e-10)
                        distance_ratios[l] = float(ratio)

                # 保存
                key = f"L{inject_l}_{dir_type}"
                if key not in results:
                    results[key] = {'inject_layer': inject_l, 'direction': dir_type, 'ratios_per_word': []}
                results[key]['ratios_per_word'].append(distance_ratios)

        if word_idx % 10 == 9:
            print(f"  Processed {word_idx+1}/50 words")
            torch.cuda.empty_cache()

    # 汇总统计
    summary = {}
    for key, data in results.items():
        inject_l = data['inject_layer']
        dir_type = data['direction']

        # 对每个后续层，计算平均距离比率
        all_ratios_by_layer = {}
        for ratios in data['ratios_per_word']:
            for l, ratio in ratios.items():
                if l not in all_ratios_by_layer:
                    all_ratios_by_layer[l] = []
                all_ratios_by_layer[l].append(ratio)

        mean_ratios = {}
        for l, ratios in all_ratios_by_layer.items():
            mean_ratios[l] = {
                'mean': float(np.mean(ratios)),
                'std': float(np.std(ratios)),
                'min': float(np.min(ratios)),
                'max': float(np.max(ratios)),
                'n_recovery': int(np.sum(np.array(ratios) < 1.0)),  # 多少词恢复了
                'n_total': len(ratios),
            }

        # 计算最终层的平均比率 (最关键指标)
        final_ratios = [ratios.get(n_layers - 1, np.nan) for ratios in data['ratios_per_word']
                        if n_layers - 1 in ratios]
        final_mean = float(np.nanmean(final_ratios)) if final_ratios else float('nan')

        # 恢复比例 (最终层ratio<1的比例)
        recovery_rate = float(np.mean(np.array(final_ratios) < 1.0)) if final_ratios else 0.0

        summary[key] = {
            'inject_layer': inject_l,
            'direction': dir_type,
            'mean_ratios_by_layer': {str(l): v for l, v in mean_ratios.items()},
            'final_layer_mean_ratio': final_mean,
            'recovery_rate': recovery_rate,  # 最终层扰动被恢复的比例
        }

        print(f"  {key}: final_ratio={final_mean:.4f}, recovery_rate={recovery_rate:.2%}")

    # 跨方向对比
    print(f"\n  === 轨迹恢复: 方向对比 ===")
    for inject_l in inject_layers:
        if inject_l + 1 >= n_layers:
            continue
        spike_key = f"L{inject_l}_spike_5"
        comp_key = f"L{inject_l}_comp_5"
        rand_key = f"L{inject_l}_random_5"

        spike_ratio = summary.get(spike_key, {}).get('final_layer_mean_ratio', float('nan'))
        comp_ratio = summary.get(comp_key, {}).get('final_layer_mean_ratio', float('nan'))
        rand_ratio = summary.get(rand_key, {}).get('final_layer_mean_ratio', float('nan'))

        spike_recovery = summary.get(spike_key, {}).get('recovery_rate', 0)
        comp_recovery = summary.get(comp_key, {}).get('recovery_rate', 0)
        rand_recovery = summary.get(rand_key, {}).get('recovery_rate', 0)

        print(f"  L{inject_l}: spike={spike_ratio:.3f}(recov={spike_recovery:.0%}), "
              f"comp={comp_ratio:.3f}(recov={comp_recovery:.0%}), "
              f"random={rand_ratio:.3f}(recov={rand_recovery:.0%})")

    return {'per_direction': results, 'summary': summary}


# ============================================================
# Exp 3: 敏感性场 ∂logit/∂(v^T h_l)
# ============================================================
def exp3_sensitivity_field(model_name, model, tokenizer, device, pca_results, n_words=100):
    """
    核心问题: 不同方向的输出敏感性如何？
    
    Phase 121发现: Causal Effect ≈ f(Energy)
    但缺失: Causal Effect = Energy × Sensitivity
    
    方法:
    1. 对每个词运行baseline，获取baseline logits
    2. 在层l的v方向上添加扰动εv
    3. 测量输出的KL散度: KL(p(y|h_l + εv) || p(y|h_l))
    4. Sensitivity(v, l) = KL / ε
    
    关键判别:
    - spike方向的Sensitivity是否更高？
    - Energy × Sensitivity 是否能解释Causal Effect？
    - 是否存在"低能高敏感"方向？
    """
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    layers = get_layers(model)

    print(f"\n{'='*60}")
    print(f"Exp 3: Sensitivity Field ({model_name})")
    print(f"n_words={n_words}")
    print(f"{'='*60}")

    all_words = get_all_words(n_words)
    template = 'Translate the word "{}" into Chinese.'

    from scipy.special import softmax as sp_softmax

    # 选取关键层
    key_layers = list(range(0, n_layers, 6)) + [n_layers - 1]
    key_layers = sorted(set([l for l in key_layers if l < n_layers]))

    # 扰动幅度
    eps_relative = 0.01  # 1% of ||h_l||

    # 方向类型: 每个方向分别测试
    direction_configs = [
        ('spike_1', 1),   # top-1 PCA direction
        ('spike_5', 5),   # top-5 PCA subspace
        ('comp_1', 1),    # 第6个PCA方向
        ('comp_5', 5),    # 第6-10个PCA方向
        ('random_1', 1),  # 随机方向
        ('random_5', 5),  # 随机5维子空间
    ]

    results = {}

    for word_idx, word in enumerate(all_words[:100]):
        prompt = template.format(word)
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        mask = attention_mask[0]
        last_pos = mask.nonzero()[-1].item() if len(mask.nonzero()) > 0 else -1

        # Baseline: 获取logits和各层hidden states
        with torch.no_grad():
            baseline_out = model(input_ids=input_ids, attention_mask=attention_mask)
        baseline_logits = baseline_out.logits[0, last_pos, :].float().cpu().numpy()
        baseline_probs = sp_softmax(baseline_logits)

        # 获取各层hidden states (用于确定扰动步长和方向)
        layer_hs = {}

        def make_hs_hook(l):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    layer_hs[l] = output[0][0, last_pos, :].detach().float().cpu().numpy()
                else:
                    layer_hs[l] = output[0, last_pos, :].detach().float().cpu().numpy()
            return hook

        hooks_hs = [layers[l].register_forward_hook(make_hs_hook(l)) for l in range(n_layers)]

        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)

        for h in hooks_hs:
            h.remove()

        # 对每个关键层和方向
        for l in key_layers:
            if l not in layer_hs or l not in pca_results:
                continue

            h_l_norm = np.linalg.norm(layer_hs[l])
            eps = eps_relative * h_l_norm

            for dir_name, n_dirs in direction_configs:
                # 获取扰动方向
                if dir_name == 'spike_1':
                    directions = [pca_results[l]['spike_5'][0]]
                elif dir_name == 'spike_5':
                    directions = list(pca_results[l]['spike_5'])
                elif dir_name == 'comp_1':
                    directions = [pca_results[l]['comp_5'][0]]
                elif dir_name == 'comp_5':
                    directions = list(pca_results[l]['comp_5'])
                elif dir_name == 'random_1':
                    np.random.seed(42 + word_idx)
                    d = np.random.randn(d_model)
                    d = d / np.linalg.norm(d)
                    directions = [d]
                elif dir_name == 'random_5':
                    np.random.seed(42 + word_idx)
                    D = np.random.randn(5, d_model)
                    Q, _ = np.linalg.qr(D.T)
                    directions = list(Q.T[:5])
                else:
                    continue

                # 对每个方向计算敏感性
                kl_values = []
                for d_idx, direction in enumerate(directions):
                    direction_tensor = torch.tensor(direction, dtype=torch.float32, device=device)

                    # 注入扰动
                    def perturb_hook(module, input, output, d_in=direction_tensor, eps_in=eps):
                        hs = output[0].clone()
                        hs[0, last_pos, :] = hs[0, last_pos, :].float() + eps_in * d_in.to(hs.dtype)
                        return (hs,) + output[1:]

                    h1 = layers[l].register_forward_hook(perturb_hook)

                    with torch.no_grad():
                        perturbed_out = model(input_ids=input_ids, attention_mask=attention_mask)

                    h1.remove()

                    perturbed_logits = perturbed_out.logits[0, last_pos, :].float().cpu().numpy()
                    perturbed_probs = sp_softmax(perturbed_logits)

                    # KL divergence
                    kl = np.sum(baseline_probs * np.log(baseline_probs / (perturbed_probs + 1e-10) + 1e-10))
                    kl_values.append(float(kl))

                # 平均KL (对该方向类型的所有方向)
                mean_kl = np.mean(kl_values)

                # 保存
                key = f"L{l}_{dir_name}"
                if key not in results:
                    results[key] = {
                        'layer': l, 'direction': dir_name,
                        'kl_per_word': [], 'energy_per_dim': [],
                    }
                results[key]['kl_per_word'].append(mean_kl)

                # 计算该方向的能量
                if dir_name.startswith('spike') and l in pca_results:
                    energy = pca_results[l].get(f'spike_frac_{n_dirs}', 0)
                elif dir_name.startswith('comp') and l in pca_results:
                    S = pca_results[l]['S']
                    total = np.sum(S**2)
                    if dir_name == 'comp_1':
                        energy = S[5]**2 / (total + 1e-10) if len(S) > 5 else 0
                    else:
                        energy = np.sum(S[5:10]**2) / (total + 1e-10) if len(S) > 10 else 0
                else:
                    energy = n_dirs / d_model  # 随机方向的期望能量密度

                results[key]['energy_per_dim'].append(energy)

        if word_idx % 20 == 19:
            print(f"  Processed {word_idx+1}/100 words")
            torch.cuda.empty_cache()

    # 汇总统计
    summary = {}
    print(f"\n  === 敏感性场: 方向对比 ===")
    print(f"  {'Layer':<6} {'Direction':<12} {'Mean KL':<12} {'Energy/dim':<12} {'KL/Energy':<12}")
    print(f"  {'-'*54}")

    for l in key_layers:
        for dir_name, n_dirs in direction_configs:
            key = f"L{l}_{dir_name}"
            if key not in results:
                continue

            mean_kl = np.mean(results[key]['kl_per_word'])
            std_kl = np.std(results[key]['kl_per_word'])
            mean_energy = np.mean(results[key]['energy_per_dim'])

            kl_per_energy = mean_kl / (mean_energy + 1e-10)

            summary[key] = {
                'layer': l,
                'direction': dir_name,
                'n_dirs': n_dirs,
                'mean_kl': float(mean_kl),
                'std_kl': float(std_kl),
                'mean_energy_per_dim': float(mean_energy),
                'kl_per_energy': float(kl_per_energy),
            }

            print(f"  L{l:<5} {dir_name:<12} {mean_kl:<12.6f} {mean_energy:<12.6f} {kl_per_energy:<12.2f}")

    # 关键对比: spike vs complement vs random 的敏感性
    print(f"\n  === 核心对比: Sensitivity × Energy 分解 ===")
    for l in key_layers:
        spike_key = f"L{l}_spike_5"
        comp_key = f"L{l}_comp_5"
        rand_key = f"L{l}_random_5"

        if spike_key in summary and comp_key in summary and rand_key in summary:
            s_kl = summary[spike_key]['mean_kl']
            c_kl = summary[comp_key]['mean_kl']
            r_kl = summary[rand_key]['mean_kl']
            s_en = summary[spike_key]['mean_energy_per_dim']
            c_en = summary[comp_key]['mean_energy_per_dim']
            r_en = summary[rand_key]['mean_energy_per_dim']

            # Causal Effect ≈ Energy × Sensitivity
            # Sensitivity ≈ KL / Energy
            s_sens = s_kl / (s_en + 1e-10)
            c_sens = c_kl / (c_en + 1e-10)
            r_sens = r_kl / (r_en + 1e-10)

            print(f"  L{l}: spike KL={s_kl:.6f} E={s_en:.6f} S={s_sens:.2f} | "
                  f"comp KL={c_kl:.6f} E={c_en:.6f} S={c_sens:.2f} | "
                  f"rand KL={r_kl:.6f} E={r_en:.6f} S={r_sens:.2f}")
            print(f"       Sensitivity ratio: spike/comp={s_sens/c_sens:.2f}x, "
                  f"spike/random={s_sens/r_sens:.2f}x")

    return {'per_direction': results, 'summary': summary}


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen3',
                        choices=['qwen3', 'glm4', 'deepseek7b'])
    parser.add_argument('--exp', type=int, default=0,
                        help='Experiment (0=all, 1=True Jacobian, 2=Trajectory Recovery, 3=Sensitivity Field)')
    parser.add_argument('--n_prompts', type=int, default=10,
                        help='Number of prompts for Exp 1 (Jacobian)')
    parser.add_argument('--n_words', type=int, default=100,
                        help='Number of words for Exp 2 & 3')
    parser.add_argument('--k_probes', type=int, default=100,
                        help='Number of probe vectors for Jacobian')
    args = parser.parse_args()

    model_name = args.model

    print(f"{'='*60}")
    print(f"Phase 122: True Jacobian + Trajectory Recovery + Sensitivity Field")
    print(f"Model: {model_name}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")

    # Load model
    try:
        model, tokenizer, device = load_model(model_name)
    except Exception as e:
        print(f"ERROR: Failed to load {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return

    info = get_model_info(model, model_name)
    print(f"Model info: {info.model_class}, {info.n_layers} layers, d_model={info.d_model}")

    # Step 0: 收集residuals并计算PCA (共享于所有实验)
    print(f"\n--- Step 0: Collecting residuals and computing PCA ---")
    all_words = get_all_words(300)
    template = 'Translate the word "{}" into Chinese.'
    prompts = [template.format(w) for w in all_words]

    t0 = time.time()
    residuals = collect_residuals(model, tokenizer, prompts, device, info.n_layers, max_batch=10)
    print(f"  Residuals collected in {time.time()-t0:.1f}s")

    pca_results = compute_pca_per_layer(residuals, n_components=50)
    print(f"  PCA computed for {len(pca_results)} layers")

    # Print key PCA info
    for l in sorted(pca_results.keys()):
        if l % 6 == 0 or l == info.n_layers - 1:
            pr = pca_results[l]['pr']
            sf5 = pca_results[l]['spike_frac_5']
            sf25 = pca_results[l]['spike_frac_25']
            print(f"    L{l}: PR={pr:.1f}, spike_frac_5={sf5:.4f}, spike_frac_25={sf25:.4f}")

    all_results = {}

    # Exp 1: True Local Jacobian
    if args.exp in [0, 1]:
        t0 = time.time()
        r1 = exp1_true_jacobian(model_name, model, tokenizer, device, pca_results,
                                n_prompts=args.n_prompts, k_probes=args.k_probes)
        all_results['exp1_true_jacobian'] = r1
        save_path = TEMP_DIR / f"phase122_exp1_{model_name}_true_jacobian.json"
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(r1, f, indent=2, default=str)
        print(f"Exp 1 done in {time.time()-t0:.1f}s, saved to {save_path}")

    # Exp 2: Trajectory Recovery
    if args.exp in [0, 2]:
        t0 = time.time()
        r2 = exp2_trajectory_recovery(model_name, model, tokenizer, device, pca_results,
                                      n_words=args.n_words)
        all_results['exp2_trajectory_recovery'] = r2
        save_path = TEMP_DIR / f"phase122_exp2_{model_name}_trajectory_recovery.json"
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(r2, f, indent=2, default=str)
        print(f"Exp 2 done in {time.time()-t0:.1f}s, saved to {save_path}")

    # Exp 3: Sensitivity Field
    if args.exp in [0, 3]:
        t0 = time.time()
        r3 = exp3_sensitivity_field(model_name, model, tokenizer, device, pca_results,
                                    n_words=args.n_words)
        all_results['exp3_sensitivity_field'] = r3
        save_path = TEMP_DIR / f"phase122_exp3_{model_name}_sensitivity_field.json"
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(r3, f, indent=2, default=str)
        print(f"Exp 3 done in {time.time()-t0:.1f}s, saved to {save_path}")

    # Save all results
    save_path = TEMP_DIR / f"phase122_{model_name}_all_results.json"
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {save_path}")

    # Release model
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()
    print("GPU memory cleared.")


if __name__ == '__main__':
    main()
