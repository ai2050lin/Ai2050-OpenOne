"""
Phase 120: 非线性放大机制 — 1%权重功率如何产生大因果效应？

核心矛盾：
- Phase 119: 所有权重矩阵对spike的功率≈1%（零假设）→ spike不是架构特征
- Phase 115-116: Ablate spike显著改变输出 → spike有强因果效应
- 解释假说: spike的1%功率通过非线性放大(ReLU/Attention)产生了不成比例的因果效应

实验设计：
Exp 1: ReLU放大分析 — MLP中spike方向在激活后占比如何变化？
Exp 2: 逐步因果追踪 — ablate spike在MLP/Attention/Residual的哪个环节产生最大效应？
Exp 3: 激活值spike占比 — 实际激活(不是权重)中spike方向的能量占比
Exp 4: 条件因果效应 — spike在"强语义"vs"弱语义"token上的因果效应差异

模型: Qwen3-4B (主), DeepSeek7B (验证, 如果可以加载)
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

MODEL_CONFIGS = {
    'qwen3': {
        'name': 'Qwen/Qwen3-4B',
        'num_layers': 36,
        'd_model': 2560,
        'intermediate_size': 9728,
    },
    'deepseek7b': {
        'name': 'D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
        'num_layers': 28,
        'd_model': 3584,
        'intermediate_size': 18944,
    }
}

TEMP_DIR = Path(__file__).parent.parent / 'glm5_temp'
TEMP_DIR.mkdir(exist_ok=True)

# 10 semantic categories × 15 words each
SEMANTIC_CATEGORIES = {
    'animals': ['cat', 'dog', 'bird', 'fish', 'lion', 'tiger', 'bear', 'wolf', 'deer', 'fox', 'eagle', 'snake', 'whale', 'shark', 'rabbit'],
    'colors': ['red', 'blue', 'green', 'yellow', 'black', 'white', 'pink', 'purple', 'orange', 'brown', 'gray', 'gold', 'silver', 'violet', 'crimson'],
    'emotions': ['happy', 'sad', 'angry', 'fear', 'love', 'joy', 'hate', 'hope', 'pride', 'shame', 'guilt', 'envy', 'calm', 'grief', 'bliss'],
    'food': ['bread', 'rice', 'meat', 'fish', 'fruit', 'cake', 'soup', 'cheese', 'milk', 'wine', 'beer', 'honey', 'sugar', 'salt', 'pepper'],
    'body': ['head', 'hand', 'foot', 'eye', 'ear', 'nose', 'mouth', 'heart', 'brain', 'arm', 'leg', 'back', 'neck', 'chest', 'finger'],
    'weather': ['rain', 'snow', 'wind', 'storm', 'cloud', 'sun', 'fog', 'ice', 'heat', 'cold', 'frost', 'thunder', 'mist', 'hail', 'dew'],
    'tools': ['hammer', 'knife', 'saw', 'drill', 'wrench', 'chisel', 'pliers', 'ruler', 'compass', 'level', 'axe', 'shovel', 'pliers', 'mallet', 'clamp'],
    'clothing': ['shirt', 'pants', 'dress', 'coat', 'hat', 'shoe', 'sock', 'glove', 'scarf', 'belt', 'jacket', 'skirt', 'boot', 'vest', 'cap'],
    'vehicles': ['car', 'bus', 'train', 'boat', 'plane', 'bike', 'truck', 'ship', 'subway', 'taxi', 'van', 'scooter', 'yacht', 'helicopter', 'tram'],
    'buildings': ['house', 'church', 'school', 'tower', 'bridge', 'castle', 'hotel', 'museum', 'palace', 'temple', 'factory', 'library', 'prison', 'market', 'barn'],
}

TASK_TEMPLATES = {
    'translate': 'Translate the word "{word}" into Chinese.',
    'continue': 'The word "{word}" is related to',
}


def get_device():
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def load_model(model_key):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    config = MODEL_CONFIGS[model_key]
    device = get_device()
    
    if model_key == 'deepseek7b':
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            config['name'], quantization_config=bnb_config,
            device_map='auto', trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config['name'], torch_dtype=torch.bfloat16,
            device_map=device, trust_remote_code=True
        )
    
    tokenizer = AutoTokenizer.from_pretrained(config['name'], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    return model, tokenizer


def collect_residuals(model, tokenizer, words_by_cat, task_key, num_layers, d_model):
    """Collect residual stream activations at each layer for given words and task."""
    import torch
    
    template = TASK_TEMPLATES[task_key]
    all_residuals = {}  # layer -> (n_words, d_model)
    
    # Process in batches of 10
    batch_size = 10
    all_words = []
    word_cats = []
    for cat, words in words_by_cat.items():
        for w in words:
            all_words.append(w)
            word_cats.append(cat)
    
    n_words = len(all_words)
    
    for layer_idx in range(num_layers):
        layer_acts = []
        
        for i in range(0, n_words, batch_size):
            batch_words = all_words[i:i+batch_size]
            prompts = [template.format(word=w) for w in batch_words]
            
            inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True).to(model.device)
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            
            # Get the last token's hidden state at this layer
            hs = outputs.hidden_states[layer_idx]  # (batch, seq_len, d_model)
            
            # Use the last non-padding token
            for j, prompt in enumerate(prompts):
                input_ids = inputs['input_ids'][j]
                # Find last non-pad position
                non_pad = (input_ids != tokenizer.pad_token_id).nonzero()
                if len(non_pad) > 0:
                    last_pos = non_pad[-1].item()
                else:
                    last_pos = -1
                layer_acts.append(hs[j, last_pos, :].cpu().float().numpy())
        
        all_residuals[layer_idx] = np.array(layer_acts)
    
    return all_residuals, word_cats


def compute_spike_dirs(residuals_dict, n_components=25):
    """Compute PCA spike directions from residuals."""
    spike_dirs = {}
    spike_coords = {}
    complements = {}
    
    for l, H in residuals_dict.items():
        H_centered = H - H.mean(axis=0, keepdims=True)
        # SVD for efficiency
        U, S, Vt = np.linalg.svd(H_centered, full_matrices=False)
        V_spike = Vt[:n_components]  # (25, d)
        
        coords = H_centered @ V_spike.T  # (n, 25)
        comp = H_centered - coords @ V_spike  # (n, d)
        
        spike_dirs[l] = V_spike
        spike_coords[l] = coords
        complements[l] = comp
    
    return spike_dirs, spike_coords, complements


# ============================================================
# Exp 1: ReLU放大分析
# ============================================================
def exp1_relu_amplification(model_key, model, tokenizer, residuals, spike_dirs, spike_coords, complements):
    """
    核心问题: MLP中spike方向在ReLU激活前后的能量占比如何变化？
    
    方法:
    1. 对于每个关键层，计算MLP输入(residual)和MLP输出(after ReLU)
    2. 分别在输入和输出中计算spike vs complement的能量占比
    3. 如果ReLU放大了spike，那么输出中spike占比应该>输入中spike占比
    
    但我们没有中间激活，所以用权重分析替代:
    - 计算 gate_proj(V_spike) vs gate_proj(V_complement) 的范数比
    - 计算 SiLU(x_spike) vs SiLU(x_complement) 的平均激活比
    """
    import torch
    
    config = MODEL_CONFIGS[model_key]
    key_layers = [0, 6, 9, 12, 15, 18, 24, 30, 35] if model_key == 'qwen3' else [0, 4, 7, 10, 13, 16, 20, 24, 27]
    key_layers = [l for l in key_layers if l < config['num_layers']]
    
    results = {}
    
    for l in key_layers:
        V_spike = spike_dirs[l]  # (25, d_model)
        H = residuals[l]  # (n_words, d_model)
        H_centered = H - H.mean(axis=0, keepdims=True)
        
        # Decompose activations into spike + complement
        spike_acts = H_centered @ V_spike.T @ V_spike  # Project onto spike subspace
        comp_acts = H_centered - spike_acts
        
        # Compute energy ratios in the residual stream (before MLP)
        spike_energy_before = np.mean(np.sum(spike_acts**2, axis=1))
        comp_energy_before = np.mean(np.sum(comp_acts**2, axis=1))
        spike_frac_before = spike_energy_before / (spike_energy_before + comp_energy_before + 1e-10)
        
        # Now analyze via weight matrices
        mlp = model.model.layers[l].mlp
        
        # gate_proj: (intermediate, d_model)
        W_gate = mlp.gate_proj.weight.detach().cpu().float().numpy()
        W_up = mlp.up_proj.weight.detach().cpu().float().numpy()
        W_down = mlp.down_proj.weight.detach().cpu().float().numpy()
        
        # For each word, compute MLP output
        # MLP(x) = W_down @ SiLU(W_gate @ x) * (W_up @ x)
        # But we can't do this efficiently for all intermediate neurons
        
        # Instead, compute the "gain" ratio:
        # For gate_proj, how does the norm of output change for spike vs complement input?
        
        # gate_proj(spike_acts): each spike component produces intermediate activations
        gate_spike = spike_acts @ W_gate.T  # (n_words, intermediate)
        gate_comp = comp_acts @ W_gate.T    # (n_words, intermediate)
        
        up_spike = spike_acts @ W_up.T
        up_comp = comp_acts @ W_up.T
        
        # SiLU activation analysis
        # SiLU(x) = x * sigmoid(x)
        # For small x, SiLU(x) ≈ x/2; for large x, SiLU(x) ≈ x
        # The key question: after SiLU, is spike's relative contribution amplified?
        
        # Compute pre-activation statistics
        gate_spike_norm = np.mean(np.sum(gate_spike**2, axis=1))
        gate_comp_norm = np.mean(np.sum(gate_comp**2, axis=1))
        
        up_spike_norm = np.mean(np.sum(up_spike**2, axis=1))
        up_comp_norm = np.mean(np.sum(up_comp**2, axis=1))
        
        # Compute the "amplification" = ratio of spike fraction after gate vs before
        spike_frac_gate = gate_spike_norm / (gate_spike_norm + gate_comp_norm + 1e-10)
        spike_frac_up = up_spike_norm / (up_spike_norm + up_comp_norm + 1e-10)
        
        gate_amplification = spike_frac_gate / (spike_frac_before + 1e-10)
        up_amplification = spike_frac_up / (spike_frac_before + 1e-10)
        
        # Now compute SiLU output analytically
        # SiLU(gate) * up, where gate and up are per-neuron
        # The MLP output for each hidden neuron i:
        #   SiLU(gate_i) * up_i
        # We need to understand how spike vs comp contribute to each gate_i and up_i
        
        # Full MLP output analysis: compute actual MLP output and decompose
        # This requires running the model, but we can approximate
        
        # Instead: compute W_down write-back analysis
        # MLP output = W_down @ h_mlp where h_mlp = SiLU(gate) * up
        # W_down writes to spike subspace with fraction:
        W_down_to_spike = V_spike @ W_down  # (25, intermediate)
        down_spike_power = np.sum(W_down_to_spike**2) / np.sum(W_down**2)
        down_comp_power = 1 - down_spike_power
        
        results[l] = {
            'spike_frac_before': float(spike_frac_before),
            'spike_frac_gate': float(spike_frac_gate),
            'spike_frac_up': float(spike_frac_up),
            'gate_amplification': float(gate_amplification),
            'up_amplification': float(up_amplification),
            'gate_spike_norm': float(gate_spike_norm),
            'gate_comp_norm': float(gate_comp_norm),
            'up_spike_norm': float(up_spike_norm),
            'up_comp_norm': float(up_comp_norm),
            'down_spike_power': float(down_spike_power),
            'down_comp_power': float(down_comp_power),
            'spike_energy_before': float(spike_energy_before),
            'comp_energy_before': float(comp_energy_before),
        }
        
        print(f"  L{l}: spike_frac before={spike_frac_before:.4f}, "
              f"gate={spike_frac_gate:.4f}(amp={gate_amplification:.2f}x), "
              f"up={spike_frac_up:.4f}(amp={up_amplification:.2f}x), "
              f"down_spike={down_spike_power:.4f}")
    
    return results


# ============================================================
# Exp 2: 逐步因果追踪 — 激活值中spike的实际能量占比
# ============================================================
def exp2_activation_spike_fraction(model_key, model, tokenizer, residuals, spike_dirs):
    """
    核心问题: 在实际激活值(不是权重)中，spike方向的能量占比是多少？
    
    Phase 119发现权重功率≈1%，但激活值可能不同，因为：
    1. 输入分布导致spike方向的激活值幅度更大
    2. 这就是"统计集中效应"的另一种表现
    
    方法:
    1. 对每层的residual stream，投影到spike子空间
    2. 计算spike分量 vs complement分量的能量比
    3. 这给出了spike在"实际计算"中的真实占比
    """
    config = MODEL_CONFIGS[model_key]
    key_layers = list(range(0, config['num_layers'], 3)) + [config['num_layers'] - 1]
    key_layers = sorted(set([l for l in key_layers if l < config['num_layers']]))
    
    results = {}
    
    for l in key_layers:
        V_spike = spike_dirs[l]  # (25, d_model)
        H = residuals[l]  # (n_words, d_model)
        H_centered = H - H.mean(axis=0, keepdims=True)
        
        # Project onto spike subspace
        spike_component = H_centered @ V_spike.T @ V_spike  # (n, d)
        comp_component = H_centered - spike_component  # (n, d)
        
        # Energy
        spike_energy = np.mean(np.sum(spike_component**2, axis=1))
        comp_energy = np.mean(np.sum(comp_component**2, axis=1))
        total_energy = spike_energy + comp_energy
        
        # Per-word spike fraction
        word_spike_frac = np.sum(spike_component**2, axis=1) / (np.sum(H_centered**2, axis=1) + 1e-10)
        
        # Per-dimension analysis
        spike_dim_energy = np.mean(np.sum(spike_component**2, axis=0))  # Average energy per spike dim
        comp_dim_energy = np.mean(np.sum(comp_component**2, axis=0)) / (comp_component.shape[1])  # Average energy per comp dim
        dim_density_ratio = spike_dim_energy / (comp_dim_energy + 1e-10)
        
        # Participation Ratio of full representation
        cov = (H_centered.T @ H_centered) / H_centered.shape[0]
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        pr = np.sum(eigenvalues)**2 / (np.sum(eigenvalues**2) + 1e-10)
        
        results[l] = {
            'spike_frac': float(spike_energy / (total_energy + 1e-10)),
            'spike_energy': float(spike_energy),
            'comp_energy': float(comp_energy),
            'word_spike_frac_mean': float(np.mean(word_spike_frac)),
            'word_spike_frac_std': float(np.std(word_spike_frac)),
            'word_spike_frac_min': float(np.min(word_spike_frac)),
            'word_spike_frac_max': float(np.max(word_spike_frac)),
            'spike_dim_energy': float(spike_dim_energy),
            'comp_dim_energy': float(comp_dim_energy),
            'dim_density_ratio': float(dim_density_ratio),
            'participation_ratio': float(pr),
            'n_spike_dims': 25,
            'n_comp_dims': int(H.shape[1] - 25),
        }
        
        print(f"  L{l}: spike_frac={spike_energy/(total_energy+1e-10):.4f}, "
              f"PR={pr:.1f}, dim_density={dim_density_ratio:.1f}x, "
              f"word_frac=[{np.min(word_spike_frac):.3f}, {np.max(word_spike_frac):.3f}]")
    
    return results


# ============================================================
# Exp 3: 逐层干预 — ablate spike在不同层的因果效应
# ============================================================
def exp3_layerwise_ablation(model_key, model, tokenizer, residuals, spike_dirs, word_cats):
    """
    核心问题: 在哪一层ablate spike产生最大因果效应？
    
    方法:
    1. 对每层，在residual stream中移除spike分量
    2. 计算对最终logit的影响
    3. 这直接测量spike的因果效应，且与层的位置相关
    
    关键: 与Phase 115-116的区别是，这里我们用更系统的方法
    """
    import torch
    
    config = MODEL_CONFIGS[model_key]
    n_layers = config['num_layers']
    
    # Select a subset of words for ablation (to save time)
    all_words = []
    selected_cats = []
    for cat, words in SEMANTIC_CATEGORIES.items():
        for w in words[:5]:  # 5 words per category = 50 total
            all_words.append(w)
            selected_cats.append(cat)
    
    task = 'translate'
    template = TASK_TEMPLATES[task]
    
    # Key layers for ablation
    key_layers = list(range(0, n_layers, 3)) + [n_layers - 1]
    key_layers = sorted(set([l for l in key_layers if l < n_layers]))
    
    results = {}
    
    for l in key_layers:
        V_spike = spike_dirs[l]  # (25, d_model)
        V_spike_t = torch.tensor(V_spike, dtype=torch.float32, device=model.device)
        
        logit_diffs = []
        
        for word in all_words:
            prompt = template.format(word=word)
            inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to(model.device)
            
            # Get baseline logit
            with torch.no_grad():
                baseline_out = model(**inputs, output_hidden_states=True)
            
            # Get the position of last non-pad token
            input_ids = inputs['input_ids'][0]
            non_pad = (input_ids != tokenizer.pad_token_id).nonzero()
            last_pos = non_pad[-1].item() if len(non_pad) > 0 else -1
            
            baseline_logit = baseline_out.logits[0, last_pos, :].cpu().float().numpy()
            
            # Ablate spike at layer l using hook
            def ablation_hook(module, input, output):
                # output is a tuple: (hidden_states, ...)
                hs = output[0]  # (batch, seq_len, d_model)
                hs_float = hs.float()
                # Project out spike component
                spike_proj = hs_float @ V_spike_t.T @ V_spike_t  # Project onto spike
                hs_ablated = (hs_float - spike_proj).to(hs.dtype)
                return (hs_ablated,) + output[1:]
            
            # Register hook at layer l
            hook = model.model.layers[l].register_forward_hook(ablation_hook)
            
            with torch.no_grad():
                ablated_out = model(**inputs)
            
            hook.remove()
            
            ablated_logit = ablated_out.logits[0, last_pos, :].cpu().float().numpy()
            
            # KL divergence between baseline and ablated
            from scipy.special import softmax
            p = softmax(baseline_logit)
            q = softmax(ablated_logit)
            kl = np.sum(p * np.log(p / (q + 1e-10) + 1e-10))
            
            # L2 difference
            l2_diff = np.linalg.norm(baseline_logit - ablated_logit)
            
            logit_diffs.append({
                'word': word,
                'category': selected_cats[all_words.index(word)],
                'kl_divergence': float(kl),
                'l2_diff': float(l2_diff),
            })
        
        # Aggregate
        mean_kl = np.mean([d['kl_divergence'] for d in logit_diffs])
        mean_l2 = np.mean([d['l2_diff'] for d in logit_diffs])
        
        # Per-category analysis
        cat_kl = {}
        for cat in set(selected_cats):
            cat_words = [d for d in logit_diffs if d['category'] == cat]
            cat_kl[cat] = float(np.mean([d['kl_divergence'] for d in cat_words]))
        
        results[l] = {
            'mean_kl': float(mean_kl),
            'mean_l2': float(mean_l2),
            'per_category_kl': cat_kl,
            'per_word': logit_diffs,
        }
        
        print(f"  L{l}: mean_KL={mean_kl:.6f}, mean_L2={mean_l2:.4f}")
    
    return results


# ============================================================
# Exp 4: 条件因果效应 — spike在不同语义强度下的效应
# ============================================================
def exp4_conditional_causal_effect(model_key, model, tokenizer, residuals, spike_dirs, word_cats):
    """
    核心问题: spike的因果效应是否取决于"语义强度"？
    
    假说: 如果spike是统计集中效应，那么：
    - 对"典型"词（高语义密度）→ spike占比高 → ablate效应大
    - 对"非典型"词（低语义密度）→ spike占比低 → ablate效应小
    
    方法:
    1. 计算每个词在spike子空间中的"密度"（spike方向上的能量占比）
    2. 按密度分组，测量每组的ablation因果效应
    3. 如果存在正相关，说明spike的因果效应来自统计集中
    """
    import torch
    
    config = MODEL_CONFIGS[model_key]
    
    # Use L12 (the spike layer) for this analysis
    target_layer = min(12, config['num_layers'] - 1)
    
    V_spike = spike_dirs[target_layer]
    V_spike_t = torch.tensor(V_spike, dtype=torch.float32, device=model.device)
    
    H = residuals[target_layer]
    H_centered = H - H.mean(axis=0, keepdims=True)
    
    # Compute spike density per word
    spike_component = H_centered @ V_spike.T @ V_spike
    word_spike_frac = np.sum(spike_component**2, axis=1) / (np.sum(H_centered**2, axis=1) + 1e-10)
    
    # Split into 3 groups: high/medium/low spike density
    percentiles = np.percentile(word_spike_frac, [33, 67])
    high_mask = word_spike_frac >= percentiles[1]
    low_mask = word_spike_frac < percentiles[0]
    mid_mask = ~high_mask & ~low_mask
    
    # For each group, measure ablation effect
    all_words = []
    for cat, words in SEMANTIC_CATEGORIES.items():
        all_words.extend(words)
    
    task = 'translate'
    template = TASK_TEMPLATES[task]
    
    groups = {'high': high_mask, 'mid': mid_mask, 'low': low_mask}
    results = {}
    
    for group_name, mask in groups.items():
        group_spike_frac = word_spike_frac[mask]
        
        # Select up to 15 words from this group
        group_indices = np.where(mask)[0][:15]
        if len(group_indices) == 0:
            continue
        
        # We need the actual words for this group
        # Since we used all_words in collect_residuals, map indices back
        kl_divs = []
        
        for idx in group_indices:
            if idx >= len(all_words):
                continue
            word = all_words[idx]
            prompt = template.format(word=word)
            inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to(model.device)
            
            with torch.no_grad():
                baseline_out = model(**inputs, output_hidden_states=True)
            
            input_ids = inputs['input_ids'][0]
            non_pad = (input_ids != tokenizer.pad_token_id).nonzero()
            last_pos = non_pad[-1].item() if len(non_pad) > 0 else -1
            baseline_logit = baseline_out.logits[0, last_pos, :].cpu().float().numpy()
            
            def ablation_hook(module, input, output):
                hs = output[0]
                hs_float = hs.float()
                spike_proj = hs_float @ V_spike_t.T @ V_spike_t
                hs_ablated = (hs_float - spike_proj).to(hs.dtype)
                return (hs_ablated,) + output[1:]
            
            hook = model.model.layers[target_layer].register_forward_hook(ablation_hook)
            
            with torch.no_grad():
                ablated_out = model(**inputs)
            
            hook.remove()
            
            ablated_logit = ablated_out.logits[0, last_pos, :].cpu().float().numpy()
            
            from scipy.special import softmax
            p = softmax(baseline_logit)
            q = softmax(ablated_logit)
            kl = np.sum(p * np.log(p / (q + 1e-10) + 1e-10))
            kl_divs.append(float(kl))
        
        results[group_name] = {
            'n_words': len(kl_divs),
            'mean_spike_frac': float(np.mean(group_spike_frac)),
            'mean_kl': float(np.mean(kl_divs)) if kl_divs else 0,
            'std_kl': float(np.std(kl_divs)) if kl_divs else 0,
        }
        
        print(f"  {group_name}: spike_frac={np.mean(group_spike_frac):.4f}, "
              f"mean_KL={np.mean(kl_divs) if kl_divs else 0:.6f}")
    
    # Correlation between spike_frac and KL
    # For this we need per-word KL, recompute for all words
    all_kl = []
    all_spike_frac = []
    
    for i, word in enumerate(all_words):
        if i >= len(word_spike_frac):
            break
        prompt = template.format(word=word)
        inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            baseline_out = model(**inputs)
        
        input_ids = inputs['input_ids'][0]
        non_pad = (input_ids != tokenizer.pad_token_id).nonzero()
        last_pos = non_pad[-1].item() if len(non_pad) > 0 else -1
        baseline_logit = baseline_out.logits[0, last_pos, :].cpu().float().numpy()
        
        def ablation_hook(module, input, output):
            hs = output[0]
            hs_float = hs.float()
            spike_proj = hs_float @ V_spike_t.T @ V_spike_t
            return ((hs_float - spike_proj).to(hs.dtype),) + output[1:]
        
        hook = model.model.layers[target_layer].register_forward_hook(ablation_hook)
        
        with torch.no_grad():
            ablated_out = model(**inputs)
        
        hook.remove()
        
        ablated_logit = ablated_out.logits[0, last_pos, :].cpu().float().numpy()
        
        from scipy.special import softmax
        p = softmax(baseline_logit)
        q = softmax(ablated_logit)
        kl = float(np.sum(p * np.log(p / (q + 1e-10) + 1e-10)))
        
        all_kl.append(kl)
        all_spike_frac.append(float(word_spike_frac[i]))
        
        if (i+1) % 20 == 0:
            print(f"    Processed {i+1}/{min(len(all_words), len(word_spike_frac))} words")
    
    # Pearson correlation
    if len(all_kl) > 2:
        from scipy.stats import pearsonr, spearmanr
        pearson_r, pearson_p = pearsonr(all_spike_frac, all_kl)
        spearman_r, spearman_p = spearmanr(all_spike_frac, all_kl)
    else:
        pearson_r = pearson_p = spearman_r = spearman_p = 0
    
    results['correlation'] = {
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
        'n_words': len(all_kl),
    }
    
    print(f"  Correlation: pearson_r={pearson_r:.4f}(p={pearson_p:.4f}), "
          f"spearman_r={spearman_r:.4f}(p={spearman_p:.4f})")
    
    return results


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen3', choices=['qwen3', 'deepseek7b'])
    parser.add_argument('--exp', type=int, default=0, help='Experiment number (0=all, 1-4)')
    args = parser.parse_args()
    
    model_key = args.model
    config = MODEL_CONFIGS[model_key]
    
    print(f"=== Phase 120: Nonlinear Amplification ({model_key}) ===")
    print(f"Exp: {args.exp}")
    
    # Load model and collect residuals
    print("\nLoading model and collecting residuals...")
    model, tokenizer = load_model(model_key)
    
    # Use all categories with 15 words each = 150 words
    task = 'translate'
    print(f"Collecting residuals for task: {task}")
    residuals, word_cats = collect_residuals(
        model, tokenizer, SEMANTIC_CATEGORIES, task,
        config['num_layers'], config['d_model']
    )
    
    print(f"Computing spike directions...")
    spike_dirs, spike_coords, complements = compute_spike_dirs(residuals, n_components=25)
    
    all_results = {}
    
    # Exp 1: ReLU amplification
    if args.exp in [0, 1]:
        print("\n--- Exp 1: ReLU Amplification Analysis ---")
        r1 = exp1_relu_amplification(model_key, model, tokenizer, residuals, spike_dirs, spike_coords, complements)
        all_results['exp1_relu'] = r1
        save_path = TEMP_DIR / f"phase120_exp1_{model_key}_relu.json"
        with open(save_path, 'w') as f:
            json.dump(r1, f, indent=2, default=str)
        print(f"Saved to {save_path}")
    
    # Exp 2: Activation spike fraction
    if args.exp in [0, 2]:
        print("\n--- Exp 2: Activation Spike Fraction ---")
        r2 = exp2_activation_spike_fraction(model_key, model, tokenizer, residuals, spike_dirs)
        all_results['exp2_activation'] = r2
        save_path = TEMP_DIR / f"phase120_exp2_{model_key}_activation.json"
        with open(save_path, 'w') as f:
            json.dump(r2, f, indent=2, default=str)
        print(f"Saved to {save_path}")
    
    # Exp 3: Layerwise ablation
    if args.exp in [0, 3]:
        print("\n--- Exp 3: Layerwise Spike Ablation ---")
        r3 = exp3_layerwise_ablation(model_key, model, tokenizer, residuals, spike_dirs, word_cats)
        all_results['exp3_ablation'] = r3
        save_path = TEMP_DIR / f"phase120_exp3_{model_key}_ablation.json"
        with open(save_path, 'w') as f:
            json.dump(r3, f, indent=2, default=str)
        print(f"Saved to {save_path}")
    
    # Exp 4: Conditional causal effect
    if args.exp in [0, 4]:
        print("\n--- Exp 4: Conditional Causal Effect ---")
        r4 = exp4_conditional_causal_effect(model_key, model, tokenizer, residuals, spike_dirs, word_cats)
        all_results['exp4_conditional'] = r4
        save_path = TEMP_DIR / f"phase120_exp4_{model_key}_conditional.json"
        with open(save_path, 'w') as f:
            json.dump(r4, f, indent=2, default=str)
        print(f"Saved to {save_path}")
    
    # Save all results
    save_path = TEMP_DIR / f"phase120_{model_key}_all_results.json"
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {save_path}")
    
    # Clean up
    import torch
    del model
    torch.cuda.empty_cache()
    print("GPU memory cleared.")


if __name__ == '__main__':
    main()
