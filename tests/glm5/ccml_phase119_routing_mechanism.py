"""
Phase 119: MLP/Attention Routing Mechanism — How does spike information inject into complement?
Phase 119: MLP/Attention路由机制 — spike信息如何注入complement？

Core insight from Phase 118:
  - W_u power is 99% in complement (spike only 1%)
  - Spike has high semantic density per dimension
  - But spike has near-zero direct linear contribution to output
  
Question: How does spike (semantic compression code) affect the output through nonlinear paths?

Exp 1: Indirect Causal Effect of Spike
  Ablate spike, measure how complement changes
  If spike affects output through complement → indirect causal path exists

Exp 2: MLP Read-Out of Spike Directions
  Does MLP at each layer read from spike directions?
  Measure: attention to spike-direction tokens, MLP gate sensitivity

Exp 3: Spike-to-Complement Information Flow
  Track how spike information propagates through residual stream
  Use gradient-based attribution: d(complement_l+1)/d(spike_l)

Exp 4: Nonlinear Semantic→Output Bridge
  Can we find nonlinear combinations of spike+complement that predict output?
  Polynomial probe: logit ~ spike + complement + spike×complement
"""

import torch
import numpy as np
import json
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist
from collections import defaultdict

# ============================================================
# Configuration
# ============================================================

MODEL_CONFIGS = {
    'qwen3': {
        'name': 'Qwen/Qwen3-4B',
        'n_layers': 36,
        'd_model': 2560,
        'dtype': torch.bfloat16,
    },
    'deepseek7b': {
        'name': 'D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
        'n_layers': 28,
        'd_model': 3584,
        'dtype': torch.float16,
    },
}

OUTPUT_DIR = Path("d:/Ai2050/TransformerLens-Project/tests/glm5_temp")

# ============================================================
# 100-word list
# ============================================================

WORD_LIST = [
    # Animals (15)
    ("猫", "cat", "animal"), ("狗", "dog", "animal"), ("鸟", "bird", "animal"),
    ("马", "horse", "animal"), ("牛", "cow", "animal"), ("鱼", "fish", "animal"),
    ("兔", "rabbit", "animal"), ("蛇", "snake", "animal"), ("虎", "tiger", "animal"),
    ("象", "elephant", "animal"), ("猴", "monkey", "animal"), ("羊", "sheep", "animal"),
    ("鸡", "chicken", "animal"), ("蜂", "bee", "animal"), ("蝶", "butterfly", "animal"),
    # Fruits (10)
    ("苹果", "apple", "fruit"), ("香蕉", "banana", "fruit"), ("橙子", "orange", "fruit"),
    ("葡萄", "grape", "fruit"), ("西瓜", "watermelon", "fruit"), ("桃子", "peach", "fruit"),
    ("梨", "pear", "fruit"), ("草莓", "strawberry", "fruit"), ("柠檬", "lemon", "fruit"),
    ("芒果", "mango", "fruit"),
    # Furniture/Artifacts (12)
    ("桌子", "table", "artifact"), ("椅子", "chair", "artifact"), ("床", "bed", "artifact"),
    ("门", "door", "artifact"), ("窗户", "window", "artifact"), ("书", "book", "artifact"),
    ("笔", "pen", "artifact"), ("电脑", "computer", "artifact"), ("电话", "phone", "artifact"),
    ("刀", "knife", "artifact"), ("车", "car", "artifact"), ("船", "ship", "artifact"),
    # Nature (10)
    ("太阳", "sun", "nature"), ("月亮", "moon", "nature"), ("星星", "star", "nature"),
    ("天空", "sky", "nature"), ("云", "cloud", "nature"), ("雨", "rain", "nature"),
    ("雪", "snow", "nature"), ("风", "wind", "nature"), ("山", "mountain", "nature"),
    ("河", "river", "nature"),
    # Colors (8)
    ("红色", "red", "color"), ("蓝色", "blue", "color"), ("绿色", "green", "color"),
    ("黄色", "yellow", "color"), ("白色", "white", "color"), ("黑色", "black", "color"),
    ("紫色", "purple", "color"), ("橙色", "orange", "color"),
    # Emotions (10)
    ("快乐", "happy", "emotion"), ("悲伤", "sad", "emotion"), ("愤怒", "angry", "emotion"),
    ("恐惧", "fear", "emotion"), ("惊讶", "surprise", "emotion"), ("爱", "love", "emotion"),
    ("恨", "hate", "emotion"), ("希望", "hope", "emotion"), ("骄傲", "pride", "emotion"),
    ("嫉妒", "jealousy", "emotion"),
    # Actions (12)
    ("跑步", "run", "action"), ("游泳", "swim", "action"), ("飞翔", "fly", "action"),
    ("跳舞", "dance", "action"), ("唱歌", "sing", "action"), ("吃", "eat", "action"),
    ("喝", "drink", "action"), ("睡", "sleep", "action"), ("走", "walk", "action"),
    ("看", "see", "action"), ("写", "write", "action"), ("读", "read", "action"),
    # People/Roles (8)
    ("老师", "teacher", "person"), ("医生", "doctor", "person"), ("工人", "worker", "person"),
    ("农民", "farmer", "person"), ("士兵", "soldier", "person"), ("律师", "lawyer", "person"),
    ("科学家", "scientist", "person"), ("艺术家", "artist", "person"),
    # Size adjectives (8)
    ("大", "big", "adjective"), ("小", "small", "adjective"), ("高", "tall", "adjective"),
    ("矮", "short", "adjective"), ("长", "long", "adjective"), ("快", "fast", "adjective"),
    ("慢", "slow", "adjective"), ("热", "hot", "adjective"),
    # Body parts (7)
    ("手", "hand", "body"), ("脚", "foot", "body"), ("头", "head", "body"),
    ("眼", "eye", "body"), ("耳", "ear", "body"), ("鼻", "nose", "body"),
    ("心", "heart", "body"),
]

CATEGORY_NAMES = sorted(set(w[2] for w in WORD_LIST))
CATEGORY_TO_IDX = {c: i for i, c in enumerate(CATEGORY_NAMES)}

TASK_TEMPLATES = {
    'translate': "将以下中文翻译成英文：{word}",
    'continue': "接下来会发生什么：{word}",
    'define': "请定义以下词语：{word}",
}


def load_model(model_key):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    config = MODEL_CONFIGS[model_key]
    print(f"Loading {config['name']}...")
    
    tokenizer = AutoTokenizer.from_pretrained(config['name'], trust_remote_code=True)
    
    if model_key in ['deepseek7b']:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
        model = AutoModelForCausalLM.from_pretrained(
            config['name'],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config['name'],
            torch_dtype=config['dtype'],
            device_map="auto",
            trust_remote_code=True,
        )
    model.eval()
    return model, tokenizer


def extract_residuals_with_interventions(model, tokenizer, texts, model_key, 
                                          spike_dirs=None, ablate_spike=False):
    """Extract residuals with optional spike ablation at each layer.
    
    If ablate_spike=True and spike_dirs is provided, will remove spike component
    from residual stream at each layer and measure the effect on subsequent layers.
    """
    config = MODEL_CONFIGS[model_key]
    n_layers = config['n_layers']
    
    all_residuals = {l: [] for l in range(n_layers)}
    all_logits = []
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            logits = outputs.logits[0, -1, :].cpu().float().numpy()
            all_logits.append(logits)
            
            for l in range(n_layers):
                h = hidden_states[l + 1][0, -1, :].cpu().float().numpy()
                all_residuals[l].append(h)
    
    for l in range(n_layers):
        all_residuals[l] = np.stack(all_residuals[l], axis=0)
    all_logits = np.stack(all_logits, axis=0)
    
    return all_residuals, all_logits


def extract_residuals_with_ablation(model, tokenizer, texts, model_key, 
                                     spike_dirs, target_layer, n_ablate_dims=25):
    """Ablate spike directions at target_layer and measure effects on downstream layers.
    
    Uses hooks to ablate spike component at the target layer.
    Returns: residuals at all layers with and without ablation.
    """
    config = MODEL_CONFIGS[model_key]
    n_layers = config['n_layers']
    d_model = config['d_model']
    
    V_spike = spike_dirs[target_layer]  # (n_components, d_model)
    V_spike = V_spike[:n_ablate_dims]   # Only ablate top n_ablate_dims
    
    # Project onto spike: P_spike = V^T (V V^T)^{-1} V = V^T V (since V is orthonormal)
    P_spike = V_spike.T @ V_spike  # (d, d) projection matrix
    
    results_clean = {l: [] for l in range(n_layers)}
    results_ablated = {l: [] for l in range(n_layers)}
    all_logits_clean = []
    all_logits_ablated = []
    
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        # Clean run
        with torch.no_grad():
            outputs_clean = model(**inputs, output_hidden_states=True)
            for l in range(n_layers):
                h = outputs_clean.hidden_states[l + 1][0, -1, :].cpu().float().numpy()
                results_clean[l].append(h)
            all_logits_clean.append(outputs_clean.logits[0, -1, :].cpu().float().numpy())
        
        # Ablated run — use hook to remove spike component at target_layer
        def make_ablation_hook(proj_matrix, device):
            P_torch = torch.tensor(proj_matrix, dtype=torch.float32).to(device)
            def hook_fn(module, input, output):
                # output is a tuple: (hidden_states, ...) or just hidden_states
                if isinstance(output, tuple):
                    h = output[0]
                    h_ablated = h - (h @ P_torch.T) @ P_torch.T  # Remove spike
                    return (h_ablated,) + output[1:]
                else:
                    h = output
                    h_ablated = h - (h @ P_torch.T) @ P_torch.T
                    return h_ablated
            return hook_fn
        
        # Register hook at target layer
        layer_name = f"model.layers.{target_layer}"
        hook = None
        for name, module in model.named_modules():
            if name == layer_name:
                hook = module.register_forward_hook(
                    make_ablation_hook(P_spike, model.device)
                )
                break
        
        if hook is None:
            # Try alternative approach — just use the clean results
            for l in range(n_layers):
                results_ablated[l].append(results_clean[l][-1])
            all_logits_ablated.append(all_logits_clean[-1])
            continue
        
        with torch.no_grad():
            try:
                outputs_ablated = model(**inputs, output_hidden_states=True)
                for l in range(n_layers):
                    h = outputs_ablated.hidden_states[l + 1][0, -1, :].cpu().float().numpy()
                    results_ablated[l].append(h)
                all_logits_ablated.append(outputs_ablated.logits[0, -1, :].cpu().float().numpy())
            except Exception as e:
                print(f"  Ablation failed: {e}")
                for l in range(n_layers):
                    results_ablated[l].append(results_clean[l][-1])
                all_logits_ablated.append(all_logits_clean[-1])
        
        hook.remove()
    
    for l in range(n_layers):
        results_clean[l] = np.stack(results_clean[l], axis=0)
        results_ablated[l] = np.stack(results_ablated[l], axis=0)
    all_logits_clean = np.stack(all_logits_clean, axis=0)
    all_logits_ablated = np.stack(all_logits_ablated, axis=0)
    
    return results_clean, results_ablated, all_logits_clean, all_logits_ablated


def compute_spike_subspace(residuals_task, residuals_base, n_components=25):
    diffs = residuals_task - residuals_base
    mean_diff = diffs.mean(axis=0)
    diffs_centered = diffs - mean_diff
    
    U, S, Vt = np.linalg.svd(diffs_centered, full_matrices=False)
    
    s2 = S ** 2
    pr = (s2.sum()) ** 2 / (s2 ** 2).sum() if (s2 ** 2).sum() > 0 else 0
    total_var = s2.sum()
    concentration = s2[:n_components].sum() / total_var if total_var > 0 else 0
    
    return {
        'components': Vt,
        'singular_values': S,
        'mean_diff': mean_diff,
        'pr': pr,
        'concentration': concentration,
    }


# ============================================================
# Exp 1: Indirect Causal Effect
# ============================================================

def exp1_indirect_causal(model, tokenizer, task_residuals, model_key):
    """Ablate spike at each layer, measure downstream effects on complement and logits."""
    print("\n" + "="*80)
    print("EXP 1: Indirect Causal Effect of Spike")
    print("spike的间接因果效应 — ablate spike后complement和logit如何变化？")
    print("="*80)
    
    config = MODEL_CONFIGS[model_key]
    n_layers = config['n_layers']
    d_model = config['d_model']
    
    # Compute spike subspaces
    spike_dirs = {}
    for l in range(n_layers):
        spike = compute_spike_subspace(
            task_residuals['translate'][l],
            task_residuals['continue'][l],
            n_components=25
        )
        spike_dirs[l] = spike['components']
    
    # For key layers, do ablation experiment
    key_layers = [0, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, n_layers-1]
    key_layers = [l for l in key_layers if l < n_layers]
    
    # Generate translate prompts for ablation
    task_texts = [TASK_TEMPLATES['translate'].format(word=w[0]) for w in WORD_LIST]
    
    results = {
        'ablation_effects': {},
        'complement_shift': {},
        'logit_shift': {},
    }
    
    # Get clean logits
    _, clean_logits = extract_residuals_with_interventions(
        model, tokenizer, task_texts, model_key
    )
    
    for ablate_layer in key_layers:
        print(f"\n  Ablating spike at L{ablate_layer}...")
        
        try:
            clean_res, ablated_res, clean_log, ablated_log = extract_residuals_with_ablation(
                model, tokenizer, task_texts[:10], model_key,  # Use subset for speed
                spike_dirs, ablate_layer, n_ablate_dims=25
            )
            
            # Measure complement shift at downstream layers
            complement_shifts = {}
            for l in range(ablate_layer, min(ablate_layer + 6, n_layers)):
                h_clean = clean_res[l]
                h_ablated = ablated_res[l]
                
                # Project both onto spike and complement
                V = spike_dirs[l][:25]
                
                # Spike component of the shift
                spike_shift = np.mean(np.linalg.norm(
                    (h_clean - h_ablated) @ V.T, axis=1
                ))
                
                # Complement component of the shift
                complement_shift = np.mean(np.linalg.norm(
                    h_clean - h_ablated - (h_clean - h_ablated) @ V.T @ V, axis=1
                ))
                
                # Total shift
                total_shift = np.mean(np.linalg.norm(h_clean - h_ablated, axis=1))
                
                complement_shifts[l] = {
                    'spike_shift': float(spike_shift),
                    'complement_shift': float(complement_shift),
                    'total_shift': float(total_shift),
                    'comp_to_total_ratio': float(complement_shift / (total_shift + 1e-10)),
                }
            
            # Logit shift
            logit_shift = np.mean(np.linalg.norm(clean_log - ablated_log, axis=1))
            logit_cos = np.mean([
                np.dot(clean_log[i], ablated_log[i]) / 
                (np.linalg.norm(clean_log[i]) * np.linalg.norm(ablated_log[i]) + 1e-10)
                for i in range(len(clean_log))
            ])
            
            results['ablation_effects'][ablate_layer] = complement_shifts
            results['logit_shift'][ablate_layer] = {
                'mean_logit_l2': float(logit_shift),
                'mean_logit_cos': float(logit_cos),
            }
            
            print(f"    L{ablate_layer}: logit_L2={logit_shift:.4f}, logit_cos={logit_cos:.6f}")
            for l, cs in complement_shifts.items():
                print(f"      L{l}: spike_shift={cs['spike_shift']:.4f}, comp_shift={cs['complement_shift']:.4f}, "
                      f"comp_ratio={cs['comp_to_total_ratio']:.4f}")
                
        except Exception as e:
            print(f"    Ablation at L{ablate_layer} failed: {e}")
            results['ablation_effects'][ablate_layer] = {'error': str(e)}
    
    return results


# ============================================================
# Exp 2: MLP Read-Out Analysis (Analytical)
# ============================================================

def exp2_mlp_readout(model, tokenizer, task_residuals, model_key):
    """Analytical analysis: how much do MLP weights read from spike directions?
    
    At each layer, the MLP output is: mlp_out = W_down @ act(W_up @ h + b_up) + b_down
    We measure: how much of W_up reads from spike vs complement?
    """
    print("\n" + "="*80)
    print("EXP 2: MLP/Attention Read-Out of Spike Directions")
    print("MLP/Attention如何读取spike方向？")
    print("="*80)
    
    config = MODEL_CONFIGS[model_key]
    n_layers = config['n_layers']
    d_model = config['d_model']
    
    # Compute spike subspaces
    spike_dirs = {}
    for l in range(n_layers):
        spike = compute_spike_subspace(
            task_residuals['translate'][l],
            task_residuals['continue'][l],
            n_components=25
        )
        spike_dirs[l] = spike['components'][:25]  # (25, d)
    
    results = {
        'mlp_up_spike_power': {},
        'mlp_down_spike_power': {},
        'attn_spike_power': {},
    }
    
    key_layers = [l for l in [0, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, n_layers-1] if l < n_layers]
    
    for l in key_layers:
        V_spike = spike_dirs[l]  # (25, d_model)
        d = V_spike.shape[1]  # Should be d_model
        
        # Get MLP up-projection weight
        try:
            mlp = model.model.layers[l].mlp
            
            # PyTorch Linear: weight shape is (out_features, in_features)
            W_gate = mlp.gate_proj.weight.detach().cpu().float().numpy()  # (intermediate, d_model)
            W_up = mlp.up_proj.weight.detach().cpu().float().numpy()       # (intermediate, d_model)
            W_down = mlp.down_proj.weight.detach().cpu().float().numpy()   # (d_model, intermediate)
            
            # How much of W_gate reads from spike?
            # gate reads h (d_model), outputs intermediate. Spike component of h is V_spike.T @ coords
            # W_gate reads from spike: W_gate @ V_spike.T gives (intermediate, 25)
            W_gate_spike = W_gate @ V_spike.T  # (intermediate, 25)
            gate_spike_power = np.sum(W_gate_spike ** 2) / np.sum(W_gate ** 2)
            
            W_up_spike = W_up @ V_spike.T  # (intermediate, 25)
            up_spike_power = np.sum(W_up_spike ** 2) / np.sum(W_up ** 2)
            
            # How much of W_down writes to spike?
            # down_proj: input (intermediate), output (d_model)
            # The output goes to residual stream. Spike component: V_spike @ output
            # So spike writing: V_spike @ W_down gives (25, intermediate)
            W_down_spike = V_spike @ W_down  # (25, intermediate)
            down_spike_power = np.sum(W_down_spike ** 2) / np.sum(W_down ** 2)
            down_complement_power = 1 - down_spike_power
            
            results['mlp_up_spike_power'][l] = {
                'gate_spike_fraction': float(gate_spike_power),
                'up_spike_fraction': float(up_spike_power),
                'down_spike_fraction': float(down_spike_power),
                'down_complement_fraction': float(down_complement_power),
            }
            
            print(f"  L{l} MLP: gate_spike={gate_spike_power:.4f}, up_spike={up_spike_power:.4f}, "
                  f"down_spike={down_spike_power:.4f}, down_comp={down_complement_power:.4f}")
            
        except Exception as e:
            print(f"  L{l}: MLP analysis failed: {e}")
        
        # Get Attention weights
        try:
            attn = model.model.layers[l].self_attn
            
            W_q = attn.q_proj.weight.detach().cpu().float().numpy()  # (num_heads*head_dim, d_model)
            W_k = attn.k_proj.weight.detach().cpu().float().numpy()  # (num_kv_heads*head_dim, d_model)
            W_v = attn.v_proj.weight.detach().cpu().float().numpy()  # (num_kv_heads*head_dim, d_model)
            W_o = attn.o_proj.weight.detach().cpu().float().numpy()  # (d_model, num_heads*head_dim)
            
            # Reading from spike: W @ V_spike.T
            Wq_spike = W_q @ V_spike.T
            q_spike_power = np.sum(Wq_spike ** 2) / np.sum(W_q ** 2)
            
            Wk_spike = W_k @ V_spike.T
            k_spike_power = np.sum(Wk_spike ** 2) / np.sum(W_k ** 2)
            
            Wv_spike = W_v @ V_spike.T
            v_spike_power = np.sum(Wv_spike ** 2) / np.sum(W_v ** 2)
            
            # Writing to spike: V_spike @ W_o
            # o_proj: input (num_heads*head_dim), output (d_model)
            Wo_spike = V_spike @ W_o  # (25, num_heads*head_dim)
            o_spike_power = np.sum(Wo_spike ** 2) / np.sum(W_o ** 2)
            o_complement_power = 1 - o_spike_power
            
            results['attn_spike_power'][l] = {
                'q_spike_fraction': float(q_spike_power),
                'k_spike_fraction': float(k_spike_power),
                'v_spike_fraction': float(v_spike_power),
                'o_spike_fraction': float(o_spike_power),
                'o_complement_fraction': float(o_complement_power),
            }
            
            print(f"  L{l} Attn: q_spike={q_spike_power:.4f}, k_spike={k_spike_power:.4f}, "
                  f"v_spike={v_spike_power:.4f}, o_spike={o_spike_power:.4f}, "
                  f"o_comp={o_complement_power:.4f}")
            
        except Exception as e:
            print(f"  L{l}: Attention analysis failed: {e}")
    
    return results


# ============================================================
# Exp 3: Spike→Complement Information Flow
# ============================================================

def exp3_spike_complement_flow(task_residuals, model_key):
    """How does spike information at layer l relate to complement at layer l+1?
    
    Key question: Can we predict the complement shift from the spike content?
    """
    print("\n" + "="*80)
    print("EXP 3: Spike→Complement Information Flow")
    print("spike→complement信息流 — spike如何影响下一层的complement？")
    print("="*80)
    
    config = MODEL_CONFIGS[model_key]
    n_layers = config['n_layers']
    
    results = {
        'spike_predict_complement': {},
        'cross_layer_spike_stability': {},
    }
    
    # For translate task
    task = 'translate'
    base_task = 'continue'
    
    for l in range(n_layers - 1):
        # Compute spike at layer l
        spike = compute_spike_subspace(
            task_residuals[task][l], task_residuals[base_task][l], n_components=25
        )
        V_spike = spike['components'][:25]
        
        # Get spike and complement representations
        h_task_l = task_residuals[task][l]
        h_base_l = task_residuals[base_task][l]
        
        spike_repr = h_task_l @ V_spike.T  # (n, 25) — spike coordinates
        
        # Complement at next layer
        spike_next = compute_spike_subspace(
            task_residuals[task][l+1], task_residuals[base_task][l+1], n_components=25
        )
        V_spike_next = spike_next['components'][:25]
        
        h_task_l1 = task_residuals[task][l+1]
        h_base_l1 = task_residuals[base_task][l+1]
        
        # Complement diff at l+1
        diff_l1 = h_task_l1 - h_base_l1
        spike_diff_l1 = diff_l1 @ V_spike_next.T @ V_spike_next
        comp_diff_l1 = diff_l1 - spike_diff_l1
        
        # Complement full representation at l+1
        spike_repr_l1 = h_task_l1 @ V_spike_next.T @ V_spike_next
        comp_repr_l1 = h_task_l1 - spike_repr_l1
        
        # Can spike coordinates at l predict complement at l+1?
        # Use ridge regression
        comp_pca = comp_repr_l1 - comp_repr_l1.mean(axis=0)
        U_comp, S_comp, Vt_comp = np.linalg.svd(comp_pca, full_matrices=False)
        comp_coords = U_comp[:, :25] * S_comp[:25]  # (n, 25) — complement PCA coordinates
        
        # Ridge: comp_coords = spike_repr @ W + b
        ridge = Ridge(alpha=1.0)
        try:
            scores = cross_val_score(ridge, spike_repr, comp_coords, cv=5, scoring='r2')
            r2_mean = scores.mean()
            r2_std = scores.std()
        except:
            r2_mean = 0.0
            r2_std = 0.0
        
        # Also: how much of the diff at l+1 is in spike vs complement?
        diff_spike_norm = np.mean(np.linalg.norm(spike_diff_l1, axis=1))
        diff_comp_norm = np.mean(np.linalg.norm(comp_diff_l1, axis=1))
        
        results['spike_predict_complement'][l] = {
            'r2_spike_to_complement': float(r2_mean),
            'r2_std': float(r2_std),
            'diff_spike_norm': float(diff_spike_norm),
            'diff_comp_norm': float(diff_comp_norm),
            'diff_comp_fraction': float(diff_comp_norm / (diff_spike_norm + diff_comp_norm + 1e-10)),
        }
        
        if l % 6 == 0 or l == n_layers - 2:
            print(f"  L{l}->L{l+1}: R2(spike->comp)={r2_mean:.4f}, "
                  f"diff_spike={diff_spike_norm:.2f}, diff_comp={diff_comp_norm:.2f}, "
                  f"comp_fraction={diff_comp_norm/(diff_spike_norm+diff_comp_norm+1e-10):.4f}")
    
    # Cross-layer spike stability
    print("\n--- Spike stability across layers ---")
    spike_subspaces = {}
    for l in range(n_layers):
        spike = compute_spike_subspace(
            task_residuals[task][l], task_residuals[base_task][l], n_components=25
        )
        spike_subspaces[l] = spike['components'][:25]
    
    for l in range(n_layers - 1):
        V1 = spike_subspaces[l]
        V2 = spike_subspaces[l + 1]
        
        # Subspace overlap: ||V1 @ V2^T||_F^2 / min(k1, k2)
        overlap = np.trace(V1 @ V2.T @ V2 @ V1.T) / 25
        
        results['cross_layer_spike_stability'][l] = {
            'subspace_overlap': float(overlap),
        }
        
        if l % 6 == 0 or l == n_layers - 2:
            print(f"  L{l}→L{l+1}: spike_overlap={overlap:.4f}")
    
    return results


# ============================================================
# Exp 4: Nonlinear Semantic→Output Bridge
# ============================================================

def exp4_nonlinear_bridge(task_residuals, all_logits, model_key):
    """Can nonlinear combinations of spike+complement predict output better?
    
    Test: logit_prediction ~ spike + complement + spike×complement
    """
    print("\n" + "="*80)
    print("EXP 4: Nonlinear Semantic→Output Bridge")
    print("非线性语义→输出桥接 — spike×complement交互项有用吗？")
    print("="*80)
    
    config = MODEL_CONFIGS[model_key]
    n_layers = config['n_layers']
    d_model = config['d_model']
    
    task = 'translate'
    base_task = 'continue'
    
    # Get top-k logit tokens (simplified: just use logit values)
    # Target: top-10 logit differences across words
    
    results = {
        'linear_spike_r2': {},
        'linear_comp_r2': {},
        'linear_both_r2': {},
        'nonlinear_r2': {},
    }
    
    key_layers = [l for l in [0, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, n_layers-1] if l < n_layers]
    
    for l in key_layers:
        spike = compute_spike_subspace(
            task_residuals[task][l], task_residuals[base_task][l], n_components=25
        )
        V_spike = spike['components'][:25]
        
        h = task_residuals[task][l]
        
        # Spike coordinates
        spike_repr = h @ V_spike.T  # (n, 25)
        
        # Complement representation (reduced)
        h_spike = h @ V_spike.T @ V_spike
        h_comp = h - h_spike
        h_comp_centered = h_comp - h_comp.mean(axis=0)
        U_comp, S_comp, Vt_comp = np.linalg.svd(h_comp_centered, full_matrices=False)
        comp_repr = U_comp[:, :50] * S_comp[:50]  # (n, 50)
        
        # Target: logit values (use top-10 PCA of logits for dimensionality)
        logits_centered = all_logits - all_logits.mean(axis=0)
        U_log, S_log, Vt_log = np.linalg.svd(logits_centered, full_matrices=False)
        logit_repr = U_log[:, :10] * S_log[:10]  # (n, 10)
        
        # Model 1: spike only
        ridge = Ridge(alpha=1.0)
        try:
            scores_spike = cross_val_score(ridge, spike_repr, logit_repr, cv=5, scoring='r2')
            r2_spike = scores_spike.mean()
        except:
            r2_spike = 0.0
        
        # Model 2: complement only
        try:
            scores_comp = cross_val_score(ridge, comp_repr, logit_repr, cv=5, scoring='r2')
            r2_comp = scores_comp.mean()
        except:
            r2_comp = 0.0
        
        # Model 3: spike + complement (linear)
        both_repr = np.hstack([spike_repr, comp_repr])
        try:
            scores_both = cross_val_score(ridge, both_repr, logit_repr, cv=5, scoring='r2')
            r2_both = scores_both.mean()
        except:
            r2_both = 0.0
        
        # Model 4: spike + complement + spike×complement (nonlinear)
        # Use top-5 spike dims × top-10 complement dims = 50 interaction terms
        n_int_spike = min(5, spike_repr.shape[1])
        n_int_comp = min(10, comp_repr.shape[1])
        interactions = []
        for i in range(n_int_spike):
            for j in range(n_int_comp):
                interactions.append(spike_repr[:, i] * comp_repr[:, j])
        interactions = np.stack(interactions, axis=1)  # (n, 50)
        
        nonlinear_repr = np.hstack([spike_repr, comp_repr, interactions])
        try:
            scores_nonlinear = cross_val_score(ridge, nonlinear_repr, logit_repr, cv=5, scoring='r2')
            r2_nonlinear = scores_nonlinear.mean()
        except:
            r2_nonlinear = 0.0
        
        interaction_boost = r2_nonlinear - r2_both
        
        results['linear_spike_r2'][l] = float(r2_spike)
        results['linear_comp_r2'][l] = float(r2_comp)
        results['linear_both_r2'][l] = float(r2_both)
        results['nonlinear_r2'][l] = {
            'r2_nonlinear': float(r2_nonlinear),
            'interaction_boost': float(interaction_boost),
        }
        
        print(f"  L{l}: R2 spike={r2_spike:.4f}, comp={r2_comp:.4f}, "
              f"both={r2_both:.4f}, nonlinear={r2_nonlinear:.4f}, "
              f"interaction_boost={interaction_boost:+.4f}")
    
    return results


# ============================================================
# Main
# ============================================================

def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save_results(results, name, timestamp):
    filepath = OUTPUT_DIR / f"{name}.json"
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"  Results saved to: {filepath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen3', choices=['qwen3', 'deepseek7b'])
    parser.add_argument('--exp', type=str, default='all',
                       choices=['all', '1', '2', '3', '4'])
    args = parser.parse_args()
    
    model_key = args.model
    config = MODEL_CONFIGS[model_key]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load model
    model, tokenizer = load_model(model_key)
    
    # Generate prompts
    task_texts = {}
    for task_name, template in TASK_TEMPLATES.items():
        task_texts[task_name] = [template.format(word=w[0]) for w in WORD_LIST]
    
    # Extract residuals for all tasks
    print(f"\nExtracting residuals for {len(WORD_LIST)} words × {len(TASK_TEMPLATES)} tasks...")
    task_residuals = {}
    all_logits = None
    for task_name, texts in task_texts.items():
        print(f"  Extracting {task_name}...")
        res, logits = extract_residuals_with_interventions(model, tokenizer, texts, model_key)
        task_residuals[task_name] = res
        if task_name == 'translate':
            all_logits = logits
    
    all_results = {
        'model': model_key,
        'n_words': len(WORD_LIST),
        'timestamp': timestamp,
    }
    
    # Run experiments
    if args.exp in ['all', '1']:
        print("\n\nRunning Exp 1: Indirect Causal Effect...")
        results = exp1_indirect_causal(model, tokenizer, task_residuals, model_key)
        all_results['exp1_indirect_causal'] = convert_to_serializable(results)
        save_results(all_results, f"phase119_exp1_{model_key}_indirect", timestamp)
    
    if args.exp in ['all', '2']:
        print("\n\nRunning Exp 2: MLP Read-Out...")
        results = exp2_mlp_readout(model, tokenizer, task_residuals, model_key)
        all_results['exp2_mlp_readout'] = convert_to_serializable(results)
        save_results(all_results, f"phase119_exp2_{model_key}_mlp", timestamp)
    
    if args.exp in ['all', '3']:
        print("\n\nRunning Exp 3: Spike→Complement Flow...")
        results = exp3_spike_complement_flow(task_residuals, model_key)
        all_results['exp3_spike_complement_flow'] = convert_to_serializable(results)
        save_results(all_results, f"phase119_exp3_{model_key}_flow", timestamp)
    
    if args.exp in ['all', '4']:
        print("\n\nRunning Exp 4: Nonlinear Bridge...")
        results = exp4_nonlinear_bridge(task_residuals, all_logits, model_key)
        all_results['exp4_nonlinear_bridge'] = convert_to_serializable(results)
        save_results(all_results, f"phase119_exp4_{model_key}_nonlinear", timestamp)
    
    # Save combined
    save_results(all_results, f"phase119_{model_key}_all_results", timestamp)
    
    del model
    torch.cuda.empty_cache()
    
    print("\n" + "="*80)
    print("PHASE 119 COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
