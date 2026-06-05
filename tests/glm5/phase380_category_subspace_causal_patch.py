"""
Phase 380: 监督类别子空间因果patch
===================================

核心目标：
1. 提取监督类别子空间（不是PCA，而是LDA/centroid方向）
2. 因果patch：替换/移除类别分量，观察logit变化
3. Jacobian线性预测误差验证

关键改进（vs Phase 379只做PCA回归）：
- Phase 379只看了单个PC与category的R²，但cat_R²(PC1)=0.02, cat_R²(PC2)=0.29
- 现在用监督方法提取完整类别子空间（可能跨多个PC）
- 真正的因果patch：在PROPER post-RMSNorm空间中操作，然后追踪到logit

实验设计：
Part 1: 监督类别子空间解码
  - LDA (Fisher判别) 提取类别子空间
  - centroid方向：每个类别vs其他类别的centroid差异
  - 子空间内分类准确率
  - 子空间维度vs分类准确率曲线

Part 2: 因果patch实验
  - remove-category patch: 从h_norm中移除类别分量，看logit变化
  - swap-category patch: 交换两个样本的类别分量
  - add-category patch: 给corrupt状态添加clean的类别分量
  - 只保留类别分量（移除其他）

Part 3: Jacobian预测误差
  - J(h_mid) · Δh vs RMSNorm(h_clean) - RMSNorm(h_corrupt)
  - cos(linear_pred, proper)
  - 类别gap在linear预测中是否也存在

目标层：
- DS7B: L4 (核心), L8, L24
- Qwen3: L4, L28
- GLM4: L4, L30

用法:
  python tests/glm5/phase380_category_subspace_causal_patch.py qwen3
  python tests/glm5/phase380_category_subspace_causal_patch.py deepseek7b
  python tests/glm5/phase380_category_subspace_causal_patch.py glm4
"""

import sys, os, time, json, gc, traceback
import torch
import numpy as np
from datetime import datetime
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, 'tests/glm5')

from model_utils import get_layers, get_model_info, release_model, get_W_U, MODEL_CONFIGS


def log(msg="", end="\n"):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", end=end, flush=True)


# ===== Binding pairs (same as Phase 379) =====
COLOR_PAIRS = [
    ("apple", "red", "blue"), ("banana", "yellow", "purple"), ("snow", "white", "black"),
    ("sky", "blue", "green"), ("cherry", "red", "blue"), ("leaf", "green", "red"),
    ("rose", "red", "blue"), ("gold", "yellow", "purple"), ("coal", "black", "white"),
    ("silver", "white", "black"), ("milk", "white", "black"), ("honey", "yellow", "blue"),
    ("ruby", "red", "green"), ("emerald", "green", "red"), ("sapphire", "blue", "red"),
    ("moon", "white", "black"), ("flame", "orange", "blue"), ("forest", "green", "white"),
    ("ocean", "blue", "yellow"), ("sun", "yellow", "purple"),
    ("tomato", "red", "blue"), ("lemon", "yellow", "purple"), ("ivory", "white", "black"),
    ("grass", "green", "red"), ("blood", "red", "blue"), ("cloud", "white", "black"),
    ("amber", "yellow", "green"), ("moss", "green", "blue"), ("raven", "black", "white"),
    ("coral", "orange", "blue"), ("teal", "blue", "red"), ("plum", "purple", "yellow"),
    ("bronze", "brown", "blue"), ("crimson", "red", "green"), ("azure", "blue", "red"),
    ("scarlet", "red", "blue"), ("turquoise", "blue", "green"),
    ("maroon", "brown", "white"), ("lime", "green", "purple"), ("peach", "orange", "blue"),
]
TEMP_PAIRS = [
    ("fire", "hot", "cold"), ("desert", "hot", "cold"), ("lava", "hot", "cold"),
    ("ice", "cold", "hot"), ("snow", "cold", "hot"), ("volcano", "hot", "cold"),
    ("furnace", "hot", "cold"), ("glacier", "cold", "hot"),
    ("oven", "hot", "cold"), ("frost", "cold", "hot"), ("magma", "hot", "cold"),
    ("winter", "cold", "hot"), ("summer", "hot", "cold"), ("arctic", "cold", "hot"),
    ("stove", "hot", "cold"), ("blizzard", "cold", "hot"), ("tundra", "cold", "hot"),
    ("inferno", "hot", "cold"), ("iceberg", "cold", "hot"),
]
MOISTURE_PAIRS = [
    ("rain", "wet", "dry"), ("ocean", "wet", "dry"), ("river", "wet", "dry"),
    ("sand", "dry", "wet"), ("dust", "dry", "wet"), ("bone", "dry", "wet"),
    ("swamp", "wet", "dry"), ("desert", "dry", "wet"),
    ("lake", "wet", "dry"), ("sponge", "wet", "dry"), ("cracker", "dry", "wet"),
    ("fog", "wet", "dry"), ("prairie", "dry", "wet"), ("puddle", "wet", "dry"),
    ("cactus", "dry", "wet"), ("waterfall", "wet", "dry"),
]
SIZE_PAIRS = [
    ("elephant", "big", "small"), ("mountain", "big", "small"), ("ant", "small", "big"),
    ("planet", "big", "small"), ("grain", "small", "big"), ("whale", "big", "small"),
    ("galaxy", "big", "small"), ("atom", "small", "big"), ("continent", "big", "small"),
    ("bacteria", "small", "big"), ("tower", "big", "small"), ("speck", "small", "big"),
    ("universe", "big", "small"), ("pixel", "small", "big"), ("castle", "big", "small"),
    ("dust_mote", "small", "big"),
]
WEIGHT_PAIRS = [
    ("boulder", "heavy", "light"), ("feather", "light", "heavy"), ("lead", "heavy", "light"),
    ("balloon", "light", "heavy"), ("steel", "heavy", "light"), ("cotton", "light", "heavy"),
    ("anchor", "heavy", "light"), ("bubble", "light", "heavy"), ("concrete", "heavy", "light"),
    ("air", "light", "heavy"), ("truck", "heavy", "light"), ("petal", "light", "heavy"),
    ("elephant", "heavy", "light"), ("cloud", "light", "heavy"),
]
SPEED_PAIRS = [
    ("cheetah", "fast", "slow"), ("turtle", "slow", "fast"), ("rocket", "fast", "slow"),
    ("snail", "slow", "fast"), ("lightning", "fast", "slow"), ("sloth", "slow", "fast"),
    ("falcon", "fast", "slow"), ("worm", "slow", "fast"), ("bullet", "fast", "slow"),
    ("glacier_motion", "slow", "fast"), ("jet", "fast", "slow"),
    ("racecar", "fast", "slow"), ("caterpillar", "slow", "fast"),
]
BRIGHT_PAIRS = [
    ("star", "bright", "dark"), ("cave", "dark", "bright"), ("sun", "bright", "dark"),
    ("shadow", "dark", "bright"), ("lamp", "bright", "dark"), ("night", "dark", "bright"),
    ("flashlight", "bright", "dark"), ("abyss", "dark", "bright"), ("diamond", "bright", "dark"),
    ("tunnel", "dark", "bright"), ("beacon", "bright", "dark"), ("eclipse", "dark", "bright"),
    ("lighthouse", "bright", "dark"), ("dungeon", "dark", "bright"),
]

ALL_PAIRS = COLOR_PAIRS + TEMP_PAIRS + MOISTURE_PAIRS + SIZE_PAIRS + WEIGHT_PAIRS + SPEED_PAIRS + BRIGHT_PAIRS

PAIR_CATEGORIES = (
    ["color"] * len(COLOR_PAIRS) +
    ["temperature"] * len(TEMP_PAIRS) +
    ["moisture"] * len(MOISTURE_PAIRS) +
    ["size"] * len(SIZE_PAIRS) +
    ["weight"] * len(WEIGHT_PAIRS) +
    ["speed"] * len(SPEED_PAIRS) +
    ["brightness"] * len(BRIGHT_PAIRS)
)

# Remove duplicates
seen = set()
unique_pairs = []
unique_cats = []
for pair, cat in zip(ALL_PAIRS, PAIR_CATEGORIES):
    key = (pair[0], pair[1])
    if key not in seen:
        seen.add(key)
        unique_pairs.append(pair)
        unique_cats.append(cat)

ALL_PAIRS = unique_pairs
PAIR_CATEGORIES = unique_cats
ALL_CATEGORIES = sorted(set(PAIR_CATEGORIES))
N_CATEGORIES = len(ALL_CATEGORIES)

CORRUPTED_BASELINE = "The item"
TEMPLATE = "The {obj} is {attr}."


# ===== Math utilities =====
def _silu(x):
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -50, 50))))

def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

def compute_pca_full(X):
    M = X - X.mean(axis=0, keepdims=True)
    try:
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
    except:
        return None, None, None, None, None
    total_var = np.sum(S**2)
    if total_var < 1e-10:
        return None, None, None, None, None
    explained = (S**2) / total_var
    eff_rank = int(np.searchsorted(np.cumsum(explained), 0.95) + 1)
    scores = U * S
    return explained, eff_rank, Vt, scores, S

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

def rms_norm_single(x, weight=None, eps=1e-6):
    d = x.shape[-1]
    rms = np.sqrt(np.mean(x**2) + eps)
    result = x / rms * np.sqrt(d)
    if weight is not None:
        result = result * weight
    return result

def rms_norm_jacobian(x, weight=None, eps=1e-6):
    """Compute Jacobian of RMSNorm at point x.
    
    J_ij = d(RMSNorm(x))_i / dx_j
         = gamma_i * [sqrt(d)/rms * delta_ij - sqrt(d)/rms^3 * x_i * x_j / d]
         = gamma_i * sqrt(d)/rms * [delta_ij - x_i*x_j / (d*rms^2)]
    """
    d = x.shape[-1]
    rms = np.sqrt(np.mean(x**2) + eps)
    rms3 = rms**3
    prefactor = np.sqrt(d) / rms
    # J = prefactor * (I - x x^T / (d * rms^2))
    J = prefactor * (np.eye(d) - np.outer(x, x) / (d * rms**2))
    if weight is not None:
        J = np.diag(weight) @ J
    return J


# ===== Model loading =====
def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = None
    for impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True, attn_implementation=impl)
            log(f"  Loaded with attn_impl={impl}")
            break
        except Exception as e:
            log(f"  Failed with {impl}: {str(e)[:80]}")
            continue
    if model is None:
        raise RuntimeError(f"Failed to load {model_name}")
    model.eval()
    
    if hasattr(model, 'hf_device_map'):
        dmap = model.hf_device_map
        gpu_count = sum(1 for v in dmap.values() if 'cuda' in str(v))
        cpu_count = sum(1 for v in dmap.values() if 'cpu' in str(v))
        log(f"  Device map: {gpu_count} GPU + {cpu_count} CPU components")
    
    return model, tokenizer


# ===== Weight loading =====
def load_mlp_weights(model, model_name, layer_idx):
    import glob
    from safetensors import safe_open
    layers = get_layers(model)
    mlp = layers[layer_idx].mlp
    gate_proj = getattr(mlp, "gate_proj", None)
    up_proj = getattr(mlp, "up_proj", None)
    down_proj = getattr(mlp, "down_proj", None)
    W_gate = W_up = W_down = None
    if gate_proj is not None and up_proj is not None and down_proj is not None:
        try:
            W_gate = gate_proj.weight.detach().cpu().float().numpy()
            W_up = up_proj.weight.detach().cpu().float().numpy()
            W_down = down_proj.weight.detach().cpu().float().numpy()
        except (NotImplementedError, RuntimeError):
            pass
    if W_gate is None:
        model_path = MODEL_CONFIGS[model_name]["path"]
        for sf_file in glob.glob(os.path.join(model_path, '*.safetensors')):
            try:
                with safe_open(sf_file, framework='pt', device='cpu') as sf:
                    for key in sf.keys():
                        if f"layers.{layer_idx}.mlp.gate_proj.weight" in key:
                            W_gate = sf.get_tensor(key).float().numpy()
                        elif f"layers.{layer_idx}.mlp.up_proj.weight" in key:
                            W_up = sf.get_tensor(key).float().numpy()
                        elif f"layers.{layer_idx}.mlp.down_proj.weight" in key:
                            W_down = sf.get_tensor(key).float().numpy()
                        elif f"layers.{layer_idx}.mlp.gate_up_proj.weight" in key:
                            full_w = sf.get_tensor(key).float().numpy()
                            half = full_w.shape[0] // 2
                            W_gate = full_w[:half]
                            W_up = full_w[half:]
                        elif f"layers.{layer_idx}.mlp.dense_h_to_4h.weight" in key:
                            full_w = sf.get_tensor(key).float().numpy()
                            half = full_w.shape[0] // 2
                            W_gate = full_w[:half]
                            W_up = full_w[half:]
                        elif f"layers.{layer_idx}.mlp.dense_4h_to_h.weight" in key:
                            W_down = sf.get_tensor(key).float().numpy()
                if W_gate is not None and W_down is not None:
                    break
            except:
                continue
    return W_gate, W_up, W_down

def _load_ln_weight(model, model_name, layer_idx):
    import glob
    from safetensors import safe_open
    layers = get_layers(model)
    for attr_name in ["post_attention_layernorm", "ln2", "input_layernorm"]:
        ln = getattr(layers[layer_idx], attr_name, None)
        if ln is not None:
            try:
                w = ln.weight.detach().cpu().float().numpy()
                if w is not None and len(w) > 0:
                    return w
            except (NotImplementedError, RuntimeError):
                pass
    model_path = MODEL_CONFIGS[model_name]["path"]
    for sf_file in glob.glob(os.path.join(model_path, '*.safetensors')):
        try:
            with safe_open(sf_file, framework='pt', device='cpu') as sf:
                for key in sf.keys():
                    for ln_name in ["post_attention_layernorm", "ln2", "input_layernorm"]:
                        if f"layers.{layer_idx}.{ln_name}.weight" in key:
                            return sf.get_tensor(key).float().numpy()
        except:
            continue
    log(f"    WARNING: Could not load LN weight for layer {layer_idx}")
    return None


# ===== Data collection =====
def collect_residual_states(model, tokenizer, model_name, target_layers):
    cfg = MODEL_CONFIGS[model_name]
    act_fn = "gelu" if model_name == "glm4" else "silu"
    layers = get_layers(model)
    input_device = next(model.parameters()).device
    n_pairs = len(ALL_PAIRS)
    
    all_data = {}
    
    for l in target_layers:
        log(f"  Collecting Layer {l}...")
        t_l = time.time()
        
        W_gate, W_up, W_down = load_mlp_weights(model, model_name, l)
        if W_gate is None:
            log(f"    SKIP: Could not load MLP weights for layer {l}")
            continue
        
        mlp_module = layers[l].mlp
        ln_weight = _load_ln_weight(model, model_name, l)
        
        pre_mlp_clean_list = []
        pre_mlp_corrupt_list = []
        h_post_clean_list = []
        h_post_corrupt_list = []
        
        for pidx, (obj, target, competitor) in enumerate(ALL_PAIRS):
            if pidx % 30 == 0:
                log(f"    Pair {pidx+1}/{n_pairs}")
            
            clean_prompt = TEMPLATE.format(obj=obj, attr=target)
            corrupt_prompt = TEMPLATE.format(obj=CORRUPTED_BASELINE, attr=target)
            
            captured = {}
            def mlp_input_hook(module, input, output=None):
                captured["mlp_input"] = input[0].detach().cpu().float()
            
            h_hook = mlp_module.register_forward_pre_hook(mlp_input_hook)
            
            with torch.no_grad():
                clean_out = model(
                    input_ids=tokenizer(clean_prompt, return_tensors="pt",
                                       truncation=True, max_length=64)["input_ids"].to(input_device),
                    attention_mask=tokenizer(clean_prompt, return_tensors="pt",
                                            truncation=True, max_length=64)["attention_mask"].to(input_device),
                    output_hidden_states=True)
            
            last_pos = tokenizer(clean_prompt, return_tensors="pt")["input_ids"].shape[1] - 1
            pre_mlp_clean = captured["mlp_input"][0, last_pos].numpy()
            h_post_clean = clean_out.hidden_states[l+1][0, last_pos].detach().cpu().float().numpy()
            pre_mlp_clean_list.append(pre_mlp_clean)
            h_post_clean_list.append(h_post_clean)
            
            del clean_out
            captured.clear()
            
            with torch.no_grad():
                corrupt_out = model(
                    input_ids=tokenizer(corrupt_prompt, return_tensors="pt",
                                       truncation=True, max_length=64)["input_ids"].to(input_device),
                    attention_mask=tokenizer(corrupt_prompt, return_tensors="pt",
                                            truncation=True, max_length=64)["attention_mask"].to(input_device),
                    output_hidden_states=True)
            
            last_pos_r = tokenizer(corrupt_prompt, return_tensors="pt")["input_ids"].shape[1] - 1
            pre_mlp_corrupt = captured["mlp_input"][0, last_pos_r].numpy()
            h_post_corrupt = corrupt_out.hidden_states[l+1][0, last_pos_r].detach().cpu().float().numpy()
            pre_mlp_corrupt_list.append(pre_mlp_corrupt)
            h_post_corrupt_list.append(h_post_corrupt)
            
            del corrupt_out
            h_hook.remove()
            if pidx % 5 == 0:
                torch.cuda.empty_cache()
        
        pre_mlp_clean = np.array(pre_mlp_clean_list)
        pre_mlp_corrupt = np.array(pre_mlp_corrupt_list)
        h_post_clean = np.array(h_post_clean_list)
        h_post_corrupt = np.array(h_post_corrupt_list)
        dh = h_post_clean - h_post_corrupt
        
        if act_fn == "silu":
            gate_act_clean = _silu(pre_mlp_clean @ W_gate.T)
            gate_act_corrupt = _silu(pre_mlp_corrupt @ W_gate.T)
        else:
            gate_act_clean = _gelu(pre_mlp_clean @ W_gate.T)
            gate_act_corrupt = _gelu(pre_mlp_corrupt @ W_gate.T)
        
        up_clean = pre_mlp_clean @ W_up.T
        up_corrupt = pre_mlp_corrupt @ W_up.T
        gate_up_clean = gate_act_clean * up_clean
        gate_up_corrupt = gate_act_corrupt * up_corrupt
        d_gate_up = gate_up_clean - gate_up_corrupt
        
        h_post_attn_clean = h_post_clean - (W_down @ gate_up_clean.T).T
        h_post_attn_corrupt = h_post_corrupt - (W_down @ gate_up_corrupt.T).T
        
        all_data[str(l)] = {
            "pre_mlp_clean": pre_mlp_clean, "pre_mlp_corrupt": pre_mlp_corrupt,
            "h_post_clean": h_post_clean, "h_post_corrupt": h_post_corrupt,
            "h_post_attn_clean": h_post_attn_clean, "h_post_attn_corrupt": h_post_attn_corrupt,
            "gate_up_clean": gate_up_clean, "gate_up_corrupt": gate_up_corrupt,
            "d_gate_up": d_gate_up, "dh": dh,
            "W_gate": W_gate, "W_up": W_up, "W_down": W_down,
            "ln_weight": ln_weight,
        }
        
        log(f"    Layer {l} done in {time.time()-t_l:.1f}s")
    
    return all_data


# ===== Part 1: Supervised Category Subspace Decoding =====
def category_subspace_decoding(all_data, model, tokenizer, model_name):
    """
    Extract supervised category subspace using multiple methods:
    1. LDA (Fisher discriminant)
    2. Centroid differences (one-vs-rest)
    3. Multi-class SVM directions (linear)
    
    Then evaluate how much category information exists in proper post-RMSNorm space.
    """
    log("\n" + "="*60)
    log("Part 1: Supervised Category Subspace Decoding")
    log("="*60)
    
    W_U = None
    try:
        W_U = get_W_U(model, model_name)
    except:
        log("  Could not load W_U")
    
    results = {}
    
    for l_str in sorted(all_data.keys(), key=int):
        d = all_data[l_str]
        W_down = d["W_down"]
        n_pairs = d["dh"].shape[0]
        d_model = d["dh"].shape[1]
        l = int(l_str)
        
        ln_weight = d.get("ln_weight", None)
        
        # Compute proper post-RMSNorm
        h_clean_norm = np.zeros_like(d["h_post_clean"])
        h_corrupt_norm = np.zeros_like(d["h_post_corrupt"])
        for i in range(n_pairs):
            h_clean_norm[i] = rms_norm_single(d["h_post_clean"][i], ln_weight)
            h_corrupt_norm[i] = rms_norm_single(d["h_post_corrupt"][i], ln_weight)
        
        dh_proper = h_clean_norm - h_corrupt_norm  # (n, d_model)
        
        # PCA for reference
        pca, rank, Vt, scores, S = compute_pca_full(dh_proper)
        if pca is None:
            log(f"  Layer {l}: PCA failed, skipping")
            continue
        
        log(f"\n  Layer {l}:")
        
        # ===== Method 1: Centroid-based category directions =====
        # For each category, compute centroid of dh_proper
        cat_centroids = {}
        cat_indices = {}
        for cat in ALL_CATEGORIES:
            idx = [j for j, c in enumerate(PAIR_CATEGORIES) if c == cat]
            cat_indices[cat] = idx
            cat_centroids[cat] = np.mean(dh_proper[idx], axis=0)
        
        overall_centroid = np.mean(dh_proper, axis=0)
        
        # One-vs-rest centroid differences
        centroid_diffs = {}
        for cat in ALL_CATEGORIES:
            # n_cat vs n_rest
            n_cat = len(cat_indices[cat])
            rest_idx = [j for j in range(n_pairs) if PAIR_CATEGORIES[j] != cat]
            rest_centroid = np.mean(dh_proper[rest_idx], axis=0)
            diff = cat_centroids[cat] - rest_centroid
            centroid_diffs[cat] = diff / (np.linalg.norm(diff) + 1e-10)
        
        # Stack centroid diffs into a subspace basis
        C_matrix = np.array([centroid_diffs[cat] for cat in ALL_CATEGORIES])  # (n_cat, d_model)
        # Orthonormalize
        try:
            U_cent, S_cent, Vt_cent = np.linalg.svd(C_matrix, full_matrices=False)
            cat_subspace_centroid = Vt_cent[:N_CATEGORIES]  # (n_cat, d_model)
        except:
            cat_subspace_centroid = None
        
        # ===== Method 2: LDA (Fisher Discriminant) =====
        # S_W = within-class scatter, S_B = between-class scatter
        # LDA direction: max w^T S_B w / w^T S_W w
        S_W = np.zeros((d_model, d_model))
        S_B = np.zeros((d_model, d_model))
        
        for cat in ALL_CATEGORIES:
            idx = cat_indices[cat]
            cat_data = dh_proper[idx]
            cat_mean = cat_centroids[cat]
            # Within-class scatter
            diff = cat_data - cat_mean
            S_W += diff.T @ diff
            # Between-class scatter
            mean_diff = (cat_mean - overall_centroid).reshape(-1, 1)
            S_B += len(idx) * mean_diff @ mean_diff.T
        
        # Regularize S_W
        S_W += 1e-6 * np.eye(d_model) * np.trace(S_W) / d_model
        
        try:
            # Solve generalized eigenvalue problem
            eigvals, eigvecs = np.linalg.eigh(np.linalg.solve(S_W, S_B))
            # Sort by eigenvalue descending
            sort_idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[sort_idx]
            eigvecs = eigvecs[:, sort_idx]
            # Take top (n_cat - 1) LDA directions
            n_lda = min(N_CATEGORIES - 1, d_model)
            lda_directions = eigvecs[:, :n_lda].T  # (n_lda, d_model)
            lda_eigenvalues = eigvals[:n_lda]
        except:
            lda_directions = None
            lda_eigenvalues = None
        
        # ===== Evaluate: Classification accuracy in category subspace =====
        # Using centroid subspace
        def classify_in_subspace(X, basis, k=5):
            """Classify X using k-dimensional subspace defined by basis rows."""
            projections = X @ basis[:k].T  # (n, k)
            # Leave-one-out nearest centroid
            correct = 0
            for i in range(len(X)):
                train_X = np.delete(projections, i, axis=0)
                train_labels = np.delete(PAIR_CATEGORIES, i)
                test_proj = projections[i]
                
                # Nearest centroid
                best_cat = None
                best_dist = float('inf')
                for cat in ALL_CATEGORIES:
                    cat_idx = [j for j, c in enumerate(train_labels) if c == cat]
                    if len(cat_idx) == 0:
                        continue
                    cat_centroid_proj = np.mean(train_X[cat_idx], axis=0)
                    dist = np.sum((test_proj - cat_centroid_proj)**2)
                    if dist < best_dist:
                        best_dist = dist
                        best_cat = cat
                if best_cat == PAIR_CATEGORIES[i]:
                    correct += 1
            return correct / len(X)
        
        # Classification using PCA subspace (for comparison)
        def classify_in_pca_subspace(X, Vt, k=5):
            """Classify X using k-dimensional PCA subspace."""
            projections = X @ Vt[:k].T  # (n, k)
            correct = 0
            for i in range(len(X)):
                train_X = np.delete(projections, i, axis=0)
                train_labels = np.delete(PAIR_CATEGORIES, i)
                test_proj = projections[i]
                
                best_cat = None
                best_dist = float('inf')
                for cat in ALL_CATEGORIES:
                    cat_idx = [j for j, c in enumerate(train_labels) if c == cat]
                    if len(cat_idx) == 0:
                        continue
                    cat_centroid_proj = np.mean(train_X[cat_idx], axis=0)
                    dist = np.sum((test_proj - cat_centroid_proj)**2)
                    if dist < best_dist:
                        best_dist = dist
                        best_cat = cat
                if best_cat == PAIR_CATEGORIES[i]:
                    correct += 1
            return correct / len(X)
        
        # Evaluate at different subspace dimensions
        n_pairs_check = min(n_pairs, 200)  # limit for speed
        
        # PCA classification (LOO might be slow for large n, use subset)
        pca_accs = {}
        for k in [1, 2, 3, 5, 7, 10, 15, 20]:
            if k > min(n_pairs_check, d_model):
                continue
            acc = classify_in_pca_subspace(dh_proper[:n_pairs_check], Vt, k)
            pca_accs[k] = round(acc, 4)
        
        # Centroid subspace classification
        centroid_accs = {}
        if cat_subspace_centroid is not None:
            for k in [1, 2, 3, 5, 7]:
                if k > N_CATEGORIES:
                    continue
                acc = classify_in_subspace(dh_proper[:n_pairs_check], cat_subspace_centroid, k)
                centroid_accs[k] = round(acc, 4)
        
        # LDA classification
        lda_accs = {}
        if lda_directions is not None:
            for k in [1, 2, 3, 5, 6]:
                if k > lda_directions.shape[0]:
                    continue
                acc = classify_in_subspace(dh_proper[:n_pairs_check], lda_directions, k)
                lda_accs[k] = round(acc, 4)
        
        # Also classify in RAW Δh space for comparison
        dh_raw = d["dh"]
        pca_raw, rank_raw, Vt_raw, scores_raw, _ = compute_pca_full(dh_raw)
        raw_pca_accs = {}
        if pca_raw is not None:
            for k in [1, 2, 3, 5, 7, 10, 15, 20]:
                if k > min(n_pairs_check, d_model):
                    continue
                acc = classify_in_pca_subspace(dh_raw[:n_pairs_check], Vt_raw, k)
                raw_pca_accs[k] = round(acc, 4)
        
        # Chance level
        chance = 1.0 / N_CATEGORIES
        
        log(f"    Chance level: {chance:.4f}")
        log(f"    PCA classification (PROPER): {pca_accs}")
        log(f"    PCA classification (RAW): {raw_pca_accs}")
        log(f"    Centroid subspace: {centroid_accs}")
        log(f"    LDA subspace: {lda_accs}")
        
        # ===== LDA directions: alignment with PCA =====
        if lda_directions is not None and Vt is not None:
            lda_pca_alignment = {}
            for i in range(min(3, lda_directions.shape[0])):
                align = []
                for j in range(min(20, Vt.shape[0])):
                    align.append(float(cosine_sim(lda_directions[i], Vt[j])))
                lda_pca_alignment[f"LDA{i}"] = [round(a, 4) for a in align]
        else:
            lda_pca_alignment = None
        
        # ===== Norm ratio correlation =====
        norm_ratio = np.sum(d["h_post_clean"]**2, axis=1) / (np.sum(d["h_post_corrupt"]**2, axis=1) + 1e-10)
        dh_proper_norm = np.linalg.norm(dh_proper, axis=1)
        
        # Project dh_proper onto LDA directions
        if lda_directions is not None:
            lda_scores = dh_proper @ lda_directions.T  # (n, n_lda)
            lda_norm_ratio_corr = []
            lda_norm_corr = []
            for i in range(min(n_lda, lda_scores.shape[1])):
                c_ratio = float(np.corrcoef(lda_scores[:, i], norm_ratio)[0, 1])
                c_norm = float(np.corrcoef(lda_scores[:, i], dh_proper_norm)[0, 1])
                lda_norm_ratio_corr.append(round(c_ratio, 4))
                lda_norm_corr.append(round(c_norm, 4))
        else:
            lda_norm_ratio_corr = None
            lda_norm_corr = None
        
        # ===== Logit effect of category subspace =====
        if W_U is not None and lda_directions is not None:
            # Project dh_proper onto LDA subspace and reconstruct
            lda_proj = lda_directions.T @ lda_directions  # projection matrix
            dh_cat_component = (lda_proj @ dh_proper.T).T  # (n, d_model)
            dh_noncat_component = dh_proper - dh_cat_component
            
            # Logit effect of cat vs noncat
            cat_logit_effect = {}
            for pidx, (obj, target, competitor) in enumerate(ALL_PAIRS):
                t_tokens = tokenizer(target, add_special_tokens=False)["input_ids"]
                c_tokens = tokenizer(competitor, add_special_tokens=False)["input_ids"]
                if len(t_tokens) != 1 or len(c_tokens) != 1:
                    continue
                t_id, c_id = t_tokens[0], c_tokens[0]
                if W_U.shape[0] == d_model:
                    logit_full = W_U.T @ dh_proper[pidx]
                    logit_cat = W_U.T @ dh_cat_component[pidx]
                    logit_noncat = W_U.T @ dh_noncat_component[pidx]
                else:
                    logit_full = W_U @ dh_proper[pidx]
                    logit_cat = W_U @ dh_cat_component[pidx]
                    logit_noncat = W_U @ dh_noncat_component[pidx]
                
                if pidx == 0:
                    cat_logit_effect["sample"] = {
                        "full_t-c": float(logit_full[t_id] - logit_full[c_id]),
                        "cat_t-c": float(logit_cat[t_id] - logit_cat[c_id]),
                        "noncat_t-c": float(logit_noncat[t_id] - logit_noncat[c_id]),
                    }
            
            # Average logit correlation
            logit_full_all = []
            logit_cat_all = []
            logit_noncat_all = []
            for pidx, (obj, target, competitor) in enumerate(ALL_PAIRS):
                t_tokens = tokenizer(target, add_special_tokens=False)["input_ids"]
                c_tokens = tokenizer(competitor, add_special_tokens=False)["input_ids"]
                if len(t_tokens) != 1 or len(c_tokens) != 1:
                    continue
                t_id, c_id = t_tokens[0], c_tokens[0]
                if W_U.shape[0] == d_model:
                    lf = W_U.T @ dh_proper[pidx]
                    lc = W_U.T @ dh_cat_component[pidx]
                    lnc = W_U.T @ dh_noncat_component[pidx]
                else:
                    lf = W_U @ dh_proper[pidx]
                    lc = W_U @ dh_cat_component[pidx]
                    lnc = W_U @ dh_noncat_component[pidx]
                logit_full_all.append(lf[t_id] - lf[c_id])
                logit_cat_all.append(lc[t_id] - lc[c_id])
                logit_noncat_all.append(lnc[t_id] - lnc[c_id])
            
            if len(logit_full_all) > 5:
                corr_cat = float(np.corrcoef(logit_full_all, logit_cat_all)[0, 1])
                corr_noncat = float(np.corrcoef(logit_full_all, logit_noncat_all)[0, 1])
                cat_logit_effect["corr_full_cat"] = round(corr_cat, 4)
                cat_logit_effect["corr_full_noncat"] = round(corr_noncat, 4)
                cat_logit_effect["mean_abs_cat"] = round(float(np.mean(np.abs(logit_cat_all))), 4)
                cat_logit_effect["mean_abs_noncat"] = round(float(np.mean(np.abs(logit_noncat_all))), 4)
        else:
            cat_logit_effect = None
        
        results[l_str] = {
            "n_pairs": n_pairs,
            "d_model": d_model,
            "pca_explained_top10": [round(float(x), 4) for x in pca[:10]],
            "lda_eigenvalues": [round(float(x), 4) for x in lda_eigenvalues[:6]] if lda_eigenvalues is not None else None,
            "classification": {
                "chance": round(chance, 4),
                "pca_proper": pca_accs,
                "pca_raw": raw_pca_accs,
                "centroid_subspace": centroid_accs,
                "lda_subspace": lda_accs,
            },
            "lda_pca_alignment": lda_pca_alignment,
            "lda_norm_ratio_corr": lda_norm_ratio_corr,
            "lda_norm_corr": lda_norm_corr,
            "cat_logit_effect": cat_logit_effect,
        }
    
    return results


# ===== Part 2: Causal Patch Experiments =====
def causal_patch_experiment(all_data, model, tokenizer, model_name):
    """
    True causal patch: modify the category component in h_norm,
    then trace through to see logit changes.
    
    Key idea:
    - We have h_clean_norm (after RMSNorm) for each sample
    - We decompose h_clean_norm into: category_component + noncategory_component
    - Then we do: h_patched = h_corrupt_norm + category_component_from_clean
    - But we need to "inverse RMSNorm" back to h_raw, then run forward
    
    Simplification for this phase:
    - We can't easily inverse-RMSNorm and re-run the full model
    - Instead, we do W_U linear readout patch (still informative)
    - For full causal patch, we'd need to modify residual stream and re-run
    
    Two patch types:
    A. W_U readout patch (linear, not truly causal but fast):
       - Remove cat component from dh_proper → see logit change
       - Swap cat component between samples → see logit swap
    
    B. Residual stream patch (truly causal):
       - Add cat_component to h_corrupt_raw (before next layer's RMSNorm)
       - Re-run model from that layer → see actual output change
       - This requires careful implementation with hooks
    """
    log("\n" + "="*60)
    log("Part 2: Causal Patch Experiments")
    log("="*60)
    
    W_U = None
    try:
        W_U = get_W_U(model, model_name)
    except:
        log("  Could not load W_U, skipping")
        return {}
    
    results = {}
    
    for l_str in sorted(all_data.keys(), key=int):
        d = all_data[l_str]
        W_down = d["W_down"]
        n_pairs = d["dh"].shape[0]
        d_model = d["dh"].shape[1]
        l = int(l_str)
        
        ln_weight = d.get("ln_weight", None)
        layers = get_layers(model)
        n_layers = len(layers)
        
        # Compute proper post-RMSNorm
        h_clean_norm = np.zeros_like(d["h_post_clean"])
        h_corrupt_norm = np.zeros_like(d["h_post_corrupt"])
        for i in range(n_pairs):
            h_clean_norm[i] = rms_norm_single(d["h_post_clean"][i], ln_weight)
            h_corrupt_norm[i] = rms_norm_single(d["h_post_corrupt"][i], ln_weight)
        
        dh_proper = h_clean_norm - h_corrupt_norm
        
        # Extract LDA subspace
        cat_centroids = {}
        cat_indices = {}
        for cat in ALL_CATEGORIES:
            idx = [j for j, c in enumerate(PAIR_CATEGORIES) if c == cat]
            cat_indices[cat] = idx
            cat_centroids[cat] = np.mean(dh_proper[idx], axis=0)
        
        overall_centroid = np.mean(dh_proper, axis=0)
        
        S_W = np.zeros((d_model, d_model))
        S_B = np.zeros((d_model, d_model))
        for cat in ALL_CATEGORIES:
            idx = cat_indices[cat]
            cat_data = dh_proper[idx]
            cat_mean = cat_centroids[cat]
            diff = cat_data - cat_mean
            S_W += diff.T @ diff
            mean_diff = (cat_mean - overall_centroid).reshape(-1, 1)
            S_B += len(idx) * mean_diff @ mean_diff.T
        S_W += 1e-6 * np.eye(d_model) * np.trace(S_W) / d_model
        
        try:
            eigvals, eigvecs = np.linalg.eigh(np.linalg.solve(S_W, S_B))
            sort_idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[sort_idx]
            eigvecs = eigvecs[:, sort_idx]
            n_lda = min(N_CATEGORIES - 1, d_model)
            lda_dirs = eigvecs[:, :n_lda].T  # (n_lda, d_model)
        except:
            log(f"  Layer {l}: LDA failed, using PCA")
            pca, _, Vt, _, _ = compute_pca_full(dh_proper)
            if pca is None:
                continue
            lda_dirs = Vt[:6]
            n_lda = 6
        
        log(f"\n  Layer {l}:")
        
        # ===== Patch Type A: W_U Readout Patch =====
        # Decompose dh_proper into category and non-category components
        lda_proj = lda_dirs.T @ lda_dirs  # (d, d) projection matrix
        dh_cat = (lda_proj @ dh_proper.T).T       # (n, d) category component
        dh_noncat = dh_proper - dh_cat              # (n, d) non-category component
        
        # Compute logit differences for different patch conditions
        def compute_logit_diffs(dh_vec, W_U, tokenizer, pairs):
            """Compute target-competitor logit diff for each pair."""
            diffs = []
            valid_pairs = []
            for pidx, (obj, target, competitor) in enumerate(pairs):
                t_tokens = tokenizer(target, add_special_tokens=False)["input_ids"]
                c_tokens = tokenizer(competitor, add_special_tokens=False)["input_ids"]
                if len(t_tokens) != 1 or len(c_tokens) != 1:
                    continue
                t_id, c_id = t_tokens[0], c_tokens[0]
                if W_U.shape[0] == dh_vec.shape[1]:
                    logit = W_U.T @ dh_vec[pidx]
                else:
                    logit = W_U @ dh_vec[pidx]
                diffs.append(float(logit[t_id] - logit[c_id]))
                valid_pairs.append(pidx)
            return np.array(diffs), valid_pairs
        
        # Baseline: full dh_proper
        logit_full, valid_idx = compute_logit_diffs(dh_proper, W_U, tokenizer, ALL_PAIRS)
        # Only category component
        logit_cat_only, _ = compute_logit_diffs(dh_cat, W_U, tokenizer, ALL_PAIRS)
        # Only non-category component
        logit_noncat_only, _ = compute_logit_diffs(dh_noncat, W_U, tokenizer, ALL_PAIRS)
        # Remove category: noncat only
        logit_removed_cat = logit_noncat_only
        # Remove non-category: cat only
        logit_removed_noncat = logit_cat_only
        
        # Correlations
        corr_full_cat = float(np.corrcoef(logit_full, logit_cat_only)[0, 1]) if len(logit_full) > 5 else None
        corr_full_noncat = float(np.corrcoef(logit_full, logit_noncat_only)[0, 1]) if len(logit_full) > 5 else None
        
        # Mean absolute logit effect
        mean_abs_full = float(np.mean(np.abs(logit_full)))
        mean_abs_cat = float(np.mean(np.abs(logit_cat_only)))
        mean_abs_noncat = float(np.mean(np.abs(logit_noncat_only)))
        
        # Fraction of logit explained by category component
        if mean_abs_full > 1e-10:
            frac_cat = mean_abs_cat / mean_abs_full
            frac_noncat = mean_abs_noncat / mean_abs_full
        else:
            frac_cat = frac_noncat = None
        
        log(f"    Logit readout patch:")
        log(f"      corr(full, cat_only) = {corr_full_cat:.4f}" if corr_full_cat else "      N/A")
        log(f"      corr(full, noncat_only) = {corr_full_noncat:.4f}" if corr_full_noncat else "      N/A")
        log(f"      |logit|_full={mean_abs_full:.4f}, |logit|_cat={mean_abs_cat:.4f}, |logit|_noncat={mean_abs_noncat:.4f}")
        log(f"      frac_cat={frac_cat:.4f}, frac_noncat={frac_noncat:.4f}" if frac_cat else "      N/A")
        
        # ===== Patch Type B: Residual Stream Causal Patch (for a subset) =====
        # Add category component to h_corrupt and see if logit changes
        # This requires running the model with modified residual stream
        
        # Select a small subset for causal patch (5 pairs per category)
        patch_pairs = []
        for cat in ALL_CATEGORIES:
            idx = cat_indices[cat]
            patch_pairs.extend(idx[:5])
        
        # For each selected pair, do:
        # 1. Run corrupt input through model
        # 2. At layer l, intercept h_post_corrupt and add dh_cat[pair_idx]
        # 3. Continue forward pass and see if output logit changes
        
        # Get the next layer's input layernorm
        next_l = l + 1
        if next_l >= n_layers:
            log(f"    Skipping causal patch (layer {l} is last layer)")
            causal_patch_results = None
        else:
            next_ln = None
            for attr_name in ["input_layernorm", "ln1"]:
                next_ln = getattr(layers[next_l], attr_name, None)
                if next_ln is not None:
                    break
            
            if next_ln is None:
                log(f"    Skipping causal patch (no LN found for layer {next_l})")
                causal_patch_results = None
            else:
                log(f"    Running causal patch on {len(patch_pairs)} pairs...")
                
                input_device = next(model.parameters()).device
                causal_results = {
                    "baseline_logit_diff": [],
                    "patched_logit_diff": [],
                    "patched_with_noncat_logit_diff": [],
                }
                
                for pidx in patch_pairs[:30]:  # Limit for speed
                    obj, target, competitor = ALL_PAIRS[pidx]
                    t_tokens = tokenizer(target, add_special_tokens=False)["input_ids"]
                    c_tokens = tokenizer(competitor, add_special_tokens=False)["input_ids"]
                    if len(t_tokens) != 1 or len(c_tokens) != 1:
                        continue
                    t_id, c_id = t_tokens[0], c_tokens[0]
                    
                    corrupt_prompt = TEMPLATE.format(obj=CORRUPTED_BASELINE, attr=target)
                    
                    # Baseline corrupt run (no patch)
                    with torch.no_grad():
                        corrupt_ids = tokenizer(corrupt_prompt, return_tensors="pt",
                                              truncation=True, max_length=64)["input_ids"].to(input_device)
                        corrupt_mask = tokenizer(corrupt_prompt, return_tensors="pt",
                                               truncation=True, max_length=64)["attention_mask"].to(input_device)
                        baseline_out = model(input_ids=corrupt_ids, attention_mask=corrupt_mask)
                    
                    baseline_logit = baseline_out.logits[0, -1].float().cpu().numpy()
                    baseline_diff = float(baseline_logit[t_id] - baseline_logit[c_id])
                    
                    # Patched run: add category component at layer l
                    patch_vector_cat = torch.tensor(dh_cat[pidx], dtype=torch.bfloat16).to(input_device)
                    patch_vector_noncat = torch.tensor(dh_noncat[pidx], dtype=torch.bfloat16).to(input_device)
                    
                    captured_patch = {}
                    
                    def make_patch_hook(patch_vec, layer_l):
                        def hook(module, input, output):
                            # input[0] is the hidden state at this layer
                            if isinstance(output, tuple):
                                h = output[0]
                                last_pos = h.shape[1] - 1
                                h_patched = h.clone()
                                h_patched[0, last_pos] = h_patched[0, last_pos] + patch_vec
                                return (h_patched,) + output[1:]
                            return output
                        return hook
                    
                    # Patch with category component
                    hook_cat = layers[l].register_forward_hook(make_patch_hook(patch_vector_cat, l))
                    with torch.no_grad():
                        patched_cat_out = model(input_ids=corrupt_ids, attention_mask=corrupt_mask)
                    hook_cat.remove()
                    
                    patched_cat_logit = patched_cat_out.logits[0, -1].float().cpu().numpy()
                    patched_cat_diff = float(patched_cat_logit[t_id] - patched_cat_logit[c_id])
                    
                    # Patch with non-category component
                    hook_noncat = layers[l].register_forward_hook(make_patch_hook(patch_vector_noncat, l))
                    with torch.no_grad():
                        patched_noncat_out = model(input_ids=corrupt_ids, attention_mask=corrupt_mask)
                    hook_noncat.remove()
                    
                    patched_noncat_logit = patched_noncat_out.logits[0, -1].float().cpu().numpy()
                    patched_noncat_diff = float(patched_noncat_logit[t_id] - patched_noncat_logit[c_id])
                    
                    del baseline_out, patched_cat_out, patched_noncat_out
                    torch.cuda.empty_cache()
                    
                    causal_results["baseline_logit_diff"].append(baseline_diff)
                    causal_results["patched_logit_diff"].append(patched_cat_diff)
                    causal_results["patched_with_noncat_logit_diff"].append(patched_noncat_diff)
                
                # Summarize
                if len(causal_results["baseline_logit_diff"]) > 3:
                    bl = np.array(causal_results["baseline_logit_diff"])
                    pc = np.array(causal_results["patched_logit_diff"])
                    pn = np.array(causal_results["patched_with_noncat_logit_diff"])
                    
                    # Clean run logit diff (for reference)
                    clean_logit_diffs = []
                    for pidx in patch_pairs[:30]:
                        obj, target, competitor = ALL_PAIRS[pidx]
                        t_tokens = tokenizer(target, add_special_tokens=False)["input_ids"]
                        c_tokens = tokenizer(competitor, add_special_tokens=False)["input_ids"]
                        if len(t_tokens) != 1 or len(c_tokens) != 1:
                            continue
                        t_id, c_id = t_tokens[0], c_tokens[0]
                        if W_U.shape[0] == d_model:
                            logit_c = W_U.T @ d["h_post_clean"][pidx]
                        else:
                            logit_c = W_U @ d["h_post_clean"][pidx]
                        clean_logit_diffs.append(float(logit_c[t_id] - logit_c[c_id]))
                    
                    cl = np.array(clean_logit_diffs) if clean_logit_diffs else None
                    
                    mean_bl = float(np.mean(bl))
                    mean_pc = float(np.mean(pc))
                    mean_pn = float(np.mean(pn))
                    mean_cl = float(np.mean(cl)) if cl is not None else None
                    
                    # Does category patch move toward clean?
                    delta_cat = mean_pc - mean_bl
                    delta_noncat = mean_pn - mean_bl
                    delta_clean = mean_cl - mean_bl if mean_cl is not None else None
                    
                    causal_patch_results = {
                        "mean_baseline_logit_diff": round(mean_bl, 4),
                        "mean_clean_logit_diff": round(mean_cl, 4) if mean_cl else None,
                        "mean_patched_cat_logit_diff": round(mean_pc, 4),
                        "mean_patched_noncat_logit_diff": round(mean_pn, 4),
                        "delta_cat_patch": round(delta_cat, 4),
                        "delta_noncat_patch": round(delta_noncat, 4),
                        "delta_clean": round(delta_clean, 4) if delta_clean else None,
                        "cat_patch_toward_clean": bool(delta_cat * delta_clean > 0) if delta_clean else None,
                        "n_patched": len(bl),
                    }
                    
                    log(f"    Causal patch results:")
                    log(f"      Baseline (corrupt): {mean_bl:.4f}")
                    log(f"      Clean: {mean_cl:.4f}" if mean_cl else "      Clean: N/A")
                    log(f"      +Cat patch: {mean_pc:.4f} (Δ={delta_cat:.4f})")
                    log(f"      +Noncat patch: {mean_pn:.4f} (Δ={delta_noncat:.4f})")
                    log(f"      Cat toward clean? {causal_patch_results['cat_patch_toward_clean']}")
                else:
                    causal_patch_results = None
        
        results[l_str] = {
            "lda_subspace_dim": int(n_lda),
            "logit_readout_patch": {
                "corr_full_cat": round(corr_full_cat, 4) if corr_full_cat else None,
                "corr_full_noncat": round(corr_full_noncat, 4) if corr_full_noncat else None,
                "mean_abs_full": round(mean_abs_full, 4),
                "mean_abs_cat": round(mean_abs_cat, 4),
                "mean_abs_noncat": round(mean_abs_noncat, 4),
                "frac_cat": round(frac_cat, 4) if frac_cat else None,
                "frac_noncat": round(frac_noncat, 4) if frac_noncat else None,
            },
            "causal_patch": causal_patch_results,
        }
    
    return results


# ===== Part 3: Jacobian Linear Prediction Error =====
def jacobian_prediction_error(all_data, model_name):
    """
    Test whether the first-order Jacobian approximation can predict
    the proper post-RMSNorm Δh.
    
    Compare:
    - proper_delta = RMSNorm(h_clean) - RMSNorm(h_corrupt)
    - linear_delta = J(h_mid) · (h_clean - h_corrupt)
    
    where h_mid = (h_clean + h_corrupt) / 2
    """
    log("\n" + "="*60)
    log("Part 3: Jacobian Linear Prediction Error")
    log("="*60)
    
    results = {}
    
    for l_str in sorted(all_data.keys(), key=int):
        d = all_data[l_str]
        n_pairs = d["dh"].shape[0]
        d_model = d["dh"].shape[1]
        l = int(l_str)
        ln_weight = d.get("ln_weight", None)
        
        log(f"\n  Layer {l}:")
        
        h_clean = d["h_post_clean"]
        h_corrupt = d["h_post_corrupt"]
        dh_raw = h_clean - h_corrupt
        
        # Compute proper post-RMSNorm
        h_clean_norm = np.zeros_like(h_clean)
        h_corrupt_norm = np.zeros_like(h_corrupt)
        for i in range(n_pairs):
            h_clean_norm[i] = rms_norm_single(h_clean[i], ln_weight)
            h_corrupt_norm[i] = rms_norm_single(h_corrupt[i], ln_weight)
        
        dh_proper = h_clean_norm - h_corrupt_norm
        
        # Compute Jacobian-based linear prediction
        # Use midpoint h_mid = (h_clean + h_corrupt) / 2
        h_mid = (h_clean + h_corrupt) / 2
        
        # For each pair, compute J(h_mid) · Δh_raw
        # This is expensive if d_model is large, so compute it efficiently
        # J(h) @ v = gamma * sqrt(d)/rms * (v - (h @ v) / (d * rms^2) * h)
        # where rms^2 = mean(h^2) + eps
        
        dh_linear = np.zeros_like(dh_proper)
        cos_proper_linear = np.zeros(n_pairs)
        relative_error = np.zeros(n_pairs)
        
        eps = 1e-6
        sqrt_d = np.sqrt(d_model)
        
        for i in range(n_pairs):
            hm = h_mid[i]
            v = dh_raw[i]
            
            rms_mid = np.sqrt(np.mean(hm**2) + eps)
            rms3 = rms_mid**3
            
            # J(h_mid) @ v = gamma * sqrt(d)/rms * (v - (hm @ v) / (d * rms^2) * hm)
            h_dot_v = np.dot(hm, v)
            Jv = sqrt_d / rms_mid * (v - h_dot_v / (d_model * (np.mean(hm**2) + eps)) * hm)
            
            if ln_weight is not None:
                Jv = ln_weight * Jv
            
            dh_linear[i] = Jv
            
            # Cosine similarity
            norm_proper = np.linalg.norm(dh_proper[i])
            norm_linear = np.linalg.norm(Jv)
            if norm_proper > 1e-10 and norm_linear > 1e-10:
                cos_proper_linear[i] = cosine_sim(dh_proper[i], Jv)
                relative_error[i] = np.linalg.norm(dh_proper[i] - Jv) / norm_proper
            else:
                cos_proper_linear[i] = 0
                relative_error[i] = 1.0
            
            if i % 30 == 0:
                log(f"    Pair {i+1}/{n_pairs}: cos={cos_proper_linear[i]:.4f}, "
                    f"rel_err={relative_error[i]:.4f}")
        
        # Also try using h_clean as the expansion point
        dh_linear_clean = np.zeros_like(dh_proper)
        cos_proper_linear_clean = np.zeros(n_pairs)
        
        for i in range(n_pairs):
            hc = h_clean[i]
            v = dh_raw[i]
            rms_c = np.sqrt(np.mean(hc**2) + eps)
            h_dot_v = np.dot(hc, v)
            Jv_c = sqrt_d / rms_c * (v - h_dot_v / (d_model * (np.mean(hc**2) + eps)) * hc)
            if ln_weight is not None:
                Jv_c = ln_weight * Jv_c
            dh_linear_clean[i] = Jv_c
            norm_p = np.linalg.norm(dh_proper[i])
            norm_l = np.linalg.norm(Jv_c)
            if norm_p > 1e-10 and norm_l > 1e-10:
                cos_proper_linear_clean[i] = cosine_sim(dh_proper[i], Jv_c)
        
        # Summary statistics
        mean_cos_mid = float(np.mean(cos_proper_linear))
        mean_cos_clean = float(np.mean(cos_proper_linear_clean))
        mean_rel_err = float(np.mean(relative_error))
        
        # Does the linear prediction preserve category structure?
        same_cos_proper = []
        cross_cos_proper = []
        same_cos_linear = []
        cross_cos_linear = []
        
        for i in range(min(n_pairs, 100)):
            for j in range(i+1, min(n_pairs, 100)):
                if PAIR_CATEGORIES[i] == PAIR_CATEGORIES[j]:
                    same_cos_proper.append(cosine_sim(dh_proper[i], dh_proper[j]))
                    same_cos_linear.append(cosine_sim(dh_linear[i], dh_linear[j]))
                else:
                    cross_cos_proper.append(cosine_sim(dh_proper[i], dh_proper[j]))
                    cross_cos_linear.append(cosine_sim(dh_linear[i], dh_linear[j]))
        
        gap_proper = float(np.mean(same_cos_proper)) - float(np.mean(cross_cos_proper))
        gap_linear = float(np.mean(same_cos_linear)) - float(np.mean(cross_cos_linear))
        
        log(f"    Mean cos(proper, linear_mid) = {mean_cos_mid:.4f}")
        log(f"    Mean cos(proper, linear_clean) = {mean_cos_clean:.4f}")
        log(f"    Mean relative error = {mean_rel_err:.4f}")
        log(f"    Category gap: proper={gap_proper:.4f}, linear_mid={gap_linear:.4f}")
        
        # How much does ||h|| / ||Δh|| ratio affect the error?
        h_norms = np.linalg.norm(h_clean, axis=1)
        dh_norms = np.linalg.norm(dh_raw, axis=1)
        ratio_h_dh = h_norms / (dh_norms + 1e-10)
        
        corr_ratio_cos = float(np.corrcoef(ratio_h_dh, cos_proper_linear)[0, 1])
        corr_ratio_relerr = float(np.corrcoef(ratio_h_dh, relative_error)[0, 1])
        
        log(f"    Corr(||h||/||Δh||, cos) = {corr_ratio_cos:.4f}")
        log(f"    Corr(||h||/||Δh||, rel_err) = {corr_ratio_relerr:.4f}")
        
        results[l_str] = {
            "mean_cos_midpoint": round(mean_cos_mid, 4),
            "mean_cos_clean_expansion": round(mean_cos_clean, 4),
            "mean_relative_error": round(mean_rel_err, 4),
            "gap_proper": round(gap_proper, 4),
            "gap_linear": round(gap_linear, 4),
            "corr_ratio_cos": round(corr_ratio_cos, 4),
            "corr_ratio_relerr": round(corr_ratio_relerr, 4),
            "mean_h_norm": round(float(np.mean(h_norms)), 2),
            "mean_dh_norm": round(float(np.mean(dh_norms)), 4),
            "mean_ratio_h_dh": round(float(np.mean(ratio_h_dh)), 2),
        }
    
    return results


# ===== Main =====
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    
    log(f"Phase 380: Category Subspace Causal Patch — {model_name}")
    log(f"Pairs: {len(ALL_PAIRS)}, Categories: {N_CATEGORIES} = {ALL_CATEGORIES}")
    
    # Target layers per model
    if model_name == "deepseek7b":
        target_layers = [4, 8, 24]
    elif model_name == "qwen3":
        target_layers = [4, 28]
    elif model_name == "glm4":
        target_layers = [4, 30]
    else:
        target_layers = [4]
    
    # Load model
    t0 = time.time()
    model, tokenizer = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    log(f"  Model loaded: {info.model_class}, {info.n_layers} layers, d={info.d_model}")
    log(f"  Load time: {time.time()-t0:.1f}s")
    
    # Collect residual states
    t0 = time.time()
    all_data = collect_residual_states(model, tokenizer, model_name, target_layers)
    log(f"  Data collection: {time.time()-t0:.1f}s")
    
    # Part 1: Supervised Category Subspace Decoding
    t0 = time.time()
    part1_results = category_subspace_decoding(all_data, model, tokenizer, model_name)
    log(f"  Part 1 done: {time.time()-t0:.1f}s")
    
    # Part 2: Causal Patch Experiments
    t0 = time.time()
    part2_results = causal_patch_experiment(all_data, model, tokenizer, model_name)
    log(f"  Part 2 done: {time.time()-t0:.1f}s")
    
    # Part 3: Jacobian Prediction Error
    t0 = time.time()
    part3_results = jacobian_prediction_error(all_data, model_name)
    log(f"  Part 3 done: {time.time()-t0:.1f}s")
    
    # Save results
    output_dir = f"results/phase380_category_subspace_causal_patch"
    os.makedirs(output_dir, exist_ok=True)
    
    output = {
        "model": model_name,
        "n_pairs": len(ALL_PAIRS),
        "n_categories": N_CATEGORIES,
        "categories": ALL_CATEGORIES,
        "target_layers": target_layers,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "part1_category_subspace": part1_results,
        "part2_causal_patch": part2_results,
        "part3_jacobian_error": part3_results,
    }
    
    output_path = os.path.join(output_dir, f"{model_name}_phase380.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    log(f"Results saved to {output_path}")
    
    # Release model
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    
    log(f"Phase 380 complete for {model_name}!")


if __name__ == "__main__":
    main()
