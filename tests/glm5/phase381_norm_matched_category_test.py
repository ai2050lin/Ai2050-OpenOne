"""
Phase 381: 范数比信号因果验证
==============================

核心问题：DS7B的类别判别是否100%来自范数差异？
- Phase 380发现：DS7B LDA0与norm_ratio相关0.92
- 如果范数匹配后类别不可分类→DS7B的"类别判别"本质上是"范数判别"

实验设计：
Part 1: 范数匹配实验
  - 对每对clean/corrupt差值，计算norm_ratio = ‖h_clean‖ / ‖h_corrupt‖
  - 构造"等范数"对照组：将corrupt的h缩放到与clean同范数
  - 在等范数条件下重做centroid分类
  - 对比：原始空间 vs 等范数空间的类别分类准确率

Part 2: 范数比信号 vs 类别信号的因果分离
  - 构造"纯范数"patch：只改变范数，不改变方向
  - 构造"纯方向"patch：只改变方向，保持范数
  - 观察两者对logit的因果效应

Part 3: 深层类别增强追踪
  - DS7B L4→L8→L12→L16→L20→L24
  - 每层：centroid分类准确率、norm_ratio相关、等范数分类准确率
  - 追踪类别信息从弱到强的增强机制

用法:
  python tests/glm5/phase381_norm_matched_category_test.py qwen3
  python tests/glm5/phase381_norm_matched_category_test.py deepseek7b
  python tests/glm5/phase381_norm_matched_category_test.py glm4
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


# ===== Binding pairs (same as Phase 379/380) =====
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
    ("violet", "purple", "yellow"), ("indigo", "blue", "red"), ("chocolate", "brown", "white"),
    ("olive", "green", "red"), ("magenta", "purple", "blue"),
]
TEMP_PAIRS = [
    ("fire", "hot", "cold"), ("desert", "hot", "cold"), ("lava", "hot", "cold"),
    ("ice", "cold", "hot"), ("snow", "cold", "hot"), ("volcano", "hot", "cold"),
    ("furnace", "hot", "cold"), ("glacier", "cold", "hot"),
    ("oven", "hot", "cold"), ("frost", "cold", "hot"), ("magma", "hot", "cold"),
    ("winter", "cold", "hot"), ("summer", "hot", "cold"), ("arctic", "cold", "hot"),
    ("stove", "hot", "cold"), ("blizzard", "cold", "hot"), ("tundra", "cold", "hot"),
    ("inferno", "hot", "cold"), ("iceberg", "cold", "hot"),
    ("sauna", "hot", "cold"), ("candle", "hot", "cold"), ("tropics", "hot", "cold"),
    ("permafrost", "cold", "hot"), ("thermos", "hot", "cold"), ("freezer", "cold", "hot"),
]
MOISTURE_PAIRS = [
    ("rain", "wet", "dry"), ("ocean", "wet", "dry"), ("river", "wet", "dry"),
    ("sand", "dry", "wet"), ("dust", "dry", "wet"), ("bone", "dry", "wet"),
    ("swamp", "wet", "dry"), ("desert", "dry", "wet"),
    ("lake", "wet", "dry"), ("sponge", "wet", "dry"), ("cracker", "dry", "wet"),
    ("fog", "wet", "dry"), ("prairie", "dry", "wet"), ("puddle", "wet", "dry"),
    ("cactus", "dry", "wet"), ("waterfall", "wet", "dry"),
    ("dew", "wet", "dry"), ("asphalt", "dry", "wet"), ("marsh", "wet", "dry"),
    ("salt", "dry", "wet"), ("towel", "wet", "dry"), ("tinder", "dry", "wet"),
    ("rainforest", "wet", "dry"), ("drought", "dry", "wet"),
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

def rms_norm_single(x, weight=None, eps=1e-6):
    d = x.shape[-1]
    rms = np.sqrt(np.mean(x**2) + eps)
    result = x / rms * np.sqrt(d)
    if weight is not None:
        result = result * weight
    return result

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


# ===== Centroid classifier (LOO) =====
def loo_centroid_accuracy(X, labels):
    """Leave-one-out nearest centroid classification."""
    n = X.shape[0]
    unique_labels = sorted(set(labels))
    correct = 0
    for i in range(n):
        # Compute centroids without sample i
        centroids = {}
        for lab in unique_labels:
            mask = [j for j in range(n) if j != i and labels[j] == lab]
            if len(mask) > 0:
                centroids[lab] = np.mean(X[mask], axis=0)
            else:
                centroids[lab] = np.zeros(X.shape[1])
        # Classify sample i
        dists = {lab: np.linalg.norm(X[i] - c) for lab, c in centroids.items()}
        pred = min(dists, key=dists.get)
        if pred == labels[i]:
            correct += 1
    return correct / n


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
        target_token_ids = []
        competitor_token_ids = []
        
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
            
            # Get target/competitor logit from clean output
            logits = clean_out.logits[0, last_pos].detach().cpu().float().numpy()
            t_id = tokenizer.encode(" " + target, add_special_tokens=False)
            c_id = tokenizer.encode(" " + competitor, add_special_tokens=False)
            target_token_ids.append(t_id[0] if t_id else -1)
            competitor_token_ids.append(c_id[0] if c_id else -1)
            
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
        
        all_data[str(l)] = {
            "pre_mlp_clean": pre_mlp_clean, "pre_mlp_corrupt": pre_mlp_corrupt,
            "h_post_clean": h_post_clean, "h_post_corrupt": h_post_corrupt,
            "ln_weight": ln_weight,
            "W_down": W_down, "W_gate": W_gate, "W_up": W_up,
            "target_token_ids": target_token_ids,
            "competitor_token_ids": competitor_token_ids,
        }
        
        log(f"    Layer {l} done in {time.time()-t_l:.1f}s")
    
    return all_data


# ===== Part 1: Norm-matched category classification =====
def part1_norm_matched_classification(all_data, model_name):
    """
    Core experiment: After norm-matching, is category still classifiable?
    
    Method:
    1. Compute dh_proper = RMSNorm(h_clean) - RMSNorm(h_corrupt) for each pair
    2. Compute norm_ratio = ‖h_clean‖ / ‖h_corrupt‖ for each pair
    3. Create norm-matched version: h_corrupt_matched = h_corrupt * (‖h_clean‖ / ‖h_corrupt‖)
       → Now ‖h_corrupt_matched‖ = ‖h_clean‖
    4. Compute dh_norm_matched = RMSNorm(h_clean) - RMSNorm(h_corrupt_matched)
    5. Compare category classification: original vs norm-matched
    
    If category accuracy drops to chance after norm-matching → category = norm
    """
    log("\n" + "="*60)
    log("Part 1: Norm-Matched Category Classification")
    log("="*60)
    
    results = {}
    
    for l_str in sorted(all_data.keys(), key=int):
        d = all_data[l_str]
        l = int(l_str)
        ln_weight = d.get("ln_weight", None)
        n_pairs = d["h_post_clean"].shape[0]
        d_model = d["h_post_clean"].shape[1]
        
        h_clean = d["h_post_clean"]      # (n, d)
        h_corrupt = d["h_post_corrupt"]  # (n, d)
        
        # ---- Compute PROPER post-RMSNorm ----
        h_clean_norm = np.zeros_like(h_clean)
        h_corrupt_norm = np.zeros_like(h_corrupt)
        for i in range(n_pairs):
            h_clean_norm[i] = rms_norm_single(h_clean[i], ln_weight)
            h_corrupt_norm[i] = rms_norm_single(h_corrupt[i], ln_weight)
        
        dh_proper = h_clean_norm - h_corrupt_norm  # (n, d)
        
        # ---- Compute norm_ratio ----
        norm_clean = np.linalg.norm(h_clean, axis=1)     # (n,)
        norm_corrupt = np.linalg.norm(h_corrupt, axis=1)  # (n,)
        norm_ratio = norm_clean / (norm_corrupt + 1e-10)  # (n,)
        norm_diff = norm_clean - norm_corrupt  # (n,)
        
        # ---- Correlation: norm_ratio vs PC1 of dh_proper ----
        # PCA on dh_proper
        M = dh_proper - dh_proper.mean(axis=0, keepdims=True)
        try:
            U, S, Vt = np.linalg.svd(M, full_matrices=False)
        except:
            log(f"  Layer {l}: SVD failed, skipping")
            continue
        pc1_scores = U[:, 0] * S[0]
        pc2_scores = U[:, 1] * S[1]
        
        norm_ratio_pc1_corr = cosine_sim(norm_ratio - norm_ratio.mean(), pc1_scores)
        norm_diff_pc1_corr = cosine_sim(norm_diff - norm_diff.mean(), pc1_scores)
        
        # ---- Category classification on dh_proper (LOO centroid) ----
        labels = [PAIR_CATEGORIES[i] for i in range(n_pairs)]
        
        # Use first 5 centroid directions for classification
        cat_centroids = {}
        for cat in ALL_CATEGORIES:
            idx = [j for j, c in enumerate(labels) if c == cat]
            cat_centroids[cat] = np.mean(dh_proper[idx], axis=0)
        
        overall_centroid = np.mean(dh_proper, axis=0)
        centroid_diffs = np.array([cat_centroids[cat] - overall_centroid for cat in ALL_CATEGORIES])
        
        # Project onto top-5 centroid directions
        Q, _ = np.linalg.qr(centroid_diffs.T)
        Q5 = Q[:, :5]
        dh_proj = dh_proper @ Q5
        
        acc_original = loo_centroid_accuracy(dh_proj, labels)
        
        # ---- NORM-MATCHED version ----
        # Scale corrupt to match clean norm: h_corrupt_matched = h_corrupt * (norm_clean / norm_corrupt)
        h_corrupt_matched = h_corrupt * (norm_clean / (norm_corrupt + 1e-10))[:, None]
        
        # Compute PROPER post-RMSNorm for norm-matched version
        h_corrupt_matched_norm = np.zeros_like(h_corrupt_matched)
        for i in range(n_pairs):
            h_corrupt_matched_norm[i] = rms_norm_single(h_corrupt_matched[i], ln_weight)
        
        dh_norm_matched = h_clean_norm - h_corrupt_matched_norm  # (n, d)
        
        # Category classification on norm-matched dh
        cat_centroids_nm = {}
        for cat in ALL_CATEGORIES:
            idx = [j for j, c in enumerate(labels) if c == cat]
            cat_centroids_nm[cat] = np.mean(dh_norm_matched[idx], axis=0)
        
        overall_centroid_nm = np.mean(dh_norm_matched, axis=0)
        centroid_diffs_nm = np.array([cat_centroids_nm[cat] - overall_centroid_nm for cat in ALL_CATEGORIES])
        Q_nm, _ = np.linalg.qr(centroid_diffs_nm.T)
        Q5_nm = Q_nm[:, :5]
        dh_proj_nm = dh_norm_matched @ Q5_nm
        
        acc_norm_matched = loo_centroid_accuracy(dh_proj_nm, labels)
        
        # ---- NORM-MATCHED: also try raw PCA projection (not centroid) ----
        M_nm = dh_norm_matched - dh_norm_matched.mean(axis=0, keepdims=True)
        try:
            U_nm, S_nm, Vt_nm = np.linalg.svd(M_nm, full_matrices=False)
            dh_pca5_nm = U_nm[:, :5] * S_nm[:5]
            acc_pca5_nm = loo_centroid_accuracy(dh_pca5_nm, labels)
        except:
            acc_pca5_nm = -1
        
        # ---- Also test on RAW h (before RMSNorm) ----
        dh_raw = h_clean - h_corrupt
        
        cat_centroids_raw = {}
        for cat in ALL_CATEGORIES:
            idx = [j for j, c in enumerate(labels) if c == cat]
            cat_centroids_raw[cat] = np.mean(dh_raw[idx], axis=0)
        overall_centroid_raw = np.mean(dh_raw, axis=0)
        centroid_diffs_raw = np.array([cat_centroids_raw[cat] - overall_centroid_raw for cat in ALL_CATEGORIES])
        Q_raw, _ = np.linalg.qr(centroid_diffs_raw.T)
        Q5_raw = Q_raw[:, :5]
        dh_raw_proj = dh_raw @ Q5_raw
        
        acc_raw = loo_centroid_accuracy(dh_raw_proj, labels)
        
        # ---- Norm-matched RAW ----
        dh_raw_nm = h_clean - h_corrupt_matched
        cat_centroids_raw_nm = {}
        for cat in ALL_CATEGORIES:
            idx = [j for j, c in enumerate(labels) if c == cat]
            cat_centroids_raw_nm[cat] = np.mean(dh_raw_nm[idx], axis=0)
        overall_centroid_raw_nm = np.mean(dh_raw_nm, axis=0)
        centroid_diffs_raw_nm = np.array([cat_centroids_raw_nm[cat] - overall_centroid_raw_nm for cat in ALL_CATEGORIES])
        Q_raw_nm, _ = np.linalg.qr(centroid_diffs_raw_nm.T)
        Q5_raw_nm = Q_raw_nm[:, :5]
        dh_raw_nm_proj = dh_raw_nm @ Q5_raw_nm
        
        acc_raw_nm = loo_centroid_accuracy(dh_raw_nm_proj, labels)
        
        # ---- Per-category norm_ratio analysis ----
        cat_norm_ratios = {}
        for cat in ALL_CATEGORIES:
            idx = [j for j, c in enumerate(labels) if c == cat]
            cat_norm_ratios[cat] = {
                "mean": float(np.mean(norm_ratio[idx])),
                "std": float(np.std(norm_ratio[idx])),
            }
        
        # ---- ANOVA: is norm_ratio different across categories? ----
        cat_groups = [norm_ratio[[j for j, c in enumerate(labels) if c == cat]] for cat in ALL_CATEGORIES]
        # F-statistic
        grand_mean = np.mean(norm_ratio)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in cat_groups)
        ss_within = sum(np.sum((g - np.mean(g))**2) for g in cat_groups)
        df_between = len(ALL_CATEGORIES) - 1
        df_within = n_pairs - len(ALL_CATEGORIES)
        f_stat = (ss_between / max(df_between, 1)) / (max(ss_within / max(df_within, 1), 1e-10))
        
        # ---- Key metric: category variance explained by norm_ratio ----
        # R² from linear regression: category label ~ norm_ratio
        # Use one-hot encoding for categories
        cat_onehot = np.zeros((n_pairs, N_CATEGORIES))
        for i, c in enumerate(labels):
            cat_onehot[i, ALL_CATEGORIES.index(c)] = 1
        
        # CCA: correlation between norm_ratio and category one-hot
        # Simplified: for each category, compute point-biserial correlation with norm_ratio
        cat_norm_corrs = {}
        for cat in ALL_CATEGORIES:
            binary = np.array([1 if c == cat else 0 for c in labels], dtype=float)
            corr = np.corrcoef(norm_ratio, binary)[0, 1]
            cat_norm_corrs[cat] = float(corr)
        
        # ---- Within-category norm-matched classification ----
        # After removing norm effects, what's left for classification?
        # Subtract norm_ratio component from dh_proper
        # Regress out norm_ratio: dh_residual = dh_proper - norm_ratio * beta
        norm_ratio_centered = (norm_ratio - norm_ratio.mean())[:, None]  # (n, 1)
        beta = np.linalg.lstsq(norm_ratio_centered, dh_proper, rcond=None)[0]  # (1, d)
        dh_no_norm = dh_proper - norm_ratio_centered @ beta  # (n, d)
        
        # Classify on dh_no_norm
        cat_centroids_nn = {}
        for cat in ALL_CATEGORIES:
            idx = [j for j, c in enumerate(labels) if c == cat]
            cat_centroids_nn[cat] = np.mean(dh_no_norm[idx], axis=0)
        overall_centroid_nn = np.mean(dh_no_norm, axis=0)
        centroid_diffs_nn = np.array([cat_centroids_nn[cat] - overall_centroid_nn for cat in ALL_CATEGORIES])
        Q_nn, _ = np.linalg.qr(centroid_diffs_nn.T)
        Q5_nn = Q_nn[:, :5]
        dh_no_norm_proj = dh_no_norm @ Q5_nn
        
        acc_no_norm = loo_centroid_accuracy(dh_no_norm_proj, labels)
        
        # Also regress out norm_diff
        norm_diff_centered = (norm_diff - norm_diff.mean())[:, None]
        beta_diff = np.linalg.lstsq(norm_diff_centered, dh_proper, rcond=None)[0]
        dh_no_normdiff = dh_proper - norm_diff_centered @ beta_diff
        
        cat_centroids_nd = {}
        for cat in ALL_CATEGORIES:
            idx = [j for j, c in enumerate(labels) if c == cat]
            cat_centroids_nd[cat] = np.mean(dh_no_normdiff[idx], axis=0)
        overall_centroid_nd = np.mean(dh_no_normdiff, axis=0)
        centroid_diffs_nd = np.array([cat_centroids_nd[cat] - overall_centroid_nd for cat in ALL_CATEGORIES])
        Q_nd, _ = np.linalg.qr(centroid_diffs_nd.T)
        Q5_nd = Q_nd[:, :5]
        dh_no_normdiff_proj = dh_no_normdiff @ Q5_nd
        
        acc_no_normdiff = loo_centroid_accuracy(dh_no_normdiff_proj, labels)
        
        # ---- Summary ----
        res = {
            "layer": l,
            "n_pairs": n_pairs,
            "norm_ratio_stats": {
                "mean": float(np.mean(norm_ratio)),
                "std": float(np.std(norm_ratio)),
                "min": float(np.min(norm_ratio)),
                "max": float(np.max(norm_ratio)),
            },
            "pc1_norm_ratio_corr": float(norm_ratio_pc1_corr),
            "pc1_norm_diff_corr": float(norm_diff_pc1_corr),
            "classification": {
                "acc_proper_centroid5": float(acc_original),
                "acc_norm_matched_centroid5": float(acc_norm_matched),
                "acc_norm_matched_pca5": float(acc_pca5_nm),
                "acc_raw_centroid5": float(acc_raw),
                "acc_raw_norm_matched_centroid5": float(acc_raw_nm),
                "acc_no_norm_ratio_centroid5": float(acc_no_norm),
                "acc_no_norm_diff_centroid5": float(acc_no_normdiff),
                "chance": float(1.0 / N_CATEGORIES),
            },
            "f_stat_norm_ratio_across_cats": float(f_stat),
            "cat_norm_corrs": cat_norm_corrs,
            "cat_norm_ratios": cat_norm_ratios,
            "norm_matched_vs_original_drop": float(acc_original - acc_norm_matched),
            "norm_regressed_vs_original_drop": float(acc_original - acc_no_norm),
        }
        
        results[l_str] = res
        
        log(f"\n  Layer {l} Results:")
        log(f"    norm_ratio: mean={np.mean(norm_ratio):.4f}, std={np.std(norm_ratio):.4f}")
        log(f"    PC1 ~ norm_ratio corr: {norm_ratio_pc1_corr:.4f}")
        log(f"    PC1 ~ norm_diff corr: {norm_diff_pc1_corr:.4f}")
        log(f"    Classification accuracy:")
        log(f"      PROPER centroid(5d):     {acc_original:.3f}")
        log(f"      Norm-matched centroid(5d): {acc_norm_matched:.3f} (drop={acc_original-acc_norm_matched:+.3f})")
        log(f"      Norm-matched PCA(5d):    {acc_pca5_nm:.3f}")
        log(f"      Raw centroid(5d):        {acc_raw:.3f}")
        log(f"      Raw norm-matched:        {acc_raw_nm:.3f} (drop={acc_raw-acc_raw_nm:+.3f})")
        log(f"      No-norm-ratio regressed: {acc_no_norm:.3f} (drop={acc_original-acc_no_norm:+.3f})")
        log(f"      No-norm-diff regressed:  {acc_no_normdiff:.3f} (drop={acc_original-acc_no_normdiff:+.3f})")
        log(f"    F-stat(norm_ratio across cats): {f_stat:.2f}")
        log(f"    Chance: {1.0/N_CATEGORIES:.3f}")
    
    return results


# ===== Part 2: Norm vs Direction causal separation =====
def part2_norm_vs_direction_causal(all_data, model, tokenizer, model_name):
    """
    Separate the causal effect of norm change vs direction change.
    
    Method:
    1. "Pure norm" patch: scale h_corrupt to match h_clean's norm, keeping direction
       h_pure_norm = h_corrupt * (‖h_clean‖ / ‖h_corrupt‖)
       This changes norm but NOT direction.
       Effect on logit: pure norm effect.
    
    2. "Pure direction" patch: rotate h_corrupt to h_clean's direction, keeping norm
       h_pure_dir = h_corrupt / ‖h_corrupt‖ * ‖h_corrupt‖  
       Actually: h_pure_dir = unit(h_clean) * ‖h_corrupt‖
       This changes direction but NOT norm.
       Effect on logit: pure direction effect.
    
    3. Full patch: replace h_corrupt with h_clean → both norm and direction change
    """
    log("\n" + "="*60)
    log("Part 2: Norm vs Direction Causal Separation")
    log("="*60)
    
    layers = get_layers(model)
    input_device = next(model.parameters()).device
    info = get_model_info(model, model_name)
    n_total = len(ALL_PAIRS)
    
    # Sample pairs for patch test (use all for thoroughness)
    test_indices = list(range(n_total))
    
    results = {}
    
    for l_str in sorted(all_data.keys(), key=int):
        d = all_data[l_str]
        l = int(l_str)
        ln_weight = d.get("ln_weight", None)
        
        if l + 1 >= info.n_layers:
            continue
        
        log(f"\n  Layer {l}: Running causal patches...")
        
        # Collect patch results
        patch_data = {
            "clean_logit_diff": [],
            "corrupt_logit_diff": [],
            "pure_norm_logit_diff": [],
            "pure_dir_logit_diff": [],
            "full_logit_diff": [],
            "norm_ratio": [],
        }
        
        # Load W_U once for this layer
        W_U = get_W_U(model, model_name)
        
        for pidx in test_indices:
            obj, target, competitor = ALL_PAIRS[pidx]
            t_id = d["target_token_ids"][pidx]
            c_id = d["competitor_token_ids"][pidx]
            if t_id < 0 or c_id < 0:
                continue
            
            # Get clean/corrupt h at layer l output
            h_clean_vec = d["h_post_clean"][pidx]  # (d,)
            h_corrupt_vec = d["h_post_corrupt"][pidx]
            
            norm_c = np.linalg.norm(h_clean_vec)
            norm_r = np.linalg.norm(h_corrupt_vec)
            patch_data["norm_ratio"].append(float(norm_c / (norm_r + 1e-10)))
            
            # Construct pure-norm and pure-dir vectors
            # Pure norm: h_corrupt direction but h_clean norm
            h_pure_norm = h_corrupt_vec * (norm_c / (norm_r + 1e-10))
            
            # Pure direction: h_clean direction but h_corrupt norm
            h_pure_dir = h_clean_vec / (norm_c + 1e-10) * norm_r
            
            # Logit lens: RMSNorm → W_U projection → logit diff
            h_clean_norm_vec = rms_norm_single(h_clean_vec, ln_weight)
            h_corrupt_norm_vec = rms_norm_single(h_corrupt_vec, ln_weight)
            h_pure_norm_normed = rms_norm_single(h_pure_norm, ln_weight)
            h_pure_dir_normed = rms_norm_single(h_pure_dir, ln_weight)
            
            logit_clean = W_U @ h_clean_norm_vec
            logit_corrupt = W_U @ h_corrupt_norm_vec
            logit_pn = W_U @ h_pure_norm_normed
            logit_pd = W_U @ h_pure_dir_normed
            
            patch_data["clean_logit_diff"].append(float(logit_clean[t_id] - logit_clean[c_id]))
            patch_data["corrupt_logit_diff"].append(float(logit_corrupt[t_id] - logit_corrupt[c_id]))
            patch_data["pure_norm_logit_diff"].append(float(logit_pn[t_id] - logit_pn[c_id]))
            patch_data["pure_dir_logit_diff"].append(float(logit_pd[t_id] - logit_pd[c_id]))
            patch_data["full_logit_diff"].append(float(logit_clean[t_id] - logit_clean[c_id]))
            
            if pidx % 30 == 0:
                log(f"    Pair {pidx+1}/{len(test_indices)}")
        
        # Summarize
        valid = [i for i in range(len(patch_data["clean_logit_diff"]))
                 if patch_data["pure_norm_logit_diff"][i] is not None
                 and patch_data["pure_dir_logit_diff"][i] is not None]
        
        if len(valid) < 5:
            log(f"  Layer {l}: Not enough valid patches, skipping")
            continue
        
        clean_ld = np.array([patch_data["clean_logit_diff"][i] for i in valid])
        corrupt_ld = np.array([patch_data["corrupt_logit_diff"][i] for i in valid])
        pure_norm_ld = np.array([patch_data["pure_norm_logit_diff"][i] for i in valid])
        pure_dir_ld = np.array([patch_data["pure_dir_logit_diff"][i] for i in valid])
        full_ld = np.array([patch_data["full_logit_diff"][i] for i in valid])
        nr = np.array([patch_data["norm_ratio"][i] for i in valid])
        
        # Effect sizes
        delta_clean = clean_ld - corrupt_ld  # total clean-corrupt difference
        delta_pure_norm = pure_norm_ld - corrupt_ld  # effect of norm change only
        delta_pure_dir = pure_dir_ld - corrupt_ld   # effect of direction change only
        delta_full = full_ld - corrupt_ld            # effect of full replacement
        
        # Fraction of clean-corrupt gap explained by each
        total_gap = np.abs(delta_clean).mean()
        norm_gap = np.abs(delta_pure_norm).mean()
        dir_gap = np.abs(delta_pure_dir).mean()
        
        norm_fraction = norm_gap / (total_gap + 1e-10)
        dir_fraction = dir_gap / (total_gap + 1e-10)
        
        res = {
            "layer": l,
            "n_valid": len(valid),
            "mean_logit_diff": {
                "clean": float(np.mean(clean_ld)),
                "corrupt": float(np.mean(corrupt_ld)),
                "pure_norm": float(np.mean(pure_norm_ld)),
                "pure_dir": float(np.mean(pure_dir_ld)),
                "full": float(np.mean(full_ld)),
            },
            "mean_delta": {
                "clean_vs_corrupt": float(np.mean(delta_clean)),
                "pure_norm_vs_corrupt": float(np.mean(delta_pure_norm)),
                "pure_dir_vs_corrupt": float(np.mean(delta_pure_dir)),
                "full_vs_corrupt": float(np.mean(delta_full)),
            },
            "gap_fraction": {
                "norm": float(norm_fraction),
                "direction": float(dir_fraction),
            },
            "norm_ratio_stats": {
                "mean": float(np.mean(nr)),
                "std": float(np.std(nr)),
            }
        }
        
        results[l_str] = res
        
        log(f"  Layer {l} Results:")
        log(f"    Mean logit diff: clean={np.mean(clean_ld):.3f}, corrupt={np.mean(corrupt_ld):.3f}")
        log(f"    Pure norm patch: {np.mean(pure_norm_ld):.3f} (Δ={np.mean(delta_pure_norm):.3f})")
        log(f"    Pure dir patch:  {np.mean(pure_dir_ld):.3f} (Δ={np.mean(delta_pure_dir):.3f})")
        log(f"    Full patch:      {np.mean(full_ld):.3f} (Δ={np.mean(delta_full):.3f})")
        log(f"    Gap fraction: norm={norm_fraction:.3f}, dir={dir_fraction:.3f}")
    
    return results


# ===== Part 3: Deep layer tracking =====
def part3_deep_layer_tracking(all_data, model_name):
    """
    Track how category structure evolves from L4 to L24 in DS7B.
    For each layer, compute:
    1. Centroid classification accuracy
    2. Norm-matched classification accuracy  
    3. PC1-norm_ratio correlation
    4. Norm vs direction gap fraction
    """
    log("\n" + "="*60)
    log("Part 3: Deep Layer Tracking")
    log("="*60)
    
    results = {}
    
    for l_str in sorted(all_data.keys(), key=int):
        d = all_data[l_str]
        l = int(l_str)
        ln_weight = d.get("ln_weight", None)
        n_pairs = d["h_post_clean"].shape[0]
        
        h_clean = d["h_post_clean"]
        h_corrupt = d["h_post_corrupt"]
        
        # PROPER post-RMSNorm
        h_clean_norm = np.zeros_like(h_clean)
        h_corrupt_norm = np.zeros_like(h_corrupt)
        for i in range(n_pairs):
            h_clean_norm[i] = rms_norm_single(h_clean[i], ln_weight)
            h_corrupt_norm[i] = rms_norm_single(h_corrupt[i], ln_weight)
        
        dh_proper = h_clean_norm - h_corrupt_norm
        
        # Norm stats
        norm_clean = np.linalg.norm(h_clean, axis=1)
        norm_corrupt = np.linalg.norm(h_corrupt, axis=1)
        norm_ratio = norm_clean / (norm_corrupt + 1e-10)
        norm_diff = norm_clean - norm_corrupt
        
        # PCA on dh_proper
        M = dh_proper - dh_proper.mean(axis=0, keepdims=True)
        try:
            U, S, Vt = np.linalg.svd(M, full_matrices=False)
        except:
            continue
        
        pc1_scores = U[:, 0] * S[0]
        total_var = np.sum(S**2)
        explained = (S**2) / (total_var + 1e-10)
        eff_rank = int(np.searchsorted(np.cumsum(explained), 0.95) + 1)
        
        pc1_norm_corr = cosine_sim(norm_ratio - norm_ratio.mean(), pc1_scores)
        pc1_normdiff_corr = cosine_sim(norm_diff - norm_diff.mean(), pc1_scores)
        
        # Category classification
        labels = [PAIR_CATEGORIES[i] for i in range(n_pairs)]
        
        # PROPER centroid(5d)
        cat_centroids = {}
        for cat in ALL_CATEGORIES:
            idx = [j for j, c in enumerate(labels) if c == cat]
            cat_centroids[cat] = np.mean(dh_proper[idx], axis=0)
        overall_centroid = np.mean(dh_proper, axis=0)
        centroid_diffs = np.array([cat_centroids[cat] - overall_centroid for cat in ALL_CATEGORIES])
        Q, _ = np.linalg.qr(centroid_diffs.T)
        dh_proj = dh_proper @ Q[:, :5]
        acc_proper = loo_centroid_accuracy(dh_proj, labels)
        
        # Norm-matched
        h_corrupt_matched = h_corrupt * (norm_clean / (norm_corrupt + 1e-10))[:, None]
        h_corrupt_matched_norm = np.zeros_like(h_corrupt_matched)
        for i in range(n_pairs):
            h_corrupt_matched_norm[i] = rms_norm_single(h_corrupt_matched[i], ln_weight)
        dh_nm = h_clean_norm - h_corrupt_matched_norm
        
        cat_centroids_nm = {}
        for cat in ALL_CATEGORIES:
            idx = [j for j, c in enumerate(labels) if c == cat]
            cat_centroids_nm[cat] = np.mean(dh_nm[idx], axis=0)
        overall_nm = np.mean(dh_nm, axis=0)
        cd_nm = np.array([cat_centroids_nm[cat] - overall_nm for cat in ALL_CATEGORIES])
        Q_nm, _ = np.linalg.qr(cd_nm.T)
        dh_nm_proj = dh_nm @ Q_nm[:, :5]
        acc_nm = loo_centroid_accuracy(dh_nm_proj, labels)
        
        # Regress out norm_ratio
        nr_centered = (norm_ratio - norm_ratio.mean())[:, None]
        beta = np.linalg.lstsq(nr_centered, dh_proper, rcond=None)[0]
        dh_no_nr = dh_proper - nr_centered @ beta
        
        cat_centroids_no = {}
        for cat in ALL_CATEGORIES:
            idx = [j for j, c in enumerate(labels) if c == cat]
            cat_centroids_no[cat] = np.mean(dh_no_nr[idx], axis=0)
        overall_no = np.mean(dh_no_nr, axis=0)
        cd_no = np.array([cat_centroids_no[cat] - overall_no for cat in ALL_CATEGORIES])
        Q_no, _ = np.linalg.qr(cd_no.T)
        dh_no_proj = dh_no_nr @ Q_no[:, :5]
        acc_no_nr = loo_centroid_accuracy(dh_no_proj, labels)
        
        # PCA of norm-matched
        M_nm = dh_nm - dh_nm.mean(axis=0, keepdims=True)
        try:
            U_nm, S_nm, Vt_nm = np.linalg.svd(M_nm, full_matrices=False)
            pc1_nm_scores = U_nm[:, 0] * S_nm[0]
            eff_rank_nm = int(np.searchsorted(np.cumsum(S_nm**2 / (np.sum(S_nm**2) + 1e-10)), 0.95) + 1)
            pc1_var_nm = float((S_nm[0]**2) / (np.sum(S_nm**2) + 1e-10))
        except:
            eff_rank_nm = -1
            pc1_var_nm = -1
        
        res = {
            "layer": l,
            "pc1_variance": float(explained[0]),
            "effective_rank": int(eff_rank),
            "pc1_norm_ratio_corr": float(pc1_norm_corr),
            "pc1_norm_diff_corr": float(pc1_normdiff_corr),
            "acc_proper_centroid5": float(acc_proper),
            "acc_norm_matched_centroid5": float(acc_nm),
            "acc_no_norm_ratio_centroid5": float(acc_no_nr),
            "norm_matched": {
                "pc1_variance": float(pc1_var_nm),
                "effective_rank": int(eff_rank_nm),
                "acc_drop": float(acc_proper - acc_nm),
            },
            "norm_ratio_stats": {
                "mean": float(np.mean(norm_ratio)),
                "std": float(np.std(norm_ratio)),
            },
        }
        
        results[l_str] = res
        
        log(f"  Layer {l}: PC1_var={explained[0]:.3f}, eff_rank={eff_rank}, "
            f"PC1~nr_corr={pc1_norm_corr:.3f}, "
            f"acc_proper={acc_proper:.3f}, acc_nm={acc_nm:.3f}, acc_no_nr={acc_no_nr:.3f}, "
            f"drop_nm={acc_proper-acc_nm:+.3f}")
    
    return results


# ===== Main =====
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    assert model_name in ("qwen3", "deepseek7b", "glm4")
    
    log(f"Phase 381: Norm-Matched Category Test — {model_name}")
    log(f"=" * 60)
    
    # Target layers
    if model_name == "deepseek7b":
        target_layers = [4, 8, 12, 16, 20, 24]
    elif model_name == "qwen3":
        target_layers = [4, 12, 20, 28]
    elif model_name == "glm4":
        target_layers = [4, 12, 20, 30]
    
    # Load model
    t0 = time.time()
    model, tokenizer = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    log(f"  Model loaded in {time.time()-t0:.1f}s: {info.model_class}, {info.n_layers} layers, d={info.d_model}")
    
    # Collect data
    log("\nCollecting residual states...")
    all_data = collect_residual_states(model, tokenizer, model_name, target_layers)
    
    # Part 1: Norm-matched classification
    log("\nRunning Part 1: Norm-Matched Category Classification...")
    part1_results = part1_norm_matched_classification(all_data, model_name)
    
    # Part 2: Norm vs Direction causal
    log("\nRunning Part 2: Norm vs Direction Causal Separation...")
    part2_results = part2_norm_vs_direction_causal(all_data, model, tokenizer, model_name)
    
    # Part 3: Deep layer tracking
    log("\nRunning Part 3: Deep Layer Tracking...")
    part3_results = part3_deep_layer_tracking(all_data, model_name)
    
    # Save results
    out_dir = f"results/phase381_norm_matched_category"
    os.makedirs(out_dir, exist_ok=True)
    
    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    full_results = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "n_pairs": len(ALL_PAIRS),
        "n_categories": N_CATEGORIES,
        "target_layers": target_layers,
        "part1_norm_matched_classification": {k: convert(v) for k, v in part1_results.items()},
        "part2_norm_vs_direction_causal": {k: convert(v) for k, v in part2_results.items()},
        "part3_deep_layer_tracking": {k: convert(v) for k, v in part3_results.items()},
    }
    
    out_file = os.path.join(out_dir, f"{model_name}_phase381.json")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False, default=convert)
    
    log(f"\nResults saved to {out_file}")
    
    # Summary
    log("\n" + "="*60)
    log("SUMMARY")
    log("="*60)
    
    log(f"\nModel: {model_name} ({info.model_class}, {info.n_layers} layers)")
    log(f"Pairs: {len(ALL_PAIRS)}, Categories: {N_CATEGORIES} ({', '.join(ALL_CATEGORIES)})")
    
    log("\nPart 1: Norm-Matched Classification")
    log("-" * 50)
    for l_str in sorted(part1_results.keys(), key=int):
        r = part1_results[l_str]
        cl = r["classification"]
        log(f"  L{r['layer']}: PROPER={cl['acc_proper_centroid5']:.3f}, "
            f"NormMatched={cl['acc_norm_matched_centroid5']:.3f}, "
            f"NormRegressed={cl['acc_no_norm_ratio_centroid5']:.3f}, "
            f"PC1~nr={r['pc1_norm_ratio_corr']:.3f}, "
            f"drop(NM)={r['norm_matched_vs_original_drop']:+.3f}, "
            f"drop(NR)={r['norm_regressed_vs_original_drop']:+.3f}")
    
    if part2_results:
        log("\nPart 2: Norm vs Direction Causal")
        log("-" * 50)
        for l_str in sorted(part2_results.keys(), key=int):
            r = part2_results[l_str]
            md = r["mean_delta"]
            gf = r["gap_fraction"]
            log(f"  L{r['layer']}: Δ(norm)={md['pure_norm_vs_corrupt']:.3f}, "
                f"Δ(dir)={md['pure_dir_vs_corrupt']:.3f}, "
                f"Δ(clean)={md['clean_vs_corrupt']:.3f}, "
                f"norm_frac={gf['norm']:.3f}, dir_frac={gf['direction']:.3f}")
    
    log("\nPart 3: Deep Layer Tracking")
    log("-" * 50)
    for l_str in sorted(part3_results.keys(), key=int):
        r = part3_results[l_str]
        log(f"  L{r['layer']}: PC1_var={r['pc1_variance']:.3f}, "
            f"eff_rank={r['effective_rank']}, "
            f"PC1~nr={r['pc1_norm_ratio_corr']:.3f}, "
            f"acc={r['acc_proper_centroid5']:.3f}, "
            f"acc_nm={r['acc_norm_matched_centroid5']:.3f}, "
            f"acc_no_nr={r['acc_no_norm_ratio_centroid5']:.3f}")
    
    # Release model
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    
    log(f"\nPhase 381 complete for {model_name}!")


if __name__ == "__main__":
    main()
