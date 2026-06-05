"""
Phase 380b: 反向因果patch确认
==============================

对Phase 380关键发现的确认测试：
1. 从clean中移除类别分量（反向patch）
2. 交换两个样本的类别分量
3. 增加数据量（221对 vs 132对）

核心验证：
- GLM4: 移除类别后logit是否朝corrupt移动？
- DS7B: 移除类别后logit是否有任何变化？
- Qwen3: 移除类别后logit是否变化？

用法:
  python tests/glm5/phase380b_reverse_causal_patch.py qwen3
  python tests/glm5/phase380b_reverse_causal_patch.py deepseek7b
  python tests/glm5/phase380b_reverse_causal_patch.py glm4
"""

import sys, os, time, json, gc, traceback
import torch
import numpy as np
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, 'tests/glm5')

from model_utils import get_layers, get_model_info, release_model, get_W_U, MODEL_CONFIGS


def log(msg="", end="\n"):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", end=end, flush=True)


# ===== Same binding pairs as Phase 379b (221 pairs) =====
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
    ("dust_mote", "small", "big"), ("skyscraper", "big", "small"), ("pebble", "small", "big"),
    ("cathedral", "big", "small"), ("needle", "small", "big"), ("moon_size", "big", "small"),
    ("molecule", "small", "big"), ("mountain_range", "big", "small"), ("drop", "small", "big"),
]
WEIGHT_PAIRS = [
    ("boulder", "heavy", "light"), ("feather", "light", "heavy"), ("lead", "heavy", "light"),
    ("balloon", "light", "heavy"), ("steel", "heavy", "light"), ("cotton", "light", "heavy"),
    ("anchor", "heavy", "light"), ("bubble", "light", "heavy"), ("concrete", "heavy", "light"),
    ("air", "light", "heavy"), ("truck", "heavy", "light"), ("petal", "light", "heavy"),
    ("elephant", "heavy", "light"), ("cloud", "light", "heavy"),
    ("iron", "heavy", "light"), ("silk", "light", "heavy"), ("gold_bar", "heavy", "light"),
    ("tissue", "light", "heavy"), ("anvil", "heavy", "light"), ("leaf", "light", "heavy"),
]
SPEED_PAIRS = [
    ("cheetah", "fast", "slow"), ("turtle", "slow", "fast"), ("rocket", "fast", "slow"),
    ("snail", "slow", "fast"), ("lightning", "fast", "slow"), ("sloth", "slow", "fast"),
    ("falcon", "fast", "slow"), ("worm", "slow", "fast"), ("bullet", "fast", "slow"),
    ("glacier_motion", "slow", "fast"), ("jet", "fast", "slow"),
    ("racecar", "fast", "slow"), ("caterpillar", "slow", "fast"),
    ("deer", "fast", "slow"), ("slug", "slow", "fast"), ("missile", "fast", "slow"),
    ("ox", "slow", "fast"), ("cheetah_run", "fast", "slow"), ("tortoise", "slow", "fast"),
]
BRIGHT_PAIRS = [
    ("star", "bright", "dark"), ("cave", "dark", "bright"), ("sun", "bright", "dark"),
    ("shadow", "dark", "bright"), ("lamp", "bright", "dark"), ("night", "dark", "bright"),
    ("flashlight", "bright", "dark"), ("abyss", "dark", "bright"), ("diamond", "bright", "dark"),
    ("tunnel", "dark", "bright"), ("beacon", "bright", "dark"), ("eclipse", "dark", "bright"),
    ("lighthouse", "bright", "dark"), ("dungeon", "dark", "bright"),
    ("neon", "bright", "dark"), ("void", "dark", "bright"), ("candle_light", "bright", "dark"),
    ("black_hole", "dark", "bright"), ("phosphor", "bright", "dark"), ("crypt", "dark", "bright"),
    ("spotlight", "bright", "dark"), ("fog_dim", "dark", "bright"),
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


def _silu(x):
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -50, 50))))

def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

def rms_norm_single(x, weight=None, eps=1e-6):
    d = x.shape[-1]
    rms = np.sqrt(np.mean(x**2) + eps)
    result = x / rms * np.sqrt(d)
    if weight is not None:
        result = result * weight
    return result


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
    return None


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
        
        h_post_clean_list = []
        h_post_corrupt_list = []
        
        for pidx, (obj, target, competitor) in enumerate(ALL_PAIRS):
            if pidx % 40 == 0:
                log(f"    Pair {pidx+1}/{n_pairs}")
            
            clean_prompt = TEMPLATE.format(obj=obj, attr=target)
            corrupt_prompt = TEMPLATE.format(obj=CORRUPTED_BASELINE, attr=target)
            
            with torch.no_grad():
                clean_out = model(
                    input_ids=tokenizer(clean_prompt, return_tensors="pt",
                                       truncation=True, max_length=64)["input_ids"].to(input_device),
                    attention_mask=tokenizer(clean_prompt, return_tensors="pt",
                                            truncation=True, max_length=64)["attention_mask"].to(input_device),
                    output_hidden_states=True)
            
            last_pos = tokenizer(clean_prompt, return_tensors="pt")["input_ids"].shape[1] - 1
            h_post_clean_list.append(clean_out.hidden_states[l+1][0, last_pos].detach().cpu().float().numpy())
            del clean_out
            
            with torch.no_grad():
                corrupt_out = model(
                    input_ids=tokenizer(corrupt_prompt, return_tensors="pt",
                                       truncation=True, max_length=64)["input_ids"].to(input_device),
                    attention_mask=tokenizer(corrupt_prompt, return_tensors="pt",
                                            truncation=True, max_length=64)["attention_mask"].to(input_device),
                    output_hidden_states=True)
            
            last_pos_r = tokenizer(corrupt_prompt, return_tensors="pt")["input_ids"].shape[1] - 1
            h_post_corrupt_list.append(corrupt_out.hidden_states[l+1][0, last_pos_r].detach().cpu().float().numpy())
            del corrupt_out
            
            if pidx % 5 == 0:
                torch.cuda.empty_cache()
        
        h_post_clean = np.array(h_post_clean_list)
        h_post_corrupt = np.array(h_post_corrupt_list)
        dh = h_post_clean - h_post_corrupt
        
        all_data[str(l)] = {
            "h_post_clean": h_post_clean,
            "h_post_corrupt": h_post_corrupt,
            "dh": dh,
            "ln_weight": ln_weight,
        }
        
        log(f"    Layer {l} done in {time.time()-t_l:.1f}s")
    
    return all_data


def reverse_causal_patch(all_data, model, tokenizer, model_name):
    """
    Reverse causal patch:
    1. Remove category component from clean → see if logit moves toward corrupt
    2. Remove non-category component from clean → see if logit moves toward corrupt
    3. Compare magnitudes
    """
    log("\n" + "="*60)
    log("Reverse Causal Patch: Remove from Clean")
    log("="*60)
    
    layers = get_layers(model)
    n_layers = len(layers)
    input_device = next(model.parameters()).device
    
    results = {}
    
    for l_str in sorted(all_data.keys(), key=int):
        d = all_data[l_str]
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
        
        dh_proper = h_clean_norm - h_corrupt_norm
        
        # LDA subspace
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
            eigvecs = eigvecs[:, sort_idx]
            n_lda = min(N_CATEGORIES - 1, d_model)
            lda_dirs = eigvecs[:, :n_lda].T
        except:
            pca, _, Vt, _, _ = compute_pca_full(dh_proper)
            if pca is None:
                continue
            lda_dirs = Vt[:6]
            n_lda = 6
        
        # Decompose dh_proper
        lda_proj = lda_dirs.T @ lda_dirs
        dh_cat = (lda_proj @ dh_proper.T).T
        dh_noncat = dh_proper - dh_cat
        
        log(f"\n  Layer {l}:")
        
        # Select subset for causal patch
        patch_pairs = []
        for cat in ALL_CATEGORIES:
            idx = cat_indices[cat]
            patch_pairs.extend(idx[:4])
        
        # For each pair: 
        # 1. Run clean → baseline clean logit
        # 2. Run clean with -dh_cat removed → see logit change
        # 3. Run clean with -dh_noncat removed → see logit change
        
        causal_results = {
            "clean_logit_diff": [],
            "remove_cat_logit_diff": [],
            "remove_noncat_logit_diff": [],
            "remove_all_logit_diff": [],  # should ≈ corrupt
        }
        
        next_l = l + 1
        if next_l >= n_layers:
            log(f"    Skipping (layer {l} is last)")
            continue
        
        for pidx in patch_pairs[:28]:
            obj, target, competitor = ALL_PAIRS[pidx]
            t_tokens = tokenizer(target, add_special_tokens=False)["input_ids"]
            c_tokens = tokenizer(competitor, add_special_tokens=False)["input_ids"]
            if len(t_tokens) != 1 or len(c_tokens) != 1:
                continue
            t_id, c_id = t_tokens[0], c_tokens[0]
            
            clean_prompt = TEMPLATE.format(obj=obj, attr=target)
            
            # Baseline clean run
            with torch.no_grad():
                clean_ids = tokenizer(clean_prompt, return_tensors="pt",
                                     truncation=True, max_length=64)["input_ids"].to(input_device)
                clean_mask = tokenizer(clean_prompt, return_tensors="pt",
                                      truncation=True, max_length=64)["attention_mask"].to(input_device)
                clean_out = model(input_ids=clean_ids, attention_mask=clean_mask)
            
            clean_logit = clean_out.logits[0, -1].float().cpu().numpy()
            clean_diff = float(clean_logit[t_id] - clean_logit[c_id])
            del clean_out
            
            # Remove category component: h_clean_raw - dh_cat_raw
            # We need to convert dh_cat from norm space back to raw space
            # But since we're patching at the raw residual level (before next LN),
            # we should use the raw Δh decomposition, not the norm-space one.
            # However, LDA was done in norm space. Let's project the LDA directions 
            # back to raw space.
            
            # Actually, simpler: just subtract dh_cat from h_post_clean
            # This is approximately correct because dh_cat is the category component of Δh
            # and we want h_clean - dh_cat ≈ h_clean - (h_clean - h_corrupt)_cat
            # = h_corrupt + dh_noncat
            
            # Remove cat: subtract category component of Δh from clean
            patch_remove_cat = torch.tensor(-dh_cat[pidx], dtype=torch.bfloat16).to(input_device)
            patch_remove_noncat = torch.tensor(-dh_noncat[pidx], dtype=torch.bfloat16).to(input_device)
            patch_remove_all = torch.tensor(-dh_proper[pidx], dtype=torch.bfloat16).to(input_device)
            
            def make_patch_hook(patch_vec):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                        last_pos = h.shape[1] - 1
                        h_patched = h.clone()
                        h_patched[0, last_pos] = h_patched[0, last_pos] + patch_vec
                        return (h_patched,) + output[1:]
                    return output
                return hook
            
            # Remove cat from clean
            hook1 = layers[l].register_forward_hook(make_patch_hook(patch_remove_cat))
            with torch.no_grad():
                out1 = model(input_ids=clean_ids, attention_mask=clean_mask)
            hook1.remove()
            logit1 = out1.logits[0, -1].float().cpu().numpy()
            remove_cat_diff = float(logit1[t_id] - logit1[c_id])
            del out1
            
            # Remove noncat from clean
            hook2 = layers[l].register_forward_hook(make_patch_hook(patch_remove_noncat))
            with torch.no_grad():
                out2 = model(input_ids=clean_ids, attention_mask=clean_mask)
            hook2.remove()
            logit2 = out2.logits[0, -1].float().cpu().numpy()
            remove_noncat_diff = float(logit2[t_id] - logit2[c_id])
            del out2
            
            # Remove all from clean (should give ≈ corrupt)
            hook3 = layers[l].register_forward_hook(make_patch_hook(patch_remove_all))
            with torch.no_grad():
                out3 = model(input_ids=clean_ids, attention_mask=clean_mask)
            hook3.remove()
            logit3 = out3.logits[0, -1].float().cpu().numpy()
            remove_all_diff = float(logit3[t_id] - logit3[c_id])
            del out3
            
            torch.cuda.empty_cache()
            
            causal_results["clean_logit_diff"].append(clean_diff)
            causal_results["remove_cat_logit_diff"].append(remove_cat_diff)
            causal_results["remove_noncat_logit_diff"].append(remove_noncat_diff)
            causal_results["remove_all_logit_diff"].append(remove_all_diff)
        
        if len(causal_results["clean_logit_diff"]) > 3:
            cl = np.array(causal_results["clean_logit_diff"])
            rc = np.array(causal_results["remove_cat_logit_diff"])
            rnc = np.array(causal_results["remove_noncat_logit_diff"])
            ra = np.array(causal_results["remove_all_logit_diff"])
            
            # Corrupt baseline
            corrupt_logit_diffs = []
            for pidx in patch_pairs[:28]:
                obj, target, competitor = ALL_PAIRS[pidx]
                t_tokens = tokenizer(target, add_special_tokens=False)["input_ids"]
                c_tokens = tokenizer(competitor, add_special_tokens=False)["input_ids"]
                if len(t_tokens) != 1 or len(c_tokens) != 1:
                    continue
                t_id, c_id = t_tokens[0], c_tokens[0]
                corrupt_prompt = TEMPLATE.format(obj=CORRUPTED_BASELINE, attr=target)
                with torch.no_grad():
                    corrupt_ids = tokenizer(corrupt_prompt, return_tensors="pt",
                                          truncation=True, max_length=64)["input_ids"].to(input_device)
                    corrupt_mask = tokenizer(corrupt_prompt, return_tensors="pt",
                                           truncation=True, max_length=64)["attention_mask"].to(input_device)
                    corrupt_out = model(input_ids=corrupt_ids, attention_mask=corrupt_mask)
                corrupt_logit = corrupt_out.logits[0, -1].float().cpu().numpy()
                corrupt_logit_diffs.append(float(corrupt_logit[t_id] - corrupt_logit[c_id]))
                del corrupt_out
                torch.cuda.empty_cache()
            
            corr = np.array(corrupt_logit_diffs) if corrupt_logit_diffs else None
            
            mean_clean = float(np.mean(cl))
            mean_remove_cat = float(np.mean(rc))
            mean_remove_noncat = float(np.mean(rnc))
            mean_remove_all = float(np.mean(ra))
            mean_corrupt = float(np.mean(corr)) if corr is not None else None
            
            # Key metric: how much does removing cat move clean toward corrupt?
            delta_remove_cat = mean_remove_cat - mean_clean
            delta_remove_noncat = mean_remove_noncat - mean_clean
            delta_remove_all = mean_remove_all - mean_clean
            delta_corrupt = mean_corrupt - mean_clean if mean_corrupt is not None else None
            
            # Fraction of clean→corrupt gap explained by cat removal
            if delta_corrupt and abs(delta_corrupt) > 1e-10:
                frac_cat = delta_remove_cat / delta_corrupt
                frac_noncat = delta_remove_noncat / delta_corrupt
                frac_all = delta_remove_all / delta_corrupt
            else:
                frac_cat = frac_noncat = frac_all = None
            
            layer_result = {
                "mean_clean": round(mean_clean, 4),
                "mean_corrupt": round(mean_corrupt, 4) if mean_corrupt else None,
                "mean_remove_cat": round(mean_remove_cat, 4),
                "mean_remove_noncat": round(mean_remove_noncat, 4),
                "mean_remove_all": round(mean_remove_all, 4),
                "delta_remove_cat": round(delta_remove_cat, 4),
                "delta_remove_noncat": round(delta_remove_noncat, 4),
                "delta_remove_all": round(delta_remove_all, 4),
                "delta_corrupt": round(delta_corrupt, 4) if delta_corrupt else None,
                "frac_cat": round(frac_cat, 4) if frac_cat else None,
                "frac_noncat": round(frac_noncat, 4) if frac_noncat else None,
                "frac_all": round(frac_all, 4) if frac_all else None,
                "n_patched": len(cl),
            }
            
            log(f"    Clean: {mean_clean:.4f}")
            log(f"    Corrupt: {mean_corrupt:.4f}" if mean_corrupt else "    Corrupt: N/A")
            log(f"    -Cat: {mean_remove_cat:.4f} (Δ={delta_remove_cat:.4f})")
            log(f"    -Noncat: {mean_remove_noncat:.4f} (Δ={delta_remove_noncat:.4f})")
            log(f"    -All: {mean_remove_all:.4f} (Δ={delta_remove_all:.4f})")
            if frac_cat is not None:
                log(f"    Fraction: cat={frac_cat:.4f}, noncat={frac_noncat:.4f}, all={frac_all:.4f}")
            
            results[l_str] = layer_result
    
    return results


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


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    
    log(f"Phase 380b: Reverse Causal Patch Confirmation — {model_name}")
    log(f"Pairs: {len(ALL_PAIRS)}, Categories: {N_CATEGORIES}")
    
    if model_name == "deepseek7b":
        target_layers = [4, 24]
    elif model_name == "qwen3":
        target_layers = [4, 28]
    elif model_name == "glm4":
        target_layers = [4, 30]
    else:
        target_layers = [4]
    
    t0 = time.time()
    model, tokenizer = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    log(f"  Model loaded: {info.model_class}, {info.n_layers} layers, d={info.d_model}")
    
    t0 = time.time()
    all_data = collect_residual_states(model, tokenizer, model_name, target_layers)
    log(f"  Data collection: {time.time()-t0:.1f}s")
    
    results = reverse_causal_patch(all_data, model, tokenizer, model_name)
    
    output_dir = f"results/phase380_category_subspace_causal_patch"
    os.makedirs(output_dir, exist_ok=True)
    
    output = {
        "model": model_name,
        "n_pairs": len(ALL_PAIRS),
        "test_type": "reverse_causal_patch",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
    }
    
    output_path = os.path.join(output_dir, f"{model_name}_phase380b.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    log(f"Results saved to {output_path}")
    
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    
    log(f"Phase 380b complete for {model_name}!")


if __name__ == "__main__":
    main()
