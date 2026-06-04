"""
Phase 375b: Confirmation Test — Top Δ(gate*up) Channel Deep Analysis
=====================================================================

Phase 376 found that masking top-10 Δ(gate*up) channels destroys 1D collapse
in DS7B L4 (PC1: 0.991→0.112). This confirmation test:

1. Uses MORE binding pairs (expanded set) to verify the top-10 finding
2. Identifies WHICH channels are the top-10 and what they encode
3. Tests if the top-10 channels are stable across different data subsets
4. Cross-references with W_down v1 direction (are these the v1-heavy channels?)
5. Tests if masking top-10 in OTHER layers also destroys 1D (specificity test)

Models: deepseek7b (primary), qwen3 (control)
"""

import sys, os, time, json, gc
import torch
import numpy as np
from datetime import datetime
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, 'tests/glm5')

def log(msg="", end="\n"):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", end=end, flush=True)


MODEL_CONFIGS = {
    "qwen3": {
        "path": "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
        "n_layers": 36, "d_model": 2560,
    },
    "glm4": {
        "path": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
        "n_layers": 40, "d_model": 4096,
    },
    "deepseek7b": {
        "path": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "n_layers": 28, "d_model": 3584,
    },
}

# ===== Expanded binding pairs (more objects per category) =====
COLOR_PAIRS = [
    ("apple", "red", "blue"), ("banana", "yellow", "purple"), ("snow", "white", "black"),
    ("sky", "blue", "green"), ("cherry", "red", "blue"), ("leaf", "green", "red"),
    ("rose", "red", "blue"), ("gold", "yellow", "purple"), ("coal", "black", "white"),
    ("silver", "white", "black"), ("milk", "white", "black"), ("honey", "yellow", "blue"),
    ("ruby", "red", "green"), ("emerald", "green", "red"), ("sapphire", "blue", "red"),
    ("moon", "white", "black"), ("flame", "orange", "blue"), ("forest", "green", "white"),
    ("ocean", "blue", "yellow"), ("sun", "yellow", "purple"),
    # Extra color pairs
    ("tomato", "red", "blue"), ("lemon", "yellow", "purple"), ("ivory", "white", "black"),
    ("grass", "green", "red"), ("blood", "red", "blue"), ("cloud", "white", "black"),
    ("amber", "yellow", "green"), ("moss", "green", "blue"), ("raven", "black", "white"),
    ("coral", "orange", "blue"), ("teal", "blue", "red"), ("plum", "purple", "yellow"),
    ("bronze", "brown", "blue"), ("crimson", "red", "green"), ("azure", "blue", "red"),
    ("ivory", "white", "black"), ("scarlet", "red", "blue"), ("turquoise", "blue", "green"),
    ("maroon", "brown", "white"), ("lime", "green", "purple"), ("peach", "orange", "blue"),
]
TEMP_PAIRS = [
    ("fire", "hot", "cold"), ("desert", "hot", "cold"), ("lava", "hot", "cold"),
    ("ice", "cold", "hot"), ("snow", "cold", "hot"), ("volcano", "hot", "cold"),
    ("furnace", "hot", "cold"), ("glacier", "cold", "hot"),
    # Extra temperature pairs
    ("oven", "hot", "cold"), ("frost", "cold", "hot"), ("magma", "hot", "cold"),
    ("winter", "cold", "hot"), ("summer", "hot", "cold"), ("arctic", "cold", "hot"),
    ("stove", "hot", "cold"), ("blizzard", "cold", "hot"), ("tundra", "cold", "hot"),
    ("inferno", "hot", "cold"), (" Equator", "hot", "cold"), ("iceberg", "cold", "hot"),
]
MOISTURE_PAIRS = [
    ("rain", "wet", "dry"), ("ocean", "wet", "dry"), ("river", "wet", "dry"),
    ("sand", "dry", "wet"), ("dust", "dry", "wet"), ("bone", "dry", "wet"),
    ("swamp", "wet", "dry"), ("desert", "dry", "wet"),
    # Extra moisture pairs
    ("lake", "wet", "dry"), ("sponge", "wet", "dry"), ("cracker", "dry", "wet"),
    ("fog", "wet", "dry"), ("prairie", "dry", "wet"), ("puddle", "wet", "dry"),
    ("cactus", "dry", "wet"), ("waterfall", "wet", "dry"),
]
SIZE_PAIRS = [
    ("elephant", "big", "small"), ("mountain", "big", "small"), ("ant", "small", "big"),
    ("planet", "big", "small"), ("grain", "small", "big"), ("whale", "big", "small"),
    # Extra size pairs
    ("galaxy", "big", "small"), ("atom", "small", "big"), ("continent", "big", "small"),
    ("bacteria", "small", "big"), ("tower", "big", "small"), ("speck", "small", "big"),
    ("universe", "big", "small"), ("pixel", "small", "big"), ("castle", "big", "small"),
    ("dust", "small", "big"),
]
WEIGHT_PAIRS = [
    ("boulder", "heavy", "light"), ("feather", "light", "heavy"), ("lead", "heavy", "light"),
    ("balloon", "light", "heavy"), ("steel", "heavy", "light"), ("cotton", "light", "heavy"),
    # Extra weight pairs
    ("anchor", "heavy", "light"), ("bubble", "light", "heavy"), ("concrete", "heavy", "light"),
    ("air", "light", "heavy"), ("truck", "heavy", "light"), ("petal", "light", "heavy"),
    ("elephant", "heavy", "light"), ("cloud", "light", "heavy"),
]
SPEED_PAIRS = [
    ("cheetah", "fast", "slow"), ("turtle", "slow", "fast"), ("rocket", "fast", "slow"),
    ("snail", "slow", "fast"), ("lightning", "fast", "slow"), ("sloth", "slow", "fast"),
    # Extra speed pairs
    ("falcon", "fast", "slow"), ("worm", "slow", "fast"), ("bullet", "fast", "slow"),
    ("glacier", "slow", "fast"), ("jet", "fast", "slow"), ("sloth", "slow", "fast"),
    ("racecar", "fast", "slow"), ("caterpillar", "slow", "fast"),
]
BRIGHT_PAIRS = [
    ("star", "bright", "dark"), ("cave", "dark", "bright"), ("sun", "bright", "dark"),
    ("shadow", "dark", "bright"), ("lamp", "bright", "dark"), ("night", "dark", "bright"),
    # Extra brightness pairs
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

# Remove duplicates (some objects appear in multiple categories)
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

CORRUPTED_BASELINE = "The item"
TEMPLATE = "The {obj} is {attr}."


def _silu(x):
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -50, 50))))

def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

def compute_pca_explained(X):
    M = X - X.mean(axis=0, keepdims=True)
    try:
        _, S, Vt = np.linalg.svd(M, full_matrices=False)
    except:
        return None, None, None, None
    total_var = np.sum(S**2)
    if total_var < 1e-10:
        return None, None, None, None
    explained = (S**2) / total_var
    eff_rank = int(np.searchsorted(np.cumsum(explained), 0.95) + 1)
    return explained, eff_rank, Vt[0], Vt

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


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
        except:
            continue
    if model is None:
        raise RuntimeError(f"Failed to load {model_name}")
    model.eval()
    return model, tokenizer, next(model.parameters()).device


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("Cannot find layers")


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
        except:
            pass
    if W_gate is None:
        model_path = MODEL_CONFIGS[model_name]["path"]
        for sf_file in glob.glob(os.path.join(model_path, '*.safetensors')):
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
    return W_gate, W_up, W_down


def run_confirmation(model_name):
    cfg = MODEL_CONFIGS[model_name]
    act_fn = "gelu" if model_name == "glm4" else "silu"
    
    log(f"\n{'='*60}")
    log(f"Phase 375b Confirmation: {model_name}")
    log(f"  n_pairs={len(ALL_PAIRS)} (expanded set)")
    log(f"{'='*60}")
    
    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    log(f"  Model loaded in {time.time()-t0:.1f}s")
    
    # Focus on key layers
    if model_name == "deepseek7b":
        target_layers = [4, 5, 8, 24]
    elif model_name == "qwen3":
        target_layers = [4, 28]
    else:
        target_layers = [4, 30]
    
    layers = get_layers(model)
    input_device = next(model.parameters()).device
    n_pairs = len(ALL_PAIRS)
    
    results = {}
    
    for l in target_layers:
        log(f"\n  Layer {l}:")
        t_layer = time.time()
        
        W_gate, W_up, W_down = load_mlp_weights(model, model_name, l)
        if W_gate is None:
            continue
        
        d_ff = W_gate.shape[0]
        mlp_module = layers[l].mlp
        
        # Collect gate*up activations
        mlp_in_clean_list = []
        mlp_in_corrupt_list = []
        h_clean_list = []
        h_corrupt_list = []
        
        for pidx, (obj, target, competitor) in enumerate(ALL_PAIRS):
            if pidx % 30 == 0:
                log(f"    Pair {pidx+1}/{n_pairs}")
            
            clean_prompt = TEMPLATE.format(obj=obj, attr=target)
            corrupt_prompt = TEMPLATE.format(obj=CORRUPTED_BASELINE, attr=target)
            
            clean_inputs = tokenizer(clean_prompt, return_tensors="pt", truncation=True, max_length=64)
            corrupt_inputs = tokenizer(corrupt_prompt, return_tensors="pt", truncation=True, max_length=64)
            
            captured = {}
            def mlp_input_hook(module, input, output=None):
                captured["mlp_input"] = input[0].detach().cpu().float()
            
            h_mlp_in = mlp_module.register_forward_pre_hook(mlp_input_hook)
            
            with torch.no_grad():
                clean_out = model(
                    input_ids=clean_inputs["input_ids"].to(input_device),
                    attention_mask=clean_inputs["attention_mask"].to(input_device),
                    output_hidden_states=True)
            last_pos_c = clean_inputs["input_ids"].shape[1] - 1
            mlp_in_clean = captured["mlp_input"][0, last_pos_c].numpy()
            h_c = clean_out.hidden_states[l+1][0, last_pos_c].detach().cpu().float().numpy()
            mlp_in_clean_list.append(mlp_in_clean)
            h_clean_list.append(h_c)
            
            captured.clear()
            with torch.no_grad():
                corrupt_out = model(
                    input_ids=corrupt_inputs["input_ids"].to(input_device),
                    attention_mask=corrupt_inputs["attention_mask"].to(input_device),
                    output_hidden_states=True)
            last_pos_r = corrupt_inputs["input_ids"].shape[1] - 1
            mlp_in_corrupt = captured["mlp_input"][0, last_pos_r].numpy()
            h_r = corrupt_out.hidden_states[l+1][0, last_pos_r].detach().cpu().float().numpy()
            mlp_in_corrupt_list.append(mlp_in_corrupt)
            h_corrupt_list.append(h_r)
            
            h_mlp_in.remove()
            del clean_out, corrupt_out
            if pidx % 5 == 0:
                torch.cuda.empty_cache()
        
        # Compute intermediate activations
        mlp_in_clean = np.array(mlp_in_clean_list)
        mlp_in_corrupt = np.array(mlp_in_corrupt_list)
        h_clean = np.array(h_clean_list)
        h_corrupt = np.array(h_corrupt_list)
        
        if act_fn == "silu":
            gate_act_clean = _silu(mlp_in_clean @ W_gate.T)
            gate_act_corrupt = _silu(mlp_in_corrupt @ W_gate.T)
        else:
            gate_act_clean = _gelu(mlp_in_clean @ W_gate.T)
            gate_act_corrupt = _gelu(mlp_in_corrupt @ W_gate.T)
        
        up_clean = mlp_in_clean @ W_up.T
        up_corrupt = mlp_in_corrupt @ W_up.T
        gate_up_clean = gate_act_clean * up_clean
        gate_up_corrupt = gate_act_corrupt * up_corrupt
        
        d_gate_up = gate_up_clean - gate_up_corrupt
        d_gate_act = gate_act_clean - gate_act_corrupt
        dh = h_clean - h_corrupt
        
        # ===== Analysis 1: Confirm top-10 finding with expanded data =====
        log(f"\n    === Analysis 1: Top-10 channel confirmation ===")
        
        pca_dgu, rank_dgu, pc1_dgu, Vt_dgu = compute_pca_explained(d_gate_up)
        pca_dh, rank_dh, _, _ = compute_pca_explained(dh)
        
        log(f"    Baseline: Δ(gate*up) PC1={pca_dgu[0]:.4f}, Δh PC1={pca_dh[0]:.4f}")
        
        dgu_energy = np.mean(d_gate_up**2, axis=0)
        dgu_top_indices = np.argsort(dgu_energy)[::-1]
        
        # Mask top-10 test
        for K in [5, 10, 20]:
            d_gate_up_masked = d_gate_up.copy()
            d_gate_up_masked[:, dgu_top_indices[:K]] = 0
            pca_masked, rank_masked, _, _ = compute_pca_explained(d_gate_up_masked)
            
            mlp_out_masked = (W_down @ d_gate_up_masked.T).T
            dh_new = dh - (W_down @ (d_gate_up - d_gate_up_masked).T).T
            pca_dh_new, rank_dh_new, _, _ = compute_pca_explained(dh_new)
            
            log(f"    Mask top-{K}: Δ(gate*up) PC1={pca_masked[0]:.4f}, Δh PC1={pca_dh_new[0]:.4f}")
        
        # ===== Analysis 2: What are the top-10 channels? =====
        log(f"\n    === Analysis 2: Top-10 channel identity ===")
        
        top10_ch = dgu_top_indices[:10]
        top20_ch = dgu_top_indices[:20]
        
        log(f"    Top-10 channel indices: {top10_ch.tolist()}")
        log(f"    Top-10 channel energies: {[f'{dgu_energy[c]:.6f}' for c in top10_ch]}")
        log(f"    Top-10 cumulative energy: {np.sum(dgu_energy[top10_ch])/np.sum(dgu_energy):.4f}")
        
        # Are these the W_down v1-heavy channels?
        U_d, S_d, Vt_d = np.linalg.svd(W_down, full_matrices=False)
        v1 = Vt_d[0]  # (d_ff,)
        
        # |v1[ch]| for top channels
        v1_abs = np.abs(v1)
        v1_ranked = np.argsort(v1_abs)[::-1]
        
        overlap_top10_v1_top10 = len(set(top10_ch) & set(v1_ranked[:10]))
        overlap_top10_v1_top50 = len(set(top10_ch) & set(v1_ranked[:50]))
        overlap_top10_v1_top100 = len(set(top10_ch) & set(v1_ranked[:100]))
        
        log(f"    Overlap top-10 Δ(gate*up) with v1 top-10: {overlap_top10_v1_top10}/10")
        log(f"    Overlap top-10 Δ(gate*up) with v1 top-50: {overlap_top10_v1_top50}/10")
        log(f"    Overlap top-10 Δ(gate*up) with v1 top-100: {overlap_top10_v1_top100}/10")
        
        # Correlation between dgu_energy and |v1|
        corr_dgu_v1 = np.corrcoef(dgu_energy, v1_abs)[0, 1]
        log(f"    Correlation dgu_energy vs |v1|: {corr_dgu_v1:.4f}")
        
        # ===== Analysis 3: Category specificity of top channels =====
        log(f"\n    === Analysis 3: Top channel category response ===")
        
        for i, ch in enumerate(top10_ch[:5]):
            ch_vals = d_gate_up[:, ch]
            cat_means = {}
            for cat in ALL_CATEGORIES:
                cat_idx = [j for j, c in enumerate(PAIR_CATEGORIES) if c == cat]
                cat_means[cat] = float(np.mean(ch_vals[cat_idx]))
            
            grand_mean = np.mean(ch_vals)
            best_cat = max(cat_means, key=lambda k: abs(cat_means[k] - grand_mean))
            
            log(f"    Ch {ch}: best_cat={best_cat}, "
                f"cat_means={', '.join(f'{k}={v:.2f}' for k, v in sorted(cat_means.items(), key=lambda x: -abs(x[1]-grand_mean))[:3])}")
        
        # ===== Analysis 4: Stability test — split-half =====
        log(f"\n    === Analysis 4: Split-half stability ===")
        
        rng = np.random.RandomState(42)
        perm = rng.permutation(n_pairs)
        half1_idx = perm[:n_pairs//2]
        half2_idx = perm[n_pairs//2:]
        
        dgu_half1 = d_gate_up[half1_idx]
        dgu_half2 = d_gate_up[half2_idx]
        
        energy_half1 = np.mean(dgu_half1**2, axis=0)
        energy_half2 = np.mean(dgu_half2**2, axis=0)
        
        top10_half1 = set(np.argsort(energy_half1)[::-1][:10])
        top10_half2 = set(np.argsort(energy_half2)[::-1][:10])
        
        stability_overlap = len(top10_half1 & top10_half2)
        log(f"    Top-10 overlap between halves: {stability_overlap}/10")
        
        # Also check top-50 overlap
        top50_half1 = set(np.argsort(energy_half1)[::-1][:50])
        top50_half2 = set(np.argsort(energy_half2)[::-1][:50])
        stability_overlap_50 = len(top50_half1 & top50_half2)
        log(f"    Top-50 overlap between halves: {stability_overlap_50}/50")
        
        # Correlation of energy profiles
        corr_energy = np.corrcoef(energy_half1, energy_half2)[0, 1]
        log(f"    Energy profile correlation between halves: {corr_energy:.4f}")
        
        # ===== Analysis 5: Mask top-10 in each half separately =====
        log(f"\n    === Analysis 5: Mask top-10 per half ===")
        
        # Use half1's top-10 to mask, test on half2
        h1_top10 = list(np.argsort(energy_half1)[::-1][:10])
        d_gate_up_masked_h1 = d_gate_up.copy()
        d_gate_up_masked_h1[:, h1_top10] = 0
        pca_masked_h1, _, _, _ = compute_pca_explained(d_gate_up_masked_h1)
        log(f"    Mask half1-top10: PC1={pca_masked_h1[0]:.4f}")
        
        # Use full top-10 to mask (same as Analysis 1)
        log(f"    Compare with full top-10 mask: PC1={pca_dgu[0] if 'pca_masked' not in dir() else 'see above'}")
        
        # ===== Store results =====
        layer_result = {
            "n_pairs": n_pairs,
            "d_ff": int(d_ff),
            "baseline_dgu_pc1": float(pca_dgu[0]),
            "baseline_dh_pc1": float(pca_dh[0]),
            "baseline_dgu_eff_rank": rank_dgu,
            "top10_channels": [int(x) for x in top10_ch],
            "top10_energies": [float(dgu_energy[c]) for c in top10_ch],
            "top10_cumulative_energy": float(np.sum(dgu_energy[top10_ch])/np.sum(dgu_energy)),
            "overlap_with_v1_top10": overlap_top10_v1_top10,
            "overlap_with_v1_top50": overlap_top10_v1_top50,
            "overlap_with_v1_top100": overlap_top10_v1_top100,
            "corr_dgu_energy_v1_abs": float(corr_dgu_v1),
            "split_half_top10_overlap": stability_overlap,
            "split_half_top50_overlap": stability_overlap_50,
            "split_half_energy_corr": float(corr_energy),
            "mask_top5_dgu_pc1": float(compute_pca_explained(
                d_gate_up * (np.arange(d_ff)[:, None] != dgu_top_indices[0]).T.astype(float)  # quick mask
            )[0]) if False else None,  # skip this, already computed above
        }
        
        # Add mask results
        mask_results = {}
        for K in [5, 10, 20]:
            d_masked = d_gate_up.copy()
            d_masked[:, dgu_top_indices[:K]] = 0
            pca_m, rank_m, _, _ = compute_pca_explained(d_masked)
            mask_results[f"mask_top{K}"] = {
                "dgu_pc1": float(pca_m[0]),
                "dgu_eff_rank": rank_m,
            }
        layer_result["mask_results"] = mask_results
        
        results[str(l)] = layer_result
        log(f"    Layer {l} done in {time.time()-t_layer:.1f}s")
    
    # Release model
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Save
    all_results = {
        "model": model_name,
        "n_pairs": n_pairs,
        "phase": "375b",
        "layer_results": results,
    }
    
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj
    
    all_results = convert_numpy(all_results)
    
    out_dir = "results/phase375b_confirmation"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{model_name}_phase375b.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log(f"  Results saved to {out_path}")
    
    # Print summary
    log(f"\n{'='*60}")
    log(f"Phase 375b Summary: {model_name} (n_pairs={n_pairs})")
    log(f"{'='*60}")
    for l in sorted(results.keys(), key=int):
        r = results[l]
        log(f"  L{int(l):>3}: base PC1={r['baseline_dgu_pc1']:.4f}, "
            f"top10_cum_energy={r['top10_cumulative_energy']:.4f}, "
            f"v1 overlap={r['overlap_with_v1_top10']}/10, "
            f"split-half overlap={r['split_half_top10_overlap']}/10, "
            f"corr(dgu,v1)={r['corr_dgu_energy_v1_abs']:.4f}")
    
    return all_results


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "deepseek7b"
    
    if model_name == "all":
        for name in ["qwen3", "deepseek7b"]:
            try:
                run_confirmation(name)
            except Exception as e:
                log(f"!!! {name} failed: {e}")
                import traceback; traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(3)
    else:
        run_confirmation(model_name)
    
    log("\nPhase 375b complete!")


if __name__ == "__main__":
    main()
