"""
Phase 375-377: Combined Analysis — Causal Validation & Semantic Decoding
========================================================================

Phase 375: up_base Semantic Decoding
  - What do the top up_base channels encode?
  - Are they category-specific? object-specific? value-specific?
  - Cross-model comparison of channel concentration

Phase 376: gate×up Causal Masking
  - Mask top-K gate×up channels → does 1D collapse disappear?
  - Mask up_base top channels → does 1D collapse disappear?
  - Shuffle up_base → does 1D collapse disappear?
  - Keep-only top-K → does 1D collapse persist?
  This turns Phase 372-373's attribution into CAUSAL evidence.

Phase 377: DS7B post-RMSNorm Geometry
  - Raw Δh geometry vs post-RMSNorm Δh geometry
  - Does category structure recover after normalization?
  - Compare same-attr cos, within-cat cos, correct/wrong PC1 cos

Models: qwen3, deepseek7b, glm4 (run one at a time)
"""

import sys, os, time, json, gc, copy
import torch
import numpy as np
from datetime import datetime
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, 'tests/glm5')

def log(msg="", end="\n"):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", end=end, flush=True)


# ===== Model Configs =====
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

# ===== Binding Pairs (84 pairs, 8 categories) =====
COLOR_PAIRS = [
    ("apple", "red", "blue"), ("banana", "yellow", "purple"), ("snow", "white", "black"),
    ("sky", "blue", "green"), ("cherry", "red", "blue"), ("leaf", "green", "red"),
    ("rose", "red", "blue"), ("gold", "yellow", "purple"), ("coal", "black", "white"),
    ("silver", "white", "black"), ("milk", "white", "black"), ("honey", "yellow", "blue"),
    ("ruby", "red", "green"), ("emerald", "green", "red"), ("sapphire", "blue", "red"),
    ("moon", "white", "black"), ("flame", "orange", "blue"), ("forest", "green", "white"),
    ("ocean", "blue", "yellow"), ("sun", "yellow", "purple"),
]
TEMP_PAIRS = [
    ("fire", "hot", "cold"), ("desert", "hot", "cold"), ("lava", "hot", "cold"),
    ("ice", "cold", "hot"), ("snow", "cold", "hot"), ("volcano", "hot", "cold"),
    ("furnace", "hot", "cold"), ("glacier", "cold", "hot"),
]
MOISTURE_PAIRS = [
    ("rain", "wet", "dry"), ("ocean", "wet", "dry"), ("river", "wet", "dry"),
    ("sand", "dry", "wet"), ("dust", "dry", "wet"), ("bone", "dry", "wet"),
    ("swamp", "wet", "dry"), ("desert", "dry", "wet"),
]
TEXTURE_PAIRS = [
    ("silk", "smooth", "rough"), ("sandpaper", "rough", "smooth"),
    ("glass", "smooth", "rough"), ("rock", "rough", "smooth"),
    ("velvet", "soft", "hard"), ("diamond", "hard", "soft"),
]
SIZE_PAIRS = [
    ("elephant", "big", "small"), ("mountain", "big", "small"), ("ant", "small", "big"),
    ("planet", "big", "small"), ("grain", "small", "big"), ("whale", "big", "small"),
]
WEIGHT_PAIRS = [
    ("boulder", "heavy", "light"), ("feather", "light", "heavy"), ("lead", "heavy", "light"),
    ("balloon", "light", "heavy"), ("steel", "heavy", "light"), ("cotton", "light", "heavy"),
]
SPEED_PAIRS = [
    ("cheetah", "fast", "slow"), ("turtle", "slow", "fast"), ("rocket", "fast", "slow"),
    ("snail", "slow", "fast"), ("lightning", "fast", "slow"), ("sloth", "slow", "fast"),
]
BRIGHT_PAIRS = [
    ("star", "bright", "dark"), ("cave", "dark", "bright"), ("sun", "bright", "dark"),
    ("shadow", "dark", "bright"), ("lamp", "bright", "dark"), ("night", "dark", "bright"),
]

ALL_PAIRS = COLOR_PAIRS + TEMP_PAIRS + MOISTURE_PAIRS + TEXTURE_PAIRS + \
            SIZE_PAIRS + WEIGHT_PAIRS + SPEED_PAIRS + BRIGHT_PAIRS

PAIR_CATEGORIES = (
    ["color"] * len(COLOR_PAIRS) +
    ["temperature"] * len(TEMP_PAIRS) +
    ["moisture"] * len(MOISTURE_PAIRS) +
    ["texture"] * len(TEXTURE_PAIRS) +
    ["size"] * len(SIZE_PAIRS) +
    ["weight"] * len(WEIGHT_PAIRS) +
    ["speed"] * len(SPEED_PAIRS) +
    ["brightness"] * len(BRIGHT_PAIRS)
)

CORRUPTED_BASELINE = "The item"
TEMPLATE = "The {obj} is {attr}."

# Category encoding for regression
ALL_CATEGORIES = sorted(set(PAIR_CATEGORIES))
CAT_TO_IDX = {c: i for i, c in enumerate(ALL_CATEGORIES)}


# ===== Utility Functions =====
def _silu(x):
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -50, 50))))

def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

def compute_pca_explained(X):
    """Compute PCA explained variance ratios. Returns (explained, eff_rank, pc1_dir, Vt)."""
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

def rms_norm_with_weight(x, weight=None, eps=1e-6):
    """RMSNorm: x / sqrt(mean(x^2) + eps) * weight (if provided) * sqrt(d)"""
    d = x.shape[-1]
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    result = x / rms * np.sqrt(d)
    if weight is not None:
        result = result * weight
    return result

def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
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


def get_mlp_activation_fn(model_name):
    if model_name == "glm4":
        return "gelu"
    else:
        return "silu"


def _load_ln_weight(model, model_name, layer_idx):
    """Load post-attention layernorm weight from model or safetensors."""
    import glob
    from safetensors import safe_open
    
    layers = get_layers(model)
    
    # Try to get from model directly
    ln = getattr(layers[layer_idx], "post_attention_layernorm", None)
    if ln is None:
        ln = getattr(layers[layer_idx], "ln2", None)
    
    if ln is not None:
        try:
            w = ln.weight.detach().cpu().float().numpy()
            if w is not None and len(w) > 0:
                return w
        except (NotImplementedError, RuntimeError):
            pass  # meta tensor, need to load from file
    
    # Fallback: load from safetensors
    model_path = MODEL_CONFIGS[model_name]["path"]
    for sf_file in glob.glob(os.path.join(model_path, '*.safetensors')):
        try:
            with safe_open(sf_file, framework='pt', device='cpu') as sf:
                for key in sf.keys():
                    if f"layers.{layer_idx}.post_attention_layernorm.weight" in key:
                        return sf.get_tensor(key).float().numpy()
                    elif f"layers.{layer_idx}.ln2.weight" in key:
                        return sf.get_tensor(key).float().numpy()
        except:
            continue
    
    log(f"    WARNING: Could not load LN weight for layer {layer_idx}")
    return None


# ===== Data Collection =====
def collect_all_activations(model, tokenizer, device, model_name, target_layers):
    """
    Collect all intermediate activations for Phase 375-377.
    
    Returns dict with keys per layer, each containing:
    - mlp_in_clean, mlp_in_corrupt: (n_pairs, d_model)
    - gate_act_clean, gate_act_corrupt: (n_pairs, d_ff)
    - up_clean, up_corrupt: (n_pairs, d_ff)
    - gate_up_clean, gate_up_corrupt: (n_pairs, d_ff)
    - h_clean, h_corrupt: (n_pairs, d_model) — hidden states at layer output
    - W_gate, W_up, W_down: weight matrices
    """
    log("\n=== Collecting all intermediate activations ===")
    
    n_pairs = len(ALL_PAIRS)
    layers = get_layers(model)
    input_device = next(model.parameters()).device
    act_fn = get_mlp_activation_fn(model_name)
    
    all_data = {}
    
    for l in target_layers:
        log(f"\n  Layer {l}:")
        t_layer = time.time()
        
        W_gate, W_up, W_down = load_mlp_weights(model, model_name, l)
        if W_gate is None or W_down is None:
            log(f"    Could not load weights, skipping")
            continue
        
        d_ff = W_gate.shape[0]
        
        mlp_module = layers[l].mlp  # Hook on MLP submodule, not full layer
        
        # Collect post-attention LN weight for Phase 377
        ln_weight = _load_ln_weight(model, model_name, l)
        
        mlp_in_clean_list = []
        mlp_in_corrupt_list = []
        h_clean_list = []
        h_corrupt_list = []
        
        for pidx, (obj, target, competitor) in enumerate(ALL_PAIRS):
            if pidx % 20 == 0:
                log(f"    Pair {pidx+1}/{n_pairs}")
            
            clean_prompt = TEMPLATE.format(obj=obj, attr=target)
            corrupt_prompt = TEMPLATE.format(obj=CORRUPTED_BASELINE, attr=target)
            
            clean_inputs = tokenizer(clean_prompt, return_tensors="pt", truncation=True, max_length=64)
            corrupt_inputs = tokenizer(corrupt_prompt, return_tensors="pt", truncation=True, max_length=64)
            
            captured = {}
            
            def mlp_input_hook(module, input, output=None):
                captured["mlp_input"] = input[0].detach().cpu().float()
            
            # Hook on the MLP submodule to get its input (post-LN attention output)
            h_mlp_in = mlp_module.register_forward_pre_hook(mlp_input_hook)
            
            # Forward - clean
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
            
            # Forward - corrupt
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
        
        # Stack
        mlp_in_clean = np.array(mlp_in_clean_list)
        mlp_in_corrupt = np.array(mlp_in_corrupt_list)
        h_clean = np.array(h_clean_list)
        h_corrupt = np.array(h_corrupt_list)
        
        # Compute intermediate activations offline
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
        
        # Differences
        d_gate_up = gate_up_clean - gate_up_corrupt
        d_gate_act = gate_act_clean - gate_act_corrupt
        d_up = up_clean - up_corrupt
        dh = h_clean - h_corrupt
        
        all_data[str(l)] = {
            "mlp_in_clean": mlp_in_clean, "mlp_in_corrupt": mlp_in_corrupt,
            "gate_act_clean": gate_act_clean, "gate_act_corrupt": gate_act_corrupt,
            "up_clean": up_clean, "up_corrupt": up_corrupt,
            "gate_up_clean": gate_up_clean, "gate_up_corrupt": gate_up_corrupt,
            "h_clean": h_clean, "h_corrupt": h_corrupt,
            "d_gate_up": d_gate_up, "d_gate_act": d_gate_act, "d_up": d_up, "dh": dh,
            "W_gate": W_gate, "W_up": W_up, "W_down": W_down,
            "ln_weight": ln_weight,  # For Phase 377 RMSNorm
        }
        
        log(f"    Layer {l} done in {time.time()-t_layer:.1f}s")
    
    return all_data


# ===== Phase 375: up_base Semantic Decoding =====
def phase375_upbase_semantic(all_data, model_name, target_layers):
    """
    Analyze what up_base (up_corrupt) top channels encode.
    
    Key analyses:
    1. Channel energy concentration
    2. Top channel semantic regression (category, object, attribute)
    3. Channel-category selectivity
    4. Cross-model comparison
    """
    log("\n" + "="*60)
    log("Phase 375: up_base Semantic Decoding")
    log("="*60)
    
    results = {}
    
    for l in target_layers:
        l_str = str(l)
        if l_str not in all_data:
            continue
        
        d = all_data[l_str]
        up_base = d["up_corrupt"]        # (n_pairs, d_ff) — up_base = up_corrupt
        gate_change = d["d_gate_act"]     # (n_pairs, d_ff)
        d_gate_up = d["d_gate_up"]        # (n_pairs, d_ff)
        d_ff = up_base.shape[1]
        n_pairs = up_base.shape[0]
        
        log(f"\n  Layer {l} (d_ff={d_ff}, n_pairs={n_pairs}):")
        
        # ===== Analysis 1: Channel energy concentration =====
        log(f"\n    === Analysis 1: up_base channel energy ===")
        
        # Energy per channel: mean |up_base[i]|^2 across pairs
        channel_energy = np.mean(up_base**2, axis=0)  # (d_ff,)
        total_energy = np.sum(channel_energy)
        energy_frac = channel_energy / total_energy
        
        # Sort channels by energy
        sorted_indices = np.argsort(energy_frac)[::-1]
        sorted_energy = energy_frac[sorted_indices]
        
        cum_energy = np.cumsum(sorted_energy)
        top1_frac = cum_energy[0]
        top10_frac = cum_energy[9]
        top50_frac = cum_energy[49]
        top100_frac = cum_energy[99] if d_ff > 100 else cum_energy[-1]
        top1000_frac = cum_energy[min(999, d_ff-1)]
        
        log(f"    up_base energy: top1={top1_frac:.6f}, top10={top10_frac:.4f}, "
            f"top50={top50_frac:.4f}, top100={top100_frac:.4f}, top1000={top1000_frac:.4f}")
        
        # Also check gate_change energy concentration
        gate_change_energy = np.mean(gate_change**2, axis=0)
        gc_total = np.sum(gate_change_energy)
        gc_frac = gate_change_energy / gc_total
        gc_sorted = np.sort(gc_frac)[::-1]
        gc_cum = np.cumsum(gc_sorted)
        
        log(f"    gate_change energy: top1={gc_cum[0]:.6f}, top10={gc_cum[9]:.4f}, top100={gc_cum[99]:.4f}")
        
        # gate×up Δ energy concentration
        dgu_energy = np.mean(d_gate_up**2, axis=0)
        dgu_total = np.sum(dgu_energy)
        dgu_frac = dgu_energy / dgu_total
        dgu_sorted = np.sort(dgu_frac)[::-1]
        dgu_cum = np.cumsum(dgu_sorted)
        
        log(f"    Δ(gate*up) energy: top1={dgu_cum[0]:.6f}, top10={dgu_cum[9]:.4f}, top100={dgu_cum[99]:.4f}")
        
        # ===== Analysis 2: Top up_base channels — what do they respond to? =====
        log(f"\n    === Analysis 2: Top up_base channel selectivity ===")
        
        top_k = min(50, d_ff)
        top_channels = sorted_indices[:top_k]
        
        # For each top channel, compute:
        # a) Which category activates it most?
        # b) Is it category-selective? (between-cat variance / total variance)
        
        channel_cat_selectivity = {}
        channel_cat_profile = {}
        
        for ch_idx in top_channels[:20]:  # Detailed analysis for top 20
            ch_vals = up_base[:, ch_idx]  # (n_pairs,)
            
            # Compute per-category mean
            cat_means = {}
            for cat in ALL_CATEGORIES:
                cat_idx = [i for i, c in enumerate(PAIR_CATEGORIES) if c == cat]
                cat_means[cat] = float(np.mean(ch_vals[cat_idx]))
            
            # Between-category variance vs total variance
            grand_mean = np.mean(ch_vals)
            total_var = np.var(ch_vals)
            
            between_var = 0
            for cat in ALL_CATEGORIES:
                cat_idx = [i for i, c in enumerate(PAIR_CATEGORIES) if c == cat]
                n_cat = len(cat_idx)
                between_var += n_cat * (cat_means[cat] - grand_mean)**2
            between_var /= n_pairs
            
            selectivity = between_var / (total_var + 1e-10)  # η²-like measure
            best_cat = max(cat_means, key=lambda k: abs(cat_means[k] - grand_mean))
            
            channel_cat_selectivity[int(ch_idx)] = {
                "selectivity": float(selectivity),
                "best_category": best_cat,
                "cat_means": cat_means,
            }
            channel_cat_profile[int(ch_idx)] = cat_means
        
        # Summary
        selectivities = [v["selectivity"] for v in channel_cat_selectivity.values()]
        log(f"    Top-20 channel category selectivity (η²): "
            f"mean={np.mean(selectivities):.4f}, max={np.max(selectivities):.4f}")
        
        # Print top 10 channels
        for i, ch_idx in enumerate(top_channels[:10]):
            sel = channel_cat_selectivity.get(int(ch_idx), {})
            log(f"      Ch {ch_idx}: energy={energy_frac[ch_idx]:.6f}, "
                f"η²={sel.get('selectivity',0):.4f}, best_cat={sel.get('best_category','?')}")
        
        # ===== Analysis 3: Do top up_base channels overlap with top gate_change channels? =====
        log(f"\n    === Analysis 3: Channel overlap ===")
        
        gate_change_top = set(np.argsort(gc_frac)[::-1][:100])
        up_base_top = set(top_channels[:100])
        dgu_top = set(np.argsort(dgu_frac)[::-1][:100])
        
        overlap_gu_ub = len(gate_change_top & up_base_top)
        overlap_dgu_ub = len(dgu_top & up_base_top)
        overlap_dgu_gc = len(dgu_top & gate_change_top)
        
        log(f"    Top-100 overlap: gate_change∩up_base={overlap_gu_ub}/100, "
            f"Δ(gate*up)∩up_base={overlap_dgu_ub}/100, Δ(gate*up)∩gate_change={overlap_dgu_gc}/100")
        
        # ===== Analysis 4: up_base channel consistency across categories =====
        log(f"\n    === Analysis 4: up_base channel consistency ===")
        
        # Are the top up_base channels the same across different attribute categories?
        cat_top_channels = {}
        for cat in ALL_CATEGORIES:
            cat_idx = [i for i, c in enumerate(PAIR_CATEGORIES) if c == cat]
            cat_up = up_base[cat_idx]
            cat_energy = np.mean(cat_up**2, axis=0)
            cat_top10 = set(np.argsort(cat_energy)[::-1][:10])
            cat_top_channels[cat] = cat_top10
        
        # Pairwise Jaccard similarity of top-10 channels between categories
        cat_pairs_jaccard = []
        for i, c1 in enumerate(ALL_CATEGORIES):
            for j, c2 in enumerate(ALL_CATEGORIES):
                if j <= i:
                    continue
                intersection = len(cat_top_channels[c1] & cat_top_channels[c2])
                union = len(cat_top_channels[c1] | cat_top_channels[c2])
                jaccard = intersection / (union + 1e-10)
                cat_pairs_jaccard.append(jaccard)
        
        log(f"    Cross-category Jaccard of top-10 up_base channels: "
            f"mean={np.mean(cat_pairs_jaccard):.4f}, min={np.min(cat_pairs_jaccard):.4f}")
        
        # ===== Analysis 5: gate_change × up_base decomposition per channel =====
        log(f"\n    === Analysis 5: Per-channel contribution to Δ(gate*up) ===")
        
        # gate_contribution = gate_change * up_base
        gate_contrib = gate_change * up_base  # (n_pairs, d_ff)
        # up_contribution = gate_base * up_change = gate_act_corrupt * d_up
        gate_act_corrupt = d["gate_act_corrupt"]
        d_up_local = d["d_up"]
        up_contrib = gate_act_corrupt * d_up_local
        interaction = gate_change * d_up_local
        
        # Per-channel contribution norms
        gate_contrib_energy = np.mean(gate_contrib**2, axis=0)
        up_contrib_energy = np.mean(up_contrib**2, axis=0)
        interact_energy = np.mean(interaction**2, axis=0)
        
        total_per_channel = gate_contrib_energy + up_contrib_energy + interact_energy
        
        # For top Δ(gate*up) channels, what fraction comes from gate_change × up_base?
        dgu_top10_ch = np.argsort(dgu_frac)[::-1][:10]
        
        for ch in dgu_top10_ch[:5]:
            gc_frac_ch = gate_contrib_energy[ch] / (total_per_channel[ch] + 1e-10)
            uc_frac_ch = up_contrib_energy[ch] / (total_per_channel[ch] + 1e-10)
            ic_frac_ch = interact_energy[ch] / (total_per_channel[ch] + 1e-10)
            log(f"      Ch {ch}: gate×up_base={gc_frac_ch:.3f}, gate_base×up_change={uc_frac_ch:.3f}, "
                f"interaction={ic_frac_ch:.3f}")
        
        # ===== Store results =====
        layer_result = {
            "d_ff": int(d_ff),
            "n_pairs": n_pairs,
            # Analysis 1: Energy concentration
            "upbase_energy_top1": float(top1_frac),
            "upbase_energy_top10": float(top10_frac),
            "upbase_energy_top50": float(top50_frac),
            "upbase_energy_top100": float(top100_frac),
            "upbase_energy_top1000": float(top1000_frac),
            "gate_change_energy_top1": float(gc_cum[0]),
            "gate_change_energy_top10": float(gc_cum[9]),
            "gate_change_energy_top100": float(gc_cum[99]),
            "dgu_energy_top1": float(dgu_cum[0]),
            "dgu_energy_top10": float(dgu_cum[9]),
            "dgu_energy_top100": float(dgu_cum[99]),
            # Analysis 2: Channel selectivity
            "top_channel_cat_selectivity": {str(k): v for k, v in channel_cat_selectivity.items()},
            "mean_cat_selectivity": float(np.mean(selectivities)),
            # Analysis 3: Channel overlap
            "overlap_gate_change_up_base_top100": int(overlap_gu_ub),
            "overlap_dgu_up_base_top100": int(overlap_dgu_ub),
            "overlap_dgu_gate_change_top100": int(overlap_dgu_gc),
            # Analysis 4: Consistency
            "cross_category_jaccard_mean": float(np.mean(cat_pairs_jaccard)),
            "cross_category_jaccard_min": float(np.min(cat_pairs_jaccard)),
            # Top channels
            "top10_up_base_channels": [int(x) for x in top_channels[:10]],
            "top10_dgu_channels": [int(x) for x in dgu_top10_ch.tolist()],
        }
        
        results[str(l)] = layer_result
    
    return results


# ===== Phase 376: gate×up Causal Masking =====
def phase376_causal_masking(all_data, model_name, target_layers):
    """
    Causal masking experiments on gate×up channels.
    
    Interventions:
    1. Mask top-K Δ(gate*up) channels (set to 0)
    2. Keep-only top-K Δ(gate*up) channels
    3. Mask top-K up_base channels in gate×up computation
    4. Shuffle up_base channels
    5. Replace up_base with mean up_base
    
    For each intervention, compute:
    - gate×up PCA (PC1, eff_rank)
    - MLP output PCA (via W_down @ masked_gate_up)
    - Post-RMSNorm Δh PCA (for DS7B)
    """
    log("\n" + "="*60)
    log("Phase 376: gate×up Causal Masking")
    log("="*60)
    
    results = {}
    
    for l in target_layers:
        l_str = str(l)
        if l_str not in all_data:
            continue
        
        d = all_data[l_str]
        d_gate_up = d["d_gate_up"]        # (n_pairs, d_ff)
        d_gate_act = d["d_gate_act"]       # (n_pairs, d_ff)
        d_up = d["d_up"]                   # (n_pairs, d_ff)
        up_base = d["up_corrupt"]          # (n_pairs, d_ff) — up_base
        gate_act_clean = d["gate_act_clean"]
        gate_act_corrupt = d["gate_act_corrupt"]
        gate_up_clean = d["gate_up_clean"]
        gate_up_corrupt = d["gate_up_corrupt"]
        W_down = d["W_down"]
        dh = d["dh"]                       # (n_pairs, d_model)
        h_clean = d["h_clean"]
        h_corrupt = d["h_corrupt"]
        d_ff = d_gate_up.shape[1]
        d_model = dh.shape[1]
        n_pairs = dh.shape[0]
        
        log(f"\n  Layer {l} (d_ff={d_ff}):")
        
        # ===== Baseline =====
        log(f"\n    === Baseline PCA ===")
        
        pca_dgu_base, rank_dgu_base, pc1_dgu_base, _ = compute_pca_explained(d_gate_up)
        pca_dh_base, rank_dh_base, pc1_dh_base, Vt_dh = compute_pca_explained(dh)
        
        # MLP output baseline
        mlp_out_base = (W_down @ d_gate_up.T).T  # (n_pairs, d_model)
        pca_mlp_base, rank_mlp_base, _, _ = compute_pca_explained(mlp_out_base)
        
        log(f"    Baseline: Δ(gate*up) PC1={pca_dgu_base[0]:.4f}, Δh PC1={pca_dh_base[0]:.4f}, "
            f"Δ(MLP out) PC1={pca_mlp_base[0]:.4f}")
        
        intervention_results = {}
        
        # ===== Intervention 1: Mask top-K Δ(gate*up) channels =====
        log(f"\n    === Intervention 1: Mask top-K Δ(gate*up) channels ===")
        
        dgu_energy = np.mean(d_gate_up**2, axis=0)
        dgu_top_indices = np.argsort(dgu_energy)[::-1]
        
        for K in [10, 50, 100]:
            mask_ch = dgu_top_indices[:K]
            d_gate_up_masked = d_gate_up.copy()
            d_gate_up_masked[:, mask_ch] = 0
            
            pca_masked, rank_masked, _, _ = compute_pca_explained(d_gate_up_masked)
            mlp_out_masked = (W_down @ d_gate_up_masked.T).T
            pca_mlp_masked, _, _, _ = compute_pca_explained(mlp_out_masked)
            
            # New Δh = old Δh - W_down @ (original - masked) 
            # = old Δh - W_down @ d_gate_up_removed
            dgu_removed = d_gate_up - d_gate_up_masked
            dh_new = dh - (W_down @ dgu_removed.T).T
            pca_dh_new, rank_dh_new, _, _ = compute_pca_explained(dh_new)
            
            log(f"    Mask top-{K}: Δ(gate*up) PC1={pca_masked[0]:.4f} (was {pca_dgu_base[0]:.4f}), "
                f"Δh PC1={pca_dh_new[0]:.4f} (was {pca_dh_base[0]:.4f})")
            
            intervention_results[f"mask_dgu_top{K}"] = {
                "dgu_pc1": float(pca_masked[0]),
                "dgu_eff_rank": rank_masked,
                "dh_pc1": float(pca_dh_new[0]),
                "dh_eff_rank": rank_dh_new,
                "mlp_out_pc1": float(pca_mlp_masked[0]),
            }
        
        # ===== Intervention 2: Keep-only top-K Δ(gate*up) channels =====
        log(f"\n    === Intervention 2: Keep-only top-K Δ(gate*up) channels ===")
        
        for K in [10, 50]:
            keep_ch = dgu_top_indices[:K]
            d_gate_up_kept = np.zeros_like(d_gate_up)
            d_gate_up_kept[:, keep_ch] = d_gate_up[:, keep_ch]
            
            pca_kept, rank_kept, _, _ = compute_pca_explained(d_gate_up_kept)
            mlp_out_kept = (W_down @ d_gate_up_kept.T).T
            pca_mlp_kept, _, _, _ = compute_pca_explained(mlp_out_kept)
            
            dgu_removed = d_gate_up - d_gate_up_kept
            dh_new = dh - (W_down @ dgu_removed.T).T
            pca_dh_new, rank_dh_new, _, _ = compute_pca_explained(dh_new)
            
            log(f"    Keep top-{K}: Δ(gate*up) PC1={pca_kept[0]:.4f}, "
                f"Δh PC1={pca_dh_new[0]:.4f}")
            
            intervention_results[f"keep_dgu_top{K}"] = {
                "dgu_pc1": float(pca_kept[0]),
                "dgu_eff_rank": rank_kept,
                "dh_pc1": float(pca_dh_new[0]),
                "dh_eff_rank": rank_dh_new,
                "mlp_out_pc1": float(pca_mlp_kept[0]),
            }
        
        # ===== Intervention 3: Mask up_base top channels =====
        log(f"\n    === Intervention 3: Mask up_base top-K channels ===")
        
        # NOTE: Correct computation for modifying up_base (up_corrupt):
        # Δ(gate*up)_new = gate_up_clean - gate_act_corrupt * up_base_new
        # (NOT decomposition approach, which gives wrong results)
        
        up_energy = np.mean(up_base**2, axis=0)
        up_top_indices = np.argsort(up_energy)[::-1]
        
        for K in [10, 50, 100]:
            mask_ch = up_top_indices[:K]
            up_base_masked = up_base.copy()
            up_base_masked[:, mask_ch] = 0
            
            # Correct: Δ(gate*up)_new = gate_up_clean - gate_act_corrupt * up_base_masked
            gate_up_corrupt_masked = gate_act_corrupt * up_base_masked
            d_gate_up_new = gate_up_clean - gate_up_corrupt_masked
            
            pca_new, rank_new, _, _ = compute_pca_explained(d_gate_up_new)
            mlp_out_new = (W_down @ d_gate_up_new.T).T
            pca_mlp_new, _, _, _ = compute_pca_explained(mlp_out_new)
            
            dgu_diff = d_gate_up - d_gate_up_new
            dh_new = dh - (W_down @ dgu_diff.T).T
            pca_dh_new, rank_dh_new, _, _ = compute_pca_explained(dh_new)
            
            log(f"    Mask up_base top-{K}: Δ(gate*up) PC1={pca_new[0]:.4f} (was {pca_dgu_base[0]:.4f}), "
                f"Δh PC1={pca_dh_new[0]:.4f} (was {pca_dh_base[0]:.4f})")
            
            intervention_results[f"mask_upbase_top{K}"] = {
                "dgu_pc1": float(pca_new[0]),
                "dgu_eff_rank": rank_new,
                "dh_pc1": float(pca_dh_new[0]),
                "dh_eff_rank": rank_dh_new,
                "mlp_out_pc1": float(pca_mlp_new[0]),
            }
        
        # ===== Intervention 4: Shuffle up_base channels =====
        log(f"\n    === Intervention 4: Shuffle up_base channels ===")
        
        rng = np.random.RandomState(42)
        n_shuffles = 5
        shuffle_pc1s = []
        shuffle_dh_pc1s = []
        
        for s in range(n_shuffles):
            up_base_shuffled = up_base.copy()
            # Shuffle each channel independently across pairs
            for ch in range(d_ff):
                rng.shuffle(up_base_shuffled[:, ch])
            
            # Correct: Δ(gate*up)_new = gate_up_clean - gate_act_corrupt * up_base_shuffled
            gate_up_corrupt_shuffled = gate_act_corrupt * up_base_shuffled
            d_gate_up_shuffled = gate_up_clean - gate_up_corrupt_shuffled
            
            pca_shuffled, rank_shuffled, _, _ = compute_pca_explained(d_gate_up_shuffled)
            shuffle_pc1s.append(float(pca_shuffled[0]))
            
            mlp_out_shuffled = (W_down @ d_gate_up_shuffled.T).T
            dgu_diff = d_gate_up - d_gate_up_shuffled
            dh_new = dh - (W_down @ dgu_diff.T).T
            pca_dh_new, _, _, _ = compute_pca_explained(dh_new)
            shuffle_dh_pc1s.append(float(pca_dh_new[0]))
        
        log(f"    Shuffle up_base: Δ(gate*up) PC1 = {np.mean(shuffle_pc1s):.4f} ± {np.std(shuffle_pc1s):.4f} "
            f"(baseline={pca_dgu_base[0]:.4f})")
        log(f"    Shuffle up_base: Δh PC1 = {np.mean(shuffle_dh_pc1s):.4f} ± {np.std(shuffle_dh_pc1s):.4f} "
            f"(baseline={pca_dh_base[0]:.4f})")
        
        intervention_results["shuffle_upbase"] = {
            "dgu_pc1_mean": float(np.mean(shuffle_pc1s)),
            "dgu_pc1_std": float(np.std(shuffle_pc1s)),
            "dgu_pc1_all": shuffle_pc1s,
            "dh_pc1_mean": float(np.mean(shuffle_dh_pc1s)),
            "dh_pc1_std": float(np.std(shuffle_dh_pc1s)),
            "dh_pc1_all": shuffle_dh_pc1s,
        }
        
        # ===== Intervention 5: Replace up_base with mean (uniform) =====
        log(f"\n    === Intervention 5: Replace up_base with uniform ===")
        
        up_base_mean_val = np.mean(up_base)
        up_base_uniform = np.full_like(up_base, up_base_mean_val)
        
        # Correct: Δ(gate*up)_new = gate_up_clean - gate_act_corrupt * up_base_uniform
        gate_up_corrupt_uniform = gate_act_corrupt * up_base_uniform
        d_gate_up_uniform = gate_up_clean - gate_up_corrupt_uniform
        
        pca_uniform, rank_uniform, _, _ = compute_pca_explained(d_gate_up_uniform)
        mlp_out_uniform = (W_down @ d_gate_up_uniform.T).T
        pca_mlp_uniform, _, _, _ = compute_pca_explained(mlp_out_uniform)
        
        dgu_diff = d_gate_up - d_gate_up_uniform
        dh_new = dh - (W_down @ dgu_diff.T).T
        pca_dh_new, rank_dh_new, _, _ = compute_pca_explained(dh_new)
        
        log(f"    Uniform up_base: Δ(gate*up) PC1={pca_uniform[0]:.4f} (was {pca_dgu_base[0]:.4f}), "
            f"Δh PC1={pca_dh_new[0]:.4f} (was {pca_dh_base[0]:.4f})")
        
        intervention_results["uniform_upbase"] = {
            "dgu_pc1": float(pca_uniform[0]),
            "dgu_eff_rank": rank_uniform,
            "dh_pc1": float(pca_dh_new[0]),
            "dh_eff_rank": rank_dh_new,
            "mlp_out_pc1": float(pca_mlp_uniform[0]),
        }
        
        # ===== Intervention 6: gate-only vs up-only patch =====
        log(f"\n    === Intervention 6: gate-only vs up-only decomposition ===")
        
        # gate-only: only the gate contribution term (d_gate_act * up_base)
        # This is the term Phase 373 identified as dominant (63.5%)
        gate_only = d_gate_act * up_base
        pca_gate_only, rank_gate_only, _, _ = compute_pca_explained(gate_only)
        mlp_out_gate_only = (W_down @ gate_only.T).T
        pca_mlp_gate_only, _, _, _ = compute_pca_explained(mlp_out_gate_only)
        
        # up-only: only gate_base * up_change (no gate_change)
        up_only = gate_act_corrupt * d_up
        pca_up_only, rank_up_only, _, _ = compute_pca_explained(up_only)
        mlp_out_up_only = (W_down @ up_only.T).T
        pca_mlp_up_only, _, _, _ = compute_pca_explained(mlp_out_up_only)
        
        log(f"    gate-only (Δgate × up_base): PC1={pca_gate_only[0]:.4f}, MLP PC1={pca_mlp_gate_only[0]:.4f}")
        log(f"    up-only (gate_base × Δup): PC1={pca_up_only[0]:.4f}, MLP PC1={pca_mlp_up_only[0]:.4f}")
        
        intervention_results["gate_only"] = {
            "dgu_pc1": float(pca_gate_only[0]),
            "dgu_eff_rank": rank_gate_only,
            "mlp_out_pc1": float(pca_mlp_gate_only[0]),
        }
        intervention_results["up_only"] = {
            "dgu_pc1": float(pca_up_only[0]),
            "dgu_eff_rank": rank_up_only,
            "mlp_out_pc1": float(pca_mlp_up_only[0]),
        }
        
        # ===== Store results =====
        layer_result = {
            "d_ff": int(d_ff),
            "n_pairs": n_pairs,
            "baseline": {
                "dgu_pc1": float(pca_dgu_base[0]),
                "dgu_eff_rank": rank_dgu_base,
                "dh_pc1": float(pca_dh_base[0]),
                "dh_eff_rank": rank_dh_base,
                "mlp_out_pc1": float(pca_mlp_base[0]),
            },
            "interventions": intervention_results,
        }
        
        results[str(l)] = layer_result
    
    return results


# ===== Phase 377: post-RMSNorm Geometry =====
def phase377_postnorm_geometry(all_data, model_name, target_layers):
    """
    Compare raw Δh geometry vs post-RMSNorm Δh geometry.
    
    Key: RMSNorm(h_clean) - RMSNorm(h_corrupt) ≠ RMSNorm(Δh)
    We need to compute: Δh_norm = RMSNorm(h_clean) - RMSNorm(h_corrupt)
    
    For DS7B, test if category structure recovers after normalization.
    """
    log("\n" + "="*60)
    log("Phase 377: post-RMSNorm Geometry")
    log("="*60)
    
    results = {}
    
    for l in target_layers:
        l_str = str(l)
        if l_str not in all_data:
            continue
        
        d = all_data[l_str]
        h_clean = d["h_clean"]       # (n_pairs, d_model)
        h_corrupt = d["h_corrupt"]   # (n_pairs, d_model)
        dh = d["dh"]                 # (n_pairs, d_model)
        ln_weight = d.get("ln_weight", None)  # For RMSNorm
        d_model = h_clean.shape[1]
        n_pairs = h_clean.shape[0]
        
        log(f"\n  Layer {l}:")
        
        # ===== Compute post-RMSNorm Δh =====
        # RMSNorm(h) = h / sqrt(mean(h^2) + eps) * sqrt(d) * weight
        # Δh_norm = RMSNorm(h_clean) - RMSNorm(h_corrupt)
        h_clean_norm = rms_norm_with_weight(h_clean, ln_weight)
        h_corrupt_norm = rms_norm_with_weight(h_corrupt, ln_weight)
        dh_norm = h_clean_norm - h_corrupt_norm
        
        # Also compute approximate: RMSNorm(Δh) for comparison
        dh_rms = rms_norm_with_weight(dh, ln_weight)
        
        # ===== Raw Δh geometry =====
        log(f"\n    === Raw Δh geometry ===")
        
        pca_raw, rank_raw, pc1_raw, Vt_raw = compute_pca_explained(dh)
        log(f"    Raw: PC1={pca_raw[0]:.4f}, PC2={pca_raw[1]:.4f}, eff_rank={rank_raw}")
        
        # Same-attribute cosine (raw)
        same_attr_cos_raw = []
        diff_attr_cos_raw = []
        for i in range(n_pairs):
            for j in range(i+1, n_pairs):
                cos = cosine_sim(dh[i], dh[j])
                if PAIR_CATEGORIES[i] == PAIR_CATEGORIES[j]:
                    same_attr_cos_raw.append(cos)
                else:
                    diff_attr_cos_raw.append(cos)
        
        mean_same_raw = np.mean(same_attr_cos_raw)
        mean_diff_raw = np.mean(diff_attr_cos_raw)
        log(f"    Raw: same-attr cos={mean_same_raw:.4f}, diff-attr cos={mean_diff_raw:.4f}")
        
        # Within-category cosine (raw)
        within_cat_cos_raw = {}
        for cat in ALL_CATEGORIES:
            idx = [i for i, c in enumerate(PAIR_CATEGORIES) if c == cat]
            if len(idx) >= 2:
                cat_vecs = dh[idx]
                cent = cat_vecs.mean(axis=0)
                cos_vals = [cosine_sim(v, cent) for v in cat_vecs]
                within_cat_cos_raw[cat] = float(np.mean(cos_vals))
        
        mean_within_raw = np.mean(list(within_cat_cos_raw.values()))
        log(f"    Raw: within-cat cos={mean_within_raw:.4f}")
        
        # ===== Post-RMSNorm Δh geometry =====
        log(f"\n    === Post-RMSNorm Δh geometry ===")
        
        pca_norm, rank_norm, pc1_norm, Vt_norm = compute_pca_explained(dh_norm)
        log(f"    Norm: PC1={pca_norm[0]:.4f}, PC2={pca_norm[1]:.4f}, eff_rank={rank_norm}")
        
        # Same-attribute cosine (norm)
        same_attr_cos_norm = []
        diff_attr_cos_norm = []
        for i in range(n_pairs):
            for j in range(i+1, n_pairs):
                cos = cosine_sim(dh_norm[i], dh_norm[j])
                if PAIR_CATEGORIES[i] == PAIR_CATEGORIES[j]:
                    same_attr_cos_norm.append(cos)
                else:
                    diff_attr_cos_norm.append(cos)
        
        mean_same_norm = np.mean(same_attr_cos_norm)
        mean_diff_norm = np.mean(diff_attr_cos_norm)
        log(f"    Norm: same-attr cos={mean_same_norm:.4f}, diff-attr cos={mean_diff_norm:.4f}")
        
        # Within-category cosine (norm)
        within_cat_cos_norm = {}
        for cat in ALL_CATEGORIES:
            idx = [i for i, c in enumerate(PAIR_CATEGORIES) if c == cat]
            if len(idx) >= 2:
                cat_vecs = dh_norm[idx]
                cent = cat_vecs.mean(axis=0)
                cos_vals = [cosine_sim(v, cent) for v in cat_vecs]
                within_cat_cos_norm[cat] = float(np.mean(cos_vals))
        
        mean_within_norm = np.mean(list(within_cat_cos_norm.values()))
        log(f"    Norm: within-cat cos={mean_within_norm:.4f}")
        
        # ===== Correct vs Wrong binding (using wrong prompts from Phase 374 data) =====
        # We need to collect wrong binding data too, but for now use the norm data
        # We'll compute correct vs wrong PC1 alignment using the category centroids
        
        # Category vs category cosine in norm space
        cat_cent_norm = {}
        for cat in ALL_CATEGORIES:
            idx = [i for i, c in enumerate(PAIR_CATEGORIES) if c == cat]
            cat_cent_norm[cat] = dh_norm[idx].mean(axis=0)
        
        between_cat_cos_norm = {}
        for i, c1 in enumerate(ALL_CATEGORIES):
            for j, c2 in enumerate(ALL_CATEGORIES):
                if j <= i:
                    continue
                cos = cosine_sim(cat_cent_norm[c1], cat_cent_norm[c2])
                between_cat_cos_norm[f"{c1}_vs_{c2}"] = float(cos)
        
        # Same in raw space
        cat_cent_raw = {}
        for cat in ALL_CATEGORIES:
            idx = [i for i, c in enumerate(PAIR_CATEGORIES) if c == cat]
            cat_cent_raw[cat] = dh[idx].mean(axis=0)
        
        between_cat_cos_raw = {}
        for i, c1 in enumerate(ALL_CATEGORIES):
            for j, c2 in enumerate(ALL_CATEGORIES):
                if j <= i:
                    continue
                cos = cosine_sim(cat_cent_raw[c1], cat_cent_raw[c2])
                between_cat_cos_raw[f"{c1}_vs_{c2}"] = float(cos)
        
        # ===== Approximate RMSNorm(Δh) geometry =====
        pca_rms, rank_rms, _, _ = compute_pca_explained(dh_rms)
        same_attr_cos_rms = []
        for i in range(n_pairs):
            for j in range(i+1, n_pairs):
                if PAIR_CATEGORIES[i] == PAIR_CATEGORIES[j]:
                    cos = cosine_sim(dh_rms[i], dh_rms[j])
                    same_attr_cos_rms.append(cos)
        mean_same_rms = np.mean(same_attr_cos_rms) if same_attr_cos_rms else 0
        
        # ===== Change metrics =====
        log(f"\n    === Change from raw → norm ===")
        pc1_change = pca_norm[0] - pca_raw[0]
        same_attr_change = mean_same_norm - mean_same_raw
        within_cat_change = mean_within_norm - mean_within_raw
        
        log(f"    PC1: {pca_raw[0]:.4f} → {pca_norm[0]:.4f} (Δ={pc1_change:+.4f})")
        log(f"    same-attr cos: {mean_same_raw:.4f} → {mean_same_norm:.4f} (Δ={same_attr_change:+.4f})")
        log(f"    within-cat cos: {mean_within_raw:.4f} → {mean_within_norm:.4f} (Δ={within_cat_change:+.4f})")
        log(f"    eff_rank: {rank_raw} → {rank_norm}")
        
        # ===== Store results =====
        layer_result = {
            "n_pairs": n_pairs,
            "d_model": d_model,
            # Raw geometry
            "raw_pc1": float(pca_raw[0]),
            "raw_pc2": float(pca_raw[1]),
            "raw_eff_rank": rank_raw,
            "raw_same_attr_cos": float(mean_same_raw),
            "raw_diff_attr_cos": float(mean_diff_raw),
            "raw_within_cat_cos": float(mean_within_raw),
            "raw_within_cat_cos_by_cat": within_cat_cos_raw,
            # Post-RMSNorm geometry
            "norm_pc1": float(pca_norm[0]),
            "norm_pc2": float(pca_norm[1]),
            "norm_eff_rank": rank_norm,
            "norm_same_attr_cos": float(mean_same_norm),
            "norm_diff_attr_cos": float(mean_diff_norm),
            "norm_within_cat_cos": float(mean_within_norm),
            "norm_within_cat_cos_by_cat": within_cat_cos_norm,
            # Approximate RMSNorm(Δh)
            "rms_pc1": float(pca_rms[0]),
            "rms_same_attr_cos": float(mean_same_rms),
            # Change metrics
            "pc1_change": float(pc1_change),
            "same_attr_cos_change": float(same_attr_change),
            "within_cat_cos_change": float(within_cat_change),
            # Between-category cosines
            "raw_between_cat_cos": between_cat_cos_raw,
            "norm_between_cat_cos": between_cat_cos_norm,
        }
        
        results[str(l)] = layer_result
    
    return results


# ===== Main Runner =====
def run_model(model_name):
    cfg = MODEL_CONFIGS[model_name]
    n_layers = cfg["n_layers"]
    d_model = cfg["d_model"]
    
    log(f"\n{'='*60}")
    log(f"Phase 375-377: {model_name}")
    log(f"{'='*60}")
    
    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    log(f"  Model loaded in {time.time()-t0:.1f}s")
    
    # Select target layers
    if model_name == "deepseek7b":
        target_layers = [3, 4, 5, 6, 8, 12, 24]
    elif model_name == "qwen3":
        target_layers = [3, 4, 5, 8, 16, 28]
    else:  # glm4
        target_layers = [3, 4, 5, 10, 20, 30]
    
    # Step 1: Collect all activations
    t1 = time.time()
    all_data = collect_all_activations(model, tokenizer, device, model_name, target_layers)
    log(f"  Data collection done in {time.time()-t1:.1f}s")
    
    # Release model early to free GPU memory for computation
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log(f"  Model released, GPU freed for computation")
    
    # Step 2: Phase 375 — up_base semantic decoding
    t2 = time.time()
    phase375_results = phase375_upbase_semantic(all_data, model_name, target_layers)
    log(f"  Phase 375 done in {time.time()-t2:.1f}s")
    
    # Step 3: Phase 376 — causal masking
    t3 = time.time()
    phase376_results = phase376_causal_masking(all_data, model_name, target_layers)
    log(f"  Phase 376 done in {time.time()-t3:.1f}s")
    
    # Step 4: Phase 377 — post-RMSNorm geometry
    t4 = time.time()
    phase377_results = phase377_postnorm_geometry(all_data, model_name, target_layers)
    log(f"  Phase 377 done in {time.time()-t4:.1f}s")
    
    # ===== Save all results =====
    all_results = {
        "model": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "n_pairs": len(ALL_PAIRS),
        "target_layers": target_layers,
        "pair_categories": PAIR_CATEGORIES,
        "phase375": phase375_results,
        "phase376": phase376_results,
        "phase377": phase377_results,
    }
    
    # Convert numpy arrays to lists for JSON serialization
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
    
    out_dir = "results/phase375_376_377_combined"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{model_name}_phase375_376_377.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log(f"  Results saved to {out_path}")
    
    # ===== Print Summary =====
    print_summary(model_name, phase375_results, phase376_results, phase377_results, target_layers)
    
    return all_results


def print_summary(model_name, p375, p376, p377, target_layers):
    """Print a comprehensive summary of all three phases."""
    log(f"\n{'='*80}")
    log(f"COMBINED SUMMARY: {model_name}")
    log(f"{'='*80}")
    
    # Phase 375 Summary
    log(f"\n--- Phase 375: up_base Semantic Decoding ---")
    log(f"{'Layer':>5} | {'up_top1':>8} | {'up_top10':>8} | {'up_top100':>9} | "
        f"{'dgu_top1':>8} | {'dgu_top10':>8} | {'cat_η²':>7} | {'Jaccard':>7}")
    log("-" * 85)
    for l in target_layers:
        ls = str(l)
        if ls not in p375:
            continue
        r = p375[ls]
        log(f"  L{l:>3} | {r.get('upbase_energy_top1',0):>8.6f} | "
            f"{r.get('upbase_energy_top10',0):>8.4f} | "
            f"{r.get('upbase_energy_top100',0):>9.4f} | "
            f"{r.get('dgu_energy_top1',0):>8.6f} | "
            f"{r.get('dgu_energy_top10',0):>8.4f} | "
            f"{r.get('mean_cat_selectivity',0):>7.4f} | "
            f"{r.get('cross_category_jaccard_mean',0):>7.4f}")
    
    # Phase 376 Summary
    log(f"\n--- Phase 376: Causal Masking ---")
    log(f"{'Layer':>5} | {'base PC1':>8} | {'mask10':>8} | {'mask50':>8} | {'mask100':>8} | "
        f"{'ub_m10':>8} | {'ub_m50':>8} | {'shuf':>8} | {'unif':>8} | {'g_only':>8}")
    log("-" * 105)
    for l in target_layers:
        ls = str(l)
        if ls not in p376:
            continue
        r = p376[ls]
        base = r.get("baseline", {})
        interventions = r.get("interventions", {})
        
        base_pc1 = base.get("dgu_pc1", 0)
        m10 = interventions.get("mask_dgu_top10", {}).get("dgu_pc1", 0)
        m50 = interventions.get("mask_dgu_top50", {}).get("dgu_pc1", 0)
        m100 = interventions.get("mask_dgu_top100", {}).get("dgu_pc1", 0)
        ub_m10 = interventions.get("mask_upbase_top10", {}).get("dgu_pc1", 0)
        ub_m50 = interventions.get("mask_upbase_top50", {}).get("dgu_pc1", 0)
        shuf = interventions.get("shuffle_upbase", {}).get("dgu_pc1_mean", 0)
        unif = interventions.get("uniform_upbase", {}).get("dgu_pc1", 0)
        g_only = interventions.get("gate_only", {}).get("dgu_pc1", 0)
        
        log(f"  L{l:>3} | {base_pc1:>8.4f} | {m10:>8.4f} | {m50:>8.4f} | {m100:>8.4f} | "
            f"{ub_m10:>8.4f} | {ub_m50:>8.4f} | {shuf:>8.4f} | {unif:>8.4f} | {g_only:>8.4f}")
    
    # Phase 377 Summary
    log(f"\n--- Phase 377: post-RMSNorm Geometry ---")
    log(f"{'Layer':>5} | {'rawPC1':>7} | {'normPC1':>8} | {'rawSame':>8} | {'normSame':>8} | "
        f"{'rawWithin':>9} | {'normWithin':>9} | {'rawRank':>7} | {'normRank':>8}")
    log("-" * 95)
    for l in target_layers:
        ls = str(l)
        if ls not in p377:
            continue
        r = p377[ls]
        log(f"  L{l:>3} | {r.get('raw_pc1',0):>7.4f} | {r.get('norm_pc1',0):>8.4f} | "
            f"{r.get('raw_same_attr_cos',0):>8.4f} | {r.get('norm_same_attr_cos',0):>8.4f} | "
            f"{r.get('raw_within_cat_cos',0):>9.4f} | {r.get('norm_within_cat_cos',0):>9.4f} | "
            f"{r.get('raw_eff_rank',0):>7} | {r.get('norm_eff_rank',0):>8}")


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "deepseek7b"
    
    if model_name == "all":
        for name in ["qwen3", "deepseek7b", "glm4"]:
            try:
                run_model(name)
            except Exception as e:
                log(f"!!! {name} failed: {e}")
                import traceback; traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(3)
    else:
        run_model(model_name)
    
    log("\nPhase 375-377 complete!")


if __name__ == "__main__":
    main()
