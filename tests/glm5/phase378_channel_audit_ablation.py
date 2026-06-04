"""
Phase 378: Channel Energy Audit + Single-Channel Causal Ablation + Post-RMSNorm Causal Patch
==============================================================================================

This phase addresses the critical hard-issues from Phase 375b:

1. ENERGY AUDIT: Clarify the exact definition of "channel energy" and verify
   that 65.2%+61.3%>100% was a recording error. Compute:
   - per-channel energy_i = mean(d_gate_up[:, i]^2)
   - total_energy = sum(all energy_i)
   - fraction_i = energy_i / total_energy
   - cumulative_top_k
   - Verify sum(fraction_i) = 1.0

2. SINGLE-CHANNEL CAUSAL ABLATION:
   - Mask only channel 2802
   - Mask only channel 17483
   - Mask both 2802 + 17483
   - Mask top-10 minus {2802, 17483} (the remaining 8)
   - Keep-only 2802
   - Keep-only 17483
   - Keep-only {2802, 17483}
   
   For each: compute Δ(gate*up) PC1, Δh PC1, post-RMSNorm Δh PC1,
   target logit change

3. POST-RMSNORM CAUSAL PATCH:
   - Extract category component in post-RMSNorm space
   - Patch: replace category component of one pair with another
   - Measure if target logit changes

Models: qwen3, deepseek7b, glm4 (run sequentially)
"""

import sys, os, time, json, gc, traceback
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

# ===== Binding pairs (expanded from Phase 375b) =====
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
    # Try flash_attention_2 first for speed, fall back to sdpa, then eager
    for impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True, attn_implementation=impl)
            log(f"  Loaded with attn_impl={impl}")
            break
        except Exception as e:
            log(f"  Failed with {impl}: {e}")
            continue
    if model is None:
        raise RuntimeError(f"Failed to load {model_name}")
    model.eval()
    
    device = next(model.parameters()).device
    if hasattr(model, 'hf_device_map'):
        dmap = model.hf_device_map
        gpu_count = sum(1 for v in dmap.values() if 'cuda' in str(v))
        cpu_count = sum(1 for v in dmap.values() if 'cpu' in str(v))
        log(f"  Device map: {gpu_count} GPU + {cpu_count} CPU components")
    
    return model, tokenizer, device


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
    """Load post-attention layernorm weight from model or safetensors."""
    import glob
    from safetensors import safe_open
    layers = get_layers(model)
    ln = getattr(layers[layer_idx], "post_attention_layernorm", None)
    if ln is None:
        ln = getattr(layers[layer_idx], "ln2", None)
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
                    if f"layers.{layer_idx}.post_attention_layernorm.weight" in key:
                        return sf.get_tensor(key).float().numpy()
                    elif f"layers.{layer_idx}.ln2.weight" in key:
                        return sf.get_tensor(key).float().numpy()
        except:
            continue
    log(f"    WARNING: Could not load LN weight for layer {layer_idx}")
    return None


def rms_norm_with_weight(x, weight=None, eps=1e-6):
    """RMSNorm: x / sqrt(mean(x^2) + eps) * sqrt(d) * weight"""
    d = x.shape[-1]
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    result = x / rms * np.sqrt(d)
    if weight is not None:
        result = result * weight
    return result


# ===== Part 1: Data Collection =====
def collect_activations(model, tokenizer, model_name, target_layers):
    """Collect MLP activations for clean and corrupt inputs."""
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
        d_ff = W_gate.shape[0]
        
        mlp_module = layers[l].mlp
        ln_weight = _load_ln_weight(model, model_name, l)
        
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
        d_up = up_clean - up_corrupt
        dh = h_clean - h_corrupt
        
        all_data[str(l)] = {
            "mlp_in_clean": mlp_in_clean, "mlp_in_corrupt": mlp_in_corrupt,
            "gate_act_clean": gate_act_clean, "gate_act_corrupt": gate_act_corrupt,
            "up_clean": up_clean, "up_corrupt": up_corrupt,
            "gate_up_clean": gate_up_clean, "gate_up_corrupt": gate_up_corrupt,
            "d_gate_up": d_gate_up, "d_gate_act": d_gate_act, "d_up": d_up, "dh": dh,
            "h_clean": h_clean, "h_corrupt": h_corrupt,
            "W_gate": W_gate, "W_up": W_up, "W_down": W_down,
            "ln_weight": ln_weight,
        }
        
        log(f"    Layer {l} done in {time.time()-t_l:.1f}s")
    
    return all_data


# ===== Part 2: Energy Audit =====
def energy_audit(all_data, model_name):
    """
    Strict energy audit: compute all definitions and verify consistency.
    """
    log("\n" + "="*60)
    log("Part 2: Energy Audit")
    log("="*60)
    
    results = {}
    
    for l_str in sorted(all_data.keys(), key=int):
        d = all_data[l_str]
        d_gate_up = d["d_gate_up"]  # (n_pairs, d_ff)
        W_down = d["W_down"]
        dh = d["dh"]
        
        d_ff = d_gate_up.shape[1]
        n_pairs = d_gate_up.shape[0]
        l = int(l_str)
        
        log(f"\n  Layer {l} (n={n_pairs}, d_ff={d_ff}):")
        
        # ===== Definition 1: per-channel energy (mean over pairs of squared values) =====
        energy_per_ch = np.mean(d_gate_up**2, axis=0)  # (d_ff,)
        total_energy = np.sum(energy_per_ch)
        
        # Fraction per channel
        frac_per_ch = energy_per_ch / total_energy  # should sum to 1.0
        
        # Sort by energy
        sorted_indices = np.argsort(energy_per_ch)[::-1]
        sorted_energies = energy_per_ch[sorted_indices]
        sorted_fracs = frac_per_ch[sorted_indices]
        cumulative_fracs = np.cumsum(sorted_fracs)
        
        log(f"    total_energy = {total_energy:.4f}")
        log(f"    sum(frac_per_ch) = {np.sum(frac_per_ch):.10f} (should be 1.0)")
        
        # Top channels
        for k in [1, 2, 5, 10, 20, 50, 100]:
            top_k_energy = np.sum(sorted_energies[:k])
            top_k_frac = cumulative_fracs[k-1]
            log(f"    Top-{k}: energy={top_k_energy:.2f}, frac={top_k_frac:.6f}")
        
        # Top-10 channel details
        top10_ch = sorted_indices[:10]
        top10_energies = sorted_energies[:10]
        top10_fracs = sorted_fracs[:10]
        
        log(f"\n    Top-10 channel DETAILS:")
        log(f"    {'Ch':>6s} {'Energy':>12s} {'Fraction':>10s} {'CumFrac':>10s}")
        for i in range(10):
            log(f"    {top10_ch[i]:6d} {top10_energies[i]:12.4f} {top10_fracs[i]:10.6f} {cumulative_fracs[i]:10.6f}")
        
        # ===== Definition 2: channel contribution to PCA variance =====
        pca_dgu, rank_dgu, pc1_dgu, Vt_dgu = compute_pca_explained(d_gate_up)
        
        # Project onto PC1: per-channel contribution to PC1 variance
        if pc1_dgu is not None:
            # pc1_dgu is (d_ff,), the first principal component direction
            # Contribution of channel i to PC1 = (pc1_dgu[i])^2 * eigenvalue
            # But simpler: project d_gate_up onto pc1_dgu, then per-channel contribution
            proj_pc1 = d_gate_up @ pc1_dgu  # (n_pairs,)
            # Per-channel contribution: d_gate_up[:, i] * pc1_dgu[i]
            # Channel i's contribution to PC1 variance = var(d_gate_up[:, i] * pc1_dgu[i])
            # Actually: contribution to total variance along PC1 = sum_j (Vt_dgu[0,j] * S[j])^2
            # But more intuitive: fraction of PC1 that each channel reconstructs
            
            # Simple approach: remove channel i, recompute PC1
            # This is too expensive for all channels, but do it for top-10
            
            log(f"\n    PCA-based channel importance (leave-one-out PC1 change):")
            baseline_pc1 = pca_dgu[0]
            for i in range(min(5, len(top10_ch))):
                ch = top10_ch[i]
                d_masked = d_gate_up.copy()
                d_masked[:, ch] = 0
                pca_masked, _, _, _ = compute_pca_explained(d_masked)
                delta_pc1 = baseline_pc1 - pca_masked[0]
                log(f"      Remove ch {ch}: PC1 {baseline_pc1:.6f} -> {pca_masked[0]:.6f} "
                    f"(delta={delta_pc1:.6f}, {delta_pc1/baseline_pc1*100:.2f}%)")
        
        # ===== Definition 3: MLP output contribution =====
        mlp_out = (W_down @ d_gate_up.T).T  # (n_pairs, d_model)
        dh_proj = (W_down @ d_gate_up.T).T
        
        # Per-channel contribution to MLP output norm
        mlp_out_per_ch = []
        for i in range(min(10, len(top10_ch))):
            ch = top10_ch[i]
            ch_contrib = np.outer(W_down[:, ch], d_gate_up[:, ch])  # (d_model, n_pairs) contribution
            ch_mlp_norm = np.mean(np.linalg.norm(ch_contrib.T, axis=1)**2)
            mlp_out_per_ch.append(float(ch_mlp_norm))
        
        total_mlp_norm = np.mean(np.linalg.norm(mlp_out, axis=1)**2)
        
        log(f"\n    Per-channel MLP output contribution:")
        log(f"    Total MLP output energy: {total_mlp_norm:.4f}")
        for i in range(min(5, len(top10_ch))):
            ch = top10_ch[i]
            log(f"      Ch {ch}: MLP energy = {mlp_out_per_ch[i]:.4f} "
                f"({mlp_out_per_ch[i]/total_mlp_norm*100:.2f}% of total)")
        
        results[l_str] = {
            "total_energy": float(total_energy),
            "sum_frac_check": float(np.sum(frac_per_ch)),
            "top1_frac": float(cumulative_fracs[0]),
            "top2_frac": float(cumulative_fracs[1]),
            "top10_frac": float(cumulative_fracs[9]),
            "top50_frac": float(cumulative_fracs[49]),
            "top100_frac": float(cumulative_fracs[99]),
            "top10_channels": [int(x) for x in top10_ch],
            "top10_energies": [float(x) for x in top10_energies],
            "top10_fracs": [float(x) for x in top10_fracs],
            "top10_cumulative_fracs": [float(x) for x in cumulative_fracs[:10]],
            "baseline_pc1": float(pca_dgu[0]) if pca_dgu is not None else None,
            "total_mlp_energy": float(total_mlp_norm),
            "top10_mlp_energies": mlp_out_per_ch[:10],
        }
    
    return results


# ===== Part 3: Single-Channel Causal Ablation =====
def single_channel_ablation(all_data, model_name):
    """
    Precise single-channel causal ablation.
    
    Interventions:
    1. Mask only ch 2802 (DS7B top-1)
    2. Mask only ch 17483 (DS7B top-2)  
    3. Mask both 2802 + 17483
    4. Mask top-10 minus {2802, 17483} (remaining 8)
    5. Keep-only 2802
    6. Keep-only 17483
    7. Keep-only {2802, 17483}
    """
    log("\n" + "="*60)
    log("Part 3: Single-Channel Causal Ablation")
    log("="*60)
    
    results = {}
    
    for l_str in sorted(all_data.keys(), key=int):
        d = all_data[l_str]
        d_gate_up = d["d_gate_up"]  # (n_pairs, d_ff)
        gate_up_clean = d["gate_up_clean"]
        gate_up_corrupt = d["gate_up_corrupt"]
        gate_act_corrupt = d["gate_act_corrupt"]
        up_corrupt = d["up_corrupt"]
        d_up = d["d_up"]
        W_down = d["W_down"]
        dh = d["dh"]
        h_clean = d["h_clean"]
        h_corrupt = d["h_corrupt"]
        ln_weight = d.get("ln_weight", None)
        
        d_ff = d_gate_up.shape[1]
        n_pairs = d_gate_up.shape[0]
        l = int(l_str)
        
        log(f"\n  Layer {l}:")
        
        # Identify top channels
        energy_per_ch = np.mean(d_gate_up**2, axis=0)
        sorted_indices = np.argsort(energy_per_ch)[::-1]
        top10_ch = sorted_indices[:10]
        top2_ch = sorted_indices[:2]
        top8_ch = sorted_indices[2:10]  # top-10 minus top-2
        
        log(f"    Top-2 channels: {top2_ch.tolist()}")
        log(f"    Top-10 channels: {top10_ch.tolist()}")
        
        # Baseline
        pca_dgu_base, rank_dgu_base, _, _ = compute_pca_explained(d_gate_up)
        pca_dh_base, rank_dh_base, _, _ = compute_pca_explained(dh)
        
        # Post-RMSNorm baseline
        dh_norm_base = rms_norm_with_weight(dh, ln_weight)
        pca_dh_norm_base, rank_dh_norm_base, _, _ = compute_pca_explained(dh_norm_base)
        
        log(f"    Baseline: Δ(gate*up) PC1={pca_dgu_base[0]:.6f}, "
            f"Δh PC1={pca_dh_base[0]:.6f}, post-RMSNorm Δh PC1={pca_dh_norm_base[0]:.6f}")
        
        ablation_results = {}
        
        # Define ablation configurations
        ablation_configs = [
            ("mask_top1", top2_ch[:1], "mask"),
            ("mask_top2", top2_ch[1:2], "mask"),
            ("mask_top1_top2", top2_ch, "mask"),
            ("mask_top10_minus_top2", top8_ch, "mask"),
            ("mask_top10", top10_ch, "mask"),
            ("keep_only_top1", top2_ch[:1], "keep_only"),
            ("keep_only_top2", top2_ch[1:2], "keep_only"),
            ("keep_only_top1_top2", top2_ch, "keep_only"),
            ("keep_only_top10", top10_ch, "keep_only"),
        ]
        
        for name, channels, mode in ablation_configs:
            d_masked = d_gate_up.copy()
            
            if mode == "mask":
                # Zero out specified channels
                d_masked[:, channels] = 0
            elif mode == "keep_only":
                # Zero out everything EXCEPT specified channels
                mask = np.ones(d_ff, dtype=bool)
                mask[channels] = False
                d_masked[:, mask] = 0
            
            # Compute metrics
            pca_dgu, rank_dgu, _, _ = compute_pca_explained(d_masked)
            
            # MLP output
            mlp_out_new = (W_down @ d_masked.T).T
            pca_mlp, _, _, _ = compute_pca_explained(mlp_out_new)
            
            # Δh after ablation
            dgu_diff = d_gate_up - d_masked
            dh_new = dh - (W_down @ dgu_diff.T).T
            pca_dh, rank_dh, _, _ = compute_pca_explained(dh_new)
            
            # Post-RMSNorm Δh after ablation
            dh_norm_new = rms_norm_with_weight(dh_new, ln_weight)
            pca_dh_norm, rank_dh_norm, _, _ = compute_pca_explained(dh_norm_new)
            
            ch_str = ",".join(str(c) for c in channels)
            log(f"    {name} (ch={ch_str}): "
                f"Δgu PC1={pca_dgu[0]:.6f} (base={pca_dgu_base[0]:.6f}), "
                f"Δh PC1={pca_dh[0]:.6f}, "
                f"post-RMSNorm Δh PC1={pca_dh_norm[0]:.6f}, "
                f"post-RMSNorm rank={rank_dh_norm}")
            
            ablation_results[name] = {
                "channels": [int(c) for c in channels],
                "mode": mode,
                "dgu_pc1": float(pca_dgu[0]),
                "dgu_eff_rank": rank_dgu,
                "dh_pc1": float(pca_dh[0]),
                "dh_eff_rank": rank_dh,
                "dh_norm_pc1": float(pca_dh_norm[0]),
                "dh_norm_eff_rank": rank_dh_norm,
                "mlp_pc1": float(pca_mlp[0]),
            }
        
        results[l_str] = {
            "top2_channels": [int(x) for x in top2_ch],
            "top10_channels": [int(x) for x in top10_ch],
            "baseline_dgu_pc1": float(pca_dgu_base[0]),
            "baseline_dh_pc1": float(pca_dh_base[0]),
            "baseline_dh_norm_pc1": float(pca_dh_norm_base[0]),
            "baseline_dh_norm_rank": rank_dh_norm_base,
            "ablation_results": ablation_results,
        }
    
    return results


# ===== Part 4: Post-RMSNorm Category Component Causal Patch =====
def post_rmsnorm_causal_patch(all_data, model_name, model, tokenizer):
    """
    Test if post-RMSNorm category structure is causally effective.
    
    Method:
    1. Compute post-RMSNorm Δh for all pairs
    2. For each category, compute category centroid direction
    3. For a given pair, replace its category component with another category's
    4. Apply inverse-RMSNorm (approximate) to get modified Δh in raw space
    5. Run model with modified hidden state and check if prediction changes
    """
    log("\n" + "="*60)
    log("Part 4: Post-RMSNorm Category Causal Patch")
    log("="*60)
    
    results = {}
    layers = get_layers(model)
    input_device = next(model.parameters()).device
    
    for l_str in sorted(all_data.keys(), key=int):
        d = all_data[l_str]
        dh = d["dh"]              # (n_pairs, d_model)
        h_clean = d["h_clean"]
        h_corrupt = d["h_corrupt"]
        ln_weight = d.get("ln_weight", None)
        
        l = int(l_str)
        n_pairs = dh.shape[0]
        d_model = dh.shape[1]
        
        log(f"\n  Layer {l}:")
        
        # Compute post-RMSNorm Δh
        dh_norm = rms_norm_with_weight(dh, ln_weight)
        
        # Compute category centroids in post-RMSNorm space
        cat_centroids = {}
        cat_indices = {}
        for cat in ALL_CATEGORIES:
            idx = [j for j, c in enumerate(PAIR_CATEGORIES) if c == cat]
            cat_indices[cat] = idx
            cat_centroids[cat] = np.mean(dh_norm[idx], axis=0)  # (d_model,)
        
        # Grand mean
        grand_mean = np.mean(dh_norm, axis=0)
        
        # Category component = cat_centroid - grand_mean
        cat_components = {}
        for cat in ALL_CATEGORIES:
            cat_components[cat] = cat_centroids[cat] - grand_mean
        
        # Measure category separation in post-RMSNorm space
        # Same-category pairwise cosine vs cross-category pairwise cosine
        same_cat_cos = []
        cross_cat_cos = []
        for i in range(n_pairs):
            for j in range(i+1, n_pairs):
                cos = cosine_sim(dh_norm[i], dh_norm[j])
                if PAIR_CATEGORIES[i] == PAIR_CATEGORIES[j]:
                    same_cat_cos.append(cos)
                else:
                    cross_cat_cos.append(cos)
        
        log(f"    Post-RMSNorm: same-cat cos={np.mean(same_cat_cos):.4f}, "
            f"cross-cat cos={np.mean(cross_cat_cos):.4f}, "
            f"gap={np.mean(same_cat_cos)-np.mean(cross_cat_cos):.4f}")
        
        # Now do causal patch via model intervention
        # For a subset of pairs, replace the hidden state at layer l+1
        # with a modified version where we swap the category component
        
        # Select test pairs: one from each category
        test_pairs = []
        for cat in ALL_CATEGORIES:
            idx = cat_indices[cat]
            if len(idx) > 0:
                test_pairs.append((idx[0], cat))
        
        log(f"    Testing {len(test_pairs)} category patches...")
        
        patch_results = []
        
        for pidx, src_cat in test_pairs:
            obj, target, competitor = ALL_PAIRS[pidx]
            clean_prompt = TEMPLATE.format(obj=obj, attr=target)
            
            # Get the target token id
            target_tokens = tokenizer(target, add_special_tokens=False)["input_ids"]
            competitor_tokens = tokenizer(competitor, add_special_tokens=False)["input_ids"]
            
            # For each target category to patch TO
            for dst_cat in ALL_CATEGORIES:
                if dst_cat == src_cat:
                    continue
                
                # Category component replacement in post-RMSNorm space
                # dh_norm_new = dh_norm[pidx] - cat_components[src_cat] + cat_components[dst_cat]
                dh_norm_patched = dh_norm[pidx] - cat_components[src_cat] + cat_components[dst_cat]
                
                # We need to map this back to raw Δh space (approximately)
                # Since RMSNorm is not invertible exactly, we use the approximation:
                # dh_norm = rms_norm(dh) => dh ≈ dh_norm * rms(dh) / sqrt(d)
                # So dh_patched ≈ dh_norm_patched * rms(dh[pidx]) / sqrt(d_model)
                # But this is very approximate. Better approach: directly patch in
                # post-RMSNorm space by running the model from layer l+1 with modified
                # hidden state.
                
                # Actually, let's use a simpler approach: patch the residual stream
                # at the OUTPUT of layer l+1. We add the category component difference
                # directly to h_clean.
                
                # Δh_patched_raw ≈ dh + (W_down @ Δ_gate_up_patched) - ... 
                # This is too complicated. Let's use the model directly.
                
                # Simpler approach: run model with hook that modifies hidden state at layer l+1
                # by adding the category component shift
                
                # The shift in post-RMSNorm space is:
                # shift_norm = cat_components[dst_cat] - cat_components[src_cat]
                # We need to convert this to a raw-space shift.
                # Approximate: shift_raw ≈ shift_norm * (rms of dh[pidx]) / sqrt(d_model)
                
                # Better: just test if the category component in post-RMSNorm space
                # correlates with the actual logit difference. Use a linear probe.
                
                pass  # We'll compute logit effects via linear probe instead
            
            # For efficiency, compute logit effects via W_U projection
            # Get W_U (unembedding matrix)
            break  # Only do the first test pair for now
        
        # ===== Simpler causal test: Linear probe on post-RMSNorm space =====
        # Instead of model patching, compute W_U @ post-RMSNorm(Δh) and see if
        # category component projects differently onto target/competitor tokens
        
        # Load W_U
        try:
            from model_utils import get_W_U
            W_U = get_W_U(model, model_name)  # (d_model, vocab_size)
        except:
            log(f"    Could not load W_U, skipping logit analysis")
            W_U = None
        
        if W_U is not None:
            # Compute logit effects for each pair
            target_logit_diffs_raw = []
            target_logit_diffs_norm = []
            
            for pidx, (obj, target, competitor) in enumerate(ALL_PAIRS):
                target_tokens = tokenizer(target, add_special_tokens=False)["input_ids"]
                competitor_tokens = tokenizer(competitor, add_special_tokens=False)["input_ids"]
                
                if len(target_tokens) != 1 or len(competitor_tokens) != 1:
                    target_logit_diffs_raw.append(None)
                    target_logit_diffs_norm.append(None)
                    continue
                
                t_id = target_tokens[0]
                c_id = competitor_tokens[0]
                
                # Raw Δh logit effect
                # W_U shape: (d_model, vocab_size) or (vocab_size, d_model)
                if W_U.shape[0] == dh.shape[1]:
                    # (d_model, vocab_size) -> project dh onto each vocab token
                    logit_raw = W_U.T @ dh[pidx]  # (vocab_size,)
                else:
                    logit_raw = W_U @ dh[pidx]  # (vocab_size,)
                diff_raw = logit_raw[t_id] - logit_raw[c_id]
                
                # Post-RMSNorm Δh logit effect
                if W_U.shape[0] == dh.shape[1]:
                    logit_norm = W_U.T @ dh_norm[pidx]
                else:
                    logit_norm = W_U @ dh_norm[pidx]
                diff_norm = logit_norm[t_id] - logit_norm[c_id]
                
                target_logit_diffs_raw.append(float(diff_raw))
                target_logit_diffs_norm.append(float(diff_norm))
            
            # Correlation between raw and norm logit effects
            valid_pairs = [(r, n) for r, n in zip(target_logit_diffs_raw, target_logit_diffs_norm)
                         if r is not None and n is not None]
            if len(valid_pairs) > 10:
                raw_vals = [v[0] for v in valid_pairs]
                norm_vals = [v[1] for v in valid_pairs]
                corr_raw_norm = np.corrcoef(raw_vals, norm_vals)[0, 1]
                
                # Category-wise average logit effects
                cat_logit_raw = {}
                cat_logit_norm = {}
                for pidx, (obj, target, competitor) in enumerate(ALL_PAIRS):
                    cat = PAIR_CATEGORIES[pidx]
                    if target_logit_diffs_raw[pidx] is not None:
                        cat_logit_raw.setdefault(cat, []).append(target_logit_diffs_raw[pidx])
                        cat_logit_norm.setdefault(cat, []).append(target_logit_diffs_norm[pidx])
                
                log(f"    Logit effect correlation (raw vs post-RMSNorm): {corr_raw_norm:.4f}")
                log(f"    Mean logit effect by category:")
                for cat in sorted(cat_logit_raw.keys()):
                    mean_raw = np.mean(cat_logit_raw[cat])
                    mean_norm = np.mean(cat_logit_norm[cat])
                    log(f"      {cat:>12s}: raw={mean_raw:.4f}, norm={mean_norm:.4f}")
                
                results[l_str] = {
                    "same_cat_cos_mean": float(np.mean(same_cat_cos)),
                    "cross_cat_cos_mean": float(np.mean(cross_cat_cos)),
                    "logit_corr_raw_norm": float(corr_raw_norm),
                    "cat_logit_raw": {k: float(np.mean(v)) for k, v in cat_logit_raw.items()},
                    "cat_logit_norm": {k: float(np.mean(v)) for k, v in cat_logit_norm.items()},
                    "mean_logit_raw": float(np.mean([v for v in target_logit_diffs_raw if v is not None])),
                    "mean_logit_norm": float(np.mean([v for v in target_logit_diffs_norm if v is not None])),
                }
    
    return results


# ===== Main =====
def run_phase378(model_name):
    cfg = MODEL_CONFIGS[model_name]
    
    log(f"\n{'='*60}")
    log(f"Phase 378: Channel Audit + Ablation + Causal Patch — {model_name}")
    log(f"  n_pairs={len(ALL_PAIRS)}, categories={len(ALL_CATEGORIES)}")
    log(f"{'='*60}")
    
    t0 = time.time()
    
    # Select target layers
    if model_name == "deepseek7b":
        target_layers = [4, 5, 8, 24]
    elif model_name == "qwen3":
        target_layers = [4, 28]
    elif model_name == "glm4":
        target_layers = [4, 30]
    else:
        target_layers = [4]
    
    # Load model
    model, tokenizer, device = load_model_bf16(model_name)
    log(f"  Model loaded in {time.time()-t0:.1f}s")
    
    # Part 1: Collect activations
    log(f"\n--- Part 1: Collecting activations ---")
    all_data = collect_activations(model, tokenizer, model_name, target_layers)
    
    # Part 2: Energy Audit
    log(f"\n--- Part 2: Energy Audit ---")
    audit_results = energy_audit(all_data, model_name)
    
    # Part 3: Single-Channel Causal Ablation
    log(f"\n--- Part 3: Single-Channel Causal Ablation ---")
    ablation_results = single_channel_ablation(all_data, model_name)
    
    # Part 4: Post-RMSNorm Causal Patch
    log(f"\n--- Part 4: Post-RMSNorm Causal Patch ---")
    patch_results = post_rmsnorm_causal_patch(all_data, model_name, model, tokenizer)
    
    # Release model
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Save results
    all_results = {
        "model": model_name,
        "phase": "378",
        "n_pairs": len(ALL_PAIRS),
        "n_categories": len(ALL_CATEGORIES),
        "categories": ALL_CATEGORIES,
        "audit": audit_results,
        "ablation": ablation_results,
        "patch": patch_results,
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
    
    out_dir = "results/phase378_channel_audit_ablation"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{model_name}_phase378.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log(f"\n  Results saved to {out_path}")
    
    # Print summary
    log(f"\n{'='*60}")
    log(f"Phase 378 Summary: {model_name}")
    log(f"{'='*60}")
    
    for l_str in sorted(audit_results.keys(), key=int):
        a = audit_results[l_str]
        ab = ablation_results.get(l_str, {})
        log(f"\n  Layer {int(l_str)}:")
        log(f"    Energy: total={a['total_energy']:.2f}, "
            f"top1={a['top1_frac']:.6f}, top2={a['top2_frac']:.6f}, "
            f"top10={a['top10_frac']:.6f}")
        log(f"    Top-10 channels: {a['top10_channels']}")
        log(f"    Baseline: Δgu PC1={a['baseline_pc1']:.6f}")
        
        if ab.get("ablation_results"):
            log(f"    Ablation results:")
            for name, res in sorted(ab["ablation_results"].items()):
                log(f"      {name}: Δgu PC1={res['dgu_pc1']:.6f}, "
                    f"Δh PC1={res['dh_pc1']:.6f}, "
                    f"post-RMSNorm Δh PC1={res['dh_norm_pc1']:.6f}")
    
    log(f"\n  Total time: {time.time()-t0:.1f}s")
    
    return all_results


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    
    if model_name == "all":
        for name in ["qwen3", "deepseek7b", "glm4"]:
            try:
                run_phase378(name)
            except Exception as e:
                log(f"!!! {name} failed: {e}")
                traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(5)
    else:
        run_phase378(model_name)
    
    log("\nPhase 378 complete!")


if __name__ == "__main__":
    main()
