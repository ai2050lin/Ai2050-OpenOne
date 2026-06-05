"""
Phase 379: RMSNorm信息重映射机制审计
=====================================

核心问题：RMSNorm到底对信号做了什么变换？

关键修正（来自Phase 378严格审视）：
1. RMSNorm作用在完整residual state (h_clean, h_corrupt)，不是Δh
2. post-RMSNorm Δh = RMSNorm(h_clean) - RMSNorm(h_corrupt) ≠ RMSNorm(Δh)
3. 需要分别对clean/corrupt状态计算RMSNorm

实验设计：
- 对5种条件（基线/仅保留top2/去掉top2/仅保留top10/去掉top10）
- 计算真实post-RMSNorm Δh = RMSNorm(h_clean_ablated) - RMSNorm(h_corrupt_ablated)
- 对比伪post-RMSNorm Δh = RMSNorm(Δh_ablated)（这是Phase 378的错误做法）
- 分析PC1/PC2+语义解码
- 回归PC score与已知变量

目标层：
- DS7B: L4 (核心), L5, L8, L24 (深层对比)
- Qwen3: L4, L28
- GLM4: L4, L30

用法:
  python tests/glm5/phase379_rmsnorm_remapping_audit.py qwen3
  python tests/glm5/phase379_rmsnorm_remapping_audit.py deepseek7b
  python tests/glm5/phase379_rmsnorm_remapping_audit.py glm4
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


# ===== Binding pairs (same as Phase 378) =====
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


# ===== Math utilities =====
def _silu(x):
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -50, 50))))

def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

def compute_pca_full(X):
    """Return explained variance ratios, effective rank, all principal components, and scores."""
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
    scores = U * S  # (n, d) — projection of centered data onto PCs
    return explained, eff_rank, Vt, scores, S

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


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


# ===== RMSNorm computation =====
def rms_norm_single(x, weight=None, eps=1e-6):
    """Apply RMSNorm to a single vector x (d_model,)."""
    d = x.shape[-1]
    rms = np.sqrt(np.mean(x**2) + eps)
    result = x / rms * np.sqrt(d)
    if weight is not None:
        result = result * weight
    return result


# ===== Data collection with proper residual states =====
def collect_residual_states(model, tokenizer, model_name, target_layers):
    """
    Collect complete residual states for clean/corrupt inputs.
    This is the key difference from Phase 378:
    We need h_clean and h_corrupt (full residual stream), not just Δh.
    """
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
        d_model = W_gate.shape[1]
        
        mlp_module = layers[l].mlp
        ln_weight = _load_ln_weight(model, model_name, l)
        
        # We need: pre-MLP residual (before layer L's MLP), post-layer residual (after layer L)
        # For proper RMSNorm: RMSNorm acts on pre-MLP residual to produce MLP input
        # So: mlp_input = RMSNorm(h_pre_mlp)
        #     mlp_output = W_down @ (act(h_pre_mlp_norm @ W_gate.T) * (h_pre_mlp_norm @ W_up.T))
        #     h_post_layer = h_pre_mlp + mlp_output  (residual connection)
        
        # Collect h_pre_mlp (residual before layer L's MLP = after layer L's attention + LN)
        # and h_post_layer (residual after layer L = used as input to layer L+1)
        # h_post_layer = h_post_attn + mlp_output = h_pre_mlp + W_down @ gate_up
        
        pre_mlp_clean_list = []
        pre_mlp_corrupt_list = []
        h_post_clean_list = []
        h_post_corrupt_list = []
        
        for pidx, (obj, target, competitor) in enumerate(ALL_PAIRS):
            if pidx % 30 == 0:
                log(f"    Pair {pidx+1}/{n_pairs}")
            
            clean_prompt = TEMPLATE.format(obj=obj, attr=target)
            corrupt_prompt = TEMPLATE.format(obj=CORRUPTED_BASELINE, attr=target)
            
            # Hook to capture pre-MLP input (after LN, before MLP)
            captured = {}
            def mlp_input_hook(module, input, output=None):
                captured["mlp_input"] = input[0].detach().cpu().float()
            
            h_hook = mlp_module.register_forward_pre_hook(mlp_input_hook)
            
            # Clean run
            with torch.no_grad():
                clean_out = model(
                    input_ids=tokenizer(clean_prompt, return_tensors="pt",
                                       truncation=True, max_length=64)["input_ids"].to(input_device),
                    attention_mask=tokenizer(clean_prompt, return_tensors="pt",
                                            truncation=True, max_length=64)["attention_mask"].to(input_device),
                    output_hidden_states=True)
            
            last_pos = tokenizer(clean_prompt, return_tensors="pt")["input_ids"].shape[1] - 1
            pre_mlp_clean = captured["mlp_input"][0, last_pos].numpy()
            # h_post_layer = hidden_states[l+1] (after this layer)
            h_post_clean = clean_out.hidden_states[l+1][0, last_pos].detach().cpu().float().numpy()
            pre_mlp_clean_list.append(pre_mlp_clean)
            h_post_clean_list.append(h_post_clean)
            
            del clean_out
            captured.clear()
            
            # Corrupt run
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
        
        pre_mlp_clean = np.array(pre_mlp_clean_list)     # (n, d_model) — MLP input after RMSNorm
        pre_mlp_corrupt = np.array(pre_mlp_corrupt_list)  # (n, d_model)
        h_post_clean = np.array(h_post_clean_list)        # (n, d_model) — residual after this layer
        h_post_corrupt = np.array(h_post_corrupt_list)    # (n, d_model)
        
        # Compute Δh at this layer
        dh = h_post_clean - h_post_corrupt
        
        # Compute gate_up in d_ff space (from MLP input, which is post-RMSNorm)
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
        
        # Also need the pre-RMSNorm residual (before LN, before MLP)
        # h_pre_mlp = mlp_input * (rms / sqrt(d)) / weight  (inverse RMSNorm)
        # But this is tricky. Instead, we can compute:
        # The residual stream BEFORE layer L's MLP is h_post_attn = hidden_states[l] + attn_output
        # But we captured hidden_states[l+1] which is AFTER this layer.
        # We need hidden_states[l] (before this layer) to get h_pre_ln.
        # Actually, h_post_clean already = h_pre_mlp + W_down @ gate_up_clean
        # So h_pre_mlp = h_post_clean - W_down @ gate_up_clean
        # Wait, that's h_post_attn (post-attention residual), which is what goes into LN before MLP.
        
        # For RMSNorm: the input to the MLP IS the post-LN version of h_post_attn.
        # We captured mlp_input = LN(h_post_attn)
        # To get h_post_attn = inverse_LN(mlp_input) ≈ mlp_input * rms_raw / sqrt(d)
        
        # Actually, what we really need for the Phase 379 experiment is:
        # Given modified gate_up (from ablation), compute:
        #   h_post_modified = h_post_attn + W_down @ gate_up_modified
        # Then: post_ln_modified = LN(h_post_modified)
        # But wait — the LN in question is the NEXT layer's input LN, not this layer's.
        # 
        # The residual flow at layer L:
        #   h_pre_attn → (attention) → h_post_attn → LN2 → mlp_input → (MLP) → mlp_output
        #   h_post_layer = h_post_attn + mlp_output
        #
        # Then at layer L+1:
        #   LN1(h_post_layer) → attention input for L+1
        #
        # So "post-RMSNorm Δh" in Phase 378 context means:
        #   LN(h_post_layer_clean) - LN(h_post_layer_corrupt)
        # where LN is the input layernorm of the NEXT layer (L+1).
        #
        # But Phase 378 computed it as: RMSNorm(Δh) where Δh = h_post_layer_clean - h_post_layer_corrupt
        # which is WRONG because RMSNorm is not linear.
        
        # For Phase 379, we need to compute the proper post-RMSNorm Δh.
        # We need: h_post_attn (pre-LN residual before MLP) to reconstruct h_post_layer.
        
        # h_post_attn = mlp_input / LN_transform  (inverse is approximate)
        # Better: we can use hidden_states[l] which is the residual before this layer's attention.
        # But h_post_attn = hidden_states[l] + attn_output (we don't have attn_output separately).
        
        # Simplest approach: we already have h_post_clean and h_post_corrupt.
        # h_post = h_post_attn + mlp_output
        # mlp_output = W_down @ gate_up
        # So h_post_attn = h_post - W_down @ gate_up
        
        # But wait: h_post_clean = h_post_attn_clean + W_down @ gate_up_clean
        # And h_post_corrupt = h_post_attn_corrupt + W_down @ gate_up_corrupt
        
        # The issue: h_post_attn_clean ≠ h_post_attn_corrupt (because attention output differs)
        # But for ablation, we modify gate_up, which only affects mlp_output:
        # h_post_ablated = h_post_attn + W_down @ gate_up_ablated
        
        # For clean: h_post_attn_clean = h_post_clean - W_down @ gate_up_clean
        # For corrupt: h_post_attn_corrupt = h_post_corrupt - W_down @ gate_up_corrupt
        
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
            "d_model": d_model,
        }
        
        log(f"    Layer {l} done in {time.time()-t_l:.1f}s")
    
    return all_data


# ===== Part 1: Proper vs Pseudo post-RMSNorm comparison =====
def compare_proper_vs_pseudo_rmsnorm(all_data, model_name):
    """
    The CORE experiment of Phase 379.
    
    Compare:
    - PROPER: post_RMSNorm_Δh = RMSNorm(h_clean_ablated) - RMSNorm(h_corrupt_ablated)
    - PSEUDO (Phase 378 style): RMSNorm(Δh_ablated)
    
    Under different ablation conditions.
    """
    log("\n" + "="*60)
    log("Part 1: Proper vs Pseudo post-RMSNorm Comparison")
    log("="*60)
    
    results = {}
    
    # Load next-layer LN weights for proper RMSNorm
    # The LN that processes h_post_layer is the next layer's input_layernorm
    # But for simplicity, we use the same RMSNorm as Phase 378 for comparison.
    # Actually, the key point is: Phase 378 computed RMSNorm(Δh), 
    # while the correct computation is RMSNorm(h_clean) - RMSNorm(h_corrupt).
    # Both use the SAME RMSNorm function (with γ weight), but applied differently.
    
    for l_str in sorted(all_data.keys(), key=int):
        d = all_data[l_str]
        W_down = d["W_down"]
        d_ff = W_down.shape[1] if W_down.shape[0] == d["d_gate_up"].shape[1] else W_down.shape[1]
        n_pairs = d["dh"].shape[0]
        d_model = d["dh"].shape[1]
        l = int(l_str)
        
        h_post_attn_clean = d["h_post_attn_clean"]   # (n, d_model)
        h_post_attn_corrupt = d["h_post_attn_corrupt"]  # (n, d_model)
        gate_up_clean = d["gate_up_clean"]             # (n, d_ff)
        gate_up_corrupt = d["gate_up_corrupt"]         # (n, d_ff)
        d_gate_up = d["d_gate_up"]
        ln_weight = d.get("ln_weight", None)
        
        # Identify top channels
        energy_per_ch = np.mean(d_gate_up**2, axis=0)
        sorted_indices = np.argsort(energy_per_ch)[::-1]
        top2_ch = sorted_indices[:2]
        top10_ch = sorted_indices[:10]
        
        log(f"\n  Layer {l}: top2={top2_ch.tolist()}, top10={top10_ch.tolist()}")
        
        # Define ablation conditions
        ablation_configs = [
            ("baseline", None, None),  # no ablation
            ("keep_only_top2", top2_ch, "keep_only"),
            ("mask_top2", top2_ch, "mask"),
            ("keep_only_top10", top10_ch, "keep_only"),
            ("mask_top10", top10_ch, "mask"),
        ]
        
        layer_results = {}
        
        for ablation_name, channels, mode in ablation_configs:
            # Create ablated gate_up
            gate_up_clean_abl = gate_up_clean.copy()
            gate_up_corrupt_abl = gate_up_corrupt.copy()
            
            if channels is not None:
                if mode == "mask":
                    gate_up_clean_abl[:, channels] = 0
                    gate_up_corrupt_abl[:, channels] = 0
                elif mode == "keep_only":
                    mask = np.ones(d_gate_up.shape[1], dtype=bool)
                    mask[channels] = False
                    gate_up_clean_abl[:, mask] = 0
                    gate_up_corrupt_abl[:, mask] = 0
            
            # Compute ablated h_post_layer
            mlp_out_clean_abl = (W_down @ gate_up_clean_abl.T).T   # (n, d_model)
            mlp_out_corrupt_abl = (W_down @ gate_up_corrupt_abl.T).T
            
            h_clean_abl = h_post_attn_clean + mlp_out_clean_abl
            h_corrupt_abl = h_post_attn_corrupt + mlp_out_corrupt_abl
            
            # Δh_ablated
            dh_abl = h_clean_abl - h_corrupt_abl
            
            # === PROPER post-RMSNorm ===
            # Apply RMSNorm to each sample individually
            h_clean_norm = np.zeros_like(h_clean_abl)
            h_corrupt_norm = np.zeros_like(h_corrupt_abl)
            for i in range(n_pairs):
                h_clean_norm[i] = rms_norm_single(h_clean_abl[i], ln_weight)
                h_corrupt_norm[i] = rms_norm_single(h_corrupt_abl[i], ln_weight)
            
            dh_proper = h_clean_norm - h_corrupt_norm  # (n, d_model)
            
            # === PSEUDO post-RMSNorm (Phase 378 style) ===
            dh_pseudo = np.zeros_like(dh_abl)
            for i in range(n_pairs):
                dh_pseudo[i] = rms_norm_single(dh_abl[i], ln_weight)
            
            # === Compare ===
            # PCA
            pca_raw, rank_raw, Vt_raw, scores_raw, _ = compute_pca_full(dh_abl)
            pca_proper, rank_proper, Vt_proper, scores_proper, _ = compute_pca_full(dh_proper)
            pca_pseudo, rank_pseudo, Vt_pseudo, scores_pseudo, _ = compute_pca_full(dh_pseudo)
            
            # Category structure
            def compute_cat_metrics(dh_vec):
                """Compute same-cat cos, cross-cat cos, gap."""
                same_cos = []
                cross_cos = []
                for i in range(n_pairs):
                    for j in range(i+1, n_pairs):
                        cos = cosine_sim(dh_vec[i], dh_vec[j])
                        if PAIR_CATEGORIES[i] == PAIR_CATEGORIES[j]:
                            same_cos.append(cos)
                        else:
                            cross_cos.append(cos)
                return float(np.mean(same_cos)), float(np.mean(cross_cos))
            
            same_raw, cross_raw = compute_cat_metrics(dh_abl)
            same_proper, cross_proper = compute_cat_metrics(dh_proper)
            same_pseudo, cross_pseudo = compute_cat_metrics(dh_pseudo)
            
            # PC1 direction consistency between proper and pseudo
            if Vt_raw is not None and Vt_proper is not None and Vt_pseudo is not None:
                cos_raw_proper = abs(cosine_sim(Vt_raw[0], Vt_proper[0]))
                cos_raw_pseudo = abs(cosine_sim(Vt_raw[0], Vt_pseudo[0]))
                cos_proper_pseudo = abs(cosine_sim(Vt_proper[0], Vt_pseudo[0]))
            else:
                cos_raw_proper = cos_raw_pseudo = cos_proper_pseudo = None
            
            # PC1 score distribution: how many positive vs negative within each category
            if scores_proper is not None:
                pc1_scores_proper = scores_proper[:, 0]
                # Per-category PC1 score statistics
                cat_pc1_stats = {}
                for cat in ALL_CATEGORIES:
                    idx = [j for j, c in enumerate(PAIR_CATEGORIES) if c == cat]
                    if len(idx) > 0:
                        scores_cat = pc1_scores_proper[idx]
                        cat_pc1_stats[cat] = {
                            "mean": float(np.mean(scores_cat)),
                            "std": float(np.std(scores_cat)),
                            "fraction_positive": float(np.mean(scores_cat > 0)),
                        }
            else:
                cat_pc1_stats = {}
            
            log(f"    {ablation_name}:")
            log(f"      raw: PC1={pca_raw[0]:.4f}, rank={rank_raw}, "
                f"same={same_raw:.4f}, cross={cross_raw:.4f}, gap={same_raw-cross_raw:.4f}")
            log(f"      PROPER: PC1={pca_proper[0]:.4f}, rank={rank_proper}, "
                f"same={same_proper:.4f}, cross={cross_proper:.4f}, gap={same_proper-cross_proper:.4f}")
            log(f"      PSEUDO: PC1={pca_pseudo[0]:.4f}, rank={rank_pseudo}, "
                f"same={same_pseudo:.4f}, cross={cross_pseudo:.4f}, gap={same_pseudo-cross_pseudo:.4f}")
            if cos_raw_proper is not None:
                log(f"      PC1 dir cos: raw↔proper={cos_raw_proper:.4f}, "
                    f"raw↔pseudo={cos_raw_pseudo:.4f}, proper↔pseudo={cos_proper_pseudo:.4f}")
            
            layer_results[ablation_name] = {
                "raw_pc1": float(pca_raw[0]) if pca_raw is not None else None,
                "raw_rank": rank_raw,
                "raw_same_cat": same_raw, "raw_cross_cat": cross_raw,
                "proper_pc1": float(pca_proper[0]) if pca_proper is not None else None,
                "proper_rank": rank_proper,
                "proper_same_cat": same_proper, "proper_cross_cat": cross_proper,
                "pseudo_pc1": float(pca_pseudo[0]) if pca_pseudo is not None else None,
                "pseudo_rank": rank_pseudo,
                "pseudo_same_cat": same_pseudo, "pseudo_cross_cat": cross_pseudo,
                "cos_raw_proper": cos_raw_proper,
                "cos_raw_pseudo": cos_raw_pseudo,
                "cos_proper_pseudo": cos_proper_pseudo,
                "cat_pc1_stats": cat_pc1_stats,
            }
        
        results[l_str] = {
            "top2_channels": [int(x) for x in top2_ch],
            "top10_channels": [int(x) for x in top10_ch],
            "ablation_results": layer_results,
        }
    
    return results


# ===== Part 2: PC semantic decoding =====
def decode_pc_semantics(all_data, model, tokenizer, model_name):
    """
    Decode what each PC encodes in the proper post-RMSNorm space.
    
    Regress PC scores against known variables:
    - Category label (one-hot)
    - 2-channel energy (for DS7B)
    - Binding strength (target-competitor logit difference from W_U)
    - Norm of Δh
    - Norm ratio (||h_clean|| / ||h_corrupt||)
    """
    log("\n" + "="*60)
    log("Part 2: PC Semantic Decoding (Proper post-RMSNorm)")
    log("="*60)
    
    # Load W_U
    try:
        W_U = get_W_U(model, model_name)
    except:
        log("  Could not load W_U, skipping logit regression")
        W_U = None
    
    results = {}
    
    for l_str in sorted(all_data.keys(), key=int):
        d = all_data[l_str]
        W_down = d["W_down"]
        d_gate_up = d["d_gate_up"]
        dh = d["dh"]
        n_pairs = dh.shape[0]
        l = int(l_str)
        
        h_post_attn_clean = d["h_post_attn_clean"]
        h_post_attn_corrupt = d["h_post_attn_corrupt"]
        gate_up_clean = d["gate_up_clean"]
        gate_up_corrupt = d["gate_up_corrupt"]
        ln_weight = d.get("ln_weight", None)
        
        # Compute proper post-RMSNorm for baseline (no ablation)
        h_clean_norm = np.zeros_like(d["h_post_clean"])
        h_corrupt_norm = np.zeros_like(d["h_post_corrupt"])
        h_clean_norm_energy = np.zeros(n_pairs)  # ||h_clean_norm||^2 per pair
        h_corrupt_norm_energy = np.zeros(n_pairs)
        h_clean_energy = np.zeros(n_pairs)  # ||h_clean||^2 per pair
        h_corrupt_energy = np.zeros(n_pairs)
        
        for i in range(n_pairs):
            h_clean_norm[i] = rms_norm_single(d["h_post_clean"][i], ln_weight)
            h_corrupt_norm[i] = rms_norm_single(d["h_post_corrupt"][i], ln_weight)
            h_clean_norm_energy[i] = np.sum(h_clean_norm[i]**2)
            h_corrupt_norm_energy[i] = np.sum(h_corrupt_norm[i]**2)
            h_clean_energy[i] = np.sum(d["h_post_clean"][i]**2)
            h_corrupt_energy[i] = np.sum(d["h_post_corrupt"][i]**2)
        
        dh_proper = h_clean_norm - h_corrupt_norm
        
        # PCA on proper post-RMSNorm Δh
        pca, rank, Vt, scores, S = compute_pca_full(dh_proper)
        if pca is None:
            log(f"  Layer {l}: PCA failed, skipping")
            continue
        
        log(f"\n  Layer {l} (proper post-RMSNorm):")
        log(f"    Top-5 PC explained: {pca[:5].tolist()}")
        log(f"    Top-10 PC explained: {pca[:10].tolist()}")
        
        # ===== Build regressor matrix =====
        # Variable 1: Category one-hot (7 categories)
        cat_onehot = np.zeros((n_pairs, len(ALL_CATEGORIES)))
        for i, cat in enumerate(PAIR_CATEGORIES):
            cat_idx = ALL_CATEGORIES.index(cat)
            cat_onehot[i, cat_idx] = 1.0
        
        # Variable 2: 2-channel energy (top2 channels' contribution to d_gate_up)
        energy_per_ch = np.mean(d_gate_up**2, axis=0)
        sorted_indices = np.argsort(energy_per_ch)[::-1]
        top2_ch = sorted_indices[:2]
        ch2_energy = np.sum(d_gate_up[:, top2_ch]**2, axis=1)  # (n,)
        
        # Variable 3: Total Δ(gate*up) energy
        total_dgu_energy = np.sum(d_gate_up**2, axis=1)  # (n,)
        
        # Variable 4: ||Δh|| (raw)
        dh_norm = np.linalg.norm(dh, axis=1)  # (n,)
        
        # Variable 5: ||h_clean|| / ||h_corrupt|| ratio
        norm_ratio = h_clean_energy / (h_corrupt_energy + 1e-10)
        
        # Variable 6: RMSNorm Δh norm
        dh_proper_norm = np.linalg.norm(dh_proper, axis=1)
        
        # Variable 7: W_U logit effect (if available)
        logit_diff_raw = None
        logit_diff_norm = None
        if W_U is not None:
            logit_diff_raw = np.zeros(n_pairs)
            logit_diff_norm = np.zeros(n_pairs)
            for pidx, (obj, target, competitor) in enumerate(ALL_PAIRS):
                t_tokens = tokenizer(target, add_special_tokens=False)["input_ids"]
                c_tokens = tokenizer(competitor, add_special_tokens=False)["input_ids"]
                if len(t_tokens) != 1 or len(c_tokens) != 1:
                    continue
                t_id, c_id = t_tokens[0], c_tokens[0]
                if W_U.shape[0] == dh.shape[1]:
                    logit_raw = W_U.T @ dh[pidx]
                    logit_norm = W_U.T @ dh_proper[pidx]
                else:
                    logit_raw = W_U @ dh[pidx]
                    logit_norm = W_U @ dh_proper[pidx]
                logit_diff_raw[pidx] = logit_raw[t_id] - logit_raw[c_id]
                logit_diff_norm[pidx] = logit_norm[t_id] - logit_norm[c_id]
        
        # ===== Regress each PC score against variables =====
        n_pcs_to_check = min(20, scores.shape[1])
        pc_regression = {}
        
        var_names = ["ch2_energy", "total_dgu_energy", "dh_raw_norm", "norm_ratio",
                     "dh_proper_norm"]
        var_data = [ch2_energy, total_dgu_energy, dh_norm, norm_ratio, dh_proper_norm]
        
        if logit_diff_raw is not None:
            var_names.extend(["logit_diff_raw", "logit_diff_norm"])
            var_data.extend([logit_diff_raw, logit_diff_norm])
        
        for pc_idx in range(n_pcs_to_check):
            pc_score = scores[:, pc_idx]
            regressors = {}
            
            # Correlation with continuous variables
            for vname, vdata in zip(var_names, var_data):
                valid = ~(np.isnan(vdata) | np.isnan(pc_score))
                if np.sum(valid) > 10:
                    corr = np.corrcoef(pc_score[valid], vdata[valid])[0, 1]
                    regressors[vname] = float(corr)
            
            # R² with category one-hot (ANOVA-like)
            # Fit: pc_score = cat_onehot @ beta + residual
            # R² = 1 - var(residual) / var(pc_score)
            try:
                beta = np.linalg.lstsq(cat_onehot, pc_score, rcond=None)[0]
                predicted = cat_onehot @ beta
                residual = pc_score - predicted
                r2_cat = 1.0 - np.var(residual) / np.var(pc_score)
                regressors["category_r2"] = float(r2_cat)
            except:
                regressors["category_r2"] = None
            
            pc_regression[f"PC{pc_idx+1}"] = regressors
        
        # Print summary
        log(f"    PC semantic decoding (correlation / R²):")
        log(f"    {'PC':>5s} {'cat_R²':>8s} {'ch2_E':>8s} {'tot_E':>8s} "
            f"{'|Δh|':>8s} {'n_ratio':>8s} {'|Δh_n|':>8s}", end="")
        if logit_diff_raw is not None:
            log(f" {'logit_r':>8s} {'logit_n':>8s}")
        else:
            log()
        
        for pc_idx in range(min(10, n_pcs_to_check)):
            r = pc_regression[f"PC{pc_idx+1}"]
            log(f"    {'PC'+str(pc_idx+1):>5s} "
                f"{r.get('category_r2', 0):8.4f} "
                f"{r.get('ch2_energy', 0):8.4f} "
                f"{r.get('total_dgu_energy', 0):8.4f} "
                f"{r.get('dh_raw_norm', 0):8.4f} "
                f"{r.get('norm_ratio', 0):8.4f} "
                f"{r.get('dh_proper_norm', 0):8.4f}", end="")
            if logit_diff_raw is not None:
                log(f" {r.get('logit_diff_raw', 0):8.4f} {r.get('logit_diff_norm', 0):8.4f}")
            else:
                log()
        
        # Also check: does PC1 score separate categories at all?
        pc1_scores = scores[:, 0]
        log(f"\n    PC1 score per category:")
        for cat in ALL_CATEGORIES:
            idx = [j for j, c in enumerate(PAIR_CATEGORIES) if c == cat]
            s = pc1_scores[idx]
            log(f"      {cat:>12s}: mean={np.mean(s):.4f}, std={np.std(s):.4f}, "
                f"frac_positive={np.mean(s > 0):.3f}")
        
        # Check: are all categories mixed on PC1 (50-50 positive/negative)?
        frac_pos_overall = np.mean(pc1_scores > 0)
        log(f"    Overall PC1 fraction positive: {frac_pos_overall:.4f}")
        
        results[l_str] = {
            "top2_channels": [int(x) for x in top2_ch],
            "pca_explained": [float(x) for x in pca[:20]],
            "pc_regression": pc_regression,
            "pc1_frac_positive": float(frac_pos_overall),
            "pc1_per_category": {
                cat: {
                    "mean": float(np.mean(pc1_scores[[j for j, c in enumerate(PAIR_CATEGORIES) if c == cat]])),
                    "std": float(np.std(pc1_scores[[j for j, c in enumerate(PAIR_CATEGORIES) if c == cat]])),
                    "frac_positive": float(np.mean(pc1_scores[[j for j, c in enumerate(PAIR_CATEGORIES) if c == cat]] > 0)),
                }
                for cat in ALL_CATEGORIES
            },
        }
    
    return results


# ===== Part 3: RMSNorm Jacobian analysis =====
def rmsnorm_jacobian_analysis(all_data, model_name):
    """
    Compute the Jacobian of RMSNorm at each pair's clean/corrupt residual states.
    
    J(x) ≈ 1/rms(x) * [I - x⊗x / ||x||²] * γ
    
    This tells us:
    - How much RMSNorm preserves direction vs rotates it
    - Whether the 2-channel signal direction is in the null space or preserved
    """
    log("\n" + "="*60)
    log("Part 3: RMSNorm Jacobian Analysis")
    log("="*60)
    
    results = {}
    
    for l_str in sorted(all_data.keys(), key=int):
        d = all_data[l_str]
        n_pairs = d["dh"].shape[0]
        d_model = d["dh"].shape[1]
        l = int(l_str)
        W_down = d["W_down"]
        d_gate_up = d["d_gate_up"]
        ln_weight = d.get("ln_weight", None)
        
        log(f"\n  Layer {l}:")
        
        # Identify top channels
        energy_per_ch = np.mean(d_gate_up**2, axis=0)
        sorted_indices = np.argsort(energy_per_ch)[::-1]
        top2_ch = sorted_indices[:2]
        
        # The 2-channel signal direction in residual space
        # Δh_from_top2 = W_down @ d_gate_up_top2
        d_gate_up_top2 = d_gate_up.copy()
        mask = np.ones(d_gate_up.shape[1], dtype=bool)
        mask[top2_ch] = False
        d_gate_up_top2[:, mask] = 0
        dh_from_top2 = (W_down @ d_gate_up_top2.T).T  # (n, d_model)
        
        # For each pair, compute how RMSNorm's Jacobian acts on the 2-channel direction
        # J_clean @ Δh_from_top2[i] vs J_corrupt @ Δh_from_top2[i]
        
        # Instead of full Jacobian (d_model × d_model), compute projection:
        # J(x) @ v ≈ (RMSNorm(x + ε*v) - RMSNorm(x)) / ε for small ε
        # This gives us the action of the Jacobian on a specific direction v.
        
        eps = 1e-3
        jacobian_preserves_top2 = []
        jacobian_rotates_top2 = []
        rmsnorm_distortion = []
        
        for i in range(n_pairs):
            h_c = d["h_post_clean"][i]   # clean residual
            h_r = d["h_post_corrupt"][i] # corrupt residual
            v = dh_from_top2[i]           # 2-channel direction in residual space
            v_norm = np.linalg.norm(v)
            if v_norm < 1e-8:
                continue
            v_unit = v / v_norm
            
            # J_clean @ v via finite difference
            h_c_plus = h_c + eps * v_unit
            h_c_minus = h_c - eps * v_unit
            Jv_clean = (rms_norm_single(h_c_plus, ln_weight) - rms_norm_single(h_c_minus, ln_weight)) / (2 * eps)
            
            # J_corrupt @ v
            h_r_plus = h_r + eps * v_unit
            h_r_minus = h_r - eps * v_unit
            Jv_corrupt = (rms_norm_single(h_r_plus, ln_weight) - rms_norm_single(h_r_minus, ln_weight)) / (2 * eps)
            
            # How much does the Jacobian preserve the direction?
            # cos(v_unit, Jv)
            cos_preserve_clean = cosine_sim(v_unit, Jv_clean)
            cos_preserve_corrupt = cosine_sim(v_unit, Jv_corrupt)
            
            # How different is J_clean @ v from J_corrupt @ v?
            # If they're very different, then RMSNorm maps the same 2-channel signal
            # differently depending on the base state (clean vs corrupt)
            cos_clean_corrupt_Jv = cosine_sim(Jv_clean, Jv_corrupt)
            
            # Proper Δ(Jv) = Jv_clean - Jv_corrupt
            dJv = Jv_clean - Jv_corrupt
            
            # Compare: does the Jacobian-transformed 2-channel direction
            # still point in the same direction as the original?
            cos_dJv_with_v = cosine_sim(dJv, v_unit)
            
            jacobian_preserves_top2.append({
                "cos_v_Jv_clean": float(cos_preserve_clean),
                "cos_v_Jv_corrupt": float(cos_preserve_corrupt),
                "cos_Jv_clean_corrupt": float(cos_clean_corrupt_Jv),
                "cos_dJv_with_v": float(cos_dJv_with_v),
                "norm_Jv_clean": float(np.linalg.norm(Jv_clean)),
                "norm_Jv_corrupt": float(np.linalg.norm(Jv_corrupt)),
                "norm_dJv": float(np.linalg.norm(dJv)),
            })
        
        # Summary statistics
        if jacobian_preserves_top2:
            cos_v_Jvc = [x["cos_v_Jv_clean"] for x in jacobian_preserves_top2]
            cos_v_Jvr = [x["cos_v_Jv_corrupt"] for x in jacobian_preserves_top2]
            cos_Jvc_Jvr = [x["cos_Jv_clean_corrupt"] for x in jacobian_preserves_top2]
            cos_dJv_v = [x["cos_dJv_with_v"] for x in jacobian_preserves_top2]
            
            log(f"    Jacobian analysis (2-channel direction):")
            log(f"      cos(v, J_clean@v): mean={np.mean(cos_v_Jvc):.4f}, std={np.std(cos_v_Jvc):.4f}")
            log(f"      cos(v, J_corrupt@v): mean={np.mean(cos_v_Jvr):.4f}, std={np.std(cos_v_Jvr):.4f}")
            log(f"      cos(J_c@v, J_r@v): mean={np.mean(cos_Jvc_Jvr):.4f}, std={np.std(cos_Jvc_Jvr):.4f}")
            log(f"      cos(ΔJv, v): mean={np.mean(cos_dJv_v):.4f}, std={np.std(cos_dJv_v):.4f}")
            
            results[l_str] = {
                "top2_channels": [int(x) for x in top2_ch],
                "mean_cos_v_Jv_clean": float(np.mean(cos_v_Jvc)),
                "mean_cos_v_Jv_corrupt": float(np.mean(cos_v_Jvr)),
                "mean_cos_Jv_clean_corrupt": float(np.mean(cos_Jvc_Jvr)),
                "mean_cos_dJv_with_v": float(np.mean(cos_dJv_v)),
                "std_cos_v_Jv_clean": float(np.std(cos_v_Jvc)),
                "std_cos_v_Jv_corrupt": float(np.std(cos_v_Jvr)),
                "per_pair_sample": jacobian_preserves_top2[:5],  # first 5 pairs as example
            }
        else:
            results[l_str] = {"note": "no valid pairs for Jacobian analysis"}
    
    return results


# ===== Main =====
def run_phase379(model_name):
    cfg = MODEL_CONFIGS[model_name]
    
    # Target layers per model
    TARGET_LAYERS = {
        "qwen3": [4, 28],
        "deepseek7b": [4, 5, 8, 24],
        "glm4": [4, 30],
    }
    target_layers = TARGET_LAYERS.get(model_name, [4])
    
    log(f"\n{'='*70}")
    log(f"Phase 379: RMSNorm Remapping Audit — {model_name}")
    log(f"Target layers: {target_layers}")
    log(f"{'='*70}")
    
    # Load model
    t0 = time.time()
    model, tokenizer = load_model_bf16(model_name)
    log(f"Model loaded in {time.time()-t0:.1f}s")
    
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log(f"GPU memory: {gpu_mem:.2f} GB")
    
    # Collect data
    log("\n--- Data Collection ---")
    all_data = collect_residual_states(model, tokenizer, model_name, target_layers)
    
    # Part 1: Proper vs Pseudo RMSNorm
    log("\n--- Part 1: Proper vs Pseudo RMSNorm ---")
    part1_results = compare_proper_vs_pseudo_rmsnorm(all_data, model_name)
    
    # Part 2: PC Semantic Decoding
    log("\n--- Part 2: PC Semantic Decoding ---")
    part2_results = decode_pc_semantics(all_data, model, tokenizer, model_name)
    
    # Part 3: Jacobian Analysis
    log("\n--- Part 3: Jacobian Analysis ---")
    part3_results = rmsnorm_jacobian_analysis(all_data, model_name)
    
    # Save results
    output = {
        "model": model_name,
        "phase": "379",
        "n_pairs": len(ALL_PAIRS),
        "n_categories": len(ALL_CATEGORIES),
        "categories": ALL_CATEGORIES,
        "timestamp": datetime.now().isoformat(),
        "part1_proper_vs_pseudo": part1_results,
        "part2_pc_semantic_decoding": part2_results,
        "part3_jacobian": part3_results,
    }
    
    os.makedirs("results/phase379_rmsnorm_remapping", exist_ok=True)
    out_path = f"results/phase379_rmsnorm_remapping/{model_name}_phase379.json"
    
    # Custom JSON serializer for numpy types
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, cls=NpEncoder, ensure_ascii=False)
    
    log(f"\nResults saved to {out_path}")
    
    # Release model
    release_model(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    log(f"\nPhase 379 complete for {model_name}")
    return output


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_phase379(model_name)
