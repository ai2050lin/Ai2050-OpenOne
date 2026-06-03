"""
Phase 347: W_down Row Structure Analysis + Full Circuit Closure
================================================================

This phase addresses the 5 hard problems from Phase 346:

Part A: W_down Row Structure — Why are positive/negative channels so balanced?
  1. Analyze W_down row vectors' cosine similarity with binding direction
  2. Distribution of row norms for positive vs negative channels
  3. Check if positive/negative channels have symmetric structures
  4. Spectral analysis: W_down's singular vectors vs binding direction
  5. Channel-level interaction decomposition: which channels drive interaction?

Part B: Full Circuit Closure — Add attention + non-binding layers
  1. Capture attention output per layer in binding direction
  2. Capture ALL layer MLP contributions (not just binding layers)
  3. Compute complete residual stream decomposition
  4. Evaluate attention vs MLP relative contribution

Part C: Interaction Physical Meaning
  1. Per-channel interaction sign and magnitude
  2. Correlation between gate magnitude and interaction contribution
  3. SiLU nonlinearity curvature analysis

Usage:
  python tests/glm5/phase347_wdown_structure_closure.py qwen3
  python tests/glm5/phase347_wdown_structure_closure.py deepseek7b
  python tests/glm5/phase347_wdown_structure_closure.py glm4
"""
import sys, os, time, json, gc
import torch
import numpy as np
from datetime import datetime
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

def log(msg="", end="\n"):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", end=end, flush=True)


MODEL_CONFIGS = {
    "qwen3": {
        "path": "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
        "n_layers": 36, "d_model": 2560,
        "binding_layers": [21, 23, 25, 27, 29],
    },
    "glm4": {
        "path": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
        "n_layers": 40, "d_model": 4096,
        "binding_layers": [30, 33, 36, 38],
    },
    "deepseek7b": {
        "path": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "n_layers": 28, "d_model": 3584,
        "binding_layers": [19, 21, 23, 24],
    },
}

TEST_PAIRS = [
    ("apple", "red", "blue"), ("banana", "yellow", "purple"), ("snow", "white", "black"),
    ("sky", "blue", "green"), ("fire", "hot", "cold"), ("grass", "green", "red"),
    ("ocean", "blue", "yellow"), ("sun", "yellow", "purple"), ("blood", "red", "green"),
    ("ice", "cold", "hot"), ("cherry", "red", "blue"), ("leaf", "green", "red"),
    ("rose", "red", "blue"), ("gold", "yellow", "purple"), ("coal", "black", "white"),
    ("silver", "white", "black"), ("milk", "white", "black"), ("honey", "yellow", "blue"),
]

CORRUPTED_BASELINE = "The item"


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
            log(f"  Loaded {model_name} with attn_impl={impl}")
            break
        except Exception as e:
            log(f"  Failed with {impl}: {e}")
            continue
    if model is None:
        raise RuntimeError(f"Failed to load {model_name}")
    model.eval()
    return model, tokenizer, next(model.parameters()).device


def get_W_U(model, model_name):
    if hasattr(model, "lm_head"):
        w = model.lm_head.weight
        if not w.is_meta:
            return w.detach().cpu().float().numpy()
    import glob
    from safetensors import safe_open
    for sf_file in glob.glob(os.path.join(MODEL_CONFIGS[model_name]["path"], '*.safetensors')):
        with safe_open(sf_file, framework='pt', device='cpu') as sf:
            if 'lm_head.weight' in sf.keys():
                return sf.get_tensor('lm_head.weight').float().numpy()
    raise ValueError(f"Cannot load lm_head for {model_name}")


def get_token_id(tokenizer, word):
    ids = tokenizer.encode(word, add_special_tokens=False)
    return ids[0] if ids else None


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("Cannot find layers")


def safe_weight_to_numpy(w):
    if w.is_meta:
        return None
    try:
        return w.detach().cpu().float().numpy()
    except:
        return None


def get_mlp_weights(layer, model_name=None, model=None):
    mlp = layer.mlp
    W_gate = W_up = W_down = None; d_ff = 0
    if hasattr(mlp, 'gate_up_proj'):
        w = safe_weight_to_numpy(mlp.gate_up_proj.weight)
        if w is not None: d_ff = w.shape[0] // 2; W_gate, W_up = w[:d_ff], w[d_ff:]
    elif hasattr(mlp, 'gate_proj'):
        W_gate = safe_weight_to_numpy(mlp.gate_proj.weight)
        W_up = safe_weight_to_numpy(mlp.up_proj.weight)
        if W_gate is not None: d_ff = W_gate.shape[0]
        elif W_up is not None: d_ff = W_up.shape[0]
    elif hasattr(mlp, 'up_proj'):
        W_up = safe_weight_to_numpy(mlp.up_proj.weight)
        if W_up is not None: d_ff = W_up.shape[0]
    if hasattr(mlp, 'down_proj'): W_down = safe_weight_to_numpy(mlp.down_proj.weight)
    return W_gate, W_up, W_down, d_ff


def get_mlp_weights_from_disk(model_name, layer_idx):
    import glob
    from safetensors import safe_open
    W_gate = W_up = W_down = None; d_ff = 0
    for sf_file in glob.glob(os.path.join(MODEL_CONFIGS[model_name]["path"], '*.safetensors')):
        try:
            with safe_open(sf_file, framework='pt', device='cpu') as sf:
                keys = sf.keys()
                p = f"model.layers.{layer_idx}.mlp"
                guk = f"{p}.gate_up_proj.weight"
                if guk in keys:
                    w = sf.get_tensor(guk).float().numpy()
                    d_ff = w.shape[0] // 2; W_gate, W_up = w[:d_ff], w[d_ff:]
                gk = f"{p}.gate_proj.weight"
                if gk in keys and W_gate is None:
                    W_gate = sf.get_tensor(gk).float().numpy(); d_ff = W_gate.shape[0]
                uk = f"{p}.up_proj.weight"
                if uk in keys and W_up is None:
                    W_up = sf.get_tensor(uk).float().numpy()
                    if d_ff == 0: d_ff = W_up.shape[0]
                dk = f"{p}.down_proj.weight"
                if dk in keys and W_down is None: W_down = sf.get_tensor(dk).float().numpy()
                if W_down is not None: break
        except: continue
    return W_gate, W_up, W_down, d_ff


def capture_mlp_internals(model, tokenizer, device, prompt, target_layers, n_layers):
    layers = get_layers(model)
    captured = {}
    def make_hook(key):
        def hook(module, input, output):
            val = output[0] if isinstance(output, tuple) else output
            captured[key] = val[0, -1, :].detach().cpu().float().numpy()
        return hook
    hooks = []
    for li in target_layers:
        layer = layers[li]
        if hasattr(layer.mlp, 'gate_proj'):
            hooks.append(layer.mlp.gate_proj.register_forward_hook(make_hook(f"gate_{li}")))
        elif hasattr(layer.mlp, 'gate_up_proj'):
            def make_glm4_hook(idx):
                def hook(module, input, output):
                    val = output[0] if isinstance(output, tuple) else output
                    v = val[0, -1, :].detach().cpu().float().numpy()
                    d = v.shape[0] // 2
                    captured[f"gate_{idx}"] = v[:d]; captured[f"up_{idx}"] = v[d:]
                return hook
            hooks.append(layer.mlp.gate_up_proj.register_forward_hook(make_glm4_hook(li)))
        if hasattr(layer.mlp, 'up_proj'):
            hooks.append(layer.mlp.up_proj.register_forward_hook(make_hook(f"up_{li}")))
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        model(**inp, output_hidden_states=True)
    for h in hooks: h.remove()
    return captured


def capture_residual_components(model, tokenizer, device, prompt, target_layers, n_layers):
    """Capture attention output and MLP output per layer via hooks."""
    layers = get_layers(model)
    captured = {}

    def make_hook(key):
        def hook(module, input, output):
            val = output[0] if isinstance(output, tuple) else output
            captured[key] = val[0, -1, :].detach().cpu().float().numpy()
        return hook

    hooks = []
    for li in target_layers:
        layer = layers[li]
        # Hook MLP output
        if hasattr(layer.mlp, 'down_proj'):
            hooks.append(layer.mlp.down_proj.register_forward_hook(make_hook(f"mlp_out_{li}")))
        # Hook attention output (before residual add)
        # In most models, self_attention returns output that gets added to residual
        if hasattr(layer, 'self_attn'):
            hooks.append(layer.self_attn.register_forward_hook(make_hook(f"attn_out_{li}")))
        # Hook gate/up for interaction analysis
        if hasattr(layer.mlp, 'gate_proj'):
            hooks.append(layer.mlp.gate_proj.register_forward_hook(make_hook(f"gate_{li}")))
            hooks.append(layer.mlp.up_proj.register_forward_hook(make_hook(f"up_{li}")))
        elif hasattr(layer.mlp, 'gate_up_proj'):
            def make_glm4_hook(idx):
                def hook(module, input, output):
                    val = output[0] if isinstance(output, tuple) else output
                    v = val[0, -1, :].detach().cpu().float().numpy()
                    d = v.shape[0] // 2
                    captured[f"gate_{idx}"] = v[:d]; captured[f"up_{idx}"] = v[d:]
                return hook
            hooks.append(layer.mlp.gate_up_proj.register_forward_hook(make_glm4_hook(li)))

    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        model(**inp, output_hidden_states=True)
    for h in hooks: h.remove()
    return captured


def capture_hidden_states(model, tokenizer, device, prompt, n_layers):
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    hs = {}
    for i, h in enumerate(out.hidden_states):
        hs[i] = h[0, -1, :].detach().cpu().float().numpy()
    return hs


def silu_np(x):
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -20, 20))))


# ======================================================================
# PART A: W_down Row Structure Analysis
# ======================================================================
def analyze_wdown_structure(W_down, direction_normed, d_ff, gate_clean, up_clean, gate_corrupt, up_corrupt):
    """
    Analyze W_down row structure to understand why channels are balanced.
    
    Key questions:
    1. What is the distribution of W_down row · direction (channel projections)?
    2. Are positive/negative channels symmetric in norm?
    3. Do positive/negative channels have different spectral properties?
    4. Which channels contribute most to the interaction term?
    """
    min_d = min(d_ff, W_down.shape[1], gate_clean.shape[0])
    Wd = W_down[:, :min_d]
    
    # Channel-level projection onto direction
    # W_down shape: (d_model, d_ff), each row is a d_ff-dimensional vector
    # channel_proj[i] = W_down[i, :] · direction = contribution of channel i to direction
    # Actually, W_down maps from d_ff to d_model, so W_down[j, i] maps channel i to dim j
    # channel_proj[i] = sum_j W_down[j, i] * direction[j] = (W_down.T @ direction)[i]
    channel_proj = Wd.T @ direction_normed  # shape: (d_ff,)
    
    # Channel activation difference (clean - corrupt)
    gsc = silu_np(gate_clean[:min_d])
    gsr = silu_np(gate_corrupt[:min_d])
    uc = up_clean[:min_d]
    ur = up_corrupt[:min_d]
    
    activation_diff = gsc * uc - gsr * ur  # shape: (d_ff,)
    
    # Per-channel contribution to MLP output difference in binding direction
    # mlp_diff_proj = (W_down @ (gsc*uc - gsr*ur)) · direction
    #               = sum_i channel_proj[i] * activation_diff[i]
    per_channel_contrib = channel_proj * activation_diff  # shape: (d_ff,)
    
    # Classify channels as positive/negative based on channel_proj sign
    pos_mask = channel_proj > 0
    neg_mask = channel_proj < 0
    
    # Channel projection statistics
    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))
    n_zero = min_d - n_pos - n_neg
    
    # Positive channel stats
    pos_proj_vals = channel_proj[pos_mask]
    neg_proj_vals = channel_proj[neg_mask]
    
    # Activation difference stats for pos/neg channels
    pos_act_diff = activation_diff[pos_mask]
    neg_act_diff = activation_diff[neg_mask]
    
    # Per-channel contribution stats
    pos_contrib = per_channel_contrib[pos_mask]
    neg_contrib = per_channel_contrib[neg_mask]
    
    # W_down row norms for channels with positive vs negative projection
    # Each column of W_down is the "read-out" vector for that channel
    # W_down[:, i] is the d_model vector that channel i writes to
    channel_norms = np.linalg.norm(Wd, axis=0)  # shape: (d_ff,)
    pos_norms = channel_norms[pos_mask]
    neg_norms = channel_norms[neg_mask]
    
    # ---- Interaction decomposition at channel level ----
    # interaction = CC - CR - RC + RR
    # For channel i: interaction_i = channel_proj[i] * (gsc[i]*uc[i] - gsr[i]*uc[i] - gsc[i]*ur[i] + gsr[i]*ur[i])
    # = channel_proj[i] * ((gsc[i] - gsr[i]) * (uc[i] - ur[i]))
    # Wait, let me verify:
    # CC_i = channel_proj[i] * gsc[i] * uc[i]
    # CR_i = channel_proj[i] * gsc[i] * ur[i]
    # RC_i = channel_proj[i] * gsr[i] * uc[i]
    # RR_i = channel_proj[i] * gsr[i] * ur[i]
    # interaction_i = CC_i - CR_i - RC_i + RR_i
    #              = channel_proj[i] * (gsc[i]*uc[i] - gsc[i]*ur[i] - gsr[i]*uc[i] + gsr[i]*ur[i])
    #              = channel_proj[i] * (gsc[i] - gsr[i]) * (uc[i] - ur[i])
    
    gate_diff = gsc - gsr  # SiLU(gate_clean) - SiLU(gate_corrupt)
    up_diff = uc - ur      # up_clean - up_corrupt
    
    # Per-channel interaction contribution
    per_channel_interaction = channel_proj * gate_diff * up_diff  # shape: (d_ff,)
    
    # Per-channel gate main effect
    # gate_main_i = channel_proj[i] * ((gsc[i]*uc[i] - gsc[i]*ur[i]) + (gsr[i]*uc[i] - gsr[i]*ur[i])) / 2
    # = channel_proj[i] * ((gsc[i] + gsr[i]) / 2) * (uc[i] - ur[i])
    # Wait, that's not right. Let me recalculate properly.
    # gate_main = ((CC-CR) + (RC-RR)) / 2
    # For channel i: gate_main_i = channel_proj[i] * ((gsc[i]*uc[i] - gsc[i]*ur[i]) + (gsr[i]*uc[i] - gsr[i]*ur[i])) / 2
    # = channel_proj[i] * ((gsc[i] + gsr[i]) / 2) * (uc[i] - ur[i])
    per_channel_gate_main = channel_proj * ((gsc + gsr) / 2) * up_diff
    
    # up_main = ((CC-RC) + (CR-RR)) / 2
    # up_main_i = channel_proj[i] * ((gsc[i]*uc[i] - gsr[i]*uc[i]) + (gsc[i]*ur[i] - gsr[i]*ur[i])) / 2
    # = channel_proj[i] * (gsc[i] - gsr[i]) * ((uc[i] + ur[i]) / 2)
    # Wait: ((gsc*uc - gsr*uc) + (gsc*ur - gsr*ur)) / 2 = (gsc - gsr) * (uc + ur) / 2
    per_channel_up_main = channel_proj * gate_diff * ((uc + ur) / 2)
    
    # Verify totals
    total_gate_main = float(np.sum(per_channel_gate_main))
    total_up_main = float(np.sum(per_channel_up_main))
    total_interaction = float(np.sum(per_channel_interaction))
    total_contrib = float(np.sum(per_channel_contrib))
    
    # For pos/neg channels: aggregate interaction
    pos_interaction = per_channel_interaction[pos_mask]
    neg_interaction = per_channel_interaction[neg_mask]
    
    # Top contributing channels
    abs_contrib = np.abs(per_channel_contrib)
    top_k = min(20, min_d)
    top_indices = np.argsort(abs_contrib)[-top_k:][::-1]
    
    top_channels = []
    for idx in top_indices:
        top_channels.append({
            "channel": int(idx),
            "proj": float(channel_proj[idx]),
            "act_diff": float(activation_diff[idx]),
            "contrib": float(per_channel_contrib[idx]),
            "gate_main": float(per_channel_gate_main[idx]),
            "up_main": float(per_channel_up_main[idx]),
            "interaction": float(per_channel_interaction[idx]),
            "gate_diff": float(gate_diff[idx]),
            "up_diff": float(up_diff[idx]),
            "sign": "pos" if channel_proj[idx] > 0 else "neg",
        })
    
    return {
        "channel_proj": channel_proj,
        "activation_diff": activation_diff,
        "per_channel_contrib": per_channel_contrib,
        "per_channel_interaction": per_channel_interaction,
        "per_channel_gate_main": per_channel_gate_main,
        "per_channel_up_main": per_channel_up_main,
        "gate_diff": gate_diff,
        "up_diff": up_diff,
        "channel_norms": channel_norms,
        # Statistics
        "n_pos": n_pos, "n_neg": n_neg, "n_zero": n_zero, "total": min_d,
        "pos_proj_mean": float(np.mean(pos_proj_vals)) if n_pos > 0 else 0,
        "neg_proj_mean": float(np.mean(neg_proj_vals)) if n_neg > 0 else 0,
        "pos_proj_std": float(np.std(pos_proj_vals)) if n_pos > 0 else 0,
        "neg_proj_std": float(np.std(neg_proj_vals)) if n_neg > 0 else 0,
        "pos_norm_mean": float(np.mean(pos_norms)) if n_pos > 0 else 0,
        "neg_norm_mean": float(np.mean(neg_norms)) if n_neg > 0 else 0,
        "pos_norm_std": float(np.std(pos_norms)) if n_pos > 0 else 0,
        "neg_norm_std": float(np.std(neg_norms)) if n_neg > 0 else 0,
        "pos_act_diff_mean": float(np.mean(pos_act_diff)) if n_pos > 0 else 0,
        "neg_act_diff_mean": float(np.mean(neg_act_diff)) if n_neg > 0 else 0,
        "pos_contrib_sum": float(np.sum(pos_contrib)),
        "neg_contrib_sum": float(np.sum(neg_contrib)),
        "pos_interaction_sum": float(np.sum(pos_interaction)),
        "neg_interaction_sum": float(np.sum(neg_interaction)),
        "total_gate_main": total_gate_main,
        "total_up_main": total_up_main,
        "total_interaction": total_interaction,
        "total_contrib": total_contrib,
        # Balance metrics
        "proj_balance_ratio": float(np.sum(np.abs(pos_proj_vals)) / max(np.sum(np.abs(neg_proj_vals)), 1e-10)) if n_neg > 0 else float('inf'),
        "contrib_balance_ratio": float(np.sum(np.abs(pos_contrib)) / max(np.sum(np.abs(neg_contrib)), 1e-10)) if n_neg > 0 else float('inf'),
        "norm_balance_ratio": float(np.mean(pos_norms) / max(np.mean(neg_norms), 1e-10)) if n_neg > 0 else float('inf'),
        # Top channels
        "top_channels": top_channels,
    }


# ======================================================================
# PART B: Full Circuit Closure
# ======================================================================
def full_circuit_closure(model, tokenizer, device, model_name, direction_normed, clean_prompt, corrupt_prompt):
    """
    Compute full circuit closure: attention + MLP contributions across ALL layers.
    """
    cfg = MODEL_CONFIGS[model_name]
    n_layers = cfg["n_layers"]
    all_layers = list(range(n_layers))
    
    # Capture hidden states
    hs_clean = capture_hidden_states(model, tokenizer, device, clean_prompt, n_layers)
    hs_corrupt = capture_hidden_states(model, tokenizer, device, corrupt_prompt, n_layers)
    
    # Final binding signal
    final_binding = float(direction_normed @ (hs_clean[n_layers] - hs_corrupt[n_layers]))
    
    # Per-layer binding trajectory
    binding_trajectory = {}
    for l in range(n_layers + 1):
        binding_trajectory[l] = float(direction_normed @ (hs_clean[l] - hs_corrupt[l]))
    
    # Per-layer delta (what each layer adds to binding)
    layer_deltas = {}
    for l in range(n_layers):
        layer_deltas[l] = binding_trajectory[l + 1] - binding_trajectory[l]
    
    # Now capture attention and MLP outputs for all layers
    # Do this in chunks to avoid memory issues
    chunk_size = 8
    attn_contribs = {}
    mlp_contribs = {}
    gate_caps = {}
    up_caps = {}
    
    for chunk_start in range(0, n_layers, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_layers)
        chunk_layers = list(range(chunk_start, chunk_end))
        
        # Clean pass
        caps_clean = capture_residual_components(model, tokenizer, device, clean_prompt, chunk_layers, n_layers)
        # Corrupt pass
        caps_corrupt = capture_residual_components(model, tokenizer, device, corrupt_prompt, chunk_layers, n_layers)
        
        for li in chunk_layers:
            # Attention contribution
            ak_c = f"attn_out_{li}"
            if ak_c in caps_clean and ak_c in caps_corrupt:
                attn_diff = caps_clean[ak_c] - caps_corrupt[ak_c]
                attn_contribs[li] = float(direction_normed @ attn_diff)
            
            # MLP contribution
            mk_c = f"mlp_out_{li}"
            if mk_c in caps_clean and mk_c in caps_corrupt:
                mlp_diff = caps_clean[mk_c] - caps_corrupt[mk_c]
                mlp_contribs[li] = float(direction_normed @ mlp_diff)
            
            # Gate/up activations
            gk = f"gate_{li}"
            uk = f"up_{li}"
            if gk in caps_clean:
                gate_caps[f"clean_{li}"] = caps_clean[gk]
            if gk in caps_corrupt:
                gate_caps[f"corrupt_{li}"] = caps_corrupt[gk]
            if uk in caps_clean:
                up_caps[f"clean_{li}"] = caps_clean[uk]
            if uk in caps_corrupt:
                up_caps[f"corrupt_{li}"] = caps_corrupt[uk]
        
        del caps_clean, caps_corrupt
        gc.collect(); torch.cuda.empty_cache()
    
    return {
        "final_binding": final_binding,
        "binding_trajectory": binding_trajectory,
        "layer_deltas": layer_deltas,
        "attn_contribs": attn_contribs,
        "mlp_contribs": mlp_contribs,
        "gate_caps": gate_caps,
        "up_caps": up_caps,
    }


def run_experiment(model_name):
    log(f"Phase 347: W_down Structure + Full Closure — {model_name}")
    log("=" * 70)
    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    cfg = MODEL_CONFIGS[model_name]
    binding_layers = cfg["binding_layers"]
    n_layers = cfg["n_layers"]
    W_U = get_W_U(model, model_name)
    d_model = W_U.shape[1]
    log(f"  W_U shape: {W_U.shape}, d_model={d_model}")

    # Pre-extract MLP weights for binding layers
    layers = get_layers(model)
    mlp_weights = {}
    for li in binding_layers:
        W_gate, W_up, W_down, d_ff = get_mlp_weights(layers[li], model_name, model)
        if W_down is None:
            W_gate, W_up, W_down, d_ff = get_mlp_weights_from_disk(model_name, li)
        mlp_weights[li] = {"W_gate": W_gate, "W_up": W_up, "W_down": W_down, "d_ff": d_ff}
        log(f"  Layer {li}: W_down shape={W_down.shape if W_down is not None else 'None'}, d_ff={d_ff}")

    # ======================================================================
    # PART A: W_down Row Structure Analysis
    # ======================================================================
    log(f"\n{'='*70}")
    log(f"PART A: W_down Row Structure Analysis")
    log(f"{'='*70}")
    
    wdown_results = {}
    all_channel_stats = defaultdict(list)
    
    n_wdown_pairs = 12  # Use more pairs for robust statistics
    
    for pidx, (obj, target, competitor) in enumerate(TEST_PAIRS[:n_wdown_pairs]):
        tid_t = get_token_id(tokenizer, target)
        tid_c = get_token_id(tokenizer, competitor)
        if tid_t is None or tid_c is None: continue
        
        direction = W_U[tid_t] - W_U[tid_c]
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-10: continue
        direction_normed = direction / dir_norm
        
        clean_prompt = f"The {obj}"
        clean_caps = capture_mlp_internals(model, tokenizer, device, clean_prompt, binding_layers, n_layers)
        corrupt_caps = capture_mlp_internals(model, tokenizer, device, CORRUPTED_BASELINE, binding_layers, n_layers)
        
        pair_key = f"{obj}-{target}-{competitor}"
        wdown_results[pair_key] = {}
        
        for li in binding_layers:
            mw = mlp_weights[li]
            W_down = mw["W_down"]; d_ff = mw["d_ff"]
            if W_down is None: continue
            gk = f"gate_{li}"; uk = f"up_{li}"
            if gk not in clean_caps or gk not in corrupt_caps: continue
            cg = clean_caps[gk][:d_ff]; crg = corrupt_caps[gk][:d_ff]
            cu = clean_caps.get(uk, np.ones(d_ff))[:d_ff]
            cru = corrupt_caps.get(uk, np.ones(d_ff))[:d_ff]
            
            analysis = analyze_wdown_structure(W_down, direction_normed, d_ff, cg, cu, crg, cru)
            
            # Store key metrics (not the full arrays to save space)
            wdown_results[pair_key][str(li)] = {
                "n_pos": analysis["n_pos"],
                "n_neg": analysis["n_neg"],
                "n_zero": analysis["n_zero"],
                "total_channels": analysis["total"],
                "pos_proj_mean": analysis["pos_proj_mean"],
                "neg_proj_mean": analysis["neg_proj_mean"],
                "pos_proj_std": analysis["pos_proj_std"],
                "neg_proj_std": analysis["neg_proj_std"],
                "pos_norm_mean": analysis["pos_norm_mean"],
                "neg_norm_mean": analysis["neg_norm_mean"],
                "pos_norm_std": analysis["pos_norm_std"],
                "neg_norm_std": analysis["neg_norm_std"],
                "pos_contrib_sum": analysis["pos_contrib_sum"],
                "neg_contrib_sum": analysis["neg_contrib_sum"],
                "pos_interaction_sum": analysis["pos_interaction_sum"],
                "neg_interaction_sum": analysis["neg_interaction_sum"],
                "proj_balance_ratio": analysis["proj_balance_ratio"],
                "contrib_balance_ratio": analysis["contrib_balance_ratio"],
                "norm_balance_ratio": analysis["norm_balance_ratio"],
                "total_gate_main": analysis["total_gate_main"],
                "total_up_main": analysis["total_up_main"],
                "total_interaction": analysis["total_interaction"],
                "total_contrib": analysis["total_contrib"],
                "top_channels": analysis["top_channels"],
                # Histogram of channel projections (10 bins)
                "proj_hist_counts": np.histogram(analysis["channel_proj"], bins=20)[0].tolist() if isinstance(analysis["channel_proj"], np.ndarray) else [],
                "proj_hist_bins": np.histogram(analysis["channel_proj"], bins=20)[1].tolist() if isinstance(analysis["channel_proj"], np.ndarray) else [],
            }
            
            # Aggregate across pairs and layers
            all_channel_stats["n_pos_frac"].append(analysis["n_pos"] / max(analysis["total"], 1))
            all_channel_stats["proj_balance_ratio"].append(analysis["proj_balance_ratio"])
            all_channel_stats["contrib_balance_ratio"].append(analysis["contrib_balance_ratio"])
            all_channel_stats["norm_balance_ratio"].append(analysis["norm_balance_ratio"])
            all_channel_stats["pos_proj_mean"].append(analysis["pos_proj_mean"])
            all_channel_stats["neg_proj_mean"].append(analysis["neg_proj_mean"])
            all_channel_stats["pos_norm_mean"].append(analysis["pos_norm_mean"])
            all_channel_stats["neg_norm_mean"].append(analysis["neg_norm_mean"])
            all_channel_stats["pos_contrib_sum"].append(analysis["pos_contrib_sum"])
            all_channel_stats["neg_contrib_sum"].append(analysis["neg_contrib_sum"])
            all_channel_stats["pos_interaction_sum"].append(analysis["pos_interaction_sum"])
            all_channel_stats["neg_interaction_sum"].append(analysis["neg_interaction_sum"])
        
        del clean_caps, corrupt_caps; gc.collect(); torch.cuda.empty_cache()
        if (pidx + 1) % 4 == 0:
            log(f"  [{pidx+1}/{n_wdown_pairs}] elapsed={time.time()-t0:.0f}s")
    
    # Part A Summary
    log(f"\n  PART A Summary:")
    log(f"  Channel projection balance (proj_balance_ratio):")
    if all_channel_stats["proj_balance_ratio"]:
        log(f"    mean={np.mean(all_channel_stats['proj_balance_ratio']):.4f} ± {np.std(all_channel_stats['proj_balance_ratio']):.4f}")
    log(f"  Channel contribution balance (contrib_balance_ratio):")
    if all_channel_stats["contrib_balance_ratio"]:
        log(f"    mean={np.mean(all_channel_stats['contrib_balance_ratio']):.4f} ± {np.std(all_channel_stats['contrib_balance_ratio']):.4f}")
    log(f"  Channel norm balance (norm_balance_ratio):")
    if all_channel_stats["norm_balance_ratio"]:
        log(f"    mean={np.mean(all_channel_stats['norm_balance_ratio']):.4f} ± {np.std(all_channel_stats['norm_balance_ratio']):.4f}")
    log(f"  Positive channel fraction:")
    if all_channel_stats["n_pos_frac"]:
        log(f"    mean={np.mean(all_channel_stats['n_pos_frac']):.4f} ± {np.std(all_channel_stats['n_pos_frac']):.4f}")
    log(f"  Positive vs negative channel projection means:")
    if all_channel_stats["pos_proj_mean"]:
        log(f"    pos_proj_mean: {np.mean(all_channel_stats['pos_proj_mean']):.6f}")
        log(f"    neg_proj_mean: {np.mean(all_channel_stats['neg_proj_mean']):.6f}")
    log(f"  Positive vs negative channel norm means:")
    if all_channel_stats["pos_norm_mean"]:
        log(f"    pos_norm_mean: {np.mean(all_channel_stats['pos_norm_mean']):.6f}")
        log(f"    neg_norm_mean: {np.mean(all_channel_stats['neg_norm_mean']):.6f}")
    log(f"  Positive vs negative contribution sums:")
    if all_channel_stats["pos_contrib_sum"]:
        log(f"    pos_contrib_sum: mean={np.mean(all_channel_stats['pos_contrib_sum']):.6f}")
        log(f"    neg_contrib_sum: mean={np.mean(all_channel_stats['neg_contrib_sum']):.6f}")
    log(f"  Positive vs negative interaction sums:")
    if all_channel_stats["pos_interaction_sum"]:
        log(f"    pos_interaction_sum: mean={np.mean(all_channel_stats['pos_interaction_sum']):.6f}")
        log(f"    neg_interaction_sum: mean={np.mean(all_channel_stats['neg_interaction_sum']):.6f}")

    # ======================================================================
    # PART B: Full Circuit Closure (3 pairs for detailed analysis)
    # ======================================================================
    log(f"\n{'='*70}")
    log(f"PART B: Full Circuit Closure")
    log(f"{'='*70}")
    
    closure_results = {}
    n_closure_pairs = 6
    
    for pidx, (obj, target, competitor) in enumerate(TEST_PAIRS[:n_closure_pairs]):
        tid_t = get_token_id(tokenizer, target)
        tid_c = get_token_id(tokenizer, competitor)
        if tid_t is None or tid_c is None: continue
        
        direction = W_U[tid_t] - W_U[tid_c]
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-10: continue
        direction_normed = direction / dir_norm
        
        clean_prompt = f"The {obj}"
        
        log(f"  Processing pair {pidx+1}: {obj}-{target}-{competitor}")
        
        circuit = full_circuit_closure(model, tokenizer, device, model_name, direction_normed, clean_prompt, CORRUPTED_BASELINE)
        
        final_binding = circuit["final_binding"]
        attn_sum = sum(v for v in circuit["attn_contribs"].values())
        mlp_sum = sum(v for v in circuit["mlp_contribs"].values())
        total_circuit = attn_sum + mlp_sum
        
        # Binding layer contributions
        binding_mlp_sum = sum(circuit["mlp_contribs"].get(li, 0) for li in binding_layers)
        binding_attn_sum = sum(circuit["attn_contribs"].get(li, 0) for li in binding_layers)
        non_binding_mlp_sum = mlp_sum - binding_mlp_sum
        non_binding_attn_sum = attn_sum - binding_attn_sum
        
        # Identify top contributing layers
        all_layer_contribs = {}
        for l in range(n_layers):
            a = circuit["attn_contribs"].get(l, 0)
            m = circuit["mlp_contribs"].get(l, 0)
            all_layer_contribs[l] = {"attn": a, "mlp": m, "total": a + m}
        
        # Sort by absolute contribution
        sorted_layers = sorted(all_layer_contribs.items(), key=lambda x: abs(x[1]["total"]), reverse=True)
        top5_layers = sorted_layers[:5]
        
        # Compute closure ratios
        circuit_closure = total_circuit / max(abs(final_binding), 1e-10)
        mlp_only_closure = mlp_sum / max(abs(final_binding), 1e-10)
        binding_mlp_closure = binding_mlp_sum / max(abs(final_binding), 1e-10)
        
        pair_key = f"{obj}-{target}-{competitor}"
        closure_results[pair_key] = {
            "final_binding": final_binding,
            "attn_sum": attn_sum,
            "mlp_sum": mlp_sum,
            "total_circuit": total_circuit,
            "binding_mlp_sum": binding_mlp_sum,
            "binding_attn_sum": binding_attn_sum,
            "non_binding_mlp_sum": non_binding_mlp_sum,
            "non_binding_attn_sum": non_binding_attn_sum,
            "circuit_closure": circuit_closure,
            "mlp_only_closure": mlp_only_closure,
            "binding_mlp_closure": binding_mlp_closure,
            "top5_layers": [(l, d) for l, d in top5_layers],
            "layer_contribs": {str(l): d for l, d in all_layer_contribs.items()},
        }
        
        log(f"    Final binding: {final_binding:.4f}")
        log(f"    Circuit total: {total_circuit:.4f} (attn={attn_sum:.4f}, mlp={mlp_sum:.4f})")
        log(f"    Binding MLP: {binding_mlp_sum:.4f}, Non-binding MLP: {non_binding_mlp_sum:.4f}")
        log(f"    Circuit closure: {circuit_closure:.4f}")
        log(f"    MLP-only closure: {mlp_only_closure:.4f}")
        log(f"    Binding-MLP closure: {binding_mlp_closure:.4f}")
        top5_str = ", ".join(f"L{l}({d['total']:.4f})" for l, d in top5_layers)
        log(f"    Top 5 layers: {top5_str}")
        
        del circuit; gc.collect(); torch.cuda.empty_cache()
    
    # Part B Summary
    log(f"\n  PART B Summary:")
    if closure_results:
        closures = [v["circuit_closure"] for v in closure_results.values()]
        mlp_closures = [v["mlp_only_closure"] for v in closure_results.values()]
        bmlp_closures = [v["binding_mlp_closure"] for v in closure_results.values()]
        final_b = [v["final_binding"] for v in closure_results.values()]
        attn_s = [v["attn_sum"] for v in closure_results.values()]
        mlp_s = [v["mlp_sum"] for v in closure_results.values()]
        total_s = [v["total_circuit"] for v in closure_results.values()]
        
        log(f"  Mean circuit closure:  {np.mean(closures):.4f} ± {np.std(closures):.4f}")
        log(f"  Mean MLP-only closure: {np.mean(mlp_closures):.4f} ± {np.std(mlp_closures):.4f}")
        log(f"  Mean binding-MLP closure: {np.mean(bmlp_closures):.4f} ± {np.std(bmlp_closures):.4f}")
        log(f"  Mean attn contribution: {np.mean(attn_s):.4f}")
        log(f"  Mean mlp contribution:  {np.mean(mlp_s):.4f}")
        log(f"  Mean total circuit:     {np.mean(total_s):.4f}")
        log(f"  Mean final binding:     {np.mean(final_b):.4f}")
        
        # Attn vs MLP fraction
        mean_abs_attn = np.mean([abs(v) for v in attn_s])
        mean_abs_mlp = np.mean([abs(v) for v in mlp_s])
        total_abs = mean_abs_attn + mean_abs_mlp
        if total_abs > 0:
            log(f"  Attn fraction of |contrib|: {mean_abs_attn/total_abs:.3f}")
            log(f"  MLP fraction of |contrib|:  {mean_abs_mlp/total_abs:.3f}")
        
        # Binding layer vs non-binding layer
        bmlp = [v["binding_mlp_sum"] for v in closure_results.values()]
        nbmlp = [v["non_binding_mlp_sum"] for v in closure_results.values()]
        log(f"  Binding MLP sum:     mean={np.mean(bmlp):.4f}")
        log(f"  Non-binding MLP sum: mean={np.mean(nbmlp):.4f}")

    # ======================================================================
    # PART C: Interaction Physical Meaning — Channel-Level Analysis
    # ======================================================================
    log(f"\n{'='*70}")
    log(f"PART C: Interaction Physical Meaning")
    log(f"{'='*70}")
    
    interaction_channel_results = {}
    
    for pidx, (obj, target, competitor) in enumerate(TEST_PAIRS[:6]):
        tid_t = get_token_id(tokenizer, target)
        tid_c = get_token_id(tokenizer, competitor)
        if tid_t is None or tid_c is None: continue
        
        direction = W_U[tid_t] - W_U[tid_c]
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-10: continue
        direction_normed = direction / dir_norm
        
        clean_prompt = f"The {obj}"
        clean_caps = capture_mlp_internals(model, tokenizer, device, clean_prompt, binding_layers, n_layers)
        corrupt_caps = capture_mlp_internals(model, tokenizer, device, CORRUPTED_BASELINE, binding_layers, n_layers)
        
        pair_key = f"{obj}-{target}-{competitor}"
        interaction_channel_results[pair_key] = {}
        
        for li in binding_layers:
            mw = mlp_weights[li]
            W_down = mw["W_down"]; d_ff = mw["d_ff"]
            if W_down is None: continue
            gk = f"gate_{li}"; uk = f"up_{li}"
            if gk not in clean_caps or gk not in corrupt_caps: continue
            cg = clean_caps[gk][:d_ff]; crg = corrupt_caps[gk][:d_ff]
            cu = clean_caps.get(uk, np.ones(d_ff))[:d_ff]
            cru = corrupt_caps.get(uk, np.ones(d_ff))[:d_ff]
            
            min_d = min(d_ff, W_down.shape[1], cg.shape[0])
            Wd = W_down[:, :min_d]
            
            gsc = silu_np(cg[:min_d]); gsr = silu_np(crg[:min_d])
            uc = cu[:min_d]; ur = cru[:min_d]
            
            # Channel projection
            channel_proj = Wd.T @ direction_normed  # (d_ff,)
            
            # Per-channel decomposition
            gate_diff = gsc - gsr
            up_diff = uc - ur
            
            per_channel_ia = channel_proj * gate_diff * up_diff
            per_channel_gm = channel_proj * ((gsc + gsr) / 2) * up_diff
            per_channel_um = channel_proj * gate_diff * ((uc + ur) / 2)
            
            # Classify channels by gate_diff and up_diff signs
            # 4 quadrants: gate+up+, gate+up-, gate-up+, gate-up-
            gg_uu = (gate_diff > 0) & (up_diff > 0)  # both increase
            gg_ud = (gate_diff > 0) & (up_diff < 0)  # gate up, up down
            gd_uu = (gate_diff < 0) & (up_diff > 0)  # gate down, up up
            gd_ud = (gate_diff < 0) & (up_diff < 0)  # both decrease
            
            def quadrant_stats(mask, label):
                if np.sum(mask) == 0:
                    return {"label": label, "count": 0, "ia_sum": 0, "gm_sum": 0, "um_sum": 0}
                return {
                    "label": label,
                    "count": int(np.sum(mask)),
                    "ia_sum": float(np.sum(per_channel_ia[mask])),
                    "gm_sum": float(np.sum(per_channel_gm[mask])),
                    "um_sum": float(np.sum(per_channel_um[mask])),
                    "ia_mean": float(np.mean(per_channel_ia[mask])),
                    "channel_proj_mean": float(np.mean(channel_proj[mask])),
                    "gate_diff_mean": float(np.mean(gate_diff[mask])),
                    "up_diff_mean": float(np.mean(up_diff[mask])),
                }
            
            quadrants = [
                quadrant_stats(gg_uu, "gate+up+"),
                quadrant_stats(gg_ud, "gate+up-"),
                quadrant_stats(gd_uu, "gate-up+"),
                quadrant_stats(gd_ud, "gate-up-"),
            ]
            
            # SiLU curvature analysis
            # SiLU'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
            # At point x, the nonlinearity "curvature" determines how much interaction is generated
            # For clean input x_c and corrupt x_r:
            # SiLU(x_c) - SiLU(x_r) ≈ SiLU'(x_mid) * (x_c - x_r) [linear approx]
            # Interaction = (SiLU(x_c) - SiLU(x_r)) * (u_c - u_r) - gate_main_like_term
            # = [SiLU(x_c) - SiLU(x_r) - SiLU'(x_mid) * (x_c - x_r)] * (u_c - u_r) + ...
            # The residual is the nonlinear part
            
            # Compute SiLU nonlinearity residual per channel
            sigmoid_c = 1.0 / (1.0 + np.exp(-np.clip(cg[:min_d], -20, 20)))
            sigmoid_r = 1.0 / np.exp(-np.clip(crg[:min_d], -20, 20))
            # Actually: sigmoid(x) = 1/(1+exp(-x))
            sigmoid_c = 1.0 / (1.0 + np.exp(-np.clip(cg[:min_d], -20, 20)))
            sigmoid_r = 1.0 / (1.0 + np.exp(-np.clip(crg[:min_d], -20, 20)))
            
            silu_deriv_c = sigmoid_c + cg[:min_d] * sigmoid_c * (1 - sigmoid_c)
            silu_deriv_r = sigmoid_r + crg[:min_d] * sigmoid_r * (1 - sigmoid_r)
            silu_deriv_mid = (silu_deriv_c + silu_deriv_r) / 2
            
            # Linear approximation of gate difference
            gate_diff_linear = silu_deriv_mid * (cg[:min_d] - crg[:min_d])
            gate_diff_actual = gsc - gsr
            gate_nonlinear_residual = gate_diff_actual - gate_diff_linear
            
            # Nonlinear interaction = nonlinear_gate_part * up_diff
            nonlinear_interaction = channel_proj * gate_nonlinear_residual * up_diff
            linear_gate_main = channel_proj * gate_diff_linear * ((uc + ur) / 2)
            
            interaction_channel_results[pair_key][str(li)] = {
                "total_interaction": float(np.sum(per_channel_ia)),
                "total_gate_main": float(np.sum(per_channel_gm)),
                "total_up_main": float(np.sum(per_channel_um)),
                "nonlinear_interaction_sum": float(np.sum(nonlinear_interaction)),
                "linear_gate_main_sum": float(np.sum(linear_gate_main)),
                "nonlinear_fraction": float(np.sum(np.abs(nonlinear_interaction)) / max(np.sum(np.abs(per_channel_ia)), 1e-10)),
                "quadrants": quadrants,
            }
        
        del clean_caps, corrupt_caps; gc.collect(); torch.cuda.empty_cache()
    
    # Part C Summary
    log(f"\n  PART C Summary:")
    all_nl_frac = []
    all_quad_counts = defaultdict(list)
    all_quad_ia = defaultdict(list)
    
    for pair_data in interaction_channel_results.values():
        for layer_data in pair_data.values():
            if "nonlinear_fraction" in layer_data:
                all_nl_frac.append(layer_data["nonlinear_fraction"])
            for q in layer_data.get("quadrants", []):
                label = q["label"]
                all_quad_counts[label].append(q["count"])
                all_quad_ia[label].append(q["ia_sum"])
    
    if all_nl_frac:
        log(f"  Nonlinear fraction of interaction: {np.mean(all_nl_frac):.4f} ± {np.std(all_nl_frac):.4f}")
    
    log(f"  Quadrant analysis (gate_diff × up_diff):")
    for label in ["gate+up+", "gate+up-", "gate-up+", "gate-up-"]:
        if all_quad_counts[label]:
            log(f"    {label}: count={np.mean(all_quad_counts[label]):.1f}, ia_sum={np.mean(all_quad_ia[label]):.6f}")

    # ======================================================================
    # Save Results
    # ======================================================================
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        elif isinstance(obj, (np.floating,)): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)): return [convert(v) for v in obj]
        return obj



    save_data = convert({
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "part_a_wdown_structure": {
            "per_pair": wdown_results,
            "aggregated_stats": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in all_channel_stats.items() if v},
        },
        "part_b_full_closure": {
            "per_pair": closure_results,
            "summary": {
                "mean_circuit_closure": float(np.mean([v["circuit_closure"] for v in closure_results.values()])) if closure_results else 0,
                "mean_mlp_only_closure": float(np.mean([v["mlp_only_closure"] for v in closure_results.values()])) if closure_results else 0,
                "mean_binding_mlp_closure": float(np.mean([v["binding_mlp_closure"] for v in closure_results.values()])) if closure_results else 0,
            } if closure_results else {},
        },
        "part_c_interaction": {
            "per_pair": interaction_channel_results,
            "mean_nonlinear_fraction": float(np.mean(all_nl_frac)) if all_nl_frac else 0,
        },
    })

    os.makedirs("results/phase347_wdown_structure_closure", exist_ok=True)
    out_path = f"results/phase347_wdown_structure_closure/{model_name}_phase347.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    log(f"\nResults saved to {out_path}")

    del model, W_U, mlp_weights; gc.collect(); torch.cuda.empty_cache()
    total_time = time.time() - t0
    log(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f}min)")


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    if model_name not in MODEL_CONFIGS: log(f"Unknown model: {model_name}"); sys.exit(1)
    run_experiment(model_name)
    log("Phase 347 complete!")
