"""
Phase 283: Deep Analysis — Complete the Causal Picture
=======================================================
Key improvements over Phase 282:

Block A: Safetensors deep layer weight extraction (GLM4/DS7B)
  → Fill the gap where device_map="auto" puts deep layers on CPU (meta tensor)

Block B: Qwen3 L18 per-head VALUE-dominant head identification
  → L18 is the only VALUE-dominant layer in Qwen3; find which heads drive this

Block C: GLM4 L0 partial RoPE decomposition (rotated 64 vs non-rotated 64)
  → GLM4 L0 is uniquely VALUE-dominant; test if partial RoPE explains this

Block D: Extended sentence types (negation, passive, conditional, recursive, translation)
  → Expand from 52 SVO pairs to 80+ pairs covering linguistic functions

Block E: Final Component Contribution Matrix
  → Function x Layer x Component → causal contribution

Usage:
  python tests/glm5/phase283_deep_analysis.py qwen3
  python tests/glm5/phase283_deep_analysis.py glm4
  python tests/glm5/phase283_deep_analysis.py deepseek7b
"""
import sys, os, json, gc, time, warnings, glob as fileglob
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from model_utils import MODEL_CONFIGS, get_model_info, get_layers, release_model

RESULT_DIR = Path("results/phase283_deep_analysis")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = Path("tmp")
TMP_DIR.mkdir(parents=True, exist_ok=True)

_log_file = None
def log_time(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_file:
        with open(_log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

# =============================================================================
# RoPE Implementation (same as Phase 282)
# =============================================================================

def compute_rope_cos_sin(seq_len, head_dim, rotary_dim, base=10000.0):
    d = np.arange(rotary_dim // 2)
    freq = base ** (-2.0 * d / rotary_dim)
    pos = np.arange(seq_len)
    angles = np.outer(pos, freq)
    angles_full = np.concatenate([angles, angles], axis=-1)
    cos = np.cos(angles_full).astype(np.float32)
    sin = np.sin(angles_full).astype(np.float32)
    if rotary_dim < head_dim:
        pad_dim = head_dim - rotary_dim
        cos = np.concatenate([cos, np.ones((seq_len, pad_dim), dtype=np.float32)], axis=-1)
        sin = np.concatenate([sin, np.zeros((seq_len, pad_dim), dtype=np.float32)], axis=-1)
    return cos, sin

def rotate_every_two(x):
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return np.concatenate([-x2, x1], axis=-1)

def apply_rope(x, cos, sin, rotary_dim):
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    cos_rot = cos[..., :rotary_dim]
    sin_rot = sin[..., :rotary_dim]
    x_flip = rotate_every_two(x_rot)
    x_rotated = x_rot * cos_rot + x_flip * sin_rot
    if rotary_dim < x.shape[-1]:
        return np.concatenate([x_rotated, x_pass], axis=-1)
    return x_rotated

MODEL_ROPE_CONFIGS = {
    "qwen3": {"rope_theta": 1000000.0, "rotary_dim": 128, "has_qk_norm": True},
    "glm4": {"rope_theta": 10000.0, "rotary_dim": 64, "has_qk_norm": False},
    "deepseek7b": {"rope_theta": 10000.0, "rotary_dim": 128, "has_qk_norm": False},
}

# =============================================================================
# Model Loading
# =============================================================================

def load_model_bf16_flash(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    log_time(f"Loading {model_name} (bf16 + flash_attention_2)...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    attn_impl = "flash_attention_2"
    try:
        import flash_attn
        log_time("  flash_attn available")
    except ImportError:
        attn_impl = "eager"
        log_time("  flash_attn not found, using eager")
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation=attn_impl,
    )
    
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log_time(f"  {model_name} loaded: device={device}, GPU={gpu_mem:.2f}GB, attn={attn_impl}")
    return model, tokenizer, device


# =============================================================================
# Utility functions
# =============================================================================

def apply_input_ln(hidden_np, layer, eps=1e-6):
    ln = None
    for ln_name in ["input_layernorm", "ln_1", "layernorm"]:
        if hasattr(layer, ln_name):
            ln = getattr(layer, ln_name)
            break
    if ln is None:
        return hidden_np
    try:
        w = ln.weight.detach().cpu().float().numpy()
    except (NotImplementedError, RuntimeError):
        return hidden_np
    rms = np.sqrt(np.mean(hidden_np ** 2, axis=-1, keepdims=True) + eps)
    return hidden_np * w / rms

def softmax_np(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

# =============================================================================
# Block A: Safetensors weight extraction for deep layers (GLM4/DS7B)
# =============================================================================

def extract_layer_weights_from_safetensors(model_name, layer_idx, n_heads_q, n_heads_kv):
    """
    Directly read Q/K/V/O weights from safetensors files.
    This bypasses the meta-tensor issue for CPU-offloaded layers.
    
    Computes d_head from actual weight dimensions (not config).
    
    Returns dict with W_q, W_k, W_v, W_o as numpy arrays, or None if failed.
    """
    cfg = MODEL_CONFIGS[model_name]
    model_path = cfg["path"]
    
    # Find all safetensors files
    sf_files = sorted(fileglob.glob(os.path.join(model_path, "*.safetensors")))
    if not sf_files:
        sf_files = sorted(fileglob.glob(os.path.join(model_path, "model-*.safetensors")))
    
    # Key patterns
    key_prefix = f"model.layers.{layer_idx}.self_attn."
    q_key = key_prefix + "q_proj.weight"
    k_key = key_prefix + "k_proj.weight"
    v_key = key_prefix + "v_proj.weight"
    o_key = key_prefix + "o_proj.weight"
    
    from safetensors import safe_open
    
    W_q, W_k, W_v, W_o = None, None, None, None
    
    for sf_file in sf_files:
        with safe_open(sf_file, framework='pt', device='cpu') as sf:
            keys = set(sf.keys())
            if q_key in keys and W_q is None:
                W_q = sf.get_tensor(q_key).float().numpy()
            if k_key in keys and W_k is None:
                W_k = sf.get_tensor(k_key).float().numpy()
            if v_key in keys and W_v is None:
                W_v = sf.get_tensor(v_key).float().numpy()
            if o_key in keys and W_o is None:
                W_o = sf.get_tensor(o_key).float().numpy()
        
        if all(x is not None for x in [W_q, W_k, W_v, W_o]):
            break
    
    if all(x is not None for x in [W_q, W_k, W_v, W_o]):
        # Compute d_head from actual weight dimensions
        q_out_dim = W_q.shape[0]  # n_heads_q * d_head
        d_head = q_out_dim // n_heads_q
        gqa_group = n_heads_q // max(n_heads_kv, 1)
        return {
            "W_q": W_q, "W_k": W_k, "W_v": W_v, "W_o": W_o,
            "n_heads_q": n_heads_q, "n_heads_kv": n_heads_kv,
            "d_head": d_head, "gqa_group": gqa_group,
        }
    return None


def detect_model_head_config(model, model_name):
    """Detect head configuration from the model, even for meta-tensor layers.
    Note: d_head is NOT returned — it depends on actual weight dimensions, not config."""
    config = model.config
    n_heads_q = getattr(config, 'num_attention_heads', 32)
    n_heads_kv = getattr(config, 'num_key_value_heads', n_heads_q)
    
    # d_model from config
    d_model = getattr(config, 'hidden_size', 
                getattr(config, 'd_model', 4096))
    
    return n_heads_q, n_heads_kv, d_model


def block_a_deep_layer_weights(model, tokenizer, device, model_info, model_name, all_pairs):
    """
    Use safetensors to load Q/K/V/O weights for ALL layers (including deep CPU-offloaded ones).
    Then run the same manual RoPE patching on all layers.
    """
    n_layers = model_info.n_layers
    rope_cfg = MODEL_ROPE_CONFIGS[model_name]
    rope_base = rope_cfg["rope_theta"]
    rotary_dim = rope_cfg["rotary_dim"]
    has_qk_norm = rope_cfg["has_qk_norm"]
    
    n_heads_q, n_heads_kv, d_model_cfg = detect_model_head_config(model, model_name)
    
    log_time(f"\n{'='*60}")
    log_time(f"Block A: Safetensors Deep Layer Weight Patching (ALL layers)")
    log_time(f"  Head config: Q={n_heads_q}, KV={n_heads_kv}, d_model={d_model_cfg}")
    log_time(f"  Attempting all {n_layers} layers via safetensors...")
    
    # Extract ALL layer weights via safetensors (d_head computed from actual weights)
    all_layer_weights = {}
    skipped = []
    for li in range(n_layers):
        w = extract_layer_weights_from_safetensors(model_name, li, n_heads_q, n_heads_kv)
        if w is not None:
            all_layer_weights[li] = w
        else:
            skipped.append(li)
    
    n_loaded = len(all_layer_weights)
    log_time(f"  Loaded {n_loaded}/{n_layers} layers via safetensors, skipped: {skipped}")
    
    if n_loaded == 0:
        log_time("  Block A: No safetensors weights found, aborting")
        return None
    
    # Also get layer objects for LN and QK norm
    layers = get_layers(model)
    
    # Pre-compute QK norm weights if needed
    qk_norms = {}
    if has_qk_norm:
        for li in range(n_layers):
            if li >= len(layers):
                continue
            layer = layers[li]
            sa = layer.self_attn
            if hasattr(sa, 'q_norm'):
                try:
                    q_norm_w = sa.q_norm.weight.detach().cpu().float().numpy()
                    k_norm_w = sa.k_norm.weight.detach().cpu().float().numpy()
                    qk_norms[li] = {"q_norm": q_norm_w, "k_norm": k_norm_w}
                except (NotImplementedError, RuntimeError):
                    pass
    
    # Determine which layers to sample (all if <15, otherwise 15 evenly-spaced)
    if n_loaded <= 15:
        sample_layers = sorted(all_layer_weights.keys())
    else:
        step = max(1, n_loaded // 14)
        sample_layers = list(range(0, n_loaded, step)) + [n_loaded - 1]
        sample_layers = sorted(set(li for li in sample_layers if li in all_layer_weights))
    
    log_time(f"  Testing {len(sample_layers)} layers: {sample_layers[:5]}...{sample_layers[-3:]}")
    
    # Now run manual RoPE patching on all sampled layers
    # We hook hidden states for ALL needed layers (using original forward)
    layers_to_hook = sample_layers
    
    def run_with_all_hooks(sentence):
        """Capture hidden states for all sample layers."""
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        captured = {}
        
        def make_hook(key):
            def hook(module, input_t, output_t):
                if isinstance(input_t, tuple) and len(input_t) > 0:
                    captured[key] = input_t[0].detach().float().cpu()
            return hook
        
        hooks = []
        for li in layers_to_hook:
            if li < len(layers):
                hooks.append(layers[li].register_forward_hook(make_hook(f"L{li}")))
        
        with torch.no_grad():
            try:
                _ = model(**inputs)
            except Exception as e:
                log_time(f"  Forward failed for '{sentence[:40]}...': {e}")
        
        for h in hooks:
            h.remove()
        return captured, inputs.input_ids.cpu()
    
    results = {}
    
    for pair_idx, pair in enumerate(all_pairs):
        pname = pair["name"]
        sent_a, sent_b = pair["A"], pair["B"]
        category = pair.get("category", "unknown")
        
        # Capture hidden states
        hA_data, _ = run_with_all_hooks(sent_a)
        hB_data, _ = run_with_all_hooks(sent_b)
        
        pair_results = {"name": pname, "category": category, "layers": {}}
        
        for li in sample_layers:
            key = f"L{li}"
            if li not in all_layer_weights:
                continue
            if key not in hA_data or key not in hB_data:
                continue
            if li >= len(layers):
                continue
            
            w = all_layer_weights[li]
            hA_pre = hA_data[key][0].numpy()
            hB_pre = hB_data[key][0].numpy()
            
            # Apply LN
            layer = layers[li]
            hA = apply_input_ln(hA_pre, layer)
            hB = apply_input_ln(hB_pre, layer)
            seqA, seqB = hA.shape[0], hB.shape[0]
            min_seq = min(seqA, seqB)
            
            W_q = w["W_q"]; W_k = w["W_k"]; W_v = w["W_v"]; W_o = w["W_o"]
            n_q = w["n_heads_q"]; n_kv = w["n_heads_kv"]
            d_head_local = w["d_head"]; gqa = w["gqa_group"]
            
            # Compute Q, K, V
            QA = hA[:min_seq] @ W_q.T
            KA = hA[:min_seq] @ W_k.T
            VA = hA[:min_seq] @ W_v.T
            QB = hB[:min_seq] @ W_q.T
            KB = hB[:min_seq] @ W_k.T
            VB = hB[:min_seq] @ W_v.T
            
            # Reshape to multi-head
            def to_mha(X, n_h, d_h):
                s = X.shape[0]
                return X.reshape(s, n_h, d_h).transpose(1, 0, 2)  # [n_h, s, d_h]
            
            QA_mha = to_mha(QA, n_q, d_head_local)
            KA_mha_raw = to_mha(KA, n_kv, d_head_local)
            VA_mha_raw = to_mha(VA, n_kv, d_head_local)
            QB_mha = to_mha(QB, n_q, d_head_local)
            KB_mha_raw = to_mha(KB, n_kv, d_head_local)
            VB_mha_raw = to_mha(VB, n_kv, d_head_local)
            
            # QK Norm
            if has_qk_norm and li in qk_norms:
                qn = qk_norms[li]["q_norm"]; kn = qk_norms[li]["k_norm"]
                eps = 1e-6
                for h in range(n_q):
                    QA_mha[h] = QA_mha[h] * qn / np.sqrt(np.mean(QA_mha[h]**2) + eps)
                    QB_mha[h] = QB_mha[h] * qn / np.sqrt(np.mean(QB_mha[h]**2) + eps)
                for h in range(n_kv):
                    KA_mha_raw[h] = KA_mha_raw[h] * kn / np.sqrt(np.mean(KA_mha_raw[h]**2) + eps)
                    KB_mha_raw[h] = KB_mha_raw[h] * kn / np.sqrt(np.mean(KB_mha_raw[h]**2) + eps)
            
            # GQA expand
            if gqa > 1:
                KA_mha = np.repeat(KA_mha_raw, gqa, axis=0)
                VA_mha = np.repeat(VA_mha_raw, gqa, axis=0)
                KB_mha = np.repeat(KB_mha_raw, gqa, axis=0)
                VB_mha = np.repeat(VB_mha_raw, gqa, axis=0)
            else:
                KA_mha = KA_mha_raw; VA_mha = VA_mha_raw
                KB_mha = KB_mha_raw; VB_mha = VB_mha_raw
            
            # Apply RoPE
            rope_cos, rope_sin = compute_rope_cos_sin(min_seq, d_head_local, rotary_dim, rope_base)
            for h in range(n_q):
                QA_mha[h] = apply_rope(QA_mha[h], rope_cos, rope_sin, rotary_dim)
                QB_mha[h] = apply_rope(QB_mha[h], rope_cos, rope_sin, rotary_dim)
                KA_mha[h] = apply_rope(KA_mha[h], rope_cos, rope_sin, rotary_dim)
                KB_mha[h] = apply_rope(KB_mha[h], rope_cos, rope_sin, rotary_dim)
            
            # Attention weights
            scale = np.sqrt(d_head_local)
            AA = np.zeros((n_q, min_seq, min_seq))
            AB = np.zeros((n_q, min_seq, min_seq))
            for h in range(n_q):
                AA[h] = softmax_np(QA_mha[h] @ KA_mha[h].T / scale)
                AB[h] = softmax_np(QB_mha[h] @ KB_mha[h].T / scale)
            
            # Mixed computations
            mixed_AW_BV = np.zeros((n_q, min_seq, d_head_local))
            mixed_BW_AV = np.zeros((n_q, min_seq, d_head_local))
            pure_AA_VA = np.zeros((n_q, min_seq, d_head_local))
            pure_AB_VB = np.zeros((n_q, min_seq, d_head_local))
            
            for h in range(n_q):
                mixed_AW_BV[h] = AA[h] @ VB_mha[h]
                mixed_BW_AV[h] = AB[h] @ VA_mha[h]
                pure_AA_VA[h] = AA[h] @ VA_mha[h]
                pure_AB_VB[h] = AB[h] @ VB_mha[h]
            
            # Project through W_o
            def flatten_mha(X, n_h, s, d_h):
                return X.transpose(1, 0, 2).reshape(s, n_h * d_h)
            
            mixed_AW_BV_o = flatten_mha(mixed_AW_BV, n_q, min_seq, d_head_local) @ W_o.T
            mixed_BW_AV_o = flatten_mha(mixed_BW_AV, n_q, min_seq, d_head_local) @ W_o.T
            pure_AA_VA_o = flatten_mha(pure_AA_VA, n_q, min_seq, d_head_local) @ W_o.T
            pure_AB_VB_o = flatten_mha(pure_AB_VB, n_q, min_seq, d_head_local) @ W_o.T
            
            total_gap = float(np.linalg.norm(pure_AA_VA_o - pure_AB_VB_o))
            weight_effect_raw = float(np.linalg.norm(mixed_BW_AV_o - pure_AB_VB_o))
            value_effect_raw = float(np.linalg.norm(mixed_AW_BV_o - pure_AB_VB_o))
            
            if total_gap > 1e-10:
                weight_effect = weight_effect_raw / total_gap
                value_effect = value_effect_raw / total_gap
            else:
                weight_effect = value_effect = 0.0
            
            # Per-head breakdown (use head-specific W_o slice)
            per_head_weight = []
            per_head_value = []
            for h in range(n_q):
                # Extract this head's portion of W_o: [d_model, d_head]
                W_o_h = W_o[:, h*d_head_local:(h+1)*d_head_local]
                # Single head output projected through its W_o slice
                pw_o = flatten_mha(pure_AB_VB[h:h+1], 1, min_seq, d_head_local) @ W_o_h.T
                if total_gap > 1e-10:
                    bw_mix = flatten_mha(mixed_BW_AV[h:h+1], 1, min_seq, d_head_local) @ W_o_h.T
                    aw_mix = flatten_mha(mixed_AW_BV[h:h+1], 1, min_seq, d_head_local) @ W_o_h.T
                    wh = float(np.linalg.norm(bw_mix - pw_o)) / total_gap
                    vh = float(np.linalg.norm(aw_mix - pw_o)) / total_gap
                else:
                    wh = vh = 0.0
                per_head_weight.append(wh)
                per_head_value.append(vh)
            
            pair_results["layers"][str(li)] = {
                "total_gap": total_gap,
                "weight_effect": weight_effect,
                "value_effect": value_effect,
                "content_dominates": value_effect > weight_effect,
                "per_head_weight": per_head_weight,
                "per_head_value": per_head_value,
                "head_content_dominates": [v > w for w, v in zip(per_head_weight, per_head_value)],
            }
        
        results[pname] = pair_results
        
        if (pair_idx + 1) % 20 == 0:
            log_time(f"  Block A: {pair_idx+1}/{len(all_pairs)} pairs done")
    
    # Aggregate
    agg = {"per_layer": {}, "per_category": defaultdict(lambda: defaultdict(list)),
           "per_head_per_layer": {}}
    
    for li in sample_layers:
        key = str(li)
        w_effs = []; v_effs = []; doms = []
        head_w = defaultdict(list); head_v = defaultdict(list)
        
        for pname, pr in results.items():
            if key in pr.get("layers", {}):
                lr = pr["layers"][key]
                w_effs.append(lr["weight_effect"])
                v_effs.append(lr["value_effect"])
                doms.append(1.0 if lr["content_dominates"] else 0.0)
                
                for h in range(len(lr.get("per_head_weight", []))):
                    head_w[h].append(lr["per_head_weight"][h])
                    head_v[h].append(lr["per_head_value"][h])
                
                cat = pr.get("category", "unknown")
                agg["per_category"][cat]["weight"].append(lr["weight_effect"])
                agg["per_category"][cat]["value"].append(lr["value_effect"])
        
        if w_effs:
            agg["per_layer"][key] = {
                "weight_effect_mean": float(np.mean(w_effs)),
                "value_effect_mean": float(np.mean(v_effs)),
                "content_dominance_rate": float(np.mean(doms)),
            }
        
        if head_w:
            agg["per_head_per_layer"][key] = {}
            for h in sorted(head_w.keys()):
                agg["per_head_per_layer"][key][str(h)] = {
                    "weight_effect": float(np.mean(head_w[h])),
                    "value_effect": float(np.mean(head_v[h])),
                    "content_dominates": float(np.mean(head_v[h])) > float(np.mean(head_w[h])),
                }
    
    log_time(f"\n  Block A Summary ({model_name}, {n_loaded} layers):")
    log_time(f"  {'Layer':>5} {'wt_eff':>8} {'val_eff':>8} {'dom%':>6} {'winner':>20}")
    for li in sorted(agg["per_layer"].keys(), key=int):
        a = agg["per_layer"][li]
        winner = "VALUE>>weight" if a["content_dominance_rate"] > 0.5 else "WEIGHT>>value"
        log_time(f"  L{int(li):>4} {a['weight_effect_mean']:8.3f} {a['value_effect_mean']:8.3f} "
                 f"{a['content_dominance_rate']:6.2f} {winner:>20}")
    
    return {"pairs": results, "aggregate": agg}


# =============================================================================
# Block B: Qwen3 L18 Per-Head VALUE-dominant Analysis
# =============================================================================

def block_b_per_head_value_analysis(model, tokenizer, device, model_info, model_name, all_pairs):
    """
    For Qwen3 only: detailed per-head analysis of layer L18 (unique VALUE-dominant layer).
    
    Uses safetensors-extracted weights + manual RoPE, then decomposes effects per attention head.
    """
    if model_name != "qwen3":
        log_time("Block B: Skipped (Qwen3 only)")
        return None
    
    n_layers = model_info.n_layers
    rope_cfg = MODEL_ROPE_CONFIGS[model_name]
    rope_base = rope_cfg["rope_theta"]
    rotary_dim = rope_cfg["rotary_dim"]
    
    n_heads_q, n_heads_kv, _ = detect_model_head_config(model, model_name)
    
    target_layers = [18]  # L18 is the only VALUE-dominant layer
    
    log_time(f"\n{'='*60}")
    log_time(f"Block B: Qwen3 L18 Per-Head VALUE-Dominant Analysis")
    log_time(f"  Layers: {target_layers}, Heads: {n_heads_q}, KV: {n_heads_kv}")
    
    # Load weights via safetensors (d_head from actual weight dimensions)
    layer_weights_sf = {}
    for li in target_layers:
        w = extract_layer_weights_from_safetensors(model_name, li, n_heads_q, n_heads_kv)
        if w:
            layer_weights_sf[li] = w
            log_time(f"  L{li}: loaded via safetensors (GQA group={n_heads_q//n_heads_kv})")
    
    # QK norm
    layers = get_layers(model)
    qk_norms = {}
    for li in target_layers:
        if li < len(layers):
            sa = layers[li].self_attn
            if hasattr(sa, 'q_norm'):
                try:
                    qk_norms[li] = {
                        "q_norm": sa.q_norm.weight.detach().cpu().float().numpy(),
                        "k_norm": sa.k_norm.weight.detach().cpu().float().numpy(),
                    }
                except: pass
    
    # Run on a larger subset: use ALL 52 pairs from Phase 282
    results = {}
    per_head_agg = {h: {"weight": [], "value": [], "content_dom": 0} for h in range(n_heads_q)}
    
    for pair_idx, pair in enumerate(all_pairs):
        pname = pair["name"]
        sent_a, sent_b = pair["A"], pair["B"]
        
        # Capture hidden states for target layers
        inputs_a = tokenizer(sent_a, return_tensors="pt").to(device)
        inputs_b = tokenizer(sent_b, return_tensors="pt").to(device)
        
        hA_data = {}; hB_data = {}
        
        def make_hook(data_dict, key):
            def hook(module, input_t, output_t):
                if isinstance(input_t, tuple) and len(input_t) > 0:
                    data_dict[key] = input_t[0].detach().float().cpu()
            return hook
        
        hooks = []
        for li in target_layers:
            if li < len(layers):
                hooks.append(layers[li].register_forward_hook(make_hook(hA_data, f"L{li}")))
        with torch.no_grad():
            _ = model(**inputs_a)
        for h in hooks:
            h.remove()
        
        hooks = []
        for li in target_layers:
            if li < len(layers):
                hooks.append(layers[li].register_forward_hook(make_hook(hB_data, f"L{li}")))
        with torch.no_grad():
            _ = model(**inputs_b)
        for h in hooks:
            h.remove()
        
        pair_results = {"name": pname, "category": pair["category"], "layers": {}}
        
        for li in target_layers:
            key = f"L{li}"
            if li not in layer_weights_sf:
                continue
            if key not in hA_data or key not in hB_data:
                continue
            
            w = layer_weights_sf[li]
            hA = apply_input_ln(hA_data[key][0].numpy(), layers[li])
            hB = apply_input_ln(hB_data[key][0].numpy(), layers[li])
            seqA, seqB = hA.shape[0], hB.shape[0]
            min_seq = min(seqA, seqB)
            
            W_q = w["W_q"]; W_k = w["W_k"]; W_v = w["W_v"]; W_o = w["W_o"]
            n_q = w["n_heads_q"]; n_kv = w["n_heads_kv"]
            d_h = w["d_head"]; gqa = w["gqa_group"]
            
            QA = (hA[:min_seq] @ W_q.T).reshape(min_seq, n_q, d_h).transpose(1, 0, 2)
            KA_raw = (hA[:min_seq] @ W_k.T).reshape(min_seq, n_kv, d_h).transpose(1, 0, 2)
            VA_raw = (hA[:min_seq] @ W_v.T).reshape(min_seq, n_kv, d_h).transpose(1, 0, 2)
            QB = (hB[:min_seq] @ W_q.T).reshape(min_seq, n_q, d_h).transpose(1, 0, 2)
            KB_raw = (hB[:min_seq] @ W_k.T).reshape(min_seq, n_kv, d_h).transpose(1, 0, 2)
            VB_raw = (hB[:min_seq] @ W_v.T).reshape(min_seq, n_kv, d_h).transpose(1, 0, 2)
            
            # QK Norm
            if li in qk_norms:
                qn = qk_norms[li]["q_norm"]; kn = qk_norms[li]["k_norm"]
                eps = 1e-6
                for h in range(n_q):
                    QA[h] = QA[h] * qn / np.sqrt(np.mean(QA[h]**2) + eps)
                    QB[h] = QB[h] * qn / np.sqrt(np.mean(QB[h]**2) + eps)
                for h in range(n_kv):
                    KA_raw[h] = KA_raw[h] * kn / np.sqrt(np.mean(KA_raw[h]**2) + eps)
                    KB_raw[h] = KB_raw[h] * kn / np.sqrt(np.mean(KB_raw[h]**2) + eps)
            
            # GQA expand
            KA = np.repeat(KA_raw, gqa, axis=0) if gqa > 1 else KA_raw
            VA = np.repeat(VA_raw, gqa, axis=0) if gqa > 1 else VA_raw
            KB = np.repeat(KB_raw, gqa, axis=0) if gqa > 1 else KB_raw
            VB = np.repeat(VB_raw, gqa, axis=0) if gqa > 1 else VB_raw
            
            # RoPE
            cos, sin = compute_rope_cos_sin(min_seq, d_h, rotary_dim, rope_base)
            for h in range(n_q):
                QA[h] = apply_rope(QA[h], cos, sin, rotary_dim)
                QB[h] = apply_rope(QB[h], cos, sin, rotary_dim)
                KA[h] = apply_rope(KA[h], cos, sin, rotary_dim)
                KB[h] = apply_rope(KB[h], cos, sin, rotary_dim)
            
            # Attention maps
            scale = np.sqrt(d_h)
            AA = np.zeros((n_q, min_seq, min_seq))
            AB = np.zeros((n_q, min_seq, min_seq))
            for h in range(n_q):
                AA[h] = softmax_np(QA[h] @ KA[h].T / scale)
                AB[h] = softmax_np(QB[h] @ KB[h].T / scale)
            
            # Per-head effects
            def flatten_h(X, s, d):
                return X.transpose(1, 0, 2).reshape(s, d)
            
            # Full-aggregate outputs (for normalization)
            full_pure_AA = np.zeros((n_q, min_seq, d_h))
            full_pure_AB = np.zeros((n_q, min_seq, d_h))
            full_AW_BV = np.zeros((n_q, min_seq, d_h))
            full_BW_AV = np.zeros((n_q, min_seq, d_h))
            for h in range(n_q):
                full_pure_AA[h] = AA[h] @ VA[h]
                full_pure_AB[h] = AB[h] @ VB[h]
                full_AW_BV[h] = AA[h] @ VB[h]
                full_BW_AV[h] = AB[h] @ VA[h]
            
            pure_AA_o = flatten_h(full_pure_AA, min_seq, n_q * d_h) @ W_o.T
            pure_AB_o = flatten_h(full_pure_AB, min_seq, n_q * d_h) @ W_o.T
            AW_BV_o = flatten_h(full_AW_BV, min_seq, n_q * d_h) @ W_o.T
            BW_AV_o = flatten_h(full_BW_AV, min_seq, n_q * d_h) @ W_o.T
            
            total_gap = float(np.linalg.norm(pure_AA_o - pure_AB_o))
            
            per_head_effects = {}
            for h in range(n_q):
                # Single head contribution
                h_pure_AA = (AA[h:h+1] @ VA[h:h+1]).transpose(1, 0, 2).reshape(min_seq, d_h)
                h_pure_AB = (AB[h:h+1] @ VB[h:h+1]).transpose(1, 0, 2).reshape(min_seq, d_h)
                h_AW_BV = (AA[h:h+1] @ VB[h:h+1]).transpose(1, 0, 2).reshape(min_seq, d_h)
                h_BW_AV = (AB[h:h+1] @ VA[h:h+1]).transpose(1, 0, 2).reshape(min_seq, d_h)
                
                # Project through W_o (use only this head's portion)
                # W_o is [d_model, n_q*d_h]; extract this head's part
                W_o_h = W_o[:, h*d_h:(h+1)*d_h]  # [d_model, d_h]
                
                pure_AA_h = h_pure_AA @ W_o_h.T  # [min_seq, d_model]
                pure_AB_h = h_pure_AB @ W_o_h.T
                AW_BV_h = h_AW_BV @ W_o_h.T
                BW_AV_h = h_BW_AV @ W_o_h.T
                
                if total_gap > 1e-10:
                    wh = float(np.linalg.norm(BW_AV_h - pure_AB_h)) / total_gap
                    vh = float(np.linalg.norm(AW_BV_h - pure_AB_h)) / total_gap
                else:
                    wh = vh = 0.0
                
                per_head_effects[h] = {"weight": wh, "value": vh, "ratio": vh / max(wh, 1e-10)}
                
                per_head_agg[h]["weight"].append(wh)
                per_head_agg[h]["value"].append(vh)
            
            pair_results["layers"][str(li)] = {
                "total_gap": total_gap,
                "per_head": per_head_effects,
            }
        
        results[pname] = pair_results
        
        if (pair_idx + 1) % 10 == 0:
            log_time(f"  Block B: {pair_idx+1}/{len(all_pairs)} pairs done")
    
    # Aggregate per-head results
    head_summary = {}
    for h in range(n_heads_q):
        if per_head_agg[h]["weight"]:
            mw = float(np.mean(per_head_agg[h]["weight"]))
            mv = float(np.mean(per_head_agg[h]["value"]))
            head_summary[str(h)] = {
                "weight_effect_mean": mw,
                "value_effect_mean": mv,
                "value_over_weight": mv / max(mw, 1e-10),
                "is_value_dominant": mv > mw,
            }
    
    # Sort heads by value/weight ratio
    sorted_by_ratio = sorted(head_summary.items(), key=lambda x: x[1]["value_over_weight"], reverse=True)
    
    log_time(f"\n  Qwen3 L18 Per-Head Analysis ({len(all_pairs)} pairs):")
    log_time(f"  {'Head':>6} {'wt_eff':>8} {'val_eff':>8} {'v/w':>8} {'dominant':>12}")
    for h_str, info in sorted_by_ratio:
        dom = "VALUE" if info["is_value_dominant"] else "WEIGHT"
        log_time(f"  H{int(h_str):>5} {info['weight_effect_mean']:8.4f} {info['value_effect_mean']:8.4f} "
                 f"{info['value_over_weight']:8.3f} {dom:>12}")
    
    # Count VALUE-dominant heads
    n_val_dom = sum(1 for _, info in sorted_by_ratio if info["is_value_dominant"])
    n_weight_dom = n_heads_q - n_val_dom
    log_time(f"\n  Summary: {n_val_dom}/{n_heads_q} heads VALUE-dominant, "
             f"{n_weight_dom}/{n_heads_q} WEIGHT-dominant")
    
    # GQA group analysis
    kv_groups = defaultdict(list)
    for h in range(n_heads_q):
        kv_groups[h // gqa].append(h)
    
    log_time(f"\n  Per KV-group VALUE-dominance (GQA group={gqa}):")
    for kv_idx in sorted(kv_groups.keys()):
        group_heads = kv_groups[kv_idx]
        n_val = sum(1 for h in group_heads if head_summary[str(h)]["is_value_dominant"])
        log_time(f"    KV group {kv_idx} (heads {group_heads}): {n_val}/{len(group_heads)} VALUE-dominant")
    
    return {"pairs": results, "head_summary": head_summary, "aggregate": per_head_agg}


# =============================================================================
# Block C: GLM4 L0 Partial RoPE Decomposition
# =============================================================================

def block_c_partial_rope_decomposition(model, tokenizer, device, model_info, model_name, all_pairs):
    """
    For GLM4 only: decompose L0 RoPE into rotated (64 dims) vs non-rotated (64 dims).
    
    Hypothesis: GLM4 L0's unique VALUE-dominance might be because the non-rotated 
    64 dimensions let value vectors carry more positional-independent information.
    
    Three variants tested:
      C1: full RoPE (same as Phase 282) — baseline
      C2: only rotated dims active, non-rotated zeroed — isolates RoPE contribution
      C3: only non-rotated dims active, rotated zeroed — isolates position-free contribution
    """
    if model_name != "glm4":
        log_time("Block C: Skipped (GLM4 only)")
        return None
    
    n_layers = model_info.n_layers
    rope_cfg = MODEL_ROPE_CONFIGS[model_name]
    rope_base = rope_cfg["rope_theta"]
    rotary_dim = rope_cfg["rotary_dim"]  # 64 (GLM4 partial RoPE)
    
    n_heads_q, n_heads_kv, d_model_cfg = detect_model_head_config(model, model_name)
    
    target_layer = 0  # L0 only
    
    log_time(f"\n{'='*60}")
    log_time(f"Block C: GLM4 L0 Partial RoPE Decomposition")
    log_time(f"  rotary_dim={rotary_dim} (config), head_dim=TBD (from weights)")
    
    # Load L0 weights (d_head from actual weight shapes)
    w = extract_layer_weights_from_safetensors(model_name, target_layer, n_heads_q, n_heads_kv)
    if w is None:
        log_time("  Failed to load L0 weights")
        return None
    
    d_head = w["d_head"]
    non_rotary_dim = d_head - rotary_dim
    log_time(f"  Actual d_head={d_head}, rotary_dim={rotary_dim}, non_rotary_dim={non_rotary_dim}")
    
    layers = get_layers(model)
    
    # Use a focused subset: the 8 'negation' pairs + 8 'human' pairs (most diverse behavior)
    focused_pairs = [p for p in all_pairs if p["category"] in ["negation", "human", "passive", "animal"]]
    focused_pairs = focused_pairs[:20]  # 20 pairs for efficiency
    
    log_time(f"  Testing {len(focused_pairs)} pairs (negation+human+passive+animal)")
    
    # Results for all three variants
    results = {"C1_full_rope": {}, "C2_rotated_only": {}, "C3_nonrotated_only": {}}
    
    for variant_name in ["C1_full_rope", "C2_rotated_only", "C3_nonrotated_only"]:
        log_time(f"\n  --- {variant_name} ---")
        
        for pair_idx, pair in enumerate(focused_pairs):
            pname = pair["name"]
            sent_a, sent_b = pair["A"], pair["B"]
            
            # Capture hidden states
            hA_data = {}; hB_data = {}
            
            def make_hook(data_dict, key):
                def hook(module, input_t, output_t):
                    if isinstance(input_t, tuple) and len(input_t) > 0:
                        data_dict[key] = input_t[0].detach().float().cpu()
                return hook
            
            inputs_a = tokenizer(sent_a, return_tensors="pt").to(device)
            inputs_b = tokenizer(sent_b, return_tensors="pt").to(device)
            
            hooks = [layers[target_layer].register_forward_hook(make_hook(hA_data, f"L{target_layer}"))]
            with torch.no_grad():
                _ = model(**inputs_a)
            hooks[0].remove()
            
            hooks = [layers[target_layer].register_forward_hook(make_hook(hB_data, f"L{target_layer}"))]
            with torch.no_grad():
                _ = model(**inputs_b)
            hooks[0].remove()
            
            key = f"L{target_layer}"
            if key not in hA_data or key not in hB_data:
                continue
            
            hA = apply_input_ln(hA_data[key][0].numpy(), layers[target_layer])
            hB = apply_input_ln(hB_data[key][0].numpy(), layers[target_layer])
            seqA, seqB = hA.shape[0], hB.shape[0]
            min_seq = min(seqA, seqB)
            
            W_q = w["W_q"]; W_k = w["W_k"]; W_v = w["W_v"]; W_o = w["W_o"]
            n_q = w["n_heads_q"]; n_kv = w["n_heads_kv"]
            d_h = w["d_head"]; gqa = w["gqa_group"]
            
            QA = (hA[:min_seq] @ W_q.T).reshape(min_seq, n_q, d_h).transpose(1, 0, 2)
            KA_raw = (hA[:min_seq] @ W_k.T).reshape(min_seq, n_kv, d_h).transpose(1, 0, 2)
            VA_raw = (hA[:min_seq] @ W_v.T).reshape(min_seq, n_kv, d_h).transpose(1, 0, 2)
            QB = (hB[:min_seq] @ W_q.T).reshape(min_seq, n_q, d_h).transpose(1, 0, 2)
            KB_raw = (hB[:min_seq] @ W_k.T).reshape(min_seq, n_kv, d_h).transpose(1, 0, 2)
            VB_raw = (hB[:min_seq] @ W_v.T).reshape(min_seq, n_kv, d_h).transpose(1, 0, 2)
            
            # GQA expand
            KA = np.repeat(KA_raw, gqa, axis=0) if gqa > 1 else KA_raw
            VA = np.repeat(VA_raw, gqa, axis=0) if gqa > 1 else VA_raw
            KB = np.repeat(KB_raw, gqa, axis=0) if gqa > 1 else KB_raw
            VB = np.repeat(VB_raw, gqa, axis=0) if gqa > 1 else VB_raw
            
            # --- Variant-specific RoPE ---
            if variant_name == "C1_full_rope":
                # Normal: rotate 64 dims, pass 64 dims
                cos, sin = compute_rope_cos_sin(min_seq, d_h, rotary_dim, rope_base)
                for h in range(n_q):
                    QA[h] = apply_rope(QA[h], cos, sin, rotary_dim)
                    QB[h] = apply_rope(QB[h], cos, sin, rotary_dim)
                    KA[h] = apply_rope(KA[h], cos, sin, rotary_dim)
                    KB[h] = apply_rope(KB[h], cos, sin, rotary_dim)
            elif variant_name == "C2_rotated_only":
                # Rotate only the first 64 dims, ZERO the last 64 dims
                # cos/sin dimension covers full head_dim (128), with pad=1/0 for non-rotary
                cos_full, sin_full = compute_rope_cos_sin(min_seq, d_h, rotary_dim, rope_base)
                # But we WANT non-rotary to be ZERO, not identity
                # So compute rope with rotary_dim=d_h (128) but only freq for first 64
                # Actually simpler: create mask that zeros non-rotary part
                cos, sin = compute_rope_cos_sin(min_seq, d_h, rotary_dim, rope_base)
                # Override: apply rope normally (which passes non-rotary as identity)
                # Then mask out non-rotary part
                for h in range(n_q):
                    QA[h] = apply_rope(QA[h], cos, sin, rotary_dim)
                    QA[h][:, rotary_dim:] = 0
                    QB[h] = apply_rope(QB[h], cos, sin, rotary_dim)
                    QB[h][:, rotary_dim:] = 0
                    KA[h] = apply_rope(KA[h], cos, sin, rotary_dim)
                    KA[h][:, rotary_dim:] = 0
                    KB[h] = apply_rope(KB[h], cos, sin, rotary_dim)
                    KB[h][:, rotary_dim:] = 0
            elif variant_name == "C3_nonrotated_only":
                # Zero the rotated part, pass non-rotated as identity
                cos, sin = compute_rope_cos_sin(min_seq, d_h, rotary_dim, rope_base)
                for h in range(n_q):
                    QA[h] = apply_rope(QA[h], cos, sin, rotary_dim)
                    QA[h][:, :rotary_dim] = 0  # zero rotated dims
                    QB[h] = apply_rope(QB[h], cos, sin, rotary_dim)
                    QB[h][:, :rotary_dim] = 0
                    KA[h] = apply_rope(KA[h], cos, sin, rotary_dim)
                    KA[h][:, :rotary_dim] = 0
                    KB[h] = apply_rope(KB[h], cos, sin, rotary_dim)
                    KB[h][:, :rotary_dim] = 0
            
            # Attention and patching (same as before)
            scale = np.sqrt(d_h)
            AA = np.zeros((n_q, min_seq, min_seq))
            AB = np.zeros((n_q, min_seq, min_seq))
            for h in range(n_q):
                AA[h] = softmax_np(QA[h] @ KA[h].T / scale)
                AB[h] = softmax_np(QB[h] @ KB[h].T / scale)
            
            def flatten_h(X, s, d):
                return X.transpose(1, 0, 2).reshape(s, d)
            
            full_pure_AA = flatten_h(np.array([AA[h] @ VA[h] for h in range(n_q)]), min_seq, n_q * d_h) @ W_o.T
            full_pure_AB = flatten_h(np.array([AB[h] @ VB[h] for h in range(n_q)]), min_seq, n_q * d_h) @ W_o.T
            full_AW_BV = flatten_h(np.array([AA[h] @ VB[h] for h in range(n_q)]), min_seq, n_q * d_h) @ W_o.T
            full_BW_AV = flatten_h(np.array([AB[h] @ VA[h] for h in range(n_q)]), min_seq, n_q * d_h) @ W_o.T
            
            total_gap = float(np.linalg.norm(full_pure_AA - full_pure_AB))
            if total_gap > 1e-10:
                weight_eff = float(np.linalg.norm(full_BW_AV - full_pure_AB)) / total_gap
                value_eff = float(np.linalg.norm(full_AW_BV - full_pure_AB)) / total_gap
            else:
                weight_eff = value_eff = 0.0
            
            results[variant_name][pname] = {
                "total_gap": total_gap,
                "weight_effect": weight_eff,
                "value_effect": value_eff,
                "content_dominates": value_eff > weight_eff,
                "category": pair["category"],
            }
            
            if (pair_idx + 1) % 10 == 0:
                log_time(f"    {pair_idx+1}/{len(focused_pairs)} pairs...")
    
    # Aggregate and compare
    log_time(f"\n  GLM4 L0 Partial RoPE Decomposition Summary:")
    log_time(f"  {'Variant':>20} {'wt_eff':>8} {'val_eff':>8} {'dom%':>6} {'winner':>12}")
    
    summary = {}
    for vname in ["C1_full_rope", "C2_rotated_only", "C3_nonrotated_only"]:
        w_effs = [r["weight_effect"] for r in results[vname].values()]
        v_effs = [r["value_effect"] for r in results[vname].values()]
        doms = [1.0 if r["content_dominates"] else 0.0 for r in results[vname].values()]
        
        if w_effs:
            mw = float(np.mean(w_effs)); mv = float(np.mean(v_effs))
            md = float(np.mean(doms))
            winner = "VALUE" if md > 0.5 else "WEIGHT"
            summary[vname] = {"weight_effect": mw, "value_effect": mv, "dom_rate": md}
            log_time(f"  {vname:>20} {mw:8.3f} {mv:8.3f} {md:6.3f} {winner:>12}")
    
    # Key insight: difference between C1 and C3 tells us how much non-rotary contributes
    if "C1_full_rope" in summary and "C3_nonrotated_only" in summary:
        c1_dom = summary["C1_full_rope"]["dom_rate"]
        c3_dom = summary["C3_nonrotated_only"]["dom_rate"]
        log_time(f"\n  C1 (full) dom_rate={c1_dom:.3f}, C3 (nonrotated only) dom_rate={c3_dom:.3f}")
        if c3_dom > c1_dom + 0.1:
            log_time(f"  → Non-rotated dims PUSH toward VALUE dominance")
        elif c1_dom > c3_dom + 0.1:
            log_time(f"  → Rotated dims PUSH toward VALUE dominance (unexpected)")
        else:
            log_time(f"  → Both dim groups contribute similarly to VALUE dominance")
    
    return {"results": results, "summary": summary}


# =============================================================================
# Block D: Extended Sentence Types
# =============================================================================

def block_d_extended_sentences():
    """
    Expand from 52 SVO pairs to include:
    - Conditional sentences (if...then...)
    - Recursive/embedded clauses
    - Translation pairs
    - Quantifier/logic
    - Negation (expanded)
    
    Returns list of new pairs.
    """
    pairs = []
    
    # Category D1: Conditional (if-then) — 6 pairs
    pairs.append({"name": "if_rain_stay", "A": "if it rains we will stay", "B": "we will stay if it rains", "category": "conditional"})
    pairs.append({"name": "if_hungry_eat", "A": "if you are hungry eat something", "B": "eat something if you are hungry", "category": "conditional"})
    pairs.append({"name": "if_tired_sleep", "A": "if she is tired she will sleep", "B": "she will sleep if she is tired", "category": "conditional"})
    pairs.append({"name": "if_cold_heater", "A": "if it gets cold turn on the heater", "B": "turn on the heater if it gets cold", "category": "conditional"})
    pairs.append({"name": "if_ready_go", "A": "if they are ready we can go", "B": "we can go if they are ready", "category": "conditional"})
    pairs.append({"name": "if_sunny_park", "A": "if tomorrow is sunny visit the park", "B": "visit the park if tomorrow is sunny", "category": "conditional"})
    
    # Category D2: Recursive/Embedded clauses — 6 pairs
    pairs.append({"name": "embed_believe_that", "A": "the scientist believes that the theory is correct", "B": "the theory is believed by the scientist to be correct", "category": "recursive"})
    pairs.append({"name": "embed_dog_ran", "A": "the dog that barked ran away", "B": "the barking dog ran away", "category": "recursive"})
    pairs.append({"name": "embed_woman_wrote", "A": "the woman who wrote the letter smiled", "B": "the letter-writing woman smiled", "category": "recursive"})
    pairs.append({"name": "embed_king_said", "A": "the king said that the queen is wise", "B": "the queen is wise said the king", "category": "recursive"})
    pairs.append({"name": "embed_teacher_think", "A": "the teacher thinks that the student learned well", "B": "the student is thought by the teacher to have learned well", "category": "recursive"})
    pairs.append({"name": "embed_door_opened", "A": "the door that was painted red opened", "B": "the red door opened", "category": "recursive"})
    
    # Category D3: Negation (expanded, more diverse than Phase 282) — 6 pairs
    pairs.append({"name": "neg_nobody_came", "A": "somebody came to the party", "B": "nobody came to the party", "category": "negation_d"})
    pairs.append({"name": "neg_never_seen", "A": "i have seen it before", "B": "i have never seen it before", "category": "negation_d"})
    pairs.append({"name": "neg_not_only", "A": "she is smart", "B": "she is not only smart but also kind", "category": "negation_d"})
    pairs.append({"name": "neg_hardly", "A": "he works hard", "B": "he hardly works", "category": "negation_d"})
    pairs.append({"name": "neg_scarcely", "A": "they had enough food", "B": "they had scarcely enough food", "category": "negation_d"})
    pairs.append({"name": "neg_double", "A": "the proposal is acceptable", "B": "the proposal is not unacceptable", "category": "negation_d"})
    
    # Category D4: Quantifier/Logic (expanded) — 6 pairs
    pairs.append({"name": "quant_every_some", "A": "some birds can fly", "B": "every bird can fly", "category": "quantifier_d"})
    pairs.append({"name": "quant_most_few", "A": "most students passed the exam", "B": "few students passed the exam", "category": "quantifier_d"})
    pairs.append({"name": "quant_always_never", "A": "she always tells the truth", "B": "she never tells the truth", "category": "quantifier_d"})
    pairs.append({"name": "quant_more_less", "A": "more people attended than expected", "B": "fewer people attended than expected", "category": "quantifier_d"})
    pairs.append({"name": "quant_only_even", "A": "only John came", "B": "even John came", "category": "quantifier_d"})
    pairs.append({"name": "quant_almost_nearly", "A": "almost everyone agreed", "B": "nearly no one agreed", "category": "quantifier_d"})
    
    # Category D5: Translation (EN↔ZH) — 8 pairs
    pairs.append({"name": "trans_dog_cat", "A": "the dog chases the cat", "B": "狗追猫", "category": "translation"})
    pairs.append({"name": "trans_sun_rise", "A": "the sun rises in the east", "B": "太阳从东方升起", "category": "translation"})
    pairs.append({"name": "trans_teacher_student", "A": "the teacher teaches the student", "B": "老师教学生", "category": "translation"})
    pairs.append({"name": "trans_bird_sky", "A": "the bird flies in the sky", "B": "鸟在天空中飞翔", "category": "translation"})
    pairs.append({"name": "trans_king_rule", "A": "the king rules the kingdom", "B": "国王统治王国", "category": "translation"})
    pairs.append({"name": "trans_water_cold", "A": "the water is very cold", "B": "水非常冷", "category": "translation"})
    pairs.append({"name": "trans_love_overcome", "A": "love overcomes everything", "B": "爱战胜一切", "category": "translation"})
    pairs.append({"name": "trans_fox_hunt", "A": "the fox hunts the rabbit", "B": "狐狸猎兔", "category": "translation"})
    
    # Category D6: Passive (expanded) — 5 pairs
    pairs.append({"name": "pass_was_eaten", "A": "the cat ate the fish", "B": "the fish was eaten by the cat", "category": "passive_d"})
    pairs.append({"name": "pass_was_written", "A": "the author wrote the book", "B": "the book was written by the author", "category": "passive_d"})
    pairs.append({"name": "pass_was_built", "A": "the workers built the bridge", "B": "the bridge was built by the workers", "category": "passive_d"})
    pairs.append({"name": "pass_is_loved", "A": "everyone loves the teacher", "B": "the teacher is loved by everyone", "category": "passive_d"})
    pairs.append({"name": "pass_was_found", "A": "the detective found the clue", "B": "the clue was found by the detective", "category": "passive_d"})
    
    return pairs


# =============================================================================
# Block E: Final Component Contribution Matrix
# =============================================================================

def block_e_final_matrix(block_a_agg, block_b_result, block_c_result, model_info, model_name):
    """
    Build comprehensive component contribution matrix:
    Function x Layer x (weight_eff, value_eff, head_dominance)
    """
    n_layers = model_info.n_layers
    
    matrix = {"model": model_name, "n_layers": n_layers, "per_layer": {}, "per_head": {}}
    
    # From Block A: all-layer summary
    if block_a_agg:
        for key, info in block_a_agg.get("per_layer", {}).items():
            matrix["per_layer"][key] = {
                "weight_effect": info.get("weight_effect_mean", 0),
                "value_effect": info.get("value_effect_mean", 0),
                "content_dominance": info.get("content_dominance_rate", 0),
                "primary_component": "VALUE" if info.get("content_dominance_rate", 0) > 0.5 else "WEIGHT",
            }
        
        # Per-head data
        for key, heads in block_a_agg.get("per_head_per_layer", {}).items():
            matrix["per_head"][key] = heads
    
    # From Block B: Qwen3 L18 specific
    if block_b_result:
        matrix["special_layers"] = matrix.get("special_layers", {})
        matrix["special_layers"]["qwen3_L18"] = {
            "head_summary": block_b_result.get("head_summary", {}),
        }
    
    # From Block C: GLM4 L0 partial RoPE
    if block_c_result:
        matrix["special_layers"] = matrix.get("special_layers", {})
        matrix["special_layers"]["glm4_L0_rope"] = block_c_result.get("summary", {})
    
    # Log the full matrix
    log_time(f"\n{'='*60}")
    log_time(f"Block E: Final Component Contribution Matrix — {model_name}")
    
    if matrix["per_layer"]:
        log_time(f"\n  Layer-level matrix:")
        log_time(f"  {'Layer':>5} {'wt_eff':>8} {'val_eff':>8} {'dom%':>6} {'primary':>12}")
        for key in sorted(matrix["per_layer"].keys(), key=int):
            e = matrix["per_layer"][key]
            log_time(f"  L{int(key):>4} {e['weight_effect']:8.3f} {e['value_effect']:8.3f} "
                     f"{e['content_dominance']:6.3f} {e['primary_component']:>12}")
    
    # Per-head VALUE analysis
    if matrix["per_head"]:
        for layer_key, heads in matrix["per_head"].items():
            n_val = sum(1 for h, info in heads.items() if info.get("content_dominates", False))
            n_total = len(heads)
            log_time(f"\n  L{int(layer_key)} per-head: {n_val}/{n_total} heads VALUE-dominant")
    
    # Special layers
    if "special_layers" in matrix:
        for key, info in matrix["special_layers"].items():
            log_time(f"\n  Special: {key}")
            if "total" in info:
                log_time(f"    Full: wt={info.get('weight_effect','N/A'):.3f}, val={info.get('value_effect','N/A'):.3f}")
    
    return matrix


# =============================================================================
# Aggregate extended sentence types for Block A
# =============================================================================

def build_all_pairs(include_extended=True):
    """Build all sentence pairs including Phase 282 base + Phase 283 extended types."""
    # Import Phase 282 pairs
    from phase282_causal_patching_rope import build_svo_pairs
    pairs = build_svo_pairs()
    
    if include_extended:
        extended = block_d_extended_sentences()
        pairs.extend(extended)
    
    return pairs


# =============================================================================
# Main
# =============================================================================

def run_phase283(model_name):
    global _log_file
    _log_file = str(TMP_DIR / f"phase283_{model_name}.txt")
    
    log_time(f"{'='*60}")
    log_time(f"Phase 283: Deep Analysis — {model_name}")
    log_time(f"{'='*60}")
    
    # Build all pairs
    all_pairs = build_all_pairs(include_extended=True)
    log_time(f"Dataset: {len(all_pairs)} pairs across {len(set(p['category'] for p in all_pairs))} categories")
    cat_counts = defaultdict(int)
    for p in all_pairs:
        cat_counts[p["category"]] += 1
    for cat, cnt in sorted(cat_counts.items()):
        log_time(f"  {cat}: {cnt} pairs")
    
    # Load model
    model, tokenizer, device = load_model_bf16_flash(model_name)
    model_info = get_model_info(model, model_name)
    log_time(f"Model: {model_info.model_class}, L={model_info.n_layers}, d={model_info.d_model}")
    
    # Warmup
    log_time("Global warmup forward...")
    warmup_inputs = tokenizer("The quick brown fox jumps over the lazy dog", return_tensors="pt").to(device)
    with torch.no_grad():
        try:
            model(**warmup_inputs)
        except:
            pass
    log_time(f"Warmup done, GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")
    
    all_results = {}
    
    try:
        # Block A: Safetensors deep layer patching (ALL layers)
        t0 = time.time()
        result_a = block_a_deep_layer_weights(model, tokenizer, device, model_info, model_name, all_pairs)
        t_a = time.time() - t0
        log_time(f"Block A done in {t_a:.1f}s ({t_a/60:.1f}min)")
        all_results["block_a"] = result_a
        
        # Save Block A
        if result_a:
            save_dict = {"aggregate": result_a["aggregate"]}
            save_dict["aggregate"]["per_category"] = dict(save_dict["aggregate"]["per_category"])
            # Convert per-head agg too
            with open(RESULT_DIR / f"{model_name}_block_a_all_layers.json", "w") as f:
                json.dump(save_dict, f, indent=2)
        
        # Block B: Per-head VALUE analysis (Qwen3 only)
        # Use only the Phase 282 base pairs (not extended) for consistency
        base_pairs = build_all_pairs(include_extended=False)
        t0 = time.time()
        result_b = block_b_per_head_value_analysis(model, tokenizer, device, model_info, model_name, base_pairs)
        t_b = time.time() - t0
        log_time(f"Block B done in {t_b:.1f}s ({t_b/60:.1f}min)")
        all_results["block_b"] = result_b
        
        if result_b:
            with open(RESULT_DIR / f"{model_name}_block_b_per_head.json", "w") as f:
                json.dump(result_b.get("head_summary", {}), f, indent=2)
        
        # Block C: Partial RoPE decomposition (GLM4 only)
        t0 = time.time()
        result_c = block_c_partial_rope_decomposition(model, tokenizer, device, model_info, model_name, all_pairs)
        t_c = time.time() - t0
        log_time(f"Block C done in {t_c:.1f}s ({t_c/60:.1f}min)")
        all_results["block_c"] = result_c
        
        if result_c:
            with open(RESULT_DIR / f"{model_name}_block_c_partial_rope.json", "w") as f:
                json.dump(result_c.get("summary", {}), f, indent=2)
        
        # Block E: Final matrix
        block_a_agg = result_a.get("aggregate", {}) if result_a else {}
        matrix = block_e_final_matrix(block_a_agg, result_b, result_c, model_info, model_name)
        all_results["matrix"] = matrix
        
        with open(RESULT_DIR / f"{model_name}_final_matrix.json", "w") as f:
            json.dump(matrix, f, indent=2)
        
        log_time(f"\nPhase 283 complete for {model_name}: A={t_a:.0f}s, B={t_b:.0f}s, C={t_c:.0f}s")
        
        return all_results
        
    finally:
        release_model(model)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    
    if model_name == "all":
        for name in ["qwen3", "glm4", "deepseek7b"]:
            try:
                r = run_phase283(name)
                log_time(f"{name} done")
            except Exception as e:
                log_time(f"!!! {name} FAILED: {e}")
                import traceback
                traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(3)
    else:
        run_phase283(model_name)
