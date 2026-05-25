"""
Phase 284: Manual Attention Calibration + Full Function Matrix
==============================================================

CORE GOAL: Verify that our manual RoPE patching results are reliable,
then build a full function × layer × component contribution matrix.

BLOCK 0 (Method Calibration):
  - Compare manual attention output vs REAL forward attention output
  - Per-layer gap statistics (cosine similarity, norm ratio, relative error)
  - This quantifies the "manual attention gap" that has been a hard concern

BLOCK 1 (Absolute + Normalized Effects):
  - Report BOTH absolute gap magnitude AND normalized effect ratios
  - This resolves the concern that deep-layer total_gap inflation masks weight contributions

BLOCK 2 (Full Function Matrix):
  - 10+ function categories: SVO, passive, negation, quantifier, conditional,
    recursive, translation, comparative, logical, temporal
  - ~120 pairs total — each function has at least 8 pairs for statistical stability
  - Per-function, per-layer: weight_effect, value_effect, absolute_gap

BLOCK 3 (DS7B SDPA verification):
  - If possible, load DS7B with SDPA attention (not eager) to reduce artifacts
  - Rerun deep layers and compare with eager results

Hardware strategy:
  - All models: bf16 + device_map="auto" + flash_attention_2
  - GLM4/DS7B: auto offload to CPU for deep layers
  - Safetensors weight extraction for all layers

Usage:
  python tests/glm5/phase284_calibrated_full_matrix.py qwen3
  python tests/glm5/phase284_calibrated_full_matrix.py glm4
  python tests/glm5/phase284_calibrated_full_matrix.py deepseek7b
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

RESULT_DIR = Path("results/phase284_calibrated_matrix")
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
# RoPE Implementation (Phase 282/283 compatible)
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
    """Load model in bf16 with flash_attention_2, device_map='auto'."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    log_time(f"Loading {model_name} (bf16 + flash_attention_2)...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try flash_attention_2 first, fall back to eager
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
    model.eval()

    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log_time(f"  {model_name} loaded: device={device}, GPU={gpu_mem:.2f}GB, attn={attn_impl}")
    return model, tokenizer, device, attn_impl

# =============================================================================
# Utility: Safetensors Weight Extraction (Phase 283 compatible)
# =============================================================================

def extract_layer_weights_from_safetensors(model_name, layer_idx, n_heads_q, n_heads_kv):
    """Read Q/K/V/O weights from safetensors."""
    cfg = MODEL_CONFIGS[model_name]
    model_path = cfg["path"]
    sf_files = sorted(fileglob.glob(os.path.join(model_path, "*.safetensors")))
    if not sf_files:
        sf_files = sorted(fileglob.glob(os.path.join(model_path, "model-*.safetensors")))

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
        q_out_dim = W_q.shape[0]
        d_head = q_out_dim // n_heads_q
        gqa_group = n_heads_q // max(n_heads_kv, 1)
        return {
            "W_q": W_q, "W_k": W_k, "W_v": W_v, "W_o": W_o,
            "n_heads_q": n_heads_q, "n_heads_kv": n_heads_kv,
            "d_head": d_head, "gqa_group": gqa_group,
        }
    return None

def detect_model_head_config(model, model_name):
    config = model.config
    n_heads_q = getattr(config, 'num_attention_heads', 32)
    n_heads_kv = getattr(config, 'num_key_value_heads', n_heads_q)
    d_model = getattr(config, 'hidden_size', getattr(config, 'd_model', 4096))
    return n_heads_q, n_heads_kv, d_model

def softmax_np(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def apply_input_ln(hidden_np, layer, eps=1e-6):
    """Apply layer's input layernorm to numpy hidden states."""
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

# =============================================================================
# BLOCK 0: Manual Attention Calibration (vs Real Forward)
# =============================================================================

def block0_calibration(model, tokenizer, device, model_name, attn_impl):
    """
    Compare manual attention computation with real forward attention output.
    
    For 5 sentences, hook both:
    - self_attn.input (h_pre) and self_attn.output (real attn_out)
    - For each layer, compute manual attn_out from h_pre using safetensors weights + RoPE
    - Compare: cosine similarity, L2 norm ratio, relative error
    """
    n_layers_model = get_model_info(model, model_name).n_layers
    n_heads_q, n_heads_kv, _ = detect_model_head_config(model, model_name)
    rope_cfg = MODEL_ROPE_CONFIGS[model_name]
    rope_base = rope_cfg["rope_theta"]
    rotary_dim = rope_cfg["rotary_dim"]
    has_qk_norm = rope_cfg["has_qk_norm"]

    log_time(f"\n{'='*60}")
    log_time(f"BLOCK 0: Manual Attention Calibration — {model_name}")
    log_time(f"  Layers={n_layers_model}, Q_heads={n_heads_q}, KV_heads={n_heads_kv}")
    log_time(f"  RoPE: theta={rope_base}, rotary_dim={rotary_dim}, QK_norm={has_qk_norm}")
    log_time(f"  Attention implementation: {attn_impl}")

    # Load all layer weights via safetensors
    all_weights = {}
    for li in range(n_layers_model):
        w = extract_layer_weights_from_safetensors(model_name, li, n_heads_q, n_heads_kv)
        if w is not None:
            all_weights[li] = w

    loaded_layers = len(all_weights)
    log_time(f"  Safetensors: loaded {loaded_layers}/{n_layers_model} layers")

    if loaded_layers == 0:
        log_time("  BLOCK 0: No weights loaded, aborting calibration")
        return None

    # Determine sample layers (all if <=20, otherwise 20 evenly spaced)
    if loaded_layers <= 20:
        sample_layers = sorted(all_weights.keys())
    else:
        step = max(1, loaded_layers // 19)
        sample_layers = sorted(set(
            list(range(0, loaded_layers, step)) + [loaded_layers - 1]
        ))
        sample_layers = [li for li in sample_layers if li in all_weights]

    log_time(f"  Calibrating {len(sample_layers)} layers: {sample_layers[:5]}...{sample_layers[-3:]}")

    # Calibration sentences: diverse short sentences
    calib_sentences = [
        "the dog chases the cat",
        "every student passed the exam",
        "love overcomes everything",
        "if it rains we will stay home",
        "the king said that the queen is wise",
    ]

    # Get layer objects for LN and QK norm
    layers = get_layers(model)

    # QK norm extraction
    qk_norms = {}
    if has_qk_norm:
        for li in sample_layers:
            if li < len(layers):
                sa = layers[li].self_attn
                if hasattr(sa, 'q_norm'):
                    try:
                        qk_norms[li] = {
                            "q_norm": sa.q_norm.weight.detach().cpu().float().numpy(),
                            "k_norm": sa.k_norm.weight.detach().cpu().float().numpy(),
                        }
                    except: pass

    # Results structure: {layer_idx: {sentence_idx: {cos_sim, norm_ratio, rel_err}}}
    calibration_results = defaultdict(lambda: defaultdict(dict))
    per_layer_agg = {}

    for sent_idx, sentence in enumerate(calib_sentences):
        log_time(f"  Calibrating sentence {sent_idx+1}/{len(calib_sentences)}: '{sentence[:50]}'")

        # Hook self_attn input and output for all sample layers
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        attn_inputs = {}
        attn_outputs = {}

        def make_attn_hook(li):
            def hook(module, input_t, output_t):
                if isinstance(input_t, tuple) and len(input_t) > 0:
                    attn_inputs[li] = input_t[0].detach().float().cpu()
                if isinstance(output_t, tuple):
                    attn_outputs[li] = output_t[0].detach().float().cpu()
                else:
                    attn_outputs[li] = output_t.detach().float().cpu()
            return hook

        hooks = []
        for li in sample_layers:
            if li < len(layers):
                sa = layers[li].self_attn
                hooks.append(sa.register_forward_hook(make_attn_hook(li)))

        with torch.no_grad():
            try:
                _ = model(**inputs)
            except Exception as e:
                log_time(f"    Forward failed: {e}")

        for h in hooks:
            h.remove()

        # Now compute manual attention for each layer and compare
        for li in sample_layers:
            if li not in attn_inputs or li not in attn_outputs:
                continue
            if li not in all_weights:
                continue

            w = all_weights[li]
            h_pre = attn_inputs[li][0].numpy()  # [seq, d_model]
            real_out = attn_outputs[li][0].numpy()  # [seq, d_model]

            # Apply LN
            if li < len(layers):
                h_pre_ln = apply_input_ln(h_pre, layers[li])
            else:
                h_pre_ln = h_pre

            seq_len = h_pre_ln.shape[0]
            W_q = w["W_q"]; W_k = w["W_k"]; W_v = w["W_v"]; W_o = w["W_o"]
            n_q = w["n_heads_q"]; n_kv = w["n_heads_kv"]
            d_h = w["d_head"]; gqa = w["gqa_group"]

            # Q, K, V projections
            Q = (h_pre_ln @ W_q.T).reshape(seq_len, n_q, d_h).transpose(1, 0, 2)  # [n_q, seq, d_h]
            K_raw = (h_pre_ln @ W_k.T).reshape(seq_len, n_kv, d_h).transpose(1, 0, 2)
            V_raw = (h_pre_ln @ W_v.T).reshape(seq_len, n_kv, d_h).transpose(1, 0, 2)

            # QK Norm
            if li in qk_norms:
                qn = qk_norms[li]["q_norm"]; kn = qk_norms[li]["k_norm"]
                eps = 1e-6
                for h in range(n_q):
                    Q[h] = Q[h] * qn / np.sqrt(np.mean(Q[h]**2) + eps)
                for h in range(n_kv):
                    K_raw[h] = K_raw[h] * kn / np.sqrt(np.mean(K_raw[h]**2) + eps)

            # GQA expand
            K = np.repeat(K_raw, gqa, axis=0) if gqa > 1 else K_raw
            V = np.repeat(V_raw, gqa, axis=0) if gqa > 1 else V_raw

            # RoPE
            cos, sin = compute_rope_cos_sin(seq_len, d_h, rotary_dim, rope_base)
            for h in range(n_q):
                Q[h] = apply_rope(Q[h], cos, sin, rotary_dim)
                K[h] = apply_rope(K[h], cos, sin, rotary_dim)

            # Manual attention
            scale = np.sqrt(d_h)
            manual_attn_out = np.zeros((n_q, seq_len, d_h))
            for h in range(n_q):
                scores = Q[h] @ K[h].T / scale
                attn_weights = softmax_np(scores)
                manual_attn_out[h] = attn_weights @ V[h]

            # Project through W_o
            manual_out = manual_attn_out.transpose(1, 0, 2).reshape(seq_len, n_q * d_h) @ W_o.T

            # Comparison metrics
            real_f32 = real_out.astype(np.float64)
            manual_f32 = manual_out.astype(np.float64)

            # Cosine similarity (flattened)
            real_flat = real_f32.ravel()
            manual_flat = manual_f32.ravel()
            cos_sim = float(np.dot(real_flat, manual_flat) /
                          (np.linalg.norm(real_flat) * np.linalg.norm(manual_flat) + 1e-10))

            # L2 norm ratio (manual / real)
            norm_real = float(np.linalg.norm(real_flat))
            norm_manual = float(np.linalg.norm(manual_flat))
            norm_ratio = norm_manual / max(norm_real, 1e-10)

            # Relative L2 error: ||manual - real|| / ||real||
            rel_err = float(np.linalg.norm(manual_flat - real_flat)) / max(norm_real, 1e-10)

            calibration_results[li][sent_idx] = {
                "cos_sim": cos_sim,
                "norm_ratio": norm_ratio,
                "rel_error": rel_err,
                "norm_real": norm_real,
                "norm_manual": norm_manual,
            }

        if (sent_idx + 1) % 3 == 0:
            log_time(f"    {sent_idx+1}/{len(calib_sentences)} done")

    # Aggregate per layer
    log_time(f"\n  Calibration Results per Layer:")
    log_time(f"  {'Layer':>5} {'cos_sim':>8} {'norm_ratio':>10} {'rel_err':>8} {'reliable?':>10}")

    for li in sorted(calibration_results.keys()):
        data = calibration_results[li]
        cos_vals = [d["cos_sim"] for d in data.values()]
        nr_vals = [d["norm_ratio"] for d in data.values()]
        re_vals = [d["rel_error"] for d in data.values()]

        m_cos = float(np.mean(cos_vals))
        m_nr = float(np.mean(nr_vals))
        m_re = float(np.mean(re_vals))

        # Reliability: cos>0.95 and norm in [0.8, 1.2] and rel_err<0.3
        reliable = (m_cos > 0.95 and 0.8 < m_nr < 1.2 and m_re < 0.3)

        per_layer_agg[str(li)] = {
            "cos_sim_mean": m_cos, "cos_sim_std": float(np.std(cos_vals)),
            "norm_ratio_mean": m_nr, "norm_ratio_std": float(np.std(nr_vals)),
            "rel_error_mean": m_re, "rel_error_std": float(np.std(re_vals)),
            "is_reliable": reliable,
        }

        flag = "RELIABLE" if reliable else "WARNING"
        log_time(f"  L{li:>4} {m_cos:8.4f} {m_nr:10.4f} {m_re:8.4f} {flag:>10}")

    if not per_layer_agg:
        log_time(f"\n  WARNING: Calibration failed for ALL layers (no data collected)")
        log_time(f"  This likely means Hook on self_attn didn't fire for this model/attention_impl")
        log_time(f"  Will PROCEED with Block 1 but mark all layers as unverified")
        return {"per_layer": {}, "calibration_failed": True, "reason": "no_hook_data"}

    # Summary stats
    reliable_count = sum(1 for v in per_layer_agg.values() if v["is_reliable"])
    log_time(f"\n  Summary: {reliable_count}/{len(per_layer_agg)} layers reliable")

    # Global stats
    all_cos = [v["cos_sim_mean"] for v in per_layer_agg.values()]
    all_re = [v["rel_error_mean"] for v in per_layer_agg.values()]
    if all_cos:
        log_time(f"  Global: cos_sim=[{min(all_cos):.4f}, {max(all_cos):.4f}] mean={np.mean(all_cos):.4f}")
    if all_re:
        log_time(f"  Global: rel_err=[{min(all_re):.4f}, {max(all_re):.4f}] mean={np.mean(all_re):.4f}")

    # Per-layer calibration error — this will be used to adjust confidence in Block 1
    return {"per_layer": dict(per_layer_agg), "per_sentence": {str(k): dict(v) for k, v in dict(calibration_results).items()}}

# =============================================================================
# BLOCK 1: Manual Attention Patching with Absolute + Normalized Effects
# =============================================================================

def block1_absolute_normalized_effects(model, tokenizer, device, model_name, all_pairs, calibration):
    """
    Run the standard manual RoPE patching (Phase 283 Block A style) but:
    1. Report BOTH absolute gap magnitude AND normalized effect ratios
    2. Flag layers where calibration shows low reliability
    3. Use per-function aggregation for the function matrix
    """
    n_layers_model = get_model_info(model, model_name).n_layers
    n_heads_q, n_heads_kv, _ = detect_model_head_config(model, model_name)
    rope_cfg = MODEL_ROPE_CONFIGS[model_name]
    rope_base = rope_cfg["rope_theta"]
    rotary_dim = rope_cfg["rotary_dim"]
    has_qk_norm = rope_cfg["has_qk_norm"]

    log_time(f"\n{'='*60}")
    log_time(f"BLOCK 1: Absolute + Normalized Effects Patching — {model_name}")
    log_time(f"  Pairs: {len(all_pairs)} from {len(set(p['category'] for p in all_pairs))} categories")

    # Load all layer weights
    all_weights = {}
    for li in range(n_layers_model):
        w = extract_layer_weights_from_safetensors(model_name, li, n_heads_q, n_heads_kv)
        if w is not None:
            all_weights[li] = w

    loaded = len(all_weights)
    log_time(f"  Safetensors: loaded {loaded}/{n_layers_model} layers")

    # Sample layers (15-20 evenly spaced)
    if loaded <= 18:
        sample_layers = sorted(all_weights.keys())
    else:
        step = max(1, loaded // 17)
        sample_layers = sorted(set(
            list(range(0, loaded, step)) + [loaded - 1]
        ))
        sample_layers = [li for li in sample_layers if li in all_weights]

    log_time(f"  Testing {len(sample_layers)} layers: {sample_layers[:4]}...{sample_layers[-3:]}")

    # Get calibration reliability for each layer
    calib_info = {}
    unreliable_layers = set()
    if calibration and isinstance(calibration, dict):
        calib_info = calibration.get("per_layer", {})
        if not calib_info and calibration.get("calibration_failed", False):
            log_time(f"  Calibration unavailable ({calibration.get('reason', 'unknown')}), all layers unverified")
        for key, info in calib_info.items():
            if not info.get("is_reliable", True):
                unreliable_layers.add(int(key))

    if unreliable_layers:
        log_time(f"  WARNING: {len(unreliable_layers)} layers marked unreliable: {sorted(unreliable_layers)[:5]}...")

    layers = get_layers(model)

    # QK norm
    qk_norms = {}
    if has_qk_norm:
        for li in sample_layers:
            if li < len(layers):
                sa = layers[li].self_attn
                if hasattr(sa, 'q_norm'):
                    try:
                        qk_norms[li] = {
                            "q_norm": sa.q_norm.weight.detach().cpu().float().numpy(),
                            "k_norm": sa.k_norm.weight.detach().cpu().float().numpy(),
                        }
                    except: pass

    # Huggingface hooks for each pair
    def capture_hidden(sentence):
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        captured = {}

        def make_hook(key):
            def hook(module, input_t, output_t):
                if isinstance(input_t, tuple) and len(input_t) > 0:
                    captured[key] = input_t[0].detach().float().cpu()
            return hook

        hooks = []
        for li in sample_layers:
            if li < len(layers):
                hooks.append(layers[li].register_forward_hook(make_hook(f"L{li}")))

        with torch.no_grad():
            try:
                _ = model(**inputs)
            except Exception as e:
                log_time(f"  Forward failed for '{sentence[:40]}': {e}")

        for h in hooks:
            h.remove()
        return captured

    # Results: {pair_name -> {layer_key -> {weight_eff, value_eff, total_gap, ...}}}
    pair_results = {}

    for pair_idx, pair in enumerate(all_pairs):
        pname = pair["name"]
        sent_a, sent_b = pair["A"], pair["B"]
        category = pair.get("category", "unknown")

        hA = capture_hidden(sent_a)
        hB = capture_hidden(sent_b)

        layer_data = {}

        for li in sample_layers:
            key = f"L{li}"
            if li not in all_weights:
                continue
            if key not in hA or key not in hB:
                continue
            if li >= len(layers):
                continue

            w = all_weights[li]
            hA_pre = hA[key][0].numpy()
            hB_pre = hB[key][0].numpy()

            # LN
            layer = layers[li]
            hA_ln = apply_input_ln(hA_pre, layer)
            hB_ln = apply_input_ln(hB_pre, layer)

            seqA, seqB = hA_ln.shape[0], hB_ln.shape[0]
            min_seq = min(seqA, seqB)

            W_q = w["W_q"]; W_k = w["W_k"]; W_v = w["W_v"]; W_o = w["W_o"]
            n_q = w["n_heads_q"]; n_kv = w["n_heads_kv"]
            d_h = w["d_head"]; gqa = w["gqa_group"]

            # QKV projections
            QA = (hA_ln[:min_seq] @ W_q.T).reshape(min_seq, n_q, d_h).transpose(1, 0, 2)
            KA_raw = (hA_ln[:min_seq] @ W_k.T).reshape(min_seq, n_kv, d_h).transpose(1, 0, 2)
            VA_raw = (hA_ln[:min_seq] @ W_v.T).reshape(min_seq, n_kv, d_h).transpose(1, 0, 2)
            QB = (hB_ln[:min_seq] @ W_q.T).reshape(min_seq, n_q, d_h).transpose(1, 0, 2)
            KB_raw = (hB_ln[:min_seq] @ W_k.T).reshape(min_seq, n_kv, d_h).transpose(1, 0, 2)
            VB_raw = (hB_ln[:min_seq] @ W_v.T).reshape(min_seq, n_kv, d_h).transpose(1, 0, 2)

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

            # Attention weights
            scale = np.sqrt(d_h)
            AA = np.zeros((n_q, min_seq, min_seq))
            AB = np.zeros((n_q, min_seq, min_seq))
            for h in range(n_q):
                AA[h] = softmax_np(QA[h] @ KA[h].T / scale)
                AB[h] = softmax_np(QB[h] @ KB[h].T / scale)

            # Mixed outputs
            def flatten_h(X, s, d):
                return X.transpose(1, 0, 2).reshape(s, d)

            full_pure_AA = flatten_h(np.array([AA[h] @ VA[h] for h in range(n_q)]), min_seq, n_q * d_h) @ W_o.T
            full_pure_AB = flatten_h(np.array([AB[h] @ VB[h] for h in range(n_q)]), min_seq, n_q * d_h) @ W_o.T
            full_AW_BV = flatten_h(np.array([AA[h] @ VB[h] for h in range(n_q)]), min_seq, n_q * d_h) @ W_o.T
            full_BW_AV = flatten_h(np.array([AB[h] @ VA[h] for h in range(n_q)]), min_seq, n_q * d_h) @ W_o.T

            # ABSOLUTE effects (unnormalized)
            total_gap_abs = float(np.linalg.norm(full_pure_AA - full_pure_AB))
            weight_effect_abs = float(np.linalg.norm(full_BW_AV - full_pure_AB))
            value_effect_abs = float(np.linalg.norm(full_AW_BV - full_pure_AB))

            # NORMALIZED effects
            if total_gap_abs > 1e-10:
                weight_eff_norm = weight_effect_abs / total_gap_abs
                value_eff_norm = value_effect_abs / total_gap_abs
            else:
                weight_eff_norm = value_eff_norm = 0.0

            is_reliable = li not in unreliable_layers

            layer_data[str(li)] = {
                "total_gap": total_gap_abs,
                "weight_effect_abs": weight_effect_abs,
                "value_effect_abs": value_effect_abs,
                "weight_effect": weight_eff_norm,
                "value_effect": value_eff_norm,
                "content_dominates": value_eff_norm > weight_eff_norm,
                "is_reliable": is_reliable,
            }

        pair_results[pname] = {"name": pname, "category": category, "layers": layer_data}

        if (pair_idx + 1) % 25 == 0:
            log_time(f"  Block1: {pair_idx+1}/{len(all_pairs)} pairs done")

    # === AGGREGATION ===
    # Per-category aggregation (primary output)
    per_category = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # per_category[category][layer][metric] = [values]

    for pname, pr in pair_results.items():
        cat = pr["category"]
        for lk, ld in pr["layers"].items():
            per_category[cat][lk]["total_gap"].append(ld["total_gap"])
            per_category[cat][lk]["weight_effect_abs"].append(ld["weight_effect_abs"])
            per_category[cat][lk]["value_effect_abs"].append(ld["value_effect_abs"])
            per_category[cat][lk]["weight_effect"].append(ld["weight_effect"])
            per_category[cat][lk]["value_effect"].append(ld["value_effect"])
            per_category[cat][lk]["content_dom"].append(1.0 if ld["content_dominates"] else 0.0)

    # Per-layer aggregation (all pairs)
    per_layer = {}
    for cat, cat_data in per_category.items():
        for lk in cat_data:
            if lk not in per_layer:
                per_layer[lk] = {"total_gap": [], "weight_effect": [], "value_effect": [],
                                 "weight_abs": [], "value_abs": [], "content_dom": []}
            per_layer[lk]["total_gap"].extend(cat_data[lk]["total_gap"])
            per_layer[lk]["weight_effect"].extend(cat_data[lk]["weight_effect"])
            per_layer[lk]["value_effect"].extend(cat_data[lk]["value_effect"])
            per_layer[lk]["weight_abs"].extend(cat_data[lk]["weight_effect_abs"])
            per_layer[lk]["value_abs"].extend(cat_data[lk]["value_effect_abs"])
            per_layer[lk]["content_dom"].extend(cat_data[lk]["content_dom"])

    # Aggregate means
    per_layer_agg = {}
    for lk, v in per_layer.items():
        if v["total_gap"]:
            per_layer_agg[lk] = {
                "total_gap_mean": float(np.mean(v["total_gap"])),
                "weight_effect_mean": float(np.mean(v["weight_effect"])),
                "value_effect_mean": float(np.mean(v["value_effect"])),
                "weight_abs_mean": float(np.mean(v["weight_abs"])),
                "value_abs_mean": float(np.mean(v["value_abs"])),
                "content_dominance_rate": float(np.mean(v["content_dom"])),
                "primary_component": "VALUE" if float(np.mean(v["content_dom"])) > 0.5 else "WEIGHT",
                "is_reliable": int(lk) not in unreliable_layers,
            }

    per_category_agg = {}
    for cat, cat_data in per_category.items():
        per_category_agg[cat] = {}
        for lk, metrics in cat_data.items():
            if metrics["total_gap"]:
                per_category_agg[cat][lk] = {
                    "total_gap_mean": float(np.mean(metrics["total_gap"])),
                    "weight_effect_mean": float(np.mean(metrics["weight_effect"])),
                    "value_effect_mean": float(np.mean(metrics["value_effect"])),
                    "weight_abs_mean": float(np.mean(metrics["weight_effect_abs"])),
                    "value_abs_mean": float(np.mean(metrics["value_effect_abs"])),
                    "content_dominance_rate": float(np.mean(metrics["content_dom"])),
                    "primary_component": "VALUE" if float(np.mean(metrics["content_dom"])) > 0.5 else "WEIGHT",
                    "n_pairs": len(metrics["total_gap"]),
                }

    # Log per-layer summary
    log_time(f"\n  Per-Layer Summary (ALL {len(all_pairs)} pairs):")
    log_time(f"  {'L':>4} {'tot_gap':>8} {'wt(N)':>7} {'val(N)':>7} {'wt(A)':>8} {'val(A)':>8} {'dom%':>6} {'winner':>10} {'cal':>8}")
    for li in sorted(sample_layers):
        if str(li) not in per_layer_agg:
            continue
        a = per_layer_agg[str(li)]
        cal = "OK" if a.get("is_reliable", True) else "WARN"
        winner = "VALUE" if a["content_dominance_rate"] > 0.5 else "WEIGHT"
        log_time(f"  L{li:>3} {a['total_gap_mean']:8.4f} {a['weight_effect_mean']:7.3f} {a['value_effect_mean']:7.3f} "
                 f"{a['weight_abs_mean']:8.4f} {a['value_abs_mean']:8.4f} "
                 f"{a['content_dominance_rate']:6.3f} {winner:>10} {cal:>8}")

    # Log per-category summary (selected categories)
    key_cats = ["animal", "human", "passive", "negation", "conditional", "recursive", "translation", "quantifier"]
    for cat in key_cats:
        if cat not in per_category_agg:
            continue
        log_time(f"\n  Category: {cat} ({per_category_agg[cat][next(iter(per_category_agg[cat]))].get('n_pairs', '?')} pairs)")
        log_time(f"  {'L':>4} {'tot_gap':>8} {'wt(N)':>7} {'val(N)':>7} {'dom%':>6} {'winner':>10}")
        for li in sorted([int(k) for k in per_category_agg[cat].keys()]):
            if str(li) not in per_category_agg[cat]:
                continue
            a = per_category_agg[cat][str(li)]
            winner = "VALUE" if a["content_dominance_rate"] > 0.5 else "WEIGHT"
            log_time(f"  L{li:>3} {a['total_gap_mean']:8.4f} {a['weight_effect_mean']:7.3f} {a['value_effect_mean']:7.3f} "
                     f"{a['content_dominance_rate']:6.3f} {winner:>10}")

    return {
        "per_layer": per_layer_agg,
        "per_category": per_category_agg,
        "model": model_name,
        "n_pairs": len(all_pairs),
        "n_layers_tested": len(sample_layers),
        "n_unreliable_layers": len(unreliable_layers),
    }

# =============================================================================
# BUILD ALL SENTENCE PAIRS (from Phase 282 + Phase 283 + Phase 284 new)
# =============================================================================

def build_all_pairs_phase284():
    """Comprehensive sentence pair builder for Phase 284."""
    pairs = []

    # === Category 1: SVO Role Exchange (animal) — 9 pairs ===
    animal_pairs = [
        ("svo_dog_cat", "the dog chases the cat", "the cat chases the dog"),
        ("svo_wolf_sheep", "the wolf hunts the sheep", "the sheep hunts the wolf"),
        ("svo_lion_deer", "the lion stalks the deer", "the deer stalks the lion"),
        ("svo_bird_snake", "the bird pecks the snake", "the snake pecks the bird"),
        ("svo_cat_mouse", "the cat catches the mouse", "the mouse catches the cat"),
        ("svo_fox_rabbit", "the fox chases the rabbit", "the rabbit chases the fox"),
        ("svo_eagle_fish", "the eagle catches the fish", "the fish catches the eagle"),
        ("svo_bear_salmon", "the bear catches the salmon", "the salmon catches the bear"),
        ("svo_shark_seal", "the shark hunts the seal", "the seal hunts the shark"),
    ]
    for name, a, b in animal_pairs:
        pairs.append({"name": name, "A": a, "B": b, "category": "animal"})

    # === Category 2: SVO Role Exchange (human) — 8 pairs ===
    human_pairs = [
        ("svo_man_woman", "the man greets the woman", "the woman greets the man"),
        ("svo_boy_girl", "the boy calls the girl", "the girl calls the boy"),
        ("svo_teacher_student", "the teacher helps the student", "the student helps the teacher"),
        ("svo_mother_child", "the mother feeds the child", "the child feeds the mother"),
        ("svo_father_son", "the father teaches the son", "the son teaches the father"),
        ("svo_doctor_patient", "the doctor examines the patient", "the patient examines the doctor"),
        ("svo_king_queen", "the king commands the queen", "the queen commands the king"),
        ("svo_brother_sister", "the brother protects the sister", "the sister protects the brother"),
    ]
    for name, a, b in human_pairs:
        pairs.append({"name": name, "A": a, "B": b, "category": "human"})

    # === Category 3: Human-Object — 6 pairs ===
    ho_pairs = [
        ("svo_child_apple", "the child eats the apple", "the apple is eaten by the child"),
        ("svo_chef_knife", "the chef uses the knife", "the knife is used by the chef"),
        ("svo_painter_brush", "the painter holds the brush", "the brush is held by the painter"),
        ("svo_driver_car", "the driver starts the car", "the car is started by the driver"),
        ("svo_writer_pen", "the writer lifts the pen", "the pen is lifted by the writer"),
        ("svo_guard_key", "the guard holds the key", "the key is held by the guard"),
    ]
    for name, a, b in ho_pairs:
        pairs.append({"name": name, "A": a, "B": b, "category": "human_object"})

    # === Category 4: Place — 6 pairs ===
    place_pairs = [
        ("svo_king_city", "the king rules the city", "the city is ruled by the king"),
        ("svo_explorer_island", "the explorer discovers the island", "the island is discovered by the explorer"),
        ("svo_tourist_museum", "the tourist visits the museum", "the museum is visited by the tourist"),
        ("svo_guard_prison", "the guard watches the prison", "the prison is watched by the guard"),
        ("svo_soldier_bridge", "the soldier defends the bridge", "the bridge is defended by the soldier"),
        ("svo_mayor_town", "the mayor governs the town", "the town is governed by the mayor"),
    ]
    for name, a, b in place_pairs:
        pairs.append({"name": name, "A": a, "B": b, "category": "place"})

    # === Category 5: Passive Voice — 8 pairs (expanded) ===
    passive_pairs = [
        ("pass_dog_cat", "the dog chases the cat", "the cat is chased by the dog"),
        ("pass_teacher_student", "the teacher teaches the student", "the student is taught by the teacher"),
        ("pass_author_book", "the author wrote the book", "the book was written by the author"),
        ("pass_wife_cake", "the wife baked the cake", "the cake was baked by the wife"),
        ("pass_workers_bridge", "the workers built the bridge", "the bridge was built by the workers"),
        ("pass_detective_clue", "the detective found the clue", "the clue was found by the detective"),
        ("pass_everyone_teacher", "everyone loves the teacher", "the teacher is loved by everyone"),
        ("pass_cat_fish", "the cat ate the fish", "the fish was eaten by the cat"),
    ]
    for name, a, b in passive_pairs:
        pairs.append({"name": name, "A": a, "B": b, "category": "passive"})

    # === Category 6: Negation — 12 pairs (expanded) ===
    neg_pairs = [
        ("neg_happy", "she is happy", "she is not happy"),
        ("neg_agree", "they agree with the proposal", "they do not agree with the proposal"),
        ("neg_found", "he found something interesting", "he found nothing interesting"),
        ("neg_remember", "i remember the meeting", "i do not remember the meeting"),
        ("neg_possible", "victory is possible", "victory is not possible"),
        ("neg_understand", "we understand the problem", "we do not understand the problem"),
        ("neg_anyone_came", "someone came to the party", "no one came to the party"),
        ("neg_ever_seen", "i have seen it before", "i have never seen it before"),
        ("neg_notonly_smart", "she is smart", "she is not only smart but also kind"),
        ("neg_hardly_works", "he works hard", "he hardly works"),
        ("neg_scarcely_enough", "they had enough food", "they had scarcely enough food"),
        ("neg_unacceptable", "the proposal is acceptable", "the proposal is not unacceptable"),
    ]
    for name, a, b in neg_pairs:
        pairs.append({"name": name, "A": a, "B": b, "category": "negation"})

    # === Category 7: Quantifier — 12 pairs (expanded) ===
    quant_pairs = [
        ("quant_few_many", "few people attended", "many people attended"),
        ("quant_some_all", "some birds can fly", "all birds can fly"),
        ("quant_slow_fast", "the car is slow", "the car is fast"),
        ("quant_small_large", "the house is small", "the house is large"),
        ("quant_quiet_loud", "the music is quiet", "the music is loud"),
        ("quant_cold_hot", "the water is cold", "the water is hot"),
        ("quant_most_few", "most students passed", "few students passed"),
        ("quant_always_never", "she always tells the truth", "she never tells the truth"),
        ("quant_more_fewer", "more people came than expected", "fewer people came than expected"),
        ("quant_only_even", "only John came", "even John came"),
        ("quant_almost_all", "almost everyone agreed", "almost no one agreed"),
        ("quant_each_both", "each student brought a book", "both students brought a book"),
    ]
    for name, a, b in quant_pairs:
        pairs.append({"name": name, "A": a, "B": b, "category": "quantifier"})

    # === Category 8: Conditional — 8 pairs (new) ===
    cond_pairs = [
        ("cond_if_rain", "if it rains we will stay home", "we will stay home if it rains"),
        ("cond_if_hungry", "if you are hungry eat something", "eat something if you are hungry"),
        ("cond_if_tired", "if she is tired she will sleep", "she will sleep if she is tired"),
        ("cond_if_cold", "if it gets cold turn on the heater", "turn on the heater if it gets cold"),
        ("cond_if_ready", "if they are ready we can go", "we can go if they are ready"),
        ("cond_if_sunny", "if tomorrow is sunny visit the park", "visit the park if tomorrow is sunny"),
        ("cond_unless", "unless you study you will fail", "you will fail unless you study"),
        ("cond_because", "because it rained we stayed home", "we stayed home because it rained"),
    ]
    for name, a, b in cond_pairs:
        pairs.append({"name": name, "A": a, "B": b, "category": "conditional"})

    # === Category 9: Recursive / Embedded — 10 pairs (expanded) ===
    rec_pairs = [
        ("rec_believe_theory", "the scientist believes that the theory is correct",
         "the theory is believed by the scientist to be correct"),
        ("rec_dog_barked", "the dog that barked ran away", "the barking dog ran away"),
        ("rec_woman_wrote", "the woman who wrote the letter smiled", "the letter-writing woman smiled"),
        ("rec_king_said", "the king said that the queen is wise", "the queen is wise said the king"),
        ("rec_teacher_think", "the teacher thinks that the student learned well",
         "the student is thought by the teacher to have learned well"),
        ("rec_door_painted", "the door that was painted red opened", "the red door opened"),
        ("rec_book_which", "the book which i read yesterday was great", "yesterday i read a book which was great"),
        ("rec_person_who", "the person who called left a message", "a message was left by the person who called"),
        ("rec_that_fact", "the fact that he lied surprised everyone",
         "everyone was surprised by the fact that he lied"),
        ("rec_nested_if", "the man who said that if it rains he will leave arrived",
         "the man arrived who said that he will leave if it rains"),
    ]
    for name, a, b in rec_pairs:
        pairs.append({"name": name, "A": a, "B": b, "category": "recursive"})

    # === Category 10: Translation (EN↔ZH) — 10 pairs (expanded) ===
    trans_pairs = [
        ("trans_dog_cat", "the dog chases the cat", "狗追猫"),
        ("trans_sun_east", "the sun rises in the east", "太阳从东方升起"),
        ("trans_teacher_teach", "the teacher teaches the student", "老师教学生"),
        ("trans_bird_sky", "the bird flies in the sky", "鸟在天空中飞翔"),
        ("trans_king_rules", "the king rules the kingdom", "国王统治王国"),
        ("trans_water_cold", "the water is very cold", "水非常冷"),
        ("trans_love_overcome", "love overcomes everything", "爱战胜一切"),
        ("trans_fox_hunt", "the fox hunts the rabbit", "狐狸猎兔"),
        ("trans_child_happy", "the child is very happy", "孩子非常快乐"),
        ("trans_mountain_high", "the mountain is extremely high", "这座山非常高"),
    ]
    for name, a, b in trans_pairs:
        pairs.append({"name": name, "A": a, "B": b, "category": "translation"})

    # === Category 11: Comparative — 8 pairs (new) ===
    comp_pairs = [
        ("comp_bigger_than", "the elephant is bigger than the mouse",
         "the mouse is smaller than the elephant"),
        ("comp_taller_than", "John is taller than Mary", "Mary is shorter than John"),
        ("comp_more_expensive", "gold is more expensive than silver",
         "silver is less expensive than gold"),
        ("comp_faster_than", "the train is faster than the bicycle",
         "the bicycle is slower than the train"),
        ("comp_stronger_than", "steel is stronger than wood",
         "wood is weaker than steel"),
        ("comp_smarter_than", "Alice is smarter than Bob", "Bob is less smart than Alice"),
        ("comp_older_than", "the grandfather is older than the father",
         "the father is younger than the grandfather"),
        ("comp_more_than", "she has more books than he does",
         "he has fewer books than she does"),
    ]
    for name, a, b in comp_pairs:
        pairs.append({"name": name, "A": a, "B": b, "category": "comparative"})

    # === Category 12: Temporal — 8 pairs (new) ===
    temp_pairs = [
        ("temp_before_after", "before eating i washed my hands",
         "after washing my hands i ate"),
        ("temp_was_is", "the cat was hungry", "the cat is hungry"),
        ("temp_will_did", "the team will win the game", "the team won the game"),
        ("temp_is_going", "she is reading a book", "she was reading a book"),
        ("temp_since_until", "the shop has been open since morning",
         "the shop was open until evening"),
        ("temp_already_yet", "he has already finished his work",
         "he has not yet finished his work"),
        ("temp_while_when", "while i was cooking the phone rang",
         "when the phone rang i was cooking"),
        ("temp_still_no", "she is still working on the project",
         "she is no longer working on the project"),
    ]
    for name, a, b in temp_pairs:
        pairs.append({"name": name, "A": a, "B": b, "category": "temporal"})

    # === Category 13: Logical operators — 8 pairs (new) ===
    logic_pairs = [
        ("logic_and_or", "the cat and the dog are sleeping",
         "the cat or the dog is sleeping"),
        ("logic_both_either", "both Alice and Bob agreed",
         "either Alice or Bob agreed"),
        ("logic_not_only", "he is not only rich but also generous",
         "he is neither rich nor generous"),
        ("logic_although", "although it rained they went out",
         "they went out although it rained"),
        ("logic_therefore", "it is raining therefore we will stay home",
         "we will stay home therefore it is raining"),
        ("logic_however", "the test was hard however she passed",
         "she passed however the test was hard"),
        ("logic_moreover", "the food is delicious moreover it is healthy",
         "the food is healthy moreover it is delicious"),
        ("logic_nevertheless", "he was tired nevertheless he continued",
         "he continued nevertheless he was tired"),
    ]
    for name, a, b in logic_pairs:
        pairs.append({"name": name, "A": a, "B": b, "category": "logical"})

    # === Category 14: Abstract concepts — 8 pairs (expanded) ===
    abstract_pairs = [
        ("abs_justice_corruption", "justice prevails over corruption",
         "corruption prevails over justice"),
        ("abs_wisdom_folly", "wisdom overcomes folly", "folly overcomes wisdom"),
        ("abs_courage_fear", "courage defeats fear", "fear defeats courage"),
        ("abs_love_hate", "love conquers hate", "hate conquers love"),
        ("abs_truth_lie", "truth defeats lies", "lies defeat truth"),
        ("abs_hope_despair", "hope overcomes despair", "despair overcomes hope"),
        ("abs_freedom_tyranny", "freedom resists tyranny", "tyranny resists freedom"),
        ("abs_knowledge_ignorance", "knowledge dispels ignorance", "ignorance dispels knowledge"),
    ]
    for name, a, b in abstract_pairs:
        pairs.append({"name": name, "A": a, "B": b, "category": "abstract"})

    return pairs

# =============================================================================
# Main runner
# =============================================================================

def run_phase284(model_name):
    global _log_file
    _log_file = str(TMP_DIR / f"phase284_{model_name}.txt")

    log_time(f"{'='*60}")
    log_time(f"Phase 284: Calibration + Full Function Matrix — {model_name}")
    log_time(f"{'='*60}")

    # Build extended pairs
    all_pairs = build_all_pairs_phase284()
    cat_counts = defaultdict(int)
    for p in all_pairs:
        cat_counts[p["category"]] += 1
    log_time(f"Dataset: {len(all_pairs)} pairs across {len(cat_counts)} categories")
    for cat, cnt in sorted(cat_counts.items()):
        log_time(f"  {cat}: {cnt}")

    # Load model
    model, tokenizer, device, attn_impl = load_model_bf16_flash(model_name)
    model_info = get_model_info(model, model_name)
    log_time(f"Model: {model_info.model_class}, L={model_info.n_layers}, d={model_info.d_model}")

    # Warmup
    log_time("Global warmup...")
    wu = tokenizer("warmup test", return_tensors="pt").to(device)
    with torch.no_grad():
        try:
            model(**wu)
        except:
            pass
    log_time(f"Warmup done, GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")

    all_results = {}

    try:
        # === Block 0: Calibration ===
        t0 = time.time()
        try:
            calib = block0_calibration(model, tokenizer, device, model_name, attn_impl)
        except Exception as e:
            log_time(f"Block 0 calibration exception: {e}")
            calib = {"per_layer": {}, "calibration_failed": True, "reason": str(e)}
        t_calib = time.time() - t0
        log_time(f"\nBlock 0 done in {t_calib:.1f}s ({t_calib/60:.1f}min)")
        all_results["calibration"] = calib

        # === Block 1: Full function matrix ===
        t0 = time.time()
        matrix = block1_absolute_normalized_effects(model, tokenizer, device, model_name, all_pairs, calib)
        t_matrix = time.time() - t0
        log_time(f"\nBlock 1 done in {t_matrix:.1f}s ({t_matrix/60:.1f}min)")
        all_results["matrix"] = matrix

        # === Save results ===
        if calib:
            with open(RESULT_DIR / f"{model_name}_block0_calibration.json", "w") as f:
                json.dump(calib, f, indent=2)

        if matrix:
            save_dict = {
                "per_layer": matrix["per_layer"],
                "per_category": {cat: dict(data) for cat, data in matrix["per_category"].items()},
                "model": matrix["model"],
                "n_pairs": matrix["n_pairs"],
                "n_layers_tested": matrix["n_layers_tested"],
                "n_unreliable_layers": matrix.get("n_unreliable_layers", 0),
            }
            with open(RESULT_DIR / f"{model_name}_block1_full_matrix.json", "w") as f:
                json.dump(save_dict, f, indent=2)

        # === Build final summary ===
        log_time(f"\n{'='*60}")
        log_time(f"Phase 284 Final Summary — {model_name}")
        log_time(f"{'='*60}")

        if calib:
            cal_agg = calib.get("per_layer", {})
            n_reliable = sum(1 for v in cal_agg.values() if v.get("is_reliable", False))
            log_time(f"Calibration: {n_reliable}/{len(cal_agg)} layers reliable")

        if matrix:
            # Count VALUE vs WEIGHT layers
            pl = matrix.get("per_layer", {})
            n_val = sum(1 for v in pl.values() if v.get("primary_component") == "VALUE")
            n_wt = sum(1 for v in pl.values() if v.get("primary_component") == "WEIGHT")
            log_time(f"Full Matrix: {n_wt} WEIGHT-dominant, {n_val} VALUE-dominant layers")

            # Key category insights
            for cat in ["animal", "passive", "negation", "conditional", "recursive", "translation"]:
                cat_data = matrix.get("per_category", {}).get(cat, {})
                if cat_data:
                    n_val_cat = sum(1 for v in cat_data.values() if v.get("primary_component") == "VALUE")
                    n_wt_cat = sum(1 for v in cat_data.values() if v.get("primary_component") == "WEIGHT")
                    log_time(f"  {cat}: {n_wt_cat}W/{n_val_cat}V")

        log_time(f"\nPhase 284 complete for {model_name}: calib={t_calib:.0f}s, matrix={t_matrix:.0f}s")

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
                r = run_phase284(name)
                log_time(f"\n{name} DONE")
            except Exception as e:
                log_time(f"!!! {name} FAILED: {e}")
                import traceback
                traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(3)
    else:
        run_phase284(model_name)
