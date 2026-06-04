"""
Phase 365: Full Attention C2R + Post-LN Logit Lens + Rigid Transfer Analysis
=============================================================================

Key improvements over Phase 364:
1. Full attention C2R at ALL layers with full-output replacement
   (Phase 364 only did key layers, and only last-token replacement)
2. Post-LN Logit Lens: apply model.model.norm before W_U projection
   - Corrects magnitude inflation in deep layers
   - Phase 364 used raw h @ W_U, which overestimates deep-layer signals
3. Rigid transfer analysis: Δh cosine similarity across layers
   - Tests whether binding signal is preserved in a "protected subspace"
   - Computes: ||Δh||, cos_sim(Δh_l, Δh_{l-1}), angle(Δh, W_U direction)
4. Dual-component layer classification: attn + MLP (Phase 364 data)

MLP C2R is NOT re-run here; Phase 364 data is loaded for comparison.

Estimated runtime:
  Qwen3: ~3 min | GLM4: ~36 min | DS7B: ~15 min
"""

import sys, os, time, json, gc
import torch
import numpy as np
from datetime import datetime
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')


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

# Full test pairs (42) - same as Phase 364 for direct comparison
TEST_PAIRS = [
    ("apple", "red", "blue"), ("banana", "yellow", "purple"), ("snow", "white", "black"),
    ("sky", "blue", "green"), ("cherry", "red", "blue"), ("leaf", "green", "red"),
    ("rose", "red", "blue"), ("gold", "yellow", "purple"), ("coal", "black", "white"),
    ("silver", "white", "black"), ("milk", "white", "black"), ("honey", "yellow", "blue"),
    ("ruby", "red", "green"), ("emerald", "green", "red"), ("sapphire", "blue", "red"),
    ("moon", "white", "black"), ("flame", "orange", "blue"), ("forest", "green", "white"),
    ("ocean", "blue", "yellow"), ("sun", "yellow", "purple"),
    ("fire", "hot", "cold"), ("desert", "hot", "cold"), ("lava", "hot", "cold"),
    ("ice", "cold", "hot"), ("snow", "cold", "hot"), ("volcano", "hot", "cold"),
    ("furnace", "hot", "cold"), ("glacier", "cold", "hot"),
    ("rain", "wet", "dry"), ("ocean", "wet", "dry"), ("river", "wet", "dry"),
    ("sand", "dry", "wet"), ("dust", "dry", "wet"), ("bone", "dry", "wet"),
    ("swamp", "wet", "dry"), ("desert", "dry", "wet"),
    ("silk", "smooth", "rough"), ("sandpaper", "rough", "smooth"),
    ("glass", "smooth", "rough"), ("rock", "rough", "smooth"),
    ("velvet", "soft", "hard"), ("diamond", "hard", "soft"),
]

CORRUPTED_BASELINE = "The item"


# ===== Model Loading =====

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
            log(f"  Failed with {impl}: {str(e)[:80]}")
            continue
    if model is None:
        raise RuntimeError(f"Failed to load {model_name}")
    model.eval()

    if hasattr(model, 'hf_device_map'):
        dmap = model.hf_device_map
        gpu_count = sum(1 for v in dmap.values() if 'cuda' in str(v))
        cpu_count = sum(1 for v in dmap.values() if 'cpu' in str(v))
        gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        log(f"  GPU={gpu_count} components, CPU={cpu_count} components, GPU mem={gpu_mem:.2f}GB")

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


def get_final_norm_params(model, model_name):
    """Get the final normalization layer parameters and type.
    Handles meta tensors (offloaded to CPU) by loading from safetensors."""
    norm_layer = None
    # Try common locations for different model architectures
    candidates = [
        ('model.model.norm', lambda m: m.model.norm),
        ('model.transformer.encoder.final_layernorm', lambda m: m.transformer.encoder.final_layernorm),
    ]
    for name, getter in candidates:
        try:
            obj = getter(model)
            if obj is not None and hasattr(obj, 'weight'):
                norm_layer = obj
                norm_name = name
                break
        except AttributeError:
            continue

    if norm_layer is None:
        raise ValueError("Cannot find final normalization layer")

    norm_type = type(norm_layer).__name__
    eps = getattr(norm_layer, 'variance_epsilon', None) or getattr(norm_layer, 'eps', 1e-6)

    # Try to get weight directly; if meta tensor, load from safetensors
    weight = None
    bias = None
    try:
        w = norm_layer.weight
        if not w.is_meta:
            weight = w.detach().cpu().float().numpy()
    except (NotImplementedError, RuntimeError):
        pass

    if weight is None:
        # Load from safetensors file
        import glob
        from safetensors import safe_open
        cfg_path = MODEL_CONFIGS[model_name]["path"]
        # Map norm_name to safetensors key
        norm_key_map = {
            'model.model.norm': 'model.norm.weight',
            'model.transformer.encoder.final_layernorm': 'transformer.encoder.final_layernorm.weight',
        }
        target_key = norm_key_map.get(norm_name, 'model.norm.weight')
        log(f"  Loading norm weight from safetensors (key={target_key})...")
        for sf_file in glob.glob(os.path.join(cfg_path, '*.safetensors')):
            with safe_open(sf_file, framework='pt', device='cpu') as sf:
                if target_key in sf.keys():
                    weight = sf.get_tensor(target_key).float().numpy()
                    # Try to get bias too
                    bias_key = target_key.replace('.weight', '.bias')
                    if bias_key in sf.keys():
                        bias = sf.get_tensor(bias_key).float().numpy()
                    break

    if weight is None:
        raise ValueError(f"Cannot load norm weight for {model_name}")

    # Also try bias from the layer object
    if bias is None and hasattr(norm_layer, 'bias') and norm_layer.bias is not None:
        try:
            if not norm_layer.bias.is_meta:
                bias = norm_layer.bias.detach().cpu().float().numpy()
        except (NotImplementedError, RuntimeError):
            pass

    log(f"  Final norm: type={norm_type}, eps={eps}, has_bias={bias is not None}, weight_shape={weight.shape}")
    return {'type': norm_type, 'weight': weight, 'bias': bias, 'eps': eps}


def rms_norm_numpy(x, weight, eps=1e-6):
    """Apply RMSNorm: x * weight / sqrt(mean(x^2) + eps)"""
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return x / rms * weight


def layer_norm_numpy(x, weight, bias=None, eps=1e-6):
    """Apply LayerNorm: (x - mean) / std * weight + bias"""
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    normalized = (x - mean) / (std + eps) * weight
    if bias is not None:
        normalized += bias
    return normalized


def apply_final_norm(x, norm_params):
    """Apply the model's final normalization to hidden states."""
    if 'RMS' in norm_params['type'] or 'Rms' in norm_params['type']:
        return rms_norm_numpy(x, norm_params['weight'], norm_params['eps'])
    else:
        return layer_norm_numpy(x, norm_params['weight'], norm_params['bias'], norm_params['eps'])


def get_token_id(tokenizer, word):
    ids = tokenizer.encode(word, add_special_tokens=False)
    return ids[0] if ids else None


def find_object_positions(tokenizer, prompt, obj_word):
    input_ids = tokenizer.encode(prompt)
    positions = []
    for i, tid in enumerate(input_ids):
        decoded = tokenizer.decode([tid]).strip().lower()
        if obj_word.lower() in decoded and decoded != '':
            positions.append(i)
    if not positions:
        positions = [1] if len(input_ids) > 1 else [0]
    return positions


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("Cannot find layers")


def compute_effect(delta_gap, base_gap, direction):
    abs_base = abs(base_gap)
    if abs_base < 1e-10:
        return 0.0
    if direction == "c2r":
        return -delta_gap / abs_base
    else:
        return delta_gap / abs_base


# ===== Hook Helpers =====

def _make_output_patch_hook_full(replacement):
    """Replace output at ALL token positions (for attention C2R)."""
    def hook(module, input, output):
        val = output[0] if isinstance(output, tuple) else output
        modified = val.clone()
        rep_t = torch.tensor(replacement, dtype=modified.dtype, device=modified.device)
        n = min(modified.shape[1], rep_t.shape[0])
        modified[0, :n, :] = rep_t[:n, :]
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified
    return hook


# ===== Load Phase 364 MLP C2R data =====

def load_phase364_mlp(model_name):
    """Load Phase 364 MLP C2R summary for dual-component classification."""
    path = f"results/phase364_layer_role/{model_name}_phase364.json"
    if not os.path.exists(path):
        log(f"  WARNING: Phase 364 data not found at {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("mlp_c2r_summary", {})


# ===== Main Experiment =====

def run_phase365(model_name):
    log(f"\n{'='*60}")
    log(f"Phase 365: {model_name}")
    log(f"{'='*60}")

    cfg = MODEL_CONFIGS[model_name]
    n_layers = cfg["n_layers"]

    # Load model
    t0_load = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    W_U = get_W_U(model, model_name)
    norm_params = get_final_norm_params(model, model_name)
    layers = get_layers(model)
    log(f"  Load time: {time.time()-t0_load:.0f}s, n_layers={n_layers}, d_model={cfg['d_model']}")

    # Load Phase 364 MLP C2R for dual-component comparison
    mlp_c2r_p364 = load_phase364_mlp(model_name)
    if mlp_c2r_p364:
        log(f"  Loaded Phase 364 MLP C2R data ({len(mlp_c2r_p364)} layers)")

    # Results storage
    results = {
        "model": model_name,
        "phase": "365",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_pairs": len(TEST_PAIRS),
        "n_layers": n_layers,
        "norm_type": norm_params['type'],
        "attention_c2r": {},        # layer -> [per_pair_effects]
        "post_ln_logit_lens": {},   # layer -> {clean_gaps, corrupt_gaps, binding_signals}
        "raw_logit_lens": {},       # layer -> {clean_gaps, corrupt_gaps, binding_signals}
        "rigid_transfer_last": {},  # layer -> {delta_norm, cos_sim_with_prev, angle_with_wu}
        "rigid_transfer_obj": {},   # layer -> {delta_norm, cos_sim_with_prev, angle_with_wu}
    }

    t0_total = time.time()
    n_pairs = len(TEST_PAIRS)
    valid_pairs = 0

    for pidx, (obj, target, competitor) in enumerate(TEST_PAIRS):
        tid_t = get_token_id(tokenizer, target)
        tid_c = get_token_id(tokenizer, competitor)
        if tid_t is None or tid_c is None:
            log(f"  Skip pair ({obj}, {target}, {competitor}): token not found")
            continue

        clean_prompt = f"The {obj}"

        # Compute W_U direction for this pair
        w_u_dir = W_U[tid_t] - W_U[tid_c]  # [d_model]
        w_u_dir_norm = w_u_dir / (np.linalg.norm(w_u_dir) + 1e-10)

        # ===== Step 1: Clean forward pass =====
        inp_clean = tokenizer(clean_prompt, return_tensors="pt", truncation=True, max_length=128).to(device)

        # Find object position in clean sequence
        obj_positions_clean = find_object_positions(tokenizer, clean_prompt, obj)
        obj_pos_clean = obj_positions_clean[0] if obj_positions_clean else 1

        with torch.no_grad():
            out_clean = model(**inp_clean, output_hidden_states=True)
        clean_logits = out_clean.logits[0, -1].float().cpu().numpy()
        clean_hs = out_clean.hidden_states  # tuple of (n_layers+1) tensors

        clean_gap = float(clean_logits[tid_t] - clean_logits[tid_c])
        clean_seq_len = inp_clean["input_ids"].shape[1]

        # Extract h at last and object positions for each layer
        clean_h_last = []
        clean_h_obj = []
        for li in range(len(clean_hs)):
            hs_np = clean_hs[li][0].float().cpu().numpy()  # [seq_len, d_model]
            clean_h_last.append(hs_np[-1])
            if obj_pos_clean < hs_np.shape[0]:
                clean_h_obj.append(hs_np[obj_pos_clean])
            else:
                clean_h_obj.append(None)

        # Raw and Post-LN Logit Lens for clean
        clean_ll_raw_gaps = []
        clean_ll_postln_gaps = []
        for li in range(len(clean_hs)):
            hs_last = clean_h_last[li]
            # Raw: h @ W_U
            raw_gap = float(hs_last @ W_U[tid_t] - hs_last @ W_U[tid_c])
            clean_ll_raw_gaps.append(raw_gap)
            # Post-LN: norm(h) @ W_U
            hs_normed = apply_final_norm(hs_last, norm_params)
            postln_gap = float(hs_normed @ W_U[tid_t] - hs_normed @ W_U[tid_c])
            clean_ll_postln_gaps.append(postln_gap)

        del clean_hs, out_clean
        gc.collect()

        # ===== Step 2: Corrupt forward pass =====
        # Capture attn_out at ALL layers (full output, all positions)
        corrupt_attn_all = {}

        def make_post_hook_attn_full(key):
            def hook(module, input, output):
                val = output[0] if isinstance(output, tuple) else output
                corrupt_attn_all[key] = val[0].detach().cpu().float().numpy()  # [seq_len, d_model]
            return hook

        hooks_2 = []
        for li in range(n_layers):
            hooks_2.append(layers[li].self_attn.register_forward_hook(
                make_post_hook_attn_full(f"attn_{li}")))

        inp_corrupt = tokenizer(CORRUPTED_BASELINE, return_tensors="pt", truncation=True, max_length=128).to(device)

        # Find object position in corrupt sequence
        obj_positions_corrupt = find_object_positions(tokenizer, CORRUPTED_BASELINE, "item")
        obj_pos_corrupt = obj_positions_corrupt[0] if obj_positions_corrupt else 1

        with torch.no_grad():
            out_corrupt = model(**inp_corrupt, output_hidden_states=True)
        corrupt_logits_val = out_corrupt.logits[0, -1].float().cpu().numpy()
        corrupt_hs = out_corrupt.hidden_states

        for h in hooks_2:
            h.remove()

        corrupt_gap = float(corrupt_logits_val[tid_t] - corrupt_logits_val[tid_c])
        base_gap = clean_gap - corrupt_gap
        corrupt_seq_len = inp_corrupt["input_ids"].shape[1]

        # Extract h at last and object positions for each layer
        corrupt_h_last = []
        corrupt_h_obj = []
        for li in range(len(corrupt_hs)):
            hs_np = corrupt_hs[li][0].float().cpu().numpy()
            corrupt_h_last.append(hs_np[-1])
            if obj_pos_corrupt < hs_np.shape[0]:
                corrupt_h_obj.append(hs_np[obj_pos_corrupt])
            else:
                corrupt_h_obj.append(None)

        # Raw and Post-LN Logit Lens for corrupt
        for li in range(len(corrupt_hs)):
            hs_last = corrupt_h_last[li]
            # Raw
            raw_gap = float(hs_last @ W_U[tid_t] - hs_last @ W_U[tid_c])
            # Post-LN
            hs_normed = apply_final_norm(hs_last, norm_params)
            postln_gap = float(hs_normed @ W_U[tid_t] - hs_normed @ W_U[tid_c])

            li_str = str(li)

            # Raw logit lens
            if li_str not in results["raw_logit_lens"]:
                results["raw_logit_lens"][li_str] = {"clean_gaps": [], "corrupt_gaps": [], "binding_signals": []}
            results["raw_logit_lens"][li_str]["clean_gaps"].append(clean_ll_raw_gaps[li])
            results["raw_logit_lens"][li_str]["corrupt_gaps"].append(raw_gap)
            results["raw_logit_lens"][li_str]["binding_signals"].append(clean_ll_raw_gaps[li] - raw_gap)

            # Post-LN logit lens
            if li_str not in results["post_ln_logit_lens"]:
                results["post_ln_logit_lens"][li_str] = {"clean_gaps": [], "corrupt_gaps": [], "binding_signals": []}
            results["post_ln_logit_lens"][li_str]["clean_gaps"].append(clean_ll_postln_gaps[li])
            results["post_ln_logit_lens"][li_str]["corrupt_gaps"].append(postln_gap)
            results["post_ln_logit_lens"][li_str]["binding_signals"].append(clean_ll_postln_gaps[li] - postln_gap)

        del corrupt_hs, out_corrupt, clean_ll_raw_gaps, clean_ll_postln_gaps
        gc.collect()

        # ===== Step 3: Rigid transfer analysis =====
        valid_pairs += 1

        # Δh at last token position
        delta_h_last = []
        for li in range(len(clean_h_last)):
            if clean_h_last[li] is not None and corrupt_h_last[li] is not None:
                delta_h_last.append(clean_h_last[li] - corrupt_h_last[li])
            else:
                delta_h_last.append(None)

        # Compute statistics for Δh at last token
        for li in range(len(delta_h_last)):
            if delta_h_last[li] is None:
                continue
            dh = delta_h_last[li]
            li_str = str(li)

            if li_str not in results["rigid_transfer_last"]:
                results["rigid_transfer_last"][li_str] = {
                    "delta_norm": [], "cos_sim_with_prev": [], "angle_with_wu": []
                }

            # Norm of Δh
            dh_norm = float(np.linalg.norm(dh))
            results["rigid_transfer_last"][li_str]["delta_norm"].append(dh_norm)

            # Cosine similarity with previous layer's Δh
            if li > 0 and delta_h_last[li-1] is not None:
                prev_norm = float(np.linalg.norm(delta_h_last[li-1]))
                if dh_norm > 1e-10 and prev_norm > 1e-10:
                    cos_sim = float(np.dot(dh, delta_h_last[li-1]) / (dh_norm * prev_norm))
                else:
                    cos_sim = 0.0
                results["rigid_transfer_last"][li_str]["cos_sim_with_prev"].append(cos_sim)

            # Angle with W_U direction
            if dh_norm > 1e-10:
                cos_wu = float(np.dot(dh, w_u_dir_norm) / dh_norm)
                angle_wu = float(np.arccos(np.clip(cos_wu, -1, 1)) * 180 / np.pi)
            else:
                angle_wu = 90.0
            results["rigid_transfer_last"][li_str]["angle_with_wu"].append(angle_wu)

        # Δh at object position (only when positions align and same seq length)
        if obj_pos_clean == obj_pos_corrupt and clean_seq_len == corrupt_seq_len:
            delta_h_obj = []
            for li in range(len(clean_h_obj)):
                if clean_h_obj[li] is not None and corrupt_h_obj[li] is not None:
                    delta_h_obj.append(clean_h_obj[li] - corrupt_h_obj[li])
                else:
                    delta_h_obj.append(None)

            for li in range(len(delta_h_obj)):
                if delta_h_obj[li] is None:
                    continue
                dh = delta_h_obj[li]
                li_str = str(li)

                if li_str not in results["rigid_transfer_obj"]:
                    results["rigid_transfer_obj"][li_str] = {
                        "delta_norm": [], "cos_sim_with_prev": [], "angle_with_wu": []
                    }

                dh_norm = float(np.linalg.norm(dh))
                results["rigid_transfer_obj"][li_str]["delta_norm"].append(dh_norm)

                if li > 0 and delta_h_obj[li-1] is not None:
                    prev_norm = float(np.linalg.norm(delta_h_obj[li-1]))
                    if dh_norm > 1e-10 and prev_norm > 1e-10:
                        cos_sim = float(np.dot(dh, delta_h_obj[li-1]) / (dh_norm * prev_norm))
                    else:
                        cos_sim = 0.0
                    results["rigid_transfer_obj"][li_str]["cos_sim_with_prev"].append(cos_sim)

                if dh_norm > 1e-10:
                    cos_wu = float(np.dot(dh, w_u_dir_norm) / dh_norm)
                    angle_wu = float(np.arccos(np.clip(cos_wu, -1, 1)) * 180 / np.pi)
                else:
                    angle_wu = 90.0
                results["rigid_transfer_obj"][li_str]["angle_with_wu"].append(angle_wu)

        if abs(base_gap) < 1e-10:
            log(f"  Pair {pidx+1} ({obj}): base_gap≈0, skip C2R")
            del corrupt_attn_all, clean_h_last, clean_h_obj, corrupt_h_last, corrupt_h_obj
            gc.collect(); torch.cuda.empty_cache()
            continue

        # ===== Step 4: Attention C2R at all layers =====
        for li in range(n_layers):
            corrupt_attn = corrupt_attn_all.get(f"attn_{li}")
            if corrupt_attn is None:
                continue

            # Full output replacement (all token positions)
            hook = layers[li].self_attn.register_forward_hook(
                _make_output_patch_hook_full(corrupt_attn))
            with torch.no_grad():
                pout = model(**inp_clean, output_hidden_states=False)
            p_logits = pout.logits[0, -1].float().cpu().numpy()
            hook.remove()

            p_gap = float(p_logits[tid_t] - p_logits[tid_c])
            delta_gap = p_gap - clean_gap
            c2r_effect = compute_effect(delta_gap, base_gap, "c2r")

            if str(li) not in results["attention_c2r"]:
                results["attention_c2r"][str(li)] = []
            results["attention_c2r"][str(li)].append(c2r_effect)

            del pout
            if li % 6 == 5:
                gc.collect(); torch.cuda.empty_cache()

        # Cleanup per-pair
        del corrupt_attn_all, clean_h_last, clean_h_obj, corrupt_h_last, corrupt_h_obj
        gc.collect(); torch.cuda.empty_cache()

        if (pidx + 1) % 3 == 0:
            elapsed = time.time() - t0_total
            gpu_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            eta = elapsed / (pidx + 1) * (n_pairs - pidx - 1)
            log(f"  [{pidx+1}/{n_pairs}] elapsed={elapsed:.0f}s ETA={eta:.0f}s GPU={gpu_gb:.1f}GB")

    # ===== Post-processing =====
    log("\n  Computing statistics...")

    # Attention C2R summary
    attn_c2r_summary = {}
    for li_str, effects in results["attention_c2r"].items():
        arr = np.array(effects)
        attn_c2r_summary[li_str] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "n": len(arr),
            "pos_rate": float(np.mean(arr > 0)),
            "neg_rate": float(np.mean(arr < 0)),
        }

    # Raw Logit Lens summary
    raw_ll_summary = {}
    for li_str, data in results["raw_logit_lens"].items():
        if len(data["clean_gaps"]) == 0:
            continue
        cg = np.array(data["clean_gaps"])
        crg = np.array(data["corrupt_gaps"])
        bs = np.array(data["binding_signals"])
        raw_ll_summary[li_str] = {
            "clean_gap_mean": float(np.mean(cg)),
            "corrupt_gap_mean": float(np.mean(crg)),
            "binding_signal_mean": float(np.mean(bs)),
            "binding_signal_std": float(np.std(bs)),
        }

    # Post-LN Logit Lens summary
    postln_ll_summary = {}
    for li_str, data in results["post_ln_logit_lens"].items():
        if len(data["clean_gaps"]) == 0:
            continue
        cg = np.array(data["clean_gaps"])
        crg = np.array(data["corrupt_gaps"])
        bs = np.array(data["binding_signals"])
        postln_ll_summary[li_str] = {
            "clean_gap_mean": float(np.mean(cg)),
            "corrupt_gap_mean": float(np.mean(crg)),
            "binding_signal_mean": float(np.mean(bs)),
            "binding_signal_std": float(np.std(bs)),
        }

    # Rigid transfer summary (last token)
    rt_last_summary = {}
    for li_str, data in results["rigid_transfer_last"].items():
        rt_last_summary[li_str] = {
            "delta_norm_mean": float(np.mean(data["delta_norm"])) if data["delta_norm"] else 0,
            "delta_norm_std": float(np.std(data["delta_norm"])) if data["delta_norm"] else 0,
            "cos_sim_mean": float(np.mean(data["cos_sim_with_prev"])) if data["cos_sim_with_prev"] else None,
            "cos_sim_std": float(np.std(data["cos_sim_with_prev"])) if data["cos_sim_with_prev"] else None,
            "angle_wu_mean": float(np.mean(data["angle_with_wu"])) if data["angle_with_wu"] else None,
            "angle_wu_std": float(np.std(data["angle_with_wu"])) if data["angle_with_wu"] else None,
        }

    # Rigid transfer summary (object position)
    rt_obj_summary = {}
    for li_str, data in results["rigid_transfer_obj"].items():
        rt_obj_summary[li_str] = {
            "delta_norm_mean": float(np.mean(data["delta_norm"])) if data["delta_norm"] else 0,
            "delta_norm_std": float(np.std(data["delta_norm"])) if data["delta_norm"] else 0,
            "cos_sim_mean": float(np.mean(data["cos_sim_with_prev"])) if data["cos_sim_with_prev"] else None,
            "cos_sim_std": float(np.std(data["cos_sim_with_prev"])) if data["cos_sim_with_prev"] else None,
            "angle_wu_mean": float(np.mean(data["angle_with_wu"])) if data["angle_with_wu"] else None,
            "angle_wu_std": float(np.std(data["angle_with_wu"])) if data["angle_with_wu"] else None,
        }

    # ===== Dual-component layer classification =====
    log("  Computing dual-component layer classification...")
    dual_roles = {}
    for li in range(n_layers):
        li_str = str(li)
        attn_mean = attn_c2r_summary.get(li_str, {}).get("mean", 0)
        mlp_mean = mlp_c2r_p364.get(li_str, {}).get("mean", 0) if mlp_c2r_p364 else 0
        postln_bs = postln_ll_summary.get(li_str, {}).get("binding_signal_mean", 0) if li < len(postln_ll_summary) else 0
        postln_bs_next = postln_ll_summary.get(str(li+1), {}).get("binding_signal_mean", 0)
        ll_delta = postln_bs_next - postln_bs

        # Classify each component
        def classify_component(c2r_val, layer_frac, ll_signal):
            if c2r_val < -0.1:
                return "calibration"
            elif c2r_val > 0.15 and layer_frac < 0.4:
                return "writing"
            elif c2r_val > 0.15 and layer_frac > 0.7:
                return "readout"
            elif c2r_val > 0.15:
                return "mid_contribution"
            elif abs(c2r_val) <= 0.1:
                return "carrying"
            else:
                return "weak_contrib"

        attn_role = classify_component(attn_mean, li / n_layers, postln_bs)
        mlp_role = classify_component(mlp_mean, li / n_layers, postln_bs)

        # Combined role
        if attn_mean < -0.1 or mlp_mean < -0.1:
            combined = "calibration"
        elif attn_mean > 0.15 and mlp_mean > 0.15:
            combined = "dual_writing" if li < n_layers * 0.4 else "dual_readout"
        elif attn_mean > 0.15:
            combined = "attn_dominant"
        elif mlp_mean > 0.15:
            combined = "mlp_dominant"
        elif abs(attn_mean) <= 0.1 and abs(mlp_mean) <= 0.1:
            combined = "pure_carrying"
        else:
            combined = "mixed"

        dual_roles[li_str] = {
            "attn_c2r_mean": attn_mean,
            "mlp_c2r_mean": mlp_mean,
            "attn_role": attn_role,
            "mlp_role": mlp_role,
            "combined_role": combined,
            "post_ln_binding": postln_bs,
            "ll_delta": ll_delta,
        }

    # Bootstrap CI for attention C2R
    log("  Computing bootstrap CIs for attention C2R...")
    bootstrap_ci = {}
    np.random.seed(42)
    for li in range(n_layers):
        li_str = str(li)
        effects = results["attention_c2r"].get(li_str, [])
        if len(effects) < 5:
            continue
        arr = np.array(effects)
        boots = []
        for _ in range(1000):
            sample = np.random.choice(arr, size=len(arr), replace=True)
            boots.append(np.mean(sample))
        bootstrap_ci[li_str] = {
            "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)),
        }

    # ===== Save results =====
    output = {
        "model": model_name,
        "phase": "365",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_pairs": valid_pairs,
        "n_layers": n_layers,
        "norm_type": norm_params['type'],
        "total_time_s": round(time.time() - t0_total, 1),
        "attention_c2r_summary": attn_c2r_summary,
        "raw_logit_lens_summary": raw_ll_summary,
        "post_ln_logit_lens_summary": postln_ll_summary,
        "rigid_transfer_last_summary": rt_last_summary,
        "rigid_transfer_obj_summary": rt_obj_summary,
        "dual_roles": dual_roles,
        "bootstrap_ci_attn": bootstrap_ci,
        "attention_c2r_per_pair": {k: [round(x, 4) for x in v] for k, v in results["attention_c2r"].items()},
    }

    os.makedirs("results/phase365_dual_component", exist_ok=True)
    out_path = f"results/phase365_dual_component/{model_name}_phase365.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log(f"  Saved to {out_path}")

    # ===== Print summary =====
    log(f"\n  === Attention C2R Summary ===")
    for li in range(n_layers):
        li_str = str(li)
        s = attn_c2r_summary.get(li_str, {})
        mean = s.get("mean", 0)
        mlp_m = mlp_c2r_p364.get(li_str, {}).get("mean", 0) if mlp_c2r_p364 else 0
        marker = ""
        if mean > 0.2:
            marker = " ***"
        elif mean < -0.1:
            marker = " (neg)"
        log(f"    L{li:2d}: Attn_C2R={mean:+.3f}  MLP_C2R={mlp_m:+.3f}{marker}")

    log(f"\n  === Post-LN vs Raw Logit Lens ===")
    sample_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    for li in sample_layers:
        li_str = str(li)
        raw_bs = raw_ll_summary.get(li_str, {}).get("binding_signal_mean", 0)
        postln_bs = postln_ll_summary.get(li_str, {}).get("binding_signal_mean", 0)
        ratio = postln_bs / raw_bs if abs(raw_bs) > 0.01 else 0
        log(f"    L{li:2d}: Raw={raw_bs:+.3f}  Post-LN={postln_bs:+.3f}  ratio={ratio:.3f}")

    log(f"\n  === Rigid Transfer (last token Δh) ===")
    for li in sample_layers:
        li_str = str(li)
        rt = rt_last_summary.get(li_str, {})
        norm = rt.get("delta_norm_mean", 0)
        cos = rt.get("cos_sim_mean", None)
        angle = rt.get("angle_wu_mean", None)
        cos_str = f"{cos:.4f}" if cos is not None else "N/A"
        angle_str = f"{angle:.1f}°" if angle is not None else "N/A"
        log(f"    L{li:2d}: ||Δh||={norm:.2f}  cos_sim(prev)={cos_str}  angle(W_U)={angle_str}")

    log(f"\n  === Dual-Component Layer Classification ===")
    role_counts = defaultdict(int)
    for v in dual_roles.values():
        role_counts[v["combined_role"]] += 1
    for role, count in sorted(role_counts.items()):
        log(f"    {role}: {count} layers")

    total_time = time.time() - t0_total
    log(f"\n  Total time: {total_time:.0f}s ({total_time/60:.1f}min)")
    log(f"  Valid pairs: {valid_pairs}")

    # Release model
    del model, W_U, layers
    gc.collect(); torch.cuda.empty_cache()

    return output


# ===== Entry Point =====
if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    if model_name not in MODEL_CONFIGS:
        log(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)
    run_phase365(model_name)
