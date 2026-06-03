"""
Phase 357: Block Patch Circuit Verification + Unified Notation
==============================================================

Goals:
1. Test multi-layer block patches to verify if binding is a circuit (layer interaction)
   rather than a single-layer process
2. Unify notation: effect = -Δgap / |base_gap| (C2R), rescue = +Δgap / |base_gap| (R2C)
3. Check super-additivity: block_effect > sum(single_effects) ?
4. Test cproj+dact coupling: separate vs combined patch at block level

Block configurations:
  Qwen3: L23-only, L21+L23, L23+L25, L21-L27, all_binding
  GLM4:  L38-only, L36+L38, L30-L38, all_binding
  DS7B:  L19-only, L19+L21, L19-L24, all_binding

For each block:
  - cproj-only patch (C2R, R2C)
  - dact-only patch (C2R, R2C)
  - cproj+dact combined patch (C2R, R2C)

Super-additivity test:
  If block_effect > sum(single_effects), binding is a circuit, not independent layers.
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
        "n_layers": 36, "d_model": 2560, "d_ff": 9728,
        "binding_layers": [21, 23, 25, 27, 29],
        "block_configs": [
            {"name": "L23_only", "layers": [23]},
            {"name": "L21+L23", "layers": [21, 23]},
            {"name": "L23+L25", "layers": [23, 25]},
            {"name": "L21-L27", "layers": [21, 23, 25, 27]},
            {"name": "L21-L29", "layers": [21, 23, 25, 27, 29]},
        ],
    },
    "glm4": {
        "path": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
        "n_layers": 40, "d_model": 4096, "d_ff": 13696,
        "binding_layers": [30, 33, 36, 38],
        "block_configs": [
            {"name": "L38_only", "layers": [38]},
            {"name": "L36+L38", "layers": [36, 38]},
            {"name": "L33-L38", "layers": [33, 36, 38]},
            {"name": "L30-L38", "layers": [30, 33, 36, 38]},
        ],
    },
    "deepseek7b": {
        "path": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "n_layers": 28, "d_model": 3584, "d_ff": 18944,
        "binding_layers": [19, 21, 23, 24],
        "block_configs": [
            {"name": "L19_only", "layers": [19]},
            {"name": "L19+L21", "layers": [19, 21]},
            {"name": "L19-L24", "layers": [19, 21, 23, 24]},
        ],
    },
}

# Extended test pairs for larger sample size
TEST_PAIRS = [
    # Color attributes (20)
    ("apple", "red", "blue"), ("banana", "yellow", "purple"), ("snow", "white", "black"),
    ("sky", "blue", "green"), ("cherry", "red", "blue"), ("leaf", "green", "red"),
    ("rose", "red", "blue"), ("gold", "yellow", "purple"), ("coal", "black", "white"),
    ("silver", "white", "black"), ("milk", "white", "black"), ("honey", "yellow", "blue"),
    ("ruby", "red", "green"), ("emerald", "green", "red"), ("sapphire", "blue", "red"),
    ("moon", "white", "black"), ("flame", "orange", "blue"), ("forest", "green", "white"),
    ("ocean", "blue", "yellow"), ("sun", "yellow", "purple"),
    # Temperature attributes (8)
    ("fire", "hot", "cold"), ("desert", "hot", "cold"), ("lava", "hot", "cold"),
    ("ice", "cold", "hot"), ("snow", "cold", "hot"), ("volcano", "hot", "cold"),
    ("furnace", "hot", "cold"), ("glacier", "cold", "hot"),
    # Wet/Dry attributes (8)
    ("rain", "wet", "dry"), ("ocean", "wet", "dry"), ("river", "wet", "dry"),
    ("sand", "dry", "wet"), ("dust", "dry", "wet"), ("bone", "dry", "wet"),
    ("swamp", "wet", "dry"), ("desert", "dry", "wet"),
    # Texture attributes (6)
    ("silk", "smooth", "rough"), ("sandpaper", "rough", "smooth"),
    ("glass", "smooth", "rough"), ("rock", "rough", "smooth"),
    ("velvet", "soft", "hard"), ("diamond", "hard", "soft"),
]

CORRUPTED_BASELINE = "The item"


def load_model_bf16(model_name):
    """BF16 + device_map=auto + flash attention"""
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
    except Exception:
        return None


def get_mlp_weights(layer, model_name=None, model=None, layer_idx=None):
    """Get MLP weight matrices, with safetensors fallback for meta device weights."""
    mlp = layer.mlp
    W_gate = W_up = W_down = None
    d_ff = 0

    if hasattr(mlp, 'gate_up_proj'):
        w = safe_weight_to_numpy(mlp.gate_up_proj.weight)
        if w is not None:
            d_ff = w.shape[0] // 2
            W_gate, W_up = w[:d_ff], w[d_ff:]
    elif hasattr(mlp, 'gate_proj'):
        W_gate = safe_weight_to_numpy(mlp.gate_proj.weight)
        W_up = safe_weight_to_numpy(mlp.up_proj.weight)
        if W_gate is not None:
            d_ff = W_gate.shape[0]
        elif W_up is not None:
            d_ff = W_up.shape[0]
    elif hasattr(mlp, 'up_proj'):
        W_up = safe_weight_to_numpy(mlp.up_proj.weight)
        if W_up is not None:
            d_ff = W_up.shape[0]

    if hasattr(mlp, 'down_proj'):
        W_down = safe_weight_to_numpy(mlp.down_proj.weight)

    # Fallback: load from safetensors if weights are on meta device
    if (W_down is None or W_gate is None) and model_name is not None and layer_idx is not None:
        import glob
        from safetensors import safe_open
        for sf_file in glob.glob(os.path.join(MODEL_CONFIGS[model_name]["path"], '*.safetensors')):
            try:
                with safe_open(sf_file, framework='pt', device='cpu') as sf:
                    dk = f"model.layers.{layer_idx}.mlp.down_proj.weight"
                    if dk in sf.keys() and W_down is None:
                        W_down = sf.get_tensor(dk).float().numpy()
                    guk = f"model.layers.{layer_idx}.mlp.gate_up_proj.weight"
                    if guk in sf.keys() and W_gate is None:
                        w = sf.get_tensor(guk).float().numpy()
                        d_ff = w.shape[0] // 2
                        W_gate, W_up = w[:d_ff], w[d_ff:]
                    gk = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
                    if gk in sf.keys() and W_gate is None:
                        W_gate = sf.get_tensor(gk).float().numpy()
                        d_ff = W_gate.shape[0]
                    uk = f"model.layers.{layer_idx}.mlp.up_proj.weight"
                    if uk in sf.keys() and W_up is None:
                        W_up = sf.get_tensor(uk).float().numpy()
                        if d_ff == 0:
                            d_ff = W_up.shape[0]
                    if W_down is not None and (W_gate is not None or W_up is not None):
                        break
            except Exception:
                continue

    return W_gate, W_up, W_down, d_ff


def silu_np(x):
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -20, 20))))


# ===== Unified Notation =====
def compute_effect(delta_gap, base_gap, direction):
    """
    Unified effect metric.
    
    For C2R (clean→corrupt): effect = -Δgap / |base_gap|
      Positive effect = binding is damaged (expected for C2R)
    For R2C (corrupt→clean): effect = +Δgap / |base_gap|
      Positive effect = binding is rescued (expected for R2C)
    
    base_gap = clean_gap - corrupt_gap (expected > 0 for binding pairs)
    """
    abs_base = abs(base_gap)
    if abs_base < 1e-10:
        return 0.0
    if direction == "c2r":
        return -delta_gap / abs_base
    else:  # r2c
        return delta_gap / abs_base


def capture_mlp_internals(model, tokenizer, device, prompt, target_layers):
    """Capture gate and up activations at target layers."""
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
                    captured[f"gate_{idx}"] = v[:d]
                    captured[f"up_{idx}"] = v[d:]
                return hook
            hooks.append(layer.mlp.gate_up_proj.register_forward_hook(make_glm4_hook(li)))
        if hasattr(layer.mlp, 'up_proj'):
            hooks.append(layer.mlp.up_proj.register_forward_hook(make_hook(f"up_{li}")))

    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=False)
    logits = out.logits[0, -1].float().cpu().numpy()
    for h in hooks:
        h.remove()
    return captured, logits


def capture_down_proj_inputs(model, tokenizer, device, prompt, target_layers):
    """Capture down_proj input activations (post-gate*up)."""
    layers = get_layers(model)
    captured = {}

    def make_pre_hook(key):
        def pre_hook(module, args):
            inp = args[0]
            captured[key] = inp[0].detach().cpu().float().numpy()
        return pre_hook

    hooks = []
    for li in target_layers:
        hooks.append(layers[li].mlp.down_proj.register_forward_pre_hook(
            make_pre_hook(f"din_{li}")))

    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=False)
    logits = out.logits[0, -1].float().cpu().numpy()
    for h in hooks:
        h.remove()
    acts = {li: captured[f"din_{li}"] for li in target_layers if f"din_{li}" in captured}
    return acts, logits


def run_model_with_block_patch(model, tokenizer, device, prompt, target_layers,
                               channels_to_patch_by_layer, replacement_acts):
    """Patch specified channels across multiple layers simultaneously."""
    layers = get_layers(model)
    hooks = []

    for li in target_layers:
        if li >= len(layers):
            continue
        ch_set = channels_to_patch_by_layer.get(li, set())
        if not ch_set:
            continue
        ch_list = sorted(ch_set)
        rep_np = replacement_acts.get(li)
        if rep_np is None:
            continue
        max_ch = max(ch_list)

        def make_patch_pre_hook(ch_indices_list, rep_numpy, max_ch_val):
            def pre_hook(module, args):
                inp = args[0]
                if inp.dim() == 3 and inp.shape[-1] > max_ch_val:
                    modified = inp.clone()
                    rep_t = torch.tensor(rep_numpy, dtype=modified.dtype, device=modified.device)
                    ch_t = torch.tensor(ch_indices_list, dtype=torch.long, device=modified.device)
                    seq_len = min(modified.shape[1], rep_t.shape[0])
                    modified[0, :seq_len, ch_t] = rep_t[:seq_len, ch_t]
                    return (modified,) + args[1:]
                return args
            return pre_hook

        hooks.append(layers[li].mlp.down_proj.register_forward_pre_hook(
            make_patch_pre_hook(ch_list, rep_np, max_ch)))

    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=False)
    logits = out.logits[0, -1].float().cpu().numpy()
    for h in hooks:
        h.remove()
    return logits


def identify_channels_per_layer(model, tokenizer, device, model_name, W_U,
                                binding_layers, ref_pairs, layers_obj, mlp_weights):
    """Identify Top 1% cproj, dact channels per layer."""
    channel_counts_cproj = defaultdict(lambda: defaultdict(int))
    channel_counts_dact = defaultdict(lambda: defaultdict(int))

    for pidx, (obj, target, competitor) in enumerate(ref_pairs):
        tid_t = get_token_id(tokenizer, target)
        tid_c = get_token_id(tokenizer, competitor)
        if tid_t is None or tid_c is None:
            continue
        direction = W_U[tid_t] - W_U[tid_c]
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-10:
            continue
        direction_normed = direction / dir_norm

        clean_caps, _ = capture_mlp_internals(model, tokenizer, device, f"The {obj}", binding_layers)
        corrupt_caps, _ = capture_mlp_internals(model, tokenizer, device, CORRUPTED_BASELINE, binding_layers)

        for li in binding_layers:
            mw = mlp_weights[li]
            W_down = mw["W_down"]
            d_ff = mw["d_ff"]
            if W_down is None:
                continue
            gk = f"gate_{li}"
            uk = f"up_{li}"
            if gk not in clean_caps or gk not in corrupt_caps:
                continue
            cg = clean_caps[gk][:d_ff]
            crg = corrupt_caps[gk][:d_ff]
            cu = clean_caps.get(uk, np.ones(d_ff))[:d_ff]
            cru = corrupt_caps.get(uk, np.ones(d_ff))[:d_ff]
            min_d = min(d_ff, W_down.shape[1], cg.shape[0])
            Wd = W_down[:, :min_d]
            gsc = silu_np(cg[:min_d])
            gsr = silu_np(crg[:min_d])
            uc = cu[:min_d]
            ur = cru[:min_d]
            dact = gsc * uc - gsr * ur
            channel_proj = Wd.T @ direction_normed
            n_top1 = max(1, min_d // 100)

            for ch in np.argsort(np.abs(channel_proj))[-n_top1:]:
                channel_counts_cproj[li][int(ch)] += 1
            for ch in np.argsort(np.abs(dact))[-n_top1:]:
                channel_counts_dact[li][int(ch)] += 1

        del clean_caps, corrupt_caps
        gc.collect()
        torch.cuda.empty_cache()

    n_ref = len(ref_pairs)
    top1_cproj = {}
    top1_dact = {}

    for li in binding_layers:
        d_ff = mlp_weights[li]["d_ff"]
        n_top1 = max(1, d_ff // 100)

        top1_cproj[li] = set(
            ch for ch, cnt in channel_counts_cproj[li].items() if cnt >= n_ref * 0.3)
        if not top1_cproj[li]:
            sorted_ch = sorted(channel_counts_cproj[li].items(), key=lambda x: -x[1])
            top1_cproj[li] = set(ch for ch, _ in sorted_ch[:n_top1])

        top1_dact[li] = set(
            ch for ch, cnt in channel_counts_dact[li].items() if cnt >= n_ref * 0.3)
        if not top1_dact[li]:
            sorted_ch = sorted(channel_counts_dact[li].items(), key=lambda x: -x[1])
            top1_dact[li] = set(ch for ch, _ in sorted_ch[:n_top1])

        log(f"  L{li}: cproj={len(top1_cproj[li])}ch, dact={len(top1_dact[li])}ch")

    return top1_cproj, top1_dact


def run_experiment(model_name):
    log(f"Phase 357: Block Patch Circuit Verification ({model_name})")
    log("=" * 70)
    t0 = time.time()
    cfg = MODEL_CONFIGS[model_name]
    binding_layers = cfg["binding_layers"]
    block_configs = cfg["block_configs"]

    # Load model
    model, tokenizer, device = load_model_bf16(model_name)
    W_U = get_W_U(model, model_name)
    layers_obj = get_layers(model)

    # Load MLP weights
    mlp_weights = {}
    for li in binding_layers:
        layer_idx = li
        W_gate, W_up, W_down, d_ff = get_mlp_weights(
            layers_obj[li], model_name, model, layer_idx)
        mlp_weights[li] = {"W_gate": W_gate, "W_up": W_up, "W_down": W_down, "d_ff": d_ff}
    log(f"  MLP weights loaded for {len(binding_layers)} layers")

    # Channel identification (use 20 ref pairs)
    log(f"\n  Part 1: Channel identification from 20 reference pairs...")
    ref_pairs = TEST_PAIRS[:20]
    top1_cproj, top1_dact = identify_channels_per_layer(
        model, tokenizer, device, model_name, W_U, binding_layers, ref_pairs, layers_obj, mlp_weights)

    # ================================================================
    # Part 2: Single-Layer Patch (re-confirm with unified notation)
    # ================================================================
    n_test = len(TEST_PAIRS)
    log(f"\n  Part 2: Single-layer patch ({n_test} pairs)...")

    single_layer_results = {li: {"cproj_c2r": [], "cproj_r2c": [],
                                 "dact_c2r": [], "dact_r2c": []}
                            for li in binding_layers}

    # Store per-pair data for super-additivity analysis
    per_pair_single = {}

    for pidx, (obj, target, competitor) in enumerate(TEST_PAIRS):
        tid_t = get_token_id(tokenizer, target)
        tid_c = get_token_id(tokenizer, competitor)
        if tid_t is None or tid_c is None:
            continue

        clean_prompt = f"The {obj}"
        clean_acts, clean_logits = capture_down_proj_inputs(
            model, tokenizer, device, clean_prompt, binding_layers)
        corrupt_acts, corrupt_logits = capture_down_proj_inputs(
            model, tokenizer, device, CORRUPTED_BASELINE, binding_layers)

        clean_target = float(clean_logits[tid_t])
        clean_compet = float(clean_logits[tid_c])
        corrupt_target = float(corrupt_logits[tid_t])
        corrupt_compet = float(corrupt_logits[tid_c])

        clean_gap = clean_target - clean_compet
        corrupt_gap = corrupt_target - corrupt_compet
        base_gap = clean_gap - corrupt_gap  # Expected > 0 for binding

        if abs(base_gap) < 1e-10:
            continue

        pair_single = {}
        for li in binding_layers:
            pair_single[li] = {}
            for gname, channels in [("cproj", top1_cproj), ("dact", top1_dact)]:
                ch_set = channels.get(li, set())
                if not ch_set:
                    continue

                # C2R
                patch_dict = {li: ch_set}
                c2r_logits = run_model_with_block_patch(
                    model, tokenizer, device, clean_prompt, [li], patch_dict, corrupt_acts)
                c2r_t = float(c2r_logits[tid_t])
                c2r_c = float(c2r_logits[tid_c])
                c2r_gap = c2r_t - c2r_c
                delta_gap_c2r = c2r_gap - clean_gap
                effect_c2r = compute_effect(delta_gap_c2r, base_gap, "c2r")

                # R2C
                r2c_logits = run_model_with_block_patch(
                    model, tokenizer, device, CORRUPTED_BASELINE, [li], patch_dict, clean_acts)
                r2c_t = float(r2c_logits[tid_t])
                r2c_c = float(r2c_logits[tid_c])
                r2c_gap = r2c_t - r2c_c
                delta_gap_r2c = r2c_gap - corrupt_gap
                effect_r2c = compute_effect(delta_gap_r2c, base_gap, "r2c")

                single_layer_results[li][f"{gname}_c2r"].append({
                    "delta_t": c2r_t - clean_target,
                    "delta_c": c2r_c - clean_compet,
                    "delta_gap": delta_gap_c2r,
                    "effect": effect_c2r,
                })
                single_layer_results[li][f"{gname}_r2c"].append({
                    "delta_t": r2c_t - corrupt_target,
                    "delta_c": r2c_c - corrupt_compet,
                    "delta_gap": delta_gap_r2c,
                    "effect": effect_r2c,
                })

                pair_single[li][gname] = {
                    "effect_c2r": effect_c2r,
                    "effect_r2c": effect_r2c,
                    "delta_gap_c2r": delta_gap_c2r,
                    "delta_gap_r2c": delta_gap_r2c,
                }

                gc.collect()
                torch.cuda.empty_cache()

        per_pair_single[pidx] = {
            "base_gap": base_gap,
            "clean_gap": clean_gap,
            "corrupt_gap": corrupt_gap,
            "layers": pair_single,
        }

        if (pidx + 1) % 5 == 0:
            elapsed = time.time() - t0
            gpu_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            log(f"  [{pidx+1}/{n_test}] single-layer done, elapsed={elapsed:.0f}s, GPU={gpu_gb:.1f}GB")

        gc.collect()
        torch.cuda.empty_cache()

    # ================================================================
    # Part 3: Block Patch
    # ================================================================
    log(f"\n  Part 3: Block patch testing ({n_test} pairs)...")

    block_results = {}
    for bcfg in block_configs:
        bname = bcfg["name"]
        blayers = bcfg["layers"]
        block_results[bname] = {
            "cproj_c2r": [], "cproj_r2c": [],
            "dact_c2r": [], "dact_r2c": [],
            "combined_c2r": [], "combined_r2c": [],
        }
        log(f"  Block: {bname} (layers={blayers})")

    for pidx, (obj, target, competitor) in enumerate(TEST_PAIRS):
        tid_t = get_token_id(tokenizer, target)
        tid_c = get_token_id(tokenizer, competitor)
        if tid_t is None or tid_c is None:
            continue

        clean_prompt = f"The {obj}"
        clean_acts, clean_logits = capture_down_proj_inputs(
            model, tokenizer, device, clean_prompt, binding_layers)
        corrupt_acts, corrupt_logits = capture_down_proj_inputs(
            model, tokenizer, device, CORRUPTED_BASELINE, binding_layers)

        clean_target = float(clean_logits[tid_t])
        clean_compet = float(clean_logits[tid_c])
        corrupt_target = float(corrupt_logits[tid_t])
        corrupt_compet = float(corrupt_logits[tid_c])
        clean_gap = clean_target - clean_compet
        corrupt_gap = corrupt_target - corrupt_compet
        base_gap = clean_gap - corrupt_gap

        if abs(base_gap) < 1e-10:
            continue

        for bcfg in block_configs:
            bname = bcfg["name"]
            blayers = bcfg["layers"]

            for gname, channels in [("cproj", top1_cproj), ("dact", top1_dact)]:
                # Collect channels for all layers in this block
                patch_dict = {}
                for li in blayers:
                    ch_set = channels.get(li, set())
                    if ch_set:
                        patch_dict[li] = ch_set
                if not patch_dict:
                    continue

                # C2R: patch block with corrupt acts
                c2r_logits = run_model_with_block_patch(
                    model, tokenizer, device, clean_prompt, blayers, patch_dict, corrupt_acts)
                c2r_t = float(c2r_logits[tid_t])
                c2r_c = float(c2r_logits[tid_c])
                c2r_gap = c2r_t - c2r_c
                delta_gap_c2r = c2r_gap - clean_gap
                effect_c2r = compute_effect(delta_gap_c2r, base_gap, "c2r")

                # R2C: patch block with clean acts
                r2c_logits = run_model_with_block_patch(
                    model, tokenizer, device, CORRUPTED_BASELINE, blayers, patch_dict, clean_acts)
                r2c_t = float(r2c_logits[tid_t])
                r2c_c = float(r2c_logits[tid_c])
                r2c_gap = r2c_t - r2c_c
                delta_gap_r2c = r2c_gap - corrupt_gap
                effect_r2c = compute_effect(delta_gap_r2c, base_gap, "r2c")

                block_results[bname][f"{gname}_c2r"].append({
                    "delta_t": c2r_t - clean_target,
                    "delta_c": c2r_c - clean_compet,
                    "delta_gap": delta_gap_c2r,
                    "effect": effect_c2r,
                })
                block_results[bname][f"{gname}_r2c"].append({
                    "delta_t": r2c_t - corrupt_target,
                    "delta_c": r2c_c - corrupt_compet,
                    "delta_gap": delta_gap_r2c,
                    "effect": effect_r2c,
                })

                gc.collect()
                torch.cuda.empty_cache()

            # Combined: cproj + dact channels together
            combined_patch = {}
            for li in blayers:
                all_ch = top1_cproj.get(li, set()) | top1_dact.get(li, set())
                if all_ch:
                    combined_patch[li] = all_ch
            if combined_patch:
                # C2R combined
                c2r_logits = run_model_with_block_patch(
                    model, tokenizer, device, clean_prompt, blayers, combined_patch, corrupt_acts)
                c2r_t = float(c2r_logits[tid_t])
                c2r_c = float(c2r_logits[tid_c])
                c2r_gap = c2r_t - c2r_c
                delta_gap_c2r = c2r_gap - clean_gap
                effect_c2r = compute_effect(delta_gap_c2r, base_gap, "c2r")

                # R2C combined
                r2c_logits = run_model_with_block_patch(
                    model, tokenizer, device, CORRUPTED_BASELINE, blayers, combined_patch, clean_acts)
                r2c_t = float(r2c_logits[tid_t])
                r2c_c = float(r2c_logits[tid_c])
                r2c_gap = r2c_t - r2c_c
                delta_gap_r2c = r2c_gap - corrupt_gap
                effect_r2c = compute_effect(delta_gap_r2c, base_gap, "r2c")

                block_results[bname]["combined_c2r"].append({
                    "delta_t": c2r_t - clean_target,
                    "delta_c": c2r_c - clean_compet,
                    "delta_gap": delta_gap_c2r,
                    "effect": effect_c2r,
                })
                block_results[bname]["combined_r2c"].append({
                    "delta_t": r2c_t - corrupt_target,
                    "delta_c": r2c_c - corrupt_compet,
                    "delta_gap": delta_gap_r2c,
                    "effect": effect_r2c,
                })

                gc.collect()
                torch.cuda.empty_cache()

        if (pidx + 1) % 5 == 0:
            elapsed = time.time() - t0
            gpu_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            log(f"  [{pidx+1}/{n_test}] block done, elapsed={elapsed:.0f}s, GPU={gpu_gb:.1f}GB")

        gc.collect()
        torch.cuda.empty_cache()

    # ================================================================
    # Part 4: Summary & Super-Additivity Analysis
    # ================================================================
    log(f"\n  {'='*70}")
    log(f"  Phase 357 Summary: {model_name}")
    log(f"  {'='*70}")

    # --- Single-layer results ---
    log(f"\n  --- Single-Layer Results (Unified Notation) ---")
    log(f"  {'Layer':>6} {'cproj_C2R_eff':>14} {'cproj_R2C_eff':>14} "
        f"{'dact_C2R_eff':>14} {'dact_R2C_eff':>14} {'n':>4}")
    log(f"  {'-'*70}")

    single_summary = {}
    for li in binding_layers:
        lr = single_layer_results[li]
        row = {}
        for key in ["cproj_c2r", "cproj_r2c", "dact_c2r", "dact_r2c"]:
            vals = lr[key]
            n = len(vals)
            if n == 0:
                row[key] = {"n": 0, "mean_effect": 0, "se_effect": 0,
                            "mean_delta_gap": 0}
                continue
            effects = [v["effect"] for v in vals]
            dgaps = [v["delta_gap"] for v in vals]
            row[key] = {
                "n": n,
                "mean_effect": float(np.mean(effects)),
                "se_effect": float(np.std(effects) / np.sqrt(n)),
                "mean_delta_gap": float(np.mean(dgaps)),
            }
        single_summary[li] = row
        cp_c2r = row["cproj_c2r"]
        cp_r2c = row["cproj_r2c"]
        da_c2r = row["dact_c2r"]
        da_r2c = row["dact_r2c"]
        log(f"  L{li:>4} {cp_c2r['mean_effect']:>+14.4f} {cp_r2c['mean_effect']:>+14.4f} "
            f"{da_c2r['mean_effect']:>+14.4f} {da_r2c['mean_effect']:>+14.4f} {cp_c2r['n']:>4}")

    # --- Block results ---
    log(f"\n  --- Block Patch Results (Unified Notation) ---")
    log(f"  {'Block':>14} {'cproj_C2R':>11} {'cproj_R2C':>11} "
        f"{'dact_C2R':>11} {'dact_R2C':>11} "
        f"{'comb_C2R':>11} {'comb_R2C':>11} {'n':>4}")
    log(f"  {'-'*100}")

    block_summary = {}
    for bcfg in block_configs:
        bname = bcfg["name"]
        blayers = bcfg["layers"]
        br = block_results[bname]
        row = {}
        for key in ["cproj_c2r", "cproj_r2c", "dact_c2r", "dact_r2c",
                     "combined_c2r", "combined_r2c"]:
            vals = br[key]
            n = len(vals)
            if n == 0:
                row[key] = {"n": 0, "mean_effect": 0, "se_effect": 0}
                continue
            effects = [v["effect"] for v in vals]
            row[key] = {
                "n": n,
                "mean_effect": float(np.mean(effects)),
                "se_effect": float(np.std(effects) / np.sqrt(n)),
            }
        block_summary[bname] = row

        cp_c2r = row["cproj_c2r"]
        cp_r2c = row["cproj_r2c"]
        da_c2r = row["dact_c2r"]
        da_r2c = row["dact_r2c"]
        cb_c2r = row["combined_c2r"]
        cb_r2c = row["combined_r2c"]
        log(f"  {bname:>14} {cp_c2r['mean_effect']:>+11.4f} {cp_r2c['mean_effect']:>+11.4f} "
            f"{da_c2r['mean_effect']:>+11.4f} {da_r2c['mean_effect']:>+11.4f} "
            f"{cb_c2r['mean_effect']:>+11.4f} {cb_r2c['mean_effect']:>+11.4f} {cp_c2r['n']:>4}")

    # --- Super-Additivity Analysis ---
    log(f"\n  --- Super-Additivity: Block vs Sum(Single) ---")
    log(f"  {'Block':>14} {'Path':>6} {'Dir':>5} "
        f"{'Block_eff':>11} {'Sum(Single)':>11} {'Ratio':>8} {'Super?':>7}")
    log(f"  {'-'*70}")

    superadd_results = {}
    for bcfg in block_configs:
        bname = bcfg["name"]
        blayers = bcfg["layers"]
        superadd_results[bname] = {}

        for gname in ["cproj", "dact"]:
            for direction in ["c2r", "r2c"]:
                key = f"{gname}_{direction}"

                # Block effect
                block_eff = block_summary[bname].get(key, {}).get("mean_effect", 0)

                # Sum of single-layer effects
                sum_single = 0
                for li in blayers:
                    if li in single_summary:
                        sum_single += single_summary[li].get(key, {}).get("mean_effect", 0)

                ratio = block_eff / sum_single if abs(sum_single) > 1e-10 else float('inf')
                is_super = "YES" if ratio > 1.3 else ("sub" if ratio < 0.7 else "additive")

                superadd_results[bname][key] = {
                    "block_eff": block_eff,
                    "sum_single": sum_single,
                    "ratio": ratio,
                    "is_superadditive": ratio > 1.3,
                }

                log(f"  {bname:>14} {gname:>6} {direction:>5} "
                    f"{block_eff:>+11.4f} {sum_single:>+11.4f} {ratio:>8.2f} {is_super:>7}")

    # --- Combined vs Separate Analysis ---
    log(f"\n  --- Combined (cproj+dact) vs Separate ---")
    log(f"  {'Block':>14} {'Dir':>5} "
        f"{'cproj':>11} {'dact':>11} {'c+d_sum':>11} {'combined':>11} {'Ratio':>8}")
    log(f"  {'-'*70}")

    coupling_results = {}
    for bcfg in block_configs:
        bname = bcfg["name"]
        coupling_results[bname] = {}

        for direction in ["c2r", "r2c"]:
            cp_eff = block_summary[bname].get(f"cproj_{direction}", {}).get("mean_effect", 0)
            da_eff = block_summary[bname].get(f"dact_{direction}", {}).get("mean_effect", 0)
            cb_eff = block_summary[bname].get(f"combined_{direction}", {}).get("mean_effect", 0)
            cd_sum = cp_eff + da_eff
            ratio = cb_eff / cd_sum if abs(cd_sum) > 1e-10 else float('inf')

            coupling_results[bname][direction] = {
                "cproj_eff": cp_eff, "dact_eff": da_eff,
                "cplusd_sum": cd_sum, "combined_eff": cb_eff,
                "ratio": ratio,
            }

            log(f"  {bname:>14} {direction:>5} "
                f"{cp_eff:>+11.4f} {da_eff:>+11.4f} {cd_sum:>+11.4f} {cb_eff:>+11.4f} {ratio:>8.2f}")

    # ================================================================
    # Save
    # ================================================================
    output = {
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "phase": "357",
        "unified_notation": {
            "C2R_effect": "-Δgap / |base_gap| (positive = binding damaged)",
            "R2C_effect": "+Δgap / |base_gap| (positive = binding rescued)",
            "base_gap": "clean_gap - corrupt_gap",
        },
        "block_configs": [{"name": b["name"], "layers": b["layers"]} for b in block_configs],
        "n_pairs": n_test,
        "single_summary": {str(k): v for k, v in single_summary.items()},
        "block_summary": block_summary,
        "superadditivity": superadd_results,
        "coupling": coupling_results,
    }

    os.makedirs("results/phase357_block_patch_circuit", exist_ok=True)
    out_path = f"results/phase357_block_patch_circuit/{model_name}_phase357.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=str, ensure_ascii=False)
    log(f"\n  Saved to {out_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    log(f"Phase 357 complete for {model_name} in {time.time()-t0:.0f}s")
    return output


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_experiment(model_name)
