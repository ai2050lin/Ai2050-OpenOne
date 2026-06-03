"""
Phase 359+360: dact Context Compatibility + cproj-dact Coupling Test
====================================================================

Core question: Why is dact R2C universally negative?

Test conditions per layer:
  1. dact_top1:  Patch top 1% dact channels at down_proj input
  2. cproj_top1: Patch top 1% cproj channels at down_proj input
  3. comb_top1:  Patch both dact and cproj top 1% channels
  4. full_mlp:   Replace entire MLP output (all d_model dims)
  5. full_resid: Replace entire residual stream (post-layer)
  6. dact_top5:  Patch top 5% dact channels (broader set)

Key comparisons:
  - dact_top1 R2C vs full_mlp R2C: if full_mlp positive but dact negative,
    dact channels alone are insufficient; other channels provide context
  - full_mlp R2C vs full_resid R2C: if full_resid positive but full_mlp negative,
    the attention contribution is also needed
  - dact_top1 vs dact_top5: if top5 helps, the problem is channel selection

Coupling analysis (Phase 360):
  - marginal_dact_given_cproj = comb_top1 - cproj_top1
  - marginal_cproj_given_dact = comb_top1 - dact_top1
  - Compare with standalone effects for interaction detection

Unified notation:
  C2R: effect = -Δgap / |base_gap| (positive = binding damaged)
  R2C: effect = +Δgap / |base_gap| (positive = binding rescued)
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
        "test_layers": [23, 27],  # L23=dact C2R negative, L27=dact C2R positive
    },
    "glm4": {
        "path": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
        "n_layers": 40, "d_model": 4096, "d_ff": 13696,
        "test_layers": [36, 38],  # L36=dact R2C positive, L38=dact R2C negative
    },
    "deepseek7b": {
        "path": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "n_layers": 28, "d_model": 3584, "d_ff": 18944,
        "test_layers": [19, 21],  # L19/L21=dact R2C very negative
    },
}

# Reference pairs for channel identification
REF_PAIRS = [
    ("apple", "red", "blue"), ("banana", "yellow", "purple"), ("snow", "white", "black"),
    ("sky", "blue", "green"), ("cherry", "red", "blue"), ("leaf", "green", "red"),
    ("rose", "red", "blue"), ("gold", "yellow", "purple"), ("coal", "black", "white"),
    ("silver", "white", "black"), ("milk", "white", "black"), ("honey", "yellow", "blue"),
    ("ruby", "red", "green"), ("emerald", "green", "red"), ("sapphire", "blue", "red"),
    ("moon", "white", "black"), ("flame", "orange", "blue"), ("forest", "green", "white"),
    ("ocean", "blue", "yellow"), ("sun", "yellow", "purple"),
]

# Full test pairs (42)
TEST_PAIRS = REF_PAIRS + [
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
    # Fallback to safetensors
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


def compute_effect(delta_gap, base_gap, direction):
    abs_base = abs(base_gap)
    if abs_base < 1e-10:
        return 0.0
    if direction == "c2r":
        return -delta_gap / abs_base
    else:
        return delta_gap / abs_base


# ===== Capture Functions =====

def capture_all_acts(model, tokenizer, device, prompt, target_layers):
    """Capture down_proj input, MLP output, and residual output at target layers."""
    layers = get_layers(model)
    captured = {}

    # Forward hook: (module, input, output) -> captures output
    def make_fwd_hook(key):
        def hook(module, input, output):
            val = output[0] if isinstance(output, tuple) else output
            captured[key] = val[0, -1, :].detach().cpu().float().numpy()
        return hook

    # Pre-hook: (module, args) -> captures full input to down_proj [seq_len, d_ff]
    def make_pre_hook(key):
        def pre_hook(module, args):
            inp = args[0]
            captured[key] = inp[0].detach().cpu().float().numpy()  # [seq_len, d_ff]
        return pre_hook

    hooks = []
    for li in target_layers:
        layer = layers[li]
        # down_proj input (d_ff space) — for channel patching
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(
            make_pre_hook(f"din_{li}")))
        # MLP output (d_model space) — for full MLP replacement
        hooks.append(layer.mlp.register_forward_hook(
            make_fwd_hook(f"mlpout_{li}")))
        # Layer output (d_model space) — for full residual replacement
        hooks.append(layer.register_forward_hook(
            make_fwd_hook(f"residout_{li}")))
        # Also capture gate/up for channel identification
        if hasattr(layer.mlp, 'gate_proj'):
            hooks.append(layer.mlp.gate_proj.register_forward_hook(
                make_fwd_hook(f"gate_{li}")))
            hooks.append(layer.mlp.up_proj.register_forward_hook(
                make_fwd_hook(f"up_{li}")))
        elif hasattr(layer.mlp, 'gate_up_proj'):
            def make_glm4_gate_hook(idx):
                def hook(module, input, output):
                    val = output[0] if isinstance(output, tuple) else output
                    v = val[0, -1, :].detach().cpu().float().numpy()
                    d = v.shape[0] // 2
                    captured[f"gate_{idx}"] = v[:d]
                    captured[f"up_{idx}"] = v[d:]
                return hook
            hooks.append(layer.mlp.gate_up_proj.register_forward_hook(
                make_glm4_gate_hook(li)))

    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=False)
    logits = out.logits[0, -1].float().cpu().numpy()

    for h in hooks:
        h.remove()

    return captured, logits


# ===== Patch Functions =====

def run_with_channel_patch(model, tokenizer, device, prompt, target_layer,
                           channel_set, replacement_din):
    """Patch specific channels at down_proj input."""
    layers = get_layers(model)
    ch_list = sorted(channel_set)
    if not ch_list:
        inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=False)
        return out.logits[0, -1].float().cpu().numpy()

    max_ch = max(ch_list)
    rep_np = replacement_din

    def make_patch_pre_hook(ch_indices, rep, max_ch_val):
        def pre_hook(module, args):
            inp = args[0]
            if inp.dim() == 3 and inp.shape[-1] > max_ch_val:
                modified = inp.clone()
                rep_t = torch.tensor(rep, dtype=modified.dtype, device=modified.device)
                ch_t = torch.tensor(ch_indices, dtype=torch.long, device=modified.device)
                seq_len = min(modified.shape[1], rep_t.shape[0])
                modified[0, :seq_len, ch_t] = rep_t[:seq_len, ch_t]
                return (modified,) + args[1:]
            return args
        return pre_hook

    hook = layers[target_layer].mlp.down_proj.register_forward_pre_hook(
        make_patch_pre_hook(ch_list, rep_np, max_ch))

    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=False)
    logits = out.logits[0, -1].float().cpu().numpy()
    hook.remove()
    return logits


def run_with_mlp_replace(model, tokenizer, device, prompt, target_layer,
                         replacement_mlp_out):
    """Replace entire MLP output at target_layer with replacement (d_model vector, last token)."""
    layers = get_layers(model)
    rep = replacement_mlp_out  # numpy array [d_model]

    def make_replace_hook(replacement):
        def hook(module, input, output):
            val = output[0] if isinstance(output, tuple) else output
            # Replace last token position only
            modified = val.clone()
            rep_t = torch.tensor(replacement, dtype=modified.dtype, device=modified.device)
            modified[0, -1, :] = rep_t
            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified
        return hook

    hook_handle = layers[target_layer].mlp.register_forward_hook(
        make_replace_hook(rep))

    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=False)
    logits = out.logits[0, -1].float().cpu().numpy()
    hook_handle.remove()
    return logits


def run_with_resid_replace(model, tokenizer, device, prompt, target_layer,
                           replacement_resid):
    """Replace entire residual stream (post-layer output) at target_layer."""
    layers = get_layers(model)
    rep = replacement_resid  # numpy array [d_model]

    def make_replace_hook(replacement):
        def hook(module, input, output):
            val = output[0] if isinstance(output, tuple) else output
            modified = val.clone()
            rep_t = torch.tensor(replacement, dtype=modified.dtype, device=modified.device)
            modified[0, -1, :] = rep_t
            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified
        return hook

    hook_handle = layers[target_layer].register_forward_hook(
        make_replace_hook(rep))

    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=False)
    logits = out.logits[0, -1].float().cpu().numpy()
    hook_handle.remove()
    return logits


# ===== Channel Identification =====

def identify_channels(model, tokenizer, device, model_name, W_U,
                      target_layers, ref_pairs, layers_obj, mlp_weights):
    """Identify top 1% and top 5% cproj/dact channels per layer."""
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

        clean_caps, _ = capture_all_acts(model, tokenizer, device, f"The {obj}", target_layers)
        corrupt_caps, _ = capture_all_acts(model, tokenizer, device, CORRUPTED_BASELINE, target_layers)

        for li in target_layers:
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
            n_top5 = max(1, min_d // 20)

            for ch in np.argsort(np.abs(channel_proj))[-n_top1:]:
                channel_counts_cproj[li][int(ch)] += 1
            for ch in np.argsort(np.abs(dact))[-n_top1:]:
                channel_counts_dact[li][int(ch)] += 1

        del clean_caps, corrupt_caps
        gc.collect()
        torch.cuda.empty_cache()

        if (pidx + 1) % 5 == 0:
            log(f"  Channel ID: {pidx+1}/{len(ref_pairs)} done")

    n_ref = len(ref_pairs)
    top1_cproj = {}
    top1_dact = {}
    top5_dact = {}

    for li in target_layers:
        d_ff = mlp_weights[li]["d_ff"]
        n_top1 = max(1, d_ff // 100)
        n_top5 = max(1, d_ff // 20)

        # cproj top 1%
        top1_cproj[li] = set(
            ch for ch, cnt in channel_counts_cproj[li].items() if cnt >= n_ref * 0.3)
        if not top1_cproj[li]:
            sorted_ch = sorted(channel_counts_cproj[li].items(), key=lambda x: -x[1])
            top1_cproj[li] = set(ch for ch, _ in sorted_ch[:n_top1])

        # dact top 1%
        top1_dact[li] = set(
            ch for ch, cnt in channel_counts_dact[li].items() if cnt >= n_ref * 0.3)
        if not top1_dact[li]:
            sorted_ch = sorted(channel_counts_dact[li].items(), key=lambda x: -x[1])
            top1_dact[li] = set(ch for ch, _ in sorted_ch[:n_top1])

        # dact top 5% (broader set)
        sorted_ch = sorted(channel_counts_dact[li].items(), key=lambda x: -x[1])
        top5_dact[li] = set(ch for ch, _ in sorted_ch[:n_top5])

        log(f"  L{li}: cproj_top1={len(top1_cproj[li])}ch, "
            f"dact_top1={len(top1_dact[li])}ch, dact_top5={len(top5_dact[li])}ch")

    return top1_cproj, top1_dact, top5_dact


# ===== Main Experiment =====

def run_experiment(model_name):
    log(f"Phase 359+360: dact Context Compatibility ({model_name})")
    log("=" * 70)
    t0 = time.time()
    cfg = MODEL_CONFIGS[model_name]
    target_layers = cfg["test_layers"]

    # Load model
    model, tokenizer, device = load_model_bf16(model_name)
    W_U = get_W_U(model, model_name)
    layers_obj = get_layers(model)

    # Load MLP weights
    mlp_weights = {}
    for li in target_layers:
        W_gate, W_up, W_down, d_ff = get_mlp_weights(
            layers_obj[li], model_name, model, li)
        mlp_weights[li] = {"W_gate": W_gate, "W_up": W_up, "W_down": W_down, "d_ff": d_ff}
    log(f"  MLP weights loaded for layers {target_layers}")

    # Channel identification
    log(f"\n  Part 1: Channel identification from {len(REF_PAIRS)} reference pairs...")
    top1_cproj, top1_dact, top5_dact = identify_channels(
        model, tokenizer, device, model_name, W_U,
        target_layers, REF_PAIRS, layers_obj, mlp_weights)

    # ================================================================
    # Part 2: Main experiment
    # ================================================================
    n_test = len(TEST_PAIRS)
    log(f"\n  Part 2: Main experiment ({n_test} pairs, {len(target_layers)} layers)...")

    # Results storage: per condition per layer
    conditions = ["dact_top1", "cproj_top1", "comb_top1",
                  "full_mlp", "full_resid", "dact_top5"]
    results = {li: {cond: {"c2r": [], "r2c": []} for cond in conditions}
               for li in target_layers}
    # Per-pair data for bootstrap
    per_pair = {}

    for pidx, (obj, target, competitor) in enumerate(TEST_PAIRS):
        tid_t = get_token_id(tokenizer, target)
        tid_c = get_token_id(tokenizer, competitor)
        if tid_t is None or tid_c is None:
            continue

        clean_prompt = f"The {obj}"

        # Capture all activations
        clean_acts, clean_logits = capture_all_acts(
            model, tokenizer, device, clean_prompt, target_layers)
        corrupt_acts, corrupt_logits = capture_all_acts(
            model, tokenizer, device, CORRUPTED_BASELINE, target_layers)

        clean_target = float(clean_logits[tid_t])
        clean_compet = float(clean_logits[tid_c])
        corrupt_target = float(corrupt_logits[tid_t])
        corrupt_compet = float(corrupt_logits[tid_c])
        clean_gap = clean_target - clean_compet
        corrupt_gap = corrupt_target - corrupt_compet
        base_gap = clean_gap - corrupt_gap

        if abs(base_gap) < 1e-10:
            del clean_acts, corrupt_acts
            gc.collect()
            torch.cuda.empty_cache()
            continue

        pair_data = {"base_gap": base_gap, "layers": {}}

        for li in target_layers:
            pair_layer = {}
            d_ff = mlp_weights[li]["d_ff"]

            # Extract captured activations for this layer
            clean_din = clean_acts.get(f"din_{li}")  # [d_ff]
            corrupt_din = corrupt_acts.get(f"din_{li}")
            clean_mlpout = clean_acts.get(f"mlpout_{li}")  # [d_model]
            corrupt_mlpout = corrupt_acts.get(f"mlpout_{li}")
            clean_residout = clean_acts.get(f"residout_{li}")  # [d_model]
            corrupt_residout = corrupt_acts.get(f"residout_{li}")

            if clean_din is None or corrupt_din is None:
                continue

            # Channel sets for this layer
            cproj_ch = top1_cproj.get(li, set())
            dact_ch1 = top1_dact.get(li, set())
            dact_ch5 = top5_dact.get(li, set())
            comb_ch = cproj_ch | dact_ch1

            # --- Condition 1: dact_top1 ---
            for direction, src_prompt, src_acts, ref_gap, ref_target, ref_compet in [
                ("c2r", clean_prompt, corrupt_acts, clean_gap, clean_target, clean_compet),
                ("r2c", CORRUPTED_BASELINE, clean_acts, corrupt_gap, corrupt_target, corrupt_compet)
            ]:
                patch_logits = run_with_channel_patch(
                    model, tokenizer, device, src_prompt, li,
                    dact_ch1, src_acts.get(f"din_{li}"))
                pt = float(patch_logits[tid_t])
                pc = float(patch_logits[tid_c])
                p_gap = pt - pc
                delta_gap = p_gap - ref_gap
                effect = compute_effect(delta_gap, base_gap, direction)
                results[li]["dact_top1"][direction].append({
                    "effect": effect, "delta_gap": delta_gap,
                    "delta_t": pt - ref_target, "delta_c": pc - ref_compet,
                })

            gc.collect()
            torch.cuda.empty_cache()

            # --- Condition 2: cproj_top1 ---
            for direction, src_prompt, src_acts, ref_gap, ref_target, ref_compet in [
                ("c2r", clean_prompt, corrupt_acts, clean_gap, clean_target, clean_compet),
                ("r2c", CORRUPTED_BASELINE, clean_acts, corrupt_gap, corrupt_target, corrupt_compet)
            ]:
                patch_logits = run_with_channel_patch(
                    model, tokenizer, device, src_prompt, li,
                    cproj_ch, src_acts.get(f"din_{li}"))
                pt = float(patch_logits[tid_t])
                pc = float(patch_logits[tid_c])
                p_gap = pt - pc
                delta_gap = p_gap - ref_gap
                effect = compute_effect(delta_gap, base_gap, direction)
                results[li]["cproj_top1"][direction].append({
                    "effect": effect, "delta_gap": delta_gap,
                    "delta_t": pt - ref_target, "delta_c": pc - ref_compet,
                })

            gc.collect()
            torch.cuda.empty_cache()

            # --- Condition 3: comb_top1 ---
            for direction, src_prompt, src_acts, ref_gap, ref_target, ref_compet in [
                ("c2r", clean_prompt, corrupt_acts, clean_gap, clean_target, clean_compet),
                ("r2c", CORRUPTED_BASELINE, clean_acts, corrupt_gap, corrupt_target, corrupt_compet)
            ]:
                patch_logits = run_with_channel_patch(
                    model, tokenizer, device, src_prompt, li,
                    comb_ch, src_acts.get(f"din_{li}"))
                pt = float(patch_logits[tid_t])
                pc = float(patch_logits[tid_c])
                p_gap = pt - pc
                delta_gap = p_gap - ref_gap
                effect = compute_effect(delta_gap, base_gap, direction)
                results[li]["comb_top1"][direction].append({
                    "effect": effect, "delta_gap": delta_gap,
                    "delta_t": pt - ref_target, "delta_c": pc - ref_compet,
                })

            gc.collect()
            torch.cuda.empty_cache()

            # --- Condition 4: full_mlp ---
            if clean_mlpout is not None and corrupt_mlpout is not None:
                for direction, src_prompt, replacement, ref_gap, ref_target, ref_compet in [
                    ("c2r", clean_prompt, corrupt_mlpout, clean_gap, clean_target, clean_compet),
                    ("r2c", CORRUPTED_BASELINE, clean_mlpout, corrupt_gap, corrupt_target, corrupt_compet)
                ]:
                    patch_logits = run_with_mlp_replace(
                        model, tokenizer, device, src_prompt, li, replacement)
                    pt = float(patch_logits[tid_t])
                    pc = float(patch_logits[tid_c])
                    p_gap = pt - pc
                    delta_gap = p_gap - ref_gap
                    effect = compute_effect(delta_gap, base_gap, direction)
                    results[li]["full_mlp"][direction].append({
                        "effect": effect, "delta_gap": delta_gap,
                        "delta_t": pt - ref_target, "delta_c": pc - ref_compet,
                    })

            gc.collect()
            torch.cuda.empty_cache()

            # --- Condition 5: full_resid ---
            if clean_residout is not None and corrupt_residout is not None:
                for direction, src_prompt, replacement, ref_gap, ref_target, ref_compet in [
                    ("c2r", clean_prompt, corrupt_residout, clean_gap, clean_target, clean_compet),
                    ("r2c", CORRUPTED_BASELINE, clean_residout, corrupt_gap, corrupt_target, corrupt_compet)
                ]:
                    patch_logits = run_with_resid_replace(
                        model, tokenizer, device, src_prompt, li, replacement)
                    pt = float(patch_logits[tid_t])
                    pc = float(patch_logits[tid_c])
                    p_gap = pt - pc
                    delta_gap = p_gap - ref_gap
                    effect = compute_effect(delta_gap, base_gap, direction)
                    results[li]["full_resid"][direction].append({
                        "effect": effect, "delta_gap": delta_gap,
                        "delta_t": pt - ref_target, "delta_c": pc - ref_compet,
                    })

            gc.collect()
            torch.cuda.empty_cache()

            # --- Condition 6: dact_top5 ---
            for direction, src_prompt, src_acts, ref_gap, ref_target, ref_compet in [
                ("c2r", clean_prompt, corrupt_acts, clean_gap, clean_target, clean_compet),
                ("r2c", CORRUPTED_BASELINE, clean_acts, corrupt_gap, corrupt_target, corrupt_compet)
            ]:
                patch_logits = run_with_channel_patch(
                    model, tokenizer, device, src_prompt, li,
                    dact_ch5, src_acts.get(f"din_{li}"))
                pt = float(patch_logits[tid_t])
                pc = float(patch_logits[tid_c])
                p_gap = pt - pc
                delta_gap = p_gap - ref_gap
                effect = compute_effect(delta_gap, base_gap, direction)
                results[li]["dact_top5"][direction].append({
                    "effect": effect, "delta_gap": delta_gap,
                    "delta_t": pt - ref_target, "delta_c": pc - ref_compet,
                })

            gc.collect()
            torch.cuda.empty_cache()

        del clean_acts, corrupt_acts
        gc.collect()
        torch.cuda.empty_cache()

        if (pidx + 1) % 5 == 0:
            elapsed = time.time() - t0
            gpu_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            log(f"  [{pidx+1}/{n_test}] pairs done, elapsed={elapsed:.0f}s, GPU={gpu_gb:.1f}GB")

    # ================================================================
    # Part 3: Summary & Analysis
    # ================================================================
    log(f"\n  {'='*90}")
    log(f"  Phase 359+360 Summary: {model_name}")
    log(f"  {'='*90}")

    # --- Per-condition results ---
    log(f"\n  --- Per-Condition Results (effect = unified notation) ---")
    header = f"  {'Layer':>6} {'Dir':>5}"
    for cond in conditions:
        header += f" {cond:>12}"
    header += f" {'n':>4}"
    log(header)
    log(f"  {'-'*90}")

    summary = {}
    for li in target_layers:
        summary[li] = {}
        for direction in ["c2r", "r2c"]:
            row_str = f"  L{li:>4} {direction:>5}"
            for cond in conditions:
                vals = results[li][cond][direction]
                n = len(vals)
                if n == 0:
                    summary[li][f"{cond}_{direction}"] = {"n": 0, "mean": 0, "se": 0}
                    row_str += f" {'N/A':>12}"
                    continue
                effects = [v["effect"] for v in vals]
                mean_eff = float(np.mean(effects))
                se_eff = float(np.std(effects) / np.sqrt(n))
                summary[li][f"{cond}_{direction}"] = {
                    "n": n, "mean": mean_eff, "se": se_eff,
                    "mean_delta_gap": float(np.mean([v["delta_gap"] for v in vals])),
                }
                row_str += f" {mean_eff:>+12.4f}"
            row_str += f" {n:>4}"
            log(row_str)

    # --- Key Comparison 1: dact_top1 R2C vs full_mlp R2C ---
    log(f"\n  --- Key Comparison: dact_top1 vs full_mlp vs full_resid (R2C) ---")
    log(f"  {'Layer':>6} {'dact_top1':>12} {'full_mlp':>12} {'full_resid':>12} "
        f"{'dact→mlp':>10} {'mlp→resid':>10} {'Diagnosis':>30}")
    log(f"  {'-'*95}")

    diagnosis_results = {}
    for li in target_layers:
        da = summary[li].get("dact_top1_r2c", {}).get("mean", 0)
        fm = summary[li].get("full_mlp_r2c", {}).get("mean", 0)
        fr = summary[li].get("full_resid_r2c", {}).get("mean", 0)

        da2fm = "↑" if fm > da else "↓"
        fm2r = "↑" if fr > fm else "↓"

        # Diagnosis
        if da < 0 and fm > 0:
            diag = "dact incomplete; full MLP rescues"
        elif da < 0 and fm < 0 and fr > 0:
            diag = "MLP insufficient; need attn too"
        elif da < 0 and fm < 0 and fr < 0:
            diag = "even full layer can't rescue"
        elif da > 0 and fm > da:
            diag = "dact partial; full MLP better"
        elif da > 0 and fm <= da:
            diag = "dact sufficient; MLP no extra"
        else:
            diag = "unclear"

        diagnosis_results[li] = {
            "dact_r2c": da, "full_mlp_r2c": fm, "full_resid_r2c": fr,
            "diagnosis": diag,
        }

        log(f"  L{li:>4} {da:>+12.4f} {fm:>+12.4f} {fr:>+12.4f} "
            f"    {da2fm:>5} {fm2r:>5} {diag:>30}")

    # --- Key Comparison 2: dact_top1 vs dact_top5 ---
    log(f"\n  --- Channel Breadth: dact_top1 vs dact_top5 (R2C) ---")
    log(f"  {'Layer':>6} {'dact_top1':>12} {'dact_top5':>12} {'Δ(top5-top1)':>14} {'Broader helps?':>16}")
    log(f"  {'-'*65}")

    for li in target_layers:
        da1 = summary[li].get("dact_top1_r2c", {}).get("mean", 0)
        da5 = summary[li].get("dact_top5_r2c", {}).get("mean", 0)
        delta = da5 - da1
        helps = "YES" if delta > 0.02 else ("marginal" if delta > 0 else "NO")
        log(f"  L{li:>4} {da1:>+12.4f} {da5:>+12.4f} {delta:>+14.4f} {helps:>16}")

    # --- Phase 360: Coupling Analysis ---
    log(f"\n  --- Phase 360: cproj-dact Coupling (R2C) ---")
    log(f"  {'Layer':>6} {'cproj_only':>12} {'dact_only':>12} {'combined':>12} "
        f"{'marg_dact':>12} {'marg_cproj':>12} {'Interaction':>12}")
    log(f"  {'-'*80}")

    coupling_results = {}
    for li in target_layers:
        cp = summary[li].get("cproj_top1_r2c", {}).get("mean", 0)
        da = summary[li].get("dact_top1_r2c", {}).get("mean", 0)
        cb = summary[li].get("comb_top1_r2c", {}).get("mean", 0)

        # Marginal effects
        marg_dact = cb - cp  # effect of adding dact given cproj
        marg_cproj = cb - da  # effect of adding cproj given dact

        # Interaction: combined - (cproj + dact)
        interaction = cb - (cp + da)

        coupling_results[li] = {
            "cproj_only": cp, "dact_only": da, "combined": cb,
            "marg_dact_given_cproj": marg_dact,
            "marg_cproj_given_dact": marg_cproj,
            "interaction": interaction,
        }

        log(f"  L{li:>4} {cp:>+12.4f} {da:>+12.4f} {cb:>+12.4f} "
            f"{marg_dact:>+12.4f} {marg_cproj:>+12.4f} {interaction:>+12.4f}")

    # --- Bootstrap CI for key conditions ---
    log(f"\n  --- Bootstrap 95% CI (1000 resamples, R2C) ---")
    np.random.seed(42)
    n_bootstrap = 1000

    for li in target_layers:
        log(f"  Layer {li}:")
        for cond in ["dact_top1", "full_mlp", "full_resid"]:
            vals = results[li][cond]["r2c"]
            if len(vals) < 5:
                log(f"    {cond}: too few samples ({len(vals)})")
                continue
            effects = np.array([v["effect"] for v in vals])
            boot_means = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(effects, size=len(effects), replace=True)
                boot_means.append(float(np.mean(sample)))
            ci_lo = float(np.percentile(boot_means, 2.5))
            ci_hi = float(np.percentile(boot_means, 97.5))
            mean_eff = float(np.mean(effects))
            log(f"    {cond}: {mean_eff:+.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}]")

    # --- Per-pair consistency for key comparison ---
    log(f"\n  --- Per-pair Consistency: dact_top1 R2C sign vs full_mlp R2C sign ---")
    for li in target_layers:
        da_vals = results[li]["dact_top1"]["r2c"]
        fm_vals = results[li]["full_mlp"]["r2c"]
        n = min(len(da_vals), len(fm_vals))
        if n == 0:
            continue
        # Count pairs where sign differs
        both_neg = sum(1 for i in range(n) if da_vals[i]["effect"] < 0 and fm_vals[i]["effect"] < 0)
        da_neg_fm_pos = sum(1 for i in range(n) if da_vals[i]["effect"] < 0 and fm_vals[i]["effect"] > 0)
        da_pos_fm_neg = sum(1 for i in range(n) if da_vals[i]["effect"] > 0 and fm_vals[i]["effect"] < 0)
        both_pos = sum(1 for i in range(n) if da_vals[i]["effect"] > 0 and fm_vals[i]["effect"] > 0)
        log(f"  L{li}: both_neg={both_neg}, da_neg_fm_pos={da_neg_fm_pos}, "
            f"da_pos_fm_neg={da_pos_fm_neg}, both_pos={both_pos} (n={n})")

    # ================================================================
    # Save
    # ================================================================
    output = {
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "phase": "359+360",
        "unified_notation": {
            "C2R_effect": "-Δgap / |base_gap| (positive = binding damaged)",
            "R2C_effect": "+Δgap / |base_gap| (positive = binding rescued)",
            "base_gap": "clean_gap - corrupt_gap",
        },
        "test_layers": target_layers,
        "conditions": conditions,
        "n_pairs": n_test,
        "summary": {str(k): v for k, v in summary.items()},
        "diagnosis": {str(k): v for k, v in diagnosis_results.items()},
        "coupling": {str(k): v for k, v in coupling_results.items()},
        "per_condition_per_pair": {},
    }

    # Save per-pair data (compact)
    for li in target_layers:
        output["per_condition_per_pair"][str(li)] = {}
        for cond in conditions:
            output["per_condition_per_pair"][str(li)][cond] = {
                "c2r_effects": [v["effect"] for v in results[li][cond]["c2r"]],
                "r2c_effects": [v["effect"] for v in results[li][cond]["r2c"]],
                "c2r_delta_gap": [v["delta_gap"] for v in results[li][cond]["c2r"]],
                "r2c_delta_gap": [v["delta_gap"] for v in results[li][cond]["r2c"]],
            }

    os.makedirs("results/phase359_dact_context_compat", exist_ok=True)
    out_path = f"results/phase359_dact_context_compat/{model_name}_phase359.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=str, ensure_ascii=False)
    log(f"\n  Saved to {out_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    log(f"Phase 359+360 complete for {model_name} in {time.time()-t0:.0f}s")
    return output


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_experiment(model_name)
