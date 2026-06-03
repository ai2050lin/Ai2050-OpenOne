"""
Phase 352: Patch Ablation — Clean/Corrupt Swap vs Zero-Out
===========================================================

Goal: Distinguish "channel presence" vs "channel activation difference" in binding causation.

Three intervention types for each channel set:
1. zero_out:  Zero channels at down_proj input during clean forward pass (Phase 351 baseline)
2. C2R:       Replace channel activations with corrupt values during clean forward pass
3. R2C:       Replace channel activations with clean values during corrupt forward pass

Key comparisons:
- If C2R ≈ zero_out → channel VALUE ≈ 0 for binding, only PRESENCE matters
- If C2R < zero_out → zero is more disruptive than corrupt values
- If C2R > zero_out → corrupt values actively oppose binding (worse than zero)
- If R2C recovers binding → these channels are SUFFICIENT for binding rescue
- If R2C ≈ FracLost(C2R) → symmetric, clean-corrupt difference is the causal signal
- If R2C > FracLost(C2R) → rescue > destruction, indicating redundancy

Four channel types:
- Top 1% cproj, Top 1% dact, Top 1% contrib, Random
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
        "d_ff": 9728,
    },
    "glm4": {
        "path": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
        "n_layers": 40, "d_model": 4096,
        "binding_layers": [30, 33, 36, 38],
        "d_ff": 13696,
    },
    "deepseek7b": {
        "path": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "n_layers": 28, "d_model": 3584,
        "binding_layers": [19, 21, 23, 24],
        "d_ff": 18944,
    },
}

TEST_PAIRS = [
    ("apple", "red", "blue"), ("banana", "yellow", "purple"), ("snow", "white", "black"),
    ("sky", "blue", "green"), ("fire", "hot", "cold"), ("grass", "green", "red"),
    ("ocean", "blue", "yellow"), ("sun", "yellow", "purple"), ("blood", "red", "green"),
    ("ice", "cold", "hot"), ("cherry", "red", "blue"), ("leaf", "green", "red"),
    ("rose", "red", "blue"), ("gold", "yellow", "purple"), ("coal", "black", "white"),
    ("silver", "white", "black"), ("milk", "white", "black"), ("honey", "yellow", "blue"),
    ("ruby", "red", "green"), ("emerald", "green", "red"), ("sapphire", "blue", "red"),
    ("rain", "wet", "dry"), ("desert", "hot", "cold"), ("moon", "white", "black"),
    ("smoke", "gray", "red"), ("flame", "orange", "blue"), ("forest", "green", "white"),
    ("night", "dark", "bright"), ("steel", "gray", "gold"), ("ivory", "white", "black"),
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
                if dk in keys and W_down is None:
                    W_down = sf.get_tensor(dk).float().numpy()
                if W_down is not None:
                    break
        except:
            continue
    return W_gate, W_up, W_down, d_ff


def get_mlp_weights(layer, model_name=None, model=None):
    mlp = layer.mlp
    W_gate = W_up = W_down = None; d_ff = 0
    if hasattr(mlp, 'gate_up_proj'):
        w = safe_weight_to_numpy(mlp.gate_up_proj.weight)
        if w is not None:
            d_ff = w.shape[0] // 2; W_gate, W_up = w[:d_ff], w[d_ff:]
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
    if W_down is None and model_name is not None:
        layers = get_layers(model)
        for i, l in enumerate(layers):
            if l is layer:
                W_gate, W_up, W_down, d_ff = get_mlp_weights_from_disk(model_name, i)
                break
    return W_gate, W_up, W_down, d_ff


def silu_np(x):
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -20, 20))))


def capture_mlp_internals(model, tokenizer, device, prompt, target_layers):
    """Capture gate/up activations for channel identification."""
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
        out = model(**inp, output_hidden_states=False)
    logits = out.logits[0, -1].float().cpu().numpy()
    for h in hooks:
        h.remove()
    return captured, logits


def capture_down_proj_inputs(model, tokenizer, device, prompt, target_layers):
    """Capture down_proj input activations (full sequence) for patching.
    
    Returns:
        acts: {layer_idx: numpy_array [seq_len, d_ff]}  — SiLU(gate)*up at down_proj input
        logits: numpy_array [vocab_size]
    """
    layers = get_layers(model)
    captured = {}
    
    def make_pre_hook(key):
        def pre_hook(module, args):
            inp = args[0]
            # Store full sequence for patching: [seq_len, d_ff]
            captured[key] = inp[0].detach().cpu().float().numpy()
        return pre_hook
    
    hooks = []
    for li in target_layers:
        hooks.append(layers[li].mlp.down_proj.register_forward_pre_hook(make_pre_hook(f"din_{li}")))
    
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=False)
    logits = out.logits[0, -1].float().cpu().numpy()
    
    for h in hooks:
        h.remove()
    return captured, logits


def run_model_with_channel_zero(model, tokenizer, device, prompt, target_layers,
                                 channels_to_zero_by_layer):
    """Zero out channels at down_proj input (Phase 351 approach)."""
    layers = get_layers(model)
    hooks = []
    for li, channels_to_zero in channels_to_zero_by_layer.items():
        if li >= len(layers) or not channels_to_zero:
            continue
        ch_list = sorted(channels_to_zero)
        ch_tensor = torch.tensor(ch_list, dtype=torch.long)
        
        def make_pre_hook(ch_indices):
            def pre_hook(module, args):
                inp = args[0]
                if inp.dim() == 3 and inp.shape[-1] > max(ch_indices):
                    modified = inp.clone()
                    modified[:, :, ch_indices] = 0.0
                    return (modified,) + args[1:]
                return args
            return pre_hook
        hooks.append(layers[li].mlp.down_proj.register_forward_pre_hook(make_pre_hook(ch_tensor)))
    
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=False)
    logits = out.logits[0, -1].float().cpu().numpy()
    for h in hooks:
        h.remove()
    return logits


def run_model_with_channel_patch(model, tokenizer, device, prompt, target_layers,
                                  channels_to_patch_by_layer, replacement_acts,
                                  model_for_device=None):
    """Patch channels at down_proj input with replacement activations.
    
    Args:
        channels_to_patch_by_layer: {layer_idx: set of channel indices to patch}
        replacement_acts: {layer_idx: numpy_array [seq_len, d_ff]} — replacement values
        model_for_device: model object for getting device info (optional)
    """
    layers = get_layers(model)
    hooks = []
    
    for li, channels_to_patch in channels_to_patch_by_layer.items():
        if li >= len(layers) or not channels_to_patch:
            continue
        ch_list = sorted(channels_to_patch)
        rep_np = replacement_acts[li]  # [seq_len, d_ff]
        max_ch = max(ch_list)
        n_ch = len(ch_list)
        
        def make_patch_pre_hook(ch_indices_list, rep_numpy, max_ch_val):
            # Store replacement as numpy, convert at hook time using input's device
            def pre_hook(module, args):
                inp = args[0]
                if inp.dim() == 3 and inp.shape[-1] > max_ch_val:
                    modified = inp.clone()
                    # Convert replacement to tensor on same device as input
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


def identify_channels(model, tokenizer, device, model_name, W_U, binding_layers, ref_pairs, layers_obj, mlp_weights):
    """Identify Top 1% channels from reference pairs (same as Phase 351b)."""
    channel_counts_cproj = defaultdict(lambda: defaultdict(int))
    channel_counts_dact = defaultdict(lambda: defaultdict(int))
    channel_counts_contrib = defaultdict(lambda: defaultdict(int))
    
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
            W_down = mw["W_down"]; d_ff = mw["d_ff"]
            if W_down is None:
                continue
            gk = f"gate_{li}"; uk = f"up_{li}"
            if gk not in clean_caps or gk not in corrupt_caps:
                continue
            cg = clean_caps[gk][:d_ff]; crg = corrupt_caps[gk][:d_ff]
            cu = clean_caps.get(uk, np.ones(d_ff))[:d_ff]
            cru = corrupt_caps.get(uk, np.ones(d_ff))[:d_ff]
            min_d = min(d_ff, W_down.shape[1], cg.shape[0])
            Wd = W_down[:, :min_d]
            gsc = silu_np(cg[:min_d]); gsr = silu_np(crg[:min_d])
            uc = cu[:min_d]; ur = cru[:min_d]
            dact = gsc * uc - gsr * ur
            channel_proj = Wd.T @ direction_normed
            contribution = channel_proj * dact
            n_top1 = max(1, min_d // 100)
            
            for ch in np.argsort(np.abs(channel_proj))[-n_top1:]:
                channel_counts_cproj[li][int(ch)] += 1
            for ch in np.argsort(np.abs(dact))[-n_top1:]:
                channel_counts_dact[li][int(ch)] += 1
            for ch in np.argsort(np.abs(contribution))[-n_top1:]:
                channel_counts_contrib[li][int(ch)] += 1
        
        del clean_caps, corrupt_caps
        gc.collect(); torch.cuda.empty_cache()
    
    n_ref = len(ref_pairs)
    top1_cproj_channels = {}
    top1_dact_channels = {}
    top1_contrib_channels = {}
    random_channels = {}
    
    for li in binding_layers:
        d_ff = mlp_weights[li]["d_ff"]
        n_top1 = max(1, d_ff // 100)
        
        top1_cproj_channels[li] = set(
            ch for ch, cnt in channel_counts_cproj[li].items() if cnt >= n_ref * 0.3)
        if not top1_cproj_channels[li]:
            sorted_ch = sorted(channel_counts_cproj[li].items(), key=lambda x: -x[1])
            top1_cproj_channels[li] = set(ch for ch, _ in sorted_ch[:n_top1])
        
        top1_dact_channels[li] = set(
            ch for ch, cnt in channel_counts_dact[li].items() if cnt >= n_ref * 0.3)
        if not top1_dact_channels[li]:
            sorted_ch = sorted(channel_counts_dact[li].items(), key=lambda x: -x[1])
            top1_dact_channels[li] = set(ch for ch, _ in sorted_ch[:n_top1])
        
        top1_contrib_channels[li] = set(
            ch for ch, cnt in channel_counts_contrib[li].items() if cnt >= n_ref * 0.3)
        if not top1_contrib_channels[li]:
            sorted_ch = sorted(channel_counts_contrib[li].items(), key=lambda x: -x[1])
            top1_contrib_channels[li] = set(ch for ch, _ in sorted_ch[:n_top1])
        
        np.random.seed(42)
        n_random = len(top1_cproj_channels[li])
        all_channels = list(range(d_ff))
        random_channels[li] = set(np.random.choice(all_channels, size=min(n_random, len(all_channels)), replace=False))
        
        log(f"  Layer {li}: cproj={len(top1_cproj_channels[li])} dact={len(top1_dact_channels[li])} "
            f"contrib={len(top1_contrib_channels[li])} random={len(random_channels[li])}")
    
    return top1_cproj_channels, top1_dact_channels, top1_contrib_channels, random_channels


def run_pair_experiment(model, tokenizer, device, model_name, W_U, obj, target, competitor,
                        binding_layers, channel_groups, mlp_weights):
    """Run all interventions for a single test pair.
    
    Returns dict with binding effects for each intervention type.
    """
    tid_t = get_token_id(tokenizer, target)
    tid_c = get_token_id(tokenizer, competitor)
    if tid_t is None or tid_c is None:
        return None
    
    clean_prompt = f"The {obj}"
    
    # Step 1: Capture clean and corrupt down_proj inputs + logits
    log(f"    Capturing clean acts...", end="")
    clean_acts_raw, clean_logits = capture_down_proj_inputs(
        model, tokenizer, device, clean_prompt, binding_layers)
    log(f" corrupt...", end="")
    corrupt_acts_raw, corrupt_logits = capture_down_proj_inputs(
        model, tokenizer, device, CORRUPTED_BASELINE, binding_layers)
    log(f" done.", end="")
    
    # Convert keys from "din_21" → 21
    clean_acts = {li: clean_acts_raw[f"din_{li}"] for li in binding_layers if f"din_{li}" in clean_acts_raw}
    corrupt_acts = {li: corrupt_acts_raw[f"din_{li}"] for li in binding_layers if f"din_{li}" in corrupt_acts_raw}
    del clean_acts_raw, corrupt_acts_raw
    
    # Compute baselines
    clean_diff_base = float(clean_logits[tid_t] - clean_logits[tid_c])
    corrupt_diff_base = float(corrupt_logits[tid_t] - corrupt_logits[tid_c])
    binding_effect_base = clean_diff_base - corrupt_diff_base
    
    if abs(binding_effect_base) < 1e-10:
        log(f" binding_effect ≈ 0, skipping")
        return None
    
    results = {
        "clean_diff_base": clean_diff_base,
        "corrupt_diff_base": corrupt_diff_base,
        "binding_effect_base": binding_effect_base,
    }
    
    # Step 2: Run interventions for each channel group
    for gname, channels in channel_groups.items():
        if not any(channels.get(li, set()) for li in binding_layers):
            continue
        
        # 2a. Zero-out on clean
        log(f" {gname}:zero...", end="")
        zero_logits = run_model_with_channel_zero(
            model, tokenizer, device, clean_prompt, binding_layers, channels)
        zero_diff = float(zero_logits[tid_t] - zero_logits[tid_c])
        zero_binding = zero_diff - corrupt_diff_base
        frac_lost_zero = (binding_effect_base - zero_binding) / abs(binding_effect_base)
        
        # 2b. C2R patch: clean prompt, replace channels with corrupt values
        log(f" C2R...", end="")
        c2r_logits = run_model_with_channel_patch(
            model, tokenizer, device, clean_prompt, binding_layers,
            channels, corrupt_acts, model)
        c2r_diff = float(c2r_logits[tid_t] - c2r_logits[tid_c])
        c2r_binding = c2r_diff - corrupt_diff_base
        frac_lost_c2r = (binding_effect_base - c2r_binding) / abs(binding_effect_base)
        
        # 2c. R2C patch: corrupt prompt, replace channels with clean values
        log(f" R2C...", end="")
        r2c_logits = run_model_with_channel_patch(
            model, tokenizer, device, CORRUPTED_BASELINE, binding_layers,
            channels, clean_acts, model)
        r2c_diff = float(r2c_logits[tid_t] - r2c_logits[tid_c])
        r2c_binding = clean_diff_base - r2c_diff
        frac_recovered_r2c = (r2c_diff - corrupt_diff_base) / abs(binding_effect_base)
        
        results[f"{gname}_zero"] = {
            "diff": zero_diff,
            "binding": zero_binding,
            "frac_lost": frac_lost_zero,
        }
        results[f"{gname}_c2r"] = {
            "diff": c2r_diff,
            "binding": c2r_binding,
            "frac_lost": frac_lost_c2r,
        }
        results[f"{gname}_r2c"] = {
            "diff": r2c_diff,
            "frac_recovered": frac_recovered_r2c,
        }
        
        gc.collect(); torch.cuda.empty_cache()
    
    log(f" ✓")
    return results


def run_experiment(model_name):
    log(f"Phase 352: Patch Ablation — C2R/R2C vs Zero-Out ({model_name})")
    log("=" * 70)
    t0 = time.time()
    cfg = MODEL_CONFIGS[model_name]
    binding_layers = cfg["binding_layers"]

    model, tokenizer, device = load_model_bf16(model_name)
    W_U = get_W_U(model, model_name)
    layers = get_layers(model)
    
    mlp_weights = {}
    for li in binding_layers:
        W_gate, W_up, W_down, d_ff = get_mlp_weights(layers[li], model_name, model)
        mlp_weights[li] = {"W_gate": W_gate, "W_up": W_up, "W_down": W_down, "d_ff": d_ff}
    log(f"  MLP weights loaded")

    # Part 1: Channel identification with 20 reference pairs
    log(f"\n  Part 1: Identifying Top 1% channels from 20 reference pairs...")
    ref_pairs = TEST_PAIRS[:20]
    top1_cproj, top1_dact, top1_contrib, random_ch = identify_channels(
        model, tokenizer, device, model_name, W_U, binding_layers, ref_pairs, layers, mlp_weights)

    channel_groups = {
        "top1_cproj": top1_cproj,
        "top1_dact": top1_dact,
        "top1_contrib": top1_contrib,
        "random": random_ch,
    }
    
    # Part 2: Run interventions on all test pairs
    n_test = len(TEST_PAIRS)
    log(f"\n  Part 2: Running patch ablation on {n_test} test pairs...")
    log(f"  Interventions per pair: zero_out, C2R, R2C × 4 channel types = 12 forward passes")
    log(f"  Plus 2 baseline captures = 14 forward passes per pair")
    
    all_results = []
    for pidx, (obj, target, competitor) in enumerate(TEST_PAIRS):
        log(f"  [{pidx+1}/{n_test}] {obj}/{target}vs{competitor}:")
        result = run_pair_experiment(
            model, tokenizer, device, model_name, W_U,
            obj, target, competitor, binding_layers, channel_groups, mlp_weights)
        if result is not None:
            all_results.append(result)
        
        if (pidx + 1) % 5 == 0:
            elapsed = time.time() - t0
            log(f"  --- Progress: {pidx+1}/{n_test}, elapsed={elapsed:.0f}s, "
                f"GPU={torch.cuda.memory_allocated()/1e9:.1f}GB ---")
    
    # Part 3: Summary
    log(f"\n  Part 3: Computing summary statistics...")
    
    # Collect frac_lost/frac_recovered for each group × intervention
    summary = {}
    for gname in ["top1_cproj", "top1_dact", "top1_contrib", "random"]:
        for itype in ["zero", "c2r", "r2c"]:
            key = f"{gname}_{itype}"
            if itype in ["zero", "c2r"]:
                vals = [r[key]["frac_lost"] for r in all_results if key in r]
                if vals:
                    summary[key] = {
                        "mean": float(np.mean(vals)),
                        "se": float(np.std(vals) / np.sqrt(len(vals))),
                        "n": len(vals),
                        "type": "frac_lost",
                    }
            else:  # r2c
                vals = [r[key]["frac_recovered"] for r in all_results if key in r]
                if vals:
                    summary[key] = {
                        "mean": float(np.mean(vals)),
                        "se": float(np.std(vals) / np.sqrt(len(vals))),
                        "n": len(vals),
                        "type": "frac_recovered",
                    }
    
    # Print summary table
    log(f"\n  ══════════════════════════════════════════════════════════════════")
    log(f"  Phase 352 Summary: {model_name}")
    log(f"  ══════════════════════════════════════════════════════════════════")
    log(f"  {'Channel':<15} {'Zero-Out':>12} {'C2R':>12} {'R2C':>12} {'C2R-Zero':>12}")
    log(f"  {'-'*65}")
    
    for gname in ["top1_cproj", "top1_dact", "top1_contrib", "random"]:
        zero_key = f"{gname}_zero"
        c2r_key = f"{gname}_c2r"
        r2c_key = f"{gname}_r2c"
        
        zero_val = summary.get(zero_key, {}).get("mean", 0)
        zero_se = summary.get(zero_key, {}).get("se", 0)
        c2r_val = summary.get(c2r_key, {}).get("mean", 0)
        c2r_se = summary.get(c2r_key, {}).get("se", 0)
        r2c_val = summary.get(r2c_key, {}).get("mean", 0)
        r2c_se = summary.get(r2c_key, {}).get("se", 0)
        diff = c2r_val - zero_val
        
        log(f"  {gname:<15} {zero_val:>+8.4f}({zero_se:>5.3f}) "
            f"{c2r_val:>+8.4f}({c2r_se:>5.3f}) "
            f"{r2c_val:>+8.4f}({r2c_se:>5.3f}) "
            f"{diff:>+8.4f}")
    
    log(f"  {'-'*65}")
    log(f"  Zero-Out / C2R: FracLost (positive = binding reduced)")
    log(f"  R2C: FracRecovered (positive = binding rescued)")
    log(f"  C2R-Zero: C2R effect minus Zero-Out effect")
    log(f"    >0: corrupt values worse than zero (anti-binding)")
    log(f"    <0: zero worse than corrupt values (signal destroyed)")
    log(f"    ≈0: corrupt values ≈ zero for binding")
    
    # Per-group detailed comparison
    log(f"\n  --- Detailed: C2R vs Zero-Out ---")
    log(f"  {'Channel':<15} {'C2R>Zero?':>12} {'Interpretation'}")
    log(f"  {'-'*65}")
    for gname in ["top1_cproj", "top1_dact", "top1_contrib", "random"]:
        zero_val = summary.get(f"{gname}_zero", {}).get("mean", 0)
        c2r_val = summary.get(f"{gname}_c2r", {}).get("mean", 0)
        diff = c2r_val - zero_val
        if abs(diff) < 0.01:
            interp = "C2R ≈ Zero → corrupt vals ≈ 0 for binding"
        elif diff > 0:
            interp = "C2R > Zero → corrupt vals actively oppose binding"
        else:
            interp = "C2R < Zero → zero more disruptive; channel value matters"
        log(f"  {gname:<15} {diff:>+12.4f}   {interp}")
    
    log(f"\n  --- Detailed: R2C Recovery ---")
    log(f"  {'Channel':<15} {'R2C FracRec':>14} {'vs C2R FracLost':>16} {'Interpretation'}")
    log(f"  {'-'*80}")
    for gname in ["top1_cproj", "top1_dact", "top1_contrib", "random"]:
        c2r_val = abs(summary.get(f"{gname}_c2r", {}).get("mean", 0))
        r2c_val = summary.get(f"{gname}_r2c", {}).get("mean", 0)
        if c2r_val > 0.005:
            ratio = r2c_val / c2r_val
            if ratio > 1.5:
                interp = "Rescue > Destroy → redundancy"
            elif ratio < 0.5:
                interp = "Destroy > Rescue → context dependency"
            else:
                interp = "Symmetric → clean-corrupt diff is causal signal"
        else:
            interp = "C2R too small for comparison"
            ratio = 0
        log(f"  {gname:<15} {r2c_val:>+14.4f} {ratio:>16.2f}   {interp}")
    
    # Per-pair breakdown for key channels (top 5 most/least affected)
    log(f"\n  --- Per-Pair Breakdown: Top 1% cproj ---")
    cproj_zero_fracs = [(i, r.get("top1_cproj_zero", {}).get("frac_lost", 0)) 
                        for i, r in enumerate(all_results)]
    cproj_zero_fracs.sort(key=lambda x: x[1], reverse=True)
    log(f"  Top 5 most affected by zero-out:")
    for idx, frac in cproj_zero_fracs[:5]:
        pair = TEST_PAIRS[idx] if idx < len(TEST_PAIRS) else ("?", "?", "?")
        log(f"    {pair[0]}/{pair[1]}vs{pair[2]}: FracLost={frac:+.4f}")
    log(f"  Top 5 least affected (or anti-binding):")
    for idx, frac in cproj_zero_fracs[-5:]:
        pair = TEST_PAIRS[idx] if idx < len(TEST_PAIRS) else ("?", "?", "?")
        log(f"    {pair[0]}/{pair[1]}vs{pair[2]}: FracLost={frac:+.4f}")
    
    # Save results
    all_results_save = []
    for r in all_results:
        r_save = {}
        for k, v in r.items():
            if isinstance(v, dict):
                r_save[k] = {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv 
                             for kk, vv in v.items()}
            elif isinstance(v, (np.floating, float)):
                r_save[k] = float(v)
            else:
                r_save[k] = v
        all_results_save.append(r_save)
    
    output = {
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "phase": "352",
        "description": "Patch ablation: C2R (clean→corrupt), R2C (corrupt→clean), Zero-Out",
        "n_test_pairs": len(all_results),
        "n_ref_pairs": 20,
        "summary": {k: v for k, v in summary.items()},
        "per_pair_results": all_results_save,
    }
    
    os.makedirs("results/phase352_patch_ablation", exist_ok=True)
    out_path = f"results/phase352_patch_ablation/{model_name}_phase352.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    log(f"\n  Saved to {out_path}")
    
    del model; gc.collect(); torch.cuda.empty_cache()
    log(f"Phase 352 complete for {model_name} in {time.time()-t0:.0f}s")
    return output


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_experiment(model_name)
