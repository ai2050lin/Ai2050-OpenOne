"""
Phase 353: dact Path Context Dependency + Four-Quadrant Causal Decomposition
==============================================================================

Goals:
1. Test why dact R2C is negative — is dact context-dependent?
2. Four-quadrant analysis: (target_up/down) x (competitor_up/down)
3. Coupled channel patch: dact alone vs dact+correlated vs dact+cproj
4. Per-pair distribution of quadrant membership

Key questions:
- Does dact need companion channels to function properly?
- Is dact's anti-binding effect in R2C due to channel context mismatch?
- What fraction of pairs fall into each quadrant?
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
    if w.is_meta: return None
    try: return w.detach().cpu().float().numpy()
    except: return None


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
        if W_gate is not None: d_ff = W_gate.shape[0]
        elif W_up is not None: d_ff = W_up.shape[0]
    elif hasattr(mlp, 'up_proj'):
        W_up = safe_weight_to_numpy(mlp.up_proj.weight)
        if W_up is not None: d_ff = W_up.shape[0]
    if hasattr(mlp, 'down_proj'):
        W_down = safe_weight_to_numpy(mlp.down_proj.weight)
    if W_down is None and model_name is not None:
        import glob
        from safetensors import safe_open
        layers = get_layers(model)
        for i, l in enumerate(layers):
            if l is layer:
                for sf_file in glob.glob(os.path.join(MODEL_CONFIGS[model_name]["path"], '*.safetensors')):
                    try:
                        with safe_open(sf_file, framework='pt', device='cpu') as sf:
                            dk = f"model.layers.{i}.mlp.down_proj.weight"
                            if dk in sf.keys():
                                W_down = sf.get_tensor(dk).float().numpy()
                            guk = f"model.layers.{i}.mlp.gate_up_proj.weight"
                            if guk in sf.keys() and W_gate is None:
                                w = sf.get_tensor(guk).float().numpy()
                                d_ff = w.shape[0]//2; W_gate=w[:d_ff]; W_up=w[d_ff:]
                            gk = f"model.layers.{i}.mlp.gate_proj.weight"
                            if gk in sf.keys() and W_gate is None:
                                W_gate = sf.get_tensor(gk).float().numpy(); d_ff=W_gate.shape[0]
                            uk = f"model.layers.{i}.mlp.up_proj.weight"
                            if uk in sf.keys() and W_up is None:
                                W_up = sf.get_tensor(uk).float().numpy()
                                if d_ff==0: d_ff=W_up.shape[0]
                            if W_down is not None: break
                    except: continue
                break
    return W_gate, W_up, W_down, d_ff


def silu_np(x):
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -20, 20))))


def capture_mlp_internals(model, tokenizer, device, prompt, target_layers):
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
    for h in hooks: h.remove()
    return captured, logits


def capture_down_proj_inputs(model, tokenizer, device, prompt, target_layers):
    layers = get_layers(model)
    captured = {}
    def make_pre_hook(key):
        def pre_hook(module, args):
            inp = args[0]
            captured[key] = inp[0].detach().cpu().float().numpy()
        return pre_hook
    hooks = []
    for li in target_layers:
        hooks.append(layers[li].mlp.down_proj.register_forward_pre_hook(make_pre_hook(f"din_{li}")))
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=False)
    logits = out.logits[0, -1].float().cpu().numpy()
    for h in hooks: h.remove()
    acts = {li: captured[f"din_{li}"] for li in target_layers if f"din_{li}" in captured}
    return acts, logits


def run_model_with_channel_patch(model, tokenizer, device, prompt, target_layers,
                                  channels_to_patch_by_layer, replacement_acts):
    """Patch specified channels at down_proj input with replacement activations."""
    layers = get_layers(model)
    hooks = []
    for li, channels_to_patch in channels_to_patch_by_layer.items():
        if li >= len(layers) or not channels_to_patch:
            continue
        ch_list = sorted(channels_to_patch)
        rep_np = replacement_acts[li]
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
    for h in hooks: h.remove()
    return logits


def identify_channels(model, tokenizer, device, model_name, W_U, binding_layers, ref_pairs, layers_obj, mlp_weights):
    """Identify Top 1% cproj, dact, and contribution channels."""
    channel_counts_cproj = defaultdict(lambda: defaultdict(int))
    channel_counts_dact = defaultdict(lambda: defaultdict(int))
    channel_counts_contrib = defaultdict(lambda: defaultdict(int))
    
    for pidx, (obj, target, competitor) in enumerate(ref_pairs):
        tid_t = get_token_id(tokenizer, target)
        tid_c = get_token_id(tokenizer, competitor)
        if tid_t is None or tid_c is None: continue
        direction = W_U[tid_t] - W_U[tid_c]
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-10: continue
        direction_normed = direction / dir_norm
        
        clean_caps, _ = capture_mlp_internals(model, tokenizer, device, f"The {obj}", binding_layers)
        corrupt_caps, _ = capture_mlp_internals(model, tokenizer, device, CORRUPTED_BASELINE, binding_layers)
        
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
    top1_cproj = {}; top1_dact = {}; top1_contrib = {}
    
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
        
        top1_contrib[li] = set(
            ch for ch, cnt in channel_counts_contrib[li].items() if cnt >= n_ref * 0.3)
        if not top1_contrib[li]:
            sorted_ch = sorted(channel_counts_contrib[li].items(), key=lambda x: -x[1])
            top1_contrib[li] = set(ch for ch, _ in sorted_ch[:n_top1])
        
        # Find dact-correlated channels: channels whose activation correlates with dact channels
        # We'll compute this per-pair during testing
        
        log(f"  Layer {li}: cproj={len(top1_cproj[li])} dact={len(top1_dact[li])} contrib={len(top1_contrib[li])}")
    
    return top1_cproj, top1_dact, top1_contrib


def find_correlated_channels(clean_acts, corrupt_acts, binding_layers, dact_channels, n_corr=50):
    """Find channels whose clean-corrupt difference correlates with dact channel differences."""
    corr_channels = {}
    for li in binding_layers:
        if li not in clean_acts or li not in corrupt_acts:
            continue
        clean_act = clean_acts[li]  # [seq_len, d_ff]
        corrupt_act = corrupt_acts[li]  # [seq_len, d_ff]
        
        # Use last position difference
        dact_full = clean_act[-1] - corrupt_act[-1]  # [d_ff]
        
        # Get dact channel differences as reference
        dact_chs = sorted(dact_channels.get(li, set()))
        if not dact_chs:
            corr_channels[li] = set()
            continue
        
        # For each dact channel, compute correlation with all other channels
        dact_signal = dact_full[dact_chs].mean()  # mean dact channel diff
        
        # Find channels with similar difference pattern
        # Simple: channels where |delta| is in top n_corr (excluding dact channels themselves)
        abs_delta = np.abs(dact_full)
        # Zero out dact channels themselves
        for ch in dact_chs:
            if ch < len(abs_delta):
                abs_delta[ch] = 0
        
        # Also zero out very small channels
        threshold = np.percentile(abs_delta[abs_delta > 0], 50) if np.any(abs_delta > 0) else 0
        
        # Get top correlated (highest delta) channels
        top_corr_idx = np.argsort(abs_delta)[-n_corr:]
        corr_channels[li] = set(int(ch) for ch in top_corr_idx if abs_delta[ch] > threshold)
    
    return corr_channels


def classify_quadrant(target_change, competitor_change):
    """Classify intervention effect into four quadrants.
    
    A: target UP + competitor DOWN → strongest binding (pro-binding)
    B: target UP + competitor UP   → shared amplification (but target more)
    C: target DOWN + competitor DOWN → shared suppression (but competitor more)
    D: target DOWN + competitor UP → anti-binding
    """
    t_up = target_change > 0
    c_down = competitor_change < 0
    
    if t_up and c_down:
        return "A_pro_binding"       # target up, competitor down
    elif t_up and not c_down:
        return "B_shared_boost"      # both up, target boost dominant
    elif not t_up and c_down:
        return "C_shared_suppress"   # both down, competitor suppress dominant
    else:
        return "D_anti_binding"      # target down, competitor up


def run_experiment(model_name):
    log(f"Phase 353: dact Context Dependency + Four-Quadrant Decomposition ({model_name})")
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

    # Part 1: Channel identification
    log(f"\n  Part 1: Identifying Top 1% channels from 20 reference pairs...")
    ref_pairs = TEST_PAIRS[:20]
    top1_cproj, top1_dact, top1_contrib = identify_channels(
        model, tokenizer, device, model_name, W_U, binding_layers, ref_pairs, layers, mlp_weights)

    # ================================================================
    # Part 2: Four-Quadrant Analysis for C2R interventions
    # ================================================================
    n_test = len(TEST_PAIRS)
    log(f"\n  Part 2: Four-Quadrant C2R analysis on {n_test} pairs...")
    
    all_pair_results = []
    
    for pidx, (obj, target, competitor) in enumerate(TEST_PAIRS):
        tid_t = get_token_id(tokenizer, target)
        tid_c = get_token_id(tokenizer, competitor)
        if tid_t is None or tid_c is None: continue
        
        clean_prompt = f"The {obj}"
        
        # Baselines
        clean_acts, clean_logits = capture_down_proj_inputs(
            model, tokenizer, device, clean_prompt, binding_layers)
        corrupt_acts, corrupt_logits = capture_down_proj_inputs(
            model, tokenizer, device, CORRUPTED_BASELINE, binding_layers)
        
        clean_target = float(clean_logits[tid_t])
        clean_compet = float(clean_logits[tid_c])
        corrupt_target = float(corrupt_logits[tid_t])
        corrupt_compet = float(corrupt_logits[tid_c])
        
        clean_diff = clean_target - clean_compet
        corrupt_diff = corrupt_target - corrupt_compet
        binding_base = clean_diff - corrupt_diff
        
        if abs(binding_base) < 1e-10:
            continue
        
        pair_result = {
            "pair": f"{obj}/{target}vs{competitor}",
            "binding_base": binding_base,
        }
        
        # === For each channel group: C2R with four-quadrant decomposition ===
        channel_groups = {
            "cproj": top1_cproj,
            "dact": top1_dact,
            "contrib": top1_contrib,
        }
        
        for gname, channels in channel_groups.items():
            if not any(channels.get(li, set()) for li in binding_layers):
                continue
            
            # C2R: clean prompt, patch channels with corrupt values
            c2r_logits = run_model_with_channel_patch(
                model, tokenizer, device, clean_prompt, binding_layers, channels, corrupt_acts)
            c2r_target = float(c2r_logits[tid_t])
            c2r_compet = float(c2r_logits[tid_c])
            
            target_change = c2r_target - clean_target
            compet_change = c2r_compet - clean_compet
            quadrant = classify_quadrant(target_change, compet_change)
            
            pair_result[f"{gname}_c2r"] = {
                "target_change": target_change,
                "compet_change": compet_change,
                "quadrant": quadrant,
                "binding_change": (c2r_target - c2r_compet) - clean_diff,
            }
            
            gc.collect(); torch.cuda.empty_cache()
        
        # ================================================================
        # Part 3: dact Context Dependency — Coupled Channel Patch
        # ================================================================
        # Find correlated channels for this pair
        corr_channels = find_correlated_channels(
            clean_acts, corrupt_acts, binding_layers, top1_dact, n_corr=50)
        
        # Combine dact + correlated channels
        dact_plus_corr = {}
        for li in binding_layers:
            dact_plus_corr[li] = top1_dact.get(li, set()) | corr_channels.get(li, set())
        
        # Combine dact + cproj channels (same-layer coupling)
        dact_plus_cproj = {}
        for li in binding_layers:
            dact_plus_cproj[li] = top1_dact.get(li, set()) | top1_cproj.get(li, set())
        
        # Test 3 patch conditions:
        # a) dact alone (already done above as "dact_c2r")
        # b) dact + correlated
        # c) dact + cproj
        # d) dact + cproj R2C (corrupt context, inject clean)
        
        for patch_name, patch_channels in [("dact_corr", dact_plus_corr), ("dact_cproj", dact_plus_cproj)]:
            if not any(patch_channels.get(li, set()) for li in binding_layers):
                continue
            
            # C2R
            c2r_logits = run_model_with_channel_patch(
                model, tokenizer, device, clean_prompt, binding_layers, patch_channels, corrupt_acts)
            c2r_target = float(c2r_logits[tid_t])
            c2r_compet = float(c2r_logits[tid_c])
            t_change = c2r_target - clean_target
            c_change = c2r_compet - clean_compet
            quadrant = classify_quadrant(t_change, c_change)
            
            pair_result[f"{patch_name}_c2r"] = {
                "target_change": t_change,
                "compet_change": c_change,
                "quadrant": quadrant,
                "binding_change": (c2r_target - c2r_compet) - clean_diff,
            }
            
            # R2C
            r2c_logits = run_model_with_channel_patch(
                model, tokenizer, device, CORRUPTED_BASELINE, binding_layers, patch_channels, clean_acts)
            r2c_target = float(r2c_logits[tid_t])
            r2c_compet = float(r2c_logits[tid_c])
            t_change_r2c = r2c_target - corrupt_target
            c_change_r2c = r2c_compet - corrupt_compet
            quadrant_r2c = classify_quadrant(t_change_r2c, c_change_r2c)
            
            pair_result[f"{patch_name}_r2c"] = {
                "target_change": t_change_r2c,
                "compet_change": c_change_r2c,
                "quadrant": quadrant_r2c,
                "binding_recovered": (r2c_target - r2c_compet) - corrupt_diff,
            }
            
            gc.collect(); torch.cuda.empty_cache()
        
        # Also do dact R2C (corrupt context, inject clean dact)
        r2c_logits_dact = run_model_with_channel_patch(
            model, tokenizer, device, CORRUPTED_BASELINE, binding_layers, top1_dact, clean_acts)
        r2c_target = float(r2c_logits_dact[tid_t])
        r2c_compet = float(r2c_logits_dact[tid_c])
        t_change = r2c_target - corrupt_target
        c_change = r2c_compet - corrupt_compet
        quadrant = classify_quadrant(t_change, c_change)
        
        pair_result["dact_r2c"] = {
            "target_change": t_change,
            "compet_change": c_change,
            "quadrant": quadrant,
            "binding_recovered": (r2c_target - r2c_compet) - corrupt_diff,
        }
        
        # cproj R2C for comparison
        r2c_logits_cproj = run_model_with_channel_patch(
            model, tokenizer, device, CORRUPTED_BASELINE, binding_layers, top1_cproj, clean_acts)
        r2c_target = float(r2c_logits_cproj[tid_t])
        r2c_compet = float(r2c_logits_cproj[tid_c])
        t_change = r2c_target - corrupt_target
        c_change = r2c_compet - corrupt_compet
        quadrant = classify_quadrant(t_change, c_change)
        
        pair_result["cproj_r2c"] = {
            "target_change": t_change,
            "compet_change": c_change,
            "quadrant": quadrant,
            "binding_recovered": (r2c_target - r2c_compet) - corrupt_diff,
        }
        
        all_pair_results.append(pair_result)
        
        if (pidx + 1) % 5 == 0:
            log(f"  [{pidx+1}/{n_test}] elapsed={time.time()-t0:.0f}s, "
                f"GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")
        
        gc.collect(); torch.cuda.empty_cache()
    
    # ================================================================
    # Part 4: Summary Statistics
    # ================================================================
    log(f"\n  ══════════════════════════════════════════════════════════════")
    log(f"  Phase 353 Summary: {model_name}")
    log(f"  ══════════════════════════════════════════════════════════════")
    
    # Four-Quadrant Distribution
    log(f"\n  --- Four-Quadrant Distribution (C2R) ---")
    log(f"  {'Channel':<15} {'A:pro_bind':>12} {'B:shared↑':>12} {'C:shared↓':>12} {'D:anti_bind':>12}")
    log(f"  {'-'*65}")
    
    quadrant_summary = {}
    for gname in ["cproj", "dact", "contrib"]:
        c2r_key = f"{gname}_c2r"
        quadrants = [r[c2r_key]["quadrant"] for r in all_pair_results if c2r_key in r]
        n = len(quadrants)
        if n == 0: continue
        qa = sum(1 for q in quadrants if q == "A_pro_binding") / n * 100
        qb = sum(1 for q in quadrants if q == "B_shared_boost") / n * 100
        qc = sum(1 for q in quadrants if q == "C_shared_suppress") / n * 100
        qd = sum(1 for q in quadrants if q == "D_anti_binding") / n * 100
        
        log(f"  {gname:<15} {qa:>11.1f}% {qb:>11.1f}% {qc:>11.1f}% {qd:>11.1f}%")
        quadrant_summary[gname] = {"A": qa, "B": qb, "C": qc, "D": qd, "n": n}
    
    # R2C quadrant distribution
    log(f"\n  --- Four-Quadrant Distribution (R2C) ---")
    log(f"  {'Channel':<15} {'A:pro_bind':>12} {'B:shared↑':>12} {'C:shared↓':>12} {'D:anti_bind':>12}")
    log(f"  {'-'*65}")
    
    r2c_quadrant_summary = {}
    for gname in ["cproj", "dact"]:
        r2c_key = f"{gname}_r2c"
        quadrants = [r[r2c_key]["quadrant"] for r in all_pair_results if r2c_key in r]
        n = len(quadrants)
        if n == 0: continue
        qa = sum(1 for q in quadrants if q == "A_pro_binding") / n * 100
        qb = sum(1 for q in quadrants if q == "B_shared_boost") / n * 100
        qc = sum(1 for q in quadrants if q == "C_shared_suppress") / n * 100
        qd = sum(1 for q in quadrants if q == "D_anti_binding") / n * 100
        
        log(f"  {gname:<15} {qa:>11.1f}% {qb:>11.1f}% {qc:>11.1f}% {qd:>11.1f}%")
        r2c_quadrant_summary[gname] = {"A": qa, "B": qb, "C": qc, "D": qd, "n": n}
    
    # dact context dependency comparison
    log(f"\n  --- dact Context Dependency: dact alone vs dact+corr vs dact+cproj ---")
    log(f"  {'Patch':<20} {'C2R_FracLost':>14} {'C2R_TΔ':>10} {'C2R_CΔ':>10} {'R2C_FracRec':>14} {'R2C_TΔ':>10} {'R2C_CΔ':>10}")
    log(f"  {'-'*90}")
    
    context_summary = {}
    for patch_name in ["dact", "dact_corr", "dact_cproj"]:
        c2r_key = f"{patch_name}_c2r"
        r2c_key = f"{patch_name}_r2c"
        
        c2r_results = [r[c2r_key] for r in all_pair_results if c2r_key in r]
        r2c_results = [r[r2c_key] for r in all_pair_results if r2c_key in r]
        
        if not c2r_results and not r2c_results:
            continue
        
        mean_c2r_tc = float(np.mean([r["target_change"] for r in c2r_results])) if c2r_results else 0
        mean_c2r_cc = float(np.mean([r["compet_change"] for r in c2r_results])) if c2r_results else 0
        mean_c2r_bc = float(np.mean([r["binding_change"] for r in c2r_results])) if c2r_results else 0
        se_c2r_bc = float(np.std([r["binding_change"] for r in c2r_results])/np.sqrt(len(c2r_results))) if c2r_results else 0
        
        mean_r2c_tc = float(np.mean([r["target_change"] for r in r2c_results])) if r2c_results else 0
        mean_r2c_cc = float(np.mean([r["compet_change"] for r in r2c_results])) if r2c_results else 0
        mean_r2c_br = float(np.mean([r["binding_recovered"] for r in r2c_results])) if r2c_results else 0
        se_r2c_br = float(np.std([r["binding_recovered"] for r in r2c_results])/np.sqrt(len(r2c_results))) if r2c_results else 0
        
        # FracLost/FracRecovered relative to binding_base
        binding_bases = [r["binding_base"] for r in all_pair_results if c2r_key in r]
        mean_bb = float(np.mean(np.abs(binding_bases))) if binding_bases else 1
        
        frac_lost = mean_c2r_bc / mean_bb if mean_bb > 0.01 else 0
        frac_rec = mean_r2c_br / mean_bb if mean_bb > 0.01 else 0
        
        log(f"  {patch_name:<20} {frac_lost:>+14.4f} {mean_c2r_tc:>+10.4f} {mean_c2r_cc:>+10.4f} "
            f"{frac_rec:>+14.4f} {mean_r2c_tc:>+10.4f} {mean_r2c_cc:>+10.4f}")
        
        context_summary[patch_name] = {
            "c2r_frac_lost": frac_lost, "c2r_se": se_c2r_bc/mean_bb if mean_bb > 0.01 else 0,
            "c2r_target": mean_c2r_tc, "c2r_compet": mean_c2r_cc,
            "r2c_frac_recovered": frac_rec, "r2c_se": se_r2c_br/mean_bb if mean_bb > 0.01 else 0,
            "r2c_target": mean_r2c_tc, "r2c_compet": mean_r2c_cc,
        }
    
    # cproj alone comparison
    cproj_c2r = [r["cproj_c2r"] for r in all_pair_results if "cproj_c2r" in r]
    cproj_r2c = [r["cproj_r2c"] for r in all_pair_results if "cproj_r2c" in r]
    if cproj_c2r:
        binding_bases = [r["binding_base"] for r in all_pair_results if "cproj_c2r" in r]
        mean_bb = float(np.mean(np.abs(binding_bases))) if binding_bases else 1
        mean_bc = float(np.mean([r["binding_change"] for r in cproj_c2r]))
        mean_br = float(np.mean([r["binding_recovered"] for r in cproj_r2c])) if cproj_r2c else 0
        
        context_summary["cproj"] = {
            "c2r_frac_lost": mean_bc/mean_bb if mean_bb > 0.01 else 0,
            "c2r_target": float(np.mean([r["target_change"] for r in cproj_c2r])),
            "c2r_compet": float(np.mean([r["compet_change"] for r in cproj_c2r])),
            "r2c_frac_recovered": mean_br/mean_bb if mean_bb > 0.01 else 0,
            "r2c_target": float(np.mean([r["target_change"] for r in cproj_r2c])) if cproj_r2c else 0,
            "r2c_compet": float(np.mean([r["compet_change"] for r in cproj_r2c])) if cproj_r2c else 0,
        }
    
    # Per-pair dact R2C analysis: how many pairs show positive vs negative recovery?
    log(f"\n  --- dact R2C: Per-Pair Sign Distribution ---")
    dact_r2c_fracs = []
    dact_r2c_positive = 0
    dact_r2c_negative = 0
    for r in all_pair_results:
        if "dact_r2c" not in r: continue
        br = r["dact_r2c"]["binding_recovered"]
        bb = abs(r["binding_base"])
        frac = br / bb if bb > 0.01 else 0
        dact_r2c_fracs.append(frac)
        if frac > 0:
            dact_r2c_positive += 1
        else:
            dact_r2c_negative += 1
    
    n_r2c = len(dact_r2c_fracs)
    log(f"  dact R2C positive (pro-binding recovery): {dact_r2c_positive}/{n_r2c} ({dact_r2c_positive/n_r2c*100:.1f}%)")
    log(f"  dact R2C negative (anti-binding): {dact_r2c_negative}/{n_r2c} ({dact_r2c_negative/n_r2c*100:.1f}%)")
    log(f"  dact R2C mean frac: {np.mean(dact_r2c_fracs):+.4f} ± {np.std(dact_r2c_fracs)/np.sqrt(n_r2c):.4f}")
    
    # dact+cproj R2C comparison
    dact_cproj_r2c_fracs = []
    for r in all_pair_results:
        if "dact_cproj_r2c" not in r: continue
        br = r["dact_cproj_r2c"]["binding_recovered"]
        bb = abs(r["binding_base"])
        frac = br / bb if bb > 0.01 else 0
        dact_cproj_r2c_fracs.append(frac)
    
    if dact_cproj_r2c_fracs:
        n_dc = len(dact_cproj_r2c_fracs)
        dc_pos = sum(1 for f in dact_cproj_r2c_fracs if f > 0)
        log(f"  dact+cproj R2C positive: {dc_pos}/{n_dc} ({dc_pos/n_dc*100:.1f}%)")
        log(f"  dact+cproj R2C mean frac: {np.mean(dact_cproj_r2c_fracs):+.4f} ± {np.std(dact_cproj_r2c_fracs)/np.sqrt(n_dc):.4f}")
        log(f"  → Adding cproj channels {'improves' if np.mean(dact_cproj_r2c_fracs) > np.mean(dact_r2c_fracs) else 'worsens'} dact R2C")
    
    # Save
    output = {
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "phase": "353",
        "quadrant_summary_c2r": quadrant_summary,
        "quadrant_summary_r2c": r2c_quadrant_summary,
        "context_dependency": context_summary,
        "dact_r2c_sign": {
            "positive": dact_r2c_positive, "negative": dact_r2c_negative,
            "mean_frac": float(np.mean(dact_r2c_fracs)) if dact_r2c_fracs else 0,
            "se_frac": float(np.std(dact_r2c_fracs)/np.sqrt(len(dact_r2c_fracs))) if dact_r2c_fracs else 0,
        },
        "n_pairs": len(all_pair_results),
        "per_pair": all_pair_results,
    }
    
    os.makedirs("results/phase353_dact_context", exist_ok=True)
    out_path = f"results/phase353_dact_context/{model_name}_phase353.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=str, ensure_ascii=False)
    log(f"\n  Saved to {out_path}")
    
    del model; gc.collect(); torch.cuda.empty_cache()
    log(f"Phase 353 complete for {model_name} in {time.time()-t0:.0f}s")
    return output


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_experiment(model_name)
