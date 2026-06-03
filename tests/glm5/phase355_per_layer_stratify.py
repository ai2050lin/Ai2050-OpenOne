"""
Phase 355: cproj/dact Per-Layer Functional Stratification
=========================================================

Goals:
1. Per-layer gap decomposition: which layers contribute most to binding gap?
2. Single-layer patch: patch only one binding layer at a time
3. Layer interaction: do early layers create gap, later layers amplify?
4. Within cproj channels: separate pro-binding (A quadrant) vs anti-binding (D quadrant) channels

Key questions:
- Is binding a single-layer or multi-layer process?
- Does cproj gap effect concentrate in specific layers?
- Is dact amplification layer-dependent?
- Can we identify "gap creator" vs "gap amplifier" layers?
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
    ("sky", "blue", "green"), ("cherry", "red", "blue"), ("leaf", "green", "red"),
    ("rose", "red", "blue"), ("gold", "yellow", "purple"), ("coal", "black", "white"),
    ("silver", "white", "black"), ("milk", "white", "black"), ("honey", "yellow", "blue"),
    ("ruby", "red", "green"), ("emerald", "green", "red"), ("sapphire", "blue", "red"),
    ("moon", "white", "black"), ("flame", "orange", "blue"), ("forest", "green", "white"),
    ("ocean", "blue", "yellow"), ("sun", "yellow", "purple"),
    ("fire", "hot", "cold"), ("desert", "hot", "cold"), ("lava", "hot", "cold"),
    ("ice", "cold", "hot"), ("snow", "cold", "hot"),
    ("rain", "wet", "dry"), ("ocean", "wet", "dry"), ("river", "wet", "dry"),
    ("sand", "dry", "wet"), ("dust", "dry", "wet"), ("bone", "dry", "wet"),
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


def identify_channels_per_layer(model, tokenizer, device, model_name, W_U, binding_layers, ref_pairs, layers_obj, mlp_weights):
    """Identify Top 1% cproj, dact channels per layer."""
    channel_counts_cproj = defaultdict(lambda: defaultdict(int))
    channel_counts_dact = defaultdict(lambda: defaultdict(int))
    
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
            n_top1 = max(1, min_d // 100)
            
            for ch in np.argsort(np.abs(channel_proj))[-n_top1:]:
                channel_counts_cproj[li][int(ch)] += 1
            for ch in np.argsort(np.abs(dact))[-n_top1:]:
                channel_counts_dact[li][int(ch)] += 1
        
        del clean_caps, corrupt_caps
        gc.collect(); torch.cuda.empty_cache()
    
    n_ref = len(ref_pairs)
    top1_cproj = {}; top1_dact = {}
    
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
        
        log(f"  Layer {li}: cproj={len(top1_cproj[li])} dact={len(top1_dact[li])}")
    
    return top1_cproj, top1_dact


def run_experiment(model_name):
    log(f"Phase 355: Per-Layer Functional Stratification ({model_name})")
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

    # Channel identification
    log(f"\n  Part 1: Identifying Top 1% channels from 20 reference pairs...")
    ref_pairs = TEST_PAIRS[:20]
    top1_cproj, top1_dact = identify_channels_per_layer(
        model, tokenizer, device, model_name, W_U, binding_layers, ref_pairs, layers, mlp_weights)

    # ================================================================
    # Part 2: Single-Layer Patch Analysis
    # ================================================================
    n_test = len(TEST_PAIRS)
    log(f"\n  Part 2: Single-layer patch analysis on {n_test} pairs...")
    
    # For each pair: capture clean/corrupt acts, then do per-layer patch
    all_layer_results = {li: {"cproj_c2r": [], "cproj_r2c": [], "dact_c2r": [], "dact_r2c": []} 
                         for li in binding_layers}
    
    for pidx, (obj, target, competitor) in enumerate(TEST_PAIRS):
        tid_t = get_token_id(tokenizer, target)
        tid_c = get_token_id(tokenizer, competitor)
        if tid_t is None or tid_c is None: continue
        
        clean_prompt = f"The {obj}"
        
        # Capture acts for ALL binding layers simultaneously
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
        binding_base = clean_gap - corrupt_gap
        
        if abs(binding_base) < 1e-10:
            continue
        
        # Single-layer patch: patch only one layer at a time
        for li in binding_layers:
            for gname, channels in [("cproj", top1_cproj), ("dact", top1_dact)]:
                ch_set = channels.get(li, set())
                if not ch_set:
                    continue
                
                # C2R: patch only this layer
                patch_dict_c2r = {li: ch_set}
                c2r_logits = run_model_with_channel_patch(
                    model, tokenizer, device, clean_prompt, [li], patch_dict_c2r, corrupt_acts)
                c2r_target = float(c2r_logits[tid_t])
                c2r_compet = float(c2r_logits[tid_c])
                c2r_gap = c2r_target - c2r_compet
                
                all_layer_results[li][f"{gname}_c2r"].append({
                    "t_change": c2r_target - clean_target,
                    "c_change": c2r_compet - clean_compet,
                    "gap_change": c2r_gap - clean_gap,
                    "frac_gap": (c2r_gap - clean_gap) / binding_base,
                })
                
                # R2C: patch only this layer
                patch_dict_r2c = {li: ch_set}
                r2c_logits = run_model_with_channel_patch(
                    model, tokenizer, device, CORRUPTED_BASELINE, [li], patch_dict_r2c, clean_acts)
                r2c_target = float(r2c_logits[tid_t])
                r2c_compet = float(r2c_logits[tid_c])
                r2c_gap = r2c_target - r2c_compet
                
                all_layer_results[li][f"{gname}_r2c"].append({
                    "t_change": r2c_target - corrupt_target,
                    "c_change": r2c_compet - corrupt_compet,
                    "gap_change": r2c_gap - corrupt_gap,
                    "frac_gap": (r2c_gap - corrupt_gap) / binding_base,
                })
                
                gc.collect(); torch.cuda.empty_cache()
        
        if (pidx + 1) % 5 == 0:
            log(f"  [{pidx+1}/{n_test}] elapsed={time.time()-t0:.0f}s, "
                f"GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")
        
        gc.collect(); torch.cuda.empty_cache()

    # ================================================================
    # Part 3: Summary
    # ================================================================
    log(f"\n  ══════════════════════════════════════════════════════════════")
    log(f"  Phase 355 Summary: {model_name}")
    log(f"  ══════════════════════════════════════════════════════════════")
    
    # Per-layer gap contribution
    log(f"\n  --- Per-Layer cproj Gap Contribution ---")
    log(f"  {'Layer':>6} {'C2R:ΔT':>10} {'C2R:ΔC':>10} {'C2R:ΔGap':>10} {'C2R:FracGap':>12} {'R2C:ΔT':>10} {'R2C:ΔC':>10} {'R2C:ΔGap':>10} {'R2C:FracGap':>12} {'n':>4}")
    log(f"  {'-'*100}")
    
    layer_summary = {}
    for li in binding_layers:
        lr = all_layer_results[li]
        row = {}
        for gname in ["cproj", "dact"]:
            for direction in ["c2r", "r2c"]:
                key = f"{gname}_{direction}"
                vals = lr[key]
                n = len(vals)
                if n == 0:
                    row[key] = {"n": 0, "t": 0, "c": 0, "gap": 0, "frac": 0, "se": 0}
                    continue
                mean_t = float(np.mean([v["t_change"] for v in vals]))
                mean_c = float(np.mean([v["c_change"] for v in vals]))
                mean_gap = float(np.mean([v["gap_change"] for v in vals]))
                mean_frac = float(np.mean([v["frac_gap"] for v in vals]))
                se_frac = float(np.std([v["frac_gap"] for v in vals])/np.sqrt(n))
                row[key] = {"n": n, "t": mean_t, "c": mean_c, "gap": mean_gap, "frac": mean_frac, "se": se_frac}
        layer_summary[li] = row
        
        c2r = row["cproj_c2r"]; r2c = row["cproj_r2c"]
        if c2r["n"] > 0:
            log(f"  L{li:>4} {c2r['t']:>+10.4f} {c2r['c']:>+10.4f} {c2r['gap']:>+10.4f} {c2r['frac']:>+12.4f} "
                f"{r2c['t']:>+10.4f} {r2c['c']:>+10.4f} {r2c['gap']:>+10.4f} {r2c['frac']:>+12.4f} {c2r['n']:>4}")
    
    log(f"\n  --- Per-Layer dact Gap Contribution ---")
    log(f"  {'Layer':>6} {'C2R:ΔT':>10} {'C2R:ΔC':>10} {'C2R:ΔGap':>10} {'C2R:FracGap':>12} {'R2C:ΔT':>10} {'R2C:ΔC':>10} {'R2C:ΔGap':>10} {'R2C:FracGap':>12} {'n':>4}")
    log(f"  {'-'*100}")
    
    for li in binding_layers:
        lr = layer_summary[li]
        c2r = lr["dact_c2r"]; r2c = lr["dact_r2c"]
        if c2r["n"] > 0:
            log(f"  L{li:>4} {c2r['t']:>+10.4f} {c2r['c']:>+10.4f} {c2r['gap']:>+10.4f} {c2r['frac']:>+12.4f} "
                f"{r2c['t']:>+10.4f} {r2c['c']:>+10.4f} {r2c['gap']:>+10.4f} {r2c['frac']:>+12.4f} {c2r['n']:>4}")
    
    # Layer role classification
    log(f"\n  --- Layer Role Classification ---")
    log(f"  {'Layer':>6} {'cproj_C2R_frac':>16} {'cproj_R2C_frac':>16} {'dact_C2R_frac':>16} {'dact_R2C_frac':>16} {'cproj_role':>20} {'dact_role':>20}")
    log(f"  {'-'*120}")
    
    layer_roles = {}
    for li in binding_layers:
        lr = layer_summary[li]
        cproj_c2r = lr["cproj_c2r"]["frac"]
        cproj_r2c = lr["cproj_r2c"]["frac"]
        dact_c2r = lr["dact_c2r"]["frac"]
        dact_r2c = lr["dact_r2c"]["frac"]
        
        # Classify cproj role
        if cproj_c2r < -0.02 and cproj_r2c > 0.02:
            cproj_role = "gap_creator"
        elif cproj_c2r < -0.02:
            cproj_role = "gap_suppressor"
        elif cproj_r2c > 0.02:
            cproj_role = "gap_amplifier"
        elif abs(cproj_c2r) < 0.02 and abs(cproj_r2c) < 0.02:
            cproj_role = "neutral"
        else:
            cproj_role = "mixed"
        
        # Classify dact role
        if abs(dact_c2r) < 0.02 and abs(dact_r2c) < 0.02:
            dact_role = "pure_amplifier"
        elif dact_c2r < -0.02:
            dact_role = "gap_suppressor"
        elif dact_r2c > 0.02:
            dact_role = "gap_creator"
        else:
            dact_role = "mixed"
        
        log(f"  L{li:>4} {cproj_c2r:>+16.4f} {cproj_r2c:>+16.4f} {dact_c2r:>+16.4f} {dact_r2c:>+16.4f} {cproj_role:>20} {dact_role:>20}")
        
        layer_roles[li] = {"cproj_role": cproj_role, "dact_role": dact_role,
                           "cproj_c2r_frac": cproj_c2r, "cproj_r2c_frac": cproj_r2c,
                           "dact_c2r_frac": dact_c2r, "dact_r2c_frac": dact_r2c}
    
    # Save
    output = {
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "phase": "355",
        "layer_summary": {str(k): v for k, v in layer_summary.items()},
        "layer_roles": {str(k): v for k, v in layer_roles.items()},
        "n_pairs": n_test,
        "per_layer_results": {str(k): {dk: dv for dk, dv in v.items()} 
                              for k, v in all_layer_results.items()},
    }
    
    os.makedirs("results/phase355_per_layer_stratify", exist_ok=True)
    out_path = f"results/phase355_per_layer_stratify/{model_name}_phase355.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=str, ensure_ascii=False)
    log(f"\n  Saved to {out_path}")
    
    del model; gc.collect(); torch.cuda.empty_cache()
    log(f"Phase 355 complete for {model_name} in {time.time()-t0:.0f}s")
    return output


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_experiment(model_name)
