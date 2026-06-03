"""
Phase 354: dact Gap Contribution + cproj Functional Stratification
==================================================================

Goals:
1. Test "amplifier" hypothesis: dact gap change ≈ 0?
2. Per-quadrant gap decomposition: what happens to binding gap in each quadrant?
3. cproj functional stratification: which layers/channels are pro-binding vs anti-binding?
4. Extended pair set with attribute-type stratification

Key predictions:
- If dact is "amplifier": gap_change ≈ 0 (both target and competitor change equally)
- If dact is "selector": gap_change > 0 (target changes more than competitor)
- cproj should show: gap_change < 0 for C2R, gap_change > 0 for R2C

Test pairs expanded to 50 with stratification by:
- color attributes (red, blue, green, yellow, white, black, orange, gray, purple)
- temperature attributes (hot, cold)
- texture attributes (wet, dry)
- other attributes (dark, bright)
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

# Expanded test pairs with attribute type stratification
TEST_PAIRS = [
    # Color attributes (30)
    ("apple", "red", "blue"), ("banana", "yellow", "purple"), ("snow", "white", "black"),
    ("sky", "blue", "green"), ("cherry", "red", "blue"), ("leaf", "green", "red"),
    ("rose", "red", "blue"), ("gold", "yellow", "purple"), ("coal", "black", "white"),
    ("silver", "white", "black"), ("milk", "white", "black"), ("honey", "yellow", "blue"),
    ("ruby", "red", "green"), ("emerald", "green", "red"), ("sapphire", "blue", "red"),
    ("moon", "white", "black"), ("flame", "orange", "blue"), ("forest", "green", "white"),
    ("ivory", "white", "black"), ("ocean", "blue", "yellow"), ("sun", "yellow", "purple"),
    ("blood", "red", "green"), ("smoke", "gray", "red"), ("steel", "gray", "gold"),
    ("grass", "green", "red"), ("night", "dark", "bright"), ("ice", "white", "black"),
    ("copper", "orange", "blue"), ("violet", "purple", "yellow"), ("coral", "red", "green"),
    # Temperature attributes (10)
    ("fire", "hot", "cold"), ("desert", "hot", "cold"), ("lava", "hot", "cold"),
    ("oven", "hot", "cold"), ("volcano", "hot", "cold"),
    ("ice", "cold", "hot"), ("snow", "cold", "hot"), ("glacier", "cold", "hot"),
    ("freezer", "cold", "hot"), ("arctic", "cold", "hot"),
    # Texture/state attributes (10)
    ("rain", "wet", "dry"), ("ocean", "wet", "dry"), ("river", "wet", "dry"),
    ("lake", "wet", "dry"), ("tear", "wet", "dry"),
    ("sand", "dry", "wet"), ("dust", "dry", "wet"), ("bone", "dry", "wet"),
    ("cracker", "dry", "wet"), ("paper", "dry", "wet"),
]

ATTR_TYPE = {}
for obj, t, c in TEST_PAIRS[:30]:
    ATTR_TYPE[(obj, t, c)] = "color"
for obj, t, c in TEST_PAIRS[30:40]:
    ATTR_TYPE[(obj, t, c)] = "temperature"
for obj, t, c in TEST_PAIRS[40:50]:
    ATTR_TYPE[(obj, t, c)] = "texture"

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
    """Identify Top 1% cproj, dact, and contribution channels with per-layer detail."""
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


def classify_quadrant(target_change, competitor_change):
    """Classify intervention effect into four quadrants."""
    t_up = target_change > 0
    c_down = competitor_change < 0
    
    if t_up and c_down:
        return "A_pro_binding"
    elif t_up and not c_down:
        return "B_shared_boost"
    elif not t_up and c_down:
        return "C_shared_suppress"
    else:
        return "D_anti_binding"


def run_experiment(model_name):
    log(f"Phase 354: dact Gap + cproj Stratification ({model_name})")
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
    log(f"\n  Part 1: Identifying Top 1% channels from 30 reference pairs...")
    ref_pairs = TEST_PAIRS[:30]
    top1_cproj, top1_dact = identify_channels(
        model, tokenizer, device, model_name, W_U, binding_layers, ref_pairs, layers, mlp_weights)

    # ================================================================
    # Part 2: Full Four-Quadrant + Gap Decomposition
    # ================================================================
    n_test = len(TEST_PAIRS)
    log(f"\n  Part 2: Gap decomposition on {n_test} pairs...")
    
    all_pair_results = []
    
    for pidx, (obj, target, competitor) in enumerate(TEST_PAIRS):
        tid_t = get_token_id(tokenizer, target)
        tid_c = get_token_id(tokenizer, competitor)
        if tid_t is None or tid_c is None: continue
        
        clean_prompt = f"The {obj}"
        attr_type = ATTR_TYPE.get((obj, target, competitor), "unknown")
        
        # Baselines
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
        
        pair_result = {
            "pair": f"{obj}/{target}vs{competitor}",
            "attr_type": attr_type,
            "binding_base": binding_base,
            "clean_gap": clean_gap,
            "corrupt_gap": corrupt_gap,
            "clean_target": clean_target,
            "clean_competitor": clean_compet,
            "corrupt_target": corrupt_target,
            "corrupt_competitor": corrupt_compet,
        }
        
        # === Test cproj and dact: C2R and R2C with full gap decomposition ===
        for gname, channels in [("cproj", top1_cproj), ("dact", top1_dact)]:
            if not any(channels.get(li, set()) for li in binding_layers):
                continue
            
            # C2R: clean prompt, patch channels with corrupt values
            c2r_logits = run_model_with_channel_patch(
                model, tokenizer, device, clean_prompt, binding_layers, channels, corrupt_acts)
            c2r_target = float(c2r_logits[tid_t])
            c2r_compet = float(c2r_logits[tid_c])
            c2r_gap = c2r_target - c2r_compet
            
            t_change = c2r_target - clean_target
            c_change = c2r_compet - clean_compet
            gap_change = c2r_gap - clean_gap
            quadrant = classify_quadrant(t_change, c_change)
            
            # Fractional gap change relative to binding_base
            frac_gap_change = gap_change / binding_base if abs(binding_base) > 1e-10 else 0
            
            pair_result[f"{gname}_c2r"] = {
                "target_change": t_change,
                "compet_change": c_change,
                "gap_change": gap_change,
                "frac_gap_change": frac_gap_change,
                "quadrant": quadrant,
                "post_gap": c2r_gap,
            }
            
            # R2C: corrupt prompt, patch channels with clean values
            r2c_logits = run_model_with_channel_patch(
                model, tokenizer, device, CORRUPTED_BASELINE, binding_layers, channels, clean_acts)
            r2c_target = float(r2c_logits[tid_t])
            r2c_compet = float(r2c_logits[tid_c])
            r2c_gap = r2c_target - r2c_compet
            
            t_change_r2c = r2c_target - corrupt_target
            c_change_r2c = r2c_compet - corrupt_compet
            gap_change_r2c = r2c_gap - corrupt_gap
            quadrant_r2c = classify_quadrant(t_change_r2c, c_change_r2c)
            
            # Fractional gap recovery relative to binding_base
            frac_gap_recovery = gap_change_r2c / binding_base if abs(binding_base) > 1e-10 else 0
            
            pair_result[f"{gname}_r2c"] = {
                "target_change": t_change_r2c,
                "compet_change": c_change_r2c,
                "gap_change": gap_change_r2c,
                "frac_gap_recovery": frac_gap_recovery,
                "quadrant": quadrant_r2c,
                "post_gap": r2c_gap,
            }
            
            gc.collect(); torch.cuda.empty_cache()
        
        # === dact+cproj combined R2C ===
        dact_plus_cproj = {}
        for li in binding_layers:
            dact_plus_cproj[li] = top1_dact.get(li, set()) | top1_cproj.get(li, set())
        
        if any(dact_plus_cproj.get(li, set()) for li in binding_layers):
            r2c_logits = run_model_with_channel_patch(
                model, tokenizer, device, CORRUPTED_BASELINE, binding_layers, dact_plus_cproj, clean_acts)
            r2c_target = float(r2c_logits[tid_t])
            r2c_compet = float(r2c_logits[tid_c])
            r2c_gap = r2c_target - r2c_compet
            
            t_change_r2c = r2c_target - corrupt_target
            c_change_r2c = r2c_compet - corrupt_compet
            gap_change_r2c = r2c_gap - corrupt_gap
            quadrant_r2c = classify_quadrant(t_change_r2c, c_change_r2c)
            frac_gap_recovery = gap_change_r2c / binding_base if abs(binding_base) > 1e-10 else 0
            
            pair_result["dact_cproj_r2c"] = {
                "target_change": t_change_r2c,
                "compet_change": c_change_r2c,
                "gap_change": gap_change_r2c,
                "frac_gap_recovery": frac_gap_recovery,
                "quadrant": quadrant_r2c,
                "post_gap": r2c_gap,
            }
            gc.collect(); torch.cuda.empty_cache()
        
        all_pair_results.append(pair_result)
        
        if (pidx + 1) % 10 == 0:
            log(f"  [{pidx+1}/{n_test}] elapsed={time.time()-t0:.0f}s, "
                f"GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")
        
        gc.collect(); torch.cuda.empty_cache()

    # ================================================================
    # Part 3: Summary Statistics
    # ================================================================
    log(f"\n  ══════════════════════════════════════════════════════════════")
    log(f"  Phase 354 Summary: {model_name}")
    log(f"  ══════════════════════════════════════════════════════════════")
    
    # === 3a: Gap decomposition per channel type ===
    log(f"\n  --- Gap Decomposition: dact vs cproj ---")
    log(f"  {'Channel':<15} {'C2R: ΔT':>10} {'C2R: ΔC':>10} {'C2R: ΔGap':>10} {'C2R: FracGap':>12} "
        f"{'R2C: ΔT':>10} {'R2C: ΔC':>10} {'R2C: ΔGap':>10} {'R2C: FracGap':>12}")
    log(f"  {'-'*110}")
    
    gap_summary = {}
    for gname in ["cproj", "dact"]:
        c2r_key = f"{gname}_c2r"
        r2c_key = f"{gname}_r2c"
        
        c2r_results = [r[c2r_key] for r in all_pair_results if c2r_key in r]
        r2c_results = [r[r2c_key] for r in all_pair_results if r2c_key in r]
        
        if not c2r_results and not r2c_results:
            continue
        
        mean_c2r_tc = float(np.mean([r["target_change"] for r in c2r_results])) if c2r_results else 0
        mean_c2r_cc = float(np.mean([r["compet_change"] for r in c2r_results])) if c2r_results else 0
        mean_c2r_gc = float(np.mean([r["gap_change"] for r in c2r_results])) if c2r_results else 0
        mean_c2r_fgc = float(np.mean([r["frac_gap_change"] for r in c2r_results])) if c2r_results else 0
        se_c2r_fgc = float(np.std([r["frac_gap_change"] for r in c2r_results])/np.sqrt(len(c2r_results))) if c2r_results else 0
        
        mean_r2c_tc = float(np.mean([r["target_change"] for r in r2c_results])) if r2c_results else 0
        mean_r2c_cc = float(np.mean([r["compet_change"] for r in r2c_results])) if r2c_results else 0
        mean_r2c_gc = float(np.mean([r["gap_change"] for r in r2c_results])) if r2c_results else 0
        mean_r2c_fgc = float(np.mean([r["frac_gap_recovery"] for r in r2c_results])) if r2c_results else 0
        se_r2c_fgc = float(np.std([r["frac_gap_recovery"] for r in r2c_results])/np.sqrt(len(r2c_results))) if r2c_results else 0
        
        log(f"  {gname:<15} {mean_c2r_tc:>+10.4f} {mean_c2r_cc:>+10.4f} {mean_c2r_gc:>+10.4f} {mean_c2r_fgc:>+12.4f} "
            f"{mean_r2c_tc:>+10.4f} {mean_r2c_cc:>+10.4f} {mean_r2c_gc:>+10.4f} {mean_r2c_fgc:>+12.4f}")
        log(f"  {'SE':>15} {'':>10} {'':>10} {'':>10} {se_c2r_fgc:>+12.4f} "
            f"{'':>10} {'':>10} {'':>10} {se_r2c_fgc:>+12.4f}")
        
        gap_summary[gname] = {
            "c2r_target": mean_c2r_tc, "c2r_compet": mean_c2r_cc,
            "c2r_gap": mean_c2r_gc, "c2r_frac_gap": mean_c2r_fgc, "c2r_se": se_c2r_fgc,
            "r2c_target": mean_r2c_tc, "r2c_compet": mean_r2c_cc,
            "r2c_gap": mean_r2c_gc, "r2c_frac_gap": mean_r2c_fgc, "r2c_se": se_r2c_fgc,
        }
    
    # dact+cproj combined
    dc_r2c_results = [r["dact_cproj_r2c"] for r in all_pair_results if "dact_cproj_r2c" in r]
    if dc_r2c_results:
        mean_r2c_tc = float(np.mean([r["target_change"] for r in dc_r2c_results]))
        mean_r2c_cc = float(np.mean([r["compet_change"] for r in dc_r2c_results]))
        mean_r2c_gc = float(np.mean([r["gap_change"] for r in dc_r2c_results]))
        mean_r2c_fgc = float(np.mean([r["frac_gap_recovery"] for r in dc_r2c_results]))
        se_r2c_fgc = float(np.std([r["frac_gap_recovery"] for r in dc_r2c_results])/np.sqrt(len(dc_r2c_results)))
        
        log(f"  {'dact+cproj':<15} {'—':>10} {'—':>10} {'—':>10} {'—':>12} "
            f"{mean_r2c_tc:>+10.4f} {mean_r2c_cc:>+10.4f} {mean_r2c_gc:>+10.4f} {mean_r2c_fgc:>+12.4f}")
        
        gap_summary["dact_cproj"] = {
            "r2c_target": mean_r2c_tc, "r2c_compet": mean_r2c_cc,
            "r2c_gap": mean_r2c_gc, "r2c_frac_gap": mean_r2c_fgc, "r2c_se": se_r2c_fgc,
        }
    
    # === 3b: Per-quadrant gap analysis ===
    log(f"\n  --- Per-Quadrant Gap Analysis ---")
    log(f"  {'Channel':<15} {'Quadrant':<20} {'n':>4} {'mean_ΔT':>10} {'mean_ΔC':>10} {'mean_ΔGap':>10} {'FracGap':>10}")
    log(f"  {'-'*85}")
    
    quadrant_gap_summary = {}
    for gname in ["cproj", "dact"]:
        for direction in ["c2r", "r2c"]:
            key = f"{gname}_{direction}"
            results = [r[key] for r in all_pair_results if key in r]
            if not results: continue
            
            for qname in ["A_pro_binding", "B_shared_boost", "C_shared_suppress", "D_anti_binding"]:
                q_results = [r for r in results if r["quadrant"] == qname]
                n_q = len(q_results)
                if n_q == 0: continue
                
                mean_tc = float(np.mean([r["target_change"] for r in q_results]))
                mean_cc = float(np.mean([r["compet_change"] for r in q_results]))
                mean_gc = float(np.mean([r["gap_change"] for r in q_results]))
                
                if direction == "c2r":
                    mean_fg = float(np.mean([r["frac_gap_change"] for r in q_results]))
                else:
                    mean_fg = float(np.mean([r["frac_gap_recovery"] for r in q_results]))
                
                log(f"  {gname:<15} {qname:<20} {n_q:>4} {mean_tc:>+10.4f} {mean_cc:>+10.4f} {mean_gc:>+10.4f} {mean_fg:>+10.4f}")
                
                qk = f"{gname}_{direction}_{qname}"
                quadrant_gap_summary[qk] = {
                    "n": n_q, "mean_tc": mean_tc, "mean_cc": mean_cc,
                    "mean_gap": mean_gc, "frac_gap": mean_fg,
                }
    
    # === 3c: Amplifier test — is dact gap_change ≈ 0? ===
    log(f"\n  --- Amplifier Test: Is dact gap_change ≈ 0? ---")
    for direction in ["c2r", "r2c"]:
        key = f"dact_{direction}"
        results = [r[key] for r in all_pair_results if key in r]
        if not results: continue
        
        if direction == "c2r":
            gap_changes = [r["gap_change"] for r in results]
            frac_gaps = [r["frac_gap_change"] for r in results]
        else:
            gap_changes = [r["gap_change"] for r in results]
            frac_gaps = [r["frac_gap_recovery"] for r in results]
        
        mean_gc = float(np.mean(gap_changes))
        se_gc = float(np.std(gap_changes)/np.sqrt(len(gap_changes)))
        mean_fg = float(np.mean(frac_gaps))
        se_fg = float(np.std(frac_gaps)/np.sqrt(len(frac_gaps)))
        
        # T-test: is gap_change significantly different from 0?
        t_stat = mean_gc / (se_gc + 1e-10)
        
        # Also compare dact gap_change magnitude to cproj gap_change magnitude
        cproj_key = f"cproj_{direction}"
        cproj_results = [r[cproj_key] for r in all_pair_results if cproj_key in r]
        if cproj_results:
            if direction == "c2r":
                cproj_gaps = [r["gap_change"] for r in cproj_results]
                cproj_fracs = [r["frac_gap_change"] for r in cproj_results]
            else:
                cproj_gaps = [r["gap_change"] for r in cproj_results]
                cproj_fracs = [r["frac_gap_recovery"] for r in cproj_results]
            cproj_mean_fg = float(np.mean(cproj_fracs))
        else:
            cproj_mean_fg = 0
        
        # Ratio: |dact gap| / |cproj gap|
        gap_ratio = abs(mean_fg) / (abs(cproj_mean_fg) + 1e-10)
        
        amplifier_verdict = "YES (amplifier)" if abs(mean_fg) < 0.02 else "NO (selector)"
        if abs(mean_fg) < 0.05:
            amplifier_verdict = "WEAK (partial amplifier)"
        
        log(f"  dact {direction.upper()}: mean_gap={mean_gc:+.4f} ± {se_gc:.4f}, "
            f"frac_gap={mean_fg:+.4f} ± {se_fg:.4f}, t={t_stat:.2f}")
        log(f"  cproj {direction.upper()}: frac_gap={cproj_mean_fg:+.4f}")
        log(f"  gap_ratio |dact|/|cproj| = {gap_ratio:.3f}")
        log(f"  → Amplifier test: {amplifier_verdict}")
    
    # === 3d: Attribute type stratification ===
    log(f"\n  --- Attribute Type Stratification ---")
    log(f"  {'AttrType':<15} {'n':>4} {'cproj_FracGapC2R':>18} {'cproj_FracGapR2C':>18} {'dact_FracGapC2R':>18} {'dact_FracGapR2C':>18}")
    log(f"  {'-'*95}")
    
    attr_summary = {}
    for attr_type in ["color", "temperature", "texture"]:
        type_results = [r for r in all_pair_results if r.get("attr_type") == attr_type]
        n_t = len(type_results)
        if n_t == 0: continue
        
        row = {"n": n_t}
        for gname in ["cproj", "dact"]:
            for direction in ["c2r", "r2c"]:
                key = f"{gname}_{direction}"
                vals = [r[key] for r in type_results if key in r]
                if not vals:
                    row[f"{gname}_{direction}"] = {"mean": 0, "se": 0, "n": 0}
                    continue
                if direction == "c2r":
                    fracs = [r["frac_gap_change"] for r in vals]
                else:
                    fracs = [r["frac_gap_recovery"] for r in vals]
                mean_f = float(np.mean(fracs))
                se_f = float(np.std(fracs)/np.sqrt(len(fracs)))
                row[f"{gname}_{direction}"] = {"mean": mean_f, "se": se_f, "n": len(fracs)}
        
        attr_summary[attr_type] = row
        
        log(f"  {attr_type:<15} {n_t:>4} "
            f"{row['cproj_c2r']['mean']:>+18.4f} {row['cproj_r2c']['mean']:>+18.4f} "
            f"{row['dact_c2r']['mean']:>+18.4f} {row['dact_r2c']['mean']:>+18.4f}")
    
    # === 3e: Quadrant distribution (C2R and R2C) ===
    log(f"\n  --- Quadrant Distribution ---")
    for direction in ["C2R", "R2C"]:
        log(f"\n  {direction}:")
        log(f"  {'Channel':<15} {'A:pro_bind':>12} {'B:shared↑':>12} {'C:shared↓':>12} {'D:anti_bind':>12} {'n':>4}")
        log(f"  {'-'*65}")
        
        dir_key = direction.lower()
        for gname in ["cproj", "dact"]:
            key = f"{gname}_{dir_key}"
            quadrants = [r[key]["quadrant"] for r in all_pair_results if key in r]
            n = len(quadrants)
            if n == 0: continue
            qa = sum(1 for q in quadrants if q == "A_pro_binding") / n * 100
            qb = sum(1 for q in quadrants if q == "B_shared_boost") / n * 100
            qc = sum(1 for q in quadrants if q == "C_shared_suppress") / n * 100
            qd = sum(1 for q in quadrants if q == "D_anti_binding") / n * 100
            log(f"  {gname:<15} {qa:>11.1f}% {qb:>11.1f}% {qc:>11.1f}% {qd:>11.1f}% {n:>4}")
    
    # === 3f: Per-pair dact R2C sign ===
    dact_r2c_fracs = []
    dact_r2c_positive = 0
    for r in all_pair_results:
        if "dact_r2c" not in r: continue
        fg = r["dact_r2c"]["frac_gap_recovery"]
        dact_r2c_fracs.append(fg)
        if fg > 0: dact_r2c_positive += 1
    
    n_r2c = len(dact_r2c_fracs)
    if n_r2c > 0:
        log(f"\n  --- dact R2C Sign Distribution ---")
        log(f"  Positive (pro-binding gap recovery): {dact_r2c_positive}/{n_r2c} ({dact_r2c_positive/n_r2c*100:.1f}%)")
        log(f"  Mean frac_gap_recovery: {np.mean(dact_r2c_fracs):+.4f} ± {np.std(dact_r2c_fracs)/np.sqrt(n_r2c):.4f}")
    
    # dact+cproj R2C
    dc_r2c_fracs = []
    for r in all_pair_results:
        if "dact_cproj_r2c" not in r: continue
        dc_r2c_fracs.append(r["dact_cproj_r2c"]["frac_gap_recovery"])
    
    if dc_r2c_fracs:
        dc_pos = sum(1 for f in dc_r2c_fracs if f > 0)
        log(f"  dact+cproj R2C positive: {dc_pos}/{len(dc_r2c_fracs)} ({dc_pos/len(dc_r2c_fracs)*100:.1f}%)")
        log(f"  dact+cproj mean frac: {np.mean(dc_r2c_fracs):+.4f} ± {np.std(dc_r2c_fracs)/np.sqrt(len(dc_r2c_fracs)):.4f}")
        log(f"  → Adding cproj {'improves' if np.mean(dc_r2c_fracs) > np.mean(dact_r2c_fracs) else 'worsens'} dact R2C gap recovery")

    # Save
    output = {
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "phase": "354",
        "gap_decomposition": gap_summary,
        "quadrant_gap": quadrant_gap_summary,
        "attr_stratification": attr_summary,
        "dact_r2c_sign": {
            "positive": dact_r2c_positive, "negative": n_r2c - dact_r2c_positive,
            "mean_frac": float(np.mean(dact_r2c_fracs)) if dact_r2c_fracs else 0,
            "se_frac": float(np.std(dact_r2c_fracs)/np.sqrt(len(dact_r2c_fracs))) if dact_r2c_fracs else 0,
        },
        "n_pairs": len(all_pair_results),
        "per_pair": all_pair_results,
    }
    
    os.makedirs("results/phase354_dact_gap_cproj_stratify", exist_ok=True)
    out_path = f"results/phase354_dact_gap_cproj_stratify/{model_name}_phase354.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=str, ensure_ascii=False)
    log(f"\n  Saved to {out_path}")
    
    del model; gc.collect(); torch.cuda.empty_cache()
    log(f"Phase 354 complete for {model_name} in {time.time()-t0:.0f}s")
    return output


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_experiment(model_name)
