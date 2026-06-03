"""
Phase 353b: cproj Path Cross-Prompt Generalization + dact Quadrant Confirmation
==============================================================================

Confirms Phase 353 with:
1. Cross-prompt generalization: test cproj channels with varied prompts
2. dact quadrant confirmation with per-pair detailed analysis
3. Channel set cross-validation: use half pairs for selection, test on other half
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

# Standard test pairs
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

# Cross-prompt variants for generalization test
PROMPT_VARIANTS = [
    "The {obj}",          # Standard
    "A {obj}",            # Indefinite article
    "{obj}",              # Bare noun
    "The {obj} is",       # With copula
    "I see the {obj}",    # Different frame
]


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


def identify_channels_with_pairs(model, tokenizer, device, model_name, W_U, binding_layers, 
                                 ref_pairs, layers_obj, mlp_weights, threshold_frac=0.3):
    """Identify channels from a specific set of reference pairs."""
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
        corrupt_caps, _ = capture_mlp_internals(model, tokenizer, device, "The item", binding_layers)
        
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
            ch for ch, cnt in channel_counts_cproj[li].items() if cnt >= n_ref * threshold_frac)
        if not top1_cproj[li]:
            sorted_ch = sorted(channel_counts_cproj[li].items(), key=lambda x: -x[1])
            top1_cproj[li] = set(ch for ch, _ in sorted_ch[:n_top1])
        
        top1_dact[li] = set(
            ch for ch, cnt in channel_counts_dact[li].items() if cnt >= n_ref * threshold_frac)
        if not top1_dact[li]:
            sorted_ch = sorted(channel_counts_dact[li].items(), key=lambda x: -x[1])
            top1_dact[li] = set(ch for ch, _ in sorted_ch[:n_top1])
    
    return top1_cproj, top1_dact


def run_experiment(model_name):
    log(f"Phase 353b: Cross-Prompt Generalization + Channel CV ({model_name})")
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

    # ================================================================
    # Part 1: Cross-Prompt Generalization for cproj C2R
    # ================================================================
    log(f"\n  Part 1: Cross-Prompt Generalization (10 pairs × 5 prompts)...")
    
    # Use standard channel identification
    ref_pairs = TEST_PAIRS[:20]
    top1_cproj, top1_dact = identify_channels_with_pairs(
        model, tokenizer, device, model_name, W_U, binding_layers, 
        ref_pairs, layers, mlp_weights)
    
    for li in binding_layers:
        log(f"  Layer {li}: cproj={len(top1_cproj.get(li, set()))} dact={len(top1_dact.get(li, set()))}")
    
    cross_prompt_results = []
    test_pairs_cross = TEST_PAIRS[:10]  # Use first 10 pairs for cross-prompt
    
    for pidx, (obj, target, competitor) in enumerate(test_pairs_cross):
        tid_t = get_token_id(tokenizer, target)
        tid_c = get_token_id(tokenizer, competitor)
        if tid_t is None or tid_c is None: continue
        
        for prompt_template in PROMPT_VARIANTS:
            prompt = prompt_template.format(obj=obj)
            corrupt_prompt = "The item"
            
            # Capture clean and corrupt activations
            clean_acts, clean_logits = capture_down_proj_inputs(
                model, tokenizer, device, prompt, binding_layers)
            corrupt_acts, corrupt_logits = capture_down_proj_inputs(
                model, tokenizer, device, corrupt_prompt, binding_layers)
            
            clean_target = float(clean_logits[tid_t])
            clean_compet = float(clean_logits[tid_c])
            corrupt_target = float(corrupt_logits[tid_t])
            corrupt_compet = float(corrupt_logits[tid_c])
            
            clean_diff = clean_target - clean_compet
            corrupt_diff = corrupt_target - corrupt_compet
            binding_base = clean_diff - corrupt_diff
            
            if abs(binding_base) < 1e-10:
                continue
            
            # C2R for cproj
            c2r_logits = run_model_with_channel_patch(
                model, tokenizer, device, prompt, binding_layers, top1_cproj, corrupt_acts)
            c2r_target = float(c2r_logits[tid_t])
            c2r_compet = float(c2r_logits[tid_c])
            c2r_diff = c2r_target - c2r_compet
            c2r_binding = c2r_diff - corrupt_diff
            frac_lost = (binding_base - c2r_binding) / abs(binding_base)
            
            # R2C for cproj
            r2c_logits = run_model_with_channel_patch(
                model, tokenizer, device, corrupt_prompt, binding_layers, top1_cproj, clean_acts)
            r2c_target = float(r2c_logits[tid_t])
            r2c_compet = float(r2c_logits[tid_c])
            r2c_diff = r2c_target - r2c_compet
            frac_recovered = (r2c_diff - corrupt_diff) / abs(binding_base)
            
            cross_prompt_results.append({
                "pair": f"{obj}/{target}vs{competitor}",
                "prompt": prompt_template,
                "binding_base": binding_base,
                "c2r_frac_lost": frac_lost,
                "r2c_frac_recovered": frac_recovered,
                "c2r_target_change": c2r_target - clean_target,
                "c2r_compet_change": c2r_compet - clean_compet,
            })
            
            gc.collect(); torch.cuda.empty_cache()
        
        log(f"  Pair {pidx+1}/10 done ({obj})")
    
    # ================================================================
    # Part 2: Channel Set Cross-Validation
    # ================================================================
    log(f"\n  Part 2: Channel Set Cross-Validation (split halves)...")
    
    # Split pairs into two halves
    half1 = TEST_PAIRS[:15]
    half2 = TEST_PAIRS[15:]
    
    # Identify channels from each half
    log(f"  Identifying channels from half1 (pairs 1-15)...")
    cproj_half1, dact_half1 = identify_channels_with_pairs(
        model, tokenizer, device, model_name, W_U, binding_layers, 
        half1, layers, mlp_weights, threshold_frac=0.2)
    
    log(f"  Identifying channels from half2 (pairs 16-30)...")
    cproj_half2, dact_half2 = identify_channels_with_pairs(
        model, tokenizer, device, model_name, W_U, binding_layers,
        half2, layers, mlp_weights, threshold_frac=0.2)
    
    # Compute overlap
    for li in binding_layers:
        s1 = cproj_half1.get(li, set())
        s2 = cproj_half2.get(li, set())
        overlap = s1 & s2
        union = s1 | s2
        jaccard = len(overlap) / len(union) if union else 0
        log(f"  Layer {li} cproj: half1={len(s1)} half2={len(s2)} overlap={len(overlap)} Jaccard={jaccard:.3f}")
    
    # Test cproj_half1 on half2 pairs and vice versa
    cv_results = []
    for train_name, train_channels, test_pairs_cv in [
        ("half1→half2", cproj_half1, half2),
        ("half2→half1", cproj_half2, half1),
    ]:
        fracs_c2r = []
        fracs_r2c = []
        
        for obj, target, competitor in test_pairs_cv:
            tid_t = get_token_id(tokenizer, target)
            tid_c = get_token_id(tokenizer, competitor)
            if tid_t is None or tid_c is None: continue
            
            clean_prompt = f"The {obj}"
            corrupt_prompt = "The item"
            
            clean_acts, clean_logits = capture_down_proj_inputs(
                model, tokenizer, device, clean_prompt, binding_layers)
            corrupt_acts, corrupt_logits = capture_down_proj_inputs(
                model, tokenizer, device, corrupt_prompt, binding_layers)
            
            clean_target = float(clean_logits[tid_t])
            clean_compet = float(clean_logits[tid_c])
            corrupt_target = float(corrupt_logits[tid_t])
            corrupt_compet = float(corrupt_logits[tid_c])
            
            binding_base = (clean_target - clean_compet) - (corrupt_target - corrupt_compet)
            if abs(binding_base) < 1e-10: continue
            
            # C2R
            c2r_logits = run_model_with_channel_patch(
                model, tokenizer, device, clean_prompt, binding_layers, train_channels, corrupt_acts)
            c2r_diff = float(c2r_logits[tid_t] - c2r_logits[tid_c])
            c2r_binding = c2r_diff - (corrupt_target - corrupt_compet)
            fracs_c2r.append((binding_base - c2r_binding) / abs(binding_base))
            
            # R2C
            r2c_logits = run_model_with_channel_patch(
                model, tokenizer, device, corrupt_prompt, binding_layers, train_channels, clean_acts)
            r2c_diff = float(r2c_logits[tid_t] - r2c_logits[tid_c])
            fracs_r2c.append((r2c_diff - (corrupt_target - corrupt_compet)) / abs(binding_base))
            
            gc.collect(); torch.cuda.empty_cache()
        
        cv_results.append({
            "train": train_name,
            "c2r_mean": float(np.mean(fracs_c2r)) if fracs_c2r else 0,
            "c2r_se": float(np.std(fracs_c2r)/np.sqrt(len(fracs_c2r))) if fracs_c2r else 0,
            "r2c_mean": float(np.mean(fracs_r2c)) if fracs_r2c else 0,
            "r2c_se": float(np.std(fracs_r2c)/np.sqrt(len(fracs_r2c))) if fracs_r2c else 0,
            "n": len(fracs_c2r),
        })
        
        log(f"  {train_name}: C2R={cv_results[-1]['c2r_mean']:+.4f}±{cv_results[-1]['c2r_se']:.4f} "
            f"R2C={cv_results[-1]['r2c_mean']:+.4f}±{cv_results[-1]['r2c_se']:.4f} (n={cv_results[-1]['n']})")
    
    # ================================================================
    # Summary
    # ================================================================
    log(f"\n  ══════════════════════════════════════════════════════════════")
    log(f"  Phase 353b Summary: {model_name}")
    log(f"  ══════════════════════════════════════════════════════════════")
    
    # Cross-prompt summary
    log(f"\n  --- Cross-Prompt Generalization (cproj C2R/R2C) ---")
    log(f"  {'Prompt':<25} {'C2R_FracLost':>14} {'R2C_FracRec':>14} {'C2R/R2C':>10} {'n':>5}")
    log(f"  {'-'*70}")
    
    cross_prompt_summary = {}
    for pt in PROMPT_VARIANTS:
        pt_results = [r for r in cross_prompt_results if r["prompt"] == pt]
        if not pt_results: continue
        mean_c2r = float(np.mean([r["c2r_frac_lost"] for r in pt_results]))
        se_c2r = float(np.std([r["c2r_frac_lost"] for r in pt_results])/np.sqrt(len(pt_results)))
        mean_r2c = float(np.mean([r["r2c_frac_recovered"] for r in pt_results]))
        se_r2c = float(np.std([r["r2c_frac_recovered"] for r in pt_results])/np.sqrt(len(pt_results)))
        ratio = mean_r2c / mean_c2r if abs(mean_c2r) > 0.005 else float('inf')
        
        short_name = pt.replace("{obj}", "X")
        log(f"  {short_name:<25} {mean_c2r:>+14.4f} {mean_r2c:>+14.4f} {ratio:>10.2f} {len(pt_results):>5}")
        
        cross_prompt_summary[pt] = {
            "c2r_mean": mean_c2r, "c2r_se": se_c2r,
            "r2c_mean": mean_r2c, "r2c_se": se_r2c,
            "ratio": ratio, "n": len(pt_results),
        }
    
    # Cross-validation summary
    log(f"\n  --- Channel Set Cross-Validation ---")
    for cvr in cv_results:
        log(f"  {cvr['train']}: C2R={cvr['c2r_mean']:+.4f}±{cvr['c2r_se']:.4f} "
            f"R2C={cvr['r2c_mean']:+.4f}±{cvr['r2c_se']:.4f}")
    
    # Channel overlap
    channel_overlap = {}
    for li in binding_layers:
        s1 = cproj_half1.get(li, set())
        s2 = cproj_half2.get(li, set())
        overlap = s1 & s2
        union = s1 | s2
        jaccard = len(overlap) / len(union) if union else 0
        channel_overlap[str(li)] = {
            "half1": len(s1), "half2": len(s2),
            "overlap": len(overlap), "jaccard": jaccard,
        }
    
    # Save
    output = {
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "phase": "353b",
        "cross_prompt_summary": cross_prompt_summary,
        "cross_validation": cv_results,
        "channel_overlap": channel_overlap,
        "n_pairs": len(cross_prompt_results),
    }
    
    os.makedirs("results/phase353_dact_context", exist_ok=True)
    out_path = f"results/phase353_dact_context/{model_name}_phase353b.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=str)
    log(f"\n  Saved to {out_path}")
    
    del model; gc.collect(); torch.cuda.empty_cache()
    log(f"Phase 353b complete for {model_name} in {time.time()-t0:.0f}s")
    return output


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_experiment(model_name)
