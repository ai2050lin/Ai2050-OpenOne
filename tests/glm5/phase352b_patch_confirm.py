"""
Phase 352b: Patch Ablation Confirmation + Target/Competitor Decomposition
=========================================================================

Confirms Phase 352 findings with:
1. Per-layer C2R/R2C for cproj channels (most stable channel type)
2. Target logit vs Competitor logit decomposition for each intervention
3. Per-pair dact analysis to understand anti-binding pattern
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
    # Convert keys
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


def run_model_with_channel_zero(model, tokenizer, device, prompt, target_layers,
                                 channels_to_zero_by_layer):
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
    for h in hooks: h.remove()
    return logits


def identify_channels(model, tokenizer, device, model_name, W_U, binding_layers, ref_pairs, layers_obj, mlp_weights):
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
    top1_cproj_channels = {}
    top1_dact_channels = {}
    
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
        
        log(f"  Layer {li}: cproj={len(top1_cproj_channels[li])} dact={len(top1_dact_channels[li])}")
    
    return top1_cproj_channels, top1_dact_channels


def run_experiment(model_name):
    log(f"Phase 352b: Patch Confirmation + Target/Competitor Decomposition ({model_name})")
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
    top1_cproj, top1_dact = identify_channels(
        model, tokenizer, device, model_name, W_U, binding_layers, ref_pairs, layers, mlp_weights)

    channel_groups = {
        "top1_cproj": top1_cproj,
        "top1_dact": top1_dact,
    }
    
    # Part 2: Target/Competitor decomposition for all pairs
    n_test = len(TEST_PAIRS)
    log(f"\n  Part 2: Target/Competitor decomposition on {n_test} pairs...")
    
    all_results = []
    for pidx, (obj, target, competitor) in enumerate(TEST_PAIRS):
        tid_t = get_token_id(tokenizer, target)
        tid_c = get_token_id(tokenizer, competitor)
        if tid_t is None or tid_c is None:
            continue
        
        clean_prompt = f"The {obj}"
        
        # Capture baselines
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
            "clean_target": clean_target, "clean_compet": clean_compet,
            "corrupt_target": corrupt_target, "corrupt_compet": corrupt_compet,
            "binding_base": binding_base,
        }
        
        # For each channel group, run C2R and R2C + decomposition
        for gname, channels in channel_groups.items():
            if not any(channels.get(li, set()) for li in binding_layers):
                continue
            
            # C2R: clean prompt, replace channels with corrupt acts
            c2r_logits = run_model_with_channel_patch(
                model, tokenizer, device, clean_prompt, binding_layers, channels, corrupt_acts)
            c2r_target = float(c2r_logits[tid_t])
            c2r_compet = float(c2r_logits[tid_c])
            c2r_diff = c2r_target - c2r_compet
            c2r_binding = c2r_diff - corrupt_diff
            c2r_target_change = c2r_target - clean_target  # positive = target boosted
            c2r_compet_change = c2r_compet - clean_compet  # positive = competitor boosted
            
            # R2C: corrupt prompt, replace channels with clean acts
            r2c_logits = run_model_with_channel_patch(
                model, tokenizer, device, CORRUPTED_BASELINE, binding_layers, channels, clean_acts)
            r2c_target = float(r2c_logits[tid_t])
            r2c_compet = float(r2c_logits[tid_c])
            r2c_diff = r2c_target - r2c_compet
            r2c_target_change = r2c_target - corrupt_target
            r2c_compet_change = r2c_compet - corrupt_compet
            
            pair_result[f"{gname}_c2r"] = {
                "target_change": c2r_target_change,
                "compet_change": c2r_compet_change,
                "frac_lost": (binding_base - c2r_binding) / abs(binding_base),
                "target_frac": c2r_target_change / abs(binding_base),
                "compet_frac": c2r_compet_change / abs(binding_base),
            }
            pair_result[f"{gname}_r2c"] = {
                "target_change": r2c_target_change,
                "compet_change": r2c_compet_change,
                "frac_recovered": (r2c_diff - corrupt_diff) / abs(binding_base),
                "target_frac": r2c_target_change / abs(binding_base),
                "compet_frac": r2c_compet_change / abs(binding_base),
            }
            
            gc.collect(); torch.cuda.empty_cache()
        
        all_results.append(pair_result)
        
        if (pidx + 1) % 10 == 0:
            log(f"  [{pidx+1}/{n_test}] elapsed={time.time()-t0:.0f}s, "
                f"GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")
    
    # Part 3: Per-layer C2R/R2C for cproj (first 10 pairs only for efficiency)
    log(f"\n  Part 3: Per-layer C2R/R2C for cproj (10 pairs)...")
    per_layer_results = {li: {"c2r": [], "r2c": []} for li in binding_layers}
    
    for pidx, (obj, target, competitor) in enumerate(TEST_PAIRS[:10]):
        tid_t = get_token_id(tokenizer, target)
        tid_c = get_token_id(tokenizer, competitor)
        if tid_t is None or tid_c is None: continue
        
        clean_prompt = f"The {obj}"
        clean_acts, clean_logits = capture_down_proj_inputs(
            model, tokenizer, device, clean_prompt, binding_layers)
        corrupt_acts, corrupt_logits = capture_down_proj_inputs(
            model, tokenizer, device, CORRUPTED_BASELINE, binding_layers)
        
        clean_diff = float(clean_logits[tid_t] - clean_logits[tid_c])
        corrupt_diff = float(corrupt_logits[tid_t] - corrupt_logits[tid_c])
        binding_base = clean_diff - corrupt_diff
        if abs(binding_base) < 1e-10: continue
        
        for li in binding_layers:
            single_layer_channels = {li: top1_cproj.get(li, set())}
            if not single_layer_channels[li]:
                continue
            
            # Per-layer C2R
            c2r_logits = run_model_with_channel_patch(
                model, tokenizer, device, clean_prompt, binding_layers,
                single_layer_channels, corrupt_acts)
            c2r_diff = float(c2r_logits[tid_t] - c2r_logits[tid_c])
            c2r_binding = c2r_diff - corrupt_diff
            frac_lost = (binding_base - c2r_binding) / abs(binding_base)
            per_layer_results[li]["c2r"].append(frac_lost)
            
            # Per-layer R2C
            r2c_logits = run_model_with_channel_patch(
                model, tokenizer, device, CORRUPTED_BASELINE, binding_layers,
                single_layer_channels, clean_acts)
            r2c_diff = float(r2c_logits[tid_t] - r2c_logits[tid_c])
            frac_rec = (r2c_diff - corrupt_diff) / abs(binding_base)
            per_layer_results[li]["r2c"].append(frac_rec)
            
            gc.collect(); torch.cuda.empty_cache()
        
        log(f"  Per-layer pair {pidx+1}/10 done")
    
    # Part 4: Summary
    log(f"\n  ══════════════════════════════════════════════════════════════")
    log(f"  Phase 352b Summary: {model_name}")
    log(f"  ══════════════════════════════════════════════════════════════")
    
    # C2R/R2C with target/competitor decomposition
    log(f"\n  --- C2R (clean→corrupt): Target vs Competitor ---")
    log(f"  {'Channel':<15} {'FracLost':>10} {'TargetΔ':>10} {'CompetΔ':>10} {'Target%':>8} {'Compet%':>8}")
    log(f"  {'-'*65}")
    
    decomp_summary = {}
    for gname in ["top1_cproj", "top1_dact"]:
        c2r_key = f"{gname}_c2r"
        r2c_key = f"{gname}_r2c"
        
        c2r_fracs = [r[c2r_key]["frac_lost"] for r in all_results if c2r_key in r]
        c2r_target_fracs = [r[c2r_key]["target_frac"] for r in all_results if c2r_key in r]
        c2r_compet_fracs = [r[c2r_key]["compet_frac"] for r in all_results if c2r_key in r]
        
        r2c_fracs = [r[r2c_key]["frac_recovered"] for r in all_results if r2c_key in r]
        r2c_target_fracs = [r[r2c_key]["target_frac"] for r in all_results if r2c_key in r]
        r2c_compet_fracs = [r[r2c_key]["compet_frac"] for r in all_results if r2c_key in r]
        
        mean_c2r = float(np.mean(c2r_fracs)) if c2r_fracs else 0
        mean_c2r_t = float(np.mean(c2r_target_fracs)) if c2r_target_fracs else 0
        mean_c2r_c = float(np.mean(c2r_compet_fracs)) if c2r_compet_fracs else 0
        se_c2r = float(np.std(c2r_fracs)/np.sqrt(len(c2r_fracs))) if c2r_fracs else 0
        
        mean_r2c = float(np.mean(r2c_fracs)) if r2c_fracs else 0
        mean_r2c_t = float(np.mean(r2c_target_fracs)) if r2c_target_fracs else 0
        mean_r2c_c = float(np.mean(r2c_compet_fracs)) if r2c_compet_fracs else 0
        se_r2c = float(np.std(r2c_fracs)/np.sqrt(len(r2c_fracs))) if r2c_fracs else 0
        
        # Target% and Compet% of C2R effect
        total_c2r_abs = abs(mean_c2r_t) + abs(mean_c2r_c)
        target_pct = abs(mean_c2r_t) / total_c2r_abs * 100 if total_c2r_abs > 0.001 else 0
        compet_pct = abs(mean_c2r_c) / total_c2r_abs * 100 if total_c2r_abs > 0.001 else 0
        
        log(f"  {gname+'_C2R':<15} {mean_c2r:>+10.4f} {mean_c2r_t:>+10.4f} {mean_c2r_c:>+10.4f} "
            f"{target_pct:>7.1f}% {compet_pct:>7.1f}%")
        
        decomp_summary[gname] = {
            "c2r_frac_lost": mean_c2r, "c2r_se": se_c2r,
            "c2r_target": mean_c2r_t, "c2r_compet": mean_c2r_c,
            "c2r_target_pct": target_pct, "c2r_compet_pct": compet_pct,
            "r2c_frac_recovered": mean_r2c, "r2c_se": se_r2c,
            "r2c_target": mean_r2c_t, "r2c_compet": mean_r2c_c,
        }
    
    log(f"\n  --- R2C (corrupt→clean): Target vs Competitor ---")
    log(f"  {'Channel':<15} {'FracRec':>10} {'TargetΔ':>10} {'CompetΔ':>10}")
    log(f"  {'-'*50}")
    for gname in ["top1_cproj", "top1_dact"]:
        ds = decomp_summary[gname]
        log(f"  {gname+'_R2C':<15} {ds['r2c_frac_recovered']:>+10.4f} "
            f"{ds['r2c_target']:>+10.4f} {ds['r2c_compet']:>+10.4f}")
    
    # Per-layer summary
    log(f"\n  --- Per-Layer C2R/R2C (cproj) ---")
    log(f"  {'Layer':>6} {'C2R_FracLost':>14} {'R2C_FracRec':>14} {'C2R/R2C':>10}")
    log(f"  {'-'*48}")
    
    per_layer_summary = {}
    for li in binding_layers:
        c2r_vals = per_layer_results[li]["c2r"]
        r2c_vals = per_layer_results[li]["r2c"]
        if not c2r_vals: continue
        mean_c2r = float(np.mean(c2r_vals))
        mean_r2c = float(np.mean(r2c_vals))
        ratio = mean_r2c / mean_c2r if abs(mean_c2r) > 0.005 else float('inf')
        
        per_layer_summary[str(li)] = {
            "c2r_frac_lost": mean_c2r,
            "r2c_frac_recovered": mean_r2c,
            "ratio": ratio,
        }
        log(f"  {li:>6} {mean_c2r:>+14.4f} {mean_r2c:>+14.4f} {ratio:>10.2f}")
    
    # Interpretation
    log(f"\n  --- Key Interpretation ---")
    for gname in ["top1_cproj", "top1_dact"]:
        ds = decomp_summary[gname]
        target_pct = ds["c2r_target_pct"]
        if target_pct > 70:
            interp = f"{gname}: TARGET BOOST dominant ({target_pct:.0f}%)"
        elif target_pct < 30:
            interp = f"{gname}: COMPETITOR SUPPRESS dominant ({100-target_pct:.0f}%)"
        else:
            interp = f"{gname}: Mixed target boost + competitor suppress"
        log(f"  {interp}")
        
        c2r = ds["c2r_frac_lost"]
        r2c = ds["r2c_frac_recovered"]
        if abs(c2r) > 0.01 and abs(r2c) > 0.01:
            if abs(abs(r2c) - abs(c2r)) / max(abs(c2r), 0.01) < 0.3:
                log(f"  → C2R ≈ R2C: Symmetric — clean-corrupt diff IS causal signal")
            elif abs(r2c) > abs(c2r):
                log(f"  → R2C > C2R: Rescue > Destroy → redundancy in binding")
            else:
                log(f"  → C2R > R2C: Destroy > Rescue → context dependency")
    
    # Save
    output = {
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "phase": "352b",
        "decomp_summary": {k: v for k, v in decomp_summary.items()},
        "per_layer_summary": per_layer_summary,
        "n_pairs": len(all_results),
    }
    
    os.makedirs("results/phase352_patch_ablation", exist_ok=True)
    out_path = f"results/phase352_patch_ablation/{model_name}_phase352b.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=str)
    log(f"\n  Saved to {out_path}")
    
    del model; gc.collect(); torch.cuda.empty_cache()
    log(f"Phase 352b complete for {model_name} in {time.time()-t0:.0f}s")
    return output


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_experiment(model_name)
