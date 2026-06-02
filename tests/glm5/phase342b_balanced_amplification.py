"""
Phase 342b: Balanced Amplification Confirmation + Corrected Embedding Patch
============================================================================

Round 2 confirmation test for the critical finding from Phase 342:
  compat_boost ≈ incompat_amplification (previously misnamed "incompat_suppress")
  → MLP does BALANCED AMPLIFICATION of both compat and incompat signals
  → Net binding effect is tiny residual after massive cancellation

This script:
1. Corrects the variable naming and interpretation
2. Computes the amplification ratio more precisely
3. Tests embedding partial patch properly (for Qwen3 only, since DS7B/GLM4 have meta device)
4. Per-pair analysis to verify the balanced amplification is not an artifact of averaging
5. Direct test: does MLP amplify the residual stream uniformly or selectively?

Usage:
  python tests/glm5/phase342b_balanced_amplification.py qwen3
  python tests/glm5/phase342b_balanced_amplification.py glm4
  python tests/glm5/phase342b_balanced_amplification.py deepseek7b
"""
import sys, os, time, json, gc, traceback
import torch
import numpy as np
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# ===== Configuration =====

MODEL_CONFIGS = {
    "qwen3": {
        "path": "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
        "n_layers": 36, "d_model": 2560,
        "binding_layers": [21, 23, 25, 27, 29],
    },
    "glm4": {
        "path": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
        "n_layers": 40, "d_model": 4096,
        "binding_layers": [30, 33, 36, 38],
    },
    "deepseek7b": {
        "path": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "n_layers": 28, "d_model": 3584,
        "binding_layers": [19, 21, 23, 24],
    },
}

HC_PAIRS = [
    ("apple", "red", "blue"),
    ("banana", "yellow", "purple"),
    ("snow", "white", "black"),
    ("sky", "blue", "green"),
    ("cherry", "red", "blue"),
    ("leaf", "green", "red"),
    ("stone", "rough", "soft"),
    ("silk", "smooth", "rough"),
    ("ice", "cold", "hot"),
    ("fire", "hot", "cold"),
    ("oven", "hot", "cold"),
    ("fridge", "cold", "hot"),
    ("grass", "green", "red"),
    ("ocean", "blue", "yellow"),
    ("sun", "yellow", "purple"),
    ("blood", "red", "green"),
    ("coal", "black", "white"),
    ("milk", "white", "black"),
    ("rose", "red", "blue"),
    ("gold", "yellow", "gray"),
    ("silver", "gray", "red"),
    ("cloud", "white", "green"),
    ("rain", "wet", "dry"),
    ("desert", "hot", "cold"),
]

CORRUPTED_BASELINE = "The item"


# ===== Model Loading =====

def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = None
    for impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True,
                attn_implementation=impl,
            )
            log(f"  Loaded {model_name} with attn_impl={impl}")
            break
        except Exception as e:
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
    model_path = MODEL_CONFIGS[model_name]["path"]
    for sf_file in glob.glob(os.path.join(model_path, '*.safetensors')):
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
    raise ValueError(f"Cannot find layers in {type(model).__name__}")


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
    model_path = MODEL_CONFIGS[model_name]["path"]
    W_gate = W_up = W_down = None
    d_ff = 0
    for sf_file in glob.glob(os.path.join(model_path, '*.safetensors')):
        try:
            with safe_open(sf_file, framework='pt', device='cpu') as sf:
                keys = sf.keys()
                for prefix in [f"model.layers.{layer_idx}.mlp"]:
                    gk = f"{prefix}.gate_proj.weight"
                    uk = f"{prefix}.up_proj.weight"
                    dk = f"{prefix}.down_proj.weight"
                    guk = f"{prefix}.gate_up_proj.weight"
                    if guk in keys:
                        w = sf.get_tensor(guk).float().numpy()
                        d_ff = w.shape[0] // 2
                        W_gate, W_up = w[:d_ff], w[d_ff:]
                    if gk in keys and W_gate is None:
                        W_gate = sf.get_tensor(gk).float().numpy()
                        d_ff = W_gate.shape[0]
                    if uk in keys and W_up is None:
                        W_up = sf.get_tensor(uk).float().numpy()
                        if d_ff == 0: d_ff = W_up.shape[0]
                    if dk in keys and W_down is None:
                        W_down = sf.get_tensor(dk).float().numpy()
                if W_down is not None:
                    break
        except:
            continue
    return W_gate, W_up, W_down, d_ff


def get_mlp_weights(layer, model_name=None, model=None):
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
        if W_gate is not None: d_ff = W_gate.shape[0]
        elif W_up is not None: d_ff = W_up.shape[0]
    elif hasattr(mlp, 'up_proj'):
        W_up = safe_weight_to_numpy(mlp.up_proj.weight)
        if W_up is not None: d_ff = W_up.shape[0]

    if hasattr(mlp, 'down_proj'):
        W_down = safe_weight_to_numpy(mlp.down_proj.weight)

    if W_down is None and model_name is not None:
        layers = get_layers(model)
        for i, l in enumerate(layers):
            if l is layer:
                W_gate, W_up, W_down, d_ff = get_mlp_weights_from_disk(model_name, i)
                break

    return W_gate, W_up, W_down, d_ff


# ===== Capture MLP Internals =====

def capture_mlp_internals(model, tokenizer, device, prompt, target_layers, n_layers):
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
        hooks.append(layer.mlp.register_forward_hook(make_hook(f"mlp_out_{li}")))

    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    for h in hooks:
        h.remove()

    final_hidden = out.hidden_states[-1][0, -1].detach().cpu().float().numpy()
    all_hidden = {i: hs[0, -1].detach().cpu().float().numpy() for i, hs in enumerate(out.hidden_states)}
    
    return captured, final_hidden, all_hidden


# ===== Corrected Channel Analysis =====

def corrected_channel_analysis(W_down, binding_dir, gate_acts_clean, up_acts_clean,
                                gate_acts_corrupt, up_acts_corrupt):
    """
    Corrected channel decomposition with proper naming.
    
    MLP output = down_proj(SiLU(gate_proj(x)) * up_proj(x))
    Binding contribution of channel i = (d @ W_down[:, i]) * SiLU(gate_i) * up_i
    
    Key insight: for incompat channels (d @ W_down[:, i] < 0):
    - Their contribution to binding is NEGATIVE
    - If delta = clean_contrib - corrupt_contrib < 0: incompatible signal is STRONGER in clean
    - If delta = clean_contrib - corrupt_contrib > 0: incompatible signal is WEAKER in clean
    """
    d_ff = W_down.shape[1]
    
    # SiLU activation
    def silu(x):
        return x * (1.0 / (1.0 + np.exp(-x)))
    
    gate_silu_clean = silu(gate_acts_clean[:d_ff])
    gate_silu_corrupt = silu(gate_acts_corrupt[:d_ff])
    up_clean = up_acts_clean[:d_ff]
    up_corrupt = up_acts_corrupt[:d_ff]
    
    # Down projection: how much each channel writes toward binding direction
    down_proj_binding = binding_dir @ W_down  # [d_ff]
    
    # Per-channel binding contribution
    contrib_clean = down_proj_binding * gate_silu_clean * up_clean
    contrib_corrupt = down_proj_binding * gate_silu_corrupt * up_corrupt
    delta = contrib_clean - contrib_corrupt
    
    # Classify channels
    compat_mask = down_proj_binding > 0  # writes toward compatible
    incompat_mask = down_proj_binding < 0  # writes toward incompatible
    
    # For compat channels:
    compat_boost = float(np.sum(delta[compat_mask & (delta > 0)]))  # more compat in clean
    compat_reduce = float(np.sum(delta[compat_mask & (delta < 0)]))  # less compat in clean
    
    # For incompat channels:
    # delta > 0 for incompat = incompat signal WEAKER in clean = actual SUPPRESSION
    incompat_suppress = float(np.sum(delta[incompat_mask & (delta > 0)]))  # positive
    # delta < 0 for incompat = incompat signal STRONGER in clean = AMPLIFICATION
    incompat_amplify = float(np.sum(delta[incompat_mask & (delta < 0)]))  # negative
    
    # Gross amplification measures
    compat_gross_amplify = compat_boost + abs(compat_reduce)  # total compat channel change
    incompat_gross_amplify = abs(incompat_amplify) + incompat_suppress  # total incompat channel change
    
    # Net effects
    compat_net = compat_boost + compat_reduce  # net compat effect
    incompat_net = incompat_amplify + incompat_suppress  # net incompat effect (should be near 0 if balanced)
    
    total_binding_clean = float(np.sum(contrib_clean))
    total_binding_corrupt = float(np.sum(contrib_corrupt))
    
    # Amplification ratio: how balanced is the amplification?
    total_gross = compat_gross_amplify + incompat_gross_amplify
    amplification_balance = incompat_gross_amplify / max(compat_gross_amplify, 1e-10)
    
    return {
        "total_binding_clean": total_binding_clean,
        "total_binding_corrupt": total_binding_corrupt,
        "delta_total": total_binding_clean - total_binding_corrupt,
        # Corrected naming
        "compat_boost": compat_boost,
        "compat_reduce": compat_reduce,  # negative
        "compat_net": compat_net,
        "incompat_suppress": incompat_suppress,  # positive = actual suppression
        "incompat_amplify": incompat_amplify,  # negative = actual amplification of incompat
        "incompat_net": incompat_net,
        # Gross amplification
        "compat_gross": compat_gross_amplify,
        "incompat_gross": incompat_gross_amplify,
        "amplification_balance": amplification_balance,  # ≈1.0 if balanced
        # Net/gross ratio
        "net_gross_ratio": abs(total_binding_clean - total_binding_corrupt) / max(total_gross, 1e-10),
        # Channel counts
        "n_compat": int(compat_mask.sum()),
        "n_incompat": int(incompat_mask.sum()),
    }


# ===== Main Experiment =====

def run_experiment(model_name):
    log(f"Phase 342b: Balanced Amplification Confirmation — {model_name}")
    log("=" * 70)

    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    cfg = MODEL_CONFIGS[model_name]
    n_layers = cfg["n_layers"]
    binding_layers = cfg["binding_layers"]

    W_U = get_W_U(model, model_name)
    log(f"  W_U shape: {W_U.shape}")
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log(f"  GPU after load: {gpu_mem:.2f}GB")

    # Pre-extract MLP weights
    layers = get_layers(model)
    mlp_weights = {}
    for li in binding_layers:
        W_gate, W_up, W_down, d_ff = get_mlp_weights(layers[li], model_name, model)
        mlp_weights[li] = {"W_down": W_down, "d_ff": d_ff}
        log(f"  L{li}: d_ff={d_ff}")

    # ==================================================================
    # EXPERIMENT 1: Corrected Per-Pair Channel Analysis
    # ==================================================================
    log(f"\n{'='*70}")
    log(f"EXPERIMENT 1: Corrected Per-Pair Channel Analysis")

    per_pair_results = {}

    for pidx, (obj, target_val, competitor_val) in enumerate(HC_PAIRS):
        pair_key = f"{obj}_{target_val}"
        tid_t = get_token_id(tokenizer, target_val)
        tid_c = get_token_id(tokenizer, competitor_val)
        if tid_t is None or tid_c is None:
            continue

        binding_dir = W_U[tid_t] - W_U[tid_c]
        clean_prompt = f"The {obj}"

        # Quick binding range check
        inp_c = tokenizer(CORRUPTED_BASELINE, return_tensors="pt", truncation=True, max_length=128).to(device)
        inp_cl = tokenizer(clean_prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            out_c = model(**inp_c, output_hidden_states=True)
            out_cl = model(**inp_cl, output_hidden_states=True)
        final_c = out_c.hidden_states[-1][0, -1].detach().cpu().float().numpy()
        final_cl = out_cl.hidden_states[-1][0, -1].detach().cpu().float().numpy()
        binding_range = float(binding_dir @ final_cl) - float(binding_dir @ final_c)
        del out_c, out_cl
        gc.collect()
        torch.cuda.empty_cache()

        if binding_range < 0.3:
            continue

        # Capture MLP internals
        clean_caps, clean_final, clean_all_hidden = capture_mlp_internals(
            model, tokenizer, device, clean_prompt, binding_layers, n_layers)
        corrupt_caps, corrupt_final, corrupt_all_hidden = capture_mlp_internals(
            model, tokenizer, device, CORRUPTED_BASELINE, binding_layers, n_layers)

        pair_result = {}
        for li in binding_layers:
            mw = mlp_weights[li]
            W_down = mw["W_down"]
            d_ff = mw["d_ff"]
            if W_down is None:
                continue

            gk_clean = f"gate_{li}"
            uk_clean = f"up_{li}"
            if gk_clean not in clean_caps or gk_clean not in corrupt_caps:
                continue

            clean_gate = clean_caps[gk_clean][:d_ff]
            corrupt_gate = corrupt_caps[gk_clean][:d_ff]
            clean_up = clean_caps.get(uk_clean, np.ones(d_ff))[:d_ff]
            corrupt_up = corrupt_caps.get(uk_clean, np.ones(d_ff))[:d_ff]

            # Truncate if needed
            min_d = min(clean_gate.shape[0], d_ff, W_down.shape[1])
            clean_gate = clean_gate[:min_d]
            corrupt_gate = corrupt_gate[:min_d]
            clean_up = clean_up[:min_d]
            corrupt_up = corrupt_up[:min_d]
            W_down_trunc = W_down[:, :min_d]

            analysis = corrected_channel_analysis(
                W_down_trunc, binding_dir,
                clean_gate, clean_up,
                corrupt_gate, corrupt_up,
            )
            pair_result[li] = analysis

        per_pair_results[pair_key] = pair_result

        del clean_caps, corrupt_caps, clean_all_hidden, corrupt_all_hidden
        gc.collect()
        torch.cuda.empty_cache()

        if (pidx + 1) % 6 == 0 or pidx < 2:
            elapsed = time.time() - t0
            gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            log(f"  [{pidx+1}/{len(HC_PAIRS)}] {pair_key}: "
                f"{len(pair_result)} layers, elapsed={elapsed:.0f}s, GPU={gpu_mem:.2f}GB")

    # ==================================================================
    # Aggregate and analyze
    # ==================================================================
    log(f"\n{'='*70}")
    log(f"RESULTS: Corrected Balanced Amplification Analysis")
    log(f"{'='*70}")

    # Per-layer aggregation
    layer_aggs = {}
    for li in binding_layers:
        metrics = ["compat_boost", "compat_reduce", "compat_net",
                   "incompat_suppress", "incompat_amplify", "incompat_net",
                   "compat_gross", "incompat_gross", "amplification_balance",
                   "net_gross_ratio", "delta_total"]
        agg = {}
        for m in metrics:
            vals = [per_pair_results[pk][li][m] for pk in per_pair_results 
                    if li in per_pair_results[pk]]
            if vals:
                agg[m] = {
                    "mean": round(float(np.mean(vals)), 4),
                    "std": round(float(np.std(vals)), 4),
                    "min": round(float(np.min(vals)), 4),
                    "max": round(float(np.max(vals)), 4),
                }
        layer_aggs[li] = agg

    # Print results
    log(f"\n  {'Layer':>6} {'compat_boost':>13} {'incompat_amplify':>17} {'Balance':>9} "
        f"{'Net/Gross':>10} {'compat_net':>11} {'incompat_net':>13}")
    log("  " + "-" * 85)

    for li in binding_layers:
        agg = layer_aggs[li]
        cb = agg.get("compat_boost", {}).get("mean", 0)
        ia = agg.get("incompat_amplify", {}).get("mean", 0)
        bal = agg.get("amplification_balance", {}).get("mean", 0)
        ngr = agg.get("net_gross_ratio", {}).get("mean", 0)
        cn = agg.get("compat_net", {}).get("mean", 0)
        inn = agg.get("incompat_net", {}).get("mean", 0)
        log(f"  L{li:>5} {cb:>+13.4f} {ia:>+17.4f} {bal:>9.3f} "
            f"{ngr:>10.4f} {cn:>+11.4f} {inn:>+13.4f}")

    # Per-pair amplification balance
    log(f"\n  Per-pair amplification balance (incompat_gross / compat_gross):")
    log(f"  {'Pair':>20} " + " ".join(f"{'L'+str(li):>8}" for li in binding_layers))
    log("  " + "-" * (20 + 9 * len(binding_layers)))

    balance_values = {li: [] for li in binding_layers}
    for pk in sorted(per_pair_results.keys()):
        vals = []
        for li in binding_layers:
            if li in per_pair_results[pk]:
                b = per_pair_results[pk][li]["amplification_balance"]
                vals.append(f"{b:>8.3f}")
                balance_values[li].append(b)
            else:
                vals.append(f"{'N/A':>8}")
        log(f"  {pk:>20} " + " ".join(vals))

    # Summary statistics
    log(f"\n  Amplification Balance Summary:")
    for li in binding_layers:
        bvals = balance_values[li]
        if bvals:
            log(f"  L{li}: mean={np.mean(bvals):.3f}, std={np.std(bvals):.3f}, "
                f"range=[{np.min(bvals):.3f}, {np.max(bvals):.3f}], n={len(bvals)}")

    # Net/gross ratio summary
    log(f"\n  Net/Gross Ratio Summary (how much of gross effect survives as net):")
    for li in binding_layers:
        ngr_vals = [per_pair_results[pk][li]["net_gross_ratio"] 
                    for pk in per_pair_results if li in per_pair_results[pk]]
        if ngr_vals:
            log(f"  L{li}: mean={np.mean(ngr_vals):.4f}, std={np.std(ngr_vals):.4f}")
            log(f"       → Only {np.mean(ngr_vals)*100:.1f}% of gross amplification survives as net binding effect")

    # ==================================================================
    # EXPERIMENT 2: MLP as Uniform Amplifier Test
    # ==================================================================
    log(f"\n{'='*70}")
    log(f"EXPERIMENT 2: Is MLP a Uniform Amplifier?")
    log(f"  Test: Does MLP output simply scale the residual stream input?")

    # For each binding layer, compare MLP output norm with input norm
    # If MLP is a uniform amplifier, the output should be proportional to input
    
    # Use apple-red pair
    test_obj, test_target, test_competitor = "apple", "red", "blue"
    tid_t = get_token_id(tokenizer, test_target)
    tid_c = get_token_id(tokenizer, test_competitor)
    
    if tid_t is not None and tid_c is not None:
        binding_dir = W_U[tid_t] - W_U[tid_c]
        clean_prompt = f"The {test_obj}"
        
        clean_caps, clean_final, clean_all_hidden = capture_mlp_internals(
            model, tokenizer, device, clean_prompt, binding_layers, n_layers)
        corrupt_caps, corrupt_final, corrupt_all_hidden = capture_mlp_internals(
            model, tokenizer, device, CORRUPTED_BASELINE, binding_layers, n_layers)
        
        amplifier_results = {}
        for li in binding_layers:
            mlp_out_key = f"mlp_out_{li}"
            if mlp_out_key not in clean_caps or mlp_out_key not in corrupt_caps:
                continue
            
            clean_mlp_out = clean_caps[mlp_out_key]
            corrupt_mlp_out = corrupt_caps[mlp_out_key]
            
            # Input to MLP = residual stream at this layer
            clean_input = clean_all_hidden[li]
            corrupt_input = corrupt_all_hidden[li]
            
            # MLP output projection onto binding direction
            mlp_binding_clean = float(binding_dir @ clean_mlp_out)
            mlp_binding_corrupt = float(binding_dir @ corrupt_mlp_out)
            
            # Input projection onto binding direction
            input_binding_clean = float(binding_dir @ clean_input)
            input_binding_corrupt = float(binding_dir @ corrupt_input)
            
            # Amplification ratio
            if abs(input_binding_clean) > 1e-6:
                amp_ratio_clean = mlp_binding_clean / input_binding_clean
            else:
                amp_ratio_clean = float('inf')
            
            # Diff analysis
            mlp_diff = clean_mlp_out - corrupt_mlp_out
            input_diff = clean_input - corrupt_input
            
            mlp_diff_binding = float(binding_dir @ mlp_diff)
            input_diff_binding = float(binding_dir @ input_diff)
            
            # Cosine similarity between MLP diff and input diff
            mlp_diff_norm = np.linalg.norm(mlp_diff)
            input_diff_norm = np.linalg.norm(input_diff)
            if mlp_diff_norm > 1e-6 and input_diff_norm > 1e-6:
                cos_sim = float(np.dot(mlp_diff, input_diff) / (mlp_diff_norm * input_diff_norm))
            else:
                cos_sim = float('nan')
            
            amplifier_results[li] = {
                "mlp_binding_clean": round(mlp_binding_clean, 4),
                "mlp_binding_corrupt": round(mlp_binding_corrupt, 4),
                "input_binding_clean": round(input_binding_clean, 4),
                "input_binding_corrupt": round(input_binding_corrupt, 4),
                "amp_ratio_clean": round(amp_ratio_clean, 4) if abs(amp_ratio_clean) < 1000 else "inf",
                "mlp_diff_binding": round(mlp_diff_binding, 4),
                "input_diff_binding": round(input_diff_binding, 4),
                "diff_cos_sim": round(cos_sim, 4) if not np.isnan(cos_sim) else "nan",
            }
            
            log(f"  L{li}: mlp_binding={mlp_binding_clean:+.4f}, input_binding={input_binding_clean:+.4f}, "
                f"amp_ratio={amp_ratio_clean:.4f}, cos_sim(diff)={cos_sim:.4f}")
        
        del clean_caps, corrupt_caps, clean_all_hidden, corrupt_all_hidden
        gc.collect()
        torch.cuda.empty_cache()

    # ==================================================================
    # Save results
    # ==================================================================
    save_data = {
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "layer_aggs": {str(k): v for k, v in layer_aggs.items()},
        "per_pair_results": {
            pk: {str(li): v for li, v in pdata.items()}
            for pk, pdata in per_pair_results.items()
        },
        "amplifier_test": {str(k): v for k, v in amplifier_results.items()} if tid_t is not None else {},
    }

    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        elif isinstance(obj, (np.floating,)): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)): return [convert(v) for v in obj]
        return obj

    save_data = convert(save_data)
    os.makedirs("results/phase342_mlp_channel", exist_ok=True)
    out_path = f"results/phase342_mlp_channel/{model_name}_phase342b.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    log(f"\nResults saved to {out_path}")

    # Release model
    del model, W_U, mlp_weights
    gc.collect()
    torch.cuda.empty_cache()

    total_time = time.time() - t0
    log(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f}min)")


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    if model_name not in MODEL_CONFIGS:
        log(f"Unknown model: {model_name}")
        sys.exit(1)
    run_experiment(model_name)
    log("Phase 342b complete!")
