"""
Phase 351b: Confirmation — Extended Reference Pairs + Contribution Ablation
===========================================================================

Confirms Phase 351 findings:
1. Uses 20 reference pairs (was 10) to identify Top 1% channels
2. Adds Top 1% |contribution| ablation group
3. Fixes GLM4 attribution calculation
4. Tests per-layer ablation (which layers contribute most)
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


def run_model_with_channel_ablation(model, tokenizer, device, prompt, target_layers,
                                     channels_to_zero_by_layer):
    layers = get_layers(model)
    hooks = []
    for li, channels_to_zero in channels_to_zero_by_layer.items():
        if li >= len(layers) or not channels_to_zero:
            continue
        mlp = layers[li].mlp
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
        hooks.append(mlp.down_proj.register_forward_pre_hook(make_pre_hook(ch_tensor)))
    
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=False)
    logits = out.logits[0, -1].float().cpu().numpy()
    for h in hooks:
        h.remove()
    return logits


def identify_channels(model, tokenizer, device, model_name, W_U, binding_layers, ref_pairs, layers_obj, mlp_weights):
    """Identify Top 1% channels from reference pairs."""
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
        
        clean_caps, _ = capture_mlp_internals(
            model, tokenizer, device, f"The {obj}", binding_layers)
        corrupt_caps, _ = capture_mlp_internals(
            model, tokenizer, device, CORRUPTED_BASELINE, binding_layers)
        
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
    
    # Select channels appearing in >= 30% of reference pairs
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


def run_ablation(model, tokenizer, device, model_name, W_U, obj, target, competitor,
                 binding_layers, ablation_channels_by_layer, mlp_weights):
    """Run causal ablation and return logit-level metrics."""
    tid_t = get_token_id(tokenizer, target)
    tid_c = get_token_id(tokenizer, competitor)
    if tid_t is None or tid_c is None:
        return None
    
    clean_prompt = f"The {obj}"
    
    # Baseline logits
    _, clean_logits = capture_mlp_internals(model, tokenizer, device, clean_prompt, binding_layers)
    _, corrupt_logits = capture_mlp_internals(model, tokenizer, device, CORRUPTED_BASELINE, binding_layers)
    
    clean_diff_base = float(clean_logits[tid_t] - clean_logits[tid_c])
    corrupt_diff_base = float(corrupt_logits[tid_t] - corrupt_logits[tid_c])
    binding_effect_base = clean_diff_base - corrupt_diff_base
    
    # Ablated logits
    if ablation_channels_by_layer:
        ablated_clean_logits = run_model_with_channel_ablation(
            model, tokenizer, device, clean_prompt, binding_layers, ablation_channels_by_layer)
        ablated_corrupt_logits = run_model_with_channel_ablation(
            model, tokenizer, device, CORRUPTED_BASELINE, binding_layers, ablation_channels_by_layer)
        clean_diff_abl = float(ablated_clean_logits[tid_t] - ablated_clean_logits[tid_c])
        corrupt_diff_abl = float(ablated_corrupt_logits[tid_t] - ablated_corrupt_logits[tid_c])
    else:
        clean_diff_abl = clean_diff_base
        corrupt_diff_abl = corrupt_diff_base
    
    binding_effect_abl = clean_diff_abl - corrupt_diff_abl
    
    # Also compute per-layer ablation effect
    per_layer_logit_effects = {}
    for li in binding_layers:
        single_layer_channels = {li: ablation_channels_by_layer.get(li, set())}
        if any(single_layer_channels[li]):
            abl_clean = run_model_with_channel_ablation(
                model, tokenizer, device, clean_prompt, binding_layers, single_layer_channels)
            abl_corrupt = run_model_with_channel_ablation(
                model, tokenizer, device, CORRUPTED_BASELINE, binding_layers, single_layer_channels)
            cd = float(abl_clean[tid_t] - abl_clean[tid_c])
            cdd = float(abl_corrupt[tid_t] - abl_corrupt[tid_c])
            per_layer_logit_effects[li] = cd - cdd
        else:
            per_layer_logit_effects[li] = binding_effect_base
    
    gc.collect(); torch.cuda.empty_cache()
    
    return {
        "binding_effect_base": binding_effect_base,
        "binding_effect_ablated": binding_effect_abl,
        "frac_lost": (binding_effect_base - binding_effect_abl) / max(abs(binding_effect_base), 1e-10),
        "per_layer_effects": per_layer_logit_effects,
    }


def run_experiment(model_name):
    log(f"Phase 351b: Confirmation — Extended Ref + Contribution Ablation ({model_name})")
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

    # Identify channels with 20 reference pairs
    log(f"\n  Identifying Top 1% channels from 20 reference pairs...")
    ref_pairs = TEST_PAIRS[:20]
    top1_cproj, top1_dact, top1_contrib, random_ch = identify_channels(
        model, tokenizer, device, model_name, W_U, binding_layers, ref_pairs, layers, mlp_weights)

    # Ablation groups
    ablation_groups = {
        "baseline": {},  # no ablation
        "top1_cproj": top1_cproj,
        "top1_dact": top1_dact,
        "top1_contrib": top1_contrib,
        "random": random_ch,
    }
    
    # Run ablation on all 30 pairs
    log(f"\n  Running ablation on {len(TEST_PAIRS)} pairs...")
    
    ablation_results = {gname: [] for gname in ablation_groups}
    
    for pidx, (obj, target, competitor) in enumerate(TEST_PAIRS):
        for gname, channels in ablation_groups.items():
            result = run_ablation(model, tokenizer, device, model_name, W_U,
                                  obj, target, competitor, binding_layers, channels, mlp_weights)
            if result is not None:
                ablation_results[gname].append(result)
        
        if (pidx + 1) % 10 == 0:
            log(f"  [{pidx+1}/{len(TEST_PAIRS)}] elapsed={time.time()-t0:.0f}s")
    
    # Summary
    log(f"\n  --- Ablation Summary ---")
    log(f"  {'Type':<15} {'BaseEff':>10} {'AblatedEff':>12} {'FracLost':>10} {'FracLost(SE)':>12}")
    log(f"  {'-'*62}")
    
    summary = {}
    for gname in ["baseline", "top1_cproj", "top1_dact", "top1_contrib", "random"]:
        results = ablation_results[gname]
        if not results:
            continue
        base_effs = [r["binding_effect_base"] for r in results]
        abl_effs = [r["binding_effect_ablated"] for r in results]
        frac_losts = [r["frac_lost"] for r in results]
        
        mean_base = float(np.mean(base_effs))
        mean_abl = float(np.mean(abl_effs))
        mean_frac = float(np.mean(frac_losts))
        se_frac = float(np.std(frac_losts) / np.sqrt(len(frac_losts)))
        
        summary[gname] = {
            "mean_base": mean_base,
            "mean_ablated": mean_abl,
            "mean_frac_lost": mean_frac,
            "se_frac_lost": se_frac,
            "n": len(results),
        }
        
        log(f"  {gname:<15} {mean_base:>+10.4f} {mean_abl:>+12.4f} {mean_frac:>+10.4f} {se_frac:>12.4f}")
    
    # Per-layer effect for top1_cproj
    log(f"\n  --- Per-Layer Ablation Effect (Top 1% cproj) ---")
    log(f"  {'Layer':>6} {'BaseEff':>10} {'AblatedEff':>12} {'FracLost':>10}")
    log(f"  {'-'*42}")
    
    per_layer_summary = {}
    for li in binding_layers:
        results = ablation_results["top1_cproj"]
        if not results:
            continue
        base_effs = [r["binding_effect_base"] for r in results]
        layer_effs = [r["per_layer_effects"].get(li, 0) for r in results]
        frac_losts = [(b - a) / max(abs(b), 1e-10) for b, a in zip(base_effs, layer_effs)]
        
        mean_frac = float(np.mean(frac_losts))
        per_layer_summary[li] = {"mean_frac_lost": mean_frac, "n": len(results)}
        log(f"  {li:>6} {np.mean(base_effs):>+10.4f} {np.mean(layer_effs):>+12.4f} {mean_frac:>+10.4f}")
    
    # Save
    all_results = {
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ablation_summary": summary,
        "per_layer_summary": {str(k): v for k, v in per_layer_summary.items()},
    }
    os.makedirs("results/phase351_top1_causal_ablation", exist_ok=True)
    out_path = f"results/phase351_top1_causal_ablation/{model_name}_phase351b.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    log(f"\n  Saved to {out_path}")
    
    del model; gc.collect(); torch.cuda.empty_cache()
    log(f"Phase 351b complete for {model_name} in {time.time()-t0:.0f}s")
    return all_results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_experiment(model_name)
