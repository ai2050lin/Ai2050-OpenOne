"""
Phase 348b: Confirmation — Kurtosis vs Net/Gross + Specialized Channel Analysis
================================================================================

Phase 348 key finding: trained W_down has higher kurtosis than init,
especially for W_U token directions and semantic directions.

This confirmation test verifies:
1. Does higher kurtosis predict higher net/gross ratio?
2. Are the "specialized channels" (large projection) the main source of binding signal?
3. Do specialized channels have systematic activation patterns?

Usage:
  python tests/glm5/phase348b_kurtosis_netgross.py qwen3
  python tests/glm5/phase348b_kurtosis_netgross.py deepseek7b
  python tests/glm5/phase348b_kurtosis_netgross.py glm4
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


def get_mlp_weights(layer, model_name=None, model=None):
    mlp = layer.mlp
    W_gate = W_up = W_down = None; d_ff = 0
    if hasattr(mlp, 'gate_up_proj'):
        w = safe_weight_to_numpy(mlp.gate_up_proj.weight)
        if w is not None: d_ff = w.shape[0] // 2; W_gate, W_up = w[:d_ff], w[d_ff:]
    elif hasattr(mlp, 'gate_proj'):
        W_gate = safe_weight_to_numpy(mlp.gate_proj.weight)
        W_up = safe_weight_to_numpy(mlp.up_proj.weight)
        if W_gate is not None: d_ff = W_gate.shape[0]
        elif W_up is not None: d_ff = W_up.shape[0]
    elif hasattr(mlp, 'up_proj'):
        W_up = safe_weight_to_numpy(mlp.up_proj.weight)
        if W_up is not None: d_ff = W_up.shape[0]
    if hasattr(mlp, 'down_proj'): W_down = safe_weight_to_numpy(mlp.down_proj.weight)
    # Fallback to disk if weights are on meta device
    if W_down is None and model_name is not None:
        layers = get_layers(model)
        for i, l in enumerate(layers):
            if l is layer:
                W_gate, W_up, W_down, d_ff = get_mlp_weights_from_disk(model_name, i)
                break
    return W_gate, W_up, W_down, d_ff


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
                if dk in keys and W_down is None: W_down = sf.get_tensor(dk).float().numpy()
                if W_down is not None: break
        except: continue
    return W_gate, W_up, W_down, d_ff


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
                    captured[f"gate_{idx}"] = v[:d]; captured[f"up_{idx}"] = v[d:]
                return hook
            hooks.append(layer.mlp.gate_up_proj.register_forward_hook(make_glm4_hook(li)))
        if hasattr(layer.mlp, 'up_proj'):
            hooks.append(layer.mlp.up_proj.register_forward_hook(make_hook(f"up_{li}")))
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        model(**inp, output_hidden_states=True)
    for h in hooks: h.remove()
    return captured


def silu_np(x):
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -20, 20))))


def run_experiment(model_name):
    log(f"Phase 348b: Kurtosis vs Net/Gross + Specialized Channel — {model_name}")
    log("=" * 70)
    t0 = time.time()
    cfg = MODEL_CONFIGS[model_name]
    binding_layers = cfg["binding_layers"]
    n_layers = cfg["n_layers"]
    
    model, tokenizer, device = load_model_bf16(model_name)
    W_U = get_W_U(model, model_name)
    d_model = W_U.shape[1]
    log(f"  W_U shape: {W_U.shape}")
    
    # Pre-extract MLP weights for binding layers
    layers = get_layers(model)
    mlp_weights = {}
    for li in binding_layers:
        W_gate, W_up, W_down, d_ff = get_mlp_weights(layers[li], model_name, model)
        mlp_weights[li] = {"W_gate": W_gate, "W_up": W_up, "W_down": W_down, "d_ff": d_ff}
    
    # ======================================================================
    # Part A: Kurtosis vs Net/Gross across semantic directions
    # ======================================================================
    log(f"\n{'='*70}")
    log(f"PART A: Kurtosis vs Net/Gross")
    log(f"{'='*70}")
    
    kurtosis_vs_netgross = []
    
    for pidx, (obj, target, competitor) in enumerate(TEST_PAIRS):
        tid_t = get_token_id(tokenizer, target)
        tid_c = get_token_id(tokenizer, competitor)
        if tid_t is None or tid_c is None: continue
        
        direction = W_U[tid_t] - W_U[tid_c]
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-10: continue
        direction_normed = direction / dir_norm
        
        clean_prompt = f"The {obj}"
        clean_caps = capture_mlp_internals(model, tokenizer, device, clean_prompt, binding_layers, n_layers)
        corrupt_caps = capture_mlp_internals(model, tokenizer, device, CORRUPTED_BASELINE, binding_layers, n_layers)
        
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
            
            # Channel projections
            channel_proj = Wd.T @ direction_normed  # (min_d,)
            
            # Compute kurtosis
            cp_mean = np.mean(channel_proj)
            cp_std = np.std(channel_proj)
            if cp_std > 1e-10:
                kurtosis = float(np.mean(((channel_proj - cp_mean) / cp_std)**4) - 3.0)
            else:
                kurtosis = 0.0
            
            # Compute net/gross
            gsc = silu_np(cg[:min_d]); gsr = silu_np(crg[:min_d])
            uc = cu[:min_d]; ur = cru[:min_d]
            
            activation_diff = gsc * uc - gsr * ur
            per_channel_contrib = channel_proj * activation_diff
            
            gross = float(np.sum(np.abs(per_channel_contrib)))
            net = float(np.sum(per_channel_contrib))
            net_gross = net / max(gross, 1e-10)
            
            # Also compute: contribution from top-k vs bottom-k channels
            abs_proj = np.abs(channel_proj)
            sorted_indices = np.argsort(abs_proj)[::-1]
            
            top_k = min(100, min_d // 10)  # top 10% channels
            top_indices = sorted_indices[:top_k]
            bottom_indices = sorted_indices[-top_k:]
            
            top_contrib = float(np.sum(per_channel_contrib[top_indices]))
            bottom_contrib = float(np.sum(per_channel_contrib[bottom_indices]))
            top_gross = float(np.sum(np.abs(per_channel_contrib[top_indices])))
            top_net_gross = top_contrib / max(top_gross, 1e-10)
            
            # Top channel fraction of gross
            top_gross_fraction = float(np.sum(np.abs(per_channel_contrib[top_indices]))) / max(gross, 1e-10)
            
            kurtosis_vs_netgross.append({
                "pair": f"{obj}-{target}-{competitor}",
                "layer": li,
                "kurtosis": kurtosis,
                "net_gross": net_gross,
                "gross": gross,
                "net": net,
                "top_net_gross": top_net_gross,
                "top_gross_fraction": top_gross_fraction,
                "top_k": top_k,
            })
        
        del clean_caps, corrupt_caps; gc.collect(); torch.cuda.empty_cache()
        if (pidx + 1) % 6 == 0:
            log(f"  [{pidx+1}/{len(TEST_PAIRS)}] elapsed={time.time()-t0:.0f}s")
    
    # Analysis
    log(f"\n  PART A Summary:")
    if kurtosis_vs_netgross:
        kurtoses = [x["kurtosis"] for x in kurtosis_vs_netgross]
        net_grosses = [x["net_gross"] for x in kurtosis_vs_netgross]
        top_fractions = [x["top_gross_fraction"] for x in kurtosis_vs_netgross]
        
        log(f"  Kurtosis: mean={np.mean(kurtoses):.4f}, range=[{np.min(kurtoses):.4f}, {np.max(kurtoses):.4f}]")
        log(f"  Net/Gross: mean={np.mean(net_grosses):.4f}, range=[{np.min(net_grosses):.4f}, {np.max(net_grosses):.4f}]")
        log(f"  Top-10% gross fraction: mean={np.mean(top_fractions):.4f}")
        
        # Correlation between kurtosis and net/gross
        if len(kurtoses) > 3:
            corr = np.corrcoef(kurtoses, net_grosses)[0, 1]
            log(f"  Correlation(kurtosis, net/gross): {corr:.4f}")
            
            # Also correlate top_fraction with net/gross
            corr2 = np.corrcoef(top_fractions, net_grosses)[0, 1]
            log(f"  Correlation(top_fraction, net/gross): {corr2:.4f}")
    
    # ======================================================================
    # Part B: Specialized Channel Analysis
    # ======================================================================
    log(f"\n{'='*70}")
    log(f"PART B: Specialized Channel Analysis")
    log(f"{'='*70}")
    
    specialized_results = []
    
    for pidx, (obj, target, competitor) in enumerate(TEST_PAIRS[:8]):
        tid_t = get_token_id(tokenizer, target)
        tid_c = get_token_id(tokenizer, competitor)
        if tid_t is None or tid_c is None: continue
        
        direction = W_U[tid_t] - W_U[tid_c]
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-10: continue
        direction_normed = direction / dir_norm
        
        clean_prompt = f"The {obj}"
        clean_caps = capture_mlp_internals(model, tokenizer, device, clean_prompt, binding_layers, n_layers)
        corrupt_caps = capture_mlp_internals(model, tokenizer, device, CORRUPTED_BASELINE, binding_layers, n_layers)
        
        for li in binding_layers[:2]:  # Only first 2 binding layers for detailed analysis
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
            
            # Channel projections
            channel_proj = Wd.T @ direction_normed
            
            # Activations
            gsc = silu_np(cg[:min_d]); gsr = silu_np(crg[:min_d])
            uc = cu[:min_d]; ur = cru[:min_d]
            activation_diff = gsc * uc - gsr * ur
            per_channel_contrib = channel_proj * activation_diff
            
            # Define specialized channels: top 1% by |channel_proj|
            abs_proj = np.abs(channel_proj)
            top1pct_k = max(1, min_d // 100)
            top1pct_indices = np.argsort(abs_proj)[-top1pct_k:]
            
            # Top 10%
            top10pct_k = max(1, min_d // 10)
            top10pct_indices = np.argsort(abs_proj)[-top10pct_k:]
            
            # Remaining 90%
            remaining_indices = np.argsort(abs_proj)[:min_d - top10pct_k]
            
            # Contribution from each group
            top1pct_gross = float(np.sum(np.abs(per_channel_contrib[top1pct_indices])))
            top10pct_gross = float(np.sum(np.abs(per_channel_contrib[top10pct_indices])))
            remaining_gross = float(np.sum(np.abs(per_channel_contrib[remaining_indices])))
            total_gross = top10pct_gross + remaining_gross
            
            top1pct_net = float(np.sum(per_channel_contrib[top1pct_indices]))
            top10pct_net = float(np.sum(per_channel_contrib[top10pct_indices]))
            remaining_net = float(np.sum(per_channel_contrib[remaining_indices]))
            total_net = top10pct_net + remaining_net
            
            # Net/gross for each group
            top1pct_ng = top1pct_net / max(top1pct_gross, 1e-10)
            top10pct_ng = top10pct_net / max(top10pct_gross, 1e-10)
            remaining_ng = remaining_net / max(remaining_gross, 1e-10)
            total_ng = total_net / max(total_gross, 1e-10)
            
            specialized_results.append({
                "pair": f"{obj}-{target}-{competitor}",
                "layer": li,
                "top1pct_gross_frac": top1pct_gross / max(total_gross, 1e-10),
                "top10pct_gross_frac": top10pct_gross / max(total_gross, 1e-10),
                "top1pct_net_frac": top1pct_net / max(abs(total_net), 1e-10),
                "top10pct_net_frac": top10pct_net / max(abs(total_net), 1e-10),
                "top1pct_ng": top1pct_ng,
                "top10pct_ng": top10pct_ng,
                "remaining_ng": remaining_ng,
                "total_ng": total_ng,
                "top1pct_k": top1pct_k,
                "top10pct_k": top10pct_k,
            })
        
        del clean_caps, corrupt_caps; gc.collect(); torch.cuda.empty_cache()
    
    # Summary
    log(f"\n  PART B Summary:")
    if specialized_results:
        t1_gf = [x["top1pct_gross_frac"] for x in specialized_results]
        t10_gf = [x["top10pct_gross_frac"] for x in specialized_results]
        t1_nf = [x["top1pct_net_frac"] for x in specialized_results]
        t10_nf = [x["top10pct_net_frac"] for x in specialized_results]
        t1_ng = [x["top1pct_ng"] for x in specialized_results]
        t10_ng = [x["top10pct_ng"] for x in specialized_results]
        rem_ng = [x["remaining_ng"] for x in specialized_results]
        tot_ng = [x["total_ng"] for x in specialized_results]
        
        log(f"  Top 1% channels: gross_frac={np.mean(t1_gf):.4f}, net_frac={np.mean(t1_nf):.4f}, net/gross={np.mean(t1_ng):.4f}")
        log(f"  Top 10% channels: gross_frac={np.mean(t10_gf):.4f}, net_frac={np.mean(t10_nf):.4f}, net/gross={np.mean(t10_ng):.4f}")
        log(f"  Remaining 90%: net/gross={np.mean(rem_ng):.4f}")
        log(f"  Total: net/gross={np.mean(tot_ng):.4f}")
        
        log(f"  → Top 1% channels contribute {np.mean(t1_gf)*100:.1f}% of gross signal")
        log(f"  → Top 10% channels contribute {np.mean(t10_gf)*100:.1f}% of gross signal")
        log(f"  → Top 10% net/gross={np.mean(t10_ng):.4f} vs Remaining net/gross={np.mean(rem_ng):.4f}")
    
    # ======================================================================
    # Part C: Random vs Semantic Direction Comparison
    # ======================================================================
    log(f"\n{'='*70}")
    log(f"PART C: Random vs Semantic Direction — Kurtosis & Net/Gross")
    log(f"{'='*70}")
    
    random_kurtoses = []
    semantic_kurtoses = []
    wu_kurtoses = []
    li = binding_layers[len(binding_layers) // 2]  # middle binding layer
    mw = mlp_weights[li]
    W_down = mw["W_down"]; d_ff = mw["d_ff"]
    if W_down is not None:
        min_d = min(d_ff, W_down.shape[1])
        Wd = W_down[:, :min_d]
        
        # Generate random directions
        rng = np.random.RandomState(42)
        n_random = 50
        random_dirs = rng.randn(n_random, d_model)
        random_dirs = random_dirs / np.linalg.norm(random_dirs, axis=1, keepdims=True)
        
        # Semantic directions
        semantic_dirs = []
        for obj, target, competitor in TEST_PAIRS:
            tid_t = get_token_id(tokenizer, target)
            tid_c = get_token_id(tokenizer, competitor)
            if tid_t is None or tid_c is None: continue
            d = W_U[tid_t] - W_U[tid_c]
            norm = np.linalg.norm(d)
            if norm > 1e-10:
                semantic_dirs.append(d / norm)
        
        # Compute kurtosis for each direction
        random_kurtoses = []
        for d in random_dirs:
            cp = Wd.T @ d
            cp_std = np.std(cp)
            if cp_std > 1e-10:
                k = float(np.mean(((cp - np.mean(cp)) / cp_std)**4) - 3.0)
            else:
                k = 0.0
            random_kurtoses.append(k)
        
        semantic_kurtoses = []
        for d in semantic_dirs:
            cp = Wd.T @ d
            cp_std = np.std(cp)
            if cp_std > 1e-10:
                k = float(np.mean(((cp - np.mean(cp)) / cp_std)**4) - 3.0)
            else:
                k = 0.0
            semantic_kurtoses.append(k)
        
        log(f"  Layer {li} kurtosis comparison:")
        log(f"    Random directions:    mean={np.mean(random_kurtoses):.4f}, std={np.std(random_kurtoses):.4f}")
        log(f"    Semantic directions:  mean={np.mean(semantic_kurtoses):.4f}, std={np.std(semantic_kurtoses):.4f}")
        log(f"    Ratio (semantic/random): {np.mean(semantic_kurtoses)/max(np.mean(random_kurtoses), 1e-10):.2f}x")
        
        # W_U token direction kurtosis
        wu_kurtoses = []
        for i in range(min(100, W_U.shape[0])):
            d = W_U[i]
            norm = np.linalg.norm(d)
            if norm < 1e-10: continue
            d_normed = d / norm
            cp = Wd.T @ d_normed
            cp_std = np.std(cp)
            if cp_std > 1e-10:
                k = float(np.mean(((cp - np.mean(cp)) / cp_std)**4) - 3.0)
            else:
                k = 0.0
            wu_kurtoses.append(k)
        
        log(f"    W_U token directions: mean={np.mean(wu_kurtoses):.4f}, std={np.std(wu_kurtoses):.4f}")
    
    # ======================================================================
    # Save Results
    # ======================================================================
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        elif isinstance(obj, (np.floating,)): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)): return [convert(v) for v in obj]
        return obj

    save_data = convert({
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "kurtosis_vs_netgross": kurtosis_vs_netgross,
        "specialized_results": specialized_results,
        "kurtosis_comparison": {
            "layer": li,
            "random_kurtosis_mean": float(np.mean(random_kurtoses)) if random_kurtoses else 0,
            "semantic_kurtosis_mean": float(np.mean(semantic_kurtoses)) if semantic_kurtoses else 0,
            "wu_token_kurtosis_mean": float(np.mean(wu_kurtoses)) if wu_kurtoses else 0,
        },
    })

    os.makedirs("results/phase348b_kurtosis_netgross", exist_ok=True)
    out_path = f"results/phase348b_kurtosis_netgross/{model_name}_phase348b.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    log(f"\nResults saved to {out_path}")

    del model, W_U, mlp_weights; gc.collect(); torch.cuda.empty_cache()
    total_time = time.time() - t0
    log(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f}min)")


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    if model_name not in MODEL_CONFIGS: log(f"Unknown model: {model_name}"); sys.exit(1)
    run_experiment(model_name)
    log("Phase 348b complete!")
