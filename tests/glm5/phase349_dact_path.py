"""
Phase 349: Δact Conditional Path Analysis — Source of Systematic Micro-Bias
============================================================================

Phase 348b showed that micro-bias (net/gross) comes from "ordinary channels" (90%),
not specialized channels. This raises the key question:

WHY do ordinary channels produce systematic directional micro-bias?

This script analyzes:
Part A: Δact distribution structure — is it concentrated or distributed?
  - Gini coefficient of |Δact|
  - Top-k Δact channel fraction
  - Overlap of top-Δact channels across different pairs

Part B: Δact vs channel_proj correlation — do large Δact channels 
  correlate with positive or negative channel_proj?
  - If Δact systematically biases towards positive-proj channels → micro-bias
  - This is the direct mechanism of micro-bias

Part C: Gate vs Up decomposition of Δact
  - Δact = SiLU(gate_c)*up_c - SiLU(gate_r)*up_r
  - Which part (gate change vs up change) correlates with channel_proj sign?
  - This tells us whether gate or up is the "path selector"

Part D: Cross-pair channel reuse
  - Do different binding pairs reuse the same high-Δact channels?
  - If yes → shared attribute-encoding channels
  - If no → pair-specific channels

Usage:
  python tests/glm5/phase349_dact_path.py qwen3
  python tests/glm5/phase349_dact_path.py deepseek7b
  python tests/glm5/phase349_dact_path.py glm4
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


def gini_coefficient(x):
    """Compute Gini coefficient of array x."""
    sorted_x = np.sort(np.abs(x))
    n = len(sorted_x)
    if n == 0: return 0
    cumx = np.cumsum(sorted_x)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n if cumx[-1] > 0 else 0


def run_experiment(model_name):
    log(f"Phase 349: Δact Conditional Path Analysis — {model_name}")
    log("=" * 70)
    t0 = time.time()
    cfg = MODEL_CONFIGS[model_name]
    binding_layers = cfg["binding_layers"]
    n_layers = cfg["n_layers"]
    
    model, tokenizer, device = load_model_bf16(model_name)
    W_U = get_W_U(model, model_name)
    d_model = W_U.shape[1]
    log(f"  W_U shape: {W_U.shape}")
    
    # Pre-extract MLP weights
    layers = get_layers(model)
    mlp_weights = {}
    for li in binding_layers:
        W_gate, W_up, W_down, d_ff = get_mlp_weights(layers[li], model_name, model)
        mlp_weights[li] = {"W_gate": W_gate, "W_up": W_up, "W_down": W_down, "d_ff": d_ff}
    
    # ======================================================================
    # Part A: Δact Distribution Structure
    # ======================================================================
    log(f"\n{'='*70}")
    log(f"PART A: Δact Distribution Structure")
    log(f"{'='*70}")
    
    all_gini = []
    all_top10_frac = []
    all_top1_frac = []
    
    # Store Δact for cross-pair analysis
    pair_dacts = defaultdict(dict)  # pair_dacts[pair_key][layer_str] = dact array
    pair_channel_proj = defaultdict(dict)
    
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
        
        pair_key = f"{obj}-{target}-{competitor}"
        
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
            
            # Δact = SiLU(gate_c)*up_c - SiLU(gate_r)*up_r
            dact = gsc * uc - gsr * ur
            channel_proj = Wd.T @ direction_normed
            
            pair_dacts[pair_key][str(li)] = dact
            pair_channel_proj[pair_key][str(li)] = channel_proj
            
            # Distribution metrics
            abs_dact = np.abs(dact)
            gini = gini_coefficient(dact)
            total_abs = np.sum(abs_dact)
            
            # Top-k fractions
            sorted_dact = np.sort(abs_dact)[::-1]
            top1_frac = float(np.sum(sorted_dact[:max(1, min_d//100)]) / max(total_abs, 1e-10))
            top10_frac = float(np.sum(sorted_dact[:max(1, min_d//10)]) / max(total_abs, 1e-10))
            
            all_gini.append(gini)
            all_top1_frac.append(top1_frac)
            all_top10_frac.append(top10_frac)
        
        del clean_caps, corrupt_caps; gc.collect(); torch.cuda.empty_cache()
        if (pidx + 1) % 6 == 0:
            log(f"  [{pidx+1}/{len(TEST_PAIRS)}] elapsed={time.time()-t0:.0f}s")
    
    log(f"\n  PART A Summary:")
    log(f"  Gini coefficient: mean={np.mean(all_gini):.4f} ± {np.std(all_gini):.4f}")
    log(f"  Top 1% |Δact| fraction: mean={np.mean(all_top1_frac):.4f}")
    log(f"  Top 10% |Δact| fraction: mean={np.mean(all_top10_frac):.4f}")
    
    # ======================================================================
    # Part B: Δact vs channel_proj Correlation — The Key Question
    # ======================================================================
    log(f"\n{'='*70}")
    log(f"PART B: Δact vs channel_proj Correlation — Why Micro-Bias Exists")
    log(f"{'='*70}")
    
    # For each (pair, layer), compute correlation between Δact and channel_proj
    # If Δact is systematically larger in positive-proj channels → micro-bias
    corr_results = []
    
    for pair_key in pair_dacts:
        for li_str in pair_dacts[pair_key]:
            dact = pair_dacts[pair_key][li_str]
            cproj = pair_channel_proj[pair_key][li_str]
            
            if len(dact) < 10: continue
            
            # Correlation between Δact and channel_proj
            corr = float(np.corrcoef(dact, cproj)[0, 1])
            
            # Separate analysis for positive vs negative proj channels
            pos_mask = cproj > 0
            neg_mask = cproj < 0
            
            pos_dact_mean = float(np.mean(dact[pos_mask])) if np.sum(pos_mask) > 0 else 0
            neg_dact_mean = float(np.mean(dact[neg_mask])) if np.sum(neg_mask) > 0 else 0
            pos_dact_abs_mean = float(np.mean(np.abs(dact[pos_mask]))) if np.sum(pos_mask) > 0 else 0
            neg_dact_abs_mean = float(np.mean(np.abs(dact[neg_mask]))) if np.sum(neg_mask) > 0 else 0
            
            # Key metric: Δact sign bias in positive vs negative proj channels
            # If pos channels have more positive Δact → contributes to net positive binding
            pos_dact_sum = float(np.sum(dact[pos_mask]))
            neg_dact_sum = float(np.sum(dact[neg_mask]))
            
            # The net binding = sum(cproj * dact)
            # = sum_pos(cproj * dact) + sum_neg(cproj * dact)
            # For net > 0, we need: sum_pos > |sum_neg|
            # i.e., positive-proj channels have dact that aligns with proj direction
            
            pos_contrib = float(np.sum(cproj[pos_mask] * dact[pos_mask]))
            neg_contrib = float(np.sum(cproj[neg_mask] * dact[neg_mask]))
            total_contrib = pos_contrib + neg_contrib
            
            # Fraction of net from positive vs negative channels
            pos_frac_of_net = pos_contrib / max(abs(total_contrib), 1e-10) if total_contrib != 0 else 0
            
            corr_results.append({
                "pair": pair_key,
                "layer": int(li_str),
                "corr_dact_cproj": corr,
                "pos_dact_mean": pos_dact_mean,
                "neg_dact_mean": neg_dact_mean,
                "pos_dact_abs_mean": pos_dact_abs_mean,
                "neg_dact_abs_mean": neg_dact_abs_mean,
                "pos_contrib": pos_contrib,
                "neg_contrib": neg_contrib,
                "total_contrib": total_contrib,
                "pos_frac_of_net": pos_frac_of_net,
            })
    
    log(f"\n  PART B Summary:")
    if corr_results:
        corrs = [x["corr_dact_cproj"] for x in corr_results]
        pos_dact = [x["pos_dact_mean"] for x in corr_results]
        neg_dact = [x["neg_dact_mean"] for x in corr_results]
        pos_abs = [x["pos_dact_abs_mean"] for x in corr_results]
        neg_abs = [x["neg_dact_abs_mean"] for x in corr_results]
        pos_frac = [x["pos_frac_of_net"] for x in corr_results]
        
        log(f"  Correlation(Δact, channel_proj): mean={np.mean(corrs):.4f} ± {np.std(corrs):.4f}")
        log(f"  Positive-proj channels: Δact_mean={np.mean(pos_dact):.6f}, |Δact|_mean={np.mean(pos_abs):.6f}")
        log(f"  Negative-proj channels: Δact_mean={np.mean(neg_dact):.6f}, |Δact|_mean={np.mean(neg_abs):.6f}")
        log(f"  Δact_mean ratio (pos/neg abs): {np.mean(pos_abs)/max(np.mean(neg_abs), 1e-10):.4f}")
        log(f"  Positive-proj fraction of net: mean={np.mean(pos_frac):.4f}")
        
        n_positive_corr = sum(1 for c in corrs if c > 0)
        log(f"  Positive correlation count: {n_positive_corr}/{len(corrs)}")
    
    # ======================================================================
    # Part C: Gate vs Up Decomposition
    # ======================================================================
    log(f"\n{'='*70}")
    log(f"PART C: Gate vs Up Decomposition of Δact")
    log(f"{'='*70}")
    
    gate_vs_up_results = []
    
    for pidx, (obj, target, competitor) in enumerate(TEST_PAIRS[:10]):
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
        
        pair_key = f"{obj}-{target}-{competitor}"
        
        for li in binding_layers[:2]:  # Only first 2 binding layers
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
            channel_proj = Wd.T @ direction_normed
            
            gsc = silu_np(cg[:min_d]); gsr = silu_np(crg[:min_d])
            uc = cu[:min_d]; ur = cru[:min_d]
            
            # Decompose Δact
            # Δact = SiLU(g_c)*u_c - SiLU(g_r)*u_r
            # = SiLU(g_c)*(u_c - u_r) + (SiLU(g_c) - SiLU(g_r))*u_r
            # = gate-driven part + up-driven part
            # More precisely:
            # Δact = SiLU(g_c)*(u_c - u_r) + (SiLU(g_c) - SiLU(g_r))*u_r
            # gate_contribution = (SiLU(g_c) - SiLU(g_r)) * u_r  [gate change at corrupt up level]
            # up_contribution = SiLU(g_c) * (u_c - u_r)  [up change at clean gate level]
            
            gate_diff = gsc - gsr
            up_diff = uc - ur
            
            # Δact = gate_diff * ur + gsc * up_diff
            # This is one decomposition (gate-driven + up-driven)
            dact = gsc * uc - gsr * ur
            gate_driven = gate_diff * ur  # gate changes, up stays at corrupt level
            up_driven = gsc * up_diff    # up changes, gate stays at clean level
            # Note: gate_driven + up_driven = dact (exact decomposition)
            
            # Project onto binding direction
            gate_driven_proj = float(direction_normed @ (Wd @ gate_driven))
            up_driven_proj = float(direction_normed @ (Wd @ up_driven))
            dact_proj = float(direction_normed @ (Wd @ dact))
            
            # Correlation of gate_diff and up_diff with channel_proj
            gate_diff_corr = float(np.corrcoef(gate_diff, channel_proj)[0, 1]) if np.std(gate_diff) > 1e-10 else 0
            up_diff_corr = float(np.corrcoef(up_diff, channel_proj)[0, 1]) if np.std(up_diff) > 1e-10 else 0
            dact_corr = float(np.corrcoef(dact, channel_proj)[0, 1])
            
            # Channel-level: which channels have largest gate_diff vs up_diff?
            abs_gate_diff = np.abs(gate_diff)
            abs_up_diff = np.abs(up_diff)
            abs_dact_arr = np.abs(dact)
            
            top_dact_k = max(1, min_d // 10)
            top_dact_indices = np.argsort(abs_dact_arr)[-top_dact_k:]
            
            # Among top-Δact channels, what fraction have positive proj?
            top_dact_pos_proj_frac = float(np.mean(channel_proj[top_dact_indices] > 0))
            
            gate_vs_up_results.append({
                "pair": pair_key,
                "layer": li,
                "gate_driven_proj": gate_driven_proj,
                "up_driven_proj": up_driven_proj,
                "dact_proj": dact_proj,
                "gate_fraction": abs(gate_driven_proj) / max(abs(dact_proj), 1e-10) if abs(dact_proj) > 1e-10 else 0,
                "up_fraction": abs(up_driven_proj) / max(abs(dact_proj), 1e-10) if abs(dact_proj) > 1e-10 else 0,
                "gate_diff_corr": gate_diff_corr,
                "up_diff_corr": up_diff_corr,
                "dact_corr": dact_corr,
                "top_dact_pos_proj_frac": top_dact_pos_proj_frac,
            })
        
        del clean_caps, corrupt_caps; gc.collect(); torch.cuda.empty_cache()
    
    log(f"\n  PART C Summary:")
    if gate_vs_up_results:
        gf = [x["gate_fraction"] for x in gate_vs_up_results]
        uf = [x["up_fraction"] for x in gate_vs_up_results]
        gc_arr = [x["gate_diff_corr"] for x in gate_vs_up_results]
        uc_arr = [x["up_diff_corr"] for x in gate_vs_up_results]
        dc_arr = [x["dact_corr"] for x in gate_vs_up_results]
        tpf = [x["top_dact_pos_proj_frac"] for x in gate_vs_up_results]
        
        log(f"  Gate-driven fraction of |Δact|: mean={np.mean(gf):.4f}")
        log(f"  Up-driven fraction of |Δact|:   mean={np.mean(uf):.4f}")
        log(f"  Corr(gate_diff, channel_proj):  mean={np.mean(gc_arr):.4f}")
        log(f"  Corr(up_diff, channel_proj):    mean={np.mean(uc_arr):.4f}")
        log(f"  Corr(Δact, channel_proj):       mean={np.mean(dc_arr):.4f}")
        log(f"  Top-Δact channels pos_proj_frac: mean={np.mean(tpf):.4f} (expect ~0.5 if random)")
    
    # ======================================================================
    # Part D: Cross-Pair Channel Reuse
    # ======================================================================
    log(f"\n{'='*70}")
    log(f"PART D: Cross-Pair Channel Reuse")
    log(f"{'='*70}")
    
    # For each binding layer, compute overlap of top-Δact channels across pairs
    for li in binding_layers[:2]:
        li_str = str(li)
        pair_top_channels = {}
        
        for pair_key in pair_dacts:
            if li_str not in pair_dacts[pair_key]: continue
            dact = pair_dacts[pair_key][li_str]
            top_k = max(1, len(dact) // 10)  # top 10%
            top_indices = set(np.argsort(np.abs(dact))[-top_k:])
            pair_top_channels[pair_key] = top_indices
        
        if len(pair_top_channels) < 2: continue
        
        # Compute pairwise Jaccard similarity
        pair_keys = list(pair_top_channels.keys())
        jaccards = []
        for i in range(len(pair_keys)):
            for j in range(i+1, len(pair_keys)):
                s1 = pair_top_channels[pair_keys[i]]
                s2 = pair_top_channels[pair_keys[j]]
                jaccard = len(s1 & s2) / max(len(s1 | s2), 1)
                jaccards.append(jaccard)
        
        # Random baseline: random top-k sets
        dact_sample = pair_dacts[pair_keys[0]][li_str]
        n_channels = len(dact_sample)
        top_k = max(1, n_channels // 10)
        rng = np.random.RandomState(42)
        random_jaccards = []
        for _ in range(100):
            s1 = set(rng.choice(n_channels, top_k, replace=False))
            s2 = set(rng.choice(n_channels, top_k, replace=False))
            random_jaccards.append(len(s1 & s2) / max(len(s1 | s2), 1))
        
        log(f"  Layer {li}:")
        log(f"    Pairwise Jaccard of top-Δact channels: mean={np.mean(jaccards):.4f}")
        log(f"    Random Jaccard baseline: mean={np.mean(random_jaccards):.4f}")
        log(f"    Ratio: {np.mean(jaccards)/max(np.mean(random_jaccards), 1e-10):.2f}x")
    
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
        "part_a_distribution": {
            "gini_mean": float(np.mean(all_gini)) if all_gini else 0,
            "top1_frac_mean": float(np.mean(all_top1_frac)) if all_top1_frac else 0,
            "top10_frac_mean": float(np.mean(all_top10_frac)) if all_top10_frac else 0,
        },
        "part_b_dact_vs_cproj": {
            "corr_mean": float(np.mean([x["corr_dact_cproj"] for x in corr_results])) if corr_results else 0,
            "pos_dact_abs_mean": float(np.mean([x["pos_dact_abs_mean"] for x in corr_results])) if corr_results else 0,
            "neg_dact_abs_mean": float(np.mean([x["neg_dact_abs_mean"] for x in corr_results])) if corr_results else 0,
            "pos_frac_of_net_mean": float(np.mean([x["pos_frac_of_net"] for x in corr_results])) if corr_results else 0,
        },
        "part_c_gate_vs_up": {
            "gate_fraction_mean": float(np.mean(gf)) if gate_vs_up_results else 0,
            "up_fraction_mean": float(np.mean(uf)) if gate_vs_up_results else 0,
            "gate_diff_corr_mean": float(np.mean(gc_arr)) if gate_vs_up_results else 0,
            "up_diff_corr_mean": float(np.mean(uc_arr)) if gate_vs_up_results else 0,
            "dact_corr_mean": float(np.mean(dc_arr)) if gate_vs_up_results else 0,
            "top_dact_pos_proj_frac": float(np.mean(tpf)) if gate_vs_up_results else 0,
        },
    })

    os.makedirs("results/phase349_dact_path", exist_ok=True)
    out_path = f"results/phase349_dact_path/{model_name}_phase349.json"
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
    log("Phase 349 complete!")
