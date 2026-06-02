"""
Phase 346: Precise Interaction Decomposition + Layer-Level Closure Test
========================================================================

Part A: Precise interaction decomposition via 4-way factorial design:
  Instead of linear approximation, we directly measure:
  1. clean gate + clean up → exact clean output
  2. clean gate + corrupt up → up main effect
  3. corrupt gate + clean up → gate main effect  
  4. corrupt gate + corrupt up → exact corrupt output

  By manually constructing SiLU(gate)*up with mixed clean/corrupt components,
  we get exact (non-approximate) decomposition.

Part B: Layer-level accumulation closure test:
  For each binding layer, compute net_l (direction-projected MLP contribution).
  Sum across layers and compare with final binding signal.

Usage:
  python tests/glm5/phase346_interaction_closure.py qwen3
  python tests/glm5/phase346_interaction_closure.py deepseek7b
  python tests/glm5/phase346_interaction_closure.py glm4
"""
import sys, os, time, json, gc
import torch
import numpy as np
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

def log(msg="", end="\n"):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", end=end, flush=True)


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

TEST_PAIRS = [
    ("apple", "red", "blue"), ("banana", "yellow", "purple"), ("snow", "white", "black"),
    ("sky", "blue", "green"), ("fire", "hot", "cold"), ("grass", "green", "red"),
    ("ocean", "blue", "yellow"), ("sun", "yellow", "purple"), ("blood", "red", "green"),
    ("ice", "cold", "hot"), ("cherry", "red", "blue"), ("leaf", "green", "red"),
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
                if dk in keys and W_down is None: W_down = sf.get_tensor(dk).float().numpy()
                if W_down is not None: break
        except: continue
    return W_gate, W_up, W_down, d_ff


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
            if l is layer: W_gate, W_up, W_down, d_ff = get_mlp_weights_from_disk(model_name, i); break
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


def capture_hidden_states(model, tokenizer, device, prompt, n_layers):
    """Capture all layer hidden states at the last position."""
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    hs = {}
    for i, h in enumerate(out.hidden_states):
        hs[i] = h[0, -1, :].detach().cpu().float().numpy()
    return hs


def precise_interaction_decomposition(W_down, direction, gate_clean, up_clean, gate_corrupt, up_corrupt):
    """
    Precise 4-way factorial decomposition:
    
    MLP_output = W_down @ (SiLU(gate) * up)
    
    Condition 1 (CC): SiLU(gate_clean) * up_clean  → clean output
    Condition 2 (CR): SiLU(gate_clean) * up_corrupt → gate=clean, up=corrupt
    Condition 3 (RC): SiLU(gate_corrupt) * up_clean → gate=corrupt, up=clean
    Condition 4 (RR): SiLU(gate_corrupt) * up_corrupt → corrupt output
    
    Effects (projected onto direction):
    - gate_main = proj(CC - CR) + proj(RC - RR)) / 2
    - up_main   = proj(CC - RC) + proj(CR - RR)) / 2
    - interaction = proj(CC) - proj(CR) - proj(RC) + proj(RR)
    
    Note: This is the standard 2×2 factorial decomposition, which is exact.
    """
    d_ff = W_down.shape[1]
    min_d = min(gate_clean.shape[0], d_ff, W_down.shape[1])
    
    def silu(x):
        return x * (1.0 / (1.0 + np.exp(-np.clip(x, -20, 20))))
    
    gsc = silu(gate_clean[:min_d]); gsr = silu(gate_corrupt[:min_d])
    uc = up_clean[:min_d]; ur = up_corrupt[:min_d]
    Wd = W_down[:, :min_d]
    
    # 4 conditions
    cc = Wd @ (gsc * uc)   # clean gate + clean up
    cr = Wd @ (gsc * ur)   # clean gate + corrupt up
    rc = Wd @ (gsr * uc)   # corrupt gate + clean up
    rr = Wd @ (gsr * ur)   # corrupt gate + corrupt up
    
    # Project onto direction
    cc_proj = float(direction @ cc)
    cr_proj = float(direction @ cr)
    rc_proj = float(direction @ rc)
    rr_proj = float(direction @ rr)
    
    # Standard 2×2 factorial decomposition (exact)
    # Total effect: CC - RR = gate_main + up_main + interaction
    gate_main = ((cc_proj - cr_proj) + (rc_proj - rr_proj)) / 2
    up_main = ((cc_proj - rc_proj) + (cr_proj - rr_proj)) / 2
    interaction = cc_proj - cr_proj - rc_proj + rr_proj
    
    total_effect = cc_proj - rr_proj
    
    # Verify decomposition
    decomp_sum = gate_main + up_main + interaction
    decomp_error = abs(decomp_sum - total_effect)
    
    return {
        "cc_proj": cc_proj, "cr_proj": cr_proj,
        "rc_proj": rc_proj, "rr_proj": rr_proj,
        "gate_main": gate_main,
        "up_main": up_main,
        "interaction": interaction,
        "total_effect": total_effect,
        "decomp_error": decomp_error,
        "gate_main_pct": abs(gate_main) / max(abs(total_effect), 1e-10),
        "up_main_pct": abs(up_main) / max(abs(total_effect), 1e-10),
        "interaction_pct": abs(interaction) / max(abs(total_effect), 1e-10),
        "interaction_contributes": abs(interaction) > 0.1 * max(abs(gate_main), abs(up_main), 1e-10),
    }


def run_experiment(model_name):
    log(f"Phase 346: Precise Interaction + Layer Closure — {model_name}")
    log("=" * 70)
    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    cfg = MODEL_CONFIGS[model_name]
    binding_layers = cfg["binding_layers"]
    W_U = get_W_U(model, model_name)
    d_model = W_U.shape[1]
    log(f"  W_U shape: {W_U.shape}")

    # Pre-extract MLP weights
    layers = get_layers(model)
    mlp_weights = {}
    for li in binding_layers:
        _, _, W_down, d_ff = get_mlp_weights(layers[li], model_name, model)
        mlp_weights[li] = {"W_down": W_down, "d_ff": d_ff}

    # ======================================================================
    # PART A: Precise Interaction Decomposition
    # ======================================================================
    log(f"\n{'='*70}")
    log(f"PART A: Precise 4-way Factorial Interaction Decomposition")
    log(f"{'='*70}")

    interaction_results = {}
    all_gate_main = []; all_up_main = []; all_interaction = []; all_total = []
    all_gate_pct = []; all_up_pct = []; all_ia_pct = []

    for pidx, (obj, target, competitor) in enumerate(TEST_PAIRS[:8]):
        tid_t = get_token_id(tokenizer, target)
        tid_c = get_token_id(tokenizer, competitor)
        if tid_t is None or tid_c is None: continue

        direction = W_U[tid_t] - W_U[tid_c]
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-10: continue
        direction_normed = direction / dir_norm

        clean_prompt = f"The {obj}"
        clean_caps = capture_mlp_internals(model, tokenizer, device, clean_prompt, binding_layers, cfg["n_layers"])
        corrupt_caps = capture_mlp_internals(model, tokenizer, device, CORRUPTED_BASELINE, binding_layers, cfg["n_layers"])

        pair_key = f"{obj}-{target}-{competitor}"
        interaction_results[pair_key] = {}

        for li in binding_layers:
            mw = mlp_weights[li]
            W_down = mw["W_down"]; d_ff = mw["d_ff"]
            if W_down is None: continue
            gk = f"gate_{li}"; uk = f"up_{li}"
            if gk not in clean_caps or gk not in corrupt_caps: continue
            cg = clean_caps[gk][:d_ff]; crg = corrupt_caps[gk][:d_ff]
            cu = clean_caps.get(uk, np.ones(d_ff))[:d_ff]
            cru = corrupt_caps.get(uk, np.ones(d_ff))[:d_ff]

            ires = precise_interaction_decomposition(W_down, direction_normed, cg, cu, crg, cru)
            interaction_results[pair_key][str(li)] = ires

            all_gate_main.append(ires["gate_main"])
            all_up_main.append(ires["up_main"])
            all_interaction.append(ires["interaction"])
            all_total.append(ires["total_effect"])
            all_gate_pct.append(ires["gate_main_pct"])
            all_up_pct.append(ires["up_main_pct"])
            all_ia_pct.append(ires["interaction_pct"])

        del clean_caps, corrupt_caps; gc.collect(); torch.cuda.empty_cache()
        if (pidx + 1) % 4 == 0:
            log(f"  [{pidx+1}/8] elapsed={time.time()-t0:.0f}s")

    # Summary
    log(f"\n  PART A Summary:")
    if all_gate_main:
        total_abs = np.mean(np.abs(all_gate_main)) + np.mean(np.abs(all_up_main)) + np.mean(np.abs(all_interaction))
        if total_abs > 1e-10:
            gm_frac = np.mean(np.abs(all_gate_main)) / total_abs
            um_frac = np.mean(np.abs(all_up_main)) / total_abs
            ia_frac = np.mean(np.abs(all_interaction)) / total_abs
        else:
            gm_frac = um_frac = ia_frac = 0

        log(f"  Gate main:    mean={np.mean(all_gate_main):+.6f}, |mean|={np.mean(np.abs(all_gate_main)):.6f}, frac={gm_frac:.3f}")
        log(f"  Up main:      mean={np.mean(all_up_main):+.6f}, |mean|={np.mean(np.abs(all_up_main)):.6f}, frac={um_frac:.3f}")
        log(f"  Interaction:  mean={np.mean(all_interaction):+.6f}, |mean|={np.mean(np.abs(all_interaction)):.6f}, frac={ia_frac:.3f}")
        log(f"  Total effect: mean={np.mean(all_total):+.6f}")
        log(f"  Decomposition error: mean={np.mean([abs(ires['decomp_error']) for pair in interaction_results.values() for ires in pair.values()]):.8f}")

        # Interaction sign analysis
        pos_ia = sum(1 for x in all_interaction if x > 0)
        neg_ia = sum(1 for x in all_interaction if x < 0)
        log(f"  Interaction sign: {pos_ia} positive, {neg_ia} negative out of {len(all_interaction)}")

        # Interaction magnitude vs main effects
        ia_significant = sum(1 for p in all_ia_pct if p > 0.1)
        log(f"  Interaction >10% of total: {ia_significant}/{len(all_ia_pct)}")

    # ======================================================================
    # PART B: Layer-Level Accumulation Closure Test
    # ======================================================================
    log(f"\n{'='*70}")
    log(f"PART B: Layer-Level Accumulation Closure")
    log(f"{'='*70}")

    closure_results = {}
    n_closure_pairs = 6

    for pidx, (obj, target, competitor) in enumerate(TEST_PAIRS[:n_closure_pairs]):
        tid_t = get_token_id(tokenizer, target)
        tid_c = get_token_id(tokenizer, competitor)
        if tid_t is None or tid_c is None: continue

        direction = W_U[tid_t] - W_U[tid_c]
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-10: continue
        direction_normed = direction / dir_norm

        clean_prompt = f"The {obj}"

        # Capture hidden states for both prompts
        hs_clean = capture_hidden_states(model, tokenizer, device, clean_prompt, cfg["n_layers"])
        hs_corrupt = capture_hidden_states(model, tokenizer, device, CORRUPTED_BASELINE, cfg["n_layers"])

        # Final binding signal
        final_clean_proj = float(direction_normed @ hs_clean[cfg["n_layers"]])
        final_corrupt_proj = float(direction_normed @ hs_corrupt[cfg["n_layers"]])
        final_binding = final_clean_proj - final_corrupt_proj

        # Per-layer binding signal
        layer_binding = {}
        for l in range(cfg["n_layers"] + 1):
            c_proj = float(direction_normed @ hs_clean[l])
            r_proj = float(direction_normed @ hs_corrupt[l])
            layer_binding[l] = c_proj - r_proj

        # Per-MLP-layer net contribution
        # MLP_l contributes: h_{l+1} - h_l (before LayerNorm)
        # But with residual: h_{l+1} = h_l + attn_l(h_l) + mlp_l(h_l)
        # Net MLP contribution in binding direction = (binding_{l+1} - binding_l) - attn_contribution

        # Simplified: just compute MLP net contribution from channel decomposition
        clean_caps = capture_mlp_internals(model, tokenizer, device, clean_prompt, binding_layers, cfg["n_layers"])
        corrupt_caps = capture_mlp_internals(model, tokenizer, device, CORRUPTED_BASELINE, binding_layers, cfg["n_layers"])

        mlp_net_per_layer = {}
        for li in binding_layers:
            mw = mlp_weights[li]
            W_down = mw["W_down"]; d_ff = mw["d_ff"]
            if W_down is None: continue
            gk = f"gate_{li}"; uk = f"up_{li}"
            if gk not in clean_caps or gk not in corrupt_caps: continue
            cg = clean_caps[gk][:d_ff]; crg = corrupt_caps[gk][:d_ff]
            cu = clean_caps.get(uk, np.ones(d_ff))[:d_ff]
            cru = corrupt_caps.get(uk, np.ones(d_ff))[:d_ff]

            def silu(x):
                return x * (1.0 / (1.0 + np.exp(-np.clip(x, -20, 20))))
            min_d = min(cg.shape[0], d_ff, W_down.shape[1])
            gsc = silu(cg[:min_d]); gsr = silu(crg[:min_d])
            uc = cu[:min_d]; ur = cru[:min_d]
            Wd = W_down[:, :min_d]

            clean_mlp_out = Wd @ (gsc * uc)
            corrupt_mlp_out = Wd @ (gsr * ur)
            net_mlp = float(direction_normed @ (clean_mlp_out - corrupt_mlp_out))
            mlp_net_per_layer[li] = net_mlp

        del clean_caps, corrupt_caps, hs_clean, hs_corrupt; gc.collect(); torch.cuda.empty_cache()

        # Sum of MLP net contributions
        mlp_net_sum = sum(mlp_net_per_layer.values())

        pair_key = f"{obj}-{target}-{competitor}"
        closure_results[pair_key] = {
            "final_binding": final_binding,
            "mlp_net_sum": mlp_net_sum,
            "mlp_net_per_layer": {str(k): v for k, v in mlp_net_per_layer.items()},
            "layer_binding_trajectory": {str(k): v for k, v in layer_binding.items()},
            "closure_ratio": mlp_net_sum / max(abs(final_binding), 1e-10),
        }

        log(f"  Pair {pidx+1}: {pair_key}")
        log(f"    Final binding: {final_binding:.4f}")
        log(f"    MLP net sum:   {mlp_net_sum:.4f}")
        log(f"    Closure ratio: {mlp_net_sum/max(abs(final_binding),1e-10):.4f}")
        log(f"    Per-layer MLP net: {', '.join(f'L{li}={v:.4f}' for li, v in sorted(mlp_net_per_layer.items()))}")

    # Overall closure
    log(f"\n  PART B Summary:")
    if closure_results:
        closure_ratios = [v["closure_ratio"] for v in closure_results.values()]
        final_bindings = [v["final_binding"] for v in closure_results.values()]
        mlp_sums = [v["mlp_net_sum"] for v in closure_results.values()]
        log(f"  Mean closure ratio: {np.mean(closure_ratios):.4f} ± {np.std(closure_ratios):.4f}")
        log(f"  Mean final binding: {np.mean(final_bindings):.4f}")
        log(f"  Mean MLP net sum:   {np.mean(mlp_sums):.4f}")
        log(f"  Correlation(final, mlp_sum): {np.corrcoef(final_bindings, mlp_sums)[0,1]:.4f}" if len(final_bindings) > 2 else "  Too few points for correlation")

    # ======================================================================
    # PART C: Binding trajectory analysis (how does binding signal evolve across ALL layers)
    # ======================================================================
    log(f"\n{'='*70}")
    log(f"PART C: Full Binding Trajectory")
    log(f"{'='*70}")

    # Use first pair for trajectory
    obj, target, competitor = TEST_PAIRS[0]
    tid_t = get_token_id(tokenizer, target)
    tid_c = get_token_id(tokenizer, competitor)
    if tid_t is not None and tid_c is not None:
        direction = W_U[tid_t] - W_U[tid_c]
        dir_norm = np.linalg.norm(direction)
        if dir_norm > 1e-10:
            direction_normed = direction / dir_norm
            hs_clean = capture_hidden_states(model, tokenizer, device, f"The {obj}", cfg["n_layers"])
            hs_corrupt = capture_hidden_states(model, tokenizer, device, CORRUPTED_BASELINE, cfg["n_layers"])

            trajectory = []
            for l in range(cfg["n_layers"] + 1):
                c_proj = float(direction_normed @ hs_clean[l])
                r_proj = float(direction_normed @ hs_corrupt[l])
                trajectory.append(c_proj - r_proj)

            # Print key points
            log(f"  Binding trajectory for '{obj}' ({target} vs {competitor}):")
            for l in [0, cfg["n_layers"]//4, cfg["n_layers"]//2, 3*cfg["n_layers"]//4, cfg["n_layers"]]:
                if l < len(trajectory):
                    log(f"    Layer {l}: binding = {trajectory[l]:.4f}")
            # Also print binding layers
            for li in binding_layers:
                if li < len(trajectory):
                    log(f"    Layer {li} (binding): binding = {trajectory[li]:.4f}")

            # Compute per-layer delta
            deltas = [trajectory[l+1] - trajectory[l] for l in range(len(trajectory)-1)]
            log(f"  Layer deltas at binding layers:")
            for li in binding_layers:
                if li < len(deltas):
                    log(f"    Layer {li}→{li+1}: delta = {deltas[li]:.4f}")

            del hs_clean, hs_corrupt; gc.collect(); torch.cuda.empty_cache()

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
        "interaction_summary": {
            "gate_main_mean": float(np.mean(all_gate_main)) if all_gate_main else 0,
            "up_main_mean": float(np.mean(all_up_main)) if all_up_main else 0,
            "interaction_mean": float(np.mean(all_interaction)) if all_interaction else 0,
            "total_effect_mean": float(np.mean(all_total)) if all_total else 0,
            "gate_main_frac": float(gm_frac) if all_gate_main else 0,
            "up_main_frac": float(um_frac) if all_up_main else 0,
            "interaction_frac": float(ia_frac) if all_interaction else 0,
        },
        "interaction_per_pair": interaction_results,
        "closure_results": closure_results,
        "closure_summary": {
            "mean_closure_ratio": float(np.mean([v["closure_ratio"] for v in closure_results.values()])) if closure_results else 0,
            "mean_final_binding": float(np.mean([v["final_binding"] for v in closure_results.values()])) if closure_results else 0,
            "mean_mlp_net_sum": float(np.mean([v["mlp_net_sum"] for v in closure_results.values()])) if closure_results else 0,
        },
    })

    os.makedirs("results/phase346_interaction_closure", exist_ok=True)
    out_path = f"results/phase346_interaction_closure/{model_name}_phase346.json"
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
    log("Phase 346 complete!")
