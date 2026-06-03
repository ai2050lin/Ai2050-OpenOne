"""
Phase 362: Residual Stream Binding Trace
==========================================

Core question: WHERE in the residual stream does binding information first appear?

Phase 361 discovered that h_in_patch ≈ full_resid at "core binding layers",
meaning binding info is already in the residual stream, NOT created by those layers.

This phase traces the binding signal through ALL layers to find:
1. At which layer does binding info first enter the residual stream?
2. How does it propagate and amplify through layers?
3. Is there a "binding creation frontier" where h_in_patch is weak but full_resid is strong?

Two methods:
  A. Logit Lens Trace (fast, approximate): project each layer's hidden state through W_U
  B. h_in_patch at sampled layers (slow, exact): direct causal measurement

Unified notation:
  C2R: effect = -Δgap / |base_gap| (positive = binding damaged)
  R2C: effect = +Δgap / |base_gap| (positive = binding rescued)
"""

import sys, os, time, json, gc
import torch
import numpy as np
from datetime import datetime
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')


def log(msg="", end="\n"):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", end=end, flush=True)


# ===== Model Configs =====
MODEL_CONFIGS = {
    "qwen3": {
        "path": "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
        "n_layers": 36, "d_model": 2560,
        "sample_layers": list(range(0, 36, 3)) + [35],  # every 3rd + last
    },
    "glm4": {
        "path": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
        "n_layers": 40, "d_model": 4096,
        "sample_layers": list(range(0, 40, 4)) + [39],  # every 4th + last
    },
    "deepseek7b": {
        "path": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "n_layers": 28, "d_model": 3584,
        "sample_layers": list(range(0, 28, 3)) + [27],  # every 3rd + last
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
    ("ice", "cold", "hot"), ("snow", "cold", "hot"), ("volcano", "hot", "cold"),
    ("furnace", "hot", "cold"), ("glacier", "cold", "hot"),
    ("rain", "wet", "dry"), ("ocean", "wet", "dry"), ("river", "wet", "dry"),
    ("sand", "dry", "wet"), ("dust", "dry", "wet"), ("bone", "dry", "wet"),
    ("swamp", "wet", "dry"), ("desert", "dry", "wet"),
    ("silk", "smooth", "rough"), ("sandpaper", "rough", "smooth"),
    ("glass", "smooth", "rough"), ("rock", "rough", "smooth"),
    ("velvet", "soft", "hard"), ("diamond", "hard", "soft"),
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
            log(f"  Failed with {impl}: {str(e)[:80]}")
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


def compute_effect(delta_gap, base_gap, direction):
    abs_base = abs(base_gap)
    if abs_base < 1e-10:
        return 0.0
    if direction == "c2r":
        return -delta_gap / abs_base
    else:
        return delta_gap / abs_base


def get_final_ln(model):
    """Find the final LayerNorm before lm_head."""
    for name in ["model.norm", "model.final_layernorm", "model.decoder.final_layer_norm"]:
        if hasattr(model, name.split(".")[0]):
            obj = model
            for part in name.split("."):
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    obj = None
                    break
            if obj is not None and hasattr(obj, 'weight'):
                return obj
    return None


# ===== Part A: Logit Lens Trace =====

def logit_lens_trace(model, tokenizer, device, W_U, model_name):
    """
    Project each layer's hidden state through W_U to get 'virtual logits'.
    Compute binding gap at each layer for both clean and corrupt.
    """
    cfg = MODEL_CONFIGS[model_name]
    n_layers = cfg["n_layers"]

    # Find final LayerNorm
    final_ln = get_final_ln(model)
    ln_weight = None
    if final_ln is not None:
        w = final_ln.weight
        if not w.is_meta:
            ln_weight = w.detach().cpu().float().numpy()
    
    # Fallback: load from safetensors
    if ln_weight is None:
        import glob
        from safetensors import safe_open
        for sf_file in glob.glob(os.path.join(MODEL_CONFIGS[model_name]["path"], '*.safetensors')):
            try:
                with safe_open(sf_file, framework='pt', device='cpu') as sf:
                    for key in ['model.norm.weight', 'model.final_layernorm.weight',
                                'model.decoder.final_layer_norm.weight']:
                        if key in sf.keys():
                            ln_weight = sf.get_tensor(key).float().numpy()
                            log(f"  Loaded final LN from safetensors: {key}")
                            break
                    if ln_weight is not None:
                        break
            except Exception:
                continue
    
    if ln_weight is None:
        log("  WARNING: No final LayerNorm found, using raw hidden states")

    layer_gaps_clean = defaultdict(list)
    layer_gaps_corrupt = defaultdict(list)
    base_gaps = []

    for pidx, (obj, target, competitor) in enumerate(TEST_PAIRS):
        tid_t = get_token_id(tokenizer, target)
        tid_c = get_token_id(tokenizer, competitor)
        if tid_t is None or tid_c is None:
            continue

        clean_prompt = f"The {obj}"

        # Run with hidden states
        clean_inp = tokenizer(clean_prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            clean_out = model(**clean_inp, output_hidden_states=True)
        clean_hs = clean_out.hidden_states  # tuple of (1, seq_len, d_model)

        corrupt_inp = tokenizer(CORRUPTED_BASELINE, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            corrupt_out = model(**corrupt_inp, output_hidden_states=True)
        corrupt_hs = corrupt_out.hidden_states

        clean_logits_final = clean_out.logits[0, -1].float().cpu().numpy()
        corrupt_logits_final = corrupt_out.logits[0, -1].float().cpu().numpy()
        base_gaps.append(float(clean_logits_final[tid_t] - clean_logits_final[tid_c]
                               - corrupt_logits_final[tid_t] + corrupt_logits_final[tid_c]))

        # Project each layer's hidden state
        for L in range(len(clean_hs)):
            # Get last token hidden state
            h_clean = clean_hs[L][0, -1].float().cpu().numpy()  # [d_model]
            h_corrupt = corrupt_hs[L][0, -1].float().cpu().numpy()

            # Apply final LayerNorm if available
            if ln_weight is not None and L > 0:
                # RMSNorm: h_normed = h / rms(h) * weight
                rms_clean = np.sqrt(np.mean(h_clean ** 2) + 1e-6)
                rms_corrupt = np.sqrt(np.mean(h_corrupt ** 2) + 1e-6)
                h_clean_proj = (h_clean / rms_clean) * ln_weight
                h_corrupt_proj = (h_corrupt / rms_corrupt) * ln_weight
            else:
                h_clean_proj = h_clean
                h_corrupt_proj = h_corrupt

            # Project through W_U
            logits_clean = W_U @ h_clean_proj
            logits_corrupt = W_U @ h_corrupt_proj

            gap_clean = float(logits_clean[tid_t] - logits_clean[tid_c])
            gap_corrupt = float(logits_corrupt[tid_t] - logits_corrupt[tid_c])

            layer_gaps_clean[L].append(gap_clean)
            layer_gaps_corrupt[L].append(gap_corrupt)

        del clean_hs, corrupt_hs, clean_out, corrupt_out
        gc.collect()
        torch.cuda.empty_cache()

        if (pidx + 1) % 10 == 0:
            log(f"  Logit lens: {pidx+1}/{len(TEST_PAIRS)} pairs done")

    # Compute binding signal at each layer
    mean_base_gap = float(np.mean([g for g in base_gaps if abs(g) > 1e-10]))

    trace = {}
    log(f"\n  --- Logit Lens Trace: {model_name} ---")
    log(f"  {'Layer':>6} {'clean_gap':>12} {'corrupt_gap':>12} {'binding_signal':>15} {'frac_of_final':>14}")
    log(f"  {'-'*65}")

    final_binding = None
    for L in sorted(layer_gaps_clean.keys()):
        cg = np.mean(layer_gaps_clean[L])
        crg = np.mean(layer_gaps_corrupt[L])
        binding = cg - crg
        if final_binding is None or L == max(layer_gaps_clean.keys()):
            final_binding = binding
        frac = binding / final_binding if abs(final_binding) > 1e-10 else 0

        trace[L] = {
            "clean_gap": float(cg),
            "corrupt_gap": float(crg),
            "binding_signal": float(binding),
            "fraction_of_final": float(frac),
        }
        log(f"  L{L:>4} {cg:>+12.4f} {crg:>+12.4f} {binding:>+15.4f} {frac:>+14.1%}")

    return trace, mean_base_gap


# ===== Part B: h_in_patch at Sampled Layers =====

def h_in_patch_trace(model, tokenizer, device, model_name, sample_layers):
    """
    Run h_in_patch at sampled layers to get exact causal measurement.
    """
    layers_obj = get_layers(model)
    results = {L: {"c2r": [], "r2c": []} for L in sample_layers}

    def make_input_patch_hook(replacement):
        def pre_hook(module, args):
            hidden_states = args[0]
            modified = hidden_states.clone()
            rep_t = torch.tensor(replacement, dtype=modified.dtype, device=modified.device)
            modified[0, -1, :] = rep_t
            return (modified,) + args[1:]
        return pre_hook

    for pidx, (obj, target, competitor) in enumerate(TEST_PAIRS):
        tid_t = get_token_id(tokenizer, target)
        tid_c = get_token_id(tokenizer, competitor)
        if tid_t is None or tid_c is None:
            continue

        clean_prompt = f"The {obj}"

        # Capture h_in at all sampled layers for both clean and corrupt
        clean_h_in = {}
        corrupt_h_in = {}

        def make_capture_hook(store, key):
            def pre_hook(module, args):
                inp = args[0]
                store[key] = inp[0, -1, :].detach().cpu().float().numpy()
            return pre_hook

        # Clean forward
        hooks = []
        for L in sample_layers:
            hooks.append(layers_obj[L].register_forward_pre_hook(
                make_capture_hook(clean_h_in, L)))
        inp = tokenizer(clean_prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            clean_out = model(**inp, output_hidden_states=False)
        clean_logits = clean_out.logits[0, -1].float().cpu().numpy()
        for h in hooks:
            h.remove()

        # Corrupt forward
        hooks = []
        for L in sample_layers:
            hooks.append(layers_obj[L].register_forward_pre_hook(
                make_capture_hook(corrupt_h_in, L)))
        inp = tokenizer(CORRUPTED_BASELINE, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            corrupt_out = model(**inp, output_hidden_states=False)
        corrupt_logits = corrupt_out.logits[0, -1].float().cpu().numpy()
        for h in hooks:
            h.remove()

        clean_target = float(clean_logits[tid_t])
        clean_compet = float(clean_logits[tid_c])
        corrupt_target = float(corrupt_logits[tid_t])
        corrupt_compet = float(corrupt_logits[tid_c])
        clean_gap = clean_target - clean_compet
        corrupt_gap = corrupt_target - corrupt_compet
        base_gap = clean_gap - corrupt_gap

        if abs(base_gap) < 1e-10:
            del clean_h_in, corrupt_h_in
            gc.collect()
            torch.cuda.empty_cache()
            continue

        # Run h_in_patch at each sampled layer
        for L in sample_layers:
            if L not in clean_h_in or L not in corrupt_h_in:
                continue

            # C2R: run clean prompt, patch with corrupt h_in
            hook = layers_obj[L].register_forward_pre_hook(
                make_input_patch_hook(corrupt_h_in[L]))
            inp = tokenizer(clean_prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
            with torch.no_grad():
                patch_out = model(**inp, output_hidden_states=False)
            patch_logits = patch_out.logits[0, -1].float().cpu().numpy()
            hook.remove()

            pt = float(patch_logits[tid_t])
            pc = float(patch_logits[tid_c])
            p_gap = pt - pc
            delta_gap = p_gap - clean_gap
            effect = compute_effect(delta_gap, base_gap, "c2r")
            results[L]["c2r"].append({"effect": effect, "delta_gap": delta_gap})

            gc.collect()
            torch.cuda.empty_cache()

            # R2C: run corrupt prompt, patch with clean h_in
            hook = layers_obj[L].register_forward_pre_hook(
                make_input_patch_hook(clean_h_in[L]))
            inp = tokenizer(CORRUPTED_BASELINE, return_tensors="pt", truncation=True, max_length=128).to(device)
            with torch.no_grad():
                patch_out = model(**inp, output_hidden_states=False)
            patch_logits = patch_out.logits[0, -1].float().cpu().numpy()
            hook.remove()

            pt = float(patch_logits[tid_t])
            pc = float(patch_logits[tid_c])
            p_gap = pt - pc
            delta_gap = p_gap - corrupt_gap
            effect = compute_effect(delta_gap, base_gap, "r2c")
            results[L]["r2c"].append({"effect": effect, "delta_gap": delta_gap})

            gc.collect()
            torch.cuda.empty_cache()

        del clean_h_in, corrupt_h_in
        gc.collect()
        torch.cuda.empty_cache()

        if (pidx + 1) % 5 == 0:
            log(f"  h_in_patch: {pidx+1}/{len(TEST_PAIRS)} pairs done")

    return results


# ===== Main Experiment =====

def run_experiment(model_name):
    log(f"Phase 362: Residual Stream Binding Trace ({model_name})")
    log("=" * 70)
    t0 = time.time()
    cfg = MODEL_CONFIGS[model_name]

    # Load model
    model, tokenizer, device = load_model_bf16(model_name)
    W_U = get_W_U(model, model_name)
    n_layers = cfg["n_layers"]
    sample_layers = sorted(set(cfg["sample_layers"]))

    log(f"\n  Model: {model_name}, n_layers={n_layers}, sample_layers={sample_layers}")

    # Part A: Logit Lens Trace
    log(f"\n  Part A: Logit Lens Trace ({len(TEST_PAIRS)} pairs)...")
    lens_trace, mean_base_gap = logit_lens_trace(model, tokenizer, device, W_U, model_name)

    # Part B: h_in_patch at sampled layers
    log(f"\n  Part B: h_in_patch Trace at {len(sample_layers)} sampled layers...")
    patch_results = h_in_patch_trace(model, tokenizer, device, model_name, sample_layers)

    # Summary
    log(f"\n  {'='*100}")
    log(f"  Phase 362 Summary: {model_name}")
    log(f"  {'='*100}")

    # Combine logit lens and h_in_patch
    log(f"\n  --- Binding Signal Trace (combined) ---")
    log(f"  {'Layer':>6} {'lens_binding':>14} {'lens_frac':>12} "
        f"{'h_in_R2C':>10} {'h_in_C2R':>10} {'n':>4}")
    log(f"  {'-'*65}")

    combined = {}
    for L in range(n_layers + 1):  # +1 for embedding layer
        lens_data = lens_trace.get(L, {})
        lens_binding = lens_data.get("binding_signal", 0)
        lens_frac = lens_data.get("fraction_of_final", 0)

        h_in_r2c = h_in_c2r = 0
        n = 0
        if L in patch_results:
            r2c_vals = patch_results[L]["r2c"]
            c2r_vals = patch_results[L]["c2r"]
            n = len(r2c_vals)
            if n > 0:
                h_in_r2c = float(np.mean([v["effect"] for v in r2c_vals]))
                h_in_c2r = float(np.mean([v["effect"] for v in c2r_vals]))

        combined[L] = {
            "lens_binding": lens_binding,
            "lens_frac": lens_frac,
            "h_in_r2c_mean": h_in_r2c,
            "h_in_c2r_mean": h_in_c2r,
            "n_pairs": n,
        }

        if L in sample_layers or L <= 2 or L >= n_layers - 2:
            log(f"  L{L:>4} {lens_binding:>+14.4f} {lens_frac:>+12.1%} "
                f"{h_in_r2c:>+10.4f} {h_in_c2r:>+10.4f} {n:>4}")

    # Find binding creation frontier
    log(f"\n  --- Binding Creation Frontier ---")
    # Find first layer where h_in_patch R2C > 0.1 (10% of base_gap)
    frontier_layer = None
    for L in sorted(sample_layers):
        if combined[L]["h_in_r2c_mean"] > 0.1 and combined[L]["n_pairs"] > 0:
            frontier_layer = L
            break

    if frontier_layer is not None:
        log(f"  First layer with h_in_patch R2C > 0.1: L{frontier_layer}")
        log(f"  → Binding info enters residual stream BEFORE L{frontier_layer}")
    else:
        log(f"  No layer found with h_in_patch R2C > 0.1")

    # Find where logit lens binding signal crosses 50%
    lens_50_layer = None
    for L in range(n_layers + 1):
        if lens_trace.get(L, {}).get("fraction_of_final", 0) > 0.5:
            lens_50_layer = L
            break

    if lens_50_layer is not None:
        log(f"  Logit lens binding > 50% of final: L{lens_50_layer}")
    else:
        log(f"  Logit lens never crosses 50%")

    # Bootstrap CI for key layers
    log(f"\n  --- Bootstrap 95% CI for h_in_patch R2C (key layers) ---")
    np.random.seed(42)
    n_bootstrap = 1000

    for L in sample_layers:
        vals = patch_results[L]["r2c"]
        if len(vals) < 5:
            continue
        effects = np.array([v["effect"] for v in vals])
        boot_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(effects, size=len(effects), replace=True)
            boot_means.append(float(np.mean(sample)))
        ci_lo = float(np.percentile(boot_means, 2.5))
        ci_hi = float(np.percentile(boot_means, 97.5))
        mean_eff = float(np.mean(effects))
        log(f"  L{L:>4}: {mean_eff:+.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}]")

    # Save
    output = {
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "phase": "362",
        "n_layers": n_layers,
        "sample_layers": sample_layers,
        "mean_base_gap": mean_base_gap,
        "logit_lens_trace": {str(k): v for k, v in lens_trace.items()},
        "h_in_patch_trace": {},
        "combined": {str(k): v for k, v in combined.items()},
    }

    for L in sample_layers:
        output["h_in_patch_trace"][str(L)] = {
            "r2c_effects": [v["effect"] for v in patch_results[L]["r2c"]],
            "c2r_effects": [v["effect"] for v in patch_results[L]["c2r"]],
            "r2c_delta_gap": [v["delta_gap"] for v in patch_results[L]["r2c"]],
            "c2r_delta_gap": [v["delta_gap"] for v in patch_results[L]["c2r"]],
        }

    os.makedirs("results/phase362_binding_trace", exist_ok=True)
    out_path = f"results/phase362_binding_trace/{model_name}_phase362.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=str, ensure_ascii=False)
    log(f"\n  Saved to {out_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    log(f"Phase 362 complete for {model_name} in {time.time()-t0:.0f}s")
    return output


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_experiment(model_name)
