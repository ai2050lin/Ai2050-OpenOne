"""
Phase 334+335: Component Causal Patching — Direct vs Indirect Effects
=====================================================================

Phase 333: MLP contributes 80-99% at key binding layers (ATTRIBUTION).
This script: Is MLP CAUSALLY necessary? Is Attention's effect direct or indirect?

Method: Activation patching at key layers.
- Clean input: "The {object}" (e.g., "The apple") -> high binding signal
- Corrupted input: "The item" -> low binding signal
- Patched runs: Run corrupted but replace specific component outputs with clean versions

Patch conditions per layer:
1. attn_patch: Replace attn_out with clean -> direct + indirect attn effect
2. mlp_patch: Replace mlp_out with clean -> direct MLP effect
3. attn_direct_only: Replace attn_out with clean + freeze mlp_out to corrupted -> ONLY direct attn effect
4. full_block: Replace both attn_out + mlp_out with clean -> full layer effect

Baselines:
5. clean: "The {object}" -> binding_clean
6. corrupted: "The item" -> binding_corrupted

Recovery metric:
  recovery_pct = (binding_patched - binding_corrupted) / (binding_clean - binding_corrupted) * 100

Key comparisons:
- If mlp_patch has high recovery -> MLP is causally necessary
- If attn_patch has low recovery -> Attention is not directly necessary
- If attn_patch > attn_direct_only -> Attention has indirect effect through MLP
- indirect_attn_pct = attn_patch_recovery - attn_direct_only_recovery

Usage:
  python tests/glm5/phase334_causal_patching.py qwen3
  python tests/glm5/phase334_causal_patching.py glm4
  python tests/glm5/phase334_causal_patching.py deepseek7b
"""
import sys, os, time, json, gc, traceback
import torch
import numpy as np
from datetime import datetime
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# ===== Configuration =====

MODEL_CONFIGS = {
    "qwen3": {
        "path": "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
        "n_layers": 36, "d_model": 2560,
    },
    "glm4": {
        "path": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
        "n_layers": 40, "d_model": 4096,
    },
    "deepseek7b": {
        "path": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "n_layers": 28, "d_model": 3584,
    },
}

# Patch layers: key binding layer + surrounding + early/mid control
# Excluding last 1-2 layers (logit lens explosion from Phase 333)
PATCH_LAYERS = {
    "qwen3": [5, 15, 25, 27, 29, 31, 33],      # key=L29, exclude L35
    "glm4": [5, 20, 30, 34, 36, 38],             # key=L38, exclude L39 (last-1)
    "deepseek7b": [5, 14, 18, 21, 23, 25],       # key=L23, exclude L27 (last)
}

# HC test pairs (12) — primary test
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
]

# NI test pairs (6) — incompatibility suppression test
NI_PAIRS = [
    ("apple", "blue", "black"),
    ("snow", "pink", "orange"),
    ("banana", "white", "black"),
    ("grass", "yellow", "purple"),
    ("sky", "red", "brown"),
    ("fire", "blue", "green"),
]

ALL_PAIRS = [(o, t, c, "HC") for o, t, c in HC_PAIRS] + \
            [(o, t, c, "NI") for o, t, c in NI_PAIRS]

CORRUPTED_PROMPT = "The item"

PATCH_TYPES = ["attn", "mlp", "attn_direct_only", "full_block"]


# ===== Model Loading =====

def load_model_bf16(model_name):
    """Load model with BF16 + device_map='auto', trying flash_attention_2 first."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]

    log(f"  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = None
    for impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            log(f"  Trying attn_implementation={impl}...")
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True,
                attn_implementation=impl,
            )
            log(f"  Loaded {model_name} with attn_impl={impl}")
            break
        except Exception as e:
            log(f"  Failed with {impl}: {e}")
            continue

    if model is None:
        raise RuntimeError(f"Failed to load {model_name}")

    model.eval()
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log(f"  Model: {type(model).__name__}, device={device}, GPU={gpu_mem:.2f}GB")

    # Show layer distribution
    if hasattr(model, 'hf_device_map'):
        dmap = model.hf_device_map
        layer_devices = {}
        for k, v in dmap.items():
            if k.startswith('model.layers.'):
                lid = k.split('.')[2]
                if lid not in layer_devices:
                    layer_devices[lid] = str(v)
        gpu_layers = sum(1 for v in layer_devices.values() if 'cuda' in v)
        cpu_layers = sum(1 for v in layer_devices.values() if 'cpu' in v)
        log(f"  Layer distribution: {gpu_layers} GPU + {cpu_layers} CPU (total {len(layer_devices)})")
        # Show deep layer devices
        for lid in sorted(layer_devices.keys(), key=int)[-3:]:
            log(f"    Layer {lid}: {layer_devices[lid]}")

    return model, tokenizer, device


# ===== Utility Functions =====

def get_W_U(model, model_name):
    """Get lm_head weight matrix."""
    if hasattr(model, "lm_head"):
        w = model.lm_head.weight
        if not w.is_meta:
            return w.detach().cpu().float().numpy()
    import glob
    from safetensors import safe_open
    model_path = MODEL_CONFIGS[model_name]["path"]
    sf_files = glob.glob(os.path.join(model_path, '*.safetensors'))
    for sf_file in sf_files:
        with safe_open(sf_file, framework='pt', device='cpu') as sf:
            if 'lm_head.weight' in sf.keys():
                w = sf.get_tensor('lm_head.weight')
                log(f"  Loaded lm_head from {os.path.basename(sf_file)}")
                return w.float().numpy()
    raise ValueError(f"Cannot load lm_head for {model_name}")


def get_token_id(tokenizer, word):
    """Get first token ID for a word."""
    ids = tokenizer.encode(word, add_special_tokens=False)
    if not ids:
        return None
    if len(ids) > 1:
        log(f"    WARN: '{word}' -> {len(ids)} tokens, using first")
    return ids[0]


def get_layers(model):
    """Get transformer layer list."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError(f"Cannot find transformer layers in {type(model).__name__}")


# ===== Capture Function =====

def run_and_capture(model, tokenizer, device, prompt, n_layers):
    """Run model and capture all attn_outs and mlp_outs at ALL layers.

    Returns:
        attn_outs: dict {layer_idx: tensor [1, seq_len, d_model] on CPU}
        mlp_outs: dict {layer_idx: tensor [1, seq_len, d_model] on CPU}
        final_hidden: numpy [d_model] — last token of last layer
        seq_len: int
    """
    captured = {}
    layers = get_layers(model)

    def make_hook(key):
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured[key] = output[0].detach().cpu()
            else:
                captured[key] = output.detach().cpu()
        return hook

    hooks = []
    for li in range(n_layers):
        layer = layers[li]
        if hasattr(layer, 'self_attn'):
            hooks.append(layer.self_attn.register_forward_hook(make_hook(f"attn_{li}")))
        if hasattr(layer, 'mlp'):
            hooks.append(layer.mlp.register_forward_hook(make_hook(f"mlp_{li}")))

    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    for h in hooks:
        h.remove()

    attn_outs = {}
    mlp_outs = {}
    for li in range(n_layers):
        if f"attn_{li}" in captured:
            attn_outs[li] = captured[f"attn_{li}"]
        if f"mlp_{li}" in captured:
            mlp_outs[li] = captured[f"mlp_{li}"]

    final_hidden = out.hidden_states[-1][0, -1].detach().cpu().float().numpy()
    seq_len = inp["input_ids"].shape[1]

    return attn_outs, mlp_outs, final_hidden, seq_len


# ===== Patched Run =====

def run_patched(model, tokenizer, device, corrupted_prompt,
                clean_attn_outs, clean_mlp_outs,
                corrupted_attn_outs, corrupted_mlp_outs,
                patch_type, patch_layer, n_layers):
    """Run corrupted input with specific patching at patch_layer.

    Patch types:
    - "attn": Replace attn_out with clean (MLP recomputes naturally)
              -> measures direct + indirect attn effect
    - "mlp": Replace mlp_out with clean
             -> measures direct MLP effect
    - "attn_direct_only": Replace attn_out with clean + freeze mlp_out to corrupted
                          -> measures ONLY direct attn effect (no indirect through MLP)
    - "full_block": Replace both attn_out + mlp_out with clean
                    -> measures full layer effect

    Returns:
        final_hidden: numpy [d_model]
    """
    layers = get_layers(model)
    hooks = []

    def make_patch_hook(replacement):
        """Create a hook that replaces the module output with `replacement`."""
        def hook(module, input, output):
            # Determine target device and dtype from original output
            if isinstance(output, tuple):
                target_device = output[0].device
                target_dtype = output[0].dtype
            else:
                target_device = output.device
                target_dtype = output.dtype

            rep = replacement.to(device=target_device, dtype=target_dtype)

            if isinstance(output, tuple):
                return (rep,) + output[1:]
            return rep
        return hook

    layer = layers[patch_layer]

    if patch_type == "attn":
        # Patch attn_out with clean; MLP recomputes on new residual naturally
        if patch_layer in clean_attn_outs and hasattr(layer, 'self_attn'):
            hooks.append(layer.self_attn.register_forward_hook(
                make_patch_hook(clean_attn_outs[patch_layer])))

    elif patch_type == "mlp":
        # Patch mlp_out with clean
        if patch_layer in clean_mlp_outs and hasattr(layer, 'mlp'):
            hooks.append(layer.mlp.register_forward_hook(
                make_patch_hook(clean_mlp_outs[patch_layer])))

    elif patch_type == "attn_direct_only":
        # Patch attn_out with clean + freeze mlp_out to corrupted
        # This isolates the DIRECT attn effect (MLP doesn't recompute)
        if patch_layer in clean_attn_outs and hasattr(layer, 'self_attn'):
            hooks.append(layer.self_attn.register_forward_hook(
                make_patch_hook(clean_attn_outs[patch_layer])))
        if patch_layer in corrupted_mlp_outs and hasattr(layer, 'mlp'):
            hooks.append(layer.mlp.register_forward_hook(
                make_patch_hook(corrupted_mlp_outs[patch_layer])))

    elif patch_type == "full_block":
        # Patch both attn_out and mlp_out with clean
        if patch_layer in clean_attn_outs and hasattr(layer, 'self_attn'):
            hooks.append(layer.self_attn.register_forward_hook(
                make_patch_hook(clean_attn_outs[patch_layer])))
        if patch_layer in clean_mlp_outs and hasattr(layer, 'mlp'):
            hooks.append(layer.mlp.register_forward_hook(
                make_patch_hook(clean_mlp_outs[patch_layer])))

    # Run forward pass
    inp = tokenizer(corrupted_prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    for h in hooks:
        h.remove()

    final_hidden = out.hidden_states[-1][0, -1].detach().cpu().float().numpy()
    return final_hidden


# ===== Main Experiment =====

def run_experiment(model_name):
    log(f"Phase 334+335: Component Causal Patching — {model_name}")
    log("=" * 70)

    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    cfg = MODEL_CONFIGS[model_name]
    n_layers = cfg["n_layers"]
    patch_layers = PATCH_LAYERS[model_name]

    W_U = get_W_U(model, model_name)
    log(f"  W_U shape: {W_U.shape}")

    if torch.cuda.is_available():
        log(f"  GPU after load: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # ============================================================
    # Step 1: Run corrupted baseline ONCE (shared across all pairs)
    # ============================================================
    log(f"\n=== Step 1: Corrupted baseline ('{CORRUPTED_PROMPT}') ===")
    corrupted_attn_outs, corrupted_mlp_outs, corrupted_hidden, corrupted_seq_len = \
        run_and_capture(model, tokenizer, device, CORRUPTED_PROMPT, n_layers)
    log(f"  Captured: {len(corrupted_attn_outs)} attn, {len(corrupted_mlp_outs)} mlp, seq_len={corrupted_seq_len}")

    if torch.cuda.is_available():
        log(f"  GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # ============================================================
    # Step 2: Per-pair experiments
    # ============================================================
    log(f"\n=== Step 2: Per-pair experiments ({len(ALL_PAIRS)} pairs) ===")

    all_results = []
    n_patch_runs = len(patch_layers) * len(PATCH_TYPES)

    for pidx, (obj, target_val, competitor_val, compat_level) in enumerate(ALL_PAIRS):
        pair_key = f"{obj}_{target_val}"
        log(f"\n  [{pidx+1}/{len(ALL_PAIRS)}] {pair_key} ({compat_level})")

        # Get token IDs
        tid_t = get_token_id(tokenizer, target_val)
        tid_c = get_token_id(tokenizer, competitor_val)
        if tid_t is None or tid_c is None:
            log(f"    SKIP: token not found")
            continue

        binding_dir = W_U[tid_t] - W_U[tid_c]  # (d_model,)

        # Run clean (The {object})
        clean_prompt = f"The {obj}"
        clean_attn_outs, clean_mlp_outs, clean_hidden, clean_seq_len = \
            run_and_capture(model, tokenizer, device, clean_prompt, n_layers)

        # Check sequence length compatibility
        if clean_seq_len != corrupted_seq_len:
            log(f"    SKIP: seq_len mismatch (clean={clean_seq_len}, corrupted={corrupted_seq_len})")
            del clean_attn_outs, clean_mlp_outs
            gc.collect()
            torch.cuda.empty_cache()
            continue

        # Compute baselines
        binding_clean = float(binding_dir @ clean_hidden)
        binding_corrupted = float(binding_dir @ corrupted_hidden)
        binding_range = binding_clean - binding_corrupted

        log(f"    binding_clean={binding_clean:+.4f}, binding_corrupted={binding_corrupted:+.4f}, "
            f"range={binding_range:+.4f}")

        pair_result = {
            "obj": obj,
            "target_val": target_val,
            "competitor_val": competitor_val,
            "compat_level": compat_level,
            "binding_clean": round(binding_clean, 4),
            "binding_corrupted": round(binding_corrupted, 4),
            "binding_range": round(binding_range, 4),
            "patches": {},
        }

        # Run all patch conditions
        patch_count = 0
        for patch_layer in patch_layers:
            for patch_type in PATCH_TYPES:
                patch_key = f"L{patch_layer}_{patch_type}"

                try:
                    patched_hidden = run_patched(
                        model, tokenizer, device, CORRUPTED_PROMPT,
                        clean_attn_outs, clean_mlp_outs,
                        corrupted_attn_outs, corrupted_mlp_outs,
                        patch_type, patch_layer, n_layers,
                    )

                    binding_patched = float(binding_dir @ patched_hidden)
                    if abs(binding_range) > 1e-8:
                        recovery_pct = 100.0 * (binding_patched - binding_corrupted) / binding_range
                    else:
                        recovery_pct = 0.0

                    pair_result["patches"][patch_key] = {
                        "binding": round(binding_patched, 4),
                        "recovery_pct": round(recovery_pct, 1),
                    }
                    patch_count += 1

                except Exception as e:
                    log(f"    ERROR at {patch_key}: {e}")
                    pair_result["patches"][patch_key] = {"error": str(e)}

        all_results.append(pair_result)

        # Free per-pair memory
        del clean_attn_outs, clean_mlp_outs
        gc.collect()
        torch.cuda.empty_cache()

        # Progress logging
        if (pidx + 1) % 3 == 0 or pidx < 2:
            elapsed = time.time() - t0
            gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            log(f"  Progress: {pidx+1}/{len(ALL_PAIRS)}, "
                f"patches_done={patch_count}/{n_patch_runs}, "
                f"elapsed={elapsed:.0f}s, GPU={gpu_mem:.2f}GB")

    # ============================================================
    # Step 3: Aggregate results by compat_level
    # ============================================================
    log(f"\n{'='*80}")
    log(f"AGGREGATE RESULTS — {model_name}")
    log(f"{'='*80}")

    aggregates = {}
    for cl in ["HC", "NI"]:
        cl_results = [r for r in all_results if r["compat_level"] == cl]
        if not cl_results:
            continue

        # Baseline averages
        avg_clean = float(np.mean([r["binding_clean"] for r in cl_results]))
        avg_corrupted = float(np.mean([r["binding_corrupted"] for r in cl_results]))
        avg_range = float(np.mean([r["binding_range"] for r in cl_results]))

        cl_aggs = {
            "n_pairs": len(cl_results),
            "avg_binding_clean": round(avg_clean, 4),
            "avg_binding_corrupted": round(avg_corrupted, 4),
            "avg_binding_range": round(avg_range, 4),
        }

        # Per-patch-condition aggregation
        for patch_layer in patch_layers:
            for patch_type in PATCH_TYPES:
                patch_key = f"L{patch_layer}_{patch_type}"
                recoveries = []
                bindings = []
                for r in cl_results:
                    if patch_key in r["patches"] and "recovery_pct" in r["patches"][patch_key]:
                        recoveries.append(r["patches"][patch_key]["recovery_pct"])
                        bindings.append(r["patches"][patch_key]["binding"])

                if recoveries:
                    cl_aggs[patch_key] = {
                        "avg_recovery_pct": round(float(np.mean(recoveries)), 1),
                        "std_recovery_pct": round(float(np.std(recoveries)), 1),
                        "min_recovery": round(float(np.min(recoveries)), 1),
                        "max_recovery": round(float(np.max(recoveries)), 1),
                        "avg_binding": round(float(np.mean(bindings)), 4),
                        "n": len(recoveries),
                    }

        # Key decompositions per layer
        for patch_layer in patch_layers:
            attn_key = f"L{patch_layer}_attn"
            mlp_key = f"L{patch_layer}_mlp"
            direct_key = f"L{patch_layer}_attn_direct_only"
            full_key = f"L{patch_layer}_full_block"

            if attn_key in cl_aggs and mlp_key in cl_aggs:
                attn_rec = cl_aggs[attn_key]["avg_recovery_pct"]
                mlp_rec = cl_aggs[mlp_key]["avg_recovery_pct"]
                total = attn_rec + mlp_rec
                mlp_share = mlp_rec / max(abs(total), 0.1) * 100

                cl_aggs[f"L{patch_layer}_decomp"] = {
                    "attn_recovery": round(attn_rec, 1),
                    "mlp_recovery": round(mlp_rec, 1),
                    "total_recovery": round(total, 1),
                    "mlp_share_pct": round(mlp_share, 1),
                }

            if attn_key in cl_aggs and direct_key in cl_aggs:
                attn_total = cl_aggs[attn_key]["avg_recovery_pct"]
                attn_direct = cl_aggs[direct_key]["avg_recovery_pct"]
                attn_indirect = attn_total - attn_direct

                cl_aggs[f"L{patch_layer}_indirect"] = {
                    "attn_total": round(attn_total, 1),
                    "attn_direct": round(attn_direct, 1),
                    "attn_indirect": round(attn_indirect, 1),
                    "indirect_share_of_attn": round(
                        attn_indirect / max(abs(attn_total), 0.1) * 100, 1),
                }

        aggregates[cl] = cl_aggs

    # ============================================================
    # Print summary tables
    # ============================================================
    for cl in ["HC", "NI"]:
        if cl not in aggregates:
            continue
        cl_aggs = aggregates[cl]

        log(f"\n--- {cl} Summary ---")
        log(f"  Baselines: clean={cl_aggs['avg_binding_clean']:+.4f}, "
            f"corrupted={cl_aggs['avg_binding_corrupted']:+.4f}, "
            f"range={cl_aggs['avg_binding_range']:+.4f}")

        log(f"\n  Recovery % by patch type and layer:")
        log(f"  {'Layer':>5} {'attn':>8} {'mlp':>8} {'attn_dir':>9} {'full':>8} "
            f"{'mlp_share':>10} {'indirect':>10}")
        log("  " + "-" * 68)

        for patch_layer in patch_layers:
            vals = {}
            for pt in PATCH_TYPES:
                pk = f"L{patch_layer}_{pt}"
                if pk in cl_aggs:
                    vals[pt] = cl_aggs[pk]["avg_recovery_pct"]
                else:
                    vals[pt] = float('nan')

            decomp_key = f"L{patch_layer}_decomp"
            indirect_key = f"L{patch_layer}_indirect"
            mlp_share = cl_aggs[decomp_key]["mlp_share_pct"] if decomp_key in cl_aggs else float('nan')
            indirect_val = cl_aggs[indirect_key]["attn_indirect"] if indirect_key in cl_aggs else float('nan')

            log(f"  L{patch_layer:>4} {vals.get('attn', float('nan')):>+8.1f} "
                f"{vals.get('mlp', float('nan')):>+8.1f} "
                f"{vals.get('attn_direct_only', float('nan')):>+9.1f} "
                f"{vals.get('full_block', float('nan')):>+8.1f} "
                f"{mlp_share:>9.1f}% {indirect_val:>+9.1f}%")

    # ============================================================
    # Key findings
    # ============================================================
    log(f"\n=== KEY FINDINGS ===")

    # Key binding layer results
    key_layers = {
        "qwen3": 29, "glm4": 38, "deepseek7b": 23,
    }
    key_layer = key_layers[model_name]

    if "HC" in aggregates:
        hc = aggregates["HC"]
        mlp_key = f"L{key_layer}_mlp"
        attn_key = f"L{key_layer}_attn"
        direct_key = f"L{key_layer}_attn_direct_only"

        if mlp_key in hc:
            log(f"  Key layer L{key_layer} MLP recovery: {hc[mlp_key]['avg_recovery_pct']:+.1f}% "
                f"(std={hc[mlp_key]['std_recovery_pct']:.1f}%)")
        if attn_key in hc:
            log(f"  Key layer L{key_layer} attn recovery: {hc[attn_key]['avg_recovery_pct']:+.1f}% "
                f"(std={hc[attn_key]['std_recovery_pct']:.1f}%)")
        if direct_key in hc:
            log(f"  Key layer L{key_layer} attn_direct recovery: {hc[direct_key]['avg_recovery_pct']:+.1f}%")

        indirect_key = f"L{key_layer}_indirect"
        if indirect_key in hc:
            ind = hc[indirect_key]
            log(f"  Key layer L{key_layer} indirect attn effect: {ind['attn_indirect']:+.1f}% "
                f"({ind['indirect_share_of_attn']:.1f}% of total attn)")

    # Release model
    del model, W_U, corrupted_attn_outs, corrupted_mlp_outs
    gc.collect()
    torch.cuda.empty_cache()
    log(f"  GPU after cleanup: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # Save results
    save_data = {
        "model": model_name,
        "n_layers": n_layers,
        "patch_layers": patch_layers,
        "patch_types": PATCH_TYPES,
        "n_pairs": len(all_results),
        "aggregates": aggregates,
        "details": all_results,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj

    save_data = convert(save_data)

    os.makedirs("results/phase334_causal_patching", exist_ok=True)
    out_path = f"results/phase334_causal_patching/{model_name}_phase334.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    log(f"Results saved to {out_path}")

    total_time = time.time() - t0
    log(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f}min)")

    return save_data


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    if model_name not in MODEL_CONFIGS:
        log(f"Unknown model: {model_name}")
        log(f"Available: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    run_experiment(model_name)
    log("Phase 334+335 complete!")
