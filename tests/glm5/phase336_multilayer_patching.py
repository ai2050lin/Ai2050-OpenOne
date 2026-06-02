"""
Phase 336+337+338: Multi-layer Block Patching + Reverse Destruction + Cross-layer Attention
===========================================================================================

Three experiments in one script (shared model loading, one model at a time):

Phase 336: Multi-layer MLP block patching
  Goal: Verify binding is distributed MLP contract
  Method: Patch multiple consecutive MLP layers simultaneously (corrupted→clean)
  Expect: Multi-layer MLP recovery >> single-layer MLP recovery

Phase 337: Reverse destruction (clean→corrupted)
  Goal: Prove MLP causal NECESSITY (not just sufficiency)
  Method: Run clean input, replace component outputs with corrupted versions
  Expect: MLP reverse destruction >> attention reverse destruction

Phase 338: Cross-layer attention indirect effects
  Goal: Test if early attention routes object identity to later MLP
  Method: Patch early attention layers with clean outputs, measure final binding
  Expect: Based on Phase 334, early attn effect should be small

Block definitions:
  Qwen3  (36L): MLP [L21-23, L24-26, L27-29, L21-29], early attn [L0-8]
  GLM4   (40L): MLP [L30-34, L35-38, L30-38], early attn [L0-10]
  DS7B   (28L): MLP [L19-21, L22-24, L19-24], early attn [L0-8]

Usage:
  python tests/glm5/phase336_multilayer_patching.py qwen3
  python tests/glm5/phase336_multilayer_patching.py glm4
  python tests/glm5/phase336_multilayer_patching.py deepseek7b
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

# Multi-layer block definitions
BLOCK_CONFIGS = {
    "qwen3": {
        # Phase 336: MLP blocks at binding layers
        "mlp_blocks": [
            {"name": "L21-23", "layers": list(range(21, 24))},
            {"name": "L24-26", "layers": list(range(24, 27))},
            {"name": "L27-29", "layers": list(range(27, 30))},
            {"name": "L21-29", "layers": list(range(21, 30))},
        ],
        # Phase 338: Early attention blocks
        "early_attn_blocks": [
            {"name": "L0-8", "layers": list(range(0, 9))},
        ],
        # Phase 337: Key layers for reverse destruction
        "key_layers": [25, 29],
    },
    "glm4": {
        "mlp_blocks": [
            {"name": "L30-34", "layers": list(range(30, 35))},
            {"name": "L35-38", "layers": list(range(35, 39))},
            {"name": "L30-38", "layers": list(range(30, 39))},
        ],
        "early_attn_blocks": [
            {"name": "L0-10", "layers": list(range(0, 11))},
        ],
        "key_layers": [30, 38],
    },
    "deepseek7b": {
        "mlp_blocks": [
            {"name": "L19-21", "layers": list(range(19, 22))},
            {"name": "L22-24", "layers": list(range(22, 25))},
            {"name": "L19-24", "layers": list(range(19, 25))},
        ],
        "early_attn_blocks": [
            {"name": "L0-8", "layers": list(range(0, 9))},
        ],
        "key_layers": [21, 23],
    },
}

# Extended HC pairs (24 pairs, from Phase 334b)
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

CORRUPTED_PROMPT = "The item"

# Patch types for multi-layer blocks
BLOCK_PATCH_TYPES = ["mlp_block", "attn_block", "full_block"]

# Reverse destruction patch types
REVERSE_PATCH_TYPES = ["mlp_reverse", "attn_reverse", "full_reverse"]


# ===== Model Loading =====

def load_model_bf16(model_name):
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
        log(f"  Layer distribution: {gpu_layers} GPU + {cpu_layers} CPU")
        for lid in sorted(layer_devices.keys(), key=int)[-3:]:
            log(f"    Layer {lid}: {layer_devices[lid]}")

    return model, tokenizer, device


# ===== Utility Functions =====

def get_W_U(model, model_name):
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
    ids = tokenizer.encode(word, add_special_tokens=False)
    if not ids:
        return None
    return ids[0]


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError(f"Cannot find transformer layers in {type(model).__name__}")


# ===== Capture Function =====

def run_and_capture(model, tokenizer, device, prompt, n_layers):
    """Run model and capture all attn_outs and mlp_outs at ALL layers."""
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


# ===== Multi-layer Patched Run =====

def run_patched_multilayer(model, tokenizer, device, base_prompt,
                           patch_specs, n_layers):
    """Run model with patches at multiple layers simultaneously.

    Args:
        base_prompt: The input prompt to use (corrupted for recovery, clean for destruction)
        patch_specs: List of (layer_idx, component_type, replacement_tensor) tuples
            component_type: "attn" or "mlp"
            replacement_tensor: The tensor to replace the component output with [1, seq_len, d_model]
        n_layers: Number of transformer layers

    Returns:
        final_hidden: numpy [d_model]
    """
    layers = get_layers(model)
    hooks = []

    def make_patch_hook(replacement):
        def hook(module, input, output):
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

    # Register hooks for all patch specifications
    for layer_idx, comp_type, replacement in patch_specs:
        layer = layers[layer_idx]
        if comp_type == "attn" and hasattr(layer, 'self_attn'):
            hooks.append(layer.self_attn.register_forward_hook(make_patch_hook(replacement)))
        elif comp_type == "mlp" and hasattr(layer, 'mlp'):
            hooks.append(layer.mlp.register_forward_hook(make_patch_hook(replacement)))

    # Run forward pass
    inp = tokenizer(base_prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    for h in hooks:
        h.remove()

    final_hidden = out.hidden_states[-1][0, -1].detach().cpu().float().numpy()
    return final_hidden


# ===== Build Patch Specifications =====

def build_block_patch_specs(block, patch_type, source_attn_outs, source_mlp_outs):
    """Build patch_specs for a multi-layer block patching condition.

    Args:
        block: dict with "name" and "layers" keys
        patch_type: "mlp_block", "attn_block", or "full_block"
        source_attn_outs: dict {layer_idx: tensor} — source attention outputs
        source_mlp_outs: dict {layer_idx: tensor} — source MLP outputs

    Returns:
        list of (layer_idx, comp_type, replacement_tensor) tuples
    """
    specs = []
    for li in block["layers"]:
        if patch_type in ("attn_block", "full_block"):
            if li in source_attn_outs:
                specs.append((li, "attn", source_attn_outs[li]))
        if patch_type in ("mlp_block", "full_block"):
            if li in source_mlp_outs:
                specs.append((li, "mlp", source_mlp_outs[li]))
    return specs


def build_single_reverse_specs(layer_idx, patch_type, source_attn_outs, source_mlp_outs):
    """Build patch_specs for a single-layer reverse destruction condition.

    Args:
        layer_idx: The layer to patch
        patch_type: "mlp_reverse", "attn_reverse", or "full_reverse"
        source_attn_outs: dict {layer_idx: tensor} — corrupted attention outputs
        source_mlp_outs: dict {layer_idx: tensor} — corrupted MLP outputs

    Returns:
        list of (layer_idx, comp_type, replacement_tensor) tuples
    """
    specs = []
    if patch_type in ("attn_reverse", "full_reverse"):
        if layer_idx in source_attn_outs:
            specs.append((layer_idx, "attn", source_attn_outs[layer_idx]))
    if patch_type in ("mlp_reverse", "full_reverse"):
        if layer_idx in source_mlp_outs:
            specs.append((layer_idx, "mlp", source_mlp_outs[layer_idx]))
    return specs


# ===== Main Experiment =====

def run_experiment(model_name):
    log(f"Phase 336+337+338: Multi-layer Patching + Reverse Destruction — {model_name}")
    log("=" * 70)

    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    cfg = MODEL_CONFIGS[model_name]
    n_layers = cfg["n_layers"]
    block_cfg = BLOCK_CONFIGS[model_name]

    W_U = get_W_U(model, model_name)
    log(f"  W_U shape: {W_U.shape}")

    if torch.cuda.is_available():
        log(f"  GPU after load: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # ============================================================
    # Step 1: Run corrupted baseline ONCE
    # ============================================================
    log(f"\n=== Step 1: Corrupted baseline ('{CORRUPTED_PROMPT}') ===")
    corrupted_attn_outs, corrupted_mlp_outs, corrupted_hidden, corrupted_seq_len = \
        run_and_capture(model, tokenizer, device, CORRUPTED_PROMPT, n_layers)
    log(f"  Captured: {len(corrupted_attn_outs)} attn, {len(corrupted_mlp_outs)} mlp")

    if torch.cuda.is_available():
        log(f"  GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # ============================================================
    # Step 2: Per-pair experiments
    # ============================================================
    log(f"\n=== Step 2: Per-pair experiments ({len(HC_PAIRS)} pairs, filtered) ===")

    all_results = []
    filtered_count = 0

    # Count total conditions for progress
    mlp_blocks = block_cfg["mlp_blocks"]
    early_attn_blocks = block_cfg["early_attn_blocks"]
    key_layers = block_cfg["key_layers"]
    n_block_conds = len(mlp_blocks) * len(BLOCK_PATCH_TYPES) + len(early_attn_blocks)
    n_reverse_conds = len(key_layers) * len(REVERSE_PATCH_TYPES)
    n_total_conds = n_block_conds + n_reverse_conds
    log(f"  Conditions per pair: {n_block_conds} block + {n_reverse_conds} reverse = {n_total_conds}")

    for pidx, (obj, target_val, competitor_val) in enumerate(HC_PAIRS):
        pair_key = f"{obj}_{target_val}"

        # Get token IDs
        tid_t = get_token_id(tokenizer, target_val)
        tid_c = get_token_id(tokenizer, competitor_val)
        if tid_t is None or tid_c is None:
            continue

        binding_dir = W_U[tid_t] - W_U[tid_c]

        # Run clean capture
        clean_prompt = f"The {obj}"
        clean_attn_outs, clean_mlp_outs, clean_hidden, clean_seq_len = \
            run_and_capture(model, tokenizer, device, clean_prompt, n_layers)

        # Check sequence length compatibility
        if clean_seq_len != corrupted_seq_len:
            log(f"  [{pidx+1}] {pair_key}: SKIP (seq_len mismatch)")
            del clean_attn_outs, clean_mlp_outs
            gc.collect()
            torch.cuda.empty_cache()
            continue

        # Compute baselines
        binding_clean = float(binding_dir @ clean_hidden)
        binding_corrupted = float(binding_dir @ corrupted_hidden)
        binding_range = binding_clean - binding_corrupted

        # Filter pairs with small/negative binding_range
        if binding_range < 0.3:
            filtered_count += 1
            if pidx < 5 or (pidx + 1) % 6 == 0:
                log(f"  [{pidx+1}] {pair_key}: FILTERED (range={binding_range:+.3f})")
            del clean_attn_outs, clean_mlp_outs
            gc.collect()
            torch.cuda.empty_cache()
            continue

        pair_result = {
            "obj": obj, "target_val": target_val, "competitor_val": competitor_val,
            "binding_clean": round(binding_clean, 4),
            "binding_corrupted": round(binding_corrupted, 4),
            "binding_range": round(binding_range, 4),
            "patches": {},
        }

        # ---- Phase 336: Multi-layer block patching (corrupted → clean) ----
        for block in mlp_blocks:
            for pt in BLOCK_PATCH_TYPES:
                cond_name = f"{block['name']}_{pt}"
                try:
                    specs = build_block_patch_specs(block, pt, clean_attn_outs, clean_mlp_outs)
                    if not specs:
                        continue
                    patched_hidden = run_patched_multilayer(
                        model, tokenizer, device, CORRUPTED_PROMPT,
                        specs, n_layers,
                    )
                    binding_patched = float(binding_dir @ patched_hidden)
                    recovery_pct = 100.0 * (binding_patched - binding_corrupted) / max(binding_range, 1e-10)
                    pair_result["patches"][cond_name] = {
                        "binding": round(binding_patched, 4),
                        "recovery_pct": round(recovery_pct, 1),
                    }
                except Exception as e:
                    pair_result["patches"][cond_name] = {"error": str(e)}

        # ---- Phase 338: Early attention block patching (corrupted → clean) ----
        for block in early_attn_blocks:
            cond_name = f"{block['name']}_attn_early"
            try:
                specs = []
                for li in block["layers"]:
                    if li in clean_attn_outs:
                        specs.append((li, "attn", clean_attn_outs[li]))
                if not specs:
                    continue
                patched_hidden = run_patched_multilayer(
                    model, tokenizer, device, CORRUPTED_PROMPT,
                    specs, n_layers,
                )
                binding_patched = float(binding_dir @ patched_hidden)
                recovery_pct = 100.0 * (binding_patched - binding_corrupted) / max(binding_range, 1e-10)
                pair_result["patches"][cond_name] = {
                    "binding": round(binding_patched, 4),
                    "recovery_pct": round(recovery_pct, 1),
                }
            except Exception as e:
                pair_result["patches"][cond_name] = {"error": str(e)}

        # ---- Phase 337: Reverse destruction (clean → corrupted) ----
        for kl in key_layers:
            for rpt in REVERSE_PATCH_TYPES:
                cond_name = f"L{kl}_{rpt}"
                try:
                    specs = build_single_reverse_specs(
                        kl, rpt, corrupted_attn_outs, corrupted_mlp_outs)
                    if not specs:
                        continue
                    # Run with CLEAN prompt, patch with CORRUPTED outputs
                    patched_hidden = run_patched_multilayer(
                        model, tokenizer, device, clean_prompt,
                        specs, n_layers,
                    )
                    binding_patched = float(binding_dir @ patched_hidden)
                    # Destruction metric: how much of binding_clean→binding_corrupted gap is explained
                    destruction_pct = 100.0 * (binding_clean - binding_patched) / max(binding_range, 1e-10)
                    pair_result["patches"][cond_name] = {
                        "binding": round(binding_patched, 4),
                        "destruction_pct": round(destruction_pct, 1),
                    }
                except Exception as e:
                    pair_result["patches"][cond_name] = {"error": str(e)}

        all_results.append(pair_result)

        # Free per-pair memory
        del clean_attn_outs, clean_mlp_outs
        gc.collect()
        torch.cuda.empty_cache()

        # Progress logging
        if (pidx + 1) % 4 == 0 or pidx < 2:
            elapsed = time.time() - t0
            gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            log(f"  [{pidx+1}/{len(HC_PAIRS)}] {pair_key}: "
                f"valid={len(all_results)}, filtered={filtered_count}, "
                f"elapsed={elapsed:.0f}s, GPU={gpu_mem:.2f}GB")

    # ============================================================
    # Step 3: Aggregate results
    # ============================================================
    log(f"\n{'='*80}")
    log(f"AGGREGATE RESULTS — {model_name}")
    log(f"{'='*80}")

    if not all_results:
        log("  No valid pairs after filtering!")
        del model, W_U, corrupted_attn_outs, corrupted_mlp_outs
        gc.collect()
        torch.cuda.empty_cache()
        return

    n_valid = len(all_results)
    avg_clean = float(np.mean([r["binding_clean"] for r in all_results]))
    avg_corrupted = float(np.mean([r["binding_corrupted"] for r in all_results]))
    avg_range = float(np.mean([r["binding_range"] for r in all_results]))

    log(f"  Valid pairs: {n_valid}/{len(HC_PAIRS)} (filtered={filtered_count})")
    log(f"  Baselines: clean={avg_clean:+.4f}, corrupted={avg_corrupted:+.4f}, range={avg_range:+.4f}")

    # ---- Phase 336: Block recovery results ----
    log(f"\n--- Phase 336: Multi-layer Block Recovery (corrupted→clean) ---")
    log(f"  {'Block':>10} {'mlp_block':>10} {'attn_block':>10} {'full_block':>10} {'mlp vs full':>12}")
    log("  " + "-" * 58)

    block_aggs = {}
    for block in mlp_blocks:
        bname = block["name"]
        bagg = {}
        for pt in BLOCK_PATCH_TYPES:
            cond = f"{bname}_{pt}"
            recs = [r["patches"][cond]["recovery_pct"] for r in all_results
                    if cond in r["patches"] and "recovery_pct" in r["patches"][cond]]
            if recs:
                bagg[pt] = {
                    "mean": round(float(np.mean(recs)), 1),
                    "std": round(float(np.std(recs)), 1),
                    "n": len(recs),
                }

        mlp_rec = bagg.get("mlp_block", {}).get("mean", float('nan'))
        attn_rec = bagg.get("attn_block", {}).get("mean", float('nan'))
        full_rec = bagg.get("full_block", {}).get("mean", float('nan'))
        mlp_vs_full = mlp_rec / max(abs(full_rec), 0.1) * 100 if abs(full_rec) > 0.1 else float('nan')

        log(f"  {bname:>10} {mlp_rec:>+10.1f} {attn_rec:>+10.1f} {full_rec:>+10.1f} {mlp_vs_full:>11.1f}%")
        block_aggs[bname] = bagg

    # ---- Phase 338: Early attention results ----
    log(f"\n--- Phase 338: Early Attention Recovery (corrupted→clean) ---")
    for block in early_attn_blocks:
        cond = f"{block['name']}_attn_early"
        recs = [r["patches"][cond]["recovery_pct"] for r in all_results
                if cond in r["patches"] and "recovery_pct" in r["patches"][cond]]
        if recs:
            avg_rec = float(np.mean(recs))
            std_rec = float(np.std(recs))
            log(f"  {block['name']}_attn_early: recovery={avg_rec:+.1f}% (std={std_rec:.1f}%, n={len(recs)})")

    # ---- Phase 337: Reverse destruction results ----
    log(f"\n--- Phase 337: Reverse Destruction (clean→corrupted) ---")
    log(f"  {'Layer':>6} {'mlp_reverse':>13} {'attn_reverse':>14} {'full_reverse':>14} {'mlp/attn':>10}")
    log("  " + "-" * 62)

    reverse_aggs = {}
    for kl in key_layers:
        klaggs = {}
        for rpt in REVERSE_PATCH_TYPES:
            cond = f"L{kl}_{rpt}"
            dests = [r["patches"][cond]["destruction_pct"] for r in all_results
                     if cond in r["patches"] and "destruction_pct" in r["patches"][cond]]
            if dests:
                klaggs[rpt] = {
                    "mean": round(float(np.mean(dests)), 1),
                    "std": round(float(np.std(dests)), 1),
                    "n": len(dests),
                }

        mlp_dest = klaggs.get("mlp_reverse", {}).get("mean", float('nan'))
        attn_dest = klaggs.get("attn_reverse", {}).get("mean", float('nan'))
        full_dest = klaggs.get("full_reverse", {}).get("mean", float('nan'))
        mlp_attn_ratio = mlp_dest / max(abs(attn_dest), 0.1) if abs(attn_dest) > 0.1 else float('nan')

        log(f"  L{kl:>5} {mlp_dest:>+13.1f} {attn_dest:>+14.1f} {full_dest:>+14.1f} {mlp_attn_ratio:>9.1f}x")
        reverse_aggs[f"L{kl}"] = klaggs

    # ---- Key comparison: Single vs Multi-layer MLP recovery ----
    log(f"\n--- Key Comparison: Single-layer (Phase 334b) vs Multi-layer (Phase 336) MLP Recovery ---")
    # Find the largest block recovery
    best_block = None
    best_mlp_rec = -999
    for block in mlp_blocks:
        bname = block["name"]
        if bname in block_aggs and "mlp_block" in block_aggs[bname]:
            rec = block_aggs[bname]["mlp_block"]["mean"]
            if rec > best_mlp_rec:
                best_mlp_rec = rec
                best_block = bname

    if best_block:
        log(f"  Best multi-layer MLP block: {best_block} = {best_mlp_rec:+.1f}%")
        log(f"  (Compare with Phase 334b single-layer: Qwen3 L25=7.9%, GLM4 L38=28.8%, DS7B L23=15.0%)")

    # ---- Key comparison: Recovery vs Destruction symmetry ----
    log(f"\n--- Recovery vs Destruction Symmetry Check ---")
    for kl in key_layers:
        # Recovery: from Phase 336, look for single-layer equivalent
        # For now, use the smallest block that includes this layer
        recovery_val = float('nan')
        for block in mlp_blocks:
            if kl in block["layers"] and block["name"] in block_aggs:
                if "mlp_block" in block_aggs[block["name"]]:
                    # This is a multi-layer block, not directly comparable
                    pass

        # Destruction: from Phase 337
        dest_key = f"L{kl}"
        mlp_dest = reverse_aggs.get(dest_key, {}).get("mlp_reverse", {}).get("mean", float('nan'))

        log(f"  L{kl}: MLP destruction = {mlp_dest:+.1f}%")

    # ============================================================
    # Step 4: Save results
    # ============================================================
    save_data = {
        "model": model_name,
        "n_valid_pairs": n_valid,
        "n_filtered": filtered_count,
        "avg_binding_clean": round(avg_clean, 4),
        "avg_binding_corrupted": round(avg_corrupted, 4),
        "avg_binding_range": round(avg_range, 4),
        "block_aggs": block_aggs,
        "reverse_aggs": reverse_aggs,
        "details": all_results,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

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

    os.makedirs("results/phase336_multilayer", exist_ok=True)
    out_path = f"results/phase336_multilayer/{model_name}_phase336.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    log(f"\nResults saved to {out_path}")

    # Release model
    del model, W_U, corrupted_attn_outs, corrupted_mlp_outs
    gc.collect()
    torch.cuda.empty_cache()

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
    log("Phase 336+337+338 complete!")
