"""
Phase 339+340+341: Multi-baseline Validation + Pipeline Combination + Identity Probe
=====================================================================================

Three experiments in one script (shared model loading, one model at a time):

Phase 339: Multi-baseline validation
  Goal: Confirm MLP block recovery is robust across different corrupted baselines
  Method: Use "The item", "The thing", "The object", "The entity" as baselines
  Key test: MLP block recovery with each baseline
  Expect: MLP block > Attention block consistently across baselines

Phase 340: Identity + Computation combined block patching
  Goal: Test if early identity block + late computation block approaches 100%
  Method: Patch L0-L2 full + L21-L29 MLP simultaneously (corrupted→clean)
  Expect: Combined block recovery significantly higher than either alone

Phase 341: Identity probe
  Goal: Verify what L0-L2 residual stream encodes (object identity vs binding)
  Method: Linear probe on residual stream at each layer to predict object identity
  Key comparison: "The apple" vs "The item" — can we read out object category?
  Expect: L0-L2 should linearly encode object identity; later layers encode compatibility

Usage:
  python tests/glm5/phase339_multibaseline_pipeline.py qwen3
  python tests/glm5/phase339_multibaseline_pipeline.py glm4
  python tests/glm5/phase339_multibaseline_pipeline.py deepseek7b
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

# Multi-baseline corrupted prompts
CORRUPTED_BASELINES = ["The item", "The thing", "The object", "The entity"]

# Pipeline block definitions per model
PIPELINE_CONFIGS = {
    "qwen3": {
        # Phase 339: Key MLP blocks to test with each baseline
        "baseline_test_blocks": [
            {"name": "L21-29_mlp", "layers": list(range(21, 30)), "type": "mlp"},
            {"name": "L21-29_full", "layers": list(range(21, 30)), "type": "full"},
            {"name": "L21-29_attn", "layers": list(range(21, 30)), "type": "attn"},
        ],
        # Phase 340: Identity + Computation combined blocks
        "pipeline_blocks": [
            # Early identity block only
            {"name": "identity_L0-2_full", "layers": list(range(0, 3)), "type": "full"},
            # Late computation block only
            {"name": "compute_L21-29_mlp", "layers": list(range(21, 30)), "type": "mlp"},
            # COMBINED: identity + computation
            {"name": "identity+compute", "identity_layers": list(range(0, 3)),
             "compute_layers": list(range(21, 30)), "type": "combined"},
            # Wider computation spans
            {"name": "compute_L18-29_mlp", "layers": list(range(18, 30)), "type": "mlp"},
            {"name": "compute_L15-29_mlp", "layers": list(range(15, 30)), "type": "mlp"},
        ],
        # Phase 341: Identity probe layers
        "probe_layers": [0, 1, 2, 3, 4, 5, 8, 12, 18, 24, 29, 35],
        # Default baseline for Phase 340/341
        "default_baseline": "The item",
    },
    "glm4": {
        "baseline_test_blocks": [
            {"name": "L30-38_mlp", "layers": list(range(30, 39)), "type": "mlp"},
            {"name": "L30-38_full", "layers": list(range(30, 39)), "type": "full"},
            {"name": "L30-38_attn", "layers": list(range(30, 39)), "type": "attn"},
        ],
        "pipeline_blocks": [
            {"name": "identity_L0-4_full", "layers": list(range(0, 5)), "type": "full"},
            {"name": "compute_L30-38_mlp", "layers": list(range(30, 39)), "type": "mlp"},
            {"name": "identity+compute", "identity_layers": list(range(0, 5)),
             "compute_layers": list(range(30, 39)), "type": "combined"},
            {"name": "compute_L25-38_mlp", "layers": list(range(25, 39)), "type": "mlp"},
            {"name": "compute_L20-38_mlp", "layers": list(range(20, 39)), "type": "mlp"},
        ],
        "probe_layers": [0, 1, 2, 3, 4, 5, 8, 12, 20, 30, 38, 39],
        "default_baseline": "The item",
    },
    "deepseek7b": {
        "baseline_test_blocks": [
            {"name": "L19-24_mlp", "layers": list(range(19, 25)), "type": "mlp"},
            {"name": "L19-24_full", "layers": list(range(19, 25)), "type": "full"},
            {"name": "L19-24_attn", "layers": list(range(19, 25)), "type": "attn"},
        ],
        "pipeline_blocks": [
            {"name": "identity_L0-2_full", "layers": list(range(0, 3)), "type": "full"},
            {"name": "compute_L19-24_mlp", "layers": list(range(19, 25)), "type": "mlp"},
            {"name": "identity+compute", "identity_layers": list(range(0, 3)),
             "compute_layers": list(range(19, 25)), "type": "combined"},
            {"name": "compute_L16-24_mlp", "layers": list(range(16, 25)), "type": "mlp"},
            {"name": "compute_L12-24_mlp", "layers": list(range(12, 25)), "type": "mlp"},
        ],
        "probe_layers": [0, 1, 2, 3, 4, 5, 8, 12, 18, 22, 24, 27],
        "default_baseline": "The item",
    },
}

# HC pairs (same as Phase 336)
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

# Identity probe objects — we need distinct object categories for probe
PROBE_OBJECTS = ["apple", "banana", "snow", "sky", "cherry", "leaf", "stone",
                 "silk", "ice", "fire", "oven", "fridge", "grass", "ocean",
                 "sun", "blood", "coal", "milk", "rose", "gold", "silver",
                 "cloud", "rain", "desert"]


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
    all_hidden = {}
    for li, hs in enumerate(out.hidden_states):
        all_hidden[li] = hs[0, -1].detach().cpu().float().numpy()
    seq_len = inp["input_ids"].shape[1]

    return attn_outs, mlp_outs, final_hidden, all_hidden, seq_len


# ===== Multi-layer Patched Run =====

def run_patched_multilayer(model, tokenizer, device, base_prompt,
                           patch_specs, n_layers):
    """Run model with patches at multiple layers simultaneously."""
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

    for layer_idx, comp_type, replacement in patch_specs:
        layer = layers[layer_idx]
        if comp_type == "attn" and hasattr(layer, 'self_attn'):
            hooks.append(layer.self_attn.register_forward_hook(make_patch_hook(replacement)))
        elif comp_type == "mlp" and hasattr(layer, 'mlp'):
            hooks.append(layer.mlp.register_forward_hook(make_patch_hook(replacement)))

    inp = tokenizer(base_prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)

    for h in hooks:
        h.remove()

    final_hidden = out.hidden_states[-1][0, -1].detach().cpu().float().numpy()
    return final_hidden


def build_block_specs(block, source_attn_outs, source_mlp_outs):
    """Build patch specs based on block type."""
    specs = []
    btype = block["type"]
    for li in block["layers"]:
        if btype in ("attn", "full"):
            if li in source_attn_outs:
                specs.append((li, "attn", source_attn_outs[li]))
        if btype in ("mlp", "full"):
            if li in source_mlp_outs:
                specs.append((li, "mlp", source_mlp_outs[li]))
    return specs


def build_combined_specs(block, source_attn_outs, source_mlp_outs):
    """Build patch specs for identity+compute combined block."""
    specs = []
    # Identity block: full (attn+mlp)
    for li in block["identity_layers"]:
        if li in source_attn_outs:
            specs.append((li, "attn", source_attn_outs[li]))
        if li in source_mlp_outs:
            specs.append((li, "mlp", source_mlp_outs[li]))
    # Compute block: MLP only
    for li in block["compute_layers"]:
        if li in source_mlp_outs:
            specs.append((li, "mlp", source_mlp_outs[li]))
    return specs


# ===== Main Experiment =====

def run_experiment(model_name):
    log(f"Phase 339+340+341: Multi-baseline + Pipeline + Identity Probe — {model_name}")
    log("=" * 70)

    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    cfg = MODEL_CONFIGS[model_name]
    n_layers = cfg["n_layers"]
    pcfg = PIPELINE_CONFIGS[model_name]

    W_U = get_W_U(model, model_name)
    log(f"  W_U shape: {W_U.shape}")

    if torch.cuda.is_available():
        log(f"  GPU after load: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # ==================================================================
    # PHASE 339: Multi-baseline validation
    # ==================================================================
    log(f"\n{'='*70}")
    log(f"PHASE 339: Multi-baseline Validation")
    log(f"{'='*70}")

    baseline_results = {}
    test_blocks = pcfg["baseline_test_blocks"]

    for baseline_idx, baseline_prompt in enumerate(CORRUPTED_BASELINES):
        log(f"\n--- Baseline: '{baseline_prompt}' ({baseline_idx+1}/{len(CORRUPTED_BASELINES)}) ---")

        # Run corrupted baseline capture
        corrupted_attn_outs, corrupted_mlp_outs, corrupted_hidden, _, corrupted_seq_len = \
            run_and_capture(model, tokenizer, device, baseline_prompt, n_layers)

        # Per-pair experiments for this baseline
        baseline_pair_results = []
        filtered_count = 0

        for pidx, (obj, target_val, competitor_val) in enumerate(HC_PAIRS):
            pair_key = f"{obj}_{target_val}"

            tid_t = get_token_id(tokenizer, target_val)
            tid_c = get_token_id(tokenizer, competitor_val)
            if tid_t is None or tid_c is None:
                continue

            binding_dir = W_U[tid_t] - W_U[tid_c]

            # Run clean capture
            clean_prompt = f"The {obj}"
            clean_attn_outs, clean_mlp_outs, clean_hidden, _, clean_seq_len = \
                run_and_capture(model, tokenizer, device, clean_prompt, n_layers)

            if clean_seq_len != corrupted_seq_len:
                del clean_attn_outs, clean_mlp_outs
                gc.collect()
                torch.cuda.empty_cache()
                continue

            # Compute baselines
            binding_clean = float(binding_dir @ clean_hidden)
            binding_corrupted = float(binding_dir @ corrupted_hidden)
            binding_range = binding_clean - binding_corrupted

            # Filter
            if binding_range < 0.3:
                filtered_count += 1
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

            # Test each block type
            for block in test_blocks:
                bname = block["name"]
                try:
                    specs = build_block_specs(block, clean_attn_outs, clean_mlp_outs)
                    if not specs:
                        continue
                    patched_hidden = run_patched_multilayer(
                        model, tokenizer, device, baseline_prompt,
                        specs, n_layers,
                    )
                    binding_patched = float(binding_dir @ patched_hidden)
                    recovery_pct = 100.0 * (binding_patched - binding_corrupted) / max(binding_range, 1e-10)
                    pair_result["patches"][bname] = {
                        "binding": round(binding_patched, 4),
                        "recovery_pct": round(recovery_pct, 1),
                    }
                except Exception as e:
                    pair_result["patches"][bname] = {"error": str(e)}

            baseline_pair_results.append(pair_result)
            del clean_attn_outs, clean_mlp_outs
            gc.collect()
            torch.cuda.empty_cache()

            if (pidx + 1) % 6 == 0 or pidx < 2:
                elapsed = time.time() - t0
                gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                log(f"  [{pidx+1}/{len(HC_PAIRS)}] valid={len(baseline_pair_results)}, "
                    f"filtered={filtered_count}, elapsed={elapsed:.0f}s, GPU={gpu_mem:.2f}GB")

        # Aggregate for this baseline
        if baseline_pair_results:
            aggs = {}
            for block in test_blocks:
                bname = block["name"]
                recs = [r["patches"][bname]["recovery_pct"] for r in baseline_pair_results
                        if bname in r["patches"] and "recovery_pct" in r["patches"][bname]]
                if recs:
                    aggs[bname] = {
                        "mean": round(float(np.mean(recs)), 1),
                        "std": round(float(np.std(recs)), 1),
                        "n": len(recs),
                    }
            baseline_results[baseline_prompt] = {
                "aggs": aggs,
                "n_valid": len(baseline_pair_results),
                "n_filtered": filtered_count,
                "details": baseline_pair_results,
            }
            log(f"  Baseline '{baseline_prompt}': {len(baseline_pair_results)} valid pairs")
            for bname, agg in aggs.items():
                log(f"    {bname}: recovery={agg['mean']:+.1f}% (std={agg['std']:.1f}%)")
        else:
            baseline_results[baseline_prompt] = {"aggs": {}, "n_valid": 0, "n_filtered": filtered_count}

        del corrupted_attn_outs, corrupted_mlp_outs
        gc.collect()
        torch.cuda.empty_cache()

    # ==================================================================
    # PHASE 340: Identity + Computation combined block patching
    # ==================================================================
    log(f"\n{'='*70}")
    log(f"PHASE 340: Pipeline Block Combination")
    log(f"{'='*70}")

    default_baseline = pcfg["default_baseline"]
    pipeline_blocks = pcfg["pipeline_blocks"]

    # Re-run corrupted baseline capture
    log(f"  Using baseline: '{default_baseline}'")
    corrupted_attn_outs, corrupted_mlp_outs, corrupted_hidden, _, corrupted_seq_len = \
        run_and_capture(model, tokenizer, device, default_baseline, n_layers)

    pipeline_pair_results = []
    filtered_count = 0

    for pidx, (obj, target_val, competitor_val) in enumerate(HC_PAIRS):
        pair_key = f"{obj}_{target_val}"

        tid_t = get_token_id(tokenizer, target_val)
        tid_c = get_token_id(tokenizer, competitor_val)
        if tid_t is None or tid_c is None:
            continue

        binding_dir = W_U[tid_t] - W_U[tid_c]

        clean_prompt = f"The {obj}"
        clean_attn_outs, clean_mlp_outs, clean_hidden, _, clean_seq_len = \
            run_and_capture(model, tokenizer, device, clean_prompt, n_layers)

        if clean_seq_len != corrupted_seq_len:
            del clean_attn_outs, clean_mlp_outs
            gc.collect()
            torch.cuda.empty_cache()
            continue

        binding_clean = float(binding_dir @ clean_hidden)
        binding_corrupted = float(binding_dir @ corrupted_hidden)
        binding_range = binding_clean - binding_corrupted

        if binding_range < 0.3:
            filtered_count += 1
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

        for block in pipeline_blocks:
            bname = block["name"]
            try:
                if block["type"] == "combined":
                    specs = build_combined_specs(block, clean_attn_outs, clean_mlp_outs)
                else:
                    specs = build_block_specs(block, clean_attn_outs, clean_mlp_outs)
                if not specs:
                    continue
                patched_hidden = run_patched_multilayer(
                    model, tokenizer, device, default_baseline,
                    specs, n_layers,
                )
                binding_patched = float(binding_dir @ patched_hidden)
                recovery_pct = 100.0 * (binding_patched - binding_corrupted) / max(binding_range, 1e-10)
                pair_result["patches"][bname] = {
                    "binding": round(binding_patched, 4),
                    "recovery_pct": round(recovery_pct, 1),
                }
            except Exception as e:
                pair_result["patches"][bname] = {"error": str(e)}

        pipeline_pair_results.append(pair_result)
        del clean_attn_outs, clean_mlp_outs
        gc.collect()
        torch.cuda.empty_cache()

        if (pidx + 1) % 6 == 0 or pidx < 2:
            elapsed = time.time() - t0
            gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            log(f"  [{pidx+1}/{len(HC_PAIRS)}] valid={len(pipeline_pair_results)}, "
                f"filtered={filtered_count}, elapsed={elapsed:.0f}s, GPU={gpu_mem:.2f}GB")

    # Aggregate pipeline results
    pipeline_aggs = {}
    if pipeline_pair_results:
        for block in pipeline_blocks:
            bname = block["name"]
            recs = [r["patches"][bname]["recovery_pct"] for r in pipeline_pair_results
                    if bname in r["patches"] and "recovery_pct" in r["patches"][bname]]
            if recs:
                pipeline_aggs[bname] = {
                    "mean": round(float(np.mean(recs)), 1),
                    "std": round(float(np.std(recs)), 1),
                    "n": len(recs),
                }

    log(f"\n--- Phase 340 Results ---")
    log(f"  {'Block':>25} {'recovery%':>10} {'std%':>8} {'n':>4}")
    log("  " + "-" * 50)
    for bname, agg in pipeline_aggs.items():
        log(f"  {bname:>25} {agg['mean']:>+10.1f} {agg['std']:>8.1f} {agg['n']:>4}")

    # ==================================================================
    # PHASE 341: Identity Probe
    # ==================================================================
    log(f"\n{'='*70}")
    log(f"PHASE 341: Identity Probe — What does the residual stream encode?")
    log(f"{'='*70}")

    probe_layers = pcfg["probe_layers"]
    log(f"  Probe layers: {probe_layers}")
    log(f"  Probe objects: {len(PROBE_OBJECTS)} objects")

    # Collect residual stream vectors for each object at each layer
    # Also collect baseline ("The item") vectors
    object_hidden_states = {}  # {layer_idx: {object_name: hidden_vector}}
    baseline_hidden_states = {}  # {layer_idx: hidden_vector}

    # Initialize
    for li in probe_layers:
        object_hidden_states[li] = {}
        baseline_hidden_states[li] = None

    # Run baseline
    log(f"  Running baseline '{default_baseline}'...")
    _, _, _, baseline_all_hidden, _ = run_and_capture(
        model, tokenizer, device, default_baseline, n_layers)
    for li in probe_layers:
        if li in baseline_all_hidden:
            baseline_hidden_states[li] = baseline_all_hidden[li]
    del baseline_all_hidden
    gc.collect()
    torch.cuda.empty_cache()

    # Run each object
    for oidx, obj in enumerate(PROBE_OBJECTS):
        clean_prompt = f"The {obj}"
        _, _, _, obj_all_hidden, _ = run_and_capture(
            model, tokenizer, device, clean_prompt, n_layers)
        for li in probe_layers:
            if li in obj_all_hidden:
                object_hidden_states[li][obj] = obj_all_hidden[li]
        del obj_all_hidden
        gc.collect()
        torch.cuda.empty_cache()

        if (oidx + 1) % 8 == 0 or oidx < 2:
            elapsed = time.time() - t0
            gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            log(f"  [{oidx+1}/{len(PROBE_OBJECTS)}] {obj}: "
                f"elapsed={elapsed:.0f}s, GPU={gpu_mem:.2f}GB")

    # Identity probe analysis
    probe_results = {}

    log(f"\n--- Phase 341: Identity Probe Results ---")
    log(f"  {'Layer':>6} {'cos_sim_obj_vs_base':>20} {'separation':>12} "
        f"{'binding_readout':>16} {'identity_readout':>17}")
    log("  " + "-" * 76)

    for li in probe_layers:
        if baseline_hidden_states[li] is None:
            continue
        base_vec = baseline_hidden_states[li]

        # 1. Cosine similarity between each object and baseline
        cos_sims = []
        for obj, obj_vec in object_hidden_states[li].items():
            cos_sim = float(np.dot(obj_vec, base_vec) /
                          (np.linalg.norm(obj_vec) * np.linalg.norm(base_vec) + 1e-10))
            cos_sims.append(cos_sim)

        # 2. Object separation: average pairwise distance between objects
        obj_names = list(object_hidden_states[li].keys())
        obj_vecs = np.array([object_hidden_states[li][n] for n in obj_names])
        n_obj = len(obj_names)

        if n_obj > 1:
            # Pairwise cosine similarity between objects
            norms = np.linalg.norm(obj_vecs, axis=1, keepdims=True)
            obj_normed = obj_vecs / (norms + 1e-10)
            sim_matrix = obj_normed @ obj_normed.T
            # Mean off-diagonal similarity (higher = less separation)
            off_diag = sim_matrix[np.triu_indices(n_obj, k=1)]
            mean_obj_sim = float(np.mean(off_diag))
            # Separation = 1 - mean_obj_sim
            separation = 1.0 - mean_obj_sim
        else:
            separation = float('nan')

        # 3. Identity readout: can we distinguish objects from baseline?
        # Use the first HC pair's binding direction as a proxy for "binding readout"
        # vs identity readout from the residual stream
        # Identity readout = how much does the object vector differ from baseline
        diffs = []
        for obj, obj_vec in object_hidden_states[li].items():
            diff = obj_vec - base_vec
            diffs.append(diff)
        diff_matrix = np.array(diffs)  # [n_obj, d_model]

        # Variance of differences across objects — higher = more identity info
        diff_var = float(np.mean(np.var(diff_matrix, axis=0)))

        # 4. Binding readout: project onto a binding direction
        # Use apple-red direction as example
        if "apple" in object_hidden_states[li] and "red" in [p[1] for p in HC_PAIRS]:
            tid_red = get_token_id(tokenizer, "red")
            tid_blue = get_token_id(tokenizer, "blue")
            if tid_red is not None and tid_blue is not None:
                binding_dir_example = W_U[tid_red] - W_U[tid_blue]
                binding_projections = [float(d @ binding_dir_example) for d in diffs]
                binding_readout_var = float(np.var(binding_projections))
            else:
                binding_readout_var = float('nan')
        else:
            binding_readout_var = float('nan')

        probe_results[f"L{li}"] = {
            "mean_cos_sim": round(float(np.mean(cos_sims)), 4),
            "std_cos_sim": round(float(np.std(cos_sims)), 4),
            "separation": round(separation, 4),
            "diff_var": round(diff_var, 6),
            "binding_readout_var": round(binding_readout_var, 6),
            "n_objects": n_obj,
        }

        log(f"  L{li:>5} {np.mean(cos_sims):>20.4f} {separation:>12.4f} "
            f"{binding_readout_var:>16.6f} {diff_var:>17.6f}")

    # ==================================================================
    # PHASE 341b: Object identity linear discriminability
    # ==================================================================
    log(f"\n--- Phase 341b: Object Identity Linear Discriminability ---")
    log(f"  Testing: can a linear probe distinguish object categories?")

    # Define object categories
    CATEGORIES = {
        "fruit": ["apple", "banana", "cherry"],
        "nature": ["snow", "sky", "leaf", "grass", "ocean", "cloud", "rain"],
        "hot": ["fire", "oven", "desert", "sun"],
        "cold": ["ice", "fridge"],
        "material": ["stone", "silk", "coal", "milk", "gold", "silver"],
        "body": ["blood"],
        "plant": ["rose"],
    }

    # Build category labels for each object
    obj_to_cat = {}
    for cat, objs in CATEGORIES.items():
        for obj in objs:
            obj_to_cat[obj] = cat

    # For each probe layer, test category discriminability
    category_probe_results = {}
    for li in probe_layers:
        if baseline_hidden_states[li] is None:
            continue
        base_vec = baseline_hidden_states[li]

        # Build feature matrix (object diffs) and labels
        features = []
        labels = []
        for obj in PROBE_OBJECTS:
            if obj in object_hidden_states[li] and obj in obj_to_cat:
                diff = object_hidden_states[li][obj] - base_vec
                features.append(diff)
                labels.append(obj_to_cat[obj])

        if len(set(labels)) < 2:
            continue

        features = np.array(features)
        labels = np.array(labels)
        n_samples = len(labels)
        n_cats = len(set(labels))

        # Simple nearest-centroid classifier (no training needed)
        # For each category, compute centroid
        centroids = {}
        for cat in set(labels):
            mask = labels == cat
            centroids[cat] = features[mask].mean(axis=0)

        # Classify each sample by nearest centroid (leave-one-out)
        correct = 0
        for i in range(n_samples):
            test_feat = features[i]
            test_label = labels[i]
            # Compute centroid without this sample
            train_mask = np.arange(n_samples) != i
            train_feats = features[train_mask]
            train_labels = labels[train_mask]

            # Nearest centroid
            best_cat = None
            best_dist = float('inf')
            for cat in set(train_labels):
                cat_mask = train_labels == cat
                if cat_mask.sum() == 0:
                    continue
                centroid = train_feats[cat_mask].mean(axis=0)
                dist = float(np.linalg.norm(test_feat - centroid))
                if dist < best_dist:
                    best_dist = dist
                    best_cat = cat

            if best_cat == test_label:
                correct += 1

        accuracy = correct / n_samples
        chance = 1.0 / n_cats

        category_probe_results[f"L{li}"] = {
            "accuracy": round(accuracy, 4),
            "chance": round(chance, 4),
            "n_samples": n_samples,
            "n_categories": n_cats,
        }

        log(f"  L{li:>5}: accuracy={accuracy:.4f} (chance={chance:.4f}), "
            f"n={n_samples}, cats={n_cats}")

    # ==================================================================
    # Final aggregation and output
    # ==================================================================
    log(f"\n{'='*80}")
    log(f"FINAL RESULTS SUMMARY — {model_name}")
    log(f"{'='*80}")

    # Phase 339 summary
    log(f"\n--- Phase 339: Multi-baseline Robustness ---")
    log(f"  {'Baseline':>12} {'MLP_block':>10} {'Full_block':>10} {'Attn_block':>10}")
    log("  " + "-" * 46)
    for baseline_prompt, bdata in baseline_results.items():
        aggs = bdata.get("aggs", {})
        mlp_rec = aggs.get("L21-29_mlp" if model_name == "qwen3" else
                          "L30-38_mlp" if model_name == "glm4" else
                          "L19-24_mlp", {}).get("mean", float('nan'))
        full_rec = aggs.get("L21-29_full" if model_name == "qwen3" else
                           "L30-38_full" if model_name == "glm4" else
                           "L19-24_full", {}).get("mean", float('nan'))
        attn_rec = aggs.get("L21-29_attn" if model_name == "qwen3" else
                           "L30-38_attn" if model_name == "glm4" else
                           "L19-24_attn", {}).get("mean", float('nan'))
        log(f"  {baseline_prompt:>12} {mlp_rec:>+10.1f} {full_rec:>+10.1f} {attn_rec:>+10.1f}")

    # Phase 340 summary
    log(f"\n--- Phase 340: Pipeline Block Combination ---")
    for bname, agg in pipeline_aggs.items():
        log(f"  {bname}: {agg['mean']:+.1f}% (std={agg['std']:.1f}%)")

    # Phase 341 summary
    log(f"\n--- Phase 341: Identity Probe Summary ---")
    for li_str, pr in probe_results.items():
        log(f"  {li_str}: cos_sim={pr['mean_cos_sim']:.4f}, "
            f"separation={pr['separation']:.4f}, diff_var={pr['diff_var']:.6f}")

    log(f"\n--- Phase 341b: Category Discriminability ---")
    for li_str, cr in category_probe_results.items():
        log(f"  {li_str}: accuracy={cr['accuracy']:.4f} vs chance={cr['chance']:.4f}")

    # ==================================================================
    # Save results
    # ==================================================================
    save_data = {
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "phase339_multibaseline": {
            baseline_prompt: {
                "aggs": bdata.get("aggs", {}),
                "n_valid": bdata.get("n_valid", 0),
                "n_filtered": bdata.get("n_filtered", 0),
            }
            for baseline_prompt, bdata in baseline_results.items()
        },
        "phase340_pipeline": {
            "aggs": pipeline_aggs,
            "n_valid": len(pipeline_pair_results),
        },
        "phase341_identity_probe": probe_results,
        "phase341b_category_probe": category_probe_results,
        "phase339_details": {
            bp: bdata.get("details", [])
            for bp, bdata in baseline_results.items()
        },
        "phase340_details": pipeline_pair_results,
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

    os.makedirs("results/phase339_multibaseline", exist_ok=True)
    out_path = f"results/phase339_multibaseline/{model_name}_phase339.json"
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
    log("Phase 339+340+341 complete!")
