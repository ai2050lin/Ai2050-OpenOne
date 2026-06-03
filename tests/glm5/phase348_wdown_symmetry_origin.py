"""
Phase 348: W_down Symmetry Origin — Initialization vs Trained + Direction Generality
===================================================================================

User's corrected conclusions from Phase 347 review:
1. "平衡放大不是训练结果" is TOO STRONG — need initialization comparison
2. "对任意方向都对称" needs broader direction testing
3. The real question: is W_down 50/50 symmetry from initialization geometry 
   or from training shaping?
4. Need to test: random directions, W_U subspace, residual PCA, etc.

This script tests:
Part A: Trained W_down symmetry across ALL layers and multiple direction types
  - Semantic directions (binding, antonym, etc.)
  - Random directions (standard Gaussian)
  - W_U column space directions (via SVD)
  - Residual stream PCA directions (from actual activations)
  
Part B: Kaiming-initialized W_down symmetry (same architecture)
  - Generate random W_down with same shape using Kaiming init
  - Test same direction types
  - Compare symmetry metrics with trained W_down

Part C: Per-layer symmetry profile
  - Does symmetry vary across layers?
  - Are early/middle/late layers different?
  - Is symmetry stronger in certain layer groups?

Usage:
  python tests/glm5/phase348_wdown_symmetry_origin.py qwen3
  python tests/glm5/phase348_wdown_symmetry_origin.py deepseek7b
  python tests/glm5/phase348_wdown_symmetry_origin.py glm4
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


def load_model_bf16(model_name):
    """BF16 + device_map=auto loading — reference model_demo_bf16.py"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = None
    # Try flash_attention_2 first, then sdpa, then eager
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


def capture_hidden_states(model, tokenizer, device, prompt, n_layers):
    inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    hs = {}
    for i, h in enumerate(out.hidden_states):
        hs[i] = h[0, -1, :].detach().cpu().float().numpy()
    return hs


def compute_symmetry_metrics(W_down, direction):
    """
    Compute W_down symmetry metrics for a given direction.
    
    For each channel i, compute:
      channel_proj[i] = (W_down.T @ direction)[i]
    
    Then measure how symmetric the positive/negative split is.
    """
    d_model = W_down.shape[0]
    d_ff = W_down.shape[1]
    
    # Channel projections
    channel_proj = W_down.T @ direction  # (d_ff,)
    
    # Positive/negative split
    pos_mask = channel_proj > 0
    neg_mask = channel_proj < 0
    zero_mask = channel_proj == 0
    
    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))
    n_zero = int(np.sum(zero_mask))
    
    pos_frac = n_pos / max(d_ff, 1)
    
    # Projection sums
    pos_proj_vals = channel_proj[pos_mask]
    neg_proj_vals = channel_proj[neg_mask]
    
    # Balance ratios
    pos_sum = np.sum(np.abs(pos_proj_vals)) if n_pos > 0 else 0
    neg_sum = np.sum(np.abs(neg_proj_vals)) if n_neg > 0 else 0
    proj_balance = pos_sum / max(neg_sum, 1e-10)
    
    # Channel norms
    channel_norms = np.linalg.norm(W_down, axis=0)
    pos_norms = channel_norms[pos_mask]
    neg_norms = channel_norms[neg_mask]
    pos_norm_mean = float(np.mean(pos_norms)) if n_pos > 0 else 0
    neg_norm_mean = float(np.mean(neg_norms)) if n_neg > 0 else 0
    norm_balance = pos_norm_mean / max(neg_norm_mean, 1e-10)
    
    # Projection distribution statistics
    proj_mean = float(np.mean(channel_proj))
    proj_std = float(np.std(channel_proj))
    proj_skew = float(np.mean(((channel_proj - proj_mean) / max(proj_std, 1e-10))**3)) if proj_std > 1e-10 else 0
    proj_kurtosis = float(np.mean(((channel_proj - proj_mean) / max(proj_std, 1e-10))**4)) - 3.0 if proj_std > 1e-10 else 0
    
    # Positive vs negative projection mean (should be symmetric around 0)
    pos_proj_mean = float(np.mean(pos_proj_vals)) if n_pos > 0 else 0
    neg_proj_mean = float(np.mean(neg_proj_vals)) if n_neg > 0 else 0
    
    return {
        "pos_frac": pos_frac,
        "proj_balance": proj_balance,
        "norm_balance": norm_balance,
        "proj_mean": proj_mean,
        "proj_std": proj_std,
        "proj_skew": proj_skew,
        "proj_kurtosis": proj_kurtosis,
        "pos_proj_mean": pos_proj_mean,
        "neg_proj_mean": neg_proj_mean,
        "pos_norm_mean": pos_norm_mean,
        "neg_norm_mean": neg_norm_mean,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_zero": n_zero,
    }


def generate_direction_sets(W_U, d_model, n_random=50, n_wu_subspace=50, n_pca=50):
    """Generate multiple direction sets for testing."""
    direction_sets = {}
    
    # 1. Standard Gaussian random directions
    rng = np.random.RandomState(42)
    random_dirs = rng.randn(n_random, d_model)
    random_dirs = random_dirs / np.linalg.norm(random_dirs, axis=1, keepdims=True)
    direction_sets["random_gaussian"] = random_dirs
    
    # 2. W_U column space directions (SVD of W_U)
    # Sample from the column space of W_U
    log("  Computing W_U SVD for subspace directions...")
    # W_U shape: (vocab, d_model), we need column space of W_U.T (d_model × vocab)
    # Actually W_U maps from d_model to vocab, so W_U columns are d_model-dimensional
    # The column space is the span of all W_U[i, :] vectors
    # Use random projection: sample random vectors in W_U column space
    # Method: take a random linear combination of W_U rows
    wu_sample_indices = rng.choice(W_U.shape[0], size=min(5000, W_U.shape[0]), replace=False)
    W_U_sample = W_U[wu_sample_indices]
    # PCA of W_U rows to get principal directions
    from numpy.linalg import svd
    U_wu, S_wu, Vt_wu = svd(W_U_sample, full_matrices=False)
    # Vt_wu rows are the principal directions of W_U row space
    # Use first n_pca directions
    n_available = min(n_pca, Vt_wu.shape[0])
    wu_pca_dirs = Vt_wu[:n_available]
    direction_sets["W_U_PCA"] = wu_pca_dirs
    
    # 3. Random directions in W_U column space
    # Generate random combinations of W_U PCA directions
    coeffs = rng.randn(n_wu_subspace, n_available)
    wu_subspace_dirs = coeffs @ wu_pca_dirs
    wu_subspace_dirs = wu_subspace_dirs / np.linalg.norm(wu_subspace_dirs, axis=1, keepdims=True)
    direction_sets["W_U_subspace_random"] = wu_subspace_dirs
    
    # 4. W_U individual token directions (top-freq tokens)
    # Use the first 50 token embedding directions
    wu_token_dirs = W_U[:50]
    wu_token_norms = np.linalg.norm(wu_token_dirs, axis=1, keepdims=True)
    wu_token_dirs = wu_token_dirs / np.maximum(wu_token_norms, 1e-10)
    direction_sets["W_U_token_directions"] = wu_token_dirs
    
    return direction_sets


def generate_semantic_directions(W_U, tokenizer, test_pairs):
    """Generate semantic directions from test pairs."""
    directions = []
    labels = []
    for obj, target, competitor in test_pairs:
        tid_t = get_token_id(tokenizer, target)
        tid_c = get_token_id(tokenizer, competitor)
        if tid_t is None or tid_c is None: continue
        d = W_U[tid_t] - W_U[tid_c]
        norm = np.linalg.norm(d)
        if norm < 1e-10: continue
        directions.append(d / norm)
        labels.append(f"{obj}-{target}-{competitor}")
    return np.array(directions), labels


def capture_residual_pca_directions(model, tokenizer, device, n_layers, d_model, n_directions=50):
    """Capture residual stream activations and compute PCA directions."""
    prompts = [
        "The apple is", "The sky is", "Fire is very", "Snow looks",
        "The ocean is", "Blood is always", "Ice feels", "Grass grows",
        "A banana is", "The sun shines",
    ]
    
    all_activations = []
    for prompt in prompts:
        inp = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64).to(device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
        # Use last layer hidden state
        hs = out.hidden_states[-1][0, -1, :].detach().cpu().float().numpy()
        all_activations.append(hs)
        del out; gc.collect(); torch.cuda.empty_cache()
    
    all_activations = np.array(all_activations)
    # PCA on activation differences (centered)
    centered = all_activations - np.mean(all_activations, axis=0)
    # Use SVD for PCA
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    # Vt rows are principal directions
    n_available = min(n_directions, Vt.shape[0])
    return Vt[:n_available]


def kaiming_init_wdown(d_model, d_ff, seed=42):
    """Generate Kaiming-initialized W_down with same architecture."""
    rng = np.random.RandomState(seed)
    # Kaiming initialization: N(0, sqrt(2/fan_in))
    # For W_down shape (d_model, d_ff), fan_in = d_ff
    std = np.sqrt(2.0 / d_ff)
    W = rng.randn(d_model, d_ff) * std
    return W


def run_experiment(model_name):
    log(f"Phase 348: W_down Symmetry Origin — {model_name}")
    log("=" * 70)
    t0 = time.time()
    cfg = MODEL_CONFIGS[model_name]
    n_layers = cfg["n_layers"]
    d_model = cfg["d_model"]
    d_ff = cfg["d_ff"]
    
    model, tokenizer, device = load_model_bf16(model_name)
    W_U = get_W_U(model, model_name)
    log(f"  W_U shape: {W_U.shape}")
    
    # ======================================================================
    # Part A: Trained W_down symmetry across ALL layers and direction types
    # ======================================================================
    log(f"\n{'='*70}")
    log(f"PART A: Trained W_down Symmetry — All Layers + Multiple Direction Types")
    log(f"{'='*70}")
    
    # Generate direction sets (only need W_U, no model inference)
    log("  Generating direction sets...")
    direction_sets = generate_direction_sets(W_U, d_model)
    
    # Generate semantic directions
    semantic_dirs, semantic_labels = generate_semantic_directions(W_U, tokenizer, TEST_PAIRS)
    if len(semantic_dirs) > 0:
        direction_sets["semantic_binding"] = semantic_dirs
    
    # Generate residual PCA directions
    log("  Capturing residual stream PCA directions...")
    residual_pca_dirs = capture_residual_pca_directions(model, tokenizer, device, n_layers, d_model)
    direction_sets["residual_PCA"] = residual_pca_dirs
    
    log(f"  Direction sets: {list(direction_sets.keys())}")
    for k, v in direction_sets.items():
        log(f"    {k}: {v.shape[0]} directions")
    
    # Extract W_down for all layers
    layers = get_layers(model)
    all_layer_wdown = {}
    log("  Extracting W_down for all layers...")
    for li in range(n_layers):
        _, _, W_down, d_ff_actual = get_mlp_weights(layers[li], model_name, model)
        if W_down is None:
            _, _, W_down, d_ff_actual = get_mlp_weights_from_disk(model_name, li)
        if W_down is not None:
            all_layer_wdown[li] = W_down
        if (li + 1) % 10 == 0:
            log(f"    Extracted {li+1}/{n_layers} layers")
    
    log(f"  Successfully extracted W_down for {len(all_layer_wdown)}/{n_layers} layers")
    
    # Test symmetry for each layer × direction type
    trained_results = {}
    sample_layers = sorted(list(all_layer_wdown.keys()))
    
    # For efficiency: sample layers (every 2nd layer for large models)
    if len(sample_layers) > 20:
        test_layers = sample_layers[::2]
        # Always include binding layers
        for bl in cfg["binding_layers"]:
            if bl not in test_layers:
                test_layers.append(bl)
        test_layers = sorted(test_layers)
    else:
        test_layers = sample_layers
    
    log(f"  Testing {len(test_layers)} layers × {len(direction_sets)} direction types")
    
    for li in test_layers:
        if li not in all_layer_wdown:
            continue
        W_down = all_layer_wdown[li]
        layer_result = {}
        
        for dir_name, dirs in direction_sets.items():
            metrics_list = []
            for d in dirs:
                m = compute_symmetry_metrics(W_down, d)
                metrics_list.append(m)
            
            # Aggregate
            avg_metrics = {}
            for key in metrics_list[0].keys():
                vals = [m[key] for m in metrics_list]
                avg_metrics[key] = float(np.mean(vals))
                avg_metrics[key + "_std"] = float(np.std(vals))
            
            layer_result[dir_name] = avg_metrics
        
        trained_results[str(li)] = layer_result
        
        if (test_layers.index(li) + 1) % 5 == 0:
            elapsed = time.time() - t0
            log(f"    [{test_layers.index(li)+1}/{len(test_layers)} layers] elapsed={elapsed:.0f}s")
    
    # Summary of trained results
    log(f"\n  PART A Summary — Trained W_down:")
    
    # Average across all layers for each direction type
    trained_summary = {}
    for dir_name in direction_sets.keys():
        layer_metrics = defaultdict(list)
        for li_str, lr in trained_results.items():
            if dir_name in lr:
                for key, val in lr[dir_name].items():
                    if not key.endswith("_std"):
                        layer_metrics[key].append(val)
        
        trained_summary[dir_name] = {key: float(np.mean(vals)) for key, vals in layer_metrics.items()}
        
        log(f"\n  Direction type: {dir_name}")
        log(f"    pos_frac:      {trained_summary[dir_name].get('pos_frac', 0):.4f}")
        log(f"    proj_balance:  {trained_summary[dir_name].get('proj_balance', 0):.4f}")
        log(f"    norm_balance:  {trained_summary[dir_name].get('norm_balance', 0):.4f}")
        log(f"    proj_skew:     {trained_summary[dir_name].get('proj_skew', 0):.6f}")
        log(f"    proj_kurtosis: {trained_summary[dir_name].get('proj_kurtosis', 0):.4f}")
    
    # ======================================================================
    # Part B: Kaiming-initialized W_down symmetry
    # ======================================================================
    log(f"\n{'='*70}")
    log(f"PART B: Kaiming-Initialized W_down Symmetry")
    log(f"{'='*70}")
    
    init_results = {}
    n_init_seeds = 3  # Test with 3 different seeds
    
    for seed in range(n_init_seeds):
        W_down_init = kaiming_init_wdown(d_model, d_ff, seed=seed)
        seed_result = {}
        
        for dir_name, dirs in direction_sets.items():
            metrics_list = []
            for d in dirs:
                m = compute_symmetry_metrics(W_down_init, d)
                metrics_list.append(m)
            
            avg_metrics = {}
            for key in metrics_list[0].keys():
                vals = [m[key] for m in metrics_list]
                avg_metrics[key] = float(np.mean(vals))
                avg_metrics[key + "_std"] = float(np.std(vals))
            
            seed_result[dir_name] = avg_metrics
        
        init_results[f"seed_{seed}"] = seed_result
        log(f"  Seed {seed} done")
    
    # Average across seeds
    init_summary = {}
    for dir_name in direction_sets.keys():
        seed_metrics = defaultdict(list)
        for seed_key, sr in init_results.items():
            if dir_name in sr:
                for key, val in sr[dir_name].items():
                    if not key.endswith("_std"):
                        seed_metrics[key].append(val)
        
        init_summary[dir_name] = {key: float(np.mean(vals)) for key, vals in seed_metrics.items()}
        
        log(f"\n  Direction type: {dir_name} (Kaiming init avg)")
        log(f"    pos_frac:      {init_summary[dir_name].get('pos_frac', 0):.4f}")
        log(f"    proj_balance:  {init_summary[dir_name].get('proj_balance', 0):.4f}")
        log(f"    norm_balance:  {init_summary[dir_name].get('norm_balance', 0):.4f}")
        log(f"    proj_skew:     {init_summary[dir_name].get('proj_skew', 0):.6f}")
        log(f"    proj_kurtosis: {init_summary[dir_name].get('proj_kurtosis', 0):.4f}")
    
    # ======================================================================
    # Part C: Per-layer symmetry profile
    # ======================================================================
    log(f"\n{'='*70}")
    log(f"PART C: Per-Layer Symmetry Profile")
    log(f"{'='*70}")
    
    # Use semantic_binding directions for per-layer profile
    profile_dir = "semantic_binding" if "semantic_binding" in direction_sets else "random_gaussian"
    profile_dirs = direction_sets[profile_dir]
    
    layer_profile = {}
    for li in sorted(all_layer_wdown.keys()):
        W_down = all_layer_wdown[li]
        metrics_list = []
        for d in profile_dirs:
            m = compute_symmetry_metrics(W_down, d)
            metrics_list.append(m)
        
        avg = {key: float(np.mean([m[key] for m in metrics_list])) for key in metrics_list[0].keys()}
        layer_profile[li] = avg
    
    # Print profile
    log(f"  Per-layer profile (direction type: {profile_dir}):")
    log(f"  {'Layer':>5} {'pos_frac':>10} {'proj_bal':>10} {'norm_bal':>10} {'skew':>10} {'kurtosis':>10}")
    
    layer_groups = {"early": [], "middle": [], "late": [], "binding": []}
    for li in sorted(layer_profile.keys()):
        p = layer_profile[li]
        log(f"  {li:>5} {p['pos_frac']:>10.4f} {p['proj_balance']:>10.4f} {p['norm_balance']:>10.4f} {p['proj_skew']:>10.6f} {p['proj_kurtosis']:>10.4f}")
        
        # Classify layer
        if li < n_layers // 3:
            layer_groups["early"].append(p)
        elif li < 2 * n_layers // 3:
            layer_groups["middle"].append(p)
        else:
            layer_groups["late"].append(p)
        if li in cfg["binding_layers"]:
            layer_groups["binding"].append(p)
    
    # Group averages
    log(f"\n  Layer group averages:")
    for group_name, profiles in layer_groups.items():
        if not profiles: continue
        avg_pos = np.mean([p["pos_frac"] for p in profiles])
        avg_bal = np.mean([p["proj_balance"] for p in profiles])
        avg_nbal = np.mean([p["norm_balance"] for p in profiles])
        avg_skew = np.mean([p["proj_skew"] for p in profiles])
        avg_kurt = np.mean([p["proj_kurtosis"] for p in profiles])
        log(f"  {group_name:>8}: pos_frac={avg_pos:.4f}, proj_bal={avg_bal:.4f}, norm_bal={avg_nbal:.4f}, skew={avg_skew:.6f}, kurtosis={avg_kurt:.4f}")
    
    # ======================================================================
    # Trained vs Init comparison
    # ======================================================================
    log(f"\n{'='*70}")
    log(f"COMPARISON: Trained vs Kaiming-Initialized W_down")
    log(f"{'='*70}")
    
    comparison = {}
    for dir_name in direction_sets.keys():
        t_data = trained_summary.get(dir_name, {})
        i_data = init_summary.get(dir_name, {})
        
        comp = {}
        for key in ["pos_frac", "proj_balance", "norm_balance", "proj_skew", "proj_kurtosis"]:
            t_val = t_data.get(key, 0)
            i_val = i_data.get(key, 0)
            comp[key] = {"trained": t_val, "init": i_val, "diff": t_val - i_val}
        
        comparison[dir_name] = comp
        
        log(f"\n  Direction: {dir_name}")
        log(f"    {'':>15} {'Trained':>10} {'Init':>10} {'Diff':>10}")
        for key in ["pos_frac", "proj_balance", "norm_balance", "proj_skew", "proj_kurtosis"]:
            log(f"    {key:>15} {comp[key]['trained']:>10.4f} {comp[key]['init']:>10.4f} {comp[key]['diff']:>10.4f}")
    
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
        "trained_summary": trained_summary,
        "init_summary": init_summary,
        "comparison": comparison,
        "layer_profile": {str(k): v for k, v in layer_profile.items()},
        "layer_groups": {k: {key: float(np.mean([p[key] for p in v])) for key in v[0].keys()} for k, v in layer_groups.items() if v},
        "trained_per_layer": trained_results,
        "init_per_seed": init_results,
        "direction_set_sizes": {k: v.shape[0] for k, v in direction_sets.items()},
    })

    os.makedirs("results/phase348_wdown_symmetry_origin", exist_ok=True)
    out_path = f"results/phase348_wdown_symmetry_origin/{model_name}_phase348.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    log(f"\nResults saved to {out_path}")

    del model, W_U, all_layer_wdown; gc.collect(); torch.cuda.empty_cache()
    total_time = time.time() - t0
    log(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f}min)")


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    if model_name not in MODEL_CONFIGS: log(f"Unknown model: {model_name}"); sys.exit(1)
    run_experiment(model_name)
    log("Phase 348 complete!")
