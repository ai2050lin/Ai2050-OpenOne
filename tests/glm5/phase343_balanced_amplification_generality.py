"""
Phase 343: Balanced Amplification Generality Test + Micro-Bias Source Decomposition
====================================================================================

Two critical experiments:

Experiment A (Phase 343): Is balanced amplification binding-specific or MLP-generic?
  - Test channel decomposition along multiple directions:
    1. Binding direction (W_U[target] - W_U[competitor])  [baseline, known balanced]
    2. Random directions (N samples)
    3. Object identity direction (W_U[object] - W_U["item"])
    4. Same-class direction (W_U[apple] - W_U[banana])
    5. Attribute value direction (W_U[red] - W_U[blue], no object context)
    6. Unrelated word direction (W_U[run] - W_U[sit])

  Key question: Is amplification_balance ≈ 1.0 for ALL directions, or only binding?

Experiment B (Phase 344): Where does the 1-3% micro-bias come from?
  - Decompose MLP(x) = down_proj(SiLU(gate_proj(x)) * up_proj(x))
  - The binding contribution of channel i is: (d @ W_down[:,i]) * SiLU(gate_i) * up_i
  - Test which component drives the micro-bias:
    a. Gate difference: does SiLU(gate_clean) - SiLU(gate_corrupt) favor compat channels?
    b. Up difference: does up_clean - up_corrupt favor compat channels?
    c. W_down structure: is there asymmetry in W_down that favors compat channels?
    d. Gate x Up interaction: does the product favor compat channels?

Usage:
  python tests/glm5/phase343_balanced_amplification_generality.py qwen3
  python tests/glm5/phase343_balanced_amplification_generality.py deepseek7b
  python tests/glm5/phase343_balanced_amplification_generality.py glm4
"""
import sys, os, time, json, gc, traceback
import torch
import numpy as np
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

def log(msg="", end="\n"):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", end=end, flush=True)


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

# Core test pairs
TEST_PAIRS = [
    ("apple", "red", "blue"),
    ("banana", "yellow", "purple"),
    ("snow", "white", "black"),
    ("sky", "blue", "green"),
    ("cherry", "red", "blue"),
    ("leaf", "green", "red"),
    ("ice", "cold", "hot"),
    ("fire", "hot", "cold"),
    ("grass", "green", "red"),
    ("ocean", "blue", "yellow"),
    ("sun", "yellow", "purple"),
    ("blood", "red", "green"),
]

# Additional word pairs for non-binding directions
IDENTITY_WORDS = {
    "same_class": [("apple", "banana"), ("cherry", "grape"), ("snow", "ice"),
                   ("fire", "oven"), ("grass", "leaf"), ("ocean", "river")],
    "attribute_only": [("red", "blue"), ("yellow", "purple"), ("white", "black"),
                       ("hot", "cold"), ("green", "red"), ("rough", "smooth")],
    "unrelated": [("run", "sit"), ("think", "sleep"), ("big", "fast"),
                  ("morning", "heavy"), ("quiet", "square"), ("young", "sharp")],
}

CORRUPTED_BASELINE = "The item"

N_RANDOM_DIRS = 10  # number of random directions to test


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


# ===== Channel Analysis for Arbitrary Direction =====

def channel_decomposition(W_down, direction, gate_acts_clean, up_acts_clean,
                          gate_acts_corrupt, up_acts_corrupt):
    """
    Channel decomposition along an arbitrary direction d.
    
    MLP output = down_proj(SiLU(gate_proj(x)) * up_proj(x))
    Contribution of channel i to direction d: (d @ W_down[:, i]) * SiLU(gate_i) * up_i
    """
    d_ff = W_down.shape[1]
    min_d = min(gate_acts_clean.shape[0], d_ff, W_down.shape[1])
    
    def silu(x):
        return x * (1.0 / (1.0 + np.exp(-x)))
    
    gate_silu_clean = silu(gate_acts_clean[:min_d])
    gate_silu_corrupt = silu(gate_acts_corrupt[:min_d])
    up_clean = up_acts_clean[:min_d]
    up_corrupt = up_acts_corrupt[:min_d]
    W_down_trunc = W_down[:, :min_d]
    
    # Projection of each channel onto direction
    dir_proj = direction @ W_down_trunc  # [min_d]
    
    # Per-channel contribution
    contrib_clean = dir_proj * gate_silu_clean * up_clean
    contrib_corrupt = dir_proj * gate_silu_corrupt * up_corrupt
    delta = contrib_clean - contrib_corrupt
    
    # Classify channels by projection sign
    pos_mask = dir_proj > 0
    neg_mask = dir_proj < 0
    
    # Gross amplification for positive and negative channels
    pos_gross = float(np.sum(np.abs(delta[pos_mask])))
    neg_gross = float(np.sum(np.abs(delta[neg_mask])))
    
    # Net contribution
    net_contrib = float(np.sum(delta))
    total_gross = pos_gross + neg_gross
    
    # Balance ratio
    balance = neg_gross / max(pos_gross, 1e-10)
    
    # Net/gross ratio
    net_gross = abs(net_contrib) / max(total_gross, 1e-10)
    
    # Separate: for pos channels, delta>0 = boost, delta<0 = reduce
    # For neg channels, delta>0 = suppress (less neg in clean), delta<0 = amplify (more neg in clean)
    pos_boost = float(np.sum(delta[pos_mask & (delta > 0)]))
    pos_reduce = float(np.sum(delta[pos_mask & (delta < 0)]))
    neg_suppress = float(np.sum(delta[neg_mask & (delta > 0)]))
    neg_amplify = float(np.sum(delta[neg_mask & (delta < 0)]))
    
    return {
        "pos_gross": pos_gross,
        "neg_gross": neg_gross,
        "total_gross": total_gross,
        "net_contrib": net_contrib,
        "balance": balance,
        "net_gross_ratio": net_gross,
        "pos_boost": pos_boost,
        "pos_reduce": pos_reduce,
        "neg_suppress": neg_suppress,
        "neg_amplify": neg_amplify,
        "n_pos": int(pos_mask.sum()),
        "n_neg": int(neg_mask.sum()),
    }


# ===== Micro-Bias Source Decomposition =====

def micro_bias_source_decomposition(W_down, W_gate, W_up, binding_dir,
                                     gate_acts_clean, up_acts_clean,
                                     gate_acts_corrupt, up_acts_corrupt):
    """
    Decompose the 1-3% micro-bias into gate, up, and W_down components.
    
    For each channel i:
      contribution_i = (d @ W_down[:,i]) * SiLU(gate_i) * up_i
    
    The delta (clean - corrupt) can be decomposed:
      delta_i = (d @ W_down[:,i]) * [SiLU(g_c_i)*u_c_i - SiLU(g_r_i)*u_r_i]
    
    Using product rule:
      delta_i ≈ (d @ W_down[:,i]) * [SiLU(g_c_i) * Δu_i + ΔSiLU(g_i) * u_r_i]
    
    where Δu_i = u_c_i - u_r_i, ΔSiLU(g_i) = SiLU(g_c_i) - SiLU(g_r_i)
    
    This separates:
    - Gate-driven bias: (d @ W_down[:,i]) * ΔSiLU(g_i) * u_r_i
    - Up-driven bias: (d @ W_down[:,i]) * SiLU(g_c_i) * Δu_i
    - Interaction: (d @ W_down[:,i]) * ΔSiLU(g_i) * Δu_i (small)
    """
    d_ff = W_down.shape[1]
    min_d = min(gate_acts_clean.shape[0], d_ff, W_down.shape[1])
    
    def silu(x):
        return x * (1.0 / (1.0 + np.exp(-x)))
    
    gate_silu_clean = silu(gate_acts_clean[:min_d])
    gate_silu_corrupt = silu(gate_acts_corrupt[:min_d])
    up_clean = up_acts_clean[:min_d]
    up_corrupt = up_acts_corrupt[:min_d]
    W_down_trunc = W_down[:, :min_d]
    
    # Direction projection
    dir_proj = binding_dir @ W_down_trunc  # [min_d]
    
    # Deltas
    delta_silu = gate_silu_clean - gate_silu_corrupt  # ΔSiLU(g)
    delta_up = up_clean - up_corrupt  # Δu
    
    # Full delta
    delta_full = dir_proj * (gate_silu_clean * up_clean - gate_silu_corrupt * up_corrupt)
    
    # Decomposed deltas
    gate_driven = dir_proj * delta_silu * up_corrupt  # gate change with base up
    up_driven = dir_proj * gate_silu_clean * delta_up  # up change with clean gate
    interaction = dir_proj * delta_silu * delta_up  # cross term
    
    # Classify channels
    pos_mask = dir_proj > 0
    neg_mask = dir_proj < 0
    
    # For each component, compute net and balance
    def component_stats(contrib_arr, pos_m, neg_m, dir_p):
        """Compute net, pos/neg gross, balance for a contribution array."""
        net = float(np.sum(contrib_arr))
        pos_gross = float(np.sum(np.abs(contrib_arr[pos_m])))
        neg_gross = float(np.sum(np.abs(contrib_arr[neg_m])))
        total_gross = pos_gross + neg_gross
        balance = neg_gross / max(pos_gross, 1e-10)
        net_gross = abs(net) / max(total_gross, 1e-10)
        return {
            "net": net, "pos_gross": pos_gross, "neg_gross": neg_gross,
            "total_gross": total_gross, "balance": balance, "net_gross_ratio": net_gross,
        }
    
    full_stats = component_stats(delta_full, pos_mask, neg_mask, dir_proj)
    gate_stats = component_stats(gate_driven, pos_mask, neg_mask, dir_proj)
    up_stats = component_stats(up_driven, pos_mask, neg_mask, dir_proj)
    inter_stats = component_stats(interaction, pos_mask, neg_mask, dir_proj)
    
    # Also check: W_down structure asymmetry
    # For pos channels (dir_proj > 0), avg |dir_proj| vs neg channels
    avg_pos_proj = float(np.mean(np.abs(dir_proj[pos_mask]))) if pos_mask.sum() > 0 else 0
    avg_neg_proj = float(np.mean(np.abs(dir_proj[neg_mask]))) if neg_mask.sum() > 0 else 0
    
    # Check: avg gate_silu magnitude for pos vs neg channels
    avg_silu_clean_pos = float(np.mean(np.abs(gate_silu_clean[pos_mask]))) if pos_mask.sum() > 0 else 0
    avg_silu_clean_neg = float(np.mean(np.abs(gate_silu_clean[neg_mask]))) if neg_mask.sum() > 0 else 0
    
    # Check: avg |delta_silu| for pos vs neg channels
    avg_delta_silu_pos = float(np.mean(np.abs(delta_silu[pos_mask]))) if pos_mask.sum() > 0 else 0
    avg_delta_silu_neg = float(np.mean(np.abs(delta_silu[neg_mask]))) if neg_mask.sum() > 0 else 0
    
    # Check: avg |delta_up| for pos vs neg channels
    avg_delta_up_pos = float(np.mean(np.abs(delta_up[pos_mask]))) if pos_mask.sum() > 0 else 0
    avg_delta_up_neg = float(np.mean(np.abs(delta_up[neg_mask]))) if neg_mask.sum() > 0 else 0
    
    # Per-channel signed bias: which channels contribute most to net bias?
    # The top net-bias channels
    top_k = 100
    abs_contrib = np.abs(delta_full)
    top_indices = np.argsort(abs_contrib)[-top_k:]
    top_net_contrib = float(np.sum(delta_full[top_indices]))
    top_pos_count = int((dir_proj[top_indices] > 0).sum())
    top_neg_count = int((dir_proj[top_indices] < 0).sum())
    
    # Correlation: does dir_proj magnitude correlate with delta_silu or delta_up?
    if np.std(delta_silu) > 1e-10 and np.std(dir_proj) > 1e-10:
        corr_silu_dir = float(np.corrcoef(delta_silu, dir_proj)[0, 1])
    else:
        corr_silu_dir = 0.0
    
    if np.std(delta_up) > 1e-10 and np.std(dir_proj) > 1e-10:
        corr_up_dir = float(np.corrcoef(delta_up, dir_proj)[0, 1])
    else:
        corr_up_dir = 0.0
    
    return {
        "full": full_stats,
        "gate_driven": gate_stats,
        "up_driven": up_stats,
        "interaction": inter_stats,
        # W_down structure
        "avg_pos_proj": avg_pos_proj,
        "avg_neg_proj": avg_neg_proj,
        "proj_asymmetry": avg_pos_proj / max(avg_neg_proj, 1e-10),
        # Gate structure
        "avg_silu_pos": avg_silu_clean_pos,
        "avg_silu_neg": avg_silu_clean_neg,
        "silu_asymmetry": avg_silu_clean_pos / max(avg_silu_clean_neg, 1e-10),
        # Delta structure
        "avg_delta_silu_pos": avg_delta_silu_pos,
        "avg_delta_silu_neg": avg_delta_silu_neg,
        "delta_silu_asymmetry": avg_delta_silu_pos / max(avg_delta_silu_neg, 1e-10),
        "avg_delta_up_pos": avg_delta_up_pos,
        "avg_delta_up_neg": avg_delta_up_neg,
        "delta_up_asymmetry": avg_delta_up_pos / max(avg_delta_up_neg, 1e-10),
        # Correlations
        "corr_silu_dir": corr_silu_dir,
        "corr_up_dir": corr_up_dir,
        # Top channels
        "top_k": top_k,
        "top_net_contrib": top_net_contrib,
        "top_pos_count": top_pos_count,
        "top_neg_count": top_neg_count,
    }


# ===== Main Experiment =====

def run_experiment(model_name):
    log(f"Phase 343: Balanced Amplification Generality + Micro-Bias Source — {model_name}")
    log("=" * 70)

    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    cfg = MODEL_CONFIGS[model_name]
    n_layers = cfg["n_layers"]
    binding_layers = cfg["binding_layers"]

    W_U = get_W_U(model, model_name)
    d_model = W_U.shape[1]
    log(f"  W_U shape: {W_U.shape}")
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log(f"  GPU after load: {gpu_mem:.2f}GB")

    # Pre-extract MLP weights for all binding layers
    layers = get_layers(model)
    mlp_weights = {}
    for li in binding_layers:
        W_gate, W_up, W_down, d_ff = get_mlp_weights(layers[li], model_name, model)
        mlp_weights[li] = {"W_gate": W_gate, "W_up": W_up, "W_down": W_down, "d_ff": d_ff}
        log(f"  L{li}: d_ff={d_ff}, W_down={W_down.shape if W_down is not None else 'None'}")

    # ==================================================================
    # EXPERIMENT A: Direction Generality Test
    # ==================================================================
    log(f"\n{'='*70}")
    log(f"EXPERIMENT A: Balanced Amplification Across Different Directions")
    log(f"{'='*70}")

    # We'll test each direction type using apple-red as the primary example
    # (plus a few more pairs for robustness)
    test_directions = {}

    # 1. Binding direction (known baseline)
    for obj, target, competitor in TEST_PAIRS[:4]:  # Use 4 pairs
        tid_t = get_token_id(tokenizer, target)
        tid_c = get_token_id(tokenizer, competitor)
        if tid_t is not None and tid_c is not None:
            d = W_U[tid_t] - W_U[tid_c]
            d_norm = np.linalg.norm(d)
            if d_norm > 0:
                test_directions[f"binding_{obj}_{target}"] = {
                    "direction": d / d_norm,
                    "type": "binding",
                    "clean_prompt": f"The {obj}",
                    "corrupt_prompt": CORRUPTED_BASELINE,
                }

    # 2. Random directions
    np.random.seed(42)
    for i in range(N_RANDOM_DIRS):
        d = np.random.randn(d_model)
        d_norm = np.linalg.norm(d)
        if d_norm > 0:
            test_directions[f"random_{i}"] = {
                "direction": d / d_norm,
                "type": "random",
                "clean_prompt": "The apple",  # use same prompts
                "corrupt_prompt": CORRUPTED_BASELINE,
            }

    # 3. Object identity direction
    for obj, _ in [("apple", "banana"), ("snow", "ice"), ("fire", "oven")]:
        oid = get_token_id(tokenizer, obj)
        iid = get_token_id(tokenizer, "item")
        if oid is not None and iid is not None:
            d = W_U[oid] - W_U[iid]
            d_norm = np.linalg.norm(d)
            if d_norm > 0:
                test_directions[f"identity_{obj}"] = {
                    "direction": d / d_norm,
                    "type": "object_identity",
                    "clean_prompt": f"The {obj}",
                    "corrupt_prompt": CORRUPTED_BASELINE,
                }

    # 4. Same-class direction (e.g., apple vs banana)
    for w1, w2 in IDENTITY_WORDS["same_class"]:
        tid1 = get_token_id(tokenizer, w1)
        tid2 = get_token_id(tokenizer, w2)
        if tid1 is not None and tid2 is not None:
            d = W_U[tid1] - W_U[tid2]
            d_norm = np.linalg.norm(d)
            if d_norm > 0:
                test_directions[f"same_class_{w1}_{w2}"] = {
                    "direction": d / d_norm,
                    "type": "same_class",
                    "clean_prompt": "The apple",  # same prompts for consistency
                    "corrupt_prompt": CORRUPTED_BASELINE,
                }

    # 5. Attribute value direction (red vs blue, no object)
    for w1, w2 in IDENTITY_WORDS["attribute_only"]:
        tid1 = get_token_id(tokenizer, w1)
        tid2 = get_token_id(tokenizer, w2)
        if tid1 is not None and tid2 is not None:
            d = W_U[tid1] - W_U[tid2]
            d_norm = np.linalg.norm(d)
            if d_norm > 0:
                test_directions[f"attribute_{w1}_{w2}"] = {
                    "direction": d / d_norm,
                    "type": "attribute_only",
                    "clean_prompt": "The apple",
                    "corrupt_prompt": CORRUPTED_BASELINE,
                }

    # 6. Unrelated word direction
    for w1, w2 in IDENTITY_WORDS["unrelated"]:
        tid1 = get_token_id(tokenizer, w1)
        tid2 = get_token_id(tokenizer, w2)
        if tid1 is not None and tid2 is not None:
            d = W_U[tid1] - W_U[tid2]
            d_norm = np.linalg.norm(d)
            if d_norm > 0:
                test_directions[f"unrelated_{w1}_{w2}"] = {
                    "direction": d / d_norm,
                    "type": "unrelated",
                    "clean_prompt": "The apple",
                    "corrupt_prompt": CORRUPTED_BASELINE,
                }

    log(f"  Total test directions: {len(test_directions)}")
    for dtype in ["binding", "random", "object_identity", "same_class", "attribute_only", "unrelated"]:
        count = sum(1 for v in test_directions.values() if v["type"] == dtype)
        log(f"    {dtype}: {count}")

    # Now run channel decomposition for each direction
    # We only need clean and corrupt activations once (same prompts)
    # Then project onto different directions
    log(f"\n  Capturing MLP internals for clean/corrupt prompts...")
    clean_caps, clean_final, clean_all_hidden = capture_mlp_internals(
        model, tokenizer, device, "The apple", binding_layers, n_layers)
    corrupt_caps, corrupt_final, corrupt_all_hidden = capture_mlp_internals(
        model, tokenizer, device, CORRUPTED_BASELINE, binding_layers, n_layers)

    direction_results = {}

    for dname, dinfo in test_directions.items():
        direction = dinfo["direction"]
        dtype = dinfo["type"]

        dir_result = {}
        for li in binding_layers:
            mw = mlp_weights[li]
            W_down = mw["W_down"]
            d_ff = mw["d_ff"]
            if W_down is None:
                continue

            gk = f"gate_{li}"
            uk = f"up_{li}"
            if gk not in clean_caps or gk not in corrupt_caps:
                continue

            clean_gate = clean_caps[gk][:d_ff]
            corrupt_gate = corrupt_caps[gk][:d_ff]
            clean_up = clean_caps.get(uk, np.ones(d_ff))[:d_ff]
            corrupt_up = corrupt_caps.get(uk, np.ones(d_ff))[:d_ff]

            analysis = channel_decomposition(
                W_down, direction, clean_gate, clean_up, corrupt_gate, corrupt_up)
            dir_result[li] = analysis

        direction_results[dname] = {
            "type": dtype,
            "layer_results": dir_result,
        }

    del clean_caps, corrupt_caps, clean_all_hidden, corrupt_all_hidden
    gc.collect()
    torch.cuda.empty_cache()

    # ==================================================================
    # Analyze Experiment A results
    # ==================================================================
    log(f"\n{'='*70}")
    log(f"EXPERIMENT A RESULTS: Balance Ratio by Direction Type")
    log(f"{'='*70}")

    type_stats = {}
    for dtype in ["binding", "random", "object_identity", "same_class", "attribute_only", "unrelated"]:
        balances_by_layer = {li: [] for li in binding_layers}
        net_gross_by_layer = {li: [] for li in binding_layers}

        for dname, dresult in direction_results.items():
            if dresult["type"] != dtype:
                continue
            for li, lr in dresult["layer_results"].items():
                balances_by_layer[li].append(lr["balance"])
                net_gross_by_layer[li].append(lr["net_gross_ratio"])

        layer_stats = {}
        for li in binding_layers:
            bvals = balances_by_layer[li]
            nvals = net_gross_by_layer[li]
            if bvals:
                layer_stats[li] = {
                    "balance_mean": round(float(np.mean(bvals)), 4),
                    "balance_std": round(float(np.std(bvals)), 4),
                    "net_gross_mean": round(float(np.mean(nvals)), 4),
                    "net_gross_std": round(float(np.std(nvals)), 4),
                    "n": len(bvals),
                }

        all_balances = []
        all_net_gross = []
        for li in binding_layers:
            all_balances.extend(balances_by_layer[li])
            all_net_gross.extend(net_gross_by_layer[li])

        if all_balances:
            type_stats[dtype] = {
                "balance_mean": round(float(np.mean(all_balances)), 4),
                "balance_std": round(float(np.std(all_balances)), 4),
                "net_gross_mean": round(float(np.mean(all_net_gross)), 4),
                "net_gross_std": round(float(np.std(all_net_gross)), 4),
                "layer_stats": layer_stats,
                "n": len(all_balances),
            }

    # Print summary table
    log(f"\n  {'Direction Type':>20} {'Balance Mean':>13} {'Balance Std':>12} "
        f"{'Net/Gross Mean':>15} {'Net/Gross Std':>14} {'N':>5}")
    log("  " + "-" * 85)

    for dtype in ["binding", "random", "object_identity", "same_class", "attribute_only", "unrelated"]:
        if dtype in type_stats:
            s = type_stats[dtype]
            log(f"  {dtype:>20} {s['balance_mean']:>13.4f} {s['balance_std']:>12.4f} "
                f"{s['net_gross_mean']:>15.4f} {s['net_gross_std']:>14.4f} {s['n']:>5}")

    # Per-layer breakdown
    log(f"\n  Per-layer balance ratio by direction type:")
    log(f"  {'Layer':>6}", end="")
    for dtype in ["binding", "random", "object_identity", "same_class", "attribute_only", "unrelated"]:
        log(f"  {dtype:>15}", end="")
    log()
    log("  " + "-" * (6 + 17 * 6))

    for li in binding_layers:
        log(f"  L{li:>5}", end="")
        for dtype in ["binding", "random", "object_identity", "same_class", "attribute_only", "unrelated"]:
            if dtype in type_stats and str(li) in type_stats[dtype]["layer_stats"]:
                val = type_stats[dtype]["layer_stats"][li]["balance_mean"]
                log(f"  {val:>15.4f}", end="")
            else:
                log(f"  {'N/A':>15}", end="")
        log()

    # Key comparison: binding vs random
    if "binding" in type_stats and "random" in type_stats:
        b_mean = type_stats["binding"]["balance_mean"]
        r_mean = type_stats["random"]["balance_mean"]
        b_ng = type_stats["binding"]["net_gross_mean"]
        r_ng = type_stats["random"]["net_gross_mean"]
        log(f"\n  KEY COMPARISON:")
        log(f"    Binding balance: {b_mean:.4f}  vs  Random balance: {r_mean:.4f}")
        log(f"    Binding net/gross: {b_ng:.4f}  vs  Random net/gross: {r_ng:.4f}")
        if abs(b_mean - r_mean) < 0.05:
            log(f"    → Balance is similar: balanced amplification appears to be MLP's GENERAL property")
        else:
            log(f"    → Balance differs: balanced amplification may be binding-specific")

    # ==================================================================
    # EXPERIMENT B: Micro-Bias Source Decomposition
    # ==================================================================
    log(f"\n{'='*70}")
    log(f"EXPERIMENT B: Micro-Bias Source Decomposition")
    log(f"  Where does the 1-3% net bias come from?")
    log(f"{'='*70}")

    # Use apple-red pair for detailed decomposition
    bias_results = {}
    test_pairs_for_bias = TEST_PAIRS[:6]  # 6 pairs for robustness

    for pidx, (obj, target_val, competitor_val) in enumerate(test_pairs_for_bias):
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

        pair_bias = {}
        for li in binding_layers:
            mw = mlp_weights[li]
            W_down = mw["W_down"]
            W_gate = mw["W_gate"]
            W_up = mw["W_up"]
            d_ff = mw["d_ff"]
            if W_down is None:
                continue

            gk = f"gate_{li}"
            uk = f"up_{li}"
            if gk not in clean_caps or gk not in corrupt_caps:
                continue

            clean_gate = clean_caps[gk][:d_ff]
            corrupt_gate = corrupt_caps[gk][:d_ff]
            clean_up = clean_caps.get(uk, np.ones(d_ff))[:d_ff]
            corrupt_up = corrupt_caps.get(uk, np.ones(d_ff))[:d_ff]

            decomp = micro_bias_source_decomposition(
                W_down, W_gate, W_up, binding_dir,
                clean_gate, clean_up, corrupt_gate, corrupt_up)
            pair_bias[li] = decomp

        bias_results[pair_key] = pair_bias

        del clean_caps, corrupt_caps, clean_all_hidden, corrupt_all_hidden
        gc.collect()
        torch.cuda.empty_cache()

        if (pidx + 1) % 3 == 0 or pidx < 2:
            elapsed = time.time() - t0
            gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            log(f"  [{pidx+1}/{len(test_pairs_for_bias)}] {pair_key}: "
                f"{len(pair_bias)} layers, elapsed={elapsed:.0f}s, GPU={gpu_mem:.2f}GB")

    # ==================================================================
    # Analyze Experiment B results
    # ==================================================================
    log(f"\n{'='*70}")
    log(f"EXPERIMENT B RESULTS: Micro-Bias Source Decomposition")
    log(f"{'='*70}")

    # Aggregate per-layer
    bias_aggs = {}
    for li in binding_layers:
        components = ["full", "gate_driven", "up_driven", "interaction"]
        metrics = ["net", "total_gross", "balance", "net_gross_ratio"]
        
        comp_aggs = {}
        for comp in components:
            metric_aggs = {}
            for metric in metrics:
                vals = []
                for pk, pdata in bias_results.items():
                    if li in pdata and comp in pdata[li]:
                        vals.append(pdata[li][comp][metric])
                if vals:
                    metric_aggs[metric] = {
                        "mean": round(float(np.mean(vals)), 6),
                        "std": round(float(np.std(vals)), 6),
                    }
            comp_aggs[comp] = metric_aggs
        
        # Structure asymmetries
        asym_metrics = ["proj_asymmetry", "silu_asymmetry", 
                       "delta_silu_asymmetry", "delta_up_asymmetry",
                       "corr_silu_dir", "corr_up_dir"]
        asym_aggs = {}
        for am in asym_metrics:
            vals = []
            for pk, pdata in bias_results.items():
                if li in pdata and am in pdata[li]:
                    vals.append(pdata[li][am])
            if vals:
                asym_aggs[am] = {
                    "mean": round(float(np.mean(vals)), 6),
                    "std": round(float(np.std(vals)), 6),
                }
        
        bias_aggs[li] = {"components": comp_aggs, "asymmetries": asym_aggs}

    # Print decomposition table
    log(f"\n  {'Layer':>6} {'Full N/G':>10} {'Gate N/G':>10} {'Up N/G':>10} "
        f"{'Gate Net':>10} {'Up Net':>10} {'Full Net':>10}")
    log("  " + "-" * 70)

    for li in binding_layers:
        ba = bias_aggs[li]
        full_ng = ba["components"]["full"]["net_gross_ratio"]["mean"] if "net_gross_ratio" in ba["components"]["full"] else 0
        gate_ng = ba["components"]["gate_driven"]["net_gross_ratio"]["mean"] if "net_gross_ratio" in ba["components"]["gate_driven"] else 0
        up_ng = ba["components"]["up_driven"]["net_gross_ratio"]["mean"] if "net_gross_ratio" in ba["components"]["up_driven"] else 0
        gate_net = ba["components"]["gate_driven"]["net"]["mean"] if "net" in ba["components"]["gate_driven"] else 0
        up_net = ba["components"]["up_driven"]["net"]["mean"] if "net" in ba["components"]["up_driven"] else 0
        full_net = ba["components"]["full"]["net"]["mean"] if "net" in ba["components"]["full"] else 0
        
        log(f"  L{li:>5} {full_ng:>10.4f} {gate_ng:>10.4f} {up_ng:>10.4f} "
            f"{gate_net:>+10.4f} {up_net:>+10.4f} {full_net:>+10.4f}")

    # Print asymmetry table
    log(f"\n  {'Layer':>6} {'|d|_asym':>10} {'SiLU_asym':>10} {'ΔSiLU_asym':>11} "
        f"{'Δup_asym':>10} {'corr(g,dir)':>12} {'corr(u,dir)':>12}")
    log("  " + "-" * 75)

    for li in binding_layers:
        ba = bias_aggs[li]
        asym = ba["asymmetries"]
        proj_a = asym.get("proj_asymmetry", {}).get("mean", 0)
        silu_a = asym.get("silu_asymmetry", {}).get("mean", 0)
        dsilu_a = asym.get("delta_silu_asymmetry", {}).get("mean", 0)
        dup_a = asym.get("delta_up_asymmetry", {}).get("mean", 0)
        cg = asym.get("corr_silu_dir", {}).get("mean", 0)
        cu = asym.get("corr_up_dir", {}).get("mean", 0)
        
        log(f"  L{li:>5} {proj_a:>10.4f} {silu_a:>10.4f} {dsilu_a:>11.4f} "
            f"{dup_a:>10.4f} {cg:>12.4f} {cu:>12.4f}")

    # Interpretation
    log(f"\n  INTERPRETATION:")
    
    # Check which component has highest net/gross ratio (most selective)
    gate_ng_all = [bias_aggs[li]["components"]["gate_driven"]["net_gross_ratio"]["mean"] 
                   for li in binding_layers if "net_gross_ratio" in bias_aggs[li]["components"]["gate_driven"]]
    up_ng_all = [bias_aggs[li]["components"]["up_driven"]["net_gross_ratio"]["mean"] 
                 for li in binding_layers if "net_gross_ratio" in bias_aggs[li]["components"]["up_driven"]]
    
    if gate_ng_all and up_ng_all:
        gate_ng_mean = float(np.mean(gate_ng_all))
        up_ng_mean = float(np.mean(up_ng_all))
        
        log(f"    Gate-driven net/gross: {gate_ng_mean:.4f}")
        log(f"    Up-driven net/gross: {up_ng_mean:.4f}")
        
        if gate_ng_mean > up_ng_mean * 1.5:
            log(f"    → GATE is the primary source of micro-bias (more selective)")
        elif up_ng_mean > gate_ng_mean * 1.5:
            log(f"    → UP projection is the primary source of micro-bias (more selective)")
        else:
            log(f"    → Both gate and up contribute comparably to micro-bias")
    
    # Check correlations
    corr_g_vals = [bias_aggs[li]["asymmetries"]["corr_silu_dir"]["mean"] 
                   for li in binding_layers if "corr_silu_dir" in bias_aggs[li]["asymmetries"]]
    corr_u_vals = [bias_aggs[li]["asymmetries"]["corr_up_dir"]["mean"] 
                   for li in binding_layers if "corr_up_dir" in bias_aggs[li]["asymmetries"]]
    
    if corr_g_vals:
        log(f"    Gate-correlation with direction: {np.mean(corr_g_vals):.4f}")
    if corr_u_vals:
        log(f"    Up-correlation with direction: {np.mean(corr_u_vals):.4f}")
    
    # Check delta_silu_asymmetry and delta_up_asymmetry
    dsilu_asym = [bias_aggs[li]["asymmetries"]["delta_silu_asymmetry"]["mean"] 
                  for li in binding_layers if "delta_silu_asymmetry" in bias_aggs[li]["asymmetries"]]
    dup_asym = [bias_aggs[li]["asymmetries"]["delta_up_asymmetry"]["mean"] 
                for li in binding_layers if "delta_up_asymmetry" in bias_aggs[li]["asymmetries"]]
    
    if dsilu_asym:
        log(f"    ΔSiLU asymmetry (pos vs neg channels): {np.mean(dsilu_asym):.4f}")
        if abs(np.mean(dsilu_asym) - 1.0) > 0.05:
            log(f"    → Gate activation changes are ASYMMETRIC across pos/neg channels")
        else:
            log(f"    → Gate activation changes are SYMMETRIC across pos/neg channels")
    
    if dup_asym:
        log(f"    Δup asymmetry (pos vs neg channels): {np.mean(dup_asym):.4f}")
        if abs(np.mean(dup_asym) - 1.0) > 0.05:
            log(f"    → Up projection changes are ASYMMETRIC across pos/neg channels")
        else:
            log(f"    → Up projection changes are SYMMETRIC across pos/neg channels")

    # ==================================================================
    # Save results
    # ==================================================================
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
        "experiment_a": {
            "type_stats": type_stats,
            "direction_results": {
                dname: {
                    "type": dr["type"],
                    "layer_results": dr["layer_results"],
                }
                for dname, dr in direction_results.items()
            },
        },
        "experiment_b": {
            "bias_aggs": bias_aggs,
            "bias_results": bias_results,
        },
    })

    os.makedirs("results/phase343_generality", exist_ok=True)
    out_path = f"results/phase343_generality/{model_name}_phase343.json"
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
    log("Phase 343 complete!")
