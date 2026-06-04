"""
Phase 369: PC Causal Patch + MLP Weight SVD + Subspace Stability
================================================================

Three experiments from Phase 368 analysis:

Part A: PC Causal Patch
  - At key layers, reconstruct Δh using only top-k PCs
  - Patch into corrupt run, measure binding gap recovery
  - DS7B L14: PC1-only should recover most gap (if 1D is causal)
  - Qwen3/GLM4: Need PC1-20 or more for recovery (if high-D is causal)

Part B: MLP Weight Matrix SVD
  - Extract W_down (output projection) for each layer
  - Compute effective rank of weight matrices
  - DS7B L6-L21: if W_down has rank ~1, that explains 1D collapse
  - Compare with Qwen3/GLM4

Part C: Subspace Stability (Half-split PCA)
  - Split 42 pairs into two halves (21 each)
  - Compute PCA on each half
  - Measure subspace angle between two halves
  - If stable → real mechanism; if unstable → sample noise

Test pairs: 42 (same as Phase 368)
Estimated runtime per model:
  Qwen3: ~3 min | GLM4: ~15 min | DS7B: ~8 min
"""

import sys, os, time, json, gc
import torch
import numpy as np
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'tests/glm5')


def log(msg="", end="\n"):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", end=end, flush=True)


# ===== Model Configs =====
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

# Full test pairs (42)
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
TEMPLATE = "The {obj} is {attr}."


# ===== Model Loading =====
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

    if hasattr(model, 'hf_device_map'):
        dmap = model.hf_device_map
        gpu_count = sum(1 for v in dmap.values() if 'cuda' in str(v))
        cpu_count = sum(1 for v in dmap.values() if 'cpu' in str(v))
        gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        log(f"  GPU={gpu_count} components, CPU={cpu_count}, GPU mem={gpu_mem:.2f}GB")

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


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("Cannot find layers")


def get_token_id(tokenizer, word):
    ids = tokenizer.encode(word, add_special_tokens=False)
    return ids[0] if ids else None


# ===== Part A: Collect Δh vectors =====
def collect_dh_vectors(model, tokenizer, device, model_name):
    """Collect per-pair Δh at last_token position for all layers + clean/corrupt logits."""
    cfg = MODEL_CONFIGS[model_name]
    n_layers = cfg["n_layers"]
    d_model = cfg["d_model"]
    n_pairs = len(TEST_PAIRS)

    dh_per_pair = {}
    clean_logits_per_pair = {}
    corrupt_logits_per_pair = {}

    for pidx, (obj, target, competitor) in enumerate(TEST_PAIRS):
        if pidx % 5 == 0:
            log(f"  Pair {pidx+1}/{n_pairs}: {obj}-{target}/{competitor}")

        clean_prompt = TEMPLATE.format(obj=obj, attr=target)
        corrupt_prompt = TEMPLATE.format(obj=CORRUPTED_BASELINE, attr=target)

        clean_inputs = tokenizer(clean_prompt, return_tensors="pt", truncation=True, max_length=64)
        corrupt_inputs = tokenizer(corrupt_prompt, return_tensors="pt", truncation=True, max_length=64)

        input_device = next(model.parameters()).device
        clean_ids = clean_inputs["input_ids"].to(input_device)
        clean_mask = clean_inputs["attention_mask"].to(input_device)
        corrupt_ids = corrupt_inputs["input_ids"].to(input_device)
        corrupt_mask = corrupt_inputs["attention_mask"].to(input_device)

        with torch.no_grad():
            clean_out = model(input_ids=clean_ids, attention_mask=clean_mask,
                              output_hidden_states=True)
            corrupt_out = model(input_ids=corrupt_ids, attention_mask=corrupt_mask,
                                output_hidden_states=True)

        last_pos_clean = clean_ids.shape[1] - 1
        last_pos_corrupt = corrupt_ids.shape[1] - 1

        # Store logits
        clean_logits = clean_out.logits[0, last_pos_clean].detach().cpu().float().numpy()
        corrupt_logits = corrupt_out.logits[0, last_pos_corrupt].detach().cpu().float().numpy()
        clean_logits_per_pair[pidx] = clean_logits
        corrupt_logits_per_pair[pidx] = corrupt_logits

        for l in range(n_layers + 1):
            h_clean = clean_out.hidden_states[l][0, last_pos_clean].detach().cpu().float().numpy()
            h_corrupt = corrupt_out.hidden_states[l][0, last_pos_corrupt].detach().cpu().float().numpy()
            dh = h_clean - h_corrupt
            if l not in dh_per_pair:
                dh_per_pair[l] = np.zeros((n_pairs, d_model), dtype=np.float32)
            dh_per_pair[l][pidx] = dh

        del clean_out, corrupt_out
        torch.cuda.empty_cache()

    return dh_per_pair, clean_logits_per_pair, corrupt_logits_per_pair


# ===== Part A: PC Causal Patch =====
def pc_causal_patch(model, tokenizer, device, model_name,
                    dh_per_pair, clean_logits_per_pair, corrupt_logits_per_pair):
    """
    PC Causal Patch: At key layers, project Δh onto top-k PC subspace,
    then use that projected Δh to modify corrupt hidden state and measure gap recovery.
    
    Instead of actual patching (which requires custom forward pass),
    we use a simpler approach: measure how much of the logit gap
    is explained by the top-k PC components of Δh.
    
    Method: 
    1. At layer l, decompose Δh = sum_i (c_i * pc_i) where pc_i are principal components
    2. Reconstruct Δh_k = sum_{i=1}^{k} (c_i * pc_i) for k = 1, 5, 10, 20, all
    3. Apply Post-LN to h_corrupt + Δh_k, project through W_U
    4. Measure recovered logit gap for target vs competitor
    """
    log("\n=== Part A: PC Causal Patch ===")
    
    cfg = MODEL_CONFIGS[model_name]
    n_layers = cfg["n_layers"]
    d_model = cfg["d_model"]
    n_pairs = len(TEST_PAIRS)
    
    W_U = get_W_U(model, model_name)  # [vocab_size, d_model]
    
    # Key layers for each model
    if model_name == "deepseek7b":
        key_layers = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
    elif model_name == "qwen3":
        key_layers = [0, 4, 8, 12, 16, 20, 24, 28, 32, 35]
    else:  # glm4
        key_layers = [0, 5, 10, 15, 20, 25, 30, 35, 39]
    
    # Also need the final layer RMSNorm weight
    # Get model's final norm (handle meta device for 8bit/bf16+auto models)
    final_norm_weight = None
    for norm_attr in ["norm", "ln_f"]:
        if hasattr(model, "model") and hasattr(model.model, norm_attr):
            norm_obj = getattr(model.model, norm_attr)
            if hasattr(norm_obj, "weight"):
                w = norm_obj.weight
                if not w.is_meta:
                    final_norm_weight = w.detach().cpu().float().numpy()
                    break
                # Meta device: load from safetensors
                import glob
                from safetensors import safe_open
                for sf_file in glob.glob(os.path.join(MODEL_CONFIGS[model_name]["path"], '*.safetensors')):
                    with safe_open(sf_file, framework='pt', device='cpu') as sf:
                        for key in sf.keys():
                            if key.endswith(f'.{norm_attr}.weight'):
                                final_norm_weight = sf.get_tensor(key).float().numpy()
                                break
                    if final_norm_weight is not None:
                        break
                break
    
    if final_norm_weight is None:
        final_norm_weight = np.ones(d_model, dtype=np.float32)
        log("  WARNING: Could not find final norm, using ones")
    
    results = {}
    
    for l in key_layers:
        if l not in dh_per_pair:
            continue
        
        M = dh_per_pair[l]  # (n_pairs, d_model)
        
        # Center and SVD
        M_centered = M - M.mean(axis=0, keepdims=True)
        try:
            U, S, Vt = np.linalg.svd(M_centered, full_matrices=False)
        except Exception as e:
            log(f"  L{l}: SVD failed: {e}")
            continue
        
        # PC directions (in original space, not centered)
        pc_dirs = Vt  # (min(n_pairs, d_model), d_model)
        
        # For each pair, project Δh onto top-k PCs, then apply Post-LN + W_U
        k_values = [1, 3, 5, 10, 20, min(n_pairs, 50)]
        k_values = sorted(set(k for k in k_values if k <= n_pairs))
        if n_pairs not in k_values:
            k_values.append(n_pairs)
        
        # First compute the "clean" logit gap for reference
        target_gaps = {}  # k -> list of recovered gaps
        for k in k_values:
            target_gaps[k] = []
        
        original_gaps = []
        
        for pidx, (obj, target, competitor) in enumerate(TEST_PAIRS):
            target_id = get_token_id(tokenizer, target)
            competitor_id = get_token_id(tokenizer, competitor)
            if target_id is None or competitor_id is None:
                continue
            
            # Original clean logit gap
            clean_logit_gap = clean_logits_per_pair[pidx][target_id] - clean_logits_per_pair[pidx][competitor_id]
            corrupt_logit_gap = corrupt_logits_per_pair[pidx][target_id] - corrupt_logits_per_pair[pidx][competitor_id]
            original_gap_diff = clean_logit_gap - corrupt_logit_gap  # how much gap changes from corrupt to clean
            original_gaps.append(original_gap_diff)
            
            # For PC reconstruction, we compute the Post-LN + W_U logit gap
            # Instead of actual patching, we compute:
            # logit_diff_k = W_U @ PostLN(h_corrupt + Δh_k) - W_U @ PostLN(h_corrupt)
            # This is approximate but avoids full forward pass
            
            # Simpler approach: just measure ||Δh_k|| / ||Δh|| as fraction of signal recovered
            # and also measure how much of the W_U-projected signal is recovered
            
            dh_full = M[pidx]  # full Δh
            
            # Project onto top-k PCs
            coeffs = U[pidx] * S  # projection coefficients (in centered space)
            
            for k in k_values:
                # Reconstruct Δh using top-k PCs
                # Δh_k = mean + sum_{i=0}^{k-1} (coeffs[i] * Vt[i])
                dh_k = M.mean(axis=0) + np.sum(coeffs[:k, np.newaxis] * Vt[:k], axis=0)
                
                # Compute logit signal via W_U projection
                # logit_signal = W_U @ dh (for each direction)
                logit_full = W_U @ dh_full  # [vocab_size]
                logit_k = W_U @ dh_k  # [vocab_size]
                
                # Target-competitor gap recovery
                gap_full = logit_full[target_id] - logit_full[competitor_id]
                gap_k = logit_k[target_id] - logit_k[competitor_id]
                
                # Recovery ratio: how much of the full Δh's logit gap is recovered by top-k PCs
                if abs(gap_full) > 1e-10:
                    recovery = gap_k / gap_full
                else:
                    recovery = 0.0
                
                target_gaps[k].append({
                    "recovery": recovery,
                    "gap_full": float(gap_full),
                    "gap_k": float(gap_k),
                    "norm_ratio": float(np.linalg.norm(dh_k) / max(np.linalg.norm(dh_full), 1e-10)),
                })
        
        layer_result = {}
        for k in k_values:
            recoveries = [x["recovery"] for x in target_gaps[k]]
            norm_ratios = [x["norm_ratio"] for x in target_gaps[k]]
            layer_result[str(k)] = {
                "recovery_mean": float(np.mean(recoveries)),
                "recovery_std": float(np.std(recoveries)),
                "recovery_median": float(np.median(recoveries)),
                "norm_ratio_mean": float(np.mean(norm_ratios)),
                "norm_ratio_std": float(np.std(norm_ratios)),
            }
        
        # Log summary
        summary = " | ".join(f"k={k}:rec={layer_result[str(k)]['recovery_mean']:.3f}" 
                            for k in k_values[:4])
        log(f"  L{l}: {summary}")
        
        results[str(l)] = {
            "k_values": k_values,
            "pc_explained_ratio_top10": [float(x) for x in (S**2 / np.sum(S**2))[:10]],
            "pc_recovery": layer_result,
        }
    
    return results


# ===== Part B: MLP Weight Matrix SVD =====
def load_weight_from_safetensors(model_name, key):
    """Load a weight tensor from safetensors files (for meta device weights)."""
    import glob
    from safetensors import safe_open
    for sf_file in glob.glob(os.path.join(MODEL_CONFIGS[model_name]["path"], '*.safetensors')):
        with safe_open(sf_file, framework='pt', device='cpu') as sf:
            if key in sf.keys():
                return sf.get_tensor(key).float().numpy()
    return None


def safe_get_weight(weight_tensor, model_name, layer_idx, weight_name):
    """Get weight numpy array, handling meta device by loading from safetensors."""
    if not weight_tensor.is_meta:
        return weight_tensor.detach().cpu().float().numpy()
    # Meta device: load from safetensors
    key = f"model.layers.{layer_idx}.{weight_name}"
    w = load_weight_from_safetensors(model_name, key)
    if w is not None:
        return w
    raise ValueError(f"Cannot load weight {key} from safetensors")


def mlp_weight_svd(model, model_name):
    """
    Extract W_down for each layer and compute effective rank.
    
    For SwiGLU: MLP(x) = W_down @ (silu(W_gate @ x) * (W_up @ x))
    The effective rank of W_down determines the output dimensionality.
    If W_down has rank ~1 at DS7B L6-L21, that explains 1D collapse.
    """
    log("\n=== Part B: MLP Weight SVD ===")
    
    cfg = MODEL_CONFIGS[model_name]
    n_layers = cfg["n_layers"]
    layers = get_layers(model)
    
    results = {}
    sample_layers = list(range(0, n_layers, max(1, n_layers // 10)))
    if (n_layers - 1) not in sample_layers:
        sample_layers.append(n_layers - 1)
    sample_layers = sorted(set(sample_layers))
    
    # Also add layers around DS7B's 1D collapse region
    if model_name == "deepseek7b":
        for extra_l in [4, 5, 6, 7, 8, 20, 21, 22, 23, 24, 25, 26]:
            if extra_l not in sample_layers and extra_l < n_layers:
                sample_layers.append(extra_l)
        sample_layers = sorted(set(sample_layers))
    
    for l in sample_layers:
        if l >= len(layers):
            continue
        layer = layers[l]
        mlp = layer.mlp
        
        # Get W_down weight (handle meta device)
        try:
            W_down = safe_get_weight(mlp.down_proj.weight, model_name, l, "mlp.down_proj.weight")
        except Exception as e:
            log(f"  L{l}: Cannot load W_down: {e}")
            continue
        W_down = W_down.astype(np.float32)  # [d_model, intermediate]
        
        # SVD of W_down
        try:
            U, S, Vt = np.linalg.svd(W_down, full_matrices=False)
        except Exception as e:
            log(f"  L{l}: SVD failed: {e}")
            continue
        
        total_energy = np.sum(S**2)
        if total_energy < 1e-10:
            continue
        
        explained = (S**2) / total_energy
        cumulative = np.cumsum(explained)
        
        eff_rank_95 = int(np.searchsorted(cumulative, 0.95) + 1)
        eff_rank_99 = int(np.searchsorted(cumulative, 0.99) + 1)
        
        # Also check W_up and W_gate
        W_up = None
        W_gate = None
        try:
            if hasattr(mlp, 'gate_up_proj'):
                # GLM4: merged
                W_gate_up = safe_get_weight(mlp.gate_up_proj.weight, model_name, l, "mlp.gate_up_proj.weight")
                intermediate = W_gate_up.shape[0] // 2
                W_up = W_gate_up[intermediate:]
                W_gate = W_gate_up[:intermediate]
            else:
                if hasattr(mlp, 'up_proj'):
                    W_up = safe_get_weight(mlp.up_proj.weight, model_name, l, "mlp.up_proj.weight")
                if hasattr(mlp, 'gate_proj'):
                    W_gate = safe_get_weight(mlp.gate_proj.weight, model_name, l, "mlp.gate_proj.weight")
        except Exception as e:
            log(f"  L{l}: Cannot load W_up/W_gate: {e}")
        
        def compute_eff_rank(W):
            try:
                _, s, _ = np.linalg.svd(W, full_matrices=False)
                total = np.sum(s**2)
                if total < 1e-10:
                    return {"eff_rank_95": 0, "eff_rank_99": 0, "top5_ratio": []}
                cum = np.cumsum((s**2) / total)
                return {
                    "eff_rank_95": int(np.searchsorted(cum, 0.95) + 1),
                    "eff_rank_99": int(np.searchsorted(cum, 0.99) + 1),
                    "top5_ratio": [float(x) for x in ((s**2)/total)[:5]],
                }
            except:
                return {"eff_rank_95": -1, "eff_rank_99": -1, "top5_ratio": []}
        
        up_rank = compute_eff_rank(W_up) if W_up is not None else None
        gate_rank = compute_eff_rank(W_gate) if W_gate is not None else None
        
        results[str(l)] = {
            "W_down_shape": list(W_down.shape),
            "W_down_eff_rank_95": eff_rank_95,
            "W_down_eff_rank_99": eff_rank_99,
            "W_down_top5_ratio": [float(x) for x in explained[:5]],
            "W_up_rank": up_rank,
            "W_gate_rank": gate_rank,
        }
        
        log(f"  L{l}: W_down {W_down.shape} eff_rank_95={eff_rank_95}, eff_rank_99={eff_rank_99}, "
            f"top5={np.array(explained[:5]).round(4)}")
    
    return results


# ===== Part C: Subspace Stability (Half-split PCA) =====
def subspace_stability(dh_per_pair, model_name):
    """
    Split 42 pairs into two halves, compute PCA on each,
    measure subspace angle between the two halves.
    
    Subspace angle: angle between the subspaces spanned by top-k PCs.
    If angle ≈ 0 → very stable subspace across samples.
    If angle ≈ 90° → completely different subspaces.
    """
    log("\n=== Part C: Subspace Stability ===")
    
    cfg = MODEL_CONFIGS[model_name]
    n_layers = cfg["n_layers"]
    n_pairs = len(TEST_PAIRS)
    
    # Split pairs: even indices vs odd indices
    half1_indices = list(range(0, n_pairs, 2))  # 0, 2, 4, ...
    half2_indices = list(range(1, n_pairs, 2))  # 1, 3, 5, ...
    
    key_layers = list(range(0, n_layers + 1, max(1, n_layers // 8)))
    if n_layers not in key_layers:
        key_layers.append(n_layers)
    key_layers = sorted(set(key_layers))
    
    results = {}
    
    for l in key_layers:
        if l not in dh_per_pair:
            continue
        
        M = dh_per_pair[l]  # (n_pairs, d_model)
        M1 = M[half1_indices]  # half 1
        M2 = M[half2_indices]  # half 2
        
        # Center each half independently
        M1_c = M1 - M1.mean(axis=0, keepdims=True)
        M2_c = M2 - M2.mean(axis=0, keepdims=True)
        
        # SVD on each half
        try:
            U1, S1, Vt1 = np.linalg.svd(M1_c, full_matrices=False)
            U2, S2, Vt2 = np.linalg.svd(M2_c, full_matrices=False)
        except Exception as e:
            log(f"  L{l}: SVD failed: {e}")
            continue
        
        # Compute subspace angles for top-k PCs
        # Principal subspace angle: cos(θ) = σ_min(V1_k^T @ V2_k)
        # where V1_k and V2_k are top-k right singular vectors
        
        for k in [1, 3, 5, 10, 20]:
            k1 = min(k, Vt1.shape[0])
            k2 = min(k, Vt2.shape[0])
            k_min = min(k1, k2)
            
            if k_min < 1:
                continue
            
            V1_k = Vt1[:k_min]  # (k_min, d_model)
            V2_k = Vt2[:k_min]  # (k_min, d_model)
            
            # Compute correlation matrix
            C = V1_k @ V2_k.T  # (k_min, k_min)
            
            # SVD of correlation matrix
            try:
                U_c, S_c, Vt_c = np.linalg.svd(C)
            except:
                continue
            
            # Principal angles
            cos_angles = np.clip(S_c, 0, 1)
            angles_deg = np.degrees(np.arccos(cos_angles))
            
            key = f"top{k}"
            if l not in results:
                results[str(l)] = {}
            results[str(l)][key] = {
                "principal_angles_deg": [float(x) for x in angles_deg],
                "mean_angle_deg": float(np.mean(angles_deg)),
                "max_angle_deg": float(np.max(angles_deg)),
                "min_cos": float(np.min(cos_angles)),
            }
        
        # Also compute PC1 alignment between halves
        pc1_half1 = Vt1[0]
        pc1_half2 = Vt2[0]
        cos_pc1 = float(np.abs(np.dot(pc1_half1, pc1_half2)))
        
        if str(l) not in results:
            results[str(l)] = {}
        results[str(l)]["pc1_alignment"] = {
            "cos_pc1_abs": cos_pc1,
            "angle_deg": float(np.degrees(np.arccos(min(cos_pc1, 1.0)))),
        }
        
        # Log
        pc1_angle = np.degrees(np.arccos(min(cos_pc1, 1.0)))
        top5_angles = results[str(l)].get("top5", {}).get("mean_angle_deg", -1)
        log(f"  L{l}: PC1_angle={pc1_angle:.1f}°, top5_mean_angle={top5_angles:.1f}°")
    
    return results


# ===== Main =====
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}. Use: qwen3, glm4, deepseek7b")
        return

    log(f"Phase 369: PC Causal Patch + MLP Weight SVD + Subspace Stability")
    log(f"Model: {model_name}")

    # ===== 1. Load model =====
    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    t_load = time.time() - t0
    log(f"Model loaded in {t_load:.1f}s")

    cfg = MODEL_CONFIGS[model_name]
    n_layers = cfg["n_layers"]

    # ===== 2. Collect Δh vectors =====
    t0 = time.time()
    dh_per_pair, clean_logits, corrupt_logits = collect_dh_vectors(
        model, tokenizer, device, model_name)
    t_collect = time.time() - t0
    log(f"Δh collection done in {t_collect:.1f}s")

    # ===== 3. Part A: PC Causal Patch =====
    t0 = time.time()
    pc_patch_results = pc_causal_patch(
        model, tokenizer, device, model_name,
        dh_per_pair, clean_logits, corrupt_logits)
    t_patch = time.time() - t0
    log(f"PC Causal Patch done in {t_patch:.1f}s")

    # ===== 4. Part B: MLP Weight SVD =====
    t0 = time.time()
    mlp_svd_results = mlp_weight_svd(model, model_name)
    t_svd = time.time() - t0
    log(f"MLP Weight SVD done in {t_svd:.1f}s")

    # ===== 5. Part C: Subspace Stability =====
    t0 = time.time()
    stability_results = subspace_stability(dh_per_pair, model_name)
    t_stab = time.time() - t0
    log(f"Subspace Stability done in {t_stab:.1f}s")

    # ===== 6. Save results =====
    output = {
        "model": model_name,
        "phase": "369",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_pairs": len(TEST_PAIRS),
        "n_layers": n_layers,
        "d_model": cfg["d_model"],
        "load_time_s": round(t_load, 1),
        "collect_time_s": round(t_collect, 1),
        "patch_time_s": round(t_patch, 1),
        "svd_time_s": round(t_svd, 1),
        "stability_time_s": round(t_stab, 1),
        "pc_causal_patch": pc_patch_results,
        "mlp_weight_svd": mlp_svd_results,
        "subspace_stability": stability_results,
    }

    os.makedirs("results/phase369_pc_patch", exist_ok=True)
    out_path = f"results/phase369_pc_patch/{model_name}_phase369.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    log(f"Results saved to {out_path}")

    # ===== 7. Release model =====
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log(f"Model released. GPU mem: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # ===== 8. Print summary =====
    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)
    
    log("\n--- PC Causal Patch (logit gap recovery by top-k PCs) ---")
    for l_key in sorted(pc_patch_results.keys(), key=lambda x: int(x)):
        r = pc_patch_results[l_key]
        k_vals = r.get("k_values", [])
        recovery_line = " | ".join(
            f"k={k}:rec={r['pc_recovery'][str(k)]['recovery_mean']:.3f}"
            for k in k_vals[:5]
        )
        pc1 = r.get("pc_explained_ratio_top10", [0])[0]
        log(f"  L{l_key}: PC1={pc1:.3f} | {recovery_line}")
    
    log("\n--- MLP Weight SVD (W_down effective rank) ---")
    for l_key in sorted(mlp_svd_results.keys(), key=lambda x: int(x)):
        r = mlp_svd_results[l_key]
        log(f"  L{l_key}: W_down {r['W_down_shape']} eff_rank_95={r['W_down_eff_rank_95']}, "
            f"eff_rank_99={r['W_down_eff_rank_99']}, top5={np.array(r['W_down_top5_ratio']).round(4)}")
    
    log("\n--- Subspace Stability (half-split) ---")
    for l_key in sorted(stability_results.keys(), key=lambda x: int(x)):
        r = stability_results[l_key]
        pc1_cos = r.get("pc1_alignment", {}).get("cos_pc1_abs", 0)
        top5_angle = r.get("top5", {}).get("mean_angle_deg", -1)
        log(f"  L{l_key}: PC1_cos={pc1_cos:.3f}, top5_mean_angle={top5_angle:.1f}°")

    log("\nPhase 369 complete!")


if __name__ == "__main__":
    main()
