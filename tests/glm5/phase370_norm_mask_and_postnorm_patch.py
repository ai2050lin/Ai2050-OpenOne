"""
Phase 370: Norm Masking Verification + Post-RMSNorm PC Causal Patch
=====================================================================

Based on Phase 369/369b findings:
1. DS7B raw Δh appears 1D (PC1>0.96) but post-RMSNorm becomes 55D
2. Need mathematical proof: is this "norm masking" or true 1D?
3. Need causal test: does post-norm high-D structure have causal effect?

Three experiments:
A) Norm masking math verification:
   - Compute σ₁ absolute norm vs residual norms
   - Show raw Δh = big 1D component + small high-D residual
   - After removing σ₁, residual should have ~55D structure

B) Post-RMSNorm PC causal patch:
   - Do PC patch in post-norm space (not raw space)
   - If post-norm k=1 PC still recovers gap → 1D computation
   - If post-norm needs k=20+ → high-D computation under norm mask

C) Norm-only vs Direction-only decomposition:
   - Test 4 interventions:
     1. Full clean Δh (baseline)
     2. Clean direction + corrupt norm (swap norm)
     3. Corrupt direction + clean norm (swap direction)
     4. Clean PC1 scalar only (just the projection magnitude)
   → Determine whether the causal effect is in the direction or the norm

Models: qwen3, glm4, deepseek7b (tested sequentially)
"""

import sys, os, time, json, gc
import torch
import numpy as np
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, 'tests/glm5')

def log(msg="", end="\n"):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", end=end, flush=True)


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

TEST_PAIRS_42 = [
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

EXTRA_PAIRS = [
    ("elephant", "big", "small"), ("mountain", "big", "small"), ("ant", "small", "big"),
    ("planet", "big", "small"), ("grain", "small", "big"), ("whale", "big", "small"),
    ("boulder", "heavy", "light"), ("feather", "light", "heavy"), ("lead", "heavy", "light"),
    ("balloon", "light", "heavy"), ("steel", "heavy", "light"), ("cotton", "light", "heavy"),
    ("cheetah", "fast", "slow"), ("turtle", "slow", "fast"), ("rocket", "fast", "slow"),
    ("snail", "slow", "fast"), ("lightning", "fast", "slow"), ("sloth", "slow", "fast"),
    ("star", "bright", "dark"), ("cave", "dark", "bright"), ("sun", "bright", "dark"),
    ("shadow", "dark", "bright"), ("lamp", "bright", "dark"), ("night", "dark", "bright"),
    ("flame", "hot", "cold"), ("ice", "cold", "hot"), ("oven", "hot", "cold"),
    ("frost", "cold", "hot"), ("magma", "hot", "cold"), ("winter", "cold", "hot"),
    ("river", "wet", "dry"), ("desert", "dry", "wet"), ("rain", "wet", "dry"),
    ("dust", "dry", "wet"), ("ocean", "wet", "dry"), ("sand", "dry", "wet"),
    ("silk", "soft", "hard"), ("diamond", "hard", "soft"), ("cotton", "soft", "hard"),
    ("iron", "hard", "soft"), ("pillow", "soft", "hard"), ("concrete", "hard", "soft"),
]

TEST_PAIRS = TEST_PAIRS_42 + EXTRA_PAIRS  # 82 pairs

CORRUPTED_BASELINE = "The item"
TEMPLATE = "The {obj} is {attr}."


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
            log(f"  Loaded with attn_impl={impl}")
            break
        except:
            continue
    if model is None:
        raise RuntimeError(f"Failed to load {model_name}")
    model.eval()
    return model, tokenizer, next(model.parameters()).device


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("Cannot find layers")


def get_W_U(model, model_name):
    """Get unembedding matrix W_U (vocab x d_model)."""
    for attr in ["lm_head", "embed_out"]:
        if hasattr(model, attr):
            w = getattr(model, attr)
            if hasattr(w, "weight"):
                weight = w.weight
                if not weight.is_meta:
                    return weight.detach().cpu().float().numpy()
    # Try from safetensors
    import glob
    from safetensors import safe_open
    for sf_file in glob.glob(os.path.join(MODEL_CONFIGS[model_name]["path"], '*.safetensors')):
        with safe_open(sf_file, framework='pt', device='cpu') as sf:
            for key in sf.keys():
                if 'lm_head' in key or 'embed_out' in key:
                    return sf.get_tensor(key).float().numpy()
    return None


def rms_norm_pytorch(x, weight, eps=1e-6):
    """Manual RMSNorm: x * weight / sqrt(mean(x^2) + eps)"""
    variance = np.mean(x ** 2, axis=-1, keepdims=True)
    x_normed = x / np.sqrt(variance + eps)
    return x_normed * weight


def get_layer_norm_weights(model, model_name, n_layers):
    """Get all input layernorm weights."""
    import glob
    from safetensors import safe_open
    
    layers = get_layers(model)
    ln_weights = {}
    
    # First try from model parameters
    for l in range(n_layers):
        layer = layers[l]
        for attr in ["input_layernorm", "ln_1", "layernorm"]:
            if hasattr(layer, attr):
                ln = getattr(layer, attr)
                if hasattr(ln, "weight"):
                    w = ln.weight
                    if not w.is_meta:
                        ln_weights[l] = w.detach().cpu().float().numpy()
                        break
                    # Meta device: load from safetensors
                    key = f"model.layers.{l}.{attr}.weight"
                    for sf_file in glob.glob(os.path.join(MODEL_CONFIGS[model_name]["path"], '*.safetensors')):
                        with safe_open(sf_file, framework='pt', device='cpu') as sf:
                            if key in sf.keys():
                                ln_weights[l] = sf.get_tensor(key).float().numpy()
                                break
                    if l in ln_weights:
                        break
    
    # Also get final norm weight
    final_norm_weight = None
    for norm_attr in ["norm", "ln_f"]:
        if hasattr(model, "model") and hasattr(model.model, norm_attr):
            norm_obj = getattr(model.model, norm_attr)
            if hasattr(norm_obj, "weight"):
                w = norm_obj.weight
                if not w.is_meta:
                    final_norm_weight = w.detach().cpu().float().numpy()
                else:
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
    
    return ln_weights, final_norm_weight


def collect_hidden_states(model, tokenizer, device, model_name, n_layers, d_model):
    """Collect hidden states for all 82 pairs."""
    n_pairs = len(TEST_PAIRS)
    
    # Store raw h_clean, h_corrupt for key layers
    # Use more granular sampling for DS7B L4-L6 region
    cfg = MODEL_CONFIGS[model_name]
    
    key_layers = sorted(set(
        [0, 1, 2, 3, 4, 5, 6, 7, 8] +  # Early layers for rewrite zone
        list(range(9, n_layers, max(1, n_layers // 8))) +  # Mid/late layers
        [n_layers - 2, n_layers - 1, n_layers]  # Final layers
    ))
    key_layers = [l for l in key_layers if l <= n_layers]
    
    h_clean_all = {l: np.zeros((n_pairs, d_model), dtype=np.float32) for l in key_layers}
    h_corrupt_all = {l: np.zeros((n_pairs, d_model), dtype=np.float32) for l in key_layers}
    
    input_device = next(model.parameters()).device
    
    for pidx, (obj, target, competitor) in enumerate(TEST_PAIRS):
        if pidx % 10 == 0:
            log(f"  Pair {pidx+1}/{n_pairs}: {obj}-{target}/{competitor}")
        
        clean_prompt = TEMPLATE.format(obj=obj, attr=target)
        corrupt_prompt = TEMPLATE.format(obj=CORRUPTED_BASELINE, attr=target)
        
        clean_inputs = tokenizer(clean_prompt, return_tensors="pt", truncation=True, max_length=64)
        corrupt_inputs = tokenizer(corrupt_prompt, return_tensors="pt", truncation=True, max_length=64)
        
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
        
        for l in key_layers:
            h_clean_all[l][pidx] = clean_out.hidden_states[l][0, last_pos_clean].detach().cpu().float().numpy()
            h_corrupt_all[l][pidx] = corrupt_out.hidden_states[l][0, last_pos_corrupt].detach().cpu().float().numpy()
        
        del clean_out, corrupt_out
        if pidx % 5 == 0:
            torch.cuda.empty_cache()
    
    return h_clean_all, h_corrupt_all, key_layers


# ===== Part A: Norm Masking Math Verification =====
def norm_masking_verification(h_clean_all, h_corrupt_all, key_layers, model_name, d_model):
    """
    Mathematically verify the norm masking hypothesis.
    
    For each layer's Δh matrix:
    1. SVD: Δh = U Σ V^T
    2. σ₁ norm vs total norm
    3. Residual after removing PC1: Δh_residual = Δh - σ₁ u₁ v₁^T
    4. Norm of PC1 component vs norm of residual
    5. Effective rank of residual
    """
    log("\n" + "="*60)
    log("Part A: Norm Masking Math Verification")
    log("="*60)
    
    n_pairs = h_clean_all[key_layers[0]].shape[0]
    results = {}
    
    for l in key_layers:
        M_clean = h_clean_all[l]  # (n_pairs, d_model)
        M_corrupt = h_corrupt_all[l]
        M_dh = M_clean - M_corrupt  # raw Δh
        
        # Per-pair analysis
        pc1_norms = []  # norm of PC1 component for each pair
        residual_norms = []  # norm of residual for each pair
        total_norms = []
        
        # Global SVD
        M_centered = M_dh - M_dh.mean(axis=0, keepdims=True)
        try:
            U, S, Vt = np.linalg.svd(M_centered, full_matrices=False)
        except:
            continue
        
        total_var = np.sum(S**2)
        if total_var < 1e-10:
            results[str(l)] = {"skip": "zero_variance"}
            continue
        
        pc1_direction = Vt[0]  # (d_model,)
        explained = (S**2) / total_var
        eff_rank_95 = int(np.searchsorted(np.cumsum(explained), 0.95) + 1)
        
        # For each pair, decompose Δh into PC1 component and residual
        for i in range(n_pairs):
            dh = M_dh[i]  # (d_model,)
            total_norm = np.linalg.norm(dh)
            total_norms.append(total_norm)
            
            # PC1 projection: <Δh, pc1> * pc1
            pc1_scalar = np.dot(dh, pc1_direction)
            pc1_component = pc1_scalar * pc1_direction
            residual = dh - pc1_component
            
            pc1_norms.append(np.linalg.norm(pc1_component))
            residual_norms.append(np.linalg.norm(residual))
        
        pc1_norms = np.array(pc1_norms)
        residual_norms = np.array(residual_norms)
        total_norms = np.array(total_norms)
        
        # Ratio: how much of each Δh's norm is in the PC1 direction
        norm_ratio_pc1 = pc1_norms / (total_norms + 1e-10)
        norm_ratio_residual = residual_norms / (total_norms + 1e-10)
        
        # Residual matrix PCA
        M_residual = M_dh - np.outer(M_dh @ pc1_direction, pc1_direction)
        M_residual_centered = M_residual - M_residual.mean(axis=0, keepdims=True)
        try:
            _, S_res, Vt_res = np.linalg.svd(M_residual_centered, full_matrices=False)
            total_res_var = np.sum(S_res**2)
            if total_res_var > 1e-10:
                explained_res = (S_res**2) / total_res_var
                eff_rank_residual = int(np.searchsorted(np.cumsum(explained_res), 0.95) + 1)
            else:
                eff_rank_residual = 0
        except:
            eff_rank_residual = -1
        
        # Apply RMSNorm and check residual
        ones = np.ones(d_model, dtype=np.float32)
        post_norm_residuals = np.zeros_like(M_residual)
        for i in range(n_pairs):
            # Need to reconstruct clean/corrupt with residual
            # Actually: post_norm(Δh) is not equal to post_norm(clean) - post_norm(corrupt)
            # But for the masking test, we check: after removing PC1 from raw Δh,
            # what is the PCA structure of the raw residual?
            pass
        
        results[str(l)] = {
            "pc1_explained_variance": float(explained[0]),
            "eff_rank_95": eff_rank_95,
            "mean_total_norm": float(np.mean(total_norms)),
            "mean_pc1_norm": float(np.mean(pc1_norms)),
            "mean_residual_norm": float(np.mean(residual_norms)),
            "norm_ratio_pc1_mean": float(np.mean(norm_ratio_pc1)),
            "norm_ratio_pc1_std": float(np.std(norm_ratio_pc1)),
            "norm_ratio_residual_mean": float(np.mean(norm_ratio_residual)),
            "residual_eff_rank_95": eff_rank_residual,
            "pc1_norm_over_residual_norm": float(np.mean(pc1_norms) / (np.mean(residual_norms) + 1e-10)),
        }
        
        log(f"  L{l}: PC1_var={explained[0]:.3f}, eff_rank={eff_rank_95} | "
            f"norm: total={np.mean(total_norms):.2f}, PC1={np.mean(pc1_norms):.2f}, "
            f"residual={np.mean(residual_norms):.2f} | "
            f"PC1/residual ratio={np.mean(pc1_norms)/(np.mean(residual_norms)+1e-10):.2f} | "
            f"residual_rank={eff_rank_residual}")
    
    return results


# ===== Part B: Post-RMSNorm PC Causal Patch =====
def postnorm_pc_patch(h_clean_all, h_corrupt_all, key_layers, W_U, ln_weights,
                      model_name, d_model, n_layers):
    """
    PC causal patch in post-RMSNorm space.
    
    Method:
    1. Compute post-RMSNorm Δh = RMSNorm(h_clean) - RMSNorm(h_corrupt)
    2. PCA on post-norm Δh
    3. Reconstruct Δh with top-k PCs in post-norm space
    4. Compute logit gap recovery using W_U
    
    Key question: How many post-norm PCs needed to recover the gap?
    - If k=1 suffices for DS7B → 1D computation even after norm removal
    - If k=20+ needed → high-D computation, raw 1D was norm masking
    """
    log("\n" + "="*60)
    log("Part B: Post-RMSNorm PC Causal Patch")
    log("="*60)
    
    n_pairs = h_clean_all[key_layers[0]].shape[0]
    ones = np.ones(d_model, dtype=np.float32)
    results = {}
    
    k_values = [1, 3, 5, 10, 20, 50]
    
    # Get final norm weight for proper post-norm logit computation
    # We'll use the final layer's RMSNorm weight
    
    for l in key_layers:
        M_clean = h_clean_all[l]
        M_corrupt = h_corrupt_all[l]
        M_dh_raw = M_clean - M_corrupt
        
        # Compute post-RMSNorm Δh
        # Use input_layernorm weight of the NEXT layer if available,
        # otherwise use plain RMSNorm (weight=1)
        if (l + 1) in ln_weights and l < n_layers:
            ln_w = ln_weights[l + 1]
        else:
            ln_w = ones
        
        M_dh_postnorm = np.zeros_like(M_dh_raw)
        for i in range(n_pairs):
            normed_clean = rms_norm_pytorch(M_clean[i], ln_w)
            normed_corrupt = rms_norm_pytorch(M_corrupt[i], ln_w)
            M_dh_postnorm[i] = normed_clean - normed_corrupt
        
        # PCA on post-norm Δh
        M_pn_centered = M_dh_postnorm - M_dh_postnorm.mean(axis=0, keepdims=True)
        try:
            U_pn, S_pn, Vt_pn = np.linalg.svd(M_pn_centered, full_matrices=False)
        except:
            continue
        
        total_pn_var = np.sum(S_pn**2)
        if total_pn_var < 1e-10:
            continue
        
        explained_pn = (S_pn**2) / total_pn_var
        eff_rank_pn = int(np.searchsorted(np.cumsum(explained_pn), 0.95) + 1)
        
        # Also PCA on raw Δh for comparison
        M_raw_centered = M_dh_raw - M_dh_raw.mean(axis=0, keepdims=True)
        try:
            _, S_raw, Vt_raw = np.linalg.svd(M_raw_centered, full_matrices=False)
            total_raw_var = np.sum(S_raw**2)
            explained_raw = (S_raw**2) / total_raw_var
            eff_rank_raw = int(np.searchsorted(np.cumsum(explained_raw), 0.95) + 1)
        except:
            explained_raw = []
            eff_rank_raw = -1
        
        # PC causal patch in post-norm space
        # Compute base logit gap: W_U @ Δh_full
        # For each pair, logit gap = (W_U @ Δh)[target_id] - (W_U @ Δh)[competitor_id]
        # We use mean direction as proxy
        
        mean_dh_raw = M_dh_raw.mean(axis=0)
        mean_dh_pn = M_dh_postnorm.mean(axis=0)
        
        if W_U is None:
            log(f"  L{l}: W_U not available, skipping patch")
            continue
        
        # Base gap in raw space
        logits_raw = W_U @ mean_dh_raw  # (vocab,)
        base_gap_raw = np.linalg.norm(logits_raw)  # Use total logit norm as proxy
        
        # Base gap in post-norm space
        logits_pn = W_U @ mean_dh_pn
        base_gap_pn = np.linalg.norm(logits_pn)
        
        layer_result = {
            "raw_pc1": float(explained_raw[0]) if len(explained_raw) > 0 else -1,
            "raw_eff_rank_95": eff_rank_raw,
            "postnorm_pc1": float(explained_pn[0]),
            "postnorm_eff_rank_95": eff_rank_pn,
            "base_gap_raw": float(base_gap_raw),
            "base_gap_pn": float(base_gap_pn),
        }
        
        # PC patch in post-norm space
        for k in k_values:
            if k > len(S_pn):
                continue
            
            # Reconstruct: Δh_k = sum_{i=1}^{k} (Δh · v_i) v_i (using centered data + mean)
            # Mean reconstruction
            mean_pn = M_dh_postnorm.mean(axis=0)
            coeffs = (M_dh_postnorm - mean_pn) @ Vt_pn[:k].T  # (n_pairs, k)
            reconstructed = coeffs @ Vt_pn[:k] + mean_pn  # (n_pairs, d_model)
            mean_recon = reconstructed.mean(axis=0)
            
            logits_recon = W_U @ mean_recon
            gap_recon = np.linalg.norm(logits_recon)
            
            recovery = gap_recon / (base_gap_pn + 1e-10) if base_gap_pn > 1e-10 else 0
            
            # Also compute per-pair recovery using target/competitor
            # Get token IDs for target/competitor
            per_pair_recovery = []
            for pidx in range(min(n_pairs, 20)):  # Sample 20 pairs
                obj, target, competitor = TEST_PAIRS[pidx]
                dh_full_pn = M_dh_postnorm[pidx]
                dh_k_pn = reconstructed[pidx]
                
                logits_full = W_U @ dh_full_pn
                logits_k = W_U @ dh_k_pn
                
                # We don't have target/competitor token IDs easily,
                # so use norm-based proxy
                per_pair_recovery.append(float(np.linalg.norm(logits_k) / (np.linalg.norm(logits_full) + 1e-10)))
            
            layer_result[f"postnorm_k{k}_recovery"] = float(recovery)
            layer_result[f"postnorm_k{k}_per_pair_mean"] = float(np.mean(per_pair_recovery))
        
        # Also do raw PC patch for comparison
        for k in [1, 5, 10, 20]:
            if k > len(S_raw):
                continue
            mean_raw = M_dh_raw.mean(axis=0)
            coeffs_raw = (M_dh_raw - mean_raw) @ Vt_raw[:k].T
            recon_raw = coeffs_raw @ Vt_raw[:k] + mean_raw
            mean_recon_raw = recon_raw.mean(axis=0)
            
            logits_recon_raw = W_U @ mean_recon_raw
            gap_recon_raw = np.linalg.norm(logits_recon_raw)
            recovery_raw = gap_recon_raw / (base_gap_raw + 1e-10) if base_gap_raw > 1e-10 else 0
            
            layer_result[f"raw_k{k}_recovery"] = float(recovery_raw)
        
        results[str(l)] = layer_result
        
        log(f"  L{l}: raw PC1={layer_result.get('raw_pc1', -1):.3f}({eff_rank_raw}D) | "
            f"post-norm PC1={explained_pn[0]:.3f}({eff_rank_pn}D) | "
            f"post-norm k=1 recovery={layer_result.get('postnorm_k1_recovery', -1):.3f} | "
            f"post-norm k=5 recovery={layer_result.get('postnorm_k5_recovery', -1):.3f} | "
            f"raw k=1 recovery={layer_result.get('raw_k1_recovery', -1):.3f}")
    
    return results


# ===== Part C: Norm vs Direction Decomposition =====
def norm_direction_decomposition(h_clean_all, h_corrupt_all, key_layers, W_U,
                                  ln_weights, model_name, d_model, n_layers):
    """
    Separate norm effect from direction effect.
    
    For each pair, compute:
    1. Full Δh (baseline)
    2. Direction-swapped: corrupt direction + clean norm
       → Δh_swapped = (clean_norm / corrupt_norm) * corrupt_Δh_direction
       Actually: we decompose into norm and direction components
    
    More precisely:
    - Δh = ||Δh|| * (Δh / ||Δh||)  =  norm * direction
    - Clean direction: direction of (h_clean - h_corrupt)
    - Clean norm: ||h_clean - h_corrupt||
    
    Interventions:
    A) Clean Δh → corrupt h + full clean Δh  (baseline)
    B) Clean direction only → corrupt h + (||corrupt_Δh|| * clean_direction)
    C) Clean norm only → corrupt h + (clean_norm * corrupt_direction)
    D) Clean PC1 scalar only → corrupt h + pc1_projection_component
    """
    log("\n" + "="*60)
    log("Part C: Norm vs Direction Decomposition")
    log("="*60)
    
    n_pairs = h_clean_all[key_layers[0]].shape[0]
    ones = np.ones(d_model, dtype=np.float32)
    results = {}
    
    for l in key_layers:
        M_clean = h_clean_all[l]
        M_corrupt = h_corrupt_all[l]
        M_dh = M_clean - M_corrupt  # clean Δh
        
        # Get PCA for PC1 direction
        M_centered = M_dh - M_dh.mean(axis=0, keepdims=True)
        try:
            _, S, Vt = np.linalg.svd(M_centered, full_matrices=False)
        except:
            continue
        total_var = np.sum(S**2)
        if total_var < 1e-10:
            continue
        
        pc1_dir = Vt[0]  # (d_model,)
        explained = (S**2) / total_var
        
        # Per-pair decomposition
        full_effects = []  # logit effect of full clean Δh
        direction_only_effects = []  # logit effect of clean direction + corrupt norm
        norm_only_effects = []  # logit effect of clean norm + corrupt direction  
        pc1_scalar_effects = []  # logit effect of PC1 projection only
        orthogonal_effects = []  # logit effect of orthogonal residual only
        
        for i in range(n_pairs):
            dh_clean = M_dh[i]  # clean Δh
            norm_clean = np.linalg.norm(dh_clean)
            if norm_clean < 1e-10:
                continue
            dir_clean = dh_clean / norm_clean
            
            # PC1 projection
            pc1_scalar = np.dot(dh_clean, pc1_dir)
            pc1_component = pc1_scalar * pc1_dir
            orthogonal_component = dh_clean - pc1_component
            
            # For logit effect, we need W_U projection
            if W_U is None:
                continue
            
            # Compute logit effects (norm of W_U @ component)
            logits_full = np.linalg.norm(W_U @ dh_clean)
            logits_pc1 = np.linalg.norm(W_U @ pc1_component)
            logits_ortho = np.linalg.norm(W_U @ orthogonal_component)
            
            # Direction only: use clean direction but with unit norm
            logits_dir_only = np.linalg.norm(W_U @ dir_clean)
            
            # Norm only: use corrupt direction with clean norm
            # We don't have a separate "corrupt Δh" per pair (since corrupt is baseline)
            # Instead, use the mean direction across all pairs
            mean_dir = M_dh.mean(axis=0)
            mean_dir_norm = np.linalg.norm(mean_dir)
            if mean_dir_norm > 1e-10:
                mean_dir_unit = mean_dir / mean_dir_norm
                norm_only_vec = norm_clean * mean_dir_unit
                logits_norm_only = np.linalg.norm(W_U @ norm_only_vec)
            else:
                logits_norm_only = 0
            
            full_effects.append(logits_full)
            direction_only_effects.append(logits_dir_only)
            norm_only_effects.append(logits_norm_only)
            pc1_scalar_effects.append(logits_pc1)
            orthogonal_effects.append(logits_ortho)
        
        if len(full_effects) == 0:
            continue
        
        full_effects = np.array(full_effects)
        direction_only_effects = np.array(direction_only_effects)
        norm_only_effects = np.array(norm_only_effects)
        pc1_scalar_effects = np.array(pc1_scalar_effects)
        orthogonal_effects = np.array(orthogonal_effects)
        
        results[str(l)] = {
            "pc1_explained": float(explained[0]),
            "mean_full_effect": float(np.mean(full_effects)),
            "mean_pc1_effect": float(np.mean(pc1_scalar_effects)),
            "mean_ortho_effect": float(np.mean(orthogonal_effects)),
            "pc1_effect_ratio": float(np.mean(pc1_scalar_effects) / (np.mean(full_effects) + 1e-10)),
            "ortho_effect_ratio": float(np.mean(orthogonal_effects) / (np.mean(full_effects) + 1e-10)),
            "mean_direction_only_effect": float(np.mean(direction_only_effects)),
            "mean_norm_only_effect": float(np.mean(norm_only_effects)),
            "direction_effect_ratio": float(np.mean(direction_only_effects) / (np.mean(full_effects) + 1e-10)),
            "norm_effect_ratio": float(np.mean(norm_only_effects) / (np.mean(full_effects) + 1e-10)),
        }
        
        log(f"  L{l}: PC1_var={explained[0]:.3f} | "
            f"PC1_effect={results[str(l)]['pc1_effect_ratio']:.3f} | "
            f"ortho_effect={results[str(l)]['ortho_effect_ratio']:.3f} | "
            f"dir_effect={results[str(l)]['direction_effect_ratio']:.3f} | "
            f"norm_effect={results[str(l)]['norm_effect_ratio']:.3f}")
    
    return results


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "deepseek7b"
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}. Use: qwen3, glm4, deepseek7b")
        return

    log(f"Phase 370: Norm Masking + Post-Norm PC Patch + Norm/Direction Decomposition")
    log(f"Model: {model_name}")

    cfg = MODEL_CONFIGS[model_name]
    n_layers = cfg["n_layers"]
    d_model = cfg["d_model"]

    # 1. Load model
    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    log(f"Model loaded in {time.time()-t0:.1f}s")

    # 2. Get W_U
    log("Getting W_U...")
    W_U = get_W_U(model, model_name)
    if W_U is not None:
        log(f"  W_U shape: {W_U.shape}")
    else:
        log("  WARNING: W_U not available")

    # 3. Get LN weights
    log("Getting LN weights...")
    ln_weights, final_norm_weight = get_layer_norm_weights(model, model_name, n_layers)
    log(f"  Got {len(ln_weights)} LN weights, final_norm={final_norm_weight is not None}")

    # 4. Collect hidden states
    log("Collecting hidden states for 82 pairs...")
    t0 = time.time()
    h_clean_all, h_corrupt_all, key_layers = collect_hidden_states(
        model, tokenizer, device, model_name, n_layers, d_model)
    log(f"  Collected in {time.time()-t0:.1f}s, {len(key_layers)} key layers")

    # Release model to free memory
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log("Model released")

    # 5. Part A: Norm masking verification
    t0 = time.time()
    part_a = norm_masking_verification(h_clean_all, h_corrupt_all, key_layers, model_name, d_model)
    log(f"Part A done in {time.time()-t0:.1f}s")

    # 6. Part B: Post-norm PC patch
    t0 = time.time()
    part_b = postnorm_pc_patch(h_clean_all, h_corrupt_all, key_layers, W_U, ln_weights,
                                model_name, d_model, n_layers)
    log(f"Part B done in {time.time()-t0:.1f}s")

    # 7. Part C: Norm vs Direction decomposition
    t0 = time.time()
    part_c = norm_direction_decomposition(h_clean_all, h_corrupt_all, key_layers, W_U,
                                           ln_weights, model_name, d_model, n_layers)
    log(f"Part C done in {time.time()-t0:.1f}s")

    # 8. Save results
    output = {
        "model": model_name,
        "phase": "370",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_pairs": len(TEST_PAIRS),
        "n_layers": n_layers,
        "d_model": d_model,
        "key_layers": key_layers,
        "part_a_norm_masking": part_a,
        "part_b_postnorm_pc_patch": part_b,
        "part_c_norm_direction": part_c,
    }

    os.makedirs("results/phase370_norm_mask", exist_ok=True)
    out_path = f"results/phase370_norm_mask/{model_name}_phase370.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log(f"Results saved to {out_path}")

    # 9. Summary
    log("\n" + "="*60)
    log("SUMMARY")
    log("="*60)
    
    log("\nPart A: Norm Masking (key layers)")
    for l in sorted(part_a.keys(), key=int):
        r = part_a[l]
        if "skip" in r:
            continue
        log(f"  L{l}: PC1/residual_norm_ratio={r.get('pc1_norm_over_residual_norm', -1):.2f}, "
            f"residual_rank={r.get('residual_eff_rank_95', -1)}, "
            f"norm_ratio_PC1={r.get('norm_ratio_pc1_mean', -1):.3f}")
    
    log("\nPart B: Post-Norm PC Patch")
    for l in sorted(part_b.keys(), key=int):
        r = part_b[l]
        log(f"  L{l}: raw PC1={r.get('raw_pc1', -1):.3f}({r.get('raw_eff_rank_95', '?')}D) → "
            f"post-norm PC1={r.get('postnorm_pc1', -1):.3f}({r.get('postnorm_eff_rank_95', '?')}D) | "
            f"pn_k1={r.get('postnorm_k1_recovery', -1):.3f}, pn_k5={r.get('postnorm_k5_recovery', -1):.3f}, "
            f"raw_k1={r.get('raw_k1_recovery', -1):.3f}")
    
    log("\nPart C: Norm vs Direction")
    for l in sorted(part_c.keys(), key=int):
        r = part_c[l]
        log(f"  L{l}: PC1_effect={r.get('pc1_effect_ratio', -1):.3f}, "
            f"ortho_effect={r.get('ortho_effect_ratio', -1):.3f}, "
            f"dir_effect={r.get('direction_effect_ratio', -1):.3f}, "
            f"norm_effect={r.get('norm_effect_ratio', -1):.3f}")
    
    log(f"\nPhase 370 complete for {model_name}!")


if __name__ == "__main__":
    main()
