"""
Phase 369b: Confirmation + RMSNorm Role in DS7B 1D Collapse
============================================================

Key questions from Phase 369:
1. Is DS7B's PC1 causal patching result stable with 82 pairs?
2. Since W_down is full-rank, what causes 1D collapse?
   - Hypothesis: RMSNorm/LayerNorm projects high-D Δh to 1D
   - Test: Measure Δh before and after RMSNorm at each layer
3. How does LayerNorm interact with Δh direction stability?

Method:
  For each layer, compute:
  - Δh before RMSNorm (pre-norm residual difference)
  - Δh after RMSNorm (post-norm residual difference)
  - Cosine similarity between pre-norm and post-norm Δh
  - PCA of pre-norm vs post-norm Δh
  
  In Transformer architectures with pre-norm:
  h_l = h_{l-1} + Attn(RMSNorm(h_{l-1})) + MLP(RMSNorm(h_{l-1}+Attn_out))
  
  So the residual stream Δh at layer l output is:
  Δh_l = Δh_{l-1} + Δ(attn_out) + Δ(mlp_out)
  
  The RMSNorm input is h_{l-1}, so:
  RMSNorm input difference = Δh_{l-1}  (high-D for DS7B L0-L5)
  RMSNorm output difference = ? (after normalization)
  
  If RMSNorm acts as a 1D projection, then:
  RMSNorm(h_clean) - RMSNorm(h_corrupt) should be near-1D
  even if h_clean - h_corrupt is high-D.

Models: deepseek7b (primary), qwen3 (control)
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

# 82 pairs (same as Phase 368b)
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


def rms_norm_pytorch(x, weight, eps=1e-6):
    """Manual RMSNorm: x * weight / sqrt(mean(x^2) + eps)"""
    # x: (d_model,), weight: (d_model,)
    variance = np.mean(x ** 2, axis=-1, keepdims=True)
    x_normed = x / np.sqrt(variance + eps)
    return x_normed * weight


def get_layer_norm_weight(layer, model_name, layer_idx):
    """Get RMSNorm weight, handling meta device."""
    # Input layernorm (pre-attention)
    for attr in ["input_layernorm", "ln_1", "layernorm"]:
        if hasattr(layer, attr):
            ln = getattr(layer, attr)
            if hasattr(ln, "weight"):
                w = ln.weight
                if not w.is_meta:
                    return w.detach().cpu().float().numpy()
                # Meta device
                import glob
                from safetensors import safe_open
                key = f"model.layers.{layer_idx}.{attr}.weight"
                for sf_file in glob.glob(os.path.join(MODEL_CONFIGS[model_name]["path"], '*.safetensors')):
                    with safe_open(sf_file, framework='pt', device='cpu') as sf:
                        if key in sf.keys():
                            return sf.get_tensor(key).float().numpy()
    return None


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "deepseek7b"
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}")
        return

    log(f"Phase 369b: RMSNorm Role + 82-pair PC Confirmation")
    log(f"Model: {model_name}, Pairs: {len(TEST_PAIRS)}")

    cfg = MODEL_CONFIGS[model_name]
    n_layers = cfg["n_layers"]
    d_model = cfg["d_model"]
    n_pairs = len(TEST_PAIRS)

    # 1. Load model
    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    log(f"Model loaded in {time.time()-t0:.1f}s")

    # 2. Collect per-pair Δh at ALL layers (both before and after LayerNorm)
    # Use hooks to capture residual stream at two points:
    #   a) After residual addition (before next layer's input LN)
    #   b) After input LN of next layer
    
    layers = get_layers(model)
    
    # Strategy: collect hidden_states from output_hidden_states=True
    # h_l = residual stream output of layer l
    # h_l after RMSNorm = input to next layer's attention/MLP
    
    # We also need the LN weights for manual RMSNorm computation
    
    # Collect LN weights
    log("Collecting LN weights...")
    ln_weights = {}
    for l in range(n_layers):
        w = get_layer_norm_weight(layers[l], model_name, l)
        if w is not None:
            ln_weights[l] = w
    log(f"  Got {len(ln_weights)} LN weights")

    # Collect Δh vectors
    log("Collecting Δh vectors...")
    dh_per_pair = {}  # l -> (n_pairs, d_model) — raw residual stream Δh
    dh_post_norm = {}  # l -> (n_pairs, d_model) — after RMSNorm applied to residual
    
    for pidx, (obj, target, competitor) in enumerate(TEST_PAIRS):
        if pidx % 10 == 0:
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

        for l in range(n_layers + 1):
            h_clean = clean_out.hidden_states[l][0, last_pos_clean].detach().cpu().float().numpy()
            h_corrupt = corrupt_out.hidden_states[l][0, last_pos_corrupt].detach().cpu().float().numpy()
            dh = h_clean - h_corrupt
            
            if l not in dh_per_pair:
                dh_per_pair[l] = np.zeros((n_pairs, d_model), dtype=np.float32)
            dh_per_pair[l][pidx] = dh

            # Compute post-RMSNorm Δh for layer l (which is input to layer l+1)
            # h_{l} is the residual stream at output of layer l
            # Input to layer l+1's attention = RMSNorm(h_l) * weight
            if l < n_layers and l in ln_weights:
                w = ln_weights[l]  # input LN weight of layer l+1
                # But wait: layer l's input LN is applied to h_{l-1}, not h_l
                # So to get post-norm Δh at layer l's input, we apply LN to h_{l-1}
                pass

        del clean_out, corrupt_out
        torch.cuda.empty_cache()

    # 3. Compute post-RMSNorm Δh using manual RMSNorm
    log("Computing post-RMSNorm Δh...")
    for l in range(n_layers + 1):
        # Post-norm Δh at this layer: apply RMSNorm to each pair's clean/corrupt h
        # This simulates what the next layer sees as input
        if l not in dh_per_pair:
            continue
        
        # For layer l, the next layer sees RMSNorm(h_l) * gamma_l+1
        # But we don't have the exact next layer's LN weight here easily
        # Instead, let's compute: RMSNorm(h_clean_l) - RMSNorm(h_corrupt_l)
        # Using the model's final norm weight as approximation, or just use
        # simple RMSNorm without learned weight
        
        post_norm_diffs = np.zeros((n_pairs, d_model), dtype=np.float32)
        for pidx in range(n_pairs):
            # Reconstruct clean and corrupt h from Δh
            # h_clean = h_corrupt + Δh, but we don't have h_corrupt individually
            # Actually we need the raw h values, not just Δh
            pass
        
        # We need to re-collect raw h values. Let's do a second pass.
        break
    
    # Actually, we need both h_clean and h_corrupt individually to compute RMSNorm difference.
    # The hidden_states give us that directly. Let me restructure.
    
    # Re-collect with raw h values stored
    log("Re-collecting raw h values for RMSNorm analysis...")
    
    h_clean_per_pair = {}  # l -> (n_pairs, d_model)
    h_corrupt_per_pair = {}  # l -> (n_pairs, d_model)
    
    for pidx, (obj, target, competitor) in enumerate(TEST_PAIRS):
        if pidx % 10 == 0:
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

        # Only collect key layers to save memory
        key_layers = list(range(0, n_layers + 1, max(1, n_layers // 7)))
        if n_layers not in key_layers:
            key_layers.append(n_layers)
        
        for l in key_layers:
            h_c = clean_out.hidden_states[l][0, last_pos_clean].detach().cpu().float().numpy()
            h_r = corrupt_out.hidden_states[l][0, last_pos_corrupt].detach().cpu().float().numpy()
            
            if l not in h_clean_per_pair:
                h_clean_per_pair[l] = np.zeros((n_pairs, d_model), dtype=np.float32)
                h_corrupt_per_pair[l] = np.zeros((n_pairs, d_model), dtype=np.float32)
            h_clean_per_pair[l][pidx] = h_c
            h_corrupt_per_pair[l][pidx] = h_r

        del clean_out, corrupt_out
        torch.cuda.empty_cache()

    # 4. RMSNorm analysis
    log("\n=== RMSNorm Analysis ===")
    
    # Compute post-norm Δh for each key layer
    # Use layer l's input LN weight (applied to h_{l-1} to get layer l's input)
    # But for simplicity, compute RMSNorm without learned weight first
    
    rmsnorm_results = {}
    
    for l in sorted(h_clean_per_pair.keys()):
        M_clean = h_clean_per_pair[l]  # (n_pairs, d_model)
        M_corrupt = h_corrupt_per_pair[l]
        M_dh = M_clean - M_corrupt  # raw Δh
        
        # Apply RMSNorm (without learned weight) to each pair
        post_norm_diffs = np.zeros((n_pairs, d_model), dtype=np.float32)
        for i in range(n_pairs):
            normed_clean = rms_norm_pytorch(M_clean[i], np.ones(d_model))
            normed_corrupt = rms_norm_pytorch(M_corrupt[i], np.ones(d_model))
            post_norm_diffs[i] = normed_clean - normed_corrupt
        
        # Also apply RMSNorm with learned weight from next layer
        if (l + 1) in ln_weights:
            w = ln_weights[l + 1]  # Actually this is layer (l+1)'s input LN
            post_norm_weighted_diffs = np.zeros((n_pairs, d_model), dtype=np.float32)
            for i in range(n_pairs):
                normed_clean = rms_norm_pytorch(M_clean[i], w)
                normed_corrupt = rms_norm_pytorch(M_corrupt[i], w)
                post_norm_weighted_diffs[i] = normed_clean - normed_corrupt
        else:
            post_norm_weighted_diffs = None
        
        # PCA on raw Δh
        M_dh_centered = M_dh - M_dh.mean(axis=0, keepdims=True)
        try:
            _, S_raw, Vt_raw = np.linalg.svd(M_dh_centered, full_matrices=False)
            total_raw = np.sum(S_raw**2)
            pc1_raw = (S_raw[0]**2 / total_raw) if total_raw > 1e-10 else 0
            eff_rank_raw = int(np.searchsorted(np.cumsum(S_raw**2 / total_raw), 0.95) + 1)
        except:
            pc1_raw = 0
            eff_rank_raw = -1
        
        # PCA on post-norm Δh (without weight)
        M_pn_centered = post_norm_diffs - post_norm_diffs.mean(axis=0, keepdims=True)
        try:
            _, S_pn, Vt_pn = np.linalg.svd(M_pn_centered, full_matrices=False)
            total_pn = np.sum(S_pn**2)
            pc1_pn = (S_pn[0]**2 / total_pn) if total_pn > 1e-10 else 0
            eff_rank_pn = int(np.searchsorted(np.cumsum(S_pn**2 / total_pn), 0.95) + 1)
        except:
            pc1_pn = 0
            eff_rank_pn = -1
        
        # PCA on post-norm Δh (with weight)
        if post_norm_weighted_diffs is not None:
            M_pnw_centered = post_norm_weighted_diffs - post_norm_weighted_diffs.mean(axis=0, keepdims=True)
            try:
                _, S_pnw, Vt_pnw = np.linalg.svd(M_pnw_centered, full_matrices=False)
                total_pnw = np.sum(S_pnw**2)
                pc1_pnw = (S_pnw[0]**2 / total_pnw) if total_pnw > 1e-10 else 0
                eff_rank_pnw = int(np.searchsorted(np.cumsum(S_pnw**2 / total_pnw), 0.95) + 1)
            except:
                pc1_pnw = 0
                eff_rank_pnw = -1
        else:
            pc1_pnw = -1
            eff_rank_pnw = -1
        
        # Cosine similarity between raw and post-norm Δh (mean directions)
        mean_raw = M_dh.mean(axis=0)
        mean_pn = post_norm_diffs.mean(axis=0)
        norm_raw = np.linalg.norm(mean_raw)
        norm_pn = np.linalg.norm(mean_pn)
        cos_raw_pn = float(np.dot(mean_raw, mean_pn) / (norm_raw * norm_pn)) if norm_raw > 1e-10 and norm_pn > 1e-10 else 0
        
        rmsnorm_results[str(l)] = {
            "pc1_raw": float(pc1_raw),
            "eff_rank_95_raw": eff_rank_raw,
            "pc1_post_norm": float(pc1_pn),
            "eff_rank_95_post_norm": eff_rank_pn,
            "pc1_post_norm_weighted": float(pc1_pnw),
            "eff_rank_95_post_norm_weighted": eff_rank_pnw,
            "cos_raw_postnorm_mean": cos_raw_pn,
        }
        
        log(f"  L{l}: raw PC1={pc1_raw:.3f}({eff_rank_raw}D) | "
            f"post-norm PC1={pc1_pn:.3f}({eff_rank_pn}D) | "
            f"post-norm+w PC1={pc1_pnw:.3f}({eff_rank_pnw}D) | "
            f"cos(raw,pn)={cos_raw_pn:.3f}")
    
    # 5. PC1 recovery with 82 pairs (DS7B only, for confirmation)
    log("\n=== 82-pair PC1 Recovery Confirmation ===")
    pc1_recovery_82 = {}
    
    for l in sorted(dh_per_pair.keys()):
        M = dh_per_pair[l]  # (82, d_model)
        if M.shape[0] < 5:
            continue
        
        M_centered = M - M.mean(axis=0, keepdims=True)
        try:
            U, S, Vt = np.linalg.svd(M_centered, full_matrices=False)
        except:
            continue
        
        total_var = np.sum(S**2)
        if total_var < 1e-10:
            continue
        
        pc1 = Vt[0]
        explained = (S**2) / total_var
        eff_rank_95 = int(np.searchsorted(np.cumsum(explained), 0.95) + 1)
        
        # PC1 alignment with each individual Δh
        cos_sims = []
        for i in range(n_pairs):
            norm_dh = np.linalg.norm(M[i])
            if norm_dh > 1e-10:
                cos_sims.append(float(np.dot(M[i], pc1) / norm_dh))
        
        # Half-split stability
        half1 = M[::2]  # even indices
        half2 = M[1::2]  # odd indices
        _, _, Vt1 = np.linalg.svd(half1 - half1.mean(axis=0), full_matrices=False)
        _, _, Vt2 = np.linalg.svd(half2 - half2.mean(axis=0), full_matrices=False)
        pc1_cos_stability = float(np.abs(np.dot(Vt1[0], Vt2[0])))
        
        pc1_recovery_82[str(l)] = {
            "n_pairs": n_pairs,
            "pc1_explained": float(explained[0]),
            "eff_rank_95": eff_rank_95,
            "cos_pc1_mean": float(np.mean(np.abs(cos_sims))),
            "cos_pc1_std": float(np.std(np.abs(cos_sims))),
            "half_split_pc1_cos": pc1_cos_stability,
        }
        
        log(f"  L{l}: PC1={explained[0]:.3f}, eff_rank={eff_rank_95}, "
            f"cos(PC1,Δh)={np.mean(np.abs(cos_sims)):.3f}±{np.std(np.abs(cos_sims)):.3f}, "
            f"stability={pc1_cos_stability:.3f}")
    
    # 6. Save results
    output = {
        "model": model_name,
        "phase": "369b",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_pairs": n_pairs,
        "n_layers": n_layers,
        "d_model": d_model,
        "rmsnorm_analysis": rmsnorm_results,
        "pc1_82pairs": pc1_recovery_82,
    }

    os.makedirs("results/phase369_pc_patch", exist_ok=True)
    out_path = f"results/phase369_pc_patch/{model_name}_phase369b.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    log(f"Results saved to {out_path}")

    # 7. Release model
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log("Model released.")

    log("\nPhase 369b complete!")


if __name__ == "__main__":
    main()
